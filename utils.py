import torch
import torchvision
import torch.nn as nn
import torchgeometry as tgm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import torch.nn.functional as F
import numpy as np

def show_image(inp, ax=None, alpha=None):
    inp = inp / torch.max(inp.view(3, -1), dim=-1).values.view(3, 1, 1)
    inp = torchvision.transforms.ToPILImage()(inp)
    if ax is None:
        if alpha is not None:
            plt.imshow(inp, interpolation=None, alpha=alpha)
        else:
            plt.imshow(inp, interpolation=None) #[0,...])
        ax=plt.gca()
        ax.xaxis.set_visible(False);  ax.yaxis.set_visible(False)
        plt.show()
    else:
        ax.xaxis.set_visible(False);  ax.yaxis.set_visible(False)
        if alpha is not None:
            ax.imshow(inp, interpolation=None, alpha=alpha)
        else:
            ax.imshow(inp, interpolation=None)

def show_masks_on_image(img, dists, ax, cmaptouse=None, show_max_points=True, show_dist=True):
    # dists: batch x channel x numrow x numcol
    batch, channel, numrow, numcol = dists.shape
    ax.xaxis.set_visible(False);  ax.yaxis.set_visible(False)
    if img is not None:
        show_image(img, ax=ax, alpha=0.8) # previously, no alpha
    #ax.imshow(img)
    colormaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys'] 
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'pink']
    for i in range(batch):
        if cmaptouse is None:
            cmaptouse = colors[i]
        if show_dist:
            ax.imshow(dists[i, 0, ...], alpha=0.5, interpolation=None, cmap=colormaps[i]) #alpha was 0.3 
        
        if show_max_points:
            dists_use = dists[i, ...]
            dists_use[torch.isnan(dists_use)] = 0
            smmed = torch.sum(torch.abs(dists_use),0)   # added abs
            amx = torch.argmax(smmed)
            obtained_maxinds = unravel_index(amx, smmed.shape)
            ax.plot(obtained_maxinds[1], obtained_maxinds[0], color=colors[i], marker='o', markersize=6)
    
def box_around_point(shp, kpx, kpy, input_R=None):
    out = torch.zeros(shp)
    out[..., kpx, kpy] = 1
    if input_R is not None:
        if input_R % 2 == 1:
            ldist = int((input_R - 1) / 2)
            rdist = ldist
        else:
            ldist = int(input_R / 2)
            rdist = ldist -1
        out[..., (kpx-ldist):(kpx+rdist+1), (kpy-ldist):(kpy+rdist+1)] = 1
    return out

def keypoints_to_input_masks(keypoints, input_R=None, numchannels=3, numrow=64, numcol=64):
    numkeypts = keypoints.shape[0]
    input_dist = torch.zeros(numkeypts, numchannels, numrow, numcol)
    for i in range(numkeypts):
        if len(keypoints.shape) == 2:
            kpx, kpy = keypoints[i, 0], keypoints[i, 1]
        else:
            kpx, kpy = keypoints[i, 0, 0], keypoints[i, 0, 1]
        input_dist[i, :, kpx, kpy] = 1
        input_dist[i,...] = box_around_point(input_dist[i,...].shape, kpx, kpy, input_R=input_R)
    return input_dist

def keypoints_to_output_masks(keypoints, transport_plan, input_R=None, out_thresh=None, numchannels=3, numrow=64, numcol=64, normalize=True):
    numkeypts = keypoints.shape[0]
    input_dist = keypoints_to_input_masks(keypoints, input_R, numchannels, numrow, numcol)
    output_dist = torch.zeros(numkeypts, numchannels, numrow, numcol)
    out_mask = transport_plan(input_dist) # does this work with batches?? if not, it should
    if normalize:
        sms = torch.sum(out_mask, [-1, -2])
        out_mask = out_mask / sms.view(out_mask.shape[0], out_mask.shape[1], 1, 1)
    if out_thresh is not None:
        out_mask[out_mask <= out_thresh] = 0
        #out_mask[out_mask > out_thresh] = 1
    output_dist = out_mask
    return input_dist, output_dist

def mapped_accuracy(output_dists, desired_output_dists, metric='IOU'):
    # both batched in first dimension
    if desired_output_dists.shape[0] != output_dists.shape[0]:
        print('Issue: input and output distributions do not have the same number of batches')
    numbatch = output_dists.shape[0]
    if metric is 'IOU':
        desired_output_dists[torch.abs(desired_output_dists) > 0] = 1
        desired_output_dists = desired_output_dists.bool()
        output_dists[torch.abs(output_dists) > 0] = 1
        output_dists = output_dists.bool()
        return torch.sum(desired_output_dists & output_dists, [1,2,3]) / (torch.sum(desired_output_dists, [1,2,3]) + torch.sum(output_dists, [1,2,3]))
    elif metric is 'dist': #if using this option, don't threshold output_dists!
        distances = torch.zeros(numbatch)
        for i in range(numbatch):
            smmed = torch.sum(output_dists[i, ...],0)
            
            amx = torch.argmax(smmed)
            obtained_maxinds = unravel_index(amx, smmed.shape)
            smmed2 = torch.sum(desired_output_dists[i, ...],0)
            amx2 = torch.argmax(smmed2)
            desired_maxinds = unravel_index(amx2, smmed2.shape)
            distances[i] = torch.sqrt((obtained_maxinds[0]-desired_maxinds[0])**2 + (obtained_maxinds[1]-desired_maxinds[1])**2)
        return distances

def d(x, y):
    # x and y can each be 2-elt lists, or they can be tensors
    return torch.sqrt( torch.tensor( (x[0] - y[0])**2 + (x[1] - y[1])**2 ) )

def normalize_im(im):
    for i in range(im.shape[0]):
        im[i, ...] = im[i, ...] / torch.sum(im[i, ...])
    return im

def normalize_K(K):
    if len(K.shape) == 4:
        sms = torch.sum(K, [-1, -2]).view(K.shape[0], K.shape[1], 1, 1)
    elif len(K.shape) == 6:
        sms = torch.sum(K, [-1, -2, -3]).view(K.shape[0], K.shape[1], K.shape[2], 1, 1, 1)
    K = K / sms
    return K

def make_K_slow(gamma, alpha, R, matched_points, numr=28, numc=28, normalize=False): # for testing 2D version; saved but shouldn't be used
    num_mps = matched_points.shape[0]
    
    matched_point_distsA = torch.zeros(num_mps, numr, numc)
    matched_point_distsB = torch.zeros(num_mps, numr, numc)
    for i in range(num_mps):
        for r in range(numr):
            for c in range(numc):
                matched_point_distsA[i, r, c] = d(matched_points[i][0], [r, c])
                matched_point_distsB[i, r, c] = d(matched_points[i][1], [r, c])
    
    # threshold on matched_point_dists by R:
    R = torch.tensor(R)
    matched_point_distsA = torch.minimum(matched_point_distsA, R)
    matched_point_distsB = torch.minimum(matched_point_distsB, R)
    
    C = torch.zeros(numr, numc, numr, numc)
    C_keypoints = torch.zeros(numr, numc, numr, numc)
    K = torch.zeros(numr, numc, numr, numc)
    for r in range(numr):
        print('r', r)
        for c in range(numc):
            for rv in range(numr):
                for cv in range(numc):
                    C[r, c, rv, cv] = d([r, c], [rv, cv])**2
                    C_keypoints[r, c, rv, cv] = torch.sum((matched_point_distsA[:, r, c] - matched_point_distsB[:, rv, cv])**2, 0).view(1)
                    #K[r, c, rv, cv] = torch.exp(-1 * (C[r, c, rv, cv] + alpha*C_keypoints[r, c, rv, cv]) / (2* gamma**2))                 
    K = torch.exp(-1 * (C + alpha*C_keypoints) / (2* gamma**2))
    if normalize:
        K = normalize_K(K)
    return K, C, C_keypoints # note: given the two C's, can try diff alpha and gamma easily

def make_K_3D(gamma, alpha, R, matched_points, dims=[28, 28, 28], normalize=False): #numr=28, numc=28):
    num_mps = matched_points.shape[0]
    numr, numc, numd = dims[:]

    # these A and B are totally unrelated to the A and B in the names above 
    rA = torch.tensor(range(numr)).float()
    rB = torch.tensor(range(numc)).float()
    rC = torch.tensor(range(numd)).float()

    # First image
    rdists = (matched_points[:, 0, 0].view(num_mps, 1) - rA.view(1, numr))**2
    cdists = (matched_points[:, 0, 1].view(num_mps, 1) - rB.view(1, numc))**2
    ddists = (matched_points[:, 0, 2].view(num_mps, 1) - rC.view(1, numd))**2
    matched_point_distsA = torch.sqrt(rdists.view(num_mps,numr,1,1) + cdists.view(num_mps,1,numc,1) + ddists.view(num_mps,1,1,numd))

    # Second image
    rdists = (matched_points[:, 1, 0].view(num_mps, 1) - rA.view(1, numr))**2
    cdists = (matched_points[:, 1, 1].view(num_mps, 1) - rB.view(1, numc))**2
    ddists = (matched_points[:, 1, 2].view(num_mps, 1) - rC.view(1, numd))**2
    matched_point_distsB = torch.sqrt(rdists.view(num_mps,numr,1,1) + cdists.view(num_mps,1,numc,1) + ddists.view(num_mps,1,1,numd))

    # threshold on matched_point_dists by R:
    R = torch.tensor(R)
    matched_point_distsA = torch.minimum(matched_point_distsA, R)
    matched_point_distsB = torch.minimum(matched_point_distsB, R)
    C_keypoints = sum((matched_point_distsA.unsqueeze(-1).unsqueeze(-1) - matched_point_distsB.unsqueeze(1).unsqueeze(1))**2, 0)
    
    rdiffs = (rA.view(numr, 1) - rA.view(1, numr))**2
    cdiffs = (rB.view(numc, 1) - rB.view(1, numc))**2
    ddiffs = (rC.view(numc, 1) - rC.view(1, numc))**2

    C = rdiffs.view(numr,1,1,numr,1,1) + cdiffs.view(1,numc,1,1,numc,1) + ddiffs.view(1,1,numd,1,1,numd)
    K = torch.exp(-1 * (C + alpha*C_keypoints) / (2* gamma**2))
    if normalize:
        K = normalize_K(K)
    return K, C, C_keypoints # note: given the two C's, can try diff alpha and gamma easily
    

def make_K(gamma, alpha, R, matched_points, numr=28, numc=28, normalize=False):
    num_mps = matched_points.shape[0]
    
    # these A and B are totally unrelated to the A and B in the names above 
    A = torch.tensor(range(numr)).float()
    rdists = (matched_points[:, 0, 0].view(num_mps, 1) - A.view(1, numr))**2
    B = torch.tensor(range(numc)).float()
    cdists = (matched_points[:, 0, 1].view(num_mps, 1) - B.view(1, numc))**2
    matched_point_distsA = torch.sqrt(rdists.view(num_mps,numr,1) + cdists.view(num_mps,1,numc))

    rdists = (matched_points[:, 1, 0].view(num_mps, 1) - A.view(1, numr))**2
    cdists = (matched_points[:, 1, 1].view(num_mps, 1) - B.view(1, numc))**2
    matched_point_distsB = torch.sqrt(rdists.view(num_mps,numr,1) + cdists.view(num_mps,1,numc))
    
    # threshold on matched_point_dists by R:
    R = torch.tensor(R)
    matched_point_distsA = torch.minimum(matched_point_distsA, R)
    matched_point_distsB = torch.minimum(matched_point_distsB, R)
    C_keypoints = sum((matched_point_distsA.unsqueeze(-1).unsqueeze(-1) - matched_point_distsB.unsqueeze(1).unsqueeze(1))**2, 0)
    
    A = torch.tensor(range(numr))
    rdiffs = (A.view(numr, 1) - A.view(1, numr))**2
    B = torch.tensor(range(numc))
    cdiffs = (B.view(numc, 1) - B.view(1, numc))**2
    C = rdiffs.view(numr,1,numr,1) + cdiffs.view(1,numc,1,numc)
    K = torch.exp(-1 * (C + alpha*C_keypoints) / (2* gamma**2))
    if normalize:
        K = normalize_K(K)
    return K, C, C_keypoints # note: given the two C's, can try diff alpha and gamma easily

def multiply_reshape(mat, vec, numr=28, numc=28, numd=1):
    num_batch = vec.shape[0]
    numchannels = vec.shape[1]
    if len(mat.shape) == 4 or len(mat.shape)==6:
        numr = mat.shape[0]
        numc = mat.shape[1]
    else:
        numr = mat.shape[1]
        numc = mat.shape[2]
        if num_batch != mat.shape[0]:
            print('Error in multiply_reshape: given K matrix with a batch dimension not matching the vector to multiply')
            print('K has shape', mat.shape, 'vec has shape', vec.shape)
    if len(mat.shape) == 4 or len(mat.shape) == 5:
        numd = 1
    elif len(mat.shape) == 6:
        numd = mat.shape[2]
    elif len(mat.shape) == 7:
        numd = mat.shape[3]
    numpix = numr*numc*numd
    
    if len(mat.shape) % 2 == 0:
        intermed = torch.matmul(mat.reshape(numpix, numpix).repeat(num_batch,numchannels,1,1),vec.reshape(num_batch,numchannels,-1, 1))
    else:
        intermed = torch.matmul(mat.reshape(num_batch,numpix, numpix).unsqueeze(1).repeat(1,numchannels,1,1),vec.reshape(num_batch,numchannels,-1, 1))
    
    if numd <= 1:
        return intermed.reshape(num_batch,numchannels, numr,numc)
    else:
        return intermed.reshape(num_batch,numchannels,numr, numc, numd)
    
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
















