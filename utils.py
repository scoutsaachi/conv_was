import torch
import torchvision
import torch.nn as nn
import torchgeometry as tgm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import torch.nn.functional as F
import numpy as np

def d(x, y):
    # x and y can each be 2-elt lists, or they can be tensors
    return torch.sqrt( torch.tensor( (x[0] - y[0])**2 + (x[1] - y[1])**2 ) )

def make_K_slow(gamma, alpha, R, matched_points, numr=28, numc=28): # for testing 2D version; saved but shouldn't be used
    num_mps = matched_points.shape[0]
    
    for i in range(num_mps):
        for r in range(28):
            for c in range(28):
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
    return K, C, C_keypoints # note: given the two C's, can try diff alpha and gamma easily

def make_K_3D(gamma, alpha, R, matched_points, dims=[28, 28, 28]): #numr=28, numc=28):
    num_mps = matched_points.shape[0]
    numr, numc, numd = dims[:]

    # these A and B are totally unrelated to the A and B in the names above 
    rA = torch.tensor(range(numr)).float()
    rB = torch.tensor(range(numc)).float()
    rC = torch.tensor(range(numd)).float()

    # First image
    rdists = (matched_points[:, 0, 0].view(num_mps, 1) - rA.view(1, numr))**2
    cdists = (matched_points[:, 0, 1].view(num_mps, 1) - rB.view(1, numc))**2
    ddists = (matched_points[:, 0, 1].view(num_mps, 1) - rC.view(1, numd))**2
    matched_point_distsA = torch.sqrt(rdists.view(num_mps,numr,1,1) + cdists.view(num_mps,1,numc,1) + ddists.view(num_mps,1,1,numd))

    # Second image
    rdists = (matched_points[:, 1, 0].view(num_mps, 1) - rA.view(1, numr))**2
    cdists = (matched_points[:, 1, 1].view(num_mps, 1) - rB.view(1, numc))**2
    ddists = (matched_points[:, 1, 1].view(num_mps, 1) - rC.view(1, numd))**2
    matched_point_distsB = torch.sqrt(rdists.view(num_mps,numr,1,1) + cdists.view(num_mps,1,numc,1) + ddists.view(num_mps,1,1,numd))

    # threshold on matched_point_dists by R:
    R = torch.tensor(R)
    matched_point_distsA = torch.minimum(matched_point_distsA, R)
    matched_point_distsB = torch.minimum(matched_point_distsB, R)
    C_keypoints = sum((matched_point_distsA.unsqueeze(-1).unsqueeze(-1) - matched_point_distsB.unsqueeze(1).unsqueeze(1))**2, 0)
    
    #A = torch.tensor(range(numr))
    rdiffs = (rA.view(numr, 1) - rA.view(1, numr))**2
    #B = torch.tensor(range(numc))
    cdiffs = (rB.view(numc, 1) - rB.view(1, numc))**2
    ddiffs = (rC.view(numc, 1) - rC.view(1, numc))**2

    C = rdiffs.view(numr,1,1,numr,1,1) + cdiffs.view(1,numc,1,1,numc,1) + ddiffs.view(1,1,numd,1,1,numd)
    K = torch.exp(-1 * (C + alpha*C_keypoints) / (2* gamma**2))
    return K, C, C_keypoints # note: given the two C's, can try diff alpha and gamma easily
    

def make_K(gamma, alpha, R, matched_points, numr=28, numc=28):
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
    return K, C, C_keypoints # note: given the two C's, can try diff alpha and gamma easily

def multiply_reshape(mat, vec, numr=28, numc=28, numd=1, numchannels=3):
    num_batch = vec.shape[0]
    numr = mat.shape[0]
    numc = mat.shape[1]
    numpix = numr*numc*numd
    intermed = torch.matmul(mat.reshape(numpix, numpix).repeat(num_batch,numchannels,1,1),vec.reshape(num_batch,numchannels,-1, 1))
    
    if numd <= 1:
        return intermed.reshape(num_batch,numchannels, numr,numc)
    else:
        return intermed.reshape(num_batch,numchannels,numr, numc, numd)
    # return torch.matmul(mat.reshape(numpix, numpix).repeat(num_batch,numchannels,1,1),vec.reshape(num_batch,numchannels,-1, 1)).reshape(num_batch,numchannels,numr,numc)
    

















