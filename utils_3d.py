import igl
import os
import meshplot as mp
import matplotlib.pyplot as plt
import torch
import numpy as np

import torchvision
import torch.nn as nn
import torchgeometry as tgm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import torch.nn.functional as F
import meshplot as mp
import matplotlib.pyplot as plt
import torchgeometry as tgm
import torch
import torch.nn as nn
import igl
import numpy as np

def make_K_3D_custom(gamma, gamma_kp, alpha, R, matched_points, dims=[28, 28, 28]): #numr=28, numc=28):
    print("gamma", gamma, "gamma_kp", gamma_kp, "alpha", alpha, "R", R, "matched_points", matched_points)
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
    matched_point_distsA = matched_point_distsA.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    matched_point_distsB = matched_point_distsB.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    C_keypoints = torch.mean((matched_point_distsA - matched_point_distsB)**2, dim=0)
#     C_keypoints = torch.max((matched_point_distsA - matched_point_distsB)**2, dim=0)[0]
    
    #A = torch.tensor(range(numr))
    rdiffs = (rA.view(numr, 1) - rA.view(1, numr))**2
    #B = torch.tensor(range(numc))
    cdiffs = (rB.view(numc, 1) - rB.view(1, numc))**2
    ddiffs = (rC.view(numc, 1) - rC.view(1, numc))**2

    C = rdiffs.view(numr,1,1,numr,1,1) + cdiffs.view(1,numc,1,1,numc,1) + ddiffs.view(1,1,numd,1,1,numd)
    C = C.double()
    C_keypoints = C_keypoints.double()
    total_cost = (1-alpha)*C / (2* gamma**2) + alpha*C_keypoints / (2* gamma_kp**2)
    K = torch.exp(-1 * total_cost)
    norm_factor = K.flatten(3, 5).sum(dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return K/norm_factor, C, C_keypoints # note: given the two C's, can try diff alpha and gamma easily

class CustomKeypoint(nn.Module):
    def __init__(self, gamma, gamma_kp, alpha, R, matched_points, dims=[28, 28, 28]):
        super().__init__()
        K = make_K_3D_custom(gamma, gamma_kp, alpha, R, matched_points, dims)[0]
        self.K = K.flatten(3, 5).permute(3, 0, 1, 2).flatten(1, 3).permute(1, 0).unsqueeze(0) # 1 x HWZ x HWZ
        print("kernel", self.K.min(), self.K.max())
        
    def forward_base(self, x, K):
        # x is N, 1, H, W, Z
        N, C, H, W, Z = x.shape
        assert C == 1
        x = x.flatten(2, 4).permute(0, 2, 1) # N, HWZ, 1
        out = K @ x # N x HWZ x 1
        out = out.permute(0, 2, 1) # N x 1 x HWZ
        out = out.reshape(N, C, H, W, Z)
        return out
    
    def forward(self, x):
        return self.forward_base(x, self.K)
        
    def forward_transpose(self, x):
        return  self.forward_base(x, self.K.permute(0, 2, 1))


class Gaussian3DBlur(torch.nn.Module):
    def __init__(self, kernel, sigma):
        super().__init__()
        self.sigma = sigma
        self.kernel = kernel
        
    def conv1d_blur(self, x):
        # x is N, 1, H, W, Z
        # we convolve over Z
        kernel_size = self.kernel
        sigma = self.sigma
        k = tgm.image.gaussian.get_gaussian_kernel(kernel_size,sigma)
        k = k.unsqueeze(0).unsqueeze(0)
        orig_shape = x.shape
        conv_inp = x.permute((4, 0, 1, 2, 3))
        permuted_shape = conv_inp.shape
        conv_inp = conv_inp.reshape(orig_shape[4], -1).permute(1,0).unsqueeze(1)
        out = torch.nn.functional.conv1d(conv_inp, k, padding=(kernel_size - 1) // 2)
        out = out.squeeze(1).permute(1,0).reshape(permuted_shape).permute(1, 2, 3, 4, 0)
        return out
    
    def forward(self, x):
        # x is N, 1, H, W, Z
        x = self.conv1d_blur(x) # convolve over Z
        
        x = x.permute(0, 1, 2, 4, 3)
        x = self.conv1d_blur(x) # convolve over W
        x = x.permute(0, 1, 2, 4, 3)
        
        x = x.permute(0, 1, 3, 4, 2) # convolve over H
        x = self.conv1d_blur(x)
        x = x.permute(0, 1, 4, 2, 3)
        return x
    
    def forward_transpose(self, x):
        return self.forward(x)
        
        
class ConvolutionalWasserstein3D(nn.Module):
    def __init__(self, nin, s, gamma, keypoint_args=None):
        super().__init__()
        self.a = torch.ones(1, nin, s, s, s).double() #/(s*s)
        self.s = s
        self.nin = nin
        self.gamma = gamma
        
        g_s = s
        if s % 2 == 0:
            g_s = s+1

        if keypoint_args is not None:
            self.kernel = CustomKeypoint(
                gamma=gamma, gamma_kp=keypoint_args['gamma_kp'],
                alpha=keypoint_args['alpha'], 
                R=keypoint_args['R'], 
                matched_points=keypoint_args['matched_points'], 
                dims=[s, s, s])
        else:
            self.kernel = Gaussian3DBlur(g_s, gamma)
        
    def sinkhorn(self, mu_0, mu_1, iters, return_plan=False):
        mu_0 = mu_0.double()
        mu_1 = mu_1.double()
        w = torch.ones(*mu_0.shape).double()
        v = torch.ones(*mu_0.shape).double()
        for i in range(iters):
            v = mu_0 /(self.kernel.forward(self.a*w))
            w = mu_1 /(self.kernel.forward_transpose(self.a*v)) # changed this one to HT
            print(i, v.flatten()[:4], w.flatten()[:4])
        out = mu_0 * torch.log(v) + mu_1 * torch.log(w)
        out = torch.flatten(out, 2, 4).unsqueeze(3) # N x C x H x W x Z
        a = torch.flatten(self.a, 2, 4).unsqueeze(2)
        out = (a @ out).squeeze(2).squeeze(2)
        if return_plan:
            return self.gamma * out, v, w
        else:
            return self.gamma * out
        
    def visualize_plan(self, pi_func, input_dist):
        # Show one channel of the input distribution, and where it is sent
        # input_dist does not have to be normalized to be a distribution
        # e.g. pi_func = lambda x: v*self.H(w*x)
        plt.figure(figsize=(15,7))
        plt.subplot(121)
        plt.imshow(input_dist[0,0,...].cpu())
        plt.title('Input distribution')
        plt.colorbar()
        
        output = pi_func(input_dist)
        plt.subplot(122)
        plt.imshow(output[0,0,...].cpu())
        plt.title('Where it is sent')
        plt.colorbar()
        plt.show()
        
    def compute_entropy(self, p):
        # p is (arbitrary batch dim) x nin x H x W x D
        # returns vector of size (arbitrary batch dim) x nin, or nin, of the separate entropies
        if len(p.shape) == 5:
            a_reshape = self.a
        elif len(p.shape) == 4:
            a_reshape = self.a[0]
        elif len(p.shape) == 3:
            a_reshape = self.a[0,0]
        z_locs = (p <= 0)
        prod = a_reshape * p * torch.log(p) 
        prod[z_locs] = 0
        out = -1*torch.sum(prod, dim=[-1, -2, -3])
        return out
    
    def entropic_sharpening(self, mu, H0):
        # mu is C x H x W
        # H0 is C 
        C, H, W, D = mu.shape
        
        # Doing this unbatched in C for now
        betas = torch.ones(C)
        for i in range(C):
            mu_ent = self.compute_entropy(mu[i,...])
            if mu_ent + torch.sum(mu[i, ...]*self.a[0, i, ...]) > H0[i] + 1:
                def f(beta):
                    mu_to_beta = torch.pow(mu[i,...], torch.tensor(beta))
                    return torch.sum(self.a[0, i, ...]*mu_to_beta) + self.compute_entropy(mu_to_beta) - (1 + H0[i])
                xx = fsolve(f, x0=[1]) # Does not include positivity constraint, not ideal.
                if xx[0] >= 0:
                    betas[i] = xx[0]
        sharpened = torch.pow(mu, betas.view(C, 1, 1, 1))
        return sharpened
    
    def wass_barycenter(self, mu_s, alphas, iters, entropic_args=None): # NOT BATCHED
        # mu_s is K x C x H x W
        # alphas is K
        # M is the number of mu_s
        # entropic_args['factor'] changes the level of sharpening by scaling the maximum entropy
        # otherwise, setting entropic_args as None means to not do entropic sharpening
        K, C, H, W, Z = mu_s.shape
        
        if entropic_args is not None:
            H0 = torch.max(self.compute_entropy(mu_s), dim=0).values
            H0 = entropic_args['factor']*H0 
        
        v = torch.ones_like(mu_s)
        w = torch.ones_like(mu_s)
        for j in range(iters):
            print(j)
            w = mu_s / self.kernel.forward_transpose(self.a * v) # changed to HT
            d = v * self.kernel.forward(self.a * w)
            mu = torch.ones(C, H, W, Z)
            for i in range(K):
                mu = mu * torch.pow(d[i], alphas[i])
            if entropic_args is not None:
                mu = self.entropic_sharpening(mu, H0)
            v = (v * mu.unsqueeze(0))/d
        return mu
    
    def wass_barycenter_obj(self, mu, mu_s, alphas):
        # mu_s is K x C x H x W
        # alphas is K
        # mu is C x H x W
        K, C, H, W = mu_s.shape
        
        mu_repped = mu.repeat(K, 1, 1, 1)
        W2dists = self.sinkhorn(mu_repped, mu_s, iters=10) # K x C
        objvals = W2dists * alphas.view(-1,1).repeat_interleave(repeats=C, dim=-1) # K x C
        return torch.sum(objvals, 0)
    

def read_offset_and_normalize_custom(path, off_type='off'):
    if off_type=='off':
        v_2, f_2, _ = igl.read_off(path)
    elif off_type=='obj':
        v_2, _, _, f_2, _, _ = igl.read_obj(path)
    v_2 = torch.tensor(v_2).cuda()
    f_2 = torch.tensor(f_2).cuda()
    for i in range(3):
        v_2[:, i] = v_2[:, i] - v_2[:, i].min()
    ranges = v_2.max(dim=0)[0]
    max_ind = torch.argmax(ranges)
    max_len = ranges[max_ind]
    scale_factor = 1/max_len
    v_2 *= scale_factor
    for i in range(3):
        axis_range = v_2[:, i].max() - v_2[:, i].min()
        v_2[:, i] += (1-axis_range)/2
    triangles = v_2[f_2.flatten()].reshape(f_2.shape[0], f_2.shape[1], 3)
    return v_2, f_2, triangles

def create_voxel_grid(res):
    voxels = np.stack(np.meshgrid(np.arange(0, 1, res), np.arange(0, 1, res), np.arange(0, 1, res)))
    voxels = voxels.transpose(1, 2, 3, 0).reshape(-1, 3).copy()
    voxels = torch.tensor(voxels)
    return voxels

def get_is_in_mesh_batched(triangles, query_points):
    def dot(a, b):
        return (a.unsqueeze(2) @ b.unsqueeze(3)).squeeze(2).squeeze(2)
    M = (triangles.unsqueeze(0) - query_points.unsqueeze(1).unsqueeze(1))
    M_norm = torch.linalg.norm(M, dim=-1)
    D = torch.linalg.det(M)
    A, B, C = M[:, :, 0], M[:, :, 1], M[:, :, 2]
    A_norm, B_norm, C_norm = M_norm[:, :, 0], M_norm[:, :, 1], M_norm[:,:, 2]
    Other = (A_norm * B_norm * C_norm) + C_norm * dot(A, B) + A_norm * dot(B, C) + B_norm * dot(A, C)
    out = torch.atan2(D, Other).sum(dim=-1)
    return (out >= 2*np.pi)

def get_voxelization(triangles, res):
    # returns: an N x 3 numpy array of the voxelized points
    # a sparse array of the voxelized grid
    voxels = create_voxel_grid(res)
    ds = torch.utils.data.TensorDataset(voxels)
    dl = torch.utils.data.DataLoader(ds, batch_size=2000, shuffle=False)
    all_out = []
    with torch.no_grad():
        for voxels_batch, in dl:
            voxels_batch = voxels_batch.cuda()
            all_out.append(get_is_in_mesh_batched(triangles, voxels_batch).cpu())
    all_out = torch.cat(all_out)
    cpu_voxels = voxels[all_out.cpu().numpy()].cpu().numpy()
    
    S = len(np.arange(0, 1, res))
    inds = np.stack(np.meshgrid(np.arange(0, S), np.arange(0, S), np.arange(0, S)))
    inds = inds.transpose(1, 2, 3, 0).reshape(-1, 3).copy()
    sparse_voxelization = torch.sparse_coo_tensor(inds[all_out].T, torch.ones(len(inds[all_out])), (S, S, S))
    return cpu_voxels, sparse_voxelization