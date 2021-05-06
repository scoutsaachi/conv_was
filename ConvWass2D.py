import torch
import torchvision
import torch.nn as nn
import torchgeometry as tgm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import torch.nn.functional as F
import numpy as np

class ConvolutionalWasserstein2D(nn.Module):
    def __init__(self, nin, s, gamma, customH=None, customHT=None):
        super().__init__()
        self.a = torch.ones(1, nin, s, s) #/(s*s)
        self.s = s
        self.nin = nin
        self.gamma = gamma
        
        g_s = s
        if s % 2 == 0:
            g_s = s+1

        
        if customH is not None:
            self.H = customH
            #self.H = (lambda x: tgm.image.GaussianBlur((g_s, g_s), (gamma, gamma))(x) + extra_cost(x)) # changed to mult
            if customHT is None:
                self.HT = customH
            else:
                self.HT = customHT #(lambda x: tgm.image.GaussianBlur((g_s, g_s), (gamma, gamma))(x) + extra_costT(x)) # changed to mult

        else:
            self.H = tgm.image.GaussianBlur((g_s, g_s), (gamma, gamma))
            self.HT = tgm.image.GaussianBlur((g_s, g_s), (gamma, gamma))
        
    def sinkhorn(self, mu_0, mu_1, iters, return_plan=False):
        w = torch.ones(*mu_0.shape)
        v = torch.ones(*mu_0.shape)
        for i in range(iters):
            v = mu_0 /(self.H(self.a*w))
            w = mu_1 /(self.HT(self.a*v)) # changed this one to HT
        out = mu_0 * torch.log(v) + mu_1 * torch.log(w)
        out = torch.flatten(out, 2, 3).unsqueeze(3)
        a = torch.flatten(out, 2, 3).unsqueeze(2)
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
        # p is (arbitrary batch dim) x nin x H x W
        # returns vector of size (arbitrary batch dim) x nin, or nin, of the separate entropies
        z_locs = (p <= 0)
        if len(p.shape) == 4:
            a_reshaped = self.a
        elif len(p.shape) == 3:
            a_reshaped = self.a[0, ...]
        else:
            a_reshaped = self.a[0, 0, ...]
        prod = a_reshaped * p * torch.log(p) 
        prod[z_locs] = 0
        return -1*torch.sum(prod, dim=[-1, -2])
    
    def entropic_sharpening(self, mu, H0):
        # mu is C x H x W
        # H0 is C 
        C, H, W = mu.shape
        
        # Doing this unbatched in C for now
        betas = torch.ones(C)
        for i in range(C):
            if self.compute_entropy(mu[i,...]) + torch.sum(mu[i, ...]*self.a[0, i, ...]) > H0[i] + 1:
                def f(beta):
                    mu_to_beta = torch.pow(mu[i,...], torch.tensor(beta))
                    return torch.sum(self.a[0, i, ...]*mu_to_beta) + self.compute_entropy(mu_to_beta) - (1 + H0[i])
                xx = fsolve(f, x0=[1]) # Does not include positivity constraint, not ideal.
                if xx[0] >= 0:
                    betas[i] = xx[0]
        sharpened = torch.pow(mu, betas.view(C, 1, 1))
        return sharpened
    
    def wass_barycenter(self, mu_s, alphas, iters, entropic_args=None): # NOT BATCHED
        # mu_s is K x C x H x W
        # alphas is K
        # M is the number of mu_s
        # entropic_args['factor'] changes the level of sharpening by scaling the maximum entropy
        # otherwise, setting entropic_args as None means to not do entropic sharpening
        K, C, H, W = mu_s.shape
        
        if entropic_args is not None:
            H0 = torch.max(self.compute_entropy(mu_s), dim=0).values
            H0 = entropic_args['factor']*H0 
        
        v = torch.ones_like(mu_s)
        w = torch.ones_like(mu_s)
        for j in range(iters):
            w = mu_s / self.HT(self.a * v) # changed to HT
            d = v * self.H(self.a * w)
            mu = torch.ones(C, H, W)
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
    