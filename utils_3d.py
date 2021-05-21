import igl
import os
import meshplot as mp
import matplotlib.pyplot as plt
import torch
import numpy as np

# def make_triangles(v, f):
#     triangles = v[f.flatten()].reshape(f.shape[0], f.shape[1], 3)
#     return triangles

# def read_offset_and_normalize(path):
#     v_2, f_2, _ = igl.read_off(path)
#     v_2 = torch.tensor(v_2).cuda()
#     f_2 = torch.tensor(f_2).cuda()
#     v_2 = v_2 - v_2.min()
#     v_2 = v_2 / v_2.max()
#     return v_2, f_2

# def read_offset_and_normalize(path, off_type='off'):
#     if off_type=='off':
#         v_2, f_2, _ = igl.read_off(path)
#     elif off_type=='obj':
#         v_2, _, _, f_2, _, _ = igl.read_obj(path)
#     v_2 = torch.tensor(v_2).cuda()
#     f_2 = torch.tensor(f_2).cuda()
#     v_2 = v_2 - v_2.min()
#     v_2 = v_2 / v_2.max()
#     triangles = v_2[f_2.flatten()].reshape(f_2.shape[0], f_2.shape[1], 3)
#     return v_2, f_2, triangles

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