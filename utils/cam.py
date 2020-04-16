import torch
import numpy as np
import torch.nn as nn
import os
from collections import Counter

'''from bts/c3d_loss.py'''
def gen_uv_grid(width, height, torch_mode):
    """
    return: uv_coords(2*H*W)
    """
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    uv_grid = np.stack(meshgrid, axis=0).astype(np.float32) # 2*H*W
    if torch_mode:
        uv_grid = torch.from_numpy(uv_grid)
    return uv_grid

'''from bts/c3d_loss.py'''
def xy1_from_uv(uv_grid, inv_K, torch_mode):
    """
    uv_grid: 2*H*W
    inv_K: 3*3
    return: uv1_flat, xy1_flat(3*N)
    """
    if torch_mode:
        uv_flat = uv_grid.reshape(2, -1) # 2*N
        dummy_ones = torch.ones((1, uv_flat.shape[1]), dtype=uv_flat.dtype, device=uv_flat.device) # 1*N
        uv1_flat = torch.cat((uv_flat, dummy_ones), dim=0) # 3*N
        xy1_flat = torch.matmul(inv_K, uv1_flat)
    else:
        uv_flat = uv_grid.reshape(2, -1) # 2*N
        dummy_ones = torch.ones((1, uv_flat.shape[1]), dtype=np.float32)
        uv1_flat = np.concatenate((uv_flat, dummy_ones), axis=0) # 3*N
        xy1_flat = np.matmul(inv_K, uv1_flat)

    return uv1_flat, xy1_flat

'''from bts/c3d_loss.py'''
def set_from_intr(width, height, K_unit, batch_size, device=None, align_corner=True):

    to_torch = True
    uv_grid = gen_uv_grid(width, height, to_torch) # 2*H*W

    K = K_unit.copy()
    effect_w = float(width - 1 if align_corner else width)
    effect_h = float(height - 1 if align_corner else height)
    K = scale_K(K, effect_w, effect_h)

    inv_K = np.linalg.inv(K)
    if to_torch:
        inv_K = torch.from_numpy(inv_K)
        K = torch.from_numpy(K)

    uv1_flat, xy1_flat = xy1_from_uv(uv_grid, inv_K, to_torch)  # 3*N

    uv1_grid = uv1_flat.reshape(3, uv_grid.shape[1], uv_grid.shape[2] ) # 3*H*W
    xy1_grid = xy1_flat.reshape(3, uv_grid.shape[1], uv_grid.shape[2] )

    xy1_grid = xy1_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) # B*3*H*W
    if device is not None:
        xy1_grid = xy1_grid.to(device=device)

    uvb_grid = uv1_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) # B*3*H*W
    for ib in range(batch_size):
        uvb_grid[ib, 2, :, :] = ib
    if device is not None:
        uvb_grid = uvb_grid.to(device=device)

    K = K.unsqueeze(0).repeat(batch_size, 1, 1) # B*3*3
    if device is not None:
        K = K.to(device=device)

    return uvb_grid, xy1_grid, width, height, K

def K_mat2py(K):
    '''
    Matlab index start from 1, python start from 0.
    The only thing needed is to offset cx, cy by 1.
    '''
    K_new = K.copy()
    K_new[0, 2] -= 1
    K_new[1, 2] -= 1
    return K_new

def scale_K(K, scale_w, scale_h, align_corner=True):
    '''
    generate new intrinsic matrix from original K and scale
    https://github.com/pytorch/pytorch/blob/5ac2593d4f2611480a5a9872e08024a665ae3c26/aten/src/ATen/native/cuda/UpSample.cuh
    see area_pixel_compute_source_index function
    '''
    K_new = np.identity(3).astype(np.float32)
    K_new[0, 0] = K[0, 0] * scale_w
    K_new[1, 1] = K[1, 1] * scale_h
    if align_corner:
        K_new[0, 2] = scale_w * K[0, 2]
        K_new[1, 2] = scale_h * K[1, 2]
    else:
        K_new[0, 2] = scale_w * (K[0, 2] + 0.5) - 0.5
        K_new[1, 2] = scale_h * (K[1, 2] + 0.5) - 0.5
    return K_new

'''from bts/bts_pre_intr.py'''
def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

'''from bts/bts_pre_intr.py'''
def lidar_to_depth(velo, extr_cam_li, K_unit, im_shape, align_corner=True):
    """extr_cam_li: 4x4, intr_K: 3x3"""
    ## recover K
    intr_K = K_unit.copy()
    effect_w = float(im_shape[1] - 1 if align_corner else im_shape[1])
    effect_h = float(im_shape[0] - 1 if align_corner else im_shape[0])
    intr_K = scale_K(intr_K, effect_w, effect_h)

    ## transform to camera frame
    velo_in_cam_frame = np.dot(extr_cam_li, velo.T).T # N*4
    velo_in_cam_frame = velo_in_cam_frame[:, :3] # N*3, xyz
    velo_in_cam_frame = velo_in_cam_frame[velo_in_cam_frame[:, 2] > 0, :]  # keep forward points

    ## project to image
    velo_proj = np.dot(intr_K, velo_in_cam_frame.T).T
    velo_proj[:, :2] = velo_proj[:, :2] / velo_proj[:, [2]]
    velo_proj[:, :2] = np.round(velo_proj[:, :2])    # -1 is for kitti dataset aligning with its matlab script, now in K_mat2py

    ## crop out-of-view points
    valid_idx = ( velo_proj[:, 0] > -0.5 ) & ( velo_proj[:, 0] < im_shape[1]-0.5 ) & ( velo_proj[:, 1] > -0.5 ) & ( velo_proj[:, 1] < im_shape[0]-0.5 )
    velo_proj = velo_proj[valid_idx, :]

    ## compose depth image
    depth_img = np.zeros((im_shape[:2]))
    depth_img[velo_proj[:, 1].astype(np.int), velo_proj[:, 0].astype(np.int)] = velo_proj[:, 2]

    ## find the duplicate points and choose the closest depth
    velo_proj_lin = sub2ind(depth_img.shape, velo_proj[:, 1], velo_proj[:, 0])
    dupe_proj_lin = [item for item, count in Counter(velo_proj_lin).items() if count > 1]
    for dd in dupe_proj_lin:
        pts = np.where(velo_proj_lin == dd)[0]
        x_loc = int(velo_proj[pts[0], 0])
        y_loc = int(velo_proj[pts[0], 1])
        depth_img[y_loc, x_loc] = velo_proj[pts, 2].min()
    depth_img[depth_img < 0] = 0

    return depth_img