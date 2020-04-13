import torch
import numpy as np

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
def set_from_intr(intr, batch_size, device=None):

    to_torch = True
    uv_grid = gen_uv_grid(intr.width, intr.height, to_torch) # 2*H*W

    K = intr.K_unit.copy()
    K[0,:] = K[0,:] * float(intr.width)
    K[1,:] = K[1,:] * float(intr.height)
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

    width_cur = intr.width
    height_cur = intr.height

    return uvb_grid, xy1_grid, width_cur, height_cur, K
