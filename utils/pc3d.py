import torch

'''
from monodepth2/cvo_utils.py
'''
def grid_from_concat_flat_func(uvb_split, flat_info, grid_shape):
    """
    uvb_split: a tuple of 3 elements of tensor N*1, only long/byte/bool tensors can be used as indices
    flat_info: 1*C*N
    grid_shape: [B,C,H,W]
    """
    C_info = flat_info.shape[1]
    grid_info = torch.zeros((grid_shape[0], C_info, grid_shape[2], grid_shape[3]), dtype=flat_info.dtype, device=flat_info.device) # B*C*H*W
    flat_info_t = flat_info.squeeze(0).transpose(0,1).unsqueeze(1) # N*1*C
    # print(max(uvb_split[2]), max(uvb_split[1]), max(uvb_split[0]))
    grid_info[uvb_split[2], :, uvb_split[1], uvb_split[0]] = flat_info_t
    return grid_info