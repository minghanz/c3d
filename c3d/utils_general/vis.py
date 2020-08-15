import torch
import numpy as np 

import cv2
from PIL import Image   ## PIL cannot handle save uint16 png, use cv2 or imageio
import imageio

import os
from .color import rgbmap

def mask_from_dep_np(depth):
    '''input is np.ndarray
    '''
    dep_mask = np.zeros_like(depth).astype(np.uint8)
    dep_mask[depth>0] = 255
    return dep_mask

def uint8_np_from_img_tensor(img):
    if img.ndim == 4:
        img_np = img.cpu().detach().numpy().transpose(0,2,3,1)
    else:
        assert img.ndim == 3, img.ndim
        img_np = img.cpu().detach().numpy().transpose(1,2,0)
        # img_np = img.permute(1,2,0).cpu().detach().numpy() # equivalent to above line
    img_np = uint8_np_from_img_np(img_np)
    img_np = img_np.squeeze()
    return img_np

def uint8_np_from_img_np(img_np):
    img_min = img_np.min()
    assert img_min >= -1e-5, img_min
    if img_min < 0:
        img_np[img_np < 0] = 0

    if img_np.max() <= 1:
        img_np = img_np * 255
    img_np = np.round(img_np).astype(np.uint8)
    return img_np

def uint16_np_from_img_tensor(img):
    # img = img.squeeze()
    if img.ndim == 4:
        img_np = img.cpu().detach().numpy().transpose(0,2,3,1)
    else:
        assert img.ndim == 3, img.ndim
        img_np = img.cpu().detach().numpy().transpose(1,2,0)
    img_np = uint16_np_from_img_np(img_np)
    img_np = img_np.squeeze()
    return img_np

def uint16_np_from_img_np(img_np):
    img_min = img_np.min()
    assert img_min >= -1e-5, img_min
    if img_min < 0:
        img_np[img_np < 0] = 0

    if img_np.max() <= 1:
        img_np = img_np * 255.0
    if img_np.max() < 255:
        img_np = img_np * 256.0
    img_np = np.round(img_np).astype(np.uint16)
    return img_np

def vis_normal(normal):
    vis = normal * 0.5 + 0.5
    return vis

# def visdepth2realdepth_np(vis_depth, ref_depth=10):
#     dum_zero = np.zeros_like(vis_depth)
#     depth = np.where(vis_depth>0, ref_depth/vis_depth - ref_depth, dum_zero)
#     return depth

# '''from bts/bts_utils.py'''
# def vis_depth(depth, ref_depth=10):        ## why normalize_result in bts_main.py want to convert it to numpy?
#     dum_zero = torch.zeros_like(depth)
#     inv_depth = torch.where(depth>0, ref_depth/(ref_depth+depth), dum_zero)
#     return inv_depth

# def vis_depth_np(depth, ref_depth=10):        ## why normalize_result in bts_main.py convert it to numpy?
#     dum_zero = np.zeros_like(depth)
#     inv_depth = np.where(depth>0, ref_depth/(ref_depth+depth), dum_zero)
#     return inv_depth
    
def visdepth2realdepth_np(vis_depth, ref_depth=10, min_depth=1, max_depth=80):
    eps_a = ref_depth / (max_depth + ref_depth)
    eps_b = min_depth
    depth = ref_depth / (vis_depth + eps_a) - ref_depth + eps_b
    return depth

"""from monodepth2 layers.py"""
def vis_depth(depth, ref_depth=10, min_depth=1, max_depth=80):
    if ref_depth > 0:
        eps_a = ref_depth / (max_depth + ref_depth)
        eps_b = min_depth
        zeros_ = torch.zeros_like(depth)
        clamped_depth = torch.where(depth > 0, torch.clamp(depth, min=min_depth, max=max_depth), zeros_ )
        disp = torch.where(clamped_depth > 0, ref_depth / (clamped_depth + ref_depth - eps_b) - eps_a, zeros_)
    else:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        # scaled_disp = torch.where(depth > 0, 1 / depth, 0)
        zeros_ = torch.zeros_like(depth)
        scaled_disp = torch.where(depth > 0, torch.clamp(1 / depth, min=min_disp, max=max_disp), zeros_)
        disp = torch.where(scaled_disp > 0, ( scaled_disp - min_disp ) / (max_disp - min_disp), zeros_)
    return disp

"""from monodepth2 layers.py"""
def vis_depth_np(depth, ref_depth=10, min_depth=1, max_depth=80):
    if ref_depth > 0:
        eps_a = ref_depth / (max_depth + ref_depth)
        eps_b = min_depth
        zeros_ = np.zeros_like(depth)
        clamped_depth = np.where(depth > 0, np.clip(depth, a_min=min_depth, a_max=max_depth), zeros_ )
        disp = np.where(clamped_depth > 0, ref_depth / (clamped_depth + ref_depth - eps_b) - eps_a, zeros_)
    else:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        # scaled_disp = torch.where(depth > 0, 1 / depth, 0)
        zeros_ = np.zeros_like(depth)
        scaled_disp = np.where(depth > 0, np.clip(1 / depth, a_min=min_disp, a_max=max_disp), zeros_)
        disp = np.where(scaled_disp > 0, ( scaled_disp - min_disp ) / (max_disp - min_disp), zeros_)
    return disp


def dep_img_bw(depth_vis, path=None, name=None):
    '''input is torch.Tensor'''
    depth_vis = uint8_np_from_img_tensor(depth_vis)
    if path is not None and name is not None:
        if not os.path.exists(path):
            os.mkdir(path)
        full_path = os.path.join(path, name)
        cv2.imwrite(full_path, depth_vis)
    return depth_vis

def vis_depth_err(depth, depth_gt, saturate_err=10):
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().detach().numpy().transpose(1,2,0)
    if isinstance(depth_gt, torch.Tensor):
        depth_gt = depth_gt.cpu().detach().numpy().transpose(1,2,0)
    
    depth_valid_mask = mask_from_dep_np(depth_gt)
    dep_diff = np.abs(depth - depth_gt)
    dep_diff[depth_valid_mask==0] = 0
    # print(dep_diff.max())
    # dep_diff_normalized = dep_diff / dep_diff.max()
    dep_diff_normalized = dep_diff / saturate_err
    dep_diff_normalized[dep_diff_normalized>1] = 1
    dep_diff_img = uint8_np_from_img_np(dep_diff_normalized)

    return dep_diff_img

def vis_pts_dist(pts_dist, saturate_err=10):
    '''input 3*H*W tensor'''
    if isinstance(pts_dist, torch.Tensor):
        pts_dist = pts_dist.cpu().detach().numpy().transpose(1,2,0)
    
    scale_min = pts_dist[..., 0].max()
    scale_max = pts_dist[..., 1].max()
    scale_mean = pts_dist[..., 2].max()
    # pts_dist[..., 0] = pts_dist[..., 0] / scale_min
    # pts_dist[..., 1] = pts_dist[..., 1] / scale_max
    # pts_dist[..., 2] = pts_dist[..., 2] / scale_mean

    pts_dist[..., 0] = pts_dist[..., 0] / 1
    pts_dist[..., 1] = pts_dist[..., 1] / saturate_err
    pts_dist[..., 2] = pts_dist[..., 2] / saturate_err
    pts_dist[pts_dist>1] = 1

    img_min = uint8_np_from_img_np(pts_dist[..., [0]])
    img_max = uint8_np_from_img_np(pts_dist[..., [1]])
    img_mean = uint8_np_from_img_np(pts_dist[..., [2]])

    return img_min, img_max, img_mean, scale_min, scale_max, scale_mean

def comment_on_img(img_vis_i, item_name, num):
    ## https://blog.csdn.net/JohinieLi/article/details/78168508
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img_vis_i,'scale of {}: {}'.format(item_name, num),(0,40),font,0.3,(255, 0, 0),1)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
    return img

'''from bts/bts_utils.py'''
def overlay_dep_on_rgb(depth, img, path=None, name=None, overlay=True):
    ''' both 3-dim: C*H*W. not including batch
    dep: output from vis_depth
    both are torch.tensor, between 0~1
    '''
    dep_np = uint8_np_from_img_tensor(depth)
    img_np= uint8_np_from_img_tensor(img)
    return overlay_dep_on_rgb_np(dep_np, img_np, path, name, overlay)

'''from bts/bts_utils.py'''
def overlay_dep_on_rgb_np(dep_255, img_np, path=None, name=None, overlay=True):
    ''' both 3-dim: H*W*C.
    both are np.array, between 0~1
    '''
    dep_mask = mask_from_dep_np(dep_255)

    r, g, b = rgbmap(dep_255, mask_zeros=True)              # return int, shape the same as input. small->r, large->b
    # dep_in_color = np.dstack((b, g, r))
    dep_in_color = np.dstack((r, g, b))                     # small: b, large: r
    dep_in_color = dep_in_color.astype(np.uint8)            # first channel: blue color, last: red

    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    if not overlay:
        img_dep = cv2.add(img_np, dep_in_color)
    else:
        inv_mask = 255 - dep_mask
        img_masked = cv2.bitwise_and(img_np, img_np, mask=inv_mask)     # the mask should be uint8 or int8, single channel
        img_dep = cv2.add(img_masked, dep_in_color)

    if path is not None and name is not None:
        if not os.path.exists(path):
            os.mkdir(path)
        # full_path = os.path.join(path, 'img'+name)
        # cv2.imwrite(full_path, img_np)
        # full_path = os.path.join(path, 'dep'+name)
        # cv2.imwrite(full_path, dep_in_color)
        full_path = os.path.join(path, name)
        cv2.imwrite(full_path, img_dep)
        # full_path = os.path.join(path, 'mask'+name)
        # cv2.imwrite(full_path, dep_mask)

    return img_dep


'''
The conversion of nparrays are in vis.py. This function does not alter the content of arrays.
'''
def save_np_to_img(np_img, filename):
    '''
    np_img: H*W*C or B*H*W*C
    mode: 
    '''
    verbose = False
    ### determine batched
    has_cnl_dim = np_img.shape[-1] <= 3
    if has_cnl_dim:
        batched = np_img.ndim == 4
        if not batched:
            assert np_img.ndim == 3
    else:
        batched = np_img.ndim == 3
        if not batched:
            assert np_img.ndim == 2
    vprint(verbose, batched, filename)
    ### determine mode
    if np_img.max() > 255:
        vprint(verbose, 'uint16 mode')
        mode = 'u16'
        assert np_img.dtype == np.uint16
    else:
        vprint(verbose, 'uint8 mode')
        mode = 'u8'
        assert np_img.dtype == np.uint8

    ### save
    if batched:
        for ib in range(np_img.shape[0]):
            fname_cur = filename + '_{}.png'.format(ib) 
            save_np_img_single(np_img[ib], fname_cur, mode)
    else:
        save_np_img_single(np_img, filename+'.png', mode)

def save_np_img_single(np_img, filename, mode):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    assert mode in ['u16', 'u8']
    if mode == 'u16':
        # ### use opencv
        # cv2.imwrite(filename, np_img)
        ### or use imageio
        imageio.imwrite(filename, np_img)
        ### above two both accept H*W or H*W*1 grayscale image
    else:
        # ### use opencv
        # cv2.imwrite(filename, np_img)
        ### or use PIL.Image. For grayscale image ony H*W is accepted
        if np_img.ndim == 2:
            img = Image.fromarray(np_img, mode='L')
        elif np_img.ndim == 3 and np_img.shape[-1] == 1:
            img = Image.fromarray(np_img[..., 0], mode='L')
        elif np_img.ndim == 3 and np_img.shape[-1] != 1:
            img = Image.fromarray(np_img, mode='RGB')
        else: 
            raise ValueError(img.shape, 'shape cannot be handled')
        img.save(filename)
    return 

def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)