import torch
import numpy as np 
import cv2
import os
from .color import rgbmap

def vis_normal(normal):
    vis = normal * 0.5 + 0.5
    return vis

'''from bts/bts_utils.py'''
def vis_depth(depth, ref_depth=10):        ## why normalize_result in bts_main.py want to convert it to numpy?
    dum_zero = torch.zeros_like(depth)
    inv_depth = torch.where(depth>0, ref_depth/(ref_depth+depth), dum_zero)
    return inv_depth

def vis_depth_np(depth, ref_depth=10):        ## why normalize_result in bts_main.py convert it to numpy?
    dum_zero = np.zeros_like(depth)
    inv_depth = np.where(depth>0, ref_depth/(ref_depth+depth), dum_zero)
    return inv_depth

'''from bts/bts_utils.py'''
def overlay_dep_on_rgb(depth, img, path=None, name=None, overlay=True):
    ''' both 3-dim: C*H*W. not including batch
    dep: output from vis_depth
    both are torch.tensor, between 0~1
    '''
    dep_np = depth.cpu().numpy().transpose(1,2,0)
    img_np= img.permute(1,2,0).cpu().numpy()
    return overlay_dep_on_rgb_np(dep_np, img_np, path, name, overlay)

'''from bts/bts_utils.py'''
def overlay_dep_on_rgb_np(dep_np, img_np, path=None, name=None, overlay=True):
    ''' both 3-dim: H*W*C.
    both are np.array, between 0~1
    '''
    dep_255 = dep_np*255

    dep_mask = np.zeros_like(dep_255).astype(np.uint8)
    dep_mask[dep_255 > 0] = 255

    r, g, b = rgbmap(dep_255, mask_zeros=True)              # return int, shape the same as input

    dep_in_color = np.dstack((b, g, r))
    dep_in_color = dep_in_color.astype(np.uint8)

    img_np= img_np * 255 # H*W*C
    img_np = img_np.astype(np.uint8)
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

    return img_np