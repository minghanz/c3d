import torch
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
# import skimage.transform
import torchvision.transforms ## no need to call this since it calls PIL internally, but here the default is BILINEAR while PIL default is BICUBIC: 
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize and https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#resize
from collections import namedtuple

from ..utils_general.calib import scale_K, crop_and_scale_K, lidar_to_depth
# class CamCrop:
#     def __init__(self, x_start, y_start, x_size, y_size):
#         self.x_start = x_start
#         self.y_start = y_start
#         self.x_start = x_start
#         self.x_start = x_start
        
# class CamScale:
#     def __init__(self, scale=None, new_width=None, new_height=None, align_corner=False):
#         assert scale is None or (new_width is None and new_height is None)
#         self.lock_ratio = scale is not None
#         if self.lock_ratio:
#             self.scale = scale
#             self.new_width = None
#             self.new_height = None
#         else:
#             self.scale = None
#             self.new_width = new_width
#             self.new_height = new_height
#         self.align_corner = align_corner

CamScale = namedtuple('CamScale', ['scale', 'new_width', 'new_height', 'align_corner'])
CamCrop = namedtuple('CamCrop', ['x_start', 'y_start', 'x_size', 'y_size'])
CamRotate = namedtuple('CamRotate', ['angle_deg', 'nearest'])
CamFlip = namedtuple('CamFlip', ['hori', 'vert'])

CamScale.__new__.__defaults__ = (None,) * len(CamScale._fields)


'''This is to get a cam_op from a batched version of it.'''
def extract_single_op(cam_op, idx=0):
    op_type = type(cam_op)
    cam_op_single = op_type(*(cam_op_item[idx] for cam_op_item in cam_op))
    return cam_op_single


'''from bts/c3d_loss.py'''
def gen_uv_grid(width, height, torch_mode):
    """
    return: uv_coords(2*H*W)
    """
    meshgrid = np.meshgrid(range(int(width)), range(int(height)), indexing='xy')
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
    if len(uv_grid.shape) == 4:
        batch_size = uv_grid.shape[0]
        if torch_mode:
            uv_flat = uv_grid.reshape(batch_size, 2, -1) # B*2*N
            dummy_ones = torch.ones((batch_size, 1, uv_flat.shape[2]), dtype=uv_flat.dtype, device=uv_flat.device) # B*1*N
            uv1_flat = torch.cat((uv_flat, dummy_ones), dim=1) # B*3*N
            xy1_flat = torch.matmul(inv_K, uv1_flat)
        else:
            uv_flat = uv_grid.reshape(batch_size, 2, -1) # B*2*N
            dummy_ones = np.ones((batch_size, 1, uv_flat.shape[2]), dtype=np.float32)
            uv1_flat = np.concatenate((uv_flat, dummy_ones), axis=1) # B*3*N
            xy1_flat = np.matmul(inv_K, uv1_flat)

    elif len(uv_grid.shape) == 3:
        if torch_mode:
            uv_flat = uv_grid.reshape(2, -1) # 2*N
            dummy_ones = torch.ones((1, uv_flat.shape[1]), dtype=uv_flat.dtype, device=uv_flat.device) # 1*N
            uv1_flat = torch.cat((uv_flat, dummy_ones), dim=0) # 3*N
            xy1_flat = torch.matmul(inv_K, uv1_flat)
        else:
            uv_flat = uv_grid.reshape(2, -1) # 2*N
            dummy_ones = np.ones((1, uv_flat.shape[1]), dtype=np.float32)
            uv1_flat = np.concatenate((uv_flat, dummy_ones), axis=0) # 3*N
            xy1_flat = np.matmul(inv_K, uv1_flat)
    else:
        raise ValueError('Dimension of uv_grid not compatible', uv_grid.shape)

    return uv1_flat, xy1_flat

'''from bts/c3d_loss.py'''
def set_from_intr(width, height, K_unit, batch_size, device=None, align_corner=False):
    '''
    K_unit is nparray here.
    '''
    to_torch = True
    uv_grid = gen_uv_grid(width, height, to_torch) # 2*H*W

    K = K_unit.copy()
    # effect_w = float(width - 1 if align_corner else width)
    # effect_h = float(height - 1 if align_corner else height)
    # scale_w, scale_h = scale_from_size(new_width=width, new_height=height, align_corner=align_corner)
    K = scale_K(K, new_width=width, new_height=height, align_corner=align_corner, torch_mode=False)

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

    return uvb_grid, xy1_grid, int(width), int(height), K

def np2Image(img_np, raw_float):
    if raw_float:
        assert img_np.dtype == np.float32
        img_255 = img_np
    else:
        assert img_np.min() >= 0, 'img min should be at least 0'
        if img_np.max() <= 1:
            img_255 = np.round(img_np * 255).astype(np.uint8)
        elif img_np.max() > 255:
            raise ValueError("image max > 255, cannot process")
        else:
            img_255 = img_np.astype(np.uint8)

    if img_255.shape[0] <= 3:
        img_255 = img_255.transpose(1,2,0)

    if raw_float:
        Imode = 'F'
    if len(img_255.shape) < 3:
        Imode = 'L'
    elif img_255.shape[2] == 1:
        Imode = 'L'
    elif img_255.shape[2] == 3:
        Imode = 'RGB'
    else:
        raise ValueError("image shape unrecognized:", img_255.shape)

    img = Image.fromarray(img_255, mode=Imode)
    return img

def scale_image(img, new_width, new_height, torch_mode, nearest, raw_float, align_corner=False):
    ## TODO: we haven't specify align_corner behavior here. 
    ## According to https://medium.com/@elagwoog/you-might-have-misundertood-the-meaning-of-align-corners-c681d0e38300, 
    ## PIL's resizing is equivalent to align_corner=False

    ## when torch_mode==True, can work with batch or not
    ## when torch_mode==False, work with a single PIL.Image object

    new_width = int(new_width)
    new_height = int(new_height)
    
    if torch_mode:
        from_pil = False
        from_np = False
        if isinstance(img, np.ndarray):
            from_np = True
            img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        if isinstance(img, Image.Image):
            from_pil = True
            img = np.array(img)
            img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        if nearest:
            if len(img.shape) == 4:
                # resized_img = scale_depth_torch_through_pil_batch(img, new_width, new_height, nearest=True) # 1
                resized_img = F.interpolate(img, size=(new_height, new_width), scale_factor=None, mode='nearest', align_corners=align_corner)
            else:
                # resized_img = scale_depth_torch_through_pil(img, new_width, new_height, nearest=True, device=img.device)
                resized_img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), scale_factor=None, mode='nearest', align_corners=align_corner).squeeze(0)
        else:
            if len(img.shape) == 4:
                # resized_img = scale_depth_torch_through_pil_batch(img, new_width, new_height, nearest=False) # 1
                resized_img = F.interpolate(img, size=(new_height, new_width), scale_factor=None, mode='bilinear', align_corners=align_corner)
            else:
                # resized_img = scale_depth_torch_through_pil(img, new_width, new_height, nearest=False, device=img.device)
                resized_img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), scale_factor=None, mode='bilinear', align_corners=align_corner).squeeze(0)
        if from_np:
            resized_img = resized_img.numpy().transpose(1,2,0)
        if from_pil:        ## do not return to PIL.Image.Image
            resized_img = resized_img.numpy().transpose(1,2,0)

    else:
        if isinstance(img, np.ndarray):
            img = np2Image(img, raw_float)
        if nearest: 
            resized_img = img.resize( (new_width, new_height), Image.NEAREST)
        else:
            resized_img = img.resize( (new_width, new_height), Image.BILINEAR) ## using BILINEAR is to be consistent to torchvision.transform.resize default

        # resized_img = img.resize( (new_width, new_height), Image.NEAREST )
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize

        # resized_img = skimage.transform.resize(img, (new_height, new_width), order=0, preserve_range=True, mode='constant', anti_aliasing=False)
        # https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
    return resized_img

def scale_depth_torch_through_pil_batch(img, new_width, new_height, nearest):
    single_list = []
    for ib in range(img.shape[0]):
        torch_img_resized = scale_depth_torch_through_pil(img[ib], new_width, new_height, nearest)
        single_list.append(torch_img_resized)
    resized_img = torch.stack(single_list, dim=0).to(device=img.device)
    return resized_img

def scale_depth_torch_through_pil(img, new_width, new_height, nearest, device=None):
    pil_img = torchvision.transforms.ToPILImage(img)
    if nearest:
        pil_img_resized = torchvision.transforms.functional.resize(pil_img, (new_height, new_width), Image.NEAREST)
    else:
        pil_img_resized = torchvision.transforms.functional.resize(pil_img, (new_height, new_width), Image.BILINEAR)
    ## do not use torchvision.transforms.functional.totensor because it changes scale:
    ## https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
    py_img_resized = np.array(pil_img_resized)
    py_img_resized = py_img_resized.transpose(2, 0, 1)
    torch_img_resized = torch.from_numpy(py_img_resized)
    if device is not None:
        torch_img_resized = torch_img_resized.to(device)
    return torch_img_resized

def scale_depth_from_lidar(velo, extr_cam_li, K_ori, new_width, new_height, old_width=2, old_height=2, align_corner=False):
    new_width = int(new_width)
    new_height = int(new_height)
    # scale_w, scale_h = scale_from_size(old_width, old_height, new_width, new_height, align_corner)
    K = scale_K(K_ori, old_width=old_width, old_height=old_height, new_width=new_width, new_height=new_height, align_corner=align_corner, torch_mode=False)

    depth_gt_scaled = lidar_to_depth(velo, extr_cam_li, im_shape=(new_height, new_width), K_ready=K, K_unit=None)
    return depth_gt_scaled

def crop_and_scale_depth_from_lidar(velo, extr_cam_li, K_ori, xy_crop, new_width, new_height, align_corner=False):
    new_width = int(new_width)
    new_height = int(new_height)
    scaled_cropped_K = crop_and_scale_K(K_ori, xy_crop, scale=None, new_width=new_width, new_height=new_height, torch_mode=False)
    
    depth_gt_scaled = lidar_to_depth(velo, extr_cam_li, im_shape=(new_height, new_width), K_ready=scaled_cropped_K, K_unit=None)
    return depth_gt_scaled
