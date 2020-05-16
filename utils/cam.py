import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import Counter
from PIL import Image
# import skimage.transform
import torchvision.transforms ## no need to call this since it calls PIL internally, but here the default is BILINEAR while PIL default is BICUBIC: 
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize and https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#resize
from collections import namedtuple

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

def extract_single_op(cam_op, idx=0):
    op_type = type(cam_op)
    cam_op_single = op_type(*(cam_op_item[idx] for cam_op_item in cam_op))
    return cam_op_single

class InExtr:
    def __init__(self):
        self.width = None
        self.height = None
        self.K_unit = None
        self.P_cam_li = None

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

def K_mat2py(K):
    '''
    Matlab index start from 1, python start from 0.
    The only thing needed is to offset cx, cy by 1.
    '''
    K_new = K.copy()
    K_new[0, 2] -= 1
    K_new[1, 2] -= 1
    return K_new

def scale_from_size(old_width=2, old_height=2, new_width=2, new_height=2, align_corner=False):
    '''
    A unit K is equivalent to the K of a 2*2 image
    '''
    scale_w = (new_width - 1) / (old_width - 1) if align_corner else new_width / old_width
    scale_h = (new_height - 1) / (old_height - 1) if align_corner else new_height / old_height
    return scale_w, scale_h

def scale_K(K, torch_mode, scale_w=None, scale_h=None, old_width=2, old_height=2, new_width=2, new_height=2, align_corner=False):
    '''
    generate new intrinsic matrix from original K and scale
    https://github.com/pytorch/pytorch/blob/5ac2593d4f2611480a5a9872e08024a665ae3c26/aten/src/ATen/native/cuda/UpSample.cuh
    see area_pixel_compute_source_index function
    '''
    if scale_w is None:
        scale_w, scale_h = scale_from_size(old_width, old_height, new_width, new_height, align_corner)

    if torch_mode:
        K_new = K.clone().detach()
    else:
        K_new = K.copy() #np.identity(3).astype(np.float32)
    if len(K.shape) == 3:
        K_new[:, 0, 0] = K[:, 0, 0] * scale_w
        K_new[:, 1, 1] = K[:, 1, 1] * scale_h
        if align_corner:
            K_new[:, 0, 2] = scale_w * K[:, 0, 2]
            K_new[:, 1, 2] = scale_h * K[:, 1, 2]
        else:
            K_new[:, 0, 2] = scale_w * (K[:, 0, 2] + 0.5) - 0.5
            K_new[:, 1, 2] = scale_h * (K[:, 1, 2] + 0.5) - 0.5
    else:
        assert len(K.shape)==2
        K_new[0, 0] = K[0, 0] * scale_w
        K_new[1, 1] = K[1, 1] * scale_h
        if align_corner:
            K_new[0, 2] = scale_w * K[0, 2]
            K_new[1, 2] = scale_h * K[1, 2]
        else:
            K_new[0, 2] = scale_w * (K[0, 2] + 0.5) - 0.5
            K_new[1, 2] = scale_h * (K[1, 2] + 0.5) - 0.5
    return K_new

def crop_K(K, w_start, h_start, torch_mode):
    '''K is np array
    '''
    if torch_mode:
        K_new = K.clone().detach()
    else:
        K_new = K.copy()
    if len(K.shape) == 3:
        K_new[:, 0, 2] -= w_start
        K_new[:, 1, 2] -= h_start
    else:
        assert len(K.shape)==2
        K_new[0, 2] -= w_start
        K_new[1, 2] -= h_start
    return K_new

def crop_and_scale_K(K, xy_crop, torch_mode, scale=None, new_width=None, new_height=None, align_corner=False):
    '''Find the intrinsic matrix equivalent to an image cropped and then scaled
    '''
    w_start = int(xy_crop[0])
    h_start = int(xy_crop[1])
    old_width = int(xy_crop[2])
    old_height = int(xy_crop[3])

    cropped_K = crop_K(K, w_start, h_start, torch_mode=torch_mode )

    if new_width is None:
        new_width = old_width * scale
        new_height = old_height * scale
    # scale_w_crop, scale_h_crop = scale_from_size(old_width=old_width, old_height=old_height, new_width=new_width, new_height=new_height, align_corner=align_corner)
    scaled_cropped_K = scale_K(cropped_K, old_width=old_width, old_height=old_height, new_width=new_width, new_height=new_height, torch_mode=torch_mode, align_corner=align_corner )

    return scaled_cropped_K

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

'''from bts/bts_pre_intr.py'''
def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

'''from bts/bts_pre_intr.py'''
def lidar_to_depth(velo, extr_cam_li, K_unit, im_shape, K_ready=None, torch_mode=False, align_corner=False, dep_dim_pre_proj=2):
    """
    if torch_mode==False, the inputs are np.array, non-batched, velo is N*4, extr_cam_li: 4x4, intr_K: 3x3
    if torch_mode==True, the inputs are torch.tensor, can be batched (3-dim, first dim batch)
    """
    velo_proj = lidar_to_proj_pts(velo, extr_cam_li, K_unit, im_shape, K_ready, torch_mode, align_corner, dep_dim_pre_proj)

    depth_img = projected_pts_to_img(velo_proj, im_shape, torch_mode)

    return depth_img

def lidar_to_proj_pts(velo, extr_cam_li, K_unit, im_shape, K_ready=None, torch_mode=False, align_corner=False, dep_dim_pre_proj=2):
    assert K_ready is None or K_unit is None
    ## recover K
    if K_ready is None:
        if torch_mode:
            intr_K = K_unit.clone().detach()
            # scale_w, scale_h = scale_from_size(new_width=im_shape[1], new_height=im_shape[0], align_corner=align_corner)
            intr_K = scale_K(intr_K, new_width=im_shape[1], new_height=im_shape[0], torch_mode=True, align_corner=align_corner)
        else:
            intr_K = K_unit.copy()
            # effect_w = float(im_shape[1] - 1 if align_corner else im_shape[1])
            # effect_h = float(im_shape[0] - 1 if align_corner else im_shape[0])
            # scale_w, scale_h = scale_from_size(new_width=im_shape[1], new_height=im_shape[0], align_corner=align_corner)
            intr_K = scale_K(intr_K, new_width=im_shape[1], new_height=im_shape[0], torch_mode=False, align_corner=align_corner)
    else:
        intr_K = K_ready

    ## transform to camera frame
    if torch_mode:
        if velo.shape[-1] == 3:
            ## not homogeneous coord
            R_cam_li = extr_cam_li[:3, :3]
            t_cam_li = extr_cam_li[:3, 3:4]
            velo_in_cam_frame = torch.matmul(R_cam_li, velo.transpose(-1, -2)) + t_cam_li # ...*3*N
        else:
            ## homogeneous coord
            velo_in_cam_frame = torch.matmul(extr_cam_li, velo.transpose(-1, -2)) # ..*4*N
    else:
        velo_in_cam_frame = np.dot(extr_cam_li, velo.T) # 4*N

    velo_in_cam_frame = velo_in_cam_frame[:3, :] # 3*N, xyz
    velo_in_cam_frame = velo_in_cam_frame[:, velo_in_cam_frame[dep_dim_pre_proj, :] > 0]  # keep forward points

    ## project to image
    if torch_mode:
        velo_proj = torch.matmul(intr_K, velo_in_cam_frame)
        velo_proj[:2, :] = velo_proj[:2, :] / velo_proj[[2], :]
        velo_proj[:2, :] = torch.round(velo_proj[:2, :])    # -1 is for kitti dataset aligning with its matlab script, now in K_mat2py
    else:
        velo_proj = np.dot(intr_K, velo_in_cam_frame)
        velo_proj[:2, :] = velo_proj[:2, :] / velo_proj[[2], :]
        velo_proj[:2, :] = np.round(velo_proj[:2, :])    # -1 is for kitti dataset aligning with its matlab script, now in K_mat2py

    ## crop out-of-view points
    valid_idx = ( velo_proj[0, :] > -0.5 ) & ( velo_proj[0, :] < im_shape[1]-0.5 ) & ( velo_proj[1, :] > -0.5 ) & ( velo_proj[1, :] < im_shape[0]-0.5 )
    velo_proj = velo_proj[:, valid_idx]
    
    return velo_proj

def projected_pts_to_img(velo_proj, im_shape, torch_mode):

    ## crop out-of-view points
    valid_idx = ( velo_proj[0, :] > -0.5 ) & ( velo_proj[0, :] < im_shape[1]-0.5 ) & ( velo_proj[1, :] > -0.5 ) & ( velo_proj[1, :] < im_shape[0]-0.5 )
    velo_proj = velo_proj[:, valid_idx]

    ## compose depth image
    if torch_mode:
        depth_img = np.zeros((im_shape[:2]))
        depth_img[velo_proj[1, :].to(dtype=int), velo_proj[0, :].to(dtype=int)] = velo_proj[2, :]
    else:
        depth_img = np.zeros((im_shape[:2]))
        depth_img[velo_proj[1, :].astype(np.int), velo_proj[0, :].astype(np.int)] = velo_proj[2, :]

    ## find the duplicate points and choose the closest depth
    velo_proj_lin = sub2ind(depth_img.shape, velo_proj[1, :], velo_proj[0, :])
    dupe_proj_lin = [item for item, count in Counter(velo_proj_lin).items() if count > 1]
    for dd in dupe_proj_lin:
        pts = np.where(velo_proj_lin == dd)[0]
        x_loc = int(velo_proj[0, pts[0]])
        y_loc = int(velo_proj[1, pts[0]])
        depth_img[y_loc, x_loc] = velo_proj[2, pts].min()
    depth_img[depth_img < 0] = 0
    return depth_img