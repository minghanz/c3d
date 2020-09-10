import numpy as np
from collections import Counter
try:
    import torch
except:
    import warnings
    warnings.warn("need to install PyTorch to use torch_mode. ")

import copy
import torchsnooper

class InExtr:
    def __init__(self):
        self.width = None
        self.height = None
        self.P_cam_li = None
        self.dist_coef = None
        self.K = None
    
    # def to_torch(self):
    #     new_inex = InExtr()
    #     new_inex.width = self.width
    #     new_inex.height = self.height
    #     if self.K is None:
    #         new_inex.K = None
    #     elif isinstance(self.K, np.ndarray):
    #         new_inex.K = torch.from_numpy(self.K)
    #     elif isinstance(self.K, torch.Tensor):
    #         new_inex.K = self.K.clone()

    def get_K(self, new_width=None, new_height=None, align_corner=None):
        assert self.K is not None
        torch_mode = not isinstance(self.K, np.ndarray)
        if new_width is None and new_height is None and align_corner is None:
            return self.K
        else:
            assert new_width is not None and new_height is not None and align_corner is not None
            K = scale_K(self.K, torch_mode, old_height=self.height, old_width=self.width, new_height=new_height, new_width=new_width, align_corner=align_corner)

            return K

    def scale(self, cam_op):
        new_width = cam_op.new_width
        new_height = cam_op.new_height
        align_corner = cam_op.align_corner
        if cam_op.scale is not None and cam_op.scale != 0 and new_width is not None and new_height is not None:
            assert new_width == int(self.width * cam_op.scale)
            assert new_height == int(self.height * cam_op.scale)

        K_scaled = self.get_K(new_width=new_width, new_height=new_height, align_corner=align_corner)

        new_inex = InExtr()
        ### renewed
        new_inex.width = new_width
        new_inex.height = new_height
        new_inex.K = K_scaled
        ### copied
        new_inex.P_cam_li = copy.copy(self.P_cam_li)
        new_inex.dist_coef = copy.copy(self.dist_coef)
        return new_inex

    def rotate(self):
        raise NotImplementedError

    def crop(self, cam_op):
        assert self.K is not None
        torch_mode = not isinstance(self.K, np.ndarray)
        
        new_width = cam_op.x_size
        new_height = cam_op.y_size
        K_cropped = crop_K(self.K, cam_op.x_start, cam_op.y_start, torch_mode)

        new_inex = InExtr()
        ### renewed
        new_inex.width = new_width
        new_inex.height = new_height
        new_inex.K = K_cropped
        ### copied
        new_inex.P_cam_li = copy.copy(self.P_cam_li)
        new_inex.dist_coef = copy.copy(self.dist_coef)
        return new_inex

    def flip(self, cam_op, fix_P_cam=True):
        ### TODO: handle when fix_P_cam is False
        ### fix_P_cam=True means that P_cam_li is left unchanged, as the lidar_pts is in camera coordinate and will flip sign
        assert self.K is not None
        torch_mode = not isinstance(self.K, np.ndarray)

        K_flipped = flip_K(self.K, self.width, self.height, torch_mode, hori=cam_op.hori, vert=cam_op.vert)

        new_inex = InExtr()
        ### renewed
        new_inex.width = self.width
        new_inex.height = self.height
        new_inex.K = K_flipped
        ### copied
        new_inex.P_cam_li = copy.copy(self.P_cam_li)
        new_inex.dist_coef = copy.copy(self.dist_coef)

        return new_inex
    
    # @torchsnooper.snoop()
    def lidar_to_depth(self, lidar_pts, timer=None):
        assert self.K is not None
        torch_mode = not isinstance(self.K, np.ndarray)

        depth_img = lidar_to_depth(lidar_pts, self.P_cam_li, K_unit=None, im_shape=(self.height, self.width), K_ready=self.K, torch_mode=torch_mode, timer=timer)    # H*W

        return depth_img

class InExtrKunit:
    def __init__(self):
        self.width = None
        self.height = None
        self.K_unit = None
        self.P_cam_li = None
        self.dist_coef = None
        
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

def flip_K(K, width, height, torch_mode, hori=True, vert=False):
    if torch_mode:
        K_new = K.clone().detach()
    else:
        K_new = K.copy()
    if len(K.shape) == 3:
        if hori:
            K_new[:, 0, 2] = width - 1 - K_new[:, 0, 2]
        if vert:
            K_new[:, 1, 2] = height - 1 - K_new[:, 1, 2]
    else:
        assert len(K.shape)==2
        if hori:
            K_new[0, 2] = width - 1 - K_new[0, 2]
        if vert:
            K_new[1, 2] = height - 1 - K_new[1, 2]
    return K_new

'''from bts/bts_pre_intr.py'''
def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

'''from bts/bts_pre_intr.py'''
def lidar_to_depth(velo, extr_cam_li, K_unit, im_shape, K_ready=None, torch_mode=False, align_corner=False, dep_dim_pre_proj=2, timer=None):
    """
    if torch_mode==False, the inputs are np.array, non-batched, velo is N*4, extr_cam_li: 4x4, intr_K: 3x3
    if torch_mode==True, the inputs are torch.tensor, can be batched (3-dim, first dim batch)
    """
    if timer is not None:
        timer.log("lidar_to_proj_pts", 2, True)
    velo_proj = lidar_to_proj_pts(velo, extr_cam_li, K_unit, im_shape, K_ready, torch_mode, align_corner, dep_dim_pre_proj, timer=timer)

    if timer is not None:
        timer.log("projected_pts_to_img", 2, True)
    depth_img = projected_pts_to_img(velo_proj, im_shape, torch_mode, timer=timer)

    return depth_img

def lidar_to_proj_pts(velo, extr_cam_li, K_unit, im_shape, K_ready=None, torch_mode=False, align_corner=False, dep_dim_pre_proj=2, timer=None):

    if timer is not None:
        timer.log("prepare K", 3, True)
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

    if timer is not None:
        timer.log("lidar to cam frame", 3, True)
    ## transform to camera frame
    if torch_mode:
        if velo.shape[-1] == 3:
            ## not homogeneous coord
            R_cam_li = extr_cam_li[:3, :3]
            t_cam_li = extr_cam_li[:3, 3:4]
            velo_in_cam_frame = R_cam_li.matmul(velo.transpose(-1, -2)) + t_cam_li # ...*3*N
        else:
            ## homogeneous coord
            velo_in_cam_frame = extr_cam_li.matmul(velo.transpose(-1, -2)) # ..*4*N
    else:
        velo_in_cam_frame = np.dot(extr_cam_li, velo.T) # 4*N

    velo_in_cam_frame = velo_in_cam_frame[:3, :] # 3*N, xyz
    velo_in_cam_frame = velo_in_cam_frame[:, velo_in_cam_frame[dep_dim_pre_proj, :] > 0]  # keep forward points

    if timer is not None:
        timer.log("project to image", 3, True)
    ## project to image
    if torch_mode:
        velo_proj = intr_K.matmul(velo_in_cam_frame)
        velo_proj[:2, :] = velo_proj[:2, :] / velo_proj[[2], :]
        velo_proj[:2, :] = velo_proj[:2, :].round()    # -1 is for kitti dataset aligning with its matlab script, now in K_mat2py
    else:
        velo_proj = np.dot(intr_K, velo_in_cam_frame)
        velo_proj[:2, :] = velo_proj[:2, :] / velo_proj[[2], :]
        velo_proj[:2, :] = np.round(velo_proj[:2, :])    # -1 is for kitti dataset aligning with its matlab script, now in K_mat2py

    ## crop out-of-view points
    valid_idx = ( velo_proj[0, :] > -0.5 ) & ( velo_proj[0, :] < im_shape[1]-0.5 ) & ( velo_proj[1, :] > -0.5 ) & ( velo_proj[1, :] < im_shape[0]-0.5 )
    velo_proj = velo_proj[:, valid_idx]
    
    return velo_proj

def projected_pts_to_img(velo_proj, im_shape, torch_mode, timer=None):

    if timer is not None:
        timer.log("crop points", 3, True)
    ## crop out-of-view points
    valid_idx = ( velo_proj[0, :] > -0.5 ) & ( velo_proj[0, :] < im_shape[1]-0.5 ) & ( velo_proj[1, :] > -0.5 ) & ( velo_proj[1, :] < im_shape[0]-0.5 )
    velo_proj = velo_proj[:, valid_idx]

    if timer is not None:
        timer.log("compose depth image", 3, True)
    ## compose depth image
    if torch_mode:
        depth_img = torch.zeros((im_shape[:2]))
        depth_img[velo_proj[1, :].to(dtype=int), velo_proj[0, :].to(dtype=int)] = velo_proj[2, :]
    else:
        depth_img = np.zeros((im_shape[:2]))
        depth_img[velo_proj[1, :].astype(np.int), velo_proj[0, :].astype(np.int)] = velo_proj[2, :]

    if timer is not None:
        timer.log("find the duplicate", 3, True)

    ## find the duplicate points and choose the closest depth
    velo_proj_lin = sub2ind(depth_img.shape, velo_proj[1, :], velo_proj[0, :])
    dupe_proj_lin = [item for item, count in Counter(velo_proj_lin).items() if count > 1]

    if timer is not None:
        timer.log("process the duplicate %d"%len(dupe_proj_lin), 3, True)

    for dd in dupe_proj_lin:
        if torch_mode:
            pts = torch.where(velo_proj_lin == dd)[0]
        else:
            pts = np.where(velo_proj_lin == dd)[0]  ### np.where() [actually np.asarray(condition).nonzero()] returns a tuple. [0] takes the array of the first dim.
        x_loc = int(velo_proj[0, pts[0]])
        y_loc = int(velo_proj[1, pts[0]])
        depth_img[y_loc, x_loc] = velo_proj[2, pts].min()

    if timer is not None:
        timer.log("process the negatives", 3, True)
    # depth_img[depth_img < 0] = 0
    if torch_mode:
        depth_img = torch.clamp(depth_img, min=0)
    else:
        depth_img = np.clip(depth_img, a_min=0, a_max=None)

    return depth_img