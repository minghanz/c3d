import numpy as np
from collections import Counter

class InExtr:
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
            velo_in_cam_frame = R_cam_li.matmul(velo.transpose(-1, -2)) + t_cam_li # ...*3*N
        else:
            ## homogeneous coord
            velo_in_cam_frame = extr_cam_li.matmul(velo.transpose(-1, -2)) # ..*4*N
    else:
        velo_in_cam_frame = np.dot(extr_cam_li, velo.T) # 4*N

    velo_in_cam_frame = velo_in_cam_frame[:3, :] # 3*N, xyz
    velo_in_cam_frame = velo_in_cam_frame[:, velo_in_cam_frame[dep_dim_pre_proj, :] > 0]  # keep forward points

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