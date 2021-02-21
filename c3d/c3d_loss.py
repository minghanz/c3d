import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
import os
import sys
# script_path = os.path.dirname(__file__)
# sys.path.append(os.path.join(script_path, '../../pytorch-unet'))
# from geometry import rgb_to_hsv
from .utils_general.color import rgb_to_hsv, hsv_to_rgb

# sys.path.append(os.path.join(script_path, '../../monodepth2'))
# from cvo_utils import *
from .cvo_funcs import *
from .utils.geometry import *
from .utils.pc3d import *
from .utils.cam_proj import *

import argparse
from .utils_general.argparse_f import init_argparser_f
from .utils_general.vis import overlay_dep_on_rgb, dep_img_bw

from .utils_general.pcl_funcs import pcl_from_flat_xyz, pcl_from_grid_xy1_dep, pcl_write

from .utils_general.timing import Timing

import torchsnooper
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import io
from PIL import Image

# import logging
class PCL_C3D_Flat:
    def __init__(self):
        self.uvb = None
        self.nb = []
        self.feature = EasyDict()

    def to_pcd(self):
        xyz = self.feature.xyz
        rgb = hsv_to_rgb(self.feature.hsv, flat=True)
        uvb = self.uvb  # 1*3*N
        batch_size = len(self.nb)
        clouds = []
        for ib in range(batch_size):
            ib_mask = uvb[0,2] == ib
            xyz_i = xyz[0,:,ib_mask]
            rgb_i = rgb[0,:,ib_mask]

            cloud_i = pcl_from_flat_xyz(xyz_i, rgb_i)
            clouds.append(cloud_i)

        return clouds

class PCL_C3D_Grid:
    def __init__(self):
        self.mask = None
        self.feature = EasyDict()

    def to_pcd(self):
        xyz = self.feature.xyz
        rgb = hsv_to_rgb(self.feature.hsv)
        xyz = xyz.permute(0,2,3,1)
        rgb = rgb.permute(0,2,3,1)  # B*H*W*C
        mask = self.mask

        batch_size = xyz.shape[0]
        clouds = []
        for ib in range(batch_size):
            mask_i = mask[ib,0]
            xyz_i = xyz[ib][mask_i]
            rgb_i = rgb[ib][mask_i]

            cloud_i = pcl_from_flat_xyz(xyz_i.transpose(0,1), rgb_i.transpose(0,1))
            clouds.append(cloud_i)

        return clouds


class PCL_C3D:
    def __init__(self):
        self.flat = PCL_C3D_Flat()
        self.grid = PCL_C3D_Grid()

# def init_pc3d():
#     pcl_c3d = EasyDict()
#     pcl_c3d.frame_id = None

#     pcl_c3d.flat = EasyDict()
#     pcl_c3d.flat.uvb = None
#     pcl_c3d.flat.feature = EasyDict()

#     pcl_c3d.grid = EasyDict()
#     pcl_c3d.grid.mask = None
#     pcl_c3d.grid.feature = EasyDict() 

#     pcl_c3d.flat_tr = EasyDict()
#     pcl_c3d.flat_tr.uvb = None
#     pcl_c3d.flat_tr.feature = EasyDict()
#     pcl_c3d.flat_tr.frame_id = None

#     return pcl_c3d

def load_simp_pc3d(pcl_c3d, mask_grid, uvb_flat, feat_grid, feat_flat):
    """
    This function loads the PCL_C3D object with dense features. 
    """
    batch_size = mask_grid.shape[0]

    ## grid features
    pcl_c3d.grid.mask = mask_grid
    for feat in feat_grid:
        pcl_c3d.grid.feature[feat] = feat_grid[feat]   # B*C*H*W

    ## flat features
    pcl_c3d.flat.uvb = []
    for feat in feat_flat:
        pcl_c3d.flat.feature[feat] = []

    #### masking out invalid points
    mask_flat = mask_grid.reshape(batch_size, 1, -1)
    for ib in range(batch_size):
        mask_vec = mask_flat[ib, 0]
        pcl_c3d.flat.nb.append(int(mask_vec.sum()))
        pcl_c3d.flat.uvb.append(uvb_flat[[ib]][:,:, mask_vec])
        for feat in feat_flat:
            pcl_c3d.flat.feature[feat].append(feat_flat[feat][[ib]][:,:, mask_vec])      # 1*C*N
    
    #### concatenation to remove the batch dimension for more efficient processing later on
    pcl_c3d.flat.uvb = torch.cat(pcl_c3d.flat.uvb, dim=2)
    for feat in feat_flat:
        pcl_c3d.flat.feature[feat] = torch.cat(pcl_c3d.flat.feature[feat], dim=2)

    return pcl_c3d


# @torchsnooper.snoop()
def load_pc3d(pcl_c3d, depth_grid, mask_grid, xy1_grid, uvb_flat, K_cur, feat_comm_grid, feat_comm_flat, sparse, use_normal, sparse_nml_opts=None, dense_nml_op=None, return_stat=False):
    assert not (sparse_nml_opts is None and dense_nml_op is None)
    """
    This function loads the PCL_C3D object from ingredients. 
    0. prepare dense features in grid (B*C*H*W) and flat (B*C*N) form

    1. load_simp_pc3d() load dense features in grid (B*C*H*W) and flat (1*C*N) form to pcl_c3d object. 
    Here in the flat form, items in the batch are concatenated. The index in the batch is saved in uvb_flat, which is also 1*3*N, where the 3 corresponds to u, v, idx_in_batch. 
    In the flat form, the points are already filtered by mask_grid, i.e. N <= B*H*W

    2. calculate sparse features (e.g. normal for sparse point-clouds), and convert it back to grid form using grid_from_concat_flat_func(). 
    This is different from step 0 as dense features usually can be conveniently calculated parallelly, while sparse feature calculation requires special care. We calculate them using operations written in cvo_ops. 
    
    sparse is a bool
    """
    feat_flat = EasyDict()
    feat_grid = EasyDict()

    for feat in feat_comm_flat:
        feat_flat[feat] = feat_comm_flat[feat]
    for feat in feat_comm_grid:
        feat_grid[feat] = feat_comm_grid[feat]

    ## xyz
    xyz_grid = xy1_grid * depth_grid

    batch_size = depth_grid.shape[0]
    xyz_flat = xyz_grid.reshape(batch_size, 3, -1)

    feat_flat['xyz'] = xyz_flat
    feat_grid['xyz'] = xyz_grid

    ## normal for dense
    if use_normal>0 and not sparse:
        normal_grid = dense_nml_op(depth_grid, K_cur)
        nres_grid = res_normal_dense(xyz_grid, normal_grid, K_cur)
        feat_grid['normal'] = normal_grid
        feat_grid['nres'] = nres_grid
        feat_flat['normal'] = normal_grid.reshape(batch_size, 3, -1)
        feat_flat['nres'] = nres_grid.reshape(batch_size, 1, -1)     
    
    ## load into pc3d object
    pcl_c3d = load_simp_pc3d(pcl_c3d, mask_grid, uvb_flat, feat_grid, feat_flat)

    ## normal for sparse
    if use_normal>0 and sparse:
        if return_stat:
            normal_flat, nres_flat, dist_stat_flat = calc_normal(pcl_c3d.flat.uvb, xyz_grid, mask_grid, sparse_nml_opts.normal_nrange, sparse_nml_opts.ignore_ib, sparse_nml_opts.min_dist_2, return_stat=return_stat)
        else:
            normal_flat, nres_flat = calc_normal(pcl_c3d.flat.uvb, xyz_grid, mask_grid, sparse_nml_opts.normal_nrange, sparse_nml_opts.ignore_ib, sparse_nml_opts.min_dist_2, return_stat=return_stat)

        ## TODO: How to deal with points with no normal?
        uvb_split = pcl_c3d.flat.uvb.to(dtype=torch.long).squeeze(0).transpose(0,1).split(1,dim=1) # a tuple of 3 elements of tensor N*1, only long/byte/bool tensors can be used as indices
        grid_xyz_shape = xyz_grid.shape
        normal_grid = grid_from_concat_flat_func(uvb_split, normal_flat, grid_xyz_shape)
        nres_grid = grid_from_concat_flat_func(uvb_split, nres_flat, grid_xyz_shape)
        if return_stat:
            dist_stat_grid = grid_from_concat_flat_func(uvb_split, dist_stat_flat, grid_xyz_shape)
            pcl_c3d.flat.feature['dist_stat'] = dist_stat_flat
            pcl_c3d.grid.feature['dist_stat'] = dist_stat_grid

        pcl_c3d.flat.feature['normal'] = normal_flat
        pcl_c3d.flat.feature['nres'] = nres_flat
        pcl_c3d.grid.feature['normal'] = normal_grid
        pcl_c3d.grid.feature['nres'] = nres_grid

    return pcl_c3d

def transform_pc3d(pcl_c3d, Ts, seq_n, K_cur, batch_n):
    """This function construct PCL_C3D_Flat objects which is transformed to the adjacent frame. 
    It assumes that a batch includes one or more rotating groups. 
    For example, when seq_n=3, batch_n=6, it means a batch [a0, a1, a2, b0, b1, b2], where a0, a1, a2 are 3 sequential frames, and b0, b1, b2 are 3 sequential frames. 
    After transformation, the batch of point clouds is transformed to the reference frame of [a1, a2, a0, b1, b2, b0]. 
    We only construct PCL_C3D_Flat objects because the calc_inn_pc3d() function takes a PCL_C3D_Flat object and a PCL_C3D_Grid object as input. The PCL_C3D_Grid form of transformed point cloud is not used thus not necessary. 
    """

    ## need to transform: flat.uvb, flat.feature['xyz'], flat.feature['normal']
    ## no need to transform grid features
    
    assert batch_n % seq_n == 0    # mode==0
    n_group = batch_n // seq_n

    ## get relative pose
    T, R, t, target_id = relative_T(Ts, seq_n, batch_n)

    ## get accumulative length
    nb = pcl_c3d.flat.nb
    acc_b = []
    acc = 0
    acc_b.append( acc )
    for ib in range(batch_n):
        acc = acc + nb[ib]
        acc_b.append( acc )

    ## process flat features
    flat_xyz = pcl_c3d.flat.feature['xyz']      # 1*C*NB
    flat_normal = pcl_c3d.flat.feature['normal']
    trans_normal_list = []
    trans_xyz_list = []
    uvb_list = []
    new_nb = []
    for ib in range(batch_n):
        ## xyz
        trans_xyz = torch.matmul(R[ib], flat_xyz[:, :, acc_b[ib]:acc_b[ib+1]]) + t[ib]
        mask_positive = trans_xyz[0, 2, :] > 0
        trans_xyz = trans_xyz[:, :, mask_positive]
        trans_xyz_list.append(trans_xyz)
        new_nb.append(trans_xyz.shape[2])

        ## normal
        trans_normal = torch.matmul(R[ib], flat_normal[:, :, acc_b[ib]:acc_b[ib+1]])
        trans_normal = trans_normal[:, :, mask_positive]
        trans_normal_list.append(trans_normal)

        ## project to uv, add b
        uvb = torch.matmul(K_cur[ib], trans_xyz)
        uvb[:, :2] = uvb[:, :2] / uvb[:, [2]] #- 1 , commented because in dataset_read.py there is a K_mat2py() function converting K from matlab to python coordinate
        uvb[:, 2, :] = target_id[ib]
        uvb_list.append(uvb)

    ## construct the new object
    tr_pcl_c3d = PCL_C3D_Flat()
    tr_pcl_c3d.feature['xyz'] = torch.cat(trans_xyz_list, dim=2)
    tr_pcl_c3d.feature['normal'] = torch.cat(trans_normal_list, dim=2)
    tr_pcl_c3d.uvb = torch.cat(uvb_list, dim=2)
    tr_pcl_c3d.nb = new_nb

    for feat_key in pcl_c3d.flat.feature:
        if feat_key not in ['xyz', 'normal']:
            tr_pcl_c3d.feature[feat_key] = pcl_c3d.flat.feature[feat_key]

    return tr_pcl_c3d
    
def relative_T(Ts, seq_n, batch_size):
    
    assert batch_size % seq_n == 0    # mode==0
    n_group = batch_size // seq_n

    ## get relative pose
    T = []
    R = []
    t = []
    target_id = []
    for n_g in range(n_group):
        for n_f in range(seq_n):
            n = n_g * seq_n + n_f
            T1 = Ts[n]
            if n_f == seq_n-1:
                tid = n_g * seq_n
            else:
                tid = n+1
            target_id.append(tid)
            T2 = Ts[tid]
            T21 = torch.matmul( torch.inverse(T2), T1 )
            T.append(T21)
            R21 = T21[:3, :3]
            R.append(R21)
            t21 = T21[:3, [3]]
            t.append(t21)
    
    return T, R, t, target_id

'''from bts/bts_pre_intr.py'''
def sub2ind(matrixSize, batchSub, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    b, h, w = matrixSize
    return batchSub * (h*w) + rowSub * w + colSub

def flow_pc3d(pcl_c3d, flow_grid, flow_mask_grid, K_cur, feat_comm_keys, use_normal, sparse_nml_opts=None, return_stat=False, timer=None):
    """
    This function construct PCL_C3D_Flat objects which is transformed by the 3D scene flow.
    """
    if timer is not None:
        timer.log("flow_pc3d start", 1, True)

    batch_size = flow_grid.shape[0]

    ### compose the flow to xyz
    xyz_grid = pcl_c3d.grid.feature['xyz']
    xyz_flat = xyz_grid.reshape(batch_size, 3, -1)
    flow_flat = flow_grid.reshape(batch_size, 3, -1)
    flow_flat = torch.cat([flow_flat[:,:2].detach(), flow_flat[:, 2:]], dim=1)      # detach the x and y dimension of the flow
    xyz_flowed_flat = xyz_flat.detach() + flow_flat         # detach so that the flowed c3d loss only affects the flow gradient instead of both flow and depth. Otherwise depth could be confused. 
    # logging.info("xyz_flat.detach(): %s"%(xyz_flat.detach().requires_grad))

    ### mask out invalid pixels and project to image uv coordinate
    xyz_mask_grid = pcl_c3d.grid.mask
    # if False:
    if flow_mask_grid is not None:
        mask_grid = xyz_mask_grid & flow_mask_grid
    else:
        mask_grid = xyz_mask_grid 
    mask_flat = mask_grid.reshape(batch_size, 1, -1)

    xyz_flowed_flat_list = [None]*batch_size
    uvb_list = [None]*batch_size
    new_nb = [None]*batch_size
    inview_mask_list = [None]*batch_size
    
    for ib in range(batch_size):
        if timer is not None:
            timer.log("uvb, inview_mask ib=%d"%ib, 2, True)
        mask_vec = mask_flat[ib, 0]
        xyz_flowed_flat_cur = xyz_flowed_flat[[ib]][:,:,mask_vec]  # 1*3*N

        uvb = torch.matmul(K_cur[ib], xyz_flowed_flat_cur) # 1*3*N
        uvb_1 = ( uvb / torch.clamp(torch.abs(uvb[:, [2]]), min=1e-6) ).round() #- 1 , commented because in dataset_read.py there is a K_mat2py() function converting K from matlab to python coordinate
        uvb_1[:, 2] = ib
        # uvb_list[ib] = uvb

        # assert (uvb[:,2] == xyz_flowed_flat_cur[:,2]).all(), "{} {}".format(uvb[0,2,0], xyz_flowed_flat_cur[0,2,0])
        # logging.info( "{} {}".format(uvb[0,2,0], xyz_flowed_flat_cur[0,2,0]) )
        ### check whether the new points are in the view of camera
        inview_mask = (uvb_1[0,0,:] > 0) & (uvb_1[0,0,:] < mask_grid.shape[3]) & (uvb_1[0,1,:] > 0) & (uvb_1[0,1,:] < mask_grid.shape[2]) & (xyz_flowed_flat_cur[0,2,:] > 0.1)
        inview_mask_list[ib] = inview_mask

        xyz_flowed_flat_cur = xyz_flowed_flat_cur[:,:,inview_mask]
        uvb_1 = uvb_1[:,:,inview_mask]
        # logging.info("diff between uvb2: {}, {}, {}".format((uvb_1-uvb_2).max(), (uvb_1-uvb_2).min(), (uvb_1[:,:2]-uvb_2[:,:2]).mean()) )
        # logging.info("uvb_1.shape: {} {}".format(uvb_1.shape, uvb.shape))
        xyz_flowed_flat_list[ib] = xyz_flowed_flat_cur
        uvb_list[ib] = uvb_1

        new_nb[ib] = uvb_1.shape[2]
    
    # print("new_nb:", new_nb)
    if timer is not None:
        timer.log("cat xyz, uvb", 1, True)

    xyz_flowed_flat = torch.cat(xyz_flowed_flat_list, dim=2)
    uvb_flat = torch.cat(uvb_list, dim=2)

    ### The occlusion check is the speed bottleneck (>0.4s), and the effect is similar to flow_mask_grid, therefore disabled
    # if timer is not None:
    #     timer.log("occlu_mask", 1, True)
    # ### find the duplicate points and filter out those not close to the camera
    # occlu_mask = torch.ones(uvb_flat.shape[2], dtype=torch.bool, device=mask_grid.device)

    # uvb_dim = [xyz_grid.shape[0], xyz_grid.shape[2], xyz_grid.shape[3]]
    # velo_proj_lin = sub2ind(uvb_dim, uvb_flat[0, 2, :], uvb_flat[0, 1, :], uvb_flat[0, 0, :] )  # B, H, W
    # dupe_proj_lin = [item for item, count in Counter(velo_proj_lin).items() if count > 1]
    # # print("# or dupe_proj_lin:", len(dupe_proj_lin))
    # for dd in dupe_proj_lin:
    #     pts = torch.where(velo_proj_lin == dd)[0] ### torch.where() [actually torch.nonzero(condition, as_tuple=True)] returns a tuple. [0] takes the array of the first dim.
    #     z_min = 1e7
    #     for pt_idx in pts:
    #         z_cur = xyz_flowed_flat[0, 2, pt_idx]
    #         if z_cur < z_min:
    #             z_min = z_cur
    #             min_idx = pt_idx
    #         else:
    #             occlu_mask[pts] = False
    #             ib = uvb_flat[0, 2, pt_idx]
    #             new_nb[ib] -= 1
    
    # # print("before occlu_mask:", xyz_flowed_flat.shape[2])
    # xyz_flowed_flat = xyz_flowed_flat[:,:,occlu_mask]
    # uvb_flat = uvb_flat[:,:,occlu_mask]
    # # print("after occlu_mask:", xyz_flowed_flat.shape[2])

    if timer is not None:
        timer.log("PCL_C3D_Flat", 1, True)
    ### construct PCL_C3D_Flat
    flow_pcl_c3d_flat = PCL_C3D_Flat()
    flow_pcl_c3d_flat.uvb = uvb_flat
    flow_pcl_c3d_flat.feature['xyz'] = xyz_flowed_flat
    flow_pcl_c3d_flat.nb = new_nb

    ### need to exit early if empty, otherwise later processing will produce unpredicted result and failure in next iteration
    if any(n <= 0 for n in new_nb):
        return flow_pcl_c3d_flat, None
    #     raise ValueError("empty pcl: {}".format(new_nb))

    if timer is not None:
        timer.log("feat_flat", 1, True)
    ### copy those shared features from original point cloud. Remember to apply the same masking.
    for feat in feat_comm_keys:
        feat_flat = pcl_c3d.grid.feature[feat].reshape(batch_size, 3, -1)
        feat_flat_list = [None]*batch_size
        for ib in range(batch_size):
            mask_vec = mask_flat[ib, 0]
            feat_flat_list[ib] = feat_flat[[ib]][:,:,mask_vec]

            ### filter out out-of-view points
            feat_flat_list[ib] = feat_flat_list[ib][:,:,inview_mask_list[ib]]

        feat_flat_concat = torch.cat(feat_flat_list, dim=2)
        ### filter out points duplicated on image
        # flow_pcl_c3d_flat.feature[feat] = feat_flat_concat[:,:,occlu_mask]
        flow_pcl_c3d_flat.feature[feat] = feat_flat_concat

    if timer is not None:
        timer.log("feat_grid", 1, True)
    ### prepare xyz_grid of the flowed point cloud
    uvb_split = uvb_flat.to(dtype=torch.long).squeeze(0).transpose(0,1).split(1,dim=1) # a tuple of 3 elements of tensor N*1, only long/byte/bool tensors can be used as indices
    xyz_flowed_grid = grid_from_concat_flat_func(uvb_split, xyz_flowed_flat, xyz_grid.shape)
    mask_flowed_grid = (xyz_flowed_grid != 0).any(1, keepdim=True)

    if timer is not None:
        timer.log("calc_normal", 1, True)
    ### calculate sparse normal
    if use_normal:
        if return_stat:
            normal_flat, nres_flat, dist_stat_flat = calc_normal(flow_pcl_c3d_flat.uvb, xyz_flowed_grid, mask_flowed_grid, sparse_nml_opts.normal_nrange, sparse_nml_opts.ignore_ib, sparse_nml_opts.min_dist_2, return_stat=return_stat)
        else:
            normal_flat, nres_flat = calc_normal(flow_pcl_c3d_flat.uvb, xyz_flowed_grid, mask_flowed_grid, sparse_nml_opts.normal_nrange, sparse_nml_opts.ignore_ib, sparse_nml_opts.min_dist_2, return_stat=return_stat)
        
        flow_pcl_c3d_flat.feature['normal'] = normal_flat
        flow_pcl_c3d_flat.feature['nres'] = nres_flat

        if return_stat:
            flow_pcl_c3d_flat.feature['dist_stat'] = dist_stat_flat

    if timer is not None:
        timer.log("PCL_C3D_Grid", 1, True)
    ### construct PCL_C3D_Grid
    flow_pcl_c3d_grid = PCL_C3D_Grid()
    flow_pcl_c3d_grid.mask = mask_flowed_grid
    flow_pcl_c3d_grid.feature['xyz'] = xyz_flowed_grid

    for feat in feat_comm_keys:
        flow_pcl_c3d_grid.feature[feat] = grid_from_concat_flat_func(uvb_split, flow_pcl_c3d_flat.feature[feat], pcl_c3d.grid.feature[feat].shape)

    if use_normal:
        flow_pcl_c3d_grid.feature['normal'] = grid_from_concat_flat_func(uvb_split, flow_pcl_c3d_flat.feature['normal'], pcl_c3d.grid.feature['normal'].shape)
        flow_pcl_c3d_grid.feature['nres'] = grid_from_concat_flat_func(uvb_split, flow_pcl_c3d_flat.feature['nres'], pcl_c3d.grid.feature['nres'].shape)
        if return_stat:
            flow_pcl_c3d_grid.feature['dist_stat'] = grid_from_concat_flat_func(uvb_split, flow_pcl_c3d_flat.feature['dist_stat'], pcl_c3d.grid.feature['dist_stat'].shape)        

    return flow_pcl_c3d_flat, flow_pcl_c3d_grid

class C3DLoss(nn.Module):
    def __init__(self, seq_frame_n=0): # , flow_mode=False
        super(C3DLoss, self).__init__()
        self.seq_frame_n = seq_frame_n      
        ### If seq_frame_n is not set (i.e. =0), it means that the target and source inputs are given separately (not mixed in a batch). 
        ### Therefore use forward_with_flow(). Otherwise use forward_with_caminfo()
        
        # self.flow_mode = flow_mode

        self.feat_inp_self = ["xyz", "hsv"]
        self.feat_inp_cross = ["xyz", "hsv"]

        self.feat_comm = ["hsv"] ### features invariant to transformation (rigid-body transformation or flow)

        self.normal_op_dense = NormalFromDepthDense()

        self.internal_count = 0     ### the internal count is to differentiate different forward passes when we debug the input and write them to files. 

        # self.timer = Timing()

    def parse_opts(self, inputs=None, f_input=None):
        # parser = argparse.ArgumentParser(description='Options for continuous 3D loss')
        parser = init_argparser_f(description='Options for continuous 3D loss')

        ## switch for enabling CVO loss
        parser.add_argument("--ell_basedist",          type=float, default=0,
                            help="if not zero, the length scale is proportional to the depth of gt points when the depth is larger than this value. If zero, ell is constant")
        parser.add_argument("--ell_keys",              nargs="+", type=str, default=['xyz', 'hsv'], 
                            help="keys of ells corresponding to ell_values")
        parser.add_argument("--ell_values_min",        nargs="+", type=float, default=[0.05, 0.1], 
                            help="min values of ells corresponding to ell_keys")
        parser.add_argument("--ell_values_rand",       nargs="+", type=float, default=[0.1, 0], 
                            help="parameter of randomness for values of ells corresponding to ell_keys")

        parser.add_argument("--ell_predpred_min",        nargs="+", type=float, default=[0.05, 0.1], 
                            help="min values of ells corresponding to ell_keys, for inner product between both predictions")
        parser.add_argument("--ell_predpred_rand",       nargs="+", type=float, default=[0.1, 0], 
                            help="parameter of randomness for values of ells corresponding to ell_keys, for inner product between both predictions")

        parser.add_argument("--use_normal",            type=int, default=0, 
                            help="if set, normal vectors of sparse pcls are from PtSampleInGridCalcNormal, while those of dense images are from NormalFromDepthDense")
        parser.add_argument("--neg_nkern_to_zero",     action="store_true",
                            help="if set, negative normal kernels are truncated to zero, otherwise use absolute value of normel kernel")
        parser.add_argument("--norm_in_dist",          action="store_true", 
                            help="if set, the normal information will be used in exp kernel besides as a coefficient term. Neet use_normal_v2 to be true to be effective")
        parser.add_argument("--res_mag_min",           type=float, default=0.1,
                            help="the minimum value for the normal kernel (or viewing it as a coefficient of geometric kernel)")
        parser.add_argument("--res_mag_max",           type=float, default=2,
                            help="the maximum value for the normal kernel (or viewing it as a coefficient of geometric kernel)")
        parser.add_argument("--norm_return_stat",      action="store_true", 
                            help="if set, calc_normal (for sparse) function will return statistics in terms of point distance")

        parser.add_argument("--neighbor_range",        type=int, default=2,
                            help="neighbor range when calculating inner product")
        parser.add_argument("--normal_nrange",         type=int, default=5,
                            help="neighbor range when calculating normal direction on sparse point cloud")
        parser.add_argument("--cross_pred_pred_weight",        type=float, default=0,
                            help="weight of c3d loss between cross-frame predictions relative to gt_pred_weight as 1. You may want to set to less than 1 because predictions are denser than gt.")
        parser.add_argument("--cross_gt_pred_weight",        type=float, default=0,
                            help="weight of c3d loss between predictions and gt from another frame, relative to gt_pred_weight as 1. You may want to set to 1. ")

        parser.add_argument("--debug_input",           action="store_true", 
                            help="if set, write the input depth, rgb, etc. to file to show whether the input is correct. ")
        parser.add_argument("--debug_path",            type=str, required=False, default=None, 
                            help="the path to output debug files. Required if debug_input is True")
        parser.add_argument("--flow_mode",           action="store_true", 
                            help="if set, consider flow in c3d loss. Otherwise only use the depth")
        parser.add_argument("--log_loss",            action="store_true", 
                            help="if set, use log of c3d_loss instead of the original sum of expoentials")

        if f_input is None:
            ### take parsed args or sys.argv[1:] as input
            self.opts, rest = parser.parse_known_args(args=inputs) # inputs can be None, in which case _sys.argv[1:] are parsed
        else:
            ### take a file as input with dedicate c3d options, or 
            arg_filename_with_prefix = '@' + f_input
            self.opts, rest = parser.parse_known_args([arg_filename_with_prefix])
            print("C3D options:")
            print(self.opts)

        self.flow_mode = self.opts.flow_mode

        self.opts.ell_min = {}
        self.opts.ell_rand = {}
        for i, ell_item in enumerate(self.opts.ell_keys):
            self.opts.ell_min[ell_item] = self.opts.ell_values_min[i]
            self.opts.ell_rand[ell_item] = self.opts.ell_values_rand[i]

        if self.opts.cross_pred_pred_weight > 0:
            self.opts.ell_min_predpred = {}
            self.opts.ell_rand_predpred = {}
            for i, ell_item in enumerate(self.opts.ell_keys):
                self.opts.ell_min_predpred[ell_item] = self.opts.ell_predpred_min[i]
                self.opts.ell_rand_predpred[ell_item] = self.opts.ell_predpred_rand[i]

        self.nml_opts = EasyDict() # nml_opts.neighbor_range, nml_opts.ignore_ib, nml_opts.min_dist_2
        self.nml_opts.normal_nrange = int(self.opts.normal_nrange)
        self.nml_opts.ignore_ib = False
        self.nml_opts.min_dist_2 = 0.05

        return rest

    def gen_rand_ell(self):
        ells = {}
        ell = {}
        for key in self.opts.ell_keys:
            ell[key] = self.opts.ell_min[key] + np.abs(self.opts.ell_rand[key]* np.random.normal()) 
        ells["pred_gt"] = ell

        if self.opts.cross_pred_pred_weight > 0:
            ell_predpred = {}
            for key in self.opts.ell_keys:
                ell_predpred[key] = self.opts.ell_min_predpred[key] + np.abs(self.opts.ell_rand_predpred[key]* np.random.normal())
            ells["pred_pred"] = ell_predpred
            
        return ells

    def forward(self, rgb=None, depth=None, depth_gt=None, depth_mask=None, depth_gt_mask=None, cam_info=None, nkern_fname=None, Ts=None, 
                depth_img_dict_1=None, depth_img_dict_2=None, flow_dict_1to2=None, flow_dict_2to1=None):
        """
        rgb: B*3*H*W
        depth, depth_gt, depth_mask, depth_gt_mask: B*1*H*W
        """
        self.internal_count += 1

        if self.seq_frame_n != 0:
            return self.forward_with_caminfo(rgb, depth, depth_gt, depth_mask, depth_gt_mask, nkern_fname, Ts, cam_info)
        else:
            return self.forward_with_flow(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info, nkern_fname)
        
        # return self.forward_with_flow(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info, nkern_fname)

    def load_pc3d(self, depth_img_dict, cam_info):
        ## ---------------------------------
        ## unpack the depth info
        ## ---------------------------------
        depth = depth_img_dict["pred"]
        depth_mask = depth_img_dict["pred_mask"]
        depth_gt = depth_img_dict["gt"]
        depth_gt_mask = depth_img_dict["gt_mask"]
        rgb = depth_img_dict['rgb']

        ## ---------------------------------
        ## load PCL_C3D objects
        ## ---------------------------------
        batch_size = rgb.shape[0]
        
        K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()

        uvb_flat_cur = uvb_grid_cur.reshape(batch_size, 3, -1)

        pc3ds = EasyDict()
        pc3ds["gt"] = PCL_C3D()
        pc3ds["pred"] = PCL_C3D()

        ## rgb to hsv
        hsv = rgb_to_hsv(rgb, flat=False)           # B*3*H*W
        hsv_flat = hsv.reshape(batch_size, 3, -1)   # B*3*N

        feat_comm_grid = {'hsv': hsv}
        feat_comm_flat = {'hsv': hsv_flat}

        assert set(list(feat_comm_grid.keys())) == set(self.feat_comm), "{}, {}".format(list(feat_comm_grid.keys()), self.feat_comm)
        
        ## generate PCL_C3D object
        pc3ds["gt"] = load_pc3d(pc3ds["gt"], depth_gt, depth_gt_mask, xy1_grid_cur, uvb_flat_cur, K_cur, feat_comm_grid, feat_comm_flat, 
                                        sparse=True, use_normal=self.opts.use_normal, sparse_nml_opts=self.nml_opts, return_stat=self.opts.norm_return_stat)
        pc3ds["pred"] = load_pc3d(pc3ds["pred"], depth, depth_mask, xy1_grid_cur, uvb_flat_cur, K_cur, feat_comm_grid, feat_comm_flat, 
                                        sparse=False, use_normal=self.opts.use_normal, dense_nml_op=self.normal_op_dense, return_stat=self.opts.norm_return_stat)
        return pc3ds

    def debug_flow_input_to_imgs(self, depth_img_dict_1, depth_img_dict_2):
        for i, depth_img_dict in enumerate([depth_img_dict_1, depth_img_dict_2]):

            depth = depth_img_dict["pred"]
            depth_mask = depth_img_dict["pred_mask"]
            depth_gt = depth_img_dict["gt"]
            depth_gt_mask = depth_img_dict["gt_mask"]
            rgb = depth_img_dict['rgb']
            # print(depth)

            batch_size = depth_gt.shape[0]

            path = self.opts.debug_path
            name = "n{:04d}_b{}_s{}_{}.jpg"
            for ib in range(batch_size):
                overlay_dep_on_rgb(depth_gt[ib], rgb[ib], path=path, name=name.format(self.internal_count, ib, i, "dep_gt"))
                overlay_dep_on_rgb(depth[ib], rgb[ib], path=path, name=name.format(self.internal_count, ib, i, "dep_pred"), overlay=False)
                dep_img_bw(depth_mask[ib], path=path, name=name.format(self.internal_count, ib, i, "mask_pred"))
                dep_img_bw(depth_gt_mask[ib], path=path, name=name.format(self.internal_count, ib, i, "mask_gt"))
        
        return

    def debug_flow_dump_pickle(self, depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info):
        dump_path = os.path.join(self.opts.debug_path, "nan_dicts.pkl")
        if not os.path.exists(os.path.dirname(dump_path)):
            os.makedirs(os.path.dirname(dump_path))

        with open(dump_path, "wb") as f:
            pickle.dump(depth_img_dict_1, f)
            pickle.dump(depth_img_dict_2, f)
            pickle.dump(flow_dict_1to2, f)
            pickle.dump(flow_dict_2to1, f)
            pickle.dump(cam_info, f)
        return

    def debug_flow_load_pickle(self, pickle_path):
        with open(pickle_path, "rb") as f:
            depth_img_dict_1 = pickle.load(f)
            depth_img_dict_2 = pickle.load(f)
            flow_dict_1to2 = pickle.load(f)
            flow_dict_2to1 = pickle.load(f)
            cam_info = pickle.load(f)
        return depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info
            
    def debug_flow_dep_to_pcd(self, depth_img_dict, cam_info):
        depth = depth_img_dict["pred"]
        depth_mask = depth_img_dict["pred_mask"]
        depth_gt = depth_img_dict["gt"]
        depth_gt_mask = depth_img_dict["gt_mask"]
        rgb = depth_img_dict['rgb']

        K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()

        clouds_pred = pcl_from_grid_xy1_dep(xy1_grid_cur, depth, rgb=rgb)
        clouds_gt = pcl_from_grid_xy1_dep(xy1_grid_cur, depth_gt, rgb=rgb)

        return clouds_pred, clouds_gt

    def debug_flow_depflow_to_pcd(self, depth_img_dict, flow_dict, cam_info):
        depth = depth_img_dict["pred"]
        depth_mask = depth_img_dict["pred_mask"]
        depth_gt = depth_img_dict["gt"]
        depth_gt_mask = depth_img_dict["gt_mask"]
        rgb = depth_img_dict['rgb']

        flow = flow_dict["pred"]
        flow_mask = flow_dict["mask"]

        K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()
        
        batch_size = depth.shape[0]

        xy1_grid_cur = xy1_grid_cur.permute(0,2,3,1)
        depth = depth.permute(0,2,3,1)  # B*H*W*C
        rgb = rgb.permute(0,2,3,1)  # B*H*W*C
        flow = flow.permute(0,2,3,1)  # B*H*W*C

        depth_mask = depth_mask.permute(0,2,3,1)  # B*H*W*C
        flow_mask = flow_mask.permute(0,2,3,1) if flow_mask is not None else flow_mask  # B*H*W*C
        mask = flow_mask & depth_mask if flow_mask is not None else depth_mask
        mask = mask.squeeze(3)  # B*H*W

        clouds = []
        for ib in range(batch_size):
            mask_i = mask[ib]
            depth_i = depth[ib][mask_i] #N*C
            xy1_i = xy1_grid_cur[ib][mask_i]
            xyz = xy1_i * depth_i
            flow_i = flow[ib][mask_i]
            xyz_flowed = xyz + flow_i
            rgb_i = rgb[ib][mask_i]
            cloud_i = pcl_from_flat_xyz(xyz_flowed.transpose(0,1), rgb=rgb_i.transpose(0,1))
            clouds.append(cloud_i)
        return clouds

    def debug_flow_input_to_pcds_raw(self, depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info):
        """This is to generate pcd files from input depth and flow"""
        ### load four point clouds predictions and two ground truths to pcd file
        clouds_pred_1, clouds_gt_1 = self.debug_flow_dep_to_pcd(depth_img_dict_1, cam_info)
        clouds_pred_2, clouds_gt_2 = self.debug_flow_dep_to_pcd(depth_img_dict_2, cam_info)

        clouds_flowed_2_from_1 = self.debug_flow_depflow_to_pcd(depth_img_dict_1, flow_dict_1to2, cam_info)
        clouds_flowed_1_from_2 = self.debug_flow_depflow_to_pcd(depth_img_dict_2, flow_dict_2to1, cam_info)

        path = self.opts.debug_path
        name = "n{:04d}_b{}_s{}_{}"
        batch_size = len(clouds_pred_1)
        assert batch_size == len(clouds_gt_1) and batch_size == len(clouds_pred_2) and batch_size == len(clouds_gt_2)
        for ib in range(batch_size):
            pcl_write(clouds_pred_1[ib], os.path.join(path, name.format(self.internal_count, ib, 0, "pred") ) )
            pcl_write(clouds_pred_2[ib], os.path.join(path, name.format(self.internal_count, ib, 1, "pred") ) )
            pcl_write(clouds_gt_1[ib], os.path.join(path, name.format(self.internal_count, ib, 0, "gt") ) )
            pcl_write(clouds_gt_2[ib], os.path.join(path, name.format(self.internal_count, ib, 1, "gt") ) )
            pcl_write(clouds_flowed_2_from_1[ib], os.path.join(path, name.format(self.internal_count, ib, 1, "flowed") ) )
            pcl_write(clouds_flowed_1_from_2[ib], os.path.join(path, name.format(self.internal_count, ib, 0, "flowed") ) )

        return

    def debug_flow_input_to_pcds_pcl_c3d(self, pcl_c3d_1, pcl_c3d_2, pcl_c3d_gt_1, pcl_c3d_gt_2, pcl_c3d_1_from_2, pcl_c3d_2_from_1):
        """This is to generate pcd files from PCL_C3D_Grid or PCL_C3D_Flat"""
        
        flats = {}
        flats["1"] = pcl_c3d_1.flat.to_pcd()
        flats["2"] = pcl_c3d_2.flat.to_pcd()
        flats["gt_1"] = pcl_c3d_gt_1.flat.to_pcd()
        flats["gt_2"] = pcl_c3d_gt_2.flat.to_pcd()
        flats["1_from_2"] = pcl_c3d_1_from_2.flat.to_pcd()
        flats["2_from_1"] = pcl_c3d_2_from_1.flat.to_pcd()

        grids = {}
        grids["1"] = pcl_c3d_1.grid.to_pcd()
        grids["2"] = pcl_c3d_2.grid.to_pcd()
        grids["gt_1"] = pcl_c3d_gt_1.grid.to_pcd()
        grids["gt_2"] = pcl_c3d_gt_2.grid.to_pcd()
        grids["1_from_2"] = pcl_c3d_1_from_2.grid.to_pcd()
        grids["2_from_1"] = pcl_c3d_2_from_1.grid.to_pcd()

        batch_size = len(flats["1"])
        for key in flats:
            assert len(flats[key]) == batch_size
        for key in grids:
            assert len(grids[key]) == batch_size

        path = self.opts.debug_path
        name = "n{:04d}_b{}_s{}_{}"
        for ib in range(batch_size):
            pcl_write(flats["1"][ib], os.path.join(path, name.format(self.internal_count, ib, 0, "pred_flat") ) )
            pcl_write(flats["2"][ib], os.path.join(path, name.format(self.internal_count, ib, 1, "pred_flat") ) )
            pcl_write(flats["gt_1"][ib], os.path.join(path, name.format(self.internal_count, ib, 0, "gt_flat") ) )
            pcl_write(flats["gt_2"][ib], os.path.join(path, name.format(self.internal_count, ib, 1, "gt_flat") ) )
            pcl_write(flats["1_from_2"][ib], os.path.join(path, name.format(self.internal_count, ib, 0, "flowed_flat") ) )
            pcl_write(flats["2_from_1"][ib], os.path.join(path, name.format(self.internal_count, ib, 1, "flowed_flat") ) )

            pcl_write(grids["1"][ib], os.path.join(path, name.format(self.internal_count, ib, 0, "pred_grid") ) )
            pcl_write(grids["2"][ib], os.path.join(path, name.format(self.internal_count, ib, 1, "pred_grid") ) )
            pcl_write(grids["gt_1"][ib], os.path.join(path, name.format(self.internal_count, ib, 0, "gt_grid") ) )
            pcl_write(grids["gt_2"][ib], os.path.join(path, name.format(self.internal_count, ib, 1, "gt_grid") ) )
            pcl_write(grids["1_from_2"][ib], os.path.join(path, name.format(self.internal_count, ib, 0, "flowed_grid") ) )
            pcl_write(grids["2_from_1"][ib], os.path.join(path, name.format(self.internal_count, ib, 1, "flowed_grid") ) )

        return 


    def debug_flow_inspect_input(self, pickle_path):
        """This is to """
        ### unpickle
        depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info = self.debug_flow_load_pickle(pickle_path)
        ### visualize input depth and mask
        self.debug_flow_input_to_imgs(depth_img_dict_1, depth_img_dict_2)
        self.debug_flow_input_to_pcds_raw(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info)

        ### calculate the inner product
        inp_total = self.forward_with_flow(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info, nkern_fname=None, debug_save_pcd=True)

        return inp_total

    def vis_pts_2D(self, pts, pts_ells, grid_h, grid_w, pts_full=None, mu=False):
        """https://www.xarg.org/2018/04/how-to-plot-a-covariance-error-ellipse/
        https://www.tensorflow.org/tensorboard/image_summaries"""
        batch_size = len(pts)

        buffs = []
        p = 0.95
        s = -2 * np.log(1 - p)
        for ib in range(batch_size):
            ells = []
            alphas = []
            sig_xs = []
            sig_ys = []
            rho_xys = []
            dets = []
            weights = []
            n_pts = pts[ib].shape[0]
            for ip in range( n_pts ):
                sig_x = pts_ells[ib][0, ip].item()
                sig_y = pts_ells[ib][1, ip].item()
                rho_xy = pts_ells[ib][2, ip].item()
                weight = pts_ells[ib][3, ip].item()
                
                a = sig_x * sig_x
                b = rho_xy * sig_x * sig_y
                c = b
                d = sig_y * sig_y

                tmp = np.sqrt((a - d) * (a - d) + 4 * b * c)

                v11 = -(tmp - a + d) / (2 * c+1e-7)
                v12 = (tmp + a - d) / (2 * c+1e-7)
                v21 = 1
                v22 = 1
                angle = np.arctan2(1, v11) / np.pi * 180    # degree

                lambda1 = (a + d - tmp) / 2
                lambda2 = (a + d + tmp) / 2
                e1 = np.sqrt(s * lambda1)
                e2 = np.sqrt(s * lambda2)

                ### avoid that the diameter shrinks too small to plot on the image
                e1 = max(e1, 1e-1)
                e2 = max(e2, 1e-1)

                det = a * d - b * c

                det = max(det, 1e-7)

                sig_xs.append(sig_x)
                sig_ys.append(sig_y)
                rho_xys.append(rho_xy)
                dets.append(det)
                weights.append(weight)
                
                x0 = pts[ib][ip, 1].item()
                y0 = pts[ib][ip, 0].item()
                if mu:
                    x0 = x0 + pts_ells[ib][5, ip].item()
                    y0 = y0 + pts_ells[ib][4, ip].item()

                ### input coordinate is x-down, y-right. Plot coordinate is X-right, Y-down.
                ### X = y, Y = -x + x0
                ### Theta = atan2(Y, X) = atan2(-x, y) = - atan2(x, y) = - (90 - atan2(y, x)) = theta - 90
                # ell = Ellipse(xy=pts[ib][ip], width=e1, height=e2, angle=angle)
                # ell = Ellipse(xy=(pts[ib][ip, 1], -pts[ib][ip, 0] + grid_h), width=3, height=1, angle=- 90)
                ell = Ellipse(xy=(x0, -y0 + grid_h), width=e1, height=e2, angle=angle - 90) # - 90
                # ell = Ellipse(xy=pts[ib][ip], width=3, height=3, angle=angle)
                alpha = weight / det

                ells.append(ell)
                alphas.append(alpha)

            sig_xs = np.array(sig_xs)
            sig_ys = np.array(sig_ys)
            rho_xys = np.array(rho_xys)
            dets = np.array(dets)
            weights = np.array(weights)

            alphas = np.array(alphas)
            alphas = alphas / np.max([alphas.max(), 1e-7]) * 0.5

            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            if pts_full is not None:
                plt.plot(pts_full[ib].cpu().detach().numpy()[:, 1], - pts_full[ib].cpu().detach().numpy()[:, 0] + grid_h, 'g.')
            plt.plot(pts[ib].cpu().detach().numpy()[:, 1], - pts[ib].cpu().detach().numpy()[:, 0] + grid_h, 'bo')
            if mu:
                plt.plot(pts[ib].cpu().detach().numpy()[:, 1] + pts_ells[ib].cpu().detach().numpy()[5, :], - (pts[ib].cpu().detach().numpy()[:, 0] + pts_ells[ib].cpu().detach().numpy()[4, :]) + grid_h, 'r.')
            for ip in range( n_pts ):
                ax.add_artist(ells[ip])
                ells[ip].set_clip_box(ax.bbox)
                ells[ip].set_alpha(alphas[ip])
                # ells[ip].set_alpha(1)
                ells[ip].set_facecolor( (0.8,0,0) )

            ax.set_xlim(0, grid_h)
            ax.set_ylim(0, grid_w)

            title_test = "sig_x: {:.2f}, {:.2f}, {:.2f}, sig_y: {:.2f}, {:.2f}, {:.2f}, rho_xy: {:.2f}, {:.2f}, {:.2f}, \n".format(
                sig_xs.min(), sig_xs.mean(), sig_xs.max(), sig_ys.min(), sig_ys.mean(), sig_ys.max(), rho_xys.min(), rho_xys.mean(), rho_xys.max()) + \
                            "det: {:.2f}, {:.2f}, {:.2f}, alpha: {:.2f}, {:.2f}, {:.2f}, weights: {:.2f}, {:.2f}, {:.2f}".format(
                dets.min(), dets.mean(), dets.max(), alphas.min(), alphas.mean(), alphas.max(), weights.min(), weights.mean(), weights.max())
            plt.title(title_test)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            pic = Image.open(buf)
            pic_array = np.array(pic)
            pic_array = pic_array.transpose(2,0,1)
            pic.close()
            buffs.append(pic_array)

        return buffs


    def forward_2D(self, grid_h, grid_w, pts_1_raw, pts_ells_1, pts_2, pts_ells_2=None, vis=False, mu=False):
        """
        This function is to conceptually implement a c3d function with covariance matrix for each point. 
        pts_1: a list of length B. Each element of the list is a tensor N*2 (or N*3 with z dim = 0). Each row is the coordinate of a point in 2D plane (image). 
        pts_ells_1: a list of length B. Each element of the list is a tensor 4*N. Each row is [sigma_x, sigma_y, rho_xy, weight] determining the covariance matrix of the corresponding point. 
        Assume pts_1 is the sparse one with covariance matrix. pts_2 is dense and only has points, therefore we do not need pts_ells_2 here. 
        """
        ### convert to list in case the input pts is a batched tensor

        # print("pts_1",  pts_1.min(), pts_1.max())
        # print("pts_ells_1", pts_ells_1.min(), pts_ells_1.max())
        # print("pts_2", pts_2[0].min(), pts_2[0].max())


        if isinstance(pts_1_raw, torch.Tensor):
            pts_1_raw = [pts_1_raw[i] for i in range(pts_1_raw.shape[0])]
        if isinstance(pts_2, torch.Tensor):
            pts_2 = [pts_2[i] for i in range(pts_2.shape[0])]
        if isinstance(pts_ells_1, torch.Tensor):
            pts_ells_1 = [pts_ells_1[i] for i in range(pts_ells_1.shape[0])]

        img_buffs = None
        if vis:
            img_buffs = self.vis_pts_2D(pts_1_raw, pts_ells_1, grid_h, grid_w, pts_2, mu=mu)

        for ib in range(len(pts_1_raw)):
            pts_1_raw[ib] = pts_1_raw[ib][:, :2]
            pts_2[ib] = pts_2[ib][:, :2]

        pts_1 = [None] * len(pts_1_raw)
        if mu:
            for ib in range(len(pts_1)):
                pts_1_temp = pts_1_raw[ib] + pts_ells_1[ib][4:6, :].transpose(0,1)
                pts_1_temp_1 = torch.clamp(pts_1_temp[:,0], min=0, max=grid_w-1)
                pts_1_temp_2 = torch.clamp(pts_1_temp[:,1], min=0, max=grid_h-1)
                pts_1[ib] = torch.stack([pts_1_temp_1, pts_1_temp_2], dim=1)
        else:
            for ib in range(len(pts_1)):
                pts_1[ib] = pts_1_raw[ib]

        pts_uvb_1 = []
        for ib in range(len(pts_1)):
            pts_uv_1_ib = pts_1[ib].round().to(dtype=int)
            pts_uvb_1_ib = torch.cat([pts_uv_1_ib, torch.ones_like(pts_uv_1_ib[:,[0]])*ib], dim=1)  # N*3 
            pts_uvb_1.append(pts_uvb_1_ib)
        pts_uvb_cat_1 = torch.cat(pts_uvb_1, dim=0)  # BigN * 3
        pts_uvb_cat_1_split = torch.split(pts_uvb_cat_1, 1, 1)
        pts_uvb_cat_1 = pts_uvb_cat_1.transpose(0,1).unsqueeze(0)   # 1*3*N

        pts_uvb_2 = []
        for ib in range(len(pts_2)):
            pts_uv_2_ib = pts_2[ib].round().to(dtype=int)
            pts_uvb_2_ib = torch.cat([pts_uv_2_ib, torch.ones_like(pts_uv_2_ib[:,[0]])*ib], dim=1)  # N*3 
            pts_uvb_2.append(pts_uvb_2_ib)
        pts_uvb_cat_2 = torch.cat(pts_uvb_2, dim=0)  # BigN * 3
        pts_uvb_cat_2_split = torch.split(pts_uvb_cat_2, 1, 1)
        pts_uvb_cat_2 = pts_uvb_cat_2.transpose(0,1).unsqueeze(0)   # 1*3*N

        pts_cat_1 = torch.cat(pts_1, dim=0).transpose(0,1).unsqueeze(0)             # 1*2*N
        pts_cat_1 = pts_cat_1[:,:2]
        pts_ells_cat_1 = torch.cat(pts_ells_1, dim=1).unsqueeze(0)   # 1*4*N
        pts_ells_cat_1 = pts_ells_cat_1[:, :4]
        pts_cat_2 = torch.cat(pts_2, dim=0).transpose(0,1).unsqueeze(0)             # 1*2*N
        pts_cat_2 = pts_cat_2[:,:2]
        if pts_ells_2 is not None:
            pts_ells_cat_2 = torch.cat(pts_ells_2, dim=1).unsqueeze(0)   # 1*4*N
        
        ### griding pts_2
        grid_shape = (len(pts_2), 2, grid_h, grid_w)
        pts_grid_2 = grid_from_concat_flat_func(pts_uvb_cat_2_split, pts_cat_2, grid_shape)

        mask_cat_2 = torch.ones_like(pts_cat_2[:,[0]]).to(dtype=torch.bool)
        grid_mask_shape = (len(pts_2), 1, grid_h, grid_w)
        mask_grid_2 = grid_from_concat_flat_func(pts_uvb_cat_2_split, mask_cat_2, grid_mask_shape)

        ### griding pts_1
        grid_shape = (len(pts_1), 2, grid_h, grid_w)
        pts_grid_1 = grid_from_concat_flat_func(pts_uvb_cat_1_split, pts_cat_1, grid_shape)

        grid_shape = (len(pts_1), 4, grid_h, grid_w)
        pts_ells_grid_1 = grid_from_concat_flat_func(pts_uvb_cat_1_split, pts_ells_cat_1, grid_shape)

        mask_cat_1 = torch.ones_like(pts_cat_1[:,[0]]).to(dtype=torch.bool)
        grid_mask_shape = (len(pts_1), 1, grid_h, grid_w)
        mask_grid_1 = grid_from_concat_flat_func(pts_uvb_cat_1_split, mask_cat_1, grid_mask_shape)

        ### convert dtype
        pts_uvb_cat_1 = pts_uvb_cat_1.float()
        pts_uvb_cat_2 = pts_uvb_cat_2.float()

        # ### center at pts_1
        # inp = PtSampleInGridSigma.apply(pts_uvb_cat_1.contiguous(), pts_cat_1.contiguous(), pts_ells_cat_1.contiguous(), pts_grid_2.contiguous(), mask_grid_2.contiguous(), self.opts.neighbor_range, False)

        # ### center at pts_1
        # # inp_sum = inp.sum()
        # # print("inp", inp.min(), inp.max(), inp.mean())
        # # print("original", inp.numel())
        # inp = inp[inp>0]
        # nel = inp.numel()
        # # print("nzero", nel)
        # if nel == 0:
        #     print("no positive output! ")
        #     inp = 0
        # else:
        #     inp = torch.clamp(inp, min=1e-7)
        #     inp_sum = 1 / ((1 / inp).mean()) 
        #     inp_sum = inp_sum * nel

        

        ### center at pts_2
        inp = PtSampleInGridSigmaGrid.apply(pts_uvb_cat_2.contiguous(), pts_cat_2.contiguous(), pts_ells_grid_1.contiguous(), pts_grid_1.contiguous(), mask_grid_1.contiguous(), self.opts.neighbor_range, False)
        inp = inp / (2*np.pi)

        # print("before", inp.shape)
        inp = inp.sum(dim=1)
        # print(inp)
        # inp, _ = inp.max(dim=1)
        # print("after", inp.shape)
        # print("inp min max", inp.min(), inp.max())

        ### select the covered points
        inp = inp[inp>0]
        # ### set a minimum for inp
        # inp = torch.clamp(inp, min=1e-7)

        ### harmonic mean
        nel = inp.numel()
        inp_sum = 1 / ((1 / inp).mean()) 
        inp_sum = inp_sum * nel
        # ### log likelihood
        # inp_sum = torch.log(inp).sum()

        return inp_sum, img_buffs

    # @torchsnooper.snoop()
    def forward_with_flow(self, depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info, nkern_fname, debug_save_pcd=False):
        if self.opts.debug_input:
            self.debug_flow_input_to_imgs(depth_img_dict_1, depth_img_dict_2)
            # self.debug_flow_input_to_pcds_raw(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info)
            # self.debug_flow_dump_pickle(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info)
        
        # self.timer.log("load_pc3d", 0, True)
        ## ---------------------------------
        ## unpack the depth info
        ## ---------------------------------
        pc3ds_1 = self.load_pc3d(depth_img_dict_1, cam_info)
        pc3ds_2 = self.load_pc3d(depth_img_dict_2, cam_info)

        # self.timer.log("flow_pc3d 1", 0, True)
        ## ---------------------------------
        ## optionally, load PCL_C3D objects after scene flow propagation
        ## ---------------------------------
        K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()

        if self.flow_mode:
            pc3ds_pred_flat_2from1, pc3ds_pred_grid_2from1 = flow_pc3d(pc3ds_1["pred"], flow_dict_1to2["pred"], flow_dict_1to2["mask"], K_cur, 
                                            self.feat_comm, use_normal=self.opts.use_normal, sparse_nml_opts=self.nml_opts, return_stat=self.opts.norm_return_stat)#, timer=self.timer)
            
            # self.timer.log("flow_pc3d 2", 0, True)

            pc3ds_pred_flat_1from2, pc3ds_pred_grid_1from2 = flow_pc3d(pc3ds_2["pred"], flow_dict_2to1["pred"], flow_dict_2to1["mask"], K_cur, 
                                            self.feat_comm, use_normal=self.opts.use_normal, sparse_nml_opts=self.nml_opts, return_stat=self.opts.norm_return_stat)#, timer=self.timer)
            
            # self.timer.log("gen_rand_ell", 0, True)
            ## ---------------------------------
            ## optional: save the pcl_c3d objects to file and generate pcd files from them
            ## ---------------------------------
            # if self.opts.debug_input or debug_save_pcd:
            #     pc3ds_1_from_2 = PCL_C3D()
            #     pc3ds_1_from_2.flat = pc3ds_pred_flat_1from2
            #     pc3ds_1_from_2.grid = pc3ds_pred_grid_1from2
            #     pc3ds_2_from_1 = PCL_C3D()
            #     pc3ds_2_from_1.flat = pc3ds_pred_flat_2from1
            #     pc3ds_2_from_1.grid = pc3ds_pred_grid_2from1
            #     self.debug_flow_input_to_pcds_pcl_c3d(pc3ds_1["pred"], pc3ds_2["pred"], pc3ds_1["gt"], pc3ds_2["gt"], pc3ds_1_from_2, pc3ds_2_from_1)

        ## ---------------------------------
        ## configure length scale of kernels
        ## ---------------------------------
        ell = self.gen_rand_ell()

        # self.timer.log("calc_inn_pc3d", 0, True)
        ## ---------------------------------
        ## calculate inner product
        ## ---------------------------------
        inp_1 = self.calc_inn_pc3d(pc3ds_1["gt"].flat, pc3ds_1["pred"].grid, ell["pred_gt"], nkern_fname)
        inp_2 = self.calc_inn_pc3d(pc3ds_2["gt"].flat, pc3ds_2["pred"].grid, ell["pred_gt"], None)
        inp_total = inp_1 + inp_2

        # print_text = "1:{:.4f}, 2:{:.4f}, ".format(inp_1.item(), inp_2.item() )

        if self.flow_mode:
            if all(n > 0 for n in pc3ds_pred_flat_1from2.nb):
                # self.timer.log("calc_inn_pc3d flow 1", 0, True)

                ### match flowed pcl with gt
                inp_flow_1 = self.calc_inn_pc3d(pc3ds_1["gt"].flat, pc3ds_pred_grid_1from2, ell["pred_gt"], None) # TODO: specify the nkern_fname here
                inp_total += inp_flow_1
                # ### match flowed pcl with pred
                # inp_flow_1 = self.calc_inn_pc3d(pc3ds_1["pred"].flat, pc3ds_pred_grid_1from2, ell["pred_pred"], None) # TODO: specify the nkern_fname here
                # inp_total += inp_flow_1 * self.opts.cross_pred_pred_weight


                # print_text = print_text + "1 from 2:{:.4f}, ".format(inp_flow_1.item() )
            
                if torch.isnan(inp_flow_1).any():
                    self.debug_flow_dump_pickle(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info)
                    raise ValueError("NaN encountered in inp_flow_1")

            if all(n > 0 for n in pc3ds_pred_flat_2from1.nb):
                # self.timer.log("calc_inn_pc3d flow 2", 0, True)

                ### match flowed pcl with gt
                inp_flow_2 = self.calc_inn_pc3d(pc3ds_2["gt"].flat, pc3ds_pred_grid_2from1, ell["pred_gt"], None) # TODO: specify the nkern_fname here
                inp_total += inp_flow_2
                # ### match flowed pcl with pred
                # inp_flow_2 = self.calc_inn_pc3d(pc3ds_2["pred"].flat, pc3ds_pred_grid_2from1, ell["pred_pred"], None) # TODO: specify the nkern_fname here
                # inp_total += inp_flow_2 * self.opts.cross_pred_pred_weight

                # print_text = print_text + "2 from 1:{:.4f}, ".format(inp_flow_2.item() )

                if torch.isnan(inp_flow_2).any():
                    self.debug_flow_dump_pickle(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info)
                    raise ValueError("NaN encountered in inp_flow_2")

            # print(print_text)

        if torch.isnan(inp_total).any():
            self.debug_flow_dump_pickle(depth_img_dict_1, depth_img_dict_2, flow_dict_1to2, flow_dict_2to1, cam_info)
            raise ValueError("NaN encountered in inp_1: {} or inp_2: {}".format( inp_1.item(), inp_2.item() ) )

        # self.timer.log("return inp_total", 0, True)
        return inp_total


    def forward_with_caminfo(self, rgb, depth, depth_gt, depth_mask, depth_gt_mask, nkern_fname, Ts, cam_info):
        
        ## ---------------------------------
        ## load PCL_C3D objects
        ## ---------------------------------
        depth_img_dict = {}

        depth_img_dict["depth"] = depth
        depth_img_dict["depth_mask"] = depth_mask
        depth_img_dict["depth_gt"] = depth_gt
        depth_img_dict["depth_gt_mask"] = depth_gt_mask
        depth_img_dict["rgb"] = rgb

        self.pc3ds = self.load_pc3d(depth_img_dict, cam_info)

        K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()
        batch_size = rgb.shape[0]
        #####################################
        # batch_size = rgb.shape[0]
        
        # K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()

        # uvb_flat_cur = uvb_grid_cur.reshape(batch_size, 3, -1)

        # self.pc3ds = EasyDict()
        # self.pc3ds["gt"] = PCL_C3D()
        # self.pc3ds["pred"] = PCL_C3D()

        # ## rgb to hsv
        # hsv = rgb_to_hsv(rgb, flat=False)           # B*3*H*W
        # hsv_flat = hsv.reshape(batch_size, 3, -1)   # B*3*N

        # feat_comm_grid = {'hsv': hsv}
        # feat_comm_flat = {'hsv': hsv_flat}
        
        # ## generate PCL_C3D object
        # self.pc3ds["gt"] = load_pc3d(self.pc3ds["gt"], depth_gt, depth_gt_mask, xy1_grid_cur, uvb_flat_cur, K_cur, feat_comm_grid, feat_comm_flat, 
        #                                 sparse=True, use_normal=self.opts.use_normal, sparse_nml_opts=self.nml_opts, return_stat=self.opts.norm_return_stat)
        # self.pc3ds["pred"] = load_pc3d(self.pc3ds["pred"], depth, depth_mask, xy1_grid_cur, uvb_flat_cur, K_cur, feat_comm_grid, feat_comm_flat, 
        #                                 sparse=False, use_normal=self.opts.use_normal, dense_nml_op=self.normal_op_dense, return_stat=self.opts.norm_return_stat)

        ## ---------------------------------
        ## optionally, load PCL_C3D objects after transformation
        ## ---------------------------------
        self.flag_cross_frame = Ts is not None and self.seq_frame_n > 1 and self.opts.cross_gt_pred_weight > 0
        self.flag_cross_frame_predpred = Ts is not None and self.seq_frame_n > 1 and self.opts.cross_pred_pred_weight > 0
        if self.flag_cross_frame:
            self.pc3ds["gt_trans_flat"] = transform_pc3d(self.pc3ds["gt"], Ts, self.seq_frame_n, K_cur, batch_size)
        if self.flag_cross_frame_predpred:
            self.pc3ds["pred_trans_flat"] = transform_pc3d(self.pc3ds["pred"], Ts, self.seq_frame_n, K_cur, batch_size)
        
        ## ---------------------------------
        ## configure length scale of kernels
        ## ---------------------------------
        ell = self.gen_rand_ell()

        ## ---------------------------------
        ## calculate inner product
        ## ---------------------------------
        inp = self.calc_inn_pc3d(self.pc3ds["gt"].flat, self.pc3ds["pred"].grid, ell["pred_gt"], nkern_fname)
        inp_total = inp

        if self.flag_cross_frame:
            inp_cross_frame = self.calc_inn_pc3d(self.pc3ds["gt_trans_flat"], self.pc3ds["pred"].grid, ell["pred_gt"], None) # TODO: specify the nkern_fname here
            inp_total = inp_total + inp_cross_frame * self.opts.cross_gt_pred_weight

        if self.flag_cross_frame_predpred:
            inp_predpred = self.calc_inn_pc3d(self.pc3ds["pred_trans_flat"], self.pc3ds["pred"].grid, ell["pred_pred"], None) # TODO: specify the nkern_fname here
            inp_total = inp_total + inp_predpred * self.opts.cross_pred_pred_weight
        
        return inp_total
    
    def calc_inn_pc3d(self, pc3d_flat, pc3d_grid, ell, nkern_fname=None):
        # assert pc3d_flat.feature.keys() == pc3d_grid.feature.keys()

        inp_feat_dict = {}

        for feat in self.feat_inp_self:
            if feat == "hsv":
                inp_feat_dict[feat] = PtSampleInGrid.apply(pc3d_flat.uvb.contiguous(), pc3d_flat.feature[feat].contiguous(), pc3d_grid.feature[feat].contiguous(), pc3d_grid.mask.contiguous(), \
                    self.opts.neighbor_range, ell[feat], False, False, self.opts.ell_basedist) # ignore_ib=False, sqr=False
            elif feat == "xyz":
                if self.opts.use_normal > 0:
                    if nkern_fname is None:
                        inp_feat_dict[feat] = PtSampleInGridWithNormal.apply(pc3d_flat.uvb.contiguous(), pc3d_flat.feature[feat].contiguous(), pc3d_grid.feature[feat].contiguous(), \
                            pc3d_grid.mask.contiguous(), pc3d_flat.feature['normal'], pc3d_grid.feature['normal'], pc3d_flat.feature['nres'], pc3d_grid.feature['nres'], \
                                self.opts.neighbor_range, ell[feat], self.opts.res_mag_max, self.opts.res_mag_min, False, self.opts.norm_in_dist, self.opts.neg_nkern_to_zero, self.opts.ell_basedist, False, None) 
                                # ignore_ib=False, return_nkern=False, filename=None
                    else:
                        inp_feat_dict[feat] = PtSampleInGridWithNormal.apply(pc3d_flat.uvb.contiguous(), pc3d_flat.feature[feat].contiguous(), pc3d_grid.feature[feat].contiguous(), \
                            pc3d_grid.mask.contiguous(), pc3d_flat.feature['normal'], pc3d_grid.feature['normal'], pc3d_flat.feature['nres'], pc3d_grid.feature['nres'], \
                                self.opts.neighbor_range, ell[feat], self.opts.res_mag_max, self.opts.res_mag_min, False, self.opts.norm_in_dist, self.opts.neg_nkern_to_zero, self.opts.ell_basedist, True, nkern_fname)
                else:
                    inp_feat_dict[feat] = PtSampleInGrid.apply(pc3d_flat.uvb.contiguous(), pc3d_flat.feature[feat].contiguous(), pc3d_grid.feature[feat].contiguous(), pc3d_grid.mask.contiguous(), \
                        self.opts.neighbor_range, ell[feat], False, False, self.opts.ell_basedist) # ignore_ib=False, sqr=False
            elif feat == "normal":
                pass
            elif feat == "panop":
                pass
            elif feat == "seman": 
                pass
            else:
                print('{} feature passed in calc_inn_pc3d'.format(feat))
                # raise ValueError("feature {} not recognized".format(feat))
                    
        inp = torch.prod( torch.cat([inp_feat_dict[feat] for feat in inp_feat_dict], dim=0), dim=0).sum()

        if self.opts.log_loss:
            inp = torch.log(inp.clamp(min=1e-8))
            
        return inp

    def get_normal_feature(self):
        return self.pc3ds["gt"].grid.feature['normal'], self.pc3ds["pred"].grid.feature['normal']

if __name__ == "__main__":
    pcl_c3d = init_pcl()
    print(pcl_c3d['flat']["feature"]['xyz'])
    # print(pcl_c3d['flat']["feature"]['abc'])