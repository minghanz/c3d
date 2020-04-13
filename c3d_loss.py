import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
import os
import sys
# script_path = os.path.dirname(__file__)
# sys.path.append(os.path.join(script_path, '../../pytorch-unet'))
# from geometry import rgb_to_hsv
from .utils.color import rgb_to_hsv

# sys.path.append(os.path.join(script_path, '../../monodepth2'))
# from cvo_utils import *
from .cvo_funcs import *
from .utils.geometry import *
from .utils.pc3d import *
from .utils.cam import *

import argparse

class PCL_C3D_Flat:
    def __init__(self):
        self.uvb = None
        self.nb = []
        self.feature = EasyDict()

class PCL_C3D_Grid:
    def __init__(self):
        self.mask = None
        self.feature = EasyDict()

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
    batch_size = mask_grid.shape[0]

    ## grid features
    pcl_c3d.grid.mask = mask_grid
    for feat in feat_grid:
        pcl_c3d.grid.feature[feat] = feat_grid[feat]   # B*C*H*W

    ## flat features
    pcl_c3d.flat.uvb = []
    for feat in feat_flat:
        pcl_c3d.flat.feature[feat] = []

    mask_flat = mask_grid.reshape(batch_size, 1, -1)
    for ib in range(batch_size):
        mask_vec = mask_flat[ib, 0]
        pcl_c3d.flat.nb.append(int(mask_vec.sum()))
        pcl_c3d.flat.uvb.append(uvb_flat[[ib]][:,:, mask_vec])
        for feat in feat_flat:
            pcl_c3d.flat.feature[feat].append(feat_flat[feat][[ib]][:,:, mask_vec])      # 1*C*N
    
    pcl_c3d.flat.uvb = torch.cat(pcl_c3d.flat.uvb, dim=2)
    for feat in feat_flat:
        pcl_c3d.flat.feature[feat] = torch.cat(pcl_c3d.flat.feature[feat], dim=2)

    return pcl_c3d

def load_pc3d(pcl_c3d, depth_grid, mask_grid, xy1_grid, uvb_flat, K_cur, feat_comm_grid, feat_comm_flat, sparse, use_normal, sparse_nml_opts=None, dense_nml_op=None):
    assert not (sparse_nml_opts is None and dense_nml_op is None)
    """sparse is a bool
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
        normal_flat, nres_flat = calc_normal(pcl_c3d.flat.uvb, xyz_grid, mask_grid, sparse_nml_opts.normal_nrange, sparse_nml_opts.ignore_ib, sparse_nml_opts.min_dist_2)
        ## TODO: How to deal with points with no normal?
        uvb_split = pcl_c3d.flat.uvb.to(dtype=torch.long).squeeze(0).transpose(0,1).split(1,dim=1) # a tuple of 3 elements of tensor N*1, only long/byte/bool tensors can be used as indices
        grid_xyz_shape = xyz_grid.shape
        normal_grid = grid_from_concat_flat_func(uvb_split, normal_flat, grid_xyz_shape)
        nres_grid = grid_from_concat_flat_func(uvb_split, nres_flat, grid_xyz_shape)

        pcl_c3d.flat.feature['normal'] = normal_flat
        pcl_c3d.flat.feature['nres'] = nres_flat
        pcl_c3d.grid.feature['normal'] = normal_grid
        pcl_c3d.grid.feature['nres'] = nres_grid

    return pcl_c3d

def transform_pc3d(pcl_c3d, Ts, seq_n, K_cur, batch_n):

    ## conduct pose transform if needed
    ## need to transform: flat.uvb, flat.feature['xyz'], flat.feature['normal']
    ## no need to transform grid features
    
    assert batch_n % seq_n == 0    # mode==0
    n_group = batch_n // seq_n

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
        uvb[:, :2] = uvb[:, :2] / uvb[:, [2]] - 1
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
    

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def preload_K(data_root):
    '''Designed for KITTI dataset. Preload intrinsic params, which is different for each date
    K_dict[(date, side)]: a dict with attrbutes: width, height, K_unit
    '''
    dates = os.listdir(data_root)
    K_dict = {}
    for date in dates:
        cam_intr_file = os.path.join(data_root, date, 'calib_cam_to_cam.txt')
        intr = read_calib_file(cam_intr_file)
        im_shape = intr["S_rect_02"][::-1].astype(np.int32) ## ZMH: [height, width]
        for side in [2,3]:
            K_dict[(date, side)] = EasyDict()
            P_rect = intr['P_rect_0'+str(side)].reshape(3, 4)
            K = P_rect[:, :3]
            K_unit = np.identity(3).astype(np.float32)
            K_unit[0] = K[0] / float(im_shape[1])
            K_unit[1] = K[1] / float(im_shape[0])

            K_dict[(date, side)].width = im_shape[1]
            K_dict[(date, side)].height = im_shape[0]
            K_dict[(date, side)].K_unit = K_unit
    return K_dict

class C3DLoss(nn.Module):
    def __init__(self, data_root, batch_size=None, seq_frame_n=1):
        super(C3DLoss, self).__init__()
        self.width = {}
        self.height = {}
        self.seq_frame_n = seq_frame_n
        # self.K = {}
        # self.uvb_flat = {}
        # self.xy1_flat = {}
        
        intr_dict = preload_K(data_root)
        for intr in intr_dict:
            uvb_grid, xy1_grid, self.width[intr], self.height[intr], K = set_from_intr(intr_dict[intr], batch_size)
            self.register_buffer("uvb_grid_{}_{}".format(intr[0], intr[1]), uvb_grid)
            self.register_buffer("xy1_grid_{}_{}".format(intr[0], intr[1]), xy1_grid)
            self.register_buffer("K_{}_{}".format(intr[0], intr[1]), K)

        self.feat_inp_self = ["xyz", "hsv"]
        self.feat_inp_cross = ["xyz", "hsv"]

        self.normal_op_dense = NormalFromDepthDense()
        self.batch_size = batch_size

    def K(self, date_side):
        date = date_side[0]
        side = date_side[1]
        return self.__getattr__("K_{}_{}".format(date, side))
    def uvb_grid(self, date_side):
        date = date_side[0]
        side = date_side[1]
        return self.__getattr__("uvb_grid_{}_{}".format(date, side))
        # return getattr(self, "uvb_grid_{}_{}".format(date, side))
    def xy1_grid(self, date_side):
        date = date_side[0]
        side = date_side[1]
        return self.__getattr__("xy1_grid_{}_{}".format(date, side))

    def parse_opts(self, inputs=None):
        parser = argparse.ArgumentParser(description='Options for continuous 3D loss')

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

        parser.add_argument("--neighbor_range",        type=int, default=2,
                            help="neighbor range when calculating inner product")
        parser.add_argument("--normal_nrange",         type=int, default=5,
                            help="neighbor range when calculating normal direction on sparse point cloud")
        parser.add_argument("--pred_pred_weight",        type=float, default=0,
                            help="weight of c3d loss between cross-frame predictions relative to gt_pred_weight as 1. You may want to set to less than 1 because predictions are denser than gt.")

        self.opts, rest = parser.parse_known_args(args=inputs) # inputs can be None, in which case _sys.argv[1:] are parsed

        self.opts.ell_min = {}
        self.opts.ell_rand = {}
        for i, ell_item in enumerate(self.opts.ell_keys):
            self.opts.ell_min[ell_item] = self.opts.ell_values_min[i]
            self.opts.ell_rand[ell_item] = self.opts.ell_values_rand[i]

        if self.opts.pred_pred_weight > 0:
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

    def forward(self, rgb, depth, depth_gt, depth_mask, depth_gt_mask, date_side=None, xy_crop=None, intr=None, nkern_fname=None, Ts=None):
        """
        rgb: B*3*H*W
        depth, depth_gt, depth_mask, depth_gt_mask: B*1*H*W
        """
        assert date_side is not None or intr is not None
        date_side = (date_side[0][0], int(date_side[1][0]) ) # originally it is a list. Take the first since the mini_batch share the same intrinsics. 

        batch_size = rgb.shape[0]       ## if drop_last is False in Sampler/DataLoader, then the batch_size is not constant. 
        if intr is not None:
            ## use input intrinsics to generate needed parameters now
            uvb_grid_cur, xy1_grid_cur, width_cur, height_cur, K_cur = set_from_intr(intr, batch_size, device=rgb.device)
        else:
            ## retrieve from preloaded parameters
            xy1_grid_cur = self.xy1_grid(date_side)
            uvb_grid_cur = self.uvb_grid(date_side)
            width_cur = self.width[date_side]
            height_cur = self.height[date_side]
            K_cur = self.K(date_side)

        ## In case the batch_size is not constant as originally set
        if batch_size != self.batch_size:
            xy1_grid_cur = xy1_grid_cur[:batch_size]
            uvb_grid_cur = uvb_grid_cur[:batch_size]
            K_cur = K_cur[:batch_size]

        ## crop the grids and modify intrinsics to match the cropped image
        if xy_crop is not None:
            x_size = xy_crop[2][0]
            y_size = xy_crop[3][0]
            uvb_grid_crop = uvb_grid_cur[:batch_size,:,:y_size, :x_size]
            xy1_grid_crop = torch.zeros((batch_size, 3, y_size, x_size), device=rgb.device, dtype=torch.float32)
            K_crop = torch.zeros((batch_size, 3, 3), device=rgb.device, dtype=torch.float32)
            for ib in range(batch_size):
                x_start = xy_crop[0][ib]
                y_start = xy_crop[1][ib]
                x_size = xy_crop[2][ib]
                y_size = xy_crop[3][ib]
                xy1_grid_crop[ib] = xy1_grid_cur[ib,:,y_start:y_start+y_size, x_start:x_start+x_size]
                K_crop[ib] = K_cur[ib]
                K_crop[ib, 0, 2] = K_crop[ib, 0, 2] - x_start
                K_crop[ib, 1, 2] = K_crop[ib, 1, 2] - y_start
            K_cur = K_crop
            width_cur = x_size      # cropped width and height are deterministic
            height_cur = y_size
            xy1_grid_cur = xy1_grid_crop
            uvb_grid_cur = uvb_grid_crop

        uvb_flat_cur = uvb_grid_cur.reshape(batch_size, 3, -1)

        pc3ds = EasyDict()
        # pc3ds["gt"] = init_pc3d()
        # pc3ds["pred"] = init_pc3d()
        pc3ds["gt"] = PCL_C3D()
        pc3ds["pred"] = PCL_C3D()

        ## rgb to hsv
        hsv = rgb_to_hsv(rgb, flat=False)           # B*3*H*W
        hsv_flat = hsv.reshape(batch_size, 3, -1)   # B*3*N

        feat_comm_grid = {}
        feat_comm_grid['hsv'] = hsv
        feat_comm_flat = {}
        feat_comm_flat['hsv'] = hsv_flat
        
        ## generate PCL_C3D object
        pc3ds["gt"] = load_pc3d(pc3ds["gt"], depth_gt, depth_gt_mask, xy1_grid_cur, uvb_flat_cur, K_cur, feat_comm_grid, feat_comm_flat, sparse=True, use_normal=self.opts.use_normal, sparse_nml_opts=self.nml_opts)
        pc3ds["pred"] = load_pc3d(pc3ds["pred"], depth, depth_mask, xy1_grid_cur, uvb_flat_cur, K_cur, feat_comm_grid, feat_comm_flat, sparse=False, use_normal=self.opts.use_normal, dense_nml_op=self.normal_op_dense)

        self.flag_cross_frame = Ts is not None and self.seq_frame_n > 1
        self.flag_cross_frame_predpred = self.flag_cross_frame and self.opts.pred_pred_weight > 0
        if self.flag_cross_frame:
            pc3ds["gt_trans_flat"] = transform_pc3d(pc3ds["gt"], Ts, self.seq_frame_n, K_cur, batch_size)
        if self.flag_cross_frame_predpred:
            pc3ds["pred_trans_flat"] = transform_pc3d(pc3ds["pred"], Ts, self.seq_frame_n, K_cur, batch_size)
        
        ## random ell
        ell = {}
        for key in self.opts.ell_keys:
            ell[key] = self.opts.ell_min[key] + np.abs(self.opts.ell_rand[key]* np.random.normal()) 
        
        if self.flag_cross_frame_predpred:
            ell_predpred = {}
            for key in self.opts.ell_keys:
                ell_predpred[key] = self.opts.ell_min_predpred[key] + np.abs(self.opts.ell_rand_predpred[key]* np.random.normal()) 

        ## calculate inner product
        inp = self.calc_inn_pc3d(pc3ds["gt"].flat, pc3ds["pred"].grid, ell, nkern_fname)
        inp_total = inp

        if self.flag_cross_frame:
            inp_cross_frame = self.calc_inn_pc3d(pc3ds["gt_trans_flat"], pc3ds["pred"].grid, ell, None) # TODO: specify the nkern_fname here
            inp_total = inp_total + inp_cross_frame
        if self.flag_cross_frame_predpred:
            inp_predpred = self.calc_inn_pc3d(pc3ds["pred_trans_flat"], pc3ds["pred"].grid, ell_predpred, None) # TODO: specify the nkern_fname here
            inp_total = inp_total + inp_predpred * self.opts.pred_pred_weight
        
        return inp_total
    
    def calc_inn_pc3d(self, pc3d_flat, pc3d_grid, ell, nkern_fname=None):
        assert pc3d_flat.feature.keys() == pc3d_grid.feature.keys()

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
                raise ValueError("feature {} not recognized".format(feat))
                    
        inp = torch.prod( torch.cat([inp_feat_dict[feat] for feat in inp_feat_dict], dim=0), dim=0).sum()

        return inp

    

if __name__ == "__main__":
    pcl_c3d = init_pcl()
    print(pcl_c3d['flat']["feature"]['xyz'])
    # print(pcl_c3d['flat']["feature"]['abc'])