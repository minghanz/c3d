import os
import numpy as np 

from torch._six import int_classes
import torch
import torch.nn as nn

from .dataset_kitti import *
from .cam import *
from ..utils_general.calib import *

from ..utils_general.timing import Timing

def CamInfo_from_InExs(inex_list):
    """construct cam_info from an InExtr object or a list of them. 
    InExtr is in numpy. CamInfo is in pytorch. """

    ### if input is a single InExtr, construct the list
    if isinstance(inex_list, InExtr):
        inex_list = [inex_list]

    ### make sure all InExtr are compatible to batch
    width0 = inex_list[0].width
    height0 = inex_list[0].height
    assert all(x.width == width0 for x in inex_list)
    assert all(x.height == height0 for x in inex_list)

    ### concatenate and batch
    K_list = [inex.K for inex in inex_list]
    P_cam_li_list = [inex.P_cam_li for inex in inex_list]

    K_batched = np.stack(K_list, axis=0)
    if any(P_cam_li is None for P_cam_li in P_cam_li_list): 
        P_cam_li_batched = None
    elif all(P_cam_li is not None for P_cam_li in P_cam_li_list):
        P_cam_li_batched = np.stack(P_cam_li_list, axis=0)
    else:
        raise ValueError("A subset of P_cam_li_list is None, need further processing. ")

    ### np.ndarray to torch.Tensor
    K_batched = torch.from_numpy(K_batched)
    P_cam_li_batched = torch.from_numpy(P_cam_li_batched) if P_cam_li_batched is not None else None
    
    ### construct CamInfo from batched input
    cam_info = CamInfo_from_K_batched(width0, height0, K_batched, P_cam_li_batched)
    return cam_info


def CamInfo_from_K_batched(width, height, K_batched, P_cam_li_batched=None):
    """ This function needs torch.Tensor input. 
    set_from_intr() assumes a single intr duplicated to the whole batch, while set_from_K_batched() assumes input of a batch of K which could be different. 
    However, the width and height of images should be the same, otherwise uvb_grid and xy1_grid cannot be batched. 
    """

    to_torch = True
    batch_size = K_batched.shape[0]
    device=K_batched.device
    dtype = K_batched.dtype

    uv_grid = gen_uv_grid(width, height, to_torch) # 2*H*W
    uv_grid = uv_grid.to(device=device)
    uv_grid_batched = uv_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) # B*2*H*W

    inv_K_batched = torch.inverse(K_batched)

    uv1_flat, xy1_flat = xy1_from_uv(uv_grid_batched, inv_K_batched, to_torch)  # 3*N

    uv1_grid = uv1_flat.reshape(batch_size, 3, uv_grid.shape[1], uv_grid.shape[2] ) # B*3*H*W
    xy1_grid = xy1_flat.reshape(batch_size, 3, uv_grid.shape[1], uv_grid.shape[2] ) # B*3*H*W
    for ib in range(batch_size):
        uv1_grid[ib, 2, :, :] = ib
    
    if P_cam_li_batched is None:
        P_cam_li_batched = torch.eye(4).to(device=device, dtype=dtype)
        P_cam_li_batched.unsqueeze(0).repeat(batch_size, 1, 1)
    
    cam_info = CamInfo(K=K_batched, width=width, height=height, xy1_grid=xy1_grid, uvb_grid=uv1_grid, P_cam_li=P_cam_li_batched)
    return cam_info

class CamInfo(nn.Module):

    def __init__(self, K=None, width=None, height=None, xy1_grid=None, uvb_grid=None, P_cam_li=None, in_extr=None, batch_size=None, align_corner=False):
        super(CamInfo, self).__init__()
        '''Valid inputs:
        1. in_extr, batch_size, (optional: align_corner)  #09102020: This is now deprecated. The in_extr accepted here is now InExtrKunit object. Use CamInfo_from_InExs to initialize a CamInfo from the new InExtr. 
        2. K, width, height, xy1_grid, uvb_grid
        '''
        if in_extr is not None:
            assert K is None and width is None and height is None and xy1_grid is None and uvb_grid is None and P_cam_li is None and batch_size is not None
            uvb_grid, xy1_grid, self.width, self.height, K = set_from_intr(in_extr.width, in_extr.height, in_extr.K_unit, batch_size, align_corner=align_corner)
            self.batch_size = batch_size
            P_cam_li = torch.from_numpy(in_extr.P_cam_li)
        else:
            self.width = int(width)     ## don't use tensor of int type
            self.height = int(height)
            self.batch_size = xy1_grid.shape[0]

        self.register_buffer('K', K.to(dtype=torch.float32))
        self.register_buffer('uvb_grid', uvb_grid.to(dtype=torch.float32))
        self.register_buffer('xy1_grid', xy1_grid.to(dtype=torch.float32))
        self.register_buffer('P_cam_li', P_cam_li.to(dtype=torch.float32))      # only P_cam_li is not batched

    def unpack(self):
        return self.K, self.width, self.height, self.xy1_grid, self.uvb_grid

    # def scale(self, new_width, new_height, align_corner=False):
    #     if not isinstance(new_width, (float, np.float, int_classes)):
    #         assert isinstance(new_width, torch.Tensor)
    #         new_width = new_width[0]
    #         new_height = new_height[0]
    #         assert isinstance(new_width, (float, np.float, int_classes))
    def scale(self, scale_op):
        if isinstance(scale_op.new_height, torch.Tensor):
            scale_op = extract_single_op(scale_op)
        new_width = scale_op.new_width
        new_height = scale_op.new_height
        align_corner = scale_op.align_corner
        if scale_op.scale is not None and scale_op.scale != 0 and new_width is not None and new_height is not None:
            assert new_width == int(self.width * scale_op.scale)
            assert new_height == int(self.height * scale_op.scale)


        # scale_w, scale_h = scale_from_size(old_width=self.width, old_height=self.height, new_width=new_width, new_height=new_height, align_corner=align_corner)
        K_scaled = scale_K(self.K, old_width=self.width, old_height=self.height, new_width=new_width, new_height=new_height, torch_mode=True, align_corner=align_corner)
        uvb_grid_scaled = self.uvb_grid[..., :int(new_height), :int(new_width)]

        uv_grid = uvb_grid_scaled[:, :2]
        inv_K_scaled = torch.inverse(K_scaled)
        _, xy1_flat = xy1_from_uv(uv_grid, inv_K_scaled, torch_mode=True)
        xy1_grid = xy1_flat.reshape(xy1_flat.shape[0], 3, uv_grid.shape[2], uv_grid.shape[3] )

        new_cam_info = CamInfo(K_scaled, new_width, new_height, xy1_grid, uvb_grid_scaled, self.P_cam_li)
        return new_cam_info

    def size(self):
        return self.uvb_grid.size()

    def crop(self, xy_crop):
        '''
        xy_crop = (x_start, y_start, x_size, y_size)
        can be a batch (each element of xy_crop is a 1D tensor), but x_size and y_size should be the same
        '''
        xy1_grid_cur = self.xy1_grid
        uvb_grid_cur = self.uvb_grid
        K_cur = self.K

        batch_sep = not isinstance(xy_crop[0], (float, np.float32, int_classes))
        if batch_sep:
            assert isinstance(xy_crop[0], torch.Tensor)
        if batch_sep:
            batch_size = xy_crop[0].shape[0]
            ## In case the batch_size is not constant as originally set
            if batch_size != self.batch_size:
                xy1_grid_cur = self.xy1_grid[:batch_size]
                uvb_grid_cur = self.uvb_grid[:batch_size]
                K_cur = self.K[:batch_size]

        ## crop the grids and modify intrinsics to match the cropped image
        if batch_sep:
            x_size = int(xy_crop[2][0])
            y_size = int(xy_crop[3][0])
            uvb_grid_crop = uvb_grid_cur[:,:,:y_size, :x_size]
            xy1_grid_crop = torch.zeros((batch_size, 3, y_size, x_size), dtype=torch.float32)
            K_crop = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
            for ib in range(batch_size):
                x_start = int(xy_crop[0][ib])
                y_start = int(xy_crop[1][ib])
                xy1_grid_crop[ib] = xy1_grid_cur[ib,:,y_start:y_start+y_size, x_start:x_start+x_size]
                K_crop[ib] = crop_K(K_cur[ib], x_start, y_start, torch_mode=True)
        else:
            x_size = int(xy_crop[2])
            y_size = int(xy_crop[3])
            x_start = int(xy_crop[0])
            y_start = int(xy_crop[1])
            uvb_grid_crop = uvb_grid_cur[:,:,:y_size, :x_size]
            xy1_grid_crop = xy1_grid_cur[:,:,y_start:y_start+y_size, x_start:x_start+x_size]
            K_crop = crop_K(K_cur, x_start, y_start, torch_mode=True)
        
        cam_info = CamInfo(K_crop, x_size, y_size, xy1_grid_crop, uvb_grid_crop, self.P_cam_li)
        return cam_info

    def lidar_to_depth(self, lidar_pts):
        batch_size = self.K.shape[0]
        if lidar_pts.ndim == 2:
            lidar_pts = lidar_pts.unsqueeze(0)
        assert lidar_pts.shape[0] == batch_size, "{}, {}".format(lidar_pts.shape, self.K.shape)

        dep_img_list = []
        for ib in range(batch_size):
            dep_img = lidar_to_depth(lidar_pts[ib], self.P_cam_li[ib], K_unit=None, K_ready=self.K[ib], im_shape=(self.height, self.width), torch_mode=True)    #H*W
            dep_img_list.append(dep_img)
        dep_img_batch = torch.stack(dep_img_list, dim=0)    # B*H*W
        
        return dep_img_batch

def seq_ops_on_cam_info(cam_info, cam_ops_list):
    for cam_op in cam_ops_list:
        if isinstance(cam_op, CamCrop):
            cam_info = cam_info.crop(cam_op)
        elif isinstance(cam_op, CamScale):
            cam_info = cam_info.scale(cam_op)
        elif isinstance(cam_op, CamRotate):
            raise ValueError("Not implemented yet! ")
        else:
            raise ValueError("op not recognized!", type(cam_op))
    return cam_info

def batch_cam_infos(list_of_cam_info):
    ### we need the image shape in all cam_infos to be the same, other properties can be different
    width = list_of_cam_info[0].width
    height = list_of_cam_info[0].height
    assert all(width == cam_info_i.width for cam_info_i in list_of_cam_info)
    assert all(height == cam_info_i.height for cam_info_i in list_of_cam_info)

    K_batched = torch.cat([cam_info_i.K for cam_info_i in list_of_cam_info], dim=0)
    xy1_grid_batched = torch.cat([cam_info_i.xy1_grid for cam_info_i in list_of_cam_info], dim=0)
    uvb_grid_batched = torch.cat([cam_info_i.uvb_grid for cam_info_i in list_of_cam_info], dim=0)
    for i in range(uvb_grid_batched.shape[0]):
        uvb_grid_batched[i,2] = i
    P_cam_li_batched = torch.cat([cam_info_i.P_cam_li for cam_info_i in list_of_cam_info], dim=0)
    
    batched_cam_info = CamInfo(K_batched, width, height, xy1_grid_batched, uvb_grid_batched, P_cam_li_batched)
    return batched_cam_info

class CamProj(nn.Module):
    def __init__(self, dataset_reader, batch_size=None):
        ## prepare uv1 and xy1 grid
        super(CamProj, self).__init__()
        # self.width = {}
        # self.height = {}
        # self.K = {}
        # self.uvb_flat = {}
        # self.xy1_flat = {}

        self.cam_infos = {}
        
        self.dataset_reader = dataset_reader
        if 'calib' in dataset_reader.ffinder.preload_ftypes:
            intr_dict = dataset_reader.preload_dict['calib']
            # for key in intr_dict:
            #     intr_dict_key0 = key
            #     break
            # self.intr_dict_key_levels = list(x for x in intr_dict_key0._fields if getattr(intr_dict_key0,x) is not None)
            # intr_dict = preload_K(data_root)
            for key in intr_dict:
                inex_list = [intr_dict[key]] * batch_size
                self.cam_infos[key] = CamInfo_from_InExs(inex_list)

                # uvb_grid, xy1_grid, self.width[key], self.height[key], K = set_from_intr(intr_dict[key].width, intr_dict[key].height, intr_dict[key].K_unit, batch_size)
                # self.register_buffer("uvb_grid_{}_{}".format(key[0], key[1]), uvb_grid)
                # self.register_buffer("xy1_grid_{}_{}".format(key[0], key[1]), xy1_grid)
                # self.register_buffer("K_{}_{}".format(key[0], key[1]), K)

        self.batch_size = batch_size

    # def K(self, date_side):
    #     date = date_side[0]
    #     side = date_side[1]
    #     return self.__getattr__("K_{}_{}".format(date, side))
    # def uvb_grid(self, date_side):
    #     date = date_side[0]
    #     side = date_side[1]
    #     return self.__getattr__("uvb_grid_{}_{}".format(date, side))
    #     # return getattr(self, "uvb_grid_{}_{}".format(date, side))
    # def xy1_grid(self, date_side):
    #     date = date_side[0]
    #     side = date_side[1]
    #     return self.__getattr__("xy1_grid_{}_{}".format(date, side))

    def prepare_cam_info(self, key=None, xy_crop=None, intr=None):
        '''
        '''
        if key is not None:
            cam_info_cur = self.cam_infos[key]
        else:
            assert intr is not None
            cam_info_cur = CamInfo(in_extr=intr, batch_size=self.batch_size)

        if xy_crop is not None:
            cam_info_cropped = cam_info_cur.crop(xy_crop)
            return cam_info_cropped
        
        return cam_info_cur
    def forward(self):
        pass