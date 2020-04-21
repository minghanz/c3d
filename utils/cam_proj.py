import os
import numpy as np 

from torch._six import int_classes

from .dataset_kitti import *
from .cam import *

class CamInfo(nn.Module):

    def __init__(self, K=None, width=None, height=None, xy1_grid=None, uvb_grid=None, P_cam_li=None, in_extr=None, batch_size=None, align_corner=False):
        super(CamInfo, self).__init__()
        '''Valid inputs:
        1. in_extr, batch_size, (optional: align_corner)
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

        self.register_buffer('K', K)
        self.register_buffer('uvb_grid', uvb_grid)
        self.register_buffer('xy1_grid', xy1_grid)
        self.register_buffer('P_cam_li', P_cam_li)      # only P_cam_li is not batched

    def unpack(self):
        return self.K, self.width, self.height, self.xy1_grid, self.uvb_grid

    def scale(self, new_width, new_height, align_corner=False):
        scale_w, scale_h = scale_from_size(old_width=self.width, old_height=self.height, new_width=new_width, new_height=new_height, align_corner=align_corner)
        K_scaled = scale_K(self.K, scale_w, scale_h, torch_mode=True, align_corner=align_corner)
        uvb_grid_scaled = self.uvb_grid[..., :int(new_height), :int(new_width)]

        uv_grid = uvb_grid_scaled[:, :2]
        inv_K_scaled = torch.inverse(K_scaled)
        _, xy1_flat = xy1_from_uv(uv_grid, inv_K_scaled, torch_mode=True)
        xy1_grid = xy1_flat.reshape(xy1_flat.shape[0], 3, uv_grid.shape[2], uv_grid.shape[3] )

        new_cam_info = CamInfo(K_scaled, new_width, new_height, xy1_grid, uvb_grid_scaled, self.P_cam_li)
        return new_cam_info

    def crop(self, xy_crop):
        '''
        xy_crop = (x_start, y_start, x_size, y_size)
        can be a batch (each element of xy_crop is a 1D tensor), but x_size and y_size should be the same
        '''
        xy1_grid_cur = self.xy1_grid
        uvb_grid_cur = self.uvb_grid
        K_cur = self.K

        batch_sep = not(isinstance(xy_crop[0], float) or isinstance(xy_crop[0], int_classes)) 
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

class CamProj(nn.Module):
    def __init__(self, data_root, batch_size=None):
        ## prepare uv1 and xy1 grid
        super(CamProj, self).__init__()
        # self.width = {}
        # self.height = {}
        # self.K = {}
        # self.uvb_flat = {}
        # self.xy1_flat = {}

        self.cam_infos = {}
        
        intr_dict = preload_K(data_root)
        for key in intr_dict:
            self.cam_infos[key] = CamInfo(in_extr=intr_dict[key], batch_size=batch_size)

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

    def prepare_cam_info(self, date_side, xy_crop=None):
        '''
        '''
        cam_info_cur = self.cam_infos[date_side]

        if xy_crop is not None:
            cam_info_cropped = cam_info_cur.crop(xy_crop)
            return cam_info_cropped
        
        return cam_info_cur
    def forward(self):
        pass