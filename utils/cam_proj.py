import os
import numpy as np 

from .dataset_kitti import *
from .cam import *

class CamInfo:
    def __init__(self, K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur):
        self.K_cur = K_cur
        self.width_cur = width_cur
        self.height_cur = height_cur
        self.xy1_grid_cur = xy1_grid_cur
        self.uvb_grid_cur = uvb_grid_cur

    def unpack(self):
        return self.K_cur, self.width_cur, self.height_cur, self.xy1_grid_cur, self.uvb_grid_cur

class CamProj(nn.Module):
    def __init__(self, data_root, batch_size=None, seq_frame_n=1):
        ## prepare uv1 and xy1 grid
        super(CamProj, self).__init__()
        self.width = {}
        self.height = {}
        self.seq_frame_n = seq_frame_n
        # self.K = {}
        # self.uvb_flat = {}
        # self.xy1_flat = {}
        
        intr_dict = preload_K(data_root)
        for intr in intr_dict:
            uvb_grid, xy1_grid, self.width[intr], self.height[intr], K = set_from_intr(intr_dict[intr].width, intr_dict[intr].height, intr_dict[intr].K_unit, batch_size)
            self.register_buffer("uvb_grid_{}_{}".format(intr[0], intr[1]), uvb_grid)
            self.register_buffer("xy1_grid_{}_{}".format(intr[0], intr[1]), xy1_grid)
            self.register_buffer("K_{}_{}".format(intr[0], intr[1]), K)

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

    def prepare_cam_info(self, date_side, xy_crop, intr, batch_size, device):
        if intr is not None:
            ## use input intrinsics to generate needed parameters now
            uvb_grid_cur, xy1_grid_cur, width_cur, height_cur, K_cur = set_from_intr(intr.width, intr.height, intr.K_unit, batch_size, device=device)
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
            xy1_grid_crop = torch.zeros((batch_size, 3, y_size, x_size), device=device, dtype=torch.float32)
            K_crop = torch.zeros((batch_size, 3, 3), device=device, dtype=torch.float32)
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
        
        cam_info = CamInfo(K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur)
        return cam_info
    def forward(self):
        pass