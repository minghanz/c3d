import torch
import numpy as np
import torch.nn as nn
import os
from easydict import EasyDict

'''from bts/c3d_loss.py'''
def gen_uv_grid(width, height, torch_mode):
    """
    return: uv_coords(2*H*W)
    """
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
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
    if torch_mode:
        uv_flat = uv_grid.reshape(2, -1) # 2*N
        dummy_ones = torch.ones((1, uv_flat.shape[1]), dtype=uv_flat.dtype, device=uv_flat.device) # 1*N
        uv1_flat = torch.cat((uv_flat, dummy_ones), dim=0) # 3*N
        xy1_flat = torch.matmul(inv_K, uv1_flat)
    else:
        uv_flat = uv_grid.reshape(2, -1) # 2*N
        dummy_ones = torch.ones((1, uv_flat.shape[1]), dtype=np.float32)
        uv1_flat = np.concatenate((uv_flat, dummy_ones), axis=0) # 3*N
        xy1_flat = np.matmul(inv_K, uv1_flat)

    return uv1_flat, xy1_flat

'''from bts/c3d_loss.py'''
def set_from_intr(intr, batch_size, device=None):

    to_torch = True
    uv_grid = gen_uv_grid(intr.width, intr.height, to_torch) # 2*H*W

    K = intr.K_unit.copy()
    K[0,:] = K[0,:] * float(intr.width)
    K[1,:] = K[1,:] * float(intr.height)
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

    width_cur = intr.width
    height_cur = intr.height

    return uvb_grid, xy1_grid, width_cur, height_cur, K

'''from bts/c3d_loss.py'''
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

'''from bts/c3d_loss.py'''
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
            uvb_grid, xy1_grid, self.width[intr], self.height[intr], K = set_from_intr(intr_dict[intr], batch_size)
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
            uvb_grid_cur, xy1_grid_cur, width_cur, height_cur, K_cur = set_from_intr(intr, batch_size, device=device)
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