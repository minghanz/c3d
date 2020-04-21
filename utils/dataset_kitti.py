import numpy as np 
import os
# from easydict import EasyDict

from .cam import scale_K, K_mat2py, scale_from_size, InExtr

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

'''from bts/c3d_loss.py, bts_pre_intr.py'''
def preload_K(data_root, align_corner=False):
    "Designed for KITTI dataset. Preload intrinsic params, which is different for each date"
    dates = os.listdir(data_root)
    K_dict = {}
    for date in dates:
        ## load cam calib file
        cam_intr_file = os.path.join(data_root, date, 'calib_cam_to_cam.txt')
        intr = read_calib_file(cam_intr_file)
        im_shape = intr["S_rect_02"][::-1].astype(np.int32) ## ZMH: [height, width]

        ## load velo calib file
        cam_lidar_extr_file = os.path.join(data_root, date, 'calib_velo_to_cam.txt')
        extr_li = read_calib_file( cam_lidar_extr_file )

        for side in [2,3]:
            K_dict[(date, side)] = InExtr() # EasyDict()

            P_rect = intr['P_rect_0'+str(side)].reshape(3, 4).astype(np.float32)

            ## intrinsics
            K = P_rect[:, :3]
            K = K_mat2py(K)
            # effect_w = float(im_shape[1] - 1 if align_corner else im_shape[1])
            # effect_h = float(im_shape[0] - 1 if align_corner else im_shape[0])
            scale_w, scale_h = scale_from_size(old_width=im_shape[1], old_height=im_shape[0], align_corner=align_corner)
            K_unit = scale_K(K, scale_w, scale_h, torch_mode=False, align_corner=align_corner)

            K_dict[(date, side)].width = im_shape[1]
            K_dict[(date, side)].height = im_shape[0]
            K_dict[(date, side)].K_unit = K_unit

            ## extrinsics
            T_cam_lidar = np.hstack((extr_li['R'].reshape(3, 3), extr_li['T'][..., np.newaxis]))
            T_cam_lidar = np.vstack((T_cam_lidar, np.array([0, 0, 0, 1.0]))).astype(np.float32)
            R_rect_cam = np.eye(4).astype(np.float32)
            R_rect_cam[:3, :3] = intr['R_rect_00'].reshape(3, 3)

            K_inv = np.linalg.inv(K)
            Kt = P_rect[:, 3:4]
            t = np.dot(K_inv, Kt)
            P_rect_t = np.identity(4).astype(np.float32)
            P_rect_t[:3, 3:4] = t # ZMH: 4*4
            
            P_rect_li = np.dot(P_rect_t, np.dot(R_rect_cam, T_cam_lidar))

            K_dict[(date, side)].P_cam_li = P_rect_li 
    return K_dict
    
'''from bts/bts_pre_intr.py'''
def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points