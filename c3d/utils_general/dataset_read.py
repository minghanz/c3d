"""
This file is to fast read out the data from file. The wanted data fields are common, so that dataloader later can benefit a unified interface. 
Data type to read: 
image, depth image, lidar, depth projection by lidar, transformation of an image (T_img), calibration info
"""

import os
import numpy as np 
from PIL import Image
# import cv2
from .dataset_find import DataFinderKITTI, DataFinderWaymo, DataFinderVKITTI2

from .io import load_velodyne_points, read_calib_file
from .calib import InExtr, K_mat2py, scale_K

class DataReader:
    """
    We assume that data are stored in two style. 
    1. Each file contains one item of data. They are the ftypes in ffinder.ftypes but not in ffinder.preset_ftypes. They are read through read_direct()
    2. Each file contains multiple items of data, or contains a single data but could be shared by multiple frames. They are the ftypes in ffinder.preset_ftypes. 
    2.1. It implies that the number of files are less than number of data items, therefore we can choose to preload them, so that we can reduce the operation of reading from files. 
    2.2. Ftypes in ffinder.preload_types are preloaded through preload_common_info(), and later use read_with_preload() to retrieve. 
    2.3. Ftypes not in ffinder.preload_types are read by opening the file every time, using read_wo_preload(). 
    2.3.1. It is different from read_direct() because in this case, the file contains multiple items of data. Extra processing is needed to figure out which part is wanted. 

    * The script is not ready for the case when ftypes in ffinder.preset_ftypes have zero or more than 1 levels in infile_level_name. NOTE: infile_level_name is now a list. 7/29/2020
    * The script is not ready for the case when a missing level in ntp does not mean it is shared at that level. 
    """
    def __init__(self, dataset, data_root):
        self.dataset_name = dataset
        if dataset == "kitti":
            self.ffinder = DataFinderKITTI(data_root=data_root)
        elif dataset == 'waymo':
            self.ffinder = DataFinderWaymo(data_root=data_root)
        elif dataset == 'vkitti2':
            self.ffinder = DataFinderVKITTI2(data_root=data_root)
        else:
            raise ValueError("dataset {} not recognized".format(self.dataset_name))

        self.preload_common_info()

    def preload_common_info(self):
        """preload the ftypes in ffinder.preload_ftypes. """
        self.preload_dict = {}
        for ftype in self.ffinder.preload_ftypes:
            self.preload_dict[ftype] = self.data_preload(ftype)

    def data_preload(self, ftype):
        """preload the data of a ftype into a dict, by looping over all filepaths."""
        assert ftype in self.ffinder.preload_ftypes

        data_dict = {}
        for fname_key, fnames in self.ffinder.fnames_preload[ftype].items():
            self.read_preset(fnames, ftype, data_dict)

        return data_dict
        
    def read_from_ntp(self, ntp, ftype):
        """The common interface to retrieve data of arbitrary ftype, given ntp. """
        if ftype not in self.ffinder.preset_ftypes:
            data = self.read_direct(ntp, ftype)
        else:
            if ftype in self.ffinder.preload_ftypes:
                data = self.read_with_preload(ntp, ftype)
            else:
                data = self.read_wo_preload(ntp, ftype)
        return data

    def read_with_preload(self, ntp, ftype):
        """If a ftype is already preloaded, we can retrieve the data from preload_dict. 
        We just need to convert the querying ntp to the ntp of the ftype wanted. """
        assert ftype in self.ffinder.preload_ftypes

        ### convert the querying ntp to the ntp of the ftype wanted
        new_ntp = self.ffinder.ntp_ftype_convert(ntp, ftype)

        ### retrieve the data from preloaded dict
        data = self.preload_dict[ftype][new_ntp]
        return data

    def read_wo_preload(self, ntp, ftype):
        """If a ftype is not preloaded but is a member of preset_ftypes, we read it from file at-need. 
        Note that these three things could be different: 
        ntp from querying file (e.g. an image, likely the most detailed), ntp of target file (e.g. calib file, likely coarse), ntp of target file and its infile variations (e.g. a certain camera in the calib file). 
        Here we only find out the target file ntp from querying ntp, so that we know which file to read, and leave remaining levels to be handled by read_preset() """
        assert ftype in self.ffinder.preset_ftypes

        ### find out which file to read
        fnames = self.ffinder.fname_from_ntp(ntp, ftype)
        
        ### separate out the remaining levels to be handled by read_preset()
        if isinstance(fnames, list):
            ntp_cur = self.ffinder.ntp_from_fname(fnames[0], ftype)       # fnames is a list, any one of the element work
        else:
            ntp_cur = self.ffinder.ntp_from_fname(fnames, ftype)          # fnames is a path
        extra_dict = self.ffinder.find_extra_dict_in_ntp(ntp, ntp_cur)
        
        ### read the file and retrieve the part we want
        data = self.read_preset(fnames, ftype, data_dict=None, **extra_dict)
        return data
            
    def read_preset(self, fnames, ftype, data_dict=None, **kwargs):
        """Read a file of ftypes from preset_ftypes. 
        kwargs could specify which part of data in the file is wanted, otherwise all is returned in a dict if there are multiple items in the file.
        If data_dict is given, it is updated with the read data, otherwise the read data is returned. 
        """
        assert ftype in self.ffinder.preset_ftypes

        ### read the file of ftypes from preset_ftypes
        preset = self.preset_func(fnames, ftype)

        ### get the ntp corresponding to this file. It may need to append some field later, if the file contains multiple items.
        if isinstance(fnames, list):
            level_ntp = self.ffinder.ntp_from_fname(fnames[0], ftype)       # fnames is a list, any one of the element work
        else:
            level_ntp = self.ffinder.ntp_from_fname(fnames, ftype)          # fnames is a path

        ### read all items from the file or a single item. 
        exist_single = self.ffinder.infile_level_name[ftype] is None                         ### the file only contains a single item of data
        query_single = kwargs != {} and all( x in kwargs for x in self.ffinder.infile_level_name[ftype])      ### kwargs specifies which item is wanted
        return_single = exist_single or query_single                                         ### otherwise more than one items will be returned

        if return_single:
            ### add a single item to the dict or return a single InExtr object
            if exist_single:
                ### no need to augment
                data_cur = self.one_from_preset(preset, ftype)
                ntp_cur = level_ntp
            else:
                ### if kwargs exists, convert it to the useful level which is ffinder.infile_level_name
                # finfo_item = kwargs[self.ffinder.infile_level_name[ftype]] ### this works when infile_level_name is a single item
                needed_kwargs = {}
                for x in self.ffinder.infile_level_name[ftype]:
                    needed_kwargs[x] = kwargs[x]
                # data_cur = self.one_from_preset(preset, ftype, finfo_item)
                data_cur = self.one_from_preset(preset, ftype, needed_kwargs)
                ntp_cur = level_ntp._replace(**needed_kwargs)

            if data_dict is None:
                ### return the value itself
                data_dict = data_cur
            else:
                ### add a single item to existing dict
                data_dict[ntp_cur] = data_cur
        else:
            ### add multiple items to the dict or return a dict of the new items
            if data_dict is None:
                data_dict = {}

            finfo_list_ftype = self.ffinder.get_finfo_list_at_level(self.ffinder.finfos, level_ntp, query_level=self.ffinder.infile_level_name[ftype])
            for finfo_item in finfo_list_ftype:
                needed_kwargs = {}
                for i, level in enumerate(self.ffinder.infile_level_name[ftype]):
                    needed_kwargs[level] = finfo_item[i]
                data_cur = self.one_from_preset(preset, ftype, needed_kwargs)
                ntp_cur = level_ntp._replace(**needed_kwargs)
                data_dict[ntp_cur] = data_cur

        return data_dict

    def read_direct(self, ntp, ftype):
        """The simpliest case when one file correspond to one item. """
        assert ftype not in self.ffinder.preset_ftypes

        fname = self.ffinder.fname_from_ntp(ntp, ftype)
        if not os.path.exists(fname):
            return False

        if ftype == "rgb":
            data = self.read_rgb_img(fname)
        elif ftype == "depth_raw":
            data = self.read_value_img(fname)
        elif ftype == "depth_dense":
            data = self.read_value_img(fname)
        elif ftype == "lidar":
            data = self.read_lidar_pts(fname)
        else:
            raise ValueError("ftype not recognized.")

        return data

    def preset_func(self, fname, ftype):
        """The function that read the file of preset_ftypes. 
        The returning data is to be further processed to get a single item. """
        assert ftype in self.ffinder.preset_ftypes

        if ftype == 'calib':
            return self.load_calib(fname)
        elif ftype == 'T_rgb':
            return self.load_Ts(fname)
        else:
            raise ValueError("ftype not implemented yet")
        return

    def one_from_preset(self, preset, ftype, aug_kwargs=None):
        """Retrieve a single item of data from the content in a file of preset_ftypes. """
        assert ftype in self.ffinder.preset_ftypes

        if ftype == 'calib':
            if aug_kwargs is None:
                return self.inex_from_calib(preset)
            else:
                return self.inex_from_calib(preset, **aug_kwargs)
        elif ftype == 'T_rgb':
            if aug_kwargs is None:
                return self.T_from_Ts(preset)
            else:
                return self.T_from_Ts(preset, **aug_kwargs)
        else:
            raise ValueError("ftype not implemented yet")
        return

    def read_rgb_img(self, fpath):
        rgb_img = Image.open(fpath)
        rgb_img = np.asarray(rgb_img, dtype=np.float32)

        return rgb_img

    def read_lidar_pts(self, fpath):
        velo = load_velodyne_points(fpath)
        return velo
           
    def read_datadict_from_img_path(self, img_path, ftype_list=None):
        """given an image path, retrieve all related info (lidar, depth, pose, etc. )"""
        if ftype_list is None:
            ftype_list = self.ffinder.ftypes

        ntp = self.ffinder.ntp_from_fname(img_path, "rgb")
        data_dict = {}
        for ftype in ftype_list:
            data_dict[ftype] = self.read_from_ntp(ntp, ftype)

        return data_dict

    def read_datadict_from_ntp(self, ntp, ftype_list=None):
        if ftype_list is None:
            ftype_list = self.ffinder.ftypes

        data_dict = {}
        for ftype in ftype_list:
            data_dict[ftype] = self.read_from_ntp(ntp, ftype)

        return data_dict

    def ntps_from_split_file(self, split_file):
        return self.ffinder.ntps_from_split_file(split_file)
 
    # def read_T(self, fname, *kwargs):
    #     Ts = self.load_Ts(fname)

    #     level_ntp = self.ffinder.ntp_from_fname(fname, 'T_rgb')    # fpaths is a list, any one of the element work

    #     exist_single = self.ffinder.infile_level_name['T_rgb'] is None                         ### calib only contains single InExtr
    #     query_single = kwargs != {} and self.ffinder.infile_level_name['T_rgb'] in kwargs      ### if kwargs does not contain the useful level, it is not used at all
    #     return_single = exist_single or query_single

    #     if return_single:
    #         ### add a single item to the dict or return a single InExtr object
    #         if exist_single:
    #             ### no need to augment
    #             T_cur = self.T_from_Ts(Ts)
    #             ntp_cur = level_ntp
    #         else:
    #             ### augment with the arg
    #             finfo_item = getattr(kwargs, self.ffinder.infile_level_name['T_rgb'])
    #             T_cur = self.T_from_Ts(Ts, finfo_item)
    #             ntp_cur = level_ntp._replace(**{self.ffinder.infile_level_name['T_rgb']: finfo_item})

    #         if inex_dict is None:
    #             ### return the value itself
    #             T_dict = T_cur
    #         else:
    #             ### add a single item to existing dict
    #             T_dict[ntp_cur] = T_cur
    #     else:
    #         ### add multiple items to the dict or return a dict of the new items
    #         if T_dict is None:
    #             T_dict = {}

    #         finfo_list_T = self.ffinder.get_finfo_list_at_level(self.ffinder.finfo_dict, level_ntp, query_level=self.ffinder.infile_level_name['T_rgb'])
    #         for finfo_item in finfo_list_T:
    #             T_cur = self.T_from_Ts(Ts, finfo_item)
    #             ntp_cur = level_ntp._replace(**{self.ffinder.infile_level_name['T_rgb']: finfo_item})
    #             T_dict[ntp_cur] = T_cur

    #     return T_cur


class DataReaderKITTI(DataReader):
    def __init__(self, *args, **kwargs):
        super(DataReaderKITTI, self).__init__(dataset='kitti', *args, **kwargs)

    def read_value_img(self, fpath):
        """read an image whose pixel values are of specific meaning instead of RGB"""
        value_img = Image.open(fpath)
        value_img = np.asarray(value_img, dtype=np.float32)
        value_img = value_img / 256.0

        return value_img

    def load_Ts(self, fpath):
        with open(fpath) as f:
            T_lines = f.readlines()

        return T_lines

    def T_from_Ts(self, T_lines, fid):
        cur_line = T_lines[fid].strip()

        T = np.array( list(map(float, cur_line.split())) ).reshape(3,4)
        T = np.vstack( (T, np.array([[0,0,0,1]]))).astype(np.float32)
        return T

    def load_calib(self, fnames):
        calib = {}
        if not isinstance(fnames, list):
            fnames = [fnames]
        for fname in fnames:
            calib.update( read_calib_file(fname) )

        return calib

    def inex_from_calib(self, calib, side):
        inex = InExtr()

        im_shape = calib["S_rect_{:02d}".format(side)][::-1].astype(np.int32) ## ZMH: [height, width]

        inex.width = im_shape[1]
        inex.height = im_shape[0]

        ## intrinsics
        P_rect = calib['P_rect_{:02d}'.format(side)].reshape(3, 4).astype(np.float32)
        K = P_rect[:, :3]
        K = K_mat2py(K)
        # K_unit = scale_K(K, old_width=im_shape[1], old_height=im_shape[0], torch_mode=False, align_corner=self.align_corner)

        # inex.K_unit = K_unit
        inex.K = K

        ## extrinsics
        T_cam_lidar = np.hstack((calib['R'].reshape(3, 3), calib['T'][..., np.newaxis]))
        T_cam_lidar = np.vstack((T_cam_lidar, np.array([0, 0, 0, 1.0]))).astype(np.float32)
        R_rect_cam = np.eye(4).astype(np.float32)
        R_rect_cam[:3, :3] = calib['R_rect_00'].reshape(3, 3)

        K_inv = np.linalg.inv(K)
        Kt = P_rect[:, 3:4]
        t = np.dot(K_inv, Kt) # in KITTI's matlab devkit, tx = Kt[0,3]/fx, ty=tz=0

        P_rect_t = np.identity(4).astype(np.float32)
        P_rect_t[:3, 3:4] = t # ZMH: 4*4
        
        P_rect_li = np.dot(P_rect_t, np.dot(R_rect_cam, T_cam_lidar))

        inex.P_cam_li = P_rect_li
        return inex


class DataReaderWaymo(DataReader):
    def __init__(self, *args, **kwargs):
        super(DataReaderWaymo, self).__init__(dataset='waymo', *args, **kwargs)

    def read_value_img(self, fpath):
        """read an image whose pixel values are of specific meaning instead of RGB"""
        value_img = Image.open(fpath)
        value_img = np.asarray(value_img, dtype=np.float32)
        value_img = value_img / 256.0

        return value_img

    def load_Ts(self, fpath):
        with open(fpath) as f:
            T_lines = f.readlines()
            
        return T_lines

    def T_from_Ts(self, T_lines, fid):
        cur_line = T_lines[fid].strip()

        cur_line_split = list(map(float, cur_line.split()))
        pose_elements = cur_line_split[1:17]
        T = np.array(pose_elements).reshape(4,4).astype(np.float32)

        return T

    def load_calib(self, fnames):
        calib = {}
        if not isinstance(fnames, list):
            fnames = [fnames]
        for fname in fnames:
            calib.update( read_calib_file(fname) )
            
        return calib

    def inex_from_calib(self, calib, side):

        inex = InExtr()

        Tr_velo_to_cam = calib['Tr_velo_to_cam_{}'.format(side)].reshape((4,4)).astype(np.float32)

        cam_intr = calib['P{}'.format(side)].reshape((3,4)).astype(np.float32)
        cam_intr = cam_intr[:, :3]
        dist_coeff = calib['Dist_{}'.format(side)].astype(np.float32)
        waymo_cam_RT=np.array([0,-1,0,0,  0,0,-1,0,   1,0,0,0,    0 ,0 ,0 ,1], dtype=np.float32).reshape(4,4)     
        Tr_velo_to_cam = waymo_cam_RT.dot(Tr_velo_to_cam)       # the axis swapping is merged into T_velo_cam

        waymo_cam_RT_inv = np.linalg.inv(waymo_cam_RT[:3, :3])
        cam_intr_normal = cam_intr.dot(waymo_cam_RT_inv)

        inex.width = 1920
        inex.height = 1280

        # K_unit = scale_K(cam_intr_normal, old_width=inex.width, old_height=inex.height, torch_mode=False, align_corner=self.align_corner)

        # inex.K_unit = K_unit
        inex.K = K

        inex.P_cam_li = Tr_velo_to_cam
        inex.dist_coef = dist_coeff

        return inex

class DataReaderVKITTI2(DataReader):
    def __init__(self, *args, **kwargs):
        super(DataReaderVKITTI2, self).__init__(dataset='vkitti2', *args, **kwargs)

    def read_value_img(self, fpath):
        """read an image whose pixel values are of specific meaning instead of RGB"""
        value_img = Image.open(fpath)
        value_img = np.asarray(value_img, dtype=np.float32)
        value_img = value_img / 100.0 # see VKITTI2Dataset in monodepth2

        return value_img

    def load_Ts(self, fpath):
        raise NotImplementedError

    def T_from_Ts(self, T_lines, fid):
        raise NotImplementedError

    def load_calib(self, fnames):
        calib = {}
        if not isinstance(fnames, list):
            fnames = [fnames]
        for fname in fnames:
            calib.update( self.read_calib_file_vkitti2(fname) )
            
        return calib

    def read_calib_file_vkitti2(self, fname):
        with open(fname) as f:
            lines = f.readlines()
        headline = lines[0]
        dataarray = [[float(x) for x in line.split(" ")[2:]] for line in lines[1:]]
        fids = [int(line.split(" ")[0]) for line in lines[1:] ]
        cams = [int(line.split(" ")[1]) for line in lines[1:] ]
        calib = dict()
        n_lines = len(lines) - 1

        for i in range(n_lines):
            key = (cams[i], fids[i])
            calib[key] = dataarray[i]

        return calib

    def inex_from_calib(self, calib, cam, fid):

        inex = InExtr()

        calib_cur = calib[(cam, fid)]
        cam_intr_normal = np.eye(3, dtype=np.float32)
        cam_intr_normal[0,0] = calib_cur[0]
        cam_intr_normal[1,1] = calib_cur[1]
        cam_intr_normal[0,2] = calib_cur[2]
        cam_intr_normal[1,0] = calib_cur[3]

        inex.width = 1242
        inex.height = 375


        # Tr_velo_to_cam = calib['Tr_velo_to_cam_{}'.format(side)].reshape((4,4)).astype(np.float32)

        # cam_intr = calib['P{}'.format(side)].reshape((3,4)).astype(np.float32)
        # cam_intr = cam_intr[:, :3]
        # dist_coeff = calib['Dist_{}'.format(side)].astype(np.float32)
        # waymo_cam_RT=np.array([0,-1,0,0,  0,0,-1,0,   1,0,0,0,    0 ,0 ,0 ,1], dtype=np.float32).reshape(4,4)     
        # Tr_velo_to_cam = waymo_cam_RT.dot(Tr_velo_to_cam)       # the axis swapping is merged into T_velo_cam

        # waymo_cam_RT_inv = np.linalg.inv(waymo_cam_RT[:3, :3])
        # cam_intr_normal = cam_intr.dot(waymo_cam_RT_inv)

        # inex.width = 1920
        # inex.height = 1280

        # K_unit = scale_K(cam_intr_normal, old_width=inex.width, old_height=inex.height, torch_mode=False, align_corner=self.align_corner)

        # inex.K_unit = K_unit
        inex.K = K

        inex.P_cam_li = np.eye(3,4)
        inex.dist_coef = np.zeros(5)

        return inex

if __name__ == "__main__":
    data_root_waymo = '/mnt/storage8t/datasets/waymo_kitti/training'
    data_root_kitti = '/mnt/storage8t/minghanz/Datasets/KITTI_data/kitti_data'
    data_root_vkitti2 = '/mnt/storage8t/minghanz/Datasets/vKITTI2'

    # dataread = DataReaderKITTI(data_root=data_root_kitti)
    # img_path = '/mnt/storage8t/minghanz/Datasets/KITTI_data/kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.jpg'
    # split_path = '/home/minghanz/bts/train_test_inputs/eigen_train_files_with_gt_nonstatic_jpg_fullpath.txt'
    # ftype_list = ["rgb", "depth_dense", "T_rgb", "lidar", "calib"]
    
    # dataread = DataReaderWaymo(data_root=data_root_waymo)
    # img_path = '/mnt/storage8t/datasets/waymo_kitti/training/1083056852838271990_4080_000_4100_000/image_00/0000000000.jpg'
    # ftype_list = ['rgb', 'depth_raw', 'lidar', 'calib', 'T_rgb']


    dataread = DataReaderVKITTI2(data_root=data_root_vkitti2)
    img_path = '/mnt/storage8t/minghanz/Datasets/vKITTI2/Scene02/clone/frames/rgb/Camera_0/rgb_00000.jpg'
    ftype_list = ["rgb", "depth_raw", "calib"]

    data_dict = dataread.read_datadict_from_img_path(img_path, ftype_list)
    print(data_dict)

    # ntps = dataread.ntps_from_split_file(split_path)
    # data_dict = dataread.read_datadict_from_ntp(ntps[0], ftype_list)
    # print(data_dict)
