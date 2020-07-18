"""
This module is to fast find corresponding files for the same frame. 
"""

import os
# import regex
from collections import namedtuple

# DatasetFInfo = namedtuple('DatasetFInfo', ['ftype', 'ext', 'level_ntp'])

# level_ntuple = {}

def set_ntp(name, level_names):
    global level_ntuple
    level_ntuple = namedtuple('level_ntuple', level_names)
    level_ntuple.__new__.__defaults__ = (None,) * len(level_ntuple._fields)

def retrieve_at_level(data, *args):
    '''This function is to read from a nested dict with given list of keys of arbitrary depth. The keys should be on consecutive levels from the top. 
    https://stackoverflow.com/a/48005385'''
    if args and data:
        element  = args[0]
        if element:
            # value = data.get(element)
            value = data[element]
            return value if len(args) == 1 else retrieve_at_level(value, *args[1:])

class DataFinder:
    def __init__(self, name, data_root):
        self.data_root = data_root
        self.name = name
        # self.level_ntuple = namedtuple('level_'+name, self.level_names)
        # self.level_ntuple.__qualname__ = 'DataFinder.level_'+name
        # self.level_ntuple = namedtuple('level_ntuple', self.level_names)
        set_ntp('level_'+name, self.level_names)
        # level_ntuple[self.name] = namedtuple('level_'+name, self.level_names)
        # level_ntuple[self.name].__new__.__defaults__ = (None,) * len(level_ntuple[self.name]._fields)   # https://stackoverflow.com/a/18348004 
        
        # ### attribute lookup failed
        # setattr(self, 'level_'+name, self.level_ntuple)

        ### the ftypes that contains more than one item in a singe file
        self.preset_ftypes = list(self.ofile_level_names.keys())

        ### get the nested dict representing the hierarchy of levels
        self.finfos = self.get_finfo_dict()

        ### get the list of filepaths for ftypes in preload_ftypes (preload_ftypes should be a subset of preset_ftypes)
        self.fnames_preload = {}
        for ftype in self.preload_ftypes:
            self.fnames_preload[ftype] = {}
            self.get_fname_dict(self.fnames_preload[ftype], finfo_dict=self.finfos, level_list=[], desired_depth=len(self.ofile_level_names[ftype]), ftype=ftype)

    def ntps_from_given_dict(self, finfo_dict, keys, dict_known, output_ntps):
        levels = list(dict_known.keys())
        # keys = sorted(keys, key=lambda x:self.level_names.index(x))
        i = len(keys)
        if i == len(self.level_names):
            output_ntps.append(level_ntuple(*keys))
        else:
            if self.level_names[i] not in levels:
                for key in finfo_dict:
                    keys_new = [*keys, key]
                    self.ntps_from_given_dict(finfo_dict[key], keys_new, dict_known, output_ntps)
            else:
                key = dict_known[self.level_names[i]]
                keys_new = [*keys, key]
                self.ntps_from_given_dict(finfo_dict[key], keys_new, dict_known, output_ntps)

        return
            

    def ntps_from_split_file(self, split_file):
        """loading a split file and return a list of ntps"""
        with open(split_file) as f:
            lines = f.readlines()

        ntps = [self.ntp_from_line_split_file(line) for line in lines]
        return ntps

    def finfos_idx_from_ntps(self, ntps):
        """generate a nested dict from a list of ntps. Assume that ntps do not have blank levels"""
        finfos = {}
        n_levels = len(ntps)
        for idx, ntp in enumerate(ntps):
            self.nested_dict_from_ntp(finfos, ntp, idx)

        return finfos
    
    def nested_dict_from_ntp(self, finfos_cur, ntp, idx):
        for i, l in enumerate(ntp):
            if i != len(ntp) - 1:
                if l not in finfos_cur:
                    finfos_cur[l] = {}
                finfos_cur = finfos_cur[l]
            else:
                finfos_cur[l] = idx
        return

    def ntp_from_fname(self, fname, ftype):
        assert ftype in self.ftypes
        if self.data_root in fname:
            fname = os.path.relpath(fname, self.data_root)
        ftype, ext, level_items = self.ntp_from_fname_parse(fname, ftype)
        level_ntp = level_ntuple(*level_items)     # create a namedtuple from an unpacked list
        # finfo = DatasetFInfo(ftype=ftype, ext=ext, level_ntp=level_ntp)
        return level_ntp

    def fname_from_ntp(self, level_ntp, new_ftype):
        assert new_ftype in self.ftypes
        fname = self.fname_from_ntp_parsed(new_ftype, *level_ntp)     # unpack the namedtuple before feeding into the function
        if isinstance(fname, list):
            fname = [ os.path.join(self.data_root, f) for f in fname ]
        else:
            fname = os.path.join(self.data_root, fname)
        return fname

    def get_fname_dict(self, fnames_dict_to_write, finfo_dict, level_list, desired_depth, ftype):
        """Get the list of all filepaths of a ftype"""
        if len(level_list) == desired_depth:
            ### with default values set for namedtuple, no need to manually fill in dummy fields
            ntp = level_ntuple(*level_list)
            fnames = self.fname_from_ntp(ntp, ftype)
            fnames_dict_to_write[ntp] = fnames
        else:
            for new_level_item in finfo_dict:
                self.get_fname_dict(fnames_dict_to_write, finfo_dict[new_level_item], level_list + [new_level_item], desired_depth, ftype)


    def get_finfo_list_at_level(self, finfo_dict, level_ntp, query_level=None):
        """read from the nested dict finfo_dict. The keys are given by level_ntp and query_level. 
        The level_ntp should be on consecutive levels, and query_level should be on a lower level. 
        query_level can be more than one level lower than level_ntp's lowest level, in which case the intermediate levels will be filled by arbitrary value. """
        level_list = [l for l in level_ntp if l is not None]

        if query_level is not None:
            query_level_idx = self.level_names.index(query_level)
            assert query_level_idx >= len(level_list), "query_level conflicts with level_ntp."

            ### fill in dummy keys between level_ntp and query_level
            while query_level_idx > len(level_list):
                finfo_dict_from_level = retrieve_at_level(self.finfos, *level_list)
                level_list.append(list(finfo_dict_from_level.keys())[0])
            
        finfo_dict_from_level = retrieve_at_level(self.finfos, *level_list)
        if isinstance(finfo_dict_from_level, dict):
            finfo_dict_from_level = list(finfo_dict_from_level.keys())
        else:
            assert isinstance(finfo_dict_from_level, list), "(sub)element in self.finfos is either dict or list, now it is {}".format(type(finfo_dict_from_level))
        
        return finfo_dict_from_level        

    def ntp_strip_fill(self, level_ntp, wanted_levels):
        """return a new level_ntuple with fields given by wanted_levels, and values from original level_ntp. 
        If wanted field is None in original level_ntp, it fills an arbitrary valid value from self.finfos. """

        tmp_dict = {}
        ### make sure the levels are in the same order of level_ntuple
        # wanted_levels = sorted(wanted_levels, key=lambda x: self.level_names.index(x))
        max_l = max([self.level_names.index(x) for x in wanted_levels])
        for i, level in enumerate(self.level_names):
            tmp_dict[level] = getattr(level_ntp, level)

            ### fill in a valid value if it is originally None
            if tmp_dict[level] == None:
                # level_list_last = wanted_levels[:i]
                level_list_last = [tmp_dict[self.level_names[l]] for l in range(i)]
                finfo_dict_from_level = retrieve_at_level(self.finfos, *level_list_last)
                tmp_dict[level] = list(finfo_dict_from_level.keys())[0]

            if i == max_l:
                break

        wanted_dict = {}
        for level in wanted_levels:
            wanted_dict[level] = tmp_dict[level]

        new_ntp = level_ntuple(**wanted_dict)

        return new_ntp

    def ntp_ftype_convert(self, ntp, ftype):
        """from a detailed ntp to a coarse ntp"""
        assert ftype in self.preset_ftypes

        ### convert the querying ntp to the ntp of the ftype wanted
        levels_out = self.ofile_level_names[ftype]
        levels_in = self.infile_level_name[ftype]
        if not isinstance(levels_out, list):
            levels_out = [levels_out]
        if not isinstance(levels_in, list):
            levels_in = [levels_in]
        wanted_levels = levels_out + levels_in
        new_ntp = self.ntp_strip_fill(ntp, wanted_levels)
        return new_ntp

    def find_extra_dict_in_ntp(self, ntp_detailed, ntp_coarse):
        extra_dict = {}
        for key in self.level_names:
            if getattr(ntp_detailed, key) is not None and getattr(ntp_coarse, key) is None:
                extra_dict[key] = getattr(ntp_detailed, key)

        return extra_dict

class DataFinderWaymo(DataFinder):
    def __init__(self, *args, **kwargs):
        self.ftypes = ['rgb', 'depth_raw', 'lidar', 'calib', 'T_rgb']
        self.level_names = ['seq', 'side', 'fid']
        self.ofile_level_names = {'calib': ['seq'], 'T_rgb': ['seq', 'side']}
        self.infile_level_name = {'calib': 'side', 'T_rgb': 'fid'}

        self.preload_ftypes = ['calib']
        # self.preload_ftypes = []

        self.calib_filenames = ['calib_all']
        super(DataFinderWaymo, self).__init__(name='waymo', *args, **kwargs)
    
    def ntp_from_fname_parse(self, fname, ftype):
        '''fname is relative to the root of the dataset'''
        if ftype in ['rgb', 'depth_raw']:
            '''training/141184560845819621_10582_560_10602_560/image_00/0000000000.jpg'''
            '''training/141184560845819621_10582_560_10602_560/depth_00/0000000000.png'''
            pre_ext, ext = fname.split('.')
            seq, side, fid = pre_ext.split('/')
            side = int(side.split('_')[-1])
            fid = int(fid)
        elif ftype == 'lidar':
            '''training/141184560845819621_10582_560_10602_560/lidar/0000000000.bin'''
            pre_ext, ext = fname.split('.')
            seq, _, fid = pre_ext.split('/')
            side = None
            fid = int(fid)
        elif ftype == 'calib':
            '''training/141184560845819621_10582_560_10602_560/calib/calib_all.txt'''
            pre_ext, ext = fname.split('.')
            seq, _, fid = pre_ext.split('/')
            side = None
            fid = None
        elif ftype == 'T_rgb':
            '''validation/1024360143612057520_3580_000_3600_000/pose_cam_00.txt'''
            pre_ext, ext = fname.split('.')
            seq, side = pre_ext.split('/')
            side = int(side.split('_')[-1])
            fid = None
        else:
            raise ValueError('ftype not seen')
            
        return ftype, ext, [seq, side, fid]

    def fname_from_ntp_parsed(self, ftype, seq, side, fid):
        if ftype == 'rgb':
            fname = os.path.join(seq, 'image_{:02d}'.format(side), '{:010d}.jpg'.format(fid))
        elif ftype == 'depth_raw':
            fname = os.path.join(seq, 'depth_{:02d}'.format(side), '{:010d}.png'.format(fid))
        elif ftype == 'lidar':
            fname = os.path.join(seq, 'lidar', '{:010d}.bin'.format(fid))
        elif ftype == 'calib':
            fname = [os.path.join(seq, 'calib', '{}.txt'.format(f)) for f in self.calib_filenames]
        elif ftype == 'T_rgb':
            fname = os.path.join(seq, 'pose_cam_{:02d}.txt'.format(side))
        else:
            raise ValueError('ftype not seen')
    
        return fname

    def get_finfo_dict(self):
        finfos = {}
        seqs = [ d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root,d) )]
        for seq in seqs:            # seq level
            finfos[seq] = {}
            cur_root = os.path.join(self.data_root, seq)
            sides = [0, 1, 2, 3, 4]
            for side in sides:      # side level
                cur_root_2 = os.path.join(cur_root,  'image_{:02d}'.format(side))
                fids = os.listdir(cur_root_2)
                fids = [ int(fid.split('.')[0]) for fid in fids]
                finfos[seq][side] = fids
        return finfos


class DataFinderKITTI(DataFinder):
    def __init__(self, *args, **kwargs):
        self.ftypes = ['rgb', 'depth_dense', 'depth_raw', 'lidar', 'calib', 'T_rgb', 'T_lidar']
        self.level_names = ['date', 'seq', 'side', 'fid']

        self.ofile_level_names = {}
        self.infile_level_name = {}
        self.ofile_level_names['calib'] = ['date']
        self.infile_level_name['calib'] = 'side'
        self.ofile_level_names['T_rgb'] = ['date', 'seq', 'side']
        self.infile_level_name['T_rgb'] = 'fid'

        self.preload_ftypes = ['calib']
        # self.preload_ftypes = []

        self.calib_filenames = ['calib_cam_to_cam', 'calib_velo_to_cam']
        super(DataFinderKITTI, self).__init__(name='kitti', *args, **kwargs)

    def ntp_from_fname_parse(self, fname, ftype):
        '''fname is relative to the root of the dataset'''

        if ftype == 'rgb':
            '''2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.jpg'''
            pre_ext, ext = fname.split('.')
            date, seq, side, _, fid = pre_ext.split('/')
            side = int(side.split('_')[-1])
            fid = int(fid)
        elif ftype in ['depth_dense', 'depth_raw']:
            '''2011_09_26/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png'''
            '''2011_09_26/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png'''
            pre_ext, ext = fname.split('.')
            date, seq, _, _, side, fid = pre_ext.split('/')
            side = int(side.split('_')[-1])
            fid = int(fid)
        elif ftype == 'lidar':
            '''2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin'''
            pre_ext, ext = fname.split('.')
            date, seq, _, _, fid = pre_ext.split('/')
            fid = int(fid)
            side = None
        elif ftype == 'calib':
            '''2011_09_26/calib_cam_to_cam.txt'''
            pre_ext, ext = fname.split('.')
            date, fid = pre_ext.split('/')
            side = None
            seq = None
            fid = None
        elif ftype == 'T_rgb':
            '''2011_09_26/2011_09_26_drive_0001_sync/poses/cam_02.txt'''
            pre_ext, ext = fname.split('.')
            date, seq, _, cam_side = pre_ext.split('/')
            side = int(cam_side.split('_')[-1])
            fid = None
        elif ftype == 'T_lidar':
            '''2011_09_26/2011_09_26_drive_0001_sync/poses/velo.txt'''
            pre_ext, ext = fname.split('.')
            date, seq, _, _ = pre_ext.split('/')
            side = None
            fid = None
        else:
            raise ValueError('ftype not seen')
        
        return ftype, ext, [date, seq, side, fid]

    def fname_from_ntp_parsed(self, ftype, date, seq, side, fid):
        if ftype == 'rgb':
            fname = os.path.join(date, seq, 'image_{:02d}'.format(side), 'data', '{:010d}.jpg'.format(fid))
        elif ftype == 'depth_dense':
            fname = os.path.join(date, seq, 'proj_depth', 'groundtruth', 'image_{:02d}'.format(side), '{:010d}.png'.format(fid))
        elif ftype == 'depth_raw':
            fname = os.path.join(date, seq, 'proj_depth', 'velodyne_raw', 'image_{:02d}'.format(side), '{:010d}.png'.format(fid))
        elif ftype == 'lidar':
            fname = os.path.join(date, seq, 'velodyne_points', 'data', '{:010d}.bin'.format(fid))
        elif ftype == 'calib':
            fname = [os.path.join(date, '{}.txt'.format(f)) for f in self.calib_filenames]
        elif ftype == 'T_rgb':
            fname = os.path.join(date, seq, 'poses', 'cam_{:02d}.txt'.format(side))
        elif ftype == 'T_lidar':
            fname = os.path.join(date, seq, 'poses', 'velo.txt')
        else:
            raise ValueError('ftype not seen')
    
        return fname

    def get_finfo_dict(self):
        finfos = {}
        dates = [ d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root,d) )]
        for date in dates:              # date level
            finfos[date] = {}
            cur_root = os.path.join(self.data_root, date)
            seqs = [ d for d in os.listdir(cur_root) if os.path.isdir(os.path.join(cur_root,d) )]
            for seq in seqs:            # seq level
                finfos[date][seq] = {}
                cur_root_2 = os.path.join(cur_root, seq)
                sides = [2, 3]
                for side in sides:      # side level
                    cur_root_3 = os.path.join(cur_root_2,  'image_{:02d}'.format(side), 'data')
                    fids = os.listdir(cur_root_3)
                    fids = [ int(fid.split('.')[0]) for fid in fids]
                    finfos[date][seq][side] = fids
        return finfos            

    def ntp_from_line_split_file(self, line):
        """from a line in split file to a ntp"""
        """bts kitti split: /home/minghanz/bts/train_test_inputs/eigen_train_files_with_gt_jpg_fullpath.txt"""
        img, gt, focal_length = line.split()
        ntp = self.ntp_from_fname(img, 'rgb')
        # path_strs = img.split('/')
        # date_str = path_strs[0]
        # seq_str = path_strs[1]
        # seq_n = int(seq_str.split('_drive_')[1].split('_')[0])  # integer of the sequence number
        # side = int(path_strs[2].split('_')[1])
        # frame = int(path_strs[-1].split('.')[0])
        # ntp = level_ntuple(date=date_str, seq=seq_str, side=side, fid=frame)
        return ntp

if __name__ == '__main__':
    ############## kitti
    # data_root = '/media/sda1/minghanz/datasets/kitti/kitti_data'
    data_root = '/mnt/storage8t/minghanz/Datasets/KITTI_data/kitti_data'
    kitti = DataFinderKITTI(data_root=data_root)
    # print(kitti.finfos)
    print(kitti.fnames_preload['calib'])

    calib_path = "2011_09_26/calib_cam_to_cam.txt"
    level_ntp = kitti.ntp_from_fname(calib_path, 'calib')    # fpaths is a list, any one of the element work
    finfo_list_calib = kitti.get_finfo_list_at_level(kitti.finfos, level_ntp, kitti.infile_level_name['calib'])
    print(finfo_list_calib)

    # fname = '2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000006.jpg'
    # ftype = 'rgb'
    # finfo = kitti.ntp_from_fname(fname, ftype)
    # print(finfo)
    # fname_depth = kitti.fname_from_ntp(finfo, 'depth_dense')
    # print(fname_depth)


    ############## waymo
    data_root = '/mnt/storage8t/datasets/waymo_kitti/training'
    waymo = DataFinderWaymo(data_root=data_root)
    print(waymo.fnames_preload['calib'])

    calib_path = "141184560845819621_10582_560_10602_560/calib/calib_all.txt"
    level_ntp = waymo.ntp_from_fname(calib_path, 'calib')    # fpaths is a list, any one of the element work
    finfo_list_calib = waymo.get_finfo_list_at_level(waymo.finfos, level_ntp, waymo.infile_level_name['calib'])
    print(finfo_list_calib)