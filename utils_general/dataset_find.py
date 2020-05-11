import os
import regex
from collections import namedtuple

# DatasetFInfo = namedtuple('DatasetFInfo', ['ftype', 'ext', 'level_ntp'])

class DataFinder:
    def __init__(self, name, data_root):
        self.data_root = data_root
        self.name=name
        self.level_ntuple = namedtuple('level_'+name, self.level_name_list)

    def finfo_from_fname(self, fname, ftype):
        assert ftype in self.ftype_list
        ftype, ext, level_items = self.finfo_from_fname_parse(fname, ftype)
        level_ntp = self.level_ntuple(*level_items)     # create a namedtuple from an unpacked list
        # finfo = DatasetFInfo(ftype=ftype, ext=ext, level_ntp=level_ntp)
        return level_ntp

    def fname_from_finfo(self, level_ntp, new_ftype):
        assert new_ftype in self.ftype_list
        fname = self.fname_from_finfo_parsed(new_ftype, *level_ntp)     # unpack the namedtuple before feeding into the function
        return fname


class DataFinderKITTI(DataFinder):
    def __init__(self, *args, **kwargs):
        self.ftype_list = ['rgb', 'depth_dense', 'depth_raw', 'lidar', 'calib']
        self.level_name_list = ['date', 'seq', 'side', 'fid']
        super(DataFinderKITTI, self).__init__(name='kitti', *args, **kwargs)

    def finfo_from_fname_parse(self, fname, ftype):
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
        else:
            raise ValueError('ftype not seen')
        
        return ftype, ext, [date, seq, side, fid]

    def fname_from_finfo_parsed(self, ftype, date, seq, side, fid):
        if ftype == 'rgb':
            fname = os.path.join(date, seq, 'image_{:02d}'.format(side), 'data', '{:010d}.jpg'.format(fid))
        elif ftype == 'depth_dense':
            fname = os.path.join(date, seq, 'proj_depth', 'groundtruth', 'image_{:02d}'.format(side), '{:010d}.png'.format(fid))
        elif ftype == 'depth_raw':
            fname = os.path.join(date, seq, 'proj_depth', 'velodyne_raw', 'image_{:02d}'.format(side), '{:010d}.png'.format(fid))
        elif ftype == 'lidar':
            fname = os.path.join(date, seq, 'velodyne_points', 'data', '{:010d}.bin'.format(fid))
        elif ftype == 'calib':
            fname = os.path.join(date, '{}.txt'.format(fid))
        else:
            raise ValueError('ftype not seen')
    
        return fname


if __name__ == '__main__':
    data_root = '/media/sda1/minghanz/datasets/kitti/kitti_data'
    kitti = DataFinderKITTI(data_root=data_root)
    fname = '2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000006.jpg'
    ftype = 'rgb'
    finfo = kitti.finfo_from_fname(fname, ftype)
    print(finfo)
    fname_depth = kitti.fname_from_finfo(finfo, 'depth_dense')
    print(fname_depth)