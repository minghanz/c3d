import torch 
from .utils_general.dataset_read import DataReaderKITTI
from .utils_general.calib import InExtr
from .utils.cam import CamScale, CamRotate, CamCrop, CamFlip
from .utils.cam_proj import CamInfo_from_InExs 
import numpy as np
import copy
from skimage.morphology import binary_dilation, binary_closing

import torchsnooper

from .utils_general.timing import Timing
# import multiprocessing

def seq_ops_on_K_pts(seq_cam_ops, in_extr, lidar_pts):
    """here we assume lidar_pts is in the camera reference frame.
    We need to include lidar_pts here to support flipping operation. """
    assert lidar_pts.shape[-1] in [3, 4]    # n*3 or n*4 shape
    for cam_op in seq_cam_ops:
        if isinstance(cam_op, CamScale):
            in_extr = in_extr.scale(cam_op)
        elif isinstance(cam_op, CamCrop):
            in_extr = in_extr.crop(cam_op)
        elif isinstance(cam_op, CamFlip):
            in_extr = in_extr.flip(cam_op)
            if cam_op.hori:
                lidar_pts[:,0] = -lidar_pts[:,0]
            if cam_op.vert:
                lidar_pts[:,1] = -lidar_pts[:,1]
        elif isinstance(cam_op, CamRotate):
            raise NotImplementedError
        else:
            raise NotImplementedError
    return in_extr, lidar_pts

def seq_ops_on_pts(seq_cam_ops, lidar_pts):
    """here we assume lidar_pts is in the camera reference frame
    We need to include lidar_pts here to support flipping operation. """
    assert lidar_pts.shape[-1] in [3, 4]    # n*3 or n*4 shape
    for cam_op in seq_cam_ops:
        if isinstance(cam_op, CamScale):
            continue
        elif isinstance(cam_op, CamCrop):
            continue
        elif isinstance(cam_op, CamFlip):
            if cam_op.hori:
                lidar_pts[:,0] = -lidar_pts[:,0]
            if cam_op.vert:
                lidar_pts[:,1] = -lidar_pts[:,1]
        elif isinstance(cam_op, CamRotate):
            raise NotImplementedError
        else:
            raise NotImplementedError
    return lidar_pts

class C3DLoader:
    """This class is to make it easy to load data needed by C3DLoss: """
    def __init__(self, datareader=None, data_root=None):
        if datareader is not None:
            self.datareader = datareader
        else:
            assert data_root is not None
            self.datareader = DataReaderKITTI(data_root=data_root)

        self.dilate_struct = np.ones((35, 35))

        # self.timer = Timing()

    def load_single(self, img_path, seq_cam_ops, lidar_pts=None, no_mask=False):
        """given the img_path and sequence of cam_ops, return cam_info, depth_gt, depth_gt_mask, depth_pred_mask. 
        (rgb is expected to be already loaded by the network's dataloader. lidar_pts can be provided if it's already avaiable in the network's dataloader. )
        This function works with np.ndarray. """

        # self.timer.log("read_datadict", 1, True)
        ### read in_extr and lidar points from dataset given the img_path
        ftype_list = ["calib"] if lidar_pts is not None else ["calib", "lidar"]
        data_dict = self.datareader.read_datadict_from_img_path(img_path, ftype_list=ftype_list)

        ### save intermediate variable to cache in case some data are to be reused
        cache_dict = {}
        cache_dict['inex_init'] = data_dict["calib"]

        inex_init = copy.deepcopy(data_dict["calib"])

        ### the calculation is in numpy instead of pytorch
        if lidar_pts is None:
            lidar_pts = data_dict['lidar']              # n*4
            cache_dict['lidar'] = lidar_pts

        elif isinstance(lidar_pts, torch.Tensor):
            lidar_pts = lidar_pts.cpu().numpy()

        # self.timer.log("lidar to cam frame", 1, True)
        ### adjust the lidar points to camera coordinate before camera operations
        extr_cam_li = inex_init.P_cam_li   # 4*4
        lidar_in_cam_frame = np.dot(extr_cam_li, lidar_pts.T).T # N*4
        lidar_in_cam_frame = lidar_in_cam_frame[lidar_in_cam_frame[:,2] > 0, :]
        inex_init.P_cam_li = np.eye(4)

        # self.timer.log("adjust K and lidar", 1, True)
        ### apply camera operations on in_extr and pts
        in_extr, lidar_pts = seq_ops_on_K_pts(seq_cam_ops, inex_init, lidar_in_cam_frame)

        cache_dict['inex_final'] = copy.deepcopy(in_extr)

        # self.timer.log("generate depth", 1, True)
        ### project to depth image
        depth_img = in_extr.lidar_to_depth(lidar_pts)
        # depth_img = in_extr.lidar_to_depth(lidar_pts, self.timer)
        mask_gt = depth_img > 0
        if not no_mask:
            # self.timer.log("binary_closing", 1, True)
            mask = binary_closing(mask_gt, self.dilate_struct)
            # mask = mask_gt

        # self.timer.log("depth_dict", 1, True)
        depth_dict = {}
        depth_dict["depth"] = depth_img.astype(np.float32)
        depth_dict["mask_gt"] = mask_gt
        if not no_mask:
            depth_dict["mask"] = mask

        # self.timer.log("load_single return", 1, True)

        return in_extr, depth_dict, cache_dict

    def load_single_known_inex(self, img_path, seq_cam_ops, inex_init, inex_final, lidar_pts=None, no_mask=False):
        inex = copy.deepcopy(inex_init)

        ### save intermediate variable to cache in case some data are to be reused
        cache_dict = {}

        if lidar_pts is None:
            ### read lidar data
            ftype_list = ["lidar"]
            data_dict = self.datareader.read_datadict_from_img_path(img_path, ftype_list=ftype_list)
            lidar_pts = data_dict["lidar"]

            cache_dict['lidar'] = lidar_pts
        
        ### adjust the lidar points to camera coordinate before camera operations
        extr_cam_li = inex.P_cam_li   # 4*4
        lidar_in_cam_frame = np.dot(extr_cam_li, lidar_pts.T).T # N*4
        lidar_in_cam_frame = lidar_in_cam_frame[lidar_in_cam_frame[:,2] > 0, :]
        inex.P_cam_li = np.eye(4)

        ### apply camera operations on in_extr and pts
        lidar_pts = seq_ops_on_pts(seq_cam_ops, lidar_in_cam_frame)

        ### project to depth image
        depth_img = inex_final.lidar_to_depth(lidar_pts)

        mask_gt = depth_img > 0
        if not no_mask:
            mask = binary_closing(mask_gt, self.dilate_struct)
            # mask = mask_gt

        depth_dict = {}
        depth_dict["depth"] = depth_img.astype(np.float32)
        depth_dict["mask_gt"] = mask_gt
        if not no_mask:
            depth_dict["mask"] = mask

        return depth_dict, cache_dict

    def load(self, img_path_batch_list, seq_cam_ops_batch_list, lidar_pts_batch_list=None, no_mask=False):
        """for batch processing. 
        img_path_batch_list: a list of img_path
        seq_cam_ops_batch_list: a list of seq_cam_ops
        lidar_pts_batch_list: a list of lidar_pts
        These inputs are in a list because they cannot be batched
        """
        ### if input is single object, convert them to list before processing
        ## img_path_batch_list is a list of str
        if isinstance(img_path_batch_list, str):
            img_path_batch_list = [img_path_batch_list]
        ## seq_cam_ops_batch_list is a list of list of cam_ops
        if isinstance(seq_cam_ops_batch_list, list) and not all(isinstance(seq_cam_ops_batch_i, list) for seq_cam_ops_batch_i in seq_cam_ops_batch_list):
            seq_cam_ops_batch_list = [seq_cam_ops_batch_list]
        assert isinstance(seq_cam_ops_batch_list, list) and all(isinstance(seq_cam_ops_batch_i, list) for seq_cam_ops_batch_i in seq_cam_ops_batch_list), seq_cam_ops_batch_list
        if isinstance(lidar_pts_batch_list, np.ndarray):
            lidar_pts_batch_list = [lidar_pts_batch_list]

        ### generate in_extr and depth image for each item
        batch_size = len(img_path_batch_list)

        ### use multiprocessing
        # # self.timer.log("get_context", 0, True)
        # ctx = multiprocessing.get_context("spawn")
        # # self.timer.log("create Pool", 0, True)
        # pool = ctx.Pool(processes=2)
        # # self.timer.log("assign jobs Pool", 0, True)
        # result_list = []
        # for ib in range(batch_size):
        #     # # self.timer.log("load_single %d"%ib, 0, True)
        #     lidar_cur = None if lidar_pts_batch_list is None else lidar_pts_batch_list[ib]
        #     result_list.append( pool.apply_async(self.load_single, (img_path_batch_list[ib], seq_cam_ops_batch_list[ib], lidar_cur, no_mask)) )

        # pool.close()
        # pool.join()
        # # self.timer.log("retrieve result", 0, True)
        # inex_list = [result.get()[0] for result in result_list]
        # depth_list = [result.get()[1] for result in result_list]
        # cache_list = [result.get()[2] for result in result_list]

        ### do not use multiprocessing
        inex_list = []
        depth_list = []
        cache_list = []
        for ib in range(batch_size):
            # self.timer.log("load_single %d"%ib, 0, True)
            lidar_cur = None if lidar_pts_batch_list is None else lidar_pts_batch_list[ib]
            in_extr, depth_dict, cache_dict = self.load_single(img_path_batch_list[ib], seq_cam_ops_batch_list[ib], lidar_cur, no_mask=no_mask)
            inex_list.append(in_extr)
            depth_list.append(depth_dict)
            cache_list.append(cache_dict)

        # self.timer.log("batching", 0, True)
        ### generate batched tensor
        array_batched_dict = {}
        for key in depth_list[0]:   # "depth", "mask", "mask_gt"
            arrays = [depth[key] for depth in depth_list]
            array_batched = np.stack(arrays, axis=0)    # B*H*W
            array_batched = torch.from_numpy(array_batched).unsqueeze(1)    # B*C*H*W
            array_batched_dict[key] = array_batched
            
        # self.timer.log("CamInfo_from_InExs", 0, True)
        cam_info_batched = CamInfo_from_InExs(inex_list)

        # self.timer.log("load return", 0, True)
        return cam_info_batched, array_batched_dict, cache_list

    def load_known_inex(self, img_path_batch_list, seq_cam_ops_batch_list, inex_init_list, inex_final_list, lidar_pts_batch_list=None, no_mask=False):

        ### if input is single object, convert them to list before processing
        ## img_path_batch_list is a list of str
        if isinstance(img_path_batch_list, str):
            img_path_batch_list = [img_path_batch_list]
        ## seq_cam_ops_batch_list is a list of list of cam_ops
        if isinstance(seq_cam_ops_batch_list, list) and not all(isinstance(seq_cam_ops_batch_i, list) for seq_cam_ops_batch_i in seq_cam_ops_batch_list):
            seq_cam_ops_batch_list = [seq_cam_ops_batch_list]
        assert isinstance(seq_cam_ops_batch_list, list) and all(isinstance(seq_cam_ops_batch_i, list) for seq_cam_ops_batch_i in seq_cam_ops_batch_list), seq_cam_ops_batch_list
        if isinstance(lidar_pts_batch_list, np.ndarray):
            lidar_pts_batch_list = [lidar_pts_batch_list]
            
        ### generate in_extr and depth image for each item
        batch_size = len(img_path_batch_list)
        depth_list = []
        cache_list = []
        for ib in range(batch_size):
            lidar_cur = None if lidar_pts_batch_list is None else lidar_pts_batch_list[ib]
            depth_dict, cache_dict = self.load_single_known_inex(img_path_batch_list[ib], seq_cam_ops_batch_list[ib], inex_init_list[ib], inex_final_list[ib], lidar_cur, no_mask=no_mask)
            depth_list.append(depth_dict)
            cache_list.append(cache_dict)

        ### generate batched tensor
        array_batched_dict = {}
        for key in depth_list[0]:   # "depth", "mask", "mask_gt"
            arrays = [depth[key] for depth in depth_list]
            array_batched = np.stack(arrays, axis=0)    # B*H*W
            array_batched = torch.from_numpy(array_batched).unsqueeze(1)    # B*C*H*W
            array_batched_dict[key] = array_batched

        return array_batched_dict, cache_list
