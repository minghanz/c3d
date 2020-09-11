"""This file is to test the C3DLoss calculated with flow is correct. 
Do not put this file at the top level of this repo, which will search c3d in this local folder first, 
but the library is installed in the environment python instead of locally, which will cause failure in importing c3d. 
"""
import numpy as np
import torch

import c3d
from c3d import C3DLoss
from c3d.utils_general.calib import InExtr
from c3d.utils.cam_proj import CamInfo_from_InExs

# from .c3d_loss import C3DLoss
# from .utils_general.calib import InExtr
# from .utils.cam_proj import CamInfo_from_InExs

def create_sample_input(width, dist, flow_z, dist_gt):
    ### CamInfo
    f = width / 2
    c = width / 2
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=np.float32)
    P_cam_li = np.eye(4)

    inex = InExtr()
    inex.width = width
    inex.height = width
    inex.K = K
    inex.P_cam_li = P_cam_li

    cam_info = CamInfo_from_InExs(inex)

    ### depth
    depth = np.ones((1, width, width), dtype=np.float32) * dist       # H*W*1
    image = np.ones((3, width, width), dtype=np.float32) * 0.8
    image[1] = 0
    image[2] = 0

    depth_mask = np.ones((1, width, width), dtype=np.bool)

    depth_gt = np.zeros((1, width, width), dtype=np.float32)
    depth_gt_mask = np.zeros((1, width, width), dtype=np.bool)
    for i in range(int(0.05*width*width)):
        w = np.random.randint(width)
        h = np.random.randint(width)
        depth_gt[0,w,h] = dist_gt
        depth_gt_mask[0,w,h] = True

    ### flow
    xy1 = cam_info.xy1_grid.numpy()
    xyz = xy1[0] * depth
    xyz_flowed = xy1[0] * (depth + flow_z)
    flow = xyz_flowed - xyz
    print(flow[2].max(), flow[2].min(), flow[1].max(), flow[1].min(), flow[0].max(), flow[0].min())
    # flow = np.zeros((3, width, width), dtype=np.float32)
    # flow[2] = flow_z
    flow_mask = np.ones((1, width, width), dtype=np.bool)

    ### construct dict
    depth_dict = dict()
    depth_dict["pred"] = depth
    depth_dict["pred_mask"] = depth_mask
    depth_dict["gt"] = depth_gt
    depth_dict["gt_mask"] = depth_gt_mask
    depth_dict['rgb'] = image
    
    flow_dict = dict()
    flow_dict["pred"] = flow
    flow_dict["mask"] = flow_mask

    ### to torch tensor on GPU
    for key, value in depth_dict.items():
        depth_dict[key] = torch.from_numpy(value).unsqueeze(0).cuda()

    for key, value in flow_dict.items():
        flow_dict[key] = torch.from_numpy(value).unsqueeze(0).cuda()

    cam_info = cam_info.cuda()

    return depth_dict, flow_dict, cam_info

if __name__ == "__main__":
    np.random.seed(0)
    depth_dict_1, flow_dict_1, cam_info = create_sample_input(width=500, dist=10, flow_z=-2, dist_gt=10)
    depth_dict_2, flow_dict_2, _ = create_sample_input(width=500, dist=8, flow_z=2, dist_gt=8)

    c3d_loss = C3DLoss(flow_mode=True)
    cfg_file = "../c3d_config_example.txt"
    c3d_loss.parse_opts(cfg_file)

    loss_c3d = c3d_loss(depth_img_dict_1=depth_dict_1, depth_img_dict_2=depth_dict_2, flow_dict_1to2=flow_dict_1, flow_dict_2to1=flow_dict_2, cam_info=cam_info)

    print(loss_c3d)


# if __name__ == "__main__":
#     c3d_loss = C3DLoss(flow_mode=True)
#     cfg_file = "/home/minghanz/self-mono-sf/scripts/c3d_config.txt"
#     pkl_file = "/home/minghanz/self-mono-sf/debug_c3d/nan_dicts.pkl"
#     c3d_loss.parse_opts(f_input=cfg_file)
#     c3d_loss.debug_flow_inspect_input(pkl_file)