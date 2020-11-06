import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_points, knn_gather, estimate_pointcloud_normals
import logging
from .utils_general.color import rgb_to_hsv
from .utils.geometry import NormalFromDepthDense, res_normal_dense, calc_normal
from easydict import EasyDict

def load_pcl_from_unpacked(mask, depth, xy1_grid, hsv, flow=None, dense_normal_op=None, K_cur=None, sparse_nml_opts=None, uvb_grid_cur=None, R=None, t=None):
    flow_mode = flow is not None
    dense_normal_mode = dense_normal_op is not None and K_cur is not None
    sparse_normal_mode = sparse_nml_opts is not None and uvb_grid_cur is not None
    assert not (dense_normal_mode and sparse_normal_mode)

    xyz_grid = xy1_grid * depth

    if dense_normal_mode:   ### for normal of flowed point cloud, use the normal of the original point cloud
        normal_grid = dense_normal_op(depth, K_cur)
        nres_grid = res_normal_dense(xyz_grid, normal_grid, K_cur)
        
    if flow_mode:
        flow = torch.cat([flow[:,:2].detach(), flow[:, 2:]], dim=1)
        xyz_grid = xyz_grid + flow

    batch_size = depth.shape[0]

    xyz_flat = xyz_grid.reshape(batch_size, 3, -1)
    if R is not None:
        xyz_flat = torch.matmul(R, xyz_flat) + t  # B*3*3 x B*3*N
    hsv_flat = hsv.reshape(batch_size, 3, -1)
    mask_flat = mask.reshape(batch_size, 1, -1)
    if dense_normal_mode:
        normal_flat = normal_grid.reshape(batch_size, 3, -1)
        if R is not None:
            normal_flat = torch.matmul(R, normal_flat)  # B*3*3 x B*3*N
        nres_flat = nres_grid.reshape(batch_size, 1, -1)
        normal_list = []
    if sparse_normal_mode:
        uvb_flat = uvb_grid_cur.reshape(batch_size, 3, -1)
        uvb_list = []
        nb_list = []

    xyz_list = []
    feature_list = []
    for ib in range(batch_size):
        mask_i = mask_flat[ib, 0]   # N
        xyz_i = xyz_flat[ib].transpose(0, 1)[mask_i]    # N_masked*3
        xyz_list.append(xyz_i)

        if dense_normal_mode:
            hsv_i = hsv_flat[ib].transpose(0, 1)[mask_i]    # N_masked*3
            normal_i = normal_flat[ib].transpose(0, 1)[mask_i]    # N_masked*3
            nres_i = nres_flat[ib].transpose(0, 1)[mask_i]    # N_masked*1
            normal_list.append(normal_i)
            feature_list.append(torch.cat([hsv_i, nres_i], dim=1))
        elif sparse_normal_mode:
            uvb_i = uvb_flat[[ib]][:,:,mask_i]
            uvb_list.append(uvb_i)
            nb_i = uvb_i.shape[2]
            if len(nb_list) > 0:
                nb_i = nb_i + nb_list[-1]
            nb_list.append(nb_i)
        else:
            hsv_i = hsv_flat[ib].transpose(0, 1)[mask_i]    # N_masked*3
            feature_list.append(hsv_i)

    if not (dense_normal_mode or sparse_normal_mode):
        pcl = Pointclouds(xyz_list, normals=None, features=feature_list)
    elif dense_normal_mode:
        pcl = Pointclouds(xyz_list, normals=normal_list, features=feature_list)
    else:
        uvb_cat = torch.cat(uvb_list, dim=2)
        normal_cat, nres_cat = calc_normal(uvb_cat, xyz_grid, mask, sparse_nml_opts.normal_nrange, sparse_nml_opts.ignore_ib, sparse_nml_opts.min_dist_2, return_stat=False)
        normal_list = []
        for ib in range(batch_size):
            mask_i = mask_flat[ib, 0]   # N
            hsv_i = hsv_flat[ib].transpose(0, 1)[mask_i]    # N_masked*3
            if ib == 0:
                normal_i = normal_cat[0,:,:nb_list[ib]].transpose(0, 1)     # N_masked*3
                nres_i = nres_cat[0,:,:nb_list[ib]].transpose(0, 1)     # N_masked*1
            else:
                normal_i = normal_cat[0,:,nb_list[ib-1]:nb_list[ib]].transpose(0, 1)     # N_masked*3
                nres_i = nres_cat[0,:,nb_list[ib-1]:nb_list[ib]].transpose(0, 1)     # N_masked*1
            if R is not None:
                normal_i = torch.matmul(R[ib], normal_i.transpose(0,1)).transpose(0,1)  # 3*3 x 3*N

            normal_list.append(normal_i)
            feature_list.append(torch.cat([hsv_i, nres_i], dim=1))
        pcl = Pointclouds(xyz_list, normals=normal_list, features=feature_list)    

    return pcl

def knn_pcl(pcl_1, pcl_2, n_neighbors=1, return_nn=False):
    dists, idxs, pcl_knn_to_1 = knn_points(pcl_1.points_padded(), pcl_2.points_padded(), pcl_1.num_points_per_cloud(), pcl_2.num_points_per_cloud(), K=n_neighbors, return_nn=return_nn)
    return dists, idxs, pcl_knn_to_1

def mask_from_pcl(pcl):
    num_points_per_cloud = pcl.num_points_per_cloud()
    mask = torch.zeros_like(pcl.points_padded()[:,:,[0]]) #(N, P, 1)
    for ib in range(len(num_points_per_cloud)):
        mask[ib, :num_points_per_cloud[ib]] = 1
    return mask

def pcl_from_knnidx(pcl, knn_idx):
    """pytorch3d has a equivalent function: pytorch3d.ops.knn_gather(), which uses torch.gather()"""
    ### (N, P2, D), (N, P1, K) -> (N, P1, K, D)
    pcl = pcl.points_padded()
    batch_size = pcl.shape[0]
    P1, K = knn_idx.shape[1:]
    P2, D = pcl.shape[1:]

    knn_idx = knn_idx.flatten(start_dim=1)
    pcl_list = []
    for ib in range(batch_size):
        idx_i = knn_idx[ib]
        pcl_knn = torch.index_select(pcl[ib], dim=0, index=knn_idx[ib]).reshape(P1, K, D)
        pcl_list.append(pcl_knn)
    pcl_knn = torch.stack(pcl_list, dim=0)  # B, P1, K, D

    return pcl_knn


class C3DLossKnn(nn.Module):
    def __init__(self):
        super(C3DLossKnn, self).__init__()
        self.num_nn = 20
        self.ell_basedist = 10
        self.ell_min = 0.05
        self.ell_rand = 0.1
        self.log_loss = False

        self.normal_op_dense = NormalFromDepthDense()
        self.sparse_nml_opts = EasyDict() # nml_opts.neighbor_range, nml_opts.ignore_ib, nml_opts.min_dist_2
        self.sparse_nml_opts.normal_nrange = 9
        self.sparse_nml_opts.ignore_ib = False
        self.sparse_nml_opts.min_dist_2 = 0.05
        self.res_mag_min = 0.1
        self.res_mag_max = 2

    def forward(self, depth_img_dict_1=None, depth_img_dict_2=None, flow_dict_1to2=None, flow_dict_2to1=None, cam_info=None):

        ### format them into point clouds 
        pcl_pred_1, pcl_gt_1, pcl_flowed_2_from_1 = self.load_pcl(cam_info, depth_img_dict_1, flow_dict_1to2)
        pcl_pred_2, pcl_gt_2, pcl_flowed_1_from_2 = self.load_pcl(cam_info, depth_img_dict_2, flow_dict_2to1)

        # logging.info("n gt: {}".format(pcl_gt_1.num_points_per_cloud()))
        # logging.info("n pred: {}".format(pcl_pred_1.num_points_per_cloud()))
        # logging.info("n flow: {}".format(pcl_flowed_1_from_2.num_points_per_cloud()))
        
        # pclpad_pred_1 = pcl_pred_1.points_padded()
        # pclpad_pred_2 = pcl_pred_2.points_padded()
        # pclpad_gt_1 = pcl_gt_1.points_padded()
        # pclpad_gt_2 = pcl_gt_2.points_padded()
        # pclpad_flowed_2_from_1 = pcl_flowed_2_from_1.points_padded()
        # pclpad_flowed_1_from_2 = pcl_flowed_1_from_2.points_padded()

        ### knn
        dists_1, idxs_1, pcl_knn_to_1 = knn_pcl(pcl_gt_1, pcl_pred_1, self.num_nn) #(N, P1, D), (N, P2, D) -> (N, P1, K), (N, P1, K), (N, P1, K, D)
        dists_2, idxs_2, pcl_knn_to_2 = knn_pcl(pcl_gt_2, pcl_pred_2, self.num_nn)
        dists_1_from_2, idxs_1_from_2, pcl_knn_flowed_to_1 = knn_pcl(pcl_gt_1, pcl_flowed_1_from_2, self.num_nn)
        dists_2_from_1, idxs_2_from_1, pcl_knn_flowed_to_2 = knn_pcl(pcl_gt_2, pcl_flowed_2_from_1, self.num_nn)

        # pcl_knn_to_1 = pcl_from_knnidx(pcl_pred_1, idxs_1)
        # pcl_knn_to_2 = pcl_from_knnidx(pcl_pred_2, idxs_2)
        # pytorch3d.ops.knn_points(pclpad_gt_1, pclpad_pred_1, num_points_per_cloud, lengths2: Optional[torch.Tensor] = None, K: int = 1, version: int = -1, return_nn: bool = False, return_sorted: bool = True)

        ### distance kernel 
        ell = self.ell_min + self.ell_rand
        length_scale_1 = ell * (pcl_gt_1.points_padded()[:,:, [2]]-self.ell_basedist) / self.ell_basedist    # points_padded is (N, P, D), length_scale is (N,P,1)
        length_scale_1 = length_scale_1.clamp(min=ell).pow(2)
        length_scale_2 = ell * (pcl_gt_2.points_padded()[:,:, [2]]-self.ell_basedist) / self.ell_basedist    # points_padded is (N, P, D), length_scale is (N,P,1)
        length_scale_2 = length_scale_2.clamp(min=ell).pow(2)

        dist_kernel_1 = torch.exp( - dists_1 / length_scale_1 )
        dist_kernel_flowed_1 = torch.exp( - dists_1_from_2 / length_scale_1 )
        dist_kernel_2 = torch.exp( - dists_2 / length_scale_2 )
        dist_kernel_flowed_2 = torch.exp( - dists_2_from_1 / length_scale_2 )

        ### color kernel
        color_scale = 0.2
        pcl_hsv_knn_to_1 = knn_gather(pcl_pred_1.features_padded()[:, :, :3], idxs_1, pcl_pred_1.num_points_per_cloud() )   # (N, P1, K, D)
        pcl_hsv_knn_to_2 = knn_gather(pcl_pred_2.features_padded()[:, :, :3], idxs_2, pcl_pred_2.num_points_per_cloud() )
        pcl_hsv_knn_flowed_to_1 = knn_gather(pcl_flowed_1_from_2.features_padded()[:, :, :3], idxs_1_from_2, pcl_flowed_1_from_2.num_points_per_cloud() )
        pcl_hsv_knn_flowed_to_2 = knn_gather(pcl_flowed_2_from_1.features_padded()[:, :, :3], idxs_2_from_1, pcl_flowed_2_from_1.num_points_per_cloud() )

        color_dist_1 = (pcl_gt_1.features_padded()[:, :, :3].unsqueeze(2) - pcl_hsv_knn_to_1).norm(-1) # .pow(2).sum(-1)  # (N, P1, K)
        color_dist_2 = (pcl_gt_2.features_padded()[:, :, :3].unsqueeze(2) - pcl_hsv_knn_to_2).norm(-1) # .pow(2).sum(-1)  # (N, P2, K)
        color_dist_flowed_1 = (pcl_gt_1.features_padded()[:, :, :3].unsqueeze(2) - pcl_hsv_knn_flowed_to_1).norm(-1) # .pow(2).sum(-1)  # (N, P1, K)
        color_dist_flowed_2 = (pcl_gt_2.features_padded()[:, :, :3].unsqueeze(2) - pcl_hsv_knn_flowed_to_2).norm(-1) # .pow(2).sum(-1)  # (N, P2, K)

        color_kernel_1 = torch.exp( - color_dist_1 / color_scale )
        color_kernel_2 = torch.exp( - color_dist_2 / color_scale )
        color_kernel_flowed_1 = torch.exp( - color_dist_flowed_1 / color_scale )
        color_kernel_flowed_2 = torch.exp( - color_dist_flowed_2 / color_scale )

        ### normal kernel

        # ### calculate normal
        # normal_gt_1 = estimate_pointcloud_normals(pcl_gt_1) # (N, P, 3)
        # normal_gt_2 = estimate_pointcloud_normals(pcl_gt_2) 
        # normal_pred_1 = estimate_pointcloud_normals(pcl_pred_1) 
        # normal_pred_2 = estimate_pointcloud_normals(pcl_pred_2) 
        # if flow_dict_1to2 is not None and flow_dict_2to1 is not None:
        #     normal_flowed_2_from_1 = estimate_pointcloud_normals(pcl_flowed_2_from_1) 
        #     normal_flowed_1_from_2 = estimate_pointcloud_normals(pcl_flowed_1_from_2)
        
        ### calculate normal
        normal_gt_1 = pcl_gt_1.normals_padded()
        normal_gt_2 = pcl_gt_2.normals_padded()
        normal_pred_1 = pcl_pred_1.normals_padded()
        normal_pred_2 = pcl_pred_2.normals_padded()
        if flow_dict_1to2 is not None and flow_dict_2to1 is not None:
            normal_flowed_2_from_1 = pcl_flowed_2_from_1.normals_padded()
            normal_flowed_1_from_2 = pcl_flowed_1_from_2.normals_padded()

        ### nres
        nres_gt_1 = pcl_gt_1.features_padded()[:,:,[3]]
        nres_gt_2 = pcl_gt_2.features_padded()[:,:,[3]]
        nres_pred_1 = pcl_pred_1.features_padded()[:,:,[3]]
        nres_pred_2 = pcl_pred_2.features_padded()[:,:,[3]]
        if flow_dict_1to2 is not None and flow_dict_2to1 is not None:
            nres_flowed_2_from_1 = pcl_flowed_2_from_1.features_padded()[:,:,[3]]
            nres_flowed_1_from_2 = pcl_flowed_1_from_2.features_padded()[:,:,[3]]

        # float res = pts_nres[0][0][in] + grid_nres[ib][0][v+innh][u+innw];
        #   float alpha = 2 * mag_min / (2*mag_min/mag_max + res);

        normal_knn_to_1 = knn_gather(normal_pred_1, idxs_1, pcl_pred_1.num_points_per_cloud() )   # (N, P1, K, D)
        normal_knn_to_2 = knn_gather(normal_pred_2, idxs_2, pcl_pred_2.num_points_per_cloud() )   # (N, P1, K, D)
        normal_knn_flowed_to_1 = knn_gather(normal_flowed_1_from_2, idxs_1_from_2, pcl_flowed_1_from_2.num_points_per_cloud() )   # (N, P1, K, D)
        normal_knn_flowed_to_2 = knn_gather(normal_flowed_2_from_1, idxs_2_from_1, pcl_flowed_2_from_1.num_points_per_cloud() )   # (N, P1, K, D)

        nres_knn_to_1 = knn_gather(nres_pred_1, idxs_1, pcl_pred_1.num_points_per_cloud() )   # (N, P1, K, D)
        nres_knn_to_2 = knn_gather(nres_pred_2, idxs_2, pcl_pred_2.num_points_per_cloud() )   # (N, P1, K, D)
        nres_knn_flowed_to_1 = knn_gather(nres_flowed_1_from_2, idxs_1_from_2, pcl_flowed_1_from_2.num_points_per_cloud() )   # (N, P1, K, D)
        nres_knn_flowed_to_2 = knn_gather(nres_flowed_2_from_1, idxs_2_from_1, pcl_flowed_2_from_1.num_points_per_cloud() )   # (N, P1, K, D)

        alpha_1 = 2 * self.res_mag_min / (2*self.res_mag_min/self.res_mag_max + nres_gt_1.unsqueeze(2) + nres_knn_to_1 ).squeeze(-1)
        alpha_2 = 2 * self.res_mag_min / (2*self.res_mag_min/self.res_mag_max + nres_gt_2.unsqueeze(2) + nres_knn_to_2 ).squeeze(-1)
        alpha_flowed_1 = 2 * self.res_mag_min / (2*self.res_mag_min/self.res_mag_max + nres_gt_1.unsqueeze(2) + nres_knn_flowed_to_1 ).squeeze(-1)
        alpha_flowed_2 = 2 * self.res_mag_min / (2*self.res_mag_min/self.res_mag_max + nres_gt_2.unsqueeze(2) + nres_knn_flowed_to_2 ).squeeze(-1)

        normal_kernel_1 = ((normal_gt_1.unsqueeze(2) * normal_knn_to_1).sum(-1)*alpha_1).clamp(min=0)
        normal_kernel_2 = ((normal_gt_2.unsqueeze(2) * normal_knn_to_2).sum(-1)*alpha_2).clamp(min=0)
        normal_kernel_flowed_1 = ((normal_gt_1.unsqueeze(2) * normal_knn_flowed_to_1).sum(-1)*alpha_flowed_1).clamp(min=0)
        normal_kernel_flowed_2 = ((normal_gt_2.unsqueeze(2) * normal_knn_flowed_to_2).sum(-1)*alpha_flowed_2).clamp(min=0)

        ### final kernel
        mask_1 = mask_from_pcl(pcl_gt_1)
        mask_2 = mask_from_pcl(pcl_gt_2)

        kernel_1 = (dist_kernel_1 * color_kernel_1 * normal_kernel_1 * mask_1).sum() / (pcl_gt_1.num_points_per_cloud().sum()*self.num_nn)
        kernel_2 = (dist_kernel_2 * color_kernel_2 * normal_kernel_2 * mask_2).sum() / (pcl_gt_2.num_points_per_cloud().sum()*self.num_nn)

        kernel_flowed_1 = (dist_kernel_flowed_1 * color_kernel_flowed_1 * normal_kernel_flowed_1 * mask_1).sum() / (pcl_gt_1.num_points_per_cloud().sum()*self.num_nn)
        kernel_flowed_2 = (dist_kernel_flowed_2 * color_kernel_flowed_2 * normal_kernel_flowed_2 * mask_2).sum() / (pcl_gt_2.num_points_per_cloud().sum()*self.num_nn)

        if self.log_loss:
            inp = kernel_1.log() + kernel_2.log() + kernel_flowed_1.log() + kernel_flowed_2.log()
        else:
            inp = kernel_1 + kernel_2 + kernel_flowed_1 + kernel_flowed_2
        
        return inp


    def load_pcl(self, cam_info, depth_img_dict, flow_dict=None ):

        ##############################################################
        ### unpack depth related data and construct pointclouds
        ##############################################################
        depth = depth_img_dict["pred"]
        depth_mask = depth_img_dict["pred_mask"]
        depth_gt = depth_img_dict["gt"]
        depth_gt_mask = depth_img_dict["gt_mask"]
        rgb = depth_img_dict['rgb']
        hsv = rgb_to_hsv(rgb, flat=False)           # B*3*H*W
    
        batch_size = rgb.shape[0]
        
        K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()

        pcl_pred = load_pcl_from_unpacked(depth_mask, depth, xy1_grid_cur, hsv, dense_normal_op=self.normal_op_dense, K_cur=K_cur)
        pcl_gt = load_pcl_from_unpacked(depth_gt_mask, depth_gt, xy1_grid_cur, hsv, sparse_nml_opts=self.sparse_nml_opts, uvb_grid_cur=uvb_grid_cur)
        
        ##############################################################
        ### unpack flow related data and construct pointclouds
        ##############################################################
        pcl_flowed = None
        if flow_dict is not None:
            flow_pred = flow_dict["pred"]
            flow_mask = flow_dict["mask"]
            flow_mask = flow_mask & depth_mask

            pcl_flowed = load_pcl_from_unpacked(flow_mask, depth, xy1_grid_cur, hsv, flow_pred, dense_normal_op=self.normal_op_dense, K_cur=K_cur)

        return pcl_pred, pcl_gt, pcl_flowed


class C3dLossKnnBtwnGT(nn.Module):
    def __init__(self):
        super(C3dLossKnnBtwnGT, self).__init__()

        self.num_nn = 20
        self.ell_basedist = 10
        self.ell_min = 0.05
        self.ell_rand = 0.1
        self.log_loss = False

        self.normal_op_dense = NormalFromDepthDense()
        self.sparse_nml_opts = EasyDict() # nml_opts.neighbor_range, nml_opts.ignore_ib, nml_opts.min_dist_2
        self.sparse_nml_opts.normal_nrange = 9
        self.sparse_nml_opts.ignore_ib = False
        self.sparse_nml_opts.min_dist_2 = 0.05
        self.res_mag_min = 0.1
        self.res_mag_max = 2


    def forward(self, depth_img_dict_1, depth_img_dict_2, cam_info, R12, t12, R21, t21):

        ### format them into point clouds 
        pcl_gt_1, pcl_gt_1_in_2 = self.load_pcl(cam_info, depth_img_dict_1, R21, t21)
        pcl_gt_2, pcl_gt_2_in_1 = self.load_pcl(cam_info, depth_img_dict_2, R12, t12)


        ### knn
        dists_1, idxs_1, pcl_knn_to_1 = knn_pcl(pcl_gt_1, pcl_gt_2_in_1, self.num_nn) #(N, P1, D), (N, P2, D) -> (N, P1, K), (N, P1, K), (N, P1, K, D)
        dists_2, idxs_2, pcl_knn_to_2 = knn_pcl(pcl_gt_2, pcl_gt_1_in_2, self.num_nn)

        ### distance kernel 
        ell = self.ell_min + self.ell_rand
        length_scale_1 = ell * (pcl_gt_1.points_padded()[:,:, [2]]-self.ell_basedist) / self.ell_basedist    # points_padded is (N, P, D), length_scale is (N,P,1)
        length_scale_1 = length_scale_1.clamp(min=ell).pow(2)
        length_scale_2 = ell * (pcl_gt_2.points_padded()[:,:, [2]]-self.ell_basedist) / self.ell_basedist    # points_padded is (N, P, D), length_scale is (N,P,1)
        length_scale_2 = length_scale_2.clamp(min=ell).pow(2)

        dist_kernel_1 = torch.exp( - dists_1 / length_scale_1 )
        dist_kernel_2 = torch.exp( - dists_2 / length_scale_2 )

        ### color kernel
        color_scale = 0.2
        pcl_hsv_knn_to_1 = knn_gather(pcl_gt_2_in_1.features_padded()[:, :, :3], idxs_1, pcl_gt_2_in_1.num_points_per_cloud() )   # (N, P1, K, D)
        pcl_hsv_knn_to_2 = knn_gather(pcl_gt_1_in_2.features_padded()[:, :, :3], idxs_2, pcl_gt_1_in_2.num_points_per_cloud() )

        color_dist_1 = (pcl_gt_1.features_padded()[:, :, :3].unsqueeze(2) - pcl_hsv_knn_to_1).norm(-1) # .pow(2).sum(-1)  # (N, P1, K)
        color_dist_2 = (pcl_gt_2.features_padded()[:, :, :3].unsqueeze(2) - pcl_hsv_knn_to_2).norm(-1) # .pow(2).sum(-1)  # (N, P2, K)

        color_kernel_1 = torch.exp( - color_dist_1 / color_scale )
        color_kernel_2 = torch.exp( - color_dist_2 / color_scale )

        ### calculate normal
        normal_gt_1 = pcl_gt_1.normals_padded()
        normal_gt_2 = pcl_gt_2.normals_padded()
        normal_gt_1_in_2 = pcl_gt_1_in_2.normals_padded()
        normal_gt_2_in_1 = pcl_gt_2_in_1.normals_padded()
        
        ### nres
        nres_gt_1 = pcl_gt_1.features_padded()[:,:,[3]]
        nres_gt_2 = pcl_gt_2.features_padded()[:,:,[3]]
        nres_gt_1_in_2 = pcl_gt_1_in_2.features_padded()[:,:,[3]]
        nres_gt_2_in_1 = pcl_gt_2_in_1.features_padded()[:,:,[3]]

        normal_knn_to_1 = knn_gather(normal_gt_2_in_1, idxs_1, pcl_gt_2_in_1.num_points_per_cloud() )   # (N, P1, K, D)
        normal_knn_to_2 = knn_gather(normal_gt_1_in_2, idxs_2, pcl_gt_1_in_2.num_points_per_cloud() )   # (N, P1, K, D)

        nres_knn_to_1 = knn_gather(nres_gt_2_in_1, idxs_1, pcl_gt_2_in_1.num_points_per_cloud() )   # (N, P1, K, D)
        nres_knn_to_2 = knn_gather(nres_gt_1_in_2, idxs_2, pcl_gt_1_in_2.num_points_per_cloud() )   # (N, P1, K, D)
        
        alpha_1 = 2 * self.res_mag_min / (2*self.res_mag_min/self.res_mag_max + nres_gt_1.unsqueeze(2) + nres_knn_to_1 ).squeeze(-1)
        alpha_2 = 2 * self.res_mag_min / (2*self.res_mag_min/self.res_mag_max + nres_gt_2.unsqueeze(2) + nres_knn_to_2 ).squeeze(-1)

        normal_kernel_1 = ((normal_gt_1.unsqueeze(2) * normal_knn_to_1).sum(-1)*alpha_1).clamp(min=0)
        normal_kernel_2 = ((normal_gt_2.unsqueeze(2) * normal_knn_to_2).sum(-1)*alpha_2).clamp(min=0)
        
        ### final kernel
        mask_1 = mask_from_pcl(pcl_gt_1)
        mask_2 = mask_from_pcl(pcl_gt_2)

        kernel_1 = (dist_kernel_1 * color_kernel_1 * normal_kernel_1 * mask_1).sum() / (pcl_gt_1.num_points_per_cloud().sum()*self.num_nn)
        kernel_2 = (dist_kernel_2 * color_kernel_2 * normal_kernel_2 * mask_2).sum() / (pcl_gt_2.num_points_per_cloud().sum()*self.num_nn)

        if self.log_loss:
            inp = (kernel_1.log() + kernel_2.log()) / 2
        else:
            inp = (kernel_1 + kernel_2) / 2

        return inp


    def load_pcl(self, cam_info, depth_img_dict, R, t ):

        ##############################################################
        ### unpack depth related data and construct pointclouds
        ##############################################################
        depth_gt = depth_img_dict["gt"]
        depth_gt_mask = depth_img_dict["gt_mask"]
        rgb = depth_img_dict['rgb']
        hsv = rgb_to_hsv(rgb, flat=False)           # B*3*H*W
    
        batch_size = rgb.shape[0]
        
        K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()

        pcl_gt = load_pcl_from_unpacked(depth_gt_mask, depth_gt, xy1_grid_cur, hsv, sparse_nml_opts=self.sparse_nml_opts, uvb_grid_cur=uvb_grid_cur)
        pcl_gt_transformed = load_pcl_from_unpacked(depth_gt_mask, depth_gt, xy1_grid_cur, hsv, sparse_nml_opts=self.sparse_nml_opts, uvb_grid_cur=uvb_grid_cur, R=R, t=t)
        
        return pcl_gt, pcl_gt_transformed
