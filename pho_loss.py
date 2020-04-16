import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils.cam_proj import *

torch_vs = (torch.__version__).split('.')
torch_version = float(torch_vs[0]) + 0.1 * float(torch_vs[1])

class PhoLoss(CamProj):

    def __init__(self, *args, **kwargs):
        super(PhoLoss, self).__init__(*args, **kwargs)
        self.ssim = SSIM()
        

    def forward(self, rgb, depth, Ts, date_side=None, xy_crop=None, intr=None):
        assert date_side is not None or intr is not None
        date_side = (date_side[0][0], int(date_side[1][0]) ) # originally it is a list. Take the first since the mini_batch share the same intrinsics. 

        batch_size = rgb.shape[0]       ## if drop_last is False in Sampler/DataLoader, then the batch_size is not constant. 

        cam_info = self.prepare_cam_info(date_side, xy_crop, intr, batch_size, rgb.device)

        rgb_recsts, reprj_errs = wrap_image(rgb, depth, Ts, cam_info, self.ssim)
        # for item in reprj_errs:
        #     print(item.shape)
        
        return rgb_recsts, torch.cat(reprj_errs).mean()

def wrap_image(rgb, depth, Ts, cam_info, ssim_op):
    '''
    back project
    transform
    project
    F.grid_sample
    calc_difference
    '''

    K_cur, width_cur, height_cur, xy1_grid_cur, uvb_grid_cur = cam_info.unpack()

    ## back project
    xyz_grid = xy1_grid_cur * depth
    xyz_flat = xyz_grid.reshape(xyz_grid.shape[0], 3, -1)

    ## get relative pose
    T, R, t = relative_T(Ts)

    ## transform
    # reverse_tid = [target_id.index(i) for i in range(len(target_id))]
    # xyz_flat_transed = torch.stack( [torch.matmul(R[i], xyz_flat[i]) + t[i] for i in reverse_tid], dim=0)
    rgb_recsts = wrap_xyz_group(xyz_flat, rgb, R, t, K_cur, width_cur, height_cur)

    ## reprojection error
    reprj_errs = reproj_error_group(rgb_recsts, rgb, ssim_op)
    return rgb_recsts, reprj_errs

def wrap_xyz_group(xyz_flat, rgb, Rs, ts, K_cur, width, height):
    batch_size = rgb.shape[0]
    assert batch_size >= 3
    n_center = batch_size - 3 + 1

    rgb_recsts = []
    for n_g in range(n_center):
        ic = n_g + 1
        iT0 = 2*n_g
        iT1 = 2*n_g + 1
        rgb_recst = wrap_xyz(xyz_flat[[ic]], Rs[iT0], ts[iT0], K_cur[[ic-1]], rgb[[ic-1]], rgb[[ic]], width, height)
        rgb_recsts.append(rgb_recst)
        # reprj_loss = compute_reprojection_loss(rgb_recst, rgb[[ic]]) ## TODO: ssim
        # reprj_losses.append(reprj_loss)
        rgb_recst = wrap_xyz(xyz_flat[[ic]], Rs[iT1], ts[iT1], K_cur[[ic+1]], rgb[[ic+1]], rgb[[ic]], width, height)
        # reprj_loss = compute_reprojection_loss(rgb_recst, rgb[[ic]]) ## TODO: ssim
        # reprj_losses.append(reprj_loss)
        rgb_recsts.append(rgb_recst)

    return rgb_recsts

def wrap_xyz(xyz, R, t, K, rgb_source, rgb_target, width, height, align_corner=True):
    xyz_trs = torch.matmul(R, xyz) + t
    uvz_trs = torch.matmul(K, xyz_trs)
    uv_trs = uvz_trs[:, :2] / uvz_trs[:, [2]]   ## matlab -1 is now in K in K_mat2py in dataset_kitti.py
    if align_corner:    # [0, w-1] -> [0, 1]
        uv_trs[:, 0] = uv_trs[:, 0] / (width - 1)
        uv_trs[:, 1] = uv_trs[:, 1] / (height - 1)
    else:               # [-0.5, w-0.5] -> [0, 1]
        uv_trs[:, 0] = (uv_trs[:, 0] + 0.5) / width
        uv_trs[:, 1] = (uv_trs[:, 1] + 0.5) / height
    uv_trs = (uv_trs - 0.5) * 2     # from [0, 1] to [-1, 1]

    uv_trs = uv_trs.reshape(rgb_target.shape[0], 2, rgb_target.shape[2], rgb_target.shape[3])
    uv_trs = uv_trs.permute(0, 2, 3, 1)
    if torch_version <= 1.2:
        if not align_corner:
            raise ValueError('torch_version <= 1.2, can only work with align_corner=True')
        rgb_recst = F.grid_sample(rgb_source, uv_trs, padding_mode="border")
    else:
        rgb_recst = F.grid_sample(rgb_source, uv_trs, padding_mode="border", align_corners=align_corner)

    return rgb_recst

def reproj_error_group(rgb_recsts, rgb, ssim_op):
    batch_size = rgb.shape[0]
    assert batch_size >= 3
    n_center = batch_size - 3 + 1

    reprj_errs = []
    for n_g in range(n_center):
        ic = n_g + 1
        iT0 = 2*n_g
        iT1 = 2*n_g + 1
        reprj_err0 = compute_reprojection_loss(rgb_recsts[iT0], rgb[[ic]], ssim_op)
        reprj_err1 = compute_reprojection_loss(rgb_recsts[iT1], rgb[[ic]], ssim_op)
        reprj_pick = torch.min(reprj_err0, reprj_err1)
        reprj_errs.append(reprj_pick)

    return reprj_errs

def compute_reprojection_loss(pred, target, ssim_op=None):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if ssim_op is None:
        reprojection_loss = l1_loss
    else:
        ssim_loss = ssim_op(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


def relative_T(Ts):
    batch_size = Ts.shape[0]
    assert batch_size >= 3
    n_center = batch_size - 3 + 1

    ## get relative pose
    T = []
    R = []
    t = []

    for n_g in range(n_center):
        ic = n_g + 1
        T1 = Ts[ic]
        T0 = Ts[ic-1]
        T2 = Ts[ic+1]
        T01 = torch.matmul( torch.inverse(T0), T1 )
        R01 = T01[:3, :3]
        t01 = T01[:3, [3]]
        T.append(T01)
        R.append(R01)
        t.append(t01)
        T21 = torch.matmul( torch.inverse(T2), T1 )
        R21 = T21[:3, :3]
        t21 = T21[:3, [3]]
        T.append(T21)
        R.append(R21)
        t.append(t21)

    return T, R, t


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)