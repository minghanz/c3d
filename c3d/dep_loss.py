import torch
import torch.nn as nn

class DepthL1Loss(nn.Module):
    def __init__(self, inbalance_to_closer):
        super(DepthL1Loss, self).__init__()
        self.inbalance_to_closer = inbalance_to_closer
    def forward(self, depth_est, depth_gt, mask):
        if self.inbalance_to_closer == 1:
            d = torch.abs(depth_est[mask] - depth_gt[mask]).mean()
        else:
            err = depth_est[mask] - depth_gt[mask]
            err_pos = self.inbalance_to_closer * err[err>0]
            err_neg = -err[err<0]
            total_num = err.numel() # mask.sum().to(dtype=torch.float32)+1e-8
            d = (err_pos.sum() + err_neg.sum()) / total_num
        return d