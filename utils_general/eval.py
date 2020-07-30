### This file is to provide a unified interface for evaluating depth prediction results agnostic to training method and network. 
import numpy as np
import cv2
import os
try:
    import torch
except ImportError:
    import warnings
    warnings.warn("torch is not imported, cannot accept torch.Tensor input")
    ### https://stackoverflow.com/questions/3891804/raise-warning-in-python-without-interrupting-program

def compute_errors(gt, pred):
    """from bts/utils/eval_with_png.py"""
    # print("pred_depth max min", pred.max(), pred.min())

    # print("gt_depth max min", gt.max(), gt.min())

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean() * 100
    d2 = (thresh < 1.25 ** 2).mean() * 100
    d3 = (thresh < 1.25 ** 3).mean() * 100
    ### Minghan: return d1, d2, d3 as percentage

    sqr_e = (gt - pred) ** 2
    mse = sqr_e.mean()
    rmse = np.sqrt(sqr_e.mean())

    mae = np.abs(gt - pred).mean()

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    lg10 = np.mean(err)

    ### this part not returned
    inv_output_km = (1e-3 * pred) ** (-1)
    inv_target_km = (1e-3 * gt) ** (-1)
    abs_inv_diff = np.abs(inv_output_km - inv_target_km)
    irmse = np.sqrt((np.power(abs_inv_diff, 2)).mean())
    imae = abs_inv_diff.mean()
    #############################

    errs = dict()
    errs.update(irmse=irmse, imae=imae, mse=mse, mae=mae, rmse=rmse, rmse_log=rmse_log, abs_rel=abs_rel, sq_rel=sq_rel, lg10=lg10, silog=silog, d1=d1, d2=d2, d3=d3)

    return errs

# def compute_errors(gt, pred):
#     """from monodepth2"""
#     """Computation of error metrics between predicted and ground truth depths
#     """
#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25     ).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()

#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())

#     rmse_log = (np.log(gt) - np.log(pred)) ** 2
#     rmse_log = np.sqrt(rmse_log.mean())

#     abs_rel = np.mean(np.abs(gt - pred) / gt)

#     sq_rel = np.mean(((gt - pred) ** 2) / gt)

#     return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def eval_preprocess(depth_pred, depth_gt, d_min, d_max, shape_unify=None, eval_crop=None, adjust_mean=False, batch_adjust_mean=None):
    """return pred and gt which consist of only valid and corresponding points, ready to calculate error metrics.
    input shape is H*W or H*W*C for np array, or H*W or C*H*W for torch tensor
    shape_unify: one of ["kb_crop", "resize", None]
    eval_crop: one of ["garg_crop", "eigen_crop", None]
    depth_gt must be uncropped full image, otherwise eval_crop will produce unintended behavior
    the inputs are not batched.
    """
    mode="torch" if isinstance(depth_pred, torch.Tensor) else "np"

    if depth_pred.ndim == 3:
        if mode == "np":
            assert depth_pred.shape[2] == 1, depth_pred.shape
            depth_pred = depth_pred[...,0]
        else:
            assert depth_pred.shape[0] == 1, depth_pred.shape
            depth_pred = depth_pred[0]

    assert depth_pred.ndim == 2

    if depth_gt.ndim == 3:
        if mode == "np":
            assert depth_gt.shape[2] == 1, depth_gt.shape
            depth_gt = depth_gt[...,0]
        else:
            assert depth_gt.shape[0] == 1, depth_gt.shape
            depth_gt = depth_gt[0]

    assert depth_gt.ndim == 2

    ### unify the shape of depth_pred and depth_gt
    gt_height = depth_gt.shape[0]
    gt_width = depth_gt.shape[1]
    if shape_unify is None:
        assert depth_pred.shape == depth_gt.shape, "pred: {} gt: {}".format(depth_pred.shape, depth_gt.shape)
        mask_shape_unify = np.ones_like(depth_gt, dtype=np.bool) if mode == "np" else torch.ones_like(depth_gt, dtype=torch.bool)

    elif shape_unify == "kb_crop":
        assert depth_pred.shape == (352, 1216), depth_pred.shape
        top_margin = int(gt_height-352)
        left_margin = int((gt_width - 1216) / 2)
        depth_pred_uncropped = np.zeros_like(depth_gt) if mode == "np" else torch.zeros_like(depth_gt)
        depth_pred_uncropped[top_margin:top_margin+352, left_margin:left_margin+1216] = depth_pred
        depth_pred = depth_pred_uncropped
        mask_shape_unify = np.zeros_like(depth_gt, dtype=np.bool) if mode == "np" else torch.zeros_like(depth_gt, dtype=torch.bool)
        mask_shape_unify[top_margin:top_margin+352, left_margin:left_margin+1216] = True

    elif shape_unify == "resize":
        if mode == "np":
            depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
            mask_shape_unify = np.ones_like(depth_gt, dtype=np.bool)
        else:
            depth_pred_dim4 = depth_pred.unsqueeze(0).unsqueeze(0)
            depth_pred_dim4 = torch.nn.functional.interpolate(depth_pred_dim4, (depth_gt.shape[0], depth_gt.shape[1]), mode='bilinear', align_corners=False)
            depth_pred = depth_pred_dim4[0,0]
            mask_shape_unify = torch.ones_like(depth_gt, dtype=torch.bool)
    else:
        raise ValueError("shape_unify {} not recognized", shape_unify)

    ### masking
    mask_valid = depth_gt>0
    mask_value = np.logical_and(depth_gt >= d_min, depth_gt <= d_max) if mode == "np" else (depth_gt >= d_min) & (depth_gt <= d_max) # logical_and does not exist in torch 1.2
    if eval_crop is None:
        mask_crop = np.ones_like(depth_gt, dtype=np.bool) if mode == "np" else torch.ones_like(depth_gt, dtype=torch.bool)
    elif eval_crop == "garg_crop":
        mask_crop = np.zeros_like(depth_gt, dtype=np.bool) if mode == "np" else torch.zeros_like(depth_gt, dtype=torch.bool)
        mask_crop[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
    elif eval_crop == "eigen_crop":
        mask_crop = np.zeros_like(depth_gt, dtype=np.bool) if mode == "np" else torch.zeros_like(depth_gt, dtype=torch.bool)
        mask_crop[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
    elif eval_crop == "vkitti2":
        mask_crop = np.zeros_like(depth_gt, dtype=np.bool) if mode == "np" else torch.zeros_like(depth_gt, dtype=torch.bool)
        mask_crop[int(0.5 * gt_height):] = 1
    else:
        raise ValueError("eval_crop {} not recognized", eval_crop)

    mask = mask_shape_unify & mask_valid & mask_value & mask_crop

    depth_pred_masked = depth_pred[mask]
    depth_gt_masked = depth_gt[mask]

    mean_pred = depth_pred_masked.mean()
    mean_gt = depth_gt_masked.mean()
    if adjust_mean:
        assert batch_adjust_mean is None
        depth_pred_masked *= mean_gt/mean_pred
    elif batch_adjust_mean is not None:
        assert adjust_mean == False
        depth_pred_masked = depth_pred_masked / batch_adjust_mean

    ### clip prediction
    depth_pred_masked[depth_pred_masked<d_min] = d_min
    depth_pred_masked[depth_pred_masked>d_max] = d_max

    extra_dict = {}
    extra_dict["mask"] = mask
    extra_dict['mean_pred'] = mean_pred
    extra_dict['mean_gt'] = mean_gt

    return depth_pred_masked, depth_gt_masked, extra_dict

def eval_depth_error(depth_pred, depth_gt, d_min, d_max, shape_unify=None, eval_crop=None, adjust_mean=False, batch_adjust_mean=None):
    """
    shape_unify: one of ["kb_crop", "resize", None]
    eval_crop: one of ["garg_crop", "eigen_crop", None]
    depth_gt must be uncropped full image, otherwise eval_crop will produce unintended behavior
    the inputs are not batched.
    """
    depth_pred_masked, depth_gt_masked, extra_dict = eval_preprocess(depth_pred, depth_gt, d_min, d_max, shape_unify, eval_crop, adjust_mean, batch_adjust_mean)

    if isinstance(depth_pred_masked, torch.Tensor):
        depth_pred_masked = depth_pred_masked.cpu().numpy()
    if isinstance(depth_gt_masked, torch.Tensor):
        depth_gt_masked = depth_gt_masked.cpu().numpy()
    
    ### calculate error
    return compute_errors(depth_gt_masked, depth_pred_masked), extra_dict
   
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        return self.sum / self.count

    def values(self):
        return self.val 

class Metrics(object):

    def __init__(self, d_min, d_max, shape_unify, eval_crop, adjust_mean=False, batch_adjust_mean=None):

        self.d_min = d_min
        self.d_max = d_max
        self.shape_unify = shape_unify
        self.eval_crop = eval_crop
        self.adjust_mean = adjust_mean
        self.batch_adjust_mean = batch_adjust_mean

        self.eval_items = ['irmse', 'imae', 'mse', 'mae', 'rmse', 'rmse_log', 'abs_rel', 'sq_rel', 'lg10', 'silog', 'd1', 'd2', 'd3']

        self.evals = dict()
        for item in self.eval_items:
            self.evals[item] = AverageMeter()

    def reset(self):
        for item in self.eval_items:
            self.evals[item].reset()

    def compute_metric(self, pred, gt):
        """the inputs are not batched.
        """
        errs, extra_dict = eval_depth_error(pred, gt, self.d_min, self.d_max, self.shape_unify, self.eval_crop, self.adjust_mean, self.batch_adjust_mean)

        for item in self.eval_items:
            self.evals[item].update(errs[item])
        
        return extra_dict

    def add_scalar(self, writer=None, tag="Test", epoch=0):
        if writer is None:
            return

        for item in self.eval_items:
            writer.add_scalar(tag+"/{}".format(item), self.evals[item].mean(), epoch)

    def get_header_row(self):
        header = ("%10s"*len(self.eval_items))% tuple(self.eval_items)
        return header

    def get_snapshot_row(self):
        row = ("%10.3f"*len(self.eval_items))% tuple(self.evals[x].values() for x in self.eval_items)
        return row

    def get_result_row(self):
        row = ("%10.3f"*len(self.eval_items))% tuple(self.evals[x].mean() for x in self.eval_items)
        return row

    def get_snapshot_info(self):
        # info = "abs_rel: %.2f" % self.abs_rel.values() + "(%.2f)" % self.abs_rel.mean()
        # info += " rmse: %.2f" % self.rmse.values() + "(%.2f)" % self.rmse.mean()
        # return info

        info = ""
        for item in self.eval_items:
            info = info + "{}: {:.3f}".format(item, self.evals[item].values())
        return info

    def get_result_info(self):
        # info = "abs_rel: %.2f" % self.abs_rel.mean() + \
        #        " rmse: %.2f" % self.rmse.mean() + \
        #        " silog: %.2f" % self.silog.mean()

        info = ""
        for item in self.eval_items:
            info = info + "{}: {:.3f}".format(item, self.evals[item].mean())
        return info
