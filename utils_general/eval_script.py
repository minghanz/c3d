"""This file is to check whether the c3d.Metrics class do evaluation correctly. 
Download the DORN result and evaluate it with c3d.Metrics. Compare it with the result reported in 
https://github.com/hufu6371/DORN/issues/13#issuecomment-514433951 
https://github.com/jahaniam/semiDepth/issues/1#issuecomment-514438234 
Turns out the results are exactly the same, meaning that the Metrics is implemented correctly. """

import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm

from .eval import Metrics

def read_ground_truth_depth(filename):
    im = Image.open(filename)
    im = np.array(im).astype(np.float32)
    im /= 256.0 # depth in metres
    return im

if __name__ == "__main__":
    pred_npy_file = "/home/minghanz/DORN_pytorch/DORN_eigen697_depth.npy"   
    ### downloaded from: https://drive.google.com/open?id=1NKFtQy3HEMSCoPUaJxVcwv4sDXLIJyKs, which is shown https://github.com/jahaniam/semiDepth/issues/1#issuecomment-514438234
    test_split_file = "eval_kitti_val.list" ### can be found in DORN_pytorch and BTS
    dataset_root = "/mnt/storage8t/minghanz/Datasets/KITTI_data/kitti_data"

    preds = np.load(pred_npy_file)
    with open(test_split_file) as f:
        lines = f.readlines()

    assert len(lines) == preds.shape[0]

    n_files = len(lines)

    metrics = Metrics(1e-3, 80, "resize", "garg_crop")  
    ### use resize here because this is what's applied in https://github.com/jahaniam/semiDepth/blob/f8d342dab3e370a0d7ab2721f5b4631f7b63ae40/utils/evaluate_kitti_depth.py#L104
    print(metrics.get_header_row())
    tbar = tqdm(enumerate(lines))
    for i, line in tbar:
        gt_file = line.split(" ")[1][:-1]
        if gt_file == "None":
            continue

        gt_path = os.path.join(dataset_root, gt_file)
        gt_depth = read_ground_truth_depth(gt_path)
        pred_depth = preds[i]

        metrics.compute_metric(pred_depth, gt_depth)

        tbar.set_description(metrics.get_snapshot_row())

    print(metrics.get_header_row())
    print(metrics.get_result_row())