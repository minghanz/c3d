# c3d
Package for computing Continuous 3D Loss. This package implements the C3D loss (continuous 3D loss) in the paper [Monocular Depth Prediction Through Continuous 3D Loss](https://arxiv.org/abs/2003.09763). 

Video Demo: 

[![Screenshot](https://img.youtube.com/vi/gDfAfD4yHuM/maxresdefault.jpg)](https://youtu.be/gDfAfD4yHuM)

## Overview
The interface for using the C3D loss is the `C3DLoss` class in `c3d_loss.py`. You can import the class to your script to augment your own depth prediction project with C3D loss. The implementation of its operations are in `cvo_funcs.py` and `cvo_ops` folder. The class also needs some other util classes to run properly, which you can find in `utils` folder. 

## Getting Started

### Prerequisite
* Python 3.6
* Pytorch 1.2
* PIL

This is the environment I used to work properly. Higher versions are likely to be fine, but I did encountered problem of divergence in no_grad mode when working on another environment with high version of Pytorch. The exact problem is not located yet. 

### Installing
To use the C3D loss, you need to build the custom operation to a shared library. 
```
cd cvo_ops
./install.sh
```
### Deploying in your own network
1. You will need to create an instance for these classes: 
* `C3DLoss` (for calculating the proposed loss), 
* `CamProj` (for generating camera intrinsic and extrinsics information needed by `C3DLoss`), and 
* `DataReaderKITTI` (enabling easier retrieval of camera intrinsic and extrinsics information from KITTI dataset). 
2. `C3DLoss` can be simply initialized as `c3d_loss = C3DLoss()`. Optionally you can create a config text file to customize the parameters. See `c3d_config_example.txt` for an example. `C3DLoss.parse_opts(f_input=$path_to_config_file$)` set the parameters specified in the config file. 
3. `CamProj` and `DataReaderKITTI` objects should be initialized in the dataset-handling part of your code. `DataReaderKITTI` takes the root path to your local KITTI dataset as initialization argument. `CamProj` takes the `DataReaderKITTI` object as initialization argument. 
4. Camera intrinsic and extrinsics information are encapsuled in the `CamInfo` class, which is output by a `CamProj` object. `CamInfo` object is needed in `C3DLoss` forward calculation, together with RGB image, depth prediction, ground truth, and validity masks. 

Implementation of the C3D loss in DORN is open-sourced. Please check [here](https://github.com/minghanz/SupervisedDepthPrediction) as an example of using the C3D loss in an existing network. 

Implementation of the C3D loss in BTS and Monodepth2 network will be released soon. 

## Note
This repo is still under heavy constructions. The implementation is subject to change. More comments and instructions would come later. Your comments are welcomed. 
