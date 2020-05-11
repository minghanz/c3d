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

## Note
This repo is still under heavy constructions. The implementation is subject to change. More comments and instructions would come later. Your comments are welcomed. 
