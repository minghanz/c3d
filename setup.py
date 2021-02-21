
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# setup(
#     name='c3d',
#     description='Package for Continuous 3D Loss',
#     author='Minghan Zhu',
#     author_email='minghanz@umich.edu',
#     url='https://github.com/minghanz/c3d',
#     packages=find_packages(),
#     )

torch_vs = (torch.__version__).split('.')
torch_version = float(torch_vs[0]) + 0.1 * float(torch_vs[1])

if torch_version > 1.2:
    ######### for pytorch 1.6
    cxx_args = ['-std=c++14']   # for pytorch 1.6. For pytorch 1.2 use c++11 or do not use extra_compile_args

    nvcc_args = [
        '-gencode', 'arch=compute_50,code=sm_50',
        '-gencode', 'arch=compute_52,code=sm_52',
        '-gencode', 'arch=compute_60,code=sm_60',
        '-gencode', 'arch=compute_61,code=sm_61',
        '-gencode', 'arch=compute_61,code=compute_61'
    ]

    extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
else:
    ######### for pytorch 1.2
    extra_compile_args = dict()

setup(
    name='c3d',
    description='Package for Continuous 3D Loss',
    author='Minghan Zhu',
    author_email='minghanz@umich.edu',
    url='https://github.com/minghanz/c3d',
    ext_modules=[
        CUDAExtension('c3d.cvo_ops.sub_cuda', [
            'c3d/cvo_ops/custom_ori/sub_cuda.cpp',
            'c3d/cvo_ops/custom_ori/sub_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.sub_norm_cuda_half_paral', [
            'c3d/cvo_ops/custom_norm/sub_norm_cuda.cpp',
            'c3d/cvo_ops/custom_norm/sub_norm_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cvo_dense_with_normal_output', [
            'c3d/cvo_ops/custom_dense_with_normal/cvo_dense_with_normal_cuda.cpp',
            'c3d/cvo_ops/custom_dense_with_normal/cvo_dense_with_normal_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cvo_dense_normal', [
            'c3d/cvo_ops/custom_dense_normal/cvo_dense_normal_cuda.cpp',
            'c3d/cvo_ops/custom_dense_normal/cvo_dense_normal_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cvo_dense_dist', [
            'c3d/cvo_ops/custom_dense_dist/cvo_dense_samp_cuda.cpp',
            'c3d/cvo_ops/custom_dense_dist/cvo_dense_samp_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cvo_dense_angle', [
            'c3d/cvo_ops/custom_dense_angle/cvo_dense_angle_cuda.cpp',
            'c3d/cvo_ops/custom_dense_angle/cvo_dense_angle_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cvo_dense_samp', [
            'c3d/cvo_ops/custom_dense/cvo_dense_samp_cuda.cpp',
            'c3d/cvo_ops/custom_dense/cvo_dense_samp_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cross_subtract_cuda', [
            'c3d/cvo_ops/custom_cross_subtract/cross_subtract_cuda.cpp',
            'c3d/cvo_ops/custom_cross_subtract/cross_subtract_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cross_prod_cuda', [
            'c3d/cvo_ops/custom_cross_prod/cross_prod_cuda.cpp',
            'c3d/cvo_ops/custom_cross_prod/cross_prod_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cvo_dense_Sigma', [
            'c3d/cvo_ops/custom_dense_Sigma/cvo_dense_Sigma_cuda.cpp',
            'c3d/cvo_ops/custom_dense_Sigma/cvo_dense_Sigma_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        CUDAExtension('c3d.cvo_ops.cvo_dense_Sigma_grid', [
            'c3d/cvo_ops/custom_dense_Sigma/cvo_dense_Sigma_grid_cuda.cpp',
            'c3d/cvo_ops/custom_dense_Sigma/cvo_dense_Sigma_grid_cuda_kernel.cu',
        ], extra_compile_args=extra_compile_args),
        
    ],
    packages=find_packages(),
    cmdclass={
        'build_ext': BuildExtension
    })
