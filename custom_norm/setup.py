from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sub_norm_cuda_half_paral',
    ext_modules=[
        CUDAExtension('sub_norm_cuda_half_paral', [
            'sub_norm_cuda.cpp',
            'sub_norm_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
