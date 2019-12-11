from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cross_subtract_cuda',
    ext_modules=[
        CUDAExtension('cross_subtract_cuda', [
            'cross_subtract_cuda.cpp',
            'cross_subtract_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
