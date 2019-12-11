from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cross_prod_cuda',
    ext_modules=[
        CUDAExtension('cross_prod_cuda', [
            'cross_prod_cuda.cpp',
            'cross_prod_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
