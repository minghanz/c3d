from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sub_cuda',
    ext_modules=[
        CUDAExtension('sub_cuda', [
            'sub_cuda.cpp',
            'sub_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
