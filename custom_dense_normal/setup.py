from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cvo_dense_normal',
    ext_modules=[
        CUDAExtension('cvo_dense_normal', [
            'cvo_dense_normal_cuda.cpp',
            'cvo_dense_normal_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
