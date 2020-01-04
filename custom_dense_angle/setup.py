from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cvo_dense_angle',
    ext_modules=[
        CUDAExtension('cvo_dense_angle', [
            'cvo_dense_angle_cuda.cpp',
            'cvo_dense_angle_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
