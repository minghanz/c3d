'''
Calculate exponential kernel exp(-|a-b|^2)/(2*l^2) or exp(-|a-b|)/l
1*C*N x B*C*H*W -> 1*NN*N
'''

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cvo_dense_samp',
    ext_modules=[
        CUDAExtension('cvo_dense_samp', [
            'cvo_dense_samp_cuda.cpp',
            'cvo_dense_samp_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
