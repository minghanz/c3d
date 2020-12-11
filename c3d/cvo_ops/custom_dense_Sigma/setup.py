'''
Calculate 2Dexponential kernel with 2D covariance matrix |Sigma|^(-1/2) exp(-1/2*(a-b)^T * Sigma^-1 * (a-b))
1*C*N x B*C*H*W -> 1*NN*N
'''

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cvo_dense_Sigma',
    ext_modules=[
        CUDAExtension('cvo_dense_Sigma', [
            'cvo_dense_Sigma_cuda.cpp',
            'cvo_dense_Sigma_cuda_kernel.cu',
        ]),
        CUDAExtension('cvo_dense_Sigma_grid', [
            'cvo_dense_Sigma_grid_cuda.cpp',
            'cvo_dense_Sigma_grid_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
