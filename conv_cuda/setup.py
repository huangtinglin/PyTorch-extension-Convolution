from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv_cuda',
    ext_modules=[
        CUDAExtension('conv_cuda',
                      sources=['conv_cuda.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
# , 'conv_cuda_kernel.cu'