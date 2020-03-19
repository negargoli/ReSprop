from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='backward_cpp',
    ext_modules=[
        CppExtension('backward_cpp', ['backward.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
