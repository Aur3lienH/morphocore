from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os
import torch

path_to_cpp = "morphocore/functional/csrc/cpp"
path_to_cuda = "morphocore/functional/csrc/cuda"

cpp_files = ["dilation.cpp", "erosion.cpp", "reshaper.cpp", "smorph.cpp", "bindings.cpp"]
cuda_files = ["dilation.cu", "erosion.cu", "smorph.cu"]

cpp_sources = [os.path.join(path_to_cpp, file) for file in cpp_files]
cuda_sources = [os.path.join(path_to_cuda, file) for file in cuda_files]

# Détection automatique de CUDA
with_cuda = torch.cuda.is_available()

if with_cuda:
    Extension = CUDAExtension
    sources = cpp_sources + cuda_sources
    define_macros = [('WITH_CUDA', '1')]
    extra_compile_args = {
        'cxx': ['-O3', '-fopenmp'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-gencode=arch=compute_86,code=sm_86',
            '--maxrregcount=72',
            '-Xcompiler=-fopenmp',
        ]
    }
else:
    Extension = CppExtension
    sources = cpp_sources
    define_macros = []
    extra_compile_args = {'cxx': ['-O3', '-fopenmp']}

setup(
    name="morphocore",
    packages=find_packages("morphocore"),
    package_dir={"": "morphocore"},
    include_package_data=True,
    
    ext_modules=[
        Extension(
            name="cpp_operation",
            sources=sources,
            include_dirs=[
                'morphocore/functional/csrc',
                'morphocore/functional/csrc/cpp', 
                'morphocore/functional/csrc/cuda'
            ],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version="0.0.1",
)