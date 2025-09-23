from setuptools import setup, find_packages
import os
import torch
import sys
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

path_to_cpp = "morphocore/functional/csrc/cpp"
path_to_cuda = "morphocore/functional/csrc/cuda"

cpp_files = ["morpho_op.cpp", "reshaper.cpp", "smorph.cpp", "bindings.cpp"]
cuda_files = ["smorph.cu", "morpho_op.cu"]

cpp_sources = [os.path.join(path_to_cpp, file) for file in cpp_files]
cuda_sources = [os.path.join(path_to_cuda, file) for file in cuda_files]

# Détection automatique de CUDA
with_cuda = os.environ.get('MORPHO_WITH_CUDA', '1') == '1' and torch.cuda.is_available()

if sys.platform.startswith('linux'):
    cxx_flags = ['-O3', '-DNDEBUG', '-std=c++17', '-ffast-math', '-fopenmp', '-Wno-unused-variable']
    link_args = ['-fopenmp']
elif sys.platform.startswith('darwin'):
    cxx_flags = ['-O3', '-DNDEBUG', '-std=c++17', '-ffast-math']
    link_args = ['-lomp']
elif sys.platform.startswith('win'):
    cxx_flags = ['/O2', '/DNDEBUG', '/std:c++17', '/openmp']
    link_args = []
else:
    cxx_flags = ['-O2', '-std=c++17']
    link_args = []

if with_cuda:
    Extension = CUDAExtension
    sources = cpp_sources + cuda_sources
    define_macros = [('WITH_CUDA', '1')]
    
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_89,code=sm_89',
            '-gencode=arch=compute_120,code=sm_120',
            '--maxrregcount=96',
            '-Xcompiler=-O3',
            '-Xcompiler=-fopenmp',
            '-Wunused-but-set-variable'
        ]
    }
else:
    Extension = CppExtension
    sources = cpp_sources
    define_macros = []
    
    extra_compile_args = {
        'cxx': cxx_flags
    }

# ✨ Setup simplifié - SEULEMENT les extensions
setup(
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
            extra_compile_args=extra_compile_args,
            extra_link_args=link_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    include_package_data=False,
    
)