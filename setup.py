from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os
import torch

path_to_cpp = "morphocore/functional/csrc/cpp"
path_to_cuda = "morphocore/functional/csrc/cuda"

cpp_files = ["morpho_op.cpp", "reshaper.cpp", "smorph.cpp", "bindings.cpp"]
cuda_files = ["smorph.cu", "morpho_op.cu"]

cpp_sources = [os.path.join(path_to_cpp, file) for file in cpp_files]
cuda_sources = [os.path.join(path_to_cuda, file) for file in cuda_files]

# Détection automatique de CUDA
with_cuda = torch.cuda.is_available()

if with_cuda:
    Extension = CUDAExtension
    sources = cpp_sources + cuda_sources
    define_macros = [('WITH_CUDA', '1')]
    
    extra_compile_args = {
        'cxx': [
            '-O3',                    # Optimisation maximale
            '-DNDEBUG',              # Retire les assertions debug  
            '-ffast-math',           # Optimisations mathématiques
            '-fopenmp',              # Parallélisation OpenMP
            '-march=native',         # Utilise toutes les instructions CPU disponibles
            '-std=c++17',
            '-Wunused-but-set-variable',
            '-finline-functions'
        ],
        'nvcc': [
            '-O3',                   # Optimisation maximale
            '--use_fast_math',       # Math rapide sur GPU
            '-std=c++17',
            
            # Support des architectures courantes (2018-2024)
            '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx, Tesla T4
            '-gencode=arch=compute_80,code=sm_80',  # RTX 30xx, A100  
            '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx Ti/Super
            '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
            
            # Optimisations registres
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
        'cxx': [
            '-O3',                    # Optimisation maximale
            '-DNDEBUG',              # Retire les assertions debug
            '-ffast-math',           # Optimisations mathématiques  
            '-fopenmp',              # Parallélisation OpenMP
            '-march=native',         # Utilise toutes les instructions CPU disponibles
            '-std=c++17',
            '-finline-functions'
        ]
    }

setup(
    name="morphocore",
    packages=find_packages(),
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
            extra_compile_args=extra_compile_args,
            extra_link_args=['-fopenmp'],  # Link OpenMP
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version="0.0.1",
)