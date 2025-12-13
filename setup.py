from setuptools import setup, find_packages
import os
import torch
import sys
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

sys.path.insert(0, os.path.abspath('.'))
from morphocore import __version__ as BASE_VERSION

path_to_cpp = "morphocore/functional/csrc/cpp"
path_to_cuda = "morphocore/functional/csrc/cuda"

cpp_files = ["morpho_op.cpp", "reshaper.cpp", "smorph.cpp", "bindings.cpp", "smorph_scm.cpp"]
cuda_files = ["smorph.cu", "morpho_op.cu", "morpho_op_kernels.cu", "smorph_kernels.cu"]

cpp_sources = [os.path.join(path_to_cpp, file) for file in cpp_files]
cuda_sources = [os.path.join(path_to_cuda, file) for file in cuda_files]

# Détection automatique de CUDA
with_cuda = os.environ.get('MORPHO_WITH_CUDA', '1') == '1'

if sys.platform.startswith('linux'):
    cxx_flags = ['-O3', '-DNDEBUG', '-std=c++17', '-fopenmp', '-Wno-unused-variable']
    link_args = ['-fopenmp']
elif sys.platform.startswith('darwin'):
    cxx_flags = ['-O3', '-DNDEBUG', '-std=c++17']
    link_args = ['-lomp']
elif sys.platform.startswith('win'):
    cxx_flags = ['/O2', '/DNDEBUG', '/std:c++17', '/openmp']
    link_args = []
else:
    cxx_flags = ['-O2', '-std=c++17']
    link_args = []
cxx_flags
torch_version = os.environ.get("TORCH_VERSION", torch.__version__.split('+')[0])
version = BASE_VERSION + "." + torch_version.replace(".", "")

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
            '-gencode=arch=compute_61,code=sm_61',     # Pascal: GTX 10XX
            '-gencode=arch=compute_70,code=sm_70',     # Volta: V100
            '-gencode=arch=compute_75,code=sm_75',     # Turing: RTX 20XX
            '-gencode=arch=compute_80,code=sm_80',     # Ampere: A100
            '-gencode=arch=compute_86,code=sm_86',     # Ampere: RTX 30XX
            '-gencode=arch=compute_89,code=sm_89',     # Ada: RTX 40XX
            '-gencode=arch=compute_90,code=sm_90',     # Hopper: H100
            '-gencode=arch=compute_89,code=compute_89', # PTX for RTX 50XX+ forward compat
            '--maxrregcount=96',
            '-Xcompiler=-O3',
            '-Xcompiler=-fopenmp'
            
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
        'build_ext': BuildExtension.with_options(
            no_python_abi_suffix=True,
            use_ninja=True,
        )
    },
    include_package_data=False,
    version=version,
    install_requires=[
        f"torch=={torch_version}" if torch_version else "torch>=2.0.0",
        "numpy>=1.19.0",
    ]
    
)