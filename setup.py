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
cuda_files = ["smorph.cu", "morpho_op.cu"]

cpp_sources = [os.path.join(path_to_cpp, file) for file in cpp_files]
cuda_sources = [os.path.join(path_to_cuda, file) for file in cuda_files]

# Détection automatique de CUDA
with_cuda = os.environ.get('MORPHO_WITH_CUDA', '1') == '1'

# CPU kernels notes:
# - Threads come from PyTorch's pool through at::parallel_for. With PyTorch's OpenMP backend that
#   template expands to `#pragma omp parallel` inside this library, so -fopenmp is required: without
#   it every kernel silently runs on a single thread.
# - No -march: the kernels are compiled for baseline x86-64 and for AVX2, and the AVX2 version is
#   picked at runtime (see morpho_cpu_simd.hpp), so one wheel is portable and fast.
# - -Wno-psabi silences a note about passing AVX vectors by value, which never happens across a call:
#   every function taking vectors is always inlined.
# - -g0 overrides the -g of the Python build flags: debug info made the library 7x larger.
# - No -std: torch's BuildExtension adds the standard its headers need (C++17, C++20 from torch 2.14)
#   to the host compiler and nvcc, and a fixed one would break the other releases.
if sys.platform.startswith('linux'):
    cxx_flags = ['-O3', '-DNDEBUG', '-fopenmp', '-Wno-unused-variable', '-Wno-psabi', '-g0']
    link_args = ['-fopenmp']
elif sys.platform.startswith('darwin'):
    cxx_flags = ['-O3', '-DNDEBUG', '-Wno-psabi']
    link_args = ['-lomp']
elif sys.platform.startswith('win'):
    cxx_flags = ['/O2', '/DNDEBUG', '/openmp']
    link_args = []
else:
    cxx_flags = ['-O2']
    link_args = []

torch_version_full = os.environ.get("TORCH_VERSION", torch.__version__.split('+')[0])
torch_version_parts = torch_version_full.split('.')
torch_major = torch_version_parts[0]
torch_minor = torch_version_parts[1] if len(torch_version_parts) > 1 else "0"


version = f"{BASE_VERSION}.{torch_major}{torch_minor}"

torch_next_minor = int(torch_minor) + 1
torch_requirement = f"torch>={torch_major}.{torch_minor}.0,<{torch_major}.{torch_next_minor}.0"

if with_cuda:
    Extension = CUDAExtension
    sources = cpp_sources + cuda_sources
    define_macros = [('WITH_CUDA', '1')]
    
    # Turing and newer: Pascal and Volta are no longer built (CUDA 13 dropped them too), which also
    # shortens every wheel build. Setting TORCH_CUDA_ARCH_LIST (e.g. "8.6") replaces this list and
    # lets torch derive the flags.
    if os.environ.get('TORCH_CUDA_ARCH_LIST'):
        arch_flags = []
    else:
        arch_flags = [
            '-gencode=arch=compute_75,code=sm_75',     # Turing: RTX 20XX
            '-gencode=arch=compute_80,code=sm_80',     # Ampere: A100
            '-gencode=arch=compute_86,code=sm_86',     # Ampere: RTX 30XX
            '-gencode=arch=compute_89,code=sm_89',     # Ada: RTX 40XX
            '-gencode=arch=compute_90,code=sm_90',     # Hopper: H100
            '-gencode=arch=compute_89,code=compute_89', # PTX for RTX 50XX+ forward compat
        ]

    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': [
            '-O3',
            '--use_fast_math',
        ] + arch_flags + [
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
        torch_requirement,
        "numpy>=1.19.0",
    ]
    
)