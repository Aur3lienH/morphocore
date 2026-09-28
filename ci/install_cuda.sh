#!/bin/bash
# Installs a CUDA compiler in a manylinux_2_28 image, with the headers torch's extension headers need.
# Usage: ci/install_cuda.sh <CUDA version, e.g. 13.0>
#
# Used as cibuildwheel's before-all for the torch releases the builder image cannot build: their
# headers need a newer GCC than its gcc-toolset-12 (manylinux_2_28 has gcc-toolset-14), and torch
# refuses an nvcc whose major version differs from its own CUDA (torch >= 2.11 uses CUDA 13).
set -euo pipefail

V=${1/./-}
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y cuda-nvcc-$V cuda-cudart-devel-$V cuda-cccl-$V cuda-nvrtc-devel-$V cuda-nvtx-$V cuda-profiler-api-$V \
    libcublas-devel-$V libcusparse-devel-$V libcusolver-devel-$V libcurand-devel-$V
dnf clean all
