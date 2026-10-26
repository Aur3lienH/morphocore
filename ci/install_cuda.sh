#!/bin/bash
# Installs a CUDA compiler in a manylinux_2_28 image, with the headers torch's extension headers need.
# Usage: ci/install_cuda.sh <CUDA version, e.g. 12.8>
#
# Used by ci/docker/builder.Dockerfile, whose image builds the Linux wheels.
set -euo pipefail

V=${1/./-}
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y cuda-nvcc-$V cuda-cudart-devel-$V cuda-cccl-$V cuda-nvrtc-devel-$V cuda-nvtx-$V cuda-profiler-api-$V \
    libcublas-devel-$V libcusparse-devel-$V libcusolver-devel-$V libcurand-devel-$V
dnf clean all
