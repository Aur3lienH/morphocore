#!/bin/bash
# Builds the Linux x86_64 CUDA wheel of one torch version.
#
# Usage: ci/build_linux_wheels.sh <torch version, e.g. 2.13> <output dir>
# Runs inside a builder image (see ci/docker): the interpreters in /opt/python, auditwheel and a
# CUDA 12 compiler.
#
# The wheel is built against the cu126 build of torch from PyTorch's index, whatever CUDA PyPI's torch
# uses: torch refuses an nvcc of another major version, and CUDA 12 code also runs on the drivers of
# the CUDA 13 torch builds. The extension bundles its own CUDA runtime.
#
# The extension only uses Python's stable ABI (see bindings.cpp), so a single build with the oldest
# supported Python gives an abi3 wheel for every Python >= 3.11. This is what cibuildwheel did,
# without its Docker in Docker: the runner keeps the builder image between jobs, where a fresh Docker
# daemon in every job downloaded it again. The wheel metadata (version, torch requirement) comes from
# setup.py, as for the Windows wheels.
set -euo pipefail

TORCH_VERSION=$1
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
SRC=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK_DIR:-$(mktemp -d)}
PY=${PYTHON:-311}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
export TORCH_VERSION
export MORPHO_WITH_CUDA=${MORPHO_WITH_CUDA:-1}

# torch's libraries come with torch at install time: leave them out of the wheel.
TORCH_LIBS=(libtorch_cpu.so libc10.so libc10_cuda.so libtorch.so libtorch_python.so libtorch_cuda.so)

echo "=== torch $TORCH_VERSION, cp$PY-abi3"
VENV=$WORK/venv
/opt/python/cp$PY-cp$PY/bin/python -m venv --clear "$VENV"
# ninja: without it torch's BuildExtension compiles the sources one after the other.
"$VENV/bin/pip" install -q --upgrade pip setuptools wheel ninja
"$VENV/bin/pip" install -q "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"

# The venv first on the PATH, for BuildExtension to find its ninja.
rm -rf "$WORK/dist"
(cd "$SRC" && PATH=$VENV/bin:$PATH "$VENV/bin/pip" wheel . --no-build-isolation --no-deps -v -w "$WORK/dist")
auditwheel repair -w "$OUT" "${TORCH_LIBS[@]/#/--exclude=}" "$WORK"/dist/*.whl
rm -rf "$VENV" "$WORK/dist"
