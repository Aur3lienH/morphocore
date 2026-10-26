#!/bin/bash
# Builds the Linux x86_64 CUDA wheels of one torch version, one per Python.
#
# Usage: ci/build_linux_wheels.sh <torch version, e.g. 2.13> <output dir>
# Runs inside a builder image (see ci/docker): the interpreters in /opt/python, auditwheel and the
# CUDA compiler matching the torch version.
#
# This is what cibuildwheel did, without its Docker in Docker: the runner keeps the builder image
# between jobs, where a fresh Docker daemon in every job downloaded it again. The wheel metadata
# (version, torch requirement) comes from setup.py, as for the Windows wheels.
set -euo pipefail

TORCH_VERSION=$1
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
SRC=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK_DIR:-$(mktemp -d)}
PYTHONS=${PYTHONS:-"311 312 313"}
export TORCH_VERSION
export MORPHO_WITH_CUDA=${MORPHO_WITH_CUDA:-1}

# torch's libraries come with torch at install time: leave them out of the wheel.
TORCH_LIBS=(libtorch_cpu.so libc10.so libc10_cuda.so libtorch.so libtorch_python.so libtorch_cuda.so)

for PY in $PYTHONS; do
    echo "=== torch $TORCH_VERSION, cp$PY"
    VENV=$WORK/venv$PY
    /opt/python/cp$PY-cp$PY/bin/python -m venv --clear "$VENV"
    # ninja: without it torch's BuildExtension compiles the sources one after the other.
    "$VENV/bin/pip" install -q --upgrade pip setuptools wheel ninja
    "$VENV/bin/pip" install -q "torch==$TORCH_VERSION"

    # The venv first on the PATH, for BuildExtension to find its ninja.
    rm -rf "$WORK/dist"
    (cd "$SRC" && PATH=$VENV/bin:$PATH "$VENV/bin/pip" wheel . --no-build-isolation --no-deps -v -w "$WORK/dist")
    auditwheel repair -w "$OUT" "${TORCH_LIBS[@]/#/--exclude=}" "$WORK"/dist/*.whl
    rm -rf "$VENV" "$WORK/dist"
done
