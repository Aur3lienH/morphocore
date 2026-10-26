#!/bin/bash
# Cross-compiles the CPU-only Windows x64 wheel of one torch version, on Linux.
#
# Usage: ci/build_windows_wheels.sh <torch version, e.g. 2.13> <output dir>
# Needs clang-cl, lld-link, curl, unzip and a Python with pip on the PATH.
#
# PyPI's torch for Windows is CPU only, so these wheels are too. The MSVC runtime and the Windows SDK
# come from xwin, torch's headers and import libraries from its Windows wheels, and Python's from the
# python NuGet package. The wheel metadata (version, torch requirement) comes from setup.py, as for
# the Linux wheels. The extension only uses Python's stable ABI (see bindings.cpp), so one build gives
# an abi3 wheel for every Python >= 3.11.
set -euo pipefail

TORCH_VERSION=$1
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
SRC=$(cd "$(dirname "$0")/.." && pwd)
CSRC=$SRC/morphocore/functional/csrc
WORK=${WORK_DIR:-$(mktemp -d)}
XWIN_VERSION=0.10.0
# The oldest supported CPython: only its headers and python3.lib, the stable ABI's import library, are used.
PY=311
PYTHON_NUGET=3.11.9

cd "$WORK"
python -m pip install -q pefile wheel setuptools

# MSVC runtime and Windows SDK.
if [ ! -d sdk ]; then
    curl -sSfL https://github.com/Jake-Shadle/xwin/releases/download/$XWIN_VERSION/xwin-$XWIN_VERSION-x86_64-unknown-linux-musl.tar.gz \
        | tar xz --strip-components=1 xwin-$XWIN_VERSION-x86_64-unknown-linux-musl/xwin
    ./xwin --accept-license --arch x86_64 --cache-dir xwin-cache splat --output sdk
    rm -rf xwin-cache
fi

# Wheel metadata: setup.py derives the version and the torch requirement from TORCH_VERSION. It
# imports torch, so any torch release will do; the extension itself is not built here.
if ! python -c "import torch" 2>/dev/null; then
    python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
rm -rf meta && mkdir meta
(cd "$SRC" && MORPHO_WITH_CUDA=0 TORCH_VERSION=$TORCH_VERSION python -c "from setuptools import build_meta; build_meta.prepare_metadata_for_build_wheel('$WORK/meta')")
DIST_INFO=$(basename meta/*.dist-info)
VERSION=$(sed -n 's/^Version: //p' meta/$DIST_INFO/METADATA)

# The torch headers require C++20 from torch 2.14, C++17 before.
CXX_STD=c++17
if [ "$(printf '%s\n' 2.14 "$TORCH_VERSION" | sort -V | head -1)" = 2.14 ]; then
    CXX_STD=c++20
fi

echo "=== torch $TORCH_VERSION, cp$PY-abi3"
PYDIR=py$PY TORCHDIR=torch$PY OBJ=obj$PY
rm -rf $PYDIR $TORCHDIR $OBJ torch-*.whl && mkdir -p $PYDIR $TORCHDIR $OBJ

curl -sSfL -o python.nupkg https://www.nuget.org/api/v2/package/python/$PYTHON_NUGET
unzip -q python.nupkg 'tools/include/*' 'tools/libs/*' -d $PYDIR && rm python.nupkg

python -m pip download -q --no-deps --only-binary=:all: --platform win_amd64 --python-version ${PY:0:1}.${PY:1} \
    "torch==$TORCH_VERSION.*" -d .
unzip -q torch-*.whl 'torch/include/*' 'torch/lib/*.lib' 'torch/lib/libiomp5md.dll' -d $TORCHDIR && rm torch-*.whl
T=$TORCHDIR/torch

# torch runs its thread pool on Intel OpenMP but ships no import library for it.
python - "$T/lib" <<'EOF'
import pefile, sys
lib = sys.argv[1]
names = [e.name.decode() for e in pefile.PE(f"{lib}/libiomp5md.dll").DIRECTORY_ENTRY_EXPORT.symbols if e.name]
open(f"{lib}/libiomp5md.def", "w").write("LIBRARY libiomp5md.dll\nEXPORTS\n" + "\n".join(names) + "\n")
EOF
lld-link /lib /def:$T/lib/libiomp5md.def /out:$T/lib/libiomp5md.lib /machine:x64

# Same sources as setup.py. /openmp is required: at::parallel_for expands to an OpenMP loop here.
CFLAGS=(--target=x86_64-pc-windows-msvc /nologo /std:$CXX_STD /O2 /EHsc /MD /DNDEBUG /openmp -w
    /imsvc sdk/crt/include /imsvc sdk/sdk/include/ucrt /imsvc sdk/sdk/include/um /imsvc sdk/sdk/include/shared
    /imsvc $T/include /imsvc $T/include/torch/csrc/api/include /imsvc $PYDIR/tools/include
    /I $CSRC /I $CSRC/cpp /I $CSRC/cuda
    -DNOMINMAX)
pids=()
for f in morpho_op reshaper smorph bindings smorph_scm; do
    clang-cl "${CFLAGS[@]}" /c $CSRC/cpp/$f.cpp /Fo$OBJ/$f.obj & pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
lld-link /nologo /dll /out:$OBJ/_C.pyd /machine:x64 \
    /libpath:sdk/crt/lib/x86_64 /libpath:sdk/sdk/lib/um/x86_64 /libpath:sdk/sdk/lib/ucrt/x86_64 \
    /libpath:$T/lib /libpath:$PYDIR/tools/libs \
    $OBJ/*.obj c10.lib torch.lib torch_cpu.lib python3.lib libiomp5md.lib

# Wheel: the Python package, the extension and the metadata; wheel pack writes the RECORD.
PKG=pkg/morphocore-$VERSION
rm -rf pkg && mkdir -p $PKG
(cd "$SRC" && find morphocore -name '*.py' -not -path '*/csrc/*' -exec cp --parents {} "$WORK/$PKG" \;)
cp $OBJ/_C.pyd $PKG/morphocore/
cp -r meta/$DIST_INFO $PKG/
printf 'Wheel-Version: 1.0\nGenerator: morphocore ci/build_windows_wheels.sh\nRoot-Is-Purelib: false\nTag: cp%s-abi3-win_amd64\n' $PY \
    > $PKG/$DIST_INFO/WHEEL
python -m wheel pack -d "$OUT" $PKG
rm -rf pkg $PYDIR $TORCHDIR $OBJ
