#!/bin/sh
# Direct build of the reference GOSDT CLI on Linux with g++ (build.sh is the macOS/clang
# version). Dependencies come from the prefix built by build_deps_linux.sh.
# Usage: ./build_linux.sh [prefix] [output name]   (defaults: ../linux_deps, gosdt)
set -e
cd "$(dirname "$0")"
mkdir -p build
SRCS=$(ls src/*.cpp | grep -v python_extension)
PREFIX=${1:-"$(pwd)/../linux_deps"}
OUT=${2:-gosdt}
LIBDIR="$PREFIX/lib"
[ -f "$LIBDIR/libtbb.so" ] || LIBDIR="$PREFIX/lib64"
g++ -O3 -std=c++17 -DNDEBUG \
  -Wno-deprecated-declarations -Wno-unused-result -Wno-unknown-pragmas \
  -I include -I "$PREFIX/include" $SRCS \
  -L"$LIBDIR" -Wl,-rpath,"$LIBDIR" -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" \
  -ltbb -ltbbmalloc -lgmp -pthread -o "build/$OUT"
echo "built build/$OUT"
