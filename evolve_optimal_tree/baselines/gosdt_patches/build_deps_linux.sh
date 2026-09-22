#!/bin/sh
# Build the C++ dependencies of the baselines (oneTBB, GMP, Boost headers) into a local
# prefix, for a Linux machine without root. macOS uses Homebrew instead (see apply.sh).
#
#   baselines/gosdt_patches/build_deps_linux.sh [prefix]     (default baselines/linux_deps)
#
# Afterwards point the builds at the prefix:
#   export CMAKE_PREFIX_PATH=$PREFIX PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig
#   export LD_LIBRARY_PATH=$PREFIX/lib:$PREFIX/lib64
# Requires cmake and ninja on PATH (`uv tool install cmake ninja`), a C++17 compiler, and curl.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=${1:-"$HERE/../linux_deps"}
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
SRC="$PREFIX/src"
mkdir -p "$SRC"
JOBS=${JOBS:-16}

TBB_VER=2022.2.0
GMP_VER=6.3.0
BOOST_VER=1_86_0

if [ ! -f "$PREFIX/lib/libtbb.so" ] && [ ! -f "$PREFIX/lib64/libtbb.so" ]; then
  echo "== oneTBB $TBB_VER"
  cd "$SRC"
  [ -d "oneTBB-$TBB_VER" ] || curl -sL "https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v$TBB_VER.tar.gz" | tar xz
  cmake -S "oneTBB-$TBB_VER" -B "oneTBB-$TBB_VER/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTBB_TEST=OFF -DTBB_STRICT=OFF -DCMAKE_INSTALL_PREFIX="$PREFIX" > /dev/null
  cmake --build "oneTBB-$TBB_VER/build" -j "$JOBS" > /dev/null
  cmake --install "oneTBB-$TBB_VER/build" > /dev/null
fi

if [ ! -f "$PREFIX/lib/libgmp.so" ]; then
  echo "== GMP $GMP_VER"
  cd "$SRC"
  [ -d "gmp-$GMP_VER" ] || curl -sL "https://ftp.gnu.org/gnu/gmp/gmp-$GMP_VER.tar.xz" | tar xJ
  cd "gmp-$GMP_VER"
  ./configure --prefix="$PREFIX" --enable-cxx > /dev/null
  make -j "$JOBS" > /dev/null
  make install > /dev/null
fi

if [ ! -d "$PREFIX/include/boost" ]; then
  echo "== Boost $BOOST_VER (headers only)"
  cd "$SRC"
  V=$(echo "$BOOST_VER" | tr _ .)
  [ -d "boost_$BOOST_VER" ] || curl -sL "https://archives.boost.io/release/$V/source/boost_$BOOST_VER.tar.gz" | tar xz "boost_$BOOST_VER/boost"
  cp -r "boost_$BOOST_VER/boost" "$PREFIX/include/"
fi

echo "deps installed under $PREFIX"
ls "$PREFIX/lib" "$PREFIX/lib64" 2>/dev/null | grep -E "libtbb|libgmp" | sort -u
