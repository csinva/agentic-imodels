#!/bin/sh
# Direct build of the reference GOSDT CLI (bypasses autotools, which hard-codes -msse4.1).
set -e
cd "$(dirname "$0")"
mkdir -p build
SRCS=$(ls src/*.cpp | grep -v python_extension)
PREFIX=${HOMEBREW_PREFIX:-/opt/homebrew}
clang++ -O3 -std=c++17 -stdlib=libc++ -DNDEBUG \
  -Wno-deprecated-declarations -Wno-unused-result -Wno-unknown-pragmas \
  -I include -I "$PREFIX/include" $SRCS \
  -L"$PREFIX/lib" -ltbb -ltbbmalloc -lgmp -pthread -o build/gosdt
echo "built build/gosdt"
