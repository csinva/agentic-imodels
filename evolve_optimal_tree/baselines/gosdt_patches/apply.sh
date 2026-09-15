#!/bin/sh
# Apply the build-compatibility patches to the reference checkout and build it.
# Usage: gosdt_patches/apply.sh [path/to/gosdt]
# Requires: brew install tbb boost gmp
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REF=${1:-"$HERE/../gosdt"}
cd "$REF"
if grep -q "patched" src/queue.hpp; then
  echo "patches already applied"
else
  patch -p1 < "$HERE/arm64-onetbb.patch"
fi
cp "$HERE/build.sh" build.sh
cp "$HERE/BUILD_PATCHES.md" BUILD_PATCHES.md
chmod +x build.sh
./build.sh
