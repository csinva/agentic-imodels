# Local build patches (arm64 macOS, oneTBB 2023, Boost 1.92)

The reference sources are unchanged except for two build-compatibility fixes:

1. `src/bitmask.hpp`, `src/index.hpp`: the `#include <simdpp/simd.h>` line is
   commented out. The header is never referenced (`simdpp::` appears nowhere in
   `src/`) and it pulls x86 intrinsics that do not compile on arm64.
2. `src/queue.hpp`: the allocator of `membership_table_type` is declared with
   `std::pair<message_type * const, bool>` instead of
   `std::pair<message_type *, bool>`. oneTBB 2021+ statically asserts that the
   allocator value type equals the map value type.

Dependencies installed with Homebrew: `tbb`, `boost`, `gmp`.

Build with `./build.sh` (writes `build/gosdt`).

Usage: `build/gosdt dataset.csv config.json`
