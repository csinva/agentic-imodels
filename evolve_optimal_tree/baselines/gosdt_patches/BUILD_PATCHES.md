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

## Optional correctness patch (not applied by `apply.sh`)

`scope-lowerbound.patch` fixes the reference's false optimality certificates.
In `Optimizer::store_children` and `Optimizer::load_children` the lower bound
of a vertex is the minimum over its splits, but splits whose bound exceeds the
vertex's current *scope* (the budget handed down by the parent) are skipped.
The resulting lower bound is only valid for that scope, yet it is cached and
never lowered (`Task::update` takes the maximum), so when a parent later widens
the scope the vertex keeps a lower bound that is too high and the search
certifies a suboptimal tree.  On `tic-tac-toe` at λ = 0.02 the unpatched
binary reports 0.324593 with a zero optimality gap; with these two lines
patched it reports 0.318330, the true optimum that pygosdt finds.

The benchmark in `benchmarks/results` was run with the *unpatched* algorithm
(only the two build fixes above), i.e. the reference as published.  To build
the corrected variant:

```
cd baselines/gosdt && patch -p1 < ../gosdt_patches/scope-lowerbound.patch && ./build.sh
```
