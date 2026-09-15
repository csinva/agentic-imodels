# pygosdt: Generalized Optimal Sparse Decision Trees without the C++ toolchain

`pygosdt` re-implements the GOSDT algorithm of the reference repository
`GeneralizedOptimalSparseDecisionTreesReference` (Lin et al., ICML 2020) as a
small Python package.  It finds a decision tree that provably minimises

    risk(tree) = misclassification loss + regularization * (number of leaves)

over all trees on the binarized features, exactly like the reference, but needs
only `numpy`, `pandas`, `scikit-learn` and (optionally, for speed) `numba`.
No Boost, GMP, TBB, WiredTiger, OpenCL or compiler is required.

Contents of this directory:

| path | what |
|---|---|
| `pygosdt/` | the package (`encoder.py`, `dataset.py`, `optimizer.py`, `fastbits.py`, `model.py`, `gosdt.py`, `cli.py`) |
| `tests/` | exactness tests against an exhaustive search and regression tests on the reference datasets |
| `REPORT.html` | benchmark comparison report (headline table, fit matrix, speed charts, stop reasons, full table) |
| `benchmarks/` | harness comparing pygosdt with the reference binary, result CSVs, summary table and plot |
| `reference_patches/` | two-line build fix and build script for the reference C++ code on arm64 macOS / oneTBB |
| `GeneralizedOptimalSparseDecisionTreesReference/` | the reference implementation (untracked; copy it here from the original location) |

## Getting the reference implementation working

The reference autotools build hard-codes `-msse4.1`, includes an unused SIMD
header that does not compile on arm64, and uses an allocator type that oneTBB
2021+ rejects.  `reference_patches/apply.sh` applies the two-line source patch
(`arm64-onetbb.patch`, see `BUILD_PATCHES.md`) and compiles the CLI directly with
clang:

```sh
brew install tbb boost gmp
reference_patches/apply.sh GeneralizedOptimalSparseDecisionTreesReference
cat GeneralizedOptimalSparseDecisionTreesReference/experiments/datasets/monk_1/data.csv \
  | GeneralizedOptimalSparseDecisionTreesReference/build/gosdt config.json
```

The reference binary reads the dataset from stdin when stdin is not a terminal
(that is how the benchmark harness invokes it).

## Installing and using pygosdt

```sh
uv sync                     # creates .venv with numpy/pandas/scikit-learn/numba
uv run pytest               # 80 tests, ~10 s
```

scikit-learn style:

```python
import pandas as pd
from pygosdt import GOSDTClassifier

df = pd.read_csv("GeneralizedOptimalSparseDecisionTreesReference/experiments/datasets/iris/data.csv")
X, y = df.iloc[:, :-1], df.iloc[:, -1]
model = GOSDTClassifier(regularization=0.02).fit(X, y)
print(model)                 # readable if/else tree
model.objective_, model.n_leaves_, model.optimal_, model.time_, model.size_
model.predict(X); model.score(X, y)
model.json()                 # same JSON schema as the reference (feature/name/relation/reference/true/false)
```

Reference-wrapper style (`python/model/gosdt.py` in the reference):

```python
from pygosdt import GOSDT
model = GOSDT({"regularization": 0.1, "time_limit": 3600}).fit(X, y)
model.tree, model.time, model.iterations, model.size
```

Command line, mirroring `gosdt dataset.csv config.json`:

```sh
uv run pygosdt data.csv config.json        # prints a JSON array with the model
```

Supported configuration keys: `regularization`, `time_limit`, `balance`,
`costs` (CSV cost matrix, reference format), `upperbound`, `look_ahead`,
`similar_support`, `feature_exchange`, `continuous_feature_exchange`,
`verbose`, `model` (output path), `engine` (`auto`/`numba`/`python`).
Multi-class targets and arbitrary cost matrices are supported.

## How it works

**Binarization** (`encoder.py`) mirrors the reference `Encoder`: numeric
columns become `x >= t` predicates at the midpoints between consecutive
observed values (reported as the upper value for integer columns), categorical
columns become `x == v` predicates, two-valued columns without missing values
become a single predicate, constant columns are dropped.  Additionally, binary
columns that induce the same row partition as an earlier one (identical or
complementary) are removed, which cannot change the optimum.

**Bitsets** (`dataset.py`) hold every binary column, every class indicator and
the *equivalent points* mask as Python integers, so a subproblem (the set of
rows reaching a node, the reference's "capture set") is one `int`, splitting is
`&`, and the memo table is a plain `dict`.

**Search** (`optimizer.py`) is a depth-first branch-and-bound dynamic program
over capture sets, memoised in a dependency graph like the reference, with a
certified interval `[lb, ub]` per subproblem and a budget (the reference's
"scope") passed down from the parent.  All bounds are exact, so the tree
returned is the true optimum of the objective on the binarized data:

* leaf risk, and the equivalent-points lower bound `min_loss + 2λ` for any split
  (`Task::Task` in the reference);
* leaf-support and incremental-accuracy conditions proving a subproblem is best
  left as a leaf;
* one-step look-ahead budgets: a child is only searched with the budget left
  after subtracting its sibling's lower bound (`send_explorers` scopes);
* similar-support bound between neighbouring binary features
  (`Dataset::distance`), computed lazily only when it could prune;
* continuous feature exchange between consecutive thresholds of one numeric
  column (`Task::continuous_feature_exchange`), applied per subproblem where it
  is provably valid, vectorised in numpy;
* a greedy incumbent and an immediate "both children as leaves" upper bound.

The per-node work, counting `|capture & class_k & feature_j|` for every
feature, is vectorised: with `engine="numba"` a small `@njit` kernel runs over
packed 64-bit words; with `engine="python"` the same counts come from
`int.bit_count` on the big integers.  Both produce identical trees.

**Output** (`model.py`) is the reference's JSON tree schema plus prediction,
scoring and structure helpers (`leaves()`, `nodes()`, `maximum_depth()`).

### Why pygosdt is faster

Both implementations do the same work per subproblem (count the class split of
every feature).  pygosdt expands far fewer subproblems, because its depth-first
search starts from a greedy incumbent and only descends into splits that can
beat it, whereas the reference's best-first message queue expands breadth-wise
before a complete tree exists to prune against (median 4.3× more expansions on
the pairs both solved, up to 267× on iris).  Each expansion is also cheaper:
one vectorised counting call and numpy filtering versus bitmask copies in every
message and several concurrent hash-map round trips (median 150 µs vs 28 µs per
expansion on pairs taking the reference over a second).  Memory follows the
same pattern.  `REPORT.html` has the full account.

### What goes wrong in the reference, and differences that affect results

* **False optimality certificates.**  When the reference computes a vertex's
  lower bound it skips splits whose bound exceeds the vertex's current scope
  (the parent's budget), caches the result, and never lowers it when the scope
  widens later.  On `tic-tac-toe` at λ = 0.02 it reports 0.324593 with a zero
  gap while pygosdt finds 0.318330 (190 errors, 6 leaves; verified
  independently, pinned in `tests/test_reference_datasets.py`), and it does so
  with every optional bound disabled.  Removing the two scope-conditional
  skips (`reference_patches/scope-lowerbound.patch`) makes the reference
  report 0.318330 too.  pygosdt records unconditional lower bounds when a
  subproblem fails its budget, so revisits with a larger budget are safe.
* The reference's pairwise `feature_exchange` bound prunes features for whole
  subtrees using bounds computed at the parent, which is not exact.  pygosdt
  only applies the provably valid per-subproblem version.
* The reference sorts integer thresholds as strings (so `"10" < "2"`), which
  breaks the threshold adjacency its continuous-feature-exchange bound relies
  on.  pygosdt sorts numerically.
* Missing numeric values never satisfy a predicate in pygosdt; the reference
  parses them as 0.  The benchmark fills missing values with 0 for both.
* Arithmetic is float64 (the reference uses float32).

### Exactness testing

`tests/test_bruteforce.py` compares the optimizer with an exhaustive
dynamic program written with plain Python sets on 50 random problems (2 and 3
classes, random cost matrices, both engines, with and without optional bounds).
`tests/test_reference_datasets.py` pins the objectives on 14 real
(dataset, λ) pairs, recomputed from the predictions.

## Benchmark

`benchmarks/run_benchmark.py` runs both implementations on the same CSV for
every (dataset, λ) pair, recomputes each returned tree's objective
independently, and records optimisation time (excluding CSV parsing and
binarization for both), graph size and iterations.  `benchmarks/summarize.py`
merges the CSVs into `benchmarks/results/summary.md` and
`benchmarks/results/benchmark.png`.

Reference settings: `worker_limit = 1` (single thread, like pygosdt), default
bounds, `time_limit = 600` s.  Hardware: Apple M5, macOS, clang 21, oneTBB
2023, Python 3.12, numpy 2.5, numba 0.67.

See `benchmarks/results/summary.md` for the full table; the headline results
are summarised at the end of this file.

## Results summary

Report with charts: `REPORT.html` (built by `benchmarks/build_report.py`). Full
table: `benchmarks/results/summary.md` (also `summary.csv`, the raw
`benchmark_final.csv`, and `benchmark.png`).  15 datasets from the reference
repository (77 to 12,381 rows, 13 to 10,001 binary features) × λ ∈ {0.1, 0.05,
0.02, 0.01, 0.005}, single thread each, 600 s time cap and 6 GB memory cap for
both implementations, run sequentially on an otherwise idle machine.

**Fit.** In every one of the 63 pairs where both implementations returned a
tree, pygosdt's objective was equal (57 pairs) or strictly lower (6 pairs);
it was never worse.  Where pygosdt certified optimality, the reference either
found the same objective or a worse one (it returns an incumbent when it hits
its time limit).  The 6 wins:

| dataset | λ | reference objective (status) | pygosdt objective (status) |
|---|---|---|---|
| tic-tac-toe | 0.02 | 0.324593 (claimed optimal) | 0.318330 (optimal) |
| iris | 0.005 | 0.045 (timed out, gap 0.025) | 0.038333 (optimal, 0.6 s) |
| gaussian_1k | 0.02 | 0.317 (timed out, gap 0.257) | 0.185 (optimal, 19 s) |
| gaussian_1k | 0.01 | 0.307 (timed out) | 0.155 (optimal, 84 s) |
| gaussian_1k | 0.005 | 0.302 (timed out) | 0.140 (optimal, 271 s) |
| sine_1k | 0.005 | 0.462 (timed out) | 0.400 (timed out) |

**Speed** (optimisation time only, both sides).  Over the 63 comparable pairs
the geometric mean of `C++ time / pygosdt time` is 13.2 (median 9.0).  pygosdt
was faster on 50 pairs; the 12 pairs where the reference was faster are all
trivial cases where its millisecond-resolution timer reports 0–2 ms and pygosdt
needs 0.1–2.7 ms.  pygosdt's depth-first search with a strong incumbent visits
far fewer subproblems than the reference's best-first message passing
(e.g. iris λ=0.01: 3,812 vs 215,881 graph nodes, 0.14 s vs 151 s), which more
than compensates for the interpreter overhead per node.

**Limits reached.** The reference hit the 600 s cap on 12 pairs (it checks
the clock only every 10,000 iterations, so it overran the cap by up to 500 s)
and the 6 GB memory cap on 12 (all five sine_10k cases in about a minute,
tic-tac-toe λ=0.005 in 23 s, four compas_processed cases, fico_1k λ=0.02,
sine_1k λ=0.05); its per-node bitmask copies are memory hungry.  pygosdt hit
the time cap on 13 pairs (fico_1k, sine_1k and sine_10k at small λ) and the memory cap on 4
(compas_processed at λ ≤ 0.05, about 2.8M memoised subproblems of 12,381-bit
keys each); in all of them it returns the incumbent with `optimal_ = False`
and a certified lower bound.  On those pairs both implementations agree
wherever the reference produced a tree at all.

**Engines.** `engine="numba"` (default) and `engine="python"` give identical
trees; numba is 1.1–2× faster on the wide numeric datasets (gaussian_1k
λ=0.05: 0.41 s vs 0.94 s) and makes little difference on narrow binary ones.
