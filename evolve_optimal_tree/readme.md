# evolve_optimal_tree — autoresearch on exact optimal sparse decision trees

Autonomous AI research on an *exact* solver for optimal sparse decision trees.

The idea: give an AI agent a working solver and a fixed benchmark and let it experiment
autonomously. It modifies `optimal_tree.py`, runs the suite, checks that every returned tree is
still optimal and whether more pairs were certified faster, keeps or discards, and repeats.

The objective solved is the GOSDT one (Lin et al., ICML 2020):

    minimise   misclassification rate + λ × (number of leaves)

over all decision trees on the binarized features. The starting solver is `pygosdt_v1`, a
pure-Python re-implementation that matches or beats the reference C++ code on every benchmark
pair and is 13× faster on geometric mean (`REPORT.html`).

## How it works

The folder has three files that matter:

- **`run_baselines.py`** — evaluates the fixed baselines (`gosdt`, the reference C++ binary,
  and `pygosdt_v1`) on the development suite and writes `results/overall_results.csv`.
  **Not modified by the agent.**
- **`optimal_tree.py`** — the single file the agent edits. Defines `OptimalTreeClassifier`
  (scikit-learn compatible; initially pygosdt_v1 flattened into one file) and an evaluation
  loop that runs the same suite and updates `results/overall_results.csv`.
  **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. Point your agent here and let it go.
  **This file is edited and iterated on by the human.**

## Metrics

Three metrics are tracked in `results/overall_results.csv`, computed by `src/evaluate.py` on
the suite in `src/suite.py` (14 datasets × λ ∈ {0.1, 0.05, 0.02, 0.01, 0.005}, 30 s cap per pair):

- **`n_wrong`** — pairs where the solver certifies a tree that disagrees with a certified
  known optimum (`src/known_optima.csv`), or crashes. Must be 0.
- **`n_solved`** — pairs certified optimal within the cap (higher is better, max 70).
- **`geo_mean_time`** — geometric mean of optimisation seconds, unsolved pairs counted at the
  cap (lower is better).

Every returned tree's objective is recomputed independently from the tree and the raw data, so
a solver cannot score by misreporting.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). The reference datasets live in
the reference repository, which is untracked here: copy
[GeneralizedOptimalSparseDecisionTrees](https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees)
to `gosdt/` (its `experiments/datasets/` is what the suite reads).

```bash
# 1. Install dependencies
uv sync

# 2. (Optional) build the reference C++ binary so it appears as a baseline
brew install tbb boost gmp
gosdt_patches/apply.sh gosdt

# 3. Run baseline evaluation (~45 min with the reference binary, ~10 min without)
uv run run_baselines.py            # add --skip-reference to skip the C++ baseline

# 4. Manually run a single experiment
uv run optimal_tree.py             # full suite, recorded
uv run optimal_tree.py --datasets iris,tic-tac-toe --lams 0.02   # quick, not recorded

# 5. Tests: exhaustive-search exactness checks for pygosdt_v1 and the flat file
uv run pytest
```

## Running the agent

Spin up Claude Code (or any LLM agent) in this folder and prompt:

```
Read and follow the instructions in `program.md`.
```

## Project structure

```
run_baselines.py     — baseline evaluation on the fixed suite (do not modify)
optimal_tree.py      — solver definition + evaluation loop (agent modifies this)
program.md           — agent instructions
src/                 — fixed suite (suite.py), scoring (evaluate.py), known optima,
                       reference-binary wrapper (reference_solver.py)
results/             — overall_results.csv (leaderboard), pair_results.csv
optimal_tree_lib/    — success/ and failure/ snapshots of every attempt
pygosdt_v1/          — the v1 package the loop starts from (importable: pygosdt_v1.GOSDTClassifier)
tests/               — exactness tests (exhaustive DP on random problems, pinned real pairs)
benchmarks/          — full 600 s benchmark of pygosdt_v1 vs the reference, results, summary, report builder
REPORT.html          — the comparison report (fit, speed, why, and what the reference gets wrong)
gosdt_patches/       — build fix + build script for the reference on arm64/oneTBB, and the
                       optional two-line correctness patch for its false optimality certificates
gosdt/               — the reference implementation (untracked)
```

## The starting solver (pygosdt_v1)

`pygosdt_v1` re-implements GOSDT with numpy, pandas, scikit-learn and numba only: binary columns,
class indicators and the equivalent-points mask are Python big-integer bitsets, a subproblem is
one `int`, and the search is a memoised depth-first branch-and-bound with the reference's bounds
(equivalent points, leaf support, incremental accuracy, one-step look-ahead, similar support,
threshold exchange) plus a greedy incumbent. The per-node counting runs in a numba kernel over
packed 64-bit words. It supports multi-class targets and cost matrices, and emits the reference's
JSON tree schema.

```python
from pygosdt_v1 import GOSDTClassifier
model = GOSDTClassifier(regularization=0.02).fit(X, y)
model.objective_, model.n_leaves_, model.optimal_, model.time_
```

Why it is faster than the reference, and what the reference gets wrong (including a verified
two-line bug that makes it certify suboptimal trees), is in `REPORT.html`.

## Design choices

- **Single file to modify.** The agent only touches `optimal_tree.py`. Diffs are small and reviewable.
- **Exactness is a gate, not a metric.** `n_wrong` must stay 0; speed and coverage are what improve.
- **Fixed suite and cap.** Same 70 pairs, same 30 s cap, objectives recomputed independently, so
  every row of the leaderboard is comparable.
- **The hard pairs are visible.** `results/pair_results.csv` shows which pairs time out; they are
  the wide numeric datasets, which is where the headroom is.
