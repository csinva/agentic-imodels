# evolve_optimal_tree — autoresearch on exact optimal sparse decision trees

Autonomous AI research on an *exact* solver for optimal sparse decision trees.

The idea: give an AI agent a working solver and a fixed benchmark and let it experiment
autonomously. Each run gets its own folder under `runs/<tag>/` (no branches, no commits); there
the agent modifies `optimal_tree.py`, runs the suite, checks that every certified tree is still
optimal and whether more pairs were certified faster, keeps or discards, and repeats.

The objective solved is the GOSDT one (Lin et al., ICML 2020):

    minimise   misclassification rate + λ × (number of leaves)

over all decision trees on the binarized features. The starting solver is `pygosdt_v1`, a
pure-Python re-implementation that matches or beats the reference C++ code on every benchmark
pair and is 13× faster on geometric mean (`baselines/REPORT.html`).

## How it works

The folder has four files that matter:

- **`run_baselines.py`** — seeds `results/overall_results.csv` with the fixed baselines
  (`gosdt`, the reference C++ binary; `pygosdt_v1`; and `streed`, the STreeD dynamic-programming
  solver), from the cached 600 s benchmark by default. **Not modified by the agent.**
- **`setup_run.py`** — creates an isolated run folder `runs/<tag>/` with a local copy of
  `optimal_tree.py`, a copy of the baseline leaderboard, a symlink to `src/`, and an empty
  snapshot folder. The agent works only inside that folder.
- **`optimal_tree.py`** — the single file the agent edits (its copy in the run folder). Defines
  `OptimalTreeClassifier` (scikit-learn compatible; initially pygosdt_v1 flattened into one
  file) and an evaluation loop that runs the suite and updates the run's
  `results/overall_results.csv`. **This file is edited and iterated on by the agent.**
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
to `baselines/gosdt/` (its `experiments/datasets/` is what the suite reads).

```bash
# 1. Install dependencies
uv sync

# 2. (Optional, only to re-run baselines) build the reference C++ binary and STreeD
brew install tbb boost gmp cmake
baselines/gosdt_patches/apply.sh baselines/gosdt
uv sync --group baselines        # builds baselines/pystreed (pybind11, C++17)

# 3. Seed the baseline leaderboard (instant, from the cached benchmark; --rerun to recompute)
uv run run_baselines.py

# 4. Create a run folder and run a single experiment inside it
uv run setup_run.py sep15-run1
cd runs/sep15-run1
uv run optimal_tree.py             # full suite (~10 min), recorded in results/
uv run optimal_tree.py --datasets iris,tic-tac-toe --lams 0.02   # quick, not recorded

# 5. Tests: exhaustive-search exactness checks for pygosdt_v1 and the flat file
uv run pytest
```

## Running the agent

Spin up Claude Code (or any LLM agent) in this folder (`evolve_optimal_tree/`) and prompt:

```
Read and follow the instructions in `program.md`.
```

## Project structure

```
run_baselines.py     — seeds the baseline leaderboard (do not modify)
setup_run.py         — creates runs/<tag>/ for one autoresearch session
optimal_tree.py      — solver definition + evaluation loop (agent modifies its copy in runs/<tag>/)
program.md           — agent instructions
src/                 — fixed suite (suite.py), scoring (evaluate.py), known optima,
                       baseline wrappers (reference_solver.py, streed_solver.py)
results/             — baseline overall_results.csv (leaderboard) and pair_results.csv
runs/<tag>/          — one folder per session: optimal_tree.py, results/, optimal_tree_lib/ snapshots
pygosdt_v1/          — the v1 package the loop starts from (importable: pygosdt_v1.GOSDTClassifier)
tests/               — exactness tests (exhaustive DP on random problems, pinned real pairs)
baselines/           — gosdt/ (the reference implementation, untracked), gosdt_patches/, pystreed/ (STreeD,
                       git metadata removed), benchmarks/ (full 600 s benchmark of every baseline through the
                       same scorer, results, report builder)
baselines/REPORT.html — the comparison report (fit, speed, why, and what the reference gets wrong)
```

To redo the full benchmark (hours; the reference hits its cap on many pairs):
`uv run baselines/benchmarks/run_benchmark.py [--resume]`, then
`uv run baselines/benchmarks/summarize.py` and `uv run baselines/benchmarks/build_report.py`.

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
two-line bug that makes it certify suboptimal trees), is in `baselines/REPORT.html`.

## Design choices

- **Single file to modify, in an isolated folder.** The agent only touches its run's
  `optimal_tree.py`; no branches or commits, and every attempt is snapshotted in the run folder.
- **Exactness is a gate, not a metric.** `n_wrong` must stay 0; speed and coverage are what improve.
- **Fixed suite and cap.** Same 70 pairs, same 30 s cap, objectives recomputed independently, so
  every row of the leaderboard is comparable.
- **The hard pairs are visible.** `results/pair_results.csv` shows which pairs time out; they are
  the wide numeric datasets, which is where the headroom is.
