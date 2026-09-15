# autoresearch — optimal sparse decision trees

This is an experiment to have the LLM autonomously research an *exact* solver for
optimal sparse decision trees: given binarized features, find the decision tree that
minimises `misclassification rate + λ × (number of leaves)` (the GOSDT objective),
and certify that it is optimal. The solver must never return a worse tree than the
known optimum; within that constraint the goal is to certify optimality on more
(dataset, λ) pairs, faster.

## Setup

To set up a new experiment, do the following:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `sep15`). The
   branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**. Read these files for full context:
   - `readme.md` — repository context.
   - `run_baselines.py` — the fixed baseline evaluation harness. Not modified by the
     agent; it has already been run for you.
   - `optimal_tree.py` — the file you modify. Solver definition and evaluation loop.
   - `src/suite.py` and `src/evaluate.py` — the fixed suite and scoring. Read-only.
   - `results/overall_results.csv` — current scores for the baselines.
   - `REPORT.html` — how the starting solver compares with the reference and why.

Then kick off the experimentation.

## Experimentation

Run an experiment with: `uv run optimal_tree.py`

This fits `OptimalTreeClassifier` on every (dataset, λ) pair of the development suite
(14 datasets × 5 values of λ, 30 s cap per pair, about 8–10 minutes for the starting
solver), recomputes each returned tree's objective independently, checks it against
the best known objective, and updates `results/overall_results.csv`.

For quick checks while developing use a subset (not recorded):
`uv run optimal_tree.py --datasets iris,tic-tac-toe,gaussian_1k --lams 0.02,0.01`

**What you CAN do:**

- Modify `optimal_tree.py` above the "Evaluation loop" banner — this is the only file
  you edit. Everything is fair game: the search strategy (depth-first, best-first,
  iterative deepening, ...), new provable bounds, the bitset engine, the numba kernels,
  memoisation and memory layout, the binarization, feature ordering heuristics,
  incumbent construction, parallelism within the process.
- Update `MODEL_NAME` (unique per attempt) and `DESCRIPTION` at the top of the file.

**What you CANNOT do:**

- Modify `run_baselines.py`, anything in `src/`, `pygosdt_v1/`, `tests/` or `benchmarks/`.
- Install new packages. Only what is already in `pyproject.toml` (numpy, pandas,
  scikit-learn, numba).
- Weaken exactness. A bound that is not provably valid is not allowed, even if it
  happens to pass the suite. State the argument for any new bound in a comment.
- Change the objective, the suite, the time cap, or how time is measured.

## Goal

Optimize the metrics in `results/overall_results.csv`:

- **`n_wrong`** — pairs where the solver *certifies* a tree that disagrees with a certified
  known optimum (worse: an unsound bound; better: a false certificate), or crashes.
  An uncertified incumbent returned at the time cap is never counted as wrong. **Must be 0.**
  Any run with `n_wrong > 0` is a discard, whatever the other metrics say.
- **`n_solved`** — pairs certified optimal within the 30 s cap (higher is better, max 70).
- **`geo_mean_time`** — geometric mean of optimisation seconds over the 70 pairs,
  unsolved pairs counted at 30 s (lower is better).

`n_solved` and `geo_mean_time` both matter; the baselines in `overall_results.csv`
show where the starting point is. The unsolved pairs (see `results/pair_results.csv`,
status `time`) are the wide numeric datasets: `fico_1k`, `sine_1k`, `gaussian_1k` and
`compas_processed` at small λ, with 600–1,400 binary thresholds. That is where the
headroom is.

**The first run**: Your very first run should always be to establish the baseline —
run the script as is, record the results.

## Output format

Once the script finishes it prints a summary like this:

```
---
model:          pygosdt_v1_flat
n_solved:       56/70 certified optimal within 30s
geo_mean_time:  0.412s
n_wrong:        0  (certified result disagreeing with a certified optimum, or a crash; must be 0)
total_seconds: 540.2s
```

It also updates `results/overall_results.csv` with the row for `MODEL_NAME`, and
`results/pair_results.csv` with the per-pair objectives, times and verdicts.

## Logging results

When an experiment is done, it should log to `results/overall_results.csv`.

The CSV has a header row and 7 columns:

```
commit,n_solved,geo_mean_time,n_wrong,status,model_name,description
```

1. git commit hash (short, 7 chars)
2. n_solved — from the script output
3. geo_mean_time — from the script output
4. n_wrong — from the script output; empty for crashes
5. status: `keep`, `discard`, or `crash` (baseline rows have status `baseline`)
6. shorthand name of the solver tried — always unique (`MODEL_NAME` in `optimal_tree.py`)
7. brief description of what this experiment tried (`DESCRIPTION` in `optimal_tree.py`)

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/sep15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit `optimal_tree.py` with an experimental idea; update `MODEL_NAME` and `DESCRIPTION`
3. git commit
4. Run the experiment: `uv run optimal_tree.py > run.log 2>&1`
5. Read results: `tail -n 8 run.log` and `grep <MODEL_NAME> results/overall_results.csv`
6. If the run crashed, check `tail -n 50 run.log` for the stack trace and attempt a fix
7. Record results in `results/overall_results.csv` (set the status column; do not commit
   this file). Record them even if results did not improve.
8. Save the current version of `optimal_tree.py` as a new file. If `n_wrong == 0` and
   either `n_solved` went up or `geo_mean_time` went down without the other getting
   worse, save it under `optimal_tree_lib/success/optimal_tree_<commit_hash>_<simple_name>.py`.
   Otherwise save it under `optimal_tree_lib/failure/optimal_tree_<commit_hash>_<simple_name>.py`.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you
should continue. Run until manually stopped. Always keep going.

**Ideas to try** (not exhaustive — be creative):

- Better incumbents: a smarter greedy/beam construction so the depth-first search prunes
  earlier; warm-start from the tree found at the previous, larger λ.
- Stronger provable lower bounds: the equivalent-points bound per class, subproblem bounds
  from already-solved supersets/subsets (risk is monotone under set inclusion), bounds
  from the similar-support distance across more than the two neighbouring thresholds.
- Exploit numeric structure: for a numeric column, thresholds are nested, so child
  statistics for all thresholds can be computed with one cumulative pass instead of one
  popcount per threshold.
- Reduce re-expansions: cache candidate lists for large subproblems, or use iterative
  budget widening so a subproblem is not re-enumerated for every budget.
- Search order: best-first within a node, breadth on the first level, or a hybrid.
- Cheaper memo keys (hash of the packed capture set) to cut memory on 12,000-row datasets.
- Read the GOSDT paper (https://arxiv.org/abs/2006.08690), the OSDT paper
  (https://arxiv.org/abs/1904.12847) and the MurTree paper
  (https://arxiv.org/abs/2007.12652) for further bounds and search ideas.
- Do not simply reduce the search space heuristically: the solver must remain exact.

Keep the solver a single class hierarchy in one file, keep it dependency-free beyond
numpy/pandas/scikit-learn/numba, and make sure every bound you add is provably valid.
BE CREATIVE!
