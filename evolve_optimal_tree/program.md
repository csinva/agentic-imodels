# autoresearch — optimal sparse decision trees

This is an experiment to have a coding agent autonomously research an *exact* solver for
optimal sparse decision trees: given binarized features, find the decision tree that
minimises `misclassification rate + λ × (number of leaves)` (the GOSDT objective), and
certify that it is optimal. The solver must never certify a wrong tree; within that
constraint the goal is to certify optimality on more (dataset, λ) pairs, faster.

## Setup

The work happens inside a fresh **run folder**. Each run is an isolated copy of the
editable file under `evolve_optimal_tree/runs/<tag>/`.

1. **Pick a unique run tag** based on today's date and a counter, e.g. `sep15-run1`.
2. **Create the run folder**: `uv run setup_run.py <tag>` (from `evolve_optimal_tree/`).
   This creates `runs/<tag>/` containing:
   - `src/` — **symlink** back to the fixed suite, scoring and known optima (do not modify)
   - `program.md` — copy of these instructions
   - `optimal_tree.py` — **local copy**, this is the file you edit
   - `results/` — the leaderboard, pre-seeded with the baseline rows (`gosdt`, `pygosdt_v1`)
   - `optimal_tree_lib/` — empty, for snapshots
3. **`cd` into the run folder** and stay there for the rest of the session:
   `cd runs/<tag>`. Do not read any of the other folders in `runs/`, only your own.
4. **Read the in-scope files**: `optimal_tree.py`, `src/suite.py`, `src/evaluate.py`,
   `results/overall_results.csv` (the two baseline rows) and `results/pair_results.csv`
   (which pairs the baselines fail to certify within the cap). `../../REPORT.html`
   explains how the starting solver compares with the reference and why.

No git branch, no commits: every change is local to the run folder.

## Experimentation

Run an experiment from inside the run folder with: `uv run optimal_tree.py`

This fits `OptimalTreeClassifier` on every (dataset, λ) pair of the development suite
(14 datasets × 5 values of λ, 30 s cap per pair, about 8–10 minutes for the starting
solver), recomputes each returned tree's objective independently, checks it against the
best known objective, and updates `results/overall_results.csv`.

For quick checks while developing use a subset (not recorded):
`uv run optimal_tree.py --datasets iris,tic-tac-toe,gaussian_1k --lams 0.02,0.01`

**What you CAN do:**

- Edit `optimal_tree.py` in the run folder, above the "Evaluation loop" banner. Everything
  is fair game: the search strategy (depth-first, best-first, iterative deepening, ...),
  new provable bounds, the bitset engine, the numba kernels, memoisation and memory layout,
  the binarization, feature ordering heuristics, incumbent construction, parallelism within
  the process.
- Update `MODEL_NAME` (must be unique within this run) and `DESCRIPTION` at the top of the file.
- Save snapshots of `optimal_tree.py` under `optimal_tree_lib/` after each attempt.

**What you CANNOT do:**

- Modify anything reached through the symlink (`src/`), or anything outside the run folder.
- Install new packages. Only what is already in `pyproject.toml` (numpy, pandas,
  scikit-learn, numba).
- Weaken exactness. A bound that is not provably valid is not allowed, even if it happens
  to pass the suite. State the argument for any new bound in a comment.
- Change the objective, the suite, the time cap, or how time is measured (the evaluation
  code in the `__main__` block must stay as is).

## Goal

Optimize the metrics in `results/overall_results.csv`:

- **`n_wrong`** — pairs where the solver *certifies* a tree that disagrees with a certified
  known optimum (worse: an unsound bound; better: a false certificate), or crashes.
  An uncertified incumbent returned at the time cap is never counted as wrong. **Must be 0.**
  Any run with `n_wrong > 0` is a discard, whatever the other metrics say.
- **`n_solved`** — pairs certified optimal within the 30 s cap (higher is better, max 70).
- **`geo_mean_time`** — geometric mean of optimisation seconds over the 70 pairs, unsolved
  pairs counted at 30 s (lower is better).

`n_solved` and `geo_mean_time` both matter. The baseline rows show the starting point:
`pygosdt_v1` certifies 56 of 70 pairs; the 14 it does not (status `time` in
`results/pair_results.csv`) are the wide numeric datasets `fico_1k`, `sine_1k`,
`gaussian_1k` and `compas_processed` at small λ, with 600–1,400 binary thresholds. That is
where the headroom is.

**The first run**: your very first run should always be to establish the baseline — run the
script as is, record the results.

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

It also updates `results/overall_results.csv`, which has the following format:

```
commit,n_solved,geo_mean_time,n_wrong,status,model_name,description
```

1. git commit hash of the repository at the time of the run (informational)
2. n_solved — from the script output
3. geo_mean_time — from the script output
4. n_wrong — from the script output; empty for crashes
5. status: `keep`, `discard`, or `crash` (baseline rows have status `baseline`)
6. shorthand unique name of the solver attempt (`MODEL_NAME` in `optimal_tree.py`)
7. brief text description of what this attempt tried (`DESCRIPTION` in `optimal_tree.py`)

Always log to this file after each experiment. `results/pair_results.csv` receives the
per-pair objectives, times and verdicts.

## The experiment loop

You are always inside the run folder (`runs/<tag>/`).

LOOP FOREVER:

1. Edit `optimal_tree.py` with one experimental idea. Update `MODEL_NAME` (unique within
   this run) and `DESCRIPTION` to reflect it.
2. Run the experiment: `uv run optimal_tree.py > run.log 2>&1`
3. Read results: `tail -n 10 run.log` and `grep <MODEL_NAME> results/overall_results.csv`
4. If the run crashed, check `tail -n 50 run.log` for the stack trace and attempt a fix.
5. Update the row in `results/overall_results.csv` with the appropriate status: `keep` if
   `n_wrong == 0` and either `n_solved` went up or `geo_mean_time` went down without the
   other getting worse (compared with the best kept row so far), otherwise `discard`, or
   `crash`.
6. Save a snapshot of `optimal_tree.py` as `optimal_tree_lib/<MODEL_NAME>.py`.
7. If the run was a `discard`, restore `optimal_tree.py` from the best kept snapshot before
   trying the next idea.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human if you should
continue. Run until manually stopped. Definitely do not stop after less than 20 iterations.

**Ideas to try** (not exhaustive — be creative):

- Better incumbents: a smarter greedy/beam construction so the depth-first search prunes
  earlier; warm-start from the tree found at the previous, larger λ.
- Stronger provable lower bounds: the equivalent-points bound per class, subproblem bounds
  from already-solved supersets/subsets (risk is monotone under set inclusion), bounds from
  the similar-support distance across more than the two neighbouring thresholds.
- Exploit numeric structure: for a numeric column, thresholds are nested, so child
  statistics for all thresholds can be computed with one cumulative pass instead of one
  popcount per threshold.
- Reduce re-expansions: cache candidate lists for large subproblems, or use iterative budget
  widening so a subproblem is not re-enumerated for every budget.
- Search order: best-first within a node, breadth on the first level, or a hybrid.
- Cheaper memo keys (hash of the packed capture set) to cut memory on 12,000-row datasets.
- Read the GOSDT paper (https://arxiv.org/abs/2006.08690), the OSDT paper
  (https://arxiv.org/abs/1904.12847) and the MurTree paper (https://arxiv.org/abs/2007.12652)
  for further bounds and search ideas.
- Do not simply reduce the search space heuristically: the solver must remain exact.

Keep the solver in one file, dependency-free beyond numpy/pandas/scikit-learn/numba, and
make sure every bound you add is provably valid. BE CREATIVE!
