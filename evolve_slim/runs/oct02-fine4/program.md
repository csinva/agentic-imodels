# autoresearch, fine track: risk scores over thousands of threshold features

This is an experiment to have a coding agent autonomously research a solver for *sparse
integer linear models*, the scoring systems used in medicine and criminal justice: pick at
most `k` of the (mostly binary) features and give each an integer number of points in
{-5, ..., 5}; the total score is turned into a risk. The goal is a solver that finds points
with a lower training loss, faster, and that stays general.

**This track** uses the `visible_fine` suite: the same 14 datasets, taken from their raw sources,
with every numeric column split at its 99 percentiles (indicators `x <= t`) instead of its 9 deciles,
so a problem has 70 to 2,365 features, most of them nested thresholds of the same column. Finer
thresholds already lower the loss far more than any algorithmic change did on the decile suite; the
aim here is a solver that exploits this structure: a much lower loss than FasterRisk on the same
features, at a small fraction of its time. The starting solver is `fine_v1`, the shipped FastRiskScore
solver (v35) unchanged; it already beats FasterRisk (mean loss 0.3412 against 0.3428, 0.068 s against
2.2 s), so the bar is v35 itself.

## The problem

For a training matrix `X` (n × d) and labels `y` in {0, 1}, the solver returns integer
points `w` with at most `k` nonzero entries, each in [-5, 5]. The harness scores `w` itself:

    criterion(w) = min over real a, b of  (1/n) Σ_i log(1 + exp(-(2 y_i - 1) (a · x_i·w + b)))

i.e. the training log loss of the score `s = X w` after the best logistic map from score to
risk (`a` is FasterRisk's 1/multiplier, `b` its intercept over the multiplier). Anything else
the model reports (intercept, multiplier) is ignored. This is the RiskSLIM / FasterRisk
problem with a real-valued intercept.

## Setup

The work happens inside a fresh **run folder**, an isolated copy of the editable file under
`evolve_slim/runs/<tag>/`.

1. **Pick a unique run tag** based on today's date and a counter, e.g. `sep26-run1`.
2. **Create the run folder**: `uv run setup_run.py <tag> --track fine` (from `evolve_slim/`). This creates
   `runs/<tag>/` containing:
   - `src/`: **symlink** back to the fixed suite, scoring, baselines and best known losses
     (do not modify)
   - `program.md`: copy of these instructions
   - `slim.py`: **local copy**, the file you edit
   - `results/`: the leaderboard, pre-seeded with the baseline rows
   - `slim_lib/`: empty, for snapshots
3. **`cd` into the run folder** and stay there for the rest of the session: `cd runs/<tag>`.
   Do not read any of the other folders in `runs/`, only your own.
4. **Read the in-scope files**: `slim.py`, `src/suite.py`, `src/evaluate.py`,
   `src/baselines.py`, `results/overall_results.csv` (the baseline rows) and
   `results/problem_results.csv` (per-problem losses, times and points of the baselines).

No git branch, no commits: every change is local to the run folder.

## Experimentation

Run an experiment from inside the run folder with: `uv run slim.py`

This fits `make_model(k, time_limit)` on every (dataset, k) problem of the development suite
(the visible_fine suite: 14 datasets × k ∈ {3, 4, 5, 7, 10} = 70 problems, 60 s limit per problem,
14 problems in parallel, each single-threaded; under a minute for the starting solver), re-scores every returned
set of points independently, and updates `results/overall_results.csv`.

For quick checks while developing use a subset (not recorded):
`uv run slim.py --datasets heart,ionosphere,magic --ks 3,10`

**What you CAN do:**

- Edit `slim.py` in the run folder, above the "Evaluation loop" banner. Everything is fair
  game: the search (beam search, local search on the integer lattice, swaps, rounding rules,
  exact search for small k, relaxations and bounds), how the score-to-risk map is handled
  during the search, data compression, numba kernels, memory layout, the default
  hyperparameters, time management.
- Update `MODEL_NAME` (must be unique within this run) and `DESCRIPTION` at the top of the file.
- Save snapshots of `slim.py` under `slim_lib/` after each attempt.

**What you CANNOT do:**

- Modify anything reached through the symlink (`src/`), or anything outside the run folder.
- Read anything under `data/` other than through `src/suite.py` for the visible_fine suite. Never
  read `data/hidden*` or any `results/hidden*` file: those are the held-out benchmarks.
- Use feature names or the file layout: the solver sees only the matrix, and must discover the
  nested structure (a column whose ones are a subset of another's) from the data.
- Install new packages. Only what is already in `pyproject.toml` (numpy, scipy, pandas,
  scikit-learn, numba).
- Use more than one core (no multiprocessing or threads), or keep state between fits (no
  caches on disk or across `fit` calls). Every fit must start from scratch.
- Special-case datasets, feature names, sizes of the suite, or values of k. The solver will be
  run on held-out datasets of other sizes, so every rule must be general.
- Change the objective, the suite, the time limit, or how time is measured (the evaluation code
  in the `__main__` block must stay as is).

## Goal

Optimize the metrics in `results/overall_results.csv`:

- **`n_invalid`**: problems where the points are not integers, fall outside [-5, 5], have
  more than k nonzero entries, or the solver crashed. **Must be 0.** Any run with
  `n_invalid > 0` is a discard, whatever the other metrics say.
- **`mean_regret`**: mean over the 70 problems of the criterion minus the best known criterion
  of that problem (`src/best_known.csv`, the best any solver has reached, including long runs).
  Lower is better; it can go below 0 if you beat the best known.
- **`geo_mean_time`**: geometric mean of the fit seconds over the 70 problems. Lower is better.
  A fit that returns after the 60 s limit is counted at its time and flagged
  (`n_over_limit`); one still running at 180 s is killed and scored as no model.

Both `mean_regret` and `geo_mean_time` matter. `mean_test_auc` (test AUC of the score on each
problem's held-out split) is reported too: it is not the target, but a change that lowers the
training loss while clearly lowering test AUC is a warning sign worth noting in the description.

**The first run**: your very first run should always be to establish the baseline: run the
script as is and record the results.

## Output format

Once the script finishes it prints a summary like this:

```
---
model:          pyfasterrisk_v1
mean_regret:    0.00412  (training log loss minus the best known, mean over 70 problems; lower is better)
geo_mean_time:  1.494s  (geometric mean of fit seconds; lower is better)
mean_test_auc:  0.8406  (test AUC of the score, mean over problems)
mean_loss:      0.33333
n_invalid:      0  (...)
n_killed:       0  (...)
n_over_limit:   0  (...)
total_seconds: 30.0s
```

It also updates `results/overall_results.csv`, which has the columns

```
commit,mean_regret,geo_mean_time,mean_test_auc,mean_loss,n_invalid,n_killed,n_over_limit,integer,status,model_name,description
```

`status` is `keep`, `discard` or `crash` (baseline rows have `baseline`); `model_name` and
`description` come from `slim.py`. `results/problem_results.csv` receives the per-problem
losses, times, AUCs and points.

## The experiment loop

You are always inside the run folder (`runs/<tag>/`).

LOOP FOREVER:

1. Edit `slim.py` with one experimental idea. Update `MODEL_NAME` (unique within this run)
   and `DESCRIPTION` to reflect it.
2. Run the experiment: `uv run slim.py > run.log 2>&1`
3. Read results: `tail -n 12 run.log` and `grep <MODEL_NAME> results/overall_results.csv`
4. If the run crashed, check `tail -n 50 run.log` for the stack trace and attempt a fix.
5. Update the row in `results/overall_results.csv` with the appropriate status: `keep` if
   `n_invalid == 0` and, compared with the best kept row so far, either `mean_regret` went down
   by at least 0.0002 without `geo_mean_time` rising by more than 10%, or `geo_mean_time` went
   down by at least 10% without `mean_regret` rising by more than 0.0002; otherwise `discard`,
   or `crash`.
6. Save a snapshot of `slim.py` as `slim_lib/<MODEL_NAME>.py`.
7. If the run was a `discard`, restore `slim.py` from the best kept snapshot before trying the
   next idea.

Times vary by a few percent between runs on a shared machine; if a keep decision rests on a
time change close to 10%, rerun once before deciding.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human if you should continue.
Run until manually stopped. Definitely do not stop after less than 30 iterations.

**Ideas to try** (not exhaustive; be creative):

- Threshold structure: columns `x <= t1 <= t2 ...` of one variable are nested. Detect the chains once
  (sort columns by count, check subset relations on the compressed rows), then compute statistics of
  all thresholds of a variable with one cumulative pass (as optimal-tree solvers do for numeric splits).
- Threshold moves in the local search: slide a chosen threshold to a neighbouring one, or split one
  line into two thresholds of the same variable (a step function), scored by the calibrated loss.
- Screening thousands of candidates: the beam and the swap screens cost O(d) per step; restrict them
  to the best thresholds per variable, or to candidates whose one-step bound can beat the incumbent.
- Two stages: solve on a coarse subset of thresholds, then refine thresholds locally around the
  chosen ones.

- Optimize the right objective: the harness re-fits the multiplier and intercept for your
  points, so the search can score integer candidates by their calibrated loss rather than by a
  fixed multiplier.
- Local search on the integer lattice after rounding: ±1 moves, swapping a support feature for
  another, moving points between features, with the calibrated loss as the acceptance rule.
- Better continuous starting points: beam width, how many diverse solutions to round, L0/L2
  paths, warm starts from k-1 to k within one fit.
- Speed: binary data has many duplicate rows (compress to unique rows with counts), the score
  takes few distinct values (sufficient statistics per score value), cache `exp(y X w)`
  updates, numba kernels.
- Search directly in integer space: beam search over integer supports and values, or an exact
  branch and bound for small k.
- Use the time budget: an anytime search that returns the best points found when the time is
  up, but do not waste time on problems that are already solved well.
- Read the FasterRisk paper (https://arxiv.org/abs/2210.05846), RiskSLIM
  (https://jmlr.org/papers/v20/18-615.html), SLIM (https://arxiv.org/abs/1502.04269), the
  rounding methods of Chevaleyre et al. (https://arxiv.org/abs/1310.2745), fastSparse
  (https://arxiv.org/abs/2202.11389) and L0Learn (https://arxiv.org/abs/2001.06471) for
  further ideas. `../../LITERATURE.md` summarises the related work.
- Do not simply turn up the search width of the starting solver: the aim is a better
  algorithm, not the same one run longer.

Keep the solver in one file, dependency-free beyond numpy/scipy/pandas/scikit-learn/numba,
single-threaded and deterministic. BE CREATIVE!
