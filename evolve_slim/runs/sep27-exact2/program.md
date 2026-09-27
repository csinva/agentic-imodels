# autoresearch, exact track: provably optimal sparse integer risk scores

This is an experiment to have a coding agent autonomously research a solver for *sparse
integer linear models*, the scoring systems used in medicine and criminal justice: pick at
most `k` of the (mostly binary) features and give each an integer number of points in
{-5, ..., 5}; the total score is turned into a risk. The goal is a solver that finds points
with a lower training loss, faster, and that stays general.

**This track** asks for more: a *certificate*. After `fit`, set `lower_bound_` to a PROVEN lower
bound on the minimum criterion over every feasible point vector. When that bound meets the loss of
the returned points (within 1e-7, and within the time limit), the problem counts as certified: the
points are provably optimal. No open-source method certifies this problem today: RiskSLIM needs
CPLEX (whose free edition refuses most of these problems) and certifies a different objective, one
without the free multiplier `a`. The starting solver `exact_v1` = the shipped heuristic (v35) for the
incumbent plus a simple certificate: enumerate every support of size k (only for k <= 4 and at most
300,000 supports), prune a support when its saturated loss (the best any function of those columns'
cells can do) is already above the incumbent, and enumerate every integer vector on the rest. It
certifies all 60 tiny problems and a handful of visible ones. The goal is to certify far more.

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
2. **Create the run folder**: `uv run setup_run.py <tag> --track exact` (from `evolve_slim/`). This creates
   `runs/<tag>/` containing:
   - `src/`: **symlink** back to the fixed suite, scoring, baselines and best known losses
     (do not modify)
   - `program.md`: copy of these instructions
   - `slim.py`: **local copy**, the file you edit
   - `results/`: the leaderboard, pre-seeded with the baseline rows
   - `slim_lib/`: empty, for snapshots
3. **`cd` into the run folder** and stay there for the rest of the session: `cd runs/<tag>`.
   Do not read any of the other folders in `runs/`, only your own.
4. **Read the in-scope files**: `slim.py` (the certification code is at its end), `src/exact.py`
   (how certificates are judged), `src/evaluate.py`, `src/suite.py`, `src/best_known.csv`.
   `results/` starts empty: your first run records the starting solver.

No git branch, no commits: every change is local to the run folder.

## Experimentation

Run an experiment from inside the run folder with: `uv run slim.py`

This fits `make_model(k, time_limit)` on every (dataset, k) problem of the development suite
(14 datasets × k ∈ {3, 4, 5, 7, 10} = 70 problems, 60 s limit per problem, 14 problems in
parallel, each single-threaded; about 30 s for the starting solver), re-scores every returned
set of points independently, and updates `results/overall_results.csv`.

For quick checks while developing use a subset (not recorded):
`uv run slim.py --datasets heart,compas,mammo --ks 3,5 --skip-tiny`

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
- Read anything under `data/` other than through `src/suite.py` for the visible suite. Never
  read `data/hidden*` or any `results/hidden*` file: those are the held-out benchmarks.
- Install new packages. Only what is already in `pyproject.toml` (numpy, scipy, pandas,
  scikit-learn, numba).
- Use more than one core (no multiprocessing or threads), or keep state between fits (no
  caches on disk or across `fit` calls). Every fit must start from scratch.
- Special-case datasets, feature names, sizes of the suite, or values of k. The solver will be
  run on held-out datasets of other sizes, so every rule must be general.
- Claim a bound you cannot prove. Every rule that raises `lower_bound_` or prunes part of the search
  must be valid for EVERY point vector it rules out; state the argument in a comment next to the
  code. Numerical slack: when a bound comes from an iterative optimisation, use a quantity that is a
  lower bound at any iterate (a dual value, or a gap bound), not the primal value, or keep a margin
  that you can justify. A bound that merely passes the suite is not allowed.
- Change the objective, the suite, the time limit, or how time is measured (the evaluation code
  in the `__main__` block must stay as is).

## Goal

Optimize the metrics in `results/overall_results.csv`:

- **`n_wrong`**: a `lower_bound_` above a loss that some point vector is known to reach (the returned
  one, the problem's best known loss, or the tiny optimum), a certified tiny answer that is not the
  optimum, invalid points, or a crash. **Must be 0.** Any run with `n_wrong > 0` is a discard.
- **`n_certified`**: visible problems proven optimal within the limit (higher is better, max 70).
  **This is the main metric.** `n_certified_tiny` must stay at 60.
- **`mean_regret`**: mean over the 70 problems of the criterion minus the best known criterion
  of that problem (`src/best_known.csv`, the best any solver has reached, including long runs).
  Lower is better; it can go below 0 if you beat the best known.
- **`geo_mean_time`**: geometric mean of the fit seconds over the 70 problems. Lower is better.
  Secondary: a change that certifies the same problems faster is a keep.
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
model:            exact_v1
n_wrong:          0  (bound above a reached loss, false certificate, invalid or crash; must be 0)
n_certified:      12/70 visible problems proven optimal
n_certified_tiny: 60/60 tiny problems proven optimal
geo_mean_time:    2.345s
mean_regret:      0.00010
mean_test_auc:    0.8424
total_seconds: 300.0s
```

and updates `results/overall_results.csv`, with the columns

```
commit,n_certified,n_certified_tiny,n_wrong,geo_mean_time,mean_regret,mean_test_auc,status,model_name,description
```

`status` is `keep`, `discard` or `crash`; `model_name` and `description` come from `slim.py`.
`results/problem_results.csv` receives, per problem, the loss, `lower_bound`, the known loss or
optimum, `certified` and the verdict.

## The experiment loop

You are always inside the run folder (`runs/<tag>/`).

LOOP FOREVER:

1. Edit `slim.py` with one experimental idea. Update `MODEL_NAME` (unique within this run)
   and `DESCRIPTION` to reflect it.
2. Run the experiment: `uv run slim.py > run.log 2>&1`
3. Read results: `tail -n 12 run.log` and `grep <MODEL_NAME> results/overall_results.csv`
4. If the run crashed, check `tail -n 50 run.log` for the stack trace and attempt a fix.
5. Update the row in `results/overall_results.csv` with the appropriate status: `keep` if
   `n_wrong == 0`, `n_certified_tiny == 60`, and, compared with the best kept row so far, either
   `n_certified` went up, or it stayed the same while `geo_mean_time` went down by at least 10%
   (and `mean_regret` did not rise by more than 0.0002); otherwise `discard`, or `crash`.
6. Save a snapshot of `slim.py` as `slim_lib/<MODEL_NAME>.py`.
7. If the run was a `discard`, restore `slim.py` from the best kept snapshot before trying the
   next idea.

Times vary by a few percent between runs on a shared machine; if a keep decision rests on a
time change close to 10%, rerun once before deciding.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human if you should continue.
Run until manually stopped. Definitely do not stop after less than 30 iterations.

**Ideas to try** (not exhaustive; be creative). The hard part is the search over supports:
C(d, k) grows too fast to enumerate beyond small k, so you need bounds that prune *families* of
supports, and exact but fast solves on the supports that survive.

- Stronger per-support bounds: the saturated loss H(S) is exact counting but loose; the continuous
  logistic optimum on S (any real coefficients, intercept) is also a valid bound, and tighter. Make it
  rigorous with the Fenchel dual of logistic regression (any feasible dual point gives a lower bound;
  H(S) is the dual with the feature constraints dropped).
- Bounds for families (a partial support T plus up to m more features from a candidate list): the
  point box (|w_j| <= 5 with a free scale a) is a cone, so plain relaxations are unbounded; look for
  valid bounds that use the cell structure of binary data, the data's distinct rows, or the incumbent.
- Screening: features that cannot belong to any support that beats the incumbent.
- Per-support integer search by branch and bound instead of enumerating 11^k vectors: fixing some
  points (up to the scale) and relaxing the rest to reals keeps the problem convex in (a, v_rest, b),
  so it gives a valid bound for the whole branch. This is what makes k > 4 reachable on a support.
- Symmetries: w and -w (and w and 2w) give the same criterion; enumerate primitive vectors only.
- Search order: supports ranked by a cheap bound, so the incumbent is confirmed early.
- Speed: numba kernels, bit-packed binary columns, incremental cell counts along a DFS over supports.

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
