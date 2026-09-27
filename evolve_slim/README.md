# evolve_slim: autoresearch on sparse integer linear models

Autonomous AI research on a solver for *sparse integer linear models*, the scoring systems
(risk scores) of medicine and criminal justice: at most `k` features, each worth an integer
number of points in {-5, ..., 5}, with the total score mapped to a risk.

The idea is the same as in [`evolve_optimal_tree`](../evolve_optimal_tree/): give an AI agent a
working solver and a fixed benchmark and let it experiment. Each run gets its own folder under
`runs/<tag>/`; there the agent edits `slim.py`, runs the suite, keeps or discards, and repeats.

The problem solved is the one of RiskSLIM (Ustun & Rudin, JMLR 2019) and FasterRisk (Liu et
al., NeurIPS 2022), with a real-valued intercept:

    minimise over integer w, |w_j| <= 5, ||w||_0 <= k:
        min over real a, b of (1/n) Σ_i log(1 + exp(-(2 y_i - 1)(a · x_i·w + b)))

The harness computes the inner minimum itself, so a solver is judged only by its points. The
starting solver `pyfasterrisk_v1` is FasterRisk flattened into one file; it reproduces the
published package's losses exactly.

## Results

Two loop sessions ran in parallel for about 90 minutes each (`runs/sep26-run1`, asked to lower the
loss; `runs/sep26-run2`, asked to go faster; prompts in [PROMPTS.md](PROMPTS.md)). Both found the same
two main changes: FasterRisk's continuous stage rebuilt with numba on compressed rows and Newton
fits, and a local search over the integer points scored by the harness's own calibrated loss. The
shipped solver is run 2's `v35_scratch` (the same points as its best kept `v34_gemm`, plus
deadlines); it became `FastRiskScoreClassifier` in imodels.

| | loss (visible) | AUC | time | loss (hidden, 27 TabArena) | AUC | time | loss (full size, k=5) | AUC | time |
|---|---|---|---|---|---|---|---|---|---|
| **v35_scratch (shipped)** | **0.3559** | **0.842** | **0.038 s** | **0.3270** | **0.794** | **0.057 s** | **0.3279** | **0.793** | **0.079 s** |
| FasterRisk | 0.3603 | 0.841 | 1.51 s | 0.3272 | 0.793 | 3.28 s | 0.3282 | 0.790 | 4.82 s |
| FasterRisk, wide search | 0.3582 | 0.841 | 18.2 s | 0.3271 | 0.793 | 47.3 s | | | |
| rounded L1 logistic | 0.4096 | 0.812 | 0.068 s | 0.3547 | 0.743 | 0.18 s | 0.3581 | 0.744 | 0.3 s |
| imodels SLIMClassifier | 0.4260 | 0.789 | 0.26 s | 0.3583 | 0.736 | 0.81 s | 0.3612 | 0.732 | 1.4 s |
| SLIM (HiGHS) | 0.4371 | 0.767 | 49 s | 0.3895 | 0.642 | 60 s | 0.3835 | 0.647 | 542 s |
| RiskSLIM (CPLEX CE) | 0.4524 | 0.718 | 14 s | 0.4046 | 0.568 | 10 s | 0.4031 | 0.560 | 16 s |
| real-valued k-sparse (not integer) | 0.3574 | 0.844 | 0.87 s | 0.3265 | 0.795 | 1.56 s | 0.3275 | 0.794 | 2.3 s |

Mean training log loss (the criterion), mean test AUC, geometric-mean fit time; one core per problem.
Per problem against FasterRisk: lower loss on 54 / 90 / 21 problems and higher on 1 / 5 / 0 (visible
70, hidden 135, full size 27); median speed-up 43x / 54x / 50x. RiskSLIM returns no score on 25 / 102 /
22 problems at the CPLEX Community Edition's size limit. On 50 problems small enough to enumerate every
score (`benchmarks/exhaustive_check.py`), v35 finds the optimum on 49 and FasterRisk on 28. All numbers
in the post come from `benchmarks/post_numbers.py`; its figure from imodels'
`docs/pages/fastriskscore_pareto.py`.

The loss gain on the held-out sets is small (0.00025 on average, a third of the gap to the real-valued
model): most of the visible-set gain came from datasets with real-valued columns, where FasterRisk's
rounding loses a feature, and every held-out dataset is binarized.

## Layout

| path | what it is |
| --- | --- |
| `program.md` | instructions for the agent (edited by the human) |
| `slim.py` | the solver the agent edits (its copy in a run folder), with the fixed evaluation loop at the bottom |
| `setup_run.py` | creates `runs/<tag>/` with a local `slim.py`, a symlink to `src/` and the baseline leaderboard |
| `src/suite.py` | the development suite: 14 datasets × k ∈ {3, 4, 5, 7, 10}, 60 s per problem |
| `src/evaluate.py` | parallel runner (single-threaded workers, hard kill at 180 s), independent re-scoring, leaderboard |
| `src/baselines.py` | FasterRisk, RiskSLIM, SLIM, rounded logistic regression, imodels' SLIMClassifier, a real-valued reference |
| `src/best_known.csv` | lowest criterion any integer solver has reached per problem (`baselines/update_best_known.py`) |
| `data/build_data.py` | builds `data/visible` (14 datasets), `data/hidden` (27 TabArena datasets) and `data/hidden_full` |
| `baselines/fasterrisk/` | FasterRisk 0.1.10, vendored (BSD 3-Clause), with a one-line numpy 2 fix |
| `baselines/setup_riskslim.sh` | fetches RiskSLIM into `baselines/risk-slim/` (untracked) and compiles its loss functions |
| `LITERATURE.md` | the related work |
| `PROMPTS.md` | the prompts that drove the search |
| `runs/<tag>/` | one folder per loop session: the agent's leaderboard and a snapshot of every attempt |
| `benchmarks/` | held-out evaluation (`run_hidden.sh`, `run_heldout_rest.sh`, `eval_solver.py`), the exhaustive check, `post_numbers.py` |
| `results/` | leaderboards and per-problem rows: visible (`*.csv`), hidden (`hidden_*`), full size (`hidden_full_t600_*`), wide FasterRisk reference (`t1200_*`) |
| `backfill_regret.py` | rewrites every leaderboard's regret after `src/best_known.csv` is refreshed |

## Metrics

- **`n_invalid`**: points not integer, outside [-5, 5], more than k nonzero, or a crash. Must be 0.
- **`mean_regret`**: mean over problems of the criterion minus the best known criterion (lower is better).
- **`geo_mean_time`**: geometric mean of fit seconds (lower is better).
- **`mean_test_auc`**: test AUC of the score on each problem's 20% held-out split (reported, not optimized).
- **`n_killed`**: no model (still running at 3× the limit, or the solver stopped without one, e.g.
  RiskSLIM at the CPLEX Community Edition's size limit); scored as the base rate.

## Quick start

```bash
uv sync --all-groups
baselines/setup_riskslim.sh                 # only to re-run the RiskSLIM baseline
uv run run_baselines.py                     # baseline rows (visible suite)
uv run setup_run.py sep26-run1 && cd runs/sep26-run1
uv run slim.py                              # one experiment iteration
```

Then point a coding agent at `program.md`.
