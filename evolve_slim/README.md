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

| solver | visible (70): loss | AUC | time | hidden (135): loss | AUC | time | full size, k=5 (27): loss | AUC | time |
|---|---|---|---|---|---|---|---|---|---|
| FastRiskScore (v35_scratch) | 0.3559 | 0.842 | 0.038 s | 0.3270 | 0.794 | 0.058 s | 0.3279 | 0.793 | 0.079 s |
| FasterRisk | 0.3603 | 0.841 | 1.49 s | 0.3272 | 0.793 | 3.27 s | 0.3282 | 0.790 | 4.82 s |
| FasterRisk, wide search | 0.3582 | 0.841 | 18.2 s | 0.3271 | 0.793 | 47.3 s |  |  |  |
| RiskSLIM (CPLEX CE) | 0.4524 | 0.718 | 13.6 s | 0.4046 | 0.568 | 9.98 s | 0.4031 | 0.560 | 16.2 s |
| cutting planes, HiGHS | 0.4097 | 0.776 | 57 s | 0.3739 | 0.667 | 60 s | 0.3602 | 0.695 | 600 s |
| SLIM (HiGHS) | 0.4371 | 0.767 | 49.3 s | 0.3895 | 0.642 | 60.2 s | 0.3835 | 0.647 | 542 s |
| abess + seq. rounding | 0.3731 | 0.842 | 0.207 s | 0.3303 | 0.789 | 0.512 s | 0.3317 | 0.788 | 0.790 s |
| fastSparse + seq. rounding | 0.3738 | 0.836 | 0.123 s | 0.3303 | 0.791 | 0.207 s | 0.3304 | 0.790 | 0.304 s |
| OKRidge + seq. rounding | 0.3654 | 0.841 | 20.4 s | 0.3286 | 0.790 | 26.5 s | 0.3297 | 0.786 | 214 s |
| OKGLM + seq. rounding | 0.3609 | 0.842 | 25.3 s | 0.3293 | 0.792 | 31.4 s | 0.3296 | 0.793 | 175 s |
| L0Learn + seq. rounding | 0.3633 | 0.843 | 0.880 s | 0.3290 | 0.792 | 2.84 s | 0.3293 | 0.790 | 4.21 s |
| skscope + seq. rounding | 0.3751 | 0.837 | 0.222 s | 0.3318 | 0.791 | 0.720 s | 0.3325 | 0.792 | 1.12 s |
| riskscores (annealscore) | 0.4876 | 0.701 | 38.3 s | 0.4097 | 0.585 | 70.1 s | 0.3821 | 0.665 | 283 s |
| riskscores (riskcd) | 0.4476 | 0.757 | 12.7 s | 0.3817 | 0.666 | 36.6 s | 0.3713 | 0.699 | 111 s |
| L1 path + seq. rounding | 0.4087 | 0.813 | 0.099 s | 0.3546 | 0.744 | 0.231 s | 0.3581 | 0.744 | 0.370 s |
| probabilistic scoring list | 0.4185 | 0.787 | 13.8 s | 0.3623 | 0.681 | 51.2 s | 0.3366 | 0.763 | 127 s |
| AutoScore | 0.4000 | 0.821 | 0.260 s | 0.3461 | 0.763 | 0.495 s | 0.3469 | 0.768 | 0.869 s |
| rounded L1 logistic | 0.4096 | 0.812 | 0.068 s | 0.3547 | 0.743 | 0.180 s | 0.3580 | 0.744 | 0.292 s |
| unit weighting | 0.4311 | 0.804 | 0.062 s | 0.3665 | 0.732 | 0.180 s | 0.3665 | 0.739 | 0.293 s |
| imodels SLIMClassifier | 0.4260 | 0.789 | 0.258 s | 0.3583 | 0.736 | 0.810 s | 0.3612 | 0.732 | 1.42 s |
| real-valued k-sparse (not integer) | 0.3574 | 0.844 | 0.867 s | 0.3265 | 0.795 | 1.56 s | 0.3275 | 0.794 | 2.27 s |

Mean training log loss (the criterion), mean test AUC, geometric-mean fit time; one core per problem.
Per problem against FasterRisk: lower loss on 54 / 90 / 21 problems and higher on 1 / 5 / 0 (visible
70, hidden 135, full size 27); median speed-up 43x / 54x / 50x. RiskSLIM returns no score on 25 / 102 /
22 problems at the CPLEX Community Edition's size limit. The probabilistic scoring list is
killed at 3x the time limit (no score) on 10 / 51 / 3 problems. None of the eight baselines added after
the first release (cutting planes with HiGHS, abess, fastSparse, OKRidge, the L1 path with rounding, PSL,
AutoScore, unit weighting) comes within 0.001 of FasterRisk's mean loss on the held-out sets; the closest
is OKRidge's optimal ridge support with FasterRisk's rounding (`uv run benchmarks/baseline_table.py`).
The five baselines added on 2026-10-03 (`benchmarks/run_new_baselines.sh`) do not change this: the closest
are L0Learn's CDPSI path and OKGLM's k-sparse logistic branch and bound, each followed by FasterRisk's rounding
(0.3290 and 0.3293 on hidden, against FasterRisk's 0.3272). riskscores' own integer search is far behind
(0.3817 with riskcd, 0.4097 with annealscore, which returns no score on 20 hidden problems within 3x the
limit); its fits often end past the limit (annealscore 23 / 66 / 5 problems, riskcd 12 / 45 / 2), because
the lambda0 path checks the clock only between fits.
On the 6 problems where the cutting planes certify RiskSLIM's optimum (breastcancer at every k, mammo
at k = 3), FastRiskScore's points have a lower calibrated loss than the certified ones on all 6:
RiskSLIM's objective fixes the score's scale, the reason FasterRisk added a multiplier. On 50 problems small enough to enumerate every
score (`benchmarks/exhaustive_check.py`), v35 finds the optimum on 49 and FasterRisk on 28. All numbers
in the post come from `benchmarks/post_numbers.py`; its figure from imodels'
`docs/pages/fastriskscore_pareto.py`.

The loss gain on the held-out sets is small (0.00025 on average, a third of the gap to the real-valued
model): most of the visible-set gain came from datasets with real-valued columns, where FasterRisk's
rounding loses a feature, and every held-out dataset is binarized.

## Baselines

All in `src/baselines.py`; run with `uv run run_baselines.py --models <names> [--suite hidden]`.
Methods that fit a real-valued model are turned into points by the same published rounding
stage, FasterRisk's star-ray search with sequential rounding (after scaling the solution into
the point box), keeping the rounding with the lowest calibrated loss.

| name | method | source / install |
| --- | --- | --- |
| `fasterrisk` | FasterRisk (Liu et al. 2022): beam search, diverse pool, star-ray sequential rounding | vendored `baselines/fasterrisk` (0.1.10, BSD-3; `tostring` fixed for numpy 2) |
| `fasterrisk_wide` | FasterRisk with every search width raised; slow reference for the best known losses | same |
| `riskslim` | RiskSLIM lattice cutting planes (Ustun & Rudin 2019) | `baselines/setup_riskslim.sh`; CPLEX Community Edition (size-limited) |
| `cpa_highs` | RiskSLIM's problem by its cutting-plane algorithm, re-solving a HiGHS MILP each round, warm-started from rounded L1-path solutions; certifies small problems | scipy |
| `slim_milp` | SLIM (Ustun & Rudin 2016): 0-1 loss MILP | scipy HiGHS |
| `abess_seqround` | abess best-subset logistic regression (Zhu et al. 2022), sizes 1..k, rounded | `abess` (GPL-3), PyPI |
| `fastsparse_seqround` | fastSparse / L0Learn L0L2 logistic path (Liu et al. 2022), boxed to [-5, 5], rounded | `fastsparsegams` (MIT), PyPI |
| `okridge_seqround` | OKRidge (Liu et al. 2023) optimal k-sparse ridge support (squared-loss proxy), logistic refit, rounded | vendored `baselines/okridge` (0.1.1, BSD-3; a removed-method call fixed) |
| `l1path_seqround` | every L1 logistic path support of size <= k, refit, rounded (FasterRisk's rounding without its beam search) | scikit-learn |
| `okglm_seqround` | OKGLM (Liu, Shafiee & Lodi 2025) branch and bound for k-sparse logistic regression with box [-5, 5] (CPU, half the limit; constant column for the intercept), its coefficients and a refit on its support rounded | vendored `baselines/okglm` (commit 461e78a, BSD-3; commercial-solver imports made optional) |
| `l0learn_seqround` | L0Learn logistic L0L2 path with CDPSI swaps (5 gammas, support <= k, unbounded), rounded | R package L0Learn 2.1.0 (MIT) via `baselines/rpkgs` |
| `skscope_seqround` | skscope ScopeSolver k-sparse logistic regression (sizes 1..k, numpy objective and gradient), rounded | `skscope` 0.1.8 (MIT), PyPI |
| `riskscores` | riskscores `risk_mod` (annealscore, the default), points in [-5, 5]; L0 penalty lambda0 walked along `cv_risk_mod`'s grid and bisected to <= k points, best of those | R package riskscores 1.3.0 (GPL-3) via `baselines/rpkgs` |
| `riskscores_cd` | the same with riskscores' `riskcd` coordinate descent | same |
| `psl` | probabilistic scoring lists (Hanselle et al. 2025), scores +-{1..5}, k greedy stages | vendored `baselines/skpsl` (scikit-psl 0.7.2, MIT; numpy 2 fix, `max_stages` added) |
| `autoscore` | AutoScore (Xie et al. 2020) on binary features: random-forest ranking, logistic regression, coefficients / smallest, rounded | reimplemented (AutoScore is R-only) |
| `rounded_lr` | L1 logistic regression tuned to k features, refit, scaled so the largest point is 5, rounded | scikit-learn |
| `unit_weighting` | the L1-selected features, each worth +-1 | scikit-learn |
| `imodels_slim` | imodels 3.0.2 `SLIMClassifier` without a MIP solver (rounded L2 logistic), penalty searched to k | reimplemented as imodels runs it |
| `continuous_beam` | reference, not integer: FasterRisk's k-sparse beam search before rounding | vendored FasterRisk |

Considered and left out (see the search notes in `LITERATURE.md`): scorepyo (abandoned, hard pins
on 2022 packages), optbinning and scorecardpy (weight-of-evidence scorecards, not sparse integer
points), the PyPI l0learn (no Python 3.12 wheel; its R package is used instead), GroupFasterRisk
(identical to FasterRisk without feature groups), and RiskSLIM written directly as a SCIP MINLP
(no incumbent within 60 s on a 500 x 30 test). The R baselines need `baselines/rpkgs/setup_r.sh` (R 4.4 via
micromamba, then CRAN); `src/baselines.py` starts one `Rscript` worker per harness process
(`baselines/rpkgs/server.R`), so R's start-up falls in the untimed warm-up, and passes data through temporary
files under `$EVOLVE_SLIM_SCRATCH`.

## Layout

| path | what it is |
| --- | --- |
| `program.md` | instructions for the agent (edited by the human) |
| `slim.py` | the solver the agent edits (its copy in a run folder), with the fixed evaluation loop at the bottom |
| `setup_run.py` | creates `runs/<tag>/` with a local `slim.py`, a symlink to `src/` and the baseline leaderboard |
| `src/suite.py` | the development suite: 14 datasets × k ∈ {3, 4, 5, 7, 10}, 60 s per problem |
| `src/evaluate.py` | parallel runner (single-threaded workers, hard kill at 180 s), independent re-scoring, leaderboard |
| `src/baselines.py` | the 19 baselines of the table above and a real-valued reference |
| `src/best_known.csv` | lowest criterion any integer solver has reached per problem (`baselines/update_best_known.py`) |
| `data/build_data.py` | builds `data/visible` (14 datasets), `data/hidden` (27 TabArena datasets) and `data/hidden_full` |
| `baselines/fasterrisk/` | FasterRisk 0.1.10, vendored (BSD 3-Clause), with a one-line numpy 2 fix |
| `baselines/okglm/` | OKGLM (src/okglm at commit 461e78a), vendored (BSD 3-Clause), patched to import without gurobipy / mosek / cvxpy |
| `baselines/rpkgs/` | the R-package baselines: `setup_r.sh` (R environment), `server.R` (persistent worker for riskscores and L0Learn) |
| `baselines/setup_riskslim.sh` | fetches RiskSLIM into `baselines/risk-slim/` (untracked) and compiles its loss functions |
| `LITERATURE.md` | the related work |
| `PROMPTS.md` | the prompts that drove the search |
| `runs/<tag>/` | one folder per loop session: the agent's leaderboard and a snapshot of every attempt |
| `benchmarks/` | held-out evaluation (`run_hidden.sh`, `run_heldout_rest.sh`, `run_more_baselines.sh`, `run_new_baselines.sh`, `eval_solver.py`), the exhaustive check, `post_numbers.py` |
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
