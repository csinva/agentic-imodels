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
| `benchmarks/` | held-out evaluation of the shipped solver and the baselines |

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
