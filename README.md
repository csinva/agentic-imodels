<h1 align="center">Agentic-imodels</h1>
<p align="center">Evolving agentic interpretability tools via autoresearch</p>

<p align="center">
<a href="#quick-start">Quick start</a> •
<a href="#repo-layout">Repo layout</a> •
<a href="#discovered-models">Discovered models</a> •
<a href="#how-the-loop-works">How the loop works</a> •
<a href="#paper">Paper</a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-3.10%2B-blue">
<img src="https://img.shields.io/badge/license-MIT-green">
<img src="https://img.shields.io/badge/uv-managed-orange">
<img src="https://img.shields.io/badge/sklearn-compatible-yellow">
</p>

> Plugin skill:
> To use the developed models in your own data-science projects, just add a pointer to the skill file at <https://github.com/csinva/agentic-imodels/tree/main/result_libs_processed/agentic-imodels> in your CLAUDE.md / AGENTS.md.

We built a library of 10 `scikit-learn`-compatible regressors whose string representations are explicitly optimized to be read by another LLM — interpretable *by agents*, not just by humans.
We did this by using coding agents with a fixed evaluation harness and a single Python file. that is optimized for:

- **Predictive performance** - root-mean squared error (RMSE) across many datasets
- **Agent Interpretability** — fraction of LLM-graded tests passed

## Quickstart

**Requirements:** Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/csinva/imodels-evolve
cd imodels-evolve
uv sync
```

### Use the curated discovered models

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from agentic_imodels import HingeEBMRegressor

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

model = HingeEBMRegressor()
model.fit(X_tr, y_tr)

print(model)               # human/LLM-readable equation
preds = model.predict(X_te)
```

Every estimator follows the standard `BaseEstimator + RegressorMixin` contract, so it drops into `Pipeline`, `cross_val_score`, `GridSearchCV`, etc.

### Run the agentic loop yourself

```bash
cd evolve
uv run run_baselines.py            # establish baseline scores
uv run interpretable_regressor.py  # one experiment iteration
```

Then point a coding agent at [`evolve/program.md`](evolve/program.md):

```
Read and follow the instructions in evolve/program.md.
```

The agent edits `evolve/interpretable_regressor.py` in a loop, commits each attempt, and logs to `evolve/results/overall_results.csv`. See [`evolve/readme.md`](evolve/readme.md) for the full protocol. A parallel setup for Codex lives in [`evolve_codex/`](evolve_codex/).

## Repo layout

| Folder | Purpose |
| --- | --- |
| [`evolve/`](evolve/) | The Claude-driven agentic loop — fixed harness (`run_baselines.py`, `src/`), agent-edited model file (`interpretable_regressor.py`), agent prompt (`program.md`). |
| [`evolve_codex/`](evolve_codex/) | Same loop, OpenAI Codex agent. |
| [`result_libs/`](result_libs/) | Raw per-run output: every regressor the agent wrote during each loop, grouped by date / agent / effort. Includes `combined_results.csv` and `pareto_evolved.csv` aggregating all runs. |
| [`result_libs_processed/agentic-imodels/`](result_libs_processed/agentic-imodels/) | Curated, installable Python package of 10 Pareto-frontier models drawn from `result_libs/`. |
| [`generalization_experiments/`](generalization_experiments/) | Re-evaluate evolved models on **new** OpenML regression suites and a **new 157-test** interpretability suite to check generalization. |
| [`e2e_experiments/`](e2e_experiments/) | Downstream end-to-end evaluation: equip Claude Code, Codex, and Copilot CLI with the evolved models and measure their performance on the [BLADE](https://github.com/behavioral-data/blade) benchmark. |
| [`paper-imodels-agentic/`](paper-imodels-agentic/) | NeurIPS 2026 paper source (`main.tex`, `figures/`, `tables/`). |

## Discovered models

Curated highlights from [`agentic-imodels`](result_libs_processed/agentic-imodels/). **Rank** is mean global RMSE rank across 65 dev datasets (lower is better). **Test interp** is fraction passed on the held-out 157-test generalization suite. Reference points: TabPFN baseline rank 94.5 / test interp 0.17; OLS baseline rank 354.5 / test interp 0.69.

| Class | Rank ↓ | Dev interp ↑ | Test interp ↑ | Idea |
| --- | ---: | ---: | ---: | --- |
| `HingeEBMRegressor` | 108.2 | 0.65 | 0.71 | Lasso on hinge basis + hidden EBM on residuals; sparse linear display. |
| `DistilledTreeBlendAtlasRegressor` | 139.7 | 1.00 | 0.71 | Ridge student distilled from GBM+RF teachers, shown with a probe-answer "atlas" card. |
| `DualPathSparseSymbolicRegressor` | 163.5 | 0.70 | 0.71 | GBM/RF/Ridge blend for `predict`, sparse symbolic equation for display. |
| `HybridGAM` | 163.8 | 0.72 | 0.68 | SmartAdditiveGAM display + hidden RF residual corrector. |
| `TeacherStudentRuleSplineRegressor` | 204.0 | 0.61 | **0.80** | GBM teacher + sparse symbolic student over hinge/step/interaction terms. |
| `SparseSignedBasisPursuitRegressor` | 272.7 | 0.67 | 0.76 | Forward-selected signed basis (linear/hinge/square/interaction) + ridge refit. |
| `HingeGAMRegressor` | 280.2 | 0.56 | 0.78 | Pure Lasso on a 10-breakpoint hinge basis; predict = display. |
| `WinsorizedSparseOLSRegressor` | 326.9 | 0.65 | 0.73 | Clip features to `[p1, p99]`, LassoCV select top-8, OLS refit. |
| `TinyDTDepth2Regressor` | 334.0 | 0.67 | 0.71 | Depth-2 decision tree (4 leaves). |
| `SmartAdditiveRegressor` | 354.3 | 0.74 | 0.73 | Adaptive-linearization GAM — Laplacian-smoothed boosted stumps per feature. |

Two stylistic patterns emerge:

- **Display-predict decoupled** (`HingeEBM`, `HybridGAM`, `DistilledTreeBlendAtlas`, `DualPathSparseSymbolic`, `TeacherStudentRuleSpline`) — a hidden corrector improves prediction while `__str__` stays a clean formula. Pick these for the lowest predictive rank.
- **Honest** (`SmoothGAM`, `HingeGAM`, `WinsorizedSparseOLS`, `SparseSignedBasisPursuit`, `TinyDT`) — `predict` and `__str__` agree, no silent corrector. Pick these when the printed formula must actually be what runs.

## How the loop works

```
        +-----------------------------+
        | program.md (agent prompt)   |
        +-----------------------------+
                       |
                       v
   +----------------------------------------+
   | edit interpretable_regressor.py        |  <-- only file the agent touches
   +----------------------------------------+
                       |
                       v
   +----------------------------------------+
   | run_baselines.py / src/performance_eval|  predictive performance (rank)
   | src/interp_eval.py                     |  43 LLM-graded interp tests
   +----------------------------------------+
                       |
                       v
   +----------------------------------------+
   | results/overall_results.csv            |  keep / discard / crash
   +----------------------------------------+
                       |
                       └──> next iteration
```

Each iteration is a single git commit. Both metrics matter — neither is a hard constraint. The agent is asked to find Pareto improvements over a strong baseline panel (OLS, Lasso, RidgeCV, EBM, RandomForest, GBM, TabPFN, …). See [`evolve/program.md`](evolve/program.md) for the exact protocol the agent follows.

## Generalization & end-to-end results

- **Generalization** ([`generalization_experiments/`](generalization_experiments/)): the evolved models retain their Pareto advantage on **new** OpenML regression suites and on a **new 157-test** interpretability suite written from scratch (separate from the 43 dev tests).
- **End-to-end ADS** ([`e2e_experiments/`](e2e_experiments/)): plugging the evolved models into Claude Code, Codex, and Copilot CLI improves their scores on the BLADE end-to-end data-science benchmark by **8%–47%** vs. standard interpretability tools.

## Related

- [`imodels`](https://github.com/csinva/imodels) — the human-designed sibling library this project extends.
- [BLADE](https://github.com/behavioral-data/blade) — end-to-end ADS benchmark used in `e2e_experiments/`.
- [TabArena](https://github.com/autogluon/tabrepo) and [PMLB](https://github.com/EpistasisLab/pmlb) — regression dataset sources.

## License

MIT.
