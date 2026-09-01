# evolve_gam — AddGP, an additive Gaussian-process GAM

An autonomous research loop (see [PROMPTS.md](PROMPTS.md)) searching for a GAM that
beats [EBM](https://github.com/interpretml/interpret) while staying interpretable
and simple. What it converged on is a generalized additive model with pairwise
interactions in which every shape function is a Gaussian process over the
quantile bins of its feature.

The model has since been contributed to `imodels`
([csinva/imodels#299](https://github.com/csinva/imodels/pull/299)) as `AddGPRegressor`.

## The idea

Fitting a GAM as a Gaussian process normally needs an `n×n` kernel matrix, which is
hopeless past a few thousand rows. But every component of a GA²M is one-dimensional
(or a 2-D grid), so once each feature is quantile-binned, every component is just a
function on a small grid. The exact marginal likelihood then touches the data
**only** through three quantities:

```
C = Z'Z      # bin co-occurrence counts
b = Z'y      # bin sums
y'y
```

One pass over the data builds them; every optimizer step afterwards uses those
alone. The fit costs `O(P³)` per step in the total bin count `P` — **independent of
the sample size**.

The same likelihood makes every structural choice: per-feature smoothness (from a
two-kernel Matérn + RBF mixture), feature relevance (amplitudes driven to zero, i.e.
ARD), which interactions to include, and how finely to grid them. There is no
cross-validation, no validation split, no bagging and no seed, so two fits on the
same data give the same model.

```python
from addgp import BinGP
model = BinGP().fit(X_train, y_train)
model.predict(X_test)
```

## Results

Four regression suites, 113 datasets, 50 → 72,000 rows, up to 1,024 features. One
80/20 split per dataset (`random_state=42`), RMSE on the train-standardized target,
all models at library defaults. Lower rank is better; **bold** marks the best
interpretable model.

| Suite | AddGP | EBM | RF | GBM | TabPFN | notes |
|---|---|---|---|---|---|---|
| imodels (65 datasets) | **4.60** | 5.00 | 6.66 | 6.38 | 3.40 | training capped at 1k rows |
| Classic-7 (full size) | **2.43** | 3.43 | 3.00 | 4.43 | 1.71 | v47 wins 6/7 head-to-head |
| TabArena-13 | **2.62** | 2.85 | 2.85 | 4.38 | 2.77 | first of six; TabPFN fails 2 |
| OpenML-CTR23 (28) | **2.11** | 2.32 | 3.07 | 3.18 | — | held out; interpretable pool |
| OpenML-CTR23 (27) | **2.85** | 3.04 | 3.81 | 4.00 | 1.96 | with TabPFN, where it runs |

AddGP is the strongest interpretable model on every suite. Only TabPFN — a black-box
foundation model, capped at 2,500 training rows by GPU memory and unable to fit
several of the wider datasets at all — ranks higher overall.

**CTR23 is the one that matters.** It was downloaded *after* the method was final and
informed no design decision, so it measures generalization rather than tuning.

**Where it loses,** consistently across all four suites: smooth deterministic
simulations and heavy-interaction data — wave-energy converters (+109% vs EBM),
building-energy simulation (+79%), robot arm dynamics (+17%), molecular fingerprints.
There the target depends on three or more inputs jointly, and no sum of pairwise
pieces can represent it. That is the model class's ceiling, not a fitting failure.

## What the search removed

The model began at 1,110 lines with a gradient-boosted tree ensemble bolted on.
Thirteen ablation rounds removed anything that could not prove its worth, leaving
686 lines and a single class. Each decision is a measured result:

**Removed** (cost of removal): the boosted tree ensemble (0.31 rank), a second
exact-kernel model class and its dispatcher (0.6 rank at small scale), half the
kernel dictionary (0.09 rank), both MAP priors (none — two datasets improved), the
per-bin z-mean machinery (none), the categorical special case (none), a binary
search over bin resolutions (≤0.3%), per-bin x-means (≤0.5%).

**Kept** (cost of removing it): ARD amplitude fitting (+9 to +24%, flips every
dataset tested — this is the mechanism, not an ornament), the blockwise pair fitting
(+25%, two separate replacement attempts failed), the log-target rule (+8 to +11%),
two alternation sweeps (+5%), the interaction screener (+10.6%), its shrinkage
constant (+3.3%), the outlier fence (+1.6%), the second kernel, the bias correction
(+4.5%, it corrects log-retransformation bias), and early stopping as the regularizer
(running the likelihood to convergence overfits).

## Layout

```
model/addgp.py            the research model (torch for the optimizer)
model/addgp_imodels.py    the dependency-free port sent to imodels (numpy/scipy,
                          analytic gradients verified to 5e-7 vs finite differences)
benchmarks/               evaluation harnesses for all four suites
results/                  per-dataset RMSEs
report/addgp_report.html  interactive write-up of the method and results
PROMPTS.md                the prompts that drove the search
```

## Reproducing

```bash
uv run benchmarks/ctr23_fetch.py     # downloads CTR23 to ~/.cache/imodels-evolve/ctr23
uv run benchmarks/ctr23_eval.py      # AddGP vs EBM/RF/GBM/Ridge, resumable
uv run benchmarks/ctr23_tabpfn.py    # adds TabPFN (needs a GPU; subprocess-isolated)
uv run benchmarks/v47_suites.py      # classic-7 and TabArena on the shipped model
```

## Caveats

- Defaults-versus-defaults on a single split per dataset. TabArena and CTR23 both
  define richer official protocols with repeated folds and hyperparameter search;
  these numbers are not comparable to published leaderboard entries.
- Baseline numbers for the classic-7 and TabArena tables were collected with the
  model as it stood a few simplification rounds earlier. The final rounds traded a
  little accuracy on near-tie datasets for a much smaller model — several TabArena
  datasets sit within 0.5% of EBM and can land on either side of the line between
  runs (float32 threading alone moves results ±0.5–1%).
- The imodels port and the research model agree to four decimals on held-out RMSE,
  but they are not bit-identical: the port derives its gradients analytically in
  numpy rather than using autograd.
