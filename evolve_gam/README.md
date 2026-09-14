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

Two later iterations (September 2026) revised the model to **AddGP_v49**, and changed how
results are measured. The single-split mean rank that drove the original search turned out
to sit inside split noise: redrawing the train/test split moves EBM's own RMSE by a median
of 5.7% (37% at the 90th percentile). Every decision from v48 on used a three-seed paired
test with a rule fixed in advance (geometric-mean RMSE ratio below 1, sign-test p < 0.05,
tail not worse), and the suite's exact duplicate datasets were collapsed before testing.

### Development suite (imodels-65), three redrawn splits

| model | vs EBM, seed 0 / 1 / 2 (GM RMSE ratio) | wins of 65 | official sibling-free rank |
|---|---|---|---|
| AddGP_v47 | 0.98 / **1.02** / 0.97 | 34 / 35 / 39 | 4.60 vs EBM 5.00 |
| **AddGP_v48 = v49 at n <= 1000** | **0.97 / 0.95 / 0.96** | 37 / 47 / 41 | **4.45 vs EBM 5.05** |

v48 vs v47, paired over 186 distinct fits: better on 115, GM 0.981, p = 0.0016. At n = 200
subsamples v48 vs EBM is 0.92 / 0.93 / 0.94.

### Held-out sets, de-duplicated (no dataset shared with the development suite)

TabArena without `houses` (California housing) and OpenML-CTR23 without its five
development-suite overlaps and seven TabArena overlaps: 35 datasets, every model refit on
identical preprocessing and the same 80/20 split.

| | wins vs EBM | GM RMSE ratio vs EBM | 90th-pct ratio |
|---|---|---|---|
| AddGP_v47 | 18/35 | 1.016 | 1.27 |
| AddGP_v48 | 20/35 | 1.006 | 1.21 |
| **AddGP_v49** | **22/35** | **0.969** | **1.10** |

Mean / median rank in the eleven-model pool of the blog post (EBM, TabPFN, RF, GBM, FIGS,
RuleFit, hierarchical shrinkage, decision tree, Ridge, MLP): TabArena-12 **v49 2.50 / 2**,
EBM 3.00 / 3; CTR23-23 TabPFN 2.87 / 2, **v49 3.13 / 2**, EBM 3.48 / 3. v49 vs v48 on the
held-out set: better on 17 of 21 non-tied datasets (the rest are small and identical by
construction), GM 0.964, p = 0.007. Largest moves: video_transcoding 0.266 -> 0.168,
naval_propulsion 0.045 -> 0.031, fps_benchmark 0.040 -> 0.028, supercon 0.334 -> 0.287
(the last two now ahead of EBM). Remaining losses: wave_energy (0.036 vs EBM 0.014),
energy_efficiency, fifa, QSAR, sarcos.

### What changed

* **v48** (three changes, each accepted by the three-seed test): one Matern and one RBF
  lengthscale learned by marginal likelihood and shared by all features, with a weak
  log-normal prior; a hierarchical prior shrinking each kernel slot's log-amplitudes toward
  their centre across features; and, below 1000 rows, 96 bins, a 2500-bin budget and up to
  16 pair surfaces at 16x16 with the pair count scaled by n.
* **v49** (one change, above 1000 rows): EBM's default fits five interaction terms per
  feature; v48 was capped at 48 because the chunked joint fit is cubic in cells. v49 keeps
  the joint fit for the first 48 pairs and backfits the rest, up to five per feature and 16
  per thousand rows, each as an exact 2-D GP on the joint model's residual with a shared
  noise level and a per-pair grid resolution chosen by marginal likelihood. Identical to
  v48 at n <= 1000 (verified on all 65 development datasets).

Falsified along the way, each with a measured cost, in `../results/generalization/` and the
prompts file: a literal shared template shape across features, a LOO predictive objective,
correlation-tied amplitudes, empirical-Bayes and Laplace priors, Laplace hyperparameter
averaging, deterministic subagging, more pairs at n <= 1000, a marginal-likelihood pair
screen, per-feature lengthscales, finer bins, removing the Tukey fence, bilinear pair
readout, an exact joint mains+pairs fit at small n, a learned per-feature level component,
pure backfitting for all pairs, and fixed-amplitude linear sweeps.

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
- Several TabArena datasets sit within 0.5% of EBM and can land on either side of
  the line between identical runs (float32 threading alone moves results ±0.5–1%),
  so treat that suite's head-to-head count as noisy.
- The imodels port and the research model agree to four decimals on held-out RMSE,
  but they are not bit-identical: the port derives its gradients analytically in
  numpy rather than using autograd.
