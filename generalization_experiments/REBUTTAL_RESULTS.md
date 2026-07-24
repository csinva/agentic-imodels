# Rebuttal experiment results (run 2026-07-23)

All experiments run locally; LLM-judge steps performed by Claude itself (no external API).

## 1. Tuned baselines (mpaA W3, TdWu, dD5H Q3)
Pool: 28 PMLB regression datasets (subsampled ≤1000×50, y-standardized), 23 models,
RandomizedSearchCV per dataset (RF/GBM n_iter=20, MLP 12, Ridge 20, EBM 5; cv=2–3).
Metric: mean RMSE rank within the shared pool (lower better).

Default → Tuned mean rank:
  GBM  10.89 → 6.57  (Δ −4.32)   RF 11.82 → 9.18 (Δ −2.64)
  EBM   8.11 → 8.21  (Δ +0.11)   Ridge 15.61 → 14.82   MLP 17.46 → 14.96
Best evolved model: **HingeEBM rank 6.00** (interp 0.71) — still #1, ahead of best tuned baseline GBM_tuned 6.57 (interp 0.535).
Group mean rank: evolved 10.95, tuned_baseline 10.75, default_baseline 14.09.

Story: tuning substantially improves the strong baselines' PREDICTIVE rank (esp. GBM/RF),
but (i) the single best predictor in the pool is still an evolved model, and (ii) tuning
does not change a baseline's interpretability (a tuned GBM/RF is exactly as unreadable as
an untuned one, interp ≈0.54/0.57). On the (interpretability × prediction) plane the evolved
models remain non-dominated: HingeEBM Pareto-dominates GBM_tuned (better rank AND ~0.17 higher
interp). Frontier claim holds.

## 2. Display-vs-predictor faithfulness (mpaA W1/Q2, TdWu Q2)
For each library model, executed the equation printed in __str__ and compared to .predict()
on held-out points across 9 real datasets (y-std, SD=1). R² of display→predict, MAE/SD, %within 0.20·SD.

  model                     intent      R²    MAE/SD  %within
  SparseSignedBasisPursuit  honest     1.000   0.000   1.000
  SmartAdditive             honest     0.989   0.034   0.967
  TinyDT                    honest     0.939   0.156   0.816
  WinsorizedSparseOLS       honest     0.762   0.084   0.910
  HingeGAM                  honest     0.314   0.521   0.283   ← honest-labeled but linearized display diverges
  HybridGAM                 decoupled  0.941   0.144   0.774
  DistilledTreeBlendAtlas   decoupled  0.755   0.301   0.552
  TeacherStudentRuleSpline  decoupled  0.581   0.375   0.477
  HingeEBM                  decoupled −1.189   0.981   0.446
  DualPathSparseSymbolic    decoupled −9.059   1.911   0.339

Story: honest models' displays are (near-)exact reconstructions of .predict(); decoupled
models diverge, gap = hidden-corrector magnitude (HingeEBM ~1 SD, DualPath ~1.9 SD). Confirms
display-faithfulness and display-simulatability are separable and should both be measured.
Nuance worth reporting: HingeGAM (labeled "honest") is only partially faithful because its
printed linearization approximates the hinge basis.

## 3. BLADE judge swap (mpaA W2)
Re-graded all 26 run-1 Claude-agent BLADE outputs (13 datasets × {standard, custom_v2}) with an
independent Claude judge, same rubric; compared to GPT-4o judge1.
  Cross-judge agreement (overall = mean of 3 dims): Pearson r = 0.839, Spearman 0.845.
    per-dim: correctness 0.817, completeness 0.837, clarity 0.816. Mean |Δ| = 1.10 / 10.
  Evolved-library gain persists under the independent judge:
    GPT-4o:  standard 6.51 → custom_v2 8.26  (+27%)
    Claude:  standard 5.51 → custom_v2 7.36  (+33%)
  custom_v2 > standard on 12/13 datasets under the independent judge (crofoot run did not invoke the custom models).

## 4. Metric-confound breakdown (mpaA point c, dD5H Limitation 2)
Categorized all 43 dev interpretability tests by grading target (verified from source):
  Graded against FITTED .predict():  35/43 (81%)  — all point-simulation (17), all complex-fn (10),
     4 sensitivity, both counterfactual, sign-of-effect, decision-region.
  Graded against DATA-GENERATING fn:  7/43 (16%)  — 5 feature-attribution (most-important, ranking,
     irrelevant, sparse-set, dominant-sample) + threshold-identification + nonlinear-threshold.
  Structural (neither):               1/43        — compactness.
Story: 81% of tests are graded against the deployed predictor's own outputs, so a display that
diverges from .predict() fails them (bounds unfaithfulness); only the 7 DGP-graded tests conflate
fit quality with simulatability (dD5H's concern) — report these as a separate sub-score.
