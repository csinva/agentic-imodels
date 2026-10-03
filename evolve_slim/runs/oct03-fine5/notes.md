# oct03-fine5 notes

Start: slim.py = i30_final (oct02-fine4): regret -0.00104, geo time 0.010-0.011 s, test AUC 0.8505, loss 0.33996.
Recorded as `j1_base` (status keep, the reference of this run).

Keep rule: n_invalid 0; (regret down >= 0.0002 with time up <= 10%) or (time down >= 10% with regret up <= 0.0002);
discard if mean test AUC is more than 0.001 below the best kept row. AUC is never a reason to keep.

Tools (copied from oct02-fine4, plus): tools/nbhd.py (exhaustive exact single-move neighbourhood of a solver's final
point), tools/rec.sh (record a variant file as an attempt with a status, restore slim.py), tools/cmp.py now also
reads model rows from results/problem_results.csv.

## Diagnosis: the final points are not exact single-move local optima
tools/nbhd.py on j1: an exhaustive exact search over all value changes, additions and swaps (every free valid column,
every value) at the final point finds an improving single move in 14/70 problems; the mean gain of the best single
move is 0.00017 (mushroom k10 -0.0051, ionosphere k7 -0.0013, australian k7 -0.00125, breastcancer k4/k5/k10
-0.0007/-0.0004/-0.0009 (value changes), mushroom k7 -0.0006, compas k10 -0.00035, spambase k3 -0.0002).
These are problems with a low loss (near separable) or small n, where the move estimate (loss at the current (a, b)
plus one trust-region Newton step) and the swap screen (gradient at the removal state with the old (a, b)) are poor.
Uniform shifts of all points (+-1, toward / away from 0) never help.

Upper bound of exact single-move polishing (tools/expolish.py: repeat {exhaustive exact best single move; ILS} at the
end of each j1 fit until no single move improves): mean loss -0.00021 (11/70 problems change; ionosphere k7 -0.0052,
mushroom k10 -0.0051 carry 2/3 of it). So a cheap exact polish is worth up to ~0.0002.

Why the moves are missed (scratch/diag_m10.py): mushroom k10's best swap passes the second-order screen but its
estimate ranks it below the 11 best of its removal (near-separable data, a = 5, b = 22: one Newton step in (a, b) is
far from the refitted map); ionosphere k7's best swap is ranked first by the estimate, but the screen only lets it
through at width 16 (at the current (a, b)) or 4 (at the refitted (a, b) of the removal state).

## Attempts
| # | MODEL_NAME | idea | mean_regret | geo_mean_time | mean_test_auc | status / why |
|---|---|---|---|---|---|---|
| 1 | j1_base | i30_final rerun | -0.00104 | 0.010 | 0.8505 | keep (reference) |
| 2 | j2_polish | exact polish of the best point: all value changes exact from (bin, x) cells, swaps screened at removal states with a refitted (a, b) (top 8 checked exactly), ILS from an improvement, up to 3 rounds | -0.00114 | 0.012 (serial A/B x1.13) | 0.8508 | discard: regret -0.00010 < 0.0002, time +13%. Split: value part alone -0.00003 at x1.05 (serial), swap part alone -0.00004 |
| 3 | j3_exactval | value changes in the main phase scored exactly (cells) instead of by the estimate; no polish | -0.00094 | 0.013 (x1.36 parallel) | 0.8506 | discard: other paths, worse loss, slower |
| 4 | j4_recalfb | in every ILS run, a failing swap phase is followed by one at refitted removal states (6 checked) | -0.00109 | 0.012 (x1.15) | 0.8510 | discard |
| 5 | j5_xpolish | exact polish: every value change, and per removal the 4 + 4 best swap-ins by the screens at the current and the refitted (a, b), scored by exact calibrated loss on (bin, x) cells (merge of two sorted cell lists; calibrations stop once below the incumbent or when twice the Newton decrement cannot reach it; 2 values per column after a fixed-(a, b) prefilter); ILS from an improvement, up to 3 rounds | -0.00118 | 0.013 (dev x1.19 parallel; serial on 6 sets x1.27 before early stopping) | 0.8508 | discard: regret -0.00014 (catches 9 of the 11 problems of the bound), time +19% |
| (probe) | | j5 with RSWAP 0 (polish instead of refit swaps) | -0.00097 | x1.09 | 0.8509 | refit swaps are not replaced by the polish |
| (probe) | | wider screens in j2's polish (n_screen 16 / 64 with 16 / 40 checked) | same as 4 | | | the estimate, not the screen width, misses mushroom k10 |
| 6 | j6_vardp | variable re-optimisation at the end: per support chain variable, the optimal step function (<= 2-3 thresholds, integer jumps) given the rest of the score, exact DP over (code, cumulative value, jumps) at the current (a, b) | -0.00104 (no point changes) | 0.027 | 0.8505 | discard: the ILS ends are already DP-optimal per variable at fixed (a, b) (slides + value changes cover it); DP cost large where the rest-score has many bins |
| (probe) | | j5 polish, exact_moves cost: on adult the colsum of the removal states' screening rows dominates (22 columns: 1.1 ms); on ionosphere k10 the calibrations of 32 x 11 candidates (6.6 ms at width 32) | | | | |
| 7 | j7_xpol2 | j5 polish cheaper: swap-ins only from the screen at the refitted (a, b) of each removal (4 per removal), 3 values per value change and 2 per swap-in by the loss at fixed (a, b) | -0.00118 | 0.012 (serial x1.12) | 0.8509 | discard: regret -0.00014 |
| (probe) | | width 16 / 32 at the refitted screen: -0.000157 / -0.000169 at x1.31 / x1.33 (row pass per column); one histogram per variable instead: x1.24; 1 value per swap-in: only -0.000095 | | | | |
| 8 | j8_xpol32 | j7 with 32 swap-ins per removal, one (bin, code) histogram per variable, tabulated fixed-(a, b) loss, only the refitted screening rows in the colsum | -0.00121 | 0.012 (dev x1.20) | 0.8510 | discard: regret -0.00017, time +20% |
| (probe) | | where the polish gains: breastcancer k4/5/10, mushroom k7/10, ionosphere k7 have a calibrated logit range a * (max - min score) of 15-70; australian k7 11, spambase k3 8.5, compas k10 4.7; most other problems 3-15 (scratch/feat.txt) | | | | |
| 9 | j9_xpolgate | j8, polish only when the logit range of the best point is >= 10 (14: -0.000143) | -0.00120 | 0.012; serial x1.11 (the gated-in problems x1.4, others x1.0-1.1) | 0.8510 | discard: regret -0.00016, time +11% (both just short of the rule) |
| 10 | j10_polall3 | j9 + polish also of the 2 next best distinct start ends | -0.00120 (same points as j9) | x1.13 vs j9 | 0.8510 | discard: other ends never beat the polished best |
| 11 | j11_rgils | j9 + ILS swap phases at refitted removal states when the logit range >= 10 | -0.00088 | x1.04 | 0.8513 | discard: mushroom k4/5/7 and breastcancer k5 end much worse (+0.002 to +0.010); refitted removal states mislead the in-path search (as in j4) |
| 12 | j12_rsfgate | j9 + refit swaps until 3 fail in gated problems | -0.00116 | x1.07 vs j9 | 0.8510 | discard: mushroom k10 +0.0028 (the extra refit swap leads to a point the polish cannot improve) |
| 13 | j13_spatgate | j9 + start patience 1 in gated problems | -0.00108 | ~same | 0.8516 | discard: the later starts matter in gated problems too (mushroom k5, heart k10 +0.002-0.003) |
| 14 | j14_fastmath | j1 with fastmath (nsz, arcp, contract, afn, reassoc) + error_model numpy on all kernels | -0.00089 | serial x0.984 (error_model alone x0.995, same points) | | discard: no speed (kernels are not vectorisable float loops), 6 problems change by rounding, +0.00016 loss |
| (probe) | | at j9's final points: exhaustive exact pair value changes (two support columns at once, all values) gain 0.00001 in mean (mushroom k10 -0.0007); rescaling round(w f) 0.000006 (tools/pairprobe.py). Exhaustive single moves still gain 0.00005 (ionosphere k7 -0.0031, compas k10, spambase k3) | | | | |
| (probe) | | why ionosphere k7's remaining swap is missed: at a = 22-25 every integer step a v overshoots the quadratic screen, so _qbest is 0 for almost every column and ties are broken by column index; the move's column was "rank 13" only among strictly negative scores | | | | |
| 15 | j15_polnewton | j9 polish with swap-ins screened by the scale-free continuous Newton gain G^2 / H at the refitted removal state (32 per removal) | -0.00125 | 0.011 (geo 0.01146 vs j1 0.01039: x1.103; back-to-back dev x1.055 / x1.12; serial x1.12) | 0.8507 | discard: regret -0.000204, but time at the 10% bar |
| 16 | j16_polglobal | j15 with one global choice of the 48 (removal, column) pairs by refitted removal loss minus Newton gain | -0.00116 | x0.97 vs j15 | 0.8509 | discard: loses mushroom k10 and ionosphere k7 |
| (probe) | | on j15: width 16: ionosphere k7 lost (-0.00119); POLV 1 or REJ 1 or a trust region / 3-8 iteration cap in the polish calibrations: mushroom k10 lost (its swap needs a converged calibration: a ~ 5, b ~ 22) | | | | |
| 17 | j17_polfast | j15 + same-point speed items: partial selection in the Newton screen (x0.989), unique-row gather fused with the nonzero-row lists and one column sum for the means of binary data (x0.993) | -0.00125 | 0.011 (0.01128, x1.086 vs j1; back-to-back x1.09 / x1.11; serial x1.10) | 0.8507 | discard: time still at the bar |
| (probe) | | removal-state calibrations with a trust region: no time change (the setup cost of ionosphere was the Newton screen's argsort over 2319 columns, fixed in j17) | | | | |
| 18 | j18_poliso | j17 + exact rejection in the polish: a candidate whose best monotone (isotonic, PAV) log loss on its cells is not below the incumbent is skipped before its calibration (same points; serial x0.970 vs j17, gated datasets x0.85-0.90) | -0.00125 | 0.011 (x1.044 vs j1 per problem) | 0.8507 | **keep**: regret -0.000204 vs j1 (-0.00125 vs -0.00104), time x1.044 recorded; confirmation: back-to-back dev x0.96 / x1.07, serial A/B x1.073; AUC +0.0002 |
| (probe) | | the same isotonic bound before the exact checks of the ILS (x1.00) and before each calibrated rounding (x1.00): no gain, left off (ISOCHK, ISORND) | | | | |
| 19 | j19_spat1 | j18 with start patience 1 | -0.00109 | serial x0.905 | 0.8512 | discard: +0.00015 loss for -9.5% time (just short of the time bar, and it gives back most of the polish gain) |
| 20 | j20_kickgate | j18 + 3 exact-ranked pair kicks (ILS + polish each) in gated problems | -0.00131 | x1.40 (1 kick: x1.30, no change) | 0.8510 | discard: -0.00006 (mushroom k10 to ~0.0000, heart k10 -0.0012) for +40% time |
| (probe) | | j18's final points are exact single-move local optima in all gated problems; exhaustive single moves still gain only in 4 ungated problems (compas k10, spambase k3, bank k3, adult k10: 0.000008 in mean). Gap of j18 to the per-problem best of all runs of the last three sessions: 0.00010 (heart k10 0.0026 from h5's pair kicks, mushroom k7, mammo k10, ilpd k10, compas k10) | | | | |
| 21 | j21_prepol | j18 + polish also before the refit swaps | -0.00111 | ~x1.1 | 0.8508 | discard: mushroom k7 +0.0098 (the refit swap from the polished point leads elsewhere) |
| 22 | j22_polns48 | j18 with 48 swap-ins per removal | -0.00125 (same points) | slower | 0.8507 | discard |
| 23 | j23_polg8 | j18 with the gate at logit range 8 | -0.00125 (spambase k3 -0.00019 only) | slower | 0.8508 | discard |
| 24 | j24_swnewton | j18 + Newton-gain screen in the ILS swap phases at logit range >= 10 | -0.00125 (same points) | same | 0.8507 | discard: ILS paths change slightly but the polished ends are the same |
| 25 | j25_affkey | j18 with affine-invariant start keys (22 of 185 starts run are affine-equivalent to an earlier one: one-hot complements on bank, mushroom, australian) | -0.00105 | ~same | 0.8508 | discard: mushroom k7/k10 +0.010/+0.003: equivalent points are different integer parametrisations with different neighbourhoods, so the "duplicate" starts are useful |
| 26 | j26_flip | j18 + complement reparametrisations (support column -> its complement with negated points, ILS + polish) in gated problems | -0.00125 (same points) | x1.45 (complement detection by bitsets is slow on ionosphere) | 0.8507 | discard: no reparametrisation leads to a better end |
| 27 | j27_rsoffgate | j18 without refit swaps in gated problems | -0.00106 | ~same | 0.8508 | discard: mushroom k7 +0.0098, breastcancer k5 +0.0023 (refit swaps still find what the polish cannot) |
| 28 | j28_poltgt | j18 with the polish swap search using the best value change as its incumbent | -0.00125 (same points) | x1.00 serial on the gated sets | 0.8507 | discard: no effect |
| 29 | j29_polg12 | j18 with the gate at logit range 12 | -0.00123 | serial x0.988 | 0.8506 | discard: loses australian k7 (logit range 11.3) for -1% time |
| 30 | j30_final | j18 with the probe knobs of j19-j29 present but off, docstring updated (same points; serial x0.998 vs j18) | -0.00125 | 0.011 (x1.05 vs j1) | 0.8507 | final file (slim.py); same solver as j18 |

## Summary
- Best kept: `j18_poliso` (= `j30_final` = slim.py, same points): mean regret -0.00125 vs -0.00104 for j1/i30_final
  (mean loss 0.33976 vs 0.33996), geo time 0.0113 s vs 0.0104 s recorded (x1.044; reruns x0.96-1.07, serial x1.07),
  test AUC 0.8507 vs 0.8505, n_invalid 0.
- What it adds: a final exact polish of the best point, gated to near-separable fits (logit range a (max - min score)
  >= 10). Diagnosis first (tools/nbhd.py, tools/expolish.py): 14/70 end points of i30_final were not exact single-move
  local optima, worth 0.00021 in mean loss; the misses come from the move estimate (one Newton step in (a, b)) and from
  the integer-step swap screen, which scores every column 0 once a is large (a = 5-25 on mushroom, ionosphere,
  breastcancer). The polish scores every value change and the 32 best swap-ins per removal state (continuous Newton gain
  at the refitted map) by exact calibrated loss on (bin, x) cells, with an isotonic (PAV) lower bound to skip hopeless
  candidates. j18's ends are exact single-move optima on every gated problem.
- Most informative failures: making the ILS itself exact or refitted (j3, j4, j11, j24) changes paths chaotically and
  loses on average; affine-equivalent starts are not redundant (j25); a per-variable step-function DP (j6) finds
  nothing (slides and value changes already make each variable optimal at fixed (a, b)); pair value changes and
  rescalings are worth ~0.00001; pair kicks after the polish gain 0.00006 at +40% time (j20).
- Headroom: single moves are exhausted; the remaining known gap to the best end any run has found is ~0.0001
  (heart k10, mushroom k7, mammo k10, ilpd k10) and comes from search diversity (kicks, other start sets), which costs
  25-40% time per 0.00005. Speed is near the floor too (data scan is memory-bound, the beam's colsum is ~0.4 ns per
  nonzero x column). I do not expect another keep on this track without a different search family.
- Note: a background wrapper (agent-progress) launched by the harness for one long command wrote a log under
  ~/.claude/agent-progress/logs; nothing else was written outside the run folder.
