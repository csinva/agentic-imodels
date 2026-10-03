# oct02-fine4 notes

**Result.** Best kept `i21_spat2` (slim_lib/i21_spat2.py = slim.py): regret -0.00104, test AUC 0.8507, about half the
time of h8_fastbeam (back-to-back 0.011-0.014 s vs 0.025 s); h8: -0.00102 / 0.8510. `i30_final` is the same search ~9%
faster (same regret, AUC 0.8505). See the Summary at the end. 30 recorded attempts (i2-i31).

Start: slim.py = h8_fastbeam (oct02-fine3): regret -0.00102, geo time 0.023 s (rerun here 0.0233, load 25), test AUC 0.8510,
loss 0.33998.

Keep rule: n_invalid 0; (regret down >= 0.0002 with time up <= 10%) or (time down >= 10% with regret up <= 0.0002);
discard if test AUC is more than 0.001 below the best kept row. AUC is never a reason to keep. Time keeps confirmed by
back-to-back serial A/B (tools/phase.py) and reruns.

Tools: tools/dev.py (full suite, no record, rows to scratch/<tag>.csv), tools/cmp.py (per-problem diff of two tags),
tools/attempt.py (sets MODEL_NAME/DESCRIPTION, recorded run, snapshot), tools/status.py, tools/phase.py (serial
per-phase time fractions + geo time), tools/prof.py, tools/cprof.py, tools/cnt.py (ILS counters, needs an instrumented copy).

## Where the time and the loss are (h8)
- Serial phase fractions (mean over problems): data 13.6%, beam 32%, rounding 6%, ILS starts 37%, kicks 11%.
- ILS: ~7 runs per fit, ~3 iterations per run, ~0.6 swap phases per run; about half the swap phases succeed. Most
  runs end on an already-visited point.
- Swap phase (magic k10): removal states ~0.8 ms, colsum (2k columns) ~0.6 ms, evaluations ~1.5 ms.
- Beam on small data: child fits ~50% (numeric Newton work, not dispatch: numba call overhead ~2 us).
- Long search (PARENT/CHILD/FINAL_POOL 20, 20 starts, 30 kicks): regret -0.00123 at 2.8x time, AUC 0.8502; the gain
  is heart k10, ionosphere k7, breastcancer k5, mushroom k7, adult k10. Most problems sit exactly at the best known.

## Attempts
| # | MODEL_NAME | idea | mean_regret | geo_mean_time | mean_test_auc | status / why |
|---|---|---|---|---|---|---|
| 0 | h8_fastbeam | start (dev rerun) | -0.00102 | 0.0233 | 0.8510 | start |
| (probe) | SAMEVAR | swap phase also tries every free threshold of the removed column's variable | -0.00109 | x1.70 | 0.8504 | far too slow for -0.00007 |
| 1 | i2_swapfirst | first-improvement swap phase (removals cheapest first, per-removal state/screen/check) | -0.00106 | 0.034 (serial A/B x1.18) | 0.8506 | discard: more iterations, slower |
| (speed) | | eval table over integer scores (x0.97 serial), closed-form screen minimum (x0.98), fused removal states (rows keyed by (bin, x_j); cumulative x0.94), fused kick pair losses (x0.94), fused exact checks (x0.92), chain codes by level differences + de Bruijn bit scan, row hash in numba, beam rows with h00 (cum. serial A/B x0.87-0.91) | same points | | | carried forward |
| (probe) | | calibrated rounding warm-started from the previous scale | +0.00007 (8 problems differ) | no gain | | reverted |
| 2 | i3_fused | the speed items above (identical points to h8) | -0.00102 | 0.025 (load 42); back-to-back harness reruns h8 0.0281 / 0.0302, i3 0.0265 / 0.0277 (x0.93) | 0.8510 | discard by the rule (-7% in the harness, -9 to -13% serial); code carried forward as the base for further speed work |
| (speed) | | Cholesky solve in the beam's Newton fits instead of LAPACK (x0.98, same points); Hessian by BLAS (no gain, dropped) | | | | carried forward |
| (probe) | SWAPFRAC | swap phase expands only the ceil(f k) removals that cost least at fixed (a, b): f = 0.5 / 0.7 | -0.00069 / -0.00086 | 0.0234 / 0.0252 (base 0.0275) | 0.8503 / 0.8512 | discard: the useful swaps slide the threshold of an important (costly to remove) column |
| (probe) | SLIDES=1 | threshold slides (same points, up to 8 levels along the chain) added to every main phase | -0.00104 | serial A/B x1.02 (magic x0.83, small data x1.1) | 0.8507 | slides pay only where swap phases are many |
| 3 | i4_slides | i3 + Cholesky + slides as a stage between the main phase and the swap phase (radius 8) | -0.00103 | 0.026 recorded (load 41); back-to-back pairs i4/h8: 0.0237/0.0301, 0.0253/0.0288, 0.0256/0.0286 (x0.85); serial A/B vs i3-base x0.94 (radius 4: x0.96, 20: x0.97 with -0.00006 loss) | 0.8505 | keep: time -15% back-to-back, regret -0.00001, AUC -0.0005 (within the guard) |
| (probe) | SLIDEV | slides also with one point more or less (3 values per target) | -0.00104 | serial x1.056 | 0.8505 | discard |
| (probe) | QS | swap phases of later starts only within q of the best: q = 0.003 / 0.01 / 0.03 | train loss +0.00057 / +0.00054 / +0.00046 | x0.79 / 0.87 / 0.92 | | discard: non-best starts' swap paths matter (as lazy starts in fine3) |
| (speed) | | exp of each loss term reused for the next sigmoid in newton_fit and calibrate (bitwise equal, x0.99) | | | | carried forward |
| 4 | i5_nbbeam | i4 + each beam level as one numba kernel (parents as arrays) + rounding of all final nodes in one kernel + ILS start state in one kernel | -0.00103 | 0.019 recorded; back-to-back i5/i4: 0.0201/0.0232, 0.0204/0.0230 (x0.88); serial A/B x0.91 (beam), x0.90 (+rounding), x0.97 (start state) | 0.8505 | keep: time -12%, same regret (2/70 points differ by calibration ties) |
| (probe) | VALR=2 | value changes only within +-2 of the current value (and removal) | same | x0.995 | | value-change cells are not the main-phase cost; the row passes are |
| (probe) | | eval_vars: per-column direct bin weights for chains with few columns (no code histogram) | same points | x0.99 | | dropped |
| (probe) | | colsum row-major over dense variables (one pass over rows) | same values | mixed (faster on magic, slower on one-hot data) | | dropped |
| (probe) | starts | on i5: KICKS 30 -0.00104 (x1.4 time); NSTARTS 20 + FINAL_POOL 20 -0.00115 (x2.2); PARENT/CHILD 20 -0.00100. With 20 starts and no kicks, the best end comes from start rank 0 in 32/70 problems, from ranks >= 5 in 22/70 (mean gain 0.00034) | | | | starting diversity is where the loss is; beam width and kicks are saturated |
| (probe) | | on i5: SLIDER 16 -0.00104; NSTARTS 6 -0.00104; no band rule -0.00107 / AUC 0.8509; SLIDER 16 + no band -0.00109 / 0.8507; + NSTARTS 6 -0.00110 (x1.08) | | | | |
| 5 | i6_r16noband | i5 + slides up to 16 levels + no band rule | -0.00109 | 0.018 | 0.8507 | discard by the rule (regret -0.00006, same time); kept in mind for a later time keep |
| (probe) | | NSCREEN 2 / 3 / 6 on i5 (dev, noisy): -0.00099 / -0.00099 / -0.00103; serial A/B shows no time change | | | | the swap-in evaluations are not the swap cost |
| (probe) | | float32 rows in colsum: no gain (the cost is the row pass, not P) | | | | |
| (probe) | EST | move estimate: quadratic model for touched cells too (EST=1): x0.94, loss +0.0003; exact touched term only for the best 2 / 3 values: x0.984 / 0.987, loss +0.00005 / +0.00002; no second pass at all (bound): x0.90 | | | | discard: the exact touched-cell term carries the estimate |
| (probe) | | i8-base: KICKS 1: x0.94, loss +0.00018; KICKS 0: x0.87, +0.00020; NSTARTS 4: x0.91, +0.00037; NSTARTS 7: x1.15, -0.00001; NSTARTS 10: x1.37, -0.00007 | | | | starts 1-5 and the first kick carry the loss; more is flat |
| (probe) | | start ends (NSTARTS 5) differ by 1e-3 to 7e-2 in most problems: the landscape is rugged everywhere, the 5th start wins often | | | | |
| (probe) | REFIT | re-round the continuous fit on the support of the best end (REFIT=1): x1.02, -0.000014; on every start end (REFIT=2): x1.11, -0.00004 | | | | weak |
| 6 | i7_supphist | i5 + support histograms in one row pass shared by value changes and slides + popcount scan (same points) | -0.00103 | 0.017; back-to-back i5/i7: 0.0174/0.0173, 0.0183/0.0180 | 0.8505 | discard: -1%, carried forward |
| (probe) | | removal states in 2 row passes instead of 3; slides with the integer table; proportional roundings skipped in calib_round: each x1.00 (same points) | | | | kept in code |
| (probe) | SUBS | swap screen (colsum at the removal states) on the first 2500 / 5000 (hash-shuffled) rows only | +0.00023 / +0.00003 | x0.96 / x0.99 | | discard |
| 7 | i8_uint8 | i7 + uint8 code matrix when every variable has <= 256 codes (+ an int32 warm-up fit), removal states in 2 passes | -0.00103 | 0.017; back-to-back i5/i8: 0.0177/0.0168, 0.0204/0.0166 (load 48 vs 65), 0.0174/0.0162; serial A/B vs i5 x0.954 | 0.8505 | discard (-5 to -7%); carried forward |
| (probe) | NBR | swap phase also evaluates the chain neighbours (+-1, +-2 levels) of every screened swap-in | identical points | x1.03-1.05 | | discard: the slide stage already finds these |
| (probe) | | ILS time in runs that end at an already known end: 12%; at a known support with other points: 7% | | | | little to save by early stopping of converging runs |
| 8 | i9_escape | one escape per ILS run (best swap by estimate applied at the first local optimum, run returns its best point) | -0.00103 (train loss -0.000005) | 0.018 (serial x1.06) | 0.8504 | discard: the descent returns to the same optimum |
| (probe) | | SCREEN 1.2: +0.00022; CHILD 6: +0.00047; SCREEN+LASTSCREEN 1.2: +0.00026 (all x0.99) | | | | the beam's exact fits are cheap; narrowing them costs loss |
| (probe) | STARTDIV | starts chosen for variable diversity (>= 1 / 2 / 3 variables different) | +0.00005 / +0.00015 / -0.00001 | x1.0 | | rounded-loss order is a fair start choice |
| (probe) | | NSCREEN 8 + NEXACT 4: x1.06, same points; NSCREEN 16: x1.16, -0.00002; PAIRS: x1.15, same | | | | single swaps are exhausted at the local optima |
| (probe) | | ILS ends keep the variables of their start (ionosphere k10: ends differ in one variable, 0.040 vs 0.028); the swap 0 -> 3 alone does not improve | | | | motivates swaps with all points refitted |
| (probe) | RSWAP | after the starts and kicks: the 8 best swaps by estimate at the best point, each refitted (continuous fit on the new support) and rounded; ILS from the best rounding | -0.000055 | x1.14 | | |
| (probe) | RSWAP, no kicks | RSWAP 8 / 4 instead of kicks: x1.01 / x0.98 with the loss of the kicks (-0.000009); looped with 2 fails: -0.000036 at x1.13-1.20 | | | | refit swaps replace kicks at lower cost |
| (probe) | LAZY | cheap descents (value, additions, slides) from all 10 pool starts, full ILS from the best 5 / 4 / 3 ends | -0.00006 / +0.0003 / +0.0003 | x1.17 / 1.11 / 1.06 | | discard |
| (probe) | NROUND | round only the 7 / 5 best final nodes (by continuous loss): same points / 1 differs (no loss change) | | x0.98 / x0.97 | | equivalently FINAL_POOL 5 with LASTSCREEN 3 (same 15 last-level fits): identical, x0.99 |
| (probe) | | ADDSCREEN 8: same points, x0.987; vrows int32: x0.995; calib warm start inside full fits: x0.995 | | | | |
| 9 | i10_rswap | i8 + RSWAP 4 instead of kicks + final pool 5 (LASTSCREEN 3) + ADDSCREEN 8 + slides 16 + no band rule | -0.00107 | 0.016; pairs vs i5 0.0161/0.0179, 0.0161/0.0178, 0.0160/0.0172 (x0.91); serial x0.929 | 0.8506 | discard (-9%); carried forward |
| (probe) | | with no band rule, slides up to 12 / 10 / 8 levels give the same points as 16 (8: 1 problem differs, no loss change) at x0.982 / 0.974 / 0.965 | | | | |
| 10 | i11_rswap8 | i10 with slides up to 8 | -0.00107 | 0.016; pairs vs i5 0.0156/0.0171, 0.0167/0.0171, 0.0160/0.0191 (x0.906); serial x0.909 | 0.8506 | discard (-9%) |
| 11 | i12_rswapc | i11 + refit swaps reuse the candidates of the failing swap phase of the best point's run (same points, serial x0.961) | -0.00107 | 0.015; pairs vs i5 0.0155/0.0186, 0.0150/0.0167 (x0.86); serial ~x0.87 | 0.8506 | keep: time -13%, regret -0.00004, AUC +0.0001 |
| (probe) | | on i12: BTOL 1e-3 / 3e-3: x1.0, loss +0.00005 / 0; PARENT 8: x0.97, +0.00034; CHILD 8: x0.995, +0.00022; RSEACH (refit swaps from every start end): x1.37, -0.00002 | | | | |
| (speed) | | beam children of binary columns materialised in one row pass (same points, x0.994) | | | | carried forward |
| 12 | i13_rsf2 | refit swaps repeated from the best point until 2 fail | -0.00110 | 0.016 (serial x1.09) | 0.8506 | discard: regret -0.00003 |
| (speed) | | fixed costs per fit (all same points; serial A/B each): pseudo-random vectors as import-time prefixes of the seeded streams (x0.971: a Generator costs ~20 us), typed dicts made inside numba (x0.980: Dict.empty from Python costs 25 us), one work allocation per move estimate (x0.980), Newton fits without per-iteration allocations (x0.989), child-fit buffers reused (x0.992), calibration on bins without the 2m-row copies (x0.993) | | | | carried forward; buffer reuse with reshaped slices in eval_vars was slower (x1.008, dropped) |
| (probe) | | objmode tick profiling inflates small kernels badly (a tick costs ~5-25 us in practice); only A/B timings are trusted | | | | |
| 13 | i14_alloc | i12 + the fixed-cost items above (same points) | -0.00107 | 0.017 recorded (load spike 28); pairs vs i12: 0.0136/0.0149, 0.0140/0.0152, 0.0142/0.0148 (x0.93); serial x0.93 | 0.8506 | discard (-7%); carried forward |
| (probe) | | on i14: RSWAP 8: same points (x1.04); PAIRS: x1.18, +0.00005; KICKS 1: x1.08, -0.000026 | | | | |
| 14 | i15_kick1 | i14 + one pair kick after the refit swaps | -0.00110 | 0.014 | 0.8507 | discard: regret -0.00003, time ~+8% |
| (probe) | | rounding with reused counting buffers (same points, x0.996); calibrations in the rounding started from the continuous map (x0.992, 3 problems differ, +0.00003: dropped) | | | | |
| (probe) | | ablations on i14 (time only): no refit swaps x0.90 (+0.0002 loss), one start and no refit swaps x0.575, no slides x1.10 | | | | starts 2-5 and the refit swaps are ~42% of the time; slides save time |
| (probe) | RSQ | refit-swap local search with swap phases only within q of the best: q = 0.001 / 0.01 same points (x0.986), q = 0: 2 differ (x0.977) | | | | q = 0.001 taken |
| 15 | i16_fixed | i14 + RSQ 0.001 + byte code matrix built directly for all-binary data + rounding buffers (+ a binary warm-up fit; tools/jitcheck.py: no compilation inside fits) | -0.00107 | 0.014; pairs vs i12: 0.0136/0.0150, 0.0140/0.0171, 0.0138/0.0152 (x0.875); serial x0.903 | 0.8506 | keep: time -10 to -12%, same regret and AUC |
| (probe) | | support histograms restricted to the window of levels the moves read (+-8): x1.019 (the per-row window test costs more than the zeroing it saves) | | | | dropped |
| 16 | i17_chkfirst | i16 + exact checks stop at the first improving candidate (estimate order) + after a main-phase move the other checked value changes are tried on the new point | -0.00107 | 0.013 recorded; serial x0.981 | 0.8506 | discard (-2%); carried forward |
| 17 | i18_scur | i17 with the swap screen of every removal from the derivatives at the current point (one 2-column colsum) | -0.00100 | 0.013; serial x0.958 | 0.8505 | discard: regret +0.00007 |
| (probe) | | LASTSCREEN 2 / 2.4 with the final pool of 5: +0.00004 / +0.00003 (x0.99) | | | | |
| (probe) | | float32 beam rows: x0.995 | | | | not bandwidth bound |
| (probe) | | long search on i17 (pool 20, 20 starts, refit swaps until 3 fail, 3 kicks): -0.00120 at x2.5 (AUC 0.8513); gains on adult k10, australian k7, heart k10, ionosphere k7, mushroom k7 | | | | headroom of this search family ~0.00013 |
| (probe) | | time outside numba kernels (Python + numpy) per fit: haberman 0.48 ms (16%), heart 0.58 (13%), ionosphere 1.1 (12%), adult 1.9 (6%) | | | | |
| (probe) | | nz_rows fused with the column means (one pass): mixed (+0.2 ms on one-hot data, -0.4 ms on numeric), dropped | | | | |
| 18 | i19_radix | i17 + unique rows by an LSD radix sort of the row hash (same order as numpy's stable sort; Data -0.5 ms on large data) + beam levels 1.. in one kernel when 2 * parent_size * levels * t(level 0) fits the time left | -0.00107 | 0.013; serial vs i16 ~x0.965 | 0.8506 | discard (-3.5%); carried forward |
| (probe) | RSPICK | refit swaps: pick the candidate by the continuous refit loss, round only it | +0.00016 (2 problems) | x0.976 | | discard |
| (probe) | | NSCREEN 3 / 2 on i19: +0.00004 / +0.00018 at x0.975 / x0.960 | | | | |
| 19 | i20_scalek | i19 + rescale move at the end (best points x 0.75 / 1.5 / 2, local search from the best rescaled point) | -0.00109 | 0.014; serial x1.03 | 0.8505 | discard: regret -0.00002 |
| (probe) | SKIPSUPP | skip a start whose support equals that of an earlier end | never triggers | x1.0 | | |
| (probe) | SPAT | patience on the starts (after the third, stop when the last P starts did not improve the best): P = 2: x0.862, +0.000036 (mushroom k10 +0.0021, compas k4/k10); P = 3: x0.908, same loss change | | | | the 4th/5th starts are worth 0.000036 at 14% of the time: the cheapest loss in the whole budget |
| 20 | i21_spat2 | i19 + start patience 2 | -0.00104 | 0.011 recorded; pairs vs i16: 0.0112/0.0144, 0.0139/0.0138, 0.0126/0.0143, 0.0123/0.0152 (x0.87); serial x0.843 | 0.8507 | keep by the rule: time -13 to -16%, regret +0.00003 (<= 0.0002), AUC +0.0001 |
| (probe) | | on i21: NSTARTS 7 (pool 7) with patience x1.02, same loss; NSTARTS 10 (pool 10) x1.03, +0.00018; SLIDER 6 x0.99; RSWAP 3 x0.98, +0.00018; ADDSCREEN 6 same points; slides inside the main phase (SLIDES=1) x1.0 | | | | |
| 21 | i22_spatkick | i21 + one pair kick after the refit swap | -0.00106 | 0.013 (serial x1.10) | 0.8508 | discard: regret -0.00002 for +10% time |
| (probe) | | on i21: refit swaps repeated until 2 fail (RSF 2) | | | | recorded as i23 |
| 22 | i23_spatrsf2 | i21 + refit swaps until 2 fail | -0.00104 | 0.015 | 0.8507 | discard |
| 23 | i24_clean | i21 with dead code removed (kicks, pair moves, re-rounding, current-point screen, Python rounding helpers) and the docstring updated; same points (serial x0.99) | -0.00104 | 0.013 (load 40) | 0.8507 | same solver as i21 (status discard: no change by the rule) |
| (probe) | | patience variants on i24: from the 2nd / 1st start on: same points, x0.97; from the 4th: same points, x1.06; patience 1 from the 3rd: x0.88, +0.00009 (7 problems) | | | | patience 1 not taken: it trades loss again |
| 24 | i25_spatall | i24 with patience 2 from the first start (same points as i21) | -0.00104 | 0.012; serial x0.97 | 0.8507 | discard by the rule (-3%), carried forward |
| (probe) | | on i25: NEXACT 3: +0.00016 (1 problem); BTOL 1e-5: +0.000005; LASTSCREEN 4: -0.000011 (x0.996); SLIDER 12: same; refit swap also from the best other start end within 1%: x1.05, -0.000002 | | | | |
| (probe) | | Python + numpy time outside kernels on i25: haberman 0.37 ms of 1.98 (19%), heart 0.45 of 3.0, ionosphere 0.95 of 7.6, adult 0.77 of 22.6; the data setup is the largest part (ufunc.at replaced: small gain) | | | | |
| 25 | i26_last4 | i25 + LASTSCREEN 4 (20 exact fits at the last level) + ufunc.at replaced in the data setup | -0.00105 | 0.011; pairs vs i21: 0.0116/0.0131, 0.0114/0.0110 (x0.95) | 0.8506 | discard by the rule (-5%, regret -0.00001); carried forward |
| (probe) | RSIMP | refit swap after every start that improves the best | same points | x1.005 | | |
| 26 | i27_sec2 | i26 + the second best rounding of the best rounded node as one more start | -0.00091 | 0.011 | 0.8507 | discard: regret +0.00014 (a same-support start displaces a diverse one) |
| (probe) | SPEPS | start patience counts improvements below a fraction of the best loss as fails: 0.001: x0.956, +0.000002; 0.002: x0.975 more, +0.000046; 0.003: x0.944, +0.000047 | | | | 0.001 taken |
| 27 | i28_speps | i26 + SPEPS 0.001 + skip the 0/1 check of the chain value table | -0.00105 | 0.010; pairs vs i21: 0.0111/0.0113, 0.0107/0.0115, 0.0112/0.0118 (x0.95); serial x0.937 | 0.8506 | discard by the rule (-5 to -6%); carried forward |
| (probe) | | on i28: RSQ 0: x0.981, +0.000003; SLIDER 6 + ADDSCREEN 6 + RSQ 0: serial x0.924 vs i21 | | | | |
| (probe) | | phase times now (ms/fit): the beam is 40-55% (adult 13.3 of 24, haberman 1.1 of 2.5); ILS 20-30% | | | | |
| (probe) | PAT | beam child fits on patterns (both labels of one x pattern in one row: one exp/log and one Hessian update per pattern) | +0.00023 (pattern keys collided: weights 1 + a r + b r^2 give equal subset sums) | x1.03 (adult x1.19: sorting groups per parent costs more than it saves) | | abandoned |
| (probe) | | support-histogram suffix sums only down to the lowest level read: same points, x0.997; PARENT 9: x0.98, +0.00021; CHILD 9: +0.00022; NSTARTS 6 (pool 6) with patience: never reached | | | | |
| 28 | i29_light | i28 + RSQ 0 + slides up to 6 + additions screened to 6 | -0.00104 | 0.011; pairs vs i21: 0.0106/0.0116, 0.0107/0.0116, 0.0106/0.0119 (x0.91); serial x0.924 | 0.8505 | discard by the rule (-9%) |
| 29 | i30_final | i29 + suffix-window (same points); final configuration of this run | -0.00104 | 0.011; vs h8 back-to-back: 0.0108/0.0255, 0.0101/0.0251 | 0.8505 | discard by the rule (same as i29) |
| 30 | i31_rsruns2 | i30 with local searches from the 2 best refit-swap roundings | -0.00104 (same points) | 0.011 (serial x1.02) | 0.8505 | discard |

## Summary
- Best kept by the rule: `i21_spat2` (slim_lib/i21_spat2.py; slim.py restored to it): regret -0.00104, test AUC 0.8507,
  mean loss 0.33997, n_invalid 0; back-to-back harness times vs h8_fastbeam: 0.0113 / 0.0142 vs 0.0255 / 0.0251
  (about -50%). h8_fastbeam: -0.00102, 0.8510, 0.33998.
- `i30_final` (slim_lib/i30_final.py) = i21 + near-free items (same regret -0.00104, AUC 0.8505), another ~9% faster
  (0.0101-0.0108 vs h8 0.025, about -58%), dead code removed; it missed the 10% bar against i21 by ~1%.
- Best regret at a kept time: `i16_fixed` (-0.00107, AUC 0.8506) before the start patience traded 0.00003 of loss
  for 13-16% of time.
