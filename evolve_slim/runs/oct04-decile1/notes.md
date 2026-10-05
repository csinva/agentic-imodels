# oct04-decile1 notes

Track: default visible decile suite (70 problems). Start: slim.py = oct03-fine5 j30_final (j18_poliso points) with
MS_SQRT=0. Goal: v35_scratch's visible loss (regret <= 0.00010) at geo time <= 0.02 s, test AUC within 0.001.

Reference rows: v35_scratch_cmp regret 0.00010, time 0.035 s, AUC 0.8424, loss 0.35592.

Keep rule (program.md): n_invalid 0; (regret down >= 0.0002 with time up <= 10%) or (time down >= 10% with regret up
<= 0.0002); discard if mean test AUC drops > 0.001 vs the best kept row.

Setup fix: j30_final's evaluation block hard-coded `suite="visible_fine"`; the first m1_base run therefore scored the
fine suite and wrote its rows into the decile results. Those rows were deleted and the block restored to the template's
(default suite). tools/dev.py likewise points at the visible suite.

Tools (copied from oct04-fine6): rec2.py (record a file as an attempt), cnt2.py (improved / worsened vs a reference),
dev.py (unrecorded full run to scratch/), status.py; new: tools/probe.py (beam supports, their roundings, start ends
on one problem, for m* files and for v35).

## Diagnosis
- m1_base vs v35: 5 better / 14 worse, mean dloss +0.00053. Excess sum: spambase k3/4/5 +0.025, ionosphere k10
  +0.0089 / k5 +0.0011, fico k4 +0.0018, australian k5/7/10 +0.002, mammo k5 +0.0011. Wins: magic k10, ilpd k7,
  heart k10.
- spambase k3 (tools/probe.py): 3 of m1's 5 final beam supports contain CapitalRunLengthLongest (raw scale up to
  ~10^3, continuous coefficient tiny), which rounds to 0 at every scale: those supports collapse to 2 columns
  (rounded loss 0.47-0.49). v35's pool of 10 still holds Remove+HP+Dollar (rounded 0.4073, its final 0.3982); m1's
  15-fit levels with one child per multiset never fit it.

## Attempts
| # | MODEL_NAME | idea | regret | geo time | AUC | vs best kept (improve / worsen) | vs v35 (improve / worsen) | status / why |
|---|---|---|---|---|---|---|---|---|
| 1 | m1_base | start (j30_final, MS_SQRT 0) | 0.00063 | 0.013 | 0.8423 | - | 5 / 14 (+0.00053) | keep (reference) |
| 2 | m2_repr | beam: a child with a coefficient below 0.0911 x its largest (rounds to 0 at every scale 0.5..5.49 of the rounding grid, so not representable by integer points) is ranked after all others (loss + 1e3) | 0.00043 | 0.013 | 0.8423 | 3 / 1 (spambase k3 -0.0107, k4 -0.0041, k5 -0.0020; ilpd k10 +0.0027) | 4 / 13 (+0.00033) | keep: -0.000201 at the same time |
| 3 | m3_swng | m2 + ILS swap phases screen swap-ins by the continuous Newton gain G^2/H everywhere (SWNG 1e-9) instead of the integer-step quadratic, which scores 0 when a * x steps overshoot (raw columns of spambase) | 0.00033 | 0.014 (x1.03) | 0.8425 | 2 / 0 (spambase k5 -0.0065: the missed single swap WordFreq0 -> WordFreqEDU found by tools/nbhd.py; bank k10) | 4 / 13 | discard: -0.000095 < 0.0002; idea carried as a candidate |
| (dev) | | FINAL_POOL 10: ionosphere k10 -0.0043, x1.13; SCREEN 3 / LASTSCREEN 6: spambase k5, ionosphere k5 better, australian k10, ilpd k10 worse, -0.00011, x1.14; SIGMAX 2: no net; NSCREEN 8: no change; SPAT 3: fico k4, mammo k5 better, -0.00004 | | | | | | not recorded |
| 4 | m4_jitwarm | m2 + the import warm-up also compiles swap_phase for every code layout with recal omitted and True. Found from the harness times: breastcancer k10, mushroom k7, mushroom k10 took ~4 s (one numba compile of swap_phase inside the first refit-swap / polish fit of a worker; tools/jitcheck.py now runs every visible problem and reports compiles per problem: none after the fix) | 0.00043 | 0.011 (x0.834) | 0.8423 | same points | 4 / 13 | keep: -17% time |
| (probe) | | Polish gate: POLISH 0 changes 1/70 (mushroom k5 +0.0015) for serial -6.5%; the logit-range gate passes on spambase (raw heavy-tailed columns: a * range 130-320, even 1-99% quantiles 40-180) where the polish never gains; extra gate POLH (loss < 0.35 x base entropy) keeps the same points at serial x0.973 | | | | | | not recorded alone |
| (probe) | | wide reference (SCREEN 10, LASTSCREEN 10, FINAL_POOL 10, no start patience): regret 0.00020 at x1.57; still loses to v35 on ionosphere k10 (+0.0046), ilpd k10, australian k5/7/10, heart k3, fico k3 | | | | | | the gap is not only beam width |
| 5 | m5_upscale | m4 + upscale restarts: on k = 3 the ILS ends at equal points (heart (-1,-1,-1), compas, fico) where v35 has (-4,-5,-4): w and f*w have the same loss, but no single +-1 change of (1,1,1) reaches the finer ratios. After the starts and at the end, a local search from f*w (largest point at bound-1 and at the bound), while it improves | 0.00041 | 0.010 (x0.935) | 0.8421 | 3 / 0 (heart k3 -0.0005, compas k3 -0.0003, fico k3 -0.0001) | 4 / 10 | discard by the rule (-0.000013), but free; carried as a candidate (UPS=2) |
| (dev) | | clipped upscales (round(f w) clipped at +-5, f = 2, 3) for points that already use the bound (mammo k5 (1,-2,1,-5,1) -> v35's (3,-4,2,-5,3)): before rswap x1.35 and australian k10 +0.0012 (path change); at the end only x1.18; without swaps x1.11 serial; with an exact value-only descent x1.31 (exact_values is a row pass per value on raw columns). Gains mammo k5/k10, ilpd k10: -0.000025 | | | | | | dropped: too costly for the gain |
| 6 | m6_reprnb | m4 with the representability penalty only when the tiny coefficient is on a non-binary column (REPRNB): binary columns with a tiny coefficient are left to the local search | 0.00039 | 0.010 (x0.913, noise: same work) | 0.8421 | 1 / 0 (ilpd k10 -0.0027, the m2 regression undone) | 5 / 12 | discard: -0.000038 |
| (A/B) | | serial A/B costs vs m5 (2 reps, 70 problems): SWNG x1.014 for -0.000095; FINAL_POOL 10 x1.127 for -0.000063; SPAT 3 x1.093 for -0.000031; SCREEN 3 / LASTSCREEN 6 x1.124 for -0.000111; POLH 0.35 x0.972, same points | | | | | | |
| 7 | m7_combo | m4 + REPRNB + SWNG + UPS 2 + POLH 0.35 (the cheap items) | 0.00028 | 0.011 (x0.992) | 0.8421 | 6 / 0 (spambase k5 -0.0065, ilpd k10 -0.0027, heart k3, compas k3, bank k10, fico k3) | 5 / 9 (+0.00018) | discard by the rule: -0.000145 < 0.0002 at the same time. Used as the working base for further combinations (knob defaults), the reference row stays m4 |
| (dev) | | knob sweep from m7 (NEXACT 3, ADDSCREEN 10, RSWAP 6, RSF 2, SLIDER 8, SPEPS 0): no point changes. The ILS ends are robust; the gap is upstream | | | | | | |
| (probe) | | **hybrid: v35's beam + calibrated rounding as the starts, m7's local search / refit swaps / polish (scratch/hyb.py, HYB=1): regret 0.00009 (beats v35's 0.00010), 7 better / 1 worse vs m7 (ionosphere k10 -0.0089, fico k4 -0.0018, australian k5/7/10, ionosphere k5), x2.9 time.** So the fine lineage's local search is as good as v35's or better; the loss gap is in the beam's starts | | | | | | |
| (dev) | | beam diversity limits of the fine track: SIGMAX 99 (several children per multiset of variables) alone: no net; PPV 0 (several thresholds of one variable per parent) alone: ionosphere k10 better, magic k4/k5 worse (+0.00008); both off at the narrow width (15 fits): +0.00036 (ionosphere k7/k10 +0.013); both off with SCREEN 10 / LASTSCREEN 10 / pool 10: regret 0.00005 at x1.37; SCREEN 5 / LASTSCREEN 5 / pool 10: same points, x1.2; SCREEN 3 / LASTSCREEN 3 / pool 10: same points, x1.04; SCREEN 4 / LASTSCREEN 2: 0.00008; pool 5 with SCREEN 10: 0.00021; CHILD 5: 0.00087 | | | | | | on decile data (about 9 thresholds per variable) the one-threshold-per-variable and one-child-per-multiset rules of the fine track remove good supports; with them off the beam needs a few more fits and a pool of 10 |
| 8 | m8_beamdiv | m7 + PPV 0, SIGMAX 99, SCREEN 3, LASTSCREEN 3, FINAL_POOL 10 (30 exact child fits per level, 10 final supports rounded, 5 best roundings as starts) | **0.00005** | 0.012 (x1.061 vs m4) | 0.8421 | 13 / 0 vs m4 (-0.00037) | **5 / 1 (-0.000047; worse only mammo k5 +0.0011)**, time x0.34 | **keep**: -0.00038 regret vs m4 at +6% time |
| (dev) | | ablations of m8: REPR 0 +0.00021 (spambase k4/k5: still needed); SWNG 0 +0.000002 (bank k10); UPS 0 no change (the wider beam already reaches heart / compas / fico k3); POLH 0 no change; POLISH 0 +0.00002 (mushroom k5); SPAT 1 +0.00004 at x0.95. NMULT 10 rounding scales: x0.95 dev, magic k7 +0.00007; UPS 0 + NMULT 10: serial x0.970; + SPAT 1: x0.87 but regret 0.00012 (> v35) | | | | | | |
| 9 | m9_rclip | m8 + 4 extra calibrated-rounding scales where the largest point is clipped at the bound (largest continuous point up to 3x the bound): when one point saturates (mammo: a rare category at -5) the other points get finer ratios ((1,-2,1,-5,1) -> (3,-4,2,-5,3)) | 0.00002 | 0.012 (x1.027) | 0.8421 | 3 / 1 (mammo k5 -0.0011, k10 -0.0006, k7 -0.0004; mushroom k5 +0.00004) | 7 / 1 | discard by the rule (-0.000029); kept as a candidate (clean, general) |
| (probe) | | phase shares of m9 (serial, mean over problems): data 0.10, beam 0.44, rounding 0.12, start runs 0.19, refit swaps 0.10, polish + upscale 0.04; geo 11.1 ms | | | | | | |
| (dev) | | speed knobs from m9: PARENT 8 same points x0.93; PARENT 6: regret 0.00259 (mushroom k3/4/5 +0.03 to +0.06, ionosphere k7/k10 +0.015/+0.022); SCREEN 2.5 same points x0.94; FINAL_POOL 8 same points; NMULT 12: +0.000015. Serial A/B: SCREEN 2.5 + UPS 0 + pool 8: x0.931 (+0.000012); + PARENT 8: x0.883 (+0.000012) | | | | | | |
| (probe) | | why PARENT 6 breaks mushroom k3: half of the beam's supports are mirror copies (one-hot complements: gill_size_eq_broad vs gill_size_eq_narrow give the same continuous fit up to the intercept), so 6 parents hold 3 distinct supports | | | | | | |
| 10 | m10_cdup | m9 + complement pairs of binary columns (x + x' = 1 on every row; matched by a hash of their bitsets and of the complemented bitsets, checked exactly) share one hash in the beam, so a mirrored support is proposed once | 0.00002 | 0.012 (x0.981; serial x0.998) | 0.8420 | 2 / 0 (tiny: bank k10, mushroom k5); 10 problems end at a mirrored point with the same loss | 7 / 0 | discard by the rule (no change); kept for robustness: with it PARENT 6 no longer breaks mushroom (ionosphere k7/k10 still do) |
| 11 | m11_lean | m10 + PARENT 8, SCREEN 2.5 (20 exact fits per level), FINAL_POOL 8, UPS 0 | 0.00004 | 0.012 (recorded x1.02 vs m8; serial A/B x0.880 vs m10) | 0.8421 | vs m8: 3 / 2 (mammo k5/7/10 better from m9; magic k4 +0.0002, k5 +0.0006) | 7 / 2 | discard as recorded (harness noise: ionosphere 8.4 ms vs 6.6, mushroom 31.9 vs 28.9 with less work); rerun |
| 12 | m12_lean_rerun | the same file as m11, rerun (program.md: rerun once when the decision rests on time) | 0.00004 | 0.010 (x0.881 vs m8) | 0.8421 | same points as m11 | 7 / 2 | **keep**: -12% time, regret -0.000017 vs m8 |
| (dev) | | from m12: REPR 0.05 / 0.15 same points (the threshold is not sensitive); SCREEN 3 same points (magic k4/k5 come from PARENT 8); POLISH 0 +0.00002 (mushroom k5); RCLIPN 8 / RCLIP 4: -0.000011 (ilpd k10, mammo k10) | | | | | | |
| 13 | m13_spat1 | m12 with start patience 1 | 0.00016 | 0.010 (x0.936) | 0.8425 | 0 / 3 (mushroom k5 +0.0056, spambase k5 +0.0021, ionosphere k5 +0.0011) | 7 / 5 | discard: -6% time only, and regret above v35's |
| 14 | m14_rclip8 | m12 with 8 clipped rounding scales up to 4x the bound | 0.00002 | 0.011 (x1.036) | 0.8422 | 2 / 0 (ilpd k10 -0.0007, mammo k10 -0.0001) | 7 / 2 | discard: -0.000011 |
| (dev) | | from m12: RSWAP 2 same points; NSTARTS 4 same points; PARENT 9 same points (magic k4/k5 not regained); CHILD 8: +0.00016 (spambase k4/k5/k7: raw columns need 10 proposals per parent). Machine load rose to ~300 (other users) during this batch, so its times are not usable | | | | | | |
| (dev) | | same-point trims from m12 (each with RSWAP 2 + NSTARTS 4): SLIDER 4, ADDSCREEN 4, NSCREEN 3, POLNS 16 leave every point unchanged; LASTSCREEN 2: spambase k5 +0.0021. RSWAP 2 + NSTARTS 4: serial x0.953 | | | | | | |
| 15 | m15_trim | m12 + all same-point trims (RSWAP 2, NSTARTS 4, SLIDER 4, ADDSCREEN 4, NSCREEN 3, POLNS 16) | 0.00004 | 0.010 (x0.996 recorded; serial A/B x0.931) | 0.8421 | 0 / 0 | 7 / 2 | discard: below the 10% bar, and it narrows every safety margin of the local search for held-out data |
| (dev) | | from m12: FINAL_POOL 10 regains magic k4/k5 (-0.000012, x1.10 dev); PARENT 10 with SCREEN 2: bank k10 only | | | | | | |
| 16 | m16_pool10trim | m12 + FINAL_POOL 10 + 8 clipped rounding scales up to 4x, paid for by m15's same-point trims (serial A/B x0.989 vs m12) | **0.00001** | 0.011 (x1.026) | 0.8422 | 4 / 0 (ilpd k10 -0.0007, magic k5 -0.0006, magic k4 -0.0002, mammo k10 -0.0001) | **7 / 0** | discard by the rule (-0.000023); the lowest loss of the run at m12's time |
| (dev) | | SCREEN 2 / LASTSCREEN 3 (16 fits per inner level, 24 at the last; pool 8) regains magic k4/k5 like pool 10 (-0.000012), serial x0.984 | | | | | | |
| 17 | m17_s2l3trim | m12 + SCREEN 2 / LASTSCREEN 3 + 8 clipped rounding scales up to 4x + m15's trims (serial A/B x0.922 vs m12) | **0.00001** | 0.011 (x1.092 recorded vs m12; x1.040 vs the m12 rerun m18) | 0.8422 | 4 / 0 (as m16) | **7 / 0** | discard by the rule (-0.000023; recorded time not lower) |
| 18 | m18_m12rerun | the m12 file rerun as a harness timing reference | 0.00004 | 0.011 (x1.050 vs m12's own record) | 0.8421 | same points | 7 / 2 | discard (reference): harness geo times of identical files differ by 5-12% between runs (m11 x1.02 vs m12 x0.88 for the same file), while serial A/B ratios repeat within ~1% |
| 19 | m19_s2l3rc8 | m12 + SCREEN 2 / LASTSCREEN 3 + 8 clipped rounding scales up to 4x, without m15's trims (serial A/B x0.992 vs m12) | **0.00001** | 0.011 (x1.020 vs m12, x0.972 vs the m18 rerun) | 0.8422 | 4 / 0 (ilpd k10, magic k5, magic k4, mammo k10) | **7 / 0** | discard by the rule (-0.000023 at the same time); the lowest-loss version at m12's time without narrower local search |
| (dev) | | per-variable cap in beam proposals (PPV k = at most k thresholds of a variable per parent; rewritten pick, PPV 0 gives m12's points): PPV 2: 3 / 4 (fico k7 +0.0012); PPV 3: 2 / 0 (magic k5, bank k10), -0.00001 | | | | | | |
| 20 | m20_ppv3 | m12 with at most 3 thresholds of one variable per parent in the beam proposals | 0.00003 | 0.012 (x1.110, noise: same work) | 0.8420 | 2 / 0 (magic k5, bank k10) | 7 / 1 | discard |
| (probe) | | generality check on the visible fine suite (99 thresholds per variable; tools/devfine.py, not recorded; reference oct04-fine6 k1_base: regret -0.00125, loss 0.33976, AUC 0.8507): m12 -0.00011 (loss 0.34090, AUC 0.8478); with MS_SQRT 1: +0.00019; PPV 3: -0.00019; PPV 1 + SIGMAX 1 (the fine rules): -0.00092. The decile beam rules cost ~0.001 on fine data: with many thresholds per variable the beam fills with neighbouring thresholds | | | | | | |
| (dev) | | threshold cells (CELLB): the beam's one-column-per-variable rule (PPV 1) and the multiset signature act on (variable, decile of the column's row fraction) instead of the variable, so thresholds of one variable that split the rows alike share a cell. Decile suite: CELLB 10 + PPV 1 (SIGMAX 1 or 99) = m12's points except bank k10 (-0.00005). Fine suite: CELLB 10 + PPV 1 + SIGMAX 99: -0.00080 (SIGMAX 1: -0.00059) | | | | | | adapts the diversity rule to the threshold density instead of a per-track setting |
| 21 | m21_cells | m12 + CELLB 10, PPV 1 (diversity rule on threshold cells) | 0.00004 | 0.011 (x1.019) | 0.8420 | 1 / 0 (bank k10 -0.00005) | 7 / 2 | discard by the rule (no decile change); kept for generality (fine suite -0.00080 vs m12 -0.00011) |
| 22 | m22_cells_s2l3 | m21 + SCREEN 2 / LASTSCREEN 3 + 8 clipped rounding scales up to 4x (m19 with cells) | 0.00003 | 0.012 (x1.119, noisy) | 0.8424 | 5 / 1 (ilpd k10, magic k5/k4, mammo k10, bank k10 better; ionosphere k5 +0.0011) | 7 / 1 | discard; fine suite -0.00065 (narrower inner levels lose some of the cell gain) |
| 23 | m23_cells_p10rc8 | m21 + FINAL_POOL 10 + 8 clipped rounding scales up to 4x (serial A/B x1.072 vs m12) | **0.00001** | 0.012 (x1.118 recorded) | 0.8421 | 5 / 0 (ilpd k10, magic k5/k4, mammo k10, bank k10) | **7 / 0** | discard by the rule (-0.000024 for +7% serial); fine suite -0.00097 (the best fine result of this run); the lowest-loss alternative for the held-out check |
| 24 | m24_noswng | m21 without the Newton-gain swap screen (SWNG 0) | 0.00004 | 0.011 (x0.993) | 0.8420 | 0 / 1 (bank k10 +0.0001) | 7 / 2 | discard: SWNG stays (cheap, bank k10) |
| 25 | m25_final | cleaned final: m12's settings (the best kept row) + the threshold-cell rule of m21; upscale code removed (off in every kept row); docstring rewritten for the decile pipeline; tools/jitcheck.py: no compilation inside any visible fit | 0.00004 | 0.011 (0.0111 s; x1.067 vs m12's 0.0104 record, x1.02 vs the m18 rerun of the same settings) | 0.8420 | vs m12: 1 / 0 (bank k10 -0.00005) | 7 / 2 (magic k4 +0.0002, k5 +0.0006) | final file (slim.py) |

## Summary
- **Best version: `m25_final` (= slim.py; the m12_lean_rerun settings plus the threshold-cell rule).** Visible decile
  suite: mean_regret 0.00004 (m1_base 0.00063, v35_scratch_cmp 0.00010), mean loss 0.35585 (0.35644, 0.35592), geo time
  0.0111 s recorded (0.0135 s, 0.0348 s; the same settings recorded 0.0104-0.0109 s in m12 / m18), test AUC 0.8420
  (0.8423, 0.8424), n_invalid 0. Versus v35: 7 problems better (magic k10 -0.0018, ilpd k7 -0.0014, heart k10 -0.0009,
  mammo k10/k7, ilpd k10, bank k10), 2 worse (magic k5 +0.0006, magic k4 +0.0002), about 3.1x faster. Versus m1_base:
  17 better, 2 worse, x0.82 time. Kept rows (program.md rule): m1 -> m2 (REPR) -> m4 (JIT warm-up) -> m8 (beam
  diversity) -> m12 (leaner beam).
- Lower-loss alternative: `m23_cells_p10rc8` (final pool 10 + 8 clipped rounding scales): regret 0.00001, better than v35 on
  7 problems and worse on none, serial x1.07 vs m25 (recorded 0.0116 s); discarded by the rule (-0.000024 only), but
  worth including in the held-out check.
- What made the difference, by size: (1) the beam's starts, not the local search (m8): the hybrid probe (v35's beam
  + this lineage's local search) already beat v35, and dropping the fine track's diversity rules (one threshold per
  variable per parent, one child per multiset of variables) with 20-30 exact fits per level and a final pool of 8-10
  gives the same supports at a third of v35's time; (2) non-binary columns: beam children whose raw-scale column
  rounds to 0 at every scale are ranked last (m2, spambase k3/k4 -0.011/-0.004); Newton-gain swap screen (spambase k5);
  clipped rounding scales for a point stuck at the bound (m9, mammo); (3) a numba compile inside timed fits (m4: ~4 s on
  3 problems per run, -17% geo time); (4) complement pairs counted once in the beam (m10, robustness: with 6 parents
  mushroom no longer collapses).
- Generality: on the visible fine suite (not the target, run with tools/devfine.py) m12's decile rules cost 0.001
  (regret -0.00011 vs -0.00125 for the fine track's k1_base); the threshold-cell rule (m21 / m25) recovers most of it
  (-0.00080; m23 -0.00097) without changing the decile points. Fine-suite AUC stays lower (0.8477-0.8482 vs 0.8507),
  mostly from MS_SQRT 0, which this task fixed.
- Timing: the harness geo time of identical files varies by 5-12% between runs (m11 vs m12); serial interleaved A/B
  (tools/ab.py) repeats within ~1%, so time decisions near the bar were checked both ways.
- Remaining gaps to the best known: heart k7 0.0018, mammo k5/k7/k10 (closed by m9-style clipped rounding in m23),
  ilpd k10, magic k4/k5 (pool 10 or SCREEN 2 / LASTSCREEN 3 close them at +2-7% time).
- Nothing written outside the run folder (uv cache in /data15/chandan/tabular/.uv-cache); scratch/ deleted. Tools
  added: tools/probe.py (beam supports, roundings, start ends on one problem), tools/batch.sh (dev runs vs a
  reference), tools/variant.sh (copy with knob defaults changed), tools/phases.py, tools/geoph.py (phase shares),
  tools/gatestat.py (polish gate statistics), tools/devfine.py; tools/jitcheck.py now reports compiles per problem.
