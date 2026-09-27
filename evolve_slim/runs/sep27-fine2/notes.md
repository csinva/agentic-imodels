# sep27-fine2 notes

Baseline fine_v1 (v35 unchanged): mean_regret 0.00019, geo_mean_time 0.067 s, test AUC 0.8496.
FasterRisk: 0.00181, 2.156 s, AUC 0.8483. best_known on this suite is mostly fine_v1's own loss, so
beating it means negative regret.

Profile of fine_v1 (magic, n=10000, d=990): data 0.275 s (transpose of X 0.11 s, per-column CSC codes 0.09 s),
beam 0.04-0.18 s, ILS 0.10-0.73 s (every eval touches the nonzeros of every column: n*d/2 per step).

## Attempts

| MODEL_NAME | idea | mean_regret | geo_mean_time | test AUC | status / why |
|---|---|---|---|---|---|
| fine_v1 | baseline | 0.00019 | 0.067 | 0.8496 | baseline |
| g2_chains | chain representation: nested binary columns found by bitset subset tests, one ordinal code per chain, gradients / move cells from per-code histograms with suffix sums; same search | 0.00019 | 0.039 | 0.8496 | keep: identical losses, -42% time (magic k=3 0.43->0.06 s) |
| g3_pervar | beam children: the 10 best *variables* by gradient, each at its best threshold (was: 10 best columns, mostly neighbouring thresholds of one variable) | -0.00091 | 0.050 | 0.8482 | discard by rule (time +28%); but the key idea: k=7,10 regret +0.41e-3 -> -1.29e-3. AUC -0.0014 |
| g4_screen | g3 + children screened by a 2-D fit (intercept + new coefficient), full refit of best 2x beam; one colsum for swap screens | -0.00096 | 0.053 | 0.8486 | discard (time); screening saves ILS time but not beam time |
| g5_pervar_fast | g4 + per-variable pick in numba, loops over rows with nonzero code only | -0.00096 | 0.046 / 0.051 rerun | 0.8486 | discard: g2 rerun under the same load gave 0.045 (serial A/B ratio 1.07) |
| g6_fastdata | g5 + vectorised column scan (bitsets word-major), chain codes by binary search over nested bitsets (data stage magic 40 -> 20 ms) | -0.00096 | 0.043 | 0.8486 | keep: vs g2 (0.045 on rerun, serial A/B ratio 1.02), regret -0.00115 |

Timing note: the machine load varies (load avg 20-33 on 176 cores), so the harness geo time of the same file moved
0.039 -> 0.045 (g2). tools/ab.py runs two files interleaved serially and gives a steadier time ratio.
| g7_slide | g6 + exact threshold slides (all other thresholds of each support column's variable, every value) in the swap phase | (A/B) -0.00070 | x1.23 | 0.8486 | discard: worse loss and slower; best-improvement trajectories change for the worse |
| (probe) | beam width ablation on g6 (k=3,5,10): parent 30 gives -1.57e-3 at 1.6x time; child 30 alone hurts (-0.54); bigger final pool (20, 40 supports rounded) no gain; wide 30/30/30 -1.92e-3 at 4x | | | | beam width on the path matters, not the final pool |
| g8_newtonscreen | beam proposals scored by a one-step Newton bound on the new coefficient for every column and parent (3 x P histogram columns per level); per parent the 10 best variables; only the best 2 x beam proposals fitted exactly (was: a 2-D fit per child, 48 us each, dominated by a row pass per child) | -0.00098 | 0.038 | 0.8480 | keep: -12% time vs g6 |
| g9_extra | g8 + the 5 best columns overall added to each parent's proposals | (A/B) +0.00030 | x0.95 | 0.8489 | discard: near-duplicate thresholds flood the global cut |
| g9b_sig_extra | g10 + 10 extra best columns per parent | (A/B) -0.00107 | x1.20 | 0.8489 | discard |
| g10_sigdiv | g8 + at most one proposal and one fitted child per multiset of variables (supports that differ only in thresholds count once) | -0.00118 | 0.039 | 0.8484 | keep: -0.00020 regret, +3% time |
| g11_ilsknobs | (A/B only) NSCREEN 8: -0.00122 x1.04; NEXACT 4: -0.00119 x1.03 | | | | discard: ILS width barely matters |
| (probe) | g10 knob sweeps: PARENT 20 -0.00135 at x1.19 (A/B); NSTARTS 3 -0.00070 at x0.77; NSTARTS 3 + PARENT 20 -0.00088 at x0.94 | | | | ILS starts matter; the beam width helps but costs |
| g12_fastils | g10 with beam row weights and Newton gains in numba (x^2 block skipped when all columns are binary); ILS swap-phase removal states and second-order screen in numba (the swap screen was 391 of 830 ms of ILS time in a profile) | -0.00118 | 0.038 / 0.037 rerun (g10 rerun 0.040) | 0.8484 | keep: identical losses, serial A/B time ratio 0.855; harness under load avg 65 shows -7.5% |
| g13_lean | g12 + swap screen returns only picked columns; signatures built incrementally, only for fitted proposals | -0.00118 | 0.035 | 0.8484 | discard by rule (-5.4% harness, serial A/B 0.919); same search, code carried into the next attempts |
| g14_polish | g13 + final polish of the best solution: ILS with exact threshold slides added to the swap phase | (A/B) -0.00119 | x1.09 | 0.8484 | discard: local optima are already nearly slide-optimal |
| g15_kicks | g13 + 2 kicks: drop the least useful feature of the best solution and rerun ILS | (A/B) -0.00124 | x1.20 | 0.8495 | discard: small gain for the time |
| g16_firstimp | g13 + first-improvement swap phase (removals from least useful, stop at first improving) | (A/B) -0.00120 | x1.10 | 0.8483 | discard: more, smaller steps; slower |
| g17_nval3 | g13 + swap-ins evaluated at their 3 best values by the second-order model only | (A/B) -0.00108 | x1.04 | 0.8483 | discard: worse and not faster (per-column candidate picks) |
| g18_warmcal | g13 + warm-started calibration across the rounding scale grid | (A/B) -0.00113 | x0.99 | 0.8483 | discard: no speed gain; tiny numeric changes move ILS paths |
| g19_leaner | g13 + starts deduplicated by score vector, partial-selection swap screen, per-kind buffers | -0.00118 | 0.030 / 0.032 rerun (g12 rerun 0.034) | 0.8484 | discard by rule (-6% on rerun; serial A/B vs g13 0.98, vs g12 ~0.90); same search, code carried forward |
| g20_swapr | g19 + swap phase over the 3 least useful features first, the rest only when those give no improving swap | (A/B) -0.00122 | x1.00 | 0.8488 | discard: no time saving (each run still ends with a full swap phase) |
| g21_perpar | g19 + at most 3 kept children per parent in the beam | (A/B) +0.00021 | x0.99 | 0.8471 | discard: much worse; strong parents need many children |
| g22_sigmax2 | g19 + two children per variable multiset | (A/B) -0.00094 | x0.94 | 0.8486 | discard |
| g23_sigset | g19 + signature as a set of variables (repeated variables collapse) | (A/B) -0.00108 | x1.02 | 0.8485 | discard |
| g24_fastcol | g19 + chain columns as comparisons, scores in numba, chain codes by branchless binary search on per-word bits, beam coefficients warm-started at the Newton step | -0.00118 | 0.042 / 0.032 / 0.030 (reruns; g12 rerun 0.034) | 0.8484 | keep: same search; serial A/B chain g12->g24 ~0.87; harness is very noisy (load avg 50-140) |
| g25_pairs | g24 + pair moves (two support points +-1 each, exact on rows grouped by (x_S, y)) at every ILS local optimum | -0.00138 | 0.036 (g24 rerun 0.032) | 0.8483 | discard by rule (+12.5% time) though -0.00020 regret; 61/69 accepted pairs move both magnitudes the same way. Variants: pairs only on the final best -0.00134 x1.08; on best 2 optima -0.00134; magnitude-coherent only -0.00133 x1.10; plus star-ray rescale moves no gain |
| g26_sharedscreen | g25 + one swap-in screen (8 columns) at the current state shared by all removals | (A/B) -0.00052 | x0.85 | 0.8479 | discard: removal-specific screens matter |
| g27_incscreen | g25 + per-removal screen from current per-code sums plus only the rows the removal changes | (A/B) -0.00138 | x1.17 | 0.8483 | discard: exact same search but slower (row-major scatter over all variables) |
