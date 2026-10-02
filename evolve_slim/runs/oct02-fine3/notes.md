# oct02-fine3 notes

**Result.** Best kept: `h8_fastbeam` (slim_lib/h8_fastbeam.py, = slim.py): mean_regret -0.00102, geo time 0.023 s
(reruns 0.024), test AUC 0.8510, mean loss 0.33998. Start g24_fastcol: -0.00118 / 0.032 s / 0.8484 / 0.33982.
v35_scratch: +0.00019 / 0.068 s / 0.8496 / 0.34119. FasterRisk: +0.00181 / 2.16 s / 0.8483 / 0.34282.
So against v35: loss -0.00121, AUC +0.0014, 3x faster; against g24: AUC +0.0026 for +0.00016 loss, 25% faster.
`h9_noband` (no band rule) is better on both (-0.00107, AUC 0.8515) but missed the 0.0002 regret bar.
`slim_lib/dev_knobs.py` is h8 with all probe knobs (env vars) used in the probes below; not an attempt.

What mattered: (1) a minimum support of sqrt(N) weighted rows on each side of every indicator (AUC +0.0023 for
+0.00035 regret; stronger supports buy AUC on haberman/ionosphere only at a large loss); (2) numba ILS / beam
kernels (time -45% with identical points), which paid for (3) two pair kicks ranked by exact joint-removal loss
with additions screened to 16 columns (-0.0002 regret, AUC +0.0004). Under the support rule, lower loss from
search no longer costs AUC until the long-search regime (heart k10).

Start: slim.py = g24_fastcol (sep27-fine2): regret -0.00118, 0.032 s, test AUC 0.8484. v35: 0.00019 / 0.068 s / 0.8496;
FasterRisk 0.00181 / 2.16 s / 0.8483.

Keep rule of this run: n_invalid == 0, test AUC no more than 0.001 below the best kept row, and (regret down >= 0.0002
with time up <= 10%, or time down >= 10% with regret up <= 0.0002); or test AUC up >= 0.002 with regret up <= 0.0005.

Tools: tools/dev.py (full suite, no record, rows to scratch/<tag>.csv), tools/sweep.sh (parallel env-var configs),
tools/cmp.py (per-dataset AUC), tools/status.py. Machine load 40-170 on 176 cores: times are noisy (g24 rerun: 0.030 to 0.053 s).

Per-dataset view of g24 vs v35: the AUC gap (-0.0012) comes from ilpd (-0.027 mean over k), australian, breastcancer;
the extra loss reduction there comes from narrow bands (e.g. heart k10 59 < age <= 63 with +3/-3, ilpd V4 1 < x <= 2.2).

| # | MODEL_NAME | idea | mean_regret | geo_mean_time | mean_test_auc | status / why |
|---|---|---|---|---|---|---|
| 0 | g24_fastcol | start (rerun) | -0.00118 | 0.030 | 0.8484 | start |
| (sweep) | MS_* | min support m = c sqrt(N) (or f N) on both sides of every indicator and between thresholds of one variable in the support. c=0.5: -0.00081 / 0.8499; c=1: -0.00083 / 0.8507; c=1.5: +0.00172 / 0.8553; c=2: +0.00373 / 0.8585; c=3: +0.00713 / 0.8596; f=0.02: -0.00066 / 0.8503; f=0.05: +0.00370 / 0.8487. c=2 ends only: +0.00321 / 0.8581; c=2 band only: -0.00115 / 0.8492 | | | | the end (indicator) support carries the effect; above c=1 the AUC gain is mostly haberman (0.61 -> 0.70, n=244) and ionosphere, at a large loss |
| 1 | h2_minsup | min support c=1 (ends and bands) | -0.00083 | 0.089 (load avg 169) | 0.8507 | keep (AUC rule: +0.0023 AUC, regret +0.00035) |
| (sweep) | MS_* | c=1 without the band rule: -0.00087 / 0.8511; c=0.75: -0.00089 / 0.8509; c=1.25: +0.00007 / 0.8513; band 0.5: -0.00085 / 0.8507; rule only on chains of >= 2 columns: c=1 identical, c=1.5 +0.00071 / 0.8516, c=2 +0.00163 / 0.8522 | | | | band rule neutral; kept c=1 with bands |
| (probe) | g25 pair moves ported (PAIRS=1) | -0.00095 | 0.045 | 0.8508 | -0.00012 only at ~+10% time |
| (probe) | width: PARENT 20 -0.00082 / 0.8499; NSTARTS 10 -0.00088 / 0.8500; CHILD 20 same; FINAL_POOL 20 + NSTARTS 10 -0.00096 / 0.8500; PARENT 6 +0.00041; PARENT 7 + pairs +0.00014 | | | | width beyond the defaults buys little; less beam hurts a lot |
| (probe) | kicks (drop 1-2 random support columns of the best, ILS again, stop after P fails): P=5 -0.00086 / 0.8510; P=25 -0.00105 / 0.8512 at 2.3x time; P=25 + pairs -0.00126 / 0.8512 at 2.4x; P=25 without min support -0.00153 / 0.8498 | | | | with min support, a lower loss now also gives a slightly higher AUC; but kicks cost 2.3x time |
| (probe) | starts: NSTARTS 1 -0.00006 (0.023 s), 2 -0.00040, 3 -0.00042, 5 -0.00083; NSTARTS 3 + 8 kicks + pairs -0.00110 at 1.7x | | | | ILS starts from different roundings are worth more than kicks per second |
| (probe) | ILS acceptance needs >= delta nats of total log-likelihood for support changes: delta 0.5 -0.00073 / 0.8505; 1 -0.00052 / 0.8498; 2 -0.00035 / 0.8491; delta 1 for all moves -0.00045 / 0.8496 | | | | an AIC-like acceptance threshold lowers both loss and AUC: discard |
| (probe) | beam knobs: SIGMAX 2 -0.00054; NSCREEN 8 -0.00091; NEXACT 3 -0.00083; SCREEN 1 / 1.5 / 2.5 / 3: -0.00023 / -0.00083 / -0.00029 / -0.00033; FINAL_POOL 15 -0.00068 | | | | regret is chaotic in the beam settings (+-0.0003), almost all of it from ionosphere (n=280, d=2365: -9.6e-4 of the -8.3e-4 mean), then mushroom and breastcancer |
| (probe) | kicks gated on the number of distinct start optima (>= 3/4/5): -0.00092 / -0.00089 / -0.00089 at 0.082 / 0.065 / 0.050 s | | | | gating loses the kick gains (mushroom has few distinct optima) |
| 2 | h3_fastforbid | h2 with the min-support mask in numba, value-change candidates vectorised (same points) | -0.00083 | 0.031 (reruns h2 0.034 / 0.037, h3 0.033 / 0.031) | 0.8507 | keep: identical points, ~-9% time (recorded h2 time 0.089 was taken at load avg 169) |
| (probe) | beam picks each variable's threshold by its Newton gain smoothed over neighbouring levels (mean / min of 3) | +0.00012 / +0.00124 | 0.042 | 0.8507 / 0.8509 | discard: higher loss, same AUC |
| (probe) | lazy starts: value/addition descent from 10 (or 5) starts, full ILS from the best 2-5 ends | -0.00056 to -0.00076 | 0.035-0.047 | 0.8507 | discard: the swap-phase paths matter |
| (probe) | coarse beam (every 3rd/5th/10th threshold of a chain), full ILS | -0.00004 / +0.00078 / +0.00061 | 0.05 | 0.8504 / 0.8498 | discard |
| (probe) | ILS starts ordered by the continuous beam loss instead of the rounded loss | -0.00084 | 0.032 | 0.8507 | no change |
| (probe) | absolute floor on the support: max(sqrt(N), A) with A = 20 / 25 / 30 / 40 | -0.00078 / +0.00203 / +0.00279 / +0.00399 | | 0.8507 / 0.8570 / 0.8585 / 0.8584 | the AUC gain beyond c=1 is haberman (0.62 -> 0.70) and ionosphere at a large loss |
| (probe) | threshold resolution: within a chain keep thresholds >= g sqrt(N) rows apart, g = 0.25 / 0.5 / 1 | +0.00137 / +0.00319 / +0.00238 | | 0.8500 / 0.8510 / 0.8510 | discard: loss up (ionosphere), AUC flat |
| (probe) | kicks: single-column drops enumerated once (patience all / 3); random 1-2 drops patience 10 | -0.00088 / -0.00087 / -0.00101 | 0.052 / 0.045 / 0.077 | 0.8512 | the gain of kicks is mostly one problem (mushroom k7, -152e-4, needs two drops) |
| 3 | h4_nbils | h3 with the ILS iteration in numba kernels (value changes + additions, the whole swap phase, the exact check) | -0.00083 | 0.047 (load avg 124-139); dev run at load 16: 0.026 | 0.8507 | keep: identical points, serial A/B time ratio vs h3 0.835 |
| (probe) | g25 pair moves on h4 (PAIRS=1); random kicks P=10; pairs + kicks P=10 / 25 | -0.00095 / -0.00110 / -0.00126 / -0.00126 | 0.034 / 0.061 / 0.067 / 0.086 (parallel sweep) | 0.8509 / 0.8512 / 0.8514 / 0.8514 | lower loss and higher AUC, but too slow |
| (probe) | kicks over pairs of support columns, cheapest combined removal first (KMODE 2), P = 3 / 6 / 12 | -0.00103 / -0.00107 / -0.00108 | 0.042 / 0.052 / 0.067 | 0.8510 | pair kicks: most of the gain in 3 kicks |
| (probe) | ILS additions screened to the m best columns by a second-order model (m = 8 / 16 / 32 / 64) | -0.00083 | same | 0.8507 | starts are full supports, so additions only matter in kicks; with kicks, m=16 cuts their cost by ~40% |
| (speed) | beam: sigmoids per row group; Newton tol 1e-7 -> 1e-4 (1e-5 / 1e-6 / 1e-4: -0.00082 / -0.00083 / -0.00084); SCREEN 2 -> 1.5 (15 exact child fits per level; 1.7: -0.00084); swap phase keeps its removal states (typed List) | -0.00083 | serial A/B vs h4 0.921 | 0.8507 | carried forward |
| 4 | h5_pairkicks | h4 + speed items above + 3 pair kicks + additions screened to 16 | -0.00107 | 0.033 (h4 rerun 0.027, h5 0.032 / 0.034) | 0.8509 | discard: regret -0.00024 but harness time +22%. (A first serial A/B said 0.89: invalid, the KICKS env var also switched kicks on in the h4 snapshot; tools/variant.py now writes variant files instead of env vars) |
| (probe) | kick descents skip the swap phase while above best x (1 + q): q = 0 / 0.001 / 0.005, P=3; P=6 q=0.001; P=2 q=0.001; NSTARTS 4 + P 3/4; NSTARTS 3 + P 5; LASTSCREEN 1 / 2 | -0.00102 / -0.00102 / -0.00103 / -0.00102 / -0.00102; -0.00060; -0.00048; -0.00059 / -0.00084 | | 0.8510-0.8512 | fewer starts or fewer last-level fits lose more than kicks gain |
| 5 | h6_kick2 | h4 + speed items + 2 pair kicks (swap phases within 0.1% of the best) + additions screened to 16 | -0.00102 (exact diff vs h4 -0.000197) | 0.033 (load 13-30; dev reruns h4 0.062 / h6 0.059 under a load spike); serial A/B 1.045 | 0.8510 | discard by the letter of the rule: regret down 0.000197 < 0.0002 (mushroom k7 -115e-4, australian k7 -12.5e-4, ionosphere k7 -13e-4; adult k10 +8.2e-4 from the beam changes) |
| (speed) | the whole ILS run (visited set as a numba Dict keyed by a random projection of w) in one numba kernel | same points | serial A/B vs h6 0.936 | | carried forward |
| (probe) | pair moves (numba) on top of kicks | -0.00102 | A/B x1.04 | 0.8512 | discard: pairs overlap with kicks |
| (probe) | kick log with 12 kicks: successes at fail index 0-1 (mushroom k7, australian k7, ionosphere k7, magic k4) and 7-11 (adult k5, heart k10, magic k10) | | | | 2 kicks catch the cheap wins |
| 6 | h7_kickexact | h6 with the kick pairs ranked by the exact loss after removing both (was: sum of single removal losses), ILS in one kernel | -0.00103 (exact diff vs h4 -0.000205) | 0.026 (h4 reruns 0.027); serial A/B vs h4 1.020 | 0.8511 | keep: regret -0.000205, time +2%, AUC +0.0004 |
| (probe) | support scale c = 0.9 / 1.1 / 1.2 / 1.3 on h7; no band rule | -0.00103 / -0.00095 / +0.00019 / +0.00048; -0.00108 | | 0.8511 / 0.8506 / 0.8520 / 0.8518; 0.8515 | c=1 stays; dropping the band rule is slightly better on both (-0.00005, +0.0004) |
| (probe) | kicks P = 4 / 8 / 12; P=8 without band rule | -0.00103 / -0.00107 / -0.00108; -0.00112 | 0.033-0.038 (parallel) | 0.8510 / 0.8506 / 0.8506; 0.8510 | saturates; the extra wins (heart k10) lower AUC |
| (probe) | long search reference: PARENT 20, CHILD 20, pool 20, 20 starts, 30 kicks (and 30/20/30/30/60, no band) | -0.00123 / -0.00131 | 0.073 / 0.096 | 0.8502 / 0.8495 | headroom under the support rule is ~0.0002-0.0003, and the extra loss reduction lowers AUC again |
| (probe) | bump structures in h7 points: 13/70 solutions have a variable with thresholds of opposite signs (fico's MSinceMostRecentInq in every k, likely its special negative codes); a monotone rule would be wrong there | | | | not pursued |
| 7 | h8_fastbeam | h7 with beam proposals + signature cut in numba (support hashes as uint64 sums), regrouping of kept children in numba, kick pair ranking in numba, rounding calibrated on distinct score values | -0.00102 | 0.023 (reruns h8 0.024 / 0.024, h7 0.027 / 0.029); serial A/B 0.899 | 0.8510 | keep: time -11% to -17%, regret +0.00001 (rounding ties) |
| (probe) | on h8: no band rule (nb) -0.00107 / 0.8515; nb + kicks 4 / 6 / 8 -0.00107 / -0.00107 / -0.00111; nb + 7 starts -0.00108; nb + kicks 8 + 7 starts -0.00112; nb + kicks 12 + 7 starts -0.00112; NSTARTS 7 -0.00103 | | 0.030-0.040 (parallel) | 0.8510-0.8515 | search saturated near -0.0011; the long-search headroom is heart (AUC 0.817 -> 0.791) and ionosphere |
| (probe) | class-aware support: msup / (4 p (1-p))^e, e = 1 / 0.5; e=1 without bands; e=1 with c=0.8 | -0.00092 / -0.00096 / -0.00100 / -0.00101 | | 0.8515 / 0.8511 / 0.8515 / 0.8510 | no better trade than c alone |
| (probe) | tie-break among local optima within eps of the best: largest smallest indicator side (eps 5e-4 / 2e-3 / 5e-3) or smallest sum of |points| (2e-3 / 5e-3) | -0.00101 / -0.00099 / -0.00089; -0.00091 / -0.00076 | | 0.8510 / 0.8511 / 0.8515; 0.8512 / 0.8518 | discard: +0.0008 AUC at most, for +0.00026 regret |
| (probe) | beam Newton tol 1e-3 / 3e-3 / 1e-2 | -0.00102 / -0.00100 / -0.00082 | ~same | 0.8511 / 0.8507 / 0.8516 | no time gain: the fits' cost is building the cell Hessians, not iterations |
| 8 | h9_noband | h8 without the band rule (indicator support only) | -0.00107 | 0.024 | 0.8515 | discard by the rule (regret -0.00005, AUC +0.0005, same time), though simpler and better on both; noted for the final choice |
| (probe) | kicks drop the best triple instead of pair (exact joint removal); triples with 4 kicks | -0.00099 / -0.00099 | | 0.8508 | discard |
| (probe) | rounding grid of 10 / 30 / 40 scales (20 now) | -0.00100 / -0.00087 / -0.00089 | | 0.8511 / 0.8510 / 0.8511 | finer grids give other starts and worse ends (chaotic) |
| (probe) | kicks also from the best end with another support (and with gate 0.3%) | -0.00102 / -0.00102 | +7% | 0.8511 | no gain |
| (probe) | swap screen picks at most one column per variable (as the beam does), n_screen 4 / 6 / 3; with no band | -0.00098 / -0.00098 / -0.00095; -0.00103 | | 0.8508; 0.8513 | discard: neighbouring thresholds are the useful swap-ins |
