# oct04-fine6 notes

Start: slim.py = j30_final (oct03-fine5; same points as j18_poliso): regret -0.00125, geo time 0.011 s, test AUC 0.8507,
loss 0.33976. Recorded as `k1_base` (status keep, the reference of this run).

Keep rule: n_invalid 0; (regret down >= 0.0002 with time up <= 10%) or (time down >= 10% with regret up <= 0.0002);
discard if mean test AUC is more than 0.001 below the best kept row. AUC is never a reason to keep.
Per attempt: problems improved / worsened (|dloss| > 1e-6) vs k1_base (tools/cnt2.py).

Tools (copied from oct03-fine5, plus): tools/rec2.py (record any solver file as an attempt without touching slim.py),
tools/cnt2.py (improved / worsened counts, gap to the per-problem best of all h/i/j/k models), tools/ilsprof.py (ILS
time split: start runs, refit swaps, extra phase, polish), tools/pair2probe.py (exact pairs of the best single moves).
All new phases sit behind env knobs in slim.py (XMODE, XN, XF, ...; off = k1's points).

## Where k1 stands
- Gap to the per-problem best of every h/i/j model of the last three runs: 0.000145 in mean, in 18/70 problems, most of
  it in heart k10 (0.0026), mushroom k10 (0.0029), mushroom k7 (0.0012), mammo k10 (0.0010), ilpd k10, compas k10,
  breastcancer k7, magic k5. A loss keep (-0.0002) needs points better than any earlier run found.
- Serial time split (sum over 70 fits, 1.19 s): data 0.21, beam 0.49, rounding 0.03, ILS 0.46 (start runs 0.28 in
  185 runs, refit swaps 0.08 of which 0.056 are the 4 continuous refits, polish 0.09 on gated problems).
- Start probe (10 starts from a pool of 10, no patience, no refit swaps / polish; scratch): start ranks 5-9 improve the
  best end in 6 problem-starts in total (0.00004 in mean); ranks 1-4 in 27 (0.00079). The rounded-loss ratio of a
  start to the incumbent does not predict which starts improve (starts > 10% above the incumbent give the largest
  gains), so no bound-based skipping of starts.
- Scale / shift moves at the final points (round(w f), |w| -+ 1): improve 2/70 problems, 0.000006 in mean.
- Exact pairs of the 40 best single moves (incl. worsening ones) at the final points (tools/pair2probe.py): improve
  5/70 problems, 0.00001 in mean. Pair moves are exhausted too.

## Attempts
| # | MODEL_NAME | idea | mean_regret | geo_mean_time | mean_test_auc | improve / worsen | status / why |
|---|---|---|---|---|---|---|---|
| 1 | k1_base | j30_final rerun | -0.00125 | 0.011 | 0.8507 | - | keep (reference) |
| 2 | k2_addrefit | after the refit swaps: add one of the 4 best swap-in columns (k + 1 columns), continuous refit + calibrated rounding, drop the column whose exact removal costs least; local search from the best | -0.00119 | 0.013 (x1.21) | 0.8508 | 2 / 1 (breastcancer k7 -0.0011, spambase k3; mushroom k10 +0.0051 because the polish then starts elsewhere) | discard |
| 3 | k3_vardrop | after the refit swaps: drop every column of one variable (4 cheapest), refit + round, local search re-adds | -0.00125 | 0.014 (x1.22) | 0.8507 | 1 / 1 (mushroom k7 -0.0010, mushroom k10 +0.0007) | discard |
| 4 | k4_vartabu2 | after the refit swaps: drop every column of one variable (2 cheapest), refit + round, local search with that variable forbidden (own visited set), then a full local search from that end | -0.00126 | 0.015 (x1.39) | 0.8507 | 1 / 1 (mushroom k7 -0.0010; mushroom k10 +0.00007) | discard |
| 5 | k5_vartabu4 | k4 with 4 variables | -0.00127 | 0.017 (x1.54) | 0.8505 | 3 / 1 (mushroom k7, compas k10 -0.0004, ilpd k10 -0.0002) | discard |
| 6 | k6_km1start | one extra start: the best rounding of the beam's k-1 column parents (beam split into levels 1..k-2 and k-1 with identical steps), local search adds a column | -0.00120 | 0.013 (x1.17) | 0.8508 | 2 / 1 (magic k7, bank k3 tiny; mushroom k10 +0.0030 via the polish) | discard (2 starts: same points, x1.27) |
| 7 | k7_beam2 | a second beam (5 x 5) with the most important variable of the best point forbidden; local search from its 2 best roundings | -0.00121 | 0.019 (x1.58) | 0.8506 | 1 / 1 (ilpd k10 -0.0005; mushroom k10 +0.0031) | discard |
| (probe) | | refit-swap refits with tol 1e-5 (same points, x0.998); refit rounding on 10 scales (1 point differs, x0.988) | | | | | refits are not a time lever |
| (probe) | | beam split (objmode ticks, scratch): colsum 30-45% of the beam on large data, < 10% on small; child fits 25-50%; materialise 15-20%. Data setup: scan of X is memory-bound (magic 5.3 of 8.6 ms), codes 0.6-1.5 ms, gather 1.1 ms on adult / bank / mushroom | | | | | |
| 8 | k8_tabuwalk | tabu walk after the refit swaps: 3 forced best-estimate swaps (removed columns may not return), local search with them forbidden, then a full local search | -0.00125 | 0.014 (x1.23) | 0.8507 | 0 / 0 | discard: the walk always falls back to the same end |
| 9 | k9_shake | 4 deterministic shakes of the best point (every point -1/0/+1), local search from the best (2 fails) | -0.00124 | 0.013 (x1.12) | 0.8508 | 3 / 1 (spambase k3, compas k4, magic k7 tiny; breastcancer k10 +0.0004) | discard |
| 10 | k10_pairkickrf | pair-drop kicks with a continuous refit of the rest (2 cheapest exact joint removals), local search re-adds | -0.00123 | 0.016 (x1.40) | 0.8509 | 2 / 1 (heart k10 -0.0012, mushroom k7 -0.0010; mushroom k10 +0.0031 via the polish) | discard |
| 11 | k11_pairs | pair moves after the refit swaps: the 12 best swaps by the estimate and every +-1 value change, every compatible pair scored by exact calibrated loss; local search from the best pair | -0.00132 | 0.018 (x1.60) | 0.8507 | 5 / 0 (mushroom k10 -0.0029, mushroom k7 -0.0012, ionosphere k10 -0.0008, mammo k10 -0.0002, magic k7) | discard: -0.00007, concentrated in near-separable problems, +60% time |
| 12 | k12_pairs8 | k11 with pairs only among the 8 single moves of lowest exact loss | -0.00126 | 0.013 (x1.18) | 0.8507 | 1 / 0 | discard: the useful pairs are made of individually bad moves |
| 13 | k13_pairsgate | k11 only when the logit range of the best point is >= 10 (polish gate) | -0.00132 | 0.015 (x1.36; serial A/B x1.23) | 0.8507 | 4 / 0 | discard: -0.00007 for +23% |
| 14 | k14_spat1 | start patience 1 | -0.00109 | 0.011 recorded (machine load 50 from other users); serial A/B x0.909 | 0.8512 | 1 / 7 (spambase k3 +0.0032, mushroom k5 +0.0034, heart k10 +0.0022, spambase k5 +0.0019, ...) | discard: +0.00015 loss spread over 7 problems for -9% time (serial); not -10% in the harness |
| 15 | k15_spat1ls3 | k14 + LASTSCREEN 3 | -0.00108 | 0.012 recorded (load 50); serial A/B x0.894 | 0.8512 | 1 / 8 | discard: by serial A/B a letter-of-the-rule time keep (-10.6%, +0.00017), but the recorded time does not show it, and it gives back most of j18's loss gain |
| 16 | k16_spat1rnm10 | k14 + start roundings on 10 scales | -0.00109 | 0.013 recorded (load 50); serial A/B x0.903 | 0.8512 | 1 / 9 | discard |
| 17 | k17_pol1 | one polish round instead of 3 | -0.00116 | 0.012 (load 50); serial A/B x0.976 | 0.8509 | 0 / 3 | discard |
| 18 | k18_pairsafter | k11's pair moves after the polish, from k1's final point (monotone; polish again only after an improvement) | -0.00127 | 0.019 (x1.69, load ~50) | 0.8507 | 4 / 0 (ionosphere k10 -0.0008, mushroom k10 -0.0007, mammo k10 -0.0002, magic k7) | discard: -0.00003; the k11 gains on mushroom k7 came from the polish chaos, not the pairs |
| 19 | k19_pairsaftg | k18 gated to logit range >= 10 | -0.00127 | 0.016 (x1.42) | 0.8507 | 3 / 0 | discard |
| 20 | k20_vtabuafter | k5's variable-tabu restarts (4 variables) after the polish (monotone) | -0.00127 | 0.019 (x1.67) | 0.8505 | 3 / 0 (mushroom k7 -0.0012, compas k10 -0.0004, ilpd k10 -0.0002) | discard |
| 21 | k21_beam2after | k7's second beam after the polish (monotone) | -0.00125 | 0.020 (x1.81) | 0.8506 | 1 / 0 (ilpd k10 -0.0005) | discard |
| 22 | k22_pairsvtabu | k18 then k20 (pairs, then variable-tabu restarts), both after the polish | -0.00130 | 0.024 (x2.13) | 0.8505 | 7 / 0 (the union of k18 and k20) | discard: -0.00005 for twice the time |
| 23 | k23_long | long-search reference: final pool 20, 20 starts, no patience, refit swaps until 3 fail, pair moves (2 fails) | -0.00126 | 0.045 (x4.0) | 0.8512 | 14 / 4 (gains: mushroom k7 -0.0037, heart k10 -0.0025, ionosphere k10, ilpd k10, compas k10/k7, spambase k3, fico k10 and 6 tiny; losses: ionosphere k7 +0.0046, mushroom k10 +0.0028, magic k10 +0.0003) | discard: at 4x time the net is -0.00001; the many small gains are real headroom, the large losses are the polish's dependence on its start point |
| 24 | k24_pool10 | final beam pool of 10 (starts = the 5 best roundings of 10 supports) | -0.00108 | 0.012 | 0.8505 | 1 / 1 (ionosphere k10 +0.0125) | discard: the 5 best roundings are almost always the first 5 supports; one chaotic loss |
| 25 | k25_sortrows | unique rows in input order (sequential gathers) instead of hash order | -0.00125 | 0.013; serial A/B x1.023 | 0.8507 | 0 / 0 | discard: slower; hash order groups similar rows |
| 26 | k26_spat1pairs | k15 + gated pair moves after the polish | -0.00110 | 0.013 | 0.8511 | 5 / 7 | discard |
| 27 | k27_speps3 | start patience counts improvements below 0.3% as fails (SPEPS 0.003) | -0.00125 | 0.012 (load); serial A/B x0.987 | 0.8510 | 0 / 1 (tiny) | discard: no time change |
| 28 | k28_spat1ls3pol1 | k15 + one polish round | -0.00099 | 0.010 (x0.87) | 0.8513 | 1 / 11 | discard: regret +0.00026 (over the 0.0002 bar) |
| 29 | k29_pairsaftg6 | k19 with 6 swaps in the pair set | -0.00127 | 0.013 (x1.21, load 14) | 0.8507 | 3 / 0 | discard |
| 30 | k30_final | k1 (same points, checked by serial A/B: 0/70 differ) with all oct04-fine6 knobs present but off and the docstring updated; final slim.py | -0.00125 | 0.012 | 0.8507 | 0 / 0 | final file (same solver as k1) |

## Summary
- No keep. Best version stays `k1_base` = `k30_final` = slim.py (= j30_final / j18_poliso points): regret -0.00125,
  loss 0.33976, geo time 0.011-0.012 s, test AUC 0.8507.
- 29 changes tried. Every diversity phase added after the starts and refit swaps (add-refit-drop, variable drops,
  variable-tabu restarts, k-1 parent starts, a second beam without the top variable, tabu walk, shakes, pair-drop refit
  kicks, exact pair moves) improves 0-5 of the 70 problems. Run before the polish, each also loses on mushroom k10
  or ionosphere k7, because the polish ends somewhere else when its start point changes. Run after the polish (XAFTER:
  monotone, nothing can worsen), the best single phase (exact pairs, k18) improves 4 problems for -0.00003 at +40-70%
  time. All of them combined (k22) improve 7 problems for -0.00005 at 2x time.
- A 4x long search (k23) improves 14 problems by small amounts but loses 2 near-separable ones to polish chaos (net
  -0.00001). Small headroom exists, spread over many problems, but this search family reaches it only at several
  times the cost.
- Time: every loss-neutral item measured <= 2% (refit tolerance and grid, rounding grid, row order, SPEPS). The only
  route to -10% is less search (start patience 1 + 15 last-level fits: serial x0.89 for +0.00017 loss over 8 problems,
  k15). By serial A/B that meets the rule, but the recorded time (machine load 50 from other users) did not show it,
  and it gives back most of j18's loss gain, so it was not kept.
- Most informative: k8 (tabu walk: 0 changes), k12 vs k11 (useful pair moves are made of two individually worsening
  moves, only in near-separable fits), k18 vs k11 (with the polish chaos removed, pairs give only -0.00003),
  k23 (4x search: 14 problems improve, net about 0), k15/k28 (time is only bought with loss).
- Headroom: on visible_fine the end points are robust to every perturbation family tried. Gap to the per-problem best
  of every h-k model: 0.000205 (19 problems), concentrated in heart k10, mushroom k7/k10, ionosphere, mammo k10,
  ilpd k10. I do not expect another keep on this track from search changes. The visible metric is dominated by
  chaotic outcomes in 2-3 near-separable problems.
- Tools kept: tools/rec2.py, cnt2.py, ilsprof.py, pair2probe.py, startprobe.py (start-rank probe), scaleprobe.py.
  scratch/ deleted. Nothing written outside the run folder (uv cache in /data15/chandan/tabular/.uv-cache).
