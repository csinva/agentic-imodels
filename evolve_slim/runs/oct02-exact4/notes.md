# oct02-exact4 notes (exact track: certify optimal sparse integer risk scores)

Start: y23_giveup3 from runs/sep28-exact3 (55/70 certified, 60/60 tiny, 0 wrong, geo 0.332 s). Proofs P1-P9 are in
runs/sep27-exact2/notes.md, P10-P25 in runs/sep28-exact3/notes.md; they are reused unchanged. New proofs start at
P26. Notation as there: subcell counts (p, n), L(p, n; x) = p sp(-x) + n sp(x), h(a, b) = a log((a+b)/a) +
b log((a+b)/b) (the saturated loss of real counts).

## Diagnosis at the start (dev driver tools/dev.py, 6 problems in parallel, no give-up)

Where the time goes (seconds out of 116, PROFILE counters): australian 7: K2 33, leaf LR 37, family 35;
heart 10: K2 7, leaf LR 83, family 18; ilpd 7: K2 78, leaf LR 6, family 11; magic 10: K2 29, leaf LR 10,
family 45; fico 7: family 43, the rest is the P1 leaf pass (143M leaves); bank 10: leaf LR 90, family 20.
Full runs (300 s, no give-up): australian 7 certifies in 116 s; heart 10 ~45% done; ionosphere 5 ~50% done
(5.7e9 leaves at 19.5M/s); ilpd 7 finds the best known loss 0.48346504 itself but is ~10% done; fico 7 < 1%;
spambase 5 ~50% (9.5k supports survive the LR and need the integer B&B).
With the best known loss given as ub (tools/dev_ub.py, 600 s): ilpd 7 ~23% done, ilpd 10 / bank 10 / magic 10
far below 1%: a better incumbent alone does not make them tractable.

Margins of the leaves the leaf LR discards (LR value - ub, nats): australian 7: 2.11M of 2.16M are >= 8 nats;
heart 10: 1.45M of 1.50M >= 8; ilpd 7: 400k of 488k >= 8. The leaves that reach the LR are not close calls: the
K relaxations before it are simply too weak there (K2 discards 17% of the P1 survivors on heart 10).

## Cardinality-aware family bounds: what was measured offline (tools/an1-an4.py, heart 10, random nodes T
built from the 25 best single columns, all other columns as candidates)

- LR(T + all candidates) is 0 (the 57 columns separate the 216 rows): the current family bound cannot work.
- The best leaf below such a node (min over the 1 or 2 added columns of the exact LR) is 8-30 nats above ub at
  every sampled node, so a tight cardinality-aware family bound would discard them all.
- chi^2 bound gain(A) <= r_A^T S_A^-1 r_A (Newton-projected dual from LR(T), KL <= chi^2): valid only when the
  projected dual stays in [0, 1]; ignoring that, it closes 40/40 nodes at m = 1 and 10/40 at m = 2; with the box
  check 2/40 and 0/40. Not usable.
- Scale bound a <= ub / E_min (E_min = LP lower bound of the perceptron loss of the integer score): E_min = 0 on
  heart (real weights with |w| <= 5 separate), so no bound on the scale. Not usable.
- Pivot cone (a T column p that every owned vector uses, P18): theta = a w gives |theta_j| <= 5 |theta_p|,
  sum over candidates |theta_j| <= 5 m |theta_p|; the dual condition is
  g_p >= 5 sum_{T \ p} |g_q| + 5 (sum of the m largest |g_j| over the candidates). With theta_p >= 0 only it
  closed 12/12 (m = 1), 6/12 (m = 2), 5/12 (m = 3); with both signs of theta_p (required: a can be negative)
  2/12, 0/12, 0/12. With ALL signs of T fixed to the LR(T) signs (a lower bound would need all 2^(t-1) sign
  patterns) 11/12, 1/12, 0/12. Reason: {theta : <= m nonzero candidates, |theta_j| <= 5a} has convex hull
  {||theta||_inf <= 5a, ||theta||_1 <= 5 m a}, which is exactly what the cone uses, so no convex relaxation in
  theta alone can be tighter; the loss of ~20 nats at m = 2 is the gap between min over the hull and min over
  the sparse set. Not used.

## New proofs

P26 (r-shift relaxation K_r, generalises P21 (r = 1) and P24 (r = 2)). Let S = T'' + R with R the last r columns
of the support (all columns two-valued). For any point vector w and (a, b), a row in the T'' cell c whose R
columns are at bits beta (beta_k = 1 iff column k of R is at its larger value v1_k) gets the logit
a w_T''.x_c + b + sum_k a w_k (v0_k + beta_k (v1_k - v0_k)) = eta_c + sum_k beta_k delta_k with
eta_c = a w_T''.x_c + b + sum_k a w_k v0_k and delta_k = a w_k (v1_k - v0_k). Hence the minimum over S is
>= K_r = inf over (eta per T'' cell, delta in R^r) of sum over subcells (c, beta) of L(p, n; eta_c + beta.delta).
Cells holding a single class are dropped (nonnegative terms, P7). Dual: for p, n >= 0 and -p <= mu <= n,
min_x [L(p, n; x) - mu x] = h(p + mu, n - mu) (minimiser sigma(x) = (p + mu)/(p + n)). So for any mu with
sum over the subcells of each kept cell = 0 and, for every k, sum over the subcells with beta_k = 1 = 0 (EXACTLY):
sum L >= sum h(p + mu, n - mu) + sum_c eta_c (sum_cell mu) + sum_k delta_k (sum_{beta_k=1} mu) = sum h(...).
mu is read from a primal Newton point, floored to the 2^-20 grid and clamped to [-p, n]; the cell sums are
absorbed on the subcell with most room; each shift sum is fixed by moving mass between two subcells of the same
cell whose bits differ only at k (keeps the cell sums and the other shift sums); finally all sums are checked
exactly (grid values with counts < 2^20: every partial sum is exact in float64) and D is evaluated with the
relative margin (terms + 8) 4 eps + 1e-12 of P21/P24. Used after K2 fails, before the leaf LR.

P27 (one-step dual of K_r from shared expansion points; no exp/log per leaf). Same relaxation and dual as P26
(any mu in [-p, n] meeting the constraints EXACTLY gives D(mu) = sum_subcells h(p + mu, n - mu) <= K_r). At a depth
s - 1 node, a primal point of the node's own relaxation (free intercept per ancestor, shifts for the r - 1 columns
of T, no j) gives every depth-t cell c a logit and t0_c = sigma(logit) (clamped to [1e-12, 1 - 1e-12]: t0 is only
an expansion point, any value in (0, 1) is allowed); log t0, log(1 - t0), H(t0) are computed once per cell.
For a leaf j: rho = n t0 - p, omega = n t0 (1 - t0) per subcell (c, j bit), one Newton step dtheta of K_r at the
node's point (arrow system), mu = rho - omega (U dtheta) (its constraint sums are g - H dtheta = 0 in exact
arithmetic), floored to the 2^-20 grid, clamped, and made exactly feasible by the P26 repair (exact check). The
value D = sum_sub n H(t), t = (p + mu)/n, is bounded below per subcell by the Taylor bound
H(t) >= H(t0) + H'(t0)(t - t0) - (t - t0)^2 / (2 min(t0 (1 - t0), t (1 - t))), H'(t0) = log((1 - t0)/t0)
(H'' = -1/(s(1 - s)), whose magnitude on the segment is largest at an end since s(1 - s) is concave).
In terms of Delta = n (t - t0) = (p + mu) - n t0: n H(t) >= n H(t0) + H'(t0) Delta - Delta^2/(2 n m).
Rounding: p + mu is exact (grid); |fl(Delta) - Delta| <= eps (n t0 + 2|Delta|) = eD; H'(t0) from the two stored
logs has absolute error <= 5 eps (|log t0| + |log(1 - t0)|) = eH; n H(t0) relative error <= 10 eps (two positive
terms); the quadratic term is evaluated with (|Delta| + eD)^2, m from t (1 - t) = (p + mu)/n (n - p - mu)/n (no
cancellation) times (1 - 8 eps), and rounded up by (1 + 8 eps); the subtracted error is
sum [10 eps n H(t0) + (|H'| + eH) eD + |Delta| eH + 2 eps |H' Delta|] + (terms + 8) 4 eps sum|terms| + 1e-12
sum|terms|. Subcells with m <= 1e-14 (t at an end of [0, 1]) are evaluated exactly with logs.
For r = 2 the layout of _k2_bound is used (_kos2_leaf); for r = KR the generic _kos_leaf. A failed one-step dual
only means the full K solves (P24, P26) and the leaf LR follow as before.

## Attempts

### z1_base (y23_giveup3 re-recorded) : n_certified 55, tiny 60, wrong 0, geo 0.338s, regret 0.00005. KEEP (baseline).

### z2_kr4 : n_certified 55, tiny 60, wrong 0, geo 0.337s, regret 0.00005. DISCARD (no change on the suite).
Idea: P26 with r = 4 after K2 (K_r warm-started from the previous leaf of the same node; dual tried after Newton
steps 0, 3, 6, 10), K2 warm-started from the node's own one-shift solution (search only) with its dual tried after
step 0. Dev (6 in parallel, no give-up): australian 7 105 -> 74 s (K_4 discards 2.09M of the 2.16M leaves that
went to the LR; ~1.5 Newton steps and 1.1 duals per call, ~8 us vs 22 us for the LR); heart 10 progress in 116 s
3.4e9 -> 7.6e9 (r = 4) / 8.6e9 (r = 5) discarded leaves. K2 warm start: K2 per call 3.9 -> 3.2 us (australian),
3.9 -> 2.5 us (ilpd), 4.1 -> 5.7 us (heart: the step-0 dual mostly fails there). Leaf LR warm start from the K2
shift (theta_j = dj / gap): slower LR (heart 55 -> 64 us per leaf), not used.
Suite: australian 7 still gives up at 1.0 s (the projection rule fires: it projects > 30x the budget at 1 s).

### z3_famstop : n_certified 53, tiny 60, wrong 0, geo 0.341s. DISCARD (lost fico 5 and bank 7).
z2 + (a) family call rule: success estimate pooled over this and deeper depths, and no call at a depth after 30
calls without a success there or deeper (ionosphere 5 spends 10-15 s on 30-700 family solves that never succeed);
(b) K2 skipped while it discards < 40% of its calls (K_4 follows; it is still run on every 32nd survivor).
(a) is wrong as a heuristic: on bank 7 / fico 5 the families succeed near the root but rarely deep, so the deep
calls were switched off and both ran to the limit. Reverted (FAMPOOL = False); (b) kept for the next attempts.
Dev on (b): heart 10 K_5 alone > K2 + K_5 (8.7e9 vs 8.0e9 leaves discarded in 116 s); australian 7 K2 + K_4 77 s
vs K_4 alone 107 s, K_3 + K_5 107 s.

### dev between z3 and z4 (not run on the suite)
- P27 generic first version: one-step duals kill almost every P1 survivor (australian 7: 8.88M of 9.03M; ilpd 7:
  15.9M of 15.9M; heart 10: 2.1M of 2.55M) but were slower than K2 per call (4 us: allocations, bool bit arrays,
  many passes). Specialised r = 2 version on the K2 layout (_kos2_leaf) and an integer-code generic version with
  preallocated work arrays (_kos_leafc, _kr_fixc): australian 7 80 -> 56 s (2 jobs in parallel), ilpd 7 K cost per
  P1 survivor 2.5 -> 1.3 us. A first _kos_leafc left stale ancestor sums of dropped ancestors, so the repair
  failed more often (weaker, still valid); fixed.
- Audits (tools/audit.py, brute force over all integer vectors): k = 4, d = 6: 34 problems; k = 3, d = 7: 12;
  k = 5, d = 6: 6+ problems: every one certified at the brute-force optimum, 0 wrong.
- Give-up projections under suite-like load (14 dev jobs): australian 7 stays at 50-115x the budget from 0.2 s to
  30 s and then finishes; ilpd 7 sits at 44-76x the whole time (it needs ~45x). No projection threshold separates
  them; the rule is relaxed (150x at 1 s, 80x after 3 s), accepting that ilpd-like problems run to the limit.

### z4_kos : n_certified 55, tiny 60, wrong 0, geo 0.368s, regret 0.00005. DISCARD (no new certificate, geo up).
z2 + P27 (one-step K_2 / K_4 duals, specialised r = 2 code and integer-code generic code) + K2 gate + give-up
relaxed (150x at 1 s, 80x after 3 s). australian 7 now runs to the limit (58.2 s) instead of giving up, but does
not finish under suite load (dev, 2 jobs: 56 s); ilpd 7 and ionosphere 5 also run to the limit (geo 0.338 -> 0.368).

### dev: KEYDEP (P13 key |theta| without sd when the data has exact column dependencies; heuristic order only)
australian 7 62 -> 48 s, adult 10 9 -> 9 s, bank 7 27 -> 22 s, mushroom 5 24 -> 39 s (dev, 9 jobs in parallel).
Profile (dev, 120 s): ionosphere 5: P1 leaf pass 72 s, pushes + packing 11 s, family 15 s (672 calls, none
succeeds), P12 counts 2.6 s; magic 7: P1 pass 29 s, family 22 s, pushes 9 s; mushroom 5: family 22 s of 39 s.

### z5_keydep : n_certified 56, tiny 60, wrong 0, geo 0.371s, regret 0.00005. KEEP (+1: australian 7).
z4 + KEYDEP. New: australian 7 (41.7 s). adult 10 8.8 -> 6.9 s, bank 7 13.5 -> 16.9 s, mushroom 5 18.8 -> 25.8 s,
magic 7 53.1 s (still tight). geo 0.338 -> 0.371 because ilpd 7 and ionosphere 5 now run to the limit (relaxed
give-up, needed for australian 7: its projection stays at 50-115x until ~30 s).
Dev (heart 10, 120 s): K_r one-step with r = 5 / 6 / 7: 8.9e9 / 1.48e10 / 1.51e10 leaves discarded; r = 7 leaves
only 24k leaves to the LR (vs 381k at r = 5).

### z6_krs : n_certified 56, tiny 60, wrong 0, geo 0.373s. DISCARD (no change on the suite).
z5 + P26/P27 width r = max(4, s - 3) (K_7 at k = 10). Dev heart 10: 1.41e10 discarded leaves in 116 s (z5: ~9e9);
the LR now sees 23k leaves instead of 381k. Kept as the base of the next attempts (valid, only changes which
relaxation is tried).

### z7_floorfix : n_certified 56, tiny 60, wrong 0, geo 0.371s. DISCARD (same as z5).
z6 + (a) the family success floor PIFLOOR applied only after a real family success (it was keyed on stats[6] > 0,
which the P23 duplicate skips also increment: on mushroom the floor kept 742 failing family solves alive, 23 s;
now 85 calls, mushroom 5 26 -> 16 s); (b) gate on the r = 2 one-step dual (heart: it rarely succeeds);
(c) one-pass ancestor repair in _kr_fixc. Dev heart 10 full run: 140 s (was 152 s).
Family success per depth (calls -> successes), dev: heart 10 depth 8: 146k -> 45k (31%), depth 7: 49k -> 15k;
australian 7 depth 4: 20k -> 1.7k (8%); magic 7 depth 3: 1.6k -> 318. A family attempt at the leaf level (depth s-1)
is almost never chosen by the cost model (FAMLEAF, kept off).

### z8_kosd : n_certified 56, tiny 60, wrong 0, geo 0.379s. DISCARD (same certificates; geo noise).
z7 + generic P27 accumulated per depth-t cell from its set T bits (no loops over all 2^r codes), Taylor-bound value
with per-cell constants (H'(t0), its error, 1/(t0(1 - t0)) rounded up, 1/n table rounded up). Dev heart 10: KOS
time 72 -> 64 s, full run 134 s. Profile of the r = 7 one-step dual on heart 10 (2.05M calls, 108 subcells on
average): build + accumulate 3.2 us, Schur + Cholesky 0.7 us, mu 1.8 us, exact repair 3.9 us, value 1.7 us.
Starting heart 10 from its optimum (0.27820094) as ub does not help (136 s): the incumbent is found early.
Audits with KR = 5 forced at k = 5 (r = s, ancestors = one cell): 10/10 and 8/8 (d = 7) certified at the brute-force
optimum, 0 wrong.

### dev: give-up projections with the z8 code (3 copies each, 9 jobs in parallel, 60 s budget)
australian 7 projects 1-2x the budget from 0.2 s on (KEYDEP: the root family succeeds after 7 children and the
early family discards are counted in the progress), and finishes in 47 s; ilpd 7 15-39x; ionosphere 5 12-20x
(75x at 0.2 s). The relaxed give-up of z4-z8 is no longer needed for australian 7: z9 restores the original rule.

### z9_giveback : n_certified 56, tiny 60, wrong 0, geo 0.347s (z5: 0.371, -6.5%). DISCARD (< 10%), base for z10.
z8 with the original give-up rule: ilpd 7 stops at 3 s again; australian 7 44 s; ionosphere 5 still runs to the
limit (its projection is ~12-13x at 3 s, just under the 12x rule in the suite).
Dev: a give-up for "no family success yet and projection > 4x after 3 s" stopped magic 7 in one dev run (its
first family successes come later); requiring >= 50 family calls without any success keeps magic 7, mushroom 5
(0 successes, projection < 1x), ionosphere 4 and spambase 4, and stops ionosphere 5 at 3 s.

### z10_nofam : n_certified 56, tiny 60, wrong 0, geo 0.359s. DISCARD (vs z5 0.371: -3%; timing noise on the
10-ms problems: heart 3 0.012 -> 0.021 s, compas 5 0.034 -> 0.054 s between z9 and z10 with the same code there).
z9 + give up after 3 s when >= 50 family solves have all failed and the projection exceeds 4x (heuristic): ionosphere
5 58.2 -> 5.8 s.

### z11_setup : n_certified 56, tiny 60, wrong 0, geo 0.333s (z5: 0.371, -10.2%), regret 0.00005. KEEP.
z10 + vectorised per-column setup for two-valued data (checked identical crow/ccode/lval/cptr/lptr/order on all
14 datasets, tools/cmpsetup.py) and no lexsort of the rows when every DFS depth is packed (any row order gives the
same counts; P22 packs per cell). fico 3 0.27 -> 0.19 s, bank 3 0.07 -> 0.06 s. Stack since z5: adaptive K_r width,
P27 per-cell accumulation, family floor fix, original give-up + no-family give-up, fast setup.

### z12_gu02 : n_certified 56, tiny 60, wrong 0, geo 0.317s (z11: 0.333, -4.8%). DISCARD (< 10%), base for next.
z11 + give up after 0.2 s when the projection exceeds 1000x (dev, 14 jobs: at 0.2 s the certified hard problems
project <= 372x (magic 7), the hopeless k = 7/10 ones 2e3-3e6x; at <= 0.1 s the projections are meaningless:
magic 7 1.5e7x, fico 5 7e5x). The hopeless k = 7/10 problems now stop at ~0.3 s instead of ~0.5 s.
Dev: a second one-step stage with r = s (one ancestor, i.e. the LR relaxation with an exact dual) on heart 10:
K_r fallbacks 152k -> 66k, but the extra stage costs more than it saves (134 -> 139 s); not kept.
Note: magic 7 certifies at 53-55 s in the suite (limit 58.2 s): little margin.

### z13_krs2 : n_certified 56, tiny 60, wrong 0, geo 0.313s. DISCARD (neutral vs z12 0.317; australian 7 43.7 -> 44.8 s).
z12 + width r = max(4, s - 2) (K_5 at k = 7, K_8 at k = 10). Reverted to KRS = 3.

### dev: word-major copy of the depth s - 2 packed columns for the leaf pass (TRANSP): slower (magic 7 P1 pass
32.6 -> 37.6 s, fico 5 65 -> 75 s, paired runs). Not used. The machine is shared (load average ~80 from other
users' jobs during this session), so only paired dev runs are compared.

### z14_gate25 : n_certified 56, tiny 60, wrong 0, geo 0.315s. DISCARD (neutral). K2 / one-step K2 gate at 25%;
reverted to 40%.

### dev: P27 repair moves from node-level lists of cell pairs (KFIXE): no gain (heart 10 KOS 65.5 -> 66.3 s,
australian 7 12.8 -> 13.3 s, paired). Not used.

### z15_starts3 : n_certified 56, tiny 60, wrong 0, geo 0.296s (z11: 0.333, -11%), regret 0.00009 (+0.00004). KEEP.
z12 + integer local search from the 3 best rounded solutions (was 5). Two uncertified problems get worse incumbents
(heart 10 0.27915 -> 0.28086, ionosphere 5 0.20649 -> 0.20744); no certificate changes. (Trade-off to keep in mind:
the certify search itself finds the better incumbents when it runs long enough, e.g. heart 10 0.27820.)

### dev profile of the mid-size certified problems (60 s budget, 8 jobs): mammo 10: 6.0 s in the integer B&B (174
supports, 98k nodes: the LR relaxation is loose there); spambase 3: 3.1 s of leaf LR (6842 leaves, continuous);
heart 7: one-step K 5.2 s, family 3.2 s, node-level K solves 1.4 s; ionosphere 4 / fico 4 / magic 5: P1 leaf pass.

### z16_cert975 : n_certified 56, tiny 60, wrong 0, geo 0.303s. DISCARD (neutral). Certificate deadline 0.975 of the
limit (58.5 s): kept in the base as a little more margin for magic 7 (53-55 s).

### z17_floor02 : n_certified 55 (lost fico 5: runs to the limit), DISCARD. Family success floor 0.2; reverted to 0.1.

### z18_floor007 : n_certified 56, geo 0.299s. DISCARD (neutral; heart 7 10.4 -> 14.2 s). Floor 0.07; reverted.

### z19_warm1d : n_certified 56, tiny 60, wrong 0, geo 0.302s. DISCARD (noise level vs z15 0.296), kept in the base.
Leaf LR warm start (search only) by up to 4 Newton steps on (theta_new, intercept) with the other coordinates fixed,
for non-binary leaves: mammo 10 5.6 -> 3.8 s, spambase 3 4.4 -> 3.9 s, spambase 4 44.5 -> 43.1 s.

### dev: the same warm start for binary leaves: no material change (heart 7 10.7 -> 10.4 s, australian 7 45.5 ->
44.6 s; the binary leaves rarely reach the LR now). Not used.

### z20_gu03 : n_certified 56, geo 0.290s (z15 0.296). DISCARD (< 10%), kept in the base. 300x give-up from 0.3 s.

### z21_gu2s : n_certified 55 (lost magic 7: given up at 2.1 s), DISCARD. 12x level from 2 s; reverted to 3 s.
magic 7's projection only drops below 12x after ~3 s in the suite; the rule is at its edge for this problem.

### z22_gu16 : n_certified 56, geo 0.286s (z15 0.296, -3.4%). DISCARD (< 10%), kept in the base: 16x after 3 s gives
magic 7 more margin at no visible cost.

### z23_starts2 : n_certified 52, geo 0.420s. DISCARD. Local search from the 2 best rounded solutions. The run was
contaminated by heavy load from another user's jobs (load average 156; ilpd 5 took 30 s instead of 1.8 s, and
spambase 4 / fico 5 / australian 7 / magic 7 failed with the SAME incumbents as z22), but ilpd 7's incumbent got
worse (0.48484 -> 0.48589): reverted to 3 starts, not retried.

### z24_base : n_certified 56, tiny 60, wrong 0, geo 0.292s, regret 0.00009 (re-run of the z22 base under normal load;
duplicate measurement, marked discard). australian 7 43.7 s, magic 7 53.0 s.

### z25_nofam30 : n_certified 56, geo 0.291s. DISCARD (small), kept in the base: no-family give-up after 30 failed
family solves (ionosphere 5 5.8 -> 3.6 s); K2 dual after the first Newton step off (K2 is rarely reached now).

### z26_famstop : n_certified 56, geo see csv. DISCARD (no effect): give-up / deadline check also before family solves
after 1 s. spambase 5 / 10 still stop at 2.7 / 5.0 s: the time goes into the integer B&B of early supports, whose
time cap (_gu_time) is at least GIVEUP_MIN = 3 s. Kept (harmless).

### z27_gufloor : n_certified 56, tiny 60, wrong 0, geo 0.292s. DISCARD (noise level), kept in the base: the
per-support LR / B&B time cap from the give-up projection is floored at 1 s instead of 3 s (spambase 5 2.7 -> 1.5 s,
spambase 10 5.0 -> 2.3 s; certified problems unaffected: their single supports need far less).

### z28_extrails : n_certified 55, tiny 60, wrong 0, geo 0.293s, regret 0.00005. DISCARD (magic 7 ran to the limit:
the run overlapped another user's jobs, load average ~99; the change only acts after a failed certificate, so it
cannot slow magic 7's search). 2 extra integer local-search starts when the certificate is not reached: regret back
to 0.00005 (heart 10 / ionosphere 5 incumbents as with 5 starts).

### z29_extrails2 : n_certified 56, tiny 60, wrong 0, geo 0.288s, regret 0.00005, AUC 0.8417. DISCARD by the rule
(vs z15: same certificates, geo -2.7% < 10%), but it is the recommended final code: same certificates as z15, lower
geo, regret back to the z1 level, more give-up margin for magic 7, faster continuous-data leaves.

### z30_cert98 : n_certified 56, tiny 60, wrong 0, geo 0.282s, regret 0.00005, AUC 0.8417. DISCARD by the rule (vs z15
-4.7%). z29 + certificate deadline 0.98 of the limit (58.8 s; magic 7 55.5 s, no fit over the limit).

## Final state (30 suite runs, z1-z30)

Best kept by the keep rule: z15_starts3 (slim_lib/z15_starts3.py; slim.py restored to it): 56/70 certified
(y23_giveup3 re-recorded as z1_base: 55/70), 60/60 tiny, 0 wrong, geo 0.296 s (z1: 0.338 s), regret 0.00009 (z1:
0.00005), test AUC 0.8421.
Recommended code: z30_cert98 (slim_lib/z30_cert98.py) = z15 + 2-D leaf LR warm start (non-binary), give-up levels
0.3 s / 300x and 3 s / 16x, no-family give-up after 30 failed family solves, family/deadline check before family
solves, per-support time cap floor 1 s, 2 extra local-search starts when not certified, deadline 0.98: 56/70, 60/60,
0 wrong, geo 0.282 s, regret 0.00005 (each step was below the 10% bar, so none is a "keep").

New certificate: australian 7 (~44 s in the suite). Beyond the 60 s limit (dev, 2-5 jobs in parallel, current code,
no give-up): heart 10 certified in ~135 s at loss 0.27820094 (below the best known 0.27914686), ilpd 7 in ~715 s
(finds and proves the best known 0.48346504), ionosphere 5 in ~624 s, spambase 5 in ~512 s.

Still uncertified (14) and the measured reason:
- heart 10: 2.3x over the budget: 4.9M leaves; one-step K_7 dual 64 s (11 us per leaf: build 3.2, repair 3.9,
  mu 1.8, value 1.7), family solves 44 s (220k calls, 31% success at depth s - 2), full K_r 12 s.
- ionosphere 5: ~11x: pure P1 enumeration of 1.1e10 leaves (P1 leaf pass 380 s of 624 s); no family solve ever
  succeeds (267 columns separate the 280 rows).
- ilpd 7: ~13x: 566M leaves (157M P1 survivors, all but 0.04% discarded by the r = 2 one-step dual at ~1.3 us),
  family solves 229 s.
- spambase 5: ~9x: continuous columns (no cell structure): ~460k leaf LRs at ~0.8 ms on 3376 rows + integer B&B.
- spambase 7 / 10, fico 7 / 10, bank 10, magic 10, australian 10, ilpd 10, ionosphere 7 / 10: projections of
  1e3-1e7x the budget at 0.2 s (heart-10-like leaf costs but 10-1000x larger trees); with the best known loss as
  ub, ilpd 10 / bank 10 / magic 10 are still < 1% done after 600 s.
- Fragile certificates: magic 7 (53-55.5 s; lost once under load average ~99 from other users' jobs), spambase 4
  (~42 s), australian 7 (~44 s), fico 5 (~38 s).

Audits: ~130 random binary problems (k = 3, 4, 5; d = 6, 7; planted duplicates / complements; KR forced to the
widest r in two batches), optimum by brute force over all integer vectors: every one certified at the optimum,
0 wrong.
