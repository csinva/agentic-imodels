# sep28-exact3 notes (exact track: certify optimal sparse integer risk scores)

Start: x7b_giveup from runs/sep27-exact2 (proofs P1-P9 are in that run's notes.md and are reused
unchanged; notation as there). New proofs below are numbered from P10.

## New proofs

P4b (B&B: first nonzero of a prefix). In int_bb a node at level m with prefix w_P = 0 (all earlier
coordinates zero) and value v != 0 has the relaxation set {alpha v z_m + sum_{i>m} beta_i z_i + b}, which
equals the parent's set {beta_m z_m + sum_{i>m} beta_i z_i + b} (alpha real free, v != 0). Same set, same
infimum, so the parent's proven bound blev[m] is the node's bound and no new solve is needed. The node is
not pruned by it (the parent was not), so the search simply descends.

P4c (B&B: non-primitive prefix). For a prefix w_P with g = gcd(|w_P|) > 1, the relaxation set
{alpha (w_P . z_P) + ...} equals that of w_P / g (alpha absorbs g). The reduced prefix has the same
sign convention and a strictly smaller first nonzero value, so in the value order of int_bb (increasing
at every level) it was visited earlier in the same call; its proven bound (stored in a dict keyed by the
reduced prefix, only bounds actually computed are stored) is a bound for this node. If it is >= ub - tol
the node is pruned; else the node descends with that bound. If the key is missing (the reduced node was
not reached because an ancestor was pruned) the node is solved normally.

P11 (Cholesky-verified lambda_min, used in gap_bound for p > 2 instead of LAPACK eigvalsh). Let A be the
computed H_R. An estimate lam of lambda_min(A) from inverse iteration is only a proposal. mu = f lam
(f in 0.98, 0.85, 0.6, 0.3) is accepted only if the floating-point Cholesky of A - (mu + marg) I completes.
By Higham (Accuracy and Stability, Thm 10.3), completion means A - (mu + marg) I + E = R^T R (PSD) with
|E| <= gamma_{p+1} |R^T||R|, so ||E||_2 <= ||E||_F <= gamma_{p+1} sum_i ||r_i||^2 <= 2 (p + 1) eps tr(A)
(for (p + 1) eps < 1/2). The margin marg = (12 p + p C + 12) eps tr(A) covers this plus the rounding of A
itself (the (p C + 10 p + 10) eps tr term of P3). Hence lambda_min(H_R) >= mu, which is all P3 needs.

## Attempts

### x7b_giveup (re-recorded) : n_certified 49, tiny 60, wrong 0, geo 0.745s, regret 0.00009. keep (baseline).
Uncertified (21): adult 10; bank 7, 10; spambase 4, 5, 7, 10; fico 5, 7, 10; australian 7, 10; heart 7, 10;
ionosphere 5, 7, 10; ilpd 7, 10; magic 7, 10. Diagnosis (single runs, no give-up, 600 s, numbers below exclude
~13 s of JIT that a fresh driver pays once):
- fico 5 (C(142,5) = 4.5e8): finishes in 180 s; 227M leaves at ~1.25M/s, 221M discarded by the root family
  bound; only 2 supports survive P1. Pure leaf-throughput problem.
- heart 7 (C(57,7) = 2.6e8): finishes in 165 s; 70% of visited leaves survive P1, leaf LR (P2/P3) takes
  ~70% of the time (35 us per leaf), family bound 16%.
- magic 7: finishes in 379 s (298M leaves).
- spambase 4 (continuous, 3376 unique rows): ~700 s; per-support integer B&B takes 75% of the time
  (~3 ms per node on ~3000 cells), leaf LR ~2 ms per leaf.
- ionosphere 5 (C(267,5) = 1.1e10): ~3000 s; 3.5M leaves/s, P1 prunes all but 1 in 50k leaves, the
  family bound never succeeds (LR on 200+ columns over 280 rows separates).
- the others project to 10^3-10^6 x the budget (family bound only succeeds at the root of the DFS).

### y2 dev attempts (not run on the suite)
- dynamic per-node candidate order (children re-sorted by P1 of child + candidate): identical leaf and
  family counts on fico 4 / magic 5 / heart 7: the family bound P9 only ever succeeds at the root node,
  where the order is the static one. Dropped.
- P10 interval bound over nested features (chains of columns ordered by inclusion of their 1-sets: in every
  cell the counts of each member lie in the box of the two chain ends, the post-split loss is concave in the
  counts, so its minimum over the box is at a corner): valid, prunes 73% of leaves on fico 4, but it needs the
  two end counts per cell (2 popcount passes), and bisection mostly ends at pairs (1.25 leaves per range):
  net 1.4x slower. Dropped.
- early P3 attempt inside Newton (before full convergence): slower (failed attempts cost a Hessian each).

### y2_bbskip : n_certified 49, tiny 60, wrong 0, geo 0.702s (from 0.745, -6%), regret 0.00009. DISCARD (rule: < 10%).
Idea: P4b (no re-solve at the first nonzero of a B&B prefix), P4c (reuse the bound of the reduced prefix
w_P / gcd), P11 (Cholesky-verified lambda_min instead of LAPACK eigvalsh in P3 for p > 2).
mammo 10: 39.3 -> 6.9 s, spambase 3: 21.7 -> 13.8 s; leaf LR 10% faster (P11). Below the 10% geo bar, but
kept as the base of the next attempts (all three are valid and only remove work).

### y3 dev (not run on the suite): P12 paired child cells at the leaf level (bits DFS)
Proof: at a depth s-2 node the exact counts of every cell c' with every later column j are computed once
(PJ). A leaf cell is c' & f or c' & ~f, so for each pair one popcount over the smaller child gives the other
child's counts exactly by subtraction (and a parent with a single nonempty child needs no popcount). Only
exact integer counts, no new bound. Measured: words per leaf on fico 5 525 -> 278, leaves/s +0-30% (noisy);
the popcount loop runs at 0.25 ns/word (AVX-512 VPOPCNTQ), so the leaf cost is dominated by per-cell overhead
(~5 cell pairs per leaf at ~45 ns each), not by words. A float32 x log x table (L1-resident) changed nothing.
Kept in the code for the next attempt.

P12 (paired child cells, exact counting only). See y3 dev above.

P13 (candidate order per node). A DFS node T with remaining candidate list L (in any order) covers the
supports T + A, A inside L, |A| = s - |T|. Child i (= T + L[i] with candidates L[i+1:]) covers exactly those
whose first element of A in list order is L[i]; for any permutation of L[i:] chosen before child i is
expanded, the children i, i+1, ... still partition the remaining supports. The family bound P9 for the
remaining children is LR over T + L[i:], whatever the order. So the remaining suffix may be re-sorted at any
time; after a failed family solve it is sorted by decreasing |theta_j| sd_j of that (partial) solution, so
that the next children take out the columns that keep LR(T + suffix) below ub. Leaves are only counted
(progress) and never skipped by the reordering.

### y3_lrorder : n_certified 50, tiny 60, wrong 0, geo 0.624s (from 0.745), regret 0.00005. KEEP.
Idea: y2 + P13 (per-node candidate lists, re-sorted by |coef| x sd of the failed family solve) in both DFS
variants + P12 paired cells. New: heart 7 (30.3 s, loss 0.31446 below the best known 0.31497). adult 7
37.3 -> 6.0 s, mammo 10 39 -> 5.6 s, australian 5 14.6 -> 4.5 s, compas 10 4.8 -> 0.7 s. Dev runs (600 s
budget): fico 5 126 -> 91 s, magic 7 ~380 -> 137 s. ionosphere 4 9.0 -> 13.7 s (P12 setup per node costs on
d = 267 with tiny cells).

P14 (row merging in the family solve). F(theta) = sum over unique rows i of pos_i sp(-u_i.theta) +
neg_i sp(u_i.theta) depends on row i only through its values on the family columns; rows equal on those
columns contribute pos sp(-z) + neg sp(z) with the same z, so merging them (adding their integer counts)
leaves F, its gradient and Hessian unchanged as functions. Rows are grouped by a hash and merged only after
exact comparison with the group's first row; equal rows left in different groups are still correct.

P15 (global exact dependencies in the family solve). Once per fit, Gram-Schmidt over [1, columns in the
DFS order] on all unique rows finds columns j that are exact combinations of the intercept and earlier
independent columns (verified with integer coefficients over a denominator <= 64, entries < 2^20, as P8).
The relation holds on every row, so in a family whose column set contains every column of j's combination,
the reachable score set {U theta} is unchanged when j is dropped (P8): same infimum, and P3 can close
(adult, bank, australian, ilpd have one-hot groups; before, every family solve there failed P3 and paid a
second solve after a per-call column reduction).

### y4_famfast : n_certified 51, tiny 60, wrong 0, geo 0.666s, regret 0.00005. KEEP (+1 certified).
Idea: P14 + P15 in the family solve. New: adult 10 (30.6 s; dev: family time 65 s -> 20 s, total 95 -> 30 s).
adult 7 6.0 -> 2.6 s, bank 5 4.0 -> 2.4 s. geo up a little because adult 10 now runs 30 s instead of giving up.

P16 (packed rows at the leaf level; exact counting only). At the first child of a depth s-2 node, the rows of
each of its cells c' (positives, then negatives) are renumbered densely and every column j the node's leaves
can add is packed into that numbering with PEXT (bits of Xb[j] at the set positions of the cell mask, in row
order). Then |c' & f & j| = popcount(packed_f & packed_j) over n_c'/64 words instead of the cell's scattered
word range (~all W words), and |c' & ~f & j| = |c' & j| - that (P12). All counts are exact integers; P1 and
its table margin are unchanged. A software PEXT is used when the host lacks BMI2. Verified: identical leaf,
survivor and B&B counts with and without packing on magic 7, fico 4, mushroom 5, compas 7, haberman 10,
ionosphere 4 (a first version recomputed the counts of zero-loss cells from the unpacked bitsets, which are
not maintained in packed mode, and produced a false improving vector on magic 7: caught by re-scoring the
returned points in the dev driver, fixed before any suite run).
Cell-major pass (VERT): the same P1 partial sums, accumulated parent cell by parent cell for all candidates
of a node, dropping a candidate once its partial sum minus the margin reaches ub - tol.

### y5_packed : n_certified 52, tiny 60, wrong 0, geo 0.654s, regret 0.00005. KEEP (+1).
Idea: P16 packed rows + cell-major pass (and P12). New: fico 5 (43.9 s; dev 84 -> 45 s). fico 4 5.1 -> 3.1 s,
ionosphere 4 13.8 -> 9.4 s, bank 5 2.4 -> 1.4 s, magic 5 6.0 -> 4.3 s. Dev: magic 7 135 -> 95 s.

P17 (one Hessian in the gap bound). For a >= 0, d >= 0: sigma'(a + d) / sigma'(a) = e^-d ((1 + e^-a) / (1 + e^-a-d))^2
>= e^-d. On the ball |t - th| <= R: |u_c.t| <= |z_c| + zerr_c + R |u_c|, so
Hess F(t) >= sum_c n_c e^{-R |u_c|} sigma'(|z_c| + zerr_c) u_c u_c^T >= e^{-R umax} H_0 (umax = max |u_c|),
H_0 being exactly the matrix P3 builds at R = 0. With mu_0 its proven lambda_min (P11 / P3 margin),
mu(R) >= e^{-R umax} mu_0. R is taken as 1.25 x the fixed point of R = (2G/mu_0) e^{R umax} and accepted only if
R mu(R) >= 2G is checked explicitly (mu(R) rounded down by 1e-12 relative); then P3 gives F - G^2 / (2 mu(R)).
Otherwise the original R iteration continues. Saves the second Hessian + eigenvalue per call.
Newton directions in lr_bound_big use a float32 Gram matrix (a search direction only: the line search uses
float64 F, and every bound comes from gap_bound_big in float64).
Tried and rejected: 2 truncated-Newton (CG) steps to detect failing family solves early (adult 10 26 -> 87 s:
the less converged coefficients make worse P13 orders); fully converged family solves (heart 7 28 -> 110 s).

### y6_onerep : n_certified 52, tiny 60, wrong 0, geo 0.614s (from 0.654, -6%), regret 0.00005. DISCARD (rule: < 10%).
P17 + float32 Newton directions. Dev: magic 7 96 -> 90 s, heart 7 31 -> 28 s. Valid and only removes work:
kept as the base for the next attempts.

P18 (support ownership in the integer B&B). Every point vector w with |supp(w)| <= s lies in many s-subsets;
assign it to exactly one: S(w) = supp(w) + the (s - |supp(w)|) smallest column indices not in supp(w) (fixed
column indices of the reduced matrix, independent of the DFS order). The DFS visits or discards (with a bound
valid for every vector inside it) every s-subset, so it suffices that the B&B of a visited S enumerates the
vectors it owns. S owns w iff its zero set Z = S \ supp(w) is the |Z| smallest indices outside supp(w), i.e.
iff every zero coordinate of S has an index below every index outside S. So in the B&B of S a coordinate
may take the value 0 only if its index is below min(complement of S); other coordinates range over
{-5..-1, 1..5}. (P4/P4b/P4c unchanged: they bound subsets of the same relaxation sets.)
Bits DFS for s <= 12 whenever all columns are binary and the cell bitsets fit in 200 MB (was s <= 8 and a
cost test); with P16 packed leaves this is faster: bank 7 ~400 -> 150 s, adult 10 26 -> 23 s, adult 7 2.6 -> 1.6 s.
Tried: the one-Hessian gap bound inside the leaf/B&B Newton loop (QUICKGAP): -10% leaf LR time on bank 7,
+40% on spambase 3: dropped.

### y7_own : n_certified 51, tiny 60, wrong 0, geo 0.598s, regret 0.00005. DISCARD (lost heart 7: 31 -> 54 s).
P18 + bits for s <= 12. heart 7 is unstable under the timing-based family call rule (dev runs: 26 s with
~40k family calls, 80 s with ~10k calls, same code): the success-rate estimate locks in low after early failures.
Dev with P18: spambase 4 266 -> 166 s (B&B nodes 198k -> 118k), mammo 10 B&B nodes 125k -> 100k.

Family call rule (heuristic only, no bound): the per-depth success estimate (succ+1)/(calls+2) is floored at 0.1.
Without the floor a few early failures can stop the calls at a depth for the rest of the run (heart 7: 26 s with
the floor, 80 s without, deterministic). Tried and rejected: floor 0.2 (fico 5 45 -> 61 s); a geometric per-node
schedule (attempt only after the leaves below dropped 2x / 1.5x): much worse everywhere (heart 7 59-84 s,
magic 7 147-205 s), because the reorders after failed attempts (P13) are what make the later attempts succeed.

### y8_floor : n_certified 52, tiny 60, wrong 0, geo 0.603s (from 0.654, -8%), regret 0.00005. DISCARD (< 10%).
y7 + success-rate floor: heart 7 certified again (26 s in dev); geo gain below the bar. Base for next attempts.

### dev (not run on the suite): P19 one-step leaf bound, rejected
P3 at theta0 = (theta_T, 0) for every surviving leaf T + j without solving: the T block of the ball Hessian
lower bound H_R is shared by all leaves of a node (weights at |z| + zerr + R |u|max), the j column and gradient
come from the cell counts in O(cells x p), lambda_min by P11, R from a grid. Valid, ~1-2 us per leaf, but it
discards only 2% (heart 7) / 21% (bank 7) of the leaves that the full LR discards (offline: the plain quadratic
model would discard 80%; the ball radius 2G/mu with G ~ the score of j makes the weights collapse), and the
total time went up. Also rejected: trying P3 once inside the Newton loop when dec <= 1e-3 (F - target):
-17% leaf LR on bank 7, +10% on heart 7, +18% on spambase 3.
Rejected (dev): P13 order by the Wald statistic theta_j^2 / [H^-1]_jj instead of |theta_j| sd_j: heart 7
26 -> 19 s but adult 10 23 s -> >300 s, bank 7 144 -> >300 s, fico 5 47 -> 56 s.

### y9_famopt : n_certified 52, tiny 60, wrong 0, geo 0.617s, regret 0.00005. DISCARD (< 10% vs y5).
y8 + family Newton loop micro-optimisations (transposed float32 Gram, incremental z recomputed exactly before
P3). adult 10 30 -> 22 s, adult 7 2.7 -> 1.2 s. Base for the next attempts.

P20 (screening the children of a B&B node with the node's quadratic lower model). Let the node's relaxation
be inf F over theta (columns U), and th any point (the node's solution). Work in theta' = S theta, U' = U S^-1
(S = diag of the column maxima; the same LR problem; the ball then suits columns of very different scales).
At th: Flow = F(th) rounded down, G >= |grad F(th)| (as P3), and for a radius R, M_R = H_R - delta I with H_R
the P3 Hessian lower bound of the ball B(th, R) (weights sigma'(|z| + zerr + R |u'|)) and delta its rounding
margin, so Hess F(t) >= M_R on the ball, and mu_R a proven lambda_min of H_R (P11) with R mu_R >= 2G.
A child is the node's set cut by a hyperplane through the origin, a.theta = 0 (beta_i = v alpha for a fixed
value v of the next coordinate when the prefix is nonzero; beta_i = 0 for the value 0). For t in the child set:
inside the ball F(t) >= Flow - G R + (1/2) D^2 with D^2 = min over a.Delta = -a.th of Delta^T M_R Delta
= (a.th)^2 / (a^T M_R^-1 a); outside the ball, by convexity along the ray from th and F >= Flow - G R +
mu_R R^2 / 2 >= Flow on the sphere, F(t) >= Flow - G R + mu_R R^2 / 2. So every child point has
F >= Flow - G R + min(D^2, mu_R R^2) / 2, and the child is discarded when this reaches ub - tol.
a^T M^-1 a is bounded above rigorously from the computed solve x of M x = a and its residual r:
a^T M^-1 a = a^T x + x^T r + r^T M^-1 r <= |a^T x| + 4 eps sum|a_i x_i| + |x||r| + |r|^2 / lambda_min(M)
(|r| inflated by the rounding of its computation, lambda_min(M) >= mu_R - 2 delta). |a.th| is rounded down.
Radii: 1.25 x 2G/mu_0 and 0.1, 0.4, 1.6 (scaled units). Prepared after every solved node and at the P4b
descents (where the point is the parent's solution in the new parametrization; at the root, the support's LR
solution: as a warm start for the solves themselves it made mammo 10 explode, so it is only a screening point).
Also: P18 fix (a disallowed zero could slip through when the value was clamped from -5 to 0).
Leaf/B&B LR: one exp and one log1p per cell per evaluation (exp(-|z|) of the accepted point reused for the
next gradient/Hessian): lr_bound 26.5 -> 19.5 us on heart-sized leaves. Dev: spambase 4 166 -> 46 s
(B&B 115 -> 7 s, leaf LR 40 -> 29 s), heart 7 26 -> 23 s.

### y10_screen : n_certified 53, tiny 60, wrong 0, geo 0.604s, regret 0.00005. KEEP (+1).
Stack since y5: P17, float32 Newton directions, P18 (+ fix), bits for s <= 12, success floor 0.1, family loop
micro-opts, P20 screening, one exp/log1p per cell. New: spambase 4 (46.1 s). adult 10 30 -> 20 s,
spambase 3 9.8 -> 4.4 s, mammo 10 8.9 -> 5.5 s. Regression: mushroom 5 20.8 -> 31.0 s (family calls never
succeed on mushroom (separable); the success floor keeps calling them).

P21 (shared-shift relaxation of a leaf, all-binary columns). For the support S = T + j, every point vector
(w, a, b) gives each row of T cell c the logit a w_T.x_cT + b + a w_j x_j = eta_c + delta h with
eta_c = a w_T.x_cT + b + a w_j v0_j, delta = a w_j (v1_j - v0_j), h in {0, 1} the indicator of the larger value
of column j. So the minimum over S is >= K_j = inf over (eta_c free per T cell, delta) of
sum_c L(P0_c, N0_c; eta_c) + L(P1_c, N1_c; eta_c + delta), a convex problem with an arrow-shaped Hessian
(diag over cells + one row/column for delta). Cells holding a single class are dropped (their infimum is 0
for any delta; dropping nonnegative terms keeps a lower bound, P7). Newton on the arrow system (O(cells) per
step), then P3 at the final point: |u| <= sqrt(2) per row, weights sigma'(|x| + zerr + R |u|), and
lambda_min of the arrow matrix from its Schur complement: lambda_min >= mu iff mu < min D_c and
a - mu - sum_c b_c^2 / (D_c - mu) >= 0, checked with a relative rounding margin 8 (m + 4) eps (bisection
on mu), entries shifted by a margin 16 eps tr. Used only in the bits DFS (binary j) before the leaf LR.
Check: on 3200 random binary leaves (heart, bank, ilpd, australian, compas, magic, fico, adult) the K_j
bound never exceeded the LR value; median relative gap to LR 0.9%. Dev: bank 7 128 -> 94 s (K_j discards
1.69M of 2.84M LR leaves), ilpd 5 6.6 -> 3.4 s, australian 5 4.3 -> 2.5 s, heart 7 22 -> 19 s.
Family-call floor applied only once some family bound has succeeded (mushroom 5 31 -> 22 s).

### y11_kj : n_certified 53, tiny 60, wrong 0, geo 0.551s (from 0.604, -8.8%), regret 0.00005. DISCARD (< 10%).
P21 + conditional floor. bank 7 no longer gives up at 3 s (it now projects within the give-up factor) and runs to
the limit, which costs geo. ilpd 5 6.8 -> 3.6 s, australian 5 4.1 -> 2.6 s, magic 5 5.5 -> 3.7 s,
mushroom 5 31 -> 23 s. Base for the next attempts.

P21 dual (replaces the P3 step of P21; no Hessian bound needed). For p, n >= 0 and -p <= mu <= n:
min_x [L(p, n; x) - mu x] = h(p + mu, n - mu) (h(a, b) = a log((a+b)/a) + b log((a+b)/b), the saturated loss of
real counts; the minimiser has sigma(x) = (p + mu)/(p + n)). Hence for every cell c and lam_c with
max(-N0, -P1) <= lam_c <= min(P0, N1) and all eta, delta:
L(P0, N0; eta) + L(P1, N1; eta + delta) >= h(P0 - lam_c, N0 + lam_c) + h(P1 + lam_c, N1 - lam_c) + lam_c delta.
Summing with sum_c lam_c = 0 EXACTLY, the delta terms cancel: K_j >= D(lam) = sum of the h terms (and D(0) is the
saturated bound P1). lam_c is read from a primal point (the residual of half 1, averaged with minus that of
half 0: the dual optimum at the primal optimum), floored to the grid 2^-20 and clamped to its interval
(integer ends), then the exact sum (counts < 2^20, so grid sums are exact in float64) is cancelled on cells with
slack. D is evaluated with a relative margin (2m + 8) 4 eps + 1e-12 (each term a log of an exact ratio of exact
values). The dual is tried after Newton steps 1, 3, 6 and at the end; any lam gives a valid bound.
Check: 3200 random binary leaves, D never above the LR value, every leaf closed (P3 version: 3126 of 3200).

P22 (packed DFS for all-binary data; exact counting only). Every depth t <= s-2 keeps, for every column the
subtree can still add, its bits in the depth-t row numbering (cells contiguous, positives then negatives),
built from depth t-1 with PEXT using the pushed column (or its complement, masked to the cell's valid rows)
as mask; depth 0 is the raw bitset. Child counts are popcounts of the packed pushed column per cell; the leaf
level uses the depth s-2 packing (P12/P16). Verified: identical leaf / survivor / family counts to the
unpacked DFS on compas 7, haberman 10, fico 4, mushroom 5, magic 7, bank 7. Measured gains were small
(bank 7 94 -> 88 s): the time is in the family solves, the leaf pass and the leaf LR, not in the pushes.
Per-node overhead cut: the family-call rule reads the clock every 256 tries (was every try), insertion sort
of <= 64 parent cells instead of argsort, no family warm-start copy for leaf-level nodes.

### y12_packed_dual : n_certified 53, tiny 60, wrong 0, geo 0.573s, regret 0.00005. DISCARD (vs y10 0.604: -5%).
y11 + P21 dual + P22 + per-node overhead. magic 7 and ionosphere 5 now project within the give-up factor and run
to the limit (54 s each, not certified), which raises geo; ilpd 5 6.8 -> 2.5 s, australian 5 4.1 -> 2.1 s.
Dev (single runs): bank 7 81 s, magic 7 77 s, heart 7 19 s.
Family solves: (a) skip a node's first family attempt when its column set equals the set its parent has just
failed on (the child T + f with suffix after f has exactly the parent's family columns); (b) row merging (P14)
skipped when the last merge on no more columns kept > 97% of the rows. Dev: magic 7 77 -> 67 s, bank 7 81 -> 76 s,
fico 5 43 -> 40 s, adult 10 17 -> 15 s.
Rejected (dev): skipping a family attempt when the columns removed since the last failure have an estimated
gain 1/2 (theta sd)^2 x curvature below the deficit: magic 7 68 -> 70 s, fico 5 40 -> 43 s, bank 7 76 -> 73 s.

### y13_famskip : n_certified 53, tiny 60, wrong 0, geo 0.566s, regret 0.00005. DISCARD (vs y10: -6%).
y12 + family-attempt skip + adaptive row merging. australian 5 1.8 s, bank 5 1.1 s, mushroom 5 17.9 s; magic 7,
bank 7, ionosphere 5 still run to the limit (dev: 67 s, 76 s, ~540 s).
Family Newton solves: exp(-|z|) of the accepted line-search point reused for the next residuals; a
single-precision Newton phase first (search only: z, F, residuals, Gram in float32; a float32 F below the
target only ends the solve as "not discarded", which is always safe), then double precision to convergence
and P3 from a float64 recomputation of z. Dev: magic 7 66 -> 56 s (family 31 -> 21 s), fico 5 40 -> 37 s;
bank 7 75 -> 90 s and adult 10 16 -> 18 s (the P13 order is sensitive to the slightly different coefficients).
Certificate deadline 0.9 -> 0.95 of the time limit (57 s; the fit returns within ~0.1 s of it).

### y14_f32fam : n_certified 54, tiny 60, wrong 0, geo 0.547s, regret 0.00005. KEEP (+1).
Stack since y10: P21 (+dual), P22, family-attempt skip, adaptive merging, float32 family phase, exp reuse,
deadline 0.95. New: magic 7 (53.9 s, tight). heart 7 23.5 -> 16.6 s, ilpd 5 6.8 -> 2.3 s, mushroom 5 31 -> 17 s.
bank 7 and ionosphere 5 run to the limit (dev ~75-90 s and ~540 s).

P23 (duplicate and complementary binary columns). Columns g and f with the same 0/1 pattern (dup) or
complementary patterns, and the same value gap |v1 - v0|, give scores w_g x_g = w_g v0_g + w_g gap h_g with
h_g = h_f (dup) or 1 - h_f (complement); so w_g x_g = (+-w_g) x_f + const, the constant absorbed by the free
intercept. Every point vector whose support contains g but not f therefore has an equivalent vector (same
criterion, coefficients still in [-5, 5]) with g replaced by f. Each class gets a representative (lowest
index); it suffices to enumerate supports S in which every non-representative column comes with its
representative, and P18 is redefined on representatives: S owns w iff S = supp(w) + the smallest
representative indices outside supp(w); a coordinate may be zero only if it is a representative with index
below every representative outside S. In the DFS, a child is skipped when the representatives missing for
its non-representative columns are not in its candidate list or exceed its free slots, and a leaf is
skipped when one is still missing; family bounds (P9) keep covering supersets, so stay valid.
Dev: bank 7 90 -> 39 s (leaf LR survivors 2.84M -> 0.75M), australian 5 1.8 -> 0.5 s, bank 5 1.1 -> 0.7 s;
same incumbents and certificates on compas 7/10, mushroom 5, adult 10.

### y15_dups : n_certified 55, tiny 60, wrong 0, geo 0.534s, regret 0.00005. KEEP (+1).
P23. New: bank 7 (38.3 s). australian 5 1.7 -> 0.6 s.

P24 (two-shift relaxation of a leaf, all-binary columns). For S = T' + f + j (f the last column of T), every
point vector gives the rows of subcell (c, sf, sj) (c a T' cell, sf / sj the indicators of the larger values of
f / j) the logit eta_c + sf delta_f + sj delta_j (the T' part is constant on c; f and j enter affinely). The
minimum over S is >= K2 = inf over (eta, delta_f, delta_j) of sum_subcells L(p, n; logit). Dual as in P21: any
mu_{c,s} in [-p, n] with, EXACTLY, sum_s mu_{c,s} = 0 for every cell, sum over sf = 1 of mu = 0 and sum over
sj = 1 of mu = 0 gives K2 >= sum h(p + mu, n - mu) (the eta, delta_f, delta_j terms cancel). mu from a primal
point (Newton with a 2-wide border, Schur complement), floored to the 2^-20 grid and clamped, then made exactly
feasible: cell sums absorbed on the subcell with most room, then the j sum by moving mass between (sf, 1) and
(sf, 0) of one cell and the f sum between (1, sj) and (0, sj) (both keep the cell sums), finally all three
families of sums checked exactly (grid sums are exact). Used after K_j fails, before the leaf LR.
Check: 2700 random binary leaves (heart, bank, ilpd, australian, compas, magic, fico, adult): never above the
LR value, closed on 2685. Dev: australian 7 148 -> 109 s, bank 7 38 -> 31 s, heart 7 17 -> 13 s.

### y16_k2 : n_certified 55, tiny 60, wrong 0, geo 0.535s, regret 0.00005. DISCARD (no change on the suite).
P24 (speeds up the LR-dominated hard problems; none crosses the limit yet). Base for the next attempts.

P25 (tried, removed): the exact-feasible entropy dual for the full leaf LR on binary columns (mu_c = cell
residual on the 2^-20 grid; total cancelled on cells with room; each column's sum over its larger value fixed by
moving mass between two cells differing only in that column, which keeps the total and the other columns;
exact re-check). Valid (2750 random binary leaves: never above the LR value; at the optimum equal to it to
1e-12), but not faster than P3 at the leaf (heart 7 17 -> 18 s, bank 7 31 -> 37 s, australian 7 116 -> 129 s:
the early dual attempts mostly fail and cost as much as a Newton step).
K2 dominates K_j (its feasible set is the subset of K_j's with eta_(c',f=1) - eta_(c',f=0) shared), so K_j is
skipped when K2 applies (t >= 1). Dev: heart 7 13.3 -> 12.5 s, australian 7 109 -> 104 s.
Rejected (dev): P3 attempted right after the float32 phase of a family solve (magic 7 57 -> 64 s: mostly fails);
P13 key |theta| instead of |theta| sd (adult 10 13 -> 8 s, bank 7 30 -> 20 s, australian 7 104 -> 70 s, but
fico 5 39 -> 51 s, magic 7 56 -> 76 s); key |theta| sqrt(sd) (bank 7 15 s, but australian 7 > 300 s): the
DFS cost is very sensitive to the order, no key is uniformly better.
Float32 design matrix of the family solve built directly in transposed layout (one pass): magic 7 family
21 -> 20 s.

### y18_f32T : n_certified 55, tiny 60, wrong 0, geo 0.527s, regret 0.00005. DISCARD (vs y15 0.534: -1%).
y16 + K2-only + one-pass float32 family design. adult 10 15.7 -> 11.2 s, bank 7 38 -> 18 s (path change).
Give-up rule (heuristic, no bound): besides "projected finish > 12 x budget after 3 s", give up after 1 s when the
projection exceeds 1000 x the budget. Dev: the hopeless problems stop at 1.0-1.5 s; every certified heavy problem
(magic 7, adult 10, fico 5, spambase 4, bank 7, heart 7, mushroom 5, ionosphere 4) still certifies.

### y19_giveup2 : n_certified 55, tiny 60, wrong 0, geo 0.458s (from 0.534, -14%), regret 0.00005. KEEP.
Stack since y15: P24 (K2), K2-only, one-pass float32 family design, two-level give-up. Hopeless problems now stop
at 1.1-1.8 s; bank 7 38 -> 18 s, adult 10 16 -> 12 s.
A support's LR / B&B is given the time at which the give-up rule would fire with the progress made so far (a
long B&B makes no progress): spambase 10 21 -> 3.7 s. Certificate deadline 0.95 -> 0.97 of the limit (58.2 s).

### y20_bbcap : n_certified 55, tiny 60, wrong 0, geo 0.454s (from 0.458, -1%), regret 0.00005. DISCARD (< 10%).
y19 + per-support solve capped by the give-up rule + deadline 0.97. spambase 10 22 -> 5 s.
Audit (scratch, brute force): 36 random binary problems (n 60-200, d 5-6 with planted duplicate and
complementary columns, k 2-3), optimum by enumerating every integer vector with the exact 2-parameter
calibration: every lower bound <= optimum, every certified loss = optimum (36/36 certified).
_jit_warmup also fits an all-binary problem (the packed DFS was compiled only by the harness's warm-up fit).
Rejected (dev): warm start of K2 from the node's one-shift solution (no gain; australian 7 path changed, 104 -> 151 s).
Give-up projections (x budget) measured at 1 s / 3 s: magic 7 15 / 8 and ionosphere 5 13 / 11 (magic 7 certifies,
the projection overestimates ~9x early), australian 7 39 / 54, ilpd 7 46 / 71, heart 10 839 / 305, adult 10 8 / 1.
Early (1 s) give-up threshold 1000 -> 30 x budget.

### y21_giveup30 : n_certified 55, tiny 60, wrong 0, geo 0.428s (from 0.458, -6.6%), regret 0.00005. DISCARD (< 10%).
y20 + early give-up at 30 x + binary warm-up. australian 7, ilpd 7, heart 10 now stop at 1.0 s.
P15 dependencies found from a Cholesky of the Gram matrix (BLAS) instead of a row-level Gram-Schmidt; every
candidate still passes the exact integer check of P8 (identical results on adult, bank, australian, ilpd,
compas, mushroom): fico setup 0.32 -> 0.012 s. Unique rows of all-two-valued data through packed bit
patterns (np.unique on a void view): ~5x faster than np.unique(axis=0). Dev: fico 3 0.57 -> 0.18 s,
magic 3 0.31 -> 0.11 s, mushroom 3 0.27 -> 0.10 s.

### y22_setup : n_certified 55, tiny 60, wrong 0, geo 0.381s (from 0.458, -17%), regret 0.00005. KEEP.
Stack since y19: per-support solve capped by the give-up rule, deadline 0.97, early give-up at 30 x, binary
warm-up at import, Gram-matrix P15, fast unique rows.
Give-up at 0.4 s for projections beyond 300 x the budget (measured at 0.4 s with the real budget: magic 7 22x,
adult 10 10x, bank 7 5x, australian 7 65x, ilpd 7 44x, spambase 5 244x, fico 7 2300x, bank 10 4e5x).

### y23_giveup3 : n_certified 55, tiny 60, wrong 0, geo 0.332s (from 0.381, -13%), regret 0.00005. KEEP.
Three-level give-up (0.4 s / 300x, 1 s / 30x, 3 s / 12x): the k = 10 hopeless problems stop at 0.4-0.7 s.

### y24_rerun : n_certified 55, tiny 60, wrong 0, geo 0.340s, regret 0.00005. DISCARD (rerun of y23: noise ~2%;
magic 7 53.0 s, certified again).
Projections at 0.1 / 0.2 / 0.3 s (x budget): magic 7 52 / 45 / 19, ionosphere 5 136 / 18 / 19, adult 10 15 / 13 / 8,
bank 7 11 / 13 / 10, australian 7 76 / 100 / 93, spambase 5 939 / 939 / 156. Fourth level: 0.2 s / 1000 x.

### y25_giveup4 : n_certified 55, tiny 60, wrong 0, geo 0.323s (from 0.332, -2.7%), regret 0.00005. DISCARD (< 10%).
Fourth give-up level (0.2 s / 1000 x): the hopeless problems are now dominated by the heuristic and setup.
Per-column codes of two-valued data without np.unique (bank 3 setup 52 -> 44 ms); the mid-size problems are
dominated by the DFS itself (heart 5: 170 ms for 4.2M supports, ionosphere 3: 205 ms for 3.1M).

### y26_codes : n_certified 55, tiny 60, wrong 0, geo 0.339s, regret 0.00005. DISCARD (noise level vs y23 0.332).
y23 + per-column codes without a sort.
P13 key without the sd factor when the data has exact column dependencies (one-hot groups): dev australian 7
149 -> 68 s, adult 10 11 -> 7.5 s, but bank 7 18 -> 23 s, mushroom 5 19 -> 26 s.

### y27_keydep : n_certified 55, tiny 60, wrong 0, geo 0.365s, regret 0.00005. DISCARD.
KEYDEP: australian 7 no longer gives up and runs to the limit (58 s, not certified), mushroom 5 19 -> 26 s.
K2 dual tried after Newton steps 0, 2, 5 (was 2, 5, 9): K2 time on australian 7 20 -> 14 s (dev 69 -> 62 s).

### y28_k2early : n_certified 55, tiny 60, wrong 0, geo 0.358s, regret 0.00005. DISCARD.
y27 + earlier K2 duals: australian 7 still runs to the limit in the suite (dev 62 s single), mushroom 5 26 s.

### y29_final : n_certified 55, tiny 60, wrong 0, geo 0.341s, regret 0.00005. DISCARD (noise level vs y23).
y23 + per-column codes + earlier K2 duals.

### y30_confirm : n_certified 55, tiny 60, wrong 0, geo 0.332s, regret 0.00005. (rerun of y23: identical metrics;
marked discard as a duplicate). slim.py restored to slim_lib/y23_giveup3.py.

## Final state
Best kept: y23_giveup3 (slim_lib/y23_giveup3.py): 55/70 certified (x7b_giveup: 49), 60/60 tiny, 0 wrong,
geo 0.332 s (x7b: 0.745 s), regret 0.00005 (x7b: 0.00009). New certificates: heart 7, adult 10, fico 5,
spambase 4, magic 7 (tight: 53-55 s), bank 7. heart 7 and heart 10 incumbents (found by the certify B&B) are
below the best known losses (0.31446 vs 0.31497; 0.27820 vs 0.27915).
Still uncertified (dev times, single process): ionosphere 5 (~520 s; 1.1e10 supports enumerated at ~25M
leaves/s, P1 prunes all but 1 in 1e7 but no family bound works on 280 rows x 267 columns), australian 7 (62-150 s
depending on the candidate order; K2 + leaf LR + family), heart 10 (~330 s; 4.8M leaves need the LR, the K
relaxations are weak with 512 cells over 216 rows), ilpd 7 (4% after 600 s; 17M leaves survive P1), bank 10,
fico 7 / 10, magic 10, australian 10, ilpd 10, ionosphere 7 / 10 (projections 1e3-1e6 x the budget: the
family bound ignores the cardinality and only prunes near the root), spambase 5 / 7 / 10 (continuous columns:
P1 is useless, every support needs an LR solve of ~1 ms on ~3000 cells).
Risk notes: magic 7 certifies with ~3-5 s of margin; the early give-up levels (0.4 s / 300x, 1 s / 30x) were
calibrated on the visible suite (magic 7 projects 22x at 0.4 s and 15x at 1 s although it finishes in time).
