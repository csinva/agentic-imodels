# sep27-exact2 notes (exact track: certify optimal sparse integer risk scores)

Notation. Unique rows grouped into cells; for a set of columns S the cells are the distinct value
combinations of X[:, S]; cell c has pos_c positives, neg_c negatives, n_c = pos_c + neg_c.
F(theta) = sum_c pos_c sp(-u_c.theta) + neg_c sp(u_c.theta), sp(z) = log(1 + e^z). Loss = F / N.

## Proofs used by the certificate

P1 (saturated bound). Any score on S is a function of the cell, so its loss >= sum_c n_c h(pos_c/n_c)
(each cell predicted at its own rate). Valid for every w with support inside S.

P2 (continuous relaxation). An integer vector w on S with multiplier a gives theta = (a w, b), a real
vector; so inf over integers >= inf over real theta of F with columns [X_S, 1] (logistic regression).

P3 (rigorous LR bound, "gap bound"). F convex, C^2. At a point th with gradient g (|g| <= G),
suppose Hess F(t) >= mu I for all t in the ball B(th, R) and R >= 2G/mu. Then inf F >= F(th) - G^2/(2 mu).
Proof: inside the ball F(t) >= F(th) + g.(t-th) + mu|t-th|^2/2 >= F(th) - G^2/(2mu). On the sphere
F >= F(th) - G R + mu R^2/2 >= F(th); by convexity along the ray from th, every point outside the ball
has F >= F(th). mu is computed as lambda_min of H_R = sum_c n_c sig'(|u_c.th| + R|u_c|) u_c u_c^T,
which is <= the Hessian at every point of the ball because sig' is even and decreasing in |t| and
|u_c.t - u_c.th| <= R|u_c|. Floating point: lambda_min from LAPACK (backward stable) minus
(10p + p C) eps trace(H_R); F inflated error 4 C eps F; G inflated by C eps sum_c n_c |u_c|.
If mu <= 0 or R cannot be closed, the bound is not used (fall back to a weaker valid bound).

P4 (branch and bound over integer points on a support). Order the coordinates; a node fixes a
prefix w_P. Every integer completion has score a (w_P.z_P) + a w_R.z_R + b, which lies in the set
{alpha (w_P.z_P) + v.z_R + b : alpha, v, b real} (convex LR, columns [w_P.z_P, z_R, 1]; the first
column dropped when w_P = 0). Children's sets are subsets of the parent's, so a parent bound is
valid for children. Symmetries: (a, w) and (-a, -w) give the same scores, so the first nonzero
coordinate is taken positive; w and g w (g integer) give the same set of scores up to the scale a,
so only primitive vectors (gcd 1) are evaluated at the leaves (w/g is in the same tree).

P5 (leaf). For a fixed w the 2-parameter calibration is P3 with p = 2; if all scores are equal the
minimum is exactly N h(P/N) (base rate). If P3 fails (separable), P1 on the cells is used.

P6 (certificate). Every w with at most k nonzeros has support inside some S with |S| = min(k, d').
The DFS visits every such S; each S is either pruned by a valid bound >= ub - tol, or resolved by
the B&B. lower_bound_ = min over every bound used to discard something and every leaf lower bound.

P7 (separated directions). F is a sum of nonnegative cell terms, so dropping any cells gives a lower
bound; when P3 cannot be closed (infimum not attained), the pure cells whose loss is already < 1e-7 are
dropped and P3 is applied to the rest. Otherwise P1 is applied to the partition of the cells by their row
of U (the score is a function of that row); rows grouped only when exactly equal (a finer grouping is
still a valid, lower, bound).

P8 (exact column reduction). If a column of U is exactly a linear combination of other kept columns
on the kept cells, the set {U theta} restricted to those cells is unchanged when it is dropped, so the
minimum is the same. A numerical dependency is only used after an exact check (integer coefficients over
a common denominator q <= 64, integer entries < 2^20, so the float sums are exact integers).

P9 (family bound). In the DFS every leaf below the remaining children of a node (chosen T, next index
jj) is a support inside T + order[jj:], so the LR over those columns (P2, rigorous by P3, P7/P8 for
degenerate cases) bounds every point vector of every one of them. The column sets shrink as jj grows,
so once the bound reaches ub - tol every remaining child of the node is discarded.

## Attempts

### exact_v1 (baseline) : n_certified 11, tiny 60, wrong 0, geo 0.287s. keep (baseline).

### x2_bnb : n_certified 42, tiny 60, wrong 0, geo 4.419s, regret 0.00005. KEEP.
Idea: rewrite the certificate. numba DFS over all C(d', s) supports (s = min(k, d')), cells kept per
depth and refined incrementally from each column's non-base rows (codes sorted so a cell's new
sub-cells are created on the fly); leaf: saturated bound P1 (incremental over touched cells) ->
continuous LR bound P2+P3 -> integer B&B P4/P5 with sign and gcd symmetry. Separated supports
(pure cells) and exactly collinear columns (one-hot groups, complementary columns) handled by P7/P8,
otherwise the bound collapses to P1 (found on mammo k=4 and australian k=4 before P7/P8).
Certified per k: 3:14 4:11 5:9 7:5 10:3. Also improves mammo 7/10 incumbents to the best known.

### x3_inplace (not run on the full suite; folded into x4)
In-place DFS: cells refined by pushing a column (rows with a non-base code move to child cells) and
undone by pop, O(nnz_j) per step instead of O(n_u) copies; leaf saturated loss from a table
Lt[m] = m log m (integer counts), margin 12 (s+1)(n_u+1) eps Lt[N] (each table entry within 1 ulp,
each cell term within 4 eps Lt[N], at most 2(s+1)(n_u+1)+1 terms/additions along a path).
Measured 1.2-1.4x on leaf throughput (adult k=7 certifies at 52 s). LR at leaves warm-started from LR(T).

### x4_family : n_certified 45, tiny 60, wrong 0, geo 3.127s, regret 0.00005. KEEP.
P9 family bound: at a DFS node, LR over T + order[jj:] (BLAS Hessians, gap bound, exact column
reduction for one-hot groups) prunes all remaining children at once when >= ub - tol. Call rule:
leaves_below * (avg nnz + survival_rate * 50 s^2) >= 0.5 n_u p^2. Offline estimate (sklearn) had
shown 99.7% of compas k=10 leaves and 94% of haberman k=10 leaves prunable this way.
New: compas 10 (6.4 s), haberman 7/10 (1.1/2.2 s); bank 5 11.8 s (was 46.8). Clock: CPU clock()
counted BLAS threads in dev runs; switched to CLOCK_MONOTONIC via ctypes.

### x5_bits : n_certified 48, tiny 60, wrong 0, geo 2.044s. KEEP.
All-binary data (after constant columns are dropped) with 2^(s-1) * words <= 4 * avg nnz per column:
raw rows bit-packed (positives then negatives), cells of each depth stored as bitsets, the leaf
saturated loss is 2 popcounts (LLVM ctpop intrinsic) per cell and word; no proof change (P1 counts are
exact integers; the table margin is the same). New: fico 4 (7.3 s), magic 5 (12.1 s), ionosphere 4
(16.1 s). Leaf accounting checked: visited + family-pruned leaves = C(d, s) on bank 5, compas 5, mammo 7.

### x6_adapt : n_certified 49, tiny 60, wrong 0, geo 1.962s. KEEP.
(a) bits leaf: cells visited in decreasing saturated loss; pure cells skipped (a pure cell stays pure);
    stop as soon as the sum of the post-split losses of the cells seen so far (others add >= 0) minus
    the margin reaches ub - tol (valid partial P1 sum). Instrumented on mushroom 5: 1.0 cell per leaf.
(b) the real mushroom bottleneck was the family bound (525 calls of LR on 6499 x 113, always failing on
    separable data). Family calls are now adaptive: called when success-rate(depth) x leaves discarded
    x measured cost per leaf >= measured seconds per (n_u p^2) x n_u p^2 (priors 1/2 and 2e-9).
New: mushroom 5 (27 s, 140M leaves, 5M leaves/s). Tried and dropped inside this attempt: rows sorted
lexicographically + per-cell word ranges (no gain).

### x7_giveup : n_certified 48, geo 0.743s, regret 0.00009. DISCARD (lost mammo 10).
Give up when elapsed > 3 s and elapsed * C(d,s) / (visited + family-discarded leaves) > 12 x budget;
ub <= 5e-8 returns lb = 0 at once (mushroom 7/10: 54 s -> 0.1 s). mammo 10 (1001 supports, heavy B&B
early) was given up at 36 s although it finishes at 40 s.

### x7b_giveup : n_certified 49, tiny 60, wrong 0, geo 0.760s (from 1.962), regret 0.00009. KEEP.
Same, give-up only when C(d, s) >= 1e6. Regret +0.00004: runs that went to 54 s had found better
incumbents on the way (e.g. heart 7), now given up at 3 s.
