# sep27-exact1 notes: certifying sparse integer risk scores

Notation. Unique x-rows i with positive count P_i and negative count N_i (total N rows). For a column set U,
x~_i = (x_iU, 1). Logistic loss l(m) = log(1 + e^-m). Criterion of integer w: min_{a,b} sum_i P_i l(a s_i + b)
+ N_i l(-(a s_i + b)), s = X w. Totals below are sums (mean = total / N). UB = loss of the incumbent.

## Proofs of the bounds used

**B1 (Fenchel dual of logistic regression).** l(m) = max_{alpha in [0,1]} H(alpha) - alpha m, with
H(alpha) = -alpha log alpha - (1-alpha) log(1-alpha) (conjugate of the logistic loss; the max is at
alpha = sigmoid(-m)). Take any alpha+_i, alpha-_i in [0,1] (for the positive / negative copies of row i) with
r := sum_i (P_i alpha+_i - N_i alpha-_i) x~_i = 0. Then for every real (v, b) on U:
sum_i P_i l(z_i) + N_i l(-z_i) >= sum_i P_i H(alpha+_i) + N_i H(alpha-_i) - (v, b) . r = D(alpha).
Every integer w with support inside U and every (a, b) is such a (v = a w, b), so the minimum criterion over
all w with support in U is >= D(alpha). (Weak duality; no optimality of alpha needed.)

**B2 (family bound).** If S is a subset of U, an alpha feasible for U is feasible for S (fewer equations), so
D(alpha) bounds every support inside U. Hence LR(U) (the continuous optimum, = max D) bounds every support
in U, and the same holds for any feasible alpha we can construct.

**B3 (constructing a feasible alpha, "Newton projection").** At any iterate (v, b) with logits z, Hessian
Hs = sum (P_i + N_i) q_i (1 - q_i) x~_i x~_i^T (q = sigmoid(z)), gradient g = -r(alpha) at
alpha+ = sigmoid(-z), alpha- = sigmoid(z), and Newton direction d solving Hs d = g:
alpha+' = alpha+ + w_i x~_i.d, alpha-' = alpha- - w_i x~_i.d, w_i = q_i(1-q_i). Then
r(alpha') = r(alpha) + Hs d = -g + g = 0 exactly (in exact arithmetic). If all alpha' lie in [0,1], D(alpha')
is a valid bound (B1). Numerical margin: we recompute the residual r' of alpha' in floating point, solve
Hs d2 = r' and subtract E = sum_i (P_i |H'(alpha+'_i)| + N_i |H'(alpha-'_i)|) w_i |x~_i.d2| (the first-order
change of D needed to reach an exactly feasible point; r' ~ 1e-13 so E is ~1e-12) plus 1e-12 N.
Together with the global TOL = 1e-9 (mean) this covers rounding.

**B4 (saturated bound).** Any score on S is a function of the cell (value combination of X_S) of a row, so its
loss >= sum over cells of the Bayes loss of the cell (entropy of its positive rate). Exact counting.

**B5 (lexicographic support tree).** Order the D non-constant columns. Every w with at most k nonzeros has a
support inside some k-subset (if D >= k; else inside all D columns). Node T = (i_1 < ... < i_t) covers all
k-subsets T u A with A inside C = {i_t + 1, ..., D-1}; the subtree is bounded (B2) by LR(T u C). For siblings
j < j' of the same parent T', U_{j'} = T' u {j', ..., D-1} is a subset of U_j, so once a child's bound
reaches UB every later sibling is pruned too (break). Constant columns only shift b, which is free.

**B6 (integer branch and bound on a support S).** (w, a) and (-w, -a) give the same logits, so the first
nonzero fixed coordinate may be taken positive; w and g w (g integer > 1) give the same criterion, so a
non-primitive leaf is skipped (its primitive version is in the box, has the same sign convention and is
enumerated elsewhere). At a node with coordinates P fixed to w_P: every completion has logits
a s_P + X_R (a w_R) + b with s_P = X_P w_P; relaxing a w_R to free reals v_R gives a logistic regression on
the columns (s_P, X_R), bounded below by B1/B3. If w_P = 0 the column s_P is dropped (support R).

## Attempts

### exact_v1 (baseline)
Starting solver. n_certified 11, tiny 60, n_wrong 0, geo 0.270 s, regret 0.00010. keep (baseline).
Measured (scratch/an2.py): per-support LR (continuous optimum) leaves only 1-6 supports below UB on
compas k<=5, haberman k<=5, mammo k<=7, adult k<=4. scratch/an3.py: lex tree with LR(T u C) family bound
(features ordered by LR drop) has 2.9k internal nodes + 6.9k leaves on compas k=10, 14k + 38k on adult k=7.

### x2_lrtree (keep)
Idea: replace support enumeration by the lexicographic support tree B5 with the family bound LR(T u C) from the
dual construction B3 (early exit once the dual reaches UB or a primal iterate goes below UB; warm start from the
last solution of each column), sibling break, leaves bounded by B4 then B1/B3 on the leaf's cells (numba
leaf_scan), and integer B&B B6 on the few surviving supports. Features ordered by the Wald statistic of the full
LR. Proofs B1-B6 above.
Result: n_certified 34 (from 11), tiny 60, n_wrong 0, geo 6.763 s, regret 0.00009.
Certifies all k of compas, mammo, haberman, breastcancer; adult k<=7; k=3 of bank/spambase/australian/heart/
ilpd/magic; k=4 australian/heart; mushroom k7/10 (loss 0). Keep.

### x3_dynenum (keep)
Idea: (a) any order of a node's candidates C gives a valid partition of its supports (B5), so each child orders
its candidates by v_j^2 var(x_j) from the node's LR solution (most important first, so later siblings exclude
them and the break comes sooner); (b) a cost model: when child i's subtree has fewer leaves than one family bound
costs (running averages of per-leaf and per-bound time), enumerate the leaves of child i and all later siblings
directly with the numba pair_scan / leaf_scan (bounds B4 then B1/B3 per leaf); (c) B4 over all columns: if the
incumbent reaches the saturated loss of the unique rows, stop (mushroom k7/k10 in 0.2 s).
No new bound. Result: n_certified 41, tiny 60, n_wrong 0, geo 3.693 s, regret 0.00009. Keep.
New: australian k5, heart k5, ilpd k4, ionosphere k3, mushroom k3, fico k3, bank k4. haberman k10 12.7 -> 1.3 s.

### x4_fastleaf (keep)
Idea (speed only, no new bound): leaf cells of T u {j} for two-valued j built from the CSC rows of j's second
value (O(nnz_j + cells) instead of O(rows)); leaf LR warm-started from LR(T) (6 Newton steps, a start point
only); own numba Cholesky instead of LAPACK solve (call overhead dominated the tiny leaf systems); the cost model
now uses deterministic operation counts (the x3 timing-based model made the search path timing dependent, so it
was not deterministic: heart k5 flipped between 10 s and a timeout); pair_scan was not compiled by the warm-up
(about 2 s of JIT inside every fit that used it), now called directly in _jit_warmup.
Result: n_certified 43, tiny 60, n_wrong 0, geo 3.241 s, regret 0.00009. Keep. New: mushroom k4, magic k4.
Long runs (600 s, 14 in parallel): adult k10 79 s, bank k5 61 s, ilpd k5 123 s, spambase k4 97 s, ionosphere k4
251 s, fico k4 345 s, magic k5 270 s, spambase k5 347 s; australian/heart/ilpd k7, bank k7, mushroom k5 did not
finish (root child 1-17 of 57-113).
Explored and rejected (scratch/an4.py, an5.py): a cardinality-aware one-Newton-step dual family bound
LR(T) - q_A/2 (q_A = score statistic of A given T). The quadratic value prunes 100% of random nodes, but the
rigorous version needs the linearised duals to stay in [0,1] and a remainder factor 1/(1-tau)^2 with
tau = max_i |x_i.theta|; tau is about the coefficient size (median 1.8-2.3 logits), so the rigorous bound
pruned only 2-9%. Not used.

### x5_pairgram (keep)
Idea (speed only): when a node enumerates the pairs {i, j} of its candidates (m = 2), one pass over the rows
(grouped by the cell of T) accumulates, per cell of T, the counts of rows at the listed value of i and of both i
and j; the 4 sub-cells of every pair follow by inclusion-exclusion, so the leaf costs O(cells) instead of
O(nnz_j). The listed value of a two-valued column is its rarer value (fewer rows). Entropies of integer counts
from an x log x table (no log calls). Removing a per-row np.sort in the gram pass made it 5x faster.
Leaf cost bank k5: 12 -> 1.3 us.
Result: n_certified 45, tiny 60, n_wrong 0, geo 2.391 s, regret 0.00009. Keep. New: bank k5, ilpd k5.
