# Run notes (sep15-run1)

- v6_dedup_thresholds: no measurable change alone on the subset; folded into v7 (snapshot kept).
- v11_cheap_threshold_{65536,131072}: raising the depth-2 "cheap" gate changes nothing on the
  deciding pairs (fico_binary λ=0.005 1.88 s vs 1.88 s, compas λ=0.05 0.89 vs 0.87 s); not run on
  the full suite, snapshot kept as v11_cheap_threshold_65536_not_run.py.
- Machine must be idle during recorded runs: tic-tac-toe λ=0.005 moved from 25 s to a 30 s timeout
  under light concurrent load in the baseline run.
- v12_children_from_kernel: creating child memo entries from kernel counts instead of leaf_stats
  changes nothing measurable (tic-tac-toe λ=0.005 3.62 vs 3.67 s); not run on the full suite.
- v13 first draft certified wrong values on 6 pairs (caught by n_wrong): moving the candidate
  preparation into a kernel changed which rejected candidates fed the node lower bound (the ones
  dropped after the depth-2 tightening were lost). Fixed by folding those back into min_pruned;
  re-validated on 200 random problems vs v10, 60 vs exhaustive search, 8 real pairs vs pygosdt_v1.
  Lesson: every refactor of the pre-loop bookkeeping must re-run the exactness checks before a run.

## Exactness audit (all versions, any dataset and lambda through the suite interface)

Standard: the implementation must return the certified optimum for every input, not just the
suite. Proof obligations checked: (1) every lower bound is valid for all trees on the subproblem;
(2) every stored ub is achieved by a tree that the extraction rebuilds; (3) `_solve(node, b)`
always leaves the node solved (lb == ub == optimum) or with lb > b, so callers never treat an
unsolved node as solved; (4) EPS comparisons only ever prune less, never more.

- pygosdt_v1_flat, v2, v3, v4, v5, v6: exact. v3's column DP is exact by the segmentation
  argument plus optimal substructure (its use as an incumbent elsewhere is only an ub). v4's
  exclusion follows from the leaf-support lemma (a leaf with potential < lambda can be removed
  for a strict gain). v5 is a line-by-line port of the numpy statistics into numba; re-derived
  and re-validated (exhaustive DP, random cost matrices, both engines).
- v7: NOT exact. When budget < 4 lambda and the best depth-2 tree exceeds the budget, the code
  stored lb = min(best, min split_lb), which can be <= budget while the node is unsolved; a
  caller then uses the node's ub as if exact. No wrong certificate appeared on the suite, but
  the implementation can produce one. Marked approximate.
- v8, v9, v10, v13: exact (v8 stores min(best, 4 lambda), the proof's quantity; v9 only changes
  the sequence of budgets passed to _solve; v10 only widens where the exact stage runs; v13's
  first draft lost rejected candidates from the node lower bound and certified wrong values on
  6 pairs, fixed before its recorded run).
- Latent issues in every version through v13, outside the suite's inputs: the inherited leaf
  test `1 - min_loss < lambda` assumes losses <= 1 (false for user cost matrices with entries
  above 1/n scale), and a user-supplied `upperbound` was treated as achievable. v14 removes the
  first (it is implied by `max_loss - min_loss < lambda` whenever losses <= 1) and turns the
  second into a pure budget.
- Baselines: gosdt (ICML 2020 C++) issues a verified false certificate, so it is flagged
  approximate as an implementation; gosdt_guesses (exact mode) and streed certify 0.318330 on
  that pair and show no violation on the suite.

## v14 — hardening + superset lower bound (keep)
63/70 solved, geo 0.01055 s (v13: 0.01126 s), 0 wrong.  Removed the three `1 - min_loss < lam`
leaf tests (they are only valid for 0/1 losses; the suite never exercised the gap, but a cost
matrix would), made `upperbound` a budget instead of a certificate, and added the superset bound
R(C) >= max over children R(child) which is sound because every tree on C restricted to a child's
capture set is a tree on that child.  Gains are on car_evaluation, coupon, compas_binned (0.4-0.6x),
small losses on tiny monk pairs (numba call overhead, ~1 ms).

## v15 — shape relaxation from the pairwise counts (keep)
63/70 solved, geo 0.00937 s (v14: 0.01055 s), 0 wrong.  With exact 1- and 2-leaf values per
child (g1 = leaf, g2 = 2λ + best 2-leaf loss), a peeled-leaf bound for 3-leaf subtrees
(g3 = 3λ + cheapest cell any split leaves as a leaf) and aλ for a >= 4, every tree with >= 4
leaves has a root split with a- and b-leaf subtrees, so min over (split, a, b), a + b >= 4, of
g_a(L) + g_b(R) bounds all of them; if the exact best depth-2 tree is no worse than that and
the leaf, the node is solved at any budget.  Children get lb = min(g1, g2, g3, 4λ) instead of
min(g1, g2, 3λ).  Both the bound and the child lbs are computed in one numba kernel
(shape_bound); the column spans are computed once per node instead of two np.unique calls.
Proof of the "solved" case: opt = min(leaf, best_d2, R(T*)) with T* the optimum if it has
>= 4 leaves, and best_d2 covers every tree with 2 or 3 leaves (shapes (1,1), (1,2), (2,1)).
Gains: tic-tac-toe 0.43-0.54x, monk_2/monk_3 ~0.5x; nothing on the seven timeouts.

## v16x — leaf-count caps (not run: null result on the subset)
Theory: in an optimal tree every leaf has potential >= λ, and replacing an a-leaf optimal
subtree on X by a leaf must not win, so loss(subtree) <= leaf loss(X) - (a-1) λ while it is
>= the equivalent-points loss of X; hence a <= cap(X) = min(pot(X)/λ, 1 + (leaf loss - eq
loss)/λ).  Children with cap <= 2 are solved exactly by the pairwise kernel and the shape
relaxation only combines feasible (a, b).  Exact (0 mismatches on all checks) but useless in
practice: iteration counts moved by < 5% (the caps only bind on tiny nodes, which the kernel
resolves anyway) and the extra per-node work cost 2-27%.  Snapshot kept for the record.

## v16 — fused expansion kernel (discard: geo 0.01000 s vs v15 0.00937 s)
63/70 solved, 0 wrong, but a regression: the fused kernel ran the pairwise stage before the
single-column DP could resolve the node, so sine_1k/gaussian_1k (999 thresholds of one column,
1 iteration each) went from ~0.5 ms to ~2.5 ms (4.6-5.5x); the binary datasets gained 0.7-0.8x.
v16b restores the v15 order (no pairwise stage on single-column nodes).  Same search and bounds as v15.  Measured
on tic-tac-toe λ=0.005: 76% of expansions are probes (pruned by the kernel stage without any
sub-expansion) costing ~16 µs each, almost all Python overhead: 23 small allocations, four
numba dispatches with 10-17 arguments each, and materialising the depth-2 incumbent before
knowing the node survives.  node_stats now derives the class counts and equivalent-points
loss itself; node_stats + prep_candidates + depth2_pairs + shape_bound run as one njit call
on a (14, m) float / (4, m) int / (2, m) bool workspace; the depth-2 tree is stored only when
the node is not pruned by the shape relaxation.  Subset: 0.73-0.96x on every pair.
Also measured: a perfect initial incumbent (upperbound = optimum + 1e-9) saves almost nothing
(tic-tac-toe 1.45 -> 1.37 s), so incumbent quality is not the bottleneck; the proof is.

## v16b — fused expansion kernel, single-column nodes excluded (keep)
63/70 solved, geo 0.00878 s (v15: 0.00937 s), 0 wrong, no pair slower; monk_2/monk_3 0.71-0.77x,
tic-tac-toe ~0.85x.  Same as v16 with the pairwise stage skipped on single-column nodes.

## v17x — pairwise similar support (not run: null result on the subset)
Theory (sound, proof in the file): for any two splits i, j of a node, a tree for split j
becomes a tree for split i by moving the points of L_i xor L_j between the children; with
the costs shifted so each class's cheapest prediction costs 0 (a per-node constant, the same
for every split) each moved point costs at most its potential, so lb(i) >= lb(j) - pot(L_i xor
L_j), with pot(L_i and L_j) read off the pairwise kernel.  This generalises the column-wide
propagation of v2 to every pair of candidates.  Exact on all checks, but it never binds: the
iteration counts are identical on tic-tac-toe, coupon, car, monk (unrelated binary features
have huge symmetric differences) and move by 1% on compas/iris, while the m x m potential
matrix and the vectorised propagation cost 5-30%.  Snapshot kept.

## v18x — row clustering + active-word kernels (not run: negative result)
Idea: sort rows lexicographically by the source columns (most informative first) so deep
capture sets cluster into few 64-bit words, and let the kernels visit only the non-zero words
(fico_binary: 40% of the words instead of 95% after sorting; compas: 87%, nodes there are
big).  Exact by construction (row order does not change the problem).  Negative: a gather
loop over active words, a loop over runs, and a loop over the [first, last) span all ran
1.1-1.8x slower on the tall datasets than the plain `range(W)` loop (LLVM vectorises the
fixed-extent popcount loop and not the others), and even with plain loops the sorted rows
were slower on compas.  Also learned: numba specialises on the read-only flag of arrays, so
passing the frombuffer mask into a kernel whose warm-up used a writable array recompiles
inside the first timed fit (0.4 s) -- subset timings must start with a warm-up pair.

## v18 — expansion cache + kernel-side child predictions (keep)
63/70 solved, geo 0.00860 s (v16b: 0.00878 s), 0 wrong, no pair slower.  The
budget-independent part of an expansion (child statistics, pairwise kernel, shape bound) is
kept on the node while its parent deepens its budget (27% of expansions are re-expansions),
probes return before slicing, and node_stats reports each child's leaf prediction so creating
a child node no longer recomputes its class counts from the big-int mask.

## v19x — depth-2 gate sweep (not run: the current gate is already at the optimum)
The pairwise stage runs when nv^2 W <= 32768 or when >= 50% (and >= 8) of the candidates
survive the cheap filter.  Sweeping the limit over 8192..524288 and the fraction over
0.25/0.5/1.1 (min 4/8) on iris, compas, fico_1k, tic-tac-toe, fico_binary, car, coupon,
monk_2 moved every pair by < 5% (noise) except disabling the survival rule, which is 5-60%
worse on iris/fico_binary.  Snapshot with the gate parametrised by environment variables.

## Where the loop stands after this session
Kept: v14 (hardening + superset lb), v15 (shape relaxation), v16b (fused expansion kernel),
v18 (expansion cache + kernel predictions): geo mean 0.01126 s (v13) -> 0.00860 s, 63/70
solved, 0 wrong throughout.  Null results documented above: leaf-count caps (v16x), pairwise
similar support (v17x), row clustering / active words (v18x), gate sweep (v19x); the
depth-3 (triple) kernel was costed and rejected before implementation (3-5x the probe work
it would replace, popcount-bound on tall data).  The seven timeouts (fico_1k lambda <= 0.05,
compas_processed lambda <= 0.02) are unsolved by every baseline within 600 s too; on them the
binding bound is 4 lambda for any child with many points, and no cheap admissible bound on
the loss of >= 4-leaf subtrees is available without O(m^3) counts.  A perfect initial
incumbent does not help (the proof, not the search for the tree, is the cost).

## v19 — exact depth-3 stage from triple counts (discard: geo 0.00870 s vs v18 0.00860 s)
New bound, exact: the class counts of every triple of candidate splits give, for every child
of the node, the exact best 3-leaf tree and the exact best (2,2) tree, so a child's achievable
risk is min over {1, 2, 3, (2,2)}-leaf trees and its lower bound min(that, 4λ + cheapest peeled
cell, 5λ + cheapest peeled cell or 2-leaf cell, 6λ).  A node is solved without expanding any
child whenever the best such tree is no worse than min over splits of the children's lower
bounds (every budget below 5λ).  Verified on 150 exhaustive trials with the extracted tree's
objective checked against the reported one, and on cost matrices.  It does what the theory
says -- iteration counts fall 2-4x (tic-tac-toe 81k -> 36k, car 8.2k -> 4.4k, monk_3 919 -> 389)
-- but the kernel costs 60 ns per triple (93 µs at 22 candidates) against ~63 µs of expansions
it removes per call, so it is break-even on tic-tac-toe and useless on fico_binary (big nodes,
many leaves affordable).  Gated to <= 14 candidates, K*W <= 64 words, budget below the
equivalent-points loss + 8λ and nodes below 64 λn points it is neutral on tic-tac-toe/coupon
and 0.6-0.9x on car/monk, which is too little to move the geometric mean (+1% in the recorded
run; the small pairs are noisy).  Also measured: node-level equivalent points (points
inseparable by the node's valid features) add nothing over the global equivalence on any
suite dataset, so that theory was dropped without a run.

## v20 — depth-3 stage with two-peeled-leaf floors (keep, best: geo 0.00851 s vs v18 0.00860 s)
63/70 solved, 0 wrong.  Same exact depth-3 stage as v19 with (a) tighter floors for the trees
it does not enumerate -- a (1,3)/(3,1) subtree has a leaf at depth 1 and a leaf at depth 2,
both cells known from the triples, so 4-leaf trees cost >= 4λ + min over splits of leaf(A) +
cheapest sub-cell of B (c4 >= c3); (2,3)/(3,2) >= 5λ + best 2-leaf(A) + cheapest sub-cell of B;
(3,3) >= 6λ + cheapest sub-cells of both sides -- and (b) the triple loop restructured so the
minima of the two (child, split) pairs whose sub-split is the largest index accumulate in
scalars (13% faster, outputs identical to v19's kernel on 45 random nodes).  Gates as v19.
Per pair: monk_2 0.55-0.84x, car 0.76-0.94x, monk_3 0.85x, iris 0.98x; tic-tac-toe and
fico_binary +2-5% (deep narrow nodes where the stage never resolves anything), coupon +11-18%.
Validation: 150 exhaustive trials with the extracted tree's objective checked, cost matrices
with losses > 1, upperbound semantics, 8 real pairs against pygosdt_v1.

## Theory summary of this round
* Node-restricted equivalent points (points inseparable by the node's valid features): sound,
  but measured to add exactly nothing over the global equivalence on every suite dataset.
* Exact depth-3 information from triple counts (v19/v20): the first bound here that gives a
  parent exact 3-leaf and (2,2) values of its children, resolving nodes with budget < 5λ
  without any child expansion; it cuts expansions 2-4x on the binary datasets but its O(m^3)
  kernel (60 ns per triple) only pays on narrow nodes, so it is gated to <= 14 candidates.
* Bounds on the loss of >= 4-leaf subtrees from pairwise counts alone are impossible beyond
  4λ (a (2,2) subtree can have zero loss on any pair of cells), which is why probes exist;
  triples are the minimum information that changes this.
* The seven timeouts (fico_1k λ <= 0.05, compas_processed λ <= 0.02) need the same triple
  information at m ~ 600-1400 candidates (10^8-10^9 triples per node) and remain out of reach;
  no exact solver in the benchmark certifies them within 600 s either.

## v21 — Python overhead round 1 (keep, best: geo 0.00809 s vs v20 0.00851 s)
63/70 solved, 0 wrong, no pair slower.  Line profile of v20 on tic-tac-toe: the memory guard
called `ps` through a subprocess every 1024 iterations, 4.4 ms per call = ~4 µs per iteration,
23% of the unprofiled fit; it is now checked at most twice a second of wall time.  The depth-2
incumbent's children were created with `_make_node` (class counts and prediction recomputed
from the big-int masks, 9 µs per call) and now come from the kernel's statistics; the candidate
refilter after the depth-2 stage (five numpy calls per node) is one numba function that also
sorts the survivors by (split_lb, split_ub2).  Subset: 0.59-0.91x everywhere.

## v22 — Python overhead round 2 (keep, best: geo 0.00801 s vs v21 0.00809 s)
63/70 solved, 0 wrong.  The node's class masks are built inside the expansion kernel from the
capture words and the dataset's mask matrix (one numpy op and one allocation less per
expansion) and the children's maximal lower bound (superset bound) comes from a numba helper
instead of two numpy reductions.  Subset 0.88-0.99x; the remaining per-expansion Python cost
is spread thin: Node construction (1.2 µs, 12% of tic-tac-toe), two big-int memo lookups per
candidate (0.3 µs each), the row views of the kernel workspace, and the loop's bookkeeping.
Line profiles: tmp/lineprof.py (uv run --with line-profiler).

## v23 — row clustering + in-kernel word compaction (keep: geo 0.00799 s vs v22 0.00801 s)
63/70 solved, 0 wrong.  Rows are sorted lexicographically by the source columns (most
informative at the root first); an expansion whose capture mask leaves >= 30% of the 64-bit
words empty copies the candidate features' words on the non-empty words only, inside the
kernel, and node_stats / depth2_pairs / depth3_triples run their unchanged contiguous loops on
the compact copy (earlier attempts with gather loops or spans lost LLVM's vectorisation).
General for tall data: fico_binary 0.70-0.91x, neutral elsewhere (compas nodes are large,
87% of words stay active; W <= 16 datasets never compact).

## v24x — lazy right child + uniform-cost leaf costs (not run: negative)
Two generic micro-optimisations, both exact.  (a) Compute the right child's mask and memo
entry only when the left child's bound plus the kernel's bound for the right child leaves the
split alive, and let memoised children take the kernel's bound when larger: neutral
(0.97-1.06x, iteration counts change by < 0.3%).  (b) With uniform costs compute a cell's
leaf cost as w * (size - largest class) instead of the K x K loop in node_stats and
depth2_pairs: 13-17% slower on tic-tac-toe and iris -- the data-dependent branch inside the
hot loops costs more than the small multiply loop it replaces (LLVM keeps the fixed-trip-count
K loop tight).  Lesson: in these kernels, branch-free fixed-size loops beat "fewer flops".

## Where the per-expansion cost stands (v23, tic-tac-toe λ=0.005, 0.89 s, 74.8k expansions)
~12 µs per expansion: kernel ~5 µs (node_stats + pairwise stage), the candidate loop ~4 µs
(two big-int memo lookups per candidate at 0.3 µs, Node construction 1.2 µs per new child),
workspace allocation/slicing ~1.5 µs, prologue ~1 µs.  Further generic gains would need a
structural change (array-based memo, loop in numba), not micro-optimisation.

## Multicore track (8 cores; `multicore` column added to the leaderboard, backfilled false)
Baseline: the reference C++ GOSDT accepts `worker_limit`; with 8 TBB workers (`gosdt_mc8`) it
certifies 38/70 within 30 s at geo 0.454 s versus 46/70 at 0.473 s single-threaded -- some
pairs speed up 3-4x (tic-tac-toe λ=0.05 0.165 -> 0.037 s) but others livelock (car λ=0.01
2.5 s -> timeout), a known issue of that code.  STreeD has no thread option; gosdt-guesses
warns that worker_limit > 1 deadlocks and was not run.

## v25 — root-parallel phase on 8 cores (keep, first multicore row: geo 0.00789 s vs v23 0.00799 s)
63/70 solved, 0 wrong.  The search runs sequentially for 50 ms; if the root is not solved by
then, 8 worker processes are forked (they inherit the memo and the root's expansion), pull
the root's surviving candidate splits from a queue and solve each split's children exactly
as the sequential loop does, with the incumbent in shared memory and only ever decreasing.
Exactness: a split is either solved (both children solved exactly) or proven above an
incumbent that is >= the final one, so the root is certified as in the sequential loop; the
winning worker sends its tree.  Verified on 120 exhaustive trials with the parallel phase
forced and on 10 real pairs against v23.  tic-tac-toe 0.32-0.42x, compas λ=0.05 0.40x,
fico_1k λ=0.1 0.88x; but fico_binary 1.19x and iris 1.29x: with few binary features the
subproblems are shared across root splits and the workers' private memos re-solve them
(188k expansions vs 58k sequential on fico_binary), plus ~50 ms of fork/queue overhead.

## v26x — shared lower-bound table between workers (not run: negative)
Exact (0 mismatches with the parallel phase forced): workers publish the lower bounds they
prove for large subproblems in a shared-memory open-addressing table keyed by the full
capture mask plus row count, and look it up when creating such a child.  Reads are lock-free
(slots are written once, bounds only increase, a partially written key can only coincide with
a strict subset, which has a different row count); writes hold a lock.  It removes 30-50% of
the duplicated expansions (tic-tac-toe 137k -> 92k, fico_binary 189k -> 105k) but each
access (mask to words, hash, probe) costs a few µs and the table itself is W x capacity words
(100 MB for compas), so wall time was 1.1-1.5x that of v25 at every threshold tried
(subproblems >= 1/16, 1/10, 1/4 of the rows); only iris gained (0.73x).
Measured with the sequential solver: the share of work under the largest root split is 61%
on fico_binary and 44% on iris (root-level parallelism caps at 1.7x / 2.2x there) but 5-6%
on tic-tac-toe and compas (8x possible) -- the next step is finer tasks, not shared memos.

## v27x — finer parallel tasks for heavy children (not run: negative)
A root split whose child holds >= half (or a quarter) of the rows is split further: the
child's kernel stage runs in the parent before forking (a loop-free `_head` copy of the
expansion), its own candidate splits become tasks and its sibling a task of its own, and the
parent combines the outcomes as the sequential epilogue would.  Exact after one fix (a
candidate whose sides were all resolved by the parent's kernel stages was never folded into
the root's value, which certified the greedy incumbent on 4 of 120 forced-parallel trials;
now 0 mismatches at both thresholds).  But without a shared memo every split-level task
re-solves the grandchildren its sibling tasks also need: tic-tac-toe 1.19M expansions vs
139k with root-level tasks (5.8-7x slower), compas 3.1x, fico_1k 4.1x, fico_binary 1.2x;
only iris gained (0.74x).  Together with v26x: process-level parallelism pays only at the
root, where the duplication factor is 1.3-3x; going finer needs shared state, and the
shared-table cost exceeded its savings.  v25 stays the multicore solver.

## Multicore summary
* Leaderboard: `multicore` column (declared, backfilled false), `gosdt_mc8` baseline
  (reference GOSDT, worker_limit=8: 38/70, geo 0.454 s; livelocks on several pairs).
* v25 (keep, multicore=true): geo 0.00789 s vs 0.00799 s sequential v23; 63/70, 0 wrong.
  Balanced problems 2.5-3x faster (tic-tac-toe, compas), unbalanced ones (fico_binary,
  iris) slightly slower; the seven timeouts unchanged (8x more search does not close them).

## v28 — the whole search compiled (keep, best sequential: geo 0.00645 s vs v23 0.00799 s)
63/70 solved, 0 wrong; every pair faster: coupon 0.29x, monk 0.33-0.38x, fico_binary
0.41-0.49x, car 0.49-0.60x, tic-tac-toe 0.55-0.64x, compas 0.73x, iris 0.82-0.84x.
Re-engineering: nodes are rows of arrays (mask words, count, leaf risk, prediction, lb, ub,
split, deferred depth-3 structure) behind an open-addressing index; the search with all its
stages (kernel probes, single-column DP with chain materialisation, depth-2 and depth-3
stages, similar-support propagation, budget deepening, superset bound) is one numba
function.  numba cannot link a self-recursive function of this size ("unresolved symbol"
once it actually recurses, with or without the disk cache), so the search is an explicit
stack of frames with per-depth workspaces and the candidate loop as a phase machine.
Python re-enters it in ~50 ms iteration chunks to enforce the time and memory limits and to
grow the store; on an interruption each open frame's incumbent is flushed to its node, so the
tree returned at a time limit is the best found (fico_1k λ=0.01 now returns the known best
0.304 instead of 0.311).  Compilation (~14 s per process) happens before the first timed fit.
Same bounds and search order as v23: verified on 150 exhaustive trials with the extracted
tree checked, cost matrices with losses > 1 and with non-zero diagonals, the budget
semantics, the time limit, and 8 real pairs.
Post-record fix to the v28 snapshot (extraction only, the search is unchanged): the compiled
port cleared a node's deferred depth-3 grandchildren in three places where the split did not
actually change (the single-column resolution writing the same split back, the premature
clears after the immediate/depth-2/depth-3 incumbent updates, and the chain materialisation
overwriting an equal-valued structure), which broke the invariant "a node's bound is realised
by its split" and produced a tree worse than the certified value when a node solved first by
a depth-3 structure was later expanded as a single-column node.  Found through the parallel
variant (a fresh worker processes candidates in an order the sequential loop never uses),
reproduced deterministically in-process, fixed by clearing the deferred structure only when
the split changes and by never replacing a strictly better structure in the chain
materialisation.  Re-validated: 150 exhaustive trials with the extracted tree checked, cost
matrices, non-zero diagonals, budget semantics, time limit.  The recorded v28 numbers stand
(same search); the snapshot in optimal_tree_lib is the fixed file.

## v29 — compiled search + root-parallel phase on 8 forked workers (keep, multicore: geo 0.00631 s vs v28 0.00645 s)
63/70 solved, 0 wrong.  The v25 design on the compiled engine: after 50 ms of sequential
search the root frame is expanded in the parent, 8 forked workers inherit the store and the
frame, pull root candidates from a queue and run the compiled search in single-candidate mode
with the shared incumbent capping the depth-0 bound; the parent combines outcomes exactly as
the sequential epilogue.  tic-tac-toe 0.38-0.55x, compas λ=0.05 0.53x, fico_binary 0.89x;
pairs of 5-40 ms pay the fork/queue overhead (1.05-1.16x).  Verified on 120 exhaustive trials
with the parallel phase forced (found and fixed the v28 extraction invariant on the way, and
the parent now writes its frame's incumbent to the root before forking).

## v30 — parallel phase with threads on private memo copies (keep, best multicore: geo 0.00603 s vs v29 0.00631 s)
63/70 solved, 0 wrong.  The compiled search releases the GIL, so 8 Python threads run it
concurrently on private copies of the memo taken after the root's expansion, with the
incumbent in a shared array; no fork, no queues, and the hand-off happens after 10 ms of
sequential search instead of 50 ms.  compas λ=0.05 0.71x, tic-tac-toe 0.74-0.94x, iris
0.80x, car 0.81x of v29; the 4-40 ms pairs are back to neutral; fico_binary λ=0.005 1.17x
(private memos duplicate its shared subproblems, as before).

## v31 — per-frame expansion cache with two slots per depth (discard: geo 0.00670 s vs v30 0.00603 s)
Exact (sequential and forced-parallel validation), and the idea works (a re-entry of a child
during budget deepening only redoes the budget-dependent steps), but the subset timing that
looked neutral hid a cost the recorded run exposed: doubling the frame slots doubled the
per-thread workspaces, whose depth-3 argument buffer was sized by the full feature count
(129 MB per workspace on fico_1k, nine workspaces allocated and zeroed per fit), so the wide
and the small pairs lost 1.6-1.7x.  Lesson: allocation costs do not show in warm repeated
fits; size buffers by what the stage can use and pool them across fits.

## v32 — lock-free shared lower-bound table for the threads (keep: geo 0.00591 s vs v30 0.00603 s; solved-only 0.00236 vs 0.00241)
The threads of v30 search private memo copies, so a subproblem reached under two root splits
is expanded by two threads.  v27x showed that sharing the memo itself (Python dicts, locks)
costs more than the duplication, so v32 shares only what is cheap and safe to share: proven
lower bounds.  One table per fit, one region per thread; a thread writes only its own region
(key words, row count, bound, then a "used" flag), every thread reads all regions before
expanding a node with at least n/16 rows and raises the node's bound to the largest value
found.  Reads need no lock: keys are written once, a partially visible key is a strict subset
of the true one and therefore has a different row count than the query (the row count is
compared first), 64-bit stores are atomic, and a bound is only ever raised, so any value a
reader sees is either 0 or a bound some thread proved for exactly that subproblem.  Bounds
are published only when raised by >= lambda/4 above the value the node had on entry, which
keeps the table small (8192 slots per thread, bounded probes).
Exactness: only lower bounds cross threads and every published bound was proved by the same
search on the same subproblem, so the certification argument of v25/v29/v30 is unchanged
(each root split is solved or proven above an incumbent that only decreases).  Validation:
sequential exhaustive DP 0 mismatches, forced-parallel exhaustive 0 mismatches, cost > 1 and
non-zero-diagonal cost matrices 0 mismatches, upperbound semantics, time limit honoured.
Also: workspaces are pooled across fits and the depth-3 buffer is sized by TRIPLE_MAX_NV,
which removes the allocation cost that sank v31 (its frame cache is kept).
Results: duplicated expansions roughly halved on the heavy pairs; compas λ=0.05 0.77x,
fico_binary λ=0.005 0.60x, tic-tac-toe 0.70-0.85x, small/wide pairs neutral (0.95-1.0x).
The two compas pairs that stop at the memory cap stop at a different time than in v30 (the
cap is hit by eight private memos); they are unsolved in both, so they do not affect the
leaderboard beyond the geo mean's noise.

## v33 — hand the root to the threads on time (keep: geo 0.00583 s vs v32 0.00591 s; solved-only 0.00226 vs 0.00236)
A per-task trace of the parallel phase showed that on iris, fico_1k and fico_binary the
sequential phase before the hand-off lasted 50-125 ms although the hand-off is meant to
happen after 10 ms: the search re-enters Python only at the end of an iteration chunk, the
first chunk was 2000 expansions, and on wide problems an expansion (118-1357 features,
depth-3 stage) costs tens of microseconds.  Now the first chunk is 500 expansions and, until
the hand-off, each chunk is sized to end at ``parallel_after`` (5 ms; 10 ms cost the 10-30 ms
pairs a slower start).  The threads' private stores are also sized by the parent's contents
(grown on demand) instead of the parent's capacity, and their key arrays are not zeroed (a
key is written in full before it is indexed): eight 21-25 MB zero-fills per fit disappear on
the wide data sets, and the compas pairs that stop at the memory cap now run to the 30 s
limit instead (fairer and still a better geo mean).
Search and bounds unchanged; validation: sequential exhaustive 0 mismatches, forced-parallel
exhaustive 0 mismatches, cost matrices 0 mismatches, upperbound and time-limit semantics.
Results: fico_1k λ=0.1 0.61x, tic-tac-toe λ=0.02 0.61x, car λ=0.005 0.70x, compas λ=0.05
0.74x, fico_binary λ=0.005 0.83x, iris λ=0.005 0.86x; nothing slower.
Trace numbers worth keeping (v32, 8 threads): the busy time summed over threads is 1.2-3x
the sequential v28 time on tic-tac-toe λ=0.005 and fico_binary λ=0.005 (private memos:
a subset reached under two root splits is expanded twice), and the parallel speed-up over
v28 was only 1.4-2.3x on the pairs above 100 ms.  Duplication is the next target.

## v33x — shared-table thresholds (not run: no signal)
Sweeping the v32 table's publication threshold (n/8 … n/64 rows, bound raised by 0 … λ/4)
and capacity (2^13 … 2^15) moved the heavy pairs by -7 … +8%, within run-to-run noise;
n/8 was the only clear loser (tic-tac-toe λ=0.005 1.27x).  Left at n/16 and λ/4.

## v34 — shared solved subproblems and a live incumbent for the threads (keep: geo 0.00580 s vs v33 0.00583 s)
Diagnosis first.  Running the parallel phase with the optimum given as the initial bound
separates the two sources of the threads' extra expansions: on iris the count drops from
7400 to 2100 (stale incumbent: a task deep in its subtree kept the budgets it started with,
improvements found by other threads reached only its root frame), on tic-tac-toe and
fico_binary it barely moves (memo duplication across the private stores, 1.4-1.6x).
Two changes:
1. The shared table gains a value column (NaN = not solved).  A node solved after a full
   frame is published with its optimum; a thread entering a subproblem whose optimum is in
   the table adopts it (lb = ub = value, solved, split = -3 - owner) instead of expanding
   it.  The tree of an adopted node is taken from the owner's store at extraction, which now
   happens after all threads joined (every store is final).  Lock-free as before: a value is
   a single 64-bit store made only for its slot's own key, so a reader sees NaN or the
   optimum of exactly the matched subproblem; the key match itself is safe because a
   partially visible key is a strict subset with a different row count.
2. When the shared incumbent drops, every frame of the task's stack is tightened: each
   child budget is re-derived from its parent's new bound exactly as it was derived when the
   child was pushed (first child: bound - lb(second); second child: bound - lb(first) while
   deepening, bound - ub(first) in the final solve).  Budgets only decrease, and a frame
   that ends with best > budget returns unsolved with a sound bound, so the conclusions are
   the ones the sequential search would draw with that incumbent.
Exactness: validated as before (sequential exhaustive 0, forced-parallel exhaustive 0 with
every adoption path exercised because the threshold is 2 rows on the tiny problems, cost
matrices 0, upperbound and time-limit semantics).
Results: iris λ=0.005 0.71x (expansions 7700 -> 5700), car λ=0.005 0.75-0.89x in the
subset runs, fico_binary λ=0.005 0.89-0.97x; tic-tac-toe unchanged (its duplication is
concurrent: both threads expand the subset before either publishes it).  Adoptions are few
(80 on iris, 4000 on tic-tac-toe out of 110k expansions), so most of the gain is the live
incumbent; the remaining duplication cannot be removed without a shared memo.

## v35 — pooled shared table, 3 ms hand-off (discard on the recorded number: geo 0.00582 s vs v34 0.00580 s; carried into v36)
Profiling the 5-10 ms pairs showed the threads' shared table costing 1-5 ms per fit to
allocate (8 regions x 65536 slots x key width), so one table per (threads, key width) is
now kept across fits and only its used flags and value column are reset before the threads
start (the reset happens-before every thread start, which keeps the lock-free reads sound:
a value read after a key match is NaN or the optimum of exactly that subproblem, and never a
value left by a previous fit).  Capacity 16384 slots per thread.  The hand-off moved from
5 ms to 3 ms (coupon and tic-tac-toe λ=0.02 0.85x in repeated timings, nothing slower).
Repeated subset timings: car λ=0.01 0.83-0.89x, coupon λ=0.005 0.85x, tic-tac-toe λ=0.02
0.91x, fico_binary λ=0.01 0.89-0.96x, car λ=0.005 0.95x, rest neutral.  The single-fit
recorded run did not show it (car λ=0.01 12.1 ms against 7.7 ms in every repeat; monk_2
λ=0.01 4.9 against 3.6), so by the loop's rule it is a discard; the changes are exact and
are kept in v36, whose recorded run decides for both.

## v36 — root frame re-armed from the cache at the hand-off (discard on the recorded number: geo 0.00587 / 0.00590 s in two runs vs v34 0.00580 s; carried into v37)
At the hand-off the parent re-expanded the root frame (kernel stages over every feature
pair) before starting the threads; on wide data that costs more than the sequential phase
itself (fico_1k λ=0.1: 6.3 ms of a 31 ms fit).  The v31 frame cache still holds the root's
expansion, so the hand-off now re-arms it (budget-dependent steps only).  Repeated timings:
fico_1k λ=0.1 0.80x, fico_binary λ=0.005 0.93x, the rest neutral; exact on every check.
Two recorded runs were nevertheless slower than v34 on fico_binary λ=0.005 (292 and 260 ms
against 178-211 in repeats), car λ=0.01 and monk_2 λ=0.01.  Investigation (harness loop
restricted to a few data sets, alternating versions, and fresh-process first-fit timings):
* monk_2 λ=0.01 is real: the 3 ms hand-off of v35 makes a 4 ms fit enter the parallel
  phase, which costs about 1 ms more than it saves -> fixed by the gate of v37;
* the heavy pairs are bimodal: one root task spans the whole parallel phase on fico_binary,
  and its wall time depends on whether it runs on a performance or an efficiency core
  (the machine has 4 + 6); in isolation v36 is faster than v34 on fico_binary
  (195-197 vs 204-222 ms) and tic-tac-toe.  Single-fit recorded runs therefore move by
  +-15% on such pairs, and the loop's decisions from here on use alternating repeats of
  the harness loop on the affected data sets in addition to the recorded run.

## v38x — half-tasks for the first root candidates (not recorded: mixed)
Each of the first two root candidates is solved as two tasks, one per child, each against
the budget bound - lb(sibling) (both solved -> exact split value; one proven above its
budget -> the split is above the incumbent), so the longest task no longer bounds the
phase.  Exact (forced-parallel exhaustive 0 mismatches, also with every candidate split).
Harness-loop timings vs v37: fico_binary λ=0.01 0.72-0.76x, λ=0.005 0.92-0.96x (the tail
case it was built for), but tic-tac-toe λ=0.01 1.14-1.26x, compas λ=0.05 1.09x, fico_1k
λ=0.1 1.10x: a split candidate loses the sibling deepening (the first child's raised bound
tightens the second's budget), and where the candidate is not a tail that costs more than
the balance gains.  Splitting every candidate is a disaster (tic-tac-toe λ=0.005 4x).  A
policy that recognises tail candidates in advance would be needed; kept in
optimal_tree_lib/v38x_half_tasks_not_run.py.

## v37 — hand-off gated on the root's remaining work (keep: geo 0.00533 s vs v34 0.00580 s, but see the caveat; solved-only 0.00221 vs 0.00224)
At the end of a sequential chunk the root frame's candidate position gives the completed
and remaining candidates; the threads start only if at least two remain and the remaining
work extrapolated from the elapsed time exceeds twice the parallel set-up cost (4 ms).
This removes the 1 ms the 3 ms hand-off cost the 4 ms fits (monk_2 λ=0.01) while keeping
the earlier start for everything larger.  Carries v35 (pooled table) and v36 (cached root
re-arm).  Exact on every check.  Recorded run: car λ=0.005 0.76x, fico_1k λ=0.1 0.80x,
tic-tac-toe 0.84-0.95x, fico_binary λ=0.005 0.88x, compas λ=0.05 0.96x, small pairs
unchanged.  Caveat: the all-pairs geo mean is flattered by the three compas pairs stopping
at the memory cap after 0.7, 9.3 and 26.6 s instead of running to 30 s.  That stop is an
artefact: the guard reads the process RSS, and on macOS freed large arrays stay in the
allocator's cache (a 2 GB array freed leaves RSS unchanged), so a fit inherits the RSS of
the previous ones.  v39 replaces the measure by the solver's own live allocations; the
fair comparison with v34 is v39's recorded run.

## v39 — memory guard on the solver's own allocations (keep: geo 0.00559 s vs v34 0.00580 s; solved-only 0.00219 vs 0.00224)
The guard now sums the bytes of the main store, of every thread's store (each thread
updates its slot of a shared array when its store grows) and of the pooled table, instead
of reading the process RSS.  Reason: on macOS a freed large array leaves RSS unchanged (the
allocator keeps it), so in a 70-pair run later fits inherited the RSS of earlier ones and
were stopped "for memory" within a second (v37: compas λ=0.005 after 0.7 s).  The cap
(6 GB) and its meaning are unchanged; the measure is deterministic.  With it the three
compas pairs stop at a genuine 6 GB after 15.0, 20.8 and 27.6 s (eight private stores of
a 12k x 621 problem; RSS is about 1.6x that because store growth copies), which accounts
for about 1.6% of the geo-mean gain over v34; the remaining ~2% is the v35-v37 chain
(pooled table, cached root re-arm, gated 3 ms hand-off): car λ=0.005 0.64x, coupon
λ=0.005 0.76x, fico_1k λ=0.1 0.78x, tic-tac-toe λ=0.02 0.83x, fico_binary λ=0.005 0.83x,
λ=0.01 0.87x, tic-tac-toe λ=0.01 0.91x; pairs under 6 ms unchanged.  Exact on every check.

## v40x — adaptive depth-3 gate by measured pay-off (not recorded: cannot discriminate)
Ablating the depth-3 stage gates (TRIPLE_MAX_NV, D3_MAX_KW, D3_MAX_LEAVES, D3_MAX_COUNT_LAM)
in the harness loop showed the stage costing car and coupon 25-35% and every other gate
change neutral or worse.  A run-time gate that counts the stage's runs and pay-offs (node
resolved by its bounds, incumbent improved, child solved through the 4-leaf rule) and
switches it off below one win in eight turned out useless: on car nearly every run "wins"
(children solved by the rule would be solved cheaply anyway) and on tic-tac-toe,
fico_binary and iris the stage never runs at all (their gates exclude it: nv > 14 until deep,
or K * W > 64), so the differences seen there were noise.  Kept in
optimal_tree_lib/v40x_adaptive_depth3_not_run.py.

## v40 — depth-3 triple stage off (keep: geo 0.00542 s vs v39 0.00559 s; solved-only 0.00212 vs 0.00219)
The stage of v20 (exact 3-leaf and (2,2) optima of every child from the feature triples,
with the 4-leaf exact rule they enable) paid off under the Python engine, where each
expansion it saved cost hundreds of microseconds.  Under the compiled parallel engine an
expansion costs a few microseconds and the stage's nv^3/6 triple kernels cost more than
what they save: in the harness loop with the stage off, car 0.6-0.7x, monk_2 0.7-0.85x,
monk_3 0.58-0.88x, coupon 0.8-0.95x, and nothing slower - it never ran on the wide or the
large data sets anyway (nv > 14 until deep, or K * W > 64).  TRIPLE_MAX_NV now defaults to 0;
the code path stays and can be re-enabled by the environment.  Recorded run: car λ=0.01
0.59x, monk_2 λ=0.01 0.85x, car λ=0.005 0.88x; fico_binary moved +6-13% in this run, the
tail-task bimodality noted under v36 (the stage does not run there).  Exact on every check
(the small-tree rule falls back to the depth-2 values, as before v20).
Lesson worth keeping: a stage's sign depends on the engine's cost model; every stage
adopted under the Python engine should be re-ablated under the compiled one (next: similar
support, feature exchange, look-ahead deepening, greedy start).

## v41 — node-keyed frame cache (discard: geo 0.00551 s vs v40 0.00542 s; solved-only 0.00214 vs 0.00212)
Measurement first: 26-36% of the expansions on fico_binary, coupon, car and monk_2 (12-15%
on tic-tac-toe and iris) are repeats of an already expanded node (budget deepening
re-enters children; chunk re-entries restart from the root), and the two-slots-per-depth
cache of v31 re-arms only about 1% of expansions.  v41 replaces it by a pool of frame slots
keyed by node (FIFO eviction, never a slot on the stack, open-addressing node -> slot map)
and, decisively, caches a frame right after its kernels rather than at the end of the
expansion, so frames that leave through a budget-dependent early return can be re-armed.
Two lessons from that:
* The first version of the early cache produced a false certificate (forced-parallel
  exhaustive trial 81: 0.3714 certified against an optimum of 0.3500).  The re-arm path
  adopted the depth-2 value as the node's bound without installing the depth-2 structure
  in the store, which the full expansion always does first; before v41 only fully expanded
  frames could be re-armed, so the omission was unreachable.  Fixed by mirroring the
  install in the re-arm (and handling a full store there).  The exhaustive validations are
  the reason such a change can be attempted at all; 420 forced-parallel trials over two
  seeds pass after the fix.
* A frame whose pair stage did not run (it runs only when enough candidates survive the
  cheap filter at the current bound) must not be cached early: re-arming it at a larger
  budget forgoes the pair bounds a fresh expansion would compute (car did 43% more
  expansions).  Only frames with the pair stage are cached early.
Re-arms rise to 10-15% of expansions.  Harness-loop timings: tic-tac-toe λ=0.005 0.92x,
iris 0.92-0.93x, fico_binary λ=0.01 0.85-0.9x, monk_2 and car λ=0.01 0.8-0.87x, but coupon
λ=0.005 1.15-1.3x, car λ=0.005 1.1-1.2x, fico_1k λ=0.1 1.06-1.1x - consistent with the pool
cycling frames through 9 MB of workspace per thread instead of 420 KB (cache locality);
a 0.5 MB pool is neutral everywhere.  Recorded run: the same split, geo slightly worse
(the compas pairs also stopped later at the memory cap).  Kept in
optimal_tree_lib/v41_node_keyed_frame_cache.py; the fix to the re-arm path is only needed
with the early cache and is not in v40.

## v41x — deepening schedule variants (not recorded: negative)
Ablating look-ahead deepening in the harness loop: without it iris and tic-tac-toe are
100-180x slower, but fico_binary λ=0.005 runs in 135 ms instead of 230-260 (its children
are cheaper to solve outright than through five doublings of a 2λ step).  Variants tried
against v40: no steps but sibling-tightened budgets (fico_binary 0.8x, tic-tac-toe 3x
slower), first step = max(2λ, slack/4) (neutral to worse), at most 1 or 2 deepening rounds
(worse everywhere).  The 2λ-doubling schedule stays; nothing in the node distinguishes the
fico_binary case in advance.

## v42x — thread count, quality-of-service classes, hardware popcount (not recorded: no lever)
* Threads: 4 and 6 are far slower than 8 on tic-tac-toe (152 and 78 ms vs 66 at λ=0.01),
  10 (all cores) is no better than 8.  8 stays.
* macOS QoS: all workers user-interactive is neutral; 4 user-interactive threads taking
  the promising candidates from the front of the order and 4 utility-class threads taking
  from the back is worse (tic-tac-toe λ=0.01 97 vs 66 ms).  The efficiency-core tail on a
  single big task cannot be steered this way.
* Kernels: the software popcount is already compiled to the hardware instruction and the
  pair loops are vectorised (a 40-feature pair sweep costs the same at 3 and at 15 words;
  hoisting the row intersection out of the class loop is neutral or slower).  The
  instruction-level side of the kernels is done; only fewer expansions can help now.
Kept in optimal_tree_lib/v42x_qos_threads_not_run.py (the QoS / thread-count development
copy of v40).

## Where the loop stands (compiled, 8 threads)
Best: v40 (geo 0.00542 s, solved-only 0.00212 s, 63/70, exact, multicore) - from v30
0.00603 (first thread version), v28 0.00645 (first compiled), v23 0.0081 (best Python).
The 7 unsolved pairs (fico_1k λ<=0.05, compas λ<=0.02) dominate the geo mean (7 x log 30 s)
and 45 of the 63 solved pairs sit under the 2 ms floor; the leaderboard is decided by
about 18 pairs between 2 and 300 ms.  Measured limits on those: the threads do 1.4-1.6x
the sequential expansions on tic-tac-toe and fico_binary (private memos; the shared table
catches the late duplicates only), a single root task can span the whole parallel phase
(fico_binary) and its wall time depends on the core it lands on, and single-fit timings
move by +-10-15% run to run for that reason - decisions in this stretch used the harness
loop with alternating repeats on the affected data sets.

## v40_sequential — v40 with n_jobs=1 (keep, sequential line: geo 0.00618 s, 63/70, exact, one core)
Recorded on the tree-blog session's request for the library: v28 0.00645 s, v23 0.00799 s in
the same file.  All seven unsolved pairs stop with status "time" at 30 s and return the best
known objective.  The gains over v28 (car 0.5x, compas λ=0.05 0.8x, fico_1k λ=0.1 0.9x) are
the depth-3 stage removal and the frame cache, neither multicore-specific.

# Phase 3 — approximate, near-optimal trees: the criterion-versus-time frontier
The user redirected the loop: the tree need not be certified optimal but must be very close;
optimise the Pareto curve of training criterion versus training time, grounded by the
approximate baselines (gosdt_guesses_guided: geo 0.0011 s, mean regret 0.044; split: 0.131 s,
0.056, no tree on 13 pairs), with nothing tuned to these data sets.  Two columns were added to
both leaderboards (src/evaluate.py, backfilled): mean_objective (training criterion of the
returned tree, single-leaf tree where none) and mean_regret (criterion minus the best known
objective, src/known_optima.csv, refreshed with the values the exact versions certified).
Starting point: v40 has geo 0.00542 s and mean regret 0.0000 (its seven uncertified pairs
return the best known objective at 30 s).
Anytime measurement (v40 under short caps): on all seven uncertified pairs the incumbent at
the earliest return already equals the best known objective; tic-tac-toe λ=0.005 is optimal
by 50 ms.  But the earliest return is 0.13-0.45 s on the wide data because the first search
chunk is 500 expansions and later chunks aim at 50 ms whatever the cap.  So the first
approximate line is an anytime mode whose chunking scales with the cap, with the cap as the
knob that traces the frontier.

## v43 (in progress) — anytime mode: cap-scaled chunking and a lookahead-greedy seed
Development findings (exact on every check, sequential and forced-parallel):
* Chunking: the first sequential chunk was 500 expansions and later chunks aimed at 50 ms,
  so a 20 ms cap returned after 0.13-0.45 s on the wide data.  Now the first chunk is 4
  expansions and every chunk (sequential and per thread) aims at min(50 ms, cap/20); caps
  are honoured within a few per cent except for the cost of the root expansion and the seed.
* Seed: the compiled engine had no starting tree at all (the greedy_init option was inherited
  but unused), so early incumbents were the shallow trees the budget-limited descents leave
  (tic-tac-toe λ=0.005 at 20 ms: 0.242 vs 0.154).  A lookahead-greedy seed now runs before
  the search: at each node the frame expansion's best tree (leaf, split with leaf children,
  or the pair stage's best two-level tree) picks the split, both children are seeded, and
  the greedy subtree value is installed when it beats the node's own; every structure it
  writes is consistent, so the search starts from a real incumbent.  Resumable and
  time-bounded (two expansions to measure the cost of one, then as many as fit a quarter of
  the cap, at most 0.1 s and 256 expansions).  Early regrets halve (tic-tac-toe λ=0.005
  0.088 -> 0.038 at 20 ms; λ=0.01 optimal at 50 ms; car λ=0.005 optimal at 16 ms; iris λ=0.005
  optimal at 40 ms).
* In exact mode (30 s cap) the seed is a big win on iris (λ=0.01 22 -> 11 ms, λ=0.005 70 -> 33
  ms) but tic-tac-toe λ=0.005 and fico_binary λ=0.01 got 1.25-1.35x slower in the harness
  loop.  Hypothesis to test: a tight incumbent at the hand-off prunes most root candidates,
  leaving few, large tasks for eight threads (the tail problem of v36 again).
Anytime rows planned: v43 capped at 10, 20, 50 and 100 ms per pair (exact = approximate).

## v43 — lookahead-greedy seed and cap-scaled chunking (keep on the exact line: geo 0.00529 s vs v40 0.00542 s, 63/70, regret 0)
Exact on every check.  A controlled comparison with the seed on and off (two fits each, same
process) settled the earlier doubt: the seed costs 0.1-0.4 ms and the exact search is faster
or equal everywhere with it (iris λ=0.005 57 -> 30 ms, fico_binary λ=0.005 208 -> 181 ms,
tic-tac-toe equal); the harness-loop slowdowns seen before were noise.  The number of root
tasks at the hand-off is unchanged (the incumbent prunes candidates inside the tasks, not
before them).

## v43 anytime rows — the first points of the criterion-versus-time frontier (status "frontier")
v43 with the per-pair time limit capped (exact = approximate: the tree is the incumbent at the
cap, certified only when the search finished earlier).  Recorded twice: first with the cap
honoured loosely on wide data (fico_1k returned after 150-190 ms under any cap because the
hand-off built 1200 root tasks for eight threads), then after three general fixes (no
hand-off when fewer than four hand-off costs remain, threads check the deadline before each
task, their first chunk is 4 expansions).  Final rows:
| cap    | geo time  | certified | mean regret | pairs above the known optimum |
| 100 ms | 0.00318 s | 61/70     | 0.0003      | fico_1k λ=0.01, 0.005 (+0.007 / +0.017: 2-leaf trees) |
|  50 ms | 0.00285 s | 59/70     | 0.0008      | + tic-tac-toe λ=0.01, 0.005 (+0.0035 / +0.0375) |
|  20 ms | 0.00247 s | 56/70     | 0.0014      | same four |
|  10 ms | 0.00220 s | 52/70     | 0.0017      | + tic-tac-toe λ=0.02 |
The remaining floor is the root expansion itself (fico_1k: 25 ms for the pair stage over
1357 features), so a 10 ms cap returns after 40 ms there.  Grounding: split 0.131 s at regret
0.101 (13 pairs without a tree score the single-leaf tree), gosdt_guesses_guided 0.0011 s at
0.044, v43 exact 0.0053 s at 0.  Every anytime row dominates split; the 10 ms row is 2x
slower than gosdt_guesses_guided with 26x less regret.

## v44x — threshold guessing (numpy stump boosting, 40 / 100 rounds): measured, not recorded
Exact search on the binary features a boosted-stump ensemble uses (as gosdt-guesses does),
reported as approximate.  At the 30 s cap: fico_1k λ=0.05 solved in 4 ms with regret 0, but
λ=0.01 and 0.005 still run to the cap (regret +0.007 / +0.016 at 40 rounds, +0.005 at 100),
compas +0.0027, iris +0.0067 (both round counts), sine_1k +0.080 (40) / +0.033 (100): the
optimum of sine_1k needs thresholds the ensemble does not pick, and the plain anytime rows
solve iris and sine_1k exactly within 40 ms.  Guessing would only lower the wide data sets'
floors at small caps (about +26% on the geo mean of a 10-20 ms row) at the price of about
+0.004 mean regret; it is kept as an option (guess_thresholds=E in v44) for a possible extra
frontier row, not as a default.  The small-cap floor on fico_1k turned out to be the
parallel hand-off itself (1200 root tasks for 8 threads under a 10 ms cap, 170 ms instead
of 41 ms single-threaded), fixed in v43 by a remaining-time gate and a per-task deadline check.

## v43 anytime 300 ms and 1 s rows (frontier): regret 0 at geo 0.00358 s and 0.00396 s
Both certify 63/70 like the exact run and return the best known objective on the seven
uncertified pairs (their incumbents reach it within 300 ms), so on this suite a 300 ms cap
costs nothing in criterion and saves a third of the exact geo mean (0.00529 s).  The frontier
now reads (geo time s, mean regret): 10 ms (0.00220, 0.0017), 20 ms (0.00247, 0.0014), 50 ms
(0.00285, 0.0008), 100 ms (0.00318, 0.0003), 300 ms (0.00358, 0), 1 s (0.00396, 0), exact
(0.00529, 0); gosdt_guesses_guided (0.0011, 0.044) and split (0.131, 0.101) for reference.

## v43bx — beam of alternative root splits for the seed (not recorded: no effect)
Seeding the next-best root splits (by two-level value) with greedy children and keeping the
best tree changes nothing on any pair (tic-tac-toe's early incumbent stays 0.1918 against
0.1543): the root choice is not what the lookahead greedy gets wrong there, the deeper
choices are, and the exact search repairs them by 100 ms.  Kept in
optimal_tree_lib/v43bx_beam_seed_not_run.py.

## v43c — pair-stage budget for short caps (frontier rows re-recorded at 5 ... 1000 ms)
Under a short cap the remaining floor was the root expansion itself: the pair kernel over
1357 features costs 25 ms, so a 10 ms cap returned after 40 ms on fico_1k.  An expansion now
skips the pair stage when its estimated cost (nv^2 * K * W word operations at the measured
0.4 ns each) exceeds a tenth of the time limit.  The rule never binds at the 30 s cap (the
largest root here is 6e7 operations against a budget of 7.5e9), so the exact line is
untouched, and validation passed again on every check; under a cap it only leaves some
bounds uncomputed, never changes one.  Caps are now honoured on every pair (5 ms cap: max
6 ms; 10 ms: 11 ms; 20 ms: 25 ms; 50 ms: 54 ms).  Recorded rows and the resulting frontier
(status computed by dominance among all approximate rows):
| row                | geo time  | certified | mean regret | status    |
| v43c 5 ms          | 0.00171 s | 50/70     | 0.0017      | frontier  |
| v43c 10 ms         | 0.00205 s | 52/70     | 0.0017      | dominated |
| v43c 20 ms         | 0.00243 s | 56/70     | 0.0014      | frontier  |
| v43c 50 ms         | 0.00284 s | 59/70     | 0.0009      | frontier  |
| v43  50 ms         | 0.00285 s | 59/70     | 0.0008      | frontier  |
| v43c 100 ms        | 0.00318 s | 60/70     | 0.0003      | frontier  |
| v43  300 ms        | 0.00358 s | 63/70     | 0.0000      | frontier  |
| v43c 300 ms / 1 s  | 0.00376 / 0.00406 s | 62-63/70 | 0.0000 | dominated (noise: same search) |
The regret left at short caps sits on five pairs: tic-tac-toe λ<=0.02 (the lookahead greedy's
deeper choices; the exact search repairs them by 100 ms) and fico_1k λ<=0.01 (2-leaf incumbents
at 5-100 ms; the best known trees need seconds).  The geo mean at 5 ms is within 4% of the
floor a 5 ms cap allows on this suite (45 pairs sit at the 1 ms floor), so lower caps are the
only way further left; 1 ms and 2 ms rows follow.

## v43c 1 ms and 2 ms rows: the fast end of the frontier
| row        | geo time  | certified | mean regret | pairs above the known optimum |
| v43c 1 ms  | 0.00108 s | 46/70     | 0.0024      | 11 (iris, monk_2, tic-tac-toe, fico_1k, car, coupon at the smallest λ) |
| v43c 2 ms  | 0.00131 s | 47/70     | 0.0018      | 8 |
The 1 ms row matches gosdt_guesses_guided's geo mean (0.0011 s) with 18x less mean regret
(0.0024 vs 0.044) and returns a tree on every pair; split (0.131 s, 0.101) is dominated by
every row.  Caps are honoured (max 1.6 ms under the 1 ms cap).

## Where the frontier stands (all rows exact = approximate except the last)
| geo time  | mean regret | row               |
| 0.00108 s | 0.0024      | v43c 1 ms         |
| 0.00131 s | 0.0018      | v43c 2 ms         |
| 0.00171 s | 0.0017      | v43c 5 ms         |
| 0.00243 s | 0.0014      | v43c 20 ms        |
| 0.00284 s | 0.0009      | v43c 50 ms        |
| 0.00318 s | 0.0003      | v43c 100 ms       |
| 0.00358 s | 0.0000      | v43 300 ms        |
| 0.00529 s | 0.0000      | v43 exact (certified 63/70) |
Everything on it is the same exact solver run as an anytime method: no bound was relaxed, no
data-set-specific rule was added.  The knob is the per-pair time cap; the machinery that makes
short caps meaningful (chunking scaled to the cap, a time-bounded lookahead-greedy seed,
hand-off rules, a pair-stage budget from a cost model) is general.  Threshold guessing (v44x)
and a root beam for the seed (v43bx) were measured and rejected.

# Phase 4 — looking for a breakthrough speed-up: where the remaining time is, and why
Profile of the hard pairs (single thread, exact v43):
* fico_1k λ=0.05, 6 s: 247k expansions - 1 at the root, 3 at depth 2, 174 at depth 3, 13.7k
  at depth 4, 227k at depth 5; every expanded node has >= 300 of the 1000 rows; not one
  expanded node is certified; the root bound stays at 0.125 against an incumbent of 0.391.
* compas λ=0.02, 6 s: the same shape (124k of 167k expansions at depth 6); fico_binary
  λ=0.005 (certified in 30 s single-threaded) by contrast certifies 21.7k of 34k expanded nodes
  and its expansions spread over depths 2-10.
* With 8 threads and a 300 s cap, fico_1k λ=0.05 reaches 14.1 million expansions before the
  6 GB memory guard stops it at 124 s, root bound still 0.125, not certified.
What the optimum is: on all seven uncertified pairs the best known tree is a stump (two
leaves: fico_1k 0.391 = 0.291 + 2λ at λ=0.05, 0.331, 0.304, 0.284 at the smaller λ; compas
0.328, 0.308, 0.298).  FICO and COMPAS are noisy: extra splits reduce the error by less than
their λ, so the answer is small and what takes forever is the proof that no tree with 3 to
1/λ leaves beats it.  With λ = 5-50 rows' worth of error and essentially no duplicate rows,
the only error floor available is the λ-per-leaf term (the equivalent-points floor is 0 on
fico_1k, and a node-local version would still be 0: every numeric column keeps dozens of
splitting thresholds deep in the tree), so certifying the stump means enumerating every
4-split path over 1357 thresholds and, for λ=0.005, every tree with up to 200 leaves whose
error could undercut 27.4% by 1.5 points.  The exact pair stage already gives every node its
exact 2-, 3- and (2,2)-leaf values, so the enumeration starts at 5 leaves; the missing piece
would be a lower bound on the error of k-leaf trees for k >= 5 that does not come from
repeated rows, and no such bound exists in the literature for this objective (GOSDT, STreeD
and gosdt-guesses all time out on the same pairs).  A several-fold engine speed-up would not
change the outcome: the search is 10-100x short of finishing.
Consequences for the leaderboard: the certified geo mean is 7 x log(30 s) plus 45 pairs at
the 1 ms floor plus about 18 pairs in between; the ideas that could still move it (shared
memo across threads, half-tasks, adaptive deepening, kernel micro-optimisation, thread QoS)
were all measured in phases 2-3 and gave at most a few per cent or nothing.  For the regime
of the hard pairs the useful product is the anytime line: the 300 ms row returns the same
stumps with regret 0 in 0.3 s instead of 30 s.

# Phase 5 — greedy algorithms to help the search (v45)
Where heuristics can pay: incumbents barely help certification (the search is bound-limited;
feeding it the optimum as an upper bound changed expansions by <10%), but they decide the
regret at short caps, where the lookahead greedy of v43 left tic-tac-toe at 0.192 against
0.154 until the exact search repaired it at 100 ms.  v45 adds two general heuristics, both
using the existing frame expansion and store, both active only under caps up to 1 s (under
long caps the search decides and they are overhead; at 30 s the file behaves as v43c):
1. A family of greedy seeds.  Besides the lookahead greedy (the expansion's best tree chooses
   the split), a gain greedy: the split with the largest impurity gain (gini; entropy and
   error are available) from the per-split class counts the expansion already computes,
   grown past unprofitable splits, then pruned bottom-up under lambda when the tree is
   installed (a node keeps its subtree only where it beats the node's own value).  Each seed
   installs only where it improves the store, so the better tree wins node by node.
2. A bottom-up exact repair of the incumbent (large-neighbourhood search with the solver
   itself): every internal node of the incumbent tree, smallest first, is solved as a
   subproblem with its current value as the budget, so it ends exactly optimal or proven
   so; improvements propagate to the ancestors' values; time-bounded (40% of the cap, at
   most 0.2 s), and everything it certifies stays in the store for the main search.
Measured on the regret pairs (2 / 5 / 20 / 50 ms caps): tic-tac-toe λ=0.005 0.0375 -> 0.0016
at 2 ms and 0 at 5 ms; λ=0.01 0.033 -> 0.008 / 0.0035; car λ=0.005 optimal at 2 ms (was 20 ms).
The gini greedy alone brings tic-tac-toe to 0.169; the repair brings it to the optimum.
Costs found and fixed on the way: v45 was first built without the pair-stage budget of v43c
(wide-data seeds paid the full root pair kernel, 24 ms on fico_1k under a 2 ms cap), and the
repair used the default 50 ms chunk target because the target was computed after the seed
block; with both fixed every cap is honoured (fico_1k and compas return in 2.0 ms under a
2 ms cap).  Exact on every check.
Recorded v45 rows (all 70 pairs, caps honoured: max 2.9 ms under the 2 ms cap, 5.5 under 5,
10.5 under 10, 22 under 20, 54 under 50, 106 under 100):
| cap    | geo time  | certified | mean regret | v43c at the same cap |
|   2 ms | 0.00133 s | 48/70     | 0.0008      | 0.00131 s, 0.0018 |
|   5 ms | 0.00170 s | 51/70     | 0.0008      | 0.00171 s, 0.0017 |
|  10 ms | 0.00203 s | 52/70     | 0.0007      | 0.00205 s, 0.0017 |
|  20 ms | 0.00240 s | 55/70     | 0.0004      | 0.00243 s, 0.0014 |
|  50 ms | 0.00280 s | 59/70     | 0.0004      | 0.00284 s, 0.0009 |
| 100 ms | 0.00309 s | 61/70     | 0.0003      | 0.00318 s, 0.0003 |
Same time, 2-3.5x less regret from 2 to 50 ms; the frontier is now v43c 1 ms (0.00108, 0.0024),
v45 2-100 ms, v43 300 ms (0.00358, 0) and the exact line.  What is left at short caps is
fico_1k λ<=0.01 (the best known trees there are not what any greedy finds; the repair has
no subtrees to work on in a stump) and tic-tac-toe λ=0.01/0.02 at the smallest caps.
Exact-mode row of v45 (30 s cap, extras off): geo 0.00532 s, 63/70, regret 0, solved-only geo
0.00208 s identical to v43's - the same search; the run folder's solver is now this file.

# Phase 6 — loop continued on the frontier: v46 (budgeted pair stage, early install, leaf repair)
What was left at short caps: fico_1k λ<=0.01 (+0.007 / +0.017 up to 100 ms, 0 at 300 ms).
Their best known trees are 4-leaf two-level trees (root NumSatisfactoryTrades>=23, both
children split on ExternalRiskEstimate), which the pair stage finds exactly - but under caps
below 240 ms the per-expansion pair budget skipped the root's full stage on 1357 features.
Steps, each measured:
* Budgeted pair stage: when the full stage is over budget, evaluate the k splits with the
  largest gini gain against all others (a new kernel, depth2_topk) for the incumbent only -
  no bounds are derived from it (lb_ge4 = -1 signals it) - charged to a global 20% budget
  per fit.  Exact: validated with the budgeted stage forced at every node (PAIR_TIME_FRAC=1e-9).
  Alone it did not find the fico_1k trees (ranking by leaf-sum or by gain misses the root
  split whose value comes from its children).
* The root may spend 2.5 per-expansion shares of the cap on its full pair stage (it is
  expanded once and cached), so from 50 ms up the root sees the two-level tree.
* The decisive fix was a bug: an expansion's best tree (leaf-sum split or two-level tree,
  children installed) was only written to the node when the candidate loop ended, so the
  seed's root expansion found the 4-leaf optimum and lost it (the seed's budget ends after
  two expansions on wide data, and the cached re-arm starts from the node's stored value).
  The node now takes the expansion's best tree immediately; this also gives the exact search
  its root incumbent at the first expansion.
* The repair also solves the incumbent's leaves as subproblems (a stump has no internal
  nodes); the threads' chunk floor is 4 expansions (it was 50: fifty wide expansions past
  the deadline).
Recorded rows (v46, 8 threads): 1 ms (0.00106 s, regret 0.0013), 2 ms (0.00132, 0.0006),
5 ms (0.00174, 0.0008), 10 ms (0.00214, 0.0006), 20 ms (0.00269, 0.0003), 50 ms (0.00293,
0.0001), 100 ms (0.00305, 0.00003), 300 ms (0.00351, 0.00003).  fico_1k λ=0.005 reaches its
optimum at 50 ms (was never below 300 ms); the 100 ms row has regret 0.00003 (61/70
certified) against 0.0003 before.  The frontier: v46 1 ms, v46 2 ms, v45 20 ms, v46 20 ms,
v46 50 ms, v46 100 ms, v43 300 ms, exact.  Open: the 300 ms cap still overshoots on fico_1k
(0.55 s), to be traced.

## v46 incident — a wrong certificate that the validations did not catch
The first recorded v46 rows had n_wrong = 2 (monk_2 λ=0.01: a 19-leaf tree of 13 errors
certified against the true 20-leaf optimum of 11; tic-tac-toe λ=0.01: 8 leaves against 9),
in the exact row and in the 50, 100 and 300 ms rows, while every validation (exhaustive
sequential and forced-parallel trials, cost matrices, semantics) had passed.  Bisection by
reverting each change one at a time (early install, chunk sizing, budgeted stage, seeds,
root multiplier) cleared none of them; the diff did: when the cache branch for the budgeted
stage was added after the full pair stage's ``FI_VALID = 1``, the two lines that refilter
the candidate list under the pair stage's bounds were left indented into the new branch,
so in exact mode the refilter no longer ran.  Two lessons recorded as rules:
* every recorded row, approximate rows included, is checked for n_wrong before it gets a
  status (the anytime rows certify when they finish early, so they can be wrong too);
* the real-pair list of the forced-parallel validation now includes monk_2 λ=0.01,
  tic-tac-toe λ=0.01, monk_3 λ=0.01, car λ=0.01, coupon λ=0.01 and monk_1 λ=0.005 - pairs
  small enough to run in the validation yet with enough structure (10-20 leaves) that an
  omitted filtering step shows.  The 150 + 300 random exhaustive trials (n <= 40 rows,
  p <= 3 features) did not exercise it.
Fixed (the refilter is back under the full pair stage; the budgeted stage's branch gets no
refilter since its two-level values are not bounds); all validations pass again on the
fixed file; every v46 row is re-recorded from it.

## v46 final rows (fixed file; every row n_wrong 0)
| cap    | geo time  | certified | mean regret | v45 at the same cap |
|   1 ms | 0.00104 s | 46/70     | 0.0013      | 0.00109, 0.0016 |
|   2 ms | 0.00131 s | 48/70     | 0.0006      | 0.00133, 0.0008 |
|   5 ms | 0.00171 s | 51/70     | 0.0008      | 0.00170, 0.0008 |
|  10 ms | 0.00215 s | 52/70     | 0.0007      | 0.00203, 0.0007 |
|  20 ms | 0.00270 s | 55/70     | 0.0003      | 0.00240, 0.0004 |
|  50 ms | 0.00297 s | 59/70     | 0.00005     | 0.00280, 0.0004 |
| 100 ms | 0.00312 s | 61/70     | 0.0000      | 0.00309, 0.0003 |
| 300 ms | 0.00352 s | 63/70     | 0.0000      | - |
| exact  | 0.00528 s | 63/70     | 0.0000      | 0.00532 (same search) |
Frontier now: v46 1 ms, v46 2 ms, v45 20 ms, v46 20 ms, v46 50 ms, v46 100 ms (regret 0 at
0.0031 s: the 300 ms rows are dominated), then the exact line.  The exact v46 row equals
v43/v45 (solved-only geo 0.00207 s); the run folder's solver is the v46 file (exact by
default; the extras and the budgeted stage engage only under caps).

# Phase 7 — speed with regret allowed, and generalisation beyond the suite (v47, v48)
## v47x — the gosdt-guesses recipe inside our search (recorded as a reference point, dominated)
Options added to the solver (all off by default): threshold guessing (the binary features a
boosted-stump reference model uses; SAMME with weighted-majority stumps on the binarised
data, numpy only), reference lower-bound guessing (the reference model's loss on a
subproblem's rows + lambda replaces the equivalent-points floor at node creation, as in
McTavish et al. 2022), and a depth budget.  Suite results at the 30 s cap, geo time / mean
regret: full recipe (40 rounds, reference bound, depth 5) 0.00189 s / 0.011; reference bound
alone 0.0048 / 0.0089; 100 rounds + reference bound 0.0031 / 0.0099; depth 5 alone 0.0052 /
0.00007; recipe under a 20 ms cap 0.00153 / 0.0105.  gosdt_guesses_guided itself: 0.0011 /
0.044.  Every one of these is dominated by the anytime rows (v46 2 ms: 0.0013 / 0.0006):
the reference bound is what costs regret - on monk_2 the stump ensemble is a poor
reference, its guessed bounds prune the real subtrees and the regret reaches 0.12-0.23 -
and the depth budget buys nothing on data the search already finishes.  Recorded as
v47_guesses_like for the leaderboard (status: dominated, reference point); the options
stay in optimal_tree_lib/v47_guesses_options.py.
Out of the suite (full FICO 10459 rows / 1883 thresholds, full COMPAS 7214 rows / 26401
binary features, full coupon 12684 rows / 118), objective after a 1 s cap, ours (anytime,
exact search) vs gosdt_guesses_guided: FICO λ=0.005 0.3088 vs 0.3101, λ=0.02 0.3401 vs 0.3401;
COMPAS equal (stumps); coupon λ=0.005 0.3561 vs 0.3559; the guesses recipe inside our search
was worse than our plain anytime on all three.  The search is not where we lose time there.

## v48 — vectorised binarisation (keep; the search is unchanged)
Profiling the out-of-suite fits: the search certified COMPAS λ=0.02 in 0.2 s while the fit
took 86 s of wall time, all in the encoder - each of the 64859 rules was applied as its own
pandas operation, and categorical rules as a Python loop over the rows (62 s), twice (the
duplicate-column pass transforms again).  gosdt-guesses' binariser takes 104 s on the same
table.  The encoder now applies one broadcast comparison per numeric column and integer
codes per categorical column: identical matrices, rules and groups on all 17 data sets
checked; COMPAS full 82 s -> 2.0 s, coupon full 0.43 -> 0.13 s.  Fit wall time on COMPAS full
86 s -> 6-7 s (the rest is BitDataset construction, profiled next), FICO full 1.4 s after
the once-per-machine compile, coupon 1.2 s.  All exactness checks pass (the encoder is
shared by every mode).
Follow-up in v48: the equivalent-points classes were found with np.unique over rows, which
sorts 3 KB records on COMPAS full (2.6 s); hashing the packed rows gives the same classes
in 0.05 s (identical majority/minority masks and cost tables on all 16 data sets checked).
Wall time per fit after the once-per-machine compile, anytime with a 1 s cap: COMPAS full
4.8 s (86 s before; 1.7 s of it is the search, the rest the duplicate-column pass, row
clustering and the bitset build), FICO full 1.2 s, coupon full 1.2 s.  gosdt_guesses_guided
on the same tables: 104 s, 3.7 s, 1.8 s (its GBDT guessing and binariser), with equal or
worse objectives.  The remaining preprocessing on very wide tables is the duplicate-column
pass (a second transform) and the per-column bit packing; both are linear and no longer
dominate.

## v49 — candidate-only pair stage, lazily grown workspaces, one binarisation pass (generalisation)
Measurements that motivated it: with the pair stage's outer loop over the surviving
candidates only, its work would be 24% of today's on iris, 76% on tic-tac-toe, 81% on
fico_binary (the kernel loops over every splitting feature, candidates or not); and the
per-thread frame workspace, 96 slots x the feature count, is 634 MB on the full COMPAS table
(x9 threads: the reason wide tables ran into the memory guard).
* depth2_cands: the pair kernel restricted to the candidates against every other split;
  non-candidates keep their bounds, their leaves stay their only achievable subtrees here,
  and trees with >= 4 leaves rooted at them are bounded by their own split bound (the
  shape bound takes a mask).  Sound, validated on every check.  First attempt ran it for
  all wide nodes with few candidates and lost 1.5-1.8x on fico_1k and compas (millisecond
  stages on nodes the old rule deliberately skipped) and broke the 1 s cap on full FICO;
  bounded to |candidates| * nv * W <= 131072 it keeps the wins and none of the losses:
  iris 2.2-2.7x faster (λ=0.005 36.8 -> 16.6 ms), the rest at parity.
* Workspaces start with 16 frame slots (depth 7) and double on demand (a new interruption
  code, handled like store growth by the sequential loop, the threads and the repair);
  the greedy seeds respect the slots available.  Full COMPAS: 106 MB per thread instead
  of 634 MB.
* The encoder's duplicate-column pass returns its matrix instead of binarising twice.
Out of the suite (anytime, 1 s cap): FICO full 1.2 s wall, COMPAS full 4.7 s (search 1.7 s).
Recorded (all n_wrong 0): exact v49 geo 0.00516 s (v46 0.00528; iris λ=0.02/0.01/0.005 at
0.36 / 0.56 / 0.54x), 63/70 - the new exact best; anytime 2 ms (0.00130 s, regret 0.0007),
5 ms (0.00168, 0.0008), 10 ms (0.00211, 0.0006), 20 ms (0.00258, 0.0003), all on or at the
frontier.  The run folder's solver is v49.
