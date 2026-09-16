"""
Evaluate the fixed baselines on the development suite and seed the leaderboard.

Baselines:
  gosdt        the reference C++ implementation (skipped if baselines/gosdt/build/gosdt is not built)
  pygosdt_v1   the pure-Python re-implementation in baselines/pygosdt_v1/
  streed       STreeD (baselines/pystreed, cost-complex-accuracy on the same binarization;
               needs `uv sync --group baselines` unless the cached rows are used)

Usage: uv run run_baselines.py [--rerun] [--skip-reference] [--run-model]

By default the baseline rows are taken from the full 600 s benchmark already in
baselines/benchmarks/results/pair_results.csv (times capped at the suite's cap, pairs counted as
solved only if certified within it), which takes a second.  --rerun evaluates
both baselines on the suite instead (~45 min with the reference binary).

Outputs (all under results/):
  overall_results.csv   one row per model: n_solved, geo_mean_time, n_wrong (leaderboard)
  pair_results.csv      one row per model x dataset x lambda

Not modified by the agent.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from evaluate import evaluate_solver, print_summary, record, rows_from_benchmark, summarize  # noqa: E402
from suite import MEMORY_LIMIT, RESULTS_DIR  # noqa: E402
import reference_solver  # noqa: E402

from pygosdt_v1 import GOSDTClassifier  # noqa: E402

def make_streed(lam, tl):
    import streed_solver  # imported lazily: needs the ``baselines`` dependency group
    return streed_solver.STreeD(lam, tl)


def make_gosdt_guesses(lam, tl):
    import guesses_solver  # the C++ core is imported in a child process; needs the ``baselines`` group
    return guesses_solver.GuessesGOSDT(lam, tl, memory_limit=MEMORY_LIMIT, guesses=False)


def make_split(lam, tl):
    import split_solver  # needs the ``baselines`` dependency group
    return split_solver.Split(lam, tl)


def make_gosdt_guesses_guided(lam, tl):
    import guesses_solver
    return guesses_solver.GuessesGOSDT(lam, tl, memory_limit=MEMORY_LIMIT, guesses=True)


# whether each baseline is an exact method (certifies optimality) or a heuristic
EXACT = {"gosdt": False,  # issues a false optimality certificate (tic-tac-toe λ=0.02)
          "pygosdt_v1": True, "streed": True, "gosdt_guesses": True, "gosdt_guesses_guided": False,
          "gosdt_mc8": False, "gosdt_guesses_mc8": True,
          "split": False}  # SPLIT is a lookahead heuristic with a depth budget
# whether each baseline uses more than one core (STreeD has no thread option; the reference
# GOSDT and gosdt-guesses take ``worker_limit``)
MULTICORE = {"gosdt_mc8": True, "gosdt_guesses_mc8": True}
WORKERS = 8

BASELINES = {
    "gosdt": (
        lambda lam, tl: reference_solver.ReferenceGOSDT(lam, tl, memory_limit=MEMORY_LIMIT),
        "reference C++ GOSDT (ICML 2020 code, default bounds, single thread) [NOT EXACT: scope-conditional lower bounds are cached unconditionally, verified false certificate on tic-tac-toe lambda=0.02 (baselines/gosdt_patches/scope-lowerbound.patch)]",
    ),
    "pygosdt_v1": (
        lambda lam, tl: GOSDTClassifier(regularization=lam, time_limit=tl, memory_limit=MEMORY_LIMIT),
        "pygosdt_v1 package: memoised depth-first branch-and-bound with the reference bounds, numba kernel",
    ),
    "streed": (
        make_streed,
        "STreeD (van der Linden et al.) cost-complex-accuracy DP on the same binarization, max depth 20",
    ),
    "gosdt_guesses": (
        make_gosdt_guesses,
        "gosdt-guesses (McTavish et al. 2022) C++ core in exact mode: same binarization, no reference labels, no depth budget",
    ),
    "split": (
        make_split,
        "SPLIT (Babbar et al. 2025): lookahead-2 prefix + optimal GOSDT leaf completion, depth budget 5, "
        "same binarization; heuristic",
    ),
    "gosdt_guesses_guided": (
        make_gosdt_guesses_guided,
        "gosdt-guesses with the paper's guesses: GBDT threshold guessing (40 stumps) and reference-label lower bounds, no depth budget; not exact",
    ),
    "gosdt_mc8": (
        lambda lam, tl: reference_solver.ReferenceGOSDT(lam, tl, memory_limit=MEMORY_LIMIT, workers=WORKERS),
        "reference C++ GOSDT with worker_limit=8 (8 TBB worker threads), otherwise as gosdt [NOT EXACT: same false certificate]",
    ),
    "gosdt_guesses_mc8": (
        lambda lam, tl: make_gosdt_guesses_mc(lam, tl),
        "gosdt-guesses C++ core in exact mode with worker_limit=8, otherwise as gosdt_guesses",
    ),
}


def make_gosdt_guesses_mc(lam, tl):
    import guesses_solver
    return guesses_solver.GuessesGOSDT(lam, tl, memory_limit=MEMORY_LIMIT, guesses=False, workers=WORKERS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true",
                        help="evaluate the baselines on the suite instead of using the cached benchmark")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--run-model", action="store_true", help="also run optimal_tree.py afterwards")
    parser.add_argument("--only", default="", help="comma-separated baseline names to (re)run; the others are left as they are")
    args = parser.parse_args()
    only = [v for v in args.only.split(",") if v]

    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, (make, description) in BASELINES.items():
        if only and name not in only:
            continue
        cached = None if args.rerun else rows_from_benchmark(name)
        if cached is not None:
            print(f"{name}: using cached rows from baselines/benchmarks/results/pair_results.csv")
            summary = summarize(cached)
        elif not args.rerun:
            print(f"skipping {name}: no complete cached rows in baselines/benchmarks/results/pair_results.csv "
                  f"(run `uv run baselines/benchmarks/run_benchmark.py --models {name} --resume`, or pass --rerun)")
            continue
        else:
            if name == "gosdt" and (args.skip_reference or not reference_solver.available()):
                print(f"skipping {name}: reference binary not built (see baselines/gosdt_patches/apply.sh)")
                continue
            print("\n" + "=" * 60 + f"\n  {name}\n" + "=" * 60)
            summary = evaluate_solver(make, name)
        record(name, description, summary, commit="baseline", status="baseline", exact=EXACT.get(name, True),
               multicore=MULTICORE.get(name, False))
        print_summary(name, summary)
    print(f"\nTotal time: {time.time() - t0:.1f}s")

    if args.run_model:
        os.system("uv run optimal_tree.py")
