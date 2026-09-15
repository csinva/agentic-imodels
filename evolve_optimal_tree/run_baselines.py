"""
Evaluate the fixed baselines on the development suite and seed the leaderboard.

Baselines:
  gosdt        the reference C++ implementation (skipped if gosdt/build/gosdt is not built)
  pygosdt_v1   the pure-Python re-implementation in pygosdt_v1/

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

BASELINES = {
    "gosdt": (
        lambda lam, tl: reference_solver.ReferenceGOSDT(lam, tl, memory_limit=MEMORY_LIMIT),
        "reference C++ GOSDT (ICML 2020 code, default bounds, single thread)",
    ),
    "pygosdt_v1": (
        lambda lam, tl: GOSDTClassifier(regularization=lam, time_limit=tl, memory_limit=MEMORY_LIMIT),
        "pygosdt_v1 package: memoised depth-first branch-and-bound with the reference bounds, numba kernel",
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true",
                        help="evaluate the baselines on the suite instead of using the cached benchmark")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--run-model", action="store_true", help="also run optimal_tree.py afterwards")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, (make, description) in BASELINES.items():
        cached = None if args.rerun else rows_from_benchmark(name)
        if cached is not None:
            print(f"{name}: using cached rows from baselines/benchmarks/results/pair_results.csv")
            summary = summarize(cached)
        else:
            if name == "gosdt" and (args.skip_reference or not reference_solver.available()):
                print(f"skipping {name}: reference binary not built (see baselines/gosdt_patches/apply.sh)")
                continue
            print("\n" + "=" * 60 + f"\n  {name}\n" + "=" * 60)
            summary = evaluate_solver(make, name)
        record(name, description, summary, commit="baseline", status="baseline")
        print_summary(name, summary)
    print(f"\nTotal time: {time.time() - t0:.1f}s")

    if args.run_model:
        os.system("uv run optimal_tree.py")
