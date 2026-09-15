"""Full benchmark of the baselines against each other, scored exactly like the loop.

Runs ``gosdt`` (the reference C++ binary), ``pygosdt_v1`` and ``streed``
(STreeD, ``baselines/pystreed``) through ``src/evaluate.evaluate_solver`` on
the development suite plus ``sine_10k``,
with a 600 s cap per pair, and writes one row per model × dataset × λ to
``results/pair_results.csv``.  ``summarize.py`` and ``build_report.py`` turn
that file into the summary table and ``baselines/REPORT.html``;
``run_baselines.py`` seeds the loop's leaderboard from it.

Usage (from ``evolve_optimal_tree``)::

    uv run baselines/benchmarks/run_benchmark.py [--models gosdt,pygosdt_v1,streed]
        [--datasets a,b] [--lams 0.1,0.05] [--time-limit 600] [--resume]

The full grid takes several hours because the reference hits the cap on many
pairs; ``--resume`` skips pairs already present in the results file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
from evaluate import PAIR_CSV_COLS, evaluate_solver, print_summary  # noqa: E402
from suite import DATA, DATASETS, LAMBDAS, MEMORY_LIMIT  # noqa: E402
import reference_solver  # noqa: E402
import suite  # noqa: E402

from pygosdt_v1 import GOSDTClassifier  # noqa: E402

RESULTS = os.path.join(HERE, "results")
PAIRS_CSV = os.path.join(RESULTS, "pair_results.csv")
FULL_TIME_LIMIT = 600.0
EXTRA_DATASETS = [("sine_10k", DATA / "sine" / "ten_thousand.csv")]

def make_streed(lam, tl):
    import streed_solver  # needs the ``baselines`` dependency group
    return streed_solver.STreeD(lam, tl)


def make_gosdt_guesses(lam, tl):
    import guesses_solver  # needs the ``baselines`` dependency group
    return guesses_solver.GuessesGOSDT(lam, tl, memory_limit=MEMORY_LIMIT, guesses=False)


def make_gosdt_guesses_guided(lam, tl):
    import guesses_solver
    return guesses_solver.GuessesGOSDT(lam, tl, memory_limit=MEMORY_LIMIT, guesses=True)


MODELS = {
    "gosdt": lambda lam, tl: reference_solver.ReferenceGOSDT(lam, tl, memory_limit=MEMORY_LIMIT),
    "pygosdt_v1": lambda lam, tl: GOSDTClassifier(regularization=lam, time_limit=tl, memory_limit=MEMORY_LIMIT),
    "streed": make_streed,
    "gosdt_guesses": make_gosdt_guesses,
    "gosdt_guesses_guided": make_gosdt_guesses_guided,
}


def load_done(model: str) -> set:
    if not os.path.exists(PAIRS_CSV):
        return set()
    with open(PAIRS_CSV, newline="") as f:
        return {(r["dataset"], float(r["lam"])) for r in csv.DictReader(f) if r["model"] == model}


def append_row(row: dict):
    os.makedirs(RESULTS, exist_ok=True)
    new = not os.path.exists(PAIRS_CSV)
    with open(PAIRS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gosdt,pygosdt_v1,streed,gosdt_guesses,gosdt_guesses_guided")
    ap.add_argument("--datasets", default="", help="subset of the benchmark datasets (default: all)")
    ap.add_argument("--lams", default="", help="subset of the λ grid (default: all)")
    ap.add_argument("--time-limit", type=float, default=FULL_TIME_LIMIT)
    ap.add_argument("--resume", action="store_true", help="skip pairs already in results/pair_results.csv")
    args = ap.parse_args()

    # the benchmark grid is the suite plus the large held-out dataset
    suite.DATASETS[:] = list(DATASETS) + [d for d in EXTRA_DATASETS if d[0] not in dict(DATASETS)]
    datasets = [d for d in args.datasets.split(",") if d] or [d for d, _ in suite.DATASETS]
    lambdas = [float(v) for v in args.lams.split(",") if v] or list(LAMBDAS)

    t0 = time.time()
    for name in [m for m in args.models.split(",") if m]:
        if name == "gosdt" and not reference_solver.available():
            print("skipping gosdt: reference binary not built (see baselines/gosdt_patches/apply.sh)")
            continue
        skip = load_done(name) if args.resume else set()
        if not args.resume and os.path.exists(PAIRS_CSV):
            # drop this model's old rows so the file holds one row per pair
            with open(PAIRS_CSV, newline="") as f:
                keep = [r for r in csv.DictReader(f) if r["model"] != name]
            with open(PAIRS_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PAIR_CSV_COLS, extrasaction="ignore")
                w.writeheader()
                w.writerows(keep)
        print("\n" + "=" * 60 + f"\n  {name}  (cap {args.time_limit:g}s, {len(skip)} pairs already done)\n" + "=" * 60)
        summary = evaluate_solver(MODELS[name], name, datasets=datasets, lambdas=lambdas,
                                  time_limit=args.time_limit, skip=skip, on_row=append_row)
        print_summary(name, summary)
    print(f"\nTotal time: {time.time() - t0:.1f}s; rows in {PAIRS_CSV}")
    print("Next: uv run baselines/benchmarks/summarize.py && uv run baselines/benchmarks/build_report.py")
