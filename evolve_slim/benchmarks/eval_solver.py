"""Evaluate solver files (slim.py snapshots) on a suite and record them.

    uv run benchmarks/eval_solver.py runs/sep26-run1/slim_lib/v12.py [more.py ...] --suite hidden
    uv run benchmarks/eval_solver.py FILE --suite hidden_full --ks 5 --time-limit 600 --jobs 27

The model name is the file stem (plus --tag). Rows go to results/<suite>_*.csv
(results/*.csv for the visible suite), with a t<limit>_ prefix for a non-default limit.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluate import evaluate_solver, print_summary, record  # noqa: E402
from suite import RESULTS_DIR, TIME_LIMIT  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--suite", default="hidden")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--ks", default="")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()
    datasets = [d for d in args.datasets.split(",") if d] or None
    ks = [int(v) for v in args.ks.split(",") if v] or None
    for path in args.files:
        name = os.path.splitext(os.path.basename(path))[0] + args.tag
        t0 = time.time()
        s = evaluate_solver(("file", os.path.abspath(path)), name, suite=args.suite, datasets=datasets, ks=ks,
                            time_limit=args.time_limit, jobs=args.jobs)
        print_summary(name, s, args.time_limit)
        print(f"total_seconds: {time.time() - t0:.1f}s")
        if not args.no_record and datasets is None:
            prefix = "" if args.suite == "visible" else f"{args.suite}_"
            if args.time_limit != TIME_LIMIT:
                prefix += f"t{args.time_limit:g}_"
            record(name, os.path.relpath(os.path.abspath(path), ROOT), s, status="eval", results_dir=RESULTS_DIR,
                   prefix=prefix)
