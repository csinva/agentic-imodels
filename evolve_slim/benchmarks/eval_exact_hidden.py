"""Certify on the held-out suite: run a certifying solver (exact track) on the 27 hidden datasets
x 5 values of k and judge every certificate the way src/exact.py does, against the hidden best
known losses (a lower bound above a loss some solver reached is WRONG).

    uv run benchmarks/eval_exact_hidden.py runs/sep28-exact3/slim_lib/y23_giveup3.py

Rows go to results/hidden_exact_problem_results.csv and results/hidden_exact_overall_results.csv.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluate import _upsert, evaluate_solver  # noqa: E402
from exact import judge  # noqa: E402
from suite import K_VALUES, RESULTS_DIR  # noqa: E402

if __name__ == "__main__":
    for path in sys.argv[1:]:
        name = os.path.splitext(os.path.basename(path))[0]
        s = evaluate_solver(("file", os.path.abspath(path)), name, suite="hidden", ks=K_VALUES, jobs=14)
        rows = judge(s["rows"], "hidden")
        n_cert = sum(r["certified"] for r in rows)
        n_wrong = sum(r["verdict"] == "WRONG" for r in rows)
        by_k = {k: sum(r["certified"] for r in rows if r["k"] == k) for k in K_VALUES}
        print(f"{name}: certified {n_cert}/{len(rows)} hidden problems (by k: {by_k}), WRONG {n_wrong}, "
              f"geo time {s['geo_mean_time']:.3f}s, mean loss {s['mean_loss']:.5f}, test AUC {s['mean_test_auc']:.4f}")
        cols = ["model", "suite", "dataset", "k", "status", "seconds", "loss", "lower_bound", "known", "certified",
                "verdict", "auc_test", "points"]
        _upsert(os.path.join(RESULTS_DIR, "hidden_exact_problem_results.csv"), cols,
                [{c: r.get(c, "") for c in cols} for r in rows], key=lambda r: r["model"])
        _upsert(os.path.join(RESULTS_DIR, "hidden_exact_overall_results.csv"),
                ["model_name", "n_certified", "n_problems", "n_wrong", "geo_mean_time", "mean_loss", "mean_test_auc"],
                [{"model_name": name, "n_certified": n_cert, "n_problems": len(rows), "n_wrong": n_wrong,
                  "geo_mean_time": f"{s['geo_mean_time']:.3f}", "mean_loss": f"{s['mean_loss']:.5f}",
                  "mean_test_auc": f"{s['mean_test_auc']:.4f}"}], key=lambda r: r["model_name"])
