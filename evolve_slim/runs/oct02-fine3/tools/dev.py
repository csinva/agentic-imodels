"""Full-suite run of a solver file without recording; saves per-problem rows to scratch/<tag>.csv.
uv run python tools/dev.py slim.py TAG   (env vars pass through to the workers)"""
import os, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
import pandas as pd
from evaluate import evaluate_solver, print_summary
from suite import TIME_LIMIT
if __name__ == "__main__":
    f, tag = os.path.abspath(sys.argv[1]), sys.argv[2]
    s = evaluate_solver(("file", f), tag, suite="visible_fine", time_limit=TIME_LIMIT, jobs=14, verbose=False)
    df = pd.DataFrame(s["rows"])
    os.makedirs(os.path.join(here, "scratch"), exist_ok=True)
    df.to_csv(os.path.join(here, "scratch", tag + ".csv"), index=False)
    print(f"{tag}: regret {s['mean_regret']:.5f} time {s['geo_mean_time']:.3f} auc {s['mean_test_auc']:.4f} "
          f"loss {s['mean_loss']:.5f} invalid {s['n_invalid']}")
