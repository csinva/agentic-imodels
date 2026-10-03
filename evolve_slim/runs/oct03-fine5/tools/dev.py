"""Full-suite (or subset) run of a solver file without recording; per-problem rows to scratch/<tag>.csv.
usage: uv run tools/dev.py <file.py> <tag> [--datasets a,b] [--ks 3,10] [--jobs 14]"""
import argparse, os, sys, time
import pandas as pd
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from evaluate import evaluate_solver
from suite import TIME_LIMIT
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("tag")
    ap.add_argument("--datasets", default=""); ap.add_argument("--ks", default=""); ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    s = evaluate_solver(("file", os.path.abspath(a.file)), a.tag, suite="visible_fine",
                        datasets=[d for d in a.datasets.split(",") if d] or None,
                        ks=[int(v) for v in a.ks.split(",") if v] or None, time_limit=TIME_LIMIT, jobs=a.jobs, verbose=False)
    df = pd.DataFrame(s["rows"])
    df.to_csv(os.path.join(here, "scratch", a.tag + ".csv"), index=False)
    print(f"{a.tag}: regret {s['mean_regret']:.5f} time {s['geo_mean_time']:.4f} auc {s['mean_test_auc']:.4f} "
          f"loss {s['mean_loss']:.5f} invalid {s['n_invalid']} total {time.time()-t0:.0f}s load {os.getloadavg()[0]:.0f}")


if __name__ == '__main__':
    main()
