"""Dev evaluation (not recorded in results/): python dev.py file.py tag [datasets] [ks]; per-problem rows -> dev/<tag>.csv"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "src"))
import pandas as pd
from evaluate import evaluate_solver, print_summary
if __name__ == "__main__":
    f, tag = sys.argv[1], sys.argv[2]
    ds = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None
    ks = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    s = evaluate_solver(("file", os.path.abspath(f)), tag, datasets=ds, ks=ks, verbose=False)
    pd.DataFrame(s["rows"]).to_csv(f"dev/{tag}.csv", index=False)
    print_summary(tag, s)
