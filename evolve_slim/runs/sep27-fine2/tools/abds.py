"""Per-dataset summary of an ab.py output: time ratio B/A and mean regret A, B (1e-3)."""
import sys, re, collections, numpy as np
rows = collections.defaultdict(list)
for ln in open(sys.argv[1]):
    m = re.match(r"(\S+)\s+k=\s*(\d+) \| t=([\d.]+) reg=([-+\d.]+) auc=[\d.]+ \| t=([\d.]+) reg=([-+\d.]+)", ln)
    if m:
        d, k, ta, ra, tb, rb = m.groups(); rows[d].append((float(ta), float(ra), float(tb), float(rb)))
for d, r in rows.items():
    r = np.array(r)
    print(f"{d:12s} tA={np.exp(np.log(r[:,0]).mean()):.3f} ratio={np.exp(np.log(r[:,2]/r[:,0]).mean()):.2f} regA={r[:,1].mean():+.3f} regB={r[:,3].mean():+.3f}")
