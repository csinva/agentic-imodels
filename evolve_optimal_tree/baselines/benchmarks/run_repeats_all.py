"""Repeat every solver in the headline comparison, to report averages not single runs.

``run_repeats.py`` covers the solvers the loop compares against; this adds the
rest of the table (the gosdt-guesses pair and the evolved solver itself) and
summarises all of them together, so every row of the comparison is a mean over
the same number of runs.

Run 0 of each solver is the one already on disk: the full benchmark converted to
the suite cap for the baselines, and the recorded run in ``runs/<tag>`` for the
evolved solver. Runs 1+ are appended here, one row per pair as it finishes, so an
interrupted sweep keeps what it measured.

Timings are the metric, so the machine must be otherwise idle and the solvers are
run one after another, never concurrently.

Usage (from ``evolve_optimal_tree``)::

    uv run baselines/benchmarks/run_repeats_all.py --models pygosdt_v1,streed --repeats 2
    uv run baselines/benchmarks/run_repeats_all.py --summary
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
from evaluate import PAIR_CSV_COLS, evaluate_solver, print_summary, rows_from_benchmark, summarize  # noqa: E402
from suite import MEMORY_LIMIT, TIME_LIMIT  # noqa: E402

RESULTS = os.path.join(HERE, "results")
REPEATS_CSV = os.path.join(RESULTS, "repeats_all.csv")
SUMMARY_CSV = os.path.join(RESULTS, "repeats_all_summary.csv")
# run 1+ of gosdt were measured by run_repeats.py; read them rather than redo them
PEER_REPEATS_CSV = os.path.join(RESULTS, "repeats.csv")
COLS = ["repeat"] + PAIR_CSV_COLS

#: evolved solvers: repeat name -> snapshot in runs/<tag>/optimal_tree_lib, whose recorded
#: row in that run's pair_results.csv is run 0
RUN_DIR = os.path.join(ROOT, "runs", "sep15-run1")
EVOLVED = {"autoopttree": "v23_word_compaction",
           "autoopttree_v40": "v40_sequential",
           "autoopttree_v46": "v46_topk_pairs",
           "autoopttree_v46_anytime100": "v46_anytime_100ms",
           "autoopttree_v49": "v49_cands_pairs_lazy_ws"}


def factories():
    from pygosdt_v1 import GOSDTClassifier
    import reference_solver

    def make_guesses(guesses):
        def make(lam, tl):
            import guesses_solver
            return guesses_solver.GuessesGOSDT(lam, tl, memory_limit=MEMORY_LIMIT, guesses=guesses)
        return make

    def make_streed(lam, tl):
        import streed_solver
        return streed_solver.STreeD(lam, tl)

    def make_guesses_hp(**kw):
        """gosdt-guesses at a setting other than the paper's headline one."""
        def make(lam, tl):
            import guesses_solver_hp
            return guesses_solver_hp.GuessesGOSDTHP(lam, tl, memory_limit=MEMORY_LIMIT, **kw)
        return make

    loaded = {}

    def make_evolved(name):
        def make(lam, tl):
            if name not in loaded:
                path = os.path.join(RUN_DIR, "optimal_tree_lib", f"{name}.py")
                spec = importlib.util.spec_from_file_location(f"_{name}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                loaded[name] = mod
            return loaded[name].make_model(lam, tl)
        return make

    return {
        "gosdt": lambda lam, tl: reference_solver.ReferenceGOSDT(lam, tl, memory_limit=MEMORY_LIMIT),
        "pygosdt_v1": lambda lam, tl: GOSDTClassifier(regularization=lam, time_limit=tl,
                                                      memory_limit=MEMORY_LIMIT),
        "streed": make_streed,
        "gosdt_guesses": make_guesses(False),
        "gosdt_guesses_guided": make_guesses(True),
        # the same baseline at other settings, so it is judged at its best on this suite:
        # the paper's depth budget, a larger and deeper threshold guesser, and the exact
        # mode with the similar-support bound its code base offers
        "gg_guided_db5": make_guesses_hp(guesses=True, depth_budget=5),
        "gg_guided_e60d2": make_guesses_hp(guesses=True, n_estimators=60, guess_depth=2),
        "gg_exact_simsup": make_guesses_hp(guesses=False, similar_support=True),
        **{rep: make_evolved(snap) for rep, snap in EVOLVED.items()},
    }


def _read(path, model=None):
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path)
    return df[df["model"] == model] if model else df


def next_repeat(model: str) -> int:
    done = [int(r) for r in _read(REPEATS_CSV, model)["repeat"]] if os.path.exists(REPEATS_CSV) else []
    return max(done, default=0) + 1


def append_row(row: dict, repeat: int):
    os.makedirs(RESULTS, exist_ok=True)
    new = not os.path.exists(REPEATS_CSV)
    with open(REPEATS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"repeat": repeat, **row})


def run_zero(model: str):
    """The run already on disk for this solver, as evaluate rows, or None."""
    if model not in EVOLVED:
        return rows_from_benchmark(model)
    name = EVOLVED[model]
    path = os.path.join(RUN_DIR, "results", "pair_results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["model"] == name]
    return df.to_dict("records") or None


def _rows_of(df):
    rows = df.to_dict("records")
    for r in rows:  # a crashed pair has no time: it spent the whole cap
        r["seconds"] = float(r["seconds"]) if not pd.isna(r["seconds"]) else TIME_LIMIT
    return rows


def write_summary():
    mine = _read(REPEATS_CSV)
    peer = _read(PEER_REPEATS_CSV)
    out = []
    for model in factories():
        runs = []
        zero = run_zero(model)
        if zero:
            runs.append((0, summarize(zero)))
        for source in (mine, peer):
            if len(source) == 0:
                continue
            for rep, d in source[source["model"] == model].groupby("repeat"):
                runs.append((int(rep), summarize(_rows_of(d))))
        if not runs:
            continue
        seen, keep = set(), []
        for rep, s in sorted(runs, key=lambda t: t[0]):  # counted once if in both files
            if rep not in seen:
                seen.add(rep)
                keep.append((rep, s))
        for rep, s in keep:
            out.append({"model": model, "repeat": rep, "n_solved": s["n_solved"],
                        "geo_mean_time": round(s["geo_mean_time"], 4), "n_wrong": s["n_wrong"],
                        "total_time": round(sum(float(r["seconds"]) for r in s["rows"]), 1)})
        if len(keep) > 1:
            d = pd.DataFrame([o for o in out if o["model"] == model])
            for stat, fn in (("mean", lambda c: c.mean()), ("std", lambda c: c.std(ddof=1))):
                out.append({"model": model, "repeat": stat,
                            "n_solved": round(fn(d["n_solved"]), 2),
                            "geo_mean_time": round(fn(d["geo_mean_time"]), 4),
                            "n_wrong": round(fn(d["n_wrong"]), 2),
                            "total_time": round(fn(d["total_time"]), 1)})
    df = pd.DataFrame(out)
    df.to_csv(SUMMARY_CSV, index=False)
    print(df.to_string(index=False))
    print(f"-> {SUMMARY_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--summary", action="store_true", help="only (re)write the summary")
    args = ap.parse_args()
    if not args.summary:
        f = factories()
        for name in [m for m in args.models.split(",") if m]:
            for _ in range(args.repeats):
                rep = next_repeat(name)
                print("\n" + "=" * 60 + f"\n  {name}  repeat {rep}  (cap {args.time_limit:g}s)\n" + "=" * 60,
                      flush=True)
                t0 = time.time()
                s = evaluate_solver(f[name], name, time_limit=args.time_limit,
                                    on_row=lambda r, rep=rep: append_row(r, rep))
                print_summary(name, s)
                print(f"repeat {rep} wall: {time.time() - t0:.0f}s", flush=True)
    write_summary()
