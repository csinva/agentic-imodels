"""Build TabArena-14: a held-out mirror of the 14-dataset development suite.

Each of the 14 development datasets gets a stand-in built from a TabArena
classification dataset (the OpenML suite ``tabarena-v0.1``) that matches it on the
things that decide how hard an exact tree search is:

* the number of rows, exactly, by stratified subsampling;
* the majority-class fraction, to within 0.05 (a source that cannot reach it is not eligible);
* the number of classes (two, or three for the iris slot);
* the number of features, exactly;
* the kind of feature, and through it the number of binary splits the solver
  searches over: where the original's features are all binary, the stand-in's are
  binarized (a numeric column at its median, a category against the rest); where
  they are numeric or ordinal, each stand-in column is cut into quantile bins, with
  one bin count chosen for all columns so the total number of splits comes as close
  as it can to the original's (for the two one-feature slots, whose originals are
  synthetic and tie-free, a real column with at least 80% of the splits is accepted).

Features are the ones with the highest mutual information with the label among the
candidates, which is what a practitioner screening a wide table would keep and
avoids stand-ins whose optimal tree is a single leaf only because the features
were drawn at random.

Pairing is a fixed rule, not a choice per dataset: originals are taken largest
first, and each takes the smallest unused eligible source that has enough rows,
enough of each class to reach the original's class balance, enough candidate
features, and (for a one-feature slot) a column with enough distinct values. TabArena's heloc and in_vehicle_coupon_recommendation are
excluded, since they are the FICO and coupon data the development suite already
contains. Everything is seeded.

    uv run --with openml baselines/benchmarks/external/build_tabarena14.py

Writes ``data/<name>.csv`` (features, then the label as the last column, all
numeric) and ``data/manifest.csv`` next to this file.
"""

import math
import os

import numpy as np
import openml
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
SEED = 0
openml.config.cache_directory = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve", "openml")

# the development suite: rows, features, binary splits, classes, majority fraction, feature kind
ORIGINALS = [
    ("chudi", 77, 2, 46, 2, 0.61, "numeric"),
    ("monk_3", 122, 11, 11, 2, 0.51, "binary"),
    ("monk_1", 124, 11, 11, 2, 0.50, "binary"),
    ("iris", 150, 4, 118, 3, 0.33, "numeric"),
    ("monk_2", 169, 11, 11, 2, 0.62, "binary"),
    ("tic-tac-toe", 958, 9, 27, 2, 0.65, "numeric"),
    ("gaussian_1k", 1000, 1, 999, 2, 0.70, "numeric"),
    ("fico_1k", 1000, 23, 1357, 2, 0.54, "numeric"),
    ("sine_1k", 1000, 1, 999, 2, 0.51, "numeric"),
    ("car_evaluation", 1728, 15, 15, 2, 0.70, "binary"),
    ("coupon_bar", 1913, 14, 14, 2, 0.59, "binary"),
    ("compas_binned", 6907, 12, 12, 2, 0.54, "binary"),
    ("fico_binary", 10459, 17, 17, 2, 0.52, "binary"),
    ("compas_processed", 12381, 22, 621, 2, 0.69, "numeric"),
]
EXCLUDED = {"heloc", "in_vehicle_coupon_recommendation"}  # already in the development suite


def load_source(did):
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                     download_features_meta_data=False)
    X, y, cat_mask, names = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    X = X.copy()
    for c, is_cat in zip(names, cat_mask):
        if is_cat or not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str).astype(object)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(X[c].median())
    return X, pd.Series(pd.factorize(y)[0], index=X.index), ds.name


def stratified_sample(y, n, majority_frac, n_classes, rng):
    """Row positions: n rows whose majority fraction is as close to the target as the classes allow."""
    counts = y.value_counts()
    classes = list(counts.index)
    if n_classes == 2:
        want = {classes[0]: int(round(majority_frac * n))}
        want[classes[1]] = n - want[classes[0]]
        # the source may be short of either class; move the shortfall to the other
        for a, b in ((classes[1], classes[0]), (classes[0], classes[1])):
            short = want[a] - counts[a]
            if short > 0:
                want[a] -= short
                want[b] += short
    else:
        base = n // n_classes
        want = {c: base for c in classes[:n_classes]}
        for c in classes[:n - base * n_classes]:
            want[c] += 1
    idx = []
    for c, k in want.items():
        pos = np.flatnonzero(y.to_numpy() == c)
        idx.extend(rng.choice(pos, size=k, replace=False))
    return np.sort(np.array(idx))


def binary_candidates(X):
    """Every column as 0/1: numeric at its median, each category against the rest."""
    out = {}
    for c in X.columns:
        col = X[c]
        if not pd.api.types.is_numeric_dtype(col):
            for v in col.value_counts().index[:10]:
                b = (col == v).astype(int)
                if 0 < b.sum() < len(b):
                    out[f"{c}={v}"] = b
        else:
            b = (col >= col.median()).astype(int)
            if 0 < b.sum() < len(b):
                out[c] = b
    return pd.DataFrame(out, index=X.index)


def numeric_candidates(X):
    """Numeric columns as they are; categories as integer codes ordered by frequency."""
    out = {}
    for c in X.columns:
        col = X[c]
        if not pd.api.types.is_numeric_dtype(col):
            order = {v: i for i, v in enumerate(col.value_counts().index)}
            col = col.map(order).astype(float)
        if col.nunique() > 1:
            out[c] = col.astype(float)
    return pd.DataFrame(out, index=X.index)


def quantize(F, splits):
    """Cut every column into at most k quantile bins, with one k for all columns chosen so
    the number of binary splits (distinct values minus one, summed) is closest to `splits`.
    The count only grows with k, so k is found by bisection."""
    ranks = {c: F[c].rank(method="first") for c in F.columns}
    distinct = {c: F[c].nunique() for c in F.columns}

    def binned(k):
        cols = {}
        for c in F.columns:
            if distinct[c] <= k:
                cols[c] = F[c].rank(method="dense").astype(int) - 1
            else:
                cols[c] = pd.qcut(ranks[c], q=k, labels=False, duplicates="drop").astype(int)
        return pd.DataFrame(cols, index=F.index)

    def count(k):
        return int(sum(min(distinct[c], k) - 1 for c in F.columns))

    lo, hi = 2, max(2, max(distinct.values()))
    while lo < hi:  # smallest k whose count reaches the target
        mid = (lo + hi) // 2
        if count(mid) >= splits:
            hi = mid
        else:
            lo = mid + 1
    k = min((lo - 1, lo), key=lambda v: (abs(count(max(v, 2)) - splits), v)) if lo > 2 else lo
    k = max(k, 2)
    return binned(k), k


def build_one(orig, source, rng):
    name, n, p, splits, K, maj, kind = orig
    X, y, src_name = source
    rows = stratified_sample(y, n, maj, K, rng)
    X, y = X.iloc[rows].reset_index(drop=True), y.iloc[rows].reset_index(drop=True)
    y = pd.Series(pd.factorize(y)[0])
    cand = binary_candidates(X) if kind == "binary" else numeric_candidates(X)
    if cand.shape[1] < p:
        return None
    discrete = kind == "binary"
    mi = mutual_info_classif(cand.to_numpy(), y.to_numpy(), discrete_features=discrete, random_state=SEED)
    order = list(cand.columns[np.argsort(-mi, kind="stable")])
    if kind == "numeric" and p == 1:
        # A one-feature slot needs a column with close to as many distinct values as rows.
        # The originals here are synthetic and have no ties; real columns almost always do,
        # so 80% of the split count is accepted (jittering ties apart would invent splits
        # between identical values).
        order = [c for c in order if cand[c].nunique() - 1 >= 0.8 * splits]
        if not order:
            return None
    chosen = order[:p]
    F = cand[chosen]
    k = None
    if kind == "numeric":
        F, k = quantize(F, splits)
    got_splits = int(sum(F[c].nunique() - 1 for c in F.columns))
    frame = F.copy()
    frame.columns = [f"x{i}" for i in range(p)]
    frame["label"] = y.to_numpy()
    info = {"name": f"ta_{name}", "mirrors": name, "source": src_name,
            "rows": n, "features": p, "binary_splits": got_splits, "target_splits": splits,
            "classes": int(y.nunique()), "majority": round(float(y.value_counts(normalize=True).max()), 3),
            "target_majority": maj, "kind": kind, "bins": k if k else "",
            "source_columns": "; ".join(chosen)}
    return frame, info


def main():
    os.makedirs(OUT, exist_ok=True)
    suite = openml.study.get_suite("tabarena-v0.1")
    tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
    clf = tasks[tasks.task_type.str.contains("Classification") & ~tasks.name.isin(EXCLUDED)]
    clf = clf.sort_values("NumberOfInstances")
    pool = list(zip(clf.did.astype(int), clf.name, clf.NumberOfInstances.astype(int), clf.NumberOfClasses.astype(int)))
    used, manifest, cache = set(), [], {}
    for orig in sorted(ORIGINALS, key=lambda o: -o[1]):
        name, n, p, splits, K, maj, kind = orig
        built = None
        for did, src, rows, classes in pool:
            if did in used or rows < n or classes != K:
                continue
            if did not in cache:
                cache[did] = load_source(did)
            X, y, _ = cache[did]
            counts = y.value_counts()
            if K > 2:
                if len(counts) < K or counts.iloc[K - 1] < n // K:
                    continue
            else:
                lo_major = int(math.ceil((maj - 0.05) * n))
                hi_major = int(math.floor((maj + 0.05) * n))
                # need some split (majority a, minority n-a) with a in range and both classes available
                if not any(counts.iloc[0] >= a and counts.iloc[1] >= n - a for a in range(max(lo_major, n - counts.iloc[1]), hi_major + 1)) \
                        and not any(counts.iloc[1] >= a and counts.iloc[0] >= n - a for a in range(lo_major, hi_major + 1)):
                    continue
            rng = np.random.default_rng(SEED)
            built = build_one(orig, cache[did], rng)
            if built is not None:
                used.add(did)
                break
        if built is None:
            raise SystemExit(f"no eligible TabArena source for {name}")
        frame, info = built
        info["openml_did"] = did
        frame.to_csv(os.path.join(OUT, f"{info['name']}.csv"), index=False)
        manifest.append(info)
        print(f"{name:17s} <- {info['source']:40s} rows={n} feats={p} splits={info['binary_splits']}/{splits} "
              f"maj={info['majority']}/{maj} classes={info['classes']}", flush=True)
    order = [o[0] for o in ORIGINALS]
    m = pd.DataFrame(manifest)
    m["_o"] = m["mirrors"].map(order.index)
    m.sort_values("_o").drop(columns="_o").to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
