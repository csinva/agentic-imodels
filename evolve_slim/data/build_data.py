"""Build the benchmark suites for evolve_slim.

Every problem is a binary classification dataset turned into a matrix of mostly
binary features, split once into train and test. The solver sees only the train
matrix; the harness scores the integer points it returns on both.

Suites
------
visible   14 datasets used during the autoresearch loop. Six are the binarized
          files distributed with RiskSLIM (github.com/ustunb/risk-slim,
          examples/data), which FasterRisk also uses; eight come from OpenML and
          are binarized here. Datasets above 12,500 rows are subsampled
          (stratified) to 12,500.
hidden    27 binary classification datasets of TabArena (OpenML suite
          tabarena-v0.1), those not already in the visible suite (bank-marketing,
          heloc and diabetes are dropped). Same binarization and row cap. Never
          used during development.
hidden_full  the same 27 at full size (no row cap).
visible_fine, hidden_fine  the visible datasets (from raw OpenML sources) and the hidden ones
          with every numeric column split at its 99 percentiles instead of its 9 deciles.

Binarization (for the OpenML datasets)
--------------------------------------
* at most 40 raw columns: when a table has more, the 40 with the highest mutual
  information with the label on the training split are kept;
* a numeric column with at most two distinct values becomes one indicator;
* any other numeric column becomes indicators ``x <= t`` at its deciles (distinct
  thresholds only), plus a missing indicator if it has missing values;
* a categorical column becomes one indicator per level that covers at least 1% of
  the training rows (at most 20 levels), plus a missing indicator.
Constant columns are dropped. Thresholds are computed on the training split.
The positive class is the minority class.

    uv run --group data data/build_data.py [--suites visible,hidden,hidden_full]
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
MAX_ROWS = 12_500
MAX_COLUMNS = 40
TEST_SIZE = 0.2
RISKSLIM_URL = "https://raw.githubusercontent.com/ustunb/risk-slim/master/examples/data/{}_data.csv"

# name -> ("riskslim", file stem) or ("openml", data id)
VISIBLE = {
    "adult": ("riskslim", "adult"),
    "bank": ("riskslim", "bank"),
    "breastcancer": ("riskslim", "breastcancer"),
    "mammo": ("riskslim", "mammo"),
    "mushroom": ("riskslim", "mushroom"),
    "spambase": ("riskslim", "spambase"),
    "fico": ("openml", 45023),
    "compas": ("openml", 42192),
    "australian": ("openml", 40981),
    "heart": ("openml", 53),
    "ionosphere": ("openml", 59),
    "ilpd": ("openml", 1480),
    "magic": ("openml", 1120),
    "haberman": ("openml", 43),
}

# the visible datasets from their raw OpenML sources, for the fine suites (the RiskSLIM files come
# pre-binarized, so their thresholds cannot be refined)
VISIBLE_RAW = dict(VISIBLE) | {"adult": ("openml", 1590), "bank": ("openml", 1461), "breastcancer": ("openml", 15),
                               "mammo": ("openml", 45557), "mushroom": ("openml", 24), "spambase": ("openml", 44)}
FINE_THRESHOLDS = 99  # percentiles 1..99 in the fine suites, deciles otherwise

# TabArena v0.1 binary classification datasets, minus those in VISIBLE
HIDDEN = {
    "Amazon_employee_access": 46905, "APSFailure": 46908, "Bank_Customer_Churn": 46911,
    "Bioresponse": 46912, "blood-transfusion": 46913, "churn": 46915,
    "coil2000_insurance_policies": 46916, "credit-g": 46918, "credit_card_clients_default": 46919,
    "customer_satisfaction_in_airline": 46920, "Diabetes130US": 46922,
    "E-CommereShippingData": 46924, "Fitness_Club": 46927, "GiveMeSomeCredit": 46929,
    "hazelnut-contaminant": 46930, "HR_Analytics": 46935, "in_vehicle_coupon": 46937,
    "Is-this-a-good-customer": 46938, "kddcup09_appetency": 46939, "Marketing_Campaign": 46940,
    "NATICUSdroid": 46969, "online_shoppers_intention": 46947, "polish_companies_bankruptcy": 46950,
    "qsar-biodeg": 46952, "seismic-bumps": 46956, "taiwanese_bankruptcy": 46962, "jm1": 46979,
}


def load_openml(did):
    import openml
    # downloads stay next to the project, not in the (slow, full) home directory
    openml.config.cache_directory = os.environ.get("OPENML_CACHE_DIR", os.path.join(os.path.dirname(HERE), ".cache", "openml"))
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                     download_features_meta_data=False)
    X, y, cat_mask, names = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    cats = {c for c, is_cat in zip(names, cat_mask) if is_cat or not pd.api.types.is_numeric_dtype(X[c])}
    y = y.astype(str).to_numpy()
    levels, counts = np.unique(y, return_counts=True)
    assert len(levels) == 2, (did, levels)
    y = (y == levels[np.argmin(counts)]).astype(np.int8)  # minority class is positive
    return X, y, cats


def load_riskslim(stem):
    raw = urllib.request.urlopen(RISKSLIM_URL.format(stem)).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    y = (df.iloc[:, 0].to_numpy() > 0).astype(np.int8)
    return df.iloc[:, 1:].astype(float), y


def subsample(X, y, max_rows):
    if max_rows is None or len(y) <= max_rows:
        return X, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=max_rows, stratify=y, random_state=SEED)
    idx = np.sort(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def binarize(Xtr, Xte, ytr, cats, n_thresholds=9):
    """Indicator features fitted on the train split; returns (Btr, Bte, names)."""
    if Xtr.shape[1] > MAX_COLUMNS:
        enc = pd.DataFrame({c: (Xtr[c].astype(str).astype("category").cat.codes if c in cats
                                else pd.to_numeric(Xtr[c], errors="coerce").fillna(Xtr[c].median() if Xtr[c].notna().any() else 0))
                            for c in Xtr.columns})
        mi = mutual_info_classif(enc.to_numpy(float), ytr, discrete_features=[c in cats for c in Xtr.columns],
                                 random_state=SEED)
        keep = [Xtr.columns[i] for i in np.argsort(-mi, kind="stable")[:MAX_COLUMNS]]
        Xtr, Xte = Xtr[keep], Xte[keep]
    cols_tr, cols_te, names = [], [], []

    def add(name, a, b):
        if a.min() != a.max():
            cols_tr.append(a.astype(np.int8)); cols_te.append(b.astype(np.int8)); names.append(name)

    for c in Xtr.columns:
        a, b = Xtr[c], Xte[c]
        if c in cats:
            a = a.astype(object).where(a.notna(), "missing").astype(str)
            b = b.astype(object).where(b.notna(), "missing").astype(str)
            freq = a.value_counts()
            for level in list(freq[freq >= 0.01 * len(a)].index[:20]):
                add(f"{c}={level}", (a == level).to_numpy(), (b == level).to_numpy())
            continue
        a = pd.to_numeric(a, errors="coerce").to_numpy(float)
        b = pd.to_numeric(b, errors="coerce").to_numpy(float)
        miss_a, miss_b = np.isnan(a), np.isnan(b)
        if miss_a.any():
            add(f"{c} missing", miss_a, miss_b)
        vals = np.unique(a[~miss_a])
        if len(vals) <= 1:
            continue
        if len(vals) == 2:
            add(f"{c}={vals[1]:g}", a == vals[1], b == vals[1])
            continue
        thresholds = np.unique(np.quantile(a[~miss_a], np.arange(1, n_thresholds + 1) / (n_thresholds + 1),
                                            method="lower"))
        for t in thresholds:
            if t < vals[-1]:
                add(f"{c}<={t:g}", (a <= t) & ~miss_a, (b <= t) & ~miss_b)
    return np.stack(cols_tr, 1), np.stack(cols_te, 1), names


def build(name, source, out_dir, max_rows, n_thresholds=9):
    kind, key = source
    if kind == "riskslim":
        X, y = load_riskslim(key)
        cats = set()
    else:
        X, y, cats = load_openml(key)
    X, y = subsample(X, y, max_rows)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    Xtr, Xte = X.iloc[idx_tr].reset_index(drop=True), X.iloc[idx_te].reset_index(drop=True)
    ytr, yte = y[idx_tr], y[idx_te]
    if kind == "riskslim":  # already binarized (a few small-integer columns); keep as is
        keep = [c for c in X.columns if Xtr[c].min() != Xtr[c].max()]
        Btr, Bte, names = Xtr[keep].to_numpy(np.float32), Xte[keep].to_numpy(np.float32), keep
    else:
        Btr, Bte, names = binarize(Xtr, Xte, ytr, cats, n_thresholds)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), Xtr=Btr, ytr=ytr, Xte=Bte, yte=yte,
                        feature_names=np.array(names))
    row = {"name": name, "source": f"{kind}:{key}", "rows": len(y), "n_train": len(ytr), "n_test": len(yte),
           "raw_columns": X.shape[1], "features": Btr.shape[1], "positive_rate": round(float(y.mean()), 3)}
    print(row, flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="visible,hidden")
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    plan = {"visible": ({k: v for k, v in VISIBLE.items()}, MAX_ROWS),
            "hidden": ({k: ("openml", v) for k, v in HIDDEN.items()}, MAX_ROWS),
            "hidden_full": ({k: ("openml", v) for k, v in HIDDEN.items()}, None),
            "visible_fine": (VISIBLE_RAW, MAX_ROWS),
            "hidden_fine": ({k: ("openml", v) for k, v in HIDDEN.items()}, MAX_ROWS)}
    for suite in args.suites.split(","):
        datasets, max_rows = plan[suite]
        nt = FINE_THRESHOLDS if suite.endswith("_fine") else 9
        rows = [build(n, s, os.path.join(HERE, suite), max_rows, nt) for n, s in datasets.items()
                if not only or n in only]
        if not only:
            pd.DataFrame(rows).to_csv(os.path.join(HERE, suite, "manifest.csv"), index=False)
