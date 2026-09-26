"""The fixed development suite for evolve_slim. Do not modify.

A problem is a (dataset, k) pair: fit integer points w_j in {-5, ..., 5} on at most
k of the dataset's (mostly binary) features. The harness then scores the points
itself (``evaluate.py``), so every solver is judged by the same rule.
"""

from __future__ import annotations

import os

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.realpath(SRC_DIR))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

DATASETS = ["adult", "bank", "breastcancer", "mammo", "mushroom", "spambase", "fico", "compas",
            "australian", "heart", "ionosphere", "ilpd", "magic", "haberman"]
K_VALUES = [3, 4, 5, 7, 10]
COEF_BOUND = 5          # every point value is an integer in [-COEF_BOUND, COEF_BOUND]
TIME_LIMIT = 60.0       # seconds per problem passed to the solver
HARD_LIMIT_FACTOR = 3   # a fit still running at this multiple of the limit is killed (no model)


def load_problem(name: str, suite: str = "visible"):
    """(Xtr, ytr, Xte, yte, feature_names); X float64, y in {0, 1}."""
    path = os.path.join(DATA_DIR, suite, f"{name}.npz")
    z = np.load(path, allow_pickle=False)
    return (z["Xtr"].astype(np.float64), z["ytr"].astype(np.int64), z["Xte"].astype(np.float64),
            z["yte"].astype(np.int64), [str(s) for s in z["feature_names"]])


def suite_datasets(suite: str = "visible"):
    if suite == "visible":
        return list(DATASETS)
    import pandas as pd
    return pd.read_csv(os.path.join(DATA_DIR, suite, "manifest.csv"))["name"].tolist()
