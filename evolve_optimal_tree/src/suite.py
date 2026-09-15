"""The fixed development suite for the optimal-tree autoresearch loop.

Do not modify: every experiment is scored on exactly these (dataset, λ) pairs
with the same time cap, so rows in ``results/overall_results.csv`` stay
comparable.  ``sine_10k`` (10,000 rows, 9,999 thresholds) is deliberately left
out of the development suite and kept for the full benchmark in
``benchmarks/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = str(ROOT / "results")
DATA = ROOT / "baselines" / "gosdt" / "experiments" / "datasets"

# (name, csv path).  The last column of every CSV is the label.
DATASETS = [
    ("chudi", DATA / "chudi" / "data.csv"),
    ("monk_3", DATA / "monk_3" / "data.csv"),
    ("monk_1", DATA / "monk_1" / "data.csv"),
    ("iris", DATA / "iris" / "data.csv"),
    ("monk_2", DATA / "monk_2" / "data.csv"),
    ("tic-tac-toe", DATA / "tic-tac-toe" / "data.csv"),
    ("gaussian_1k", DATA / "gaussian" / "thousand.csv"),
    ("fico_1k", DATA / "fico" / "data.csv"),
    ("sine_1k", DATA / "sine" / "thousand.csv"),
    ("car_evaluation", DATA / "car_evaluation" / "data.csv"),
    ("coupon_bar", DATA / "coupon" / "bar-7.csv"),
    ("compas_binned", DATA / "compas" / "binned.csv"),
    ("fico_binary", DATA / "fico" / "fico-binary.csv"),
    ("compas_processed", DATA / "compas" / "processed.csv"),
]

LAMBDAS = [0.1, 0.05, 0.02, 0.01, 0.005]

# Seconds of optimisation allowed per (dataset, λ) pair.  A pair that is not
# certified optimal within the cap counts as unsolved, and its time counts as
# the cap.
TIME_LIMIT = 30.0

# Resident memory (bytes) at which a fit is stopped, to protect the machine.
MEMORY_LIMIT = 6 * (1 << 30)

KNOWN_OPTIMA = HERE / "known_optima.csv"


def load_dataset(name: str) -> pd.DataFrame:
    """Load a suite dataset with missing values filled by 0 (label is the last column)."""
    path = dict(DATASETS)[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: copy the reference repository to {DATA.parent.parent} (see readme.md)")
    frame = pd.read_csv(path)
    if frame.isna().any().any():
        frame = frame.fillna(0)
    return frame


def pairs():
    for name, _ in DATASETS:
        for lam in LAMBDAS:
            yield name, lam


def load_known_optima() -> dict:
    """{(dataset, lam): (objective, certified)} for the best known objective per pair."""
    table = pd.read_csv(KNOWN_OPTIMA)
    return {(r.dataset, float(r.lam)): (float(r.objective), bool(r.certified)) for r in table.itertuples()}
