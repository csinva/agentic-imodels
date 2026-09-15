"""Regression tests on the reference datasets.

Expected objectives were produced by the reference C++ implementation and
cross-checked with pygosdt (see benchmarks/results).  For tic-tac-toe at
λ = 0.02 the reference returns 0.324593 but the tree below is verifiably better
(190 errors on 958 rows with 6 leaves); pygosdt must find it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pygosdt_v1 import GOSDTClassifier

DATA = Path(__file__).resolve().parent.parent / "gosdt" / "experiments" / "datasets"

CASES = [
    ("monk_1/data.csv", 0.1, 0.466129), ("monk_1/data.csv", 0.05, 0.338710), ("monk_1/data.csv", 0.02, 0.16),
    ("monk_2/data.csv", 0.02, 0.358935), ("monk_3/data.csv", 0.02, 0.242951),
    ("iris/data.csv", 0.1, 0.34), ("iris/data.csv", 0.05, 0.19), ("iris/data.csv", 0.02, 0.10),
    ("tic-tac-toe/data.csv", 0.05, 0.396555), ("tic-tac-toe/data.csv", 0.02, 0.318330),
    ("car_evaluation/data.csv", 0.02, 0.204676), ("car_evaluation/data.csv", 0.01, 0.145231),
    ("chudi/data.csv", 0.02, 0.10), ("gaussian/thousand.csv", 0.05, 0.275),
]


@pytest.mark.skipif(not DATA.exists(), reason="reference datasets not available")
@pytest.mark.parametrize("path,lam,expected", CASES)
@pytest.mark.parametrize("engine", ["numba", "python"])
def test_reference_dataset_objective(path, lam, expected, engine):
    frame = pd.read_csv(DATA / path)
    X, y = frame.iloc[:, :-1], frame.iloc[:, -1]
    model = GOSDTClassifier(regularization=lam, engine=engine).fit(X, y)
    assert model.optimal_
    # objective recomputed from predictions, independent of the stored leaf losses
    loss = float(np.mean(model.predict(X) != y.to_numpy()))
    assert loss + lam * model.n_leaves_ == pytest.approx(expected, abs=2e-6)
    assert model.objective_ == pytest.approx(expected, abs=2e-6)
