"""The agent-edited single file must stay an exact solver: check it against the
exhaustive DP and against pygosdt_v1 on real pairs."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pygosdt_v1 import GOSDTClassifier

from test_bruteforce import exhaustive_optimum

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "baselines" / "gosdt" / "experiments" / "datasets"


def load_flat():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("optimal_tree", ROOT / "optimal_tree.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["optimal_tree"] = module  # numba's cache needs the module to be importable by name
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", range(8))
def test_flat_file_matches_exhaustive_search(seed):
    flat = load_flat()
    rng = np.random.default_rng(1000 + seed)
    n, m, K = int(rng.integers(8, 24)), int(rng.integers(2, 6)), int(rng.choice([2, 3]))
    Xb = rng.random((n, m)) < 0.5
    y = rng.integers(0, K, size=n)
    lam = float(rng.choice([0.1, 0.05, 0.02]))
    model = flat.make_model(lam, 60.0).fit(pd.DataFrame(Xb.astype(int)), y)
    assert model.optimal_
    expected = exhaustive_optimum(model.encoder_.transform(pd.DataFrame(Xb.astype(int))),
                                  model.target_encoder_.transform(y), K, lam)
    assert model.objective_ == pytest.approx(expected, abs=1e-9)


@pytest.mark.skipif(not DATA.exists(), reason="reference datasets not available")
@pytest.mark.parametrize("path,lam", [("iris/data.csv", 0.02), ("tic-tac-toe/data.csv", 0.02),
                                      ("monk_2/data.csv", 0.01), ("gaussian/thousand.csv", 0.05)])
def test_flat_file_matches_pygosdt_v1(path, lam):
    flat = load_flat()
    frame = pd.read_csv(DATA / path)
    X, y = frame.iloc[:, :-1], frame.iloc[:, -1]
    a = flat.make_model(lam, 120.0).fit(X, y)
    b = GOSDTClassifier(regularization=lam).fit(X, y)
    assert a.optimal_ and b.optimal_
    assert a.objective_ == pytest.approx(b.objective_, abs=1e-9)
