"""Wrapper that runs SPLIT (Babbar et al., ICML 2025; ``baselines/split``) with the
solver interface used by ``evaluate.evaluate_solver``.

SPLIT optimises a shallow prefix of the tree with a GOSDT-style search in
which subtrees beyond ``lookahead_depth`` are evaluated greedily, then fills
each prefix leaf with an optimal GOSDT subtree under the remaining depth
budget.  It uses the same regularization λ (its ``reg``) and, through this
wrapper, the same binarization as every other baseline, but it needs a depth
budget and is a heuristic by construction, so it never certifies optimality:
rows are recorded with status ``heuristic`` and ``exact = approximate``.  It
only supports binary classification, so multi-class pairs (iris) are recorded
with status ``unsupported`` and no tree.

Defaults follow the paper's quick-start (lookahead depth 2, full depth 5).
Each fit runs in a child process so a runaway leaf completion (the gosdt-guesses
core) can be killed at the memory cap.  Requires the ``baselines`` dependency
group (``uv sync --group baselines``).  Do not modify.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import NoModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

LOOKAHEAD_DEPTH = 2
FULL_DEPTH = 5


def _rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
        return int(out.stdout.strip() or 0) * 1024
    except (ValueError, OSError):
        return 0


def _fit_child(payload: dict) -> dict:
    """Runs inside the child process: binarize, fit SPLIT, return a JSON-able tree."""
    from split import SPLIT
    from split._tree import Leaf

    sys.path.insert(0, str(ROOT))
    from pygosdt_v1.encoder import BinaryEncoder, TargetEncoder

    X = pd.DataFrame(payload["X"], columns=payload["columns"])
    y = np.asarray(payload["y"])
    enc = BinaryEncoder().fit(X)
    Xb = pd.DataFrame(enc.transform(X).astype(np.int64), columns=[f"b{j}" for j in range(enc.n_binary_features)])
    tenc = TargetEncoder().fit(y)
    y_idx = tenc.transform(y)
    lam = float(payload["regularization"])
    model = SPLIT(lookahead_depth_budget=payload["lookahead_depth"], full_depth_budget=payload["full_depth"],
                  reg=lam, time_limit=max(1, int(payload["time_limit"])), verbose=False, binarize=False)
    t0 = time.perf_counter()
    model.fit(Xb, pd.Series(y_idx))
    seconds = time.perf_counter() - t0
    classes = list(model.clf.classes_)

    def convert(node, Xb_, y_):
        n = Xb_.shape[0]
        if isinstance(node, Leaf):
            label = int(classes[node.prediction])
            return {"prediction": tenc.inverse(label), "name": payload["target_name"],
                    "loss": float(np.sum(y_ != label)) / n, "complexity": lam}
        j = int(node.feature)
        rule = enc.rules[j]
        mask = Xb_.iloc[:, j].to_numpy() == 1
        return {"feature": int(rule["feature"]), "name": rule["name"], "relation": rule["relation"],
                "reference": rule["reference"], "type": rule["type"],
                "true": convert(node.left_child, Xb_[mask], y_[mask]),
                "false": convert(node.right_child, Xb_[~mask], y_[~mask])}

    return {"tree": convert(model.tree, Xb, y_idx), "time": seconds, "n_binary_features": int(Xb.shape[1])}


class Split:
    def __init__(self, regularization: float, time_limit: float, memory_limit: int = 6 * (1 << 30),
                 lookahead_depth: int = LOOKAHEAD_DEPTH, full_depth: int = FULL_DEPTH):
        self.regularization = regularization
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.lookahead_depth = lookahead_depth
        self.full_depth = full_depth

    def fit(self, X: pd.DataFrame, y):
        y_arr = np.asarray(y)
        if len(np.unique(y_arr)) != 2:
            raise NoModel("unsupported", "SPLIT only supports binary classification")
        payload = {"X": X.to_numpy().tolist(), "columns": [str(c) for c in X.columns], "y": y_arr.tolist(),
                   "regularization": self.regularization, "time_limit": self.time_limit,
                   "lookahead_depth": self.lookahead_depth, "full_depth": self.full_depth,
                   "target_name": str(getattr(y, "name", "class") or "class")}
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp) / "in.pkl", Path(tmp) / "out.json"
            inp.write_bytes(pickle.dumps(payload))
            t0 = time.perf_counter()
            proc = subprocess.Popen([sys.executable, __file__, str(inp), str(out)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            killed = ""
            while proc.poll() is None:
                time.sleep(0.25)
                if time.perf_counter() - t0 > 2 * self.time_limit + 60:
                    proc.kill(); killed = "time"
                elif self.memory_limit and _rss_bytes(proc.pid) > self.memory_limit:
                    proc.kill(); killed = "memory"
            self.wall_ = time.perf_counter() - t0
            stderr = proc.stderr.read() if proc.stderr else ""
            if killed:
                raise NoModel(killed, f"SPLIT stopped at the {killed} cap without a model")
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"SPLIT child failed: {stderr[-800:]}")
            res = json.loads(out.read_text())
        self.tree_ = res["tree"]
        self.time_ = float(res["time"])
        self.n_binary_features_ = res["n_binary_features"]
        self.optimal_ = False
        self.stop_reason_ = "heuristic"
        self.size_ = ""
        self.iterations_ = ""
        self.lowerbound_ = ""
        self.upperbound_ = ""
        return self


if __name__ == "__main__":  # child-process entry point
    _payload = pickle.loads(Path(sys.argv[1]).read_bytes())
    Path(sys.argv[2]).write_text(json.dumps(_fit_child(_payload)))
