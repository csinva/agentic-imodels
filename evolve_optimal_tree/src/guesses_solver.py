"""Wrapper that runs gosdt-guesses (McTavish et al., AAAI 2022; ``baselines/gosdt_guesses``)
with the solver interface used by ``evaluate.evaluate_solver``.

Two configurations:

* ``guesses=False`` (model name ``gosdt_guesses``): the exact solver of that
  code base on the same binarization as every other baseline (pygosdt_v1's
  encoder, which mirrors the original GOSDT encoder), no reference labels, no
  depth budget.  This is the newer C++ implementation of GOSDT solving the
  same problem as ``gosdt`` and ``pygosdt_v1``, so its results are directly
  comparable.
* ``guesses=True`` (model name ``gosdt_guesses_guided``): the paper's recipe,
  threshold guessing with a gradient-boosted ensemble of 40 stumps and the
  ensemble's predictions as reference labels for lower-bound guessing, still
  without a depth budget.  This changes the search space, so its objective can
  be worse than the true optimum; it is reported for speed, not as an exact
  solver.

Each fit runs in a child process so a runaway search can be killed on the
memory cap (the C++ core cannot be interrupted in-process).  Used by
``run_baselines.py --rerun`` and ``baselines/benchmarks/run_benchmark.py``.
Do not modify.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def available() -> bool:
    return importlib.util.find_spec("gosdt") is not None


def _rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
        return int(out.stdout.strip() or 0) * 1024
    except (ValueError, OSError):
        return 0


def _fit_child(payload: dict) -> dict:
    """Runs inside the child process: binarize, fit gosdt-guesses, return a JSON-able result."""
    from gosdt import GOSDTClassifier, Status, ThresholdGuessBinarizer
    from sklearn.ensemble import GradientBoostingClassifier

    sys.path.insert(0, str(ROOT))
    from pygosdt_v1 import BinaryEncoder

    X = pd.DataFrame(payload["X"], columns=payload["columns"])
    y = np.asarray(payload["y"])
    classes = np.unique(y)
    y_ref = None
    enc = BinaryEncoder().fit(X)
    Xb = enc.transform(X)
    rules = enc.rules
    negate = [False] * len(rules)
    if payload["guesses"]:
        # Threshold guessing on the exact binarization: the GBDT stumps select a
        # subset of the exact thresholds (the guesser needs numeric input, and the
        # suite has categorical columns), and its predictions give the reference
        # labels for lower-bound guessing.
        frame = pd.DataFrame(Xb.astype(float), columns=[f"b{j}" for j in range(Xb.shape[1])])
        guesser = ThresholdGuessBinarizer(n_estimators=40, max_depth=1, random_state=2021)
        guesser.set_output(transform="pandas")
        Xg = guesser.fit_transform(frame, y)
        gb = GradientBoostingClassifier(n_estimators=40, max_depth=1, random_state=42)
        gb.fit(Xg, y)
        y_ref = gb.predict(Xg)
        if set(np.unique(y_ref)) != set(classes):
            y_ref = None  # gosdt-guesses requires every class in y_ref; keep threshold guessing only
        # column "b<j> <= 0.5" is the negation of exact rule j
        picked = []
        negate = []
        for name in Xg.columns:
            feat, _, thr = str(name).rpartition(" <= ")
            j = int(feat[1:])
            picked.append(j)
            negate.append(float(thr) < 1.0)
        rules = [rules[j] for j in picked]
        Xb = Xg.to_numpy(dtype=bool)
    clf = GOSDTClassifier(regularization=payload["regularization"], time_limit=int(payload["time_limit"]),
                          allow_small_reg=True, similar_support=payload["similar_support"], verbose=False)
    clf.fit(Xb, y, y_ref=y_ref)
    r = clf.get_result()
    model = json.loads(r["models_string"])[0]
    tree = _decode(model, rules, negate, classes, payload["target_name"], payload["regularization"])
    return {
        "tree": tree, "time": float(r["time"]), "status": str(r["status"]).split(".")[-1],
        "graph_size": int(r["graph_size"]), "n_iterations": int(r["n_iterations"]),
        "lower_bound": float(r["lower_bound"]), "upper_bound": float(r["upper_bound"]),
        "n_binary_features": int(Xb.shape[1]),
    }


def _decode(node, rules, negate, classes, target_name, lam):
    if "prediction" in node:
        pred = classes[int(node["prediction"])]
        pred = pred.item() if isinstance(pred, np.generic) else pred
        return {"prediction": pred, "name": target_name, "loss": float(node.get("loss", 0.0)),
                "complexity": float(lam)}
    k = int(node["feature"])
    rule = rules[k]
    yes, no = node["true"], node["false"]
    if negate[k]:
        yes, no = no, yes
    return {"feature": int(rule["feature"]), "name": rule["name"], "relation": rule["relation"],
            "reference": rule["reference"], "type": rule["type"],
            "true": _decode(yes, rules, negate, classes, target_name, lam),
            "false": _decode(no, rules, negate, classes, target_name, lam)}


class GuessesGOSDT:
    def __init__(self, regularization: float, time_limit: float, memory_limit: int = 6 * (1 << 30),
                 guesses: bool = False, similar_support: bool = False):
        self.regularization = regularization
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.guesses = guesses
        self.similar_support = similar_support

    def fit(self, X: pd.DataFrame, y):
        payload = {"X": X.to_numpy().tolist(), "columns": [str(c) for c in X.columns],
                   "y": np.asarray(y).tolist(), "regularization": self.regularization,
                   "time_limit": self.time_limit, "guesses": self.guesses,
                   "similar_support": self.similar_support,
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
                raise RuntimeError(f"gosdt-guesses stopped at the {killed} cap without a model")
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"gosdt-guesses child failed: {stderr[-800:]}")
            res = json.loads(out.read_text())
        self.tree_ = res["tree"]
        self.time_ = res["time"]
        converged = res["status"] == "CONVERGED"
        # with guesses the search space is restricted, so convergence is not a
        # certificate of global optimality: report it as a heuristic result
        self.optimal_ = converged and not self.guesses
        if self.guesses:
            self.stop_reason_ = "heuristic" if converged else "time"
        else:
            self.stop_reason_ = "optimal" if converged else ("time" if res["status"] == "TIMEOUT" else res["status"].lower())
        self.size_ = res["graph_size"]
        self.iterations_ = res["n_iterations"]
        self.lowerbound_ = res["lower_bound"]
        self.upperbound_ = res["upper_bound"]
        self.n_binary_features_ = res["n_binary_features"]
        return self


if __name__ == "__main__":  # child entry point
    payload = pickle.loads(Path(sys.argv[1]).read_bytes())
    result = _fit_child(payload)
    Path(sys.argv[2]).write_text(json.dumps(result))
