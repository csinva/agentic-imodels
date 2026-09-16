"""Wrapper that runs the reference C++ GOSDT binary with the solver interface used
by ``evaluate.evaluate_solver`` (``fit``, ``tree_``, ``optimal_``, ``time_``).

Used by ``run_baselines.py --rerun`` and ``baselines/benchmarks/run_benchmark.py``.
Requires ``baselines/gosdt/build/gosdt`` (see ``baselines/gosdt_patches/apply.sh``).
Do not modify.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from evaluate import NoModel
from suite import ROOT

BINARY = ROOT / "baselines" / "gosdt" / "build" / "gosdt"


def available() -> bool:
    return BINARY.exists()


def _rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
        return int(out.stdout.strip() or 0) * 1024
    except (ValueError, OSError):
        return 0


class ReferenceGOSDT:
    """Run ``gosdt`` on a DataFrame ``X`` with labels ``y`` (written as the last CSV column)."""

    def __init__(self, regularization: float, time_limit: float, memory_limit: int = 6 * (1 << 30),
                 workers: int = 1):
        self.regularization = regularization
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.workers = workers

    def fit(self, X: pd.DataFrame, y):
        frame = X.copy()
        frame["class"] = pd.Series(y).to_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "data.csv"
            frame.to_csv(csv, index=False)
            model_path = Path(tmp) / "model.json"
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(json.dumps({
                "regularization": self.regularization, "verbose": True, "worker_limit": int(self.workers),
                "model_limit": 1, "time_limit": int(self.time_limit), "model": str(model_path)}))
            with open(csv, "rb") as fh, open(Path(tmp) / "out.txt", "w+") as out_fh:
                t0 = time.perf_counter()
                proc = subprocess.Popen([str(BINARY), str(cfg)], stdin=fh, stdout=out_fh,
                                        stderr=subprocess.STDOUT, text=True)
                killed = ""
                while proc.poll() is None:
                    time.sleep(0.25)
                    # the binary checks its clock only every 10,000 iterations
                    if time.perf_counter() - t0 > 2 * self.time_limit + 30:
                        proc.kill()
                        killed = "timeout"
                    elif self.memory_limit and _rss_bytes(proc.pid) > self.memory_limit:
                        proc.kill()
                        killed = "memory"
                self.wall_ = time.perf_counter() - t0
                out_fh.seek(0)
                out = out_fh.read()
            m = re.search(r"Training Duration: ([0-9.eE+-]+) seconds", out)
            self.time_ = float(m.group(1)) if m else self.wall_
            m = re.search(r"Optimality Gap: ([0-9.eE+-]+)", out)
            gap = float(m.group(1)) if m else float("nan")
            self.optimal_ = (killed == "" and gap == 0.0)
            self.stop_reason_ = killed or ("optimal" if self.optimal_ else "time")
            if self.stop_reason_ == "timeout":
                self.stop_reason_ = "time"
            m = re.search(r"Size of Graph: (\d+)", out)
            self.size_ = int(m.group(1)) if m else ""
            m = re.search(r"Number of Iterations: (\d+)", out)
            self.iterations_ = int(m.group(1)) if m else ""
            m = re.search(r"Binary Dataset Dimension: (\d+) x (\d+)", out)
            self.n_binary_features_ = int(m.group(2)) if m else ""
            m = re.search(r"Objective Boundary: \[([0-9.eE+-]+), ([0-9.eE+-]+)\]", out)
            self.lowerbound_, self.upperbound_ = (float(m.group(1)), float(m.group(2))) if m else ("", "")
            if not model_path.exists():
                if killed:
                    raise NoModel("time" if killed == "timeout" else killed,
                                  f"reference stopped at the {killed} cap without a model")
                raise RuntimeError(f"reference produced no model ({self.stop_reason_})")
            models = json.loads(model_path.read_text())
            if not models:
                raise RuntimeError("reference produced an empty model list")
            self.tree_ = models[0]
        return self
