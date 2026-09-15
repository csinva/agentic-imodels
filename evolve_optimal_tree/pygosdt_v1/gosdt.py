"""User-facing estimators.

``GOSDTClassifier`` is a scikit-learn compatible estimator.  ``GOSDT`` mirrors
the reference ``python/model/gosdt.py`` wrapper (``GOSDT(config).fit(X, y)``
with ``.time``, ``.iterations``, ``.size`` and ``.tree`` attributes).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .dataset import BitDataset
from .encoder import BinaryEncoder, TargetEncoder, _to_dataframe
from .model import TreeClassifier
from .optimizer import Optimizer


class GOSDTClassifier(BaseEstimator, ClassifierMixin):
    """Optimal sparse decision tree minimising ``loss + regularization * leaves``.

    Parameters
    ----------
    regularization : float, default 0.05
        Penalty per leaf.  The reference implementation recommends values
        above ``1 / n_samples``.
    time_limit : float, default 0
        Seconds after which the search stops and the best tree found so far is
        returned (``optimal_`` is then ``False``).  ``0`` means unlimited.
    balance : bool, default False
        Weigh every class equally (balanced accuracy).
    costs : (K, K) array-like, optional
        ``costs[i, j]`` is the cost of predicting class ``i`` when the true class
        is ``j``.  Overrides ``balance``.
    upperbound : float, default 0
        Optional trusted upper bound on the optimal risk used for pruning.
    look_ahead, similar_support, feature_exchange, continuous_feature_exchange : bool
        Enable the corresponding bounds (all exact; they only affect speed).
    greedy_init : bool, default True
        Seed the search with a greedy tree optimising the same objective.
    engine : {"auto", "numba", "python"}, default "auto"
        Bit-counting backend; ``auto`` uses numba when it is installed.  Both
        engines produce identical trees.
    memory_limit : int, default 0
        Resident memory (bytes) above which the search stops like a time limit.
    verbose : bool, default False
    """

    def __init__(self, regularization: float = 0.05, time_limit: float = 0.0,
                 balance: bool = False, costs=None, upperbound: float = 0.0,
                 look_ahead: bool = True, similar_support: bool = True,
                 feature_exchange: bool = True, continuous_feature_exchange: bool = True,
                 greedy_init: bool = True, engine: str = "auto", memory_limit: int = 0,
                 verbose: bool = False):
        self.regularization = regularization
        self.memory_limit = memory_limit
        self.time_limit = time_limit
        self.balance = balance
        self.costs = costs
        self.upperbound = upperbound
        self.look_ahead = look_ahead
        self.similar_support = similar_support
        self.feature_exchange = feature_exchange
        self.continuous_feature_exchange = continuous_feature_exchange
        self.greedy_init = greedy_init
        self.engine = engine
        self.verbose = verbose

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, target_name: str = "class"):
        X = _to_dataframe(X)
        if isinstance(y, pd.DataFrame):
            target_name = str(y.columns[0])
            y = y.iloc[:, 0]
        elif isinstance(y, pd.Series) and y.name is not None:
            target_name = str(y.name)
        y = np.asarray(y).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have different numbers of rows")

        t0 = time.perf_counter()
        self.encoder_ = BinaryEncoder().fit(X)
        Xb = self.encoder_.transform(X)
        self.target_encoder_ = TargetEncoder().fit(y)
        y_idx = self.target_encoder_.transform(y)
        self.classes_ = self.target_encoder_.classes_
        data = BitDataset(Xb, y_idx, len(self.classes_), costs=self.costs, balance=self.balance)
        self.encoding_time_ = time.perf_counter() - t0
        self.n_binary_features_ = data.m

        if self.verbose:
            print(f"Dataset Dimensions: {data.n} x {data.m} x {data.K}")

        opt = Optimizer(
            data, self.regularization, groups=self.encoder_.groups,
            time_limit=self.time_limit, look_ahead=self.look_ahead,
            similar_support=self.similar_support, feature_exchange=self.feature_exchange,
            continuous_feature_exchange=self.continuous_feature_exchange,
            greedy_init=self.greedy_init, upperbound=self.upperbound, engine=self.engine,
            memory_limit=self.memory_limit, verbose=self.verbose,
        )
        root = opt.run()
        self.optimal_ = opt.optimal
        self.stop_reason_ = opt.stop_reason
        self.time_ = opt.elapsed
        self.iterations_ = opt.iterations
        self.size_ = len(opt.memo)
        self.lowerbound_ = root.lb
        self.upperbound_ = root.ub
        if not self.optimal_:
            warnings.warn(f"{self.stop_reason_} limit reached before optimality was certified; "
                          "returning the best tree found", RuntimeWarning)

        raw = opt.extract(root)
        self.tree_ = self._decode(raw, data, target_name)
        self.tree = TreeClassifier(self.tree_)
        self.objective_ = self.tree.risk()
        self.n_leaves_ = self.tree.leaves()
        if self.verbose:
            print(f"Training Duration: {self.time_:.4f} seconds")
            print(f"Number of Iterations: {self.iterations_}")
            print(f"Size of Graph: {self.size_}")
            print(f"Objective Boundary: [{root.lb}, {root.ub}]")
            print(f"Loss: {self.tree.loss()}  Complexity: {self.tree.complexity()}")
        return self

    def _decode(self, node: dict, data: BitDataset, target_name: str) -> dict:
        if "prediction" in node:
            key = node["key"]
            _, _, max_loss, _, _, prediction = data.leaf_stats(key)
            return {
                "prediction": self.target_encoder_.inverse(prediction),
                "name": target_name,
                "loss": float(max_loss),
                "complexity": float(self.regularization),
            }
        rule = self.encoder_.rules[node["feature"]]
        return {
            "feature": int(rule["feature"]),
            "name": rule["name"],
            "relation": rule["relation"],
            "reference": rule["reference"],
            "type": rule["type"],
            "true": self._decode(node["true"], data, target_name),
            "false": self._decode(node["false"], data, target_name),
        }

    # -------------------------------------------------------------- predict
    def predict(self, X):
        X = _to_dataframe(X)
        pred = self.tree.predict_fast(X)
        return np.array(pred.tolist())

    def score(self, X, y, sample_weight=None):
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    # -------------------------------------------------------------- reports
    def leaves(self) -> int:
        return self.tree.leaves()

    def nodes(self) -> int:
        return self.tree.nodes()

    def max_depth(self) -> int:
        return self.tree.maximum_depth()

    def json(self) -> str:
        return self.tree.json()

    def __str__(self):
        return str(self.tree)


class GOSDT:
    """Drop-in replacement for the reference Python wrapper.

    >>> model = GOSDT({"regularization": 0.1, "time_limit": 3600})
    >>> model.fit(X, y)
    >>> model.tree, model.time, model.iterations, model.size
    """

    def __init__(self, configuration: dict | None = None):
        self.configuration = dict(configuration or {})
        self.time = 0.0
        self.iterations = 0
        self.size = 0
        self.tree = None
        self.model_ = None

    def fit(self, X, y):
        cfg = self.configuration
        if "objective" in cfg and cfg["objective"] not in (None, "acc"):
            raise NotImplementedError("only the accuracy objective is supported")
        kwargs = dict(
            regularization=cfg.get("regularization", 0.05),
            time_limit=cfg.get("time_limit", 0.0),
            balance=cfg.get("balance", False),
            upperbound=cfg.get("upperbound", 0.0),
            look_ahead=cfg.get("look_ahead", True),
            similar_support=cfg.get("similar_support", True),
            feature_exchange=cfg.get("feature_exchange", True),
            continuous_feature_exchange=cfg.get("continuous_feature_exchange", True),
            verbose=cfg.get("verbose", False),
            engine=cfg.get("engine", "auto"),
        )
        if cfg.get("costs"):
            kwargs["costs"] = _read_cost_matrix(cfg["costs"], y)
        self.model_ = GOSDTClassifier(**kwargs).fit(X, y)
        self.tree = self.model_.tree
        self.time = self.model_.time_
        self.iterations = self.model_.iterations_
        self.size = self.model_.size_
        return self

    def status(self):
        return 0 if self.model_.optimal_ else 1

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y, weight=None):
        return self.tree.score(X, y, weight=weight)

    def error(self, X, y, weight=None):
        return self.tree.error(X, y, weight=weight)

    def leaves(self):
        return self.tree.leaves()

    def nodes(self):
        return self.tree.nodes()

    def max_depth(self):
        return self.tree.maximum_depth()

    def json(self):
        return self.tree.json()

    def __len__(self):
        return self.tree.leaves()


def _read_cost_matrix(path: str, y) -> np.ndarray:
    """Parse the reference cost CSV (header of class names, then rows)."""
    table = pd.read_csv(path)
    classes = np.unique(np.asarray(y).ravel())
    names = [str(c) for c in table.columns]
    K = len(classes)
    C = np.zeros((K, K))
    col_of = {name: i for i, name in enumerate(names)}
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i == j:
                continue
            if table.shape[0] == 1:
                C[i, j] = float(table.iloc[0, col_of[str(cj)]])
            else:
                C[i, j] = float(table.iloc[col_of[str(ci)], col_of[str(cj)]])
    return C
