"""Binarization of arbitrary tabular data into boolean split features.

This mirrors the semantics of the reference C++ ``Encoder``:

* numeric columns become threshold predicates ``x >= t`` where ``t`` is the
  midpoint between consecutive observed values (for integer columns the
  reference value is reported as the upper observed value, which induces the
  same partition);
* categorical columns become equality predicates ``x == v`` for each value;
* columns with exactly two values and no missing entries become a single
  ``x == v`` predicate on the larger value;
* constant columns are dropped;
* missing values never satisfy a predicate.

Additionally, binary columns that induce the same partition of the rows as an
earlier column (identical or complementary) are dropped, because they can never
change the optimal objective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MISSING_STRINGS = {"", "NULL", "null", "Null", "NA", "na", "NaN", "nan", "N/A", "n/a"}


def _to_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])


def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    return pd.api.types.is_numeric_dtype(s)


class BinaryEncoder:
    """Fit/transform arbitrary features into a boolean split matrix.

    Attributes after ``fit``:

    rules : list of dict
        One entry per binary feature with keys ``feature`` (source column
        index), ``name`` (source column name), ``relation`` (``">="`` or
        ``"=="``), ``reference`` (threshold or category value) and ``type``
        (``"integral"``, ``"rational"`` or ``"categorical"``).
    groups : list of list of int
        Indices of binary features (in threshold order) that belong to the
        same ordinal source column with more than one threshold.
    """

    def __init__(self, drop_duplicate_columns: bool = True):
        self.drop_duplicate_columns = drop_duplicate_columns
        self.rules: list[dict] = []
        self.groups: list[list[int]] = []
        self.feature_names: list[str] = []
        self.n_source_features = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X) -> "BinaryEncoder":
        X = _to_dataframe(X)
        self.feature_names = [str(c) for c in X.columns]
        self.n_source_features = X.shape[1]
        rules: list[dict] = []
        groups: list[list[int]] = []

        for j, col in enumerate(X.columns):
            s = X[col]
            name = str(col)
            if _is_numeric_series(s):
                values = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                has_missing = finite.shape[0] != values.shape[0]
                uniq = np.unique(finite)
                if uniq.shape[0] <= 1:
                    continue
                integral = bool(np.all(np.equal(np.mod(uniq, 1), 0)))
                kind = "integral" if integral else "rational"
                if uniq.shape[0] == 2 and not has_missing:
                    ref = uniq[1]
                    rules.append({
                        "feature": j, "name": name, "relation": "==",
                        "reference": int(ref) if integral else float(ref), "type": kind,
                    })
                    continue
                start = len(rules)
                for a, b in zip(uniq[:-1], uniq[1:]):
                    if integral:
                        ref = int(b)
                    else:
                        ref = float(0.5 * (a + b))
                    rules.append({
                        "feature": j, "name": name, "relation": ">=",
                        "reference": ref, "type": kind,
                    })
                groups.append(list(range(start, len(rules))))
            else:
                raw = s.to_numpy(dtype=object)
                mask = np.array([not _is_missing(v) for v in raw], dtype=bool)
                present = raw[mask]
                has_missing = present.shape[0] != raw.shape[0]
                uniq = sorted(set(present.tolist()), key=lambda v: str(v))
                if len(uniq) <= 1:
                    continue
                if len(uniq) == 2 and not has_missing:
                    uniq = uniq[1:]
                for v in uniq:
                    rules.append({
                        "feature": j, "name": name, "relation": "==",
                        "reference": v, "type": "categorical",
                    })

        self.rules = rules
        self.groups = groups

        if self.drop_duplicate_columns and rules:
            Xb = self._apply_rules(X, rules)
            keep = _unique_partitions(Xb)
            if keep.shape[0] != len(rules):
                remap = {old: new for new, old in enumerate(keep.tolist())}
                self.rules = [rules[i] for i in keep.tolist()]
                self.groups = [
                    g2 for g2 in ([remap[i] for i in g if i in remap] for g in groups)
                    if len(g2) > 1
                ]
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X) -> np.ndarray:
        X = _to_dataframe(X)
        if X.shape[1] != self.n_source_features:
            raise ValueError(
                f"expected {self.n_source_features} feature columns, got {X.shape[1]}"
            )
        return self._apply_rules(X, self.rules)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_binary_features(self) -> int:
        return len(self.rules)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _apply_rules(X: pd.DataFrame, rules: list[dict]) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros((n, len(rules)), dtype=bool)
        cache: dict[int, np.ndarray] = {}
        for k, rule in enumerate(rules):
            j = rule["feature"]
            if rule["relation"] == ">=":
                if j not in cache:
                    cache[j] = pd.to_numeric(X.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
                col = cache[j]
                with np.errstate(invalid="ignore"):
                    out[:, k] = col >= rule["reference"]
            else:
                if rule["type"] == "categorical":
                    col = X.iloc[:, j].to_numpy(dtype=object)
                    ref = rule["reference"]
                    out[:, k] = np.array([(not _is_missing(v)) and v == ref for v in col], dtype=bool)
                else:
                    if j not in cache:
                        cache[j] = pd.to_numeric(X.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
                    col = cache[j]
                    with np.errstate(invalid="ignore"):
                        out[:, k] = col == rule["reference"]
        return out


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v in _MISSING_STRINGS:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _unique_partitions(Xb: np.ndarray) -> np.ndarray:
    """Return indices of columns whose induced row partition is new.

    A column and its complement induce the same partition, so columns are
    canonicalised by flipping them when their first row is ``True``.
    """
    if Xb.shape[1] == 0:
        return np.arange(0)
    canon = Xb ^ Xb[0:1, :]
    packed = np.packbits(canon, axis=0)
    seen: dict[bytes, int] = {}
    keep = []
    for k in range(packed.shape[1]):
        key = packed[:, k].tobytes()
        if key in seen:
            continue
        seen[key] = k
        keep.append(k)
    return np.array(keep, dtype=np.int64)


class TargetEncoder:
    """Map arbitrary labels to contiguous integer class indices."""

    def __init__(self):
        self.classes_: np.ndarray | None = None

    def fit(self, y) -> "TargetEncoder":
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        y = np.asarray(y).ravel()
        idx = np.searchsorted(self.classes_, y)
        if np.any(idx >= self.classes_.shape[0]) or np.any(self.classes_[np.minimum(idx, len(self.classes_) - 1)] != y):
            raise ValueError("labels contain classes not seen during fit")
        return idx.astype(np.int64)

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse(self, idx: int):
        v = self.classes_[idx]
        if isinstance(v, np.generic):
            return v.item()
        return v
