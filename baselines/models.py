"""
Shared model definitions for both interpretability tests and TabArena evaluation.

Exports:
  REGRESSOR_DEFS  — list of (name, regressor) for run_all_interpretability_tests.py
  CLASSIFIER_DEFS — list of (name, classifier) for run_tabarena.py
  MODEL_GROUPS    — dict mapping group name → set of model names
  GROUP_COLORS    — dict mapping group name → hex color
"""

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression, Lasso, LassoCV, RidgeCV,
    LogisticRegression, LogisticRegressionCV,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Regressor definitions (used by run_all_interpretability_tests.py)
# ---------------------------------------------------------------------------

REGRESSOR_DEFS = [
    ("DT_mini",    DecisionTreeRegressor(max_leaf_nodes=8,  random_state=42)),
    ("DT_large",   DecisionTreeRegressor(max_leaf_nodes=20, random_state=42)),
    ("OLS",        LinearRegression()),
    ("LASSO",      Lasso(alpha=0.1)),
    ("LassoCV",    LassoCV(cv=5)),
    ("RidgeCV",    RidgeCV()),
    ("RF",         RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ("GBM",        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
    ("MLP",        MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000,
                                random_state=42, learning_rate_init=0.01)),
]

try:
    from pygam import LinearGAM
    REGRESSOR_DEFS = [("GAM", LinearGAM(n_splines=10))] + REGRESSOR_DEFS
except ImportError:
    pass

try:
    from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor, TreeGAMRegressor
    REGRESSOR_DEFS += [
        ("FIGS_mini",    FIGSRegressor(max_rules=8,  random_state=42)),
        ("FIGS_large",   FIGSRegressor(max_rules=20, random_state=42)),
        ("RuleFit",      RuleFitRegressor(max_rules=20, random_state=42)),
        ("HSTree_mini",  HSTreeRegressor(max_leaf_nodes=8,  random_state=42)),
        ("HSTree_large", HSTreeRegressor(max_leaf_nodes=20, random_state=42)),
        ("TreeGAM",      TreeGAMRegressor(n_boosting_rounds=5, max_leaf_nodes=4, random_state=42)),
    ]
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Classifier definitions (used by run_tabarena.py)
# ---------------------------------------------------------------------------

def _lr(penalty, **kw):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(penalty=penalty, solver="saga",
                                                max_iter=200, random_state=42, **kw))])

def _lr_cv(penalty):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegressionCV(penalty=penalty, solver="saga",
                                                  Cs=10, cv=5, max_iter=200,
                                                  random_state=42))])

CLASSIFIER_DEFS = [
    ("DT_mini",    DecisionTreeClassifier(max_leaf_nodes=8,  random_state=42)),
    ("DT_large",   DecisionTreeClassifier(max_leaf_nodes=20, random_state=42)),
    ("OLS",        _lr("l2", C=1e6)),   # near-unregularized logistic ≈ OLS
    ("LASSO",      _lr("l1", C=10.0)),
    ("LassoCV",    _lr_cv("l1")),
    ("RidgeCV",    _lr_cv("l2")),
    ("RF",         RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ("GBM",        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ("MLP",        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000,
                                 random_state=42, learning_rate_init=0.01)),
]

try:
    from imodels import (
        FIGSClassifier, RuleFitClassifier, HSTreeClassifier, TreeGAMClassifier,
    )
    CLASSIFIER_DEFS += [
        ("FIGS_mini",    FIGSClassifier(max_rules=8)),
        ("FIGS_large",   FIGSClassifier(max_rules=20)),
        ("RuleFit",      RuleFitClassifier(max_rules=20, random_state=42)),
        ("HSTree_mini",  HSTreeClassifier(max_leaf_nodes=8,  random_state=42)),
        ("HSTree_large", HSTreeClassifier(max_leaf_nodes=20, random_state=42)),
        ("TreeGAM",      TreeGAMClassifier(n_boosting_rounds=5, max_leaf_nodes=4,
                                           random_state=42)),
    ]
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Shared metadata: groups and colors
# ---------------------------------------------------------------------------

MODEL_GROUPS = {
    "black-box": {"RF", "GBM", "MLP"},
    "imodels":   {"FIGS_mini", "FIGS_large", "RuleFit", "HSTree_mini", "HSTree_large", "TreeGAM"},
    "linear":    {"OLS", "LASSO", "LassoCV", "RidgeCV"},
    "tree":      {"DT_mini", "DT_large"},
    "gam":       {"GAM"},
}

GROUP_COLORS = {
    "black-box": "#e74c3c",
    "imodels":   "#27ae60",
    "linear":    "#2980b9",
    "tree":      "#e67e22",
    "gam":       "#8e44ad",
}
