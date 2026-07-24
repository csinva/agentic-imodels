"""
Tuned-baselines experiment (rebuttal: mpaA W3, TdWu, dD5H Q3).

Compares, on a common pool of PMLB + OpenML regression datasets:
  (a) default-hyperparameter baselines (as in the paper),
  (b) hyperparameter-TUNED baselines (RandomizedSearchCV), and
  (c) the 10 evolved agentic-imodels library models.

Reports mean RMSE rank (lower = better) within the shared pool, so we can
see whether tuning the strong baselines changes the Pareto/frontier claim.

Protocol mirrors evolve/src/performance_eval.py:
  - subsample to <=1000 train samples, <=50 features (seed 42)
  - normalize y with train-set mean/std
  - 80/20 train/test split (seed 42), RMSE on test.

Run:  python tuned_baselines_experiment.py
Out:  scratchpad/tuned_baselines/{perf_long.csv, summary.csv}
"""
import os, sys, warnings, time, json
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all="ignore")
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from pmlb import fetch_data, regression_dataset_names

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "result_libs_processed", "agentic-imodels"))
from agentic_imodels import (
    HingeEBMRegressor, HybridGAM, SmartAdditiveRegressor, HingeGAMRegressor,
    TeacherStudentRuleSplineRegressor, DualPathSparseSymbolicRegressor,
    SparseSignedBasisPursuitRegressor, DistilledTreeBlendAtlasRegressor,
    WinsorizedSparseOLSRegressor, TinyDTDepth2Regressor)

OUT = "/private/tmp/claude-501/-Users-chandansingh-Downloads/f32dcc95-9574-4888-b12b-77a8bb0d41d5/scratchpad/tuned_baselines"
os.makedirs(OUT, exist_ok=True)
MAX_SAMPLES, MAX_FEATURES, SEED = 1000, 50, 42
N_ITER, CV = 20, 3
N_DATASETS = 30  # tractable representative pool

def subsample(Xtr, Xte, ytr, yte):
    rng = np.random.RandomState(SEED)
    if Xtr.shape[1] > MAX_FEATURES:
        fi = np.sort(rng.choice(Xtr.shape[1], MAX_FEATURES, replace=False))
        Xtr, Xte = Xtr[:, fi], Xte[:, fi]
    if len(Xtr) > MAX_SAMPLES:
        idx = rng.choice(len(Xtr), MAX_SAMPLES, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    return Xtr, Xte, ytr, yte

# ---- tuned-baseline factories (fresh estimator per dataset) ----
def tuned_rf():
    return RandomizedSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=1),
        {"n_estimators": randint(50, 250), "max_depth": [3,5,8,12,None],
         "min_samples_leaf": randint(1,10), "max_features": ["sqrt","log2",1.0,0.5]},
        n_iter=N_ITER, cv=CV, random_state=SEED, n_jobs=1)
def tuned_gbm():
    return RandomizedSearchCV(GradientBoostingRegressor(random_state=SEED),
        {"n_estimators": randint(50, 400), "max_depth": randint(2,6),
         "learning_rate": loguniform(1e-2, 3e-1), "subsample": uniform(0.6,0.4),
         "min_samples_leaf": randint(1,10)},
        n_iter=N_ITER, cv=CV, random_state=SEED, n_jobs=1)
def tuned_mlp():
    return RandomizedSearchCV(MLPRegressor(random_state=SEED, max_iter=300, early_stopping=True),
        {"hidden_layer_sizes": [(64,),(128,),(64,64),(128,64),(100,50,25)],
         "alpha": loguniform(1e-5,1e-1), "learning_rate_init": loguniform(1e-4,1e-2),
         "activation": ["relu","tanh"]},
        n_iter=12, cv=CV, random_state=SEED, n_jobs=1)
def tuned_ridge():
    return RandomizedSearchCV(Ridge(random_state=SEED),
        {"alpha": loguniform(1e-3, 1e3)}, n_iter=20, cv=CV, random_state=SEED, n_jobs=1)
def tuned_ebm():
    # EBM fits are ~10s each, so keep the search small (n_iter=5, cv=2 => 10 fits)
    return RandomizedSearchCV(ExplainableBoostingRegressor(random_state=SEED, outer_bags=2),
        {"max_bins": [128,256], "learning_rate": loguniform(5e-3,3e-1),
         "interactions": [0,5], "min_samples_leaf": randint(2,10)},
        n_iter=5, cv=2, random_state=SEED, n_jobs=1)

def build_models():
    m = {}
    # default baselines (mirroring run_baselines.py)
    m["OLS"] = LinearRegression()
    m["RidgeCV"] = RidgeCV()
    m["LassoCV"] = LassoCV(cv=3)
    m["DT_large"] = DecisionTreeRegressor(max_leaf_nodes=20, random_state=SEED)
    m["RF_default"] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=SEED)
    m["GBM_default"] = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED)
    m["MLP_default"] = MLPRegressor(random_state=SEED)
    m["EBM_default"] = ExplainableBoostingRegressor(random_state=SEED, outer_bags=3, max_rounds=1000)
    # tuned baselines
    m["RF_tuned"] = tuned_rf()
    m["GBM_tuned"] = tuned_gbm()
    m["MLP_tuned"] = tuned_mlp()
    m["Ridge_tuned"] = tuned_ridge()
    m["EBM_tuned"] = tuned_ebm()
    # evolved library
    m["HingeEBM"] = HingeEBMRegressor()
    m["HybridGAM"] = HybridGAM()
    m["SmartAdditive"] = SmartAdditiveRegressor()
    m["HingeGAM"] = HingeGAMRegressor()
    m["TeacherStudentRuleSpline"] = TeacherStudentRuleSplineRegressor()
    m["DualPathSparseSymbolic"] = DualPathSparseSymbolicRegressor()
    m["SparseSignedBasisPursuit"] = SparseSignedBasisPursuitRegressor()
    m["DistilledTreeBlendAtlas"] = DistilledTreeBlendAtlasRegressor()
    m["WinsorizedSparseOLS"] = WinsorizedSparseOLSRegressor()
    m["TinyDT"] = TinyDTDepth2Regressor()
    return m

MODEL_GROUP = {}
for k in ["OLS","RidgeCV","LassoCV","DT_large","RF_default","GBM_default","MLP_default","EBM_default"]:
    MODEL_GROUP[k] = "default_baseline"
for k in ["RF_tuned","GBM_tuned","MLP_tuned","Ridge_tuned","EBM_tuned"]:
    MODEL_GROUP[k] = "tuned_baseline"
for k in ["HingeEBM","HybridGAM","SmartAdditive","HingeGAM","TeacherStudentRuleSpline",
          "DualPathSparseSymbolic","SparseSignedBasisPursuit","DistilledTreeBlendAtlas",
          "WinsorizedSparseOLS","TinyDT"]:
    MODEL_GROUP[k] = "evolved"

def eval_dataset(name):
    from copy import deepcopy
    try:
        df = fetch_data(name, local_cache_dir="/tmp/pmlbcache")
    except Exception as e:
        return name, {}, "load_fail:%s" % str(e)[:60]
    y = df["target"].values.astype(float)
    X = df.drop(columns=["target"]).values.astype(np.float32)
    valid = ~np.isnan(y); X, y = X[valid], y[valid]
    if len(X) < 30 or X.shape[1] < 1:
        return name, {}, "too_small"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    Xtr, Xte, ytr, yte = subsample(Xtr, Xte, ytr, yte)
    ym, ys = float(ytr.mean()), float(ytr.std())
    if ys > 0:
        ytr = (ytr - ym) / ys; yte = (yte - ym) / ys
    rmses = {}
    for mname, model in build_models().items():
        try:
            mm = deepcopy(model); mm.fit(Xtr, ytr)
            p = mm.predict(Xte)
            rmses[mname] = float(np.sqrt(mean_squared_error(yte, p)))
        except Exception:
            rmses[mname] = np.nan
    try:
        with open(os.path.join(OUT, "progress.txt"), "a") as pf:
            pf.write(name + "\n")
    except Exception:
        pass
    return name, rmses, "ok"

if __name__ == "__main__":
    names = sorted(n for n in regression_dataset_names if "_fri_" not in n)[:N_DATASETS]
    print(f"[tuned-baselines] {len(names)} PMLB datasets, {len(build_models())} models, "
          f"n_iter={N_ITER}, cv={CV}", flush=True)
    open(os.path.join(OUT, "progress.txt"), "w").close()
    from joblib import Parallel, delayed
    t0 = time.time()
    results = Parallel(n_jobs=6)(
        delayed(eval_dataset)(n) for n in names)
    rows = []
    for name, rmses, status in results:
        print(f"  {name:<28} {status}", flush=True)
        for mname, v in rmses.items():
            rows.append({"dataset": name, "model": mname, "group": MODEL_GROUP[mname], "rmse": v})
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(OUT, "perf_long.csv"), index=False)

    # rank within each dataset over models present on ALL datasets
    piv = long.pivot_table(index="dataset", columns="model", values="rmse")
    complete = piv.dropna(axis=1, how="any")  # models valid on every dataset
    ranks = complete.rank(axis=1, method="average")  # 1 = best (lowest rmse)
    summary = pd.DataFrame({
        "mean_rank": ranks.mean(axis=0),
        "median_rank": ranks.median(axis=0),
        "mean_rmse": complete.mean(axis=0),
        "n_datasets": ranks.shape[0],
    })
    summary["group"] = [MODEL_GROUP[m] for m in summary.index]
    summary = summary.sort_values("mean_rank")
    summary.to_csv(os.path.join(OUT, "summary.csv"))
    print("\n==== SUMMARY (mean rank over %d datasets, %d models in complete pool) ====" %
          (ranks.shape[0], complete.shape[1]), flush=True)
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"), flush=True)
    print(f"\n[done] {time.time()-t0:.0f}s", flush=True)
