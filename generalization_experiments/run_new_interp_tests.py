"""Run the 157 NEW interpretability tests and 16-OpenML performance eval on
baselines + every evolved model found in `../result_libs/*/interpretable_regressors_lib/success/`.

For each result_libs folder, writes
    interpretability_results_test.csv
    overall_results_test.csv
    interpretability_vs_performance_test.png

Caching is keyed by (unique_model_name, test_name, model_instance) via joblib
Memory so that a single baseline is computed once across all folders.

Usage: uv run run_new_interp_tests.py
"""

import csv
import importlib.util
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from joblib import Memory, Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "evolve" / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evolve"))

from performance_eval import (  # noqa: E402
    MIN_SAMPLES, MIN_FEATURES,
    subsample_dataset, OVERALL_CSV_COLS,
)
from interp_eval import CHECKPOINT  # noqa: E402
from visualize import plot_interp_vs_performance  # noqa: E402

import imodelsx.llm  # noqa: E402

from new_interp_tests import ALL_TESTS, _ALL_TEST_FNS, category_of  # noqa: E402

# Baselines
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LassoCV as SkLassoCV, LinearRegression, RidgeCV as SkRidgeCV  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from pygam import LinearGAM  # noqa: E402
from imodels import FIGSRegressor, HSTreeRegressorCV, RuleFitRegressor  # noqa: E402
from interpret.glassbox import ExplainableBoostingRegressor  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402


class TabPFN200(TabPFNRegressor):
    """TabPFN wrapper that caps training data at 200 samples — keeps CPU
    latency tractable on the 16 held-out OpenML datasets (the underlying model
    prints a `>200 samples on CPU may be slow` warning, which we heed)."""

    def fit(self, X, y):
        import numpy as _np
        X = _np.asarray(X); y = _np.asarray(y)
        if X.shape[0] > 200:
            rng = _np.random.RandomState(42)
            idx = rng.choice(X.shape[0], 200, replace=False)
            X, y = X[idx], y[idx]
        return super().fit(X, y)

import openml  # noqa: E402


RESULT_LIBS_DIR = SCRIPT_DIR.parent / "result_libs"
CACHE_ROOT = SCRIPT_DIR / ".cache_new_tests"
CACHE_ROOT.mkdir(exist_ok=True)

# Separate caches for interp vs perf to keep sizes manageable.
_interp_cache = Memory(location=str(CACHE_ROOT / "interp"), verbose=0)
_perf_cache = Memory(location=str(CACHE_ROOT / "perf"), verbose=0)


NEW_OPENML_IDS = [
    44065, 44066, 44068, 44069, 45048, 45041,
    45043, 45047, 45045, 45046, 44055, 44056,
    44059, 44061, 44062, 44063,
]

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve")


BASELINE_DEFS = [
    ("PyGAM",        LinearGAM(n_splines=10)),
    ("DT_mini",      DecisionTreeRegressor(max_leaf_nodes=8,  random_state=42)),
    ("DT_large",     DecisionTreeRegressor(max_leaf_nodes=20, random_state=42)),
    ("OLS",          LinearRegression()),
    ("LassoCV",      SkLassoCV(cv=3)),
    ("RidgeCV",      SkRidgeCV(cv=3)),
    ("RF",           RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ("GBM",          GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
    ("MLP",          MLPRegressor(random_state=42)),
    ("FIGS_mini",    FIGSRegressor(max_rules=8,  random_state=42)),
    ("FIGS_large",   FIGSRegressor(max_rules=20, random_state=42)),
    ("RuleFit",      RuleFitRegressor(max_rules=20, random_state=42)),
    ("HSTree_mini",  HSTreeRegressorCV(max_leaf_nodes=8,  random_state=42)),
    ("HSTree_large", HSTreeRegressorCV(max_leaf_nodes=20, random_state=42)),
    ("EBM",          ExplainableBoostingRegressor(random_state=42, outer_bags=3, max_rounds=1000)),
    ("TabPFN",       TabPFN200(device="cpu", random_state=42,
                                n_estimators=2, fit_mode="low_memory")),
]

BASELINE_DESCRIPTIONS = {
    "PyGAM":        "generalized additive model with 10 splines per feature and default settings",
    "DT_mini":      "small decision tree with up to 8 max_leaf_nodes",
    "DT_large":     "large decision tree with up to 20 max_leaf_nodes",
    "OLS":          "ordinary least squares linear regression",
    "LassoCV":      "Lasso linear model with cross-validation to select the regularization parameter",
    "RidgeCV":      "Ridge linear model with cross-validation to select the regularization parameter",
    "RF":           "random forest with 50 tree estimators, each with max_depth of 5",
    "GBM":          "gradient boosting machine with 100 tree estimators, each with max_depth of 3",
    "MLP":          "multi-layer perceptron with default hidden layer size, ReLU activation, and Adam solver",
    "FIGS_mini":    "small FIGS with up to 8 max_rules",
    "FIGS_large":   "large FIGS with up to 20 max_rules",
    "RuleFit":      "RuleFit with up to 20 max_rules",
    "HSTree_mini":  "small HSTree with up to 8 max_leaf_nodes",
    "HSTree_large": "large HSTree with up to 20 max_leaf_nodes",
    "EBM":          "explainable boosting machine (InterpretML) with 3 outer bags and 1000 max rounds",
    "TabPFN":       "TabPFN foundation model for tabular data (training data capped at 200 samples, 2 ensemble members, low-memory fit mode for CPU latency)",
}

BASELINE_NAMES = {n for n, _ in BASELINE_DEFS}


# ---------------------------------------------------------------------------
# Load evolved models from a given folder
# ---------------------------------------------------------------------------

def load_evolved_models(lib_root):
    """Return [(shorthand_name, model_instance, description), ...] from a
    folder's interpretable_regressors_lib/{success,failure}/ sub-directories.

    We include failure/ models as well so Fig 3 reflects the full pool of
    evolved candidates (same scope as the folder's overall_results.csv)."""
    lib_dir = Path(lib_root) / "interpretable_regressors_lib"
    defs = []
    seen = set()
    if not lib_dir.exists():
        return defs
    py_files = []
    for sub in ("success", "failure"):
        sub_dir = lib_dir / sub
        if sub_dir.exists():
            py_files.extend(sorted(sub_dir.glob("*.py")))
    for py_file in py_files:
        mod_name = f"{lib_root.name}__{py_file.parent.name}__{py_file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"  WARNING: failed to load {py_file.name}: {e}")
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, "fit") and hasattr(obj, "predict"):
                obj.__module__ = mod_name
        name = getattr(mod, "model_shorthand_name", None)
        desc = getattr(mod, "model_description", "")
        mdefs = getattr(mod, "model_defs", None)
        if name and mdefs and name not in seen:
            seen.add(name)
            defs.append((name, mdefs[0][1], desc))
    return defs


# ---------------------------------------------------------------------------
# Interpretability: one cached test invocation
# ---------------------------------------------------------------------------

# imodelsx.llm retries failed API calls every LLM_REPEAT_DELAY seconds forever.
# On sustained 429s that hangs the whole run. Disable its retry loop — we do
# our own bounded backoff below so stuck models fail fast and we move on.
imodelsx.llm.LLM_CONFIG["LLM_REPEAT_DELAY"] = None

# imodelsx.llm creates AzureOpenAI clients with no I/O timeout for gpt-5, so
# a stalled connection hangs the worker forever (thread-based timeouts don't
# help — the worker thread can't be killed and the hang is inside C-level
# socket code that holds the GIL briefly, causing massive thread leaks).
# Patch the client to have a bounded I/O timeout + few retries.
_orig_llm_chat_init = imodelsx.llm.LLM_Chat.__init__
def _patched_llm_chat_init(self, *args, **kwargs):
    _orig_llm_chat_init(self, *args, **kwargs)
    if getattr(self, "client", None) is not None:
        self.client = self.client.with_options(timeout=90.0, max_retries=2)
imodelsx.llm.LLM_Chat.__init__ = _patched_llm_chat_init


def _is_transient(err_msg):
    msg = err_msg.lower()
    return any(tok in msg for tok in ("429", "too many", "timeout", "timed out",
                                      "connection", "temporarily"))


class _TransientLLMError(Exception):
    """Raised when retries are exhausted on a transient (e.g. 429) error;
    NOT caught by the cache, so the test will be retried on the next run."""


def _run_test_with_bounded_retry(test_fn, model, llm, test_name,
                                  max_retries=1, base_delay=10):
    """Per-test budget ≈ 2 minutes: the client-level I/O timeout (90s) bounds
    each attempt; we allow 1 retry for transient errors (429/network blip),
    then fail. No thread-based wrapper — that leaks httpx pool threads from
    inside test_fn and causes GIL meltdown at scale."""
    import time as _time
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return test_fn(model, llm)
        except AssertionError as e:
            return dict(test=test_name, passed=False, error=f"Assertion: {e}",
                        ground_truth=None, response=None)
        except Exception as e:
            last_err = str(e)
            is_timeout = ("timeout" in last_err.lower() or
                          "timed out" in last_err.lower())
            if is_timeout:
                # Treat I/O timeout as a test failure (user-requested): do not
                # retry, cache the failure so re-runs skip it.
                return dict(test=test_name, passed=False,
                            error=f"Timeout: {last_err}",
                            ground_truth=None, response=None)
            transient = _is_transient(last_err)
            if not transient:
                return dict(test=test_name, passed=False, error=last_err,
                            ground_truth=None, response=None)
            if attempt == max_retries:
                raise _TransientLLMError(last_err)
            _time.sleep(base_delay * (2 ** attempt))


_SHARED_LLM = None
_SHARED_LLM_LOCK = None


def _get_shared_llm():
    """Create the imodelsx LLM client once per process. Creating a fresh
    client per test triggers Azure CLI auth subprocesses in every worker
    thread, which pile up and stall the run. The client is thread-safe for
    concurrent __call__ (it uses per-request httpx calls with a pool)."""
    global _SHARED_LLM, _SHARED_LLM_LOCK
    if _SHARED_LLM_LOCK is None:
        import threading
        _SHARED_LLM_LOCK = threading.Lock()
    if _SHARED_LLM is None:
        with _SHARED_LLM_LOCK:
            if _SHARED_LLM is None:
                _SHARED_LLM = imodelsx.llm.get_llm(CHECKPOINT)
    return _SHARED_LLM


@_interp_cache.cache(ignore=["model"])
def run_one_interp_test(unique_model_name, test_name, model):
    llm = _get_shared_llm()
    test_fn = _ALL_TEST_FNS[test_name]
    result = _run_test_with_bounded_retry(test_fn, model, llm, test_name)
    result["model"] = unique_model_name
    result.setdefault("test", test_name)
    return result


def _run_one_safe(unique_model_name, test_name, model):
    """Wrapper that converts _TransientLLMError into an uncached error dict.
    The raise inside run_one_interp_test prevents joblib.Memory from caching
    the failure; this wrapper returns a placeholder so the Parallel call
    doesn't abort the whole model."""
    try:
        return run_one_interp_test(unique_model_name, test_name, model)
    except _TransientLLMError as e:
        return dict(test=test_name, model=unique_model_name, passed=False,
                    error=f"Transient LLM error: {e}",
                    ground_truth=None, response=None)


_MODEL_WALL_TIMEOUT_SEC = 4 * 60  # hard per-model wall-clock budget.


def _run_all_interp_inproc(unique_model_name, model, n_jobs):
    """Actual test-loop body. Called either in-process (baselines) or inside
    a subprocess worker (evolved models)."""
    test_names = [fn.__name__ for fn in ALL_TESTS]
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_one_safe)(unique_model_name, tn, model) for tn in test_names
    )
    return results


def _subprocess_worker(q, unique_model_name, model, n_jobs):
    """multiprocessing.Process target — runs tests and puts results on q.
    Runs in a forked child: reset any shared state that shouldn't be inherited."""
    global _SHARED_LLM, _SHARED_LLM_LOCK
    _SHARED_LLM = None       # fresh httpx client — don't inherit parent's threads.
    _SHARED_LLM_LOCK = None
    try:
        results = _run_all_interp_inproc(unique_model_name, model, n_jobs)
        q.put(("ok", results))
    except BaseException as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def run_all_interp_for_model(unique_model_name, model, n_jobs=8,
                              use_subprocess=True):
    """Run 157 tests for a model with a hard wall-clock timeout. Evolved
    models run in a forked subprocess so that hung model.fit / deadlocked
    sklearn nested parallelism can be hard-killed after `_MODEL_WALL_TIMEOUT_SEC`.
    Uses fork (not spawn) because evolved model classes come from dynamically-
    imported modules and cannot be pickled for spawn transport."""
    if not use_subprocess:
        return _run_all_interp_inproc(unique_model_name, model, n_jobs)

    import multiprocessing as _mp
    import queue as _queue
    ctx = _mp.get_context("fork")  # avoid pickling evolved model classes.
    q = ctx.Queue()
    p = ctx.Process(target=_subprocess_worker,
                    args=(q, unique_model_name, model, n_jobs))
    p.start()

    # Drain the queue FIRST (with the wall-clock timeout), THEN join. If we
    # `p.join()` before `q.get()`, the child can deadlock writing a large
    # payload (157 test dicts ≈ 100KB) to the pipe while parent waits for
    # child to exit — classic multiprocessing.Queue hazard.
    test_names_all = [fn.__name__ for fn in ALL_TESTS]
    try:
        tag, payload = q.get(timeout=_MODEL_WALL_TIMEOUT_SEC)
    except _queue.Empty:
        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join(5)
        print(f"  ! {unique_model_name}: wall-clock timeout "
              f"({_MODEL_WALL_TIMEOUT_SEC}s); marking all tests as Timeout.")
        return [dict(test=tn, model=unique_model_name, passed=False,
                     error=f"Model wall-clock timeout ({_MODEL_WALL_TIMEOUT_SEC}s)",
                     ground_truth=None, response=None) for tn in test_names_all]

    # Got the payload; now child should exit cleanly.
    p.join(10)
    if p.is_alive():
        p.terminate()
        p.join(5)

    if tag == "ok":
        return payload
    print(f"  ! {unique_model_name}: subprocess error: {payload}")
    return [dict(test=tn, model=unique_model_name, passed=False,
                 error=f"subprocess-error: {payload}",
                 ground_truth=None, response=None) for tn in test_names_all]


# ---------------------------------------------------------------------------
# Performance: 16 OpenML datasets
# ---------------------------------------------------------------------------

def _load_openml_dataset_by_id(dataset_id):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    import pandas as pd

    path = os.path.join(_CACHE_DIR, f"openml_id_{dataset_id}.parquet")
    if not os.path.exists(path):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        openml.config.cache_directory = os.path.join(_CACHE_DIR, "openml")
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.DataFrame(X, columns=attribute_names)
        df["__target__"] = y
        df.to_parquet(path, index=False)
    else:
        df = pd.read_parquet(path)

    y_raw = df["__target__"].values
    X_raw = df.drop(columns=["__target__"])
    y = pd.to_numeric(pd.Series(y_raw), errors="coerce").values.astype(float)
    valid = ~np.isnan(y)
    y = y[valid]
    X_raw = X_raw[valid]
    cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    for col in num_cols:
        median = X_tr[col].median()
        X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").fillna(median)
        X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(median)
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))
    return X_tr.astype(np.float32).values, X_te.astype(np.float32).values, y_tr, y_te


_PERF_TIMEOUT_SEC = 120  # per-dataset fit+predict budget for evolved models.


class _PerfTimeout(Exception):
    pass


def _perf_alarm_handler(signum, frame):
    raise _PerfTimeout("perf fit/predict exceeded budget")


@_perf_cache.cache(ignore=["model"])
def _fit_and_rmse(unique_model_name, dataset_id, model):
    import gc
    import signal as _signal
    from sklearn.metrics import mean_squared_error
    # TabPFN on CPU is intractable at 1000 samples; skip perf for it and
    # report NaN. Interpretability tests (which use 100-sample data) still
    # run TabPFN; only the 16-dataset performance eval is skipped.
    if unique_model_name == "TabPFN":
        return float("nan")
    X_tr, X_te, y_tr, y_te = _load_openml_dataset_by_id(dataset_id)
    X_tr, X_te, y_tr, y_te = subsample_dataset(X_tr, X_te, y_tr, y_te)
    if len(X_tr) < MIN_SAMPLES or X_tr.shape[1] < MIN_FEATURES:
        return float("nan")
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std())
    if y_std > 0:
        y_tr = (y_tr - y_mean) / y_std
        y_te = (y_te - y_mean) / y_std
    # SIGALRM-based timeout: works because _fit_and_rmse is called from the
    # main thread (perf loop is serial, not joblib.Parallel). Prevents a
    # user-written model.fit from stalling the whole run.
    prev = _signal.signal(_signal.SIGALRM, _perf_alarm_handler)
    _signal.alarm(_PERF_TIMEOUT_SEC)
    try:
        for attempt in range(2):
            try:
                m = deepcopy(model)
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
                gc.collect()
                return rmse
            except _PerfTimeout:
                print(f"    {unique_model_name} @ openml/{dataset_id}: TIMEOUT ({_PERF_TIMEOUT_SEC}s)")
                return float("nan")
            except Exception as e:
                if attempt == 0:
                    gc.collect()
                    _signal.alarm(_PERF_TIMEOUT_SEC)  # reset for retry
                    continue
                print(f"    {unique_model_name} @ openml/{dataset_id}: ERROR — {e}")
                return float("nan")
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, prev)


def run_all_perf_for_model(unique_model_name, model):
    """Evaluate model on all 16 OpenML datasets. Returns {ds_name: rmse}."""
    rmses = {}
    for did in NEW_OPENML_IDS:
        try:
            rmse = _fit_and_rmse(unique_model_name, did, model)
        except Exception as e:
            print(f"  ERROR on openml/{did}: {e}")
            rmse = float("nan")
        rmses[f"openml/{did}"] = rmse
    return rmses


# ---------------------------------------------------------------------------
# Per-folder workflow
# ---------------------------------------------------------------------------

def process_folder(folder, n_jobs_interp=8, verbose=True):
    """Load folder's evolved models + baselines, run tests + perf, write outputs."""
    folder_name = folder.name
    print(f"\n{'='*70}\n  Processing: {folder_name}\n{'='*70}")

    evolved = load_evolved_models(folder)
    print(f"  Loaded {len(evolved)} evolved models from {folder_name}")

    # Build model list: baselines first, then evolved.
    # Unique name: baselines use their plain name (cache shared); evolved prefixed.
    models = []  # list of dict(unique_name, display_name, model, description, is_baseline)
    for bname, bmodel in BASELINE_DEFS:
        models.append(dict(
            unique_name=bname,
            display_name=bname,
            model=bmodel,
            description=BASELINE_DESCRIPTIONS.get(bname, bname),
            is_baseline=True,
        ))
    for ename, emodel, edesc in evolved:
        models.append(dict(
            unique_name=f"{folder_name}::{ename}",
            display_name=ename,
            model=emodel,
            description=edesc,
            is_baseline=False,
        ))

    # --- interpretability ---
    print(f"  Running {len(ALL_TESTS)} interp tests × {len(models)} models …")
    interp_rows = []
    for entry in models:
        if verbose:
            print(f"    [interp] {entry['display_name']}", flush=True)
        # Baselines run in-process (trusted, fast); evolved models run in a
        # subprocess with a wall-clock timeout so a hung user-written fit()
        # or a sklearn nested-parallelism deadlock cannot block the whole
        # suite.
        results = run_all_interp_for_model(
            entry["unique_name"], entry["model"], n_jobs=n_jobs_interp,
            use_subprocess=not entry["is_baseline"],
        )
        passed = sum(1 for r in results if r.get("passed"))
        if verbose:
            print(f"           {passed}/{len(results)} passed")
        for r in results:
            interp_rows.append({
                "model": entry["display_name"],
                "test": r["test"],
                "category": category_of(r["test"]),
                "passed": r.get("passed", False),
                "ground_truth": r.get("ground_truth", ""),
                "response": (r.get("response") or "")[:500],
                "error": r.get("error", ""),
            })

    interp_csv = folder / "interpretability_results_test.csv"
    with open(interp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "test", "category", "passed",
                                          "ground_truth", "response", "error"])
        w.writeheader()
        w.writerows(interp_rows)
    print(f"  Saved → {interp_csv}")

    # --- performance ---
    print(f"  Running perf on 16 OpenML datasets × {len(models)} models …")
    perf_by_model = {}
    for entry in models:
        if verbose:
            print(f"    [perf] {entry['display_name']}")
        perf_by_model[entry["display_name"]] = run_all_perf_for_model(
            entry["unique_name"], entry["model"]
        )

    # Compute mean_rank within this folder's model set.
    model_names = [e["display_name"] for e in models]
    avg_rank = {}
    for ds_name in [f"openml/{d}" for d in NEW_OPENML_IDS]:
        pairs = []
        for mn in model_names:
            v = perf_by_model[mn].get(ds_name, float("nan"))
            if not np.isnan(v):
                pairs.append((mn, v))
        pairs.sort(key=lambda x: x[1])
        for rank_idx, (mn, _) in enumerate(pairs, 1):
            avg_rank.setdefault(mn, []).append(rank_idx)

    mean_rank = {mn: float(np.mean(v)) if v else float("nan")
                 for mn, v in avg_rank.items()}

    # per-model interp pass rate
    frac_pass = defaultdict(lambda: [0, 0])
    for r in interp_rows:
        frac_pass[r["model"]][1] += 1
        if r["passed"]:
            frac_pass[r["model"]][0] += 1

    overall_rows = []
    for entry in models:
        mn = entry["display_name"]
        rank = mean_rank.get(mn, float("nan"))
        p, t = frac_pass[mn]
        frac = p / t if t else float("nan")
        overall_rows.append({
            "commit": "baseline" if entry["is_baseline"] else "",
            "mean_rank": f"{rank:.2f}" if not np.isnan(rank) else "nan",
            "frac_interpretability_tests_passed": f"{frac:.4f}" if not np.isnan(frac) else "nan",
            "status": "baseline" if entry["is_baseline"] else "evolved",
            "model_name": mn,
            "description": entry["description"],
        })

    overall_csv = folder / "overall_results_test.csv"
    with open(overall_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        w.writeheader()
        w.writerows(overall_rows)
    print(f"  Saved → {overall_csv}")

    # --- plot ---
    plot_path = folder / "interpretability_vs_performance_test.png"
    plot_interp_vs_performance(str(overall_csv), str(plot_path))

    return overall_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_pooled_outputs(folders):
    """After per-folder outputs are written, build a single pooled table with
    globally-computed ranks — used by fig_generalization_scatter.py."""
    pooled_dir = SCRIPT_DIR / "new_results_test"
    pooled_dir.mkdir(exist_ok=True)

    # Collect baselines once + all evolved models across folders.
    models = []
    seen = set()
    for bname, bmodel in BASELINE_DEFS:
        models.append(dict(unique_name=bname, display_name=bname,
                           model=bmodel, description=BASELINE_DESCRIPTIONS.get(bname, bname),
                           is_baseline=True, run=""))
        seen.add(bname)
    for folder in folders:
        for ename, emodel, edesc in load_evolved_models(folder):
            uniq = f"{folder.name}::{ename}"
            display = ename if ename not in seen else f"{ename}@{folder.name}"
            seen.add(display)
            models.append(dict(unique_name=uniq, display_name=display,
                               model=emodel, description=edesc,
                               is_baseline=False, run=folder.name))

    # RMSE per (model, dataset) — use only previously-cached values. Stale
    # cache entries (e.g., from folders processed before _fit_and_rmse's body
    # changed) become NaN rather than triggering re-fits; this keeps the
    # pooled phase bounded when old caches don't match the new function hash.
    perf_rows = []
    _cached_call = _fit_and_rmse.call_and_shelve  # noqa: F841 (placeholder)
    _check_cached = _fit_and_rmse.check_call_in_cache
    for entry in models:
        for did in NEW_OPENML_IDS:
            if _check_cached(entry["unique_name"], did, entry["model"]):
                rmse = _fit_and_rmse(entry["unique_name"], did, entry["model"])
            else:
                rmse = float("nan")
            perf_rows.append((entry["display_name"], f"openml/{did}", rmse))

    # Global ranks.
    by_ds = defaultdict(list)
    for mn, ds, v in perf_rows:
        if not np.isnan(v):
            by_ds[ds].append((mn, v))
    ranks_per_model = defaultdict(list)
    for ds, pairs in by_ds.items():
        pairs.sort(key=lambda p: p[1])
        for i, (mn, _) in enumerate(pairs, 1):
            ranks_per_model[mn].append(i)
    mean_rank = {mn: float(np.mean(v)) if v else float("nan")
                 for mn, v in ranks_per_model.items()}

    # Pool interp pass rates from already-computed cache only. Same pattern as
    # perf above: if the cache entry isn't present (e.g., function hash drift,
    # or the model was never run), count it as a non-pass rather than
    # triggering a fresh LLM call that would stall the pool phase.
    print(f"Pooling (cache-only) interp tests for pooled table …", flush=True)
    _check_interp_cached = run_one_interp_test.check_call_in_cache
    test_names_all = [fn.__name__ for fn in ALL_TESTS]
    frac_pass = {}
    for entry in models:
        passed = 0
        cached = 0
        for tn in test_names_all:
            if _check_interp_cached(entry["unique_name"], tn, entry["model"]):
                cached += 1
                r = run_one_interp_test(entry["unique_name"], tn, entry["model"])
                if r.get("passed"):
                    passed += 1
        frac_pass[entry["display_name"]] = (
            passed / len(test_names_all) if cached == len(test_names_all)
            else float("nan")
        )

    out_csv = pooled_dir / "overall_results_pooled.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS + ["run"])
        w.writeheader()
        for entry in models:
            dn = entry["display_name"]
            r = mean_rank.get(dn, float("nan"))
            frac = frac_pass.get(dn, float("nan"))
            w.writerow({
                "commit":        "baseline" if entry["is_baseline"] else "",
                "mean_rank":     f"{r:.2f}" if not np.isnan(r) else "nan",
                "frac_interpretability_tests_passed": f"{frac:.4f}" if not np.isnan(frac) else "nan",
                "status":        "baseline" if entry["is_baseline"] else "evolved",
                "model_name":    dn,
                "description":   entry["description"],
                "run":           entry["run"],
            })
    print(f"Pooled table → {out_csv}")


def main():
    t0 = time.time()
    folders = sorted([p for p in RESULT_LIBS_DIR.iterdir() if p.is_dir()])
    folders = [p for p in folders if (p / "interpretable_regressors_lib" / "success").exists()]
    print(f"Found {len(folders)} result_libs folders:", flush=True)
    for p in folders:
        n = len(list((p / "interpretable_regressors_lib" / "success").glob("*.py")))
        print(f"  {p.name}: {n} evolved models", flush=True)

    for folder in folders:
        done_marker = folder / "overall_results_test.csv"
        if done_marker.exists():
            print(f"  [skip] {folder.name}: {done_marker.name} already exists", flush=True)
            continue
        process_folder(folder)

    write_pooled_outputs(folders)
    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
