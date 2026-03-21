"""
Evaluate a fixed set of baseline classifiers on TabArena classification datasets
(subsampled) and produce performance rank results.

Usage: uv run run_tabarena.py
Outputs:
  results/tabarena_scores.json   — avg rank and mean AUC per model
  results/tabarena_results.csv   — per-dataset per-model AUC
"""

import csv
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np
from joblib import Memory
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prepare import get_all_datasets
from models import CLASSIFIER_DEFS as MODEL_DEFS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_memory = Memory(location=os.path.join(RESULTS_DIR, "cache"), verbose=0)

# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

MAX_SAMPLES = 1000
MAX_FEATURES = 25
SUBSAMPLE_SEED = 42


def subsample_dataset(X_train, X_test, y_train, y_test,
                      max_samples=MAX_SAMPLES, max_features=MAX_FEATURES,
                      seed=SUBSAMPLE_SEED):
    """Cap training samples and features with a fixed random seed."""
    rng = np.random.RandomState(seed)
    if X_train.shape[1] > max_features:
        feat_idx = rng.choice(X_train.shape[1], max_features, replace=False)
        feat_idx.sort()
        X_train = X_train[:, feat_idx]
        X_test  = X_test[:, feat_idx]
    if len(X_train) > max_samples:
        idx = rng.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@_memory.cache
def _run_one_classifier(model_name, ds_name, clf,
                        X_train, X_test, y_train, y_test):
    """Fit one classifier on one dataset and return AUC. Cached by joblib."""
    n_classes = len(np.unique(y_train))
    try:
        m = deepcopy(clf)
        m.fit(X_train, y_train)
        if n_classes == 2:
            proba = m.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        else:
            proba = m.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        return float(auc)
    except Exception as e:
        return str(e)   # cache the error message too


def evaluate_all_classifiers(model_defs):
    """Evaluate all classifiers on every TabArena dataset (subsampled).

    Returns:
        dataset_aucs : {dataset_name: {model_name: auc}}
    """
    dataset_aucs = {}
    for ds_name, X_train, X_test, y_train, y_test in get_all_datasets():
        X_train, X_test, y_train, y_test = subsample_dataset(
            X_train, X_test, y_train, y_test)
        n_classes = len(np.unique(y_train))
        print(f"\n  Dataset: {ds_name} — {X_train.shape[1]} features, "
              f"{len(X_train)} train samples, {n_classes} classes")
        dataset_aucs[ds_name] = {}

        for name, clf in model_defs:
            result = _run_one_classifier(name, ds_name, clf,
                                         X_train, X_test, y_train, y_test)
            if isinstance(result, float):
                dataset_aucs[ds_name][name] = result
                print(f"    {name:<15}: {result:.4f}")
            else:
                print(f"    {name:<15}: ERROR — {result}")
                dataset_aucs[ds_name][name] = float("nan")

    return dataset_aucs


def compute_rank_scores(dataset_aucs):
    """For each dataset rank models by AUC (1=best), then average ranks."""
    all_model_names = set()
    for d in dataset_aucs.values():
        all_model_names.update(d.keys())

    ranks_per_model = {n: [] for n in all_model_names}
    mean_auc_per_model = {n: [] for n in all_model_names}

    for ds_name, model_aucs in dataset_aucs.items():
        valid = [(n, v) for n, v in model_aucs.items() if not np.isnan(v)]
        sorted_models = sorted(valid, key=lambda x: x[1], reverse=True)
        rank_map = {n: r + 1 for r, (n, _) in enumerate(sorted_models)}
        for name in all_model_names:
            if name in model_aucs and not np.isnan(model_aucs[name]):
                ranks_per_model[name].append(rank_map[name])
                mean_auc_per_model[name].append(model_aucs[name])

    avg_rank = {n: float(np.mean(v)) for n, v in ranks_per_model.items() if v}
    avg_auc  = {n: float(np.mean(v)) for n, v in mean_auc_per_model.items() if v}
    return avg_rank, avg_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    print("Evaluating classifiers on TabArena datasets "
          f"(max {MAX_SAMPLES} train samples, max {MAX_FEATURES} features)...")
    dataset_aucs = evaluate_all_classifiers(MODEL_DEFS)
    avg_rank, avg_auc = compute_rank_scores(dataset_aucs)

    print("\n\nTabArena summary (sorted by avg rank):")
    for name, rank in sorted(avg_rank.items(), key=lambda x: x[1]):
        print(f"  {name:<15}: avg_rank={rank:.2f}  mean_auc={avg_auc.get(name, float('nan')):.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # JSON summary
    scores = {
        "tabarena_avg_rank":     avg_rank,
        "tabarena_mean_auc":     avg_auc,
        "tabarena_per_dataset":  dataset_aucs,
    }
    json_path = os.path.join(RESULTS_DIR, "tabarena_scores.json")
    with open(json_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores saved → {json_path}")

    # CSV: one row per (dataset, model)
    csv_path = os.path.join(RESULTS_DIR, "tabarena_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "auc", "rank"])
        for ds_name, model_aucs in dataset_aucs.items():
            valid = [(n, v) for n, v in model_aucs.items() if not np.isnan(v)]
            sorted_models = sorted(valid, key=lambda x: x[1], reverse=True)
            rank_map = {n: r + 1 for r, (n, _) in enumerate(sorted_models)}
            for name, auc in model_aucs.items():
                rank = rank_map.get(name, "")
                writer.writerow([ds_name, name, "" if np.isnan(auc) else f"{auc:.6f}", rank])
    print(f"Per-dataset results saved → {csv_path}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
