"""Write the 14 TabArena sources of TabArena-14 at full size, unsubsampled.

TabArena-14 subsamples each source to match a development dataset's rows, features
and class balance. This writes the same 14 sources whole: every row, every feature,
categorical columns as integer codes and missing values filled (median, or a
"missing" category), with the label last. Nothing else is changed, so the solvers
see the datasets as they are rather than as a mirror of the development suite.

    uv run --with openml baselines/benchmarks/external/build_tabarena14_full.py
"""

import os

import numpy as np
import openml
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_full")
openml.config.cache_directory = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve", "openml")


def main():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(HERE, "data", "manifest.csv"))
    rows = []
    for r in man.itertuples():
        ds = openml.datasets.get_dataset(int(r.openml_did), download_data=True, download_qualities=False,
                                         download_features_meta_data=False)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
        X = X.copy()
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
            else:
                codes = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
                order = {v: i for i, v in enumerate(codes.value_counts().index)}
                X[c] = codes.map(order).astype(int)
        X.columns = [f"x{i}" for i in range(X.shape[1])]
        X["label"] = pd.factorize(y)[0]
        name = f"taf_{r.mirrors}"
        X.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        splits = int(sum(X[c].nunique() - 1 for c in X.columns[:-1]))
        rows.append({"name": name, "source": r.source, "openml_did": int(r.openml_did), "rows": len(X),
                     "features": X.shape[1] - 1, "binary_splits": splits, "classes": int(X["label"].nunique()),
                     "majority": round(float(X["label"].value_counts(normalize=True).max()), 3)})
        print(f"{name:22s} {r.source[:38]:38s} rows={len(X):6d} feats={X.shape[1]-1:5d} splits={splits:8,d}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
