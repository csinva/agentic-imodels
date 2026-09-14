"""Command line interface mirroring ``gosdt dataset.csv config.json``.

Prints a JSON array with the optimal model (same schema as the reference).
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .gosdt import GOSDT
from .model import NumpyEncoder


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pygosdt", description=__doc__)
    parser.add_argument("dataset", help="CSV file; last column is the label")
    parser.add_argument("config", nargs="?", help="JSON configuration file")
    args = parser.parse_args(argv)

    config = {}
    if args.config:
        with open(args.config) as fh:
            config = json.load(fh)
    frame = pd.read_csv(args.dataset)
    X = frame.iloc[:, :-1]
    y = frame.iloc[:, -1]
    model = GOSDT(config).fit(X, y)
    if config.get("verbose"):
        print(f"Training Duration: {model.time} seconds")
        print(f"Number of Iterations: {model.iterations} iterations")
        print(f"Size of Graph: {model.size} nodes")
        print(f"Loss: {model.tree.loss()}")
        print(f"Complexity: {model.tree.complexity()}")
    output = json.dumps([model.tree.source], indent=2, cls=NumpyEncoder)
    if config.get("model"):
        with open(config["model"], "w") as fh:
            fh.write(output)
    if not config.get("model") or config.get("verbose"):
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
