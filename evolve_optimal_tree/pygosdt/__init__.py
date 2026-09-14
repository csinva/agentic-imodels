"""pygosdt: Generalized Optimal Sparse Decision Trees in pure Python."""

from .encoder import BinaryEncoder, TargetEncoder
from .dataset import BitDataset
from .optimizer import Optimizer
from .model import TreeClassifier
from .gosdt import GOSDT, GOSDTClassifier

__all__ = [
    "BinaryEncoder", "TargetEncoder", "BitDataset", "Optimizer",
    "TreeClassifier", "GOSDT", "GOSDTClassifier",
]
__version__ = "0.1.0"
