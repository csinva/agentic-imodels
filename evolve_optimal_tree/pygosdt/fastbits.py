"""numba kernels over packed 64-bit words for the per-subproblem statistics.

The optimizer keeps capture sets as Python integers (hashable memo keys) but
the expensive step, counting ``|capture & target_k & feature_j|`` for every
feature ``j``, is done here on ``uint64`` word arrays.  The result is bit-for-bit
identical to the pure Python loop; only the speed differs.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco if not (args and callable(args[0])) else args[0]


@njit(cache=True, nogil=True)
def _popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=True, nogil=True)
def child_counts(F, masks, out):
    """out[j, r] = popcount(F[j] & masks[r]) for every feature j and mask r."""
    m, W = F.shape
    R = masks.shape[0]
    for j in range(m):
        for r in range(R):
            acc = np.uint64(0)
            for w in range(W):
                acc += _popcount64(F[j, w] & masks[r, w])
            out[j, r] = acc
    return out


@njit(cache=True, nogil=True)
def child_counts_subset(F, feats, masks, out):
    """Same as ``child_counts`` restricted to the rows ``feats`` of ``F``."""
    W = F.shape[1]
    R = masks.shape[0]
    for t in range(feats.shape[0]):
        j = feats[t]
        for r in range(R):
            acc = np.uint64(0)
            for w in range(W):
                acc += _popcount64(F[j, w] & masks[r, w])
            out[t, r] = acc
    return out


def pack_columns(Xb: np.ndarray) -> np.ndarray:
    """(n, m) bool -> (m, W) uint64 with row i of the data in bit i."""
    n, m = Xb.shape
    W = (n + 63) // 64
    packed = np.packbits(np.ascontiguousarray(Xb.T), axis=1, bitorder="little")
    padded = np.zeros((m, W * 8), dtype=np.uint8)
    padded[:, :packed.shape[1]] = packed
    return np.ascontiguousarray(padded.view(np.uint64))


def int_to_words(value: int, W: int) -> np.ndarray:
    return np.frombuffer(value.to_bytes(W * 8, "little"), dtype=np.uint64)


def warm_up():
    """Trigger JIT compilation (cached on disk afterwards)."""
    if not HAVE_NUMBA:
        return
    F = np.zeros((2, 1), dtype=np.uint64)
    masks = np.zeros((1, 1), dtype=np.uint64)
    child_counts(F, masks, np.zeros((2, 1), dtype=np.uint64))
    child_counts_subset(F, np.zeros(1, dtype=np.int64), masks, np.zeros((1, 1), dtype=np.uint64))
