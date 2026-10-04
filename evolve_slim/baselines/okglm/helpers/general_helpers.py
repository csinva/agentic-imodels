import numpy as np

try:
    import cupy as cp
except Exception:  # cupy may be unavailable on CPU-only installs
    cp = None


def _get_array_module(use_gpu):
    if use_gpu:
        if cp is None:
            raise RuntimeError("cupy is required for GPU operations.")
        return cp
    return np


def to_gpu(arr):
    """Move numpy arrays to GPU when cupy is available."""
    if cp is None:
        return arr
    if isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_cpu(arr):
    """Move cupy arrays to CPU when cupy is available."""
    if cp is None:
        return arr
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def assert_gpu_available():
    if cp is None:
        raise RuntimeError("GPU requested but cupy is not available.")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise RuntimeError("GPU requested but CUDA runtime is unavailable.") from exc
    if device_count < 1:
        raise RuntimeError("GPU requested but no CUDA devices are available.")
