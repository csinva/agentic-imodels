import sys

from okglm.helpers.general_helpers import _get_array_module, to_cpu, to_gpu


def estimate_largest_eigenvalue_of_XTX(X, max_iter=1000, tol=1e-6, use_gpu=False):
    xp = _get_array_module(use_gpu)
    X = to_gpu(X) if use_gpu else to_cpu(X)

    n, p = X.shape
    eig_vec = xp.random.randn(p)
    eig_vec = eig_vec / xp.linalg.norm(eig_vec)

    lmbd_max_prev = 0.0
    lmbd_max = 1e8

    if n < p:
        i = 0
        while i < max_iter:
            XTXv = X.T @ (X @ eig_vec)
            eig_vec = XTXv / xp.linalg.norm(XTXv)
            lmbd_max = float(eig_vec @ XTXv)
            if abs(lmbd_max - lmbd_max_prev) < tol:
                break
            lmbd_max_prev = lmbd_max
            i += 1
        if i == max_iter:
            sys.stderr.write(
                f"Power iteration did not converge after {max_iter} iterations\n"
            )
    else:
        XTX = X.T @ X
        i = 0
        while i < max_iter:
            XTXv = XTX @ eig_vec
            eig_vec = XTXv / xp.linalg.norm(XTXv)
            lmbd_max = float(eig_vec @ XTXv)
            if abs(lmbd_max - lmbd_max_prev) < tol:
                break
            lmbd_max_prev = lmbd_max
            i += 1
        if i == max_iter:
            sys.stderr.write(
                f"Power iteration did not converge after {max_iter} iterations\n"
            )

    return lmbd_max


def estimate_largest_eigenvalue_of_XTX_gpu(X, max_iter=1000, tol=1e-6):
    return estimate_largest_eigenvalue_of_XTX(
        X, max_iter=max_iter, tol=tol, use_gpu=True
    )
