import time

try:
    import cupy as cp
    from cupyx.scipy.special import xlogy as cupy_xlogy
except Exception:  # cupy may be unavailable on CPU-only installs
    cp = None
    cupy_xlogy = None
from scipy.special import xlogy as scipy_xlogy

from okglm.helpers.general_helpers import _get_array_module, to_cpu, to_gpu
from okglm.helpers.linalg_helpers import estimate_largest_eigenvalue_of_XTX


class base_data_GLM_gpu:
    """
    A class for the data for the generalized linear models (GLM) optimization problem with the following form: F(beta) = F_y(X^T beta) = f(X^T beta, y).
    """

    def __init__(self, X, y, L=1e8, use_gpu=False):
        self.use_gpu = use_gpu
        self.xp = _get_array_module(use_gpu)
        self._xlogy = cupy_xlogy if use_gpu else scipy_xlogy

        self.n, self.p = X.shape
        self.X = to_gpu(X) if use_gpu else to_cpu(X)
        self.y = to_gpu(y) if use_gpu else to_cpu(y)

        self.L = L
        self.preprocess_data()

    def get_L(self):
        return self.L

    def preprocess_data(self):
        pass

    def _preprocess_eigenvalues(self, store_xtx=False, log_timing=False):
        self.lmbd_min = 0
        self.lmbd_max = 1e8
        if store_xtx:
            self.XTX = None

        if self.n >= self.p:
            XTX = self.X.T @ self.X
            if store_xtx:
                self.XTX = XTX
            if log_timing:
                start_time = time.time()
                lmbd = self.xp.linalg.eigvalsh(XTX)
                print(
                    f"calculating largest eigenvalue of XTX took {time.time() - start_time} seconds"
                )
            else:
                lmbd = self.xp.linalg.eigvalsh(XTX)
            self.lmbd_min = float(lmbd[0]) * 0.95
            self.lmbd_max = float(lmbd[-1])
        else:
            self.lmbd_max = estimate_largest_eigenvalue_of_XTX(
                self.X, use_gpu=self.use_gpu
            )

    def get_grad_F(self, beta):
        pass

    def get_F(self, beta):
        pass

    def get_zeta_from_beta(self, beta):
        pass

    def get_F_fenchel(self, zeta):
        pass


class data_LinearRegression_gpu(base_data_GLM_gpu):
    """
    A class for the data for the linear regression optimization problem with the following form: F(beta) = F_y(X^T beta) = ||X^T beta - y||^2.
    """

    def __init__(self, X, y, L=1e8, use_gpu=False):
        super().__init__(X, y, L, use_gpu=use_gpu)

    def preprocess_data(self):
        self.XTX = None
        self.XTy = self.X.T @ self.y
        self.yTy = self.y @ self.y
        self.lmbd_min = 0
        self.lmbd_max = 1e8

        if abs(self.L - 1e8) < 1e-6:
            self._preprocess_eigenvalues(store_xtx=True)
            self.L = 2 * self.lmbd_max

    def get_grad_F(self, beta):
        return 2 * (self.X.T @ (self.X @ beta - self.y))

    def get_F(self, beta):
        return float(self.xp.sum((self.X @ beta - self.y) ** 2))

    def get_zeta_from_beta(self, beta):
        Xbeta = self.X @ beta
        return (-2) * (Xbeta - self.y)

    def get_F_fenchel(self, zeta):
        return float(1 / 4 * self.xp.sum(zeta ** 2) + zeta @ self.y)


class data_LogisticRegression_gpu(base_data_GLM_gpu):
    """
    A class for the data for the logistic regression optimization problem with the following form: F(beta) = F_y(X^T beta) = \sum_{i=1}^n log(1 + exp(-y_i X_i^T beta)).
    """

    def __init__(self, X, y, L=1e8, use_gpu=False):
        super().__init__(X, y, L, use_gpu=use_gpu)

    def preprocess_data(self):
        self.yX = self.y.reshape(-1, 1) * self.X
        self.lmbd_min = 0
        self.lmbd_max = 1e8

        if abs(self.L - 1e8) < 1e-6:
            self._preprocess_eigenvalues()
            self.L = 1.0 / 4 * self.lmbd_max

    def get_grad_F(self, beta):
        yXbeta = self.yX @ beta
        return - (1 / (1 + self.xp.exp(yXbeta))) @ self.yX

    def get_F(self, beta):
        return float(self._compute_logistic_loss(self.yX @ beta))

    def get_zeta_from_beta(self, beta):
        yXbeta = self.yX @ beta
        return 1 / (1 + self.xp.exp(yXbeta)) * (self.y)

    def get_F_fenchel(self, zeta):
        zeta_over_neg_y = zeta / (-self.y)
        one_minus_zeta_over_neg_y = 1 - zeta_over_neg_y

        return float(
            self.xp.sum(
                self.xp.multiply(zeta_over_neg_y, self.xp.log(zeta_over_neg_y))
                + self.xp.multiply(one_minus_zeta_over_neg_y, self.xp.log(one_minus_zeta_over_neg_y))
            )
        )

    def _compute_logistic_loss(self, yXbeta):
        return self.xp.sum(self.xp.log(1 + self.xp.exp(-yXbeta)))


class data_PoissonRegression_gpu(base_data_GLM_gpu):
    """
    A class for the data for the Poisson regression optimization problem with the following form: F(beta) = F_y(X^T beta) = \sum_{i=1}^n (exp(X_i^T beta) - y_i X_i^T beta).
    """

    def __init__(self, X, y, L=1e8, use_gpu=False):
        super().__init__(X, y, L, use_gpu=use_gpu)

    def preprocess_data(self, M=2.0):
        self.yX = self.y.reshape(-1, 1) * self.X
        self.XT = self.X.T
        self.XTX = None

        if abs(self.L - 1e8) < 1e-6:
            self._preprocess_eigenvalues(store_xtx=True, log_timing=True)
            X_abs = self.xp.abs(self.X)
            self.X_abs = self.xp.partition(X_abs, -10, axis=1)[:, -10:]
            self.X_abs_top_k_sum = self.xp.sum(self.X_abs[:10], axis=1)

            self.L = self.lmbd_max * 0.75 * 1e4

            print(f"max(self.X_abs_top_k_sum): {float(self.xp.max(self.X_abs_top_k_sum))}")
            print(f"lmbd_max: {self.lmbd_max}")
            print(f"The Lipschitz constant L is estimated to be {self.L}")

    def get_grad_F(self, beta):
        return self.XT @ (self.xp.exp(self.X @ beta) - self.y)

    def get_F(self, beta):
        Xbeta = self.X @ beta
        return float(self.xp.sum(self.xp.exp(Xbeta)) - self.y.dot(Xbeta))

    def get_zeta_from_beta(self, beta):
        Xbeta = self.X @ beta
        return self.y - self.xp.exp(Xbeta)

    def get_F_fenchel(self, zeta):
        zeta_and_y = zeta + self.y
        return float(self._xlogy(zeta_and_y, zeta_and_y).sum() - self.xp.sum(zeta_and_y))
