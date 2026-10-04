import numpy as np

from okglm.helpers.general_helpers import _get_array_module, to_cpu, to_gpu
from okglm.prox_operators import (
    compute_fenchel_kyFan_HuberLoss,
    compute_fenchel_kyFan_HuberLoss_bnb,
    compute_kyFan_HuberLoss_bnb,
    prox_fenchel_kyFan_HuberLoss_pava,
    prox_fenchel_kyFan_HuberLoss_pava_bnb,
)

def prox_fenchel_kyFan_HuberLoss_pava_gpu(x, k, rho, M=1e6):
    x_cpu = to_cpu(x)
    prox_cpu = prox_fenchel_kyFan_HuberLoss_pava(x_cpu, k, rho, M)
    return to_gpu(prox_cpu)


class base_regularizer_gpu:
    """
    A class for the regularizer for the composite convex optimization problem with the following form: L(beta) = F(beta) + G(beta), where G(beta) is the regularizer.
    """

    def __init__(self, X, use_gpu=False, **reg_params):
        self.X = X
        self.use_gpu = use_gpu
        self.xp = _get_array_module(use_gpu)
        self.reg_params = reg_params

    def _require_param(self, name):
        if name not in self.reg_params:
            raise ValueError(f"Missing regularizer parameter: {name}")
        return self.reg_params[name]

    def get_G(self, beta):
        pass

    def get_G_fenchel(self, zeta):
        pass

    def get_prox_G(self, beta, stepsize):
        pass

    def get_feasible_zeta(self, zeta):
        return zeta


class regularizer_fenchel_kyFan_HuberLoss_gpu(base_regularizer_gpu):
    """
    A class for the regularizer G(beta) = 2\lambda_2 g(beta), where
        g(beta) = \min_z \sum_{j=1}^p beta_j^2 / z_j, s.t. 0 <= z_j <= 1, \sum_{j=1}^p z_j <= k, |beta_j| <= M * z_j
    """

    def __init__(self, X, **reg_params):
        super().__init__(X, **reg_params)
        self.lambda2 = self._require_param("lambda2")
        self.twoLambda2 = 2 * self.lambda2
        self.k = self._require_param("k")
        self.M = reg_params.get("M", 1e6)

    def get_G(self, beta):
        beta_cpu = to_cpu(beta)
        value1 = compute_fenchel_kyFan_HuberLoss(beta_cpu, self.k, self.M)
        return self.twoLambda2 * value1

    def get_G_fenchel(self, zeta):
        result = self.X.T @ zeta / self.twoLambda2
        return self.twoLambda2 * self._compute_kyfan_huber_loss(result)

    def get_prox_G(self, beta, stepsize):
        if self.use_gpu:
            return prox_fenchel_kyFan_HuberLoss_pava_gpu(
                beta, self.k, self.twoLambda2 * stepsize, self.M
            )
        return prox_fenchel_kyFan_HuberLoss_pava(
            beta, self.k, self.twoLambda2 * stepsize, self.M
        )

    def _compute_kyfan_huber_loss(self, x):
        abs_x = self.xp.abs(x)
        top_k_abs_x = self.xp.partition(abs_x, -self.k)[-self.k:]
        return float(
            self.xp.sum(
                self.xp.where(
                    top_k_abs_x <= self.M,
                    0.5 * top_k_abs_x ** 2,
                    self.M * top_k_abs_x - 0.5 * self.M ** 2,
                )
            )
        )


class regularizer_fenchel_kyFan_HuberLoss_BnB_gpu(base_regularizer_gpu):
    """
    KyFan-HuberLoss regularizer that supports BnB z-lb/z-ub constraints.
    """

    def __init__(self, X, **reg_params):
        super().__init__(X, **reg_params)
        self.lambda2 = self._require_param("lambda2")
        self.twoLambda2 = 2 * self.lambda2
        self.k = self._require_param("k")
        self.M = reg_params.get("M", 1e6)

        self.p = X.shape[1]
        self.z_lb = np.zeros(self.p, dtype=bool)
        self.z_ub = np.ones(self.p, dtype=bool)
        self._update_z_masks()

    def _update_z_masks(self):
        self.k_reduced = self.k - int(np.sum(self.z_lb))
        self.z_ub_reversed = ~self.z_ub
        self.z_free = (~self.z_lb) & self.z_ub

    def reset_z_lb_and_z_ub(self, z_lb, z_ub):
        self.z_lb = np.asarray(to_cpu(z_lb)).astype(bool).copy()
        self.z_ub = np.asarray(to_cpu(z_ub)).astype(bool).copy()
        self._update_z_masks()

    def get_G(self, beta):
        beta_cpu = to_cpu(beta)
        value1 = compute_fenchel_kyFan_HuberLoss_bnb(
            beta_cpu,
            self.k_reduced,
            self.z_lb,
            self.z_ub_reversed,
            self.z_free,
            self.M,
        )
        return self.twoLambda2 * value1

    def get_G_fenchel(self, zeta):
        x = (self.X.T @ zeta) / self.twoLambda2
        x_cpu = to_cpu(x)
        return self.twoLambda2 * compute_kyFan_HuberLoss_bnb(
            x_cpu,
            self.k_reduced,
            self.z_lb,
            self.z_ub_reversed,
            self.z_free,
            self.M,
        )

    def get_prox_G(self, beta, stepsize):
        beta_cpu = to_cpu(beta)
        prox_cpu = prox_fenchel_kyFan_HuberLoss_pava_bnb(
            beta_cpu,
            self.k_reduced,
            self.twoLambda2 * stepsize,
            self.z_lb,
            self.z_ub_reversed,
            self.z_free,
            self.M,
        )
        return to_gpu(prox_cpu) if self.use_gpu else prox_cpu

class regularizer_l1Regularized_gpu(base_regularizer_gpu):
    """
    A class for the L1 regularizer G(beta) = lambda * ||beta||_1
    """

    def __init__(self, X, **reg_params):
        super().__init__(X, **reg_params)
        self.lambda1 = self._require_param("lambda2")

    def get_G(self, beta):
        return self.lambda1 * float(self.xp.sum(self.xp.abs(beta)))

    def get_G_fenchel(self, zeta):
        return 0.0

    def get_prox_G(self, beta, stepsize):
        return self.xp.sign(beta) * self.xp.maximum(
            self.xp.abs(beta) - self.lambda1 * stepsize, 0
        )

    def get_feasible_zeta(self, zeta):
        result = self.xp.max(self.xp.abs(self.X.T @ zeta))
        if result <= self.lambda1:
            return zeta
        return zeta * (self.lambda1 / result)


class regularizer_l1Constrained_gpu(base_regularizer_gpu):
    """
    A class for the L1 constrained regularizer G(beta) = 0 if ||beta||_1 <= s, +inf otherwise
    """

    def __init__(self, X, **reg_params):
        super().__init__(X, **reg_params)
        self.s = self._require_param("M")

    def get_G(self, beta):
        return 0.0

    def get_G_fenchel(self, zeta):
        result = self.xp.max(self.xp.abs(self.X.T @ zeta))
        return float(self.s * result)

    def get_prox_G(self, beta, stepsize):
        beta_cpu = to_cpu(beta)
        u = np.abs(beta_cpu)
        if np.sum(u) <= self.s:
            return beta
        w = np.sort(u)[::-1]
        sv = np.cumsum(w)
        rho = np.nonzero(w - (sv - self.s) / (np.arange(len(w)) + 1) > 0)[0]
        if len(rho) == 0:
            theta = 0
        else:
            rho = rho[-1]
            theta = (sv[rho] - self.s) / (rho + 1)
        w_thresh = np.maximum(u - theta, 0)
        prox_beta_cpu = np.sign(beta_cpu) * w_thresh
        return to_gpu(prox_beta_cpu)
