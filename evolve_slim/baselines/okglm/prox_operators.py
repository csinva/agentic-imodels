import os
import sys
import numpy as np
import time
import scipy

try:  # evolve_slim: cvxpy only serves the reference solvers, not the BnB
    import cvxpy
except ImportError:
    cvxpy = None
import numba

from okglm.pava_algorithms import pava_numba_nonincreasing_Huber
from sklearn.isotonic import IsotonicRegression
isotonic_clf = IsotonicRegression()

def prox_kyFan_l2Norm_squared(x, k, rho):
    # want to find \argmin_{gamma} 1/2 ||gamma - x||_2^2 + rho * h(gamma), where
    # h(gamma) = 1/2 \sum_{j=1}^k gamma_[j]^2, where gamma_[j] is the j-th largest element of gamma (in absolute value)

    abs_x = np.abs(x)
    argmax_k_abs_x = np.argpartition(abs_x, -k)[-k:]
    w = np.ones_like(x)
    w[argmax_k_abs_x] = 1 + rho # / 2
    b = abs_x / w

    v = isotonic_clf.fit_transform(X=abs_x, y=b, sample_weight=w)
    return v * np.sign(x)

def prox_fenchel_kyFan_l2Norm_squared(x, k, rho):
    # want to find \argmin_{beta} 1/2 ||beta - x||_2^2 + rho * h^*(beta), where
    # h^*(beta) is the Fenchel conjugate of h(beta), 
    # h(beta) = 1/2 \sum_{j=1}^k beta_[j]^2, 
    # beta_[j] is the j-th largest element of beta (in absolute value)

    return x - rho * prox_kyFan_l2Norm_squared(x/rho, k, 1./rho)

def prox_HuberLoss(x, rho, M=1e6):
    # want to find \argmin_{gamma} 1/2 ||gamma - x||_2^2 + rho * h(gamma), where
    # h(gamma) = H_M(gamma), 
    # H_M(\alpha) = 1/2 * \alpha^2 if |\alpha| <= M, and H_M(\alpha) = M * |\alpha| - 1/2 * M^2 if |\alpha| > M

    abs_x = np.abs(x)
    sign_x = np.sign(x)
    threshold = M * (1 + rho)

    # Apply element-wise condition
    gamma = np.where(abs_x <= threshold, x / (1 + rho), x - sign_x * rho * M)
    return gamma

def prox_kyFan_HuberLoss_pava(x, k, rho, M=1e6):
    # want to find \argmin_{gamma} 1/2 ||gamma - x||_2^2 + rho * h(gamma), where
    # h(gamma) = \sum_{j=1}^k H_M(gamma_[j]), 
    # gamma_[j] is the j-th largest element of gamma (in absolute value),
    # H_M(\alpha) = 1/2 * \alpha^2 if |\alpha| <= M, and H_M(\alpha) = M * |\alpha| - 1/2 * M^2 if |\alpha| > M

    abs_x = np.abs(x)
    order_abs_x = np.argsort(-abs_x)

    abs_x_sorted = abs_x[order_abs_x]
    w = 0.5 * np.ones(len(x))

    # time_start = time.time()
    v= pava_numba_nonincreasing_Huber(abs_x_sorted, w, k, rho, M)
    # print(f"pava_numba_nonincreasing_Huber time: {time.time() - time_start}")

    v_sorted = np.zeros(len(x))
    v_sorted[order_abs_x] = v

    return v_sorted * np.sign(x)

def prox_kyFan_HuberLoss_pava_bnb(x, k_reduced, rho, z_lb, z_ub_reversed, z_free, M=1e6):
    # Special handling for z bounds.
    v_returned = np.zeros(len(x))
    v_returned[z_ub_reversed] = x[z_ub_reversed]
    v_returned[z_lb] = prox_HuberLoss(x[z_lb], rho, M)

    if k_reduced <= 0:
        v_returned[z_free] = x[z_free]
        return v_returned

    x_free = x[z_free]
    if x_free.size == 0:
        return v_returned

    abs_x = np.abs(x_free)
    order_abs_x = np.argsort(-abs_x)
    abs_x_sorted = abs_x[order_abs_x]
    w = 0.5 * np.ones(len(x_free))

    v = pava_numba_nonincreasing_Huber(abs_x_sorted, w, k_reduced, rho, M)

    v_sorted = np.zeros(len(x_free))
    v_sorted[order_abs_x] = v
    v_returned[z_free] = v_sorted * np.sign(x_free)
    return v_returned

def prox_fenchel_kyFan_HuberLoss_pava(x, k, rho, M=1e6):
    # want to find \argmin_{beta} 1/2 ||beta - x||_2^2 + rho * h^*(beta), where 
    # h^*(beta) is the Fenchel conjugate of h(beta), 
    # h(beta) = \sum_{j=1}^k H_M(beta_[j]),
    # beta_[j] is the j-th largest element of beta (in absolute value),
    # H_M(\alpha) = 1/2 * \alpha^2 if |\alpha| <= M, and H_M(\alpha) = M * |\alpha| - 1/2 * M^2 if |\alpha| > M

    return x - rho * prox_kyFan_HuberLoss_pava(x/rho, k, 1./rho, M)

def prox_fenchel_kyFan_HuberLoss_pava_bnb(x, k_reduced, rho, z_lb, z_ub_reversed, z_free, M=1e6):
    return x - rho * prox_kyFan_HuberLoss_pava_bnb(x / rho, k_reduced, 1.0 / rho, z_lb, z_ub_reversed, z_free, M)

def compute_kyFan_l2Norm_squared(x, k):
    abs_x = np.abs(x)
    top_k_abs_x = np.partition(abs_x, -k)[-k:]
    return 0.5 * np.sum(top_k_abs_x ** 2)

# @numba.njit this makes it slower...
def compute_fenchel_kyFan_l2Norm_squared(x, k):
    u = np.zeros(k)
    sorted_abs_x = np.sort(np.abs(x))
    cumsum_sorted_abs_x = np.cumsum(sorted_abs_x)
    for j in range(k):
        reverse_j = -j - 1
        avg = cumsum_sorted_abs_x[reverse_j] / (k - j)
        if avg >= sorted_abs_x[reverse_j]:
            u[j:] = avg
            break
        else:
            u[j] = sorted_abs_x[reverse_j]
    return 0.5 * np.sum(u ** 2)

def compute_kyFan_HuberLoss(x, k, M=1e6):
    abs_x = np.abs(x)
    top_k_abs_x = np.partition(abs_x, -k)[-k:]
    return np.sum(np.where(top_k_abs_x <= M, 0.5 * top_k_abs_x ** 2, M * top_k_abs_x - 0.5 * M ** 2))

def compute_kyFan_HuberLoss_bnb(x, k_reduced, z_lb, z_ub_reversed, z_free, M=1e6):
    abs_x_z_lb = np.abs(x[z_lb])
    sum1 = np.sum(
        np.where(
            abs_x_z_lb <= M,
            0.5 * abs_x_z_lb ** 2,
            M * abs_x_z_lb - 0.5 * M ** 2,
        )
    )
    if k_reduced <= 0:
        return sum1

    abs_x = np.abs(x[z_free])
    if abs_x.size == 0:
        return sum1
    top_k_reduced_abs_x = np.partition(abs_x, -k_reduced)[-k_reduced:]
    sum2 = np.sum(
        np.where(
            top_k_reduced_abs_x <= M,
            0.5 * top_k_reduced_abs_x ** 2,
            M * top_k_reduced_abs_x - 0.5 * M ** 2,
        )
    )
    return sum1 + sum2

def compute_fenchel_kyFan_HuberLoss(x, k, M=1e6):

    # abs_x = np.abs(x)

    # abs_x_max = abs_x.max()
    # if abs_x_max > M + 1e-8:
    #     raise ValueError(f"The input x is not in the domain of the Fenchel conjugate of the kyFan Huber loss; abs_x.max ({abs_x_max}) > M ({M})")
    
    # sum_abs_x = np.sum(abs_x)
    # if sum_abs_x > k * M + 1e-8:
    #     raise ValueError(f"The input x is not in the domain of the Fenchel conjugate of the kyFan Huber loss; sum_abs_x ({sum_abs_x}) > k * M ({k * M})")
    
    return compute_fenchel_kyFan_l2Norm_squared(x, k)

def compute_fenchel_kyFan_l2Norm_squared_bnb(x, k_reduced, z_lb, z_ub_reversed, z_free):
    if k_reduced <= 0:
        return 0.5 * np.sum(x[z_lb] ** 2)

    x_free = x[z_free]
    if x_free.size == 0:
        return 0.5 * np.sum(x[z_lb] ** 2)

    u = np.zeros(k_reduced)
    sorted_abs_x = np.sort(np.abs(x_free))
    cumsum_sorted_abs_x = np.cumsum(sorted_abs_x)
    for j in range(k_reduced):
        reverse_j = -j - 1
        avg = cumsum_sorted_abs_x[reverse_j] / (k_reduced - j)
        if avg >= sorted_abs_x[reverse_j]:
            u[j:] = avg
            break
        else:
            u[j] = sorted_abs_x[reverse_j]
    return 0.5 * np.sum(u ** 2) + 0.5 * np.sum(x[z_lb] ** 2)

def compute_fenchel_kyFan_HuberLoss_bnb(x, k_reduced, z_lb, z_ub_reversed, z_free, M=1e6):
    return compute_fenchel_kyFan_l2Norm_squared_bnb(
        x, k_reduced, z_lb, z_ub_reversed, z_free
    )

def compute_fenchel_kyFan_HuberLoss_cvxpy(x, k, M=1e6):
    p = len(x)
    s = cvxpy.Variable(p)
    z = cvxpy.Variable(p)

    constraints = [
        z >= 0, 
        z <= 1, 
        cvxpy.sum(z) <= k, 
        np.abs(x) <= M * z, 
        cvxpy.SOC(t=s+z, X=cvxpy.vstack([2 * x, s - z]))
    ]

    loss = 0.5 * cvxpy.sum(s)

    problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)
    problem.solve(solver=cvxpy.CLARABEL)

    return problem.value

def prox_kyFan_HuberLoss_iterative(x, k, rho, M=1e6):
    # want to find \argmin_{gamma} 1/2 ||gamma - x||_2^2 + rho * h(gamma), where
    # h(gamma) = 1/2 \sum_{j=1}^k H_M(gamma_[j])^2, where gamma_[j] is the j-th largest element of gamma (in absolute value)

    abs_x = np.abs(x)
    u = np.zeros(k)
    order_abs_x = np.argsort(-abs_x)

    abs_x_sorted = abs_x[order_abs_x]
    w = np.ones(len(x))
    w[:k] = 1 + rho

    y = abs_x_sorted.copy()
    v_sorted = np.zeros(len(x))
    v = abs_x.copy()

    for iter in range(1000):
        u = np.sign(v[:k]) * np.maximum(0, np.abs(v[:k]) - M)
        y[:k] = (abs_x_sorted[:k] + rho * u) / (1 + rho)
        v = scipy.optimize.isotonic_regression(y, weights=w, increasing=False).x

    v_sorted[order_abs_x] = v
    return v_sorted * np.sign(x)

def compute_total_loss_prox_g(x, gamma, k, rho, M=1e6):
    """Compute total loss for prox_g case.
    
    Args:
        x: Input vector
        gamma: Current solution vector
        k: Number of top elements to consider
        rho: Regularization parameter
        M: Huber loss parameter (default: 1e6)
    
    Returns:
        Total loss value: 1/2 ||gamma - x||_2^2 + rho * h(gamma)
        where h(gamma) = sum_{j=1}^k H_M(gamma_[j])
    """
    diff_norm_squared = 0.5 * np.sum((gamma - x) ** 2)
    huber_loss = compute_kyFan_HuberLoss(gamma, k, M)
    return diff_norm_squared + rho * huber_loss

def compute_total_loss_prox_g_fenchel(x, beta, k, rho, M=1e6):
    """Compute total loss for prox_g_fenchel case.
    
    Args:
        x: Input vector
        beta: Current solution vector
        k: Number of top elements to consider
        rho: Regularization parameter
        M: Huber loss parameter (default: 1e6)
    
    Returns:
        Total loss value: 1/2 ||beta - x||_2^2 + rho * h^*(beta)
        where h^*(beta) is the Fenchel conjugate of h(beta)
    """
    diff_norm_squared = 0.5 * np.sum((beta - x) ** 2)
    fenchel_loss = compute_fenchel_kyFan_HuberLoss(beta, k, M)
    return diff_norm_squared + rho * fenchel_loss
