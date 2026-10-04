import os
import numpy as np
import time

from okglm.helpers.linalg_helpers import (
    estimate_largest_eigenvalue_of_XTX as _estimate_largest_eigenvalue_of_XTX,
)
from okglm.optimizer_factory import (
    get_data_class,
    get_baseline_optimizer_class,
    is_gpu_method,
)

# generate 2D iid Gaussian matrix as desgin matrix
def generate_gaussian_matrix(m, n):
    return np.random.randn(m, n)

# generaet 2D Toeplitz covariance matrix as design matrix
def generate_toeplitz_matrix_old(rho, m, n, seed=0):

    rng = np.random.default_rng(seed)

    # # method 1
    # if rho <= 0 or rho >= 1:
    #     raise ValueError("rho must be in (0, 1)")

    # feature_indices = np.arange(n)
    # Sigma_exponent = np.abs(np.subtract.outer(feature_indices, feature_indices))
    # Sigma = rho ** Sigma_exponent

    # mean = np.zeros(n)
    # return rng.multivariate_normal(mean=np.zeros(n), cov=Sigma, size=m)

    # method 2, using Cholesky decomposition, which is more efficient
    L = generate_toeplitz_matrix_cholesky_decomposition(rho, n)
    base_random_variable = rng.normal(size=(n, m))
    return (L @ base_random_variable).T

# generaet 2D Toeplitz covariance matrix as design matrix
def generate_toeplitz_matrix(rho, m, n, seed=0):

    rng = np.random.default_rng(seed)

    # # method 1
    # if rho <= 0 or rho >= 1:
    #     raise ValueError("rho must be in (0, 1)")

    # feature_indices = np.arange(n)
    # Sigma_exponent = np.abs(np.subtract.outer(feature_indices, feature_indices))
    # Sigma = rho ** Sigma_exponent

    # mean = np.zeros(n)
    # return rng.multivariate_normal(mean=np.zeros(n), cov=Sigma, size=m)

    # method 2, using Cholesky decomposition, which is more efficient
    start_time = time.time()
    L = generate_toeplitz_matrix_cholesky_decomposition(rho, n)
    print(f"generating L takes time: {time.time() - start_time}")
    start_time = time.time()
    base_random_variable = rng.normal(size=(m, n))
    # base_random_variable = rng.normal(size=(n, m))
    print(f"generating base_random_variable takes time: {time.time() - start_time}")
    start_time = time.time()
    output =  base_random_variable @ (L.T)
    # output = (L @ base_random_variable).T
    print(f"generating base_random_variable @ L.T takes time: {time.time() - start_time}")
    return output

def generate_toeplitz_matrix_cholesky_decomposition(rho, n):
    """
    Computes the Cholesky decomposition of a covariance matrix
    with entries Sigma[i,j] = rho^|i-j|.

    Parameters:
    - rho: float, the correlation parameter (0 < rho < 1)
    - n: int, the size of the matrix

    Returns:
    - L: numpy array, the lower triangular Cholesky factor
    """
    if rho <= 0 or rho >= 1:
        raise ValueError("rho must be in (0, 1)")
    
    feature_indices = np.arange(n)
    L_exponent = np.abs(np.subtract.outer(feature_indices, feature_indices))
    L = np.tril(rho ** L_exponent)
    L[:, 1:] *= np.sqrt(1 - rho ** 2)

    return L

def generate_k_equally_spaced_sparse_vector(k, n, coeff_val=1):
    if k > n:
        raise ValueError("k must be less than or equal to n")
    beta = np.zeros(n)
    beta[::n//k] = coeff_val
    return beta

def generate_linear_regression_data_old(m, n, rho, k, coeff_val, snr=5, seed=0):
    rng = np.random.default_rng(seed)
    X = generate_toeplitz_matrix_old(rho, m, n, seed)
    beta = generate_k_equally_spaced_sparse_vector(k, n, coeff_val)
    Xbeta = X @ beta
    sig_noise = np.linalg.norm(Xbeta) / snr
    noise = rng.normal(0, sig_noise, m)
    y = X @ beta + noise
    return X, y, beta

def generate_linear_regression_data(m, n, rho, k, coeff_val, snr=5, seed=0):
    rng = np.random.default_rng(seed)
    start_time = time.time()
    X = generate_toeplitz_matrix(rho, m, n, seed)
    print(f"generating X takes time: {time.time() - start_time}")
    beta = generate_k_equally_spaced_sparse_vector(k, n, coeff_val)
    Xbeta = X @ beta
    sig_noise = np.sqrt((np.std(Xbeta, ddof=1)) ** 2 / snr)
    noise = rng.normal(0, sig_noise, m)
    y = X @ beta + noise
    return X, y, beta

def generate_logistic_regression_data(m, n, rho, k, coeff_val, seed=0):
    rng = np.random.default_rng(seed)
    X = generate_toeplitz_matrix(rho, m, n, seed)
    beta = generate_k_equally_spaced_sparse_vector(k, n, coeff_val)
    Xbeta = X @ beta
    prob = 1 / (1 + np.exp(-Xbeta))
    y = rng.binomial(1, prob, m) * 2 - 1
    return X, y, beta

def generate_poisson_regression_data(m, n, rho, k, coeff_val, seed=0):
    rng = np.random.default_rng(seed)
    X = generate_toeplitz_matrix(rho, m, n, seed)
    beta = generate_k_equally_spaced_sparse_vector(k, n, coeff_val)
    Xbeta = X @ beta
    lambdas = np.exp(Xbeta)
    y = rng.poisson(lambdas)
    return X, y, beta

# generate data sampled from a mixture of Gaussian distribution and Dirac delta distribution
def generate_mixture_of_gaussian_and_Dirac(m, n, mixture_ratio=0.5):
    total_num = m * n
    x = np.random.randn(total_num)
    x[:int(total_num * mixture_ratio)] = 0
    np.random.shuffle(x) # shuffle the data in place
    return x.reshape(m, n)

def generate_mixture_of_Bernoulli_and_Dirac(m, n, mixture_ratio=0.5):
    total_num = m * n
    x = np.random.choice([-1, 1], total_num, p=[0.5, 0.5])
    x[:int(total_num * mixture_ratio)] = 0
    np.random.shuffle(x) # shuffle the data in place
    return x.reshape(m, n)

def generate_mixture_of_Continuous_and_Dirac(m, n, mixture_ratio=0.5, lb=-1, ub=1):
    total_num = m * n
    x = np.random.uniform(lb, ub, total_num)
    x[:int(total_num * mixture_ratio)] = 0
    np.random.shuffle(x) # shuffle the data in place
    return x.reshape(m, n)

def generate_synthetic_data_for_prox_g(p, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0, scale=1, size=p)

def generate_synthetic_data(args):
    dataPath = os.path.join(
        args.dataDir,
        f"GLMLossType={args.GLMLossType}_n={args.n}_p={args.p}_k={args.k}_rho={args.rho}_seed={args.seed}.npz",
    )
    L = 1e8
    use_gpu = is_gpu_method(args.method)
    data_class = get_data_class(args.GLMLossType)
    baseline_optimizer_class = get_baseline_optimizer_class(args.method, args.GLMLossType)

    if args.GLMLossType == "linear":
        if os.path.exists(dataPath):
            print(f"Data already exists; loading data from {dataPath}")
            data = np.load(dataPath)
            X, y, beta_true, L = data['X'], data['y'], data['beta_true'], data['L']
        else:
            X, y, beta_true = generate_linear_regression_data(
                args.n, args.p, rho=args.rho, k=args.k, coeff_val=args.coeff_val, snr=5
            )
            data_class_instance = data_class(X, y, use_gpu=use_gpu)
            L = data_class_instance.get_L()
            np.savez(dataPath, X=X, y=y, beta_true=beta_true, L=L)

    elif args.GLMLossType == "logistic":
        if os.path.exists(dataPath):
            print(f"Data already exists; loading data from {dataPath}")
            data = np.load(dataPath)
            X, y, beta_true, L = data['X'], data['y'], data['beta_true'], data['L']
        else:
            X, y, beta_true = generate_logistic_regression_data(
                args.n, args.p, rho=args.rho, k=args.k, coeff_val=args.coeff_val, seed=args.seed
            )
            data_class_instance = data_class(X, y, use_gpu=use_gpu)
            L = data_class_instance.get_L()
            np.savez(dataPath, X=X, y=y, beta_true=beta_true, L=L)

    elif args.GLMLossType == "poisson":
        if os.path.exists(dataPath):
            print(f"Data already exists; loading data from {dataPath}")
            data = np.load(dataPath)
            X, y, beta_true, L = data['X'], data['y'], data['beta_true'], data['L']
        else:
            X, y, beta_true = generate_poisson_regression_data(
                args.n, args.p, rho=args.rho, k=args.k, coeff_val=args.coeff_val, seed=args.seed
            )
            data_class_instance = data_class(X, y, use_gpu=use_gpu)
            L = data_class_instance.get_L()
            np.savez(dataPath, X=X, y=y, beta_true=beta_true, L=L)

    else:
        raise ValueError(f"Invalid GLMLossType: {args.GLMLossType}")

    return X, y, beta_true, data_class, baseline_optimizer_class, L

# estimate the largest eigenvalue of a matrix X.T @ X through power iteration
def estimate_largest_eigenvalue_of_XTX(X, max_iter=1000, tol=1e-6):
    return _estimate_largest_eigenvalue_of_XTX(X, max_iter=max_iter, tol=tol)

def preprocess_data(X, y, lambda2):
    n, d = X.shape

    XTX = None
    XTy = X.T @ y
    yTy = y @ y

    lmbd_min = 0
    lmbd_max = 1e8

    start_time = time.time()
    if n >= d:
        XTX = X.T @ X
        lmbd = np.linalg.eigvalsh(XTX)
        lmbd_min = lmbd[0] * 0.95
        lmbd_max = lmbd[-1]
    else:
        eig_vec = np.random.randn(d)
        for i in range(1000):
            eig_vec = X.T @ (X @ eig_vec)
            eig_vec = eig_vec / np.linalg.norm(eig_vec)
        lmbd_max = np.linalg.norm(X @ eig_vec)
    print(f"calculating eigenvalues takes time: {time.time() - start_time}")

    if lmbd_min > 1e-6:
        lambda2 += lmbd_min
        XTX -= lmbd_min * np.eye(d)

    return XTX, XTy, yTy, lambda2, lmbd_min, lmbd_max

def generate_more_balanced_data(X, y, seed=0):
    
    # Check that y only has two unique values: 1 and -1
    unique_values = np.unique(y)
    if len(unique_values) != 2 or not (np.array_equal(np.sort(unique_values), [-1, 1])):
        raise ValueError("y must contain exactly two unique values: -1 and 1")

    y_pos_indices = np.where(y == 1)[0]
    y_neg_indices = np.where(y == -1)[0]

    X_pos = X[y_pos_indices]
    X_neg = X[y_neg_indices]
    y_pos = y[y_pos_indices]
    y_neg = y[y_neg_indices]

    n = len(y)
    n_pos = len(y_pos)
    n_neg = len(y_neg)

    rng = np.random.default_rng(seed)

    new_y_pos_indices = rng.choice(n_pos, size=n, replace=True)
    new_y_neg_indices = rng.choice(n_neg, size=n, replace=True)

    X_pos = X_pos[new_y_pos_indices]
    y_pos = y_pos[new_y_pos_indices]
    X_neg = X_neg[new_y_neg_indices]
    y_neg = y_neg[new_y_neg_indices]

    X_new = np.vstack((X_pos, X_neg))
    y_new = np.hstack((y_pos, y_neg))

    return X_new, y_new

if __name__ == "__main__":
    # print(generate_gaussian_matrix(10, 10))
    A = generate_toeplitz_matrix(0.5, 5, 5)
    print(A)
    # print(generate_mixture_of_gaussian_and_Dirac(10, 10))
    # print(generate_mixture_of_Bernoulli_and_Dirac(10, 10))
    # print(generate_mixture_of_Continuous_and_Dirac(10, 10))
