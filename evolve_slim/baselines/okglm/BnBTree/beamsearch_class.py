import numpy as np
import time
from scipy.optimize import minimize
import os
import gc
from okglm.utils import convert_GB_to_bytes

from okglm.data_generation_and_preprocessing import (
    generate_linear_regression_data,
    generate_logistic_regression_data,
    generate_more_balanced_data,
)

def compute_logistic_loss_expyXbeta(ExpyXbeta):
    return np.sum(np.log1p(1. / ExpyXbeta))

class base_data_GLM:
    """
    A class for the data for the generalized linear models (GLM) optimization problem with the following form: F(beta) = F_y(X^T beta) = f(X^T beta, y).
    """

    def __init__(self, X, y):
        self.n, self.p = X.shape
        self.X = X
        self.y = y

        self.XT = X.T

        self.L = np.ones(self.p) * 1e8

        self.preprocess_data()
    
    def preprocess_data(self):
        pass

    def get_grad_F_j(self, intermediate_var, j):
        pass

    def get_grad_F(self, intermediate_var):
        pass

    def get_F(self, intermediate_var):
        pass

    def update_intermediate_var(self, intermediate_var, j, diff_beta_j):
        pass

    def compute_intermediate_var(self, beta):
        # for linear regression, intermediate_var is y - X^T beta; 
        # for logistic regression, intermediate_var is exp(y * X^T beta)
        pass

    def compute_intermediate_var_betaSub(self, betaSub, intermediate_var2):
        # for linear regression, intermediate_var2 is XSub;
        # for logistic regression, intermediate_var2 is yXSub
        pass

    def get_scipy_loss_func_intermediate_var2_on_support_indices(self, support_indices):
        pass

    def get_scipy_loss_func(self, intermediate_var2, lambda2=0):
        # for linear regression, intermediate_var2 is XSub; 
        # for logistic regression, intermediate_var2 is yXSub
        pass

class data_LinearRegression(base_data_GLM):
    """
    A class for the data for the linear regression optimization problem with the following form: F(beta) = F_y(X^T beta) = ||X^T beta - y||^2.
    """

    def __init__(self, X, y):
        super().__init__(X, y)
    
    def preprocess_data(self):
        self.L = np.sum(self.X ** 2, axis=0)
    
    def get_grad_F_j(self, y_minus_Xbeta, j):
        return -2 * y_minus_Xbeta.dot(self.X[:, j])

    def get_grad_F(self, y_minus_Xbeta):
        return -2 * self.X.T.dot(y_minus_Xbeta)
    
    def get_F(self, y_minus_Xbeta):
        return np.sum(y_minus_Xbeta ** 2)
    
    def update_intermediate_var(self, y_minus_Xbeta, j, diff_beta_j):
        y_minus_Xbeta -= self.X[:, j] * diff_beta_j
    
    def compute_intermediate_var(self, beta):
        return self.y - self.X @ beta

    def compute_intermediate_var_betaSub(self, betaSub, XSub):
        return self.y - XSub @ betaSub
    
    def get_scipy_loss_func_intermediate_var2_on_support_indices(self, support_indices):
        return self.X[:, support_indices]
    
    def get_scipy_loss_func(self, XSub, lambda2=0):
        def loss_func(beta):
            return np.sum((XSub @ beta - self.y) ** 2) + lambda2 * np.sum(beta ** 2)
        return loss_func

class data_LogisticRegression(base_data_GLM):
    """
    A class for the data for the logistic regression optimization problem with the following form: F(beta) = F_y(X^T beta) = \sum_{i=1}^n log(1 + exp(-y_i X_i^T beta)).
    """

    def __init__(self, X, y):
        super().__init__(X, y)
    
    def preprocess_data(self):
        self.yX = self.y.reshape(-1, 1) * self.X # yX[i] = y[i] * X[i]
        self.L = np.sum(self.X ** 2, axis=0) * 0.25
    
    def get_grad_F_j(self, ExpyXbeta, j):
        return - self.yX[:, j].dot(1. / (1. + ExpyXbeta))
    
    def get_grad_F(self, ExpyXbeta):
        return - self.yX.T.dot(1. / (1. + ExpyXbeta))
    
    def get_F(self, ExpyXbeta):
        return compute_logistic_loss_expyXbeta(ExpyXbeta)
    
    def update_intermediate_var(self, ExpyXbeta, j, diff_beta_j):
        ExpyXbeta *= np.exp(self.yX[:, j] * diff_beta_j)
    
    def compute_intermediate_var(self, beta):
        return np.exp(self.yX @ beta)
    
    def compute_intermediate_var_betaSub(self, betaSub, yXSub):
        return np.exp(yXSub @ betaSub)
    
    def get_scipy_loss_func_intermediate_var2_on_support_indices(self, support_indices):
        return self.yX[:, support_indices]
    
    def get_scipy_loss_func(self, yXSub, lambda2=0):
        def loss_func(beta):
            return np.sum(np.log1p(1. / np.exp(yXSub @ beta))) + lambda2 * np.sum(beta ** 2)
        return loss_func

class GLMOptimizer:
    """
    A class for training the GLMs (generalized linear models) with the coordinate descent (CD) algorithm.
    """

    def __init__(self, X, y, lambda2, M, data_class):
        self.n, self.p = X.shape
        self.lambda2 = lambda2
        self.twoLambda2 = 2 * lambda2
        self.M = M
        self.data_class = data_class(X, y)
        self.L = self.data_class.L + self.twoLambda2 # Lipschitz constants for the gradient of F at each coordinate

        self.beta = np.zeros(self.p)
        self.intermediate_var = self.data_class.compute_intermediate_var(self.beta) # for linear regression, intermediate_var is y - X^T beta; for logistic regression, intermediate_var is exp(y * X^T beta)

        self.bounds = [(-M, M) for _ in range(self.p)]  # Box constraints for each beta_j
    
    def reset_beta(self, beta):
        self.beta = beta
        self.intermediate_var = self.data_class.compute_intermediate_var(self.beta)
    
    def get_beta(self):
        return self.beta
    
    def get_intermediate_var(self):
        return self.intermediate_var
    
    def get_total_loss(self, beta, intermediate_var):
        return self.data_class.get_F(intermediate_var) + self.twoLambda2 * np.sum(beta ** 2)

    def get_grad_at_coord_j(self, j, beta, intermediate_var):
        return self.data_class.get_grad_F_j(intermediate_var, j) + self.twoLambda2 * self.beta[j]
    
    def optimize_1step_at_coord(self, beta, intermediate_var, j):
        grad_j = self.get_grad_at_coord_j(j, beta, intermediate_var)
        step_at_j = grad_j / self.L[j]

        prev_beta_j = beta[j]
        curr_beta_j = prev_beta_j - step_at_j
        curr_beta_j = max(-self.M, min(self.M, curr_beta_j))

        diff_beta_j = curr_beta_j - prev_beta_j
        beta[j] = curr_beta_j
        self.data_class.update_intermediate_var(intermediate_var, j, diff_beta_j)

    # def finetune_on_current_support(self, beta, intermediate_var, max_iter=1000, tol=1e-6):
    #     support_indices = np.where(beta != 0)[0]
    #     loss_prev = self.get_total_loss(beta, intermediate_var)

    #     for curr_iter in range(max_iter):
    #         for j in support_indices:
    #             self.optimize_1step_at_coord(beta, intermediate_var, j)
    #             tmp_loss = self.get_total_loss(beta, intermediate_var)
            
    #         if curr_iter % 10 == 0:
    #             loss_curr = self.get_total_loss(beta, intermediate_var)
    #             print(f"iter {curr_iter}, prev_loss: {loss_prev} loss: {loss_curr}")

    #             if abs((loss_curr - loss_prev) / loss_prev) < tol:
    #                 curr_gap = abs((loss_curr - loss_prev) / loss_prev)
    #                 print(f"curr_gap: {curr_gap}, break!")
    #                 break
                
    #             loss_prev = loss_curr
        
    #     loss_final = self.get_total_loss(beta, intermediate_var)
    #     return beta, intermediate_var, loss_final

    def finetune_on_current_support(self, supp_mask):

        scipy_loss_func_intermediate_var2_on_support_indices = self.data_class.get_scipy_loss_func_intermediate_var2_on_support_indices(supp_mask)
        loss_func = self.data_class.get_scipy_loss_func(scipy_loss_func_intermediate_var2_on_support_indices, self.lambda2)
        supp_size = np.sum(supp_mask)
        bounds = self.bounds[:supp_size]
        beta_init = np.zeros(supp_size)
        scipy_optimizer = minimize(loss_func, beta_init, bounds=bounds)

        beta_on_supp = scipy_optimizer.x
        loss_final = scipy_optimizer.fun
        intermediate_var_final = self.data_class.compute_intermediate_var_betaSub(beta_on_supp, scipy_loss_func_intermediate_var2_on_support_indices)

        return beta_on_supp, intermediate_var_final, loss_final

    
class BeamSearchOptimizer:
    """
    A class for the Proximal Gradient Descent (PGD) optimizer to solve the unconstrained convex composite optimization problem for generalized linear models (GLM) with the following form:
    minimize F(beta) + G(beta), where
        F(beta) = F_y(X^T beta) = f(X^T beta, y),
        G(beta) = 2\lambda_2 g(beta), where
            f is the GLM loss function,
            g(beta) = \min_z \sum_{j=1}^p beta_j^2 / z_j, s.t. 0 <= z_j <= 1, \sum_{j=1}^p z_j <= k, |beta_j| <= M * z_j
    """
    
    def __init__(self, X, y, lambda2, M, data_class, parent_size=10, child_size=10, allowed_supp_mask=None, max_memory_GB=50):
        self.n, self.p = X.shape
        self.lambda2 = lambda2
        self.M = M
        self.twoLambda2 = 2 * lambda2

        self.solution = None
        self.solver_time = None
        self.solution_status = "not converged; reached max_iter"

        self.beta = np.zeros(self.p)
        self.z_lb = np.zeros(self.p).astype(bool)
        self.z_ub = np.ones(self.p).astype(bool)
        self.z_ub_reversed = ~self.z_ub
        self.z_free = (~self.z_lb) & self.z_ub

        # beam search parameters
        # new
        self.GLMOptimizer = GLMOptimizer(X, y, lambda2, M, data_class)

        # old
        self.parent_size = parent_size
        self.child_size = child_size

        self.supp_mask_arr_parent = np.zeros((parent_size, self.p)).astype(bool)
        self.num_parent = 1

        self.total_child_size = self.parent_size * self.child_size
        self.supp_mask_arr_child = np.zeros((self.total_child_size, self.p)).astype(bool)
        self.loss_arr_child = np.zeros(self.total_child_size)

        if allowed_supp_mask is None:
            self.allowed_supp_mask = np.ones(self.p).astype(bool)
        else:
            self.allowed_supp_mask = allowed_supp_mask

        self.saved_solution = {}
        supp_mask_all_False = np.zeros(self.p).astype(bool)
        tmp_support_str = supp_mask_all_False.tobytes()
        intermediate_var = self.GLMOptimizer.data_class.compute_intermediate_var(self.beta)
        self.saved_solution[tmp_support_str] = (None, intermediate_var, self.GLMOptimizer.data_class.get_F(intermediate_var))

        self.max_memory_GB = max_memory_GB
        tmp_support_bytes = supp_mask_all_False.nbytes
        entry_bytes = intermediate_var.nbytes + self.beta.nbytes + tmp_support_bytes
        self.max_saved_solutions = max(1, int(convert_GB_to_bytes(self.max_memory_GB) / max(entry_bytes, 1)))

    def get_beta(self):
        return self.beta
    
    def get_beta_path(self):
        return self.beta_path
    
    def get_intermediate_var(self):
        return self.intermediate_var
    
    def get_loss(self):
        return self.loss
    
    def get_total_loss(self, beta, intermediate_var):
        return self.GLMOptimizer.data_class.get_F(intermediate_var) + self.twoLambda2 * np.sum(beta ** 2)
    
    def reset_fixed_supp_and_allowed_supp(self, fixed_supp_mask, allowed_supp_mask):
        """Reset the fixed support and allowed support

        Args:
            fixed_supp_mask (np.array): 1D array of boolean values indicating the fixed support
            allowed_supp_mask (np.array): 1D array of boolean values indicating the allowed support
        """
        self.fixed_supp_mask = fixed_supp_mask
        self.allowed_supp_mask = allowed_supp_mask
        self.beta.fill(0)
        tmp_support_str = fixed_supp_mask.tobytes()
        if tmp_support_str in self.saved_solution:
            beta_on_supp_tmp, self.r, self.loss = self.saved_solution[tmp_support_str]
        else:
            beta_on_supp_tmp, self.r, self.loss = self.GLMOptimizer.finetune_on_current_support(fixed_supp_mask)
        self.beta[fixed_supp_mask] = beta_on_supp_tmp
    
    def get_sparse_sol_via_OMP(self, k):
        nonzero_indices_set = set(np.where(np.abs(self.beta) > 1e-6)[0])
        num_nonzero = len(nonzero_indices_set)
        zero_indices_set = set(np.where(self.allowed_supp_mask)[0]) - nonzero_indices_set

        if len(zero_indices_set) == 0:
            return
    
        self.supp_mask_arr_parent[0] = np.abs(self.beta) > 1e-9

        self.num_parent = 1
        self.forbidden_support = set()

        while num_nonzero < min(k, self.p):
            num_nonzero += 1
            self.beamSearch_multipleSupports_via_OMP_by_1()
        
        del self.forbidden_support
        gc.collect()

        best_sol_supp_mask = self.supp_mask_arr_parent[0]
        beta_on_supp, self.intermediate_var, self.loss = self.GLMOptimizer.finetune_on_current_support(best_sol_supp_mask)
        self.beta.fill(0.0)
        self.beta[best_sol_supp_mask] = beta_on_supp

    def reset_state(self):
        """Reset cached state so the optimizer behaves like a fresh instance."""
        self.beta.fill(0.0)
        self.z_lb.fill(False)
        self.z_ub.fill(True)
        self.z_ub_reversed = ~self.z_ub
        self.z_free = (~self.z_lb) & self.z_ub

        self.supp_mask_arr_parent.fill(False)
        self.num_parent = 1
        self.supp_mask_arr_child.fill(False)
        self.loss_arr_child.fill(0.0)

        self.saved_solution = {}
        supp_mask_all_false = np.zeros(self.p, dtype=bool)
        tmp_support_str = supp_mask_all_false.tobytes()
        intermediate_var = self.GLMOptimizer.data_class.compute_intermediate_var(self.beta)
        self.saved_solution[tmp_support_str] = (None, intermediate_var, self.GLMOptimizer.data_class.get_F(intermediate_var))
    
    def get_sparse_sol_via_OMP_path(self, k):
        nonzero_indices_set = set(np.where(np.abs(self.beta) > 1e-6)[0])
        num_nonzero = len(nonzero_indices_set)
        zero_indices_set = set(np.where(self.allowed_supp_mask)[0]) - nonzero_indices_set

        if len(zero_indices_set) == 0:
            raise ValueError("There are no allowed indices to add to the support")
    
        self.supp_mask_arr_parent[0] = np.abs(self.beta) > 1e-9


        self.num_parent = 1
        self.forbidden_support = set()

        self.beta_path = []
        while num_nonzero < min(k, self.p):
            num_nonzero += 1
            self.beamSearch_multipleSupports_via_OMP_by_1()

            best_sol_supp_mask = self.supp_mask_arr_parent[0]
            beta_on_supp, self.intermediate_var, self.loss = self.GLMOptimizer.finetune_on_current_support(best_sol_supp_mask)
            beta_tmp = np.zeros(self.p)
            beta_tmp[best_sol_supp_mask] = beta_on_supp
            self.beta_path.append(beta_tmp)
        
        del self.forbidden_support
        gc.collect()

        self.beta_path = np.vstack(self.beta_path)

    def beamSearch_multipleSupports_via_OMP_by_1(self):
        self.loss_arr_child.fill(1e32)
        self.total_child_added = 0

        for i in range(self.num_parent):
            self.expand_parent_i_support_via_OMP_by_1(i)
        
        child_indices = np.argsort(self.loss_arr_child)[:min(self.parent_size, self.total_child_added)] # get indices of children which have the smallest losses
        num_child_indices = len(child_indices)

        self.supp_mask_arr_parent[:num_child_indices] = self.supp_mask_arr_child[child_indices]

        self.num_parent = num_child_indices
    
    def expand_parent_i_support_via_OMP_by_1(self, i):
        fixed_supp_mask = self.supp_mask_arr_parent[i]
        unfixed_and_allowed_mask = np.logical_xor(self.allowed_supp_mask, fixed_supp_mask)
        unfixed_and_allowed_indicies = np.where(unfixed_and_allowed_mask)[0]

        tmp_support_str = self.supp_mask_arr_parent[i].tobytes()
        if tmp_support_str in self.saved_solution:
            _, intermediate_var_parent_i, _ = self.saved_solution[tmp_support_str]
        else:
            beta_on_supp_tmp, intermediate_var_parent_i, loss_tmp = self.GLMOptimizer.finetune_on_current_support(self.supp_mask_arr_parent[i])
            if len(self.saved_solution) < self.max_saved_solutions:
                self.saved_solution[tmp_support_str] = (beta_on_supp_tmp, intermediate_var_parent_i, loss_tmp)
        # half_grad_on_unfixed_and_allowed_supp = intermediate_var_parent_i[unfixed_and_allowed_indicies] - self.XTy[unfixed_and_allowed_indicies]
        # abs_half_grad_on_unfixed_and_allowed_supp =  half_grad_on_unfixed_and_allowed_supp ** 2 / self.half_Lipschitz[unfixed_and_allowed_indicies]
        grad_on_unfixed_and_allowed_supp = self.GLMOptimizer.data_class.get_grad_F(intermediate_var_parent_i)[unfixed_and_allowed_indicies]
        loss_decrease_on_unfixed_and_allowed_supp = grad_on_unfixed_and_allowed_supp ** 2 / self.GLMOptimizer.data_class.L[unfixed_and_allowed_indicies]

        num_new_js = min(self.child_size, len(unfixed_and_allowed_indicies))
        # new_js = unfixed_and_allowed_indicies[np.argsort(-abs_half_grad_on_unfixed_and_allowed_supp)][:num_new_js]
        new_js = unfixed_and_allowed_indicies[np.argsort(-loss_decrease_on_unfixed_and_allowed_supp)][:num_new_js]
        child_start, child_end = i * self.child_size, i*self.child_size + num_new_js

        self.supp_mask_arr_child[child_start:child_end] = self.supp_mask_arr_parent[i]

        for l in range(num_new_js):
            child_id = child_start + l
            self.supp_mask_arr_child[child_id, new_js[l]] = True
            tmp_support_str = self.supp_mask_arr_child[child_id].tobytes()
            if tmp_support_str not in self.forbidden_support:
                self.total_child_added += 1
                self.forbidden_support.add(tmp_support_str)

                if tmp_support_str in self.saved_solution:
                    _, _, self.loss_arr_child[child_id] = self.saved_solution[tmp_support_str]
                else:
                    beta_on_supp_tmp, r_tmp, self.loss_arr_child[child_id] = self.GLMOptimizer.finetune_on_current_support(self.supp_mask_arr_child[child_id])
                    if len(self.saved_solution) <= self.max_saved_solutions:
                        self.saved_solution[tmp_support_str] = (beta_on_supp_tmp, r_tmp, self.loss_arr_child[child_id])

def run_GLMOptimizer_finetune():
    np.random.seed(0)
    n, p = 10000, 300
    X = np.random.randn(n, p)
    beta_true = np.ones(p)
    lambda2 = 1e-2
    M = 2

    y = X @ beta_true + 0.1 * np.random.randn(n)
    data_class = data_LinearRegression

    # y = np.random.binomial(1, 1. / (1. + np.exp(-X @ beta_true)))
    # y = 2 * y - 1
    # yX = y.reshape(-1, 1) * X

    # data_class = data_LogisticRegression

    optimizer = GLMOptimizer(X, y, lambda2, M, data_class)

    supp_mask_init = np.ones(p).astype(bool)

    start_time = time.time()
    beta_final, intermediate_var_final, loss_final = optimizer.finetune_on_current_support(supp_mask_init)
    CD_time = time.time() - start_time

    # solve linear regression using scipy
    beta_scipy = np.random.randn(p)
    
    start_time = time.time()
    bounds = [(-M, M) for _ in range(p)]  # Box constraints for each beta_j

    def loss_LinearRegression_func(beta):
        return np.sum((X @ beta - y) ** 2) + lambda2 * np.sum(beta ** 2)
    
    def loss_LogisticRegression_func(beta):
        return np.sum(np.log1p(1. / np.exp(yX @ beta))) + lambda2 * np.sum(beta ** 2)

    scipy_optimizer = minimize(loss_LinearRegression_func, beta_scipy, bounds=bounds)
    # scipy_optimizer = minimize(loss_LogisticRegression_func, beta_scipy, bounds=bounds)
    scipy_time = time.time() - start_time

    print("CD loss:", loss_final)
    print("CD beta:", beta_final)
    print("CD time:", CD_time)

    print("scipy loss:", scipy_optimizer.fun)
    print("scipy beta:", scipy_optimizer.x)
    print("scipy time:", scipy_time)

def run_beamsearch_cv(
    openML_dataset_name,
    data_class,
    lambda2=1e0,
    M=10.0,
    k=5,
    n_fold=5,
    random_state=0,
    data_dir=None,
    parent_size=5,
    child_size=5,
):
    from sklearn.model_selection import KFold, StratifiedKFold

    def compute_metric_SE(y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def compute_metric_logistic(y_pred, y_true):
        return float(np.mean(np.log(1 + np.exp(-y_pred * y_true))))

    def compute_metric_acc(y_pred, y_true):
        return float(np.mean(np.sign(y_pred) == y_true))

    def compute_metric_auc(y_pred, y_true):
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) > 2:
            return np.nan
        return float(roc_auc_score(y_true, y_pred))

    if data_dir is None:
        data_dir = os.environ.get("data_dir") or os.environ.get("DATA_DIR")
        if not data_dir:
            raise EnvironmentError("Missing environment variable: data_dir")

    from python_expt_scripts.BnB_expt_pipeline import download_and_load_realworld_dataset, realworld_data_dict

    X, y = download_and_load_realworld_dataset(openML_dataset_name, data_dir, realworld_data_dict)

    if data_class == data_LogisticRegression:
        y_pos_indices = np.where(y == 1)[0]
        y_neg_indices = np.where(y == -1)[0]

        X_pos = X[y_pos_indices]
        X_neg = X[y_neg_indices]
        y_pos = y[y_pos_indices]
        y_neg = y[y_neg_indices]

        n_pos = len(y_pos)
        n_neg = len(y_neg)
        print(f"n_pos: {n_pos}, n_neg: {n_neg}, X_pos.shape: {X_pos.shape}, X_neg.shape: {X_neg.shape}")

        X = np.vstack((X_pos, X_neg))
        y = np.hstack((y_pos, y_neg))

    X_mean = np.mean(X, axis=0)
    X = X - X_mean
    X_norm = np.linalg.norm(X, axis=0)
    zero_norm_indices = X_norm == 0
    X = X[:, ~zero_norm_indices]
    X = X / X_norm[~zero_norm_indices]

    if data_class == data_LinearRegression:
        y_mean = np.mean(y)
        y = y - y_mean
        y = y / np.std(y)

    print(f"norm of each column: {np.linalg.norm(X, axis=0)}")
    if data_class == data_LogisticRegression:
        print(f"y: {y}")
        print(f"unique y: {np.unique(y)}")
        print(f"count of y -1: {np.sum(y == -1)}, count of y 1: {np.sum(y == 1)}")

    metrics_to_plot = {}
    if data_class == data_LogisticRegression:
        problem_type = "classification"
        metrics_to_plot["Logistic Loss"] = compute_metric_logistic
        metrics_to_plot["Accuracy"] = compute_metric_acc
        metrics_to_plot["AUC"] = compute_metric_auc
    elif data_class == data_LinearRegression:
        problem_type = "regression"
        metrics_to_plot["Squared Error"] = compute_metric_SE
    else:
        raise ValueError("Unsupported data_class for beamsearch CV experiment.")

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    all_train_metrics = {name: [] for name in metrics_to_plot.keys()}
    all_test_metrics = {name: [] for name in metrics_to_plot.keys()}
    all_beta_abs_max = []

    if data_class == data_LogisticRegression:
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    else:
        skf = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        if data_class == data_LogisticRegression:
            # Balance only the training split; keep the test split imbalanced.
            X_train, y_train = generate_more_balanced_data(X_train, y_train, seed=random_state)

        optimizer = BeamSearchOptimizer(
            X_train,
            y_train,
            lambda2,
            M,
            data_class,
            parent_size=parent_size,
            child_size=child_size,
        )
        time_start = time.time()
        optimizer.get_sparse_sol_via_OMP_path(k)
        print(f"time for beam search: {time.time() - time_start}")
        time_start = time.time()
        beta_path = optimizer.get_beta_path()

        current_fold_train_metrics = {name: [] for name in metrics_to_plot.keys()}
        current_fold_test_metrics = {name: [] for name in metrics_to_plot.keys()}

        beta_abs_max = []
        for l in range(len(beta_path)):
            beta = beta_path[l]
            beta_abs_max.append(max(np.abs(beta)))
            y_pred_train = X_train @ beta
            y_pred_test = X_test @ beta

            for metric_name, metric_func in metrics_to_plot.items():
                train_metric = metric_func(y_pred_train, y_train)
                test_metric = metric_func(y_pred_test, y_test)
                current_fold_train_metrics[metric_name].append(train_metric)
                current_fold_test_metrics[metric_name].append(test_metric)

        for metric_name in metrics_to_plot.keys():
            all_train_metrics[metric_name].append(current_fold_train_metrics[metric_name])
            all_test_metrics[metric_name].append(current_fold_test_metrics[metric_name])

        print(f"beta_abs_max: {np.asarray(beta_abs_max)}")
        all_beta_abs_max.append(np.asarray(beta_abs_max))
        print(f"Fold {i + 1}/{n_fold} finished")
        print(f"time for all other evaluation tasks: {time.time() - time_start}")

    for metric_name in metrics_to_plot.keys():
        all_train_metrics[metric_name] = np.vstack(all_train_metrics[metric_name])
        all_test_metrics[metric_name] = np.vstack(all_test_metrics[metric_name])

    support_size_path = np.arange(1, len(beta_path) + 1)

    return {
        "support_size_path": support_size_path,
        "train_metrics": all_train_metrics,
        "test_metrics": all_test_metrics,
        "beta_abs_max": all_beta_abs_max,
        "problem_type": problem_type,
        "dataset_name": openML_dataset_name,
        "lambda2": lambda2,
        "M": M,
        "k": k,
    }


def run_BeamSearchOptimizer():
    n = 6000
    p = 6000
    k = 10
    rho = 0.9
    coeff_val = 1.0
    snr = 5.0
    M = 2
    lambda2 = 1e-2
    
    X, y, true_beta = generate_linear_regression_data(m=n, n=p, rho=rho, k=k, coeff_val=coeff_val, snr=snr)
    data_class = data_LinearRegression

    X, y, true_beta = generate_logistic_regression_data(m=n, n=p, rho=rho, k=k, coeff_val=coeff_val)
    yX = y.reshape(-1, 1) * X
    data_class = data_LogisticRegression

    optimizer = BeamSearchOptimizer(X, y, lambda2, M, data_class)
    start_time = time.time()
    optimizer.get_sparse_sol_via_OMP(k)
    solver_time = time.time() - start_time
    print(f"solver_time: {solver_time}")

    print(f"there are {len(optimizer.saved_solution)} saved solutions")

    estimated_beta = optimizer.get_beta()
    nonzero_indices = np.where(np.abs(estimated_beta) > 1e-6)[0]
    print(f"nonzero indices: {nonzero_indices}, with beta values: {estimated_beta[nonzero_indices]}")


    X_sub = X[:, nonzero_indices]
    yX_sub = yX[:, nonzero_indices]
    bounds = [(-M, M) for _ in range(len(nonzero_indices))]  # Box constraints for each beta_j

    def loss_LinearRegression_func(beta):
        return np.sum((X_sub @ beta - y) ** 2) + lambda2 * np.sum(beta ** 2)
    
    def loss_LogisticRegression_func(beta):
        return np.sum(np.log1p(1. / np.exp(yX_sub @ beta))) + lambda2 * np.sum(beta ** 2)

    beta_scipy = np.random.randn(len(nonzero_indices))
    # scipy_optimizer = minimize(loss_LinearRegression_func, beta_scipy, bounds=bounds)
    scipy_optimizer = minimize(loss_LogisticRegression_func, beta_scipy, bounds=bounds)

    print("scipy loss:", scipy_optimizer.fun)
    print("scipy beta:", scipy_optimizer.x)



if __name__ == "__main__":
    # run_GLMOptimizer_finetune()
    # run_BeamSearchOptimizer()
    pass
