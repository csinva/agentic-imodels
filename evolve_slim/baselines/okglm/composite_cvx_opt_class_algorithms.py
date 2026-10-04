import os
import time
import math

from okglm.helpers.general_helpers import _get_array_module, to_cpu, to_gpu
from okglm.composite_cvx_opt_class_data import (
    base_data_GLM_gpu,
    data_LinearRegression_gpu,
    data_LogisticRegression_gpu,
    data_PoissonRegression_gpu,
)
from okglm.composite_cvx_opt_class_regularizers import (
    base_regularizer_gpu,
    regularizer_fenchel_kyFan_HuberLoss_gpu,
    regularizer_fenchel_kyFan_HuberLoss_BnB_gpu,
    regularizer_l1Regularized_gpu,
    regularizer_l1Constrained_gpu,
)

if os.environ.get("my_gurobi_license_path"):  # evolve_slim: optional license paths
    os.environ["GRB_LICENSE_FILE"] = os.environ["my_gurobi_license_path"]
if os.environ.get("my_mosek_license_path"):  # evolve_slim: optional license paths
    os.environ["MOSEKLM_LICENSE_FILE"] = os.environ["my_mosek_license_path"]


class PGD_optimizer:
    """
    A class for the Proximal Gradient Descent (PGD) optimizer to solve the unconstrained convex composite optimization problem for generalized linear models (GLM) with the following form:
    minimize F(beta) + G(beta), where
        F(beta) = F_y(X^T beta) = f(X^T beta, y),
        G(beta) = 2\lambda_2 g(beta), where
            f is the GLM loss function,
            g(beta) = \min_z \sum_{j=1}^p beta_j^2 / z_j, s.t. 0 <= z_j <= 1, \sum_{j=1}^p z_j <= k, |beta_j| <= M * z_j
    """
    
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, reg_params=None, use_gpu=False):
        self.n, self.p = X.shape
        self.k = k
        self.lambda2 = lambda2
        self.M = M
        self.twoLambda2 = 2 * lambda2
        self.use_gpu = use_gpu
        self.xp = _get_array_module(use_gpu)

        self.data_class = data_class(X, y, L, use_gpu=use_gpu)
        self.L = self.data_class.L
        self.twoLambda2_over_L = self.twoLambda2 / self.L

        self.reg_params = {"lambda2": lambda2, "k": k, "M": M}
        if reg_params:
            self.reg_params.update(reg_params)
        self.reg_class = reg_class(self.data_class.X, use_gpu=use_gpu, **self.reg_params)

        self.solution = None
        self.solver_time = None
        self.solution_status = "not converged; reached max_iter"

        self.beta = self.xp.zeros(self.p)

        self.postprocess_freq = postprocess_freq
        self.verbose = verbose
    
    def reset_beta(self, beta):
        beta_copy = beta.copy()
        self.beta = to_gpu(beta_copy) if self.use_gpu else to_cpu(beta_copy)

    def get_solution_status(self):
        return self.solution_status

    def get_grad_F(self, beta):
        return self.data_class.get_grad_F(beta)

    def get_F(self, beta):
        return self.data_class.get_F(beta)

    def get_zeta_from_beta(self, beta):
        zeta = self.data_class.get_zeta_from_beta(beta)
        return self.reg_class.get_feasible_zeta(zeta)
        
    def get_F_fenchel(self, zeta):
        return self.data_class.get_F_fenchel(zeta)
    
    def take_gradient_step(self, beta, grad, step_size):
        return beta - step_size * grad

    def get_G(self, beta):
        # # Move to CPU for proximal operations
        # beta_cpu = to_cpu(beta)
        # value1 = compute_fenchel_kyFan_HuberLoss(beta_cpu, self.k, self.M)
        # return self.twoLambda2 * value1
        return self.reg_class.get_G(beta)

    def get_G_fenchel(self, zeta):
        # # Handle GPU operations for matrix multiply
        # result = self.data_class.X.T @ zeta / self.twoLambda2
        # return self.twoLambda2 * compute_kyFan_HuberLoss_gpu(result, self.k, self.M)
        return self.reg_class.get_G_fenchel(zeta)
    
    def get_prox_G(self, beta, stepsize):
        return self.reg_class.get_prox_G(beta, stepsize)

    def get_primal_loss(self, beta):
        return self.get_F(beta) + self.get_G(beta)

    def get_dual_loss(self, beta):
        zeta = self.get_zeta_from_beta(beta)
        return -self.get_F_fenchel(-zeta) - self.get_G_fenchel(zeta)
    
    def get_primal_dual_losses_and_gap(self, beta):
        primal_loss = self.get_primal_loss(beta)
        dual_loss = self.get_dual_loss(beta)
        primal_dual_diff = primal_loss - dual_loss
        optimality_gap = (primal_loss - dual_loss) / (1e-12 + abs(dual_loss))
        return primal_loss, dual_loss, primal_dual_diff, optimality_gap
    
    def print_primal_dual_losses_and_gap(self, primal_loss, dual_loss, optimality_gap, iter):
        print(f"iter: {iter}, primal_loss: {primal_loss}, dual_loss: {dual_loss}, optimality_gap: {optimality_gap}")
    
    def get_solution(self):
        return self.solution
    
    def get_solver_time(self):
        return self.solver_time
    
    def solve(self, num_iter=1000, tol=1e-6, timeLimit=1800):
        self._initialize_solve(num_iter, tol, timeLimit)

        early_stopping = False
        for iter in range(0, self.num_iter):

            self._optimize_in_one_iteration()

            early_stopping = self.postprocess_after_one_iteration(iter)

            if early_stopping:
                break

        self.solution = self.beta
        self.solver_time = time.time() - self.time_start

    def _initialize_solve(self, num_iter, tol, timeLimit):
        self.time_start = time.time()

        self.num_iter = num_iter
        self.tol = tol
        self.timeLimit = timeLimit

        self.internal_iter = 0

    def _optimize_in_one_iteration(self):
        # gradient descent step
        grad_beta = self.get_grad_F(self.beta)
        self.beta = self.take_gradient_step(self.beta, grad_beta, 1 / self.L)
        # proximal step
        self.beta = self.get_prox_G(self.beta, 1 / self.L)
        self.internal_iter += 1

    def postprocess_after_one_iteration(self, iter):
        early_stopping = False

        if self.check_time_limit(iter):
            early_stopping = True
            return early_stopping

        if self.internal_iter % self.postprocess_freq != 0:
            return early_stopping

        primal_loss, dual_loss, primal_dual_diff, optimality_gap = self.get_primal_dual_losses_and_gap(self.beta)

        if (self.verbose):
            self.print_primal_dual_losses_and_gap(primal_loss, dual_loss, optimality_gap, iter)
        
        if self.check_optimality_gap_convergence(optimality_gap):
            early_stopping = True
            return early_stopping

        self._restart(primal_loss, dual_loss, primal_dual_diff)

        return early_stopping

    def check_time_limit(self, iter):
        if time.time() - self.time_start > self.timeLimit:
            print(f"Time limit {self.timeLimit}s exceeded at iteration {iter}")
            self.solution_status = "not converged; reached time limit"
            return True
        return False
    
    def check_optimality_gap_convergence(self, optimality_gap):
        if optimality_gap < self.tol:
            self.solution_status = "converged"
            return True
        return False

    def _restart(self, primal_loss, dual_loss, primal_dual_diff):
        pass

class ACFGM_optimizer(PGD_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, seed=0, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, use_gpu=use_gpu)
        self.rho = (1 - math.sqrt(6)/3)
        self.L_init_in_find_eta_1 = None
        self.rng = self.xp.random.RandomState(seed)

    def get_tau(self, iter):
        if iter == 1:
            return 0
        elif iter >= 2:
            return iter / 2
        else:
            raise ValueError("Invalid iteration")
    
    def get_rho(self, iter):
        if iter == 1:
            return 0
        elif iter >= 2:
            return self.rho
        else:
            raise ValueError("Invalid iteration")
    
    def find_eta_1(self, beta_init, r=1.1):
        grad_beta_init = self.get_grad_F(beta_init)
        print("before linesearch, self.L_init_in_find_eta_1:", self.L_init_in_find_eta_1)
        if self.L_init_in_find_eta_1 is None:
            beta_init2 = self.rng.normal(scale=1e-6, size=beta_init.shape) + beta_init
            grad_beta_init2 = self.get_grad_F(beta_init2)
            self.L_init_in_find_eta_1 = float(self.xp.linalg.norm(grad_beta_init2 - grad_beta_init) / self.xp.linalg.norm(beta_init2 - beta_init))

        # beta_init2 = beta_init + self.rng.normal(scale=1e-6, size=beta_init.shape)
        # grad_beta_init2 = self.get_grad_F(beta_init2)
        # self.L_init_in_find_eta_1 = float(cp.linalg.norm(grad_beta_init2 - grad_beta_init) / cp.linalg.norm(beta_init2 - beta_init))
        
        iter = 0
        primal_loss_init = self.get_primal_loss(beta_init)
        print("before linesearch, primal_loss_init:", primal_loss_init)

        grad_beta_init = self.get_grad_F(beta_init)
        while True:
            ################ implementation from paper
            eta_iter = 1 / (4 * (1 - self.rho) * self.L_init_in_find_eta_1 * (r ** iter) )
            beta_intermediate = self.take_gradient_step(beta_init, grad_beta_init, eta_iter)
            beta_iter = self.get_prox_G(beta_intermediate, eta_iter)
            grad_beta_iter = self.get_grad_F(beta_iter)
            L_iter = float(self.xp.linalg.norm(grad_beta_iter - grad_beta_init) / self.xp.linalg.norm(beta_iter - beta_init))
            primal_loss_iter = self.get_primal_loss(beta_iter)
            print(f"during linesearch, eta_iter at iter {iter}: {eta_iter}")
            print(f"during linesearch, primal_loss_iter at iter {iter}: {primal_loss_iter}")
            print()

            if eta_iter < 2. / (5 * L_iter):
                # self.L_init_in_find_eta_1 = None
                self.L_init_in_find_eta_1 = L_iter
                print(f"after linesearch, eta_iter is {to_cpu(eta_iter)}, L_iter is {to_cpu(L_iter)}, self.L is {self.L}, 5/2 * L_iter - 1 /  eta_iter is {to_cpu(5/2 * L_iter - 1 / eta_iter)}")
                return eta_iter
            
            # ################ implementation from the author's github code
            # L_iter = r ** iter * self.L_init_in_find_eta_1 / 4
            # eta_iter = 1 / 2.5 / L_iter
            # beta_intermediate = beta_init - eta_iter * self.get_grad_F(beta_init)
            # beta_iter = prox_fenchel_kyFan_HuberLoss_pava_gpu(beta_intermediate, self.k, self.twoLambda2 * eta_iter, self.M)
            # grad_beta_iter = self.get_grad_F(beta_iter)

            # tmp1 = cp.linalg.norm(grad_beta_iter-grad_beta_init)**2/2/L_iter
            # tmp2 = L_iter * (cp.linalg.norm(beta_iter-beta_init))**2 / 2
            # if tmp1 <= tmp2:
            #     self.L_init_in_find_eta_1 = L_iter
            #     return eta_iter

            iter += 1

    def get_eta(self, iter, beta_init, eta_prev=None, L_prev=None):
        if iter == 1:
            return self.find_eta_1(beta_init)
        elif iter == 2:
            if L_prev == 0:
                return (1 - self.rho) * eta_prev
            return min((1 - self.rho) * eta_prev, 1. / (4. * L_prev))
        elif iter == 3:
            if L_prev == 0:
                return eta_prev
            return min(eta_prev, 1. / (4. * L_prev))
        elif iter >= 4:
            if L_prev == 0:
                return iter / (iter-1) * eta_prev
            return min(iter / (iter-1) * eta_prev, (iter-1) / (8 * L_prev))
        else:
            raise ValueError("Invalid iteration")
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)
        self.alpha = self.beta.copy()
        self.gamma = self.beta.copy()

        self.eta_prev = None
        self.L_prev = None

        self.grad_beta_iter = self.get_grad_F(self.beta)
    
    def _optimize_in_one_iteration(self):
        tau_iter_next = self.get_tau(self.internal_iter + 1)
        rho_iter_next = self.get_rho(self.internal_iter + 1)
        eta_iter_next = self.get_eta(self.internal_iter + 1, self.beta, eta_prev=self.eta_prev, L_prev=self.L_prev)
        eta_iter_next = float(eta_iter_next)  # Ensure Python scalar

        intermediate = self.take_gradient_step(self.alpha, self.grad_beta_iter, eta_iter_next)
        self.gamma = self.get_prox_G(intermediate, eta_iter_next)
        self.alpha = (1 - rho_iter_next) * self.alpha + rho_iter_next * self.gamma
        beta_new = tau_iter_next / (1 + tau_iter_next) * self.beta + 1 / (1 + tau_iter_next) * self.gamma

        grad_beta_iter_next = self.get_grad_F(beta_new)

        numerator = self.get_F(beta_new) - self.get_F(self.beta) + grad_beta_iter_next @ (self.beta - beta_new)

        if numerator >= 0:
            self.L_prev = 0
        else:
            self.L_prev = - 1. / 2 * float(self.xp.linalg.norm(grad_beta_iter_next - self.grad_beta_iter)**2) / numerator

        if self.internal_iter == 0:
            self.L_prev = float(self.xp.linalg.norm(grad_beta_iter_next - self.grad_beta_iter) / self.xp.linalg.norm(beta_new - self.beta))

        self.eta_prev = eta_iter_next
        self.beta = beta_new.copy()
        self.grad_beta_iter = grad_beta_iter_next.copy()

        self.internal_iter += 1

class PD_Restarted_ACFGM_optimizer(ACFGM_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, seed=0, restart_exponent=1, dynamic_restart=False, restart_check_freq=1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, seed, use_gpu=use_gpu)
        self.restart_exponent = restart_exponent
        self.dynamic_restart = dynamic_restart
        self.postprocess_freq = restart_check_freq # override postprocess_freq to restart_check_freq
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)

        primal_loss, dual_loss, primal_dual_diff_init, optimality_gap = self.get_primal_dual_losses_and_gap(self.beta)
        self.primal_dual_diff_init = primal_dual_diff_init
        if self.verbose:
            print(f"Initial primal-dual difference: {primal_dual_diff_init}")
    
    def _restart(self, primal_loss, dual_loss, primal_dual_diff):
        if primal_dual_diff / self.primal_dual_diff_init >= 1. / math.e**self.restart_exponent:
            if self.dynamic_restart:
                self.postprocess_freq *= 2
            return
        if self.verbose:
            print(f"Restart!!! Primal-dual difference reduced by a factor of e^{self.restart_exponent} at internal iteration {self.internal_iter}")

        self.internal_iter = 0
        self.primal_dual_diff_init = primal_dual_diff
        self.alpha = self.beta.copy()
        self.gamma = self.beta.copy()
        self.L_prev = None
        self.eta_prev = None

class ACFGM2_optimizer(ACFGM_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, seed=0, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, seed, use_gpu=use_gpu)

        self.c = 0.1
    
    def get_tau(self, iter, tau_prev=None, L_prev=None, eta=None):
        if iter == 1:
            return 0
        elif iter == 2:
            return 1
        elif iter >= 3:
            return tau_prev + self.c / 2 + 2 * (1 - self.c) * eta * L_prev / tau_prev
        else:
            raise ValueError("Invalid iteration")

    def get_eta(self, iter, beta_init, tau_prev=None, tau_prev_prev=None, eta_prev=None, L_prev=None):
        if iter == 1:
            return self.find_eta_1(beta_init)
        elif iter == 2:
            if L_prev == 0:
                return (1 - self.rho) * eta_prev
            return min((1 - self.rho) * eta_prev, 1. / (4. * L_prev))
        elif iter >= 3:
            if L_prev == 0:
                return min(4/3*eta_prev, (tau_prev_prev + 1) / tau_prev * eta_prev)
            return min(4/3*eta_prev, (tau_prev_prev + 1) / tau_prev * eta_prev, tau_prev / (4 * L_prev))
        else:
            raise ValueError("Invalid iteration")
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)
    
        self.tau_prev = None
        self.tau_prev_prev = None

    def _optimize_in_one_iteration(self):
        rho_iter_next = self.get_rho(self.internal_iter + 1)
        eta_iter_next = self.get_eta(self.internal_iter + 1, self.beta, tau_prev=self.tau_prev, tau_prev_prev=self.tau_prev_prev, eta_prev=self.eta_prev, L_prev=self.L_prev)
        eta_iter_next = float(eta_iter_next)  # Ensure Python scalar
        tau_iter_next = self.get_tau(self.internal_iter + 1, tau_prev=self.tau_prev, L_prev=self.L_prev, eta=eta_iter_next)

        intermediate = self.take_gradient_step(self.alpha, self.grad_beta_iter, eta_iter_next)
        self.gamma = self.get_prox_G(intermediate, eta_iter_next)
        self.alpha = (1 - rho_iter_next) * self.alpha + rho_iter_next * self.gamma
        beta_new = tau_iter_next / (1 + tau_iter_next) * self.beta + 1 / (1 + tau_iter_next) * self.gamma

        grad_beta_iter_next = self.get_grad_F(beta_new)

        numerator = self.get_F(beta_new) - self.get_F(self.beta) + grad_beta_iter_next @ (self.beta - beta_new)

        if numerator >= 0:
            self.L_prev = 0
        else:
            self.L_prev = - 1. / 2 * float(self.xp.linalg.norm(grad_beta_iter_next - self.grad_beta_iter)**2) / numerator
        if self.internal_iter == 0:
            self.L_prev = float(self.xp.linalg.norm(grad_beta_iter_next - self.grad_beta_iter) / self.xp.linalg.norm(beta_new - self.beta))

        self.eta_prev = eta_iter_next
        self.beta = beta_new.copy()
        self.grad_beta_iter = grad_beta_iter_next.copy()

        self.tau_prev_prev = self.tau_prev
        self.tau_prev = tau_iter_next

        self.internal_iter += 1

class PD_Restarted_ACFGM2_optimizer(ACFGM2_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, seed=0, restart_exponent=1, dynamic_restart=False, restart_check_freq=1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose,  postprocess_freq, seed, use_gpu=use_gpu)
        self.restart_exponent = restart_exponent
        self.dynamic_restart = dynamic_restart
        self.postprocess_freq = restart_check_freq # override postprocess_freq to restart_check_freq
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)

        primal_loss, dual_loss, primal_dual_diff_init, optimality_gap = self.get_primal_dual_losses_and_gap(self.beta)
        self.primal_dual_diff_init = primal_dual_diff_init
        if self.verbose:
            print(f"Initial primal-dual difference: {primal_dual_diff_init}")

    def _restart(self, primal_loss, dual_loss, primal_dual_diff):
        if primal_dual_diff / self.primal_dual_diff_init >= 1. / math.e**self.restart_exponent:
            if self.dynamic_restart:
                self.postprocess_freq *= 2
            return
        if self.verbose:
            print(f"Restart!!! Primal-dual difference reduced by a factor of e^{self.restart_exponent} at internal iteration {self.internal_iter}")

        self.internal_iter = 0
        self.primal_dual_diff_init = primal_dual_diff
        self.alpha = self.beta.copy()
        self.gamma = self.beta.copy()
        self.L_prev = None
        self.eta_prev = None
        self.tau_prev = None
        self.tau_prev_prev = None

class Beck_FISTA_optimizer(PGD_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, use_gpu=use_gpu)
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)
    
        self.beta_old = self.beta.copy()
        self.gamma = self.beta.copy()

        self.rho_curr = 1.0
        self.rho_prev = 1.0
    
    def _optimize_in_one_iteration(self):
        # gradient descent step
        grad_gamma = self.get_grad_F(self.gamma)
        beta_intermediate = self.take_gradient_step(self.gamma, grad_gamma, 1 / self.L)
        self.beta_old[:] = self.beta[:]

        # proximal step
        self.beta = self.get_prox_G(beta_intermediate, 1 / self.L)

        # acceleration step
        self.rho_prev = self.rho_curr
        self.rho_curr = (1 + math.sqrt(1 + 4 * self.rho_prev ** 2)) / 2
        self.gamma = self.beta + (self.rho_prev - 1) / self.rho_curr * (self.beta - self.beta_old)

        self.internal_iter += 1

class Restarted_Beck_FISTA_optimizer(Beck_FISTA_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, use_gpu=use_gpu)

        if postprocess_freq != 1:
            raise ValueError("postprocess_freq must be 1 for Restarted_Beck_FISTA_optimizer")
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)
    
        primal_loss, dual_loss, primal_dual_diff, optimality_gap = self.get_primal_dual_losses_and_gap(self.beta)
        self.primal_loss_prev = primal_loss
        if self.verbose:
            print(f"primal loss: {primal_loss}")
        
    def _restart(self, primal_loss, dual_loss, primal_dual_diff):
        if primal_loss <= self.primal_loss_prev:
            self.primal_loss_prev = primal_loss
            return
        if self.verbose:
            print(f"Restart!!! Primal loss increased at internal iteration {self.internal_iter}")

        self.rho_curr = 1.0
        self.rho_prev = 1.0
        self.gamma = self.beta.copy()
        self.beta_old = self.beta.copy()
        self.primal_loss_prev = primal_loss
        self.internal_iter = 0

class PD_Restarted_Beck_FISTA_optimizer(Beck_FISTA_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, restart_exponent=1, dynamic_restart=False, restart_check_freq=1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, use_gpu=use_gpu)
        self.restart_exponent = restart_exponent
        self.dynamic_restart = dynamic_restart
        self.postprocess_freq = restart_check_freq # override postprocess_freq to restart_check_freq
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)

        primal_loss, dual_loss, primal_dual_diff_init, optimality_gap = self.get_primal_dual_losses_and_gap(self.beta)
        self.primal_dual_diff_init = primal_dual_diff_init
        if self.verbose:
            print(f"Initial primal-dual difference: {primal_dual_diff_init}")

    def _restart(self, primal_loss, dual_loss, primal_dual_diff):
        if primal_dual_diff / self.primal_dual_diff_init >= 1. / math.e**self.restart_exponent:
            if self.dynamic_restart:
                self.postprocess_freq *= 2
            return
        if self.verbose:
            print(f"Restart!!! Primal-dual difference reduced by a factor of e^{self.restart_exponent} at internal iteration {self.internal_iter}")

        self.internal_iter = 0
        self.primal_dual_diff_init = primal_dual_diff
        self.beta_old = self.beta.copy()
        self.gamma = self.beta.copy()
        self.rho_curr = 1.0
        self.rho_prev = 1.0

class Beck_FISTALineSearch_optimizer(PGD_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=-1.0, verbose=False, postprocess_freq=1, seed=0, linesearch_multiplier=1.1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, use_gpu=use_gpu)
        self.rng = self.xp.random.RandomState(seed)
        self.linesearch_multiplier = linesearch_multiplier

        if L == -1.0:
            self.L_initial = self.get_L_initial()
    
    def get_L_initial(self):
        beta2 = self.beta + self.rng.normal(scale=1e-6, size=self.p)
        grad_beta2 = self.get_grad_F(beta2)
        grad_beta = self.get_grad_F(self.beta)

        return float(self.xp.linalg.norm(grad_beta2 - grad_beta) / self.xp.linalg.norm(beta2 - self.beta))
    
    def perform_line_search(self, beta):
        L_tmp = self.L_initial / self.linesearch_multiplier

        grad_F_beta = self.get_grad_F(beta)
        beta_intermediate = self.take_gradient_step(beta, grad_F_beta, 1 / L_tmp)
        beta_new = self.get_prox_G(beta_intermediate, 1 / L_tmp)

        F_beta = self.get_F(beta)
        F_beta_new = self.get_F(beta_new)
        SurrogateF_beta_new = F_beta + float(grad_F_beta @ (beta_new - beta)) + L_tmp / 2 * float(self.xp.linalg.norm(beta_new - beta) ** 2)
        # print(f"before linesearch, F_beta: {F_beta}")
        # print(f"Initial L_tmp: {L_tmp}, F_beta_new: {F_beta_new}")

        line_search_iter = 0

        if F_beta_new > SurrogateF_beta_new:

            while F_beta_new >= SurrogateF_beta_new:
                L_tmp = L_tmp * self.linesearch_multiplier

                beta_intermediate = self.take_gradient_step(beta, grad_F_beta, 1 / L_tmp)
                beta_new = self.get_prox_G(beta_intermediate, 1 / L_tmp)

                F_beta_new = self.get_F(beta_new)
                SurrogateF_beta_new = F_beta + float(grad_F_beta @ (beta_new - beta)) + L_tmp / 2 * float(self.xp.linalg.norm(beta_new - beta) ** 2)
                line_search_iter += 1
            # print(f"Line search iteration {line_search_iter}: L_tmp = {L_tmp}, F_beta_new = {F_beta_new}")

            self.L_initial = L_tmp
            return float(self.L_initial)  # Convert to Python scalar
        else:

            while F_beta_new <= SurrogateF_beta_new:
                L_tmp = L_tmp / self.linesearch_multiplier

                beta_intermediate = self.take_gradient_step(beta, grad_F_beta, 1 / L_tmp)
                beta_new = self.get_prox_G(beta_intermediate, 1 / L_tmp)

                F_beta_new = self.get_F(beta_new)
                SurrogateF_beta_new = F_beta + float(grad_F_beta @ (beta_new - beta)) + L_tmp / 2 * float(self.xp.linalg.norm(beta_new - beta) ** 2)
                line_search_iter += 1
            
            # print(f"Line search iteration {line_search_iter}: L_tmp = {L_tmp * self.linesearch_multiplier}, F_beta_new = {F_beta_new}")

            self.L_initial = L_tmp * self.linesearch_multiplier
            return float(self.L_initial)  # Convert to Python scalar
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)

        self.beta_old = self.beta.copy()
        self.gamma = self.beta.copy()
        self.rho_curr = 1.0
        self.rho_prev = 1.0
    
    def _optimize_in_one_iteration(self):
        L_curr = self.perform_line_search(self.gamma)

        # gradient descent step
        grad_gamma = self.get_grad_F(self.gamma)
        beta_intermediate = self.take_gradient_step(self.gamma, grad_gamma, 1 / L_curr)
        self.beta_old[:] = self.beta[:]

        # proximal step
        self.beta = self.get_prox_G(beta_intermediate, 1 / L_curr)

        # acceleration step
        self.rho_prev = self.rho_curr
        self.rho_curr = (1 + math.sqrt(1 + 4 * self.rho_prev ** 2)) / 2
        self.gamma = self.beta + (self.rho_prev - 1) / self.rho_curr * (self.beta - self.beta_old)

        self.internal_iter += 1

class PD_Restarted_Beck_FISTALineSearch_optimizer(Beck_FISTALineSearch_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=-1.0, verbose=False, postprocess_freq=1, seed=0, linesearch_multiplier=1.1, restart_exponent=1, dynamic_restart=False, restart_check_freq=1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, seed, linesearch_multiplier, use_gpu=use_gpu)
        self.restart_exponent = restart_exponent
        self.dynamic_restart = dynamic_restart
        self.postprocess_freq = restart_check_freq # override postprocess_freq to restart_check_freq
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)

        primal_loss, dual_loss, primal_dual_diff_init, optimality_gap = self.get_primal_dual_losses_and_gap(self.beta)
        self.primal_dual_diff_init = primal_dual_diff_init
        if self.verbose:
            print(f"Initial primal-dual difference: {primal_dual_diff_init}")
    
    def _restart(self, primal_loss, dual_loss, primal_dual_diff):
        if primal_dual_diff / self.primal_dual_diff_init >= 1. / math.e**self.restart_exponent:
            if self.dynamic_restart:
                self.postprocess_freq *= 2
            return
        if self.verbose:
            print(f"Restart!!! Primal-dual difference reduced by a factor of e^{self.restart_exponent} at internal iteration {self.internal_iter}")

        self.internal_iter = 0
        self.primal_dual_diff_init = primal_dual_diff
        self.beta_old = self.beta.copy()
        self.gamma = self.beta.copy()
        self.rho_curr = 1.0
        self.rho_prev = 1.0
        self.L_initial = self.get_L_initial()

class FISTA_optimizer(PGD_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, use_gpu=use_gpu)
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)
        self.beta_old = self.beta.copy()
        self.internal_iter = 1
    
    def _optimize_in_one_iteration(self):
        beta_intermediate = self.beta + (self.internal_iter / (self.internal_iter + 3)) * (self.beta - self.beta_old)
        self.beta_old = self.beta.copy()

        # gradient descent step
        grad_beta_intermediate = self.get_grad_F(beta_intermediate)
        beta_intermediate = self.take_gradient_step(beta_intermediate, grad_beta_intermediate, 1 / self.L)

        self.beta = self.get_prox_G(beta_intermediate, 1 / self.L)
        self.internal_iter += 1

class Restarted_FISTA_optimizer(FISTA_optimizer):
    def __init__(self, X, y, k, lambda2, M, data_class, reg_class=regularizer_fenchel_kyFan_HuberLoss_gpu, L=1e8, verbose=False, postprocess_freq=1, use_gpu=False):
        super().__init__(X, y, k, lambda2, M, data_class, reg_class, L, verbose, postprocess_freq, use_gpu=use_gpu)

        if postprocess_freq != 1:
            raise ValueError("postprocess_freq must be 1 for Restarted_FISTA_optimizer")
    
    def _initialize_solve(self, num_iter, tol, timeLimit):
        super()._initialize_solve(num_iter, tol, timeLimit)
    
        primal_loss, dual_loss, primal_dual_diff, optimality_gap = self.get_primal_dual_losses_and_gap(self.beta)
        self.primal_loss_prev = primal_loss
        if self.verbose:
            print(f"primal loss: {primal_loss}")
    
    def _restart(self, primal_loss, dual_loss, primal_dual_diff):
        if primal_loss <= self.primal_loss_prev:
            self.primal_loss_prev = primal_loss
            return
        if self.verbose:
            print(f"Restarting! Primal loss increased at internal iteration {self.internal_iter}")

        self.internal_iter = 1
        self.beta_old = self.beta.copy()
        self.primal_loss_prev = primal_loss

