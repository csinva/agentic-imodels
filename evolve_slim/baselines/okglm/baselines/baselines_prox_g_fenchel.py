import numpy as np
import scipy
import time
import sys
import os

if os.environ.get("my_gurobi_license_path"):  # evolve_slim: optional license paths
    os.environ["GRB_LICENSE_FILE"] = os.environ["my_gurobi_license_path"]
if os.environ.get("my_mosek_license_path"):  # evolve_slim: optional license paths
    os.environ["MOSEKLM_LICENSE_FILE"] = os.environ["my_mosek_license_path"]

import gurobipy as gp
import mosek
import mosek.fusion as msk

import cvxpy

from okglm.mosek_helpers import mosek_solve_and_retrieve_variable

# want to find \argmin_{alpha} 1/ 2 ||alpha - mu||^2 + rho TopSum_k(Huber_M(alpha)), where
# Huber_M(alpha)_j = 1/2 alpha_j^2 if |alpha_j| <= M, and M |alpha_j| - 1/2 M^2 otherwise
# TopSum_k(Huber_M(alpha)) = \sum_{j=1}^k Huber_M(alpha)_[j]
# Huber_M(alpha)_[j] is the j-th largest element of Huber_M(alpha)

def prox_kyFan_HuberLoss_gurobi(mu, k, rho, M=1e6, verbose=False):
    p = len(mu)
    model = gp.Model("prox_kyFan_HuberLoss_gurobi")

    # prox variable
    alpha = model.addMVar(shape=p, name="alpha", lb=-gp.GRB.INFINITY)

    # Huber variables
    t = model.addMVar(shape=p, name="t", lb=-gp.GRB.INFINITY)
    v = model.addMVar(shape=p, name="v", lb=-gp.GRB.INFINITY)
    v_abs = model.addMVar(shape=p, name="v_abs", lb=-gp.GRB.INFINITY)

    # top_sum_k variables
    b = model.addVar(name="t_kth_largest", lb=-gp.GRB.INFINITY)
    max_t_minus_b_and_0 = model.addMVar(shape=p, name="max_t_minus_b_and_0", lb=0)

    # add constraints
    model.addConstr(v_abs >= v)
    model.addConstr(v_abs >= -v)
    model.addConstr(t >= 1/2 * (v - alpha) * (v - alpha) + M * v_abs)
    model.addConstr(max_t_minus_b_and_0 >= t - b)

    TopSum_k_t = k * b + gp.quicksum(max_t_minus_b_and_0)

    loss = 1/2 * (alpha - mu) @ (alpha - mu) + rho * TopSum_k_t
    model.setObjective(loss, gp.GRB.MINIMIZE)

    model.setParam(gp.GRB.Param.OutputFlag, verbose)
    model.optimize()

    return alpha.X

def prox_kyFan_HuberLoss_mosek(mu, k, rho, M=1e6, verbose=False):
    p = len(mu)
    
    model = msk.Model("prox_kyFan_HuberLoss_mosek")
    alpha = model.variable("alpha", p, msk.Domain.unbounded())
    t = model.variable("t", p, msk.Domain.greaterThan(0.0))
    v = model.variable("v", p, msk.Domain.unbounded())
    v_abs = model.variable("v_abs", p, msk.Domain.greaterThan(0.0))
    b = model.variable("b", 1, msk.Domain.unbounded())
    max_t_minus_b_and_0 = model.variable("max_t_minus_b_and_0", p, msk.Domain.greaterThan(0.0))
    first_term = model.variable("first_term", 1, msk.Domain.unbounded())

    model.constraint("v_abs >= v", msk.Expr.sub(v_abs, v), msk.Domain.greaterThan(0.0))
    model.constraint("v_abs >= -v", msk.Expr.add(v_abs, v), msk.Domain.greaterThan(0.0))
    model.constraint("t >= 1/2 * (v - alpha) * (v - alpha) + M * v_abs", msk.Expr.hstack(msk.Expr.constTerm(p, 1.0), msk.Expr.sub(t, msk.Expr.mul(M, v_abs)), msk.Expr.sub(v, alpha)), msk.Domain.inRotatedQCone())
    model.constraint("max_t_minus_b_and_0 >= t - b", msk.Expr.sub(max_t_minus_b_and_0, msk.Expr.sub(t, msk.Var.repeat(b, p))), msk.Domain.greaterThan(0.0))
    model.constraint("first_term >= 1/2 * ||alpha - mu||^2", msk.Expr.vstack(first_term, 1, msk.Expr.sub(alpha, mu)), msk.Domain.inRotatedQCone())

    TopSum_k_t = msk.Expr.add(msk.Expr.mul(k, b), msk.Expr.sum(max_t_minus_b_and_0))

    loss = msk.Expr.add(first_term, msk.Expr.mul(rho, TopSum_k_t))

    model.objective(msk.ObjectiveSense.Minimize, loss)

    model.setLogHandler(sys.stdout if verbose else None)
    model.solve()

    return alpha.level()

def prox_kyFan_HuberLoss_cvxpy_formulation(mu, k, rho, M=1e6):
    p = len(mu)
    alpha = cvxpy.Variable(p)
    t = cvxpy.Variable(p)
    v = cvxpy.Variable(p)
    v_abs = cvxpy.Variable(p)
    b = cvxpy.Variable()
    max_t_minus_b_and_0 = cvxpy.Variable(p)

    variables = {"alpha": alpha, "t": t, "v": v, "v_abs": v_abs, "b": b, "max_t_minus_b_and_0": max_t_minus_b_and_0}

    constraints = [
        # t >= 0,
        # v_abs >= 0,
        max_t_minus_b_and_0 >= 0,
        v_abs >= v,
        v_abs >= -v,
        t >= 1/2 * cvxpy.square(v - alpha) + M * v_abs,
        max_t_minus_b_and_0 >= t - b,
    ]

    TopSum_k_t = k * b + cvxpy.sum(max_t_minus_b_and_0)

    loss = 1/2 * cvxpy.sum_squares(alpha - mu) + rho * TopSum_k_t
    objective = cvxpy.Minimize(loss)
    problem = cvxpy.Problem(objective, constraints)

    return variables, problem

class prox_kyFan_HuberLoss_cvxpy_optimizer:
    def __init__(self, mu, k, rho, M):
        self.variables, self.problem = prox_kyFan_HuberLoss_cvxpy_formulation(mu, k, rho, M)
    
    def solve(self, solverName, verbose=False, timeLimit=300):
        if solverName == "gurobi":
            gurobi_env = gp.Env()
            gurobi_env.setParam("TimeLimit", timeLimit)
            self.problem.solve(solver=cvxpy.GUROBI, verbose=verbose, env=gurobi_env)
        elif solverName == "mosek":
            mosek_params={
                'MSK_DPAR_OPTIMIZER_MAX_TIME': timeLimit,
            }
            self.problem.solve(solver=cvxpy.MOSEK, mosek_params=mosek_params, verbose=verbose)
        elif solverName == "clarabel":
            clarabel_settings = {
                'time_limit': timeLimit
            }
            self.problem.solve(solver=cvxpy.CLARABEL, verbose=verbose,  **clarabel_settings)
        elif solverName == "scs":
            self.problem.solve(solver=cvxpy.SCS, verbose=verbose, time_limit_secs=timeLimit)
        else:
            raise ValueError("Invalid solverName")
    
    def get_solution(self):
        return self.variables["alpha"].value
    
    def get_solver_time(self):
        return self.problem.solver_stats.solve_time
