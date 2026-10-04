import os

if os.environ.get("my_gurobi_license_path"):  # evolve_slim: optional license paths
    os.environ["GRB_LICENSE_FILE"] = os.environ["my_gurobi_license_path"]
if os.environ.get("my_mosek_license_path"):  # evolve_slim: optional license paths
    os.environ["MOSEKLM_LICENSE_FILE"] = os.environ["my_mosek_license_path"]

import numpy as np
import scipy
import time
import sys
import gurobipy as gp
import mosek
import mosek.fusion as msk
import cvxpy

from okglm.mosek_helpers import mosek_solve_and_retrieve_variable


# want to find \argmin_{beta} 1/2 ||beta - x||_2^2 + rho * h^*(beta), where
# h^*(beta) is the Fenchel conjugate of h(beta),
# h(beta) = \sum_{j=1}^k H_M(beta_[j]),
# beta_[j] is the j-th largest element of beta (in absolute value),
# H_M(\alpha) = 1/2 * \alpha^2 if |\alpha| <= M, and H_M(\alpha) = M * |\alpha| - 1/2 * M^2 if |\alpha| > M

# this is equivalent to solving the following optimization problem:
# minimize 1/2 ||beta - x||_2^2 + rho * 1/2 * \sum_{j=1}^p beta_j^2 / z_j, s.t. 0 <= z_j <= 1, \sum_{j=1}^p z_j <= k, |beta_j| <= M * z_j



def prox_fenchel_kyFan_HuberLoss_gurobi(x, k, rho, M=1e6, verbose=False):

    p = len(x)
    model = gp.Model("prox_fenchel_kyFan_HuberLoss_gurobi")

    beta = model.addMVar(shape=p, name="beta", lb=-gp.GRB.INFINITY)
    s = model.addMVar(shape=p, name="s", lb=0)
    z = model.addMVar(shape=p, name="z", lb=0, ub=1)

    # beta_j^2 <= s_j * z_j
    model.addConstr(beta * beta <= s * z)
    # model.addConstr((s+z) * (s+z) >= (2 * beta) * (2 * beta) + (s - z) * (s - z))

    # \sum_{j=1}^p z_j <= k
    model.addConstr(z.sum() <= k)

    # |beta_j| <= M * z_j
    model.addConstr(beta <= M * z)
    model.addConstr(beta >= -M * z)

    # objective
    beta_minus_x = beta - x
    loss = 0.5 * beta_minus_x @ beta_minus_x + rho * 0.5 * gp.quicksum(s)
    model.setObjective(loss, gp.GRB.MINIMIZE)

    model.setParam("OutputFlag", 0)
    if verbose:
        model.setParam("OutputFlag", 1)

    model.optimize()

    return beta.X

def prox_fenchel_kyFan_HuberLoss_mosek(x, k, rho, M=1e6, verbose=False):

    model = msk.Model("prox_fenchel_kyFan_norm_2")
    beta = model.variable("beta", len(x), msk.Domain.unbounded())
    z = model.variable("z", len(x), msk.Domain.inRange(0, 1))
    s = model.variable("s", len(x), msk.Domain.greaterThan(0.0))
    t = model.variable("t", 1, msk.Domain.greaterThan(0.0))

    variables = {"beta": beta, "z": z, "s": s, "t": t}

    model.constraint(msk.Expr.hstack(s, z, beta), msk.Domain.inRotatedQCone()) # beta_j^2 <= 2 * s_j * z_j
    model.constraint(msk.Expr.sub(beta, msk.Expr.mul(M, z)), msk.Domain.lessThan(0))
    model.constraint(msk.Expr.add(beta, msk.Expr.mul(M, z)), msk.Domain.greaterThan(0))

    model.constraint(msk.Expr.sum(z), msk.Domain.lessThan(k))
    model.constraint(msk.Expr.vstack(t, 1.0, msk.Expr.sub(beta, x)), msk.Domain.inRotatedQCone()) # \sum_{j=1}^p (beta_j - x_j)^2 <= 2 * t

    loss = msk.Expr.mul(rho, msk.Expr.sum(s))
    # loss = msk.Expr.add(loss, msk.Expr.dot(-x, beta))
    loss = msk.Expr.add(loss, msk.Expr.mul(1, t))

    model.objective(msk.ObjectiveSense.Minimize, loss)
    if verbose:
        model.setLogHandler(sys.stdout)

    # model.setSolverParam("intpntCoTolRelGap", 1e-3)

    beta = mosek_solve_and_retrieve_variable(model, variables, "beta")
    return beta

def prox_fenchel_kyFan_HuberLoss_cvxpy_formulation(x, k, rho, M=1e6, verbose=False):
    
    p = len(x)
    beta = cvxpy.Variable(p, name="beta")
    z = cvxpy.Variable(p, name="z")
    s = cvxpy.Variable(p, name="s")

    variables = {"beta": beta, "z": z, "s": s}

    constraints = [
        cvxpy.SOC(t=s + z, X=cvxpy.vstack([2 * beta, s - z])),
        cvxpy.sum(z) <= k,
        beta <= M * z,
        beta >= -M * z,
        z <= 1,
        z >= 0,
        # s >= 0,
    ]

    loss = 0.5 * cvxpy.sum_squares(beta - x) + rho * 0.5 * cvxpy.sum(s)

    problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)

    return variables, problem

class prox_fenchel_kyFan_HuberLoss_cvxpy_optimizer:
    def __init__(self, mu, k, rho, M):
        self.variables, self.problem = prox_fenchel_kyFan_HuberLoss_cvxpy_formulation(mu, k, rho, M)
    
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
            self.problem.solve(solver=cvxpy.CLARABEL, verbose=verbose, **clarabel_settings)
        elif solverName == "scs":
            self.problem.solve(solver=cvxpy.SCS, verbose=verbose, time_limit_secs=timeLimit)
        else:
            raise ValueError("Invalid solverName")
    
    def get_solution(self):
        return self.variables["beta"].value
    
    def get_solver_time(self):
        return self.problem.solver_stats.solve_time
