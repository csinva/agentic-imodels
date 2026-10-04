from __future__ import annotations  # evolve_slim: gurobipy / mosek annotations without the packages
import os
import numpy as np
import sys
import scipy
import math
import time

# License envs are optional; only set if provided
_grb_path = os.environ.get("my_gurobi_license_path")
if _grb_path:
    os.environ["GRB_LICENSE_FILE"] = _grb_path
_msk_path = os.environ.get("my_mosek_license_path")
if _msk_path:
    os.environ["MOSEKLM_LICENSE_FILE"] = _msk_path

try:  # evolve_slim: commercial solvers serve only the reference relaxation solvers, not the BnB
    import gurobipy as gp
except ImportError:
    gp = None
try:
    import mosek.fusion as msk
except ImportError:
    msk = None
try:
    import cvxpy
except ImportError:
    cvxpy = None


def softplus(M, t, u):
    """
    Enforce t >= log(1 + exp(u)) componentwise using two exponential cone constraints.
    """
    n = t.getShape()[0]
    z1 = M.variable(n, msk.Domain.greaterThan(0.0))
    z2 = M.variable(n, msk.Domain.greaterThan(0.0))
    ones = msk.Expr.constTerm(n, 1.0)
    M.constraint(msk.Expr.add(z1, z2), msk.Domain.equalsTo(1.0))
    M.constraint(msk.Expr.hstack(z1, ones, msk.Expr.sub(u, t)), msk.Domain.inPExpCone())
    M.constraint(msk.Expr.hstack(z2, ones, msk.Expr.neg(t)), msk.Domain.inPExpCone())

class solver_base_class():
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        self.n, self.p = X.shape
        self.X = X
        self.y = y
        self.k = k
        self.lambda2 = lambda2
        self.M = M
        self.z_is_boolean = z_is_boolean
        self.relaxation_type = relaxation_type

        self.variables, self.problem = self.get_problem_formulation()

        self.solution = None
        self.solver_time = None
        self.solution_status = None
        self.objective_value = None
    
    def get_problem_formulation(self):
        raise NotImplementedError
    
    def solve(self, verbose=False, tol=1e-6, timeLimit=1800):
        raise NotImplementedError
    
    def get_solution(self):
        return self.solution
    
    def get_solver_time(self):
        return self.solver_time
    
    def get_solution_status(self):
        return self.solution_status

    def get_objective_value(self):
        return self.objective_value
    
class cvxpy_solver_base_class(solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        super().__init__(X, y, k, lambda2, M, z_is_boolean, relaxation_type)
    
    def get_variables_and_constraints(self):
        beta = cvxpy.Variable(self.p)

        if self.relaxation_type == "perspective":
            s = cvxpy.Variable(self.p)
            z = None
            if self.z_is_boolean:
                z = cvxpy.Variable(self.p, boolean=True)
            else:
                z = cvxpy.Variable(self.p)

            variables = {
                "beta": beta,
                "s": s,
                "z": z
            }

            constraints = [
                beta <= self.M * z,
                beta >= -self.M * z,
            ]
            if self.z_is_boolean is False:
                constraints += [
                    z >= 0,
                    z <= 1,
                ]
            constraints += [
                cvxpy.sum(z) <= self.k,
                cvxpy.SOC(t=s + z, X=cvxpy.vstack([2 * beta, s - z])),
            ]

            return variables, constraints
        
        elif self.relaxation_type == "l1":
            variables = {
                "beta": beta,
            }

            constraints =  [
                beta >= -self.M,
                beta <= self.M,
                cvxpy.sum(cvxpy.abs(beta)) <= self.k * self.M,
            ]
            return variables, constraints
        
        elif self.relaxation_type == "l1_no_constraint":
            variables = {
                "beta": beta,
            }

            constraints =  [ ]
            return variables, constraints
        
        else:
            raise ValueError("Invalid relaxation type")
    
    def get_problem_formulation(self):
        pass

    def solve(self, solverName="gurobi", verbose=False, tol=1e-6, timeLimit=1800, beta_warmstart=None):
        print("tol = ", tol)
        if solverName == "gurobi":
            gurobi_env = gp.Env()
            gurobi_env.setParam("BarQCPConvTol", tol)
            gurobi_env.setParam("TimeLimit", timeLimit)
            gurobi_env.setParam("MIPGap", tol)  # Set MIP gap tolerance
            self.problem.solve(solver=cvxpy.GUROBI, env=gurobi_env, verbose=verbose)
        elif solverName == "mosek":
            print("tol = ", tol)
            mosek_params = {
                'MSK_DPAR_OPTIMIZER_MAX_TIME': float(timeLimit),
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
            }
            if self.z_is_boolean:
                mosek_params['MSK_DPAR_MIO_MAX_TIME'] = float(timeLimit)
                mosek_params['MSK_DPAR_MIO_TOL_REL_GAP'] = tol  # Set MIP gap tolerance for Mosek
            self.problem.solve(solver=cvxpy.MOSEK, mosek_params=mosek_params, verbose=verbose)
        elif solverName == "clarabel":
            clarabel_settings = {
                'tol_gap_abs': tol,
                'tol_gap_rel': tol,
                'tol_feas': tol,
                'time_limit': timeLimit
            }
            self.problem.solve(solver=cvxpy.CLARABEL, verbose=verbose, **clarabel_settings)
        elif solverName == "scs":
            self.problem.solve(solver=cvxpy.SCS, eps=tol, verbose=verbose, time_limit_secs=timeLimit)
        elif solverName == "scsGPU":
            self.problem.solve(solver=cvxpy.SCS, gpu=True, use_indirect=True, eps=tol, verbose=verbose, time_limit_secs=timeLimit)
        else:
            raise ValueError("Invalid solverName")
        
        print(f"Solver training is complete; solution status: {self.problem.status}")

        self.solution = self.variables["beta"].value
        self.solver_time = self.problem.solver_stats.solve_time
        self.solution_status = self.problem.status
        self.objective_value = self.problem.value

class LinearRegression_cvxpy(cvxpy_solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        super().__init__(X, y, k, lambda2, M, z_is_boolean, relaxation_type)
    
    def get_problem_formulation(self):
        variables, constraints = self.get_variables_and_constraints()

        loss = None
        if self.relaxation_type == "perspective":
            loss = cvxpy.sum_squares(self.X @ variables["beta"] - self.y) + self.lambda2 * cvxpy.sum(variables["s"])
        elif self.relaxation_type == "l1":
            loss = cvxpy.sum_squares(self.X @ variables["beta"] - self.y) + self.lambda2 * cvxpy.sum(cvxpy.abs(variables["beta"]))
        elif self.relaxation_type == "l1_no_constraint":
            loss = cvxpy.sum_squares(self.X @ variables["beta"] - self.y) + self.lambda2 * cvxpy.sum(cvxpy.abs(variables["beta"]))
        else:
            raise ValueError("Invalid relaxation type")
        problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)

        return variables, problem

class LogisticRegression_cvxpy(cvxpy_solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        self.yX = y.reshape(-1, 1) * X
        super().__init__(X, y, k, lambda2, M, z_is_boolean, relaxation_type)
    
    def get_problem_formulation(self):
        variables, constraints = self.get_variables_and_constraints()

        loss = None
        if self.relaxation_type == "perspective":
            loss = cvxpy.sum(cvxpy.logistic(-self.yX @ variables["beta"])) + self.lambda2 * cvxpy.sum(variables["s"])
        elif self.relaxation_type == "l1":
            loss = cvxpy.sum(cvxpy.logistic(-self.yX @ variables["beta"])) + self.lambda2 * cvxpy.sum_squares(variables["beta"])
        else:
            raise ValueError("Invalid relaxation type")
        problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)

        return variables, problem

class PoissonRegression_cvxpy(cvxpy_solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        self.yX = y.reshape(-1, 1) * X
        super().__init__(X, y, k, lambda2, M, z_is_boolean, relaxation_type)
    
    def get_problem_formulation(self):
        variables, constraints = self.get_variables_and_constraints()

        loss = None
        if self.relaxation_type == "perspective":
            loss = cvxpy.sum(cvxpy.exp(self.X @ variables["beta"])) + cvxpy.sum(-self.yX @ variables["beta"]) + self.lambda2 * cvxpy.sum(variables["s"])
        elif self.relaxation_type == "l1":
            loss = cvxpy.sum(cvxpy.exp(self.X @ variables["beta"])) + cvxpy.sum(-self.yX @ variables["beta"]) + self.lambda2 * cvxpy.sum_squares(variables["beta"])
        else:
            raise ValueError("Invalid relaxation type")
        problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)

        return variables, problem
    
class LinearRegression_mosek(solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        super().__init__(X, y, k, lambda2, M, z_is_boolean=z_is_boolean, relaxation_type=relaxation_type)
    
    def get_problem_formulation(self):
        problem = msk.Model("Linear Regression")

        beta = problem.variable("beta", self.p, msk.Domain.unbounded())
        variables = {"beta": beta}

        # Residual norm via rotated cone
        t = problem.variable("t", 1, msk.Domain.greaterThan(0))
        problem.constraint(
            msk.Expr.vstack(t, 0.5, msk.Expr.sub(msk.Expr.mul(self.X, beta), self.y)),
            msk.Domain.inRotatedQCone(),
        )

        if self.relaxation_type == "perspective":
            s = problem.variable("s", self.p, msk.Domain.greaterThan(0))
            z_domain = msk.Domain.binary() if self.z_is_boolean else msk.Domain.inRange(0, 1)
            z = problem.variable("z", self.p, z_domain)
            variables.update({"s": s, "z": z, "t": t})

            # Vectorized per-coordinate rotated cones and box/cardinality constraints
            problem.constraint(msk.Expr.hstack(msk.Expr.mul(0.5, s), z, beta), msk.Domain.inRotatedQCone())
            problem.constraint(msk.Expr.sub(beta, msk.Expr.mul(self.M, z)), msk.Domain.lessThan(0))
            problem.constraint(msk.Expr.add(beta, msk.Expr.mul(self.M, z)), msk.Domain.greaterThan(0))
            problem.constraint(msk.Expr.sum(z), msk.Domain.lessThan(self.k))

            loss = msk.Expr.add(t, msk.Expr.mul(self.lambda2, msk.Expr.sum(s)))

        elif self.relaxation_type in ["l1", "l1_no_constraint"]:
            abs_beta = problem.variable("abs_beta", self.p, msk.Domain.greaterThan(0))
            variables.update({"abs_beta": abs_beta, "t": t})
            problem.constraint(msk.Expr.sub(abs_beta, beta), msk.Domain.greaterThan(0))
            problem.constraint(msk.Expr.add(abs_beta, beta), msk.Domain.greaterThan(0))

            if self.relaxation_type == "l1":
                problem.constraint(beta, msk.Domain.lessThan(self.M))
                problem.constraint(beta, msk.Domain.greaterThan(-self.M))
                problem.constraint(msk.Expr.sum(abs_beta), msk.Domain.lessThan(self.k * self.M))

            loss = msk.Expr.add(t, msk.Expr.mul(self.lambda2, msk.Expr.sum(abs_beta)))
        else:
            raise ValueError(f"Invalid relaxation type: {self.relaxation_type}")

        problem.objective(msk.ObjectiveSense.Minimize, loss)
        return variables, problem

    def solve(self, verbose=False, tol=1e-6, timeLimit=1800, solverName=None, beta_warmstart=None):
        mosek_params = {
            'optimizerMaxTime': float(timeLimit),
            'intpntCoTolPfeas': tol,
            'intpntCoTolDfeas': tol,
            'intpntCoTolRelGap': tol,
        }
        if self.z_is_boolean:
            mosek_params['mioMaxTime'] = float(timeLimit)
            mosek_params['mioTolRelGap'] = tol
            if beta_warmstart is not None:
                z0 = np.zeros(self.p, dtype=float)
                k_eff = min(self.k, self.p)
                idx = np.argpartition(-np.abs(beta_warmstart), k_eff-1)[:k_eff]
                z0[idx] = 1.0
                self.variables["z"].setLevel(z0.tolist())
                # Tell MOSEK to construct a feasible incumbent from integer values
                self.problem.setSolverParam("mioConstructSol", "on")
        for key, value in mosek_params.items():
            self.problem.setSolverParam(key, value)
        if verbose:
            self.problem.setLogHandler(sys.stdout)

        time_start = time.time()
        self.problem.solve()
        self.solver_time = time.time() - time_start

        try:
            self.solution = self.variables["beta"].level()
        except Exception:
            self.solution = None
        try:
            self.objective_value = self.problem.primalObjValue()
        except Exception:
            self.objective_value = None
        
        self.solution_status = self.problem.getProblemStatus()


class LogisticRegression_mosek(solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        super().__init__(X, y, k, lambda2, M, z_is_boolean=z_is_boolean, relaxation_type=relaxation_type)

    def get_problem_formulation(self):
        problem = msk.Model("Logistic Regression")

        beta = problem.variable("beta", self.p, msk.Domain.unbounded())
        variables = {"beta": beta}

        yX = msk.Matrix.dense(self.y.reshape(-1, 1) * self.X)
        margins = msk.Expr.mul(yX, beta)  # y * (X @ beta)
        # softplus via exponential cone: softplus(u) = log(1+exp(u))
        t_soft = problem.variable("t_soft", self.n)
        softplus(problem, t_soft, msk.Expr.neg(margins))

        if self.relaxation_type == "perspective":
            s = problem.variable("s", self.p, msk.Domain.greaterThan(0))
            z_domain = msk.Domain.binary() if self.z_is_boolean else msk.Domain.inRange(0, 1)
            z = problem.variable("z", self.p, z_domain)
            variables.update({"s": s, "z": z, "t_soft": t_soft})

            problem.constraint(msk.Expr.hstack(msk.Expr.mul(0.5, s), z, beta), msk.Domain.inRotatedQCone())
            problem.constraint(msk.Expr.sub(beta, msk.Expr.mul(self.M, z)), msk.Domain.lessThan(0))
            problem.constraint(msk.Expr.add(beta, msk.Expr.mul(self.M, z)), msk.Domain.greaterThan(0))
            problem.constraint(msk.Expr.sum(z), msk.Domain.lessThan(self.k))

            loss = msk.Expr.add(msk.Expr.sum(t_soft), msk.Expr.mul(self.lambda2, msk.Expr.sum(s)))

        elif self.relaxation_type == "l1":
            abs_beta = problem.variable("abs_beta", self.p, msk.Domain.greaterThan(0))
            t_reg = problem.variable("t_reg", 1, msk.Domain.greaterThan(0.0))
            variables.update({"abs_beta": abs_beta, "t_soft": t_soft, "t_reg": t_reg})
            problem.constraint(msk.Expr.sub(abs_beta, beta), msk.Domain.greaterThan(0))
            problem.constraint(msk.Expr.add(abs_beta, beta), msk.Domain.greaterThan(0))
            problem.constraint(beta, msk.Domain.lessThan(self.M))
            problem.constraint(beta, msk.Domain.greaterThan(-self.M))
            problem.constraint(msk.Expr.sum(abs_beta), msk.Domain.lessThan(self.k * self.M))
            problem.constraint(msk.Expr.vstack(t_reg, 0.5, beta), msk.Domain.inRotatedQCone())
            loss = msk.Expr.add(msk.Expr.sum(t_soft), msk.Expr.mul(self.lambda2, t_reg))
        else:
            raise ValueError(f"Invalid relaxation type: {self.relaxation_type}")

        problem.objective(msk.ObjectiveSense.Minimize, loss)
        return variables, problem

    def solve(self, verbose=False, tol=1e-6, timeLimit=1800, solverName=None, beta_warmstart=None):
        mosek_params = {
            'optimizerMaxTime': float(timeLimit),
            'intpntCoTolPfeas': tol,
            'intpntCoTolDfeas': tol,
            'intpntCoTolRelGap': tol,
        }
        if self.z_is_boolean:
            mosek_params['mioMaxTime'] = float(timeLimit)
            mosek_params['mioTolRelGap'] = tol
            if beta_warmstart is not None:
                z0 = np.zeros(self.p, dtype=float)
                k_eff = min(self.k, self.p)
                idx = np.argpartition(-np.abs(beta_warmstart), k_eff-1)[:k_eff]
                z0[idx] = 1.0
                self.variables["z"].setLevel(z0.tolist())
                # Tell MOSEK to construct a feasible incumbent from integer values
                self.problem.setSolverParam("mioConstructSol", "on")
        for key, value in mosek_params.items():
            self.problem.setSolverParam(key, value)
        if verbose:
            self.problem.setLogHandler(sys.stdout)

        time_start = time.time()
        self.problem.solve()
        self.solver_time = time.time() - time_start

        try:
            self.solution = self.variables["beta"].level()
        except Exception:
            self.solution = None
        try:
            self.objective_value = self.problem.primalObjValue()
        except Exception:
            self.objective_value = None
        self.solution_status = self.problem.getProblemStatus()


class PoissonRegression_mosek(solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        super().__init__(X, y, k, lambda2, M, z_is_boolean=z_is_boolean, relaxation_type=relaxation_type)

    def get_problem_formulation(self):
        problem = msk.Model("Poisson Regression")

        beta = problem.variable("beta", self.p, msk.Domain.unbounded())
        variables = {"beta": beta}

        X_mat = msk.Matrix.dense(self.X)
        margin = msk.Expr.mul(X_mat, beta)  # X @ beta
        uexp = problem.variable("uexp", self.n, msk.Domain.greaterThan(0.0))
        problem.constraint(msk.Expr.hstack(uexp, msk.Expr.constTerm(self.n, 1.0), margin), msk.Domain.inPExpCone())
        linear_term = msk.Expr.dot(self.y.tolist(), margin)

        if self.relaxation_type == "perspective":
            s = problem.variable("s", self.p, msk.Domain.greaterThan(0))
            z_domain = msk.Domain.binary() if self.z_is_boolean else msk.Domain.inRange(0, 1)
            z = problem.variable("z", self.p, z_domain)
            variables.update({"s": s, "z": z})

            problem.constraint(msk.Expr.hstack(msk.Expr.mul(0.5, s), z, beta), msk.Domain.inRotatedQCone())
            problem.constraint(msk.Expr.sub(beta, msk.Expr.mul(self.M, z)), msk.Domain.lessThan(0))
            problem.constraint(msk.Expr.add(beta, msk.Expr.mul(self.M, z)), msk.Domain.greaterThan(0))
            problem.constraint(msk.Expr.sum(z), msk.Domain.lessThan(self.k))

            reg = msk.Expr.mul(self.lambda2, msk.Expr.sum(s))
        elif self.relaxation_type == "l1":
            abs_beta = problem.variable("abs_beta", self.p, msk.Domain.greaterThan(0))
            t_reg = problem.variable("t_reg", 1, msk.Domain.greaterThan(0.0))
            variables.update({"abs_beta": abs_beta, "t_reg": t_reg})
            problem.constraint(msk.Expr.sub(abs_beta, beta), msk.Domain.greaterThan(0))
            problem.constraint(msk.Expr.add(abs_beta, beta), msk.Domain.greaterThan(0))
            problem.constraint(beta, msk.Domain.lessThan(self.M))
            problem.constraint(beta, msk.Domain.greaterThan(-self.M))
            problem.constraint(msk.Expr.sum(abs_beta), msk.Domain.lessThan(self.k * self.M))
            problem.constraint(msk.Expr.vstack(t_reg, 0.5, beta), msk.Domain.inRotatedQCone())
            reg = msk.Expr.mul(self.lambda2, t_reg)
        else:
            raise ValueError(f"Invalid relaxation type: {self.relaxation_type}")

        loss = msk.Expr.add(msk.Expr.sub(msk.Expr.sum(uexp), linear_term), reg)
        problem.objective(msk.ObjectiveSense.Minimize, loss)
        return variables, problem

    def solve(self, verbose=False, tol=1e-6, timeLimit=1800, solverName=None, beta_warmstart=None):
        mosek_params = {
            'optimizerMaxTime': float(timeLimit),
            'intpntCoTolPfeas': tol,
            'intpntCoTolDfeas': tol,
            'intpntCoTolRelGap': tol,
        }
        if self.z_is_boolean:
            mosek_params['mioMaxTime'] = float(timeLimit)
            mosek_params['mioTolRelGap'] = tol
            if beta_warmstart is not None:
                z0 = np.zeros(self.p, dtype=float)
                k_eff = min(self.k, self.p)
                idx = np.argpartition(-np.abs(beta_warmstart), k_eff-1)[:k_eff]
                z0[idx] = 1.0
                self.variables["z"].setLevel(z0.tolist())
                # Tell MOSEK to construct a feasible incumbent from integer values
                self.problem.setSolverParam("mioConstructSol", "on")
        for key, value in mosek_params.items():
            self.problem.setSolverParam(key, value)
        if verbose:
            self.problem.setLogHandler(sys.stdout)

        time_start = time.time()
        self.problem.solve()
        self.solver_time = time.time() - time_start

        try:
            self.solution = self.variables["beta"].level()
        except Exception:
            self.solution = None
        try:
            self.objective_value = self.problem.primalObjValue()
        except Exception:
            self.objective_value = None
        self.solution_status = self.problem.getProblemStatus()


class LinearRegression_gurobi(solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        super().__init__(X, y, k, lambda2, M, z_is_boolean=z_is_boolean, relaxation_type=relaxation_type)

    def get_problem_formulation(self):
        model = gp.Model("LinearRegression_gurobi")
        model.Params.OutputFlag = 0  # silence by default; controlled via solve(verbose)

        beta = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="beta")
        variables = {"beta": beta}

        r = model.addMVar(self.n, lb=-gp.GRB.INFINITY, name="residual")
        variables["r"] = r
        model.addConstr(r == self.X @ beta - self.y, name="residuals")

        if self.relaxation_type == "perspective":
            s = model.addMVar(self.p, lb=0.0, name="s")
            z_vtype = gp.GRB.BINARY if self.z_is_boolean else gp.GRB.CONTINUOUS
            z = model.addMVar(self.p, lb=0.0, ub=1.0, vtype=z_vtype, name="z")
            variables.update({"s": s, "z": z})

            model.addConstr(beta * beta <= s * z, name="rotated_cone")
            model.addConstr(beta <= self.M * z, name="beta_pos")
            model.addConstr(-beta <= self.M * z, name="beta_neg")
            model.addConstr(z.sum() <= self.k, name="cardinality")

            obj = r @ r + self.lambda2 * s.sum()

        elif self.relaxation_type in ["l1", "l1_no_constraint"]:
            abs_beta = model.addMVar(self.p, lb=0.0, name="abs_beta")
            variables["abs_beta"] = abs_beta
            model.addConstr(abs_beta >= beta, name="abs_pos")
            model.addConstr(abs_beta >= -beta, name="abs_neg")
            if self.relaxation_type == "l1":
                model.addConstr(beta <= self.M, name="box_pos")
                model.addConstr(beta >= -self.M, name="box_neg")
                model.addConstr(abs_beta.sum() <= self.k * self.M, name="l1_card")

            obj = r @ r + self.lambda2 * abs_beta.sum()
        else:
            raise ValueError(f"Invalid relaxation type: {self.relaxation_type}")

        model.setObjective(obj, gp.GRB.MINIMIZE)

        return variables, model

    def solve(self, verbose=False, tol=1e-6, timeLimit=1800, solverName=None, beta_warmstart=None):
        if verbose:
            self.problem.Params.OutputFlag = 1
        self.problem.Params.TimeLimit = timeLimit
        # Tolerances
        self.problem.Params.BarQCPConvTol = tol
        self.problem.Params.OptimalityTol = tol
        self.problem.Params.FeasibilityTol = tol
        if self.z_is_boolean:
            self.problem.Params.MIPGap = tol
            if beta_warmstart is not None:
                z_start = np.zeros(self.p)
                idx = np.argpartition(-np.abs(beta_warmstart), self.k-1)[:self.k]
                z_start[idx] = 1.0
                self.variables["z"].Start = z_start   # MVar Start
                self.variables["beta"].Start = beta_warmstart  # MVar Start

        time_start = time.time()
        self.problem.optimize()
        self.solver_time = time.time() - time_start

        if (self.problem.SolCount or 0) > 0:
            self.solution = self.variables["beta"].X
            self.objective_value = self.problem.ObjVal
        else:
            self.solution = None
            self.objective_value = None
        self.solution_status = self.problem.status


class LogisticRegression_gurobi(solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        self.yX = y.reshape(-1, 1) * X
        super().__init__(X, y, k, lambda2, M, z_is_boolean=z_is_boolean, relaxation_type=relaxation_type)

    def get_problem_formulation(self):
        model = gp.Model("LogisticRegression_gurobi")
        model.Params.OutputFlag = 0

        beta = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="beta")
        variables = {"beta": beta}

        # Margins and logistic loss using general exp/log
        xexp = model.addMVar(self.n, lb=-gp.GRB.INFINITY, name="xexp")
        model.addConstr(xexp == -(self.yX @ beta), name="xexp_def")
        exp_val = model.addMVar(self.n, lb=0.0, name="exp_val")
        w = model.addMVar(self.n, lb=1.0, name="w_log1p")
        t = model.addMVar(self.n, lb=-gp.GRB.INFINITY, name="t_softplus")
        for i in range(self.n):
            model.addGenConstrExp(xexp[i], exp_val[i], name=f"exp_constr_{i}")
            model.addConstr(w[i] == 1.0 + exp_val[i], name=f"one_plus_exp_{i}")
            # model.addGenConstrLog(w[i], t[i], name=f"log_constr_{i}")
            model.addGenConstrExp(t[i], w[i], name=f"log_constr_{i}_via_exp")

        if self.relaxation_type == "perspective":
            s = model.addMVar(self.p, lb=0.0, name="s")
            z_vtype = gp.GRB.BINARY if self.z_is_boolean else gp.GRB.CONTINUOUS
            z = model.addMVar(self.p, lb=0.0, ub=1.0, vtype=z_vtype, name="z")
            variables.update({"s": s, "z": z})

            model.addConstr(beta * beta <= s * z, name="rotated_cone")
            model.addConstr(beta <= self.M * z, name="beta_pos")
            model.addConstr(-beta <= self.M * z, name="beta_neg")
            model.addConstr(z.sum() <= self.k, name="cardinality")

            obj = t.sum() + self.lambda2 * s.sum()

        elif self.relaxation_type == "l1":
            abs_beta = model.addMVar(self.p, lb=0.0, name="abs_beta")
            variables["abs_beta"] = abs_beta
            model.addConstr(abs_beta >= beta, name="abs_pos")
            model.addConstr(abs_beta >= -beta, name="abs_neg")
            model.addConstr(beta <= self.M, name="box_pos")
            model.addConstr(beta >= -self.M, name="box_neg")
            model.addConstr(abs_beta.sum() <= self.k * self.M, name="l1_card")
            obj = t.sum() + self.lambda2 * (beta @ beta)
        else:
            raise ValueError(f"Invalid relaxation type: {self.relaxation_type}")

        model.setObjective(obj, gp.GRB.MINIMIZE)
        return variables, model

    def solve(self, verbose=False, tol=1e-6, timeLimit=1800, solverName=None, beta_warmstart=None):
        if verbose:
            self.problem.Params.OutputFlag = 1
        self.problem.Params.TimeLimit = timeLimit
        self.problem.Params.BarQCPConvTol = tol
        self.problem.Params.OptimalityTol = tol
        self.problem.Params.FeasibilityTol = tol
        self.problem.Params.FuncNonlinear = 1
        if self.z_is_boolean:
            self.problem.Params.MIPGap = tol
            if beta_warmstart is not None:
                z_start = np.zeros(self.p)
                idx = np.argpartition(-np.abs(beta_warmstart), self.k-1)[:self.k]
                z_start[idx] = 1.0
                self.variables["z"].Start = z_start   # MVar Start
                self.variables["beta"].Start = beta_warmstart  # MVar Start

        time_start = time.time()
        self.problem.optimize()
        self.solver_time = time.time() - time_start

        if (self.problem.SolCount or 0) > 0:
            self.solution = self.variables["beta"].X
            self.objective_value = self.problem.ObjVal
        else:
            self.solution = None
            self.objective_value = None
        self.solution_status = self.problem.status


class PoissonRegression_gurobi(solver_base_class):
    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        super().__init__(X, y, k, lambda2, M, z_is_boolean=z_is_boolean, relaxation_type=relaxation_type)

    def get_problem_formulation(self):
        model = gp.Model("PoissonRegression_gurobi")
        model.Params.OutputFlag = 0

        beta = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="beta")
        variables = {"beta": beta}

        margin = model.addMVar(self.n, lb=-gp.GRB.INFINITY, name="margin")
        model.addConstr(margin == self.X @ beta, name="margin_def")
        exp_val = model.addMVar(self.n, lb=0.0, name="exp_val")
        linear_expr = 0.0
        for i in range(self.n):
            model.addGenConstrExp(margin[i], exp_val[i], name=f"exp_constr_{i}")
            linear_expr += -self.y[i] * margin[i]

        if self.relaxation_type == "perspective":
            s = model.addMVar(self.p, lb=0.0, name="s")
            z_vtype = gp.GRB.BINARY if self.z_is_boolean else gp.GRB.CONTINUOUS
            z = model.addMVar(self.p, lb=0.0, ub=1.0, vtype=z_vtype, name="z")
            variables.update({"s": s, "z": z})

            model.addConstr(beta * beta <= s * z, name="rotated_cone")
            model.addConstr(beta <= self.M * z, name="beta_pos")
            model.addConstr(-beta <= self.M * z, name="beta_neg")
            model.addConstr(z.sum() <= self.k, name="cardinality")

            obj = exp_val.sum() + linear_expr + self.lambda2 * s.sum()
        elif self.relaxation_type == "l1":
            abs_beta = model.addMVar(self.p, lb=0.0, name="abs_beta")
            variables["abs_beta"] = abs_beta
            model.addConstr(abs_beta >= beta, name="abs_pos")
            model.addConstr(abs_beta >= -beta, name="abs_neg")
            model.addConstr(beta <= self.M, name="box_pos")
            model.addConstr(beta >= -self.M, name="box_neg")
            model.addConstr(abs_beta.sum() <= self.k * self.M, name="l1_card")
            obj = exp_val.sum() + linear_expr + self.lambda2 * (beta @ beta)
        else:
            raise ValueError(f"Invalid relaxation type: {self.relaxation_type}")

        model.setObjective(obj, gp.GRB.MINIMIZE)
        return variables, model

    def solve(self, verbose=False, tol=1e-6, timeLimit=1800, solverName=None, beta_warmstart=None):
        if verbose:
            self.problem.Params.OutputFlag = 1
        self.problem.Params.TimeLimit = timeLimit
        self.problem.Params.BarQCPConvTol = tol
        self.problem.Params.OptimalityTol = tol
        self.problem.Params.FeasibilityTol = tol
        # self.problem.Params.FuncNonlinear = 1
        self.problem.Params.FuncNonlinear = 0
        self.problem.Params.FuncPieces = -1
        self.problem.Params.FuncPieceError = 1e-4   # try 1e-4 first if model explodes

        if self.z_is_boolean:
            self.problem.Params.MIPGap = tol
            if beta_warmstart is not None:
                z_start = np.zeros(self.p)
                idx = np.argpartition(-np.abs(beta_warmstart), self.k-1)[:self.k]
                z_start[idx] = 1.0
                self.variables["z"].Start = z_start   # MVar Start
                self.variables["beta"].Start = beta_warmstart  # MVar Start

        time_start = time.time()
        self.problem.optimize()
        self.solver_time = time.time() - time_start

        if (self.problem.SolCount or 0) > 0:
            self.solution = self.variables["beta"].X
            self.objective_value = self.problem.ObjVal
        else:
            self.solution = None
            self.objective_value = None
        self.solution_status = self.problem.status

class LinearRegression_gurobiOA(solver_base_class):
    """
    Gurobi Outer-Approximation (cutting-plane) formulation for linear regression
    with perspective/SOC constraints.

    - If z_is_boolean == False: continuous cutting-plane loop (Kelley).
    - If z_is_boolean == True: lazy-constraint OA in a MIP callback (cbLazy).

    Squared loss:
        phi(beta) = sum_i (x_i^T beta - y_i)^2
    Cuts:
        t >= phi(beta_k) + grad(phi)(beta_k)^T (beta - beta_k)
           = (phi(beta_k) - grad^T beta_k) + grad^T beta
    """

    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        super().__init__(X, y, k, lambda2, M, z_is_boolean=z_is_boolean, relaxation_type=relaxation_type)

    # ---------- loss/gradient ----------
    def _loss_and_grad(self, beta_vec: np.ndarray):
        """
        Returns (loss, grad) for phi(beta) = ||X beta - y||^2.
        """
        r = (self.X @ beta_vec - self.y).ravel()
        loss = float(r @ r)
        grad = 2.0 * (self.X.T @ r)
        return loss, grad

    def _add_oa_cut(self, model: gp.Model, t_var, beta_var, beta_at: np.ndarray, name: str):
        """
        Adds: t >= c + g^T beta, where
            g = grad phi(beta_at),
            c = phi(beta_at) - g^T beta_at
        """
        f, g = self._loss_and_grad(beta_at)
        c = f - float(g @ beta_at)
        model.addConstr(t_var >= c + g @ beta_var, name=name)

    # ---------- model ----------
    def get_problem_formulation(self):
        model = gp.Model("LinearRegression_gurobiOA")
        model.Params.OutputFlag = 0  # controlled in solve()

        beta = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="beta")
        t_epi = model.addVar(lb=0.0, name="t_epi")  # epigraph of squared loss (lower bounded by 0)

        variables = {"beta": beta, "t_epi": t_epi}

        if self.relaxation_type == "perspective":
            s = model.addMVar(self.p, lb=0.0, name="s")

            z_vtype = gp.GRB.BINARY if self.z_is_boolean else gp.GRB.CONTINUOUS
            z = model.addMVar(self.p, lb=0.0, ub=1.0, vtype=z_vtype, name="z")
            variables.update({"s": s, "z": z})

            # -Mz <= beta <= Mz
            model.addConstr(beta <= self.M * z, name="beta_pos")
            model.addConstr(-beta <= self.M * z, name="beta_neg")
            model.addConstr(z.sum() <= self.k, name="cardinality")

            # SOC representation of beta_j^2 <= s_j z_j:
            # Introduce a,b,u with:
            #   a = 2 beta,  b = s - z,  u = s + z (u>=0)
            # and enforce: a^2 + b^2 <= u^2  (elementwise)
            a = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="a_soc")
            b = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="b_soc")
            u = model.addMVar(self.p, lb=0.0, name="u_soc")  # must be >=0 for SOC meaning
            model.addConstr(a == 2.0 * beta, name="a_def")
            model.addConstr(b == s - z, name="b_def")
            model.addConstr(u == s + z, name="u_def")

            # elementwise SOC: a_j^2 + b_j^2 <= u_j^2
            # IMPORTANT: use a*a not a**2 (MLinExpr doesn't support pow)
            model.addConstr(a * a + b * b <= u * u, name="persp_soc")

            obj = t_epi + self.lambda2 * s.sum()

        elif self.relaxation_type == "l1":
            abs_beta = model.addMVar(self.p, lb=0.0, name="abs_beta")
            variables["abs_beta"] = abs_beta

            model.addConstr(abs_beta >= beta, name="abs_pos")
            model.addConstr(abs_beta >= -beta, name="abs_neg")
            model.addConstr(beta <= self.M, name="box_pos")
            model.addConstr(beta >= -self.M, name="box_neg")
            model.addConstr(abs_beta.sum() <= self.k * self.M, name="l1_card")

            obj = t_epi + self.lambda2 * (beta @ beta)

        else:
            raise ValueError(f"Invalid relaxation type: {self.relaxation_type}")

        model.setObjective(obj, gp.GRB.MINIMIZE)
        return variables, model

    # ---------- solve ----------
    def solve(self, verbose=False, tol=1e-6, timeLimit=1800, solverName=None, beta_warmstart=None):
        """
        If z is continuous: Kelley cutting-plane loop.
        If z is binary: lazy-constraint OA via MIP callback.
        """
        m = self.problem
        beta = self.variables["beta"]
        s = self.variables["s"]
        z = self.variables["z"]
        t = self.variables["t_epi"]

        # Base params
        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.OptimalityTol = tol
        m.Params.FeasibilityTol = tol
        m.Params.BarQCPConvTol = tol
        m.Params.TimeLimit = float(timeLimit)

        # Add a couple of initial cuts to avoid an overly-loose start
        cut_id = 0
        beta0 = np.zeros(self.p)
        self._add_oa_cut(m, t, beta, beta0, name=f"oa_cut_init_{cut_id}")
        cut_id += 1
        if beta_warmstart is not None:
            self._add_oa_cut(m, t, beta, beta_warmstart, name=f"oa_cut_init_{cut_id}")
            cut_id += 1

        # Warm start (only meaningful for MIP)
        if self.z_is_boolean and ("z" in self.variables) and (beta_warmstart is not None):
            # Build a k-sparse start z
            z_start = np.zeros(self.p)
            k_eff = min(self.k, self.p)
            idx = np.argpartition(-np.abs(beta_warmstart), k_eff - 1)[:k_eff]
            z_start[idx] = 1.0

            self.variables["z"].Start = z_start
            self.variables["beta"].Start = beta_warmstart
            m.update()

        start_time = time.time()

        # ----- Case 1: continuous z -> manual cutting-plane loop -----
        if not self.z_is_boolean:
            max_cuts = int(1e3)  # you can tune
            while True:
                elapsed = time.time() - start_time
                remaining = float(timeLimit) - elapsed
                if remaining <= 0:
                    break
                m.Params.TimeLimit = remaining
                # verbose is False here to avoid double logging
                m.Params.OutputFlag = 0

                m.optimize()
                self.solution_status = m.Status

                if m.SolCount == 0:
                    break

                beta_sol = beta.X
                s_sol = s.X
                z_sol = z.X
                print(f"max beta_sol: {np.max(np.abs(beta_sol))}")
                obj_val = m.ObjVal
                t_sol = float(t.X)
                f_sol, g_sol = self._loss_and_grad(beta_sol)
                vio = f_sol - t_sol

                if verbose:
                    print(f"[OA] f(beta)={f_sol:.6e}, [OA] obj={obj_val:.6e}, t={t_sol:.6e}, violation={vio:.3e}, cuts={cut_id}")

                if vio <= 10.0 * tol:
                    # tight enough
                    break

                # Add violated cut at current solution
                c = f_sol - float(g_sol @ beta_sol)
                m.addConstr(t >= c + g_sol @ beta, name=f"oa_cut_{cut_id}")
                cut_id += 1

                if cut_id >= max_cuts:
                    break

            self.solver_time = time.time() - start_time

            if m.SolCount > 0:
                self.solution = beta.X
                self.objective_value = m.ObjVal
            else:
                self.solution = None
                self.objective_value = None
            self.solution_status = m.Status
            return

        # ----- Case 2: binary z -> lazy-constraint OA in callback -----
        # Lazy constraints must be enabled to add constraints during MIP search. :contentReference[oaicite:1]{index=1}
        m.Params.LazyConstraints = 1

        # If you later also add cbCut user cuts, PreCrush is recommended so cuts aren't ignored. :contentReference[oaicite:2]{index=2}
        # We keep it on here to be safe with OA-style callbacks.
        m.Params.PreCrush = 1

        m.Params.MIPGap = tol

        # Attach for callback
        m._oa_beta = beta
        m._oa_t = t


        def oa_cb(model, where):
            # ---- 1) User cuts at fractional node relaxations ----
            if where == gp.GRB.Callback.MIPNODE:
                # Only if the node relaxation was solved to optimality
                if model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) != gp.GRB.OPTIMAL:
                    print(f"[OA Callback] Node relaxation not optimal, skipping user cut.")
                    return

                beta_val = np.array(model.cbGetNodeRel(model._oa_beta))
                t_val = float(model.cbGetNodeRel(model._oa_t))

                f_val, g_val = self._loss_and_grad(beta_val)

                # If epigraph violated, add a *user cut* to tighten the relaxation
                if f_val > t_val + 10.0 * tol:
                    c_val = f_val - float(g_val @ beta_val)
                    model.cbCut(model._oa_t >= c_val + g_val @ model._oa_beta)
                    # print(f"[OA Callback] Added user cut at node relaxation, f={f_val:.6e}, t={t_val:.6e}")
                    self.num_user_cuts += 1
                    # print(f"[OA Callback] Total user cuts added so far: {self.num_user_cuts}")
                else:
                    print(f"[OA Callback] No violation at node relaxation, f={f_val:.6e}, t={t_val:.6e}")
                return

            # ---- 2) Lazy constraints at integer-feasible incumbents ----
            # only adding this is too weak, that's why we also have user cuts above
            if where == gp.GRB.Callback.MIPSOL:
                beta_val = np.array(model.cbGetSolution(model._oa_beta))
                t_val = float(model.cbGetSolution(model._oa_t))

                f_val, g_val = self._loss_and_grad(beta_val)

                # If violated at an incumbent, it MUST be enforced for correctness -> lazy constraint
                if f_val > t_val + 10.0 * tol:
                    c_val = f_val - float(g_val @ beta_val)
                    model.cbLazy(model._oa_t >= c_val + g_val @ model._oa_beta)
                    # print(f"[OA Callback] Added lazy constraint at incumbent, f={f_val:.6e}, t={t_val:.6e}")
                else:
                    print(f"[OA Callback] No violation at incumbent, f={f_val:.6e}, t={t_val:.6e}")
                return

        self.num_user_cuts = 0
        m.optimize(oa_cb)
        self.solver_time = time.time() - start_time

        if (m.SolCount or 0) > 0:
            self.solution = beta.X
            self.objective_value = m.ObjVal
        else:
            self.solution = None
            self.objective_value = None
        self.solution_status = m.Status

class LogisticRegression_guriobiOA(solver_base_class):
    """
    Gurobi Outer-Approximation (cutting-plane) formulation for logistic regression
    with perspective/SOC constraints.

    - If z_is_boolean == False: continuous cutting-plane loop (Kelley).
    - If z_is_boolean == True: lazy-constraint OA in a MIP callback (cbLazy).

    Logistic loss:
        phi(beta) = sum_i log(1 + exp(- y_i * x_i^T beta))
    Cuts:
        t >= phi(beta_k) + grad(phi)(beta_k)^T (beta - beta_k)
           = (phi(beta_k) - grad^T beta_k) + grad^T beta
    """

    def __init__(self, X, y, k, lambda2, M, z_is_boolean=False, relaxation_type="perspective"):
        # store yX = diag(y) X for fast loss/grad
        self.yX = y.reshape(-1, 1) * X
        super().__init__(X, y, k, lambda2, M, z_is_boolean=z_is_boolean, relaxation_type=relaxation_type)

    # ---------- loss/gradient ----------
    def _loss_and_grad(self, beta_vec: np.ndarray):
        """
        Returns (loss, grad) for phi(beta) = sum log(1+exp(-yXbeta)).
        Uses stable primitives:
            loss = sum logaddexp(0, -m), m = yX @ beta
            grad = - yX^T * sigmoid(-m)
        """
        m = self.yX @ beta_vec                      # shape (n,)
        loss = np.logaddexp(0.0, -m).sum()
        sigma = scipy.special.expit(-m)             # shape (n,)
        grad = -(self.yX.T @ sigma)                 # shape (p,)
        return float(loss), grad

    def _add_oa_cut(self, model: gp.Model, t_var, beta_var, beta_at: np.ndarray, name: str):
        """
        Adds: t >= c + g^T beta, where
            g = grad phi(beta_at),
            c = phi(beta_at) - g^T beta_at
        """
        f, g = self._loss_and_grad(beta_at)
        c = f - float(g @ beta_at)
        model.addConstr(t_var >= c + g @ beta_var, name=name)

    # ---------- model ----------
    def get_problem_formulation(self):
        model = gp.Model("LogisticRegression_gurobiOA")
        model.Params.OutputFlag = 0  # controlled in solve()

        beta = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="beta")
        t_epi = model.addVar(lb=0.0, name="t_epi")  # epigraph of logistic loss (lower bounded by 0)

        variables = {"beta": beta, "t_epi": t_epi}

        if self.relaxation_type == "perspective":
            s = model.addMVar(self.p, lb=0.0, name="s")

            z_vtype = gp.GRB.BINARY if self.z_is_boolean else gp.GRB.CONTINUOUS
            z = model.addMVar(self.p, lb=0.0, ub=1.0, vtype=z_vtype, name="z")
            variables.update({"s": s, "z": z})

            # -Mz <= beta <= Mz
            model.addConstr(beta <= self.M * z, name="beta_pos")
            model.addConstr(-beta <= self.M * z, name="beta_neg")
            model.addConstr(z.sum() <= self.k, name="cardinality")

            # SOC representation of beta_j^2 <= s_j z_j:
            # Introduce a,b,u with:
            #   a = 2 beta,  b = s - z,  u = s + z (u>=0)
            # and enforce: a^2 + b^2 <= u^2  (elementwise)
            a = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="a_soc")
            b = model.addMVar(self.p, lb=-gp.GRB.INFINITY, name="b_soc")
            u = model.addMVar(self.p, lb=0.0, name="u_soc")  # must be >=0 for SOC meaning
            model.addConstr(a == 2.0 * beta, name="a_def")
            model.addConstr(b == s - z, name="b_def")
            model.addConstr(u == s + z, name="u_def")

            # elementwise SOC: a_j^2 + b_j^2 <= u_j^2
            # IMPORTANT: use a*a not a**2 (MLinExpr doesn't support pow)
            model.addConstr(a * a + b * b <= u * u, name="persp_soc")

            obj = t_epi + self.lambda2 * s.sum()

        elif self.relaxation_type == "l1":
            # Match your template: logistic loss epigraph + ridge penalty, with l1-ball + box constraints
            abs_beta = model.addMVar(self.p, lb=0.0, name="abs_beta")
            variables["abs_beta"] = abs_beta

            model.addConstr(abs_beta >= beta, name="abs_pos")
            model.addConstr(abs_beta >= -beta, name="abs_neg")
            model.addConstr(beta <= self.M, name="box_pos")
            model.addConstr(beta >= -self.M, name="box_neg")
            model.addConstr(abs_beta.sum() <= self.k * self.M, name="l1_card")

            obj = t_epi + self.lambda2 * (beta @ beta)

        else:
            raise ValueError(f"Invalid relaxation type: {self.relaxation_type}")

        model.setObjective(obj, gp.GRB.MINIMIZE)
        return variables, model

    # ---------- solve ----------
    def solve(self, verbose=False, tol=1e-6, timeLimit=1800, solverName=None, beta_warmstart=None):
        """
        If z is continuous: Kelley cutting-plane loop.
        If z is binary: lazy-constraint OA via MIP callback.
        """
        m = self.problem
        beta = self.variables["beta"]
        s = self.variables["s"]
        z = self.variables["z"]
        t = self.variables["t_epi"]

        # Base params
        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.OptimalityTol = tol
        m.Params.FeasibilityTol = tol
        m.Params.BarQCPConvTol = tol
        m.Params.TimeLimit = float(timeLimit)

        # Add a couple of initial cuts to avoid an overly-loose start
        cut_id = 0
        beta0 = np.zeros(self.p)
        self._add_oa_cut(m, t, beta, beta0, name=f"oa_cut_init_{cut_id}")
        cut_id += 1
        if beta_warmstart is not None:
            self._add_oa_cut(m, t, beta, beta_warmstart, name=f"oa_cut_init_{cut_id}")
            cut_id += 1

        # Warm start (only meaningful for MIP)
        if self.z_is_boolean and ("z" in self.variables) and (beta_warmstart is not None):
            # Build a k-sparse start z
            z_start = np.zeros(self.p)
            k_eff = min(self.k, self.p)
            idx = np.argpartition(-np.abs(beta_warmstart), k_eff - 1)[:k_eff]
            z_start[idx] = 1.0

            self.variables["z"].Start = z_start
            self.variables["beta"].Start = beta_warmstart
            m.update()

        start_time = time.time()

        # ----- Case 1: continuous z -> manual cutting-plane loop -----
        if not self.z_is_boolean:
            max_cuts = int(1e3)  # you can tune
            while True:
                elapsed = time.time() - start_time
                remaining = float(timeLimit) - elapsed
                if remaining <= 0:
                    break
                m.Params.TimeLimit = remaining
                # verbose is False here to avoid double logging
                m.Params.OutputFlag = 0

                m.optimize()
                self.solution_status = m.Status

                if m.SolCount == 0:
                    break

                beta_sol = beta.X
                s_sol = s.X
                z_sol = z.X
                print(f"max beta_sol: {np.max(np.abs(beta_sol))}")
                obj_val = m.ObjVal
                t_sol = float(t.X)
                f_sol, g_sol = self._loss_and_grad(beta_sol)
                vio = f_sol - t_sol

                if verbose:
                    print(f"[OA] f(beta)={f_sol:.6e}, [OA] obj={obj_val:.6e}, t={t_sol:.6e}, violation={vio:.3e}, cuts={cut_id}")

                if vio <= 10.0 * tol:
                    # tight enough
                    break

                # Add violated cut at current solution
                c = f_sol - float(g_sol @ beta_sol)
                m.addConstr(t >= c + g_sol @ beta, name=f"oa_cut_{cut_id}")
                cut_id += 1

                if cut_id >= max_cuts:
                    break

            self.solver_time = time.time() - start_time

            if m.SolCount > 0:
                self.solution = beta.X
                self.objective_value = m.ObjVal
            else:
                self.solution = None
                self.objective_value = None
            self.solution_status = m.Status
            return

        # ----- Case 2: binary z -> lazy-constraint OA in callback -----
        # Lazy constraints must be enabled to add constraints during MIP search. :contentReference[oaicite:1]{index=1}
        m.Params.LazyConstraints = 1

        # If you later also add cbCut user cuts, PreCrush is recommended so cuts aren't ignored. :contentReference[oaicite:2]{index=2}
        # We keep it on here to be safe with OA-style callbacks.
        m.Params.PreCrush = 1

        m.Params.MIPGap = tol

        # Attach for callback
        m._oa_beta = beta
        m._oa_t = t

        def oa_cb(model, where):
            # ---- 1) User cuts at fractional node relaxations ----
            if where == gp.GRB.Callback.MIPNODE:
                # Only if the node relaxation was solved to optimality
                if model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) != gp.GRB.OPTIMAL:
                    return

                beta_val = np.array(model.cbGetNodeRel(model._oa_beta))
                t_val = float(model.cbGetNodeRel(model._oa_t))

                f_val, g_val = self._loss_and_grad(beta_val)

                # If epigraph violated, add a *user cut* to tighten the relaxation
                if f_val > t_val + 10.0 * tol:
                    c_val = f_val - float(g_val @ beta_val)
                    model.cbCut(model._oa_t >= c_val + g_val @ model._oa_beta)
                    # model.cbLazy(model._oa_t >= c_val + g_val @ model._oa_beta)
                return

            # ---- 2) Lazy constraints at integer-feasible incumbents ----
            # only adding this is too weak, that's why we also have user cuts above
            if where == gp.GRB.Callback.MIPSOL:
                beta_val = np.array(model.cbGetSolution(model._oa_beta))
                t_val = float(model.cbGetSolution(model._oa_t))

                f_val, g_val = self._loss_and_grad(beta_val)

                # If violated at an incumbent, it MUST be enforced for correctness -> lazy constraint
                if f_val > t_val + 10.0 * tol:
                    c_val = f_val - float(g_val @ beta_val)
                    model.cbLazy(model._oa_t >= c_val + g_val @ model._oa_beta)
                return

        m.optimize(oa_cb)
        self.solver_time = time.time() - start_time

        if (m.SolCount or 0) > 0:
            self.solution = beta.X
            self.objective_value = m.ObjVal
        else:
            self.solution = None
            self.objective_value = None
        self.solution_status = m.Status
