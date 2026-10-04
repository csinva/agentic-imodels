import os
import numpy as np
import sys
import scipy
import math
import time

if os.environ.get("my_gurobi_license_path"):  # evolve_slim: optional license paths
    os.environ["GRB_LICENSE_FILE"] = os.environ["my_gurobi_license_path"]
if os.environ.get("my_mosek_license_path"):  # evolve_slim: optional license paths
    os.environ["MOSEKLM_LICENSE_FILE"] = os.environ["my_mosek_license_path"]

import gurobipy as gp
import mosek.fusion as msk
import cvxpy

class solver_compute_g_class():
    def __init__(self, x, k, M=1e6):
        self.x = x
        self.k = k
        self.M = M
        self.g_value = -1
        self.solver_time = None
        self.solution_status = None

        self.variables, self.problem = self.get_problem_formulation()
    
    def get_variables_and_constraints(self):
        p = len(self.x)
        s = cvxpy.Variable(p)
        z = cvxpy.Variable(p)

        variables = {
            "s": s,
            "z": z
        }

        constraints = [
            z >= 0, 
            z <= 1, 
            cvxpy.sum(z) <= self.k, 
            np.abs(self.x) <= self.M * z, 
            cvxpy.SOC(t=s+z, X=cvxpy.vstack([2 * self.x, s - z]))
        ]

        return variables, constraints
    
    def get_solver_time(self):
        return self.solver_time

    def get_g_value(self):
        return self.g_value

    def get_problem_formulation(self):

        variables, constraints = self.get_variables_and_constraints()
        loss = 0.5 * cvxpy.sum(variables["s"])
        problem = cvxpy.Problem(cvxpy.Minimize(loss), constraints)

        return variables, problem
    
    def solve(self, solverName="gurobi", verbose=False, tol=1e-6, timeLimit=1800):
        print("tol = ", tol)
        if solverName == "gurobi":
            gurobi_env = gp.Env()
            gurobi_env.setParam("BarConvTol", tol)
            gurobi_env.setParam("TimeLimit", timeLimit)
            gurobi_env.setParam("MIPGap", tol)  # Set MIP gap tolerance
            self.problem.solve(solver=cvxpy.GUROBI, env=gurobi_env, verbose=verbose)
        elif solverName == "mosek":
            print("tol = ", tol)
            mosek_params = {
                'MSK_DPAR_OPTIMIZER_MAX_TIME': timeLimit,
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                'MSK_DPAR_MIO_TOL_REL_GAP': tol,  # Set MIP gap tolerance for Mosek
            }
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
        
        self.g_value = self.problem.value
        self.solver_time = self.problem.solver_stats.solve_time
        self.solution_status = self.problem.status