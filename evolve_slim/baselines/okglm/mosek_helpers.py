import os

if os.environ.get("my_gurobi_license_path"):  # evolve_slim: optional license paths
    os.environ["GRB_LICENSE_FILE"] = os.environ["my_gurobi_license_path"]
if os.environ.get("my_mosek_license_path"):  # evolve_slim: optional license paths
    os.environ["MOSEKLM_LICENSE_FILE"] = os.environ["my_mosek_license_path"]

import mosek
import mosek.fusion as msk


def mosek_solve_and_retrieve_variable(model, variables, variable_name):
    # model.solve()
    # model.acceptedSolutionStatus(msk.AccSolutionStatus.Optimal)
    # return model.getPrimalSolution(parameter_name)

    try:
        model.solve()
        # Ensure the solution status is optimal
        model.acceptedSolutionStatus(msk.AccSolutionStatus.Optimal)
        return variables[variable_name].level()

    except msk.SolutionError as e:
        print("Solution status is not optimal.")
        prosta = model.getProblemStatus()
        if prosta == msk.ProblemStatus.DualInfeasible:
            print("Dual infeasibility certificate found.")
        elif prosta == msk.ProblemStatus.PrimalInfeasible:
            print("Primal infeasibility certificate found.")
        elif prosta == msk.ProblemStatus.Unknown:
            print("The solution status is unknown.")
            symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(model.getSolverIntInfo("optimizeResponse"))))
            print(f"Termination code: {symname} {desc}")
        else:
            print(f"Unexpected problem status: {prosta}")
        return None

    except msk.OptimizeError as e:
        print(f"Optimization failed. Error: {e}")
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None