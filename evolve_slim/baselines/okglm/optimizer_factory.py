from okglm.composite_cvx_opt_class_algorithms import (
    Restarted_FISTA_optimizer,
    FISTA_optimizer,
    PGD_optimizer,
    Beck_FISTA_optimizer,
    Beck_FISTALineSearch_optimizer,
    ACFGM_optimizer,
    ACFGM2_optimizer,
    Restarted_Beck_FISTA_optimizer,
    PD_Restarted_Beck_FISTA_optimizer,
    PD_Restarted_Beck_FISTALineSearch_optimizer,
    PD_Restarted_ACFGM_optimizer,
    PD_Restarted_ACFGM2_optimizer,
    data_LinearRegression_gpu,
    data_LogisticRegression_gpu,
    data_PoissonRegression_gpu,
    regularizer_l1Regularized_gpu,
    regularizer_l1Constrained_gpu,
)
from okglm.baselines.baselines_solve_relaxation import (
    LinearRegression_cvxpy,
    LogisticRegression_cvxpy,
    PoissonRegression_cvxpy,
    LinearRegression_gurobi,
    LogisticRegression_gurobi,
    PoissonRegression_gurobi,
    LinearRegression_mosek,
    LogisticRegression_mosek,
    PoissonRegression_mosek,
    LinearRegression_gurobiOA,
    LogisticRegression_guriobiOA,
)
from okglm.helpers.general_helpers import assert_gpu_available


OUR_METHODS = {
    "ours",
    "oursGPU",
    "oursPGD",
    "oursGPUPGD",
    "oursBeckFISTA",
    "oursGPUBeckFISTA",
    "oursBeckFISTALineSearch",
    "oursBeckFISTALineSearche2",
    "oursBeckFISTALineSearche3",
    "oursBeckFISTALineSearche4",
    "oursBeckFISTALineSearche5",
    "oursGPUBeckFISTALineSearch",
    "oursGPUBeckFISTALineSearche2",
    "oursGPUBeckFISTALineSearche3",
    "oursGPUBeckFISTALineSearche4",
    "oursGPUBeckFISTALineSearche5",
    "oursACFGM",
    "oursGPUACFGM",
    "oursACFGM2",
    "oursGPUACFGM2",
    "oursRestartedBeckFISTA",
    "oursGPURestartedBeckFISTA",
    "oursPDRestartedBeckFISTA",
    "oursGPUPDRestartedBeckFISTA",
    "oursPDRestartedBeckFISTALineSearch",
    "oursGPUPDRestartedBeckFISTALineSearch",
    "oursPDRestartedACFGM",
    "oursGPUPDRestartedACFGM",
    "oursPDRestartedACFGM2",
    "oursGPUPDRestartedACFGM2",
    "oursPDRestartedBeckFISTAe2",
    "oursPDRestartedBeckFISTAe3",
    "oursPDRestartedBeckFISTAe4",
    "oursPDRestartedBeckFISTAe5",
    "oursPDRestartedBeckFISTALineSearche2",
    "oursPDRestartedBeckFISTALineSearche3",
    "oursPDRestartedBeckFISTALineSearche4",
    "oursPDRestartedBeckFISTALineSearche5",
    "oursPDRestartedBeckFISTALineSearche3Dynamic",
    "oursPDRestartedACFGMe2",
    "oursPDRestartedACFGMe3",
    "oursPDRestartedACFGMe4",
    "oursPDRestartedACFGMe5",
    "oursPDRestartedACFGM2e2",
    "oursPDRestartedACFGM2e3",
    "oursPDRestartedACFGM2e4",
    "oursPDRestartedACFGM2e5",
    "oursGPUPDRestartedBeckFISTAe2",
    "oursGPUPDRestartedBeckFISTAe3",
    "oursGPUPDRestartedBeckFISTAe4",
    "oursGPUPDRestartedBeckFISTAe5",
    "oursGPUPDRestartedBeckFISTALineSearche2",
    "oursGPUPDRestartedBeckFISTALineSearche3",
    "oursGPUPDRestartedBeckFISTALineSearche4",
    "oursGPUPDRestartedBeckFISTALineSearche5",
    "oursGPUPDRestartedACFGMe2",
    "oursGPUPDRestartedACFGMe3",
    "oursGPUPDRestartedACFGMe4",
    "oursGPUPDRestartedACFGMe5",
    "oursGPUPDRestartedACFGM2e2",
    "oursGPUPDRestartedACFGM2e3",
    "oursGPUPDRestartedACFGM2e4",
    "oursGPUPDRestartedACFGM2e5",
    "oursGPUPGDl1Regularized",
    "oursGPUBeckFISTAl1Regularized",
    "oursGPUBeckFISTALineSearchl1Regularized",
    "oursGPUACFGMl1Regularized",
    "oursGPUACFGM2l1Regularized",
    "oursGPURestartedBeckFISTAl1Regularized",
    "oursGPUPDRestartedBeckFISTAl1Regularized",
    "oursGPUPDRestartedBeckFISTALineSearchl1Regularized",
    "oursGPUPDRestartedACFGMl1Regularized",
    "oursGPUPDRestartedACFGM2l1Regularized",
    "oursGPUPGDl1Constrained",
    "oursGPUBeckFISTAl1Constrained",
    "oursGPUBeckFISTALineSearchl1Constrained",
    "oursGPUACFGMl1Constrained",
    "oursGPUACFGM2l1Constrained",
    "oursGPURestartedBeckFISTAl1Constrained",
    "oursGPUPDRestartedBeckFISTAl1Constrained",
    "oursGPUPDRestartedBeckFISTALineSearchl1Constrained",
    "oursGPUPDRestartedACFGMl1Constrained",
    "oursGPUPDRestartedACFGM2l1Constrained",
}


BASELINE_METHODS = {
    "gurobi",
    "mosek",
    "scs",
    "clarabel",
    "gurobiNative",
    "gurobiNativeWarmstart",
    "gurobiNativeOA",
    "gurobiNativeOAWarmstart",
    "mosekNative",
    "mosekNativeWarmstart",
}


def is_gpu_method(method):
    return "GPU" in method


def normalize_method(method):
    if method.startswith("oursGPU"):
        return "ours" + method[len("oursGPU"):]
    return method


def _parse_restart_exponent(method):
    for exponent in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        if method.endswith(f"e{exponent}") or method.endswith(f"e{exponent}Dynamic"):
            return exponent
    return None


def _get_reg_class(method):
    if "l1Regularized" in method:
        return regularizer_l1Regularized_gpu
    if "l1Constrained" in method:
        return regularizer_l1Constrained_gpu
    return None


def get_data_class(glm_loss_type):
    if glm_loss_type == "linear":
        return data_LinearRegression_gpu
    if glm_loss_type == "logistic":
        return data_LogisticRegression_gpu
    if glm_loss_type == "poisson":
        return data_PoissonRegression_gpu
    raise ValueError(f"Invalid GLMLossType: {glm_loss_type}")


def get_baseline_optimizer_class(method, glm_loss_type):
    if glm_loss_type == "linear":
        if method in ("gurobiNative", "gurobiNativeWarmstart"):
            return LinearRegression_gurobi
        if method in ("gurobiNativeOA", "gurobiNativeOAWarmstart"):
            return LinearRegression_gurobiOA
        if method in ("mosekNative", "mosekNativeWarmstart"):
            return LinearRegression_mosek
        if method in ("gurobi", "mosek", "scs", "clarabel"):
            return LinearRegression_cvxpy
    elif glm_loss_type == "logistic":
        if method in ("gurobiNative", "gurobiNativeWarmstart"):
            return LogisticRegression_gurobi
        if method in ("gurobiNativeOA", "gurobiNativeOAWarmstart"):
            return LogisticRegression_guriobiOA
        if method in ("mosekNative", "mosekNativeWarmstart"):
            return LogisticRegression_mosek
        if method in ("gurobi", "mosek", "scs", "clarabel"):
            return LogisticRegression_cvxpy
    elif glm_loss_type == "poisson":
        if method in ("gurobiNative", "gurobiNativeWarmstart"):
            return PoissonRegression_gurobi
        if method in ("mosekNative", "mosekNativeWarmstart"):
            return PoissonRegression_mosek
        if method in ("gurobi", "mosek", "scs", "clarabel"):
            return PoissonRegression_cvxpy
    return None


def build_optimizer(method, X, y, k, lambda2, M, data_class, L, verbose, use_gpu):
    if use_gpu:
        assert_gpu_available()
    normalized = normalize_method(method)
    reg_class = _get_reg_class(normalized)
    restart_exponent = _parse_restart_exponent(normalized)
    dynamic_restart = normalized.endswith("Dynamic")

    if normalized == "ours":
        cls = Restarted_FISTA_optimizer
        kwargs = {}
    elif normalized.startswith("oursPGD"):
        cls = PGD_optimizer
        kwargs = {}
    elif "PDRestartedBeckFISTALineSearch" in normalized:
        cls = PD_Restarted_Beck_FISTALineSearch_optimizer
        kwargs = {"restart_exponent": restart_exponent} if restart_exponent else {}
        if dynamic_restart:
            kwargs["dynamic_restart"] = True
    elif "PDRestartedBeckFISTA" in normalized:
        cls = PD_Restarted_Beck_FISTA_optimizer
        kwargs = {"restart_exponent": restart_exponent} if restart_exponent else {}
    elif "RestartedBeckFISTA" in normalized:
        cls = Restarted_Beck_FISTA_optimizer
        kwargs = {}
    elif "BeckFISTALineSearch" in normalized:
        cls = Beck_FISTALineSearch_optimizer
        kwargs = {}
    elif normalized.startswith("oursBeckFISTA"):
        cls = Beck_FISTA_optimizer
        kwargs = {}
    elif "PDRestartedACFGM2" in normalized:
        cls = PD_Restarted_ACFGM2_optimizer
        kwargs = {"restart_exponent": restart_exponent} if restart_exponent else {}
    elif "PDRestartedACFGM" in normalized:
        cls = PD_Restarted_ACFGM_optimizer
        kwargs = {"restart_exponent": restart_exponent} if restart_exponent else {}
    elif normalized.startswith("oursACFGM2"):
        cls = ACFGM2_optimizer
        kwargs = {}
    elif normalized.startswith("oursACFGM"):
        cls = ACFGM_optimizer
        kwargs = {}
    else:
        raise ValueError(f"Invalid method: {method}")

    if reg_class is not None:
        kwargs["reg_class"] = reg_class

    optimizer_L = -1.0 if "LineSearch" in normalized else L

    return cls(
        X,
        y,
        k,
        lambda2,
        M,
        data_class,
        L=optimizer_L,
        verbose=verbose,
        use_gpu=use_gpu,
        **kwargs,
    )


def build_dual_loss_optimizer(X, y, k, lambda2, M, data_class, use_gpu):
    """Construct a lightweight optimizer solely to evaluate dual loss for baseline solvers."""
    if use_gpu:
        assert_gpu_available()
    return Restarted_FISTA_optimizer(
        X, y, k, lambda2, M, data_class, use_gpu=use_gpu
    )
