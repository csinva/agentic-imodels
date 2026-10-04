from okglm.BnBTree.composite_cvx_opt_class_algorithms_bnb import (
    PGD_optimizer_BnB,
    ACFGM_optimizer_BnB,
    PD_Restarted_ACFGM_optimizer_BnB,
    ACFGM2_optimizer_BnB,
    PD_Restarted_ACFGM2_optimizer_BnB,
    Beck_FISTA_optimizer_BnB,
    Restarted_FISTA_optimizer_BnB,
    Restarted_Beck_FISTA_optimizer_BnB,
    PD_Restarted_Beck_FISTA_optimizer_BnB,
    Beck_FISTALineSearch_optimizer_BnB,
    PD_Restarted_Beck_FISTALineSearch_optimizer_BnB,
)
from okglm.composite_cvx_opt_class_regularizers import (
    regularizer_fenchel_kyFan_HuberLoss_BnB_gpu,
)
from okglm.optimizer_factory import normalize_method, _parse_restart_exponent, is_gpu_method
from okglm.helpers.general_helpers import assert_gpu_available

_BASE_BNB_METHODS = {
    "ours",
    "oursPGD",
    "oursBeckFISTA",
    "oursBeckFISTALineSearch",
    "oursACFGM",
    "oursACFGM2",
    "oursRestartedBeckFISTA",
    "oursPDRestartedBeckFISTA",
    "oursPDRestartedBeckFISTALineSearch",
    "oursPDRestartedBeckFISTAe2",
    "oursPDRestartedBeckFISTAe3",
    "oursPDRestartedBeckFISTAe4",
    "oursPDRestartedBeckFISTAe5",
    "oursPDRestartedBeckFISTALineSearche2",
    "oursPDRestartedBeckFISTALineSearche3",
    "oursPDRestartedBeckFISTALineSearche4",
    "oursPDRestartedBeckFISTALineSearche5",
    "oursPDRestartedBeckFISTALineSearche3Dynamic",
    "oursPDRestartedACFGM",
    "oursPDRestartedACFGM2",
    "oursPDRestartedACFGMe2",
    "oursPDRestartedACFGMe3",
    "oursPDRestartedACFGMe4",
    "oursPDRestartedACFGMe5",
    "oursPDRestartedACFGM2e2",
    "oursPDRestartedACFGM2e3",
    "oursPDRestartedACFGM2e4",
    "oursPDRestartedACFGM2e5",
}
OUR_BNB_LOWER_BOUND_METHODS = _BASE_BNB_METHODS | {
    method.replace("ours", "oursGPU", 1) for method in _BASE_BNB_METHODS
}


def build_bnb_lower_solver(method, tree_data, data_class):
    use_gpu = is_gpu_method(method)
    if use_gpu:
        assert_gpu_available()

    normalized = normalize_method(method)
    restart_exponent = _parse_restart_exponent(normalized) or 1
    dynamic_restart = normalized.endswith("Dynamic")

    if normalized == "ours":
        cls = Restarted_FISTA_optimizer_BnB
        kwargs = {}
    elif normalized.startswith("oursPGD"):
        cls = PGD_optimizer_BnB
        kwargs = {}
    elif "PDRestartedBeckFISTALineSearch" in normalized:
        cls = PD_Restarted_Beck_FISTALineSearch_optimizer_BnB
        kwargs = {"restart_exponent": restart_exponent}
        if dynamic_restart:
            kwargs["dynamic_restart"] = True
    elif "PDRestartedBeckFISTA" in normalized:
        cls = PD_Restarted_Beck_FISTA_optimizer_BnB
        kwargs = {"restart_exponent": restart_exponent}
        if dynamic_restart:
            kwargs["dynamic_restart"] = True
    elif "RestartedBeckFISTA" in normalized:
        cls = Restarted_Beck_FISTA_optimizer_BnB
        kwargs = {}
    elif "BeckFISTALineSearch" in normalized:
        cls = Beck_FISTALineSearch_optimizer_BnB
        kwargs = {}
    elif normalized.startswith("oursBeckFISTA"):
        cls = Beck_FISTA_optimizer_BnB
        kwargs = {}
    elif "PDRestartedACFGM2" in normalized:
        cls = PD_Restarted_ACFGM2_optimizer_BnB
        kwargs = {"restart_exponent": restart_exponent}
        if dynamic_restart:
            kwargs["dynamic_restart"] = True
    elif "PDRestartedACFGM" in normalized:
        cls = PD_Restarted_ACFGM_optimizer_BnB
        kwargs = {"restart_exponent": restart_exponent}
        if dynamic_restart:
            kwargs["dynamic_restart"] = True
    elif normalized.startswith("oursACFGM2"):
        cls = ACFGM2_optimizer_BnB
        kwargs = {}
    elif normalized.startswith("oursACFGM"):
        cls = ACFGM_optimizer_BnB
        kwargs = {}
    else:
        raise ValueError(f"Unsupported lower_bound_method for BnB: {method}")

    return cls(
        X=tree_data.X,
        y=tree_data.y,
        k=tree_data.k,
        lambda2=tree_data.lambda2,
        M=tree_data.M,
        data_class=data_class,
        reg_class=regularizer_fenchel_kyFan_HuberLoss_BnB_gpu,
        use_gpu=use_gpu,
        **kwargs,
    )
