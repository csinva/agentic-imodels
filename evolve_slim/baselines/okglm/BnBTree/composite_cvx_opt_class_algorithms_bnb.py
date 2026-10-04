from okglm.composite_cvx_opt_class_algorithms import (
    PGD_optimizer,
    ACFGM_optimizer,
    PD_Restarted_ACFGM_optimizer,
    ACFGM2_optimizer,
    PD_Restarted_ACFGM2_optimizer,
    Beck_FISTA_optimizer,
    Restarted_Beck_FISTA_optimizer,
    PD_Restarted_Beck_FISTA_optimizer,
    Beck_FISTALineSearch_optimizer,
    PD_Restarted_Beck_FISTALineSearch_optimizer,
    Restarted_FISTA_optimizer,
)
from okglm.composite_cvx_opt_class_regularizers import (
    regularizer_fenchel_kyFan_HuberLoss_BnB_gpu,
)


class _BnBStoppingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upper_bound = 1e12

    def reset_z_lb_and_z_ub(self, z_lb, z_ub):
        if not hasattr(self.reg_class, "reset_z_lb_and_z_ub"):
            raise AttributeError("Regularizer does not support z-bound updates.")
        self.reg_class.reset_z_lb_and_z_ub(z_lb, z_ub)

    def solve(self, num_iter=1000, tol=1e-6, timeLimit=1800, upper_bound=1e12):
        self.upper_bound = upper_bound
        super().solve(num_iter=num_iter, tol=tol, timeLimit=timeLimit)

    def postprocess_after_one_iteration(self, iter):
        # Return value is the early_stopping boolean.
        if self.check_time_limit(iter):
            return True

        if self.internal_iter % self.postprocess_freq != 0:
            return False

        primal_loss, dual_loss, primal_dual_diff, optimality_gap = self.get_primal_dual_losses_and_gap(self.beta)

        if self.verbose:
            self.print_primal_dual_losses_and_gap(primal_loss, dual_loss, optimality_gap, iter)

        if primal_loss < self.upper_bound:
            return True

        if dual_loss >= self.upper_bound:
            return True

        if self.check_optimality_gap_convergence(optimality_gap):
            return True

        self._restart(primal_loss, dual_loss, primal_dual_diff)
        return False


_BNB_CLASS_CACHE = {}


def _make_bnb_optimizer_class(base_cls):
    if base_cls not in _BNB_CLASS_CACHE:
        _BNB_CLASS_CACHE[base_cls] = type(
            f"{base_cls.__name__}_BnB",
            (_BnBStoppingMixin, base_cls),
            {},
        )
    return _BNB_CLASS_CACHE[base_cls]


PGD_optimizer_BnB = _make_bnb_optimizer_class(PGD_optimizer)
Beck_FISTA_optimizer_BnB = _make_bnb_optimizer_class(Beck_FISTA_optimizer)
Beck_FISTALineSearch_optimizer_BnB = _make_bnb_optimizer_class(Beck_FISTALineSearch_optimizer)
ACFGM_optimizer_BnB = _make_bnb_optimizer_class(ACFGM_optimizer)
ACFGM2_optimizer_BnB = _make_bnb_optimizer_class(ACFGM2_optimizer)
PD_Restarted_Beck_FISTA_optimizer_BnB = _make_bnb_optimizer_class(PD_Restarted_Beck_FISTA_optimizer)
PD_Restarted_Beck_FISTALineSearch_optimizer_BnB = _make_bnb_optimizer_class(PD_Restarted_Beck_FISTALineSearch_optimizer)
PD_Restarted_ACFGM_optimizer_BnB = _make_bnb_optimizer_class(PD_Restarted_ACFGM_optimizer)
PD_Restarted_ACFGM2_optimizer_BnB = _make_bnb_optimizer_class(PD_Restarted_ACFGM2_optimizer)
Restarted_Beck_FISTA_optimizer_BnB = _make_bnb_optimizer_class(Restarted_Beck_FISTA_optimizer)
Restarted_FISTA_optimizer_BnB = _make_bnb_optimizer_class(Restarted_FISTA_optimizer)
