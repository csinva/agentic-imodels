import numpy as np
import gc

gc.enable()

from okglm.helpers.general_helpers import to_cpu


class Node:
    def __init__(self, parent, zlb, zub, **kwargs):
        """Initialize a node in the branch and bound tree.

        Args:
            parent (CustomClass): parent node
            zlb (np.array): 1D array indicating lower bound of each binary variable
            zub (np.array): 1D array indicating upper bound of each binary variable
        """

        self.data = kwargs.get("data", parent.data if parent else None)
        self.inherit_parent_upper_sol = kwargs.get("inherit_parent_upper_sol", False)
        self.inherit_parent_lower_sol = kwargs.get("inherit_parent_lower_sol", False)
        if parent is not None:
            self.parent_lower_bound = parent.lower_bound
        else:
            self.parent_lower_bound = -1e8

        self.level = parent.level + 1 if parent else 0

        if zlb is not None:
            self.zlb = zlb
        else:
            self.zlb = np.zeros((self.data.p,), dtype=bool)
        if zub is not None:
            self.zub = zub
        else:
            self.zub = np.ones((self.data.p,), dtype=bool)

        if self.inherit_parent_upper_sol:
            self.upper_bound = parent.upper_bound
            self.upper_beta = parent.upper_beta
            self.upper_r = parent.upper_r
        else:
            self.upper_bound = None
            self.upper_beta = None
            self.upper_r = None

        if self.inherit_parent_lower_sol:
            # self.lower_bound = parent.lower_bound
            self.lower_beta = parent.lower_beta.copy()
            self.lower_bound = parent.lower_bound
        else:
            self.lower_beta = np.zeros((self.data.p,))
            self.lower_bound = -1e12

        self.allowed_support = np.flatnonzero(self.zub)
        self.fixed_support_on_allowed_support = np.flatnonzero(self.zlb[self.allowed_support])
        self.unfixed_support_on_allowed_support = np.flatnonzero(~self.zlb[self.allowed_support])

    def delete_storedData_on_allowed_support(self):
        """Delete the stored data on the allowed support to save memory."""
        del self.allowed_support
        del self.fixed_support_on_allowed_support
        del self.unfixed_support_on_allowed_support

    def clear_solution_buffers(self):
        """Drop solution arrays after a node is fully processed to free memory."""
        self.upper_beta = self.upper_r = self.lower_beta = None

    def solve_upper_bound_with_cache(self, k, upper_solver):
        """Solve the k-sparse ridge regression for the current node using heuristic method

        Args:
            k (int): cardinality constraint
            upper_solver (CustomClass): solver that can search for the k-sparse solution

        Returns:
            float: loss of the k-sparse solution
        """

        upper_solver.reset_fixed_supp_and_allowed_supp(self.zlb, self.zub)

        upper_solver.get_sparse_sol_via_OMP(k=k)

        self.upper_beta = upper_solver.get_beta()
        self.upper_bound = upper_solver.get_loss()

        return self.upper_bound

    def solve_lower_bound_with_PGM(self, upper_bound, lower_solver):
        # lower_solver.reset_beta handles CPU->GPU transfer when use_gpu=True.
        lower_solver.reset_beta(self.lower_beta)
        lower_solver.reset_z_lb_and_z_ub(self.zlb, self.zub)

        lower_solver.solve(upper_bound=upper_bound, tol=1e-4)

        lower_beta = lower_solver.get_solution()
        self.lower_bound = lower_solver.get_dual_loss(lower_beta)
        self.lower_beta = to_cpu(lower_beta)

        # print("lower_beta is", self.lower_beta)
        # print("nonzero indices of lower_beta are", np.nonzero(self.lower_beta))
        # print("nonzero lower_beta are", self.lower_beta[np.nonzero(self.lower_beta)])
        # print("lower_bound is", self.lower_bound)
        # print("upper_bound is", upper_bound)
        # sys.exit()

        return self.lower_bound

def new_z(node, index):
    """Create two new z vectors by branching on the index-th variable

    Args:
        node (CustomClass): parent node
        index (int): index of the variable to branch on

    Returns:
        np.array: 1D array of left z vector
        np.array: 1D array of right z vector
    """
    new_zlb = node.zlb.copy()
    new_zub = node.zub.copy()
    new_zlb[index] = 1
    new_zub[index] = 0
    return new_zlb, new_zub


def branch(current_node, upper_solver):
    """Branch on the current node

    Args:
        current_node (CustomClass): parent node
        k (int): cardinality constraint

    Returns:
        CustomClass: left child node
        CustomClass: right child node
    """
    unfixed_and_allowed_support = current_node.allowed_support[
        current_node.unfixed_support_on_allowed_support
    ]
    nonzero_unfixed_and_allowed_support = unfixed_and_allowed_support[np.flatnonzero(current_node.upper_beta[unfixed_and_allowed_support])]

    intermediate_var = upper_solver.get_intermediate_var()
    grad_beta = upper_solver.GLMOptimizer.data_class.get_grad_F(intermediate_var)
    delta_beta = -current_node.upper_beta[nonzero_unfixed_and_allowed_support]
    increase_in_loss = (
        0.5 * upper_solver.GLMOptimizer.L[nonzero_unfixed_and_allowed_support]
        * delta_beta ** 2
        + grad_beta[nonzero_unfixed_and_allowed_support] * delta_beta
    )
    branching_variable = nonzero_unfixed_and_allowed_support[
        np.argmax(increase_in_loss)
    ]

    new_zlb, new_zub = new_z(current_node, branching_variable)
    # right_node = Node(current_node, new_zlb, current_node.zub.copy(), inherit_parent_upper_sol=(len(current_node.fixed_support_on_allowed_support) == k-1))
    right_node = Node(current_node, new_zlb, current_node.zub.copy(), inherit_parent_lower_sol=True)
    left_node = Node(current_node, current_node.zlb.copy(), new_zub, inherit_parent_lower_sol=True)
    return left_node, right_node
