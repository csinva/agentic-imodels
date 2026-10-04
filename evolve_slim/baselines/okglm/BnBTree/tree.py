import time
import queue
import sys

import numpy as np

from .node import Node, branch  # , presolve


from .beamsearch_class import (
    data_LinearRegression as BeamSearch_data_LinearRegression,
    data_LogisticRegression as BeamSearch_data_LogisticRegression,
    BeamSearchOptimizer
)

from okglm.composite_cvx_opt_class_data import (
    data_LinearRegression_gpu as GLM_data_LinearRegression,
    data_LogisticRegression_gpu as GLM_data_LogisticRegression,
)
from okglm.BnBTree.bnb_lower_bound_factory import build_bnb_lower_solver

import gc
from okglm.utils import (
    get_RAM_available_in_GB,
    get_RAM_used_in_GB,
)


class TreeDataClass:
    def __init__(self, X, y, lambda2, k, M):
        """Create a DataClass object used to store data for the BnBTree class

        Args:
            X (np.array): n x p numpy array of features
            y (np.array): 1 dimensional numpy array of size n of predictions
            lambda2 (float): coefficient of l2 regularization
        """
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.lambda2 = lambda2
        self.k = k
        self.M = M

class BNBTree:
    def __init__(
        self,
        X,
        y,
        k=10,
        lambda2=1e-5,
        max_memory_GB=300,
        M=10,
        GLMLossType="linear",
        beamsize=10,
        lower_bound_method="oursPDRestartedBeckFISTALineSearche3",
    ):
        """Initialize the BnBTree class with the data and parameters

        Args:
            X (np.array): n x p numpy array of features
            y (np.array): 1 dimensional numpy array of size n of predictions
            lambda2 (float, optional): coefficient of l2 regularization. Defaults to 1e-5.
            max_memory_GB (int, optional): max memory to use to store all unprocessed nodes and heuristic solutions. Defaults to 300.
            useBruteForce (bool, optional): use brute force search when the number of enumeration is small. Defaults to False.
            tighten_bound_via_ADMM (bool, optional): whether to use ADMM to tighten the lower bound computed by the Fast Solve method. Defaults to True.
        """
        self._initialize_loss_classes(GLMLossType)
        self.tree_data = TreeDataClass(X, y, lambda2, k, M)

        self._initialize_memory_limits(max_memory_GB)

        self._initialize_upper_solver(self.tree_data, beamsize)
        self.lower_bound_method = lower_bound_method
        self._initialize_lower_solver(self.tree_data, lower_bound_method)
        self._initialize_bnb_variables()

    def get_RAM_used_since_start(self):
        return get_RAM_used_in_GB() - self.RAM_used_GB_start

    def _initialize_memory_limits(self, max_memory_GB):
        available_memory_GB = get_RAM_available_in_GB()
        if max_memory_GB is None:
            print("No max_memory_GB is given. Using all available memory ({} GB) in the machine".format(available_memory_GB))
            self.max_memory_GB = available_memory_GB
        elif max_memory_GB > available_memory_GB:
            print("max_memory_GB is larger than available memory. Using all available memory ({} GB) in the machine".format(available_memory_GB))
            self.max_memory_GB = available_memory_GB
        else:
            print("Using max memory ({} GB)".format(max_memory_GB))
            self.max_memory_GB = max_memory_GB
        self.safe_max_memory_GB = 0.95 * self.max_memory_GB
        self.RAM_used_GB_start = get_RAM_used_in_GB()

    def _initialize_loss_classes(self, GLMLossType):
        if GLMLossType == "linear":
            self.BeamSearch_data_class = BeamSearch_data_LinearRegression
            self.lower_solver_GLM_data_class = GLM_data_LinearRegression
            return
        if GLMLossType == "logistic":
            self.BeamSearch_data_class = BeamSearch_data_LogisticRegression
            self.lower_solver_GLM_data_class = GLM_data_LogisticRegression
            return
        raise ValueError("GLMLossType must be either 'linear' or 'logistic'")

    def _initialize_upper_solver(self, tree_data, beamsize):
        self.upper_solver_with_cache = BeamSearchOptimizer(
            tree_data.X,
            tree_data.y,
            tree_data.lambda2,
            tree_data.M,
            self.BeamSearch_data_class,
            parent_size=beamsize,
            child_size=beamsize,
            max_memory_GB=50,
        )

    def _initialize_lower_solver(self, tree_data, lower_bound_method):
        self.lower_solver = build_bnb_lower_solver(
            lower_bound_method,
            tree_data,
            self.lower_solver_GLM_data_class,
        )

    def _initialize_bnb_variables(self):
        self.root = Node(None, None, None, data=self.tree_data)
        self.bfs_queue = queue.Queue()
        self.dfs_queue = queue.LifoQueue()
        self.bfs_queue.put(self.root)
        self.num_lower_bound_solves = 0
        self.best_upper_bound = None
        self.best_beta = None
        self.start_time = None
        self.min_lower_bounds_per_level = {}
        self.num_open_nodes_by_level = {0: 1}
        self.min_open_level = 0
        self.bnb_lower_bound = -sys.maxsize
        self.bnb_gap = sys.maxsize

    def reset_start_time(self):
        self.start_time = time.time()

    def reset(self):
        """Reset mutable state so solve() starts from a clean slate."""
        self._initialize_memory_limits(self.max_memory_GB)

        if self.upper_solver_with_cache is not None:
            self.upper_solver_with_cache.reset_state()
        if self.lower_solver is not None:
            self.lower_solver.reset_beta(np.zeros(self.tree_data.p))
        self._initialize_bnb_variables()

    def _maybe_collect_garbage(self, RAM_used_GB_since_start):
        if RAM_used_GB_since_start >= 0.9 * self.safe_max_memory_GB or self.num_lower_bound_solves % 50 == 0:
            gc.collect()

    def _queues_have_nodes(self):
        return self.bfs_queue.qsize() > 0 or self.dfs_queue.qsize() > 0

    def _next_node(self):
        return self.dfs_queue.get() if self.dfs_queue.qsize() > 0 else self.bfs_queue.get()

    def _should_prune_by_parent(self, curr_node):
        return curr_node.parent_lower_bound and self.best_upper_bound <= curr_node.parent_lower_bound

    def _update_best_upper_bound_and_beta(self, curr_node):
        if curr_node.upper_bound < self.best_upper_bound:
            self.best_upper_bound = curr_node.upper_bound
            self.best_beta = curr_node.upper_beta.copy()
            self.bnb_gap = (self.best_upper_bound - self.bnb_lower_bound) / abs(self.best_upper_bound)

    def _update_min_lower_bound_per_level(self, curr_node, curr_lower_bound):
        self.min_lower_bounds_per_level[curr_node.level] = min(curr_lower_bound, self.min_lower_bounds_per_level.get(curr_node.level, sys.maxsize))
        self.num_open_nodes_by_level[curr_node.level] -= 1

    def _print_BnB_tree_status_header(self):
        print(
            "'l' -> level(depth) of BnB tree, ",
            "'d' -> best dual bound, ",
            "'u' -> best upper(primal) bound, ",
            "'g' -> optimiality gap, ",
            "'# nodes' -> number of nodes processed, ",
            "'t' -> time",
        )

    def _print_BnB_tree_status_line(self, level):
        print(
            "l: {}, ".format(level).ljust(8),
            "d: {:.10f}, ".format(self.bnb_lower_bound).ljust(25),
            "u: {:.10f}, ".format(self.best_upper_bound).ljust(25),
            "g: {:.10f}, ".format(self.bnb_gap).ljust(15),
            "# nodes: {}".format(self.num_lower_bound_solves).ljust(12),
            "t: {:.5f} s".format(time.time() - self.start_time).ljust(12),
        )

    def _update_bnb_lower_bound_and_gap_if_level_closed(self):
        if self.num_open_nodes_by_level[self.min_open_level] == 0:
            del self.num_open_nodes_by_level[self.min_open_level]
            self.bnb_lower_bound = max(j for i, j in self.min_lower_bounds_per_level.items() if i <= self.min_open_level)
            self.bnb_gap = (self.best_upper_bound - self.bnb_lower_bound) / abs(self.best_upper_bound)
            self.min_open_level += 1

    def _ask_node_to_branch(self, curr_node, gap_tol, number_of_dfs_levels):
        curr_gap = (curr_node.upper_bound - curr_node.lower_bound) / abs(curr_node.upper_bound)
        if curr_gap <= gap_tol:
            return
        if curr_node.lower_bound < self.best_upper_bound and len(curr_node.fixed_support_on_allowed_support) < self.tree_data.k:
            left_node, right_node = branch(curr_node, self.upper_solver_with_cache)
            self.num_open_nodes_by_level[curr_node.level + 1] = self.num_open_nodes_by_level.get(curr_node.level + 1, 0) + 2 # two new nodes added; if level not exist, initialize to 0 first
            if curr_node.level < self.min_open_level + number_of_dfs_levels:
                self.dfs_queue.put(right_node)
                self.dfs_queue.put(left_node)
            else:
                self.bfs_queue.put(right_node)
                self.bfs_queue.put(left_node)

    def _ask_node_to_solve_upper_bound(self, curr_node):
        return curr_node.solve_upper_bound_with_cache(self.tree_data.k, self.upper_solver_with_cache)

    def _ask_node_to_solve_lower_bound(self, curr_node):
        """Solve the perspective relaxation for a node and return a lower bound.

        Args:
            curr_node (CustomClass): current node to be solved

        Returns:
            float: dual value of the perspective relaxation of the current node
        """
        time_start = time.time()
        curr_lower_bound = curr_node.solve_lower_bound_with_PGM(self.best_upper_bound, self.lower_solver)
        print("Restarted FISTA method time used is {}".format(time.time() - time_start))

        return curr_lower_bound

    def _pack_results(self):
        return (
            self.best_upper_bound,
            self.best_beta,
            self.bnb_gap,
            self.bnb_lower_bound,
            self.num_lower_bound_solves,
            time.time() - self.start_time,
        )

    def solve(
        self,
        gap_tol=1e-2,
        number_of_dfs_levels=0,
        verbose=False,
        time_limit=3600,
    ):
        """Solve the k-sparse ridge regression problem using branch and bound

        Args:
            gap_tol (float, optional): optimality gap tolerance hyperparameter. Defaults to 1e-2.
            number_of_dfs_levels (int, optional): number of levels for depth-first search during branch and bound. Defaults to 0.
            verbose (bool, optional): whether to print informations into terminal. Defaults to False.
            time_limit (int, optional): time limit (in seconds) of running branch and bound. Defaults to 3600.

        Returns:
            float: cost or loss of the best solution found
            np.array: best solution found
            float: best gap found
            float: best lower bound found
            float: running time of branch and bound
        """

        self.reset_start_time()

        if verbose:
            if number_of_dfs_levels > 0:
                print("using depth-first search for the first {} levels".format(number_of_dfs_levels))
            else:
                print("using breadth-first search")

        self.best_upper_bound = self._ask_node_to_solve_upper_bound(self.root)
        self.best_beta = self.root.upper_beta.copy()
        RAM_used_GB_since_start = self.get_RAM_used_since_start()

        self._print_BnB_tree_status_header()

        # keep searching through the queue if the queue is not empty AND time limit is not reached AND RAM used is within limit
        while (
            self._queues_have_nodes()
            and (time.time() - self.start_time < time_limit)
            and (RAM_used_GB_since_start < self.safe_max_memory_GB)
        ):

            RAM_used_GB_since_start = self.get_RAM_used_since_start()

            curr_node = self._next_node()

            # Prune when a better incumbent makes this subtree provably suboptimal via the parent's lower bound.
            if self._should_prune_by_parent(curr_node):
                self.num_open_nodes_by_level[curr_node.level] -= 1
                curr_node.clear_solution_buffers()
                curr_node.delete_storedData_on_allowed_support()
                continue

            self.num_lower_bound_solves += 1
            self._maybe_collect_garbage(RAM_used_GB_since_start)
            curr_lower_bound = self._ask_node_to_solve_lower_bound(curr_node)
            self._update_min_lower_bound_per_level(curr_node, curr_lower_bound)

            # Prune when the newly calculated node lower bound is already worse than the best known upper bound.
            if curr_lower_bound >= self.best_upper_bound:
                curr_node.clear_solution_buffers()
                curr_node.delete_storedData_on_allowed_support()
                continue

            self._ask_node_to_solve_upper_bound(curr_node)
            self._update_best_upper_bound_and_beta(curr_node)

            prev_min_open_level = self.min_open_level
            self._update_bnb_lower_bound_and_gap_if_level_closed()
            if verbose and self.min_open_level != prev_min_open_level:
                self._print_BnB_tree_status_line(prev_min_open_level)

            if self.bnb_gap <= gap_tol:
                return self._pack_results()

            self._ask_node_to_branch(curr_node, gap_tol, number_of_dfs_levels,)

            curr_node.clear_solution_buffers()
            curr_node.delete_storedData_on_allowed_support()

        if not self._queues_have_nodes():
            # update lower bound and gap
            self.bnb_lower_bound = self.best_upper_bound
            self.bnb_gap = 0.0
        elif RAM_used_GB_since_start >= self.safe_max_memory_GB:
            print("RAM used since start is greater than 0.95*max_memory(0.95*{} GB = {} GB)!".format(self.max_memory_GB, self.safe_max_memory_GB))
        else:
            print("Time limit is reached!")
        return self._pack_results()
