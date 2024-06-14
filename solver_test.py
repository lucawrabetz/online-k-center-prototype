import pytest
from typing import List, Dict
from solvers import IFLSolver
from allowed_types import _SOLVERS

# 1. x_0 = (0, 1), x_1 = (1, 1), x_2 = (1, 0)
# (a). gamma = 0: solution should be build all facilities: objective = 0.0 (for all solvers).
# (b). gamma = 100: solution should be build no facilities and serve all points with x_0:
# - objective (same for all solvers since "empty" facility set for all 3): (1) + (1) = 2
# - \sum_t max_min_service_cost(t)


class HomogeneousSolverTestSuite:
    """
    Homogeneous - as in the correct solution is the same for all solvers.
    For now, just take a set name. The set should only have one instance,
    so we can just use the set name to get the instance, and set up the correct solutions in this class in a dictionary keyed by the parameter gamma used for the instance.
    """

    def __init__(self, set_name: str):
        self.solvers: List[IFLSolver] = _SOLVERS
        self.set_name: str = set_name
        self.gamma_to_correct_objective: Dict[float, float]


def main():
    test_suite = HomogeneousSolverTestSuite("test")


if __name__ == "__main__":
    main()
