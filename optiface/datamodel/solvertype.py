import logging
from typing import Any, Dict, List

from optiface import logger
from optiface.datamodel import feature


class SolverType:
    """
    Class to store all "lightweight" non-functional metadata about a solver type:
        - the convention when an instance of this class is a member of a solver class to identify it is to store it in the _typeid attribute, to distinguish from _id, which we will reserve for repetitions of the same type (for example with instances)
    """

    def __init__(self, name: str, parameters: List[feature.Feature] = []) -> None:
        self._name: str = name
        self._parameters: List[feature.Feature] = parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> List[feature.Feature]:
        return self._parameters

    def print(self) -> None:
        logger.log_body(f"Name: {self.name}")
        logger.log_body(f"Parameters:")
        for param in self.parameters:
            param.print()


def main():
    GUROBI_SYMMETRY = feature.Feature("g_sym", 0, int, "Gurobi Symmetry", "GSym")
    MIP = SolverType("MIP", [])
    SMIP = SolverType("SMIP", [GUROBI_SYMMETRY])
    MIP.print()
    SMIP.print()


if __name__ == "__main__":
    main()
