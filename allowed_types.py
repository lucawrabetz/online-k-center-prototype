import logging
from log_config import _LOGGER
from typing import List, Dict


class FLSolverType:
    """
    Class to store all (id) information about a solver type:
    """

    def __init__(self, name: str, parameters: Dict[str, str] = {}) -> None:
        self._name: str = name
        self._parameters: Dict[str, str] = parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> Dict[str, str]:
        return self._parameters


class FLInstanceType:
    """
    Class to store all (id) information about an instance type:
    """

    def __init__(self, n: int = 2, T: int = 3) -> None:
        self.n: int = n
        self.T: int = T

    def print(self) -> None:
        _LOGGER.log_body(f"n: {self.n}")
        _LOGGER.log_body(f"T: {self.T}")


_OffMIP = FLSolverType("OffMIP")
_StMIP = FLSolverType("StMIP")
_CVCTA = FLSolverType("CVCTA")
_SOLVERS = [_OffMIP, _StMIP, _CVCTA]
