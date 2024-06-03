import logging
from log_config import _LOGGER


class FLSolverType:
    """
    Class to store all information about a solver type:
    """


class FLInstanceType:
    """
    Class to store all information about an instance type:
    """

    def __init__(self, n: int = 2, T: int = 3) -> None:
        self.n: int = n
        self.T: int = T

    def print(self) -> None:
        _LOGGER.log_body(f"n: {self.n}")
        _LOGGER.log_body(f"T: {self.T}")
