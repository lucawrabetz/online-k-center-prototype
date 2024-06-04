import os
import logging
from log_config import _LOGGER
from typing import List, Dict, Any
from util import _DAT


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

    def __init__(self, set_name: str = "test", n: int = 2, T: int = 3) -> None:
        self.set_name: str = set_name
        self.n: int = n
        self.T: int = T
        # Unique identifier if all instance parameters already exist in another instance, only set for saved instances.
        self.id: int = -1

    def from_filename(self, filename: str) -> None:
        split = filename.split("_")
        if len(split) != 4:
            raise ValueError("Invalid filename")
        self.set_name = split[0]
        self.n = int(split[1])
        self.T = int(split[2])
        self.id = int(split[3].replace(".csv", ""))

    def file_path(self) -> str:
        if self.id is not None:
            return os.path.join(
                _DAT, self.set_name, f"{self.set_name}_{self.n}_{self.T}_{self.id}.csv"
            )
        else:
            # TODO: decouple/wrap instead of checking.
            raise ValueError("Instance has no id so is not saved.")

    def print(self) -> None:
        _LOGGER.log_body(f"n: {self.n}")
        _LOGGER.log_body(f"T: {self.T}")


_TEST_SHAPE = FLInstanceType()

_OffMIP = FLSolverType("OffMIP")
_StMIP = FLSolverType("StMIP")
_CVCTA = FLSolverType("CVCTA")
_SOLVERS: List[Any] = [_OffMIP, _StMIP, _CVCTA]

_BUILT_IN_TYPES: List[Any] = [str, int, float, bool]
_ALLOWED_TYPES: List[Any] = _SOLVERS.copy()
_ALLOWED_TYPES.extend(_BUILT_IN_TYPES)
