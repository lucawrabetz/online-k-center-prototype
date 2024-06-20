import logging
from typing import Any, Dict, List

from optiface import logger
from optiface.datamodel import feature


class SolverType:
    """
    Class to store all (id) information about a solver type:
        - the convention when an instance of this class is a member
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
