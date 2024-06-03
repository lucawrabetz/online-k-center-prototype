from typing import List, Any
from allowed_types import FLInstanceType, FLSolverType


class FLExperiment:
    def __init__(self):
        pass

    def configure_experiment(
        self,
        instances: List[FLInstanceType],
        solvers: List[FLSolverType],
        params: Any = None,
    ):
        """
        params = None means use all params for all solvers
        """
        pass

    def run(self):
        pass
