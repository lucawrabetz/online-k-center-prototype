from typing import Any, List
from feature_interface import IFeature
from allowed_types import FLSolverType, _StMIP, _SOLVERS


class DataModel:
    def __init__(self, features: List[IFeature]) -> None:
        self.features: List[IFeature] = features


# Instance parameter features
SET_NAME = IFeature("set_name", "test", str, "Set Name", "Set")
DIMENSION = IFeature("n", 2, int, "n", "n")
ID = IFeature("id", 0, int, "ID", "ID")
# TODO: TIME_PERIODS is a good example of potential run vs instance parameter
TIME_PERIODS = IFeature("time_periods", 2, int, "Time Periods", "T")

# SOLVER is REQUIRED.
SOLVER = IFeature("solver", _StMIP, FLSolverType, "Solver", "Sol", _SOLVERS)

# Outputs
OBJECTIVE = IFeature("objective", 0.0, float, "Objective", "Obj")
UNBOUNDED = IFeature("unbounded", "NOT_UNBOUNDED", str, "Unbounded", "Unb")
OPTIMAL = IFeature("optimal", "OPTIMAL", str, "Optimal", "Opt")
TIME = IFeature("time", 0.0, float, "Running Time (ms)", "T (ms)")
TIME_S = IFeature("time_s", 0.0, float, "Running Time (s)", "T (s)")

features = [
    SET_NAME,
    DIMENSION,
    ID,
    TIME_PERIODS,
    SOLVER,
    OBJECTIVE,
    UNBOUNDED,
    OPTIMAL,
    TIME,
    TIME_S,
]
_DATA_MODEL = DataModel(features)
