import pandas as pd
from typing import Any, List
from feature_interface import IFeature
from util import _FINALDB
from allowed_types import FLSolverType, _OMIP, _SOLVERS


def get_next_run_id() -> int:
    """
    Get the next run ID by reading the final database.
    """
    try:
        df = pd.read_csv(_FINALDB)
        return df["run_id"].max() + 1
    except FileNotFoundError:
        return 0


class DataModel:
    def __init__(self, features: List[IFeature]) -> None:
        self.features: List[IFeature] = features


# Instance parameter features
# TODO: refactor RUN_ID = migrate to databases
RUN_ID = IFeature("run_id", 0, int, "Run ID", "Run ID")
SET_NAME = IFeature("set_name", "test", str, "Set Name", "Set")
DIMENSION = IFeature("n", 2, int, "n", "n")
ID = IFeature("id", 0, int, "ID", "ID")
# TODO: TIME_PERIODS and GAMMA is a good example of potential run vs instance parameter
TIME_PERIODS = IFeature("T", 2, int, "Time Periods", "T")
GAMMA = IFeature("Gamma", 2, int, "Gamma", "Gamma")
# if row[GAMMA_RUN] = -1, then gamma = row[GAMMA], otherwise
# gamma = row[GAMMA_RUN]
GAMMA_RUN = IFeature("Gamma_run", -1, int, "Gamma", "Gamma")
TIME_PERIODS_RUN = IFeature("T_run", -1, int, "Time Periods", "T")

# SOLVER is REQUIRED.
SOLVER = IFeature("solver", _OMIP, FLSolverType, "Solver", "Sol", _SOLVERS)

# Outputs
OBJECTIVE = IFeature("objective", 0.0, float, "Objective", "Obj")
UNBOUNDED = IFeature("unbounded", "NOT_UNBOUNDED", str, "Unbounded", "Unb")
OPTIMAL = IFeature("optimal", "OPTIMAL", str, "Optimal", "Opt")
TIME = IFeature("time", 0.0, float, "Running Time (ms)", "T (ms)")
TIME_S = IFeature("time_s", 0.0, float, "Running Time (s)", "T (s)")
IT_TIME = IFeature("it_time", -1.0, float, "Iteration Time (ms)", "It T (ms)")

features = [
    SET_NAME,
    DIMENSION,
    ID,
    TIME_PERIODS,
    TIME_PERIODS_RUN,
    GAMMA,
    GAMMA_RUN,
    SOLVER,
    OBJECTIVE,
    UNBOUNDED,
    OPTIMAL,
    TIME,
    TIME_S,
    IT_TIME,
]
_DATA_MODEL = DataModel(features)
