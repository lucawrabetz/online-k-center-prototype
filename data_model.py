import pandas as pd
from typing import Any, List, Dict
from feature_interface import IFeature
from util import _FINALDB
from allowed_types import FLSolverType, _OMIP, _SOLVERS

# opti-face


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
        self.features: Dict[str, IFeature] = {
            feature.name: feature for feature in features
        }


# implementer

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
PERMUTATION = IFeature(
    "perm", "none", str, "Permutation", "Perm", ["none", "start", "end", "full"]
)
PERMUTATION_ORDER = IFeature(
    "perm_order", "none", str, "Permutation Order", "Perm Order"
)

# SOLVER is REQUIRED.
SOLVER = IFeature("solver", _OMIP, FLSolverType, "Solver", "Sol", _SOLVERS)

# Outputs
OBJECTIVE = IFeature("objective", 0.0, float, "Objective", "Obj")
UNBOUNDED = IFeature("unbounded", "NOT_UNBOUNDED", str, "Unbounded", "Unb")
OPTIMAL = IFeature("optimal", "OPTIMAL", str, "Optimal", "Opt")
# unspecified time is in ms
# TODO: refactor time type
TIME = IFeature("time", -1.0, float, "Running Time (ms)", "T (ms)")
TIME_S = IFeature("time_s", -1.0, float, "Running Time (s)", "T (s)")
IT_TIME = IFeature("it_time", -1.0, float, "Iteration Time (ms)", "It T (ms)")
NUM_FACILITIES = IFeature("num_facilities", 0, int, "Number of Facilities", "K")
FACILITIES_STR = IFeature("facilities_str", "0", str, "Facilities", "Fac")

features = [
    SET_NAME,
    DIMENSION,
    ID,
    TIME_PERIODS,
    TIME_PERIODS_RUN,
    GAMMA,
    GAMMA_RUN,
    PERMUTATION,
    PERMUTATION_ORDER,
    SOLVER,
    OBJECTIVE,
    UNBOUNDED,
    OPTIMAL,
    TIME,
    TIME_S,
    IT_TIME,
    NUM_FACILITIES,
    FACILITIES_STR,
]
_COLUMN_INDEX = [
    RUN_ID.name,
    SET_NAME.name,
    DIMENSION.name,
    ID.name,
    TIME_PERIODS.name,
    TIME_PERIODS_RUN.name,
    GAMMA.name,
    GAMMA_RUN.name,
    PERMUTATION.name,
    PERMUTATION_ORDER.name,
    SOLVER.name,
    OBJECTIVE.name,
    UNBOUNDED.name,
    OPTIMAL.name,
    TIME.name,
    TIME_S.name,
    IT_TIME.name,
    NUM_FACILITIES.name,
    FACILITIES_STR.name,
]
_DATA_MODEL = DataModel(features)
