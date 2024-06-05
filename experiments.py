import os
import pandas as pd
import logging
from typing import Dict, List, Any
from allowed_types import FLInstanceType, FLSolverType
from allowed_types import _StMIP, _OffMIP, _CVCTA
from problem import FLOfflineInstance, FLSolution
from solvers import OfflineMIP, OnlineCVCTAlgorithm, StaticMIP, _SOLVER_FACTORY
from solvers import IFLSolver
from data_model import _DATA_MODEL, OBJECTIVE, TIME
from feature_interface import IFeature
from log_config import _LOGGER
from util import _DAT


class OutputRow:
    def __init__(self, row: Dict[IFeature, Any] = None) -> None:
        self.row: Dict[IFeature, Any] = row if row is not None else {}

    def from_run(
        self, instance: FLOfflineInstance, solver: IFLSolver, solution: FLSolution
    ) -> None:
        self.row = {}
        # TODO...
        for feature in _DATA_MODEL.features:
            if feature.name == "set_name":
                self.row[feature] = instance.id.set_name
            elif feature.name == "n":
                self.row[feature] = instance.id.n
            elif feature.name == "id":
                # TODO: refactor id.id to <something_else>.id throughout codebase
                self.row[feature] = instance.id.id
            elif feature.name == "time_periods":
                self.row[feature] = instance.id.T
            elif feature.name == "solver":
                self.row[feature] = solver.id.name
            elif feature.name == "objective":
                self.row[feature] = solution.objective
            elif feature.name == "unbounded":
                self.row[feature] = solution.unbounded
            elif feature.name == "optimal":
                self.row[feature] = solution.optimal
            elif feature.name == "time":
                self.row[feature] = solution.running_time_ms
            elif feature.name == "time_s":
                self.row[feature] = solution.running_time_s

    def series(self) -> pd.Series:
        data = {}
        for feature, value in self.row.items():
            data[feature.name] = value
        return pd.Series(data)

    def validate(self) -> bool:
        return True


class OutputTable:
    def __init__(self) -> None:
        self.rows: List[OutputRow] = []

    def add_row(self, row: OutputRow) -> None:
        if not row.validate():
            raise ValueError("Row is not valid")
        self.rows.append(row.series())

    def dataframe(self) -> pd.DataFrame:
        data = {}
        for row in self.rows:
            for feature, value in row.row.items():
                data[feature.name] = value
        return pd.DataFrame(self.rows)


class CSVWrapper:
    def __init__(self, path: str) -> None:
        self.path: str = path

    def write_line(self, table: OutputTable) -> None:
        table.dataframe().to_csv(self.path, index=False)


class FLRuns:
    def __init__(
        self, instance_id: FLInstanceType, solver_ids: List[FLSolverType]
    ) -> None:
        self.instance_id: FLInstanceType = instance_id
        self.solver_ids: List[FLSolverType] = solver_ids

    def single_run(
        self, solver_id: FLSolverType, instance: FLOfflineInstance
    ) -> OutputRow:
        solver = _SOLVER_FACTORY.solver(solver_id)
        solver.configure_solver(instance)
        solution: FLSolution = solver.solve(instance)
        row = OutputRow()
        row.from_run(instance, solver, solution)
        return row

    def run(self) -> OutputTable:
        instance = FLOfflineInstance(self.instance_id)
        instance.read()
        table = OutputTable()
        _LOGGER.log_body(
            f"Running solvers {', '.join([s.name for s in self.solver_ids])} on instance {self.instance_id.file_path()}"
        )
        summary: List[str] = []
        for solver_id in self.solver_ids:
            row = self.single_run(solver_id, instance)
            table.add_row(row)
            summary.append(
                f"{solver_id.name}: {row.row[OBJECTIVE]}, time (ms): {row.row[TIME]}"
            )
        _LOGGER.log_body("; ".join(summary))
        _LOGGER.separator_line()
        return table


class FLExperiment:
    def __init__(self, set_name: str) -> None:
        self.set_name: str = set_name
        self.data_path: str = os.path.join(_DAT, set_name)
        self.instance_ids: List[FLInstanceType] = []
        self.solver_ids: List[FLSolverType] = []
        self.solvers: List[IFLSolver] = []

    def configure_experiment(
        self,
        instance_ids: List[FLInstanceType] = None,
        solver_ids: List[FLSolverType] = None,
        params: Any = None,
    ):
        """
        initially:
            instances: just all files in directory dat/set_name
            solvers: passed explicitly or just stmip
        """
        filenames = os.listdir(self.data_path)
        for filename in filenames:
            instance_id = FLInstanceType()
            instance_id.from_filename(filename)
            self.instance_ids.append(instance_id)
        if solver_ids:
            self.solver_ids = solver_ids
        else:
            self.solver_ids = [_StMIP]

    def run(self):
        _LOGGER.log_header(f"Running experiment for set {self.set_name}")
        for instance_id in self.instance_ids:
            _LOGGER.log_subheader(
                f"Running for instance {instance_id.file_path()}  ---> T = {instance_id.T}, n = {instance_id.n}"
            )
            # in here will require some sort of registry
            # solver id -> factory method to construct solver
            run = FLRuns(instance_id, self.solver_ids)
            table = run.run()
