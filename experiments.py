import os
import csv
import pandas as pd
import logging
from typing import Dict, List, Any
from allowed_types import FLInstanceType, FLSolverType, _OMIP, _SOMIP, _CCTA
from problem import FLOfflineInstance, FLSolution
from solvers import IFLSolver, OfflineMIP, SemiOfflineMIP, CCTAlgorithm, _SOLVER_FACTORY
from data_model import RUN_ID, _DATA_MODEL, OBJECTIVE, TIME, get_next_run_id
from feature_interface import IFeature
from log_config import _LOGGER, throwaway_gurobi_model
from util import _DAT, _FINALDB, _SERVICEDB, _TIMEDB


class OutputRow:
    def __init__(self, row: Dict[IFeature, Any] = None) -> None:
        self.row: Dict[IFeature, Any] = row if row is not None else {}

    def from_run(
        self,
        run_id: int,
        instance: FLOfflineInstance,
        solver: IFLSolver,
        solution: FLSolution,
    ) -> None:
        self.row = {}
        # TODO... so hacky, everything clicks if you add a new feature to data model except for this
        self.row[RUN_ID] = run_id
        for feature in _DATA_MODEL.features:
            if feature.name == "set_name":
                self.row[feature] = instance.id.set_name
            elif feature.name == "n":
                self.row[feature] = instance.id.n
            elif feature.name == "id":
                # TODO: refactor id.id to <something_else>.id throughout codebase
                self.row[feature] = instance.id.id
            elif feature.name == "T":
                self.row[feature] = instance.id.T
            elif feature.name == "T_run":
                # TODO: add solver suite functionality for decoupled T_run, and refactor gamma_run and T_run
                self.row[feature] = -1
            # Workflow for decoupled run / instance parameter.
            # Defaulting the decoupled run proxy to -1 essentially takes care of everything. Original parameter (gamma) is updated if the run parameter is set, so all functionality still calls gamma and works for the run. We achieve this by maintaining 3 gamma attributes on the problem class because gamma is a "decoupled parameter", so one attribute is for the instance parameter, one is for the run_parameter when the instance is being used for a run, and one is for actual run functionality (which we don't look at here.)
            elif feature.name == "Gamma_run":
                self.row[feature] = instance.set_Gamma
            elif feature.name == "Gamma":
                self.row[feature] = instance.original_Gamma
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
            elif feature.name == "it_time_ms":
                self.row[feature] = solution.iteration_time_ms

    def validate(self) -> bool:
        return True


class OutputTable:
    def __init__(self) -> None:
        self.rows: List[OutputRow] = []

    def add_row(self, row: OutputRow) -> None:
        if not row.validate():
            raise ValueError("Row is not valid")
        self.rows.append(row)

    def dataframe(self) -> pd.DataFrame:
        series_data = []
        for row in self.rows:
            data = {}
            for feature, value in row.row.items():
                data[feature.name] = value
            series_data.append(pd.Series(data))
        return pd.DataFrame(series_data)


class CSVWrapper:
    def __init__(self, path: str = _FINALDB) -> None:
        self.path: str = path

    def write_table(self, table: OutputTable) -> None:
        # TODO: port in and refactor from "sheet_cat.py" in ARSPI.
        new_df = table.dataframe()
        if os.path.exists(self.path):
            existing_df = pd.read_csv(self.path)
            if existing_df.shape[0] == 0:
                new_df.to_csv(self.path, index=False)
            if set(existing_df.columns) == set(new_df.columns):
                new_df.to_csv(self.path, mode="a", header=False, index=False)
            else:
                # Existing dataframe has data and columns do not match, raise an error
                raise ValueError("Columns do not match. Cannot append to the CSV file.")
        else:
            new_df.to_csv(self.path, index=False)


class HorizonCSVWrapper:
    def __init__(self, path: str = _SERVICEDB) -> None:
        self.path: str = path

    def write_horizon(self, horizon: List[Any], run_id: int) -> None:
        with open(self.path, "a", newline="") as file:
            writer = csv.writer(file)
            row = [run_id] + horizon
            writer.writerow(row)


class FLExperiment:
    def __init__(self, set_name: str) -> None:
        self.set_name: str = set_name
        self.data_path: str = os.path.join(_DAT, set_name)
        self.instance_ids: List[FLInstanceType] = []
        self.solver_ids: List[FLSolverType] = []
        self.solvers: List[IFLSolver] = []
        self.csv_wrapper: CSVWrapper = CSVWrapper()
        self.service_wrapper: HorizonCSVWrapper = HorizonCSVWrapper()
        self.time_wrapper: HorizonCSVWrapper = HorizonCSVWrapper(_TIMEDB)
        self.run_id: int = get_next_run_id()
        self.gamma: float = -1.0
        throwaway_gurobi_model()
        _LOGGER.clear_page()

    def configure_experiment(
        self,
        instance_ids: List[FLInstanceType] = None,
        solver_ids: List[FLSolverType] = None,
        gamma: float = None,
        params: Any = None,
    ):
        """
        initially:
            instances: just all files in directory dat/set_name
            solvers: passed explicitly or just fully offline mip
            gamma: use instance gamma
        """
        filenames = os.listdir(self.data_path)
        for filename in filenames:
            instance_id = FLInstanceType()
            instance_id.from_filename(filename)
            self.instance_ids.append(instance_id)
        if solver_ids:
            self.solver_ids = solver_ids
        else:
            self.solver_ids = [_OMIP]
        if gamma is not None:
            self.gamma = gamma

    def single_run(
        self, solver_id: FLSolverType, instance: FLOfflineInstance
    ) -> OutputRow:
        if self.gamma >= 0:
            instance.set_gamma_run(self.gamma)
        solver = _SOLVER_FACTORY.solver(solver_id)
        solver.configure_solver(instance)
        solution: FLSolution = solver.solve(instance)
        row = OutputRow()
        row.from_run(self.run_id, instance, solver, solution)
        self.service_wrapper.write_horizon(solution.service_costs, self.run_id)
        if solver_id == _CCTA:
            self.time_wrapper.write_horizon(solution.iteration_time_ms, self.run_id)
        self.run_id += 1
        return row

    def run(self) -> None:
        gamma_str: str = ""
        if self.gamma >= 0:
            gamma_str = str(self.gamma)
        else:
            gamma_str = "instance"
        _LOGGER.log_header(
            f"Running experiment for set {self.set_name} with {gamma_str} gamma"
        )
        for instance_id in self.instance_ids:
            instance = FLOfflineInstance(instance_id)
            instance.read()
            table = OutputTable()
            if len(self.solver_ids) == 1:
                _LOGGER.log_body(
                    f"Running solver {self.solver_ids[0].name} on instance {instance_id.file_path}"
                )
            else:
                _LOGGER.log_body(
                    f"Running solvers {', '.join([s.name for s in self.solver_ids])} on instance {instance_id.file_path}"
                )
            summary: List[str] = []
            _LOGGER.separator_line()
            for solver_id in self.solver_ids:
                row = self.single_run(solver_id, instance)
                table.add_row(row)
                summary.append(
                    f"{solver_id.name}: {row.row[OBJECTIVE]}, time (ms): {row.row[TIME]}"
                )
            _LOGGER.log_body("; ".join(summary))
            self.csv_wrapper.write_table(table)
            _LOGGER.separator_line()
            # _LOGGER.log_subheader(
            #     f"Running for instance {instance_id.file_path}  ---> T = {instance_id.T}, n = {instance_id.n}"
            # )
            # run = FLRuns(instance_id, self.solver_ids, self.csv_wrapper)
            # table = run.run()
