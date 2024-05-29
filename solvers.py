import sys
import time
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Tuple
from gurobipy import Model, GRB, quicksum
from problem import FLOfflineInstance, FLSolution, CVCTState
from log_config import gurobi_log_file, _LOGGER
from util import append_date

GRBVar = Any


class IFLSolver(ABC):
    NAME: str

    def __init__(self) -> None:
        self.T: int = 0
        self.objective: float = 0.0
        self.running_time_s: float = 0.0
        self.optimal: bool = False
        pass

    @abstractmethod
    def configure_solver(self, instance: FLOfflineInstance) -> None:
        pass

    @abstractmethod
    def solve(self, instance: FLOfflineInstance) -> FLSolution:
        pass


class OfflineMIP(IFLSolver):
    NAME = "OffMIP"

    def __init__(self) -> None:
        """Constructs blank gurobi model."""
        super().__init__()
        self.model: Model = Model("OfflineMIP")
        self.model.setParam("LogToConsole", 0)
        self.model.setParam("LogFile", gurobi_log_file())
        self.zeta: List[GRBVar] = []
        self.z: List[GRBVar] = []
        self.y: List[GRBVar] = []

    def add_variables(self, instance: FLOfflineInstance) -> None:
        self.y = [
            [
                self.model.addVar(vtype=GRB.BINARY, name=f"y^{t}_{j}")
                for j in range(1, t + 1)
            ]
            for t in range(1, self.T + 1)
        ]
        self.z = [
            [
                [
                    self.model.addVar(vtype=GRB.BINARY, name=f"z^{t}_{i}_{j}")
                    for j in range(0, t + 1)
                ]
                for i in range(1, t + 1)
            ]
            for t in range(1, self.T + 1)
        ]
        self.zeta = [
            self.model.addVar(vtype=GRB.CONTINUOUS, name=f"zeta^{t}")
            for t in range(1, self.T + 1)
        ]
        self.model.update()

    def add_zetalowerbound_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(
                    self.zeta[t_index]
                    >= quicksum(
                        instance.get_distance(i, j) * self.z[t_index][i_index][j]
                        for j in range(t + 1)
                    ),
                    name=f"C_zeta_lb_{t}_{i}",
                )
        self.model.update()

    def add_linkzy_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                for j in range(1, t + 1):
                    # logging.debug(f"Adding constraint for t = {t}, i = {i}, j = {j}")
                    self.model.addConstr(
                        self.z[t_index][i_index][j] <= self.y[t_index][j - 1],
                        name=f"C_linkzy_{t}_{i}_{j}",
                    )
        self.model.update()

    def add_assignment_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(
                    quicksum(self.z[t_index][i_index][j] for j in range(t + 1)) == 1,
                    name=f"C_assignment_{t}_{i}",
                )
        self.model.update()

    def add_nondecreasingy_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T):
            t_index = t - 1
            for j in range(1, t + 1):
                j_index = j - 1
                self.model.addConstr(
                    self.y[t_index][j_index] <= self.y[t_index + 1][j_index],
                    name=f"C_nondecreasingy_{t}_{j}",
                )
        self.model.update()

    def add_atmostonefacility_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T):
            t_index = t - 1
            self.model.addConstr(
                quicksum(self.y[t_index][j] for j in range(t)) + 1
                >= quicksum(self.y[t_index + 1][j] for j in range(t + 1)),
                name=f"C_atmostonefacility_{t}",
            )
        self.model.update()

    def add_objective_function(self, instance: FLOfflineInstance) -> None:
        self.model.setObjective(
            quicksum(self.y[self.T - 1][j] for j in range(self.T)) * instance.Gamma
            + quicksum(self.zeta[t] for t in range(self.T)),
            GRB.MINIMIZE,
        )
        self.model.update()

    def configure_solver(self, instance: FLOfflineInstance) -> None:
        """
        Set up variables, constraints, and objective function given a full instance.
        """
        if not instance.is_set:
            raise ValueError("Instance must be set before configuring solver.")
        self.T = instance.shape.T
        self.add_variables(instance)
        self.add_zetalowerbound_constraints(instance)
        self.add_linkzy_constraints(instance)
        self.add_assignment_constraints(instance)
        self.add_nondecreasingy_constraints(instance)
        self.add_atmostonefacility_constraints(instance)
        self.add_objective_function(instance)

    def final_facilities_service_costs(self) -> Tuple[List[int], List[float]]:
        built = [False for _ in range(self.T)]
        facilities = [
            0
        ]  # from 0, ..., T, index t represents what we built at time t, -1 for nothing
        service_costs = [
            0.0
        ]  # from 0, ..., T, index t represents the service cost at time t
        for t in range(1, self.T + 1):
            facility = -1
            t_index = t - 1
            service_costs.append(self.zeta[t_index].X)
            for j in range(1, t + 1):
                j_index = j - 1
                if built[j_index]:
                    continue
                if self.y[t_index][j_index].X > 0.5:
                    built[j_index] = True
                    facility = j
                    break
            facilities.append(facility)
        return (facilities, service_costs)

    def solve(self, instance: FLOfflineInstance) -> FLSolution:
        """Solve model."""
        start = time.time()
        self.model.optimize()
        self.running_time_s = time.time() - start
        self.optimal = self.model.status == GRB.OPTIMAL
        solution = FLSolution()
        state = CVCTState()
        state.bare_final_state(instance, self.final_facilities_service_costs())
        solution.from_cvtca(state, self.running_time_s, self.optimal, self.NAME)
        return solution

    def write_model(self, filename: str = "OfflineMIP.lp") -> None:
        """Write model to file."""
        self.model.write(filename)


class StaticMIP(IFLSolver):
    NAME = "StMIP"

    def __init__(self) -> None:
        """Constructs blank gurobi model."""
        super().__init__()
        self.model: Model = Model("StaticMIP")
        self.model.setParam("LogToConsole", 0)
        self.model.setParam("LogFile", gurobi_log_file())
        self.zeta: List[GRBVar] = []
        self.z: List[GRBVar] = []
        self.y: List[GRBVar] = []

    def add_variables(self, instance: FLOfflineInstance) -> None:
        self.y = [
            self.model.addVar(vtype=GRB.BINARY, name=f"y^{j}")
            for j in range(1, self.T + 1)
        ]
        self.z = [
            [
                self.model.addVar(vtype=GRB.BINARY, name=f"z^{i}_{j}")
                for j in range(0, self.T + 1)
            ]
            for i in range(1, self.T + 1)
        ]
        self.zeta = [
            self.model.addVar(vtype=GRB.CONTINUOUS, name=f"zeta^{t}")
            for t in range(1, self.T + 1)
        ]
        self.model.update()

    def add_zetalowerbound_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(
                    self.zeta[t_index]
                    >= quicksum(
                        instance.get_distance(i, j) * self.z[i_index][j]
                        for j in range(self.T + 1)
                    ),
                    name=f"C_zeta_lb_{t}_{i}",
                )
        self.model.update()

    def add_linkzy_constraints(self, instance: FLOfflineInstance) -> None:
        for i in range(1, self.T + 1):
            i_index = i - 1
            for j in range(1, self.T + 1):
                # logging.debug(
                #     f"Adding linkzy constraint for i = {i}, j = {j}. i_index = {i_index}."
                # )
                # logging.debug(f"y: {self.y}, z: {self.z}")
                self.model.addConstr(
                    self.z[i_index][j] <= self.y[j - 1],
                    name=f"C_linkzy_{i}_{j}",
                )
        self.model.update()

    def add_oneassignment_constraints(self, instance: FLOfflineInstance) -> None:
        for i in range(1, self.T + 1):
            i_index = i - 1
            self.model.addConstr(
                quicksum(self.z[i_index][j] for j in range(self.T + 1)) == 1,
                name=f"C_oneassignment_{i}",
            )
        self.model.update()

    def add_objective_function(self, instance: FLOfflineInstance) -> None:
        self.model.setObjective(
            quicksum(self.y[j] for j in range(self.T)) * instance.Gamma
            + quicksum(self.zeta[t] for t in range(self.T)),
            GRB.MINIMIZE,
        )
        self.model.update()

    def configure_solver(self, instance: FLOfflineInstance) -> None:
        """
        Set up variables, constraints, and objective function given a full instance.
        """
        if not instance.is_set:
            raise ValueError("Instance must be set before configuring solver.")
        self.T = instance.shape.T
        self.add_variables(instance)
        self.add_zetalowerbound_constraints(instance)
        self.add_linkzy_constraints(instance)
        self.add_oneassignment_constraints(instance)
        self.add_objective_function(instance)

    def final_facilities_service_costs(self) -> Tuple[List[int], List[float]]:
        built = [False for _ in range(self.T)]
        facilities = [0]  # static facility set, length depends on facilities built
        service_costs = [
            0.0
        ]  # from 0, ..., T, index t represents the service cost at time t
        for t in range(1, self.T + 1):
            t_index = t - 1
            service_costs.append(self.zeta[t_index].X)

        for j in range(1, t + 1):
            j_index = j - 1
            if self.y[j_index].X > 0.5:
                facilities.append(j)

        return (facilities, service_costs)

    def solve(self, instance: FLOfflineInstance) -> FLSolution:
        """Solve model."""
        start = time.time()
        self.model.optimize()
        self.running_time_s = time.time() - start
        self.optimal = self.model.status == GRB.OPTIMAL
        solution = FLSolution()
        state = CVCTState()
        state.bare_final_state(instance, self.final_facilities_service_costs())
        solution.from_cvtca(state, self.running_time_s, self.optimal, self.NAME)
        return solution

    def write_model(self) -> None:
        """Write model to file."""
        filename = append_date(self.NAME) + ".lp"
        self.model.write(filename)


class OnlineCVCTAlgorithm(IFLSolver):
    NAME = "CVTCA"

    def __init__(self) -> None:
        self.offline_instance: FLOfflineInstance = FLOfflineInstance()
        self.T: int = 0
        self.Gamma: float = 0.0
        self.cum_var_cost: float = 0.0
        self.state: CVCTState = CVCTState()

    def configure_solver(self, instance: FLOfflineInstance) -> None:
        if not instance.is_set:
            raise ValueError("Instance must be set before configuring solver.")
        self.offline_instance = instance
        self.T = instance.shape.T
        self.Gamma = instance.Gamma
        self.state.configure_state(instance)

    def greedy_facility_selection(self) -> int:
        """Distances updated with new point on previous facility set before calling."""
        ell = -1
        max_distance = sys.float_info.min
        for i in range(1, self.state.t_index + 1):
            if self.state.distance_to_closest_facility[i] > max_distance:
                ell = i
                max_distance = self.state.distance_to_closest_facility[i]
        return ell

    def add_facility(self) -> None:
        ell = self.greedy_facility_selection()
        self.state.update(self.offline_instance, ell)

    def no_facility_update(self, service_cost: float) -> None:
        self.state.update(self.offline_instance, service_cost=service_cost)

    def single_iteration(self) -> None:
        _LOGGER.log_subheader(f"Starting iteration at time {self.state.t_index}")
        self.state.update(self.offline_instance)
        nobuild_service_cost = self.state.compute_service_cost()
        _LOGGER.log_body(
            f"(cumVarCost: {self.state.cum_var_cost}) + (no build service cost: {nobuild_service_cost}) = {self.state.cum_var_cost + nobuild_service_cost}, gamma: {self.Gamma}"
        )
        if self.state.cum_var_cost + nobuild_service_cost > self.Gamma:
            self.add_facility()
        else:
            self.no_facility_update(nobuild_service_cost)

    def solve(self, instance: FLOfflineInstance) -> FLSolution:
        start = time.time()
        while self.state.t_index <= self.T:
            self.single_iteration()
        self.running_time_s = time.time() - start
        self.optimal = False
        solution = FLSolution()
        solution.from_cvtca(self.state, self.running_time_s, self.optimal, self.NAME)
        return solution
