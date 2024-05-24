import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from gurobipy import Model, GRB, quicksum
from problem import FLOfflineInstance, FLSolution, CVCTState

class IFLSolver(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def configure_solver(self, instance: FLOfflineInstance) -> None:
        pass

    @abstractmethod
    def solve(self) -> None:
        pass

class OfflineMIP(IFLSolver):
    def __init__(self) -> None:
        '''Constructs blank gurobi model.'''
        self.model: GRBModel = Model("OfflineMIP")
        self.zeta: List[GRBVar] = None
        self.z: List[GRBVar] = None
        self.y: List[GRBVar] = None
        self.T: List[GRBVar] = None

    def add_variables(self, instance: FLOfflineInstance) -> None:
        self.y = [[self.model.addVar(vtype=GRB.BINARY, name=f"y^{t}_{j}") for j in range(1, t + 1)] for t in range(1, self.T + 1)]
        self.z = [[[self.model.addVar(vtype=GRB.BINARY, name=f"z^{t}_{i}_{j}") for j in range(0, t + 1)] for i in range(1, t + 1)] for t in range(1, self.T + 1)]
        self.zeta = [self.model.addVar(vtype=GRB.CONTINUOUS, name=f"zeta^{t}") for t in range(1, self.T + 1)]
        self.model.update()

    def add_zetalowerbound_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(self.zeta[t_index] >= quicksum(instance.get_distance(i, j) * self.z[t_index][i_index][j] for j in range(t + 1)), name = f"C_zeta_lb_{t}_{i}")
        self.model.update()

    def add_linkzy_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                for j in range(1, t + 1):
                    # print(f"Adding constraint for t = {t}, i = {i}, j = {j}")
                    self.model.addConstr(self.z[t_index][i_index][j] <= self.y[t_index][j-1], name = f"C_linkzy_{t}_{i}_{j}")
        self.model.update()

    def add_assignment_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(quicksum(self.z[t_index][i_index][j] for j in range(t + 1)) == 1, name = f"C_assignment_{t}_{i}")
        self.model.update()

    def add_nondecreasingy_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T):
            t_index = t - 1
            for j in range(1, t + 1):
                j_index = j - 1
                self.model.addConstr(self.y[t_index][j_index] <= self.y[t_index + 1][j_index], name = f"C_nondecreasingy_{t}_{j}")
        self.model.update()

    def add_atmostonefacility_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T):
            t_index = t - 1
            self.model.addConstr(quicksum(self.y[t_index][j] for j in range(t)) + 1 >= quicksum(self.y[t_index + 1][j] for j in range(t+1)), name = f"C_atmostonefacility_{t}")
        self.model.update()

    def add_objective_function(self, instance: FLOfflineInstance) -> None:
        self.model.setObjective(quicksum(self.y[self.T - 1][j] for j in range(self.T)) * instance.Gamma + quicksum(self.zeta[t] for t in range(self.T)), GRB.MINIMIZE)
        self.model.update()

    def configure_solver(self, instance: FLOfflineInstance) -> None:
        '''
        Set up variables, constraints, and objective function given a full instance.
        '''
        if not instance.is_set: raise ValueError("Instance must be set before configuring solver.")
        self.T = instance.shape.T
        self.add_variables(instance)
        self.add_zetalowerbound_constraints(instance)
        self.add_linkzy_constraints(instance)
        self.add_assignment_constraints(instance)
        self.add_nondecreasingy_constraints(instance)
        self.add_atmostonefacility_constraints(instance)
        self.add_objective_function(instance)

    def solve(self) -> None:
        '''Solve model.'''
        self.model.optimize()
        declared_built = [False for _ in range(self.T)]
        # for t in range(1, self.T + 1):
        #     t_index = t - 1
        #     for i in range(1, t + 1):
        #         i_index = i - 1
        #         for j in range(0, t + 1):
        #             if self.z[t_index][i_index][j].x > 0.5:
        #                 print(f"    - Facility {j} serves demand point {i}")
        #                 print(f"    - z_{t}_{i}_{j} = {self.z[t_index][i_index][j].x}")
        #                 if j == 0:
        #                     print(f"    - Facility 0 is not constrained by y.")
        #                 else:
        #                     print(f"    - y_{t}_{j} = {self.y[t_index][j-1].x}")

    def write_model(self, filename: str = "OfflineMIP.lp") -> None:
        '''Write model to file.'''
        self.model.write(filename)

class OnlineCVCTAlgorithm(IFLSolver):
    def __init__(self) -> None:
        self.offline_instance: FLOfflineInstance = None
        self.T: int = None
        self.Gamma: float = None
        self.cum_var_cost: float = None
        self.state: CVCTState = CVCTState()

    def configure_solver(self, instance: FLOfflineInstance) -> None:
        if not instance.is_set: raise ValueError("Instance must be set before configuring solver.")
        self.offline_instance = instance
        self.T = instance.shape.T
        self.Gamma = instance.Gamma
        self.state.configure_state(instance)

    def greedy_facility_selection(self) -> int:
        '''Distances updated with new point on previous facility set before calling.'''
        ell = -1
        max_distance = -1
        for i in range(1, self.state.t_index + 1):
            if self.state.distance_to_closest_facility[i] > max_distance:
                ell = i
                max_distance = self.state.distance_to_closest_facility[i]
        return ell

    def add_facility(self) -> None:
        ell = self.greedy_facility_selection()
        self.state.update(self.offline_instance, ell)

    def no_facility_update(self, service_cost: float) -> None:
        self.state.update(self.offline_instance, service_cost = service_cost)

    def single_iteration(self) -> None:
        print(f"Starting iteration at time {self.state.t_index}")
        self.state.update(self.offline_instance)
        nobuild_service_cost = self.state.service_cost()
        print(f"cumVarCost: {self.state.cum_var_cost}, gamma: {self.Gamma}, no build service cost: {nobuild_service_cost}")
        if self.state.cum_var_cost + nobuild_service_cost > self.Gamma:
            self.add_facility()
        else:
            self.no_facility_update(nobuild_service_cost)
        print()
        
    def solve(self) -> FLSolution:
        while self.state.t_index <= self.T:
            self.single_iteration()
        solution = FLSolution()
        solution.from_cvtca(self.state)
        return solution
