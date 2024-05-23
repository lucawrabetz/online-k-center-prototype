import sys
import numpy as np
from typing import List
from gurobipy import Model, GRB, quicksum
from util import FLFullInstance


class OfflineMIP:
    def __init__(self) -> None:
        '''Constructs blank gurobi model.'''
        self.model: GRBModel = Model("OfflineMIP")
        self.zeta: List[GRBVar] = None
        self.z: List[GRBVar] = None
        self.y: List[GRBVar] = None
        self.T: List[GRBVar] = None

    def add_variables(self, instance: FLFullInstance) -> None:
        self.y = [[self.model.addVar(vtype=GRB.BINARY, name=f"y^{t}_{j}") for j in range(1, t + 1)] for t in range(1, self.T + 1)]
        self.z = [[[self.model.addVar(vtype=GRB.BINARY, name=f"z^{t}_{i}_{j}") for j in range(0, t + 1)] for i in range(1, t + 1)] for t in range(1, self.T + 1)]
        self.zeta = [self.model.addVar(vtype=GRB.CONTINUOUS, name=f"zeta^{t}") for t in range(1, self.T + 1)]
        self.model.update()

    def add_zetalowerbound_constraints(self, instance: FLFullInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(self.zeta[t_index] >= quicksum(instance.get_distance(i, j) * self.z[t_index][i_index][j] for j in range(t + 1)), name = f"C_zeta_lb_{t}_{i}")
        self.model.update()

    def add_linkzy_constraints(self, instance: FLFullInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                for j in range(1, t + 1):
                    # print(f"Adding constraint for t = {t}, i = {i}, j = {j}")
                    self.model.addConstr(self.z[t_index][i_index][j] <= self.y[t_index][j-1], name = f"C_linkzy_{t}_{i}_{j}")
        self.model.update()

    def add_assignment_constraints(self, instance: FLFullInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(quicksum(self.z[t_index][i_index][j] for j in range(t + 1)) == 1, name = f"C_assignment_{t}_{i}")
        self.model.update()

    def add_nondecreasingy_constraints(self, instance: FLFullInstance) -> None:
        for t in range(1, self.T):
            t_index = t - 1
            for j in range(1, t + 1):
                j_index = j - 1
                self.model.addConstr(self.y[t_index][j_index] <= self.y[t_index + 1][j_index], name = f"C_nondecreasingy_{t}_{j}")
        self.model.update()

    def add_atmostonefacility_constraints(self, instance: FLFullInstance) -> None:
        for t in range(1, self.T):
            t_index = t - 1
            self.model.addConstr(quicksum(self.y[t_index][j] for j in range(t)) + 1 >= quicksum(self.y[t_index + 1][j] for j in range(t+1)), name = f"C_atmostonefacility_{t}")
        self.model.update()

    def add_objective_function(self, instance: FLFullInstance) -> None:
        self.model.setObjective(quicksum(self.y[self.T - 1][j] for j in range(self.T)) * instance.Gamma + quicksum(self.zeta[t] for t in range(self.T)), GRB.MINIMIZE)
        self.model.update()

    def configure_solver(self, instance: FLFullInstance) -> None:
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

class CVCTState:
    def __init__(self) -> None:
        self.T: int = None
        self.num_facilities: int = None
        self.t_index: int = None
        self.cum_var_cost: float = None
        self.facilities: List[int] = None
        self.distance_to_closest_facility: List[float] = None
        self.service_costs: List[float] = None

    def configure_state(self, instance: FLFullInstance) -> None:
        '''
        Configure algorithm global state.
            - self.facilities: List indexed by time t, index: t -> 0: 0, ... T: T, where self.facilities[t] == -1 means no facility is built at time t, and self.facilities[t] == i, i \in [0, T] means facility i is built at time t. The list is unique, of course, and initialized as [0, -1, ... , -1].
            - self.distance_to_closest_facility: List indexed by time t, index: t -> 0: 0.0, ... T: 0.0, where self.distance_to_closest_facility[t] == -1 means the algorithm has not reached t, so the point "doesn't exist" in the online setting and the distance is uninitialized. self.distance_to_closest_facility[t] is equivalent to s_t(F_t)_i, i.e. s_t(F_t) for a fixed i -> min_{j \in F_t} d(x_i, x_j), in the paper.
            - self.facility_service_costs: List indexed by time t, index: t -> 0: 0.0, ... T: 0.0, where self.facility_service_costs[t] == s_t(F_t) in the paper, which is evaluated and summed to compute the objective function c_t(F_t).
        '''
        self.T = instance.shape.T
        self.num_facilities = 1
        self.t_index = 1
        self.cum_var_cost = 0.0
        self.facilities = [-1 if t > 0 else 0 for t in range(self.T + 1)]
        self.distance_to_closest_facility = [-1 if t > 0 else 0.0 for t in range(self.T + 1)]
        self.service_costs = [-1 if t > 0 else 0.0 for t in range(self.T + 1)]

    def update_distances_new_facility(self, instance: FLFullInstance, ell: int) -> None:
        '''For all points i = 1, ..., self.t_index, update self.distance_to_closest_facility[i] if the new facility is closer to x_i.'''
        for i in range(1, self.t_index + 1):
            new_distance = instance.get_distance(i, ell)
            if new_distance < self.distance_to_closest_facility[i]:
                self.distance_to_closest_facility[i] = new_distance

    def update_distances_new_point(self, instance: FLFullInstance) -> None:
        '''compute closest facility to point, update self.distance_to_closest_facility (only at self.t_index).'''
        min_distance = sys.float_info.max
        for j in self.facilities:
            if j == -1:
                continue
            new_distance = instance.get_distance(self.t_index, j)
            min_distance = min(min_distance, new_distance)
        self.distance_to_closest_facility[self.t_index] = min_distance

    def update(self, instance: FLFullInstance, ell: int = None, service_cost: float = None) -> None:
        '''
        Update algorithm state. This can be triggered by:
            - A new point self.offline_instance.points[self.t_index] arriving (ell = None, service_cost = None)
            - A new facility (ell) has been built
            - An iteration has ended and we have the final service cost for the time period (previous + new point on previous facility set)
        '''
        if ell and service_cost:
            raise ValueError("ell and service_cost cannot be set at the same time.")
        elif ell:
            print(f"***** Updating state at time {self.t_index}, to add facility {ell}: {instance.points[ell].x} *****")
            self.facilities[self.t_index] = ell
            self.num_facilities += 1
            self.update_distances_new_facility(instance, ell)
            self.cum_var_cost = self.service_cost(new_facility = True)
            self.t_index += 1
            print(f"Facilities: {self.facilities}, distances: {self.distance_to_closest_facility}.")
        elif service_cost:
            print(f"Updating state at time {self.t_index}, for service cost {service_cost}.")
            self.facilities[self.t_index] = -1
            self.service_costs[self.t_index] = service_cost
            self.cum_var_cost += service_cost
            self.t_index += 1
            print(f"Facilities: {self.facilities}, distances: {self.distance_to_closest_facility}.")
        else:
            print(f"Updating state at time {self.t_index} to add demand point {self.t_index}: {instance.points[self.t_index].x}.")
            self.update_distances_new_point(instance)
            print(f"Distances after update: {self.distance_to_closest_facility}.")

    def service_cost(self, new_facility: bool = False) -> float:
        '''
        The updated self.distance_to_closest_facility[self.t_index] encodes the correct current facility set. 
            - if no facility has been built since the last distance update, self.service_costs[self.t_index - 1] is a lower bound.
            - if a new facility has been built we have to check the whole distance list as the new faciility may have better served some point that was the previous max min.
        '''
        if new_facility: return max(self.distance_to_closest_facility[:self.t_index + 1])
        else: return max(self.distance_to_closest_facility[self.t_index], self.service_costs[self.t_index - 1])

class OnlineCVCTAlgorithm:
    def __init__(self) -> None:
        self.offline_instance: FLFullInstance = None
        self.T: int = None
        self.Gamma: float = None
        self.cum_var_cost: float = None
        self.state: CVCTState = CVCTState()

    def configure_solver(self, instance: FLFullInstance) -> None:
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
        

    def solve(self):
        while self.state.t_index <= self.T:
            self.single_iteration()
