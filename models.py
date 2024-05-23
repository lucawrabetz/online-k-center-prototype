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


class OnlineCAlgorithm:
    def __init__(self) -> None:
        self.offline_instance: FLFullInstance = None
        self.T: int = None
        self.Gamma: float = None
        self.serve_cost_count: float = None
        # See comment in configure_solver for explanation self.facilities.
        self.facilities: List[int] = None
        self.facilities_last_update: List[int] = None
        self.last_update_tindex: int = None
        self.facility_delta: int = None
        self.assignment: List[int] = None
        self.distance_to_assigned: List[float] = None
        self.num_facilities: int = None
        self.t_index: int = None

    def configure_solver(self, instance: FLFullInstance) -> None:
        '''
        Configure algorithm global state.
            - self.facilities: List indexed by time t, index: t -> 0: 0, ... T: T, where self.facilities[t] == -1 means no facility is built at time t, and self.facilities[t] == i, i \in [0, T] means facility i is built at time t. The list is unique, of course, and initialized as [0, -1, ... , -1].
            - self.distance_to_assigned: List indexed by time t, index: t -> 0: 0.0, ... T: 0.0, where self.distance_to_assigned[t] == -1 means the algorithm has not reached t, so the point "doesn't exist" in the online setting and the distance is uninitialized. self.distance_to_assigned[t] is equivalent to s_t(F_t)_i, i.e. s_t(F_t) for a fixed i -> min_{j \in F_t} d(x_i, x_j), in the paper.
            - self.facility_service_costs: List indexed by time t, index: t -> 0: 0.0, ... T: 0.0, where self.facility_service_costs[t] == s_t(F_t) in the paper, which is evaluated and summed to compute the objective function c_t(F_t).
        '''
        if not instance.is_set: raise ValueError("Instance must be set before configuring solver.")
        self.offline_instance = instance
        self.T = instance.shape.T
        self.Gamma = instance.Gamma
        self.serve_cost_count = 0.0
        self.facilities = [-1 if t > 0 else 0 for t in range(self.T + 1)]
        self.facilities_last_update = self.facilities.copy()
        self.last_update_tindex = 0
        self.facility_delta = -1
        self.assignment = [-1 if t > 0 else 0 for t in range(self.T + 1)]
        self.distance_to_assigned = [-1 if t > 0 else 0.0 for t in range(self.T + 1)]
        self.facility_service_costs = [0.0 for t in range(self.T + 1)]
        self.num_facilities = 1
        self.t_index = 1


    def next_facility_service_cost_same_set(self) -> float:
        '''Called at the start of an iteration, haven't decided to build a new facility yet, but distances and facility_service_costs are updated to t_index with assignments assuming no facility is built.'''
        return max(self.distance_to_assigned[self.t_index], self.facility_service_costs[self.t_index - 1])

    # def objective_cost_evaluation(self, facilities: List[int] = None) -> float:
    #     if facilities:
    #     else:
    #         self.facility_service_cost_evaluation()
    #         return self.Gamma * (self.num_facilities - 1) + sum(self.facility_service_costs[t] for t in range(1, self.t_index + 1))
    
    def greedy_facility_selection(self) -> int:
        pass

    def update_assignment(self) -> None:
        pass

    def update_facility_service_costs(self) -> None:
        distances = [self.distances_to_assigned[i] for i in range(0, self.t_index + 1)]
        self.facility_service_costs[self.t_index] = max(distances)

    def update_distances_new_facility(self, ell: int) -> None:
        '''For all points i = 1, ..., self.t_index, update self.distance_to_assigned[i] if the new facility is closer to x_i.'''
        for i in range(self.t_index):
            new_distance = self.offline_instance.get_distance(i, ell)
            if new_distance < self.distance_to_assigned[i]:
                self.distance_to_assigned[i] = new_distance

    def update_distances_new_point(self) -> None:
        '''compute closest facility to point, update self.distance_to_assigned (only at self.t_index).'''
        min_distance = sys.float_info.max
        for t in range(self.t_index):
            if self.facilities[t] != -1:
                j = self.facilities[t]
                min_distance = min(min_distance, self.offline_instance.get_distance(self.t_index, j))
        self.distance_to_assigned[self.t_index] = min_distance

    def update_state(self, ell: int = None) -> None:
        '''
        Update algorithm state. This can be triggered by two events in an iteration:
            - A new point self.offline_instance.points[self.t_index] has arrived in an online fashion.
                - compute closest facility to point, update self.distance_to_assigned (only at self.t_index)
            - A new facility has been built.
                - for all points i = 1, ..., self.t_index, update self.distance_to_assigned[i] if the new facility is closer to x_i.
        If ell, the facility index, is None, then we assume a new point has arrived (offline_instance.points[self.t_index] is the new point). 
        '''
        print(f"Updating state at time {self.t_index}")
        print(f"Facilities: {self.facilities}")
        print(f"Distance to assigned: {self.distance_to_assigned}")
        if ell:
            print(f"Updating for new facility {ell}")
            self.update_distances_new_facility(ell)
        else:
            print(f"Updating for new point {self.t_index}")
            self.update_distances_new_point()
        print(f"Distance to assigned after update: {self.distance_to_assigned}")
        print(f"\n")

    def single_iteration(self) -> None:
        print(f"\nStarting iteration at time {self.t_index}")
        self.update_state()
        if self.serve_cost_count + self.next_facility_service_cost_same_set() > self.Gamma:
            ell = self.greedy_facility_selection()
            self.facilities[self.t_index] = ell
            self.num_facilities += 1
            update_state()
            self.serve_cost_count = self.facility_service_costs[self.t_index]
        else:
            self.facilities[self.t_index] = -1
            self.update_state()
            self.serve_cost_count += self.facility_service_costs[self.t_index]
        

    def solve(self):
        while self.t_index <= self.T:
            self.single_iteration()
            self.t_index += 1
