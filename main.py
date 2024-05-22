from gurobipy import *
from util import *

class OfflineMIP:
    def __init__(self) -> None:
        '''Constructs blank gurobi model.'''
        self.model = Model("OfflineMIP")
        self.zeta = None
        self.z = None
        self.y = None
        self.T = None

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
        print(self.z)
        print(self.zeta)
        print(self.y)
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                for j in range(1, t + 1):
                    print(f"Adding constraint for t = {t}, i = {i}, j = {j}")
                    print(f"y_{t}_{j} = {self.y[t_index][j-1]}")
                    print(f"z_{t}_{i}_{j} = {self.z[t_index][i_index][j]}")
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
        self.model.setObjective(quicksum(self.y[self.T - 1][j] for j in range(self.T)) * instance.shape.Gamma + quicksum(self.zeta[t] for t in range(self.T)), GRB.MINIMIZE)
        self.model.update()

    def configure_model(self, instance: FLFullInstance) -> None:
        '''
        Set up variables, constraints, and objective function given a full instance.
        '''
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
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                print(f"t = {t}, i = {i}")
                for j in range(0, t + 1):
                    if self.z[t_index][i_index][j].x > 0.5:
                        print(f"    - Facility {j} serves demand point {i}")
                        print(f"    - z_{t}_{i}_{j} = {self.z[t_index][i_index][j].x}")
                        if j == 0:
                            print(f"    - Facility 0 is not constrained by y.")
                        else:
                            print(f"    - y_{t}_{j} = {self.y[t_index][j-1].x}")

    def write_model(self, filename: str = "OfflineMIP.lp") -> None:
        '''Write model to file.'''
        self.model.write(filename)

def main():
    instance = FLFullInstance(INSTANCE_SHAPES["test"])
    instance.set_x_random()
    instance.print()
    mip = OfflineMIP()
    mip.configure_model(instance)
    mip.write_model("test.lp")
    mip.solve()

if __name__ == '__main__':
    main()
