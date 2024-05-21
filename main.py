from gurobipy import *
from models import *

# Data
# FLInstanceGenerator(n, T): A class to generate random instances, and construct 
# FLFullInstance: Passive struct to hold an entire offline instance.
# FLInstanceDistributor(FLFullInstance): A class to hold an entire offline instance, distribute it to offline algorithms (return it entirely), and distribute it to online algorithms (return it partially / incrementally).

# Full Instance description (necessary data held in FLFullInstance):
# n: dimension of points (int)
# T: total number of time periods (int)
# x_0, x_1, ..., x_T: points in R^n 
# Gamma: model parameter (float)

class OfflineMIP:
    def __init__(self) -> None:
        '''Constructs blank gurobi model.'''
        self.model = Model("OfflineMIP")
        self.zeta = None
        self.z = None
        self.y = None

    def configure_model(self, instance: FLFullInstance) -> None:
        '''
        Set up variables, constraints, and objective function given instance.
        '''
        # Add variables:
        self.y = [[self.model.addVar(vtype=GRB.BINARY, name=f"y^{t}_{i}") for i in range(1, t + 1)] for t in range(1, instance.shape.T + 1)]
        self.z = [[[self.model.addVar(vtype=GRB.BINARY, name=f"z^{t}_{i}_{j}") for j in range(0, t + 1)] for i in range(1, t + 1)] for t in range(1, instance.shape.T + 1)]
        self.zeta = [self.model.addVar(vtype=GRB.CONTINUOUS, name=f"zeta^{t}") for t in range(1, instance.shape.T + 1)]

    def write_model(self, filename: str = "OfflineMIP.lp") -> None:
        '''Write model to file.'''
        self.model.write(filename)

def main():
    instance = FLFullInstance(INSTANCE_SHAPES["test"])
    mip = OfflineMIP()
    mip.configure_model(instance)
    mip.write_model("test.lp")

if __name__ == '__main__':
    main()
