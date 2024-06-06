from typing import List, Dict
from solvers import OfflineMIP, SemiOfflineMIP, CCTAlgorithm, IFLSolver, _SOLVER_FACTORY
from problem import FLOfflineInstance, FLSolution
from log_config import setup_logging, throwaway_gurobi_model, _LOGGER
from allowed_types import FLInstanceType, _SOLVERS

setup_logging()
import logging


class InteractiveExperiment:
    def __init__(self) -> None:
        # T: int = int(input("Enter T: "))
        # n: int = int(input("Enter n: "))
        throwaway_gurobi_model()
        _LOGGER.clear_page()
        T: int = 3
        n: int = 2
        self.shape: FLInstanceType = FLInstanceType(n=n, T=T)
        self.instance: FLOfflineInstance = FLOfflineInstance(self.shape)
        # Gamma_raw: str = input("Enter Gamma (r to generate randomly): ")
        # if Gamma_raw == "r":
        #    low: int = int(input("Enter low: "))
        #    high: int = int(input("Enter high: "))
        #    self.instance.set_random(low=low, high=high)
        # else:
        #    Gamma: float = float(Gamma_raw)
        #    self.instance.set_random(Gamma=Gamma)
        self.instance.set_random(Gamma=0.75)
        self.solvers: List[IFLSolver] = [
            _SOLVER_FACTORY.solver(s_id) for s_id in _SOLVERS
        ]
        for solver in self.solvers:
            solver.configure_solver(self.instance)
        self.solver_to_solution: Dict[str, FLSolution] = {}

    def print(self) -> None:
        _LOGGER.log_header(
            f"Interactive Experiment for randomly generated instance with T = {self.instance.id.T}, n = {self.instance.id.n}, gamma = {self.instance.Gamma}"
        )
        _LOGGER.separator_line()
        for solver_name, solution in self.solver_to_solution.items():
            solution.print(self.instance, solver_name)
            _LOGGER.separator_line()

    def run(self) -> None:
        for solver in self.solvers:
            sol = solver.solve(self.instance)
            self.solver_to_solution[solver.id.name] = sol
        self.print()


def main():
    exp = InteractiveExperiment()
    exp.run()


if __name__ == "__main__":
    main()
