from typing import List, Dict
from solvers import OfflineMIP, OnlineCVCTAlgorithm, StaticMIP, IFLSolver
from problem import FLOfflineInstance, FLSolution
from log_config import setup_logging, _LOGGER
from allowed_types import FLInstanceType, _INSTANCE_SHAPES

setup_logging()
import logging


class InteractiveExperiment:
    def __init__(self) -> None:
        T: int = int(input("Enter T: "))
        n: int = int(input("Enter n: "))
        self.shape: FLInstanceType = FLInstanceType(n=n, T=T)
        self.instance: FLOfflineInstance = FLOfflineInstance(self.shape)
        Gamma_raw: str = input("Enter Gamma (r to generate randomly): ")
        if Gamma_raw == "r":
            low: int = int(input("Enter low: "))
            high: int = int(input("Enter high: "))
            self.instance.set_random(low=low, high=high)
        else:
            Gamma: float = float(Gamma_raw)
            self.instance.set_random(Gamma=Gamma)
        self.solvers: List[IFLSolver] = [
            OfflineMIP(),
            StaticMIP(),
            OnlineCVCTAlgorithm(),
        ]
        for solver in self.solvers:
            solver.configure_solver(self.instance)
        self.solver_to_solution: Dict[str, FLSolution] = {}

    def print(self) -> None:
        _LOGGER.separator_block()
        for solver_name, solution in self.solver_to_solution.items():
            solution.print(self.instance, solver_name)
            _LOGGER.separator_block()

    def run(self) -> None:
        for solver in self.solvers:
            sol = solver.solve(self.instance)
            self.solver_to_solution[solver.id.name] = sol
        self.print()


def main():
    exp = InteractiveExperiment()
    exp.run()
    # logging.info("Started")
    # instance = FLOfflineInstance(_INSTANCE_SHAPES["small"])
    # instance.set_random()
    # instance.print()
    # mip = OfflineMIP()
    # mip.configure_solver(instance)
    # logging.info("SOLVING OFFLINE MIP...")
    # mip_solution = mip.solve()
    # logging.info("SOLVING ONLINE CVTCA...")
    # algo = OnlineCVCTAlgorithm()
    # algo.configure_solver(instance)
    # algo_solution = algo.solve()
    # mip_solution.print("MIP")
    # algo_solution.print("CVTCA")
    # logging.info("Finished")


if __name__ == "__main__":
    main()
