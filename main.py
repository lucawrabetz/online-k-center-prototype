from solvers import OfflineMIP, OnlineCVCTAlgorithm
from problem import FLOfflineInstance, INSTANCE_SHAPES

from log_config import setup_logging

setup_logging()
import logging


def main():
    logging.info("Started")
    instance = FLOfflineInstance(INSTANCE_SHAPES["small"])
    instance.set_random()
    instance.print()
    mip = OfflineMIP()
    mip.configure_solver(instance)
    logging.info("SOLVING OFFLINE MIP...")
    mip_solution = mip.solve()
    logging.info("SOLVING ONLINE CVTCA...")
    algo = OnlineCVCTAlgorithm()
    algo.configure_solver(instance)
    algo_solution = algo.solve()
    mip_solution.print("MIP")
    algo_solution.print("CVTCA")
    logging.info("Finished")


if __name__ == "__main__":
    main()
