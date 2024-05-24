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
    solution = mip.solve()
    # solution.print()
    logging.info("SOLVING ONLINE CVTCA...")
    algo = OnlineCVCTAlgorithm()
    algo.configure_solver(instance)
    solution = algo.solve()
    solution.print()
    logging.info("Finished")


if __name__ == "__main__":
    main()
