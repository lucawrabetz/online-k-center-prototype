import logging
from argparse import ArgumentParser
from experiments import FLExperiment
from allowed_types import _SOLVERS, _CCTA

from log_config import setup_logging, _LOGGER

setup_logging()


def main():
    parser = ArgumentParser()
    parser.add_argument("--set_name", type=str, default="test")
    parser.add_argument("--gamma", type=int, default=-1)
    if parser.parse_args().gamma == -1:
        experiment = FLExperiment(parser.parse_args().set_name)
        experiment.configure_experiment(solver_ids=_SOLVERS)
        experiment.run()
    else:
        for g in range(parser.parse_args().gamma + 1):
            experiment = FLExperiment(parser.parse_args().set_name)
            experiment.configure_experiment(solver_ids=_SOLVERS, gamma=g)
            experiment.run()


if __name__ == "__main__":
    main()
