import logging
from argparse import ArgumentParser
from experiments import FLExperiment
from allowed_types import _SOLVERS

from log_config import setup_logging, _LOGGER

setup_logging()


def main():
    parser = ArgumentParser()
    parser.add_argument("--set_name", type=str, default="test")
    experiment = FLExperiment(parser.parse_args().set_name)
    experiment.configure_experiment(solver_ids=_SOLVERS)
    experiment.run()


if __name__ == "__main__":
    main()
