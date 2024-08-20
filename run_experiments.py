import logging
from argparse import ArgumentParser
from experiments import FLExperiment
from allowed_types import _SOLVERS, _CCTA

from log_config import setup_logging, _LOGGER
from problem import euclidean_distance, taxicab_distance

setup_logging()


def main():
    DISTANCES = {"euclidean": euclidean_distance, "taxicab": taxicab_distance}
    parser = ArgumentParser()
    parser.add_argument("--set_name", type=str, default="test")
    parser.add_argument("--distance", type=str, default="euclidean")
    parser.add_argument("--write", type=bool, default=True)
    parser.add_argument("--gamma", type=int, default=-1)
    parser.add_argument(
        "--perm", type=str, default="none", choices=["none", "start", "end", "full"]
    )
    distance = parser.parse_args().distance
    write = parser.parse_args().write
    if distance not in DISTANCES.keys():
        raise ValueError(
            "Invalid distance type. Must be one of {}".format(DISTANCES.keys())
        )
    distance_function = DISTANCES[distance]
    permutation = parser.parse_args().perm
    if parser.parse_args().gamma == -1:
        experiment = FLExperiment(
            parser.parse_args().set_name, distance=distance_function, write=write
        )
        experiment.configure_experiment(solver_ids=_SOLVERS)
        experiment.run(permutation=permutation)
    else:
        experiment = FLExperiment(
            parser.parse_args().set_name, distance=distance_function, write=write
        )
        experiment.configure_experiment(
            solver_ids=_SOLVERS,
            gamma=parser.parse_args().gamma,
        )
        experiment.run(permutation=permutation)


if __name__ == "__main__":
    main()
