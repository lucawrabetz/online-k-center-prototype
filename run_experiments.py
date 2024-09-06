import logging
from argparse import ArgumentParser
from experiments import FLExperiment
from allowed_types import _SOLVERS, _CCTA, _OMIP

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
    parser.add_argument("--reps", type=int, default=-1)
    parser.add_argument(
        "--perm",
        type=str,
        default="none",
        choices=["none", "start", "end", "full", "nearest", "farthest"],
    )
    parser.add_argument("--firstf", type=int, default=-1)
    distance = parser.parse_args().distance
    write = parser.parse_args().write
    first_facility = (
        None if parser.parse_args().firstf == -1 else parser.parse_args().firstf
    )
    if distance not in DISTANCES.keys():
        raise ValueError(
            "Invalid distance type. Must be one of {}".format(DISTANCES.keys())
        )
    distance_function = DISTANCES[distance]
    permutation = parser.parse_args().perm
    SOLVERS = [_CCTA, _OMIP]
    if parser.parse_args().reps == -1:
        experiment = FLExperiment(
            parser.parse_args().set_name, distance=distance_function, write=write
        )
        experiment.configure_experiment(solver_ids=SOLVERS)
        experiment.run(permutation=permutation, first_facility=first_facility)
    else:
        for gamma in range(150000, parser.parse_args().gamma + 1, 5000):
            for first_facility in range(parser.parse_args().reps):
                experiment = FLExperiment(
                    parser.parse_args().set_name,
                    distance=distance_function,
                    write=write,
                )
                experiment.configure_experiment(
                    solver_ids=SOLVERS,
                    gamma=gamma,
                )
                experiment.run(permutation=permutation, first_facility=first_facility)


if __name__ == "__main__":
    main()
