from experiments import FLExperiment
from allowed_types import _SOLVERS


def main():
    instances = []
    experiment = FLExperiment()
    experiment.configure_experiment(instances, _SOLVERS)
    experiment.run()


if __name__ == "__main__":
    main()
