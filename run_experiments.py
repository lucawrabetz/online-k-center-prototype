from experiments import FLExperiment


def main():
    instances = []
    solvers = []
    experiment = FLExperiment()
    experiment.configure_experiment(instances, solvers)
    experiment.run()


if __name__ == "__main__":
    main()
