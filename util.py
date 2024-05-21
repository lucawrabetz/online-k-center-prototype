import numpy as np

class DPoint:
    def __init__(self, x: np.ndarray) -> None:
        self.x: np.ndarray = np.asarray(x)
        self.n: int = len(x)

class FLInstanceShape:
    def __init__(self, n: int = 2, T: int = 3, Gamma: float = 10.0) -> None:
        self.n: int = n
        self.T: int = T
        self.Gamma: np.float64 = np.float64(Gamma)

    def print(self) -> None:
        print(f"n: {self.n}")
        print(f"T: {self.T}")

INSTANCE_SHAPES = {
    "test": FLInstanceShape(),
}

class FLDistribution:
    '''
    Class to hold "distributions" for generating demand points for FLFullInstances.
    '''
    def __init__(self, shape: FLInstanceShape) -> None:
        self.shape: FLInstanceShape = shape
        self.rng: np.random.Generator = np.random.default_rng()

    def unit_square(self) -> np.ndarray:
        '''
        Generate points uniformly at random from the unit square of dimension n.
        '''
        return np.asarray([DPoint(self.rng.random((self.shape.n,))) for _ in range(self.shape.T + 1)])
        


class FLFullInstance:
    def __init__(self, shape: FLInstanceShape) -> None:
        self.shape: FLInstanceShape = shape
        self.x: np.ndarray = np.asarray([DPoint(np.zeros(self.shape.n)) for _ in range(self.shape.T + 1)])

    def set_x_random(self) -> None:
        '''
        Set demand points by instantiating and calling on an FLDistribution object.
        '''
        dist = FLDistribution(self.shape)
        self.x = dist.unit_square()
        print("Set x randomly.")
        self.print()

    def read_x_from_file(self, filename: str) -> None:
        '''
        Read demand points from file.
        '''
        pass

    def print(self) -> None:
        print(f"n: {self.shape.n}")
        print(f"T: {self.shape.T}")
        print(f"Gamma: {self.shape.Gamma}")
        for t in range(self.shape.T + 1):
            print(f"x_{t}: {self.x[t].x}")

# class OnlineWrapper (?)

def main():
    instance = FLFullInstance(INSTANCE_SHAPES["test"])
    instance.print()
    instance.set_x_random()

if __name__ == '__main__':
    main()
