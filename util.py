import numpy as np
from typing import Callable, List

class DPoint:
    def __init__(self, x: np.ndarray) -> None:
        self.x: np.ndarray = np.asarray(x)
        self.n: int = len(x)


class FLInstanceShape:
    def __init__(self, n: int = 2, T: int = 3) -> None:
        self.n: int = n
        self.T: int = T

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

    def gamma_interval(self, low: int, high: int) -> float:
        '''
        Generate Gamma uniformly at random from [low, high].
        '''
        return self.rng.uniform(low, high)

    def x_unit_square(self) -> List[DPoint]:
        '''
        Generate points uniformly at random from the unit square of dimension n.
        '''
        return [DPoint(self.rng.random((self.shape.n,))) for _ in range(self.shape.T + 1)]

### distance functions
def euclidean_distance(point_a: DPoint, point_b: DPoint) -> float:
    return np.linalg.norm(point_a.x - point_b.x)

class FLFullInstance:
    def __init__(self, shape: FLInstanceShape) -> None:
        self.shape: FLInstanceShape = shape
        self.x: List[DPoint] = [DPoint(np.zeros(self.shape.n)) for _ in range(self.shape.T + 1)]
        self.Gamma: float = 0.0
        self.distances: np.ndarray = np.zeros((self.shape.T + 1, self.shape.T + 1))
        self.distance: Callable[[DPoint, DPoint], float] = euclidean_distance
        self.is_set: bool = False

    def set_distance_matrix(self) -> None:
        '''
        Set distance matrix based on demand points.
        '''
        for i in range(self.shape.T):
            for j in range(i + 1, self.shape.T + 1):
                if i == j:
                    self.distances[i][j] = 0.0
                else:
                    self.distances[i][j] = self.distance(self.x[i], self.x[j])
                    self.distances[j][i] = self.distances[i][j]

    def get_distance(self, i: int, j: int) -> float:
        '''
        Return distance between points i and j.
        '''
        return self.distances[i][j]

    def set_x_random(self, low: int = 5, high: int = 15, Gamma: float = None) -> None:
        '''
        Set demand points by instantiating and calling on an FLDistribution object.
        Update distance matrix.
        '''
        dist = FLDistribution(self.shape)
        if Gamma:
            self.Gamma = Gamma
        else:
            self.Gamma = dist.gamma_interval(low, high)
        self.x = dist.x_unit_square()
        self.set_distance_matrix()
        self.is_set = True

    def read_x_from_file(self, filename: str) -> None:
        '''
        Read demand points from file.
        Update distance matrix.
        '''
        pass
        self.is_set = True
        self.set_distance_matrix()

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
