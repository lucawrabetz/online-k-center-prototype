import sys
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
    "small": FLInstanceShape(10)
}

class FLDistribution:
    '''
    Class to hold "distributions" for generating demand points for FLOfflineInstances.
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

class FLOfflineInstance:
    def __init__(self, shape: FLInstanceShape) -> None:
        self.shape: FLInstanceShape = shape
        self.points: List[DPoint] = [DPoint(np.zeros(self.shape.n)) for _ in range(self.shape.T + 1)]
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
                    self.distances[i][j] = self.distance(self.points[i], self.points[j])
                    self.distances[j][i] = self.distances[i][j]

    def get_distance(self, i: int, j: int) -> float:
        '''
        Return distance between points i and j.
        '''
        return self.distances[i][j]

    def set_random(self, low: int = 5, high: int = 15, Gamma: float = None) -> None:
        '''
        Set demand points by instantiating and calling on an FLDistribution object.
        Update distance matrix.
        '''
        dist = FLDistribution(self.shape)
        if Gamma:
            self.Gamma = Gamma
        else:
            self.Gamma = dist.gamma_interval(low, high)
        self.points = dist.x_unit_square()
        self.set_distance_matrix()
        self.is_set = True

    def read_from_file(self, filename: str) -> None:
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
        print(f"Gamma: {self.Gamma}")
        for t in range(self.shape.T + 1):
            print(f"x_{t}: {self.points[t].x}")

class CVCTState:
    def __init__(self) -> None:
        self.T: int = None
        self.n: int = None
        self.Gamma: float = None
        self.num_facilities: int = None
        self.t_index: int = None
        self.cum_var_cost: float = None
        self.facilities: List[int] = None
        self.distance_to_closest_facility: List[float] = None
        self.service_costs: List[float] = None

    def configure_state(self, instance: FLOfflineInstance) -> None:
        '''
        Configure algorithm global state.
        '''
        self.T = instance.shape.T
        self.n = instance.shape.n
        self.Gamma = instance.Gamma
        self.num_facilities = 1
        self.t_index = 1
        self.cum_var_cost = 0.0
        self.facilities = [-1 if t > 0 else 0 for t in range(self.T + 1)]
        self.distance_to_closest_facility = [-1 if t > 0 else 0.0 for t in range(self.T + 1)]
        self.service_costs = [-1 if t > 0 else 0.0 for t in range(self.T + 1)]

    def update_distances_new_facility(self, instance: FLOfflineInstance, ell: int) -> None:
        '''For all points i = 1, ..., self.t_index, update self.distance_to_closest_facility[i] if the new facility is closer to x_i.'''
        for i in range(1, self.t_index + 1):
            new_distance = instance.get_distance(i, ell)
            if new_distance < self.distance_to_closest_facility[i]:
                self.distance_to_closest_facility[i] = new_distance

    def update_distances_new_point(self, instance: FLOfflineInstance) -> None:
        '''compute closest facility to point, update self.distance_to_closest_facility (only at self.t_index).'''
        min_distance = sys.float_info.max
        for j in self.facilities:
            if j == -1:
                continue
            new_distance = instance.get_distance(self.t_index, j)
            min_distance = min(min_distance, new_distance)
        self.distance_to_closest_facility[self.t_index] = min_distance

    def update(self, instance: FLOfflineInstance, ell: int = None, service_cost: float = None) -> None:
        '''
        Update algorithm state. This can be triggered by:
            - A new point self.offline_instance.points[self.t_index] arriving (ell = None, service_cost = None)
            - A new facility (ell) has been built
            - An iteration has ended and we have the final service cost for the time period (previous + new point on previous facility set)
        '''
        if ell and service_cost:
            raise ValueError("ell and service_cost cannot be set at the same time.")
        elif ell:
            print(f"***** Updating state at time {self.t_index}, to add facility {ell}: {instance.points[ell].x} *****")
            self.facilities[self.t_index] = ell
            self.num_facilities += 1
            self.update_distances_new_facility(instance, ell)
            self.cum_var_cost = self.service_cost(new_facility = True)
            self.t_index += 1
            print(f"Facilities: {self.facilities}, distances: {self.distance_to_closest_facility}.")
        elif service_cost:
            print(f"Updating state at time {self.t_index}, for service cost {service_cost}.")
            self.facilities[self.t_index] = -1
            self.service_costs[self.t_index] = service_cost
            self.cum_var_cost += service_cost
            self.t_index += 1
            print(f"Facilities: {self.facilities}, distances: {self.distance_to_closest_facility}.")
        else:
            print(f"Updating state at time {self.t_index} to add demand point {self.t_index}: {instance.points[self.t_index].x}.")
            self.update_distances_new_point(instance)
            print(f"Distances after update: {self.distance_to_closest_facility}.")

    def service_cost(self, new_facility: bool = False) -> float:
        '''
        The updated self.distance_to_closest_facility[self.t_index] encodes the correct current facility set. 
            - if no facility has been built since the last distance update, self.service_costs[self.t_index - 1] is a lower bound.
            - if a new facility has been built we have to check the whole distance list as the new faciility may have better served some point that was the previous max min.
        '''
        if new_facility: return max(self.distance_to_closest_facility[:self.t_index + 1])
        else: return max(self.distance_to_closest_facility[self.t_index], self.service_costs[self.t_index - 1])

class FLSolution:
    def __init__(self) -> None:
        self.n: int = None
        self.T: int = None
        self.Gamma: float = None
        self.num_facilities: int = None
        self.facilities: List[int] = None
        self.service_costs: List[float] = None

    def print(self) -> None:
        print(f"Built {self.num_facilities} facilities: {self.facilities}")
        print(f"Final service distances: {self.distance_to_closest_facility}")

    def from_cvtca(self, state: CVCTState) -> None:
        self.n = state.n
        self.T = state.T
        self.Gamma = state.Gamma
        self.num_facilities = state.num_facilities
        self.facilities = state.facilities
        self.distance_to_closest_facility = state.distance_to_closest_facility
# class OnlineWrapper (?)

def main():
    instance = FLOfflineInstance(INSTANCE_SHAPES["test"])
    instance.print()
    instance.set_random()

if __name__ == '__main__':
    main()
