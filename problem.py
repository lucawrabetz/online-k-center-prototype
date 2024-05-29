import sys
import logging
import numpy as np
from typing import Callable, List, Tuple
from util import Data
from log_config import _LOGGER


class DPoint:
    """
    Class to hold a demand point in n-dimensional space. Doesn't have any setup methods, so doesn't inherit Data.
    """

    def __init__(self, x: np.ndarray) -> None:
        self.x: np.ndarray = np.asarray(x)
        self.n: int = len(x)

    def print(self) -> None:
        with np.printoptions(precision=4, suppress=True):
            _LOGGER.log_body(self.x)


class FLInstanceShape:
    """
    Class to hold define instance shapes. Doesn't have any setup methods, so doesn't inherit Data.
    """

    def __init__(self, n: int = 2, T: int = 3) -> None:
        self.n: int = n
        self.T: int = T

    def print(self) -> None:
        _LOGGER.log_body(f"n: {self.n}")
        _LOGGER.log_body(f"T: {self.T}")


INSTANCE_SHAPES = {
    "test": FLInstanceShape(),
    "small": FLInstanceShape(10),
    "blank": FLInstanceShape(0, 0),
}


class FLDistribution:
    """
    Class to hold "distributions" for generating demand points for FLOfflineInstances.
    """

    def __init__(self, shape: FLInstanceShape) -> None:
        self.shape: FLInstanceShape = shape
        self.rng: np.random.Generator = np.random.default_rng()

    def gamma_interval(self, low: int, high: int) -> float:
        """
        Generate Gamma uniformly at random from [low, high].
        """
        return self.rng.uniform(low, high)

    def x_unit_square(self) -> List[DPoint]:
        """
        Generate points uniformly at random from the unit square of dimension n.
        """
        return [
            DPoint(self.rng.random((self.shape.n,))) for _ in range(self.shape.T + 1)
        ]


### distance functions - metric for instances
### instance class has a callable attribute for this
def euclidean_distance(point_a: DPoint, point_b: DPoint) -> float:
    return np.linalg.norm(point_a.x - point_b.x)


class FLOfflineInstance(Data):
    """
    Class to hold offline instance data for the online min max center problem.
    """

    def __init__(self, shape: FLInstanceShape = INSTANCE_SHAPES["blank"]) -> None:
        super().__init__()
        self.shape: FLInstanceShape = shape
        self.points: List[DPoint] = [
            DPoint(np.zeros(self.shape.n)) for _ in range(self.shape.T + 1)
        ]
        self.Gamma: float = 0.0
        self.distances: np.ndarray = np.zeros((self.shape.T + 1, self.shape.T + 1))
        self.distance: Callable[[DPoint, DPoint], float] = euclidean_distance

    def set_distance_matrix(self) -> None:
        """
        Set distance matrix based on demand points.
        """
        for i in range(self.shape.T):
            for j in range(i + 1, self.shape.T + 1):
                if i == j:
                    self.distances[i][j] = 0.0
                else:
                    self.distances[i][j] = self.distance(self.points[i], self.points[j])
                    self.distances[j][i] = self.distances[i][j]

    def get_distance(self, i: int, j: int) -> float:
        """
        Return distance between points i and j.
        """
        return self.distances[i][j]

    def set_random(self, low: int = 5, high: int = 15, Gamma: float = None) -> None:
        """
        Set demand points by instantiating and calling on an FLDistribution object.
        Update distance matrix.
        """
        dist = FLDistribution(self.shape)
        if Gamma:
            self.Gamma = Gamma
        else:
            self.Gamma = dist.gamma_interval(low, high)
        self.points = dist.x_unit_square()
        self.set_distance_matrix()
        self._is_set = True

    def read_from_file(self, filename: str) -> None:
        """
        Read demand points from file.
        Update distance matrix.
        """
        pass
        self._is_set = True
        self.set_distance_matrix()

    def print(self) -> None:
        _LOGGER.log_body(f"n: {self.shape.n}, T: {self.shape.T}, Gamma: {self.Gamma}")
        for t in range(self.shape.T + 1):
            self.points[t].print()


class CVCTState(Data):
    """
    Class to hold state for the CVCTCA algorithm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.T: int = 0
        self.n: int = 0
        self.Gamma: float = 0.0
        self.objective: float = 0.0
        self.num_facilities: int = 0
        self.t_index: int = 0
        self.cum_var_cost: float = 0.0
        self.facilities: List[int] = []
        self.distance_to_closest_facility: List[float] = []
        self.service_costs: List[float] = []

    def configure_state(self, instance: FLOfflineInstance) -> None:
        """
        Configure algorithm global state.
        """
        if not instance.is_set:
            raise ValueError("Instance must be set before configuring state.")
        self.T = instance.shape.T
        self.n = instance.shape.n
        self.Gamma = instance.Gamma
        self.num_facilities = 1
        self.t_index = 1
        self.cum_var_cost = 0.0
        self.facilities = [-1 if t > 0 else 0 for t in range(self.T + 1)]
        self.distance_to_closest_facility = [
            -1 if t > 0 else 0.0 for t in range(self.T + 1)
        ]
        self.service_costs = [-1 if t > 0 else 0.0 for t in range(self.T + 1)]
        self._is_set = True

    def update_distances_new_facility(
        self, instance: FLOfflineInstance, ell: int
    ) -> None:
        """
        Check if newly built facility ell is closer to any points than their current closest facility, update distances accordingly.
        """
        for i in range(1, self.t_index + 1):
            new_distance = instance.get_distance(i, ell)
            if new_distance < self.distance_to_closest_facility[i]:
                self.distance_to_closest_facility[i] = new_distance

    def update_distances_new_point(self, instance: FLOfflineInstance) -> None:
        """
        Compute closest facility to point, update distances accordingly (only at self.t_index).
        """
        min_distance = sys.float_info.max
        for j in self.facilities:
            if j == -1:
                continue
            new_distance = instance.get_distance(self.t_index, j)
            min_distance = min(min_distance, new_distance)
        self.distance_to_closest_facility[self.t_index] = min_distance

    def update(
        self, instance: FLOfflineInstance, ell: int = None, service_cost: float = None
    ) -> None:
        """
        Update algorithm state. This can be triggered by:
            - A new point self.offline_instance.points[self.t_index] arriving (ell = None, service_cost = None)
            - A new facility (ell) has been built
            - An iteration has ended and we have the final service cost for the time period (previous + new point on previous facility set)
        """
        if ell and service_cost:
            raise ValueError("ell and service_cost cannot be set at the same time.")
        elif ell:
            _LOGGER.log_special(
                f"Updating state at time {self.t_index}, to add facility {ell}: {instance.points[ell].x}"
            )
            self.facilities[self.t_index] = ell
            self.num_facilities += 1
            self.update_distances_new_facility(instance, ell)
            self.cum_var_cost = self.compute_service_cost(new_facility=True)
            self.service_costs[self.t_index] = self.cum_var_cost
            self.t_index += 1
            logging.debug(
                f"Facilities: {self.facilities}, distances: {self.distance_to_closest_facility}."
            )
        elif service_cost:
            logging.debug(
                f"Updating state at time {self.t_index}, for service cost {service_cost}."
            )
            self.facilities[self.t_index] = -1
            self.service_costs[self.t_index] = service_cost
            self.cum_var_cost += service_cost
            self.t_index += 1
            logging.debug(
                f"Facilities: {self.facilities}, distances: {self.distance_to_closest_facility}."
            )
        else:
            logging.debug(
                f"Updating state at time {self.t_index} to add demand point {self.t_index}: {instance.points[self.t_index].x}."
            )
            self.update_distances_new_point(instance)
            _LOGGER.log_body(
                f"Distances after update: {self.distance_to_closest_facility}"
            )

    def compute_service_cost(self, new_facility: bool = False) -> float:
        """
        Compute service cost for current time period.
        The updated self.distance_to_closest_facility[self.t_index] encodes the correct current facility set.
            - if no facility has been built since the last distance update, self.service_costs[self.t_index - 1] is a lower bound.
            - if a new facility has been built we have to check the whole distance list as the new facility may have better served some point that was the previous max min.
        """
        if new_facility:
            return max(self.distance_to_closest_facility[: self.t_index + 1])
        else:
            return max(
                self.distance_to_closest_facility[self.t_index],
                self.service_costs[self.t_index - 1],
            )

    def bare_final_state(
        self, facilities_service_costs: Tuple[List[int], List[float]]
    ) -> None:
        """
        Set final state without any additional processing.
        """
        self.facilities = facilities_service_costs[0]
        self.service_costs = facilities_service_costs[1]
        for x in self.facilities:
            if x != -1:
                self.num_facilities += 1
        self.t_index = self.T
        self._is_set = True

    def set_final_objective(self) -> None:
        """
        Set final objective value.
        """
        logging.debug("Final service costs:")
        for t, service_cost in enumerate(self.service_costs):
            logging.debug(f"t: {t}, service cost: {service_cost}")
        logging.debug(f"Sum of service costs: {sum(self.service_costs)}")
        self.objective = self.Gamma * (self.num_facilities - 1) + sum(
            self.service_costs
        )
        logging.debug(f"Objective: {self.objective}")


class FLSolution(Data):
    """
    Class to hold a solution for the online min max center problem.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n: int = 0
        self.T: int = 0
        self.Gamma: float = 0.0
        self.objective: float = 0.0
        self.running_time_s: float = 0.0
        self.running_time_ms: float = 0.0
        self.optimal: bool = False
        self.num_facilities: int = 0
        self.facilities: List[int] = []
        self.distance_to_closest_facility: List[float] = []
        self.service_costs: List[float] = []
        self.solver: str = ""

    def print(self, name: str = "Final") -> None:
        _LOGGER.log_subheader(
            f" {name} Solution:  Objective: {self.objective:.2f}, Optimal: {self.optimal}, Running Time: {self.running_time_ms:.2f} ms"
        )
        if self.num_facilities == 1:
            _LOGGER.log_body("Built no additional facilities")
        elif self.num_facilities == 2:
            _LOGGER.log_body(
                f"Built 1 facility (in addition to x_0): {self.facilities}"
            )
        else:
            _LOGGER.log_body(
                f"Built {self.num_facilities - 1} facilities (in addition to x_0): {self.facilities}"
            )
        _LOGGER.log_body(
            f"Final service distances (closest distance to a facility, for all i): {self.distance_to_closest_facility}"
        )
        _LOGGER.log_body(
            f"Time horizon service costs (max min cost, for all t): {self.service_costs}"
        )

    def set_running_time(self, time: float) -> None:
        self.running_time_s = time
        self.running_time_ms = time * 1000

    def set_optimal(self, optimal: bool) -> None:
        self.optimal = optimal

    def set_solver(self, solver: str) -> None:
        self.solver = solver

    def from_cvtca(
        self, state: CVCTState, time: float, optimal: bool, solver: str
    ) -> None:
        self.n = state.n
        self.T = state.T
        self.Gamma = state.Gamma
        state.set_final_objective()
        self.objective = state.objective
        self.num_facilities = state.num_facilities
        self.facilities = state.facilities
        self.distance_to_closest_facility = state.distance_to_closest_facility
        self.service_costs = state.service_costs
        self.set_running_time(time)
        self.set_optimal(optimal)
        self.set_solver(solver)
        self._is_set = True


def main():
    instance = FLOfflineInstance(INSTANCE_SHAPES["test"])
    instance.set_random()


if __name__ == "__main__":
    main()
