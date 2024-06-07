import os
import logging
from typing import List
from argparse import ArgumentParser
from allowed_types import FLSolverType, FLInstanceType
from util import _DAT
from problem import FLOfflineInstance

from log_config import setup_logging, _LOGGER

setup_logging()

# Testbeds could be defined:
#  (a) "explicitly": a list of instance type objects
#  (b) "theoretically": a list of lists of values for instance parameters, leading to a list of instance type objects of all possible combinations

_RESERVED_SETNAMES = os.listdir(_DAT)


class TestbedGenerator:
    """
    Populate a test directory associated with a new set name with randomly generated instances.
    """

    def __init__(
        self, set_name: str, num_instances: int, instance_ids: List[FLInstanceType]
    ) -> None:
        """
        set_name: name of the testbed
        num_instances: number of instances to generate per instance type / shape / parameter combination
        """
        if set_name in _RESERVED_SETNAMES:
            raise ValueError(f"Set name {set_name} is already used.")
        self.set_name = set_name
        self.num_instances = num_instances
        self.instance_ids = instance_ids
        self.final_ids: List[FLInstanceType] = []

    def generate_for_one_instanceid(self, instance_id: FLInstanceType) -> None:
        if instance_id.set_name != self.set_name:
            raise ValueError(
                f"Instance set name {instance_id.set_name} does not match testbed set name {self.set_name}."
            )
        if instance_id.id != -1:
            raise ValueError(
                f"Instance type {instance_id.set_name} already has id {instance_id.id}."
            )
        for i in range(self.num_instances):
            new_id = FLInstanceType(
                set_name=self.set_name, n=instance_id.n, T=instance_id.T
            )
            new_id.set_id(i)
            self.final_ids.append(new_id)
            instance = FLOfflineInstance(new_id)
            # for now, our random experiments always use the unit "square" and the interval [0, 1] for gamma, both uniform
            instance.set_random(low=0, high=1)
            instance.write_to_csv()

    def write(self) -> List[FLInstanceType]:
        """
        Write instances to disk.
        Returns list of instance types with id's set.
        """
        if not os.path.exists(_DAT):
            os.makedirs(_DAT)
        if not os.path.exists(os.path.join(_DAT, self.set_name)):
            os.makedirs(os.path.join(_DAT, self.set_name))
        for instance_id in self.instance_ids:
            self.generate_for_one_instanceid(instance_id)
        return self.final_ids


def id_factory(set_name: str, n: List[int], T: List[int]) -> List[FLInstanceType]:
    """
    Generate instance types with all combinations of n and T.
    """
    ids = []
    for n_val in n:
        for T_val in T:
            ids.append(FLInstanceType(set_name=set_name, n=n_val, T=T_val))
    return ids


def main():
    parser = ArgumentParser()
    parser.add_argument("--set_name", type=str, default="test")
    set_name = parser.parse_args().set_name
    num_instances = 30
    dimensions = [100000, 1000000]
    time_periods = [1000]
    ids = id_factory(set_name, dimensions, time_periods)
    generator = TestbedGenerator(set_name, num_instances, ids)
    generator.write()


if __name__ == "__main__":
    main()
