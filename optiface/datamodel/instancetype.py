import os

from optiface import paths
from optiface import logger


class InstanceType:
    """
    Class to store all non-functional metadata about an instance type.
    """

    def __init__(
        self, set_name: str = "test", n: int = 2, T: int = 3, instance_id: int = None
    ) -> None:
        self._set_name: str = set_name
        self._n: int = n
        self._T: int = T
        # Unique identifier if all instance parameters already exist in another instance, only set for saved instances.
        self._id: int = -1 if instance_id is None else instance_id
        self._file_path: str = ""
        self.update_filepath()

    @property
    def set_name(self) -> str:
        return self._set_name

    @property
    def n(self) -> int:
        return self._n

    @property
    def T(self) -> int:
        return self._T

    @property
    def file_path(self) -> str:
        self.update_filepath()
        if self._file_path == "":
            # TODO: add warning here
            return ""
        else:
            return self._file_path

    def update_filepath(self) -> None:
        if self._id >= -1:
            self._file_path = os.path.join(
                paths._DAT_DIR,
                self._set_name,
                f"{self._set_name}_{self._n}_{self._T}_{self._id}.csv",
            )
        else:
            self._file_path = ""

    def set_id(self, instance_id: int) -> None:
        self._id = instance_id
        self.update_filepath()

    def from_filename(self, filename: str) -> None:
        split = filename.split("_")
        if len(split) != 4:
            raise ValueError("Invalid filename")
        self._set_name = split[0]
        self._n = int(split[1])
        self._T = int(split[2])
        self._id = int(split[3].replace(".csv", ""))

    def print(self) -> None:
        logger.log_body(f"n: {self._n}, T: {self._T}")
