import logging
from typing import Any, List, Dict, Tuple, Type, Optional

# TODO: why was this here? - from allowed_types import _ALLOWED_TYPES
from optiface import logger


class Feature:
    """
    Fully public STRUCT for a feature.
    """

    def __init__(
        self,
        name: str = "objective",
        default: Any = 100.0,
        feature_type: Type = float,
        pretty_output_name: str = "Objective Value",
        compressed_output_name: str = "Obj",
        allowed_values: List[Any] = None,
    ) -> None:
        """
        self.set_default_and_type handles type checking for default value.
        """
        self._name: str
        self.set_name(name)
        # self._default: Any but after self.set_default_and_type, it is guaranteed to be of type feature_type
        self._type: Type
        self._default: Any
        self.set_default_and_type(default, feature_type)
        self._pretty_output_name: str
        self._compressed_output_name: str
        self.set_strings(name, pretty_output_name, compressed_output_name)
        self.set_allowed_values(allowed_values)

    def print(self) -> None:
        logger.log_body(f"Name: {self.name}")
        logger.log_body(f"Type: {self.type}")
        logger.log_body(f"Default: {self.default}")
        logger.log_body(f"Pretty Output Name: {self.pretty_output_name}")
        logger.log_body(f"Compressed Output Name: {self.compressed_output_name}")
        logger.log_body(f"Allowed Values: {self.allowed_values}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> Type:
        return self._type

    @property
    def default(self) -> Any:
        return self._default

    @property
    def pretty_output_name(self) -> str:
        return self._pretty_output_name

    @property
    def compressed_output_name(self) -> str:
        return self._compressed_output_name

    def set_name(self, name: str) -> None:
        if name is None:
            raise TypeError("Name cannot be None")
        self._name = name

    def set_default_and_type(self, default: Any, feature_type: Type) -> None:
        """
        Default must:
            - not be None
            - be of type feature_type
        """
        if default is None:
            raise TypeError("Default value cannot be None")
        if type(default) != feature_type:
            logger.log_error(
                f"Default value {default} does not match feature type: {feature_type}"
            )
            raise TypeError(
                f"Default value {default} does not match feature type: {feature_type}"
            )
        self._type = type(default)
        self._default = default

    def set_strings(
        self, name: str, pretty_output_name: str, compressed_output_name: str
    ) -> None:
        if pretty_output_name is None:
            self._pretty_output_name = name
        else:
            self._pretty_output_name = pretty_output_name
        if compressed_output_name is None:
            self._compressed_output_name = name
        else:
            self._compressed_output_name = compressed_output_name

    def set_allowed_values(self, allowed_values: Optional[List[Any]]) -> None:
        if allowed_values is None:
            self.allowed_values = []
            return
        # static typing says Any for the values, but we check them here
        for v in allowed_values:
            if type(v) != self._type:
                logger.log_error(
                    f"Allowed value {v} does not match feature type {self._type}"
                )
                raise TypeError(
                    f"Allowed value {v} does not match feature type {self._type}"
                )
        self.allowed_values = allowed_values


def main() -> None:
    RUN_ID = Feature("run_id", 0, int, "Run ID", "RunID")
    SET_NAME = Feature("set_name", "test", str, "Set Name", "SNAME")
    RUN_ID.print()
    SET_NAME.print()


if __name__ == "__main__":
    main()
