import logging
from log_config import _LOGGER
from abc import ABC, abstractmethod
from allowed_types import _ALLOWED_TYPES
from typing import Any, List, Dict, Tuple


class IFeature(ABC):
    def __init__(
        self,
        name: str,
        default: Any = None,
        feature_type: type = str,
        pretty_output_name: str = None,
        compressed_output_name: str = None,
        allowed_values: List[Any] = None,
    ) -> None:
        """
        If default exists, feature_type is only used as a sanity check, we set the type of default to self.type.
        TODO: put init in a configure function rather than the constructor.
        """
        self.name = name
        if not default:
            self.default = None
            self.type = feature_type
        elif type(default) != feature_type:
            raise ValueError(f"Default value {default} is not of type {feature_type}")
        else:
            self.default = default
            self.type = type(default)

        if pretty_output_name:
            self.pretty_output_name = pretty_output_name
        else:
            self.pretty_output_name = name
        if compressed_output_name:
            self.compressed_output_name = compressed_output_name
        else:
            self.compressed_output_name = name
        if allowed_values:
            self.allowed_values = allowed_values
        else:
            self.allowed_values = []
