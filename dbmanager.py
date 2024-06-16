import logging
import pandas as pd
from util import _FINALDB
from data_model import _DATA_MODEL, _COLUMN_INDEX
from feature_interface import IFeature

from log_config import setup_logging, _LOGGER

setup_logging()


def add_feature_column(df, feature: IFeature) -> None:
    df[feature.name] = feature.default
    _LOGGER.log_body(f"Added feature {feature.name} to the table")


def add_missing_features(table: str = _FINALDB) -> None:
    df = pd.read_csv(table)
    _LOGGER.log_header(f"Adding missing features to the table {table}")
    for name, feature in _DATA_MODEL.features.items():
        if feature.name not in df.columns:
            add_feature_column(df, feature)
    df = df[_COLUMN_INDEX]
    df.to_csv(table, index=False)


def main():
    add_missing_features()


if __name__ == "__main__":
    main()
