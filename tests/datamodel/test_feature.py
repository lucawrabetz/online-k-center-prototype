import pytest
from optiface.datamodel.feature import Feature

# TODO: add logging for tests.
# Global variables for expected values
RUN_ID_FEATURE_NAME = "run_id"
RUN_ID_FEATURE_DEFAULT = 0
RUN_ID_FEATURE_TYPE = int
RUN_ID_FEATURE_PRETTY_OUTPUT_NAME = "Run ID"
RUN_ID_FEATURE_COMPRESSED_OUTPUT_NAME = "RunID"

SET_NAME_FEATURE_NAME = "set_name"
SET_NAME_FEATURE_DEFAULT = "test"
SET_NAME_FEATURE_TYPE = str
SET_NAME_FEATURE_PRETTY_OUTPUT_NAME = "Set Name"
SET_NAME_FEATURE_COMPRESSED_OUTPUT_NAME = "SNAME"


# Fixtures for valid Feature instances
@pytest.fixture
def run_id_feature():
    return Feature(
        name=RUN_ID_FEATURE_NAME,
        default=RUN_ID_FEATURE_DEFAULT,
        feature_type=RUN_ID_FEATURE_TYPE,
        pretty_output_name=RUN_ID_FEATURE_PRETTY_OUTPUT_NAME,
        compressed_output_name=RUN_ID_FEATURE_COMPRESSED_OUTPUT_NAME,
    )


@pytest.fixture
def set_name_feature():
    return Feature(
        name=SET_NAME_FEATURE_NAME,
        default=SET_NAME_FEATURE_DEFAULT,
        feature_type=SET_NAME_FEATURE_TYPE,
        pretty_output_name=SET_NAME_FEATURE_PRETTY_OUTPUT_NAME,
        compressed_output_name=SET_NAME_FEATURE_COMPRESSED_OUTPUT_NAME,
    )


# Test that attributes are correctly set.
def test_run_id_feature_attributes(run_id_feature):
    assert run_id_feature.name == RUN_ID_FEATURE_NAME
    assert run_id_feature.default == RUN_ID_FEATURE_DEFAULT
    assert run_id_feature.type == RUN_ID_FEATURE_TYPE
    assert run_id_feature.pretty_output_name == RUN_ID_FEATURE_PRETTY_OUTPUT_NAME
    assert (
        run_id_feature.compressed_output_name == RUN_ID_FEATURE_COMPRESSED_OUTPUT_NAME
    )


def test_set_name_feature_attributes(set_name_feature):
    assert set_name_feature.name == SET_NAME_FEATURE_NAME
    assert set_name_feature.default == SET_NAME_FEATURE_DEFAULT
    assert set_name_feature.type == SET_NAME_FEATURE_TYPE
    assert set_name_feature.pretty_output_name == SET_NAME_FEATURE_PRETTY_OUTPUT_NAME
    assert (
        set_name_feature.compressed_output_name
        == SET_NAME_FEATURE_COMPRESSED_OUTPUT_NAME
    )


# Tests to ensure errors are triggered correctly.
def test_name_none():
    with pytest.raises(TypeError, match="Name cannot be None"):
        Feature(name=None)


def test_default_none():
    with pytest.raises(TypeError, match="Default value cannot be None"):
        Feature(name=RUN_ID_FEATURE_NAME, default=None)


def test_default_type_mismatch():
    with pytest.raises(
        TypeError,
        match=f"Default value test does not match feature type: {RUN_ID_FEATURE_TYPE}",
    ):
        Feature(
            name=RUN_ID_FEATURE_NAME, default="test", feature_type=RUN_ID_FEATURE_TYPE
        )


def test_invalid_allowed_values_type():
    with pytest.raises(
        TypeError,
        match=f"Allowed value test does not match feature type {RUN_ID_FEATURE_TYPE}",
    ):
        Feature(
            name=RUN_ID_FEATURE_NAME,
            default=RUN_ID_FEATURE_DEFAULT,
            feature_type=RUN_ID_FEATURE_TYPE,
            allowed_values=["test"],
        )


# Test that the pretty_output_name and compressed_output_name are set to the name if not passed.
def test_names_none():
    feature = Feature(
        name=RUN_ID_FEATURE_NAME,
        default=RUN_ID_FEATURE_DEFAULT,
        feature_type=RUN_ID_FEATURE_TYPE,
        pretty_output_name=None,
        compressed_output_name=None,
    )
    assert feature.pretty_output_name == RUN_ID_FEATURE_NAME
    assert feature.compressed_output_name == RUN_ID_FEATURE_NAME
