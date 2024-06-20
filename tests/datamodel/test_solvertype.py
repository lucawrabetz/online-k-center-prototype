import pytest
from typing import List

from optiface.datamodel.feature import Feature
from optiface.datamodel.solvertype import SolverType

_FEATURE_GUROBI_SYMMETRY: Feature = Feature("g_sym", 0, int, "Gurobi Symmetry", "GSym")
_SOLVER_MIP_NAME: str = "MIP"
_SOLVER_MIP_PARAMETERS: List[Feature] = []
_SOLVER_SMIP_NAME: str = "SMIP"
_SOLVER_SMIP_PARAMETERS: List[Feature] = [_FEATURE_GUROBI_SYMMETRY]


@pytest.fixture
def solver_mip():
    return SolverType(_SOLVER_MIP_NAME, _SOLVER_MIP_PARAMETERS)


@pytest.fixture
def solver_smip():
    return SolverType(_SOLVER_SMIP_NAME, _SOLVER_SMIP_PARAMETERS)


def test_solver_mip_attributes(solver_mip):
    assert solver_mip.name == _SOLVER_MIP_NAME
    assert solver_mip.parameters == _SOLVER_MIP_PARAMETERS


def test_solver_smip_attributes(solver_smip):
    assert solver_smip.name == _SOLVER_SMIP_NAME
    assert len(solver_smip.parameters) == 1
    assert solver_smip.parameters[0].name == _FEATURE_GUROBI_SYMMETRY.name
    assert solver_smip.parameters[0].default == _FEATURE_GUROBI_SYMMETRY.default
    assert solver_smip.parameters[0].type == _FEATURE_GUROBI_SYMMETRY.type
    assert (
        solver_smip.parameters[0].pretty_output_name
        == _FEATURE_GUROBI_SYMMETRY.pretty_output_name
    )
    assert (
        solver_smip.parameters[0].compressed_output_name
        == _FEATURE_GUROBI_SYMMETRY.compressed_output_name
    )
