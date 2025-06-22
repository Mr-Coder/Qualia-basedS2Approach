import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from processors.equation_builder import EquationBuilder


@pytest.fixture
def builder():
    return EquationBuilder({})

def test_equation_building_basic(builder):
    extraction_result = {
        'explicit_relations': [
            {'relation': 'distance=speed*time', 'var_entity': {'distance': 's', 'speed': 'v', 'time': 't'}}
        ],
        'implicit_relations': []
    }
    result = builder.build_equations(extraction_result)
    assert 'equations' in result
    assert isinstance(result['equations'], list)
    assert any('distance' in eq or 's' in eq for eq in result['equations']) 