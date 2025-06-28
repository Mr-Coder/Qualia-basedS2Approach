import pytest

from src.processors.relation_builder import RelationBuilder


def test_construct_implicit_relations():
    builder = RelationBuilder()
    text = "Ice cubes, each with a volume of 200 cm³, are dropped into a tank containing 5L of water"
    matches = []  # 可以根据需要添加匹配结果
    
    relations = builder.construct_implicit_relations(matches, text)
    
    # 验证是否提取到了所有需要的隐含关系
    assert len(relations) >= 3
    relation_types = [r['relation_type'] for r in relations]
    assert 'volume_displacement' in relation_types
    assert 'water_level_change' in relation_types
    assert 'capacity_constraint' in relation_types

def test_build_equation_system():
    builder = RelationBuilder()
    relations = {
        'explicit': [
            {
                'type': 'numerical',
                'expression': 'volume=200*cm³',
                'value': '200',
                'unit': 'cm³'
            }
        ],
        'implicit': [
            {
                'type': 'implicit',
                'relation_type': 'volume_displacement',
                'value': '200',
                'unit': 'cm³'
            }
        ]
    }
    
    equation_system = builder.build_equation_system(relations)
    
    # 验证方程组系统的结构
    assert 'equations' in equation_system
    assert 'variables' in equation_system
    assert 'constraints' in equation_system
    assert 'units' in equation_system