import logging
from typing import Dict, List

import pytest

from models.structures import ProcessedText
from processors.relation_extractor import ExtractionResult, RelationExtractor


@pytest.fixture
def ice_cube_text() -> ProcessedText:
    """准备冰块问题的测试数据"""
    return ProcessedText(
        raw_text="Ice cubes, each with a volume of 200 cm³, are dropped into a tank containing 5L of water",
        segmentation=[
            "Ice", "cubes", "each", "with", "volume", "of", "200", "cm³", "are", 
            "dropped", "into", "tank", "containing", "5L", "of", "water"
        ],
        pos_tags=[
            "n", "n", "nz", "v", "n", "wp", "m", "q", "v", 
            "v", "p", "n", "v", "m", "wp", "n"
        ],
        dependencies=[
            ("Ice", "ATT", "cubes"),
            ("cubes", "SBV", "dropped"),
            ("200", "ATT", "cm³"),
            ("tank", "VOB", "containing"),
            ("5L", "ATT", "water")
        ],
        semantic_roles={}
    )

@pytest.fixture
def extractor():
    """创建关系提取器实例"""
    return RelationExtractor()

def test_relation_extractor_creation(extractor):
    """测试关系提取器的创建"""
    assert extractor is not None
    assert hasattr(extractor, 'process_text')
    assert hasattr(extractor, 'patterns')
    assert len(extractor.patterns) > 0

def test_simple_extraction(extractor, caplog):
    """测试简单文本的提取"""
    caplog.set_level(logging.DEBUG)
    result = extractor.process_text(
        "测试文本",
        ["测试", "文本"],
        ["n", "n"],
        []  # Empty dependencies list
    )
    
    assert result is not None
    assert 'status' in result
    assert result['status'] == 'success'
    assert 'explicit_relations' in result
    assert 'implicit_relations' in result
    assert isinstance(result['explicit_relations'], list)
    assert isinstance(result['implicit_relations'], list)

def test_ice_cube_problem_extraction(extractor, ice_cube_text, caplog):
    """测试冰块问题的关系提取"""
    caplog.set_level(logging.DEBUG)
    
    result = extractor.process_text(
        ice_cube_text.raw_text,
        ice_cube_text.segmentation,
        ice_cube_text.pos_tags,
        ice_cube_text.dependencies
    )
    
    assert result['status'] == 'success'
    explicit = result['explicit_relations']
    implicit = result['implicit_relations']
    
    # 验证数值关系
    numerical_relations = [
        rel for rel in explicit 
        if rel.get('type') == 'numerical'
    ]
    assert len(numerical_relations) > 0
    assert any('200' in str(rel) for rel in numerical_relations)
    assert any('cm³' in str(rel) for rel in numerical_relations)
    
    # 验证动作关系
    action_relations = [
        rel for rel in explicit 
        if rel.get('type') == 'action'
    ]
    assert len(action_relations) > 0
    assert any('dropped' in str(rel) for rel in action_relations)

def test_error_handling(extractor):
    """测试错误处理情况"""
    # 测试空文本
    result = extractor.process_text("", [], [], [])
    assert result['status'] == 'success'
    assert len(result['explicit_relations']) == 0
    assert len(result['implicit_relations']) == 0
    
    # 测试无效文本
    result = extractor.process_text(
        "测试",
        ["测试"],
        ["n", "v"],  # 长度不匹配
        []
    )
    assert result['status'] == 'success'
    assert len(result['explicit_relations']) == 0
    assert len(result['implicit_relations']) == 0

# 更新测试用例
test_cases = [
    (
        ProcessedText(
            raw_text="The temperature is 25 degrees",
            segmentation=["temperature", "is", "25", "degrees"],
            pos_tags=["n", "v", "m", "n"],
            dependencies=[
                ("temperature", "SBV", "is"),
                ("25", "ATT", "degrees")
            ],
            semantic_roles={}
        ),
        {
            "explicit": {
                "numerical": 1,
                "action": 0
            },
            "implicit": 0
        }
    ),
    (
        ProcessedText(
            raw_text="Water flows at 2 liters per second",
            segmentation=["Water", "flows", "at", "2", "liters", "per", "second"],
            pos_tags=["n", "v", "p", "m", "n", "p", "n"],
            dependencies=[
                ("Water", "SBV", "flows"),
                ("2", "ATT", "liters")
            ],
            semantic_roles={}
        ),
        {
            "explicit": {
                "numerical": 1,
                "action": 1
            },
            "implicit": 0
        }
    )
]

@pytest.mark.parametrize("input_text,expected_counts", test_cases)
def test_relation_counts(extractor, input_text, expected_counts):
    """测试不同文本的关系提取数量和类型"""
    result = extractor.process_text(
        input_text.raw_text,
        input_text.segmentation,
        input_text.pos_tags,
        input_text.dependencies
    )
    
    # 验证显式关系数量和类型
    for rel_type, count in expected_counts["explicit"].items():
        actual_count = len([r for r in result['explicit_relations'] if r.get('type') == rel_type])
        assert actual_count == count, f"Expected {count} {rel_type} relations, got {actual_count}"
    
    # 验证隐式关系总数
    assert len(result['implicit_relations']) == expected_counts["implicit"]

def test_logging_output(extractor, ice_cube_text, caplog):
    """测试日志输出的完整性"""
    caplog.set_level(logging.DEBUG)
    
    result = extractor.process_text(
        ice_cube_text.raw_text,
        ice_cube_text.segmentation,
        ice_cube_text.pos_tags,
        ice_cube_text.dependencies
    )
    
    print("\n=== Ice Cube Problem Test Results ===\n")
    
    print("Explicit Relations:")
    for rel in result['explicit_relations']:
        print(f"- {rel}")
    
    print("\nImplicit Relations:")
    implicit_relations = result['implicit_relations']
    assert len(implicit_relations) > 0, "应该有隐含关系被提取出来"
    
    # 验证水位变化关系
    water_level_relations = [rel for rel in implicit_relations if rel.get('relation_type') == 'water_level_change']
    assert len(water_level_relations) > 0, "应该包含水位变化关系"
    
    # 验证体积置换关系
    volume_relations = [rel for rel in implicit_relations if rel.get('relation_type') == 'volume_displacement']
    assert len(volume_relations) > 0, "应该包含体积置换关系"
    
    # 验证容量限制关系
    capacity_relations = [rel for rel in implicit_relations if rel.get('relation_type') == 'capacity_constraint']
    assert len(capacity_relations) > 0, "应该包含容量限制关系"
    
    for rel in implicit_relations:
        print(f"- {rel}")
