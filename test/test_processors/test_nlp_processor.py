import json
import logging
import os
import sys
from typing import Any, Dict

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.structures import ProcessedText
from processors.nlp_processor import NLPProcessor

# 添加测试数据
TEST_PROBLEMS = {
    "ice_cube_problem": {
        "text": "Ice cubes, each with a volume of 200 cm³, are dropped into a tank containing 5L of water at a rate of one cube per minute. Simultaneously, water is leaking from the tank through a tube at a rate of 2 mL per second. How long will it take for the water level in the tank to rise to 9L?",
        "expected_solution": {
            "time": 120,
            "water_change": 4
        }
    }
}

@pytest.fixture
def nlp_processor():
    """创建 NLPProcessor 实例的 fixture"""
    config = {
        'use_mps': False,
        'use_gpu': False
    }
    return NLPProcessor(config)

def test_initialization():
    """测试初始化"""
    processor = NLPProcessor()
    assert processor is not None
    assert processor.device in ['cpu', 'cuda', 'mps']

def test_basic_process(nlp_processor, caplog):
    """测试基本文本处理"""
    caplog.set_level(logging.DEBUG)
    
    test_text = TEST_PROBLEMS["ice_cube_problem"]["text"]
    result = nlp_processor.process(test_text)
    
    assert isinstance(result, ProcessedText)
    assert len(result.segmentation) > 0
    assert len(result.pos_tags) > 0
    assert isinstance(result.dependencies, list)
    assert all(isinstance(dep, tuple) and len(dep) == 3 for dep in result.dependencies)
    assert isinstance(result.semantic_roles, dict)
    
    print("\n=== 基本处理测试结果 ===")
    print(f"原始文本: {result.raw_text}")
    print(f"分词结果: {result.segmentation}")
    print(f"词性标注: {result.pos_tags}")
    print(f"依存关系: {result.dependencies}")
    print(f"语义角色: {result.semantic_roles}")

def test_error_handling(nlp_processor):
    """测试错误处理"""
    with pytest.raises(ValueError):
        nlp_processor.process("")
    
    with pytest.raises(ValueError):
        nlp_processor.process(None)
