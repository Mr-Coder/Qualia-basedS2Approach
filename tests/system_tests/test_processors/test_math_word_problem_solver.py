#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

# 添加 src 目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from config.config import ConfigError
from models.structures import ProcessedText
from processors.math_word_problem_solver import MathProblemSolver
from processors.nlp_processor import NLPProcessor
from processors.relation_extractor import RelationExtractor
from processors.equation_builder import EquationBuilder
from processors.solution_generator import SolutionGenerator
from processors.inference_tracker import InferenceTracker

# 测试用的配置数据
TEST_CONFIG = {
    'version': '1.0.0',
    'log_level': 'INFO',
    'log_file': 'math_solver.log',
    'use_gpu': False,
    'use_mps': True,
    'num_threads': 8,
    'batch_size': 32,
    'nlp': {
        'model_path': 'models/nlp',
        'language': 'zh',
        'max_length': 512,
        'device_settings': {
            'use_mps': True,
            'use_gpu': False,
            'fallback_to_cpu': True,
            'device_priority': ['mps', 'cpu']
        }
    },
    'output': {
        'save_steps': True,
        'save_path': 'results',
        'format': 'json'
    },
    'solver': {
        'max_iterations': 1000,
        'tolerance': 1e-6,
        'device': 'mps'
    }
}

# 测试用的问题数据
TEST_PROBLEM = "小明有5个苹果，给了小红3个，还剩多少个苹果？"

# 测试用的处理结果
TEST_PROCESSED_TEXT = ProcessedText(
    raw_text=TEST_PROBLEM,
    segmentation=['小明', '有', '5', '个', '苹果', '，', '给', '了', '小红', '3', '个'],
    pos_tags=['n', 'v', 'm', 'q', 'n', 'w', 'v', 'u', 'n', 'm', 'q'],
    dependencies=[
        ('有', '小明', 'SBV'),
        ('有', '苹果', 'VOB'),
        ('苹果', '5', 'NUM'),
        ('苹果', '个', 'M'),
        ('给', '小红', 'VOB'),
        ('给', '3', 'NUM'),
        ('3', '个', 'M')
    ]
)

TEST_RELATIONS = {
    'explicit': [
        {'relation': '初始数量', 'value': 5, 'unit': '个', 'object': '苹果'},
        {'relation': '转移数量', 'value': 3, 'unit': '个', 'object': '苹果'}
    ],
    'implicit': [
        {'relation': '剩余数量', 'expression': 'x = initial - transfer', 'unit': '个', 'object': '苹果'}
    ]
}

TEST_EQUATIONS = ['x = 5 - 3']

TEST_SOLUTION = {'x': 2}

@pytest.fixture
def config_file():
    """创建临时配置文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(TEST_CONFIG, f)
        return Path(f.name)

@pytest.fixture
def solver(config_file):
    """创建 MathProblemSolver 实例"""
    return MathProblemSolver(config_path=str(config_file))

class TestMathProblemSolver:
    
    def test_init_with_valid_config(self, config_file):
        """测试使用有效配置初始化"""
        solver = MathProblemSolver(config_path=str(config_file))
        assert solver.config == TEST_CONFIG
        
    def test_init_without_config(self):
        """测试不使用配置文件初始化"""
        with pytest.raises(ConfigError):
            MathProblemSolver()
            
    def test_init_with_invalid_config_path(self):
        """测试使用无效配置文件路径初始化"""
        with pytest.raises(ConfigError):
            MathProblemSolver(config_path="invalid_path.json")
            
    @patch('processors.math_word_problem_solver.NLPProcessor')
    @patch('processors.math_word_problem_solver.RelationExtractor')
    @patch('processors.math_word_problem_solver.EquationBuilder')
    @patch('processors.math_word_problem_solver.SolutionGenerator')
    @patch('processors.math_word_problem_solver.InferenceTracker')
    def test_solve_valid_problem(self, mock_tracker, mock_solver, mock_builder, 
                               mock_extractor, mock_nlp, solver):
        """测试解决有效的数学问题"""
        # 配置模拟对象
        mock_nlp.return_value.process.return_value = TEST_PROCESSED_TEXT
        mock_extractor.return_value.extract.return_value = TEST_RELATIONS
        mock_builder.return_value.build.return_value = TEST_EQUATIONS
        mock_solver.return_value.solve.return_value = TEST_SOLUTION
        mock_tracker.return_value.get_inference_history.return_value = [
            {
                'step_name': '初始化问题',
                'input': TEST_PROBLEM,
                'output': TEST_PROCESSED_TEXT,
                'metadata': {'stage': 'nlp_processing'}
            },
            {
                'step_name': '提取关系',
                'input': TEST_PROCESSED_TEXT,
                'output': TEST_RELATIONS,
                'metadata': {'stage': 'relation_extraction'}
            }
        ]
        
        # 执行测试
        result = solver.solve(TEST_PROBLEM)
        
        # 验证结果
        assert result['solution'] == TEST_SOLUTION
        assert 'inference_steps' in result
        assert 'problem_structure' in result
        
        # 验证调用
        mock_nlp.return_value.process.assert_called_once_with(TEST_PROBLEM)
        mock_extractor.return_value.extract.assert_called_once_with(TEST_PROCESSED_TEXT)
        mock_builder.return_value.build.assert_called_once_with(TEST_RELATIONS)
        mock_solver.return_value.solve.assert_called_once_with(TEST_EQUATIONS)
        
    def test_solve_invalid_input(self, solver):
        """测试处理无效输入"""
        invalid_inputs = [None, "", 123, [], {}]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                solver.solve(invalid_input)
                
    def test_build_problem_structure(self, solver):
        """测试问题结构构建"""
        structure = solver._build_problem_structure(
            TEST_PROCESSED_TEXT,
            TEST_RELATIONS,
            TEST_EQUATIONS,
            TEST_SOLUTION
        )
        
        assert 'text_structure' in structure
        assert 'math_structure' in structure
        assert 'metadata' in structure
        
        assert structure['text_structure']['raw_text'] == TEST_PROCESSED_TEXT.raw_text
        assert structure['text_structure']['segmentation'] == TEST_PROCESSED_TEXT.segmentation
        assert structure['math_structure']['relations'] == TEST_RELATIONS
        assert structure['math_structure']['equations'] == TEST_EQUATIONS
        assert structure['math_structure']['solution'] == TEST_SOLUTION

if __name__ == '__main__':
    pytest.main(['-v'])
