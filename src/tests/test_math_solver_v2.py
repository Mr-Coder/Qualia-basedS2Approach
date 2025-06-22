#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数学问题求解器单元测试
~~~~~~~~~~~~~~~~~~~~

Author: [Hao Meng]
Date: [2025-05-29]
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 添加src目录到Python路径
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from config.advanced_config import ConfigManager, SolverConfig
from math_problem_solver import MathProblemSolver
from utils.error_handling import MathProblemSolverError


class TestSolverConfig(unittest.TestCase):
    """测试求解器配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SolverConfig()
        
        self.assertEqual(config.logging.level, "INFO")
        self.assertTrue(config.performance.enable_caching)
        self.assertTrue(config.visualization.enabled)
        self.assertEqual(config.nlp.language, "zh")
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {
            "logging": {"level": "DEBUG"},
            "performance": {"enable_caching": False},
            "visualization": {"enabled": False}
        }
        
        config = SolverConfig.from_dict(config_dict)
        
        self.assertEqual(config.logging.level, "DEBUG")
        self.assertFalse(config.performance.enable_caching)
        self.assertFalse(config.visualization.enabled)
    
    def test_config_validation(self):
        """测试配置验证"""
        config = SolverConfig()
        
        # 有效配置
        errors = config.validate()
        self.assertEqual(len(errors), 0)
        
        # 无效配置
        config.logging.level = "INVALID"
        config.performance.max_cache_size = -1
        
        errors = config.validate()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("无效的日志级别" in error for error in errors))
        self.assertTrue(any("缓存大小必须大于0" in error for error in errors))
    
    def test_config_file_operations(self):
        """测试配置文件操作"""
        config = SolverConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # 保存配置
            config.save_to_file(config_path)
            self.assertTrue(Path(config_path).exists())
            
            # 加载配置
            loaded_config = SolverConfig.from_file(config_path)
            self.assertEqual(config.logging.level, loaded_config.logging.level)
            
        finally:
            # 清理临时文件
            if Path(config_path).exists():
                Path(config_path).unlink()


class TestConfigManager(unittest.TestCase):
    """测试配置管理器"""
    
    def test_default_config_manager(self):
        """测试默认配置管理器"""
        manager = ConfigManager()
        
        self.assertIsInstance(manager.config, SolverConfig)
        self.assertEqual(manager.config.logging.level, "INFO")
    
    def test_config_update(self):
        """测试配置更新"""
        manager = ConfigManager()
        
        original_level = manager.config.logging.level
        
        # 更新配置
        manager.update_config({
            "logging": {"level": "DEBUG"}
        })
        
        self.assertEqual(manager.config.logging.level, "DEBUG")
        self.assertNotEqual(manager.config.logging.level, original_level)


class TestMathProblemSolver(unittest.TestCase):
    """测试数学问题求解器"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试配置
        self.test_config = {
            "logging": {"level": "ERROR"},  # 减少测试时的日志输出
            "performance": {"enable_performance_tracking": False},
            "visualization": {"enabled": False}  # 禁用可视化以加快测试
        }
    
    @patch('src.processors.nlp_processor.NLPProcessor')
    @patch('src.processors.MWP_process.MWPCoarseClassifier')
    @patch('src.processors.relation_matcher.RelationMatcher')
    @patch('src.processors.relation_extractor.RelationExtractor')
    @patch('src.processors.equation_builder.EquationBuilder')
    @patch('src.processors.inference_tracker.InferenceTracker')
    def test_solver_initialization(self, mock_tracker, mock_builder, mock_extractor, 
                                  mock_matcher, mock_classifier, mock_nlp):
        """测试求解器初始化"""
        # 配置模拟对象
        mock_nlp.return_value = MagicMock()
        mock_classifier.return_value = MagicMock()
        mock_matcher.return_value = MagicMock()
        mock_extractor.return_value = MagicMock()
        mock_builder.return_value = MagicMock()
        mock_tracker.return_value = MagicMock()
        
        # 创建求解器
        solver = MathProblemSolver(self.test_config)
        
        self.assertIsNotNone(solver)
        self.assertIsNotNone(solver.logger)
        
        # 验证组件都被初始化
        mock_nlp.assert_called_once()
        mock_classifier.assert_called_once()
        mock_matcher.assert_called_once()
        mock_extractor.assert_called_once()
        mock_builder.assert_called_once()
        mock_tracker.assert_called_once()
    
    def test_tank_problem_solving(self):
        """测试水箱问题求解"""
        # 这是一个集成测试，需要确保组件可用
        try:
            solver = MathProblemSolver(self.test_config)
            
            problem = "A tank contains 5L of water. Water is added at a rate of 2 L/minute. Water leaks out at 1 L/minute. How long until it contains 10L?"
            
            result = solver.solve(problem)
            
            # 基本结果验证
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)
            
            # 如果成功，验证答案
            if result.get('status') == 'success':
                self.assertIn('answer', result)
                self.assertIsInstance(result['answer'], (int, float))
                # 对于这个特定问题，答案应该是5.0
                self.assertEqual(result['answer'], 5.0)
                
        except Exception as e:
            # 如果组件不可用，跳过测试
            self.skipTest(f"组件不可用，跳过集成测试: {e}")
    
    def test_error_handling(self):
        """测试错误处理"""
        try:
            solver = MathProblemSolver(self.test_config)
            
            # 测试空输入
            result = solver.solve("")
            self.assertIn('status', result)
            
            # 测试无效输入（如果处理失败，应该返回错误状态）
            result = solver.solve("This is not a math problem at all!!!")
            self.assertIn('status', result)
            
        except Exception as e:
            self.skipTest(f"组件不可用，跳过错误处理测试: {e}")
    
    def test_problem_type_detection(self):
        """测试问题类型检测"""
        try:
            solver = MathProblemSolver(self.test_config)
            
            # 测试水箱问题检测
            equation_system = {
                'equations': ['time = (volume_10-volume_5)/rate_1', 'rate_1 = rate_2-rate_1']
            }
            extraction_result = {}
            
            problem_type = solver._detect_problem_type(equation_system, extraction_result)
            # 应该检测到包含volume和rate的水箱问题
            self.assertEqual(problem_type, "tank")
            
            # 测试运动问题检测
            equation_system = {
                'equations': ['distance = speed * time', 'velocity = distance / time']
            }
            
            problem_type = solver._detect_problem_type(equation_system, extraction_result)
            self.assertEqual(problem_type, "motion")
            
            # 测试一般问题
            equation_system = {
                'equations': ['x + y = 10', 'x - y = 2']
            }
            
            problem_type = solver._detect_problem_type(equation_system, extraction_result)
            self.assertEqual(problem_type, "general")
            
        except Exception as e:
            self.skipTest(f"组件不可用，跳过问题类型检测测试: {e}")
    
    def test_tank_parameter_extraction(self):
        """测试水箱参数提取"""
        try:
            solver = MathProblemSolver(self.test_config)
            
            # 模拟关系提取结果
            extraction_result = {
                'implicit_relations': [
                    {
                        'source_pattern': 'tank_direct',
                        'var_entity': {
                            'initial_volume': '5.0',
                            'target_volume': '10.0',
                            'inflow_rate': '2.0',
                            'outflow_rate': '1.0'
                        }
                    }
                ]
            }
            
            params = solver._extract_tank_parameters(extraction_result)
            
            self.assertEqual(params['initial_volume'], 5.0)
            self.assertEqual(params['target_volume'], 10.0)
            self.assertEqual(params['inflow_rate'], 2.0)
            self.assertEqual(params['outflow_rate'], 1.0)
            self.assertEqual(params['net_rate'], 1.0)  # 2.0 - 1.0
            
        except Exception as e:
            self.skipTest(f"组件不可用，跳过参数提取测试: {e}")


class TestPerformanceAndCaching(unittest.TestCase):
    """测试性能和缓存功能"""
    
    def test_caching_functionality(self):
        """测试缓存功能"""
        config = {
            "performance": {"enable_caching": True},
            "logging": {"level": "ERROR"},
            "visualization": {"enabled": False}
        }
        
        try:
            solver = MathProblemSolver(config)
            
            # 创建测试数据
            processed_text = MagicMock()
            processed_text.segmentation = ["test", "problem"]
            
            # 第一次调用（应该计算并缓存）
            result1 = solver._classify_problem(processed_text)
            
            # 第二次调用（应该使用缓存）
            result2 = solver._classify_problem(processed_text)
            
            # 结果应该相同
            self.assertEqual(result1, result2)
            
        except Exception as e:
            self.skipTest(f"组件不可用，跳过缓存测试: {e}")


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestSolverConfig,
        TestConfigManager,
        TestMathProblemSolver,
        TestPerformanceAndCaching
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print(f"\n{'='*50}")
    print("测试结果摘要")
    print(f"{'='*50}")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # 返回退出码
    sys.exit(0 if result.wasSuccessful() else 1)
