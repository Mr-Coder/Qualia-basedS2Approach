"""
Evaluation Module - Comprehensive Test Suite
===========================================

测试覆盖范围：
- 正常情况测试
- 边界条件测试  
- 异常情况测试

Author: AI Assistant
Date: 2024-07-13
"""

import logging
import os
import sys
import unittest
from typing import Any, Dict

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from evaluation import (EvaluationAPI, EvaluationOrchestrator, evaluation_api,
                        evaluation_orchestrator)


class TestEvaluationModule(unittest.TestCase):
    """Evaluation模块测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.api = EvaluationAPI()
        self.orchestrator = EvaluationOrchestrator()
        
        # 测试数据
        self.valid_results = [
            {"accuracy": 0.95, "quality": 0.88},
            {"accuracy": 0.92, "quality": 0.85},
            {"accuracy": 0.89, "quality": 0.82}
        ]
        self.valid_config = {"metrics": ["accuracy", "quality"]}
        
        # 边界测试数据
        self.empty_results = []
        self.none_input = None
        self.large_results = [{"result": i} for i in range(1000)]
        
        # 异常测试数据
        self.invalid_results = "not_a_list"
        self.invalid_type = 12345
        
    def tearDown(self):
        """测试后清理"""
        if hasattr(self.api, '_initialized') and self.api._initialized:
            self.api.shutdown()
    
    # ==================== 正常情况测试 ====================
    
    def test_module_import(self):
        """测试模块导入"""
        self.assertIsNotNone(evaluation_api)
        self.assertIsNotNone(evaluation_orchestrator)
        self.assertIsInstance(evaluation_api, EvaluationAPI)
        self.assertIsInstance(evaluation_orchestrator, EvaluationOrchestrator)
    
    def test_api_initialization(self):
        """测试API初始化"""
        result = self.api.initialize()
        self.assertTrue(result)
        self.assertTrue(self.api._initialized)
    
    def test_evaluation_performance_normal(self):
        """测试正常性能评估"""
        self.api.initialize()
        
        result = self.api.evaluate_performance(self.valid_results)
        self.assertEqual(result['status'], 'success')
        self.assertIn('evaluation_results', result)
        
        # 检查评估结果结构
        eval_results = result['evaluation_results']
        self.assertIn('accuracy', eval_results)
        self.assertIn('quality_score', eval_results)
        self.assertIn('total_samples', eval_results)
    
    def test_evaluation_with_config(self):
        """测试带配置的评估"""
        self.api.initialize()
        
        result = self.api.evaluate_performance(self.valid_results, self.valid_config)
        self.assertEqual(result['status'], 'success')
    
    def test_orchestrator_operations(self):
        """测试协调器操作"""
        self.orchestrator.initialize_orchestrator()
        
        # 测试性能评估协调
        result = self.orchestrator.orchestrate(
            "evaluate_performance", 
            results=self.valid_results
        )
        self.assertEqual(result['status'], 'success')
        
        # 测试健康检查协调
        result = self.orchestrator.orchestrate("health_check")
        self.assertIn('overall_health', result)
    
    # ==================== 边界条件测试 ====================
    
    def test_empty_results(self):
        """测试空结果列表"""
        self.api.initialize()
        
        result = self.api.evaluate_performance(self.empty_results)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['evaluation_results']['total_samples'], 0)
    
    def test_none_input(self):
        """测试None输入"""
        self.api.initialize()
        
        result = self.api.evaluate_performance(self.none_input)
        self.assertEqual(result['status'], 'error')
    
    def test_large_results(self):
        """测试大结果集"""
        self.api.initialize()
        
        result = self.api.evaluate_performance(self.large_results)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['evaluation_results']['total_samples'], 1000)
    
    def test_single_result(self):
        """测试单个结果"""
        self.api.initialize()
        
        single_result = [{"accuracy": 0.95}]
        result = self.api.evaluate_performance(single_result)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['evaluation_results']['total_samples'], 1)
    
    def test_special_characters_in_results(self):
        """测试结果中的特殊字符"""
        self.api.initialize()
        
        special_results = [
            {"metric": "accuracy", "value": 0.95, "note": "!@#$%^&*()"},
            {"metric": "quality", "value": 0.88, "note": "★☆♠♣♥♦"}
        ]
        result = self.api.evaluate_performance(special_results)
        self.assertEqual(result['status'], 'success')
    
    # ==================== 异常情况测试 ====================
    
    def test_invalid_results_type(self):
        """测试无效结果类型"""
        self.api.initialize()
        
        result = self.api.evaluate_performance(self.invalid_results)
        self.assertEqual(result['status'], 'error')
    
    def test_invalid_data_type(self):
        """测试无效数据类型"""
        self.api.initialize()
        
        result = self.api.evaluate_performance(self.invalid_type)
        self.assertEqual(result['status'], 'error')
    
    def test_api_not_initialized(self):
        """测试API未初始化"""
        api = EvaluationAPI()
        
        result = api.evaluate_performance(self.valid_results)
        self.assertEqual(result['status'], 'error')
    
    def test_orchestrator_invalid_operation(self):
        """测试协调器无效操作"""
        self.orchestrator.initialize_orchestrator()
        
        with self.assertRaises(ValueError):
            self.orchestrator.orchestrate("invalid_operation")
    
    def test_component_registration(self):
        """测试组件注册"""
        self.orchestrator.initialize_orchestrator()
        
        # 注册测试组件
        test_component = {"name": "test_evaluator", "value": 789}
        self.orchestrator.register_component("test_component", test_component)
        
        # 获取组件
        retrieved = self.orchestrator.get_component("test_component")
        self.assertEqual(retrieved, test_component)
        
        # 测试获取不存在的组件
        with self.assertRaises(AttributeError):
            self.orchestrator.get_component("non_existent")
    
    def test_module_status(self):
        """测试模块状态"""
        self.api.initialize()
        
        status = self.api.get_module_status()
        self.assertIn('module_name', status)
        self.assertIn('initialized', status)
        self.assertIn('version', status)
        self.assertTrue(status['initialized'])
    
    def test_health_check(self):
        """测试健康检查"""
        self.api.initialize()
        
        health = self.api.health_check()
        self.assertIn('overall_health', health)
        self.assertEqual(health['overall_health'], 'healthy')
    
    def test_shutdown_functionality(self):
        """测试关闭功能"""
        self.api.initialize()
        self.assertTrue(self.api._initialized)
        
        result = self.api.shutdown()
        self.assertTrue(result)
        self.assertFalse(self.api._initialized)
    
    def test_error_handling(self):
        """测试错误处理"""
        self.api.initialize()
        
        # 测试各种错误情况
        error_cases = [
            (None, "None输入"),
            ("not_a_list", "字符串"),
            (123, "数字类型"),
            ({"invalid": "data"}, "字典类型"),
        ]
        
        for test_input, description in error_cases:
            with self.subTest(description):
                result = self.api.evaluate_performance(test_input)
                self.assertIn('status', result)
    
    def test_evaluation_metrics(self):
        """测试评估指标"""
        self.api.initialize()
        
        # 测试不同的评估指标组合
        metric_combinations = [
            [{"accuracy": 0.95}],
            [{"quality": 0.88}],
            [{"accuracy": 0.95, "quality": 0.88}],
            [{"efficiency": 0.92, "robustness": 0.85}],
        ]
        
        for metrics in metric_combinations:
            with self.subTest(metrics=metrics):
                result = self.api.evaluate_performance(metrics)
                self.assertEqual(result['status'], 'success')


class TestEvaluationIntegration(unittest.TestCase):
    """Evaluation模块集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.api = EvaluationAPI()
        self.api.initialize()
    
    def test_full_evaluation_pipeline(self):
        """测试完整评估流水线"""
        # 1. 准备评估数据
        evaluation_data = [
            {"accuracy": 0.95, "quality": 0.88, "efficiency": 0.92},
            {"accuracy": 0.92, "quality": 0.85, "efficiency": 0.89},
            {"accuracy": 0.89, "quality": 0.82, "efficiency": 0.86}
        ]
        
        # 2. 执行评估
        result = self.api.evaluate_performance(evaluation_data)
        self.assertEqual(result['status'], 'success')
        
        # 3. 检查评估结果
        eval_results = result['evaluation_results']
        self.assertIn('accuracy', eval_results)
        self.assertIn('quality_score', eval_results)
        self.assertIn('total_samples', eval_results)
        self.assertEqual(eval_results['total_samples'], 3)
    
    def test_evaluation_configuration_options(self):
        """测试评估配置选项"""
        configs = [
            {"metrics": ["accuracy"]},
            {"metrics": ["quality"]},
            {"metrics": ["accuracy", "quality"]},
            {"threshold": 0.8},
        ]
        
        for config in configs:
            with self.subTest(config=config):
                result = self.api.evaluate_performance(self.api.valid_results, config)
                self.assertEqual(result['status'], 'success')
    
    def test_performance_under_load(self):
        """测试负载下的性能"""
        # 创建大量评估数据
        large_evaluation_data = [
            {"accuracy": 0.9 + (i % 10) * 0.01, "quality": 0.8 + (i % 10) * 0.01}
            for i in range(100)
        ]
        
        # 执行评估
        result = self.api.evaluate_performance(large_evaluation_data)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['evaluation_results']['total_samples'], 100)
    
    def test_evaluation_edge_cases(self):
        """测试评估边界情况"""
        edge_cases = [
            # 空结果
            [],
            # 单个结果
            [{"accuracy": 1.0}],
            # 极值结果
            [{"accuracy": 0.0}, {"accuracy": 1.0}],
            # 缺失指标
            [{"quality": 0.8}],
            # 混合指标
            [{"accuracy": 0.9}, {"quality": 0.8}, {"efficiency": 0.7}],
        ]
        
        for case in edge_cases:
            with self.subTest(case=case):
                result = self.api.evaluate_performance(case)
                self.assertIn('status', result)
    
    def test_evaluation_error_recovery(self):
        """测试评估错误恢复"""
        # 先执行一个无效评估（应该失败）
        invalid_result = self.api.evaluate_performance("invalid_data")
        self.assertEqual(invalid_result['status'], 'error')
        
        # 然后执行一个有效评估（应该成功）
        valid_result = self.api.evaluate_performance([{"accuracy": 0.95}])
        self.assertEqual(valid_result['status'], 'success')
        
        # 验证API仍然正常工作
        status = self.api.get_module_status()
        self.assertTrue(status['initialized'])


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2) 