"""
Models Module - Comprehensive Test Suite
=======================================

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

from models import (ModelsAPI, ModelsOrchestrator, models_api,
                    models_orchestrator)


class TestModelsModule(unittest.TestCase):
    """Models模块测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.api = ModelsAPI()
        self.orchestrator = ModelsOrchestrator()
        
        # 测试数据
        self.valid_model_data = {
            "type": "equation",
            "data": "x + y = 10"
        }
        self.valid_equation = "2x + 3y = 15"
        self.valid_relation = {
            "source": "小明",
            "target": "小红", 
            "type": "比...高"
        }
        
        # 边界测试数据
        self.empty_model = {}
        self.none_input = None
        self.large_model = {
            "type": "complex",
            "data": "x" * 1000
        }
        
        # 异常测试数据
        self.invalid_model = {"wrong_key": "value"}
        self.invalid_type = 12345
        
    def tearDown(self):
        """测试后清理"""
        if hasattr(self.api, '_initialized') and self.api._initialized:
            self.api.shutdown()
    
    # ==================== 正常情况测试 ====================
    
    def test_module_import(self):
        """测试模块导入"""
        self.assertIsNotNone(models_api)
        self.assertIsNotNone(models_orchestrator)
        self.assertIsInstance(models_api, ModelsAPI)
        self.assertIsInstance(models_orchestrator, ModelsOrchestrator)
    
    def test_api_initialization(self):
        """测试API初始化"""
        result = self.api.initialize()
        self.assertTrue(result)
        self.assertTrue(self.api._initialized)
    
    def test_model_creation_normal(self):
        """测试正常模型创建"""
        self.api.initialize()
        
        # 测试方程模型
        result = self.api.create_model(self.valid_model_data, "equation")
        self.assertEqual(result['status'], 'success')
        self.assertIn('model', result)
        
        # 测试关系模型
        result = self.api.create_model(self.valid_relation, "relation")
        self.assertEqual(result['status'], 'success')
        
        # 测试通用模型
        result = self.api.create_model({"type": "general", "data": "test"})
        self.assertEqual(result['status'], 'success')
    
    def test_equation_processing_normal(self):
        """测试正常方程处理"""
        self.api.initialize()
        
        result = self.api.process_equations(self.valid_equation)
        self.assertEqual(result['status'], 'success')
        self.assertIn('model', result)
    
    def test_relation_processing_normal(self):
        """测试正常关系处理"""
        self.api.initialize()
        
        result = self.api.process_relations(self.valid_relation)
        self.assertEqual(result['status'], 'success')
        self.assertIn('model', result)
    
    def test_orchestrator_operations(self):
        """测试协调器操作"""
        self.orchestrator.initialize_orchestrator()
        
        # 测试模型创建协调
        result = self.orchestrator.orchestrate(
            "create_model", 
            model_data=self.valid_model_data,
            model_type="equation"
        )
        self.assertEqual(result['status'], 'success')
        
        # 测试方程处理协调
        result = self.orchestrator.orchestrate(
            "process_equations",
            equations=self.valid_equation
        )
        self.assertEqual(result['status'], 'success')
        
        # 测试健康检查协调
        result = self.orchestrator.orchestrate("health_check")
        self.assertIn('overall_health', result)
    
    # ==================== 边界条件测试 ====================
    
    def test_empty_model_data(self):
        """测试空模型数据"""
        self.api.initialize()
        
        result = self.api.create_model(self.empty_model)
        self.assertEqual(result['status'], 'error')
    
    def test_none_input(self):
        """测试None输入"""
        self.api.initialize()
        
        result = self.api.create_model(self.none_input)
        self.assertEqual(result['status'], 'error')
    
    def test_large_model_data(self):
        """测试大模型数据"""
        self.api.initialize()
        
        result = self.api.create_model(self.large_model)
        self.assertEqual(result['status'], 'success')
    
    def test_minimal_valid_model(self):
        """测试最小有效模型"""
        self.api.initialize()
        
        minimal_model = {"type": "minimal", "data": "test"}
        result = self.api.create_model(minimal_model)
        self.assertEqual(result['status'], 'success')
    
    def test_special_characters_in_model(self):
        """测试模型中的特殊字符"""
        self.api.initialize()
        
        special_model = {
            "type": "special",
            "data": "x + y = @#$%^&*()_+-=[]{}|;':\",./<>?"
        }
        result = self.api.create_model(special_model)
        self.assertEqual(result['status'], 'success')
    
    def test_unicode_characters_in_model(self):
        """测试模型中的Unicode字符"""
        self.api.initialize()
        
        unicode_model = {
            "type": "unicode",
            "data": "中文变量 + English变量 = 结果★☆♠♣♥♦"
        }
        result = self.api.create_model(unicode_model)
        self.assertEqual(result['status'], 'success')
    
    # ==================== 异常情况测试 ====================
    
    def test_invalid_model_structure(self):
        """测试无效模型结构"""
        self.api.initialize()
        
        result = self.api.create_model(self.invalid_model)
        self.assertEqual(result['status'], 'error')
    
    def test_invalid_data_type(self):
        """测试无效数据类型"""
        self.api.initialize()
        
        result = self.api.create_model(self.invalid_type)
        self.assertEqual(result['status'], 'error')
    
    def test_api_not_initialized(self):
        """测试API未初始化"""
        api = ModelsAPI()
        
        result = api.create_model(self.valid_model_data)
        self.assertEqual(result['status'], 'error')
        self.assertIn('未初始化', result['error_message'])
    
    def test_orchestrator_invalid_operation(self):
        """测试协调器无效操作"""
        self.orchestrator.initialize_orchestrator()
        
        with self.assertRaises(ValueError):
            self.orchestrator.orchestrate("invalid_operation")
    
    def test_validation_failure(self):
        """测试验证失败"""
        self.api.initialize()
        
        # 测试缺少必要字段的模型
        invalid_model = {"type": "test"}  # 缺少data字段
        result = self.api.create_model(invalid_model)
        self.assertEqual(result['status'], 'error')
    
    def test_component_registration(self):
        """测试组件注册"""
        self.orchestrator.initialize_orchestrator()
        
        # 注册测试组件
        test_component = {"name": "test_model", "value": 456}
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
            ({}, "空字典"),
            (123, "数字类型"),
            ({"invalid": "data"}, "无效结构"),
        ]
        
        for test_input, description in error_cases:
            with self.subTest(description):
                result = self.api.create_model(test_input)
                self.assertIn('status', result)
                # 应该返回错误或验证失败状态
    
    def test_model_types(self):
        """测试不同模型类型"""
        self.api.initialize()
        
        model_types = ["equation", "relation", "structure", "general"]
        
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                test_data = {"type": model_type, "data": f"test_{model_type}"}
                result = self.api.create_model(test_data, model_type)
                self.assertEqual(result['status'], 'success')


class TestModelsIntegration(unittest.TestCase):
    """Models模块集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.api = ModelsAPI()
        self.api.initialize()
    
    def test_full_model_pipeline(self):
        """测试完整模型流水线"""
        # 1. 创建方程模型
        equation_result = self.api.create_model(
            {"type": "equation", "data": "x + y = 10"}, 
            "equation"
        )
        self.assertEqual(equation_result['status'], 'success')
        
        # 2. 处理方程
        process_result = self.api.process_equations("2x + 3y = 15")
        self.assertEqual(process_result['status'], 'success')
        
        # 3. 处理关系
        relation_result = self.api.process_relations({
            "source": "小明", "target": "小红", "type": "比...高"
        })
        self.assertEqual(relation_result['status'], 'success')
    
    def test_model_configuration_options(self):
        """测试模型配置选项"""
        configs = [
            {"model_type": "equation"},
            {"model_type": "relation"},
            {"model_type": "structure"},
            {"model_type": "general"},
        ]
        
        for config in configs:
            with self.subTest(config=config):
                result = self.api.create_model(
                    {"type": "test", "data": "test"}, 
                    config=config
                )
                self.assertEqual(result['status'], 'success')
    
    def test_performance_under_load(self):
        """测试负载下的性能"""
        # 创建大量模型数据
        model_data_list = [
            {"type": f"model_{i}", "data": f"data_{i}"} 
            for i in range(50)
        ]
        
        # 批量创建模型
        success_count = 0
        for model_data in model_data_list:
            result = self.api.create_model(model_data)
            if result['status'] == 'success':
                success_count += 1
        
        # 检查成功率
        success_rate = success_count / len(model_data_list)
        self.assertGreater(success_rate, 0.8)  # 至少80%成功率
    
    def test_model_validation_edge_cases(self):
        """测试模型验证边界情况"""
        edge_cases = [
            # 最小有效模型
            {"type": "minimal", "data": "a"},
            # 最大长度模型
            {"type": "max", "data": "x" * 1000},
            # 特殊字符模型
            {"type": "special", "data": "!@#$%^&*()"},
            # Unicode模型
            {"type": "unicode", "data": "中文★☆♠♣♥♦"},
        ]
        
        for case in edge_cases:
            with self.subTest(case=case):
                result = self.api.create_model(case)
                self.assertIn('status', result)
    
    def test_model_error_recovery(self):
        """测试模型错误恢复"""
        # 先创建一个无效模型（应该失败）
        invalid_result = self.api.create_model({"invalid": "data"})
        self.assertEqual(invalid_result['status'], 'error')
        
        # 然后创建一个有效模型（应该成功）
        valid_result = self.api.create_model({
            "type": "valid", "data": "valid_data"
        })
        self.assertEqual(valid_result['status'], 'success')
        
        # 验证API仍然正常工作
        status = self.api.get_module_status()
        self.assertTrue(status['initialized'])


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2) 