"""
Data Module - Comprehensive Test Suite
=====================================

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

from data import DataAPI, DataOrchestrator, data_api, data_orchestrator


class TestDataModule(unittest.TestCase):
    """Data模块测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.api = DataAPI()
        self.orchestrator = DataOrchestrator()
        
        # 测试数据
        self.valid_dataset_name = "test_dataset"
        self.valid_method_name = "test_method"
        
        # 边界测试数据
        self.empty_string = ""
        self.none_input = None
        self.long_name = "a" * 1000
        
        # 异常测试数据
        self.invalid_type = 12345
        
    def tearDown(self):
        """测试后清理"""
        if hasattr(self.api, '_initialized') and self.api._initialized:
            self.api.shutdown()
    
    # ==================== 正常情况测试 ====================
    
    def test_module_import(self):
        """测试模块导入"""
        self.assertIsNotNone(data_api)
        self.assertIsNotNone(data_orchestrator)
        self.assertIsInstance(data_api, DataAPI)
        self.assertIsInstance(data_orchestrator, DataOrchestrator)
    
    def test_api_initialization(self):
        """测试API初始化"""
        result = self.api.initialize()
        self.assertTrue(result)
        self.assertTrue(self.api._initialized)
    
    def test_get_dataset_information_normal(self):
        """测试正常获取数据集信息"""
        self.api.initialize()
        
        # 测试获取所有数据集信息
        result = self.api.get_dataset_information()
        self.assertEqual(result['status'], 'success')
        self.assertIn('all_datasets', result)
        
        # 测试获取特定数据集信息
        result = self.api.get_dataset_information(self.valid_dataset_name)
        self.assertEqual(result['status'], 'success')
    
    def test_get_performance_data_normal(self):
        """测试正常获取性能数据"""
        self.api.initialize()
        
        # 测试获取特定方法性能数据
        result = self.api.get_performance_data(self.valid_method_name)
        self.assertEqual(result['status'], 'success')
        
        # 测试获取所有方法性能数据
        result = self.api.get_performance_data()
        self.assertEqual(result['status'], 'success')
        self.assertIn('message', result)
    
    def test_orchestrator_operations(self):
        """测试协调器操作"""
        self.orchestrator.initialize_orchestrator()
        
        # 测试数据集信息获取协调
        result = self.orchestrator.orchestrate(
            "get_dataset_info", 
            dataset_name=self.valid_dataset_name
        )
        self.assertEqual(result['status'], 'success')
        
        # 测试性能数据获取协调
        result = self.orchestrator.orchestrate(
            "get_performance_data",
            method_name=self.valid_method_name
        )
        self.assertEqual(result['status'], 'success')
        
        # 测试健康检查协调
        result = self.orchestrator.orchestrate("health_check")
        self.assertIn('overall_health', result)
    
    # ==================== 边界条件测试 ====================
    
    def test_empty_dataset_name(self):
        """测试空数据集名称"""
        self.api.initialize()
        
        result = self.api.get_dataset_information(self.empty_string)
        self.assertEqual(result['status'], 'success')
    
    def test_none_input(self):
        """测试None输入"""
        self.api.initialize()
        
        result = self.api.get_dataset_information(self.none_input)
        self.assertEqual(result['status'], 'success')
    
    def test_long_dataset_name(self):
        """测试长数据集名称"""
        self.api.initialize()
        
        result = self.api.get_dataset_information(self.long_name)
        self.assertEqual(result['status'], 'success')
    
    def test_special_characters_in_names(self):
        """测试名称中的特殊字符"""
        self.api.initialize()
        
        special_name = "test_dataset_@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = self.api.get_dataset_information(special_name)
        self.assertEqual(result['status'], 'success')
    
    def test_unicode_characters_in_names(self):
        """测试名称中的Unicode字符"""
        self.api.initialize()
        
        unicode_name = "测试数据集_中文★☆♠♣♥♦"
        result = self.api.get_dataset_information(unicode_name)
        self.assertEqual(result['status'], 'success')
    
    # ==================== 异常情况测试 ====================
    
    def test_invalid_data_type(self):
        """测试无效数据类型"""
        self.api.initialize()
        
        result = self.api.get_dataset_information(self.invalid_type)
        self.assertEqual(result['status'], 'error')
    
    def test_api_not_initialized(self):
        """测试API未初始化"""
        api = DataAPI()
        
        result = api.get_dataset_information(self.valid_dataset_name)
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
        test_component = {"name": "test_data", "value": 101}
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
            (123, "数字类型"),
            ({"invalid": "data"}, "字典类型"),
            ([], "列表类型"),
        ]
        
        for test_input, description in error_cases:
            with self.subTest(description):
                result = self.api.get_dataset_information(test_input)
                self.assertIn('status', result)
    
    def test_data_operations(self):
        """测试数据操作"""
        self.api.initialize()
        
        # 测试不同的数据集名称
        dataset_names = [
            "dataset_1",
            "dataset_2", 
            "dataset_3",
            "special_dataset",
        ]
        
        for name in dataset_names:
            with self.subTest(name=name):
                result = self.api.get_dataset_information(name)
                self.assertEqual(result['status'], 'success')
        
        # 测试不同的方法名称
        method_names = [
            "method_1",
            "method_2",
            "method_3",
            "special_method",
        ]
        
        for name in method_names:
            with self.subTest(name=name):
                result = self.api.get_performance_data(name)
                self.assertEqual(result['status'], 'success')


class TestDataIntegration(unittest.TestCase):
    """Data模块集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.api = DataAPI()
        self.api.initialize()
    
    def test_full_data_pipeline(self):
        """测试完整数据流水线"""
        # 1. 获取数据集信息
        dataset_result = self.api.get_dataset_information("test_dataset")
        self.assertEqual(dataset_result['status'], 'success')
        
        # 2. 获取性能数据
        performance_result = self.api.get_performance_data("test_method")
        self.assertEqual(performance_result['status'], 'success')
        
        # 3. 获取所有数据集信息
        all_datasets_result = self.api.get_dataset_information()
        self.assertEqual(all_datasets_result['status'], 'success')
        self.assertIn('all_datasets', all_datasets_result)
    
    def test_data_configuration_options(self):
        """测试数据配置选项"""
        # 测试不同的数据集查询方式
        query_options = [
            None,  # 获取所有数据集
            "dataset_1",  # 获取特定数据集
            "dataset_2",
            "dataset_3",
        ]
        
        for option in query_options:
            with self.subTest(option=option):
                result = self.api.get_dataset_information(option)
                self.assertEqual(result['status'], 'success')
        
        # 测试不同的性能数据查询方式
        performance_options = [
            None,  # 获取所有方法
            "method_1",  # 获取特定方法
            "method_2",
            "method_3",
        ]
        
        for option in performance_options:
            with self.subTest(option=option):
                result = self.api.get_performance_data(option)
                self.assertEqual(result['status'], 'success')
    
    def test_performance_under_load(self):
        """测试负载下的性能"""
        # 创建大量查询请求
        dataset_names = [f"dataset_{i}" for i in range(50)]
        method_names = [f"method_{i}" for i in range(50)]
        
        # 批量查询数据集信息
        success_count = 0
        for name in dataset_names:
            result = self.api.get_dataset_information(name)
            if result['status'] == 'success':
                success_count += 1
        
        # 检查成功率
        success_rate = success_count / len(dataset_names)
        self.assertGreater(success_rate, 0.8)  # 至少80%成功率
        
        # 批量查询性能数据
        success_count = 0
        for name in method_names:
            result = self.api.get_performance_data(name)
            if result['status'] == 'success':
                success_count += 1
        
        # 检查成功率
        success_rate = success_count / len(method_names)
        self.assertGreater(success_rate, 0.8)  # 至少80%成功率
    
    def test_data_edge_cases(self):
        """测试数据边界情况"""
        edge_cases = [
            # 空字符串
            "",
            # 最小长度名称
            "a",
            # 最大长度名称
            "a" * 1000,
            # 特殊字符名称
            "!@#$%^&*()",
            # Unicode名称
            "中文★☆♠♣♥♦",
        ]
        
        for case in edge_cases:
            with self.subTest(case=case):
                # 测试数据集信息查询
                result = self.api.get_dataset_information(case)
                self.assertIn('status', result)
                
                # 测试性能数据查询
                result = self.api.get_performance_data(case)
                self.assertIn('status', result)
    
    def test_data_error_recovery(self):
        """测试数据错误恢复"""
        # 先执行一个无效查询（应该失败）
        invalid_result = self.api.get_dataset_information(123)
        self.assertEqual(invalid_result['status'], 'error')
        
        # 然后执行一个有效查询（应该成功）
        valid_result = self.api.get_dataset_information("test_dataset")
        self.assertEqual(valid_result['status'], 'success')
        
        # 验证API仍然正常工作
        status = self.api.get_module_status()
        self.assertTrue(status['initialized'])
    
    def test_data_consistency(self):
        """测试数据一致性"""
        # 多次查询相同的数据，结果应该一致
        dataset_name = "test_dataset"
        
        result1 = self.api.get_dataset_information(dataset_name)
        result2 = self.api.get_dataset_information(dataset_name)
        
        self.assertEqual(result1['status'], result2['status'])
        
        # 性能数据查询也应该一致
        method_name = "test_method"
        
        result3 = self.api.get_performance_data(method_name)
        result4 = self.api.get_performance_data(method_name)
        
        self.assertEqual(result3['status'], result4['status'])


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2) 