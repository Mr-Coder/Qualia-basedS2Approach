"""
Modular Architecture - Integration Test Suite
============================================

综合集成测试：测试所有模块的协同工作

测试覆盖范围：
- 模块间集成测试
- 端到端流程测试
- 性能压力测试
- 错误恢复测试

Author: AI Assistant
Date: 2024-07-13
"""

import logging
import os
import sys
import time
import unittest
from typing import Any, Dict

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data import DataAPI, data_api
from evaluation import EvaluationAPI, evaluation_api
from models import ModelsAPI, models_api
from processors import ProcessorsAPI, processors_api


class TestModularArchitectureIntegration(unittest.TestCase):
    """模块化架构集成测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 初始化所有模块
        self.processors_api = ProcessorsAPI()
        self.models_api = ModelsAPI()
        self.evaluation_api = EvaluationAPI()
        self.data_api = DataAPI()
        
        # 测试数据
        self.test_text = "小明有5个苹果，给了小红3个，还剩几个？"
        self.test_model_data = {"type": "equation", "data": "x + y = 10"}
        self.test_results = [{"accuracy": 0.95, "quality": 0.88}]
        self.test_dataset_name = "test_dataset"
        
    def tearDown(self):
        """测试后清理"""
        # 关闭所有模块
        for api in [self.processors_api, self.models_api, 
                   self.evaluation_api, self.data_api]:
            if hasattr(api, '_initialized') and api._initialized:
                api.shutdown()
    
    # ==================== 模块间集成测试 ====================
    
    def test_all_modules_initialization(self):
        """测试所有模块初始化"""
        # 初始化所有模块
        processors_init = self.processors_api.initialize()
        models_init = self.models_api.initialize()
        evaluation_init = self.evaluation_api.initialize()
        data_init = self.data_api.initialize()
        
        # 验证所有模块都成功初始化
        self.assertTrue(processors_init)
        self.assertTrue(models_init)
        self.assertTrue(evaluation_init)
        self.assertTrue(data_init)
        
        # 验证所有模块都处于初始化状态
        self.assertTrue(self.processors_api._initialized)
        self.assertTrue(self.models_api._initialized)
        self.assertTrue(self.evaluation_api._initialized)
        self.assertTrue(self.data_api._initialized)
    
    def test_cross_module_communication(self):
        """测试模块间通信"""
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 1. 使用processors处理文本
        text_result = self.processors_api.process_text(self.test_text)
        self.assertEqual(text_result['status'], 'success')
        
        # 2. 使用models创建模型
        model_result = self.models_api.create_model(self.test_model_data)
        self.assertEqual(model_result['status'], 'success')
        
        # 3. 使用evaluation评估结果
        eval_result = self.evaluation_api.evaluate_performance(self.test_results)
        self.assertEqual(eval_result['status'], 'success')
        
        # 4. 使用data获取信息
        data_result = self.data_api.get_dataset_information(self.test_dataset_name)
        self.assertEqual(data_result['status'], 'success')
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 完整的端到端流程
        workflow_results = []
        
        # 步骤1: 文本处理
        text_result = self.processors_api.process_text(self.test_text)
        workflow_results.append(("text_processing", text_result['status']))
        
        # 步骤2: 模型创建
        model_result = self.models_api.create_model(self.test_model_data)
        workflow_results.append(("model_creation", model_result['status']))
        
        # 步骤3: 性能评估
        eval_result = self.evaluation_api.evaluate_performance(self.test_results)
        workflow_results.append(("evaluation", eval_result['status']))
        
        # 步骤4: 数据查询
        data_result = self.data_api.get_dataset_information()
        workflow_results.append(("data_query", data_result['status']))
        
        # 验证所有步骤都成功
        for step_name, status in workflow_results:
            with self.subTest(step=step_name):
                self.assertEqual(status, 'success')
    
    # ==================== 性能压力测试 ====================
    
    def test_performance_under_load(self):
        """测试负载下的性能"""
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 创建大量测试数据
        test_texts = [f"测试文本{i}" for i in range(100)]
        test_models = [{"type": f"model_{i}", "data": f"data_{i}"} for i in range(100)]
        test_results = [{"accuracy": 0.9 + (i % 10) * 0.01} for i in range(100)]
        test_datasets = [f"dataset_{i}" for i in range(100)]
        
        # 记录开始时间
        start_time = time.time()
        
        # 并行执行所有模块的操作
        processors_results = []
        models_results = []
        evaluation_results = []
        data_results = []
        
        # 批量处理
        for i in range(100):
            # Processors操作
            result = self.processors_api.process_text(test_texts[i])
            processors_results.append(result['status'])
            
            # Models操作
            result = self.models_api.create_model(test_models[i])
            models_results.append(result['status'])
            
            # Evaluation操作
            result = self.evaluation_api.evaluate_performance([test_results[i]])
            evaluation_results.append(result['status'])
            
            # Data操作
            result = self.data_api.get_dataset_information(test_datasets[i])
            data_results.append(result['status'])
        
        # 记录结束时间
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证性能
        self.assertLess(execution_time, 30.0)  # 应该在30秒内完成
        
        # 验证成功率
        processors_success_rate = processors_results.count('success') / len(processors_results)
        models_success_rate = models_results.count('success') / len(models_results)
        evaluation_success_rate = evaluation_results.count('success') / len(evaluation_results)
        data_success_rate = data_results.count('success') / len(data_results)
        
        self.assertGreater(processors_success_rate, 0.8)
        self.assertGreater(models_success_rate, 0.8)
        self.assertGreater(evaluation_success_rate, 0.8)
        self.assertGreater(data_success_rate, 0.8)
    
    # ==================== 错误恢复测试 ====================
    
    def test_error_recovery_across_modules(self):
        """测试跨模块错误恢复"""
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 1. 先执行一些无效操作（应该失败）
        invalid_operations = [
            (self.processors_api.process_text, None),
            (self.models_api.create_model, None),
            (self.evaluation_api.evaluate_performance, "invalid"),
            (self.data_api.get_dataset_information, 123),
        ]
        
        for operation, invalid_input in invalid_operations:
            result = operation(invalid_input)
            self.assertEqual(result['status'], 'error')
        
        # 2. 然后执行有效操作（应该成功）
        valid_operations = [
            (self.processors_api.process_text, self.test_text),
            (self.models_api.create_model, self.test_model_data),
            (self.evaluation_api.evaluate_performance, self.test_results),
            (self.data_api.get_dataset_information, self.test_dataset_name),
        ]
        
        for operation, valid_input in valid_operations:
            result = operation(valid_input)
            self.assertEqual(result['status'], 'success')
        
        # 3. 验证所有模块仍然正常工作
        for api in [self.processors_api, self.models_api, 
                   self.evaluation_api, self.data_api]:
            status = api.get_module_status()
            self.assertTrue(status['initialized'])
    
    def test_module_isolation(self):
        """测试模块隔离性"""
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 测试一个模块的错误不会影响其他模块
        # 1. 在processors中制造错误
        processors_result = self.processors_api.process_text(None)
        self.assertEqual(processors_result['status'], 'error')
        
        # 2. 验证其他模块仍然正常工作
        models_result = self.models_api.create_model(self.test_model_data)
        self.assertEqual(models_result['status'], 'success')
        
        evaluation_result = self.evaluation_api.evaluate_performance(self.test_results)
        self.assertEqual(evaluation_result['status'], 'success')
        
        data_result = self.data_api.get_dataset_information(self.test_dataset_name)
        self.assertEqual(data_result['status'], 'success')
    
    # ==================== 配置和状态测试 ====================
    
    def test_module_configuration_consistency(self):
        """测试模块配置一致性"""
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 检查所有模块的状态
        module_statuses = [
            self.processors_api.get_module_status(),
            self.models_api.get_module_status(),
            self.evaluation_api.get_module_status(),
            self.data_api.get_module_status(),
        ]
        
        # 验证所有模块都有正确的状态结构
        for status in module_statuses:
            self.assertIn('module_name', status)
            self.assertIn('initialized', status)
            self.assertIn('version', status)
            self.assertTrue(status['initialized'])
    
    def test_health_check_across_modules(self):
        """测试跨模块健康检查"""
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 检查所有模块的健康状态
        health_checks = [
            self.processors_api.health_check(),
            self.models_api.health_check(),
            self.evaluation_api.health_check(),
            self.data_api.health_check(),
        ]
        
        # 验证所有模块都是健康的
        for health in health_checks:
            self.assertIn('overall_health', health)
            self.assertEqual(health['overall_health'], 'healthy')
    
    # ==================== 并发测试 ====================
    
    def test_concurrent_module_operations(self):
        """测试并发模块操作"""
        import threading

        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 定义并发操作函数
        def processors_operation():
            for i in range(10):
                result = self.processors_api.process_text(f"测试文本{i}")
                self.assertEqual(result['status'], 'success')
        
        def models_operation():
            for i in range(10):
                result = self.models_api.create_model({"type": f"model_{i}", "data": f"data_{i}"})
                self.assertEqual(result['status'], 'success')
        
        def evaluation_operation():
            for i in range(10):
                result = self.evaluation_api.evaluate_performance([{"accuracy": 0.9 + i * 0.01}])
                self.assertEqual(result['status'], 'success')
        
        def data_operation():
            for i in range(10):
                result = self.data_api.get_dataset_information(f"dataset_{i}")
                self.assertEqual(result['status'], 'success')
        
        # 创建线程
        threads = [
            threading.Thread(target=processors_operation),
            threading.Thread(target=models_operation),
            threading.Thread(target=evaluation_operation),
            threading.Thread(target=data_operation),
        ]
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有模块仍然正常工作
        for api in [self.processors_api, self.models_api, 
                   self.evaluation_api, self.data_api]:
            status = api.get_module_status()
            self.assertTrue(status['initialized'])
    
    # ==================== 资源管理测试 ====================
    
    def test_resource_cleanup(self):
        """测试资源清理"""
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
        
        # 执行一些操作
        self.processors_api.process_text(self.test_text)
        self.models_api.create_model(self.test_model_data)
        self.evaluation_api.evaluate_performance(self.test_results)
        self.data_api.get_dataset_information(self.test_dataset_name)
        
        # 关闭所有模块
        processors_shutdown = self.processors_api.shutdown()
        models_shutdown = self.models_api.shutdown()
        evaluation_shutdown = self.evaluation_api.shutdown()
        data_shutdown = self.data_api.shutdown()
        
        # 验证所有模块都成功关闭
        self.assertTrue(processors_shutdown)
        self.assertTrue(models_shutdown)
        self.assertTrue(evaluation_shutdown)
        self.assertTrue(data_shutdown)
        
        # 验证所有模块都处于未初始化状态
        self.assertFalse(self.processors_api._initialized)
        self.assertFalse(self.models_api._initialized)
        self.assertFalse(self.evaluation_api._initialized)
        self.assertFalse(self.data_api._initialized)


class TestModularArchitectureStress(unittest.TestCase):
    """模块化架构压力测试"""
    
    def setUp(self):
        """测试前设置"""
        self.processors_api = ProcessorsAPI()
        self.models_api = ModelsAPI()
        self.evaluation_api = EvaluationAPI()
        self.data_api = DataAPI()
        
        # 初始化所有模块
        self.processors_api.initialize()
        self.models_api.initialize()
        self.evaluation_api.initialize()
        self.data_api.initialize()
    
    def tearDown(self):
        """测试后清理"""
        for api in [self.processors_api, self.models_api, 
                   self.evaluation_api, self.data_api]:
            if hasattr(api, '_initialized') and api._initialized:
                api.shutdown()
    
    def test_stress_test_rapid_operations(self):
        """压力测试：快速操作"""
        # 快速连续执行操作
        for i in range(50):
            # Processors操作
            result = self.processors_api.process_text(f"快速测试文本{i}")
            self.assertEqual(result['status'], 'success')
            
            # Models操作
            result = self.models_api.create_model({"type": f"stress_model_{i}", "data": f"stress_data_{i}"})
            self.assertEqual(result['status'], 'success')
            
            # Evaluation操作
            result = self.evaluation_api.evaluate_performance([{"accuracy": 0.9 + (i % 10) * 0.01}])
            self.assertEqual(result['status'], 'success')
            
            # Data操作
            result = self.data_api.get_dataset_information(f"stress_dataset_{i}")
            self.assertEqual(result['status'], 'success')
    
    def test_stress_test_memory_usage(self):
        """压力测试：内存使用"""
        # 创建大量数据并执行操作
        large_texts = ["大文本" * 100 for _ in range(20)]
        large_models = [{"type": f"large_model_{i}", "data": "x" * 1000} for i in range(20)]
        large_results = [[{"accuracy": 0.9 + (i % 10) * 0.01} for _ in range(10)] for i in range(20)]
        
        for i in range(20):
            # 处理大文本
            result = self.processors_api.process_text(large_texts[i])
            self.assertEqual(result['status'], 'success')
            
            # 创建大模型
            result = self.models_api.create_model(large_models[i])
            self.assertEqual(result['status'], 'success')
            
            # 评估大结果集
            result = self.evaluation_api.evaluate_performance(large_results[i])
            self.assertEqual(result['status'], 'success')


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2) 