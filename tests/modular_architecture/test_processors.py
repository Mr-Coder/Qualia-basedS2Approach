"""
Processors Module - Comprehensive Test Suite
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

from processors import (ProcessorsAPI, ProcessorsOrchestrator, batch_process,
                        classify_complexity, extract_relations,
                        get_module_info, initialize_module, process_dataset,
                        process_nlp, process_text, processors_api,
                        processors_orchestrator)


class TestProcessorsModule(unittest.TestCase):
    """Processors模块测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.api = ProcessorsAPI()
        self.orchestrator = ProcessorsOrchestrator()
        
        # 测试数据
        self.valid_text = "小明有5个苹果，给了小红3个，还剩几个？"
        self.valid_dict = {"problem": "2 + 3 = ?", "context": "数学题"}
        self.valid_list = ["问题1", "问题2", "问题3"]
        
        # 边界测试数据
        self.empty_string = ""
        self.whitespace_string = "   "
        self.none_input = None
        self.large_text = "测试" * 1000
        
        # 异常测试数据
        self.invalid_dict = {"wrong_key": "value"}
        self.invalid_type = 12345
        
    def tearDown(self):
        """测试后清理"""
        if hasattr(self.api, '_initialized') and self.api._initialized:
            self.api.shutdown()
    
    # ==================== 正常情况测试 ====================
    
    def test_module_import(self):
        """测试模块导入"""
        self.assertIsNotNone(processors_api)
        self.assertIsNotNone(processors_orchestrator)
        self.assertIsInstance(processors_api, ProcessorsAPI)
        self.assertIsInstance(processors_orchestrator, ProcessorsOrchestrator)
    
    def test_module_info(self):
        """测试模块信息获取"""
        info = get_module_info()
        self.assertIn('version', info)
        self.assertIn('architecture', info)
        self.assertEqual(info['architecture'], 'modular')
        self.assertEqual(info['version'], '2.0.0')
    
    def test_api_initialization(self):
        """测试API初始化"""
        result = self.api.initialize()
        self.assertTrue(result)
        self.assertTrue(self.api._initialized)
    
    def test_text_processing_normal(self):
        """测试正常文本处理"""
        self.api.initialize()
        
        # 测试字符串输入
        result = self.api.process_text(self.valid_text)
        self.assertEqual(result['status'], 'success')
        self.assertIn('processed_data', result)
        
        # 测试字典输入
        result = self.api.process_text(self.valid_dict)
        self.assertEqual(result['status'], 'success')
        
        # 测试列表输入
        result = self.api.process_text(self.valid_list)
        self.assertEqual(result['status'], 'success')
    
    def test_dataset_processing_normal(self):
        """测试正常数据集处理"""
        self.api.initialize()
        
        dataset = [
            {"problem": "1 + 1 = ?"},
            {"problem": "2 + 2 = ?"},
            {"problem": "3 + 3 = ?"}
        ]
        
        result = self.api.process_dataset(dataset)
        self.assertEqual(result['status'], 'success')
        self.assertIn('processed_samples', result)
        self.assertEqual(len(result['processed_samples']), 3)
    
    def test_relation_extraction_normal(self):
        """测试正常关系提取"""
        self.api.initialize()
        
        text = "小明比小红高，小红比小李高"
        result = self.api.extract_relations(text)
        self.assertEqual(result['status'], 'success')
    
    def test_complexity_classification_normal(self):
        """测试正常复杂度分类"""
        self.api.initialize()
        
        text = "简单的加法题：1 + 1 = ?"
        result = self.api.classify_complexity(text)
        self.assertEqual(result['status'], 'success')
    
    def test_nlp_processing_normal(self):
        """测试正常NLP处理"""
        self.api.initialize()
        
        text = "这是一个自然语言处理测试"
        result = self.api.process_nlp(text)
        self.assertEqual(result['status'], 'success')
    
    def test_batch_processing_normal(self):
        """测试正常批量处理"""
        self.api.initialize()
        
        inputs = ["文本1", "文本2", "文本3"]
        result = self.api.batch_process(inputs)
        self.assertEqual(result['status'], 'success')
        self.assertIn('statistics', result)
    
    def test_orchestrator_operations(self):
        """测试协调器操作"""
        self.orchestrator.initialize_orchestrator()
        
        # 测试文本处理协调
        result = self.orchestrator.orchestrate(
            "process_text", 
            text=self.valid_text
        )
        self.assertEqual(result['status'], 'success')
        
        # 测试健康检查协调
        result = self.orchestrator.orchestrate("health_check")
        self.assertIn('overall_health', result)
    
    # ==================== 边界条件测试 ====================
    
    def test_empty_string_input(self):
        """测试空字符串输入"""
        self.api.initialize()
        
        result = self.api.process_text(self.empty_string)
        self.assertEqual(result['status'], 'error')
        self.assertIn('error_message', result)
    
    def test_whitespace_only_input(self):
        """测试仅空白字符输入"""
        self.api.initialize()
        
        result = self.api.process_text(self.whitespace_string)
        self.assertEqual(result['status'], 'error')
    
    def test_none_input(self):
        """测试None输入"""
        self.api.initialize()
        
        result = self.api.process_text(self.none_input)
        self.assertEqual(result['status'], 'error')
    
    def test_large_text_input(self):
        """测试大文本输入"""
        self.api.initialize()
        
        result = self.api.process_text(self.large_text)
        self.assertEqual(result['status'], 'success')
    
    def test_empty_dataset(self):
        """测试空数据集"""
        self.api.initialize()
        
        result = self.api.process_dataset([])
        self.assertEqual(result['status'], 'error')
    
    def test_single_item_dataset(self):
        """测试单项目数据集"""
        self.api.initialize()
        
        result = self.api.process_dataset([{"problem": "test"}])
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['processed_samples']), 1)
    
    def test_large_dataset(self):
        """测试大数据集"""
        self.api.initialize()
        
        large_dataset = [{"problem": f"问题{i}"} for i in range(100)]
        result = self.api.process_dataset(large_dataset)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['processed_samples']), 100)
    
    def test_special_characters(self):
        """测试特殊字符"""
        self.api.initialize()
        
        special_text = "测试文本：@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = self.api.process_text(special_text)
        self.assertEqual(result['status'], 'success')
    
    def test_unicode_characters(self):
        """测试Unicode字符"""
        self.api.initialize()
        
        unicode_text = "测试文本：中文、English、123、特殊符号：★☆♠♣♥♦"
        result = self.api.process_text(unicode_text)
        self.assertEqual(result['status'], 'success')
    
    # ==================== 异常情况测试 ====================
    
    def test_invalid_dict_structure(self):
        """测试无效字典结构"""
        self.api.initialize()
        
        result = self.api.process_text(self.invalid_dict)
        self.assertEqual(result['status'], 'error')
    
    def test_invalid_data_type(self):
        """测试无效数据类型"""
        self.api.initialize()
        
        result = self.api.process_text(self.invalid_type)
        self.assertEqual(result['status'], 'error')
    
    def test_api_not_initialized(self):
        """测试API未初始化"""
        api = ProcessorsAPI()
        
        result = api.process_text(self.valid_text)
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
        
        # 测试无效配置
        invalid_config = {"invalid_key": "invalid_value"}
        result = self.api.process_text(self.valid_text, invalid_config)
        self.assertEqual(result['status'], 'success')  # 应该降级处理
    
    def test_component_registration(self):
        """测试组件注册"""
        self.orchestrator.initialize_orchestrator()
        
        # 注册测试组件
        test_component = {"name": "test", "value": 123}
        self.orchestrator.register_component("test_component", test_component)
        
        # 获取组件
        retrieved = self.orchestrator.get_component("test_component")
        self.assertEqual(retrieved, test_component)
        
        # 测试获取不存在的组件
        with self.assertRaises(AttributeError):
            self.orchestrator.get_component("non_existent")
    
    def test_operation_history(self):
        """测试操作历史"""
        self.orchestrator.initialize_orchestrator()
        
        # 执行一些操作
        self.orchestrator.orchestrate("process_text", text="test")
        self.orchestrator.orchestrate("health_check")
        
        # 检查历史记录
        history = self.orchestrator.get_operation_history()
        self.assertGreater(len(history), 0)
        
        # 清空历史
        self.orchestrator.clear_operation_history()
        history = self.orchestrator.get_operation_history()
        self.assertEqual(len(history), 0)
    
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
    
    def test_convenience_functions(self):
        """测试便捷函数"""
        # 测试便捷函数
        result = process_text(self.valid_text)
        self.assertEqual(result['status'], 'success')
        
        result = process_dataset([{"problem": "test"}])
        self.assertEqual(result['status'], 'success')
        
        result = extract_relations(self.valid_text)
        self.assertEqual(result['status'], 'success')
        
        result = classify_complexity(self.valid_text)
        self.assertEqual(result['status'], 'success')
        
        result = process_nlp(self.valid_text)
        self.assertEqual(result['status'], 'success')
        
        result = batch_process([self.valid_text])
        self.assertEqual(result['status'], 'success')
    
    def test_error_handling(self):
        """测试错误处理"""
        self.api.initialize()
        
        # 测试各种错误情况
        error_cases = [
            (None, "None输入"),
            ("", "空字符串"),
            (123, "数字类型"),
            ({"invalid": "data"}, "无效字典"),
        ]
        
        for test_input, description in error_cases:
            with self.subTest(description):
                result = self.api.process_text(test_input)
                self.assertIn('status', result)
                # 应该返回错误或验证失败状态


class TestProcessorsIntegration(unittest.TestCase):
    """Processors模块集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.api = ProcessorsAPI()
        self.api.initialize()
    
    def test_full_processing_pipeline(self):
        """测试完整处理流水线"""
        # 1. 文本处理
        text_result = self.api.process_text("测试文本")
        self.assertEqual(text_result['status'], 'success')
        
        # 2. 关系提取
        relation_result = self.api.extract_relations("小明比小红高")
        self.assertEqual(relation_result['status'], 'success')
        
        # 3. 复杂度分类
        complexity_result = self.api.classify_complexity("简单问题")
        self.assertEqual(complexity_result['status'], 'success')
        
        # 4. NLP处理
        nlp_result = self.api.process_nlp("自然语言文本")
        self.assertEqual(nlp_result['status'], 'success')
        
        # 5. 批量处理
        batch_result = self.api.batch_process(["文本1", "文本2"])
        self.assertEqual(batch_result['status'], 'success')
    
    def test_configuration_options(self):
        """测试配置选项"""
        configs = [
            {"processing_mode": "nlp"},
            {"processing_mode": "relation"},
            {"processing_mode": "classification"},
            {"processing_mode": "comprehensive"},
        ]
        
        for config in configs:
            with self.subTest(config=config):
                result = self.api.process_text("测试文本", config)
                self.assertEqual(result['status'], 'success')
    
    def test_performance_under_load(self):
        """测试负载下的性能"""
        # 创建大量测试数据
        test_data = [f"测试文本{i}" for i in range(50)]
        
        # 批量处理
        result = self.api.batch_process(test_data)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['processed_samples']), 50)
        
        # 检查统计信息
        self.assertIn('statistics', result)
        stats = result['statistics']
        self.assertIn('total_samples', stats)
        self.assertIn('successful_samples', stats)
        self.assertIn('success_rate', stats)


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2) 