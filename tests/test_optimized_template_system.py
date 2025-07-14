#!/usr/bin/env python3
"""
优化后的模板系统测试
测试动态模板管理系统的各项功能
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append('src')

from template_management import TemplateManager
from template_management.template_matcher import MatchResult, TemplateMatcher
from template_management.template_registry import (TemplateDefinition,
                                                   TemplateMetadata,
                                                   TemplatePattern,
                                                   TemplateRegistry)
from template_management.template_validator import TemplateValidator


class TestOptimizedTemplateRegistry(unittest.TestCase):
    """测试优化后的模板注册表"""
    
    def setUp(self):
        """设置测试环境"""
        self.registry = TemplateRegistry()
    
    def test_register_template(self):
        """测试注册模板"""
        # 创建测试模板
        template = TemplateDefinition(
            template_id="test_template",
            name="测试模板",
            category="test",
            patterns=[
                TemplatePattern(
                    pattern_id="test_pattern",
                    regex_pattern=r"测试.*(\d+)",
                    confidence_weight=0.9,
                    description="测试模式",
                    examples=["测试123"]
                )
            ],
            solution_template="测试结果: {operand1}",
            variables=["operand1"]
        )
        
        # 注册模板
        success = self.registry.register_template(template)
        self.assertTrue(success)
        
        # 验证模板已注册
        registered_template = self.registry.get_template("test_template")
        self.assertIsNotNone(registered_template)
        self.assertEqual(registered_template.name, "测试模板")
    
    def test_unregister_template(self):
        """测试注销模板"""
        # 先注册模板
        template = TemplateDefinition(
            template_id="test_unregister",
            name="测试注销模板",
            category="test",
            patterns=[
                TemplatePattern(
                    pattern_id="test_pattern",
                    regex_pattern=r"测试.*(\d+)",
                    confidence_weight=0.9
                )
            ],
            solution_template="测试结果: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template)
        
        # 注销模板
        success = self.registry.unregister_template("test_unregister")
        self.assertTrue(success)
        
        # 验证模板已注销
        registered_template = self.registry.get_template("test_unregister")
        self.assertIsNone(registered_template)
    
    def test_get_active_templates(self):
        """测试获取活跃模板"""
        # 创建启用和禁用的模板
        enabled_template = TemplateDefinition(
            template_id="enabled_template",
            name="启用模板",
            category="test",
            patterns=[TemplatePattern("pattern1", r"启用.*(\d+)", 0.9)],
            solution_template="启用结果: {operand1}",
            variables=["operand1"]
        )
        enabled_template.metadata.enabled = True
        
        disabled_template = TemplateDefinition(
            template_id="disabled_template",
            name="禁用模板",
            category="test",
            patterns=[TemplatePattern("pattern2", r"禁用.*(\d+)", 0.9)],
            solution_template="禁用结果: {operand1}",
            variables=["operand1"]
        )
        disabled_template.metadata.enabled = False
        
        self.registry.register_template(enabled_template)
        self.registry.register_template(disabled_template)
        
        # 获取活跃模板
        active_templates = self.registry.get_active_templates()
        
        # 验证只有启用的模板被返回
        self.assertEqual(len(active_templates), 1)
        self.assertEqual(active_templates[0].template_id, "enabled_template")
    
    def test_search_templates(self):
        """测试搜索模板"""
        # 创建测试模板
        template1 = TemplateDefinition(
            template_id="search_test1",
            name="加法运算",
            category="arithmetic",
            patterns=[TemplatePattern("pattern1", r"加.*(\d+)", 0.9)],
            solution_template="加法: {operand1}",
            variables=["operand1"]
        )
        template1.metadata.description = "基本加法运算"
        template1.metadata.tags = ["加法", "运算"]
        
        template2 = TemplateDefinition(
            template_id="search_test2",
            name="减法运算",
            category="arithmetic",
            patterns=[TemplatePattern("pattern2", r"减.*(\d+)", 0.9)],
            solution_template="减法: {operand1}",
            variables=["operand1"]
        )
        template2.metadata.description = "基本减法运算"
        template2.metadata.tags = ["减法", "运算"]
        
        self.registry.register_template(template1)
        self.registry.register_template(template2)
        
        # 搜索包含"加法"的模板
        results = self.registry.search_templates("加法")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].template_id, "search_test1")
        
        # 搜索包含"运算"的模板
        results = self.registry.search_templates("运算")
        self.assertEqual(len(results), 2)
    
    def test_update_template_usage(self):
        """测试更新模板使用统计"""
        template = TemplateDefinition(
            template_id="usage_test",
            name="使用统计测试",
            category="test",
            patterns=[TemplatePattern("pattern1", r"统计.*(\d+)", 0.9)],
            solution_template="统计: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template)
        
        # 更新使用统计
        self.registry.update_template_usage("usage_test", success=True)
        self.registry.update_template_usage("usage_test", success=False)
        
        # 验证统计更新
        updated_template = self.registry.get_template("usage_test")
        self.assertEqual(updated_template.metadata.usage_count, 2)
        self.assertIsNotNone(updated_template.metadata.last_used)
    
    def test_export_import_templates(self):
        """测试导出导入模板"""
        # 创建测试模板
        template = TemplateDefinition(
            template_id="export_test",
            name="导出测试模板",
            category="test",
            patterns=[TemplatePattern("pattern1", r"导出.*(\d+)", 0.9)],
            solution_template="导出: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template)
        
        # 导出模板
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            success = self.registry.export_templates(export_file)
            self.assertTrue(success)
            
            # 验证导出文件
            with open(export_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assertIn("templates", data)
                self.assertEqual(len(data["templates"]), 1)
                self.assertEqual(data["templates"][0]["template_id"], "export_test")
            
            # 清空注册表
            self.registry.unregister_template("export_test")
            self.assertEqual(len(self.registry.get_all_templates()), 0)
            
            # 导入模板
            imported_count = self.registry.import_templates(export_file)
            self.assertEqual(imported_count, 1)
            
            # 验证导入成功
            imported_template = self.registry.get_template("export_test")
            self.assertIsNotNone(imported_template)
            self.assertEqual(imported_template.name, "导出测试模板")
            
        finally:
            # 清理临时文件
            Path(export_file).unlink(missing_ok=True)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        # 创建测试模板
        template = TemplateDefinition(
            template_id="stats_test",
            name="统计测试模板",
            category="test",
            patterns=[TemplatePattern("pattern1", r"统计.*(\d+)", 0.9)],
            solution_template="统计: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template)
        
        # 获取统计信息
        stats = self.registry.get_stats()
        
        # 验证统计信息
        self.assertIn("total_templates", stats)
        self.assertIn("active_templates", stats)
        self.assertIn("categories", stats)
        self.assertIn("template_count_by_category", stats)
        
        self.assertEqual(stats["total_templates"], 1)
        self.assertEqual(stats["active_templates"], 1)
        self.assertIn("test", stats["categories"])


class TestOptimizedTemplateMatcher(unittest.TestCase):
    """测试优化后的模板匹配器"""
    
    def setUp(self):
        """设置测试环境"""
        self.registry = TemplateRegistry()
        self.matcher = TemplateMatcher(self.registry)
    
    def test_match_text_best(self):
        """测试最佳匹配"""
        # 创建测试模板
        template = TemplateDefinition(
            template_id="match_test",
            name="匹配测试模板",
            category="test",
            patterns=[
                TemplatePattern("pattern1", r"匹配.*(\d+)", 0.9),
                TemplatePattern("pattern2", r"测试.*(\d+)", 0.8)
            ],
            solution_template="匹配结果: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template)
        
        # 测试匹配
        result = self.matcher.match_text_best("匹配123")
        self.assertIsNotNone(result)
        self.assertEqual(result.template_id, "match_test")
        self.assertEqual(result.confidence, 0.9)
        
        # 测试无匹配
        result = self.matcher.match_text_best("不匹配的文本")
        self.assertIsNone(result)
    
    def test_match_text_multiple(self):
        """测试多匹配结果"""
        # 创建多个测试模板
        template1 = TemplateDefinition(
            template_id="high_confidence",
            name="高置信度模板",
            category="test",
            patterns=[TemplatePattern("pattern1", r"高.*(\d+)", 0.95)],
            solution_template="高置信度: {operand1}",
            variables=["operand1"]
        )
        
        template2 = TemplateDefinition(
            template_id="low_confidence",
            name="低置信度模板",
            category="test",
            patterns=[TemplatePattern("pattern2", r"低.*(\d+)", 0.8)],
            solution_template="低置信度: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template1)
        self.registry.register_template(template2)
        
        # 测试多匹配（应该按置信度排序）
        results = self.matcher.match_text("高123低456")
        
        # 验证结果按置信度排序
        self.assertGreater(len(results), 0)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].confidence, results[i + 1].confidence)
    
    def test_extract_variables(self):
        """测试变量提取"""
        template = TemplateDefinition(
            template_id="variable_test",
            name="变量测试模板",
            category="test",
            patterns=[TemplatePattern("pattern1", r"变量.*(\d+).*(\d+)", 0.9)],
            solution_template="变量结果: {operand1} + {operand2}",
            variables=["operand1", "operand2"]
        )
        
        self.registry.register_template(template)
        
        # 测试变量提取
        text = "变量123和456"
        variables = self.matcher.extract_variables(text, template)
        
        # 验证变量提取
        self.assertIn("operand1", variables)
        self.assertIn("operand2", variables)
        self.assertEqual(variables["operand1"], 123.0)
        self.assertEqual(variables["operand2"], 456.0)
    
    def test_get_match_statistics(self):
        """测试获取匹配统计"""
        # 创建测试模板并进行匹配
        template = TemplateDefinition(
            template_id="stats_test",
            name="统计测试模板",
            category="test",
            patterns=[TemplatePattern("pattern1", r"统计.*(\d+)", 0.9)],
            solution_template="统计: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template)
        
        # 进行一些匹配
        self.matcher.match_text_best("统计123")
        self.matcher.match_text_best("统计456")
        
        # 获取统计信息
        stats = self.matcher.get_match_statistics()
        
        # 验证统计信息
        self.assertIn("total_matches", stats)
        self.assertIn("successful_matches", stats)
        self.assertIn("average_confidence", stats)
        self.assertIn("total_templates", stats)
        self.assertIn("active_templates", stats)


class TestOptimizedTemplateValidator(unittest.TestCase):
    """测试优化后的模板验证器"""
    
    def setUp(self):
        """设置测试环境"""
        self.validator = TemplateValidator()
    
    def test_validate_template(self):
        """测试模板验证"""
        # 创建有效模板
        valid_template = TemplateDefinition(
            template_id="valid_test",
            name="有效测试模板",
            category="test",
            patterns=[
                TemplatePattern("pattern1", r"有效.*(\d+)", 0.9)
            ],
            solution_template="有效结果: {operand1}",
            variables=["operand1"]
        )
        
        # 验证有效模板
        result = self.validator.validate_template(valid_template)
        self.assertTrue(result)
    
    def test_validate_invalid_template(self):
        """测试无效模板验证"""
        # 创建无效模板（缺少必需字段）
        invalid_template = TemplateDefinition(
            template_id="",  # 无效ID
            name="",  # 无效名称
            category="",  # 无效分类
            patterns=[],  # 空模式列表
            solution_template="",
            variables=[]
        )
        
        # 验证无效模板
        result = self.validator.validate_template(invalid_template)
        self.assertFalse(result)
    
    def test_validate_template_dict(self):
        """测试模板字典验证"""
        # 有效模板字典
        valid_dict = {
            "template_id": "dict_test",
            "name": "字典测试模板",
            "category": "test",
            "patterns": [
                {
                    "pattern_id": "pattern1",
                    "regex_pattern": r"字典.*(\d+)",
                    "confidence_weight": 0.9,
                    "description": "字典模式",
                    "examples": ["字典123"]
                }
            ],
            "solution_template": "字典结果: {operand1}",
            "variables": ["operand1"]
        }
        
        # 验证有效字典
        result = self.validator.validate_template_dict(valid_dict)
        self.assertTrue(result)
        
        # 无效模板字典
        invalid_dict = {
            "template_id": "",  # 无效ID
            "name": "",
            "category": "",
            "patterns": [],  # 空模式列表
            "solution_template": ""
        }
        
        # 验证无效字典
        result = self.validator.validate_template_dict(invalid_dict)
        self.assertFalse(result)
    
    def test_validate_pattern(self):
        """测试模式验证"""
        # 有效模式
        valid_pattern = TemplatePattern(
            pattern_id="valid_pattern",
            regex_pattern=r"有效.*(\d+)",
            confidence_weight=0.9,
            description="有效模式",
            examples=["有效123"]
        )
        
        # 验证有效模式
        result = self.validator.validate_pattern(valid_pattern)
        self.assertTrue(result)
        
        # 无效模式
        invalid_pattern = TemplatePattern(
            pattern_id="",  # 无效ID
            regex_pattern="",  # 无效正则表达式
            confidence_weight=1.5  # 无效置信度
        )
        
        # 验证无效模式
        result = self.validator.validate_pattern(invalid_pattern)
        self.assertFalse(result)


class TestOptimizedTemplateManager(unittest.TestCase):
    """测试优化后的模板管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = TemplateManager()
    
    def test_match_template(self):
        """测试模板匹配"""
        # 测试匹配
        result = self.manager.match_template("5 plus 3")
        self.assertIsNotNone(result)
        self.assertIn("template_id", result)
        self.assertIn("confidence", result)
        self.assertIn("template_name", result)
    
    def test_get_templates(self):
        """测试获取模板列表"""
        templates = self.manager.get_templates()
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)
        
        # 验证模板结构
        if templates:
            template = templates[0]
            self.assertIn("template_id", template)
            self.assertIn("name", template)
            self.assertIn("category", template)
            self.assertIn("patterns", template)
            self.assertIn("solution_template", template)
            self.assertIn("variables", template)
            self.assertIn("metadata", template)
    
    def test_add_template(self):
        """测试添加模板"""
        template_data = {
            "template_id": "test_add",
            "name": "测试添加模板",
            "category": "test",
            "patterns": [
                {
                    "pattern_id": "add_pattern",
                    "regex_pattern": r"添加.*(\d+)",
                    "confidence_weight": 0.9,
                    "description": "添加模式",
                    "examples": ["添加123"]
                }
            ],
            "solution_template": "添加结果: {operand1}",
            "variables": ["operand1"]
        }
        
        success = self.manager.add_template(template_data)
        self.assertTrue(success)
        
        # 验证模板已添加
        result = self.manager.match_template("添加456")
        self.assertIsNotNone(result)
        self.assertEqual(result["template_id"], "test_add")
    
    def test_remove_template(self):
        """测试移除模板"""
        # 先添加模板
        template_data = {
            "template_id": "test_remove",
            "name": "测试移除模板",
            "category": "test",
            "patterns": [
                {
                    "pattern_id": "remove_pattern",
                    "regex_pattern": r"移除.*(\d+)",
                    "confidence_weight": 0.9
                }
            ],
            "solution_template": "移除结果: {operand1}",
            "variables": ["operand1"]
        }
        
        self.manager.add_template(template_data)
        
        # 移除模板
        success = self.manager.remove_template("test_remove")
        self.assertTrue(success)
        
        # 验证模板已移除
        result = self.manager.match_template("移除123")
        self.assertIsNone(result)
    
    def test_update_template(self):
        """测试更新模板"""
        # 先添加模板
        template_data = {
            "template_id": "test_update",
            "name": "原始模板",
            "category": "test",
            "patterns": [
                {
                    "pattern_id": "update_pattern",
                    "regex_pattern": r"更新.*(\d+)",
                    "confidence_weight": 0.9
                }
            ],
            "solution_template": "原始结果: {operand1}",
            "variables": ["operand1"]
        }
        
        self.manager.add_template(template_data)
        
        # 更新模板
        updates = {
            "name": "更新后的模板",
            "solution_template": "更新后的结果: {operand1}"
        }
        
        success = self.manager.update_template("test_update", updates)
        self.assertTrue(success)
        
        # 验证模板已更新
        templates = self.manager.get_templates()
        updated_template = next((t for t in templates if t["template_id"] == "test_update"), None)
        self.assertIsNotNone(updated_template)
        self.assertEqual(updated_template["name"], "更新后的模板")
    
    def test_get_template_statistics(self):
        """测试获取模板统计信息"""
        stats = self.manager.get_template_statistics()
        
        # 验证统计信息结构
        self.assertIn("total_templates", stats)
        self.assertIn("active_templates", stats)
        self.assertIn("categories", stats)
        self.assertIn("total_operations", stats)
        self.assertIn("average_response_time", stats)
        self.assertIn("success_rate", stats)
    
    def test_search_templates(self):
        """测试搜索模板"""
        # 添加测试模板
        template_data = {
            "template_id": "search_test",
            "name": "搜索测试模板",
            "category": "test",
            "patterns": [
                {
                    "pattern_id": "search_pattern",
                    "regex_pattern": r"搜索.*(\d+)",
                    "confidence_weight": 0.9
                }
            ],
            "solution_template": "搜索结果: {operand1}",
            "variables": ["operand1"]
        }
        
        self.manager.add_template(template_data)
        
        # 搜索模板
        results = self.manager.search_templates("搜索")
        self.assertGreater(len(results), 0)
        
        # 验证搜索结果
        found = any(r["template_id"] == "search_test" for r in results)
        self.assertTrue(found)
    
    def test_export_import_templates(self):
        """测试导出导入模板"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            # 导出模板
            success = self.manager.export_templates(export_file)
            self.assertTrue(success)
            
            # 导入模板
            imported_count = self.manager.import_templates(export_file)
            self.assertGreaterEqual(imported_count, 0)
            
        finally:
            # 清理临时文件
            Path(export_file).unlink(missing_ok=True)
    
    def test_reload_templates(self):
        """测试重新加载模板"""
        success = self.manager.reload_templates()
        self.assertTrue(success)


class TestOptimizedTemplateSystemIntegration(unittest.TestCase):
    """测试优化后的模板系统集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = TemplateManager()
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 1. 添加自定义模板
        custom_template = {
            "template_id": "integration_test",
            "name": "集成测试模板",
            "category": "integration",
            "patterns": [
                {
                    "pattern_id": "integration_pattern",
                    "regex_pattern": r"集成.*(\d+)",
                    "confidence_weight": 0.9,
                    "description": "集成测试模式",
                    "examples": ["集成123"]
                }
            ],
            "solution_template": "集成结果: {operand1}",
            "variables": ["operand1"]
        }
        
        success = self.manager.add_template(custom_template)
        self.assertTrue(success)
        
        # 2. 测试模板匹配
        result = self.manager.match_template("集成456")
        self.assertIsNotNone(result)
        self.assertEqual(result["template_id"], "integration_test")
        
        # 3. 获取统计信息
        stats = self.manager.get_template_statistics()
        self.assertGreater(stats["total_templates"], 0)
        
        # 4. 搜索模板
        search_results = self.manager.search_templates("集成")
        self.assertEqual(len(search_results), 1)
        
        # 5. 移除模板
        remove_success = self.manager.remove_template("integration_test")
        self.assertTrue(remove_success)
        
        # 6. 验证移除成功
        result = self.manager.match_template("集成789")
        self.assertIsNone(result)
    
    def test_performance_under_load(self):
        """测试负载下的性能"""
        # 添加多个模板
        for i in range(10):
            template_data = {
                "template_id": f"perf_test_{i}",
                "name": f"性能测试模板{i}",
                "category": "performance",
                "patterns": [
                    {
                        "pattern_id": f"perf_pattern_{i}",
                        "regex_pattern": rf"性能.*{i}.*(\d+)",
                        "confidence_weight": 0.9
                    }
                ],
                "solution_template": f"性能结果{i}: {{operand1}}",
                "variables": ["operand1"]
            }
            self.manager.add_template(template_data)
        
        # 进行多次匹配测试
        import time
        start_time = time.time()
        
        for i in range(100):
            result = self.manager.match_template(f"性能{i}123")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证性能（应该在合理时间内完成）
        self.assertLess(total_time, 5.0)  # 5秒内完成100次匹配
        
        # 验证统计信息
        stats = self.manager.get_template_statistics()
        self.assertGreater(stats["total_operations"], 0)
    
    def test_concurrent_access(self):
        """测试并发访问"""
        import queue
        import threading
        
        results = queue.Queue()
        
        def worker(worker_id):
            """工作线程函数"""
            try:
                # 每个线程进行一些模板操作
                for i in range(10):
                    template_data = {
                        "template_id": f"concurrent_{worker_id}_{i}",
                        "name": f"并发测试模板{worker_id}_{i}",
                        "category": "concurrent",
                        "patterns": [
                            {
                                "pattern_id": f"concurrent_pattern_{worker_id}_{i}",
                                "regex_pattern": rf"并发.*{worker_id}.*{i}.*(\d+)",
                                "confidence_weight": 0.9
                            }
                        ],
                        "solution_template": f"并发结果{worker_id}_{i}: {{operand1}}",
                        "variables": ["operand1"]
                    }
                    
                    success = self.manager.add_template(template_data)
                    results.put(success)
                    
                    # 测试匹配
                    result = self.manager.match_template(f"并发{worker_id}{i}123")
                    results.put(result is not None)
                    
            except Exception as e:
                results.put(f"Error in worker {worker_id}: {e}")
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        success_count = 0
        error_count = 0
        
        while not results.empty():
            result = results.get()
            if isinstance(result, bool):
                if result:
                    success_count += 1
            elif isinstance(result, str) and result.startswith("Error"):
                error_count += 1
        
        # 验证大部分操作成功
        self.assertGreater(success_count, 0)
        self.assertEqual(error_count, 0)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2) 