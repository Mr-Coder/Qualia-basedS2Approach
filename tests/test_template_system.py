#!/usr/bin/env python3
"""
模板系统测试
测试动态模板管理系统的各项功能
"""

import json
# 添加src目录到路径
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


class TestTemplateRegistry(unittest.TestCase):
    """测试模板注册表"""
    
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
        # 先注册一个模板
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
    
    def test_get_templates_by_category(self):
        """测试按分类获取模板"""
        # 注册多个模板
        template1 = TemplateDefinition(
            template_id="test1",
            name="测试模板1",
            category="arithmetic",
            patterns=[TemplatePattern("p1", r"(\d+)", 0.9)],
            solution_template="结果: {operand1}",
            variables=["operand1"]
        )
        
        template2 = TemplateDefinition(
            template_id="test2", 
            name="测试模板2",
            category="arithmetic",
            patterns=[TemplatePattern("p2", r"(\d+)", 0.9)],
            solution_template="结果: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template1)
        self.registry.register_template(template2)
        
        # 获取arithmetic分类的模板
        templates = self.registry.get_templates_by_category("arithmetic")
        self.assertEqual(len(templates), 2)
    
    def test_search_templates(self):
        """测试搜索模板"""
        # 注册测试模板
        template = TemplateDefinition(
            template_id="search_test",
            name="搜索测试模板",
            category="test",
            patterns=[TemplatePattern("p1", r"(\d+)", 0.9)],
            solution_template="结果: {operand1}",
            variables=["operand1"],
            metadata=TemplateMetadata(
                template_id="search_test",
                name="搜索测试模板",
                category="test",
                description="这是一个搜索测试模板",
                tags=["测试", "搜索"]
            )
        )
        
        self.registry.register_template(template)
        
        # 搜索模板
        results = self.registry.search_templates("搜索")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "搜索测试模板")


class TestTemplateMatcher(unittest.TestCase):
    """测试模板匹配器"""
    
    def setUp(self):
        """设置测试环境"""
        self.registry = TemplateRegistry()
        self.matcher = TemplateMatcher(self.registry)
    
    def test_match_text(self):
        """测试文本匹配"""
        # 注册测试模板
        template = TemplateDefinition(
            template_id="match_test",
            name="匹配测试模板",
            category="test",
            patterns=[
                TemplatePattern(
                    pattern_id="match_pattern",
                    regex_pattern=r"测试.*(\d+)",
                    confidence_weight=0.9,
                    description="匹配测试模式",
                    examples=["测试123"]
                )
            ],
            solution_template="匹配结果: {operand1}",
            variables=["operand1"]
        )
        
        self.registry.register_template(template)
        
        # 测试匹配
        matches = self.matcher.match_text("测试456")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_id, "match_test")
        self.assertGreater(matches[0].confidence, 0.8)
    
    def test_extract_numbers(self):
        """测试数字提取"""
        text = "这里有123和456两个数字"
        numbers = self.matcher.extract_numbers(text)
        self.assertEqual(numbers, [123.0, 456.0])
    
    def test_extract_variables(self):
        """测试变量提取"""
        template = TemplateDefinition(
            template_id="var_test",
            name="变量测试模板",
            category="test",
            patterns=[TemplatePattern("p1", r"(\d+)", 0.9)],
            solution_template="结果: {operand1}",
            variables=["operand1", "operand2"]
        )
        
        text = "测试123和456"
        variables = self.matcher.extract_variables(text, template)
        
        self.assertIn("numbers", variables)
        self.assertIn("operand1", variables)
        self.assertIn("operand2", variables)
        self.assertEqual(variables["operand1"], 123.0)
        self.assertEqual(variables["operand2"], 456.0)


class TestTemplateValidator(unittest.TestCase):
    """测试模板验证器"""
    
    def setUp(self):
        """设置测试环境"""
        self.validator = TemplateValidator()
    
    def test_validate_template(self):
        """测试模板验证"""
        # 创建有效模板
        template = TemplateDefinition(
            template_id="valid_template",
            name="有效模板",
            category="test",
            patterns=[
                TemplatePattern(
                    pattern_id="valid_pattern",
                    regex_pattern=r"(\d+)",
                    confidence_weight=0.9
                )
            ],
            solution_template="结果: {operand1}",
            variables=["operand1"]
        )
        
        # 验证模板
        is_valid = self.validator.validate_template(template)
        self.assertTrue(is_valid)
    
    def test_validate_invalid_template(self):
        """测试无效模板验证"""
        # 创建无效模板（缺少必需字段）
        template = TemplateDefinition(
            template_id="",  # 空的模板ID
            name="",
            category="",
            patterns=[],
            solution_template="",
            variables=[]
        )
        
        # 验证模板
        is_valid = self.validator.validate_template(template)
        self.assertFalse(is_valid)
    
    def test_validate_regex_pattern(self):
        """测试正则表达式验证"""
        # 有效正则表达式
        is_valid = self.validator._validate_regex_pattern(r"(\d+)")
        self.assertTrue(is_valid)
        
        # 无效正则表达式
        is_valid = self.validator._validate_regex_pattern(r"([")
        self.assertFalse(is_valid)


class TestTemplateManager(unittest.TestCase):
    """测试模板管理器"""
    
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
    
    def test_get_templates(self):
        """测试获取模板列表"""
        templates = self.manager.get_templates()
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)
    
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
    
    def test_remove_template(self):
        """测试移除模板"""
        # 先添加一个模板
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
    
    def test_export_import_templates(self):
        """测试导出导入模板"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # 导出模板
            success = self.manager.export_templates(temp_file)
            self.assertTrue(success)
            
            # 检查文件是否存在
            self.assertTrue(Path(temp_file).exists())
            
            # 导入模板
            imported_count = self.manager.import_templates(temp_file)
            self.assertGreater(imported_count, 0)
            
        finally:
            # 清理临时文件
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        stats = self.manager.get_template_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_templates", stats)
        self.assertIn("active_templates", stats)
        self.assertIn("categories", stats)


class TestTemplateSystemIntegration(unittest.TestCase):
    """测试模板系统集成"""
    
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
    
    def test_performance_benchmark(self):
        """测试性能基准"""
        import time

        # 准备测试数据
        test_texts = [
            "5 plus 3",
            "10 minus 4",
            "6 times 7", 
            "20 divided by 5",
            "打8折",
            "长5宽3",
            "30%折扣",
            "平均分85"
        ] * 10  # 重复10次
        
        # 性能测试
        start_time = time.time()
        matches = 0
        
        for text in test_texts:
            result = self.manager.match_template(text)
            if result:
                matches += 1
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证性能
        self.assertLess(execution_time, 5.0)  # 应该在5秒内完成
        self.assertGreater(matches, 0)  # 应该有匹配结果
        
        print(f"性能测试结果: {len(test_texts)} 次匹配, {execution_time:.3f} 秒, {matches} 次成功")


def run_template_system_tests():
    """运行模板系统测试"""
    print("🧪 运行模板系统测试")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestTemplateRegistry,
        TestTemplateMatcher, 
        TestTemplateValidator,
        TestTemplateManager,
        TestTemplateSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print(f"\n📊 测试结果:")
    print(f"  运行测试: {result.testsRun}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n❌ 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 部分测试失败!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_template_system_tests()
    exit(0 if success else 1) 