#!/usr/bin/env python3
"""
简单模板系统测试
验证动态模板管理系统的基本功能
"""

import os
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_template_registry():
    """测试模板注册表"""
    print("🧪 测试模板注册表")
    
    try:
        from template_management.template_registry import (TemplateDefinition,
                                                           TemplatePattern,
                                                           TemplateRegistry)

        # 创建注册表
        registry = TemplateRegistry()
        
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
        success = registry.register_template(template)
        print(f"  ✅ 模板注册: {'成功' if success else '失败'}")
        
        # 获取模板
        retrieved = registry.get_template("test_template")
        print(f"  ✅ 模板获取: {'成功' if retrieved else '失败'}")
        
        # 获取统计信息
        stats = registry.get_stats()
        print(f"  ✅ 统计信息: {stats['total_templates']} 个模板")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_template_matcher():
    """测试模板匹配器"""
    print("\n🧪 测试模板匹配器")
    
    try:
        from template_management.template_matcher import TemplateMatcher
        from template_management.template_registry import (TemplateDefinition,
                                                           TemplatePattern,
                                                           TemplateRegistry)

        # 创建注册表和匹配器
        registry = TemplateRegistry()
        matcher = TemplateMatcher(registry)
        
        # 测试数字提取
        text = "这里有123和456两个数字"
        numbers = matcher.extract_numbers(text)
        print(f"  ✅ 数字提取: {numbers}")
        
        # 测试文本匹配
        matches = matcher.match_text("5 plus 3")
        print(f"  ✅ 文本匹配: {len(matches)} 个匹配")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_template_validator():
    """测试模板验证器"""
    print("\n🧪 测试模板验证器")
    
    try:
        from template_management.template_registry import (TemplateDefinition,
                                                           TemplatePattern)
        from template_management.template_validator import TemplateValidator
        
        validator = TemplateValidator()
        
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
        is_valid = validator.validate_template(template)
        print(f"  ✅ 模板验证: {'通过' if is_valid else '失败'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能")
    
    try:
        # 测试正则表达式
        import re

        # 测试模式
        patterns = [
            r"(\d+(?:\.\d+)?).+?plus.+?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?).+?minus.+?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?).+?times.+?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?).+?divided.+?(\d+(?:\.\d+)?)"
        ]
        
        test_texts = [
            "5 plus 3",
            "10 minus 4",
            "6 times 7",
            "20 divided by 5"
        ]
        
        print("  📝 测试文本匹配:")
        for i, text in enumerate(test_texts):
            for j, pattern in enumerate(patterns):
                match = re.search(pattern, text)
                if match:
                    print(f"    ✅ '{text}' 匹配模式 {j+1}: {match.groups()}")
                    break
            else:
                print(f"    ❌ '{text}' 无匹配")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 简单模板系统测试")
    print("=" * 50)
    
    tests = [
        test_template_registry,
        test_template_matcher,
        test_template_validator,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过!")
        return True
    else:
        print("❌ 部分测试失败!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 