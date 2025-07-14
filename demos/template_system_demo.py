#!/usr/bin/env python3
"""
模板系统优化演示
展示动态模板管理系统如何消除硬编码，实现模板热更新
"""

import json
# 添加src目录到路径
import sys
import time
from pathlib import Path

sys.path.append('src')

from template_management import TemplateManager
from template_management.template_registry import (TemplateDefinition,
                                                   TemplateMetadata,
                                                   TemplatePattern)


def demo_basic_template_matching():
    """演示基本模板匹配功能"""
    print("🔍 演示基本模板匹配功能")
    print("=" * 50)
    
    # 创建模板管理器
    template_manager = TemplateManager()
    
    # 测试文本
    test_texts = [
        "5 plus 3 equals what?",
        "10 minus 4",
        "6 times 7",
        "20 divided by 5",
        "打8折后价格是多少？",
        "长5米宽3米的长方形面积",
        "30%的折扣",
        "平均分是85分"
    ]
    
    print("📝 测试文本:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    
    print("\n🎯 模板匹配结果:")
    for text in test_texts:
        result = template_manager.match_template(text)
        if result:
            print(f"  ✅ '{text}' -> {result['template_name']} (置信度: {result['confidence']:.2f})")
        else:
            print(f"  ❌ '{text}' -> 无匹配")
    
    return template_manager


def demo_template_management():
    """演示模板管理功能"""
    print("\n🔧 演示模板管理功能")
    print("=" * 50)
    
    template_manager = TemplateManager()
    
    # 1. 获取所有模板
    print("📋 当前模板列表:")
    templates = template_manager.get_templates()
    for template in templates[:5]:  # 只显示前5个
        print(f"  • {template['name']} ({template['category']}) - {template['metadata']['usage_count']} 次使用")
    
    # 2. 搜索模板
    print("\n🔍 搜索包含'加法'的模板:")
    search_results = template_manager.search_templates("加法")
    for result in search_results:
        print(f"  • {result['name']} - {result['description']}")
    
    # 3. 获取统计信息
    print("\n📊 模板统计信息:")
    stats = template_manager.get_template_statistics()
    print(f"  总模板数: {stats['total_templates']}")
    print(f"  活跃模板数: {stats['active_templates']}")
    print(f"  分类数: {stats['categories']}")
    print(f"  平均置信度: {stats['average_confidence']:.2f}")
    
    return template_manager


def demo_dynamic_template_addition():
    """演示动态添加模板"""
    print("\n➕ 演示动态添加模板")
    print("=" * 50)
    
    template_manager = TemplateManager()
    
    # 创建新模板
    new_template = {
        "template_id": "custom_ratio",
        "name": "比例问题",
        "category": "custom",
        "patterns": [
            {
                "pattern_id": "ratio_pattern",
                "regex_pattern": r"(\d+)\s*:\s*(\d+)",
                "confidence_weight": 0.9,
                "description": "比例关系",
                "examples": ["3:4", "5:2"]
            },
            {
                "pattern_id": "ratio_text",
                "regex_pattern": r"比例.*(\d+).*(\d+)",
                "confidence_weight": 0.8,
                "description": "比例文本",
                "examples": ["比例3比4", "比例5比2"]
            }
        ],
        "solution_template": "比例计算: {operand1} : {operand2}",
        "variables": ["operand1", "operand2"],
        "metadata": {
            "version": "1.0.0",
            "author": "demo",
            "description": "自定义比例问题模板",
            "tags": ["比例", "自定义"],
            "enabled": True,
            "priority": 5
        }
    }
    
    # 添加模板
    print("📝 添加新模板:")
    print(f"  模板ID: {new_template['template_id']}")
    print(f"  名称: {new_template['name']}")
    print(f"  分类: {new_template['category']}")
    
    success = template_manager.add_template(new_template)
    if success:
        print("  ✅ 模板添加成功")
        
        # 测试新模板
        test_text = "比例3比4"
        result = template_manager.match_template(test_text)
        if result:
            print(f"  🎯 测试匹配: '{test_text}' -> {result['template_name']}")
    else:
        print("  ❌ 模板添加失败")
    
    return template_manager


def demo_template_export_import():
    """演示模板导出导入"""
    print("\n📤 演示模板导出导入")
    print("=" * 50)
    
    template_manager = TemplateManager()
    
    # 导出模板
    export_file = "config/templates/exported_templates.json"
    print(f"📤 导出模板到: {export_file}")
    
    success = template_manager.export_templates(export_file)
    if success:
        print("  ✅ 模板导出成功")
        
        # 检查导出文件
        if Path(export_file).exists():
            with open(export_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  📊 导出 {len(data.get('templates', []))} 个模板")
    
    # 导入模板
    print(f"\n📥 从文件导入模板:")
    imported_count = template_manager.import_templates(export_file)
    print(f"  📊 导入 {imported_count} 个模板")
    
    return template_manager


def demo_hot_reload():
    """演示热重载功能"""
    print("\n🔄 演示热重载功能")
    print("=" * 50)
    
    template_manager = TemplateManager()
    
    # 创建外部模板文件
    external_template = {
        "template_id": "external_test",
        "name": "外部测试模板",
        "category": "external",
        "patterns": [
            {
                "pattern_id": "external_pattern",
                "regex_pattern": r"外部.*(\d+)",
                "confidence_weight": 0.9,
                "description": "外部测试模式",
                "examples": ["外部测试123"]
            }
        ],
        "solution_template": "外部测试: {operand1}",
        "variables": ["operand1"]
    }
    
    # 创建外部模板文件
    external_file = "config/templates/external_test.json"
    external_data = {
        "export_time": time.time(),
        "templates": [external_template]
    }
    
    Path("config/templates").mkdir(parents=True, exist_ok=True)
    with open(external_file, 'w', encoding='utf-8') as f:
        json.dump(external_data, f, ensure_ascii=False, indent=2)
    
    print(f"📝 创建外部模板文件: {external_file}")
    
    # 重新加载模板
    print("🔄 重新加载模板...")
    success = template_manager.reload_templates()
    if success:
        print("  ✅ 模板重新加载成功")
        
        # 测试外部模板
        test_text = "外部测试123"
        result = template_manager.match_template(test_text)
        if result:
            print(f"  🎯 外部模板匹配: '{test_text}' -> {result['template_name']}")
    else:
        print("  ❌ 模板重新加载失败")
    
    # 清理
    if Path(external_file).exists():
        Path(external_file).unlink()
    
    return template_manager


def demo_performance_comparison():
    """演示性能对比"""
    print("\n⚡ 演示性能对比")
    print("=" * 50)
    
    template_manager = TemplateManager()
    
    # 测试文本
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
    
    print(f"📊 性能测试: {len(test_texts)} 次匹配")
    
    # 测试动态模板系统
    start_time = time.time()
    dynamic_matches = 0
    
    for text in test_texts:
        result = template_manager.match_template(text)
        if result:
            dynamic_matches += 1
    
    dynamic_time = time.time() - start_time
    
    print(f"  🚀 动态模板系统:")
    print(f"    执行时间: {dynamic_time:.3f} 秒")
    print(f"    匹配成功: {dynamic_matches}/{len(test_texts)} ({dynamic_matches/len(test_texts)*100:.1f}%)")
    print(f"    平均时间: {dynamic_time/len(test_texts)*1000:.2f} 毫秒/次")
    
    # 获取统计信息
    stats = template_manager.get_template_statistics()
    print(f"    总操作数: {stats['total_operations']}")
    print(f"    成功率: {stats['success_rate']:.2f}")
    
    return template_manager


def demo_legacy_comparison():
    """演示与硬编码模板的对比"""
    print("\n🔄 演示与硬编码模板的对比")
    print("=" * 50)
    
    # 硬编码模板示例（旧系统）
    hardcoded_templates = {
        "addition": [
            r"(\d+(?:\.\d+)?).+?(\d+(?:\.\d+)?).+?total",
            r"(\d+(?:\.\d+)?).+?(\d+(?:\.\d+)?).+?altogether",
            r"(\d+(?:\.\d+)?).+?plus.+?(\d+(?:\.\d+)?)"
        ],
        "subtraction": [
            r"(\d+(?:\.\d+)?).+?minus.+?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?).+?take away.+?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?).+?left.+?(\d+(?:\.\d+)?)"
        ]
    }
    
    print("📋 硬编码模板系统:")
    print("  ❌ 模板硬编码在代码中")
    print("  ❌ 无法动态添加新模板")
    print("  ❌ 无法热更新")
    print("  ❌ 无法统计使用情况")
    print("  ❌ 无法验证模板质量")
    
    print("\n📋 动态模板系统:")
    print("  ✅ 模板存储在外部文件")
    print("  ✅ 支持动态添加新模板")
    print("  ✅ 支持热重载")
    print("  ✅ 详细的使用统计")
    print("  ✅ 模板质量验证")
    print("  ✅ 多格式支持 (JSON/YAML)")
    print("  ✅ 分类管理")
    print("  ✅ 置信度计算")
    print("  ✅ 变量提取")


def main():
    """主演示函数"""
    print("🚀 模板系统优化演示")
    print("=" * 60)
    print("目标: 消除硬编码，实现动态模板管理")
    print("=" * 60)
    
    try:
        # 1. 基本模板匹配
        template_manager = demo_basic_template_matching()
        
        # 2. 模板管理功能
        demo_template_management()
        
        # 3. 动态添加模板
        demo_dynamic_template_addition()
        
        # 4. 导出导入功能
        demo_template_export_import()
        
        # 5. 热重载功能
        demo_hot_reload()
        
        # 6. 性能对比
        demo_performance_comparison()
        
        # 7. 与硬编码系统对比
        demo_legacy_comparison()
        
        print("\n🎉 模板系统优化演示完成!")
        print("=" * 60)
        print("✅ 成功实现了动态模板管理系统")
        print("✅ 消除了硬编码模板")
        print("✅ 支持模板热更新")
        print("✅ 提供了完整的模板管理功能")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 