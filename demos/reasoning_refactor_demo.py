#!/usr/bin/env python3
"""
推理引擎重构演示
展示策略模式重构后的推理引擎功能和优势
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json

from src.core.interfaces import ReasoningContext
from src.reasoning.new_reasoning_engine import ModernReasoningEngine
from src.reasoning.strategy_manager import StrategyType


def print_section(title: str, content: str = ""):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)
    if content:
        print(content)

def print_result(result, show_details=True):
    """打印推理结果"""
    print(f"✅ 成功: {result.success}")
    print(f"📊 答案: {result.result}")
    print(f"🎯 置信度: {result.confidence:.3f}")
    print(f"⏱️  处理时间: {result.processing_time:.3f}秒")
    
    if show_details and result.metadata:
        strategy = result.metadata.get("strategy_used", "未知")
        steps = len(result.metadata.get("reasoning_steps", []))
        print(f"🧠 使用策略: {strategy}")
        print(f"📝 推理步骤: {steps}步")

def demo_basic_functionality():
    """演示基础功能"""
    print_section("🚀 基础功能演示")
    
    engine = ModernReasoningEngine()
    
    test_problems = [
        "计算 15 + 27",
        "小明有20个苹果，吃了8个，还剩多少个？",
        "一个长方形长12米，宽8米，面积是多少平方米？",
        "100元商品打7折后的价格是多少？"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n问题 {i}: {problem}")
        print("-" * 50)
        
        start_time = time.time()
        result = engine.reason(problem)
        end_time = time.time()
        
        print_result(result)
        
        if result.success:
            print("✨ 推理成功!")
        else:
            print("❌ 推理失败:", result.error_message)

def demo_strategy_comparison():
    """演示策略对比"""
    print_section("🔄 策略对比演示", "展示不同策略处理同一问题的效果")
    
    engine = ModernReasoningEngine()
    
    # 复杂问题，适合展示策略差异
    complex_problem = """
    一个班级有42名学生，其中男生人数比女生多6人。
    如果要将所有学生分成若干个小组，每组人数相等且每组至少有3人，
    最多可以分成多少组？每组有多少人？
    """
    
    print(f"测试问题: {complex_problem}")
    
    # 获取可用策略
    strategies = engine.get_available_strategies()
    print(f"\n可用策略: {strategies}")
    
    # 让引擎自动选择策略
    print("\n🎯 自动策略选择:")
    result = engine.reason(complex_problem)
    print_result(result)
    
    # 显示推理步骤
    if result.metadata and "reasoning_steps" in result.metadata:
        steps = result.metadata["reasoning_steps"]
        print(f"\n📋 推理步骤详情 ({len(steps)}步):")
        for i, step in enumerate(steps[:5], 1):  # 只显示前5步
            action = step.get("action", "未知操作")
            description = step.get("description", "无描述")
            confidence = step.get("confidence", 0)
            print(f"  {i}. [{action}] {description} (置信度: {confidence:.2f})")
        
        if len(steps) > 5:
            print(f"  ... 还有 {len(steps)-5} 个步骤")

def demo_confidence_analysis():
    """演示置信度分析"""
    print_section("📊 置信度分析演示", "展示置信度计算的详细信息")
    
    engine = ModernReasoningEngine()
    
    test_cases = [
        {
            "problem": "5 + 3 = ?",
            "description": "简单算术 - 应该有高置信度"
        },
        {
            "problem": "如果x + 2 = 7，那么x等于多少？",
            "description": "简单方程 - 中等置信度"
        },
        {
            "problem": "根据量子力学原理计算电子的位置",
            "description": "超出范围问题 - 应该有低置信度"
        }
    ]
    
    for case in test_cases:
        print(f"\n🔍 {case['description']}")
        print(f"问题: {case['problem']}")
        print("-" * 50)
        
        result = engine.reason(case["problem"])
        print_result(result)
        
        # 详细置信度分析
        if result.metadata and "confidence_details" in result.metadata:
            conf_details = result.metadata["confidence_details"]
            print("\n🧮 置信度组成:")
            
            components = conf_details.get("component_confidences", {})
            for component, confidence in components.items():
                print(f"  • {component}: {confidence:.3f}")
            
            factors = conf_details.get("confidence_factors", [])
            if factors:
                print(f"  主要因子: {', '.join(factors)}")

def demo_performance_monitoring():
    """演示性能监控"""
    print_section("⚡ 性能监控演示", "展示引擎性能统计和监控功能")
    
    engine = ModernReasoningEngine()
    
    # 批量处理问题
    batch_problems = [
        "10 + 20",
        "50 - 15", 
        "6 × 8",
        "144 ÷ 12",
        "求2的3次方",
        "小王买了5支笔，每支3元，花了多少钱？",
        "圆的半径是7米，面积是多少？",
        "从1加到10等于多少？"
    ]
    
    print(f"批量处理 {len(batch_problems)} 个问题...")
    
    start_time = time.time()
    results = []
    
    for i, problem in enumerate(batch_problems, 1):
        print(f"处理问题 {i}/{len(batch_problems)}: {problem[:30]}...")
        result = engine.reason(problem)
        results.append(result)
    
    total_time = time.time() - start_time
    
    # 性能统计
    print(f"\n📈 性能统计:")
    print(f"  总处理时间: {total_time:.2f}秒")
    print(f"  平均每题时间: {total_time/len(batch_problems):.3f}秒")
    
    successful = [r for r in results if r.success]
    print(f"  成功率: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    if successful:
        avg_confidence = sum(r.confidence for r in successful) / len(successful)
        print(f"  平均置信度: {avg_confidence:.3f}")
    
    # 获取详细性能报告
    performance_report = engine.get_performance_report()
    print(f"\n📊 详细性能报告:")
    
    reasoning_stats = performance_report.get("reasoning_stats", {})
    print(f"  累计处理问题: {reasoning_stats.get('total_problems', 0)}")
    print(f"  累计成功问题: {reasoning_stats.get('successful_problems', 0)}")
    print(f"  整体成功率: {performance_report.get('success_rate', 0):.3f}")
    
    # 策略使用统计
    strategy_usage = reasoning_stats.get("strategy_usage", {})
    if strategy_usage:
        print(f"\n🎯 策略使用统计:")
        for strategy, count in strategy_usage.items():
            print(f"  {strategy}: {count}次")

def demo_error_handling():
    """演示错误处理"""
    print_section("🛡️ 错误处理演示", "展示引擎的错误处理和恢复能力")
    
    engine = ModernReasoningEngine()
    
    error_cases = [
        {
            "problem": "",
            "description": "空问题"
        },
        {
            "problem": "这完全不是一个数学问题，而是关于天气的讨论",
            "description": "非数学问题"
        },
        {
            "problem": "计算除以零的结果",
            "description": "数学错误"
        },
        {
            "problem": "🚀🌟💎🎉🔥",
            "description": "表情符号问题"
        }
    ]
    
    for case in error_cases:
        print(f"\n🧪 测试 - {case['description']}")
        print(f"问题: '{case['problem']}'")
        print("-" * 40)
        
        try:
            result = engine.reason(case["problem"])
            
            if result.success:
                print("✅ 意外成功处理")
                print(f"答案: {result.result}")
            else:
                print("❌ 预期失败，引擎正确处理")
                if result.error_message:
                    print(f"错误信息: {result.error_message}")
            
            print(f"置信度: {result.confidence:.3f}")
            
        except Exception as e:
            print(f"🚨 捕获异常: {type(e).__name__}: {str(e)}")

def demo_custom_strategy():
    """演示自定义策略"""
    print_section("🔧 自定义策略演示", "展示如何添加和使用自定义推理策略")
    
    from src.reasoning.strategy_manager.strategy_base import (
        ReasoningStrategy, StrategyComplexity, StrategyResult, StrategyType)
    
    class SimpleMultiplicationStrategy(ReasoningStrategy):
        """简单乘法策略示例"""
        
        def __init__(self):
            super().__init__(
                name="simple_multiplication",
                strategy_type=StrategyType.CHAIN_OF_THOUGHT,
                complexity=StrategyComplexity.SIMPLE
            )
        
        def can_handle(self, problem_text: str, context=None) -> bool:
            """只处理包含乘法的简单问题"""
            return any(word in problem_text.lower() for word in ["乘", "×", "*", "倍"])
        
        def estimate_complexity(self, problem_text: str, context=None) -> float:
            """估计为简单复杂度"""
            return 0.2
        
        def _execute_reasoning(self, problem_text: str, context=None) -> StrategyResult:
            """执行简单乘法推理"""
            import re
            import time
            
            start_time = time.time()
            
            # 提取数字
            numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
            numbers = [float(n) for n in numbers]
            
            if len(numbers) >= 2:
                result = numbers[0] * numbers[1]
                
                reasoning_steps = [
                    {
                        "step": 1,
                        "action": "number_extraction",
                        "description": f"提取数字: {numbers[0]}, {numbers[1]}",
                        "confidence": 0.95
                    },
                    {
                        "step": 2,
                        "action": "multiplication",
                        "description": f"计算 {numbers[0]} × {numbers[1]} = {result}",
                        "confidence": 0.98
                    }
                ]
                
                return StrategyResult(
                    success=True,
                    answer=str(result),
                    confidence=0.96,
                    reasoning_steps=reasoning_steps,
                    strategy_used=self.name,
                    execution_time=time.time() - start_time,
                    metadata={"numbers_found": len(numbers)}
                )
            
            return StrategyResult(
                success=False,
                answer="无法处理",
                confidence=0.0,
                reasoning_steps=[],
                strategy_used=self.name,
                execution_time=time.time() - start_time,
                metadata={}
            )
    
    # 创建引擎并添加自定义策略
    engine = ModernReasoningEngine()
    custom_strategy = SimpleMultiplicationStrategy()
    
    print("📝 添加自定义策略...")
    success = engine.add_strategy(custom_strategy)
    print(f"添加结果: {'成功' if success else '失败'}")
    
    print(f"\n可用策略: {engine.get_available_strategies()}")
    
    # 测试自定义策略
    multiplication_problem = "计算 6 × 9"
    print(f"\n🧮 测试乘法问题: {multiplication_problem}")
    
    result = engine.reason(multiplication_problem)
    print_result(result)
    
    if result.metadata.get("strategy_used") == "simple_multiplication":
        print("🎉 成功使用了自定义策略!")
    else:
        print("📌 使用了其他策略")

def main():
    """主函数"""
    print("🤖 推理引擎重构演示")
    print("展示策略模式重构后的现代化推理引擎")
    
    try:
        # 基础功能演示
        demo_basic_functionality()
        
        # 策略对比演示
        demo_strategy_comparison()
        
        # 置信度分析演示
        demo_confidence_analysis()
        
        # 性能监控演示
        demo_performance_monitoring()
        
        # 错误处理演示
        demo_error_handling()
        
        # 自定义策略演示
        demo_custom_strategy()
        
        print_section("🎉 演示完成", """
重构后的推理引擎具有以下优势:

✅ 策略模式 - 可扩展的推理算法架构
✅ 模块化设计 - 清晰的职责分离
✅ 置信度计算 - 多维度的结果可信度评估
✅ 性能监控 - 全面的运行时统计和监控
✅ 错误处理 - 健壮的异常处理和恢复机制
✅ 可测试性 - 每个组件都可以独立测试

这种架构使得推理引擎更加:
• 可维护 - 代码结构清晰，易于修改
• 可扩展 - 容易添加新的推理策略
• 可靠 - 完善的错误处理和监控
• 高效 - 智能的策略选择和优化
        """)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 