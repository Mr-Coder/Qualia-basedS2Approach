#!/usr/bin/env python3
"""
Enhanced Integration Demo
Demonstrating the fusion of Intelligent Tutor with COT-DIR Method
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.base_model import ModelInput
from src.models.hybrid_model import HybridModel
from src.models.intelligent_tutor import (IntelligentTutor, ProblemContext,
                                          StudentState)
from src.models.proposed_model import COTDIRModel


def demo_integration():
    """Demonstrate the integration of Intelligent Tutor with COT-DIR"""
    
    print("🎓 智能辅导系统与COT-DIR方法融合演示")
    print("=" * 80)
    print("🔗 融合架构：")
    print("   • 智能辅导系统：责任链 + 状态机 + 策略组合 + 观察者")
    print("   • COT-DIR方法：思维链 + 定向隐含推理")
    print("   • 混合求解器：模式匹配 + LLM回退")
    print("=" * 80)
    
    # Initialize components
    print("\n📋 初始化组件...")
    
    # 1. Intelligent Tutor
    intelligent_tutor = IntelligentTutor()
    print("   ✅ 智能辅导系统已初始化")
    
    # 2. COT-DIR Model
    cotdir_config = {
        "enable_ird": True,
        "enable_mlr": True,
        "enable_cv": True
    }
    cotdir_model = COTDIRModel(cotdir_config)
    cotdir_success = cotdir_model.initialize()
    print(f"   {'✅' if cotdir_success else '❌'} COT-DIR模型{'已初始化' if cotdir_success else '初始化失败'}")
    
    # 3. Hybrid Model
    hybrid_model = HybridModel("demo_hybrid")
    hybrid_success = hybrid_model.initialize()
    print(f"   {'✅' if hybrid_success else '❌'} 混合模型{'已初始化' if hybrid_success else '初始化失败'}")
    
    # Test problems
    test_problems = [
        {
            "text": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
            "type": "simple_addition",
            "expected": "8",
            "description": "简单加法问题 - 适合智能辅导"
        },
        {
            "text": "一个复杂的数学问题，涉及多个变量和关系，需要深入推理和分析。",
            "type": "complex_reasoning", 
            "expected": "unknown",
            "description": "复杂推理问题 - 适合COT-DIR"
        },
        {
            "text": "小华有10个糖果，他给了小明3个，还剩多少个？",
            "type": "subtraction",
            "expected": "7", 
            "description": "减法问题 - 适合混合方法"
        }
    ]
    
    student_id = "integration_demo_student"
    
    print(f"\n🧪 开始融合测试...")
    print("-" * 80)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n📝 问题 {i}: {problem['description']}")
        print(f"   题目: {problem['text']}")
        print(f"   类型: {problem['type']}")
        
        # Test different approaches
        print(f"\n   🔄 测试不同方法:")
        
        # 1. Intelligent Tutor approach
        if intelligent_tutor:
            print(f"   1️⃣ 智能辅导系统:")
            problem_context = ProblemContext(
                problem_text=problem['text'],
                problem_id=f"demo_{i}",
                difficulty_level=1,
                concept_tags=["addition"] if "add" in problem['text'].lower() else ["subtraction"],
                expected_answer=problem['expected']
            )
            
            tutor_response = intelligent_tutor.solve_problem(student_id, problem_context)
            print(f"      响应类型: {tutor_response.response_type}")
            print(f"      置信度: {tutor_response.confidence_level:.2f}")
            print(f"      消息: {tutor_response.message[:60]}...")
        
        # 2. COT-DIR approach
        if cotdir_success:
            print(f"   2️⃣ COT-DIR方法:")
            model_input = ModelInput(
                problem_text=problem['text'],
                problem_id=f"demo_{i}"
            )
            
            cotdir_result = cotdir_model.solve_problem(model_input)
            print(f"      答案: {cotdir_result.answer}")
            print(f"      置信度: {cotdir_result.confidence_score:.2f}")
            print(f"      推理步骤: {len(cotdir_result.reasoning_chain)}")
            print(f"      复杂度: {cotdir_result.metadata.get('complexity', 'unknown')}")
        
        # 3. Hybrid approach
        if hybrid_success:
            print(f"   3️⃣ 混合方法:")
            model_input = ModelInput(
                problem_text=problem['text'],
                problem_id=f"demo_{i}"
            )
            
            hybrid_result = hybrid_model.solve_problem(model_input)
            print(f"      答案: {hybrid_result.answer}")
            print(f"      置信度: {hybrid_result.confidence_score:.2f}")
            print(f"      求解器类型: {hybrid_result.metadata.get('solver_type', 'unknown')}")
            print(f"      LLM回退: {hybrid_result.metadata.get('llm_fallback_used', False)}")
    
    # Show integration benefits
    print(f"\n🎯 融合优势分析:")
    print("-" * 80)
    
    benefits = [
        "✅ 智能选择：根据问题复杂度和学生状态自动选择最佳方法",
        "✅ 渐进式辅导：从简单提示到深度推理的完整学习路径", 
        "✅ 关系发现：COT-DIR的隐含关系发现能力",
        "✅ 状态管理：智能辅导系统的学习状态跟踪",
        "✅ 实时反馈：观察者模式的实时学习反馈",
        "✅ 策略组合：灵活的教学策略组合",
        "✅ 性能优化：混合方法的效率平衡",
        "✅ 可扩展性：易于添加新的求解方法和教学策略"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    # Show architecture diagram
    print(f"\n🏗️ 融合架构图:")
    print("-" * 80)
    print("""
    学生问题输入
         ↓
    ┌─────────────────────────────────────┐
    │        智能方法选择器                │
    │  (基于问题复杂度 + 学生状态)        │
    └─────────────────────────────────────┘
         ↓
    ┌─────────────┬─────────────┬─────────────┐
    │  智能辅导   │  COT-DIR    │   混合方法   │
    │  系统       │  方法       │             │
    │             │             │             │
    │ • 责任链    │ • 思维链    │ • 模式匹配  │
    │ • 状态机    │ • 隐含推理  │ • LLM回退   │
    │ • 策略组合  │ • 关系发现  │ • 置信度    │
    │ • 观察者    │ • 验证      │ • 自适应    │
    └─────────────┴─────────────┴─────────────┘
         ↓
    ┌─────────────────────────────────────┐
    │        统一响应整合器                │
    │  (组合所有方法的优势)               │
    └─────────────────────────────────────┘
         ↓
    个性化学习响应
    """)
    
    print("\n✅ 融合演示完成！")
    print("\n💡 核心价值：")
    print("   • 智能辅导系统提供个性化学习体验")
    print("   • COT-DIR方法提供深度推理能力") 
    print("   • 混合方法提供效率和准确性的平衡")
    print("   • 三者融合创造最佳的学习效果")


if __name__ == "__main__":
    demo_integration() 