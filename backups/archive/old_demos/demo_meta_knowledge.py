#!/usr/bin/env python3
"""
元知识功能演示
展示如何利用元知识增强数学推理能力
"""

import os
import sys

sys.path.append('src')

from reasoning_core.meta_knowledge import MetaKnowledge, MetaKnowledgeReasoning
from reasoning_core.reasoning_engine import ReasoningEngine


def demo_meta_knowledge_basic():
    """演示元知识基本功能"""
    print("=" * 60)
    print("元知识基本功能演示")
    print("=" * 60)
    
    # 初始化元知识
    meta_knowledge = MetaKnowledge()
    
    # 1. 概念识别
    test_texts = [
        "一个长方形的面积是24平方厘米，长是6厘米，求宽",
        "商品打8折后价格是80元，求原价",
        "小明有100元，花了30%，还剩多少钱"
    ]
    
    print("\n1. 概念识别:")
    for text in test_texts:
        concepts = meta_knowledge.identify_concepts_in_text(text)
        print(f"文本: {text}")
        print(f"识别概念: {concepts}")
        
        for concept in concepts:
            concept_info = meta_knowledge.get_concept_info(concept)
            if concept_info:
                print(f"  - {concept}: {concept_info['definition']}")
        print()
    
    # 2. 策略推荐
    print("\n2. 解题策略推荐:")
    for text in test_texts:
        strategies = meta_knowledge.suggest_strategies(text)
        print(f"文本: {text}")
        print(f"推荐策略: {strategies}")
        
        for strategy in strategies:
            strategy_info = meta_knowledge.get_strategy_info(strategy)
            if strategy_info:
                print(f"  - {strategy}: {strategy_info['description']}")
        print()
    
    # 3. 单位转换
    print("\n3. 单位转换:")
    conversions = [
        (5, "米", "厘米", "长度"),
        (1000, "克", "千克", "重量"),
        (2, "小时", "分钟", "时间")
    ]
    
    for value, from_unit, to_unit, unit_type in conversions:
        result = meta_knowledge.convert_units(value, from_unit, to_unit, unit_type)
        print(f"{value}{from_unit} = {result}{to_unit}")
    
    # 4. 表达式验证
    print("\n4. 数学表达式验证:")
    expressions = [
        "2 + 3 = 5",
        "10 / 0 = 0",
        "5 * (3 + 2",
        "100 - 50 = 50"
    ]
    
    for expr in expressions:
        validation = meta_knowledge.validate_mathematical_expression(expr)
        print(f"表达式: {expr}")
        print(f"  有效: {validation['is_valid']}")
        if validation['errors']:
            print(f"  错误: {validation['errors']}")
        if validation['warnings']:
            print(f"  警告: {validation['warnings']}")
        print()


def demo_meta_knowledge_reasoning():
    """演示元知识推理增强"""
    print("\n" + "=" * 60)
    print("元知识推理增强演示")
    print("=" * 60)
    
    # 初始化推理增强器
    meta_knowledge = MetaKnowledge()
    meta_reasoning = MetaKnowledgeReasoning(meta_knowledge)
    
    # 测试问题
    test_problems = [
        {
            "text": "一个长方形的面积是24平方厘米，长是6厘米，求宽",
            "reasoning_steps": [
                {"step": 1, "action": "extract_info", "description": "提取已知信息"}
            ]
        },
        {
            "text": "商品打8折后价格是80元，求原价",
            "reasoning_steps": [
                {"step": 1, "action": "extract_info", "description": "提取已知信息"}
            ]
        }
    ]
    
    for problem in test_problems:
        print(f"\n问题: {problem['text']}")
        
        # 增强推理
        enhanced = meta_reasoning.enhance_reasoning(
            problem['text'], 
            problem['reasoning_steps']
        )
        
        print("增强结果:")
        print(f"  识别概念: {enhanced['concept_analysis']['identified_concepts']}")
        print(f"  推荐策略: {enhanced['suggested_strategies']}")
        
        if enhanced['meta_knowledge_enhancements']:
            print("  元知识增强:")
            for enhancement in enhanced['meta_knowledge_enhancements']:
                print(f"    - {enhancement['type']}: {enhancement.get('strategy', enhancement.get('concept', ''))}")
        
        if enhanced['error_prevention']:
            print("  错误预防:")
            for error in enhanced['error_prevention']:
                print(f"    - {error['concept']}: {error['mistake']}")


def demo_integrated_reasoning():
    """演示集成元知识的推理引擎"""
    print("\n" + "=" * 60)
    print("集成元知识的推理引擎演示")
    print("=" * 60)
    
    # 初始化推理引擎
    engine = ReasoningEngine()
    
    # 测试样本
    test_samples = [
        {
            "problem": "一个长方形的面积是24平方厘米，长是6厘米，求宽",
            "cleaned_text": "一个长方形的面积是24平方厘米，长是6厘米，求宽"
        },
        {
            "problem": "商品打8折后价格是80元，求原价",
            "cleaned_text": "商品打8折后价格是80元，求原价"
        },
        {
            "problem": "小明有100元，花了30%，还剩多少钱",
            "cleaned_text": "小明有100元，花了30%，还剩多少钱"
        }
    ]
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{i}. 问题: {sample['problem']}")
        
        # 求解
        result = engine.solve(sample)
        
        print(f"   最终答案: {result['final_answer']}")
        print(f"   使用策略: {result['strategy_used']}")
        print(f"   置信度: {result['confidence']:.2f}")
        
        # 显示元知识增强信息
        if 'meta_knowledge_enhancement' in result:
            meta = result['meta_knowledge_enhancement']
            if meta['concept_analysis']['identified_concepts']:
                print(f"   识别概念: {meta['concept_analysis']['identified_concepts']}")
            if meta['suggested_strategies']:
                print(f"   推荐策略: {meta['suggested_strategies']}")
        
        # 显示解决方案验证
        if 'solution_validation' in result:
            validation = result['solution_validation']
            print(f"   解决方案验证: {'有效' if validation['is_valid'] else '无效'}")
            if validation['warnings']:
                print(f"   警告: {validation['warnings']}")
            if validation['suggestions']:
                print(f"   建议: {validation['suggestions']}")


def demo_concept_relationships():
    """演示概念关系"""
    print("\n" + "=" * 60)
    print("概念关系演示")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    concepts = ["分数", "百分比", "面积", "体积", "速度"]
    
    for concept in concepts:
        print(f"\n{concept}:")
        concept_info = meta_knowledge.get_concept_info(concept)
        if concept_info:
            print(f"  定义: {concept_info['definition']}")
            print(f"  性质: {', '.join(concept_info['properties'])}")
            print(f"  常见错误: {', '.join(concept_info['common_mistakes'])}")
        
        # 查找相关概念
        related = meta_knowledge.get_related_concepts(concept)
        if related:
            print(f"  相关概念: {', '.join(related)}")


if __name__ == "__main__":
    print("元知识功能演示")
    print("本演示展示如何利用元知识增强数学推理能力")
    
    try:
        # 基本功能演示
        demo_meta_knowledge_basic()
        
        # 推理增强演示
        demo_meta_knowledge_reasoning()
        
        # 集成推理演示
        demo_integrated_reasoning()
        
        # 概念关系演示
        demo_concept_relationships()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("元知识为数学推理提供了丰富的背景知识，包括:")
        print("- 数学概念定义和性质")
        print("- 解题策略推荐")
        print("- 常见错误预防")
        print("- 单位转换支持")
        print("- 表达式验证")
        print("- 概念关系分析")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 