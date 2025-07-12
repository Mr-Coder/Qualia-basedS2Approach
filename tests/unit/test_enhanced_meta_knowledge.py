#!/usr/bin/env python3
"""
增强元知识功能测试
测试新增的数学概念、解题策略和智能识别功能
"""

import os
import sys

sys.path.append('src')

from reasoning_core.meta_knowledge import MetaKnowledge, MetaKnowledgeReasoning
from reasoning_core.reasoning_engine import ReasoningEngine


def test_enhanced_concept_recognition():
    """测试增强的概念识别功能"""
    print("=" * 60)
    print("测试增强的概念识别功能")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    test_cases = [
        ("商品打8折后价格是80元，求原价", ["折扣"]),
        ("小明有100元，花了30%，还剩多少钱", ["百分比"]),
        ("一个长方形的面积是24平方厘米，长是6厘米，求宽", ["面积"]),
        ("商店进价50元，售价80元，求利润率", ["利润"]),
        ("三个数的平均数是15，求总和", ["平均数"]),
        ("甲比乙多20%，求比例", ["比例", "百分比"]),
        ("设未知数为x，建立方程求解", ["方程"]),
        ("长方体体积是120立方厘米，长5厘米，宽4厘米，求高", ["体积", "面积"]),
        ("汽车速度是60千米每小时，行驶2小时，求距离", ["速度"]),
        ("1/3 + 1/6 = ?", ["分数"])
    ]
    
    for text, expected_concepts in test_cases:
        identified = meta_knowledge.identify_concepts_in_text(text)
        print(f"文本: {text}")
        print(f"期望概念: {expected_concepts}")
        print(f"识别概念: {identified}")
        print(f"匹配: {set(identified) == set(expected_concepts)}")
        print()


def test_enhanced_strategy_suggestion():
    """测试增强的策略推荐功能"""
    print("=" * 60)
    print("测试增强的策略推荐功能")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    test_cases = [
        ("商品打8折后价格是80元，求原价", ["逆向思维"]),
        ("如果x>0，求|x|的值", ["分类讨论"]),
        ("画图分析这个几何问题", ["数形结合"]),
        ("用等量代换简化表达式", ["等量代换"]),
        ("求未知数x的值", ["设未知数"]),
        ("列出所有可能的情况", ["列表法"]),
        ("假设答案正确，验证一下", ["假设法"]),
        ("从整体角度考虑这个问题", ["整体思想"]),
        ("三个数的平均数是15，求总和", ["逆向思维", "整体思想"]),
        ("甲比乙多20%，求比例", ["等量代换", "整体思想"])
    ]
    
    for text, expected_strategies in test_cases:
        suggested = meta_knowledge.suggest_strategies(text)
        print(f"文本: {text}")
        print(f"期望策略: {expected_strategies}")
        print(f"推荐策略: {suggested}")
        print(f"匹配: {set(suggested) >= set(expected_strategies)}")
        print()


def test_meta_knowledge_integration():
    """测试元知识集成功能"""
    print("=" * 60)
    print("测试元知识集成功能")
    print("=" * 60)
    
    engine = ReasoningEngine()
    
    test_samples = [
        {
            "problem": "商品打8折后价格是80元，求原价",
            "cleaned_text": "商品打8折后价格是80元，求原价"
        },
        {
            "problem": "商店进价50元，售价80元，求利润率",
            "cleaned_text": "商店进价50元，售价80元，求利润率"
        },
        {
            "problem": "三个数的平均数是15，求总和",
            "cleaned_text": "三个数的平均数是15，求总和"
        },
        {
            "problem": "甲比乙多20%，求比例",
            "cleaned_text": "甲比乙多20%，求比例"
        }
    ]
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{i}. 问题: {sample['problem']}")
        
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
            
            # 显示概念详细信息
            for concept in meta['concept_analysis']['identified_concepts']:
                if concept in meta['concept_analysis']:
                    concept_info = meta['concept_analysis'][concept]
                    print(f"   {concept}定义: {concept_info['definition']}")
                    print(f"   {concept}性质: {', '.join(concept_info['properties'][:2])}...")
        
        # 显示解决方案验证
        if 'solution_validation' in result:
            validation = result['solution_validation']
            print(f"   解决方案验证: {'有效' if validation['is_valid'] else '无效'}")
            if validation['warnings']:
                print(f"   警告: {validation['warnings']}")
            if validation['suggestions']:
                print(f"   建议: {validation['suggestions']}")


def test_concept_relationships():
    """测试概念关系功能"""
    print("=" * 60)
    print("测试概念关系功能")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    concepts = ["分数", "百分比", "面积", "体积", "速度", "折扣", "利润", "平均数", "比例", "方程"]
    
    for concept in concepts:
        print(f"\n{concept}:")
        concept_info = meta_knowledge.get_concept_info(concept)
        if concept_info:
            print(f"  定义: {concept_info['definition']}")
            print(f"  性质: {', '.join(concept_info['properties'])}")
            print(f"  运算: {', '.join(concept_info['operations'][:2])}...")
            print(f"  常见错误: {', '.join(concept_info['common_mistakes'])}")
        
        # 查找相关概念
        related = meta_knowledge.get_related_concepts(concept)
        if related:
            print(f"  相关概念: {', '.join(related)}")


def test_unit_conversions():
    """测试单位转换功能"""
    print("=" * 60)
    print("测试单位转换功能")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    test_conversions = [
        (5, "米", "厘米", "长度"),
        (1000, "克", "千克", "重量"),
        (2, "小时", "分钟", "时间"),
        (10, "平方米", "平方厘米", "面积"),
        (1, "立方米", "立方厘米", "体积")
    ]
    
    for value, from_unit, to_unit, unit_type in test_conversions:
        result = meta_knowledge.convert_units(value, from_unit, to_unit, unit_type)
        print(f"{value}{from_unit} = {result}{to_unit}")


def test_expression_validation():
    """测试表达式验证功能"""
    print("=" * 60)
    print("测试表达式验证功能")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    test_expressions = [
        "2 + 3 = 5",
        "10 / 0 = 0",
        "5 * (3 + 2",
        "100 - 50 = 50",
        "x + y = z",
        "2 * 3 + 4 = 10",
        "1/2 + 1/3 = 5/6",
        "a * b + c * d"
    ]
    
    for expr in test_expressions:
        validation = meta_knowledge.validate_mathematical_expression(expr)
        print(f"表达式: {expr}")
        print(f"  有效: {validation['is_valid']}")
        if validation['errors']:
            print(f"  错误: {validation['errors']}")
        if validation['warnings']:
            print(f"  警告: {validation['warnings']}")
        print()


if __name__ == "__main__":
    print("增强元知识功能测试")
    print("本测试验证新增的数学概念、解题策略和智能识别功能")
    
    try:
        # 测试增强的概念识别
        test_enhanced_concept_recognition()
        
        # 测试增强的策略推荐
        test_enhanced_strategy_suggestion()
        
        # 测试元知识集成
        test_meta_knowledge_integration()
        
        # 测试概念关系
        test_concept_relationships()
        
        # 测试单位转换
        test_unit_conversions()
        
        # 测试表达式验证
        test_expression_validation()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("增强的元知识功能包括:")
        print("- 10个数学概念的定义、性质和常见错误")
        print("- 8种解题策略的详细描述和适用场景")
        print("- 智能概念识别和策略推荐")
        print("- 概念关系分析和单位转换")
        print("- 数学表达式验证和错误检查")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 