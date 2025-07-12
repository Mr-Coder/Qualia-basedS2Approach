#!/usr/bin/env python3
"""
增强策略库功能测试
测试新增的解题策略和智能推荐算法
"""

import os
import sys

sys.path.append('src')

from reasoning_core.meta_knowledge import MetaKnowledge


def test_enhanced_strategies():
    """测试增强的策略库"""
    print("=" * 60)
    print("测试增强的策略库")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    # 测试所有策略
    print(f"策略总数: {len(meta_knowledge.strategies)}")
    print("所有策略列表:")
    for i, (strategy_name, strategy_info) in enumerate(meta_knowledge.strategies.items(), 1):
        print(f"{i:2d}. {strategy_name}")
        print(f"    描述: {strategy_info['description']}")
        print(f"    难度: {strategy_info.get('difficulty', '未知')}")
        print(f"    成功率: {strategy_info.get('success_rate', 0.0):.2f}")
        print(f"    适用问题: {', '.join(strategy_info['applicable_problems'])}")
        print()


def test_strategy_recommendation():
    """测试策略推荐功能"""
    print("=" * 60)
    print("测试策略推荐功能")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    test_cases = [
        {
            "problem": "已知一个长方形的面积是24平方厘米，长是6厘米，求宽",
            "expected_strategies": ["逆向思维", "数形结合", "设未知数"]
        },
        {
            "problem": "如果x>0，求|x|的值；如果x<0，求|x|的值",
            "expected_strategies": ["分类讨论", "数轴法"]
        },
        {
            "problem": "证明不存在最大的质数",
            "expected_strategies": ["反证法", "构造法"]
        },
        {
            "problem": "求函数f(x)=x²+2x+1的最小值",
            "expected_strategies": ["极值法", "配方法"]
        },
        {
            "problem": "数列1, 3, 6, 10, 15...的第n项是多少？",
            "expected_strategies": ["递推法", "归纳法"]
        },
        {
            "problem": "解不等式|x-3|<5",
            "expected_strategies": ["数轴法", "分类讨论"]
        },
        {
            "problem": "用换元法求积分∫(x+1)²dx",
            "expected_strategies": ["换元法", "配方法"]
        },
        {
            "problem": "因式分解x³-8",
            "expected_strategies": ["因式分解", "配凑法"]
        },
        {
            "problem": "列出所有可能的排列组合情况",
            "expected_strategies": ["列表法", "分类讨论"]
        },
        {
            "problem": "假设答案正确，验证一下",
            "expected_strategies": ["假设法", "验证"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 问题: {test_case['problem']}")
        
        # 基本策略推荐
        strategies = meta_knowledge.suggest_strategies(test_case['problem'])
        print(f"   推荐策略: {strategies}")
        
        # 带优先级的策略推荐
        strategies_with_priority = meta_knowledge.suggest_strategies_with_priority(test_case['problem'])
        print("   策略优先级:")
        for strategy_info in strategies_with_priority[:3]:  # 显示前3个
            print(f"     - {strategy_info['strategy']}: 优先级={strategy_info['priority']:.2f}, "
                  f"成功率={strategy_info['success_rate']:.2f}, 难度={strategy_info['difficulty']}")
        
        # 检查期望策略
        expected = test_case['expected_strategies']
        matched = [s for s in strategies if s in expected]
        print(f"   匹配期望策略: {matched}/{len(expected)}")


def test_strategy_priority():
    """测试策略优先级计算"""
    print("\n" + "=" * 60)
    print("测试策略优先级计算")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    test_problems = [
        "简单的加法问题：1+2=?",
        "复杂的证明题：证明不存在最大的质数",
        "中等难度的应用题：小明有100元，花了30%，还剩多少钱？",
        "困难的数列问题：求斐波那契数列的第n项",
        "基础的几何问题：求长方形的面积"
    ]
    
    for problem in test_problems:
        print(f"\n问题: {problem}")
        
        strategies_with_priority = meta_knowledge.suggest_strategies_with_priority(problem)
        
        print("策略优先级排序:")
        for i, strategy_info in enumerate(strategies_with_priority[:5], 1):
            print(f"  {i}. {strategy_info['strategy']}")
            print(f"     优先级: {strategy_info['priority']:.3f}")
            print(f"     成功率: {strategy_info['success_rate']:.3f}")
            print(f"     难度: {strategy_info['difficulty']}")
            print(f"     描述: {strategy_info['description']}")


def test_strategy_details():
    """测试策略详细信息"""
    print("\n" + "=" * 60)
    print("测试策略详细信息")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    # 测试几个重要策略的详细信息
    important_strategies = ["递推法", "反证法", "构造法", "极值法", "归纳法"]
    
    for strategy_name in important_strategies:
        strategy_info = meta_knowledge.get_strategy_info(strategy_name)
        if strategy_info:
            print(f"\n{strategy_name}:")
            print(f"  描述: {strategy_info['description']}")
            print(f"  适用问题: {', '.join(strategy_info['applicable_problems'])}")
            print(f"  解题步骤: {', '.join(strategy_info['steps'])}")
            print(f"  示例: {', '.join(strategy_info['examples'])}")
            print(f"  难度: {strategy_info.get('difficulty', '未知')}")
            print(f"  成功率: {strategy_info.get('success_rate', 0.0):.2f}")


def test_complexity_based_recommendation():
    """测试基于复杂度的策略推荐"""
    print("\n" + "=" * 60)
    print("测试基于复杂度的策略推荐")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    complexity_test_cases = [
        {
            "complexity": "简单",
            "problem": "简单的加法计算：1+2+3=?",
            "expected": ["列表法", "整体思想"]
        },
        {
            "complexity": "中等",
            "problem": "中等难度的应用题：小明有100元，花了30%，还剩多少钱？",
            "expected": ["设未知数", "等量代换", "整体思想"]
        },
        {
            "complexity": "困难",
            "problem": "复杂的证明题：证明不存在最大的质数，需要详细的数学推导和逻辑推理",
            "expected": ["反证法", "构造法", "归纳法"]
        }
    ]
    
    for test_case in complexity_test_cases:
        print(f"\n复杂度: {test_case['complexity']}")
        print(f"问题: {test_case['problem']}")
        
        strategies = meta_knowledge.suggest_strategies(test_case['problem'])
        print(f"推荐策略: {strategies}")
        
        # 检查是否包含期望的简单策略
        expected = test_case['expected']
        matched = [s for s in strategies if s in expected]
        print(f"匹配期望策略: {matched}/{len(expected)}")


def test_concept_strategy_mapping():
    """测试概念与策略的映射关系"""
    print("\n" + "=" * 60)
    print("测试概念与策略的映射关系")
    print("=" * 60)
    
    meta_knowledge = MetaKnowledge()
    
    concept_test_cases = [
        "比例问题：甲比乙多20%，求比例",
        "分数问题：1/3 + 1/6 = ?",
        "方程问题：设未知数为x，建立方程求解",
        "面积问题：求长方形的面积",
        "速度问题：汽车速度是60千米每小时",
        "折扣问题：商品打8折",
        "利润问题：求利润率",
        "平均数问题：求平均值"
    ]
    
    for problem in concept_test_cases:
        print(f"\n问题: {problem}")
        
        # 识别概念
        concepts = meta_knowledge.identify_concepts_in_text(problem)
        print(f"识别概念: {concepts}")
        
        # 推荐策略
        strategies = meta_knowledge.suggest_strategies(problem)
        print(f"推荐策略: {strategies}")
        
        # 带优先级的策略
        strategies_with_priority = meta_knowledge.suggest_strategies_with_priority(problem)
        if strategies_with_priority:
            best_strategy = strategies_with_priority[0]
            print(f"最佳策略: {best_strategy['strategy']} (优先级: {best_strategy['priority']:.3f})")


if __name__ == "__main__":
    print("增强策略库功能测试")
    print("本测试验证新增的解题策略和智能推荐算法")
    
    try:
        # 测试增强的策略库
        test_enhanced_strategies()
        
        # 测试策略推荐功能
        test_strategy_recommendation()
        
        # 测试策略优先级
        test_strategy_priority()
        
        # 测试策略详细信息
        test_strategy_details()
        
        # 测试基于复杂度的推荐
        test_complexity_based_recommendation()
        
        # 测试概念策略映射
        test_concept_strategy_mapping()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("增强的策略库功能包括:")
        print("- 18种解题策略（从8种扩展到18种）")
        print("- 智能策略推荐算法")
        print("- 策略优先级评分")
        print("- 基于复杂度的策略选择")
        print("- 概念与策略的智能映射")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 