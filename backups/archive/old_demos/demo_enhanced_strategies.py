#!/usr/bin/env python3
"""
增强策略库演示
展示新增的解题策略和智能推荐算法的实际应用
"""

import os
import sys

sys.path.append('src')

from reasoning_core.meta_knowledge import MetaKnowledge, MetaKnowledgeReasoning


def demo_strategy_recommendation():
    """演示策略推荐功能"""
    print("=" * 80)
    print("增强策略库演示 - 智能策略推荐")
    print("=" * 80)
    
    meta_knowledge = MetaKnowledge()
    
    # 不同类型的数学问题
    problems = [
        {
            "title": "基础几何问题",
            "problem": "已知一个长方形的面积是24平方厘米，长是6厘米，求宽",
            "concepts": ["面积", "几何"],
            "expected_strategies": ["数形结合", "设未知数", "逆向思维"]
        },
        {
            "title": "绝对值问题",
            "problem": "解不等式|x-3|<5，求x的取值范围",
            "concepts": ["绝对值", "不等式"],
            "expected_strategies": ["数轴法", "分类讨论"]
        },
        {
            "title": "数列问题",
            "problem": "数列1, 3, 6, 10, 15...的第n项是多少？求通项公式",
            "concepts": ["数列", "递推"],
            "expected_strategies": ["递推法", "归纳法"]
        },
        {
            "title": "证明题",
            "problem": "证明不存在最大的质数",
            "concepts": ["质数", "证明"],
            "expected_strategies": ["反证法", "构造法"]
        },
        {
            "title": "最值问题",
            "problem": "求函数f(x)=x²+2x+1的最小值",
            "concepts": ["函数", "最值"],
            "expected_strategies": ["极值法", "配方法"]
        },
        {
            "title": "复杂计算",
            "problem": "计算(1+1/2)(1+1/3)(1+1/4)...(1+1/100)",
            "concepts": ["分数", "计算"],
            "expected_strategies": ["整体思想", "配凑法"]
        }
    ]
    
    for i, problem_info in enumerate(problems, 1):
        print(f"\n{i}. {problem_info['title']}")
        print(f"   问题: {problem_info['problem']}")
        print(f"   涉及概念: {', '.join(problem_info['concepts'])}")
        
        # 识别概念
        identified_concepts = meta_knowledge.identify_concepts_in_text(problem_info['problem'])
        print(f"   识别概念: {identified_concepts}")
        
        # 推荐策略
        strategies = meta_knowledge.suggest_strategies(problem_info['problem'])
        print(f"   推荐策略: {strategies}")
        
        # 带优先级的策略推荐
        strategies_with_priority = meta_knowledge.suggest_strategies_with_priority(problem_info['problem'])
        print("   策略优先级排序:")
        for j, strategy_info in enumerate(strategies_with_priority[:3], 1):
            print(f"     {j}. {strategy_info['strategy']}")
            print(f"        优先级: {strategy_info['priority']:.3f}")
            print(f"        成功率: {strategy_info['success_rate']:.3f}")
            print(f"        难度: {strategy_info['difficulty']}")
            print(f"        描述: {strategy_info['description']}")
        
        # 检查期望策略匹配
        expected = problem_info['expected_strategies']
        matched = [s for s in strategies if s in expected]
        print(f"   匹配期望策略: {matched}/{len(expected)}")
        
        print("-" * 60)


def demo_strategy_application():
    """演示策略应用"""
    print("\n" + "=" * 80)
    print("增强策略库演示 - 策略应用示例")
    print("=" * 80)
    
    meta_knowledge = MetaKnowledge()
    
    # 策略应用示例
    strategy_examples = [
        {
            "strategy": "递推法",
            "problem": "斐波那契数列：1, 1, 2, 3, 5, 8, 13...",
            "steps": [
                "1. 建立递推关系：F(n) = F(n-1) + F(n-2)",
                "2. 确定初始条件：F(1) = 1, F(2) = 1",
                "3. 逐步计算：F(3) = 1+1 = 2, F(4) = 1+2 = 3...",
                "4. 验证结果：检查递推关系是否成立"
            ]
        },
        {
            "strategy": "反证法",
            "problem": "证明不存在最大的质数",
            "steps": [
                "1. 假设结论不成立：存在最大的质数p",
                "2. 构造数N = 2×3×5×...×p + 1",
                "3. 推导矛盾：N不能被任何质数整除",
                "4. 否定假设：不存在最大的质数"
            ]
        },
        {
            "strategy": "极值法",
            "problem": "求函数f(x)=x²+2x+1的最小值",
            "steps": [
                "1. 识别极值条件：二次函数开口向上",
                "2. 建立函数关系：f(x) = (x+1)²",
                "3. 求极值：x = -1时取得最小值",
                "4. 验证结果：f(-1) = 0"
            ]
        },
        {
            "strategy": "配方法",
            "problem": "配方：x²+2x+1",
            "steps": [
                "1. 识别配方形式：x²+2x",
                "2. 进行配方：x²+2x+1 = (x+1)²",
                "3. 简化表达式：得到完全平方形式",
                "4. 验证结果：展开验证"
            ]
        },
        {
            "strategy": "数轴法",
            "problem": "解不等式|x-3|<5",
            "steps": [
                "1. 画数轴：标注关键点3",
                "2. 分析区间：|x-3|<5等价于-5<x-3<5",
                "3. 求解：-2<x<8",
                "4. 确定解集：x∈(-2,8)"
            ]
        }
    ]
    
    for example in strategy_examples:
        print(f"\n策略: {example['strategy']}")
        print(f"问题: {example['problem']}")
        print("解题步骤:")
        for step in example['steps']:
            print(f"  {step}")
        
        # 获取策略信息
        strategy_info = meta_knowledge.get_strategy_info(example['strategy'])
        if strategy_info:
            print(f"策略信息:")
            print(f"  难度: {strategy_info.get('difficulty', '未知')}")
            print(f"  成功率: {strategy_info.get('success_rate', 0.0):.2f}")
            print(f"  适用问题: {', '.join(strategy_info['applicable_problems'])}")
        
        print("-" * 60)


def demo_complexity_analysis():
    """演示复杂度分析"""
    print("\n" + "=" * 80)
    print("增强策略库演示 - 复杂度分析")
    print("=" * 80)
    
    meta_knowledge = MetaKnowledge()
    
    # 不同复杂度的问题
    complexity_problems = [
        {
            "complexity": "简单",
            "problems": [
                "计算1+2+3+4+5",
                "求长方形的面积",
                "解简单方程x+5=10"
            ]
        },
        {
            "complexity": "中等",
            "problems": [
                "小明有100元，花了30%，还剩多少钱？",
                "解不等式2x+3>7",
                "求等差数列的前n项和"
            ]
        },
        {
            "complexity": "困难",
            "problems": [
                "证明不存在最大的质数",
                "求函数f(x)=x³-3x+1的极值",
                "用数学归纳法证明数列求和公式"
            ]
        }
    ]
    
    for complexity_info in complexity_problems:
        print(f"\n{complexity_info['complexity']}问题分析:")
        
        for problem in complexity_info['problems']:
            print(f"\n  问题: {problem}")
            
            # 推荐策略
            strategies = meta_knowledge.suggest_strategies(problem)
            print(f"  推荐策略: {strategies}")
            
            # 策略分布分析
            difficulty_distribution = {"简单": 0, "中等": 0, "困难": 0}
            for strategy in strategies:
                strategy_info = meta_knowledge.get_strategy_info(strategy)
                if strategy_info:
                    difficulty = strategy_info.get("difficulty", "中等")
                    difficulty_distribution[difficulty] += 1
            
            print(f"  策略难度分布: {difficulty_distribution}")
            
            # 平均成功率
            success_rates = []
            for strategy in strategies:
                strategy_info = meta_knowledge.get_strategy_info(strategy)
                if strategy_info:
                    success_rates.append(strategy_info.get("success_rate", 0.0))
            
            if success_rates:
                avg_success_rate = sum(success_rates) / len(success_rates)
                print(f"  平均成功率: {avg_success_rate:.3f}")


def demo_integration_with_reasoning():
    """演示与推理引擎的集成"""
    print("\n" + "=" * 80)
    print("增强策略库演示 - 与推理引擎集成")
    print("=" * 80)
    
    meta_knowledge = MetaKnowledge()
    reasoning_engine = MetaKnowledgeReasoning(meta_knowledge)
    
    # 测试问题
    test_problem = "已知一个长方形的面积是24平方厘米，长是6厘米，求宽"
    
    print(f"测试问题: {test_problem}")
    
    # 基本推理步骤
    current_reasoning = [
        {"step": 1, "action": "分析问题", "content": "这是一个几何问题，已知面积和长，求宽"},
        {"step": 2, "action": "识别概念", "content": "涉及面积概念"},
        {"step": 3, "action": "建立关系", "content": "面积 = 长 × 宽"}
    ]
    
    # 增强推理
    enhanced_reasoning = reasoning_engine.enhance_reasoning(test_problem, current_reasoning)
    
    print("\n增强推理结果:")
    print(f"识别概念: {enhanced_reasoning['concept_analysis']['identified_concepts']}")
    print(f"推荐策略: {enhanced_reasoning['suggested_strategies']}")
    print(f"错误预防: {len(enhanced_reasoning['error_prevention'])}个预防建议")
    print(f"元知识增强: {len(enhanced_reasoning['meta_knowledge_enhancements'])}个增强项")
    
    # 显示概念分析详情
    if enhanced_reasoning['concept_analysis']['identified_concepts']:
        print("\n概念分析详情:")
        for concept in enhanced_reasoning['concept_analysis']['identified_concepts']:
            if concept in enhanced_reasoning['concept_analysis']:
                concept_info = enhanced_reasoning['concept_analysis'][concept]
                print(f"  {concept}: {concept_info['definition']}")
                print(f"    性质: {', '.join(concept_info['properties'])}")
                print(f"    常见错误: {', '.join(concept_info['common_mistakes'])}")
    
    # 显示策略建议详情
    if enhanced_reasoning['suggested_strategies']:
        print("\n策略建议详情:")
        for strategy in enhanced_reasoning['suggested_strategies'][:3]:  # 显示前3个
            strategy_info = meta_knowledge.get_strategy_info(strategy)
            if strategy_info:
                print(f"  {strategy}: {strategy_info['description']}")
                print(f"    步骤: {', '.join(strategy_info['steps'])}")
                print(f"    成功率: {strategy_info.get('success_rate', 0.0):.2f}")
    
    # 显示错误预防建议
    if enhanced_reasoning['error_prevention']:
        print("\n错误预防建议:")
        for prevention in enhanced_reasoning['error_prevention'][:3]:  # 显示前3个
            print(f"  {prevention['concept']}: 注意{prevention['mistake']}")
    
    # 显示相关概念
    related_concepts = []
    for enhancement in enhanced_reasoning['meta_knowledge_enhancements']:
        if enhancement['type'] == 'related_concepts':
            related_concepts.extend(enhancement['related'])
    
    if related_concepts:
        print(f"\n相关概念: {', '.join(list(set(related_concepts)))}")
    
    # 验证解决方案
    solution = "设宽为x，则6x=24，解得x=4"
    calculation_steps = ["6x=24", "x=24÷6", "x=4"]
    
    validation_result = reasoning_engine.validate_solution(test_problem, solution, calculation_steps)
    
    print(f"\n解决方案验证:")
    print(f"验证结果: {validation_result['is_valid']}")
    if validation_result['errors']:
        print(f"错误: {len(validation_result['errors'])}个")
        for error in validation_result['errors'][:2]:  # 显示前2个错误
            print(f"  - {error['type']}: {error['description']}")
    if validation_result['warnings']:
        print(f"警告: {len(validation_result['warnings'])}个")
        for warning in validation_result['warnings'][:2]:  # 显示前2个警告
            print(f"  - {warning}")
    if validation_result['suggestions']:
        print(f"建议: {len(validation_result['suggestions'])}个")
        for suggestion in validation_result['suggestions']:
            print(f"  - {suggestion}")


def demo_strategy_effectiveness():
    """演示策略效果分析"""
    print("\n" + "=" * 80)
    print("增强策略库演示 - 策略效果分析")
    print("=" * 80)
    
    meta_knowledge = MetaKnowledge()
    
    # 策略效果统计
    print("策略库统计信息:")
    print(f"总策略数: {len(meta_knowledge.strategies)}")
    
    # 按难度分类
    difficulty_stats = {"简单": 0, "中等": 0, "困难": 0}
    success_rate_stats = {"简单": [], "中等": [], "困难": []}
    
    for strategy_name, strategy_info in meta_knowledge.strategies.items():
        difficulty = strategy_info.get("difficulty", "中等")
        difficulty_stats[difficulty] += 1
        success_rate = strategy_info.get("success_rate", 0.0)
        success_rate_stats[difficulty].append(success_rate)
    
    print("\n按难度分类:")
    for difficulty, count in difficulty_stats.items():
        avg_success = sum(success_rate_stats[difficulty]) / len(success_rate_stats[difficulty]) if success_rate_stats[difficulty] else 0
        print(f"  {difficulty}: {count}个策略，平均成功率: {avg_success:.3f}")
    
    # 成功率最高的策略
    print("\n成功率最高的策略:")
    sorted_strategies = sorted(meta_knowledge.strategies.items(), 
                              key=lambda x: x[1].get("success_rate", 0.0), reverse=True)
    
    for i, (strategy_name, strategy_info) in enumerate(sorted_strategies[:5], 1):
        success_rate = strategy_info.get("success_rate", 0.0)
        difficulty = strategy_info.get("difficulty", "未知")
        print(f"  {i}. {strategy_name}: {success_rate:.3f} ({difficulty})")
    
    # 最困难的策略
    print("\n最困难的策略:")
    hard_strategies = [(name, info) for name, info in meta_knowledge.strategies.items() 
                      if info.get("difficulty") == "困难"]
    
    for i, (strategy_name, strategy_info) in enumerate(hard_strategies, 1):
        success_rate = strategy_info.get("success_rate", 0.0)
        print(f"  {i}. {strategy_name}: 成功率 {success_rate:.3f}")
        print(f"     描述: {strategy_info['description']}")


if __name__ == "__main__":
    print("增强策略库演示")
    print("本演示展示新增的解题策略和智能推荐算法的实际应用")
    
    try:
        # 演示策略推荐功能
        demo_strategy_recommendation()
        
        # 演示策略应用
        demo_strategy_application()
        
        # 演示复杂度分析
        demo_complexity_analysis()
        
        # 演示与推理引擎的集成
        demo_integration_with_reasoning()
        
        # 演示策略效果分析
        demo_strategy_effectiveness()
        
        print("\n" + "=" * 80)
        print("增强策略库演示完成！")
        print("主要改进:")
        print("- 策略数量从8种扩展到18种")
        print("- 新增高级策略：递推法、反证法、构造法、极值法、归纳法等")
        print("- 智能策略推荐算法，基于关键词、概念、复杂度")
        print("- 策略优先级评分系统")
        print("- 与推理引擎的深度集成")
        print("=" * 80)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 