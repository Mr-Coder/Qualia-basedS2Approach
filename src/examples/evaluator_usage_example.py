"""
评估器使用示例
==============

演示如何使用各种评估器类来评估数学问题求解系统的性能。

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators import (PerformanceEvaluator, ReasoningChainEvaluator,
                        RelationDiscoveryEvaluator)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def performance_evaluator_example():
    """性能评估器使用示例"""
    print("=" * 60)
    print("性能评估器使用示例")
    print("=" * 60)
    
    # 创建评估器
    evaluator = PerformanceEvaluator()
    
    # 示例数据
    predictions = [15, 23, 8, 12, 45, 33, 7, 19, 25, 11]
    ground_truth = [15, 25, 8, 10, 45, 33, 9, 19, 25, 11]
    complexity_labels = ["L0", "L1", "L0", "L2", "L1", "L3", "L0", "L2", "L1", "L0"]
    
    print(f"预测结果: {predictions}")
    print(f"真实答案: {ground_truth}")
    print(f"复杂度标签: {complexity_labels}")
    print()
    
    # 评估整体准确率
    overall_acc = evaluator.evaluate_overall_accuracy(predictions, ground_truth)
    print(f"整体准确率: {overall_acc:.4f}")
    
    # 按复杂度级别评估
    level_results = evaluator.evaluate_by_complexity_level(predictions, ground_truth, complexity_labels)
    print("\n按复杂度级别的准确率:")
    for level, acc in level_results.items():
        print(f"  {level}: {acc:.4f}")
    
    # 计算鲁棒性分数
    robustness = evaluator.calculate_robustness_score(level_results)
    print(f"\n鲁棒性分数: {robustness:.4f}")
    
    # 计算详细指标
    detailed_metrics = evaluator.calculate_detailed_metrics(predictions, ground_truth, complexity_labels)
    print("\n详细性能指标:")
    for metric, value in detailed_metrics.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for sub_metric, sub_value in value.items():
                print(f"    {sub_metric}: {sub_value}")
        else:
            print(f"  {metric}: {value}")
    
    # 生成性能报告
    report = evaluator.generate_performance_report(detailed_metrics)
    print("\n" + "="*60)
    print("性能评估报告:")
    print("="*60)
    print(report)


def relation_discovery_evaluator_example():
    """关系发现评估器使用示例"""
    print("\n" + "=" * 60)
    print("关系发现评估器使用示例")
    print("=" * 60)
    
    # 创建评估器
    evaluator = RelationDiscoveryEvaluator()
    
    # 示例数据：发现的关系
    discovered_relations = [
        {"type": "mathematical_operations", "match": "addition", "position": (5, 15)},
        {"type": "unit_conversions", "match": "meters to kilometers", "position": (20, 35)},
        {"type": "proportional_relations", "match": "speed and time", "position": (40, 55)},
        {"type": "temporal_relations", "match": "before and after", "position": (60, 75)}
    ]
    
    # 真实关系
    true_relations = [
        {"type": "mathematical_operations", "match": "addition", "position": (5, 15)},
        {"type": "unit_conversions", "match": "meters to kilometers", "position": (20, 35)},
        {"type": "physical_constraints", "match": "gravity constraint", "position": (30, 45)},
        {"type": "proportional_relations", "match": "speed and time", "position": (40, 55)}
    ]
    
    print("发现的关系:")
    for i, rel in enumerate(discovered_relations):
        print(f"  {i+1}. {rel['type']}: {rel['match']}")
    
    print("\n真实关系:")
    for i, rel in enumerate(true_relations):
        print(f"  {i+1}. {rel['type']}: {rel['match']}")
    
    # 评估关系发现质量
    evaluation_result = evaluator.evaluate_relation_discovery(discovered_relations, true_relations)
    print(f"\n关系发现评估结果:")
    for metric, value in evaluation_result.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    # 按关系类型评估
    type_results = evaluator.evaluate_by_relation_type(discovered_relations, true_relations)
    print(f"\n按关系类型的评估结果:")
    for rel_type, result in type_results.items():
        print(f"  {rel_type}:")
        print(f"    精确率: {result['precision']:.3f}")
        print(f"    召回率: {result['recall']:.3f}")
        print(f"    F1分数: {result['f1']:.3f}")
    
    # 计算覆盖度指标
    coverage_metrics = evaluator.calculate_coverage_metrics(discovered_relations, true_relations)
    print(f"\n覆盖度指标:")
    for metric, value in coverage_metrics.items():
        print(f"  {metric}: {value:.4f}")


def reasoning_chain_evaluator_example():
    """推理链评估器使用示例"""
    print("\n" + "=" * 60)
    print("推理链评估器使用示例")
    print("=" * 60)
    
    # 创建评估器
    evaluator = ReasoningChainEvaluator()
    
    # 示例推理链
    reasoning_chain = [
        {
            "type": "analysis",
            "content": "理解题目：小明有15个苹果，给了小红3个，问还剩多少个？",
            "input": "15个苹果，给出3个",
            "operation": "problem_analysis",
            "output": "需要计算减法"
        },
        {
            "type": "method",
            "content": "选择解题方法：使用减法运算",
            "input": "减法问题",
            "operation": "method_selection",
            "output": "减法运算"
        },
        {
            "type": "calculation",
            "content": "执行计算：15 - 3 = 12",
            "input": "15, 3",
            "operation": "subtraction",
            "output": "12"
        },
        {
            "type": "verification",
            "content": "验证结果：12 + 3 = 15，正确",
            "input": "12, 3",
            "operation": "addition_check",
            "output": "验证通过"
        },
        {
            "type": "conclusion",
            "content": "答案：小明还剩12个苹果",
            "input": "12",
            "operation": "final_answer",
            "output": "12个苹果"
        }
    ]
    
    print("推理链步骤:")
    for i, step in enumerate(reasoning_chain):
        print(f"  步骤{i+1} ({step['type']}): {step['content']}")
    
    # 评估推理链质量
    quality_scores = evaluator.evaluate_reasoning_chain_quality(reasoning_chain)
    print(f"\n推理链质量评估:")
    for dimension, score in quality_scores.items():
        print(f"  {dimension}: {score:.4f}")
    
    # 示例失败的推理链（用于错误传播分析）
    failed_chains = [
        [
            {"type": "analysis", "content": "错误理解题目", "has_error": True},
            {"type": "calculation", "content": "15 + 3 = 18", "has_error": True},
            {"type": "answer", "content": "答案是18"}
        ],
        [
            {"type": "analysis", "content": "正确理解题目"},
            {"type": "calculation", "content": "15 - 3 = 11", "has_error": True},
            {"type": "answer", "content": "答案是11"}
        ],
        [
            {"type": "analysis", "content": "正确理解题目"},
            {"type": "calculation", "content": "15 - 3 = 12"},
            {"type": "verification", "content": "验证错误", "has_error": True}
        ]
    ]
    
    # 分析错误传播
    error_patterns = evaluator.analyze_error_propagation(failed_chains)
    print(f"\n错误传播分析:")
    for pattern, count in error_patterns.items():
        print(f"  {pattern}: {count}")


def comprehensive_evaluation_example():
    """综合评估示例"""
    print("\n" + "=" * 60)
    print("综合评估示例")
    print("=" * 60)
    
    # 模拟一个完整的评估流程
    print("模拟数学问题求解系统的综合评估...")
    
    # 1. 性能评估
    performance_evaluator = PerformanceEvaluator()
    predictions = [42, 15, 28, 9, 33]
    ground_truth = [42, 16, 28, 9, 35]
    complexity_labels = ["L1", "L0", "L2", "L0", "L3"]
    
    performance_metrics = performance_evaluator.calculate_detailed_metrics(
        predictions, ground_truth, complexity_labels
    )
    
    # 2. 关系发现评估
    relation_evaluator = RelationDiscoveryEvaluator()
    discovered_relations = [
        {"type": "mathematical_operations", "match": "multiplication"},
        {"type": "unit_conversions", "match": "cm to m"}
    ]
    true_relations = [
        {"type": "mathematical_operations", "match": "multiplication"},
        {"type": "proportional_relations", "match": "direct proportion"}
    ]
    
    relation_metrics = relation_evaluator.evaluate_relation_discovery(
        discovered_relations, true_relations
    )
    
    # 3. 推理链评估
    reasoning_evaluator = ReasoningChainEvaluator()
    sample_chain = [
        {"type": "analysis", "content": "分析问题", "input": "问题", "operation": "analyze", "output": "理解"},
        {"type": "calculation", "content": "计算", "input": "数据", "operation": "compute", "output": "结果"}
    ]
    
    reasoning_metrics = reasoning_evaluator.evaluate_reasoning_chain_quality(sample_chain)
    
    # 综合结果
    comprehensive_results = {
        "performance_metrics": performance_metrics,
        "relation_discovery_metrics": relation_metrics,
        "reasoning_chain_metrics": reasoning_metrics,
        "overall_system_score": (
            performance_metrics.get("overall_accuracy", 0) * 0.4 +
            relation_metrics.get("f1", 0) * 0.3 +
            reasoning_metrics.get("overall", 0) * 0.3
        )
    }
    
    print(f"系统综合评估结果:")
    print(f"  性能准确率: {performance_metrics.get('overall_accuracy', 0):.4f}")
    print(f"  关系发现F1: {relation_metrics.get('f1', 0):.4f}")
    print(f"  推理链质量: {reasoning_metrics.get('overall', 0):.4f}")
    print(f"  系统综合分数: {comprehensive_results['overall_system_score']:.4f}")
    
    # 导出结果
    output_path = "evaluation_results.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n评估结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")


def main():
    """主函数"""
    print("数学问题求解系统评估器使用示例")
    print("="*80)
    
    try:
        # 运行各个示例
        performance_evaluator_example()
        relation_discovery_evaluator_example()
        reasoning_chain_evaluator_example()
        comprehensive_evaluation_example()
        
        print("\n" + "="*80)
        print("所有示例运行完成！")
        
    except Exception as e:
        logger.error(f"运行示例时出错: {e}")
        raise


if __name__ == "__main__":
    main() 