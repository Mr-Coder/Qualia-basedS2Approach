"""
数据集分析示例
=============

这个示例展示如何使用DatasetLoader、ComplexityClassifier和ImplicitRelationAnnotator
来加载数据集、分析问题复杂度和标注隐式关系。

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from processors.complexity_classifier import ComplexityClassifier
# 导入处理器类
from processors.dataset_loader import DatasetLoader
from processors.implicit_relation_annotator import ImplicitRelationAnnotator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dataset_example():
    """
    数据集分析示例
    
    演示完整的数据集加载、复杂度分析和隐式关系标注流程
    """
    logger.info("开始数据集分析示例")
    
    # 1. 初始化组件
    dataset_loader = DatasetLoader()
    complexity_classifier = ComplexityClassifier()
    relation_annotator = ImplicitRelationAnnotator()
    
    # 2. 创建示例数据（实际使用时从文件加载）
    sample_problems = [
        {
            "id": 1,
            "question": "小明有20个苹果，吃了5个，还剩多少个？",
            "answer": "15",
            "dataset": "sample"
        },
        {
            "id": 2,
            "question": "一个长方形的长是8米，宽是6米，它的面积是多少平方米？",
            "answer": "48",
            "dataset": "sample"
        },
        {
            "id": 3,
            "question": "小华以每小时60千米的速度行驶了2小时，然后以每小时80千米的速度行驶了1.5小时，总共行驶了多少千米？",
            "answer": "240",
            "dataset": "sample"
        },
        {
            "id": 4,
            "question": "一个水池可以装500升水，现在已经装了300升，如果每分钟可以装25升，还需要多少分钟才能装满？",
            "answer": "8",
            "dataset": "sample"
        }
    ]
    
    logger.info(f"加载了 {len(sample_problems)} 个示例问题")
    
    # 3. 复杂度分析
    logger.info("开始复杂度分析...")
    classified_problems = complexity_classifier.batch_classify_problems(sample_problems)
    
    # 计算DIR分数
    complexity_analysis = complexity_classifier.analyze_dataset_complexity(classified_problems)
    
    logger.info("复杂度分析结果：")
    logger.info(f"DIR分数: {complexity_analysis['dir_score']:.3f}")
    logger.info(f"级别分布: {complexity_analysis['level_distribution']}")
    logger.info(f"复杂度摘要: {complexity_analysis['complexity_summary']}")
    
    # 4. 隐式关系标注
    logger.info("开始隐式关系标注...")
    annotated_problems = relation_annotator.create_ground_truth_relations(classified_problems)
    
    # 分析关系分布
    relation_analysis = relation_annotator.analyze_relation_distribution(annotated_problems)
    
    logger.info("隐式关系分析结果：")
    logger.info(f"总问题数: {relation_analysis['total_problems']}")
    logger.info(f"总关系数: {relation_analysis['total_relations']}")
    logger.info(f"平均每题关系数: {relation_analysis['avg_relations_per_problem']:.2f}")
    logger.info(f"关系类型分布: {relation_analysis['relation_type_percentages']}")
    
    # 5. 详细展示每个问题的分析结果
    logger.info("\n详细分析结果：")
    for i, problem in enumerate(annotated_problems):
        logger.info(f"\n问题 {i+1}: {problem['question']}")
        logger.info(f"复杂度级别: {problem['complexity_level']}")
        logger.info(f"推理深度: {problem['inference_depth']}")
        logger.info(f"知识依赖: {problem['knowledge_dependency']}")
        logger.info(f"隐式关系数量: {problem['implicit_relations_count']}")
        
        if problem['implicit_relations_true']:
            logger.info("识别的隐式关系:")
            for relation in problem['implicit_relations_true']:
                logger.info(f"  - 类型: {relation['type']}, 匹配: '{relation['match']}'")
    
    # 6. 导出结果
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # 导出复杂度分析
    with open(output_dir / "complexity_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(complexity_analysis, f, ensure_ascii=False, indent=2)
    
    # 导出关系分析
    with open(output_dir / "relation_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(relation_analysis, f, ensure_ascii=False, indent=2)
    
    # 导出标注结果
    with open(output_dir / "annotated_problems.json", 'w', encoding='utf-8') as f:
        json.dump(annotated_problems, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n分析结果已导出到 {output_dir} 目录")
    
    return {
        "complexity_analysis": complexity_analysis,
        "relation_analysis": relation_analysis,
        "annotated_problems": annotated_problems
    }


def load_real_dataset_example():
    """
    真实数据集加载示例
    
    演示如何加载真实的数学题数据集
    """
    logger.info("真实数据集加载示例")
    
    dataset_loader = DatasetLoader()
    
    # 示例：加载不同类型的数据集
    # 注意：这些文件路径需要根据实际情况调整
    
    datasets_to_load = [
        # ("examples/math23k_sample.json", "load_math23k"),
        # ("examples/gsm8k_sample.json", "load_gsm8k"),
        # ("examples/mawps_sample.json", "load_mawps")
    ]
    
    for file_path, load_method in datasets_to_load:
        if Path(file_path).exists():
            try:
                loader_func = getattr(dataset_loader, load_method)
                data = loader_func(file_path)
                logger.info(f"成功加载 {load_method}: {len(data)} 个问题")
            except Exception as e:
                logger.error(f"加载 {file_path} 失败: {e}")
        else:
            logger.warning(f"文件不存在: {file_path}")
    
    # 获取数据集统计信息
    stats = dataset_loader.get_dataset_stats()
    if stats:
        logger.info("数据集统计信息:")
        for name, stat in stats.items():
            logger.info(f"  {name}: {stat}")
    else:
        logger.info("没有加载任何数据集")


def complexity_classification_example():
    """
    复杂度分类示例
    
    演示不同复杂度级别的问题分类
    """
    logger.info("复杂度分类示例")
    
    classifier = ComplexityClassifier()
    
    # 不同复杂度的示例问题
    test_problems = [
        {
            "text": "3 + 5 = ?",
            "expected_level": "L0",
            "description": "简单算术，显式问题"
        },
        {
            "text": "小明有10个苹果，给了小红3个，还剩几个？",
            "expected_level": "L1",
            "description": "简单减法，浅层隐式"
        },
        {
            "text": "一个长方形花园，长是宽的2倍，如果周长是24米，求面积。",
            "expected_level": "L2",
            "description": "需要建立方程，中等隐式"
        },
        {
            "text": "甲乙两人从A地出发到B地，甲先走1小时后乙才出发，甲的速度是4km/h，乙的速度是6km/h，乙出发后多长时间能追上甲？",
            "expected_level": "L3",
            "description": "复杂推理链，深度隐式"
        }
    ]
    
    for i, problem in enumerate(test_problems):
        level = classifier.classify_problem_complexity(problem["text"])
        inference_depth = classifier.calculate_inference_depth(problem["text"])
        knowledge_dependency = classifier.calculate_knowledge_dependency(problem["text"])
        
        logger.info(f"\n问题 {i+1}: {problem['text']}")
        logger.info(f"预期级别: {problem['expected_level']}, 实际级别: {level}")
        logger.info(f"推理深度: {inference_depth}, 知识依赖: {knowledge_dependency}")
        logger.info(f"描述: {problem['description']}")
        
        if level == problem['expected_level']:
            logger.info("✓ 分类正确")
        else:
            logger.info("✗ 分类可能需要调整")


def implicit_relation_annotation_example():
    """
    隐式关系标注示例
    
    演示如何标注数学问题中的隐式关系
    """
    logger.info("隐式关系标注示例")
    
    annotator = ImplicitRelationAnnotator()
    
    # 包含不同类型隐式关系的示例
    test_problems = [
        {
            "text": "小明买了3千克苹果，每千克8元，一共花了多少钱？",
            "expected_relations": ["mathematical_operations", "unit_conversions"]
        },
        {
            "text": "一个圆形水池的半径是5米，它的面积是多少平方米？",
            "expected_relations": ["geometric_properties"]
        },
        {
            "text": "汽车以每小时60千米的速度行驶2小时，行驶了多少千米？",
            "expected_relations": ["proportional_relations", "unit_conversions"]
        },
        {
            "text": "水箱最多能装500升水，现在装了300升，还能装多少升？",
            "expected_relations": ["physical_constraints", "mathematical_operations"]
        }
    ]
    
    for i, problem in enumerate(test_problems):
        relations = annotator.annotate_implicit_relations(problem["text"])
        
        logger.info(f"\n问题 {i+1}: {problem['text']}")
        logger.info(f"预期关系类型: {problem['expected_relations']}")
        logger.info(f"识别的关系数量: {len(relations)}")
        
        if relations:
            logger.info("识别的关系:")
            for relation in relations:
                logger.info(f"  - 类型: {relation['type']}")
                logger.info(f"    匹配: '{relation['match']}'")
                logger.info(f"    位置: {relation['position']}")
        else:
            logger.info("未识别到隐式关系")


if __name__ == "__main__":
    """运行所有示例"""
    
    print("=" * 60)
    print("数据集分析工具示例")
    print("=" * 60)
    
    try:
        # 1. 完整的数据集分析示例
        print("\n1. 完整数据集分析示例")
        print("-" * 30)
        analyze_dataset_example()
        
        # 2. 真实数据集加载示例
        print("\n2. 真实数据集加载示例")
        print("-" * 30)
        load_real_dataset_example()
        
        # 3. 复杂度分类示例
        print("\n3. 复杂度分类示例")
        print("-" * 30)
        complexity_classification_example()
        
        # 4. 隐式关系标注示例
        print("\n4. 隐式关系标注示例")
        print("-" * 30)
        implicit_relation_annotation_example()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"运行示例时出错: {e}")
        raise 