#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GNN Enhancement Demo
===================

演示GNN增强功能的完整示例

展示如何使用GNN来增强COT-DIR1的：
1. 隐式关系发现 (IRD)
2. 多层级推理 (MLR)
3. 链式验证 (CV)

Author: AI Assistant
Date: 2024-07-13
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 添加src路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# 导入GNN模块
try:
    from gnn_enhancement import (GNNIntegrator, GNNUtils, GraphBuilder,
                                 MathConceptGNN, ReasoningGNN, VerificationGNN,
                                 get_gnn_status, initialize_gnn_module)
    GNN_AVAILABLE = True
except ImportError as e:
    print(f"警告: GNN模块导入失败: {e}")
    GNN_AVAILABLE = False


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def demo_math_concept_gnn():
    """演示数学概念GNN"""
    print_section("1. 数学概念GNN演示")
    
    if not GNN_AVAILABLE:
        print("❌ GNN模块不可用，跳过演示")
        return
    
    # 创建MathConceptGNN实例
    concept_gnn = MathConceptGNN()
    
    # 示例问题
    problem_text = "一个长方形的长是8米，宽是5米，求这个长方形的面积。"
    entities = ["8", "米", "5", "米", "长方形", "面积"]
    
    print(f"📝 问题文本: {problem_text}")
    print(f"🔍 识别实体: {entities}")
    
    # 构建概念图
    print_subsection("构建概念图")
    concept_graph = concept_gnn.build_concept_graph(problem_text, entities)
    
    if "error" not in concept_graph:
        print(f"✅ 概念图构建成功")
        print(f"   - 概念数量: {concept_graph.get('num_concepts', 0)}")
        print(f"   - 关系数量: {concept_graph.get('num_relations', 0)}")
        
        # 显示概念
        concepts = concept_graph.get("concepts", [])[:3]  # 显示前3个
        for i, concept in enumerate(concepts):
            print(f"   - 概念{i+1}: {concept.get('text', 'N/A')} ({concept.get('type', 'unknown')})")
    else:
        print(f"❌ 概念图构建失败: {concept_graph['error']}")
    
    # 测试概念相似度
    print_subsection("概念相似度计算")
    similarity = concept_gnn.get_concept_similarity("长方形", "面积")
    print(f"🔗 '长方形' 与 '面积' 的相似度: {similarity:.3f}")
    
    # 增强隐式关系发现
    print_subsection("增强隐式关系发现")
    existing_relations = [
        {"source": "长方形", "target": "面积", "type": "geometric_relation", "confidence": 0.8}
    ]
    
    enhanced_relations = concept_gnn.enhance_implicit_relations(problem_text, existing_relations)
    print(f"🔄 原始关系数量: {len(existing_relations)}")
    print(f"✨ 增强后关系数量: {len(enhanced_relations)}")
    
    # 显示模块信息
    print_subsection("模块信息")
    module_info = concept_gnn.get_module_info()
    print(f"📊 模块名称: {module_info['name']}")
    print(f"📊 版本: {module_info['version']}")
    print(f"📊 概念数量: {module_info['num_concepts']}")


def demo_reasoning_gnn():
    """演示推理GNN"""
    print_section("2. 推理GNN演示")
    
    if not GNN_AVAILABLE:
        print("❌ GNN模块不可用，跳过演示")
        return
    
    # 创建ReasoningGNN实例
    reasoning_gnn = ReasoningGNN()
    
    # 示例推理步骤
    reasoning_steps = [
        {
            "id": 0,
            "description": "识别长方形的长和宽",
            "action": "extraction",
            "inputs": ["问题文本"],
            "outputs": ["长=8米", "宽=5米"],
            "confidence": 0.9
        },
        {
            "id": 1,
            "description": "应用长方形面积公式",
            "action": "calculation",
            "inputs": ["长=8米", "宽=5米"],
            "outputs": ["面积=长×宽"],
            "confidence": 0.8
        },
        {
            "id": 2,
            "description": "计算具体数值",
            "action": "calculation",
            "inputs": ["面积=长×宽", "长=8", "宽=5"],
            "outputs": ["面积=40平方米"],
            "confidence": 0.9
        }
    ]
    
    problem_context = {
        "problem_type": "geometry",
        "difficulty": "basic",
        "domain": "area_calculation"
    }
    
    print(f"🔄 推理步骤数量: {len(reasoning_steps)}")
    
    # 构建推理图
    print_subsection("构建推理图")
    reasoning_graph = reasoning_gnn.build_reasoning_graph(reasoning_steps, problem_context)
    
    if "error" not in reasoning_graph:
        print(f"✅ 推理图构建成功")
        print(f"   - 步骤数量: {reasoning_graph.get('num_steps', 0)}")
        print(f"   - 依赖关系: {reasoning_graph.get('num_dependencies', 0)}")
    else:
        print(f"❌ 推理图构建失败: {reasoning_graph['error']}")
    
    # 优化推理路径
    print_subsection("优化推理路径")
    optimized_steps = reasoning_gnn.optimize_reasoning_path(reasoning_steps, problem_context)
    print(f"🔄 原始步骤数量: {len(reasoning_steps)}")
    print(f"✨ 优化后步骤数量: {len(optimized_steps)}")
    
    # 计算推理质量分数
    print_subsection("推理质量评估")
    quality_score = reasoning_gnn.get_reasoning_quality_score(reasoning_steps, problem_context)
    print(f"📊 推理质量分数: {quality_score:.3f}")
    
    # 显示模块信息
    print_subsection("模块信息")
    module_info = reasoning_gnn.get_module_info()
    print(f"📊 模块名称: {module_info['name']}")
    print(f"📊 版本: {module_info['version']}")
    print(f"📊 步骤类型数量: {module_info['num_step_types']}")


def demo_verification_gnn():
    """演示验证GNN"""
    print_section("3. 验证GNN演示")
    
    if not GNN_AVAILABLE:
        print("❌ GNN模块不可用，跳过演示")
        return
    
    # 创建VerificationGNN实例
    verification_gnn = VerificationGNN()
    
    # 使用之前的推理步骤
    reasoning_steps = [
        {
            "id": 0,
            "description": "识别长方形的长和宽",
            "action": "extraction",
            "confidence": 0.9
        },
        {
            "id": 1,
            "description": "应用长方形面积公式",
            "action": "calculation",
            "confidence": 0.8
        },
        {
            "id": 2,
            "description": "计算具体数值",
            "action": "calculation",
            "confidence": 0.9
        }
    ]
    
    verification_context = {
        "problem_type": "geometry",
        "expected_answer": "40平方米"
    }
    
    # 构建验证图
    print_subsection("构建验证图")
    verification_graph = verification_gnn.build_verification_graph(reasoning_steps, verification_context)
    
    if "error" not in verification_graph:
        print(f"✅ 验证图构建成功")
        print(f"   - 验证步骤数量: {verification_graph.get('num_verification_steps', 0)}")
        print(f"   - 依赖关系: {verification_graph.get('num_dependencies', 0)}")
    else:
        print(f"❌ 验证图构建失败: {verification_graph['error']}")
    
    # 执行验证
    print_subsection("执行验证")
    verification_result = verification_gnn.perform_verification(reasoning_steps, verification_context)
    
    if "error" not in verification_result:
        print(f"✅ 验证执行成功")
        print(f"   - 整体结果: {verification_result.get('overall_result', 'unknown')}")
        print(f"   - 置信度: {verification_result.get('confidence_score', 0.0):.3f}")
        print(f"   - 通过检查: {verification_result.get('passed_checks', 0)}/{verification_result.get('total_checks', 0)}")
    else:
        print(f"❌ 验证执行失败: {verification_result['error']}")
    
    # 增强验证准确性
    print_subsection("增强验证准确性")
    existing_verification = {
        "confidence_score": 0.7,
        "verification_details": [
            {"result": "pass", "confidence": 0.8, "verification_type": "calculation_check"}
        ]
    }
    
    enhanced_verification = verification_gnn.enhance_verification_accuracy(
        reasoning_steps, existing_verification
    )
    
    if "error" not in enhanced_verification:
        original_confidence = existing_verification.get("confidence_score", 0.0)
        enhanced_confidence = enhanced_verification.get("confidence_score", 0.0)
        improvement = enhanced_confidence - original_confidence
        
        print(f"🔄 原始置信度: {original_confidence:.3f}")
        print(f"✨ 增强后置信度: {enhanced_confidence:.3f}")
        print(f"📈 提升幅度: {improvement:.3f}")
    else:
        print(f"❌ 验证增强失败: {enhanced_verification['error']}")


def demo_graph_builder():
    """演示图构建器"""
    print_section("4. 图构建器演示")
    
    if not GNN_AVAILABLE:
        print("❌ GNN模块不可用，跳过演示")
        return
    
    # 创建GraphBuilder实例
    graph_builder = GraphBuilder()
    
    # 示例数据
    problem_text = "一个长方形的长是8米，宽是5米，求这个长方形的面积。"
    reasoning_steps = [
        {"id": 0, "description": "识别长方形的长和宽", "action": "extraction"},
        {"id": 1, "description": "应用长方形面积公式", "action": "calculation"},
        {"id": 2, "description": "计算具体数值", "action": "calculation"}
    ]
    context = {"problem_type": "geometry"}
    
    # 构建所有类型的图
    print_subsection("构建所有图")
    all_graphs = graph_builder.build_all_graphs(problem_text, reasoning_steps, context)
    
    if "error" not in all_graphs:
        print(f"✅ 图构建成功")
        
        # 显示图统计
        stats = graph_builder.get_graph_statistics(all_graphs)
        print(f"📊 概念图: {stats['concept_graph'].get('num_concepts', 0)} 概念, {stats['concept_graph'].get('num_relations', 0)} 关系")
        print(f"📊 推理图: {stats['reasoning_graph'].get('num_steps', 0)} 步骤, {stats['reasoning_graph'].get('num_dependencies', 0)} 依赖")
        print(f"📊 验证图: {stats['verification_graph'].get('num_verification_steps', 0)} 验证步骤")
        
        # 验证图结构
        print_subsection("验证图结构")
        validation = graph_builder.validate_graphs(all_graphs)
        print(f"✅ 图结构验证: {'通过' if validation['valid'] else '失败'}")
        if validation['errors']:
            print(f"❌ 错误: {validation['errors']}")
        if validation['warnings']:
            print(f"⚠️ 警告: {validation['warnings']}")
    else:
        print(f"❌ 图构建失败: {all_graphs['error']}")


def demo_gnn_integrator():
    """演示GNN集成器"""
    print_section("5. GNN集成器演示")
    
    if not GNN_AVAILABLE:
        print("❌ GNN模块不可用，跳过演示")
        return
    
    # 创建GNNIntegrator实例
    integrator = GNNIntegrator()
    
    # 显示集成状态
    print_subsection("集成状态")
    status = integrator.get_integration_status()
    print(f"📊 集成器状态: {'✅ 已初始化' if status['integrator_initialized'] else '❌ 未初始化'}")
    
    components = status['components_status']
    for component, available in components.items():
        status_icon = "✅" if available else "❌"
        print(f"   - {component}: {status_icon}")
    
    # 示例数据
    problem_text = "一个长方形的长是8米，宽是5米，求这个长方形的面积。"
    reasoning_steps = [
        {"id": 0, "description": "识别长方形的长和宽", "action": "extraction"},
        {"id": 1, "description": "应用长方形面积公式", "action": "calculation"},
        {"id": 2, "description": "计算具体数值", "action": "calculation"}
    ]
    
    # 模拟现有处理结果
    processing_result = {
        "relation_results": {
            "relations": [
                {"source": "长方形", "target": "面积", "type": "geometric_relation"}
            ]
        }
    }
    
    # 模拟现有评估结果
    evaluation_result = {
        "confidence_score": 0.7,
        "verification_details": [
            {"result": "pass", "confidence": 0.8}
        ]
    }
    
    # 综合集成演示
    print_subsection("综合集成")
    comprehensive_result = integrator.comprehensive_integration(
        problem_text, reasoning_steps, processing_result, evaluation_result
    )
    
    if "error" not in comprehensive_result:
        print(f"✅ 综合集成成功")
        
        # 显示增强效果
        summary = comprehensive_result.get("enhancement_summary", {})
        print(f"📈 IRD提升比例: {summary.get('ird_improvement', 1.0):.2f}")
        print(f"📈 MLR质量分数: {summary.get('mlr_quality', 0.5):.3f}")
        print(f"📈 CV置信度: {summary.get('cv_confidence', 0.5):.3f}")
    else:
        print(f"❌ 综合集成失败: {comprehensive_result['error']}")


def demo_gnn_utils():
    """演示GNN工具"""
    print_section("6. GNN工具演示")
    
    if not GNN_AVAILABLE:
        print("❌ GNN模块不可用，跳过演示")
        return
    
    # 示例图数据
    sample_graph = {
        "nodes": [
            {"id": 0, "text": "长方形", "type": "concept"},
            {"id": 1, "text": "面积", "type": "concept"},
            {"id": 2, "text": "8米", "type": "number"},
            {"id": 3, "text": "5米", "type": "number"}
        ],
        "edges": [
            {"source": 0, "target": 1, "type": "geometric_relation", "weight": 0.8},
            {"source": 0, "target": 2, "type": "unit_relation", "weight": 0.9},
            {"source": 0, "target": 3, "type": "unit_relation", "weight": 0.9}
        ]
    }
    
    # 验证图结构
    print_subsection("图结构验证")
    validation = GNNUtils.validate_graph_structure(sample_graph)
    print(f"✅ 图结构验证: {'通过' if validation['valid'] else '失败'}")
    
    # 提取图特征
    print_subsection("图特征提取")
    features = GNNUtils.extract_graph_features(sample_graph)
    print(f"📊 节点数量: {features['num_nodes']}")
    print(f"📊 边数量: {features['num_edges']}")
    print(f"📊 图密度: {features['density']:.3f}")
    print(f"📊 平均度: {features['avg_degree']:.2f}")
    
    # 计算图度量
    print_subsection("图度量计算")
    metrics = GNNUtils.calculate_graph_metrics(sample_graph)
    print(f"📊 最大度: {metrics['max_degree']}")
    print(f"📊 最小度: {metrics['min_degree']}")
    print(f"📊 度标准差: {metrics['degree_std']:.3f}")
    print(f"📊 连通性: {'✅ 连通' if metrics['is_connected'] else '❌ 不连通'}")
    
    # 格式化用于可视化
    print_subsection("可视化格式化")
    viz_data = GNNUtils.format_graph_for_visualization(sample_graph)
    print(f"📊 可视化节点: {len(viz_data['nodes'])}")
    print(f"📊 可视化链接: {len(viz_data['links'])}")


def main():
    """主函数"""
    print("🚀 GNN Enhancement Demo for COT-DIR1")
    print("=" * 60)
    
    # 检查GNN模块状态
    if GNN_AVAILABLE:
        print("✅ GNN模块加载成功")
        
        # 初始化GNN模块
        if initialize_gnn_module():
            print("✅ GNN模块初始化成功")
            
            # 获取GNN状态
            status = get_gnn_status()
            print(f"📊 GNN版本: {status['version']}")
            print(f"📊 PyTorch可用: {'✅' if status['torch_available'] else '❌'}")
            print(f"📊 DGL可用: {'✅' if status['dgl_available'] else '❌'}")
        else:
            print("❌ GNN模块初始化失败")
            return
    else:
        print("❌ GNN模块不可用")
        print("请安装必要的依赖: pip install torch dgl networkx")
        return
    
    try:
        # 运行各个演示
        demo_math_concept_gnn()
        demo_reasoning_gnn()
        demo_verification_gnn()
        demo_graph_builder()
        demo_gnn_integrator()
        demo_gnn_utils()
        
        print_section("演示完成")
        print("🎉 所有GNN功能演示完成！")
        print("💡 您可以根据需要在实际项目中使用这些功能。")
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 