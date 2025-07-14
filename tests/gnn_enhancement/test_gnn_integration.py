#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GNN Integration Tests
====================

测试GNN集成功能的单元测试

Author: AI Assistant
Date: 2024-07-13
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# 添加src路径
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# 尝试导入GNN模块
try:
    from gnn_enhancement import (GNNIntegrator, GNNUtils, GraphBuilder,
                                 MathConceptGNN, ReasoningGNN, VerificationGNN,
                                 get_gnn_status, initialize_gnn_module)
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False


class TestMathConceptGNN(unittest.TestCase):
    """测试MathConceptGNN类"""
    
    def setUp(self):
        """设置测试环境"""
        if not GNN_AVAILABLE:
            self.skipTest("GNN模块不可用")
        
        self.concept_gnn = MathConceptGNN()
        self.sample_problem = "一个长方形的长是8米，宽是5米，求面积。"
        self.sample_entities = ["8", "米", "5", "米", "长方形", "面积"]
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.concept_gnn)
        self.assertEqual(self.concept_gnn.concept_dim, 128)
        self.assertEqual(self.concept_gnn.hidden_dim, 256)
        self.assertIsInstance(self.concept_gnn.relation_types, dict)
    
    def test_build_concept_graph(self):
        """测试构建概念图"""
        result = self.concept_gnn.build_concept_graph(self.sample_problem, self.sample_entities)
        
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertIn("concepts", result)
            self.assertIn("relations", result)
            self.assertIn("num_concepts", result)
            self.assertIn("num_relations", result)
    
    def test_concept_similarity(self):
        """测试概念相似度计算"""
        similarity = self.concept_gnn.get_concept_similarity("长方形", "面积")
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_enhance_implicit_relations(self):
        """测试增强隐式关系发现"""
        existing_relations = [
            {"source": "长方形", "target": "面积", "type": "geometric_relation", "confidence": 0.8}
        ]
        
        enhanced_relations = self.concept_gnn.enhance_implicit_relations(
            self.sample_problem, existing_relations
        )
        
        self.assertIsInstance(enhanced_relations, list)
        self.assertGreaterEqual(len(enhanced_relations), len(existing_relations))
    
    def test_module_info(self):
        """测试模块信息"""
        info = self.concept_gnn.get_module_info()
        self.assertIsInstance(info, dict)
        self.assertIn("name", info)
        self.assertIn("version", info)
        self.assertEqual(info["name"], "MathConceptGNN")


class TestReasoningGNN(unittest.TestCase):
    """测试ReasoningGNN类"""
    
    def setUp(self):
        """设置测试环境"""
        if not GNN_AVAILABLE:
            self.skipTest("GNN模块不可用")
        
        self.reasoning_gnn = ReasoningGNN()
        self.sample_steps = [
            {
                "id": 0,
                "description": "识别长方形的长和宽",
                "action": "extraction",
                "confidence": 0.9
            },
            {
                "id": 1,
                "description": "应用面积公式",
                "action": "calculation",
                "confidence": 0.8
            }
        ]
        self.sample_context = {"problem_type": "geometry"}
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.reasoning_gnn)
        self.assertEqual(self.reasoning_gnn.step_dim, 128)
        self.assertEqual(self.reasoning_gnn.hidden_dim, 256)
        self.assertIsInstance(self.reasoning_gnn.step_types, dict)
    
    def test_build_reasoning_graph(self):
        """测试构建推理图"""
        result = self.reasoning_gnn.build_reasoning_graph(self.sample_steps, self.sample_context)
        
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertIn("steps", result)
            self.assertIn("dependencies", result)
            self.assertIn("num_steps", result)
    
    def test_optimize_reasoning_path(self):
        """测试优化推理路径"""
        optimized_steps = self.reasoning_gnn.optimize_reasoning_path(
            self.sample_steps, self.sample_context
        )
        
        self.assertIsInstance(optimized_steps, list)
        self.assertGreaterEqual(len(optimized_steps), 0)
    
    def test_reasoning_quality_score(self):
        """测试推理质量分数"""
        quality_score = self.reasoning_gnn.get_reasoning_quality_score(
            self.sample_steps, self.sample_context
        )
        
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)


class TestVerificationGNN(unittest.TestCase):
    """测试VerificationGNN类"""
    
    def setUp(self):
        """设置测试环境"""
        if not GNN_AVAILABLE:
            self.skipTest("GNN模块不可用")
        
        self.verification_gnn = VerificationGNN()
        self.sample_steps = [
            {
                "id": 0,
                "description": "识别长方形的长和宽",
                "action": "extraction",
                "confidence": 0.9
            }
        ]
        self.sample_context = {"problem_type": "geometry"}
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.verification_gnn)
        self.assertEqual(self.verification_gnn.verification_dim, 128)
        self.assertIsInstance(self.verification_gnn.verification_types, dict)
    
    def test_build_verification_graph(self):
        """测试构建验证图"""
        result = self.verification_gnn.build_verification_graph(
            self.sample_steps, self.sample_context
        )
        
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertIn("verification_steps", result)
            self.assertIn("dependencies", result)
    
    def test_perform_verification(self):
        """测试执行验证"""
        result = self.verification_gnn.perform_verification(
            self.sample_steps, self.sample_context
        )
        
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertIn("overall_result", result)
            self.assertIn("confidence_score", result)
    
    def test_enhance_verification_accuracy(self):
        """测试增强验证准确性"""
        existing_verification = {
            "confidence_score": 0.7,
            "verification_details": []
        }
        
        enhanced = self.verification_gnn.enhance_verification_accuracy(
            self.sample_steps, existing_verification
        )
        
        self.assertIsInstance(enhanced, dict)
        if "error" not in enhanced:
            self.assertIn("confidence_score", enhanced)


class TestGraphBuilder(unittest.TestCase):
    """测试GraphBuilder类"""
    
    def setUp(self):
        """设置测试环境"""
        if not GNN_AVAILABLE:
            self.skipTest("GNN模块不可用")
        
        self.graph_builder = GraphBuilder()
        self.sample_problem = "一个长方形的长是8米，宽是5米，求面积。"
        self.sample_steps = [
            {"id": 0, "description": "识别长方形的长和宽", "action": "extraction"}
        ]
        self.sample_context = {"problem_type": "geometry"}
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.graph_builder)
        self.assertIsNotNone(self.graph_builder.concept_builder)
        self.assertIsNotNone(self.graph_builder.reasoning_builder)
        self.assertIsNotNone(self.graph_builder.verification_builder)
    
    def test_build_concept_graph(self):
        """测试构建概念图"""
        result = self.graph_builder.build_concept_graph(self.sample_problem, self.sample_context)
        self.assertIsInstance(result, dict)
    
    def test_build_all_graphs(self):
        """测试构建所有图"""
        result = self.graph_builder.build_all_graphs(
            self.sample_problem, self.sample_steps, self.sample_context
        )
        
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertIn("concept_graph", result)
            self.assertIn("reasoning_graph", result)
            self.assertIn("verification_graph", result)
    
    def test_validate_graphs(self):
        """测试验证图结构"""
        sample_graphs = {
            "concept_graph": {"concepts": [], "relations": []},
            "reasoning_graph": {"steps": [], "dependencies": []},
            "verification_graph": {"verification_steps": [], "dependencies": []}
        }
        
        validation = self.graph_builder.validate_graphs(sample_graphs)
        self.assertIsInstance(validation, dict)
        self.assertIn("valid", validation)
        self.assertIn("errors", validation)
        self.assertIn("warnings", validation)


class TestGNNIntegrator(unittest.TestCase):
    """测试GNNIntegrator类"""
    
    def setUp(self):
        """设置测试环境"""
        if not GNN_AVAILABLE:
            self.skipTest("GNN模块不可用")
        
        self.integrator = GNNIntegrator()
        self.sample_problem = "一个长方形的长是8米，宽是5米，求面积。"
        self.sample_steps = [
            {"id": 0, "description": "识别长方形的长和宽", "action": "extraction"}
        ]
        self.sample_relations = [
            {"source": "长方形", "target": "面积", "type": "geometric_relation"}
        ]
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.integrator)
        self.assertIsNotNone(self.integrator.math_concept_gnn)
        self.assertIsNotNone(self.integrator.reasoning_gnn)
        self.assertIsNotNone(self.integrator.verification_gnn)
        self.assertIsNotNone(self.integrator.graph_builder)
    
    def test_get_integration_status(self):
        """测试获取集成状态"""
        status = self.integrator.get_integration_status()
        self.assertIsInstance(status, dict)
        self.assertIn("integrator_initialized", status)
        self.assertIn("components_status", status)
        self.assertTrue(status["integrator_initialized"])
    
    def test_enhance_ird_module(self):
        """测试增强IRD模块"""
        result = self.integrator.enhance_ird_module(self.sample_problem, self.sample_relations)
        
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertIn("enhanced_relations", result)
            self.assertIn("concept_graph", result)
    
    def test_enhance_mlr_module(self):
        """测试增强MLR模块"""
        context = {"problem_type": "geometry"}
        result = self.integrator.enhance_mlr_module(self.sample_steps, context)
        
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertIn("optimized_steps", result)
            self.assertIn("reasoning_graph", result)
    
    def test_enhance_cv_module(self):
        """测试增强CV模块"""
        existing_verification = {"confidence_score": 0.7, "verification_details": []}
        result = self.integrator.enhance_cv_module(self.sample_steps, existing_verification)
        
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertIn("enhanced_verification", result)
            self.assertIn("verification_graph", result)


class TestGNNUtils(unittest.TestCase):
    """测试GNNUtils类"""
    
    def setUp(self):
        """设置测试环境"""
        if not GNN_AVAILABLE:
            self.skipTest("GNN模块不可用")
        
        self.sample_graph = {
            "nodes": [
                {"id": 0, "text": "长方形", "type": "concept"},
                {"id": 1, "text": "面积", "type": "concept"}
            ],
            "edges": [
                {"source": 0, "target": 1, "type": "geometric_relation", "weight": 0.8}
            ]
        }
    
    def test_validate_graph_structure(self):
        """测试验证图结构"""
        validation = GNNUtils.validate_graph_structure(self.sample_graph)
        self.assertIsInstance(validation, dict)
        self.assertIn("valid", validation)
        self.assertIn("errors", validation)
        self.assertIn("warnings", validation)
    
    def test_extract_graph_features(self):
        """测试提取图特征"""
        features = GNNUtils.extract_graph_features(self.sample_graph)
        self.assertIsInstance(features, dict)
        self.assertIn("num_nodes", features)
        self.assertIn("num_edges", features)
        self.assertIn("density", features)
    
    def test_calculate_graph_metrics(self):
        """测试计算图度量"""
        metrics = GNNUtils.calculate_graph_metrics(self.sample_graph)
        self.assertIsInstance(metrics, dict)
        self.assertIn("num_nodes", metrics)
        self.assertIn("num_edges", metrics)
        self.assertIn("density", metrics)
    
    def test_format_graph_for_visualization(self):
        """测试格式化图用于可视化"""
        viz_data = GNNUtils.format_graph_for_visualization(self.sample_graph)
        self.assertIsInstance(viz_data, dict)
        self.assertIn("nodes", viz_data)
        self.assertIn("links", viz_data)
        self.assertIsInstance(viz_data["nodes"], list)
        self.assertIsInstance(viz_data["links"], list)


class TestGNNModule(unittest.TestCase):
    """测试GNN模块级别功能"""
    
    def test_get_gnn_status(self):
        """测试获取GNN状态"""
        if not GNN_AVAILABLE:
            self.skipTest("GNN模块不可用")
        
        status = get_gnn_status()
        self.assertIsInstance(status, dict)
        self.assertIn("version", status)
        self.assertIn("gnn_available", status)
        self.assertIn("components", status)
    
    def test_initialize_gnn_module(self):
        """测试初始化GNN模块"""
        if not GNN_AVAILABLE:
            self.skipTest("GNN模块不可用")
        
        result = initialize_gnn_module()
        self.assertIsInstance(result, bool)


def run_tests():
    """运行所有测试"""
    print("🧪 运行GNN集成测试...")
    
    if not GNN_AVAILABLE:
        print("❌ GNN模块不可用，跳过测试")
        print("请安装必要的依赖: pip install torch dgl networkx")
        return False
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestMathConceptGNN,
        TestReasoningGNN,
        TestVerificationGNN,
        TestGraphBuilder,
        TestGNNIntegrator,
        TestGNNUtils,
        TestGNNModule
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
        return True
    else:
        print(f"\n❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 