"""
QS²增强IRD引擎测试套件
====================

为QS²增强隐式关系发现系统提供全面的测试覆盖。
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# 导入待测试的模块
from src.reasoning.qs2_enhancement import (
    QualiaRole,
    QualiaStructure,
    QualiaStructureConstructor,
    CompatibilityEngine,
    CompatibilityResult,
    EnhancedIRDEngine,
    EnhancedRelation,
    DiscoveryResult,
    RelationType,
    RelationStrength,
    QS2Config,
    DataSerializer,
    BatchProcessor,
    ValidationUtils,
    create_default_config
)


class TestQualiaStructureConstructor(unittest.TestCase):
    """QualiaStructureConstructor测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.constructor = QualiaStructureConstructor()
        self.test_context = "小明有10个苹果，他吃了3个苹果，还剩多少个苹果？"
    
    def test_construct_basic_structure(self):
        """测试基本语义结构构建"""
        entity = {"name": "苹果", "type": "physical_object"}
        structure = self.constructor.construct_qualia_structure(
            entity, self.test_context
        )
        
        self.assertIsInstance(structure, QualiaStructure)
        self.assertEqual(structure.entity, "苹果")
        self.assertEqual(structure.entity_type, "physical_object")
        self.assertIsInstance(structure.formal_roles, list)
        self.assertIsInstance(structure.telic_roles, list)
        self.assertIsInstance(structure.agentive_roles, list)
        self.assertIsInstance(structure.constitutive_roles, list)
        self.assertTrue(0.0 <= structure.confidence <= 1.0)
    
    def test_construct_number_structure(self):
        """测试数字实体结构构建"""
        entity = "10"
        structure = self.constructor.construct_qualia_structure(
            entity, self.test_context
        )
        
        self.assertEqual(structure.entity, "10")
        self.assertEqual(structure.entity_type, "number")
        self.assertIn("整数", structure.formal_roles)
        self.assertIn("正数", structure.formal_roles)
    
    def test_construct_unit_structure(self):
        """测试单位实体结构构建"""
        entity = "个"
        structure = self.constructor.construct_qualia_structure(
            entity, self.test_context
        )
        
        self.assertEqual(structure.entity, "个")
        self.assertEqual(structure.entity_type, "unit")
        self.assertIn("计数单位", structure.formal_roles)
        self.assertIn("用于度量", structure.telic_roles)
    
    def test_batch_construct_structures(self):
        """测试批量构建语义结构"""
        entities = [
            {"name": "苹果", "type": "physical_object"},
            {"name": "10", "type": "number"},
            {"name": "个", "type": "unit"}
        ]
        
        structures = self.constructor.batch_construct_structures(
            entities, self.test_context
        )
        
        self.assertEqual(len(structures), 3)
        for structure in structures:
            self.assertIsInstance(structure, QualiaStructure)
            self.assertTrue(0.0 <= structure.confidence <= 1.0)
    
    def test_context_feature_extraction(self):
        """测试上下文特征提取"""
        entity = "苹果"
        structure = self.constructor.construct_qualia_structure(
            entity, self.test_context
        )
        
        context_features = structure.context_features
        self.assertIn("surrounding_words", context_features)
        self.assertIn("related_verbs", context_features)
        self.assertIn("related_numbers", context_features)
        self.assertIn("problem_type", context_features)
    
    def test_invalid_entity_handling(self):
        """测试无效实体处理"""
        invalid_entity = None
        structure = self.constructor.construct_qualia_structure(
            invalid_entity, self.test_context
        )
        
        self.assertIsInstance(structure, QualiaStructure)
        self.assertEqual(structure.confidence, 0.0)


class TestCompatibilityEngine(unittest.TestCase):
    """CompatibilityEngine测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.engine = CompatibilityEngine()
        self.constructor = QualiaStructureConstructor()
        self.test_context = "小明有10个苹果，他吃了3个苹果，还剩多少个苹果？"
    
    def test_compute_compatibility_same_type(self):
        """测试同类型实体兼容性计算"""
        entity1 = {"name": "苹果", "type": "physical_object"}
        entity2 = {"name": "橙子", "type": "physical_object"}
        
        structure1 = self.constructor.construct_qualia_structure(
            entity1, self.test_context
        )
        structure2 = self.constructor.construct_qualia_structure(
            entity2, self.test_context
        )
        
        compatibility = self.engine.compute_compatibility(
            structure1, structure2
        )
        
        self.assertTrue(0.0 <= compatibility <= 1.0)
        self.assertGreater(compatibility, 0.3)  # 同类型应该有一定兼容性
    
    def test_compute_compatibility_different_type(self):
        """测试不同类型实体兼容性计算"""
        entity1 = {"name": "10", "type": "number"}
        entity2 = {"name": "苹果", "type": "physical_object"}
        
        structure1 = self.constructor.construct_qualia_structure(
            entity1, self.test_context
        )
        structure2 = self.constructor.construct_qualia_structure(
            entity2, self.test_context
        )
        
        compatibility = self.engine.compute_compatibility(
            structure1, structure2
        )
        
        self.assertTrue(0.0 <= compatibility <= 1.0)
        # 不同类型兼容性较低，但在数学问题中可能有关联
        self.assertGreaterEqual(compatibility, 0.0)
    
    def test_compute_detailed_compatibility(self):
        """测试详细兼容性计算"""
        entity1 = {"name": "苹果", "type": "physical_object"}
        entity2 = {"name": "橙子", "type": "physical_object"}
        
        structure1 = self.constructor.construct_qualia_structure(
            entity1, self.test_context
        )
        structure2 = self.constructor.construct_qualia_structure(
            entity2, self.test_context
        )
        
        result = self.engine.compute_detailed_compatibility(
            structure1, structure2
        )
        
        self.assertIsInstance(result, CompatibilityResult)
        self.assertEqual(result.entity1, "苹果")
        self.assertEqual(result.entity2, "橙子")
        self.assertTrue(0.0 <= result.overall_score <= 1.0)
        self.assertIn("formal", result.detailed_scores)
        self.assertIn("telic", result.detailed_scores)
        self.assertIn("agentive", result.detailed_scores)
        self.assertIn("constitutive", result.detailed_scores)
        self.assertIn("contextual", result.detailed_scores)
    
    def test_batch_compute_compatibility(self):
        """测试批量兼容性计算"""
        entities = [
            {"name": "苹果", "type": "physical_object"},
            {"name": "橙子", "type": "physical_object"},
            {"name": "10", "type": "number"}
        ]
        
        structures = [
            self.constructor.construct_qualia_structure(entity, self.test_context)
            for entity in entities
        ]
        
        results = self.engine.batch_compute_compatibility(structures)
        
        # 应该有3个实体的 C(3,2) = 3 个组合
        self.assertEqual(len(results), 3)
        for i, j, compatibility in results:
            self.assertTrue(0 <= i < len(structures))
            self.assertTrue(0 <= j < len(structures))
            self.assertTrue(i < j)  # 确保不重复
            self.assertTrue(0.0 <= compatibility <= 1.0)
    
    def test_high_compatibility_pairs(self):
        """测试高兼容性实体对识别"""
        entities = [
            {"name": "苹果", "type": "physical_object"},
            {"name": "橙子", "type": "physical_object"},
            {"name": "书", "type": "physical_object"}
        ]
        
        structures = [
            self.constructor.construct_qualia_structure(entity, self.test_context)
            for entity in entities
        ]
        
        high_pairs = self.engine.get_high_compatibility_pairs(
            structures, threshold=0.4
        )
        
        self.assertIsInstance(high_pairs, list)
        for struct1, struct2, compatibility in high_pairs:
            self.assertIsInstance(struct1, QualiaStructure)
            self.assertIsInstance(struct2, QualiaStructure)
            self.assertGreaterEqual(compatibility, 0.4)


class TestEnhancedIRDEngine(unittest.TestCase):
    """EnhancedIRDEngine测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.engine = EnhancedIRDEngine()
        self.test_problem = "小明有10个苹果，他吃了3个苹果，还剩多少个苹果？"
    
    def test_discover_relations_basic(self):
        """测试基本关系发现"""
        result = self.engine.discover_relations(self.test_problem)
        
        self.assertIsInstance(result, DiscoveryResult)
        self.assertIsInstance(result.relations, list)
        self.assertGreaterEqual(result.entity_count, 1)
        self.assertGreaterEqual(result.processing_time, 0)
        self.assertGreaterEqual(result.total_pairs_evaluated, 0)
    
    def test_discover_relations_with_entities(self):
        """测试带实体列表的关系发现"""
        entities = [
            {"name": "苹果", "type": "physical_object"},
            {"name": "10", "type": "number"},
            {"name": "3", "type": "number"},
            {"name": "个", "type": "unit"}
        ]
        
        result = self.engine.discover_relations(
            self.test_problem, entities
        )
        
        self.assertEqual(result.entity_count, 4)
        self.assertGreater(result.total_pairs_evaluated, 0)
        
        # 检查关系
        for relation in result.relations:
            self.assertIsInstance(relation, EnhancedRelation)
            self.assertTrue(0.0 <= relation.strength <= 1.0)
            self.assertTrue(0.0 <= relation.confidence <= 1.0)
            self.assertIn(relation.relation_type, list(RelationType))
            self.assertIn(relation.strength_level, list(RelationStrength))
    
    def test_relation_filtering(self):
        """测试关系过滤"""
        result = self.engine.discover_relations(self.test_problem)
        
        # 按强度过滤
        strong_relations = self.engine.filter_relations_by_strength(
            result.relations, 0.6
        )
        for relation in strong_relations:
            self.assertGreaterEqual(relation.strength, 0.6)
        
        # 按类型过滤
        semantic_relations = self.engine.filter_relations_by_type(
            result.relations, RelationType.SEMANTIC
        )
        for relation in semantic_relations:
            self.assertEqual(relation.relation_type, RelationType.SEMANTIC)
    
    def test_entity_relations(self):
        """测试实体关系获取"""
        entities = [
            {"name": "苹果", "type": "physical_object"},
            {"name": "10", "type": "number"}
        ]
        
        result = self.engine.discover_relations(
            self.test_problem, entities
        )
        
        if result.relations:
            entity_name = result.relations[0].entity1
            entity_relations = self.engine.get_entity_relations(
                entity_name, result.relations
            )
            
            for relation in entity_relations:
                self.assertTrue(
                    relation.entity1 == entity_name or 
                    relation.entity2 == entity_name
                )
    
    def test_export_to_graph(self):
        """测试导出为图结构"""
        result = self.engine.discover_relations(self.test_problem)
        
        graph = self.engine.export_relations_to_graph(result.relations)
        
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)
        self.assertIsInstance(graph["nodes"], list)
        self.assertIsInstance(graph["edges"], list)
        
        for edge in graph["edges"]:
            self.assertIn("source", edge)
            self.assertIn("target", edge)
            self.assertIn("weight", edge)
            self.assertIn("type", edge)
    
    def test_configuration(self):
        """测试配置调整"""
        # 测试阈值配置
        self.engine.configure_thresholds(
            min_strength=0.5,
            max_relations_per_entity=5
        )
        
        self.assertEqual(self.engine.min_strength_threshold, 0.5)
        self.assertEqual(self.engine.max_relations_per_entity, 5)
        
        # 测试配置后的关系发现
        result = self.engine.discover_relations(self.test_problem)
        
        for relation in result.relations:
            self.assertGreaterEqual(relation.strength, 0.5)


class TestQS2Config(unittest.TestCase):
    """QS2Config测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = create_default_config()
        
        self.assertIsInstance(config, QS2Config)
        self.assertTrue(config.validate())
    
    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        config = QS2Config()
        self.assertTrue(config.validate())
        
        # 无效配置 - 权重和不为1
        config.compatibility_config["compatibility_weights"] = {
            "formal": 0.5,
            "telic": 0.3,
            "agentive": 0.1,
            "constitutive": 0.1,
            "contextual": 0.2  # 总和 > 1
        }
        
        with self.assertRaises(Exception):
            config.validate()
    
    def test_config_serialization(self):
        """测试配置序列化"""
        config = QS2Config()
        
        # 转换为字典
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        
        # 从字典重建
        config2 = QS2Config.from_dict(config_dict)
        self.assertEqual(config.compatibility_config, config2.compatibility_config)
    
    def test_config_file_io(self):
        """测试配置文件读写"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # 保存配置
            config = QS2Config()
            config.save_to_file(config_path)
            
            # 加载配置
            loaded_config = QS2Config.load_from_file(config_path)
            
            self.assertEqual(
                config.compatibility_config,
                loaded_config.compatibility_config
            )


class TestDataSerializer(unittest.TestCase):
    """DataSerializer测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.constructor = QualiaStructureConstructor()
        self.test_context = "测试上下文"
    
    def test_serialize_qualia_structure(self):
        """测试语义结构序列化"""
        entity = {"name": "苹果", "type": "physical_object"}
        structure = self.constructor.construct_qualia_structure(
            entity, self.test_context
        )
        
        # 序列化
        serialized = DataSerializer.serialize_qualia_structure(structure)
        self.assertIsInstance(serialized, dict)
        
        # 反序列化
        deserialized = DataSerializer.deserialize_qualia_structure(serialized)
        self.assertEqual(structure.entity, deserialized.entity)
        self.assertEqual(structure.entity_type, deserialized.entity_type)
    
    def test_file_io(self):
        """测试文件读写"""
        test_data = {"test": "data", "number": 42}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_data.json"
            
            # 保存数据
            DataSerializer.save_to_file(test_data, file_path, "json")
            
            # 加载数据
            loaded_data = DataSerializer.load_from_file(file_path, "json")
            
            self.assertEqual(test_data, loaded_data)


class TestValidationUtils(unittest.TestCase):
    """ValidationUtils测试"""
    
    def test_validate_qualia_structure(self):
        """测试语义结构验证"""
        # 有效结构
        valid_structure = QualiaStructure(
            entity="苹果",
            entity_type="physical_object",
            formal_roles=["水果"],
            telic_roles=["食用"],
            agentive_roles=["自然生长"],
            constitutive_roles=["果皮", "果肉"],
            context_features={},
            confidence=0.8
        )
        
        self.assertTrue(ValidationUtils.validate_qualia_structure(valid_structure))
        
        # 无效结构 - 缺少实体名称
        invalid_structure = QualiaStructure(
            entity="",
            entity_type="physical_object",
            formal_roles=[],
            telic_roles=[],
            agentive_roles=[],
            constitutive_roles=[],
            context_features={},
            confidence=0.8
        )
        
        self.assertFalse(ValidationUtils.validate_qualia_structure(invalid_structure))
    
    def test_validate_enhanced_relation(self):
        """测试增强关系验证"""
        # 创建有效的兼容性结果
        compatibility_result = CompatibilityResult(
            entity1="苹果",
            entity2="橙子",
            overall_score=0.8,
            detailed_scores={},
            compatibility_reasons=[],
            incompatibility_reasons=[],
            confidence=0.7
        )
        
        # 有效关系
        valid_relation = EnhancedRelation(
            entity1="苹果",
            entity2="橙子",
            relation_type=RelationType.SEMANTIC,
            strength=0.8,
            strength_level=RelationStrength.STRONG,
            compatibility_result=compatibility_result,
            confidence=0.7
        )
        
        self.assertTrue(ValidationUtils.validate_enhanced_relation(valid_relation))
        
        # 无效关系 - 强度超出范围
        invalid_relation = EnhancedRelation(
            entity1="苹果",
            entity2="橙子",
            relation_type=RelationType.SEMANTIC,
            strength=1.5,  # 超出范围
            strength_level=RelationStrength.STRONG,
            compatibility_result=compatibility_result,
            confidence=0.7
        )
        
        self.assertFalse(ValidationUtils.validate_enhanced_relation(invalid_relation))


class TestBatchProcessor(unittest.TestCase):
    """BatchProcessor测试"""
    
    def test_process_problems_batch(self):
        """测试批量问题处理"""
        config = create_default_config()
        processor = BatchProcessor(config)
        engine = EnhancedIRDEngine()
        
        problems = [
            "小明有10个苹果，他吃了3个苹果，还剩多少个苹果？",
            "一个长方形的长是8米，宽是5米，面积是多少？"
        ]
        
        results = processor.process_problems_batch(problems, engine)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, DiscoveryResult)
    
    def test_aggregate_results(self):
        """测试结果聚合"""
        config = create_default_config()
        processor = BatchProcessor(config)
        
        # 创建模拟结果
        mock_results = [
            DiscoveryResult(
                relations=[],
                processing_time=1.0,
                entity_count=3,
                total_pairs_evaluated=3,
                high_strength_relations=1,
                statistics={"relation_type_distribution": {"semantic": 2}}
            ),
            DiscoveryResult(
                relations=[],
                processing_time=2.0,
                entity_count=4,
                total_pairs_evaluated=6,
                high_strength_relations=2,
                statistics={"relation_type_distribution": {"functional": 1}}
            )
        ]
        
        aggregated = processor.aggregate_results(mock_results)
        
        self.assertEqual(aggregated["total_problems"], 2)
        self.assertEqual(aggregated["total_processing_time"], 3.0)
        self.assertEqual(aggregated["total_entities"], 7)
        self.assertEqual(aggregated["relation_type_distribution"]["semantic"], 2)
        self.assertEqual(aggregated["relation_type_distribution"]["functional"], 1)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流水线"""
        # 创建配置
        config = create_default_config()
        config.discovery_config["min_strength_threshold"] = 0.2
        
        # 创建引擎
        engine = EnhancedIRDEngine(config.discovery_config)
        
        # 处理问题
        problem = "小明有10个苹果，他吃了3个苹果，还剩多少个苹果？"
        result = engine.discover_relations(problem)
        
        # 验证结果
        self.assertIsInstance(result, DiscoveryResult)
        self.assertGreaterEqual(result.entity_count, 1)
        self.assertGreaterEqual(result.processing_time, 0)
        
        # 验证关系
        for relation in result.relations:
            self.assertTrue(ValidationUtils.validate_enhanced_relation(relation))
        
        # 导出图结构
        graph = engine.export_relations_to_graph(result.relations)
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)
        
        # 获取统计信息
        stats = engine.get_global_stats()
        self.assertIn("total_discoveries", stats)
        self.assertIn("total_relations_found", stats)
    
    def test_error_handling(self):
        """测试错误处理"""
        engine = EnhancedIRDEngine()
        
        # 测试空问题
        result = engine.discover_relations("")
        self.assertIsInstance(result, DiscoveryResult)
        self.assertEqual(result.entity_count, 0)
        
        # 测试无效实体
        result = engine.discover_relations("测试", [])
        self.assertIsInstance(result, DiscoveryResult)
        self.assertEqual(result.entity_count, 0)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)