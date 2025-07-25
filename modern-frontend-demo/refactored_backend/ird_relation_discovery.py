#!/usr/bin/env python3
"""
IRD隐式关系发现模块
基于QS²语义分析结果发现实体间的隐式关系
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from itertools import combinations
from qs2_semantic_analyzer import SemanticEntity, CompatibilityResult, QS2SemanticAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class ImplicitRelation:
    """隐式关系"""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    strength: float
    confidence: float
    evidence: List[str]
    discovery_method: str
    properties: Dict[str, Any]

@dataclass
class RelationNetwork:
    """关系网络"""
    entities: List[SemanticEntity]
    relations: List[ImplicitRelation]
    network_metrics: Dict[str, float]

class IRDRelationDiscovery:
    """IRD隐式关系发现器"""
    
    def __init__(self, qs2_analyzer: QS2SemanticAnalyzer):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.qs2_analyzer = qs2_analyzer
        
        # 关系发现阈值
        self.thresholds = {
            "direct_relation": 0.3,        # 直接关系阈值
            "transitive_relation": 0.2,    # 传递关系阈值
            "context_relation": 0.25       # 上下文关系阈值
        }
        
        # 关系类型定义
        self.relation_types = {
            "ownership": "拥有关系",
            "quantity": "数量关系", 
            "functional": "功能关系",
            "semantic": "语义关系",
            "contextual": "上下文关系",
            "transitive": "传递关系",
            "mathematical": "数学关系",
            "causal": "因果关系"
        }
        
        # 领域知识库
        self.domain_knowledge = self._initialize_domain_knowledge()

    def _initialize_domain_knowledge(self) -> Dict[str, Dict[str, float]]:
        """初始化领域知识库"""
        return {
            # 人-物关系强度
            "person_object": {
                ("person", "object"): 0.8,
                ("person", "苹果"): 0.9,
                ("person", "书"): 0.8,
                ("person", "钱"): 0.9
            },
            # 数量关系强度
            "quantity_relations": {
                ("number", "object"): 0.9,
                ("number", "person"): 0.7
            },
            # 数学运算关系
            "mathematical_relations": {
                "addition": 0.95,
                "subtraction": 0.90,
                "multiplication": 0.85,
                "division": 0.85
            }
        }

    def discover_relations(self, semantic_entities: List[SemanticEntity], 
                         context: str) -> RelationNetwork:
        """
        发现实体间的隐式关系
        
        Args:
            semantic_entities: 语义实体列表
            context: 上下文信息
            
        Returns:
            RelationNetwork: 关系网络
        """
        try:
            self.logger.info(f"开始IRD关系发现，实体数量: {len(semantic_entities)}")
            
            relations = []
            
            # Layer 1: 直接语义关系发现
            direct_relations = self._discover_direct_relations(semantic_entities, context)
            relations.extend(direct_relations)
            
            # Layer 2: 基于上下文的关系增强
            context_relations = self._discover_context_relations(semantic_entities, context)
            relations.extend(context_relations)
            
            # Layer 3: 传递性关系推理
            transitive_relations = self._discover_transitive_relations(relations, semantic_entities)
            relations.extend(transitive_relations)
            
            # 关系过滤和优化
            filtered_relations = self._filter_and_optimize_relations(relations)
            
            # 计算网络指标
            network_metrics = self._calculate_network_metrics(semantic_entities, filtered_relations)
            
            relation_network = RelationNetwork(
                entities=semantic_entities,
                relations=filtered_relations,
                network_metrics=network_metrics
            )
            
            self.logger.info(f"IRD关系发现完成，发现{len(filtered_relations)}个关系")
            return relation_network
            
        except Exception as e:
            self.logger.error(f"IRD关系发现失败: {e}")
            return RelationNetwork(
                entities=semantic_entities,
                relations=[],
                network_metrics={}
            )

    def _discover_direct_relations(self, entities: List[SemanticEntity], 
                                 context: str) -> List[ImplicitRelation]:
        """发现直接语义关系"""
        
        direct_relations = []
        relation_id = 1
        
        for entity1, entity2 in combinations(entities, 2):
            # 计算兼容性
            compatibility = self.qs2_analyzer.compute_compatibility(entity1, entity2, context)
            
            if compatibility.compatibility_score >= self.thresholds["direct_relation"]:
                # 确定关系类型
                relation_type = self._determine_relation_type(entity1, entity2, compatibility, context)
                
                # 生成证据
                evidence = self._generate_evidence(entity1, entity2, compatibility, context)
                
                # 创建关系
                relation = ImplicitRelation(
                    relation_id=f"direct_{relation_id}",
                    source_entity_id=entity1.entity_id,
                    target_entity_id=entity2.entity_id,
                    relation_type=relation_type,
                    strength=compatibility.compatibility_score,
                    confidence=min(entity1.confidence * entity2.confidence, 1.0),
                    evidence=evidence,
                    discovery_method="direct_semantic",
                    properties={
                        "role_similarities": compatibility.role_similarities,
                        "context_boost": compatibility.context_boost
                    }
                )
                
                direct_relations.append(relation)
                relation_id += 1
        
        self.logger.debug(f"发现{len(direct_relations)}个直接关系")
        return direct_relations

    def _discover_context_relations(self, entities: List[SemanticEntity], 
                                   context: str) -> List[ImplicitRelation]:
        """基于上下文发现关系"""
        
        context_relations = []
        relation_id = 1
        
        # 分析上下文中的关系指示词
        relation_indicators = {
            "ownership": ["有", "拥有", "属于", "的"],
            "quantity": ["个", "只", "本", "支", "块", "元"],
            "mathematical": ["一共", "总共", "合计", "加起来", "总数"],
            "functional": ["用于", "用来", "为了", "目的"],
            "causal": ["因为", "所以", "导致", "引起"]
        }
        
        for entity1, entity2 in combinations(entities, 2):
            for rel_type, indicators in relation_indicators.items():
                context_strength = self._calculate_context_strength(
                    entity1, entity2, indicators, context
                )
                
                if context_strength >= self.thresholds["context_relation"]:
                    relation = ImplicitRelation(
                        relation_id=f"context_{relation_id}",
                        source_entity_id=entity1.entity_id,
                        target_entity_id=entity2.entity_id,
                        relation_type=rel_type,
                        strength=context_strength,
                        confidence=0.8,  # 上下文关系的基础置信度
                        evidence=[f"上下文指示: {indicators}"],
                        discovery_method="context_analysis",
                        properties={"context_indicators": indicators}
                    )
                    
                    context_relations.append(relation)
                    relation_id += 1
        
        self.logger.debug(f"发现{len(context_relations)}个上下文关系")
        return context_relations

    def _discover_transitive_relations(self, existing_relations: List[ImplicitRelation], 
                                     entities: List[SemanticEntity]) -> List[ImplicitRelation]:
        """发现传递性关系"""
        
        transitive_relations = []
        relation_id = 1
        
        # 创建实体索引
        entity_dict = {e.entity_id: e for e in entities}
        
        # 构建关系图
        relation_graph = {}
        for relation in existing_relations:
            if relation.source_entity_id not in relation_graph:
                relation_graph[relation.source_entity_id] = []
            relation_graph[relation.source_entity_id].append(relation)
        
        # 寻找传递路径
        for entity_id in relation_graph:
            for relation1 in relation_graph[entity_id]:
                target_id = relation1.target_entity_id
                if target_id in relation_graph:
                    for relation2 in relation_graph[target_id]:
                        # 检查是否可以构成传递关系
                        if self._can_form_transitive_relation(relation1, relation2):
                            transitive_strength = self._calculate_transitive_strength(
                                relation1, relation2
                            )
                            
                            if transitive_strength >= self.thresholds["transitive_relation"]:
                                relation = ImplicitRelation(
                                    relation_id=f"transitive_{relation_id}",
                                    source_entity_id=relation1.source_entity_id,
                                    target_entity_id=relation2.target_entity_id,
                                    relation_type="transitive",
                                    strength=transitive_strength,
                                    confidence=min(relation1.confidence * relation2.confidence, 1.0),
                                    evidence=[f"传递路径: {relation1.relation_id} -> {relation2.relation_id}"],
                                    discovery_method="transitive_inference",
                                    properties={
                                        "intermediate_relation1": relation1.relation_id,
                                        "intermediate_relation2": relation2.relation_id
                                    }
                                )
                                
                                transitive_relations.append(relation)
                                relation_id += 1
        
        self.logger.debug(f"发现{len(transitive_relations)}个传递关系")
        return transitive_relations

    def _determine_relation_type(self, entity1: SemanticEntity, entity2: SemanticEntity,
                               compatibility: CompatibilityResult, context: str) -> str:
        """确定关系类型"""
        
        # 基于实体类型判断
        if entity1.entity_type == "person" and entity2.entity_type == "object":
            return "ownership"
        elif entity1.entity_type == "number" and entity2.entity_type == "object":
            return "quantity"
        elif entity1.entity_type == "person" and entity2.entity_type == "number":
            return "quantity"
        
        # 基于上下文关键词
        if any(word in context for word in ["有", "拥有"]):
            return "ownership"
        elif any(word in context for word in ["一共", "总共", "合计"]):
            return "mathematical"
        elif any(word in context for word in ["用于", "为了"]):
            return "functional"
        
        # 基于Qualia相似度最高的角色
        max_role = max(compatibility.role_similarities.items(), key=lambda x: x[1])
        if max_role[0] == "telic":
            return "functional"
        elif max_role[0] == "formal":
            return "semantic"
        
        return "semantic"  # 默认类型

    def _generate_evidence(self, entity1: SemanticEntity, entity2: SemanticEntity,
                         compatibility: CompatibilityResult, context: str) -> List[str]:
        """生成关系证据"""
        
        evidence = []
        
        # 兼容性证据
        evidence.append(f"语义兼容性: {compatibility.compatibility_score:.3f}")
        
        # Qualia角色证据
        for role, similarity in compatibility.role_similarities.items():
            if similarity > 0.3:
                evidence.append(f"{role}角色相似度: {similarity:.3f}")
        
        # 上下文证据
        if compatibility.context_boost > 0:
            evidence.append(f"上下文增强: {compatibility.context_boost:.3f}")
        
        # 实体类型证据
        evidence.append(f"实体类型组合: {entity1.entity_type}-{entity2.entity_type}")
        
        return evidence

    def _calculate_context_strength(self, entity1: SemanticEntity, entity2: SemanticEntity,
                                  indicators: List[str], context: str) -> float:
        """计算上下文关系强度"""
        
        strength = 0.0
        
        # 检查指示词在上下文中的出现
        for indicator in indicators:
            if indicator in context:
                # 基于实体名称与指示词的距离
                entity1_pos = context.find(entity1.name)
                entity2_pos = context.find(entity2.name)
                indicator_pos = context.find(indicator)
                
                if entity1_pos != -1 and entity2_pos != -1 and indicator_pos != -1:
                    # 计算距离权重
                    avg_entity_pos = (entity1_pos + entity2_pos) / 2
                    distance = abs(avg_entity_pos - indicator_pos)
                    distance_weight = max(0, 1 - distance / 50)  # 距离越近权重越高
                    strength += 0.3 * distance_weight
        
        # 基于实体类型的上下文适配性
        domain_strength = self._get_domain_relation_strength(
            entity1.entity_type, entity2.entity_type
        )
        strength += domain_strength * 0.2
        
        return min(strength, 1.0)

    def _get_domain_relation_strength(self, type1: str, type2: str) -> float:
        """获取领域知识中的关系强度"""
        
        # 检查人-物关系
        if "person_object" in self.domain_knowledge:
            for key, strength in self.domain_knowledge["person_object"].items():
                if (type1, type2) == key or (type2, type1) == key:
                    return strength
        
        # 检查数量关系
        if "quantity_relations" in self.domain_knowledge:
            for key, strength in self.domain_knowledge["quantity_relations"].items():
                if (type1, type2) == key or (type2, type1) == key:
                    return strength
        
        return 0.1  # 默认较低强度

    def _can_form_transitive_relation(self, relation1: ImplicitRelation, 
                                    relation2: ImplicitRelation) -> bool:
        """判断两个关系是否可以形成传递关系"""
        
        # 检查关系连接性
        if relation1.target_entity_id != relation2.source_entity_id:
            return False
        
        # 检查关系类型兼容性
        compatible_types = {
            ("ownership", "quantity"): True,
            ("quantity", "mathematical"): True,
            ("functional", "semantic"): True,
            ("semantic", "contextual"): True
        }
        
        type_pair = (relation1.relation_type, relation2.relation_type)
        return compatible_types.get(type_pair, False)

    def _calculate_transitive_strength(self, relation1: ImplicitRelation, 
                                     relation2: ImplicitRelation) -> float:
        """计算传递关系强度"""
        
        # 几何平均
        geometric_mean = math.sqrt(relation1.strength * relation2.strength)
        
        # 置信度衰减
        confidence_decay = 0.8  # 传递关系的置信度衰减因子
        
        return geometric_mean * confidence_decay

    def _filter_and_optimize_relations(self, relations: List[ImplicitRelation]) -> List[ImplicitRelation]:
        """过滤和优化关系"""
        
        # 去重：相同实体对的关系只保留最强的
        relation_dict = {}
        for relation in relations:
            key = (relation.source_entity_id, relation.target_entity_id)
            reverse_key = (relation.target_entity_id, relation.source_entity_id)
            
            # 使用较小的实体ID作为键，确保一致性
            if key[0] < key[1]:
                final_key = key
            else:
                final_key = reverse_key
            
            if final_key not in relation_dict or relation_dict[final_key].strength < relation.strength:
                relation_dict[final_key] = relation
        
        # 强度过滤
        filtered_relations = [
            rel for rel in relation_dict.values() 
            if rel.strength >= 0.2  # 最低强度阈值
        ]
        
        # 按强度排序
        filtered_relations.sort(key=lambda x: x.strength, reverse=True)
        
        self.logger.debug(f"关系过滤: {len(relations)} -> {len(filtered_relations)}")
        return filtered_relations

    def _calculate_network_metrics(self, entities: List[SemanticEntity], 
                                 relations: List[ImplicitRelation]) -> Dict[str, float]:
        """计算网络指标"""
        
        metrics = {}
        
        # 基础指标
        metrics["entity_count"] = len(entities)
        metrics["relation_count"] = len(relations)
        metrics["relation_density"] = len(relations) / max(len(entities) * (len(entities) - 1) / 2, 1)
        
        # 平均关系强度
        if relations:
            metrics["average_relation_strength"] = sum(r.strength for r in relations) / len(relations)
            metrics["max_relation_strength"] = max(r.strength for r in relations)
            metrics["min_relation_strength"] = min(r.strength for r in relations)
        else:
            metrics["average_relation_strength"] = 0.0
            metrics["max_relation_strength"] = 0.0
            metrics["min_relation_strength"] = 0.0
        
        # 关系类型分布
        relation_types = {}
        for relation in relations:
            relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1
        metrics["relation_type_diversity"] = len(relation_types)
        
        return metrics

# 测试函数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from problem_preprocessor import ProblemPreprocessor
    from qs2_semantic_analyzer import QS2SemanticAnalyzer
    
    # 创建分析器
    preprocessor = ProblemPreprocessor()
    qs2_analyzer = QS2SemanticAnalyzer()
    ird_discovery = IRDRelationDiscovery(qs2_analyzer)
    
    # 测试问题
    test_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    print(f"测试问题: {test_problem}")
    print("="*60)
    
    # 预处理
    processed = preprocessor.preprocess(test_problem)
    print(f"预处理实体: {[e.name for e in processed.entities]}")
    
    # QS²语义分析
    semantic_entities = qs2_analyzer.analyze_semantics(processed)
    print(f"语义实体数量: {len(semantic_entities)}")
    
    # IRD关系发现
    relation_network = ird_discovery.discover_relations(semantic_entities, test_problem)
    
    print(f"\n关系网络分析:")
    print(f"实体数量: {len(relation_network.entities)}")
    print(f"关系数量: {len(relation_network.relations)}")
    
    # 显示发现的关系
    print(f"\n发现的关系:")
    for relation in relation_network.relations:
        source_name = next(e.name for e in semantic_entities if e.entity_id == relation.source_entity_id)
        target_name = next(e.name for e in semantic_entities if e.entity_id == relation.target_entity_id)
        print(f"  {source_name} --[{relation.relation_type}]--> {target_name}")
        print(f"    强度: {relation.strength:.3f}, 置信度: {relation.confidence:.3f}")
        print(f"    证据: {', '.join(relation.evidence[:2])}")
        print(f"    发现方法: {relation.discovery_method}")
    
    # 显示网络指标
    print(f"\n网络指标:")
    for metric, value in relation_network.network_metrics.items():
        print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")