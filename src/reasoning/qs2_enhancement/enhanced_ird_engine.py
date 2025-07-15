"""
增强型隐式关系发现引擎 - QS²算法集成
==============================

集成QualiaStructureConstructor和CompatibilityEngine，
实现基于语义结构的增强隐式关系发现。

核心功能：
1. 基于QS²算法的语义结构构建
2. 多维度兼容性计算
3. 增强的隐式关系发现
4. 关系强度评估和排序
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入QS²组件
from .qualia_constructor import QualiaStructureConstructor, QualiaStructure
from .compatibility_engine import CompatibilityEngine, CompatibilityResult

# 尝试导入现有系统组件
try:
    from ..implicit_relation_discovery import ImplicitRelationDiscovery
    from ...models.structures import Entity, Problem, Relation
    from ...processors.nlp_processor import NLPProcessor
except ImportError:
    # 如果现有组件不可用，提供基础替代
    ImplicitRelationDiscovery = None
    Entity = Dict[str, Any]
    Problem = Dict[str, Any]
    Relation = Dict[str, Any]
    NLPProcessor = None

logger = logging.getLogger(__name__)


class RelationStrength(Enum):
    """关系强度级别"""
    VERY_WEAK = "very_weak"     # 0.0 - 0.2
    WEAK = "weak"               # 0.2 - 0.4
    MODERATE = "moderate"       # 0.4 - 0.6
    STRONG = "strong"           # 0.6 - 0.8
    VERY_STRONG = "very_strong" # 0.8 - 1.0


class RelationType(Enum):
    """关系类型"""
    SEMANTIC = "semantic"           # 语义关系
    FUNCTIONAL = "functional"       # 功能关系
    CONTEXTUAL = "contextual"       # 上下文关系
    STRUCTURAL = "structural"       # 结构关系
    QUANTITATIVE = "quantitative"   # 数量关系


@dataclass
class EnhancedRelation:
    """增强关系结构"""
    entity1: str
    entity2: str
    relation_type: RelationType
    strength: float
    strength_level: RelationStrength
    compatibility_result: CompatibilityResult
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    discovery_method: str = "qs2_enhanced"
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entity1": self.entity1,
            "entity2": self.entity2,
            "relation_type": self.relation_type.value,
            "strength": self.strength,
            "strength_level": self.strength_level.value,
            "compatibility_result": self.compatibility_result.to_dict(),
            "evidence": self.evidence,
            "confidence": self.confidence,
            "discovery_method": self.discovery_method,
            "timestamp": self.timestamp
        }


@dataclass
class DiscoveryResult:
    """发现结果"""
    relations: List[EnhancedRelation]
    processing_time: float
    entity_count: int
    total_pairs_evaluated: int
    high_strength_relations: int
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "relations": [r.to_dict() for r in self.relations],
            "processing_time": self.processing_time,
            "entity_count": self.entity_count,
            "total_pairs_evaluated": self.total_pairs_evaluated,
            "high_strength_relations": self.high_strength_relations,
            "statistics": self.statistics
        }


class EnhancedIRDEngine:
    """增强型隐式关系发现引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化QS²组件
        self.qualia_constructor = QualiaStructureConstructor(
            self.config.get("qualia_config", {})
        )
        self.compatibility_engine = CompatibilityEngine(
            self.config.get("compatibility_config", {})
        )
        
        # 初始化原始IRD引擎（如果可用）
        self.original_ird = ImplicitRelationDiscovery() if ImplicitRelationDiscovery else None
        
        # 关系发现参数
        self.min_strength_threshold = self.config.get("min_strength_threshold", 0.3)
        self.max_relations_per_entity = self.config.get("max_relations_per_entity", 10)
        self.enable_parallel_processing = self.config.get("enable_parallel_processing", True)
        self.max_workers = self.config.get("max_workers", 4)
        
        # 关系类型权重
        self.relation_type_weights = self.config.get("relation_type_weights", {
            RelationType.SEMANTIC: 0.3,
            RelationType.FUNCTIONAL: 0.35,
            RelationType.CONTEXTUAL: 0.15,
            RelationType.STRUCTURAL: 0.1,
            RelationType.QUANTITATIVE: 0.1
        })
        
        # 统计信息
        self.stats = {
            "total_discoveries": 0,
            "total_relations_found": 0,
            "average_processing_time": 0.0,
            "entity_type_distribution": {},
            "relation_type_distribution": {},
            "strength_distribution": {}
        }
        
        self.logger.info("增强型隐式关系发现引擎初始化完成")
    
    def discover_relations(
        self, 
        problem: Union[Problem, str], 
        entities: Optional[List[Entity]] = None,
        context_weight: float = 1.0
    ) -> DiscoveryResult:
        """
        发现实体间的隐式关系
        
        Args:
            problem: 数学问题或问题文本
            entities: 实体列表（可选，如果不提供将自动提取）
            context_weight: 上下文权重
            
        Returns:
            DiscoveryResult: 发现结果
        """
        start_time = time.time()
        
        try:
            # 标准化输入
            problem_text = self._normalize_problem(problem)
            entities = entities or self._extract_entities(problem_text)
            
            self.logger.info(f"开始关系发现，实体数量: {len(entities)}")
            
            # 构建语义结构
            qualia_structures = self._build_qualia_structures(entities, problem_text)
            
            # 发现关系
            relations = self._discover_enhanced_relations(
                qualia_structures, problem_text, context_weight
            )
            
            # 后处理和优化
            relations = self._post_process_relations(relations)
            
            # 计算统计信息
            processing_time = time.time() - start_time
            statistics = self._calculate_statistics(relations, qualia_structures)
            
            # 构建结果
            result = DiscoveryResult(
                relations=relations,
                processing_time=processing_time,
                entity_count=len(entities),
                total_pairs_evaluated=len(qualia_structures) * (len(qualia_structures) - 1) // 2,
                high_strength_relations=len([r for r in relations if r.strength > 0.7]),
                statistics=statistics
            )
            
            # 更新全局统计
            self._update_global_stats(result)
            
            self.logger.info(
                f"关系发现完成: {len(relations)} 个关系，"
                f"处理时间: {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"关系发现失败: {str(e)}")
            return DiscoveryResult(
                relations=[],
                processing_time=time.time() - start_time,
                entity_count=len(entities) if entities else 0,
                total_pairs_evaluated=0,
                high_strength_relations=0,
                statistics={"error": str(e)}
            )
    
    def _normalize_problem(self, problem: Union[Problem, str]) -> str:
        """标准化问题输入"""
        if isinstance(problem, str):
            return problem
        elif isinstance(problem, dict):
            return problem.get("text", problem.get("question", str(problem)))
        else:
            return str(problem)
    
    def _extract_entities(self, problem_text: str) -> List[Entity]:
        """提取实体（使用现有系统或简单规则）"""
        if self.original_ird and hasattr(self.original_ird, 'extract_entities'):
            return self.original_ird.extract_entities(problem_text)
        
        # 简单实体提取
        entities = []
        
        # 提取数字
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
        for num in numbers:
            entities.append({"name": num, "type": "number"})
        
        # 提取常见实体
        common_entities = [
            "苹果", "书", "车", "钱", "米", "小时", "分钟", "元", "个", "只", "本",
            "面积", "周长", "速度", "时间", "距离", "价格", "成本", "利润"
        ]
        
        for entity in common_entities:
            if entity in problem_text:
                entities.append({"name": entity, "type": "general"})
        
        return entities
    
    def _build_qualia_structures(
        self, 
        entities: List[Entity], 
        context: str
    ) -> List[QualiaStructure]:
        """构建语义结构"""
        structures = []
        
        if self.enable_parallel_processing and len(entities) > 5:
            # 并行构建
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.qualia_constructor.construct_qualia_structure,
                        entity, context
                    ): entity for entity in entities
                }
                
                for future in as_completed(futures):
                    try:
                        structure = future.result()
                        structures.append(structure)
                    except Exception as e:
                        entity = futures[future]
                        self.logger.error(f"构建实体 {entity} 的语义结构失败: {e}")
        else:
            # 串行构建
            for entity in entities:
                try:
                    structure = self.qualia_constructor.construct_qualia_structure(
                        entity, context
                    )
                    structures.append(structure)
                except Exception as e:
                    self.logger.error(f"构建实体 {entity} 的语义结构失败: {e}")
        
        self.logger.debug(f"成功构建 {len(structures)} 个语义结构")
        return structures
    
    def _discover_enhanced_relations(
        self, 
        structures: List[QualiaStructure], 
        context: str,
        context_weight: float
    ) -> List[EnhancedRelation]:
        """发现增强关系"""
        relations = []
        
        # 计算所有实体对的兼容性
        compatibility_results = []
        
        if self.enable_parallel_processing and len(structures) > 10:
            # 并行计算兼容性
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for i in range(len(structures)):
                    for j in range(i + 1, len(structures)):
                        future = executor.submit(
                            self.compatibility_engine.compute_detailed_compatibility,
                            structures[i], structures[j], context_weight
                        )
                        futures.append((future, i, j))
                
                for future, i, j in futures:
                    try:
                        result = future.result()
                        compatibility_results.append((i, j, result))
                    except Exception as e:
                        self.logger.error(f"计算兼容性失败 ({i}, {j}): {e}")
        else:
            # 串行计算
            for i in range(len(structures)):
                for j in range(i + 1, len(structures)):
                    try:
                        result = self.compatibility_engine.compute_detailed_compatibility(
                            structures[i], structures[j], context_weight
                        )
                        compatibility_results.append((i, j, result))
                    except Exception as e:
                        self.logger.error(f"计算兼容性失败 ({i}, {j}): {e}")
        
        # 基于兼容性结果构建关系
        for i, j, compatibility_result in compatibility_results:
            if compatibility_result.overall_score >= self.min_strength_threshold:
                # 确定关系类型
                relation_type = self._determine_relation_type(
                    structures[i], structures[j], compatibility_result
                )
                
                # 计算关系强度
                strength = self._calculate_relation_strength(
                    compatibility_result, relation_type
                )
                
                # 确定强度级别
                strength_level = self._determine_strength_level(strength)
                
                # 生成证据
                evidence = self._generate_evidence(
                    structures[i], structures[j], compatibility_result, context
                )
                
                # 计算置信度
                confidence = self._calculate_relation_confidence(
                    structures[i], structures[j], compatibility_result
                )
                
                # 创建关系
                relation = EnhancedRelation(
                    entity1=structures[i].entity,
                    entity2=structures[j].entity,
                    relation_type=relation_type,
                    strength=strength,
                    strength_level=strength_level,
                    compatibility_result=compatibility_result,
                    evidence=evidence,
                    confidence=confidence
                )
                
                relations.append(relation)
        
        self.logger.debug(f"发现 {len(relations)} 个增强关系")
        return relations
    
    def _determine_relation_type(
        self, 
        struct1: QualiaStructure, 
        struct2: QualiaStructure,
        compatibility: CompatibilityResult
    ) -> RelationType:
        """确定关系类型"""
        scores = compatibility.detailed_scores
        
        # 基于最高分数确定主要关系类型
        if scores.get("telic", 0) > 0.6:
            return RelationType.FUNCTIONAL
        elif scores.get("formal", 0) > 0.6:
            return RelationType.SEMANTIC
        elif scores.get("contextual", 0) > 0.6:
            return RelationType.CONTEXTUAL
        elif scores.get("constitutive", 0) > 0.6:
            return RelationType.STRUCTURAL
        else:
            # 基于实体类型推断
            if struct1.entity_type == "number" and struct2.entity_type == "number":
                return RelationType.QUANTITATIVE
            else:
                return RelationType.SEMANTIC
    
    def _calculate_relation_strength(
        self, 
        compatibility: CompatibilityResult, 
        relation_type: RelationType
    ) -> float:
        """计算关系强度"""
        base_strength = compatibility.overall_score
        
        # 基于关系类型调整权重
        type_weight = self.relation_type_weights.get(relation_type, 1.0)
        
        # 基于置信度调整
        confidence_factor = compatibility.confidence
        
        # 综合强度
        strength = base_strength * type_weight * confidence_factor
        
        return min(1.0, max(0.0, strength))
    
    def _determine_strength_level(self, strength: float) -> RelationStrength:
        """确定强度级别"""
        if strength < 0.2:
            return RelationStrength.VERY_WEAK
        elif strength < 0.4:
            return RelationStrength.WEAK
        elif strength < 0.6:
            return RelationStrength.MODERATE
        elif strength < 0.8:
            return RelationStrength.STRONG
        else:
            return RelationStrength.VERY_STRONG
    
    def _generate_evidence(
        self, 
        struct1: QualiaStructure, 
        struct2: QualiaStructure,
        compatibility: CompatibilityResult,
        context: str
    ) -> List[str]:
        """生成关系证据"""
        evidence = []
        
        # 添加兼容性原因作为证据
        evidence.extend(compatibility.compatibility_reasons)
        
        # 添加语义结构证据
        if struct1.entity_type == struct2.entity_type:
            evidence.append(f"实体类型相同: {struct1.entity_type}")
        
        # 添加上下文证据
        if ("problem_type" in struct1.context_features and 
            "problem_type" in struct2.context_features and
            struct1.context_features["problem_type"] == struct2.context_features["problem_type"]):
            evidence.append(f"问题类型相同: {struct1.context_features['problem_type']}")
        
        # 添加共现证据
        if struct1.entity in context and struct2.entity in context:
            evidence.append("实体在同一上下文中出现")
        
        return evidence
    
    def _calculate_relation_confidence(
        self, 
        struct1: QualiaStructure, 
        struct2: QualiaStructure,
        compatibility: CompatibilityResult
    ) -> float:
        """计算关系置信度"""
        # 基于兼容性置信度
        base_confidence = compatibility.confidence
        
        # 基于语义结构完整性
        struct1_completeness = struct1.confidence
        struct2_completeness = struct2.confidence
        
        # 基于证据数量
        evidence_count = len(compatibility.compatibility_reasons)
        evidence_factor = min(1.0, evidence_count / 3.0)
        
        # 综合置信度
        confidence = (base_confidence * 0.4 + 
                     struct1_completeness * 0.25 + 
                     struct2_completeness * 0.25 + 
                     evidence_factor * 0.1)
        
        return min(1.0, max(0.0, confidence))
    
    def _post_process_relations(self, relations: List[EnhancedRelation]) -> List[EnhancedRelation]:
        """后处理关系"""
        # 按强度排序
        relations.sort(key=lambda r: r.strength, reverse=True)
        
        # 限制每个实体的关系数量
        entity_relation_count = {}
        filtered_relations = []
        
        for relation in relations:
            entity1_count = entity_relation_count.get(relation.entity1, 0)
            entity2_count = entity_relation_count.get(relation.entity2, 0)
            
            if (entity1_count < self.max_relations_per_entity and 
                entity2_count < self.max_relations_per_entity):
                filtered_relations.append(relation)
                entity_relation_count[relation.entity1] = entity1_count + 1
                entity_relation_count[relation.entity2] = entity2_count + 1
        
        # 去重（基于实体对）
        seen_pairs = set()
        unique_relations = []
        
        for relation in filtered_relations:
            pair = tuple(sorted([relation.entity1, relation.entity2]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_relations.append(relation)
        
        self.logger.debug(f"后处理完成: {len(unique_relations)} 个关系")
        return unique_relations
    
    def _calculate_statistics(
        self, 
        relations: List[EnhancedRelation], 
        structures: List[QualiaStructure]
    ) -> Dict[str, Any]:
        """计算统计信息"""
        stats = {}
        
        # 关系类型分布
        relation_type_count = {}
        for relation in relations:
            relation_type = relation.relation_type.value
            relation_type_count[relation_type] = relation_type_count.get(relation_type, 0) + 1
        stats["relation_type_distribution"] = relation_type_count
        
        # 强度分布
        strength_levels = {}
        for relation in relations:
            level = relation.strength_level.value
            strength_levels[level] = strength_levels.get(level, 0) + 1
        stats["strength_distribution"] = strength_levels
        
        # 实体类型分布
        entity_types = {}
        for struct in structures:
            entity_type = struct.entity_type
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        stats["entity_type_distribution"] = entity_types
        
        # 平均值统计
        if relations:
            avg_strength = sum(r.strength for r in relations) / len(relations)
            avg_confidence = sum(r.confidence for r in relations) / len(relations)
            stats["average_strength"] = avg_strength
            stats["average_confidence"] = avg_confidence
        
        # 结构完整性统计
        if structures:
            avg_structure_confidence = sum(s.confidence for s in structures) / len(structures)
            stats["average_structure_confidence"] = avg_structure_confidence
        
        return stats
    
    def _update_global_stats(self, result: DiscoveryResult):
        """更新全局统计信息"""
        self.stats["total_discoveries"] += 1
        self.stats["total_relations_found"] += len(result.relations)
        
        # 更新平均处理时间
        current_avg = self.stats["average_processing_time"]
        total_discoveries = self.stats["total_discoveries"]
        new_avg = ((current_avg * (total_discoveries - 1) + result.processing_time) / total_discoveries)
        self.stats["average_processing_time"] = new_avg
        
        # 更新分布统计
        for relation_type, count in result.statistics.get("relation_type_distribution", {}).items():
            self.stats.setdefault("relation_type_distribution", {})[relation_type] = \
                self.stats["relation_type_distribution"].get(relation_type, 0) + count
        
        for entity_type, count in result.statistics.get("entity_type_distribution", {}).items():
            self.stats.setdefault("entity_type_distribution", {})[entity_type] = \
                self.stats["entity_type_distribution"].get(entity_type, 0) + count
    
    def get_relation_by_entities(
        self, 
        entity1: str, 
        entity2: str, 
        relations: List[EnhancedRelation]
    ) -> Optional[EnhancedRelation]:
        """根据实体获取关系"""
        for relation in relations:
            if ((relation.entity1 == entity1 and relation.entity2 == entity2) or
                (relation.entity1 == entity2 and relation.entity2 == entity1)):
                return relation
        return None
    
    def filter_relations_by_strength(
        self, 
        relations: List[EnhancedRelation], 
        min_strength: float
    ) -> List[EnhancedRelation]:
        """按强度过滤关系"""
        return [r for r in relations if r.strength >= min_strength]
    
    def filter_relations_by_type(
        self, 
        relations: List[EnhancedRelation], 
        relation_type: RelationType
    ) -> List[EnhancedRelation]:
        """按类型过滤关系"""
        return [r for r in relations if r.relation_type == relation_type]
    
    def get_entity_relations(
        self, 
        entity: str, 
        relations: List[EnhancedRelation]
    ) -> List[EnhancedRelation]:
        """获取实体的所有关系"""
        return [r for r in relations if r.entity1 == entity or r.entity2 == entity]
    
    def export_relations_to_graph(
        self, 
        relations: List[EnhancedRelation]
    ) -> Dict[str, Any]:
        """导出关系为图结构"""
        nodes = set()
        edges = []
        
        for relation in relations:
            nodes.add(relation.entity1)
            nodes.add(relation.entity2)
            edges.append({
                "source": relation.entity1,
                "target": relation.entity2,
                "weight": relation.strength,
                "type": relation.relation_type.value,
                "evidence": relation.evidence
            })
        
        return {
            "nodes": [{"id": node} for node in nodes],
            "edges": edges
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_discoveries": 0,
            "total_relations_found": 0,
            "average_processing_time": 0.0,
            "entity_type_distribution": {},
            "relation_type_distribution": {},
            "strength_distribution": {}
        }
        self.logger.info("增强IRD引擎统计信息已重置")
    
    def configure_thresholds(
        self, 
        min_strength: Optional[float] = None,
        max_relations_per_entity: Optional[int] = None
    ):
        """配置阈值参数"""
        if min_strength is not None:
            self.min_strength_threshold = min_strength
        if max_relations_per_entity is not None:
            self.max_relations_per_entity = max_relations_per_entity
        
        self.logger.info(f"阈值配置更新: min_strength={self.min_strength_threshold}, "
                        f"max_relations={self.max_relations_per_entity}")