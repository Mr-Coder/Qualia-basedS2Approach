"""
语义兼容性计算引擎 - QS²算法核心组件
=================================

基于Qualia Structure计算实体间的语义兼容性，
为隐式关系发现提供语义相似度评估。

核心功能：
1. 计算两个语义结构之间的兼容性分数
2. 多维度语义相似度评估
3. 上下文感知的兼容性计算
4. 支持批量兼容性计算
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from .qualia_constructor import QualiaStructure, QualiaRole

logger = logging.getLogger(__name__)


class CompatibilityType(Enum):
    """兼容性类型"""
    FORMAL = "formal"           # 形式兼容性
    TELIC = "telic"            # 功能兼容性
    AGENTIVE = "agentive"      # 起源兼容性
    CONSTITUTIVE = "constitutive"  # 构成兼容性
    CONTEXTUAL = "contextual"   # 上下文兼容性
    OVERALL = "overall"         # 整体兼容性


@dataclass
class CompatibilityResult:
    """兼容性计算结果"""
    entity1: str
    entity2: str
    overall_score: float
    detailed_scores: Dict[str, float]
    compatibility_reasons: List[str]
    incompatibility_reasons: List[str]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entity1": self.entity1,
            "entity2": self.entity2,
            "overall_score": self.overall_score,
            "detailed_scores": self.detailed_scores,
            "compatibility_reasons": self.compatibility_reasons,
            "incompatibility_reasons": self.incompatibility_reasons,
            "confidence": self.confidence
        }


class CompatibilityEngine:
    """语义兼容性计算引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 兼容性权重配置
        self.compatibility_weights = self.config.get("compatibility_weights", {
            "formal": 0.25,
            "telic": 0.35,
            "agentive": 0.15,
            "constitutive": 0.15,
            "contextual": 0.10
        })
        
        # 兼容性阈值
        self.compatibility_threshold = self.config.get("compatibility_threshold", 0.6)
        
        # 语义相似度计算方法
        self.similarity_method = self.config.get("similarity_method", "jaccard")
        
        # 初始化语义相似度规则
        self.semantic_similarity_rules = self._initialize_semantic_rules()
        
        # 统计信息
        self.stats = {
            "total_computations": 0,
            "high_compatibility_pairs": 0,
            "average_compatibility": 0.0,
            "computation_time_stats": {}
        }
        
        self.logger.info("语义兼容性计算引擎初始化完成")
    
    def compute_compatibility(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure,
        context_weight: float = 1.0
    ) -> float:
        """
        计算两个语义结构的兼容性分数
        
        Args:
            structure1: 第一个语义结构
            structure2: 第二个语义结构
            context_weight: 上下文权重
            
        Returns:
            float: 兼容性分数 (0.0 - 1.0)
        """
        try:
            # 计算详细兼容性结果
            detailed_result = self.compute_detailed_compatibility(
                structure1, structure2, context_weight
            )
            
            # 更新统计信息
            self._update_stats(detailed_result.overall_score)
            
            return detailed_result.overall_score
            
        except Exception as e:
            self.logger.error(f"兼容性计算失败: {str(e)}")
            return 0.0
    
    def compute_detailed_compatibility(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure,
        context_weight: float = 1.0
    ) -> CompatibilityResult:
        """
        计算详细的兼容性结果
        
        Args:
            structure1: 第一个语义结构
            structure2: 第二个语义结构
            context_weight: 上下文权重
            
        Returns:
            CompatibilityResult: 详细兼容性结果
        """
        try:
            # 计算各维度兼容性
            formal_score = self._compute_formal_compatibility(structure1, structure2)
            telic_score = self._compute_telic_compatibility(structure1, structure2)
            agentive_score = self._compute_agentive_compatibility(structure1, structure2)
            constitutive_score = self._compute_constitutive_compatibility(structure1, structure2)
            contextual_score = self._compute_contextual_compatibility(structure1, structure2)
            
            # 计算加权总分
            overall_score = (
                formal_score * self.compatibility_weights["formal"] +
                telic_score * self.compatibility_weights["telic"] +
                agentive_score * self.compatibility_weights["agentive"] +
                constitutive_score * self.compatibility_weights["constitutive"] +
                contextual_score * self.compatibility_weights["contextual"] * context_weight
            )
            
            # 构建详细分数字典
            detailed_scores = {
                "formal": formal_score,
                "telic": telic_score,
                "agentive": agentive_score,
                "constitutive": constitutive_score,
                "contextual": contextual_score,
                "overall": overall_score
            }
            
            # 生成兼容性和不兼容性原因
            compatibility_reasons = self._generate_compatibility_reasons(
                structure1, structure2, detailed_scores
            )
            incompatibility_reasons = self._generate_incompatibility_reasons(
                structure1, structure2, detailed_scores
            )
            
            # 计算置信度
            confidence = self._calculate_compatibility_confidence(
                structure1, structure2, detailed_scores
            )
            
            result = CompatibilityResult(
                entity1=structure1.entity,
                entity2=structure2.entity,
                overall_score=overall_score,
                detailed_scores=detailed_scores,
                compatibility_reasons=compatibility_reasons,
                incompatibility_reasons=incompatibility_reasons,
                confidence=confidence
            )
            
            self.logger.debug(
                f"兼容性计算完成: {structure1.entity} <-> {structure2.entity}, "
                f"总分: {overall_score:.3f}, 置信度: {confidence:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"详细兼容性计算失败: {str(e)}")
            return CompatibilityResult(
                entity1=structure1.entity,
                entity2=structure2.entity,
                overall_score=0.0,
                detailed_scores={},
                compatibility_reasons=[],
                incompatibility_reasons=[f"计算失败: {str(e)}"],
                confidence=0.0
            )
    
    def _compute_formal_compatibility(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure
    ) -> float:
        """计算形式兼容性"""
        if not structure1.formal_roles or not structure2.formal_roles:
            return 0.0
        
        # 计算Jaccard相似度
        set1 = set(structure1.formal_roles)
        set2 = set(structure2.formal_roles)
        
        if self.similarity_method == "jaccard":
            return self._jaccard_similarity(set1, set2)
        elif self.similarity_method == "cosine":
            return self._cosine_similarity(set1, set2)
        else:
            return self._jaccard_similarity(set1, set2)
    
    def _compute_telic_compatibility(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure
    ) -> float:
        """计算功能兼容性"""
        if not structure1.telic_roles or not structure2.telic_roles:
            return 0.0
        
        set1 = set(structure1.telic_roles)
        set2 = set(structure2.telic_roles)
        
        # 功能兼容性更重要，使用语义相似度增强
        base_similarity = self._jaccard_similarity(set1, set2)
        
        # 检查语义相似的功能
        semantic_bonus = self._compute_semantic_similarity(
            structure1.telic_roles, structure2.telic_roles, "telic"
        )
        
        return min(1.0, base_similarity + semantic_bonus * 0.3)
    
    def _compute_agentive_compatibility(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure
    ) -> float:
        """计算起源兼容性"""
        if not structure1.agentive_roles or not structure2.agentive_roles:
            return 0.0
        
        set1 = set(structure1.agentive_roles)
        set2 = set(structure2.agentive_roles)
        
        return self._jaccard_similarity(set1, set2)
    
    def _compute_constitutive_compatibility(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure
    ) -> float:
        """计算构成兼容性"""
        if not structure1.constitutive_roles or not structure2.constitutive_roles:
            return 0.0
        
        set1 = set(structure1.constitutive_roles)
        set2 = set(structure2.constitutive_roles)
        
        return self._jaccard_similarity(set1, set2)
    
    def _compute_contextual_compatibility(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure
    ) -> float:
        """计算上下文兼容性"""
        if not structure1.context_features or not structure2.context_features:
            return 0.0
        
        # 比较上下文特征
        compatibility_score = 0.0
        total_features = 0
        
        # 比较问题类型
        if ("problem_type" in structure1.context_features and 
            "problem_type" in structure2.context_features):
            total_features += 1
            if (structure1.context_features["problem_type"] == 
                structure2.context_features["problem_type"]):
                compatibility_score += 1.0
        
        # 比较相关动词
        if ("related_verbs" in structure1.context_features and 
            "related_verbs" in structure2.context_features):
            total_features += 1
            verbs1 = set(structure1.context_features["related_verbs"])
            verbs2 = set(structure2.context_features["related_verbs"])
            if verbs1 and verbs2:
                compatibility_score += self._jaccard_similarity(verbs1, verbs2)
        
        # 比较周围词汇
        if ("surrounding_words" in structure1.context_features and 
            "surrounding_words" in structure2.context_features):
            total_features += 1
            words1 = set(structure1.context_features["surrounding_words"])
            words2 = set(structure2.context_features["surrounding_words"])
            if words1 and words2:
                compatibility_score += self._jaccard_similarity(words1, words2)
        
        return compatibility_score / total_features if total_features > 0 else 0.0
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """计算Jaccard相似度"""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """计算余弦相似度（简化版）"""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        magnitude1 = math.sqrt(len(set1))
        magnitude2 = math.sqrt(len(set2))
        
        return intersection / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0.0
    
    def _compute_semantic_similarity(
        self, 
        roles1: List[str], 
        roles2: List[str], 
        role_type: str
    ) -> float:
        """计算语义相似度"""
        similarity_score = 0.0
        
        # 获取该角色类型的语义相似度规则
        rules = self.semantic_similarity_rules.get(role_type, {})
        
        for role1 in roles1:
            for role2 in roles2:
                # 检查直接匹配
                if role1 == role2:
                    similarity_score += 1.0
                    continue
                
                # 检查语义相似度规则
                for similar_group in rules.get("similar_groups", []):
                    if role1 in similar_group and role2 in similar_group:
                        similarity_score += 0.7
                        break
                
                # 检查上下位关系
                for hypernym, hyponyms in rules.get("hypernym_relations", {}).items():
                    if ((role1 == hypernym and role2 in hyponyms) or 
                        (role2 == hypernym and role1 in hyponyms)):
                        similarity_score += 0.5
                        break
        
        # 标准化分数
        max_possible = len(roles1) * len(roles2)
        return similarity_score / max_possible if max_possible > 0 else 0.0
    
    def _generate_compatibility_reasons(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure, 
        scores: Dict[str, float]
    ) -> List[str]:
        """生成兼容性原因"""
        reasons = []
        
        # 基于各维度得分生成原因
        if scores["formal"] > 0.6:
            common_formal = set(structure1.formal_roles).intersection(
                set(structure2.formal_roles)
            )
            if common_formal:
                reasons.append(f"形式特征相似: {', '.join(common_formal)}")
        
        if scores["telic"] > 0.6:
            common_telic = set(structure1.telic_roles).intersection(
                set(structure2.telic_roles)
            )
            if common_telic:
                reasons.append(f"功能目的相似: {', '.join(common_telic)}")
        
        if scores["agentive"] > 0.6:
            common_agentive = set(structure1.agentive_roles).intersection(
                set(structure2.agentive_roles)
            )
            if common_agentive:
                reasons.append(f"起源方式相似: {', '.join(common_agentive)}")
        
        if scores["constitutive"] > 0.6:
            common_constitutive = set(structure1.constitutive_roles).intersection(
                set(structure2.constitutive_roles)
            )
            if common_constitutive:
                reasons.append(f"构成成分相似: {', '.join(common_constitutive)}")
        
        if scores["contextual"] > 0.6:
            reasons.append("上下文环境相似")
        
        return reasons
    
    def _generate_incompatibility_reasons(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure, 
        scores: Dict[str, float]
    ) -> List[str]:
        """生成不兼容性原因"""
        reasons = []
        
        # 基于各维度得分生成原因
        if scores["formal"] < 0.3:
            reasons.append("形式特征差异较大")
        
        if scores["telic"] < 0.3:
            reasons.append("功能目的不同")
        
        if scores["agentive"] < 0.3:
            reasons.append("起源方式不同")
        
        if scores["constitutive"] < 0.3:
            reasons.append("构成成分不同")
        
        if scores["contextual"] < 0.3:
            reasons.append("上下文环境不匹配")
        
        # 检查实体类型不兼容
        if structure1.entity_type != structure2.entity_type:
            incompatible_types = [
                ("number", "physical_object"),
                ("abstract_concept", "physical_object"),
                ("unit", "math_concept")
            ]
            type_pair = (structure1.entity_type, structure2.entity_type)
            if type_pair in incompatible_types or type_pair[::-1] in incompatible_types:
                reasons.append(f"实体类型不兼容: {structure1.entity_type} vs {structure2.entity_type}")
        
        return reasons
    
    def _calculate_compatibility_confidence(
        self, 
        structure1: QualiaStructure, 
        structure2: QualiaStructure, 
        scores: Dict[str, float]
    ) -> float:
        """计算兼容性置信度"""
        # 基于语义结构的完整性
        completeness1 = self._calculate_structure_completeness(structure1)
        completeness2 = self._calculate_structure_completeness(structure2)
        
        # 基于分数的一致性
        score_values = [scores[key] for key in ["formal", "telic", "agentive", "constitutive"]]
        score_variance = sum((s - sum(score_values)/len(score_values))**2 for s in score_values) / len(score_values)
        consistency = 1.0 - min(1.0, score_variance)
        
        # 综合置信度
        confidence = (completeness1 + completeness2 + consistency) / 3.0
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_structure_completeness(self, structure: QualiaStructure) -> float:
        """计算语义结构的完整性"""
        role_counts = [
            len(structure.formal_roles),
            len(structure.telic_roles),
            len(structure.agentive_roles),
            len(structure.constitutive_roles)
        ]
        
        # 非空角色数量
        non_empty_roles = sum(1 for count in role_counts if count > 0)
        
        # 总角色数量
        total_roles = sum(role_counts)
        
        # 结构置信度
        structure_confidence = structure.confidence
        
        # 综合完整性
        completeness = (non_empty_roles / 4.0) * 0.5 + (min(1.0, total_roles / 8.0)) * 0.3 + structure_confidence * 0.2
        
        return min(1.0, max(0.0, completeness))
    
    def _initialize_semantic_rules(self) -> Dict[str, Dict[str, Any]]:
        """初始化语义相似度规则"""
        return {
            "formal": {
                "similar_groups": [
                    ["整数", "数字", "数值"],
                    ["长度单位", "距离单位", "度量单位"],
                    ["水果", "食物", "可食用"],
                    ["学习用品", "文具", "工具"]
                ],
                "hypernym_relations": {
                    "数字": ["整数", "小数", "正数", "负数"],
                    "单位": ["长度单位", "时间单位", "货币单位"],
                    "物体": ["水果", "工具", "交通工具"]
                }
            },
            "telic": {
                "similar_groups": [
                    ["计算", "运算", "求解"],
                    ["测量", "度量", "衡量"],
                    ["比较", "对比", "评估"],
                    ["食用", "消费", "使用"]
                ],
                "hypernym_relations": {
                    "数学操作": ["计算", "测量", "比较"],
                    "日常使用": ["食用", "阅读", "运输"]
                }
            },
            "agentive": {
                "similar_groups": [
                    ["自然生长", "自然形成", "天然"],
                    ["人工制造", "人造", "制作"],
                    ["计算得出", "推算", "求得"],
                    ["测量获得", "测得", "量得"]
                ],
                "hypernym_relations": {
                    "自然过程": ["自然生长", "自然形成"],
                    "人工过程": ["人工制造", "制作", "生产"]
                }
            },
            "constitutive": {
                "similar_groups": [
                    ["数字符号", "符号", "表示"],
                    ["材料", "成分", "组成"],
                    ["部分", "组件", "元素"]
                ],
                "hypernym_relations": {
                    "物理构成": ["材料", "成分", "物质"],
                    "抽象构成": ["符号", "概念", "元素"]
                }
            }
        }
    
    def _update_stats(self, compatibility_score: float):
        """更新统计信息"""
        self.stats["total_computations"] += 1
        
        if compatibility_score > self.compatibility_threshold:
            self.stats["high_compatibility_pairs"] += 1
        
        # 更新平均兼容性
        current_avg = self.stats["average_compatibility"]
        total_comps = self.stats["total_computations"]
        new_avg = ((current_avg * (total_comps - 1) + compatibility_score) / total_comps)
        self.stats["average_compatibility"] = new_avg
    
    def batch_compute_compatibility(
        self, 
        structures: List[QualiaStructure],
        context_weight: float = 1.0
    ) -> List[Tuple[int, int, float]]:
        """
        批量计算兼容性
        
        Args:
            structures: 语义结构列表
            context_weight: 上下文权重
            
        Returns:
            List[Tuple[int, int, float]]: (索引1, 索引2, 兼容性分数)
        """
        results = []
        
        for i in range(len(structures)):
            for j in range(i + 1, len(structures)):
                try:
                    compatibility = self.compute_compatibility(
                        structures[i], structures[j], context_weight
                    )
                    results.append((i, j, compatibility))
                except Exception as e:
                    self.logger.error(f"批量计算失败 ({i}, {j}): {e}")
                    results.append((i, j, 0.0))
        
        self.logger.info(f"批量计算完成，处理 {len(results)} 个实体对")
        return results
    
    def get_high_compatibility_pairs(
        self, 
        structures: List[QualiaStructure],
        threshold: Optional[float] = None
    ) -> List[Tuple[QualiaStructure, QualiaStructure, float]]:
        """获取高兼容性实体对"""
        threshold = threshold or self.compatibility_threshold
        high_pairs = []
        
        for i in range(len(structures)):
            for j in range(i + 1, len(structures)):
                compatibility = self.compute_compatibility(structures[i], structures[j])
                if compatibility > threshold:
                    high_pairs.append((structures[i], structures[j], compatibility))
        
        # 按兼容性排序
        high_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return high_pairs
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_computations": 0,
            "high_compatibility_pairs": 0,
            "average_compatibility": 0.0,
            "computation_time_stats": {}
        }
        self.logger.info("兼容性引擎统计信息已重置")