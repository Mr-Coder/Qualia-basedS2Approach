"""
Relation Discovery Evaluator Module
===================================

This module provides functionality to evaluate the quality of implicit relation discovery
in mathematical problem solving systems.

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RelationDiscoveryEvaluator:
    """
    关系发现评估器
    
    用于评估隐式关系发现的质量，包括：
    - 精确率、召回率和F1分数计算
    - 语义准确性评估
    - 关系类型分析
    - 发现质量统计
    """
    
    def __init__(self):
        """初始化关系发现评估器"""
        self.evaluation_history = []
        self.relation_type_weights = {
            "mathematical_operations": 1.0,
            "unit_conversions": 0.9,
            "physical_constraints": 0.8,
            "temporal_relations": 0.7,
            "geometric_properties": 0.8,
            "proportional_relations": 0.9
        }
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("RelationDiscoveryEvaluator initialized")
    
    def evaluate_relation_discovery(self, discovered_relations: List[Dict[str, Any]], 
                                   true_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估隐式关系发现质量
        
        Args:
            discovered_relations: 发现的关系列表
            true_relations: 真实关系列表
            
        Returns:
            Dict: 评估结果，包含精确率、召回率、F1分数等
        """
        try:
            # 标准化关系格式
            discovered_set = self._normalize_relations(discovered_relations)
            true_set = self._normalize_relations(true_relations)
            
            # 计算精确率
            if len(discovered_set) == 0:
                precision = 1.0 if len(true_set) == 0 else 0.0
            else:
                correct_discovered = len(discovered_set & true_set)
                precision = correct_discovered / len(discovered_set)
            
            # 计算召回率
            if len(true_set) == 0:
                recall = 1.0 if len(discovered_set) == 0 else 0.0
            else:
                correct_discovered = len(discovered_set & true_set)
                recall = correct_discovered / len(true_set)
            
            # 计算F1分数
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            # 计算语义准确性
            semantic_accuracy = self.evaluate_semantic_accuracy(discovered_relations, true_relations)
            
            # 计算加权分数
            weighted_f1 = self._calculate_weighted_f1(discovered_relations, true_relations)
            
            result = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "semantic_accuracy": semantic_accuracy,
                "weighted_f1": weighted_f1,
                "avg_relations": len(discovered_relations),
                "true_relations_count": len(true_relations),
                "correct_relations": len(discovered_set & true_set),
                "false_positives": len(discovered_set - true_set),
                "false_negatives": len(true_set - discovered_set)
            }
            
            self.logger.info(f"Relation discovery evaluation: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating relation discovery: {e}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "semantic_accuracy": 0.0,
                "weighted_f1": 0.0,
                "avg_relations": 0,
                "true_relations_count": 0,
                "correct_relations": 0,
                "false_positives": 0,
                "false_negatives": 0
            }
    
    def evaluate_semantic_accuracy(self, discovered_relations: List[Dict[str, Any]], 
                                 true_relations: List[Dict[str, Any]]) -> float:
        """
        评估发现关系的语义准确性
        
        Args:
            discovered_relations: 发现的关系列表
            true_relations: 真实关系列表
            
        Returns:
            float: 语义准确性分数 (0-1之间)
        """
        if not discovered_relations:
            return 1.0 if not true_relations else 0.0
        
        try:
            correct_semantic = 0
            for rel in discovered_relations:
                if self.is_semantically_valid(rel, true_relations):
                    correct_semantic += 1
            
            semantic_accuracy = correct_semantic / len(discovered_relations)
            self.logger.debug(f"Semantic accuracy: {semantic_accuracy:.3f}")
            return semantic_accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluating semantic accuracy: {e}")
            return 0.0
    
    def is_semantically_valid(self, relation: Dict[str, Any], 
                            true_relations: List[Dict[str, Any]]) -> bool:
        """
        检查关系是否在语义上有效
        
        Args:
            relation: 待检查的关系
            true_relations: 真实关系列表
            
        Returns:
            bool: 是否语义有效
        """
        try:
            relation_type = relation.get("type", "")
            relation_match = relation.get("match", "")
            
            # 检查类型匹配
            for true_rel in true_relations:
                true_type = true_rel.get("type", "")
                true_match = true_rel.get("match", "")
                
                # 完全匹配
                if relation_type == true_type and relation_match == true_match:
                    return True
                
                # 语义相似匹配
                if relation_type == true_type and self._is_semantically_similar(relation_match, true_match):
                    return True
                
                # 同义关系类型匹配
                if self._are_related_types(relation_type, true_type) and relation_match == true_match:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking semantic validity: {e}")
            return False
    
    def evaluate_by_relation_type(self, discovered_relations: List[Dict[str, Any]], 
                                true_relations: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        按关系类型评估发现质量
        
        Args:
            discovered_relations: 发现的关系列表
            true_relations: 真实关系列表
            
        Returns:
            Dict: 各关系类型的评估结果
        """
        type_results = {}
        
        # 按类型分组
        discovered_by_type = defaultdict(list)
        true_by_type = defaultdict(list)
        
        for rel in discovered_relations:
            rel_type = rel.get("type", "unknown")
            discovered_by_type[rel_type].append(rel)
        
        for rel in true_relations:
            rel_type = rel.get("type", "unknown")
            true_by_type[rel_type].append(rel)
        
        # 计算各类型的指标
        all_types = set(discovered_by_type.keys()) | set(true_by_type.keys())
        
        for rel_type in all_types:
            discovered_type = discovered_by_type.get(rel_type, [])
            true_type = true_by_type.get(rel_type, [])
            
            type_result = self.evaluate_relation_discovery(discovered_type, true_type)
            type_results[rel_type] = type_result
            
            self.logger.debug(f"Type {rel_type}: P={type_result['precision']:.3f}, "
                            f"R={type_result['recall']:.3f}, F1={type_result['f1']:.3f}")
        
        return type_results
    
    def calculate_coverage_metrics(self, discovered_relations: List[Dict[str, Any]], 
                                 true_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算覆盖度指标
        
        Args:
            discovered_relations: 发现的关系列表
            true_relations: 真实关系列表
            
        Returns:
            Dict: 覆盖度指标
        """
        coverage_metrics = {
            "type_coverage": 0.0,
            "pattern_coverage": 0.0,
            "position_coverage": 0.0,
            "overall_coverage": 0.0
        }
        
        try:
            if not true_relations:
                return coverage_metrics
            
            # 类型覆盖度
            true_types = set(rel.get("type", "") for rel in true_relations)
            discovered_types = set(rel.get("type", "") for rel in discovered_relations)
            type_coverage = len(discovered_types & true_types) / len(true_types) if true_types else 0.0
            
            # 模式覆盖度
            true_patterns = set(rel.get("pattern", "") for rel in true_relations)
            discovered_patterns = set(rel.get("pattern", "") for rel in discovered_relations)
            pattern_coverage = len(discovered_patterns & true_patterns) / len(true_patterns) if true_patterns else 0.0
            
            # 位置覆盖度（基于文本位置的重叠）
            position_coverage = self._calculate_position_coverage(discovered_relations, true_relations)
            
            # 总体覆盖度
            overall_coverage = (type_coverage + pattern_coverage + position_coverage) / 3
            
            coverage_metrics.update({
                "type_coverage": type_coverage,
                "pattern_coverage": pattern_coverage,
                "position_coverage": position_coverage,
                "overall_coverage": overall_coverage
            })
            
            self.logger.info(f"Coverage metrics: Type={type_coverage:.3f}, "
                           f"Pattern={pattern_coverage:.3f}, Overall={overall_coverage:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating coverage metrics: {e}")
        
        return coverage_metrics
    
    def analyze_discovery_patterns(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析关系发现模式
        
        Args:
            evaluation_results: 多次评估结果列表
            
        Returns:
            Dict: 发现模式分析结果
        """
        if not evaluation_results:
            return {}
        
        analysis = {
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "precision_std": 0.0,
            "recall_std": 0.0,
            "f1_std": 0.0,
            "consistency_score": 0.0,
            "improvement_trend": "stable"
        }
        
        try:
            precisions = [result.get("precision", 0) for result in evaluation_results]
            recalls = [result.get("recall", 0) for result in evaluation_results]
            f1_scores = [result.get("f1", 0) for result in evaluation_results]
            
            # 计算平均值
            analysis["avg_precision"] = sum(precisions) / len(precisions)
            analysis["avg_recall"] = sum(recalls) / len(recalls)
            analysis["avg_f1"] = sum(f1_scores) / len(f1_scores)
            
            # 计算标准差
            if len(precisions) > 1:
                import statistics
                analysis["precision_std"] = statistics.stdev(precisions)
                analysis["recall_std"] = statistics.stdev(recalls)
                analysis["f1_std"] = statistics.stdev(f1_scores)
            
            # 一致性分数（基于标准差）
            avg_std = (analysis["precision_std"] + analysis["recall_std"] + analysis["f1_std"]) / 3
            analysis["consistency_score"] = max(0, 1 - avg_std)
            
            # 改进趋势分析
            if len(f1_scores) >= 3:
                recent_avg = sum(f1_scores[-3:]) / 3
                early_avg = sum(f1_scores[:3]) / 3
                if recent_avg > early_avg + 0.05:
                    analysis["improvement_trend"] = "improving"
                elif recent_avg < early_avg - 0.05:
                    analysis["improvement_trend"] = "declining"
                else:
                    analysis["improvement_trend"] = "stable"
            
            self.logger.info(f"Discovery patterns: Avg F1={analysis['avg_f1']:.3f}, "
                           f"Consistency={analysis['consistency_score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing discovery patterns: {e}")
        
        return analysis
    
    def _normalize_relations(self, relations: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
        """将关系列表标准化为集合格式"""
        normalized = set()
        for rel in relations:
            rel_type = rel.get("type", "")
            rel_match = rel.get("match", "")
            normalized.add((rel_type, rel_match))
        return normalized
    
    def _calculate_weighted_f1(self, discovered_relations: List[Dict[str, Any]], 
                             true_relations: List[Dict[str, Any]]) -> float:
        """计算加权F1分数"""
        try:
            type_results = self.evaluate_by_relation_type(discovered_relations, true_relations)
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for rel_type, result in type_results.items():
                weight = self.relation_type_weights.get(rel_type, 0.5)
                weighted_sum += result["f1"] * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating weighted F1: {e}")
            return 0.0
    
    def _is_semantically_similar(self, text1: str, text2: str) -> bool:
        """检查两个文本是否语义相似"""
        # 简单的语义相似性检查
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower())
        
        # 检查包含关系
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return True
        
        # 检查关键词重叠
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total > 0.5 if total > 0 else False
    
    def _are_related_types(self, type1: str, type2: str) -> bool:
        """检查两个关系类型是否相关"""
        related_groups = [
            {"mathematical_operations", "proportional_relations"},
            {"unit_conversions", "physical_constraints"},
            {"temporal_relations", "geometric_properties"}
        ]
        
        for group in related_groups:
            if type1 in group and type2 in group:
                return True
        
        return False
    
    def _calculate_position_coverage(self, discovered_relations: List[Dict[str, Any]], 
                                   true_relations: List[Dict[str, Any]]) -> float:
        """计算位置覆盖度"""
        try:
            if not true_relations:
                return 1.0
            
            covered_positions = 0
            total_positions = len(true_relations)
            
            for true_rel in true_relations:
                true_pos = true_rel.get("position", (0, 0))
                
                for disc_rel in discovered_relations:
                    disc_pos = disc_rel.get("position", (0, 0))
                    
                    # 检查位置重叠
                    if self._positions_overlap(true_pos, disc_pos):
                        covered_positions += 1
                        break
            
            return covered_positions / total_positions
            
        except Exception as e:
            self.logger.warning(f"Error calculating position coverage: {e}")
            return 0.0
    
    def _positions_overlap(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """检查两个位置是否重叠"""
        try:
            start1, end1 = pos1
            start2, end2 = pos2
            
            # 检查区间重叠
            return not (end1 <= start2 or end2 <= start1)
            
        except Exception:
            return False
    
    def export_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        """导出评估结果"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"Evaluation results exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting evaluation results: {e}")
            raise 