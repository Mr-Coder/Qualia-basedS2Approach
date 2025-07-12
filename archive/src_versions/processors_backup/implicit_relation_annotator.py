"""
Implicit Relation Annotator Module
==================================

This module provides functionality to annotate implicit relations in mathematical problems.
It identifies and categorizes different types of implicit relationships that are not
explicitly stated in the problem text but are necessary for solving the problem.

Relation Types:
- Mathematical Operations (35.2%)
- Unit Conversions (18.7%)
- Physical Constraints (16.4%)
- Temporal Relations (12.3%)
- Geometric Properties (10.8%)
- Proportional Relations (6.6%)

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ImplicitRelationAnnotator:
    """
    隐式关系标注器
    
    用于识别和标注数学问题中的隐式关系，包括：
    - 数学运算关系 (35.2%)
    - 单位转换关系 (18.7%)
    - 物理约束关系 (16.4%)
    - 时间关系 (12.3%)
    - 几何属性关系 (10.8%)
    - 比例关系 (6.6%)
    """
    
    def __init__(self):
        """初始化隐式关系标注器"""
        self.relation_types = {
            "mathematical_operations": 0.352,  # 35.2%
            "unit_conversions": 0.187,         # 18.7%
            "physical_constraints": 0.164,     # 16.4%
            "temporal_relations": 0.123,       # 12.3%
            "geometric_properties": 0.108,     # 10.8%
            "proportional_relations": 0.066    # 6.6%
        }
        
        self.relation_patterns = self._initialize_relation_patterns()
        self.unit_conversion_rules = self._initialize_unit_conversion_rules()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ImplicitRelationAnnotator initialized")
    
    def _initialize_relation_patterns(self) -> Dict[str, List[str]]:
        """初始化关系模式"""
        patterns = {
            "mathematical_operations": [
                r"总共|一共|合计",
                r"剩下|还剩|余下",
                r"每.*需要|每.*用",
                r"平均|均匀",
                r"增加.*倍|减少.*倍",
                r"比.*多|比.*少"
            ],
            "unit_conversions": [
                r"\d+\s*(米|厘米|毫米|千米)",
                r"\d+\s*(千克|克|吨)",
                r"\d+\s*(小时|分钟|秒)",
                r"\d+\s*(元|角|分)",
                r"\d+\s*(升|毫升)"
            ],
            "physical_constraints": [
                r"装满|盛满",
                r"最多|最少|至多|至少",
                r"不能超过|不得少于",
                r"容量|体积|面积"
            ],
            "temporal_relations": [
                r"之前|之后|同时",
                r"先.*后|首先.*然后",
                r"开始.*结束",
                r"第.*天|第.*小时"
            ],
            "geometric_properties": [
                r"长方形|正方形|圆形|三角形",
                r"长.*宽|半径|直径",
                r"周长|面积|体积",
                r"平行|垂直|相交"
            ],
            "proportional_relations": [
                r"正比|反比|成比例",
                r"速度.*时间|工作效率",
                r"单价.*数量",
                r"密度.*体积"
            ]
        }
        return patterns
    
    def _initialize_unit_conversion_rules(self) -> Dict[str, Dict[str, float]]:
        """
        初始化单位转换规则
        
        Returns:
            Dict: 单位转换规则
        """
        rules = {
            "length": {
                "千米": 1000,
                "米": 1,
                "分米": 0.1,
                "厘米": 0.01,
                "毫米": 0.001
            },
            "weight": {
                "吨": 1000,
                "千克": 1,
                "克": 0.001,
                "毫克": 0.000001
            },
            "time": {
                "小时": 3600,
                "分钟": 60,
                "秒": 1
            },
            "currency": {
                "元": 1,
                "角": 0.1,
                "分": 0.01
            },
            "volume": {
                "升": 1,
                "毫升": 0.001
            }
        }
        return rules
    
    def annotate_implicit_relations(self, problem_text: str) -> List[Dict[str, Any]]:
        """
        标注问题中的隐式关系
        
        Args:
            problem_text: 问题文本
            
        Returns:
            List[Dict]: 标注的隐式关系列表
        """
        relations = []
        
        try:
            # 数学运算关系
            math_ops = self.extract_mathematical_operations(problem_text)
            relations.extend(math_ops)
            
            # 单位转换关系
            unit_conversions = self.extract_unit_conversions(problem_text)
            relations.extend(unit_conversions)
            
            # 物理约束关系
            physical_constraints = self.extract_physical_constraints(problem_text)
            relations.extend(physical_constraints)
            
            # 时间关系
            temporal_relations = self.extract_temporal_relations(problem_text)
            relations.extend(temporal_relations)
            
            # 几何属性关系
            geometric_properties = self.extract_geometric_properties(problem_text)
            relations.extend(geometric_properties)
            
            # 比例关系
            proportional_relations = self.extract_proportional_relations(problem_text)
            relations.extend(proportional_relations)
            
            self.logger.debug(f"Extracted {len(relations)} implicit relations from problem text")
            return relations
            
        except Exception as e:
            self.logger.error(f"Error annotating implicit relations: {e}")
            return []
    
    def extract_mathematical_operations(self, text: str) -> List[Dict[str, Any]]:
        """提取数学运算关系"""
        relations = []
        for pattern in self.relation_patterns["mathematical_operations"]:
            matches = re.finditer(pattern, text)
            for match in matches:
                relations.append({
                    "type": "mathematical_operations",
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span()
                })
        return relations
    
    def extract_unit_conversions(self, text: str) -> List[Dict[str, Any]]:
        """提取单位转换关系"""
        relations = []
        for pattern in self.relation_patterns["unit_conversions"]:
            matches = re.finditer(pattern, text)
            for match in matches:
                relations.append({
                    "type": "unit_conversions",
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span()
                })
        return relations
    
    def extract_physical_constraints(self, text: str) -> List[Dict[str, Any]]:
        """提取物理约束关系"""
        relations = []
        for pattern in self.relation_patterns["physical_constraints"]:
            matches = re.finditer(pattern, text)
            for match in matches:
                relations.append({
                    "type": "physical_constraints",
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span()
                })
        return relations
    
    def extract_temporal_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        提取时间关系
        
        Args:
            text: 文本
            
        Returns:
            List[Dict]: 时间关系列表
        """
        relations = []
        
        for temporal_type, patterns in self.relation_patterns["temporal_relations"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    relations.append({
                        "type": "temporal_relations",
                        "subtype": temporal_type,
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.span(),
                        "confidence": 0.75
                    })
        
        return relations
    
    def extract_geometric_properties(self, text: str) -> List[Dict[str, Any]]:
        """
        提取几何属性关系
        
        Args:
            text: 文本
            
        Returns:
            List[Dict]: 几何属性关系列表
        """
        relations = []
        
        for geo_type, patterns in self.relation_patterns["geometric_properties"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    relations.append({
                        "type": "geometric_properties",
                        "subtype": geo_type,
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.span(),
                        "confidence": 0.85
                    })
        
        return relations
    
    def extract_proportional_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        提取比例关系
        
        Args:
            text: 文本
            
        Returns:
            List[Dict]: 比例关系列表
        """
        relations = []
        
        for prop_type, patterns in self.relation_patterns["proportional_relations"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    relations.append({
                        "type": "proportional_relations",
                        "subtype": prop_type,
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.span(),
                        "confidence": 0.8
                    })
        
        return relations
    
    def create_ground_truth_relations(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为问题创建隐式关系的真值标注
        
        Args:
            problems: 问题列表
            
        Returns:
            List[Dict]: 带有隐式关系真值标注的问题列表
        """
        annotated_problems = []
        
        for i, problem in enumerate(problems):
            try:
                problem_text = problem.get("question", problem.get("text", ""))
                relations = self.annotate_implicit_relations(problem_text)
                
                # 添加隐式关系标注
                annotated_problem = problem.copy()
                annotated_problem["implicit_relations_true"] = relations
                annotated_problem["implicit_relations_count"] = len(relations)
                annotated_problem["relation_types_present"] = list(set(
                    rel["type"] for rel in relations
                ))
                
                annotated_problems.append(annotated_problem)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(problems)} problems")
                    
            except Exception as e:
                self.logger.error(f"Error processing problem {i}: {e}")
                # 添加空标注以保持一致性
                annotated_problem = problem.copy()
                annotated_problem["implicit_relations_true"] = []
                annotated_problem["implicit_relations_count"] = 0
                annotated_problem["relation_types_present"] = []
                annotated_problems.append(annotated_problem)
        
        self.logger.info(f"Completed annotation for {len(annotated_problems)} problems")
        return annotated_problems
    
    def analyze_relation_distribution(self, annotated_problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析隐式关系分布
        
        Args:
            annotated_problems: 标注的问题列表
            
        Returns:
            Dict: 关系分布分析结果
        """
        relation_counts = defaultdict(int)
        subtype_counts = defaultdict(int)
        total_relations = 0
        
        for problem in annotated_problems:
            relations = problem.get("implicit_relations_true", [])
            total_relations += len(relations)
            
            for relation in relations:
                relation_type = relation["type"]
                subtype = relation.get("subtype", "unknown")
                
                relation_counts[relation_type] += 1
                subtype_counts[f"{relation_type}.{subtype}"] += 1
        
        # 计算分布百分比
        relation_percentages = {}
        if total_relations > 0:
            for rel_type, count in relation_counts.items():
                relation_percentages[rel_type] = (count / total_relations) * 100
        
        analysis = {
            "total_problems": len(annotated_problems),
            "total_relations": total_relations,
            "avg_relations_per_problem": total_relations / len(annotated_problems) if annotated_problems else 0,
            "relation_type_counts": dict(relation_counts),
            "relation_type_percentages": relation_percentages,
            "subtype_counts": dict(subtype_counts),
            "expected_distribution": {
                "mathematical_operations": 35.2,
                "unit_conversions": 18.7,
                "physical_constraints": 16.4,
                "temporal_relations": 12.3,
                "geometric_properties": 10.8,
                "proportional_relations": 6.6
            }
        }
        
        return analysis
    
    def validate_annotations(self, annotated_problems: List[Dict[str, Any]], 
                           sample_size: int = 100) -> Dict[str, Any]:
        """
        验证标注质量
        
        Args:
            annotated_problems: 标注的问题列表
            sample_size: 验证样本大小
            
        Returns:
            Dict: 验证结果
        """
        import random
        
        if len(annotated_problems) < sample_size:
            sample_size = len(annotated_problems)
        
        sample_problems = random.sample(annotated_problems, sample_size)
        
        validation_results = {
            "sample_size": sample_size,
            "problems_with_relations": 0,
            "avg_confidence": 0.0,
            "relation_type_coverage": set(),
            "potential_issues": []
        }
        
        total_confidence = 0
        relation_count = 0
        
        for problem in sample_problems:
            relations = problem.get("implicit_relations_true", [])
            
            if relations:
                validation_results["problems_with_relations"] += 1
                
                for relation in relations:
                    confidence = relation.get("confidence", 0.5)
                    total_confidence += confidence
                    relation_count += 1
                    
                    validation_results["relation_type_coverage"].add(relation["type"])
                    
                    # 检查潜在问题
                    if confidence < 0.5:
                        validation_results["potential_issues"].append({
                            "type": "low_confidence",
                            "relation": relation,
                            "problem_id": problem.get("id", "unknown")
                        })
        
        if relation_count > 0:
            validation_results["avg_confidence"] = total_confidence / relation_count
        
        validation_results["relation_type_coverage"] = list(validation_results["relation_type_coverage"])
        
        return validation_results
    
    def export_annotations(self, annotated_problems: List[Dict[str, Any]], 
                          output_path: str, include_analysis: bool = True) -> None:
        """
        导出标注结果
        
        Args:
            annotated_problems: 标注的问题列表
            output_path: 输出文件路径
            include_analysis: 是否包含分析结果
        """
        try:
            export_data = {
                "problems": annotated_problems,
                "metadata": {
                    "total_problems": len(annotated_problems),
                    "annotation_version": "1.0.0",
                    "relation_types": list(self.relation_types.keys())
                }
            }
            
            if include_analysis:
                export_data["analysis"] = self.analyze_relation_distribution(annotated_problems)
                export_data["validation"] = self.validate_annotations(annotated_problems)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Annotations exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting annotations: {e}")
            raise 