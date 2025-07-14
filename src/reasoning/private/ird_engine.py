"""
隐式关系发现引擎 (Implicit Relation Discovery Engine)

专注于从数学问题文本中发现隐式的数学关系。
这是COT-DIR算法的第一个核心组件。
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class RelationType(Enum):
    """隐式关系类型"""
    ARITHMETIC = "arithmetic"          # 算术关系
    PROPORTION = "proportion"          # 比例关系
    COMPARISON = "comparison"          # 比较关系
    TEMPORAL = "temporal"              # 时间关系
    CAUSAL = "causal"                 # 因果关系
    CONSTRAINT = "constraint"          # 约束关系
    SPATIAL = "spatial"               # 空间关系
    FUNCTIONAL = "functional"          # 函数关系


@dataclass
class ImplicitRelation:
    """隐式关系数据结构"""
    relation_type: RelationType
    entities: List[str]
    confidence: float
    description: str
    mathematical_expression: Optional[str] = None
    source_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.relation_type.value,
            "entities": self.entities,
            "confidence": self.confidence,
            "description": self.description,
            "expression": self.mathematical_expression,
            "source": self.source_text
        }


@dataclass
class IRDResult:
    """IRD处理结果"""
    relations: List[ImplicitRelation]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    
    def get_relations_by_type(self, relation_type: RelationType) -> List[ImplicitRelation]:
        """按类型获取关系"""
        return [r for r in self.relations if r.relation_type == relation_type]
    
    def get_high_confidence_relations(self, threshold: float = 0.7) -> List[ImplicitRelation]:
        """获取高置信度关系"""
        return [r for r in self.relations if r.confidence >= threshold]


class ImplicitRelationDiscoveryEngine:
    """隐式关系发现引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化IRD引擎"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 配置参数
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.max_relations = self.config.get("max_relations", 10)
        self.enable_advanced_patterns = self.config.get("enable_advanced_patterns", True)
        
        # 关系模式库
        self.relation_patterns = self._initialize_relation_patterns()
        self.entity_extractors = self._initialize_entity_extractors()
        
        # 统计信息
        self.stats = {
            "total_processed": 0,
            "relations_found": 0,
            "average_confidence": 0.0,
            "relation_type_counts": {rt.value: 0 for rt in RelationType}
        }
        
        self.logger.info("隐式关系发现引擎初始化完成")
    
    def discover_relations(self, problem_text: str, context: Optional[Dict[str, Any]] = None) -> IRDResult:
        """
        发现问题文本中的隐式关系
        
        Args:
            problem_text: 问题文本
            context: 可选的上下文信息
            
        Returns:
            IRDResult: 发现的关系列表及相关信息
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"开始隐式关系发现: {problem_text[:50]}...")
            
            # 文本预处理
            cleaned_text = self._preprocess_text(problem_text)
            
            # 实体提取
            entities = self._extract_entities(cleaned_text)
            
            # 关系发现
            relations = []
            
            # 基于模式的关系发现
            pattern_relations = self._discover_pattern_based_relations(cleaned_text, entities)
            relations.extend(pattern_relations)
            
            # 基于语义的关系发现
            if self.enable_advanced_patterns:
                semantic_relations = self._discover_semantic_relations(cleaned_text, entities)
                relations.extend(semantic_relations)
            
            # 关系验证和过滤
            validated_relations = self._validate_relations(relations, cleaned_text)
            
            # 关系合并和去重
            final_relations = self._merge_and_deduplicate_relations(validated_relations)
            
            # 计算整体置信度
            overall_confidence = self._calculate_overall_confidence(final_relations)
            
            # 更新统计信息
            self._update_stats(final_relations)
            
            processing_time = time.time() - start_time
            
            result = IRDResult(
                relations=final_relations,
                confidence_score=overall_confidence,
                processing_time=processing_time,
                metadata={
                    "entities_found": len(entities),
                    "pattern_relations": len(pattern_relations),
                    "semantic_relations": len(semantic_relations) if self.enable_advanced_patterns else 0,
                    "original_text": problem_text,
                    "cleaned_text": cleaned_text
                }
            )
            
            self.logger.info(f"IRD完成: 发现{len(final_relations)}个关系，置信度{overall_confidence:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"隐式关系发现失败: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        
        # 标准化文本
        cleaned = text.strip()
        
        # 标准化标点符号
        replacements = {
            "？": "?", "。": ".", "，": ",", "；": ";", "：": ":",
            "（": "(", "）": ")", "【": "[", "】": "]"
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # 标准化空白字符
        cleaned = " ".join(cleaned.split())
        
        return cleaned
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体"""
        entities = {
            "numbers": [],
            "units": [],
            "objects": [],
            "actions": [],
            "conditions": []
        }
        
        # 提取数字
        number_pattern = r'\d+(?:\.\d+)?'
        entities["numbers"] = re.findall(number_pattern, text)
        
        # 提取单位
        unit_patterns = [
            r'(?:元|米|厘米|千米|公里|分钟|小时|天|年|个|只|本|斤|公斤|吨)',
            r'(?:平方米|立方米|平方厘米|立方厘米)'
        ]
        for pattern in unit_patterns:
            entities["units"].extend(re.findall(pattern, text))
        
        # 提取对象（名词）
        object_patterns = [
            r'(?:苹果|桔子|书|笔|桌子|椅子|汽车|火车|飞机|学生|老师|工人)',
            r'(?:长方形|正方形|圆形|三角形|梯形)'
        ]
        for pattern in object_patterns:
            entities["objects"].extend(re.findall(pattern, text))
        
        # 提取动作
        action_patterns = [
            r'(?:买|卖|给|拿|走|来|去|做|用|花|赚|失去|得到)'
        ]
        for pattern in action_patterns:
            entities["actions"].extend(re.findall(pattern, text))
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _discover_pattern_based_relations(self, text: str, entities: Dict[str, List[str]]) -> List[ImplicitRelation]:
        """基于模式的关系发现"""
        relations = []
        
        # 算术关系模式
        arithmetic_relations = self._find_arithmetic_relations(text, entities)
        relations.extend(arithmetic_relations)
        
        # 比例关系模式
        proportion_relations = self._find_proportion_relations(text, entities)
        relations.extend(proportion_relations)
        
        # 比较关系模式
        comparison_relations = self._find_comparison_relations(text, entities)
        relations.extend(comparison_relations)
        
        # 时间关系模式
        temporal_relations = self._find_temporal_relations(text, entities)
        relations.extend(temporal_relations)
        
        # 约束关系模式
        constraint_relations = self._find_constraint_relations(text, entities)
        relations.extend(constraint_relations)
        
        return relations
    
    def _find_arithmetic_relations(self, text: str, entities: Dict[str, List[str]]) -> List[ImplicitRelation]:
        """发现算术关系"""
        relations = []
        
        # 加法关系
        add_patterns = [
            r'(\d+(?:\.\d+)?)[^0-9]*加上[^0-9]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)[^0-9]*和[^0-9]*(\d+(?:\.\d+)?)[^0-9]*一共',
            r'总共[^0-9]*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in add_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                numbers = [match.group(i) for i in range(1, match.lastindex + 1) if match.group(i)]
                if len(numbers) >= 2:
                    relation = ImplicitRelation(
                        relation_type=RelationType.ARITHMETIC,
                        entities=numbers,
                        confidence=0.8,
                        description=f"加法关系: {' + '.join(numbers)}",
                        mathematical_expression=f"{numbers[0]} + {numbers[1]}",
                        source_text=match.group(0)
                    )
                    relations.append(relation)
        
        # 减法关系
        sub_patterns = [
            r'(\d+(?:\.\d+)?)[^0-9]*减去[^0-9]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)[^0-9]*少[^0-9]*(\d+(?:\.\d+)?)',
            r'剩下[^0-9]*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in sub_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                numbers = [match.group(i) for i in range(1, match.lastindex + 1) if match.group(i)]
                if len(numbers) >= 2:
                    relation = ImplicitRelation(
                        relation_type=RelationType.ARITHMETIC,
                        entities=numbers,
                        confidence=0.8,
                        description=f"减法关系: {numbers[0]} - {numbers[1]}",
                        mathematical_expression=f"{numbers[0]} - {numbers[1]}",
                        source_text=match.group(0)
                    )
                    relations.append(relation)
        
        return relations
    
    def _find_proportion_relations(self, text: str, entities: Dict[str, List[str]]) -> List[ImplicitRelation]:
        """发现比例关系"""
        relations = []
        
        # 百分比模式
        percent_patterns = [
            r'(\d+(?:\.\d+)?)%',
            r'百分之(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)[^0-9]*成'
        ]
        
        for pattern in percent_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                percentage = match.group(1)
                relation = ImplicitRelation(
                    relation_type=RelationType.PROPORTION,
                    entities=[percentage],
                    confidence=0.9,
                    description=f"百分比关系: {percentage}%",
                    mathematical_expression=f"{percentage}/100",
                    source_text=match.group(0)
                )
                relations.append(relation)
        
        # 比率模式
        ratio_patterns = [
            r'(\d+(?:\.\d+)?)[^0-9]*比[^0-9]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)[^0-9]*∶[^0-9]*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in ratio_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                numbers = [match.group(1), match.group(2)]
                relation = ImplicitRelation(
                    relation_type=RelationType.PROPORTION,
                    entities=numbers,
                    confidence=0.8,
                    description=f"比例关系: {numbers[0]}:{numbers[1]}",
                    mathematical_expression=f"{numbers[0]}/{numbers[1]}",
                    source_text=match.group(0)
                )
                relations.append(relation)
        
        return relations
    
    def _find_comparison_relations(self, text: str, entities: Dict[str, List[str]]) -> List[ImplicitRelation]:
        """发现比较关系"""
        relations = []
        
        comparison_patterns = [
            r'比[^0-9]*(\d+(?:\.\d+)?)[^0-9]*多[^0-9]*(\d+(?:\.\d+)?)',
            r'比[^0-9]*(\d+(?:\.\d+)?)[^0-9]*少[^0-9]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)[^0-9]*倍',
            r'大于[^0-9]*(\d+(?:\.\d+)?)',
            r'小于[^0-9]*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                numbers = [match.group(i) for i in range(1, match.lastindex + 1) if match.group(i)]
                if numbers:
                    relation = ImplicitRelation(
                        relation_type=RelationType.COMPARISON,
                        entities=numbers,
                        confidence=0.7,
                        description=f"比较关系: {match.group(0)}",
                        source_text=match.group(0)
                    )
                    relations.append(relation)
        
        return relations
    
    def _find_temporal_relations(self, text: str, entities: Dict[str, List[str]]) -> List[ImplicitRelation]:
        """发现时间关系"""
        relations = []
        
        temporal_patterns = [
            r'(\d+(?:\.\d+)?)[^0-9]*小时',
            r'(\d+(?:\.\d+)?)[^0-9]*分钟',
            r'(\d+(?:\.\d+)?)[^0-9]*天',
            r'第(\d+)[^0-9]*天',
            r'(\d+(?:\.\d+)?)[^0-9]*年前',
            r'(\d+(?:\.\d+)?)[^0-9]*年后'
        ]
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                time_value = match.group(1)
                relation = ImplicitRelation(
                    relation_type=RelationType.TEMPORAL,
                    entities=[time_value],
                    confidence=0.8,
                    description=f"时间关系: {match.group(0)}",
                    source_text=match.group(0)
                )
                relations.append(relation)
        
        return relations
    
    def _find_constraint_relations(self, text: str, entities: Dict[str, List[str]]) -> List[ImplicitRelation]:
        """发现约束关系"""
        relations = []
        
        constraint_patterns = [
            r'如果[^，。]*则[^，。]*',
            r'当[^，。]*时[^，。]*',
            r'假设[^，。]*',
            r'条件[^，。]*',
            r'要求[^，。]*',
            r'限制[^，。]*'
        ]
        
        for pattern in constraint_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                relation = ImplicitRelation(
                    relation_type=RelationType.CONSTRAINT,
                    entities=[],
                    confidence=0.6,
                    description=f"约束关系: {match.group(0)}",
                    source_text=match.group(0)
                )
                relations.append(relation)
        
        return relations
    
    def _discover_semantic_relations(self, text: str, entities: Dict[str, List[str]]) -> List[ImplicitRelation]:
        """基于语义的关系发现（高级模式）"""
        relations = []
        
        # 这里可以集成更复杂的NLP模型
        # 目前实现简化版本的语义分析
        
        # 功能关系发现
        functional_patterns = [
            r'速度.*距离.*时间',
            r'面积.*长.*宽',
            r'体积.*长.*宽.*高',
            r'利润.*成本.*售价'
        ]
        
        for pattern in functional_patterns:
            if re.search(pattern, text):
                relation = ImplicitRelation(
                    relation_type=RelationType.FUNCTIONAL,
                    entities=[],
                    confidence=0.7,
                    description=f"函数关系: {pattern}",
                    source_text=text
                )
                relations.append(relation)
        
        return relations
    
    def _validate_relations(self, relations: List[ImplicitRelation], text: str) -> List[ImplicitRelation]:
        """验证关系"""
        validated = []
        
        for relation in relations:
            # 置信度过滤
            if relation.confidence < self.confidence_threshold:
                continue
            
            # 实体有效性检查
            if relation.entities and not self._validate_entities(relation.entities, text):
                relation.confidence *= 0.8  # 降低置信度
            
            # 关系一致性检查
            if self._is_relation_consistent(relation, text):
                validated.append(relation)
        
        return validated
    
    def _validate_entities(self, entities: List[str], text: str) -> bool:
        """验证实体有效性"""
        for entity in entities:
            if entity not in text:
                return False
        return True
    
    def _is_relation_consistent(self, relation: ImplicitRelation, text: str) -> bool:
        """检查关系一致性"""
        # 简化的一致性检查
        if relation.source_text and relation.source_text not in text:
            return False
        return True
    
    def _merge_and_deduplicate_relations(self, relations: List[ImplicitRelation]) -> List[ImplicitRelation]:
        """合并和去重关系"""
        if not relations:
            return []
        
        # 按类型和实体进行去重
        unique_relations = []
        seen = set()
        
        for relation in relations:
            # 创建唯一标识
            key = (
                relation.relation_type.value,
                tuple(sorted(relation.entities)),
                relation.description
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # 如果重复，选择置信度更高的
                for i, existing in enumerate(unique_relations):
                    existing_key = (
                        existing.relation_type.value,
                        tuple(sorted(existing.entities)),
                        existing.description
                    )
                    if existing_key == key and relation.confidence > existing.confidence:
                        unique_relations[i] = relation
                        break
        
        # 限制最大数量
        if len(unique_relations) > self.max_relations:
            unique_relations.sort(key=lambda x: x.confidence, reverse=True)
            unique_relations = unique_relations[:self.max_relations]
        
        return unique_relations
    
    def _calculate_overall_confidence(self, relations: List[ImplicitRelation]) -> float:
        """计算整体置信度"""
        if not relations:
            return 0.0
        
        # 加权平均置信度
        total_confidence = sum(r.confidence for r in relations)
        avg_confidence = total_confidence / len(relations)
        
        # 基于关系数量的调整
        count_factor = min(len(relations) / 5, 1.0)  # 5个关系为满分
        
        return avg_confidence * count_factor
    
    def _update_stats(self, relations: List[ImplicitRelation]):
        """更新统计信息"""
        self.stats["total_processed"] += 1
        self.stats["relations_found"] += len(relations)
        
        if relations:
            # 更新平均置信度
            total_confidence = sum(r.confidence for r in relations)
            current_avg = self.stats["average_confidence"]
            new_avg = ((current_avg * (self.stats["total_processed"] - 1) + 
                       total_confidence / len(relations)) / self.stats["total_processed"])
            self.stats["average_confidence"] = new_avg
            
            # 更新关系类型计数
            for relation in relations:
                self.stats["relation_type_counts"][relation.relation_type.value] += 1
    
    def _initialize_relation_patterns(self) -> Dict[str, Any]:
        """初始化关系模式库"""
        return {
            "arithmetic_patterns": [],
            "proportion_patterns": [],
            "comparison_patterns": [],
            "temporal_patterns": [],
            "constraint_patterns": []
        }
    
    def _initialize_entity_extractors(self) -> Dict[str, Any]:
        """初始化实体提取器"""
        return {
            "number_extractor": None,
            "unit_extractor": None,
            "object_extractor": None
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_processed": 0,
            "relations_found": 0,
            "average_confidence": 0.0,
            "relation_type_counts": {rt.value: 0 for rt in RelationType}
        }
        self.logger.info("IRD引擎统计信息已重置")