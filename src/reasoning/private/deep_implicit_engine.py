"""
深度隐含关系发现引擎 (Deep Implicit Relation Discovery Engine)

实现三大核心能力：
1. 语义蕴含推理逻辑 - 从表层文本推导深层逻辑关系
2. 隐含约束条件挖掘 - 发现题目未明确表达的约束条件  
3. 多层关系建模机制 - 构建层次化的实体关系网络

与前端物性关系图深度集成，提供可视化展示能力。
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


class SemanticRelationType(Enum):
    """语义关系类型"""
    # 显式关系
    EXPLICIT_OWNERSHIP = "explicit_ownership"      # 明确拥有
    EXPLICIT_QUANTITY = "explicit_quantity"        # 明确数量
    EXPLICIT_OPERATION = "explicit_operation"      # 明确运算
    
    # 隐含关系  
    IMPLICIT_DEPENDENCY = "implicit_dependency"    # 隐含依赖
    IMPLICIT_CONSTRAINT = "implicit_constraint"    # 隐含约束
    IMPLICIT_EQUIVALENCE = "implicit_equivalence"  # 隐含等价
    
    # 深层关系
    DEEP_CAUSALITY = "deep_causality"              # 深层因果
    DEEP_CONSERVATION = "deep_conservation"        # 深层守恒
    DEEP_INVARIANCE = "deep_invariance"            # 深层不变性


class ConstraintType(Enum):
    """约束类型"""
    # 物理约束
    CONSERVATION_LAW = "conservation_law"          # 守恒定律
    CONTINUITY_CONSTRAINT = "continuity_constraint"  # 连续性约束
    NON_NEGATIVITY = "non_negativity"              # 非负性约束
    
    # 逻辑约束
    MUTUAL_EXCLUSION = "mutual_exclusion"          # 互斥约束
    COMPLETENESS = "completeness"                  # 完整性约束
    CONSISTENCY = "consistency"                    # 一致性约束
    
    # 语义约束
    TYPE_COMPATIBILITY = "type_compatibility"      # 类型兼容性
    ROLE_CONSTRAINT = "role_constraint"            # 角色约束
    CONTEXT_BOUNDARY = "context_boundary"          # 上下文边界


class RelationDepth(Enum):
    """关系深度级别"""
    SURFACE = "surface"        # 表层关系 - 直接从文本提取
    SHALLOW = "shallow"        # 浅层关系 - 简单推理得出
    MEDIUM = "medium"          # 中层关系 - 多步推理链
    DEEP = "deep"              # 深层关系 - 复杂语义推理


@dataclass
class DeepImplicitRelation:
    """深度隐含关系"""
    id: str
    source_entity: str
    target_entity: str
    relation_type: SemanticRelationType
    depth: RelationDepth
    confidence: float
    semantic_evidence: List[str]
    logical_basis: str
    constraint_implications: List[str]
    mathematical_expression: Optional[str]
    frontend_display_data: Dict[str, Any]  # 前端显示数据
    
    def to_frontend_format(self) -> Dict[str, Any]:
        """转换为前端可视化格式"""
        return {
            "id": self.id,
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relation_type.value,
            "depth": self.depth.value,
            "confidence": self.confidence,
            "label": self._generate_display_label(),
            "evidence": self.semantic_evidence,
            "constraints": self.constraint_implications,
            "visualization": self.frontend_display_data
        }
    
    def _generate_display_label(self) -> str:
        """生成前端显示标签"""
        type_labels = {
            SemanticRelationType.EXPLICIT_OWNERSHIP: "拥有关系",
            SemanticRelationType.EXPLICIT_QUANTITY: "数量关系", 
            SemanticRelationType.IMPLICIT_DEPENDENCY: "依赖关系",
            SemanticRelationType.IMPLICIT_CONSTRAINT: "约束关系",
            SemanticRelationType.DEEP_CAUSALITY: "因果关系",
            SemanticRelationType.DEEP_CONSERVATION: "守恒关系"
        }
        base_label = type_labels.get(self.relation_type, "未知关系")
        
        depth_indicator = {
            RelationDepth.SURFACE: "📄",
            RelationDepth.SHALLOW: "🔍", 
            RelationDepth.MEDIUM: "🧠",
            RelationDepth.DEEP: "⚡"
        }
        
        return f"{depth_indicator[self.depth]} {base_label} ({self.confidence:.1%})"


@dataclass 
class ImplicitConstraint:
    """隐含约束"""
    id: str
    constraint_type: ConstraintType
    description: str
    affected_entities: List[str]
    constraint_expression: str
    discovery_method: str
    confidence: float
    frontend_visualization: Dict[str, Any]
    
    def to_frontend_format(self) -> Dict[str, Any]:
        """转换为前端约束展示格式"""
        return {
            "id": self.id,
            "type": self.constraint_type.value,
            "description": self.description,
            "entities": self.affected_entities,
            "expression": self.constraint_expression,
            "confidence": self.confidence,
            "icon": self._get_constraint_icon(),
            "color": self._get_constraint_color(),
            "visualization": self.frontend_visualization
        }
    
    def _get_constraint_icon(self) -> str:
        """获取约束图标"""
        icons = {
            ConstraintType.CONSERVATION_LAW: "⚖️",
            ConstraintType.CONTINUITY_CONSTRAINT: "🔗",
            ConstraintType.NON_NEGATIVITY: "➕",
            ConstraintType.MUTUAL_EXCLUSION: "⚔️",
            ConstraintType.COMPLETENESS: "🔄",
            ConstraintType.TYPE_COMPATIBILITY: "🔧"
        }
        return icons.get(self.constraint_type, "📋")
    
    def _get_constraint_color(self) -> str:
        """获取约束颜色"""
        colors = {
            ConstraintType.CONSERVATION_LAW: "#16a085",
            ConstraintType.CONTINUITY_CONSTRAINT: "#3498db", 
            ConstraintType.NON_NEGATIVITY: "#27ae60",
            ConstraintType.MUTUAL_EXCLUSION: "#e74c3c",
            ConstraintType.COMPLETENESS: "#9b59b6",
            ConstraintType.TYPE_COMPATIBILITY: "#f39c12"
        }
        return colors.get(self.constraint_type, "#95a5a6")


class DeepImplicitEngine:
    """深度隐含关系发现引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化深度隐含引擎"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config or {}
        
        # 语义推理配置
        self.semantic_depth_threshold = self.config.get("semantic_depth_threshold", 0.6)
        self.max_reasoning_hops = self.config.get("max_reasoning_hops", 5)
        self.constraint_discovery_enabled = self.config.get("constraint_discovery", True)
        
        # 语义知识库
        self.semantic_patterns = self._initialize_semantic_patterns()
        self.constraint_rules = self._initialize_constraint_rules()
        self.domain_knowledge = self._initialize_domain_knowledge()
        
        self.logger.info("深度隐含关系发现引擎初始化完成")
    
    def discover_deep_relations(
        self, 
        problem_text: str, 
        entities: List[Dict[str, Any]], 
        surface_relations: List[Dict[str, Any]]
    ) -> Tuple[List[DeepImplicitRelation], List[ImplicitConstraint]]:
        """
        发现深度隐含关系和约束
        
        Args:
            problem_text: 问题文本
            entities: 实体列表
            surface_relations: 表层关系
            
        Returns:
            Tuple[深度关系列表, 隐含约束列表]
        """
        self.logger.info("开始深度隐含关系发现")
        
        # 第一步：语义蕴含推理
        semantic_relations = self._perform_semantic_implication_reasoning(
            problem_text, entities, surface_relations
        )
        
        # 第二步：隐含约束挖掘
        implicit_constraints = self._discover_implicit_constraints(
            problem_text, entities, semantic_relations
        )
        
        # 第三步：多层关系建模
        deep_relations = self._build_multilayer_relation_model(
            entities, semantic_relations, implicit_constraints
        )
        
        # 第四步：为前端生成可视化数据
        self._enhance_for_frontend_visualization(deep_relations, implicit_constraints)
        
        self.logger.info(f"发现 {len(deep_relations)} 个深度关系，{len(implicit_constraints)} 个隐含约束")
        return deep_relations, implicit_constraints
    
    def _perform_semantic_implication_reasoning(
        self, 
        problem_text: str, 
        entities: List[Dict[str, Any]], 
        surface_relations: List[Dict[str, Any]]
    ) -> List[DeepImplicitRelation]:
        """执行语义蕴含推理"""
        relations = []
        
        # 1. 基于语义模式的推理
        pattern_relations = self._apply_semantic_patterns(problem_text, entities)
        relations.extend(pattern_relations)
        
        # 2. 基于实体语义类型的推理
        type_relations = self._infer_from_entity_types(entities)
        relations.extend(type_relations)
        
        # 3. 基于上下文语义的推理
        context_relations = self._infer_from_context_semantics(problem_text, entities)
        relations.extend(context_relations)
        
        # 4. 基于数学语义的推理
        math_relations = self._infer_mathematical_semantics(problem_text, entities)
        relations.extend(math_relations)
        
        return relations
    
    def _apply_semantic_patterns(self, problem_text: str, entities: List[Dict[str, Any]]) -> List[DeepImplicitRelation]:
        """应用语义模式推理"""
        relations = []
        
        for pattern_name, pattern_data in self.semantic_patterns.items():
            pattern_regex = pattern_data["pattern"]
            relation_type = pattern_data["relation_type"]
            confidence_base = pattern_data["confidence"]
            
            matches = re.finditer(pattern_regex, problem_text, re.IGNORECASE)
            
            for match in matches:
                # 识别匹配的实体
                matched_entities = self._extract_entities_from_match(match, entities)
                
                if len(matched_entities) >= 2:
                    relation = DeepImplicitRelation(
                        id=f"semantic_{len(relations)}",
                        source_entity=matched_entities[0]["name"],
                        target_entity=matched_entities[1]["name"],
                        relation_type=relation_type,
                        depth=RelationDepth.SHALLOW,
                        confidence=confidence_base,
                        semantic_evidence=[f"语义模式匹配: {pattern_name}", match.group()],
                        logical_basis=f"基于语义模式 '{pattern_name}' 的推理",
                        constraint_implications=[],
                        mathematical_expression=pattern_data.get("math_expr"),
                        frontend_display_data={
                            "pattern_name": pattern_name,
                            "match_text": match.group(),
                            "reasoning_type": "semantic_pattern"
                        }
                    )
                    relations.append(relation)
        
        return relations
    
    def _infer_from_entity_types(self, entities: List[Dict[str, Any]]) -> List[DeepImplicitRelation]:
        """基于实体类型推理隐含关系"""
        relations = []
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                
                # 人物-物品的隐含拥有关系
                if (entity1.get("type") == "person" and entity2.get("type") == "object"):
                    relation = DeepImplicitRelation(
                        id=f"ownership_{i}_{j}",
                        source_entity=entity1["name"],
                        target_entity=entity2["name"],
                        relation_type=SemanticRelationType.IMPLICIT_DEPENDENCY,
                        depth=RelationDepth.SHALLOW,
                        confidence=0.7,
                        semantic_evidence=[f"{entity1['name']} 作为人物实体", f"{entity2['name']} 作为物品实体"],
                        logical_basis="人物实体对物品实体的潜在拥有关系",
                        constraint_implications=["非负数量约束", "整数约束"],
                        mathematical_expression=f"ownership({entity1['name']}, {entity2['name']}) ≥ 0",
                        frontend_display_data={
                            "relationship_nature": "potential_ownership",
                            "entity_types": [entity1.get("type"), entity2.get("type")],
                            "reasoning_type": "type_inference"
                        }
                    )
                    relations.append(relation)
                
                # 数量实体间的隐含等价关系
                elif (entity1.get("type") == "concept" and entity2.get("type") == "concept"):
                    if "总" in entity1["name"] or "一共" in entity1["name"]:
                        relation = DeepImplicitRelation(
                            id=f"aggregation_{i}_{j}",
                            source_entity=entity1["name"],
                            target_entity=entity2["name"],
                            relation_type=SemanticRelationType.IMPLICIT_EQUIVALENCE,
                            depth=RelationDepth.MEDIUM,
                            confidence=0.8,
                            semantic_evidence=[f"{entity1['name']} 表示聚合概念", f"{entity2['name']} 为构成部分"],
                            logical_basis="聚合概念与组成部分的等价关系",
                            constraint_implications=["加法守恒定律", "部分小于整体"],
                            mathematical_expression=f"{entity1['name']} = Σ({entity2['name']})",
                            frontend_display_data={
                                "relationship_nature": "aggregation_equivalence",
                                "aggregation_type": "summation",
                                "reasoning_type": "concept_inference"
                            }
                        )
                        relations.append(relation)
        
        return relations
    
    def _infer_from_context_semantics(self, problem_text: str, entities: List[Dict[str, Any]]) -> List[DeepImplicitRelation]:
        """基于上下文语义推理"""
        relations = []
        
        # 分析上下文中的隐含信息
        context_indicators = {
            "买": {"type": SemanticRelationType.DEEP_CAUSALITY, "implications": ["货币减少", "物品增加", "价值交换"]},
            "给": {"type": SemanticRelationType.IMPLICIT_DEPENDENCY, "implications": ["转移关系", "数量重分配"]},
            "剩": {"type": SemanticRelationType.DEEP_CONSERVATION, "implications": ["减法运算", "余量保持"]},
            "一共": {"type": SemanticRelationType.IMPLICIT_EQUIVALENCE, "implications": ["加法聚合", "总量守恒"]}
        }
        
        for indicator, properties in context_indicators.items():
            if indicator in problem_text:
                # 找到相关实体对
                relevant_entities = [e for e in entities if e["name"] in problem_text]
                
                for i, entity1 in enumerate(relevant_entities):
                    for entity2 in relevant_entities[i+1:]:
                        relation = DeepImplicitRelation(
                            id=f"context_{indicator}_{len(relations)}",
                            source_entity=entity1["name"],
                            target_entity=entity2["name"],
                            relation_type=properties["type"],
                            depth=RelationDepth.MEDIUM,
                            confidence=0.75,
                            semantic_evidence=[f"上下文指示词: {indicator}", f"文本片段: {problem_text}"],
                            logical_basis=f"基于上下文指示词 '{indicator}' 的语义推理",
                            constraint_implications=properties["implications"],
                            mathematical_expression=self._generate_context_math_expr(indicator, entity1, entity2),
                            frontend_display_data={
                                "context_indicator": indicator,
                                "semantic_role": properties["type"].value,
                                "reasoning_type": "context_semantics"
                            }
                        )
                        relations.append(relation)
        
        return relations
    
    def _infer_mathematical_semantics(self, problem_text: str, entities: List[Dict[str, Any]]) -> List[DeepImplicitRelation]:
        """基于数学语义推理"""
        relations = []
        
        # 提取数字实体
        numbers = [e for e in entities if e.get("type") == "number" or e["name"].isdigit()]
        
        # 运算语义推理
        if "面积" in problem_text and len(numbers) >= 2:
            # 长度×宽度→面积的深层关系
            relation = DeepImplicitRelation(
                id="geometric_multiplication",
                source_entity=f"dimensions({numbers[0]['name']}, {numbers[1]['name']})",
                target_entity="面积",
                relation_type=SemanticRelationType.DEEP_INVARIANCE,
                depth=RelationDepth.DEEP,
                confidence=0.9,
                semantic_evidence=["几何运算语义", "长方形面积公式"],
                logical_basis="几何学中长度和宽度决定面积的不变性关系",
                constraint_implications=["长度非负", "宽度非负", "面积非负", "乘法交换律"],
                mathematical_expression=f"Area = {numbers[0]['name']} × {numbers[1]['name']}",
                frontend_display_data={
                    "operation_type": "geometric_multiplication",
                    "formula": "长 × 宽 = 面积",
                    "reasoning_type": "mathematical_semantics"
                }
            )
            relations.append(relation)
        
        return relations
    
    def _discover_implicit_constraints(
        self, 
        problem_text: str, 
        entities: List[Dict[str, Any]], 
        relations: List[DeepImplicitRelation]
    ) -> List[ImplicitConstraint]:
        """发现隐含约束条件"""
        constraints = []
        
        # 1. 基于实体类型的约束
        type_constraints = self._discover_type_constraints(entities)
        constraints.extend(type_constraints)
        
        # 2. 基于关系的约束
        relation_constraints = self._discover_relation_constraints(relations)
        constraints.extend(relation_constraints)
        
        # 3. 基于问题域的约束
        domain_constraints = self._discover_domain_constraints(problem_text, entities)
        constraints.extend(domain_constraints)
        
        # 4. 基于数学运算的约束
        math_constraints = self._discover_mathematical_constraints(problem_text, entities)
        constraints.extend(math_constraints)
        
        return constraints
    
    def _discover_type_constraints(self, entities: List[Dict[str, Any]]) -> List[ImplicitConstraint]:
        """基于实体类型发现约束"""
        constraints = []
        
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            
            if entity_type == "object":
                # 物品数量非负约束
                constraint = ImplicitConstraint(
                    id=f"non_negative_{entity['name']}",
                    constraint_type=ConstraintType.NON_NEGATIVITY,
                    description=f"{entity['name']}的数量必须为非负整数",
                    affected_entities=[entity["name"]],
                    constraint_expression=f"count({entity['name']}) ≥ 0 ∧ count({entity['name']}) ∈ ℤ",
                    discovery_method="entity_type_analysis",
                    confidence=0.95,
                    frontend_visualization={
                        "constraint_nature": "quantity_non_negative",
                        "entity_type": entity_type,
                        "visual_indicator": "border_green"
                    }
                )
                constraints.append(constraint)
            
            elif entity_type == "money":
                # 货币守恒约束
                constraint = ImplicitConstraint(
                    id=f"money_conservation_{entity['name']}",
                    constraint_type=ConstraintType.CONSERVATION_LAW,
                    description=f"{entity['name']}在交易过程中遵循守恒定律",
                    affected_entities=[entity["name"]],
                    constraint_expression=f"Σ_before({entity['name']}) = Σ_after({entity['name']})",
                    discovery_method="conservation_principle",
                    confidence=0.9,
                    frontend_visualization={
                        "constraint_nature": "money_conservation",
                        "visual_indicator": "border_gold",
                        "flow_arrows": True
                    }
                )
                constraints.append(constraint)
        
        return constraints
    
    def _discover_relation_constraints(self, relations: List[DeepImplicitRelation]) -> List[ImplicitConstraint]:
        """基于关系发现约束"""
        constraints = []
        
        for relation in relations:
            if relation.relation_type == SemanticRelationType.IMPLICIT_EQUIVALENCE:
                # 等价关系的一致性约束
                constraint = ImplicitConstraint(
                    id=f"equivalence_consistency_{relation.id}",
                    constraint_type=ConstraintType.CONSISTENCY,
                    description=f"{relation.source_entity} 和 {relation.target_entity} 的等价关系一致性",
                    affected_entities=[relation.source_entity, relation.target_entity],
                    constraint_expression=f"{relation.source_entity} ⟺ {relation.target_entity}",
                    discovery_method="relation_analysis",
                    confidence=relation.confidence * 0.9,
                    frontend_visualization={
                        "constraint_nature": "equivalence_consistency",
                        "relation_id": relation.id,
                        "visual_indicator": "double_arrow"
                    }
                )
                constraints.append(constraint)
            
            elif relation.relation_type == SemanticRelationType.DEEP_CONSERVATION:
                # 守恒关系的平衡约束
                constraint = ImplicitConstraint(
                    id=f"conservation_balance_{relation.id}",
                    constraint_type=ConstraintType.CONSERVATION_LAW,
                    description=f"{relation.source_entity} 到 {relation.target_entity} 的守恒平衡",
                    affected_entities=[relation.source_entity, relation.target_entity],
                    constraint_expression=f"Δ{relation.source_entity} + Δ{relation.target_entity} = 0",
                    discovery_method="conservation_analysis",
                    confidence=relation.confidence,
                    frontend_visualization={
                        "constraint_nature": "conservation_balance",
                        "relation_id": relation.id,
                        "visual_indicator": "balance_scale"
                    }
                )
                constraints.append(constraint)
        
        return constraints
    
    def _discover_domain_constraints(self, problem_text: str, entities: List[Dict[str, Any]]) -> List[ImplicitConstraint]:
        """基于问题域发现约束"""
        constraints = []
        
        # 购物场景约束
        if any(word in problem_text for word in ["买", "购", "付", "花", "钱"]):
            money_entities = [e for e in entities if e.get("type") == "money"]
            for money_entity in money_entities:
                constraint = ImplicitConstraint(
                    id=f"shopping_constraint_{money_entity['name']}",
                    constraint_type=ConstraintType.CONTEXT_BOUNDARY,
                    description=f"购物场景中{money_entity['name']}的使用约束",
                    affected_entities=[money_entity["name"]],
                    constraint_expression=f"spent({money_entity['name']}) ≤ available({money_entity['name']})",
                    discovery_method="domain_knowledge",
                    confidence=0.85,
                    frontend_visualization={
                        "constraint_nature": "shopping_limit",
                        "domain": "shopping",
                        "visual_indicator": "wallet_limit"
                    }
                )
                constraints.append(constraint)
        
        # 几何场景约束
        if any(word in problem_text for word in ["面积", "周长", "长", "宽", "高"]):
            dimension_entities = [e for e in entities if any(dim in e["name"] for dim in ["长", "宽", "高"])]
            for dim_entity in dimension_entities:
                constraint = ImplicitConstraint(
                    id=f"geometric_constraint_{dim_entity['name']}",
                    constraint_type=ConstraintType.NON_NEGATIVITY,
                    description=f"几何维度{dim_entity['name']}的非负约束",
                    affected_entities=[dim_entity["name"]],
                    constraint_expression=f"{dim_entity['name']} > 0",
                    discovery_method="geometric_domain",
                    confidence=0.95,
                    frontend_visualization={
                        "constraint_nature": "geometric_positive",
                        "domain": "geometry",
                        "visual_indicator": "ruler_positive"
                    }
                )
                constraints.append(constraint)
        
        return constraints
    
    def _discover_mathematical_constraints(self, problem_text: str, entities: List[Dict[str, Any]]) -> List[ImplicitConstraint]:
        """基于数学运算发现约束"""
        constraints = []
        
        # 加法约束
        if any(word in problem_text for word in ["加", "和", "一共", "总共"]):
            constraint = ImplicitConstraint(
                id="addition_constraint",
                constraint_type=ConstraintType.COMPLETENESS,
                description="加法运算的完整性约束",
                affected_entities=[e["name"] for e in entities if e.get("type") in ["number", "object"]],
                constraint_expression="Σ(parts) = total",
                discovery_method="operation_analysis",
                confidence=0.9,
                frontend_visualization={
                    "constraint_nature": "addition_completeness",
                    "operation": "addition",
                    "visual_indicator": "sum_symbol"
                }
            )
            constraints.append(constraint)
        
        # 乘法约束
        if any(word in problem_text for word in ["乘", "倍", "×", "面积"]):
            constraint = ImplicitConstraint(
                id="multiplication_constraint",
                constraint_type=ConstraintType.TYPE_COMPATIBILITY,
                description="乘法运算的类型兼容性约束",
                affected_entities=[e["name"] for e in entities if e.get("type") == "number"],
                constraint_expression="∀a,b: multiply(a,b) → compatible_units(a,b)",
                discovery_method="operation_analysis",
                confidence=0.85,
                frontend_visualization={
                    "constraint_nature": "multiplication_compatibility",
                    "operation": "multiplication",
                    "visual_indicator": "multiply_symbol"
                }
            )
            constraints.append(constraint)
        
        return constraints
    
    def _build_multilayer_relation_model(
        self, 
        entities: List[Dict[str, Any]], 
        relations: List[DeepImplicitRelation], 
        constraints: List[ImplicitConstraint]
    ) -> List[DeepImplicitRelation]:
        """构建多层关系建模"""
        multilayer_relations = relations.copy()
        
        # 第一层：实体直接关系
        direct_relations = [r for r in relations if r.depth in [RelationDepth.SURFACE, RelationDepth.SHALLOW]]
        
        # 第二层：基于约束的间接关系
        constraint_relations = self._derive_relations_from_constraints(entities, constraints)
        multilayer_relations.extend(constraint_relations)
        
        # 第三层：传递性关系推理
        transitive_relations = self._derive_transitive_relations(direct_relations)
        multilayer_relations.extend(transitive_relations)
        
        # 第四层：整体性关系推理
        holistic_relations = self._derive_holistic_relations(entities, multilayer_relations, constraints)
        multilayer_relations.extend(holistic_relations)
        
        return multilayer_relations
    
    def _derive_relations_from_constraints(
        self, 
        entities: List[Dict[str, Any]], 
        constraints: List[ImplicitConstraint]
    ) -> List[DeepImplicitRelation]:
        """从约束中推导关系"""
        relations = []
        
        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.CONSERVATION_LAW:
                # 守恒约束暗示实体间的平衡关系
                affected = constraint.affected_entities
                if len(affected) >= 2:
                    relation = DeepImplicitRelation(
                        id=f"constraint_derived_{constraint.id}",
                        source_entity=affected[0],
                        target_entity=affected[1],
                        relation_type=SemanticRelationType.DEEP_CONSERVATION,
                        depth=RelationDepth.DEEP,
                        confidence=constraint.confidence * 0.8,
                        semantic_evidence=[f"来自约束: {constraint.description}"],
                        logical_basis=f"基于{constraint.constraint_type.value}约束的推导",
                        constraint_implications=[constraint.description],
                        mathematical_expression=constraint.constraint_expression,
                        frontend_display_data={
                            "derived_from_constraint": constraint.id,
                            "constraint_type": constraint.constraint_type.value,
                            "reasoning_type": "constraint_derivation"
                        }
                    )
                    relations.append(relation)
        
        return relations
    
    def _derive_transitive_relations(self, relations: List[DeepImplicitRelation]) -> List[DeepImplicitRelation]:
        """推导传递性关系"""
        transitive_relations = []
        
        for i, rel1 in enumerate(relations):
            for j, rel2 in enumerate(relations[i+1:], i+1):
                # 查找传递性连接：A→B, B→C ⇒ A→C
                if rel1.target_entity == rel2.source_entity:
                    confidence = min(rel1.confidence, rel2.confidence) * 0.7  # 传递性降低置信度
                    
                    transitive_relation = DeepImplicitRelation(
                        id=f"transitive_{rel1.id}_{rel2.id}",
                        source_entity=rel1.source_entity,
                        target_entity=rel2.target_entity,
                        relation_type=SemanticRelationType.IMPLICIT_DEPENDENCY,
                        depth=RelationDepth.DEEP,
                        confidence=confidence,
                        semantic_evidence=[
                            f"传递性推理: {rel1.source_entity}→{rel1.target_entity}→{rel2.target_entity}",
                            f"基于关系: {rel1.id}, {rel2.id}"
                        ],
                        logical_basis="传递性关系推理",
                        constraint_implications=rel1.constraint_implications + rel2.constraint_implications,
                        mathematical_expression=f"transitive({rel1.source_entity}, {rel2.target_entity})",
                        frontend_display_data={
                            "relation_chain": [rel1.id, rel2.id],
                            "reasoning_type": "transitive_inference",
                            "transitivity_depth": 2
                        }
                    )
                    transitive_relations.append(transitive_relation)
        
        return transitive_relations
    
    def _derive_holistic_relations(
        self, 
        entities: List[Dict[str, Any]], 
        relations: List[DeepImplicitRelation], 
        constraints: List[ImplicitConstraint]
    ) -> List[DeepImplicitRelation]:
        """推导整体性关系"""
        holistic_relations = []
        
        # 识别系统级的整体性关系
        entity_groups = self._group_entities_by_semantic_role(entities)
        
        for group_name, group_entities in entity_groups.items():
            if len(group_entities) > 1:
                # 创建组内实体的整体性关系
                for i, entity1 in enumerate(group_entities):
                    for entity2 in group_entities[i+1:]:
                        holistic_relation = DeepImplicitRelation(
                            id=f"holistic_{group_name}_{entity1['name']}_{entity2['name']}",
                            source_entity=entity1["name"],
                            target_entity=entity2["name"],
                            relation_type=SemanticRelationType.IMPLICIT_CONSTRAINT,
                            depth=RelationDepth.DEEP,
                            confidence=0.6,
                            semantic_evidence=[f"同属语义组: {group_name}"],
                            logical_basis=f"基于语义组 '{group_name}' 的整体性关系",
                            constraint_implications=[f"组内实体一致性约束"],
                            mathematical_expression=f"same_group({entity1['name']}, {entity2['name']})",
                            frontend_display_data={
                                "semantic_group": group_name,
                                "group_size": len(group_entities),
                                "reasoning_type": "holistic_inference"
                            }
                        )
                        holistic_relations.append(holistic_relation)
        
        return holistic_relations
    
    def _enhance_for_frontend_visualization(
        self, 
        relations: List[DeepImplicitRelation], 
        constraints: List[ImplicitConstraint]
    ):
        """为前端可视化增强数据"""
        
        # 为关系增加可视化属性
        for relation in relations:
            relation.frontend_display_data.update({
                "depth_color": self._get_depth_color(relation.depth),
                "confidence_size": relation.confidence * 40 + 20,  # 节点大小
                "relation_width": relation.confidence * 5 + 1,     # 连线宽度
                "animation_delay": hash(relation.id) % 100 * 0.01,  # 动画延迟
                "hover_info": {
                    "title": relation._generate_display_label(),
                    "details": relation.semantic_evidence,
                    "constraints": relation.constraint_implications
                }
            })
        
        # 为约束增加可视化属性
        for constraint in constraints:
            constraint.frontend_visualization.update({
                "constraint_priority": self._get_constraint_priority(constraint.constraint_type),
                "visualization_layer": self._get_constraint_layer(constraint.constraint_type),
                "animation_type": self._get_constraint_animation(constraint.constraint_type),
                "detail_panel": {
                    "title": constraint.description,
                    "expression": constraint.constraint_expression,
                    "method": constraint.discovery_method,
                    "entities": constraint.affected_entities
                }
            })
    
    def _get_depth_color(self, depth: RelationDepth) -> str:
        """获取深度对应的颜色"""
        colors = {
            RelationDepth.SURFACE: "#bdc3c7",    # 浅灰
            RelationDepth.SHALLOW: "#3498db",    # 蓝色
            RelationDepth.MEDIUM: "#9b59b6",     # 紫色
            RelationDepth.DEEP: "#e74c3c"        # 红色
        }
        return colors.get(depth, "#95a5a6")
    
    def _get_constraint_priority(self, constraint_type: ConstraintType) -> int:
        """获取约束优先级"""
        priorities = {
            ConstraintType.CONSERVATION_LAW: 1,
            ConstraintType.NON_NEGATIVITY: 2,
            ConstraintType.CONSISTENCY: 3,
            ConstraintType.TYPE_COMPATIBILITY: 4,
            ConstraintType.COMPLETENESS: 5
        }
        return priorities.get(constraint_type, 9)
    
    # 其他辅助方法...
    def _initialize_semantic_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化语义模式"""
        return {
            "ownership_pattern": {
                "pattern": r"(\w+)有(\d+)个?(\w+)",
                "relation_type": SemanticRelationType.EXPLICIT_OWNERSHIP,
                "confidence": 0.9,
                "math_expr": "owns(person, count, object)"
            },
            "aggregation_pattern": {
                "pattern": r"一共|总共|合计",
                "relation_type": SemanticRelationType.IMPLICIT_EQUIVALENCE,
                "confidence": 0.8,
                "math_expr": "total = sum(parts)"
            },
            "transaction_pattern": {
                "pattern": r"(\w+)买(\w+)花了?(\d+)元",
                "relation_type": SemanticRelationType.DEEP_CAUSALITY,
                "confidence": 0.85,
                "math_expr": "transaction(buyer, item, cost)"
            }
        }
    
    def _initialize_constraint_rules(self) -> Dict[str, Any]:
        """初始化约束规则"""
        return {
            "quantity_non_negative": {"type": ConstraintType.NON_NEGATIVITY, "confidence": 0.95},
            "money_conservation": {"type": ConstraintType.CONSERVATION_LAW, "confidence": 0.9},
            "operation_consistency": {"type": ConstraintType.CONSISTENCY, "confidence": 0.8}
        }
    
    def _initialize_domain_knowledge(self) -> Dict[str, Any]:
        """初始化领域知识"""
        return {
            "mathematics": {
                "operations": ["加", "减", "乘", "除"],
                "properties": ["交换律", "结合律", "分配律"],
                "constraints": ["非负性", "连续性", "守恒性"]
            },
            "geometry": {
                "shapes": ["长方形", "正方形", "圆形"],
                "measures": ["长度", "宽度", "面积", "周长"],
                "relations": ["长×宽=面积", "2×(长+宽)=周长"]
            }
        }
    
    def _extract_entities_from_match(self, match, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从匹配中提取实体"""
        matched_entities = []
        match_text = match.group()
        
        for entity in entities:
            if entity["name"] in match_text:
                matched_entities.append(entity)
        
        return matched_entities
    
    def _generate_context_math_expr(self, indicator: str, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> str:
        """生成上下文数学表达式"""
        expressions = {
            "买": f"transaction({entity1['name']}, {entity2['name']})",
            "给": f"transfer({entity1['name']}, {entity2['name']})",
            "剩": f"remainder({entity1['name']}, {entity2['name']})",
            "一共": f"sum({entity1['name']}, {entity2['name']})"
        }
        return expressions.get(indicator, f"relation({entity1['name']}, {entity2['name']})")
    
    def _group_entities_by_semantic_role(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按语义角色分组实体"""
        groups = {
            "agents": [],      # 主体实体
            "objects": [],     # 客体实体
            "quantities": [],  # 数量实体
            "concepts": []     # 概念实体
        }
        
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            if entity_type == "person":
                groups["agents"].append(entity)
            elif entity_type == "object":
                groups["objects"].append(entity)
            elif entity_type == "number":
                groups["quantities"].append(entity)
            else:
                groups["concepts"].append(entity)
        
        return groups
    
    def _get_constraint_layer(self, constraint_type: ConstraintType) -> str:
        """获取约束可视化层级"""
        layers = {
            ConstraintType.CONSERVATION_LAW: "background",
            ConstraintType.NON_NEGATIVITY: "foreground",
            ConstraintType.CONSISTENCY: "overlay"
        }
        return layers.get(constraint_type, "default")
    
    def _get_constraint_animation(self, constraint_type: ConstraintType) -> str:
        """获取约束动画类型"""
        animations = {
            ConstraintType.CONSERVATION_LAW: "pulse",
            ConstraintType.NON_NEGATIVITY: "highlight",
            ConstraintType.CONSISTENCY: "fade"
        }
        return animations.get(constraint_type, "none")