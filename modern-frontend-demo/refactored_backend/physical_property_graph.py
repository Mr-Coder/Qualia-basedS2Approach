#!/usr/bin/env python3
"""
物性图谱模块
Physical Property Graph Module
基于物理属性和约束关系构建数学问题的物性推理图谱
"""

import logging
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from problem_preprocessor import ProcessedProblem
from qs2_semantic_analyzer import SemanticEntity
from ird_relation_discovery import RelationNetwork, ImplicitRelation

logger = logging.getLogger(__name__)

class PhysicalPropertyType(Enum):
    """物理属性类型"""
    CONSERVATION = "conservation"       # 守恒性
    DISCRETENESS = "discreteness"       # 离散性 
    CONTINUITY = "continuity"          # 连续性
    ADDITIVITY = "additivity"          # 可加性
    MEASURABILITY = "measurability"    # 可测量性
    LOCALITY = "locality"              # 局域性
    TEMPORALITY = "temporality"        # 时间性
    CAUSALITY = "causality"            # 因果性

class ConstraintType(Enum):
    """约束类型"""
    CONSERVATION_LAW = "conservation_law"     # 守恒定律
    NON_NEGATIVE = "non_negative"             # 非负约束
    INTEGER_CONSTRAINT = "integer_constraint" # 整数约束
    UPPER_BOUND = "upper_bound"               # 上界约束
    LOWER_BOUND = "lower_bound"               # 下界约束
    EQUIVALENCE = "equivalence"               # 等价约束
    ORDERING = "ordering"                     # 序关系约束
    EXCLUSIVITY = "exclusivity"               # 互斥约束

@dataclass
class PhysicalProperty:
    """物理属性"""
    property_id: str
    property_type: PhysicalPropertyType
    entity_id: str
    value: Any
    unit: str
    certainty: float
    measurement_method: str
    constraints: List[str]
    dependencies: List[str]
    
@dataclass
class PhysicalConstraint:
    """物理约束"""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    mathematical_expression: str
    involved_entities: List[str]
    involved_properties: List[str]
    strength: float
    violation_penalty: float
    enforcement_method: str

@dataclass
class PhysicalRelation:
    """物理关系"""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    physical_basis: str
    strength: float
    properties_involved: List[str]
    constraints_imposed: List[str]
    causal_direction: Optional[str]

@dataclass
class PropertyGraph:
    """物性图谱"""
    entities: List[SemanticEntity]
    properties: List[PhysicalProperty]
    constraints: List[PhysicalConstraint]
    relations: List[PhysicalRelation]
    graph_metrics: Dict[str, float]
    consistency_score: float

class PhysicalPropertyGraphBuilder:
    """物性图谱构建器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 物理属性识别规则
        self.property_rules = {
            "count": {
                "type": PhysicalPropertyType.DISCRETENESS,
                "constraints": [ConstraintType.NON_NEGATIVE, ConstraintType.INTEGER_CONSTRAINT],
                "unit": "个",
                "additive": True
            },
            "mass": {
                "type": PhysicalPropertyType.CONSERVATION,
                "constraints": [ConstraintType.NON_NEGATIVE],
                "unit": "kg",
                "additive": True
            },
            "area": {
                "type": PhysicalPropertyType.CONTINUITY,
                "constraints": [ConstraintType.NON_NEGATIVE],
                "unit": "m²",
                "additive": False
            },
            "money": {
                "type": PhysicalPropertyType.DISCRETENESS,
                "constraints": [ConstraintType.NON_NEGATIVE],
                "unit": "元",
                "additive": True
            }
        }
        
        # 物理约束传播规则
        self.constraint_propagation_rules = {
            "conservation": self._apply_conservation_constraint,
            "additivity": self._apply_additivity_constraint,
            "ordering": self._apply_ordering_constraint,
            "boundary": self._apply_boundary_constraint
        }

    def build_property_graph(self, processed_problem: ProcessedProblem,
                           semantic_entities: List[SemanticEntity],
                           relation_network: RelationNetwork) -> PropertyGraph:
        """
        构建物性图谱
        
        Args:
            processed_problem: 预处理后的问题
            semantic_entities: 语义实体列表
            relation_network: 关系网络
            
        Returns:
            PropertyGraph: 物性图谱
        """
        try:
            self.logger.info(f"开始构建物性图谱，实体数量: {len(semantic_entities)}")
            
            # Step 1: 分析实体的物理属性
            properties = self._analyze_physical_properties(semantic_entities, processed_problem)
            
            # Step 2: 识别物理约束
            constraints = self._identify_physical_constraints(properties, processed_problem)
            
            # Step 3: 构建物理关系
            physical_relations = self._build_physical_relations(
                semantic_entities, relation_network, properties, constraints
            )
            
            # Step 4: 约束传播和一致性检查
            consistency_score = self._propagate_constraints_and_check_consistency(
                properties, constraints, physical_relations
            )
            
            # Step 5: 计算图谱指标
            graph_metrics = self._calculate_graph_metrics(
                semantic_entities, properties, constraints, physical_relations
            )
            
            property_graph = PropertyGraph(
                entities=semantic_entities,
                properties=properties,
                constraints=constraints,
                relations=physical_relations,
                graph_metrics=graph_metrics,
                consistency_score=consistency_score
            )
            
            self.logger.info(f"物性图谱构建完成，属性{len(properties)}个，约束{len(constraints)}个，关系{len(physical_relations)}个")
            return property_graph
            
        except Exception as e:
            self.logger.error(f"物性图谱构建失败: {e}")
            return PropertyGraph(
                entities=semantic_entities,
                properties=[],
                constraints=[],
                relations=[],
                graph_metrics={},
                consistency_score=0.0
            )

    def _analyze_physical_properties(self, entities: List[SemanticEntity], 
                                   problem: ProcessedProblem) -> List[PhysicalProperty]:
        """分析实体的物理属性"""
        
        properties = []
        property_id = 1
        
        for entity in entities:
            # 根据实体类型和上下文识别物理属性
            if entity.entity_type == "number":
                # 数字实体的物理属性
                value = float(entity.name)
                
                # 识别数字的物理意义
                physical_meaning = self._identify_physical_meaning(entity, problem)
                
                prop = PhysicalProperty(
                    property_id=f"prop_{property_id}",
                    property_type=physical_meaning["type"],
                    entity_id=entity.entity_id,
                    value=value,
                    unit=physical_meaning["unit"],
                    certainty=entity.confidence,
                    measurement_method="direct_observation",
                    constraints=[c.value for c in physical_meaning["constraints"]],
                    dependencies=[]
                )
                properties.append(prop)
                property_id += 1
                
            elif entity.entity_type in ["person", "object"]:
                # 实体的拥有关系属性
                ownership_prop = PhysicalProperty(
                    property_id=f"prop_{property_id}",
                    property_type=PhysicalPropertyType.LOCALITY,
                    entity_id=entity.entity_id,
                    value="possessor",
                    unit="entity",
                    certainty=entity.confidence,
                    measurement_method="semantic_analysis",
                    constraints=[ConstraintType.EXCLUSIVITY.value],
                    dependencies=[]
                )
                properties.append(ownership_prop)
                property_id += 1
        
        return properties

    def _identify_physical_meaning(self, entity: SemanticEntity, 
                                 problem: ProcessedProblem) -> Dict[str, Any]:
        """识别数字的物理意义"""
        
        text = problem.cleaned_text
        
        # 根据上下文判断物理意义
        if any(unit in text for unit in ["个", "只", "本", "支"]):
            return self.property_rules["count"]
        elif any(unit in text for unit in ["元", "角", "分"]):
            return self.property_rules["money"]
        elif any(unit in text for unit in ["米", "厘米", "平方米"]):
            if "面积" in text or "平方" in text:
                return self.property_rules["area"]
            else:
                return {
                    "type": PhysicalPropertyType.CONTINUITY,
                    "constraints": [ConstraintType.NON_NEGATIVE],
                    "unit": "米",
                    "additive": False
                }
        else:
            # 默认为计数属性
            return self.property_rules["count"]

    def _identify_physical_constraints(self, properties: List[PhysicalProperty],
                                     problem: ProcessedProblem) -> List[PhysicalConstraint]:
        """识别物理约束"""
        
        constraints = []
        constraint_id = 1
        
        # 1. 守恒约束
        additive_properties = [p for p in properties if self._is_additive_property(p)]
        if len(additive_properties) >= 2:
            conservation_constraint = PhysicalConstraint(
                constraint_id=f"cons_{constraint_id}",
                constraint_type=ConstraintType.CONSERVATION_LAW,
                description="数量守恒定律",
                mathematical_expression="∑(individual_quantities) = total_quantity",
                involved_entities=[p.entity_id for p in additive_properties],
                involved_properties=[p.property_id for p in additive_properties],
                strength=1.0,
                violation_penalty=1000.0,
                enforcement_method="hard_constraint"
            )
            constraints.append(conservation_constraint)
            constraint_id += 1
        
        # 2. 非负约束
        for prop in properties:
            if ConstraintType.NON_NEGATIVE.value in prop.constraints:
                non_negative_constraint = PhysicalConstraint(
                    constraint_id=f"cons_{constraint_id}",
                    constraint_type=ConstraintType.NON_NEGATIVE,
                    description=f"{prop.entity_id}的值必须非负",
                    mathematical_expression=f"value({prop.entity_id}) ≥ 0",
                    involved_entities=[prop.entity_id],
                    involved_properties=[prop.property_id],
                    strength=1.0,
                    violation_penalty=100.0,
                    enforcement_method="hard_constraint"
                )
                constraints.append(non_negative_constraint)
                constraint_id += 1
        
        # 3. 整数约束
        for prop in properties:
            if ConstraintType.INTEGER_CONSTRAINT.value in prop.constraints:
                integer_constraint = PhysicalConstraint(
                    constraint_id=f"cons_{constraint_id}",
                    constraint_type=ConstraintType.INTEGER_CONSTRAINT,
                    description=f"{prop.entity_id}必须为整数",
                    mathematical_expression=f"value({prop.entity_id}) ∈ ℤ",
                    involved_entities=[prop.entity_id],
                    involved_properties=[prop.property_id],
                    strength=1.0,
                    violation_penalty=50.0,
                    enforcement_method="hard_constraint"
                )
                constraints.append(integer_constraint)
                constraint_id += 1
        
        return constraints

    def _build_physical_relations(self, entities: List[SemanticEntity],
                                relation_network: RelationNetwork,
                                properties: List[PhysicalProperty],
                                constraints: List[PhysicalConstraint]) -> List[PhysicalRelation]:
        """构建物理关系"""
        
        physical_relations = []
        relation_id = 1
        
        if not relation_network or not relation_network.relations:
            return physical_relations
        
        for semantic_relation in relation_network.relations:
            # 识别关系的物理基础
            physical_basis = self._identify_physical_basis(semantic_relation, properties)
            
            # 确定涉及的属性
            involved_properties = self._get_involved_properties(
                semantic_relation, properties
            )
            
            # 确定施加的约束
            imposed_constraints = self._get_imposed_constraints(
                semantic_relation, constraints
            )
            
            # 确定因果方向
            causal_direction = self._determine_causal_direction(semantic_relation)
            
            physical_relation = PhysicalRelation(
                relation_id=f"phys_rel_{relation_id}",
                source_entity_id=semantic_relation.source_entity_id,
                target_entity_id=semantic_relation.target_entity_id,
                relation_type=semantic_relation.relation_type,
                physical_basis=physical_basis,
                strength=semantic_relation.strength,
                properties_involved=involved_properties,
                constraints_imposed=imposed_constraints,
                causal_direction=causal_direction
            )
            
            physical_relations.append(physical_relation)
            relation_id += 1
        
        return physical_relations

    def _propagate_constraints_and_check_consistency(self, 
                                                   properties: List[PhysicalProperty],
                                                   constraints: List[PhysicalConstraint],
                                                   relations: List[PhysicalRelation]) -> float:
        """约束传播和一致性检查"""
        
        consistency_score = 1.0
        violations = []
        
        # 检查每个约束是否被满足
        for constraint in constraints:
            violation = self._check_constraint_violation(constraint, properties)
            if violation:
                violations.append(violation)
                consistency_score *= (1 - violation["severity"])
        
        # 检查关系的一致性
        for relation in relations:
            relation_consistency = self._check_relation_consistency(relation, properties)
            consistency_score *= relation_consistency
        
        if violations:
            self.logger.warning(f"发现{len(violations)}个约束违反")
        
        return max(consistency_score, 0.0)

    def _calculate_graph_metrics(self, entities: List[SemanticEntity],
                               properties: List[PhysicalProperty],
                               constraints: List[PhysicalConstraint],
                               relations: List[PhysicalRelation]) -> Dict[str, float]:
        """计算图谱指标"""
        
        metrics = {}
        
        # 基础统计指标
        metrics["entity_count"] = len(entities)
        metrics["property_count"] = len(properties)
        metrics["constraint_count"] = len(constraints)
        metrics["relation_count"] = len(relations)
        
        # 密度指标
        max_relations = len(entities) * (len(entities) - 1) / 2
        metrics["relation_density"] = len(relations) / max(max_relations, 1)
        
        # 约束强度指标
        if constraints:
            metrics["average_constraint_strength"] = sum(c.strength for c in constraints) / len(constraints)
            metrics["hard_constraint_ratio"] = sum(1 for c in constraints if c.enforcement_method == "hard_constraint") / len(constraints)
        else:
            metrics["average_constraint_strength"] = 0.0
            metrics["hard_constraint_ratio"] = 0.0
        
        # 物理属性分布
        property_types = {}
        for prop in properties:
            prop_type = prop.property_type.value
            property_types[prop_type] = property_types.get(prop_type, 0) + 1
        metrics["property_type_diversity"] = len(property_types)
        
        # 因果关系指标
        causal_relations = [r for r in relations if r.causal_direction]
        metrics["causal_relation_ratio"] = len(causal_relations) / max(len(relations), 1)
        
        return metrics

    def _is_additive_property(self, prop: PhysicalProperty) -> bool:
        """判断属性是否具有可加性"""
        additive_types = [
            PhysicalPropertyType.DISCRETENESS,
            PhysicalPropertyType.CONSERVATION
        ]
        return prop.property_type in additive_types

    def _identify_physical_basis(self, relation: ImplicitRelation, 
                               properties: List[PhysicalProperty]) -> str:
        """识别关系的物理基础"""
        
        if relation.relation_type == "ownership":
            return "locality_principle"
        elif relation.relation_type == "quantity":
            return "additivity_principle"
        elif relation.relation_type == "mathematical":
            return "conservation_principle"
        else:
            return "semantic_association"

    def _get_involved_properties(self, relation: ImplicitRelation,
                               properties: List[PhysicalProperty]) -> List[str]:
        """获取关系涉及的属性"""
        
        involved = []
        for prop in properties:
            if prop.entity_id in [relation.source_entity_id, relation.target_entity_id]:
                involved.append(prop.property_id)
        
        return involved

    def _get_imposed_constraints(self, relation: ImplicitRelation,
                               constraints: List[PhysicalConstraint]) -> List[str]:
        """获取关系施加的约束"""
        
        imposed = []
        for constraint in constraints:
            if (relation.source_entity_id in constraint.involved_entities or
                relation.target_entity_id in constraint.involved_entities):
                imposed.append(constraint.constraint_id)
        
        return imposed

    def _determine_causal_direction(self, relation: ImplicitRelation) -> Optional[str]:
        """确定因果方向"""
        
        if relation.relation_type == "ownership":
            return f"{relation.source_entity_id} -> {relation.target_entity_id}"
        elif relation.relation_type == "quantity":
            return f"{relation.target_entity_id} -> {relation.source_entity_id}"
        else:
            return None

    def _check_constraint_violation(self, constraint: PhysicalConstraint,
                                  properties: List[PhysicalProperty]) -> Optional[Dict[str, Any]]:
        """检查约束违反"""
        
        # 简化的约束检查逻辑
        for prop_id in constraint.involved_properties:
            prop = next((p for p in properties if p.property_id == prop_id), None)
            if prop:
                if constraint.constraint_type == ConstraintType.NON_NEGATIVE:
                    if isinstance(prop.value, (int, float)) and prop.value < 0:
                        return {
                            "constraint_id": constraint.constraint_id,
                            "violation_type": "non_negative",
                            "severity": 0.8
                        }
        
        return None

    def _check_relation_consistency(self, relation: PhysicalRelation,
                                  properties: List[PhysicalProperty]) -> float:
        """检查关系一致性"""
        
        # 简化的一致性检查
        source_props = [p for p in properties if p.entity_id == relation.source_entity_id]
        target_props = [p for p in properties if p.entity_id == relation.target_entity_id]
        
        if source_props and target_props:
            # 检查属性类型兼容性
            compatible_types = self._check_property_type_compatibility(source_props, target_props)
            return 0.9 if compatible_types else 0.7
        
        return 0.8

    def _check_property_type_compatibility(self, source_props: List[PhysicalProperty],
                                         target_props: List[PhysicalProperty]) -> bool:
        """检查属性类型兼容性"""
        
        source_types = set(p.property_type for p in source_props)
        target_types = set(p.property_type for p in target_props)
        
        # 简化的兼容性规则
        compatible_combinations = [
            (PhysicalPropertyType.LOCALITY, PhysicalPropertyType.DISCRETENESS),
            (PhysicalPropertyType.DISCRETENESS, PhysicalPropertyType.DISCRETENESS),
            (PhysicalPropertyType.CONSERVATION, PhysicalPropertyType.CONSERVATION)
        ]
        
        for source_type in source_types:
            for target_type in target_types:
                if (source_type, target_type) in compatible_combinations:
                    return True
        
        return False

    # 约束传播方法
    def _apply_conservation_constraint(self, constraint: PhysicalConstraint,
                                     properties: List[PhysicalProperty]) -> Dict[str, Any]:
        """应用守恒约束"""
        return {"status": "applied", "method": "conservation_propagation"}

    def _apply_additivity_constraint(self, constraint: PhysicalConstraint,
                                   properties: List[PhysicalProperty]) -> Dict[str, Any]:
        """应用可加性约束"""
        return {"status": "applied", "method": "additivity_propagation"}

    def _apply_ordering_constraint(self, constraint: PhysicalConstraint,
                                 properties: List[PhysicalProperty]) -> Dict[str, Any]:
        """应用序关系约束"""
        return {"status": "applied", "method": "ordering_propagation"}

    def _apply_boundary_constraint(self, constraint: PhysicalConstraint,
                                 properties: List[PhysicalProperty]) -> Dict[str, Any]:
        """应用边界约束"""
        return {"status": "applied", "method": "boundary_propagation"}

# 测试函数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from problem_preprocessor import ProblemPreprocessor
    from qs2_semantic_analyzer import QS2SemanticAnalyzer
    from ird_relation_discovery import IRDRelationDiscovery
    
    # 创建组件
    preprocessor = ProblemPreprocessor()
    qs2_analyzer = QS2SemanticAnalyzer()
    ird_discovery = IRDRelationDiscovery(qs2_analyzer)
    property_graph_builder = PhysicalPropertyGraphBuilder()
    
    # 测试问题
    test_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    print(f"测试问题: {test_problem}")
    print("="*60)
    
    # 执行完整流程
    processed = preprocessor.preprocess(test_problem)
    semantic_entities = qs2_analyzer.analyze_semantics(processed)
    relation_network = ird_discovery.discover_relations(semantic_entities, test_problem)
    
    # 构建物性图谱
    property_graph = property_graph_builder.build_property_graph(
        processed, semantic_entities, relation_network
    )
    
    print(f"物性图谱分析结果:")
    print(f"实体数量: {len(property_graph.entities)}")
    print(f"物理属性数量: {len(property_graph.properties)}")
    print(f"物理约束数量: {len(property_graph.constraints)}")
    print(f"物理关系数量: {len(property_graph.relations)}")
    print(f"一致性得分: {property_graph.consistency_score:.3f}")
    
    print(f"\n物理属性详情:")
    for prop in property_graph.properties:
        print(f"  {prop.property_id}: {prop.property_type.value} = {prop.value}{prop.unit} (确定性: {prop.certainty:.2f})")
    
    print(f"\n物理约束详情:")
    for constraint in property_graph.constraints:
        print(f"  {constraint.constraint_id}: {constraint.description}")
        print(f"    数学表达: {constraint.mathematical_expression}")
        print(f"    约束强度: {constraint.strength:.2f}")
    
    print(f"\n物理关系详情:")
    for relation in property_graph.relations:
        print(f"  {relation.relation_id}: {relation.source_entity_id} -> {relation.target_entity_id}")
        print(f"    物理基础: {relation.physical_basis}")
        print(f"    关系强度: {relation.strength:.3f}")
        if relation.causal_direction:
            print(f"    因果方向: {relation.causal_direction}")
    
    print(f"\n图谱指标:")
    for metric, value in property_graph.graph_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")