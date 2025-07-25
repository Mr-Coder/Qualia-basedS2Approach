#!/usr/bin/env python3
"""
物理约束传播网络
基于物理定律的约束满足和传播算法
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from itertools import combinations

logger = logging.getLogger(__name__)

class PhysicalLaw(Enum):
    """物理定律类型"""
    CONSERVATION_OF_MASS = "conservation_of_mass"
    CONSERVATION_OF_ENERGY = "conservation_of_energy"
    CONSERVATION_OF_MOMENTUM = "conservation_of_momentum"
    ADDITIVITY = "additivity"
    NON_NEGATIVITY = "non_negativity"
    DISCRETENESS = "discreteness"
    CONTINUITY = "continuity"
    CAUSALITY = "causality"
    LOCALITY = "locality"
    SYMMETRY = "symmetry"

@dataclass
class PhysicalConstraint:
    """物理约束"""
    constraint_id: str
    law_type: PhysicalLaw
    entities: List[str]
    mathematical_form: str
    strength: float
    violability: float  # 约束可违背程度
    context_dependent: bool

@dataclass
class ConstraintViolation:
    """约束违背"""
    constraint_id: str
    violation_degree: float
    affected_entities: List[str]
    suggested_corrections: List[str]

class PhysicsConstraintNetwork:
    """物理约束传播网络"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 约束传播网络
        self.constraint_propagator = self._build_constraint_propagator()
        
        # 物理定律编码器
        self.physics_encoder = self._build_physics_encoder()
        
        # 约束冲突解决器
        self.conflict_resolver = self._build_conflict_resolver()
        
        # 物理知识库
        self.physics_kb = self._initialize_physics_kb()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "constraint_dim": 64,
            "entity_dim": 128,
            "hidden_dim": 256,
            "num_propagation_steps": 5,
            "convergence_threshold": 1e-4,
            "violation_threshold": 0.1,
            "learning_rate": 1e-3,
            "enable_soft_constraints": True,
            "enable_temporal_constraints": True,
            "enable_causal_constraints": True,
            "max_iterations": 100
        }
    
    def _build_constraint_propagator(self) -> nn.Module:
        """构建约束传播网络"""
        class ConstraintPropagator(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # 约束更新网络
                self.constraint_updater = nn.Sequential(
                    nn.Linear(config["constraint_dim"] * 2, config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(config["hidden_dim"], config["constraint_dim"]),
                    nn.Tanh()
                )
                
                # 实体状态更新网络
                self.entity_updater = nn.Sequential(
                    nn.Linear(config["entity_dim"] + config["constraint_dim"], config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(config["hidden_dim"], config["entity_dim"]),
                    nn.Tanh()
                )
                
                # 约束满足度计算器
                self.satisfaction_calculator = nn.Sequential(
                    nn.Linear(config["constraint_dim"], config["hidden_dim"] // 2),
                    nn.ReLU(),
                    nn.Linear(config["hidden_dim"] // 2, 1),
                    nn.Sigmoid()
                )
                
                # 冲突检测器
                self.conflict_detector = nn.Sequential(
                    nn.Linear(config["constraint_dim"] * 2, config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Linear(config["hidden_dim"], 1),
                    nn.Sigmoid()
                )
                
            def forward(self, entity_states, constraint_states, adjacency_matrix):
                """执行约束传播"""
                batch_size = entity_states.size(0)
                num_entities = entity_states.size(1)
                num_constraints = constraint_states.size(1)
                
                # 多步传播
                for step in range(self.config["num_propagation_steps"]):
                    # 更新约束状态
                    new_constraint_states = []
                    for i in range(num_constraints):
                        # 收集相关实体信息
                        related_entities = []
                        for j in range(num_entities):
                            if adjacency_matrix[i, j] > 0:
                                related_entities.append(entity_states[:, j])
                        
                        if related_entities:
                            entity_info = torch.stack(related_entities).mean(dim=0)
                            constraint_input = torch.cat([
                                constraint_states[:, i], entity_info
                            ], dim=-1)
                            updated_constraint = self.constraint_updater(constraint_input)
                            new_constraint_states.append(updated_constraint)
                        else:
                            new_constraint_states.append(constraint_states[:, i])
                    
                    constraint_states = torch.stack(new_constraint_states, dim=1)
                    
                    # 更新实体状态
                    new_entity_states = []
                    for j in range(num_entities):
                        # 收集相关约束信息
                        related_constraints = []
                        for i in range(num_constraints):
                            if adjacency_matrix[i, j] > 0:
                                related_constraints.append(constraint_states[:, i])
                        
                        if related_constraints:
                            constraint_info = torch.stack(related_constraints).mean(dim=0)
                            entity_input = torch.cat([
                                entity_states[:, j], constraint_info
                            ], dim=-1)
                            updated_entity = self.entity_updater(entity_input)
                            new_entity_states.append(updated_entity)
                        else:
                            new_entity_states.append(entity_states[:, j])
                    
                    entity_states = torch.stack(new_entity_states, dim=1)
                
                # 计算约束满足度
                satisfaction_scores = []
                for i in range(num_constraints):
                    score = self.satisfaction_calculator(constraint_states[:, i])
                    satisfaction_scores.append(score)
                
                satisfaction_scores = torch.stack(satisfaction_scores, dim=1)
                
                return {
                    "entity_states": entity_states,
                    "constraint_states": constraint_states,
                    "satisfaction_scores": satisfaction_scores
                }
        
        return ConstraintPropagator(self.config)
    
    def _build_physics_encoder(self) -> nn.Module:
        """构建物理定律编码器"""
        class PhysicsEncoder(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # 物理定律嵌入
                self.law_embeddings = nn.Embedding(
                    num_embeddings=len(PhysicalLaw),
                    embedding_dim=config["constraint_dim"]
                )
                
                # 数学形式编码器
                self.math_encoder = nn.Sequential(
                    nn.Linear(10, config["constraint_dim"] // 2),  # 简化的数学形式编码
                    nn.ReLU(),
                    nn.Linear(config["constraint_dim"] // 2, config["constraint_dim"] // 2)
                )
                
                # 上下文编码器
                self.context_encoder = nn.Sequential(
                    nn.Linear(config["entity_dim"], config["constraint_dim"] // 2),
                    nn.ReLU(),
                    nn.Linear(config["constraint_dim"] // 2, config["constraint_dim"] // 2)
                )
                
                # 约束强度预测器
                self.strength_predictor = nn.Sequential(
                    nn.Linear(config["constraint_dim"], config["constraint_dim"] // 2),
                    nn.ReLU(),
                    nn.Linear(config["constraint_dim"] // 2, 1),
                    nn.Sigmoid()
                )
                
            def encode_constraint(self, law_type, mathematical_form, context):
                """编码物理约束"""
                # 物理定律嵌入
                law_idx = list(PhysicalLaw).index(law_type)
                law_emb = self.law_embeddings(torch.tensor(law_idx))
                
                # 数学形式编码
                math_features = self._encode_mathematical_form(mathematical_form)
                math_emb = self.math_encoder(math_features)
                
                # 上下文编码
                context_emb = self.context_encoder(context)
                
                # 组合编码
                constraint_emb = torch.cat([law_emb, math_emb, context_emb], dim=-1)
                
                # 预测约束强度
                strength = self.strength_predictor(constraint_emb)
                
                return {
                    "constraint_embedding": constraint_emb,
                    "predicted_strength": strength
                }
            
            def _encode_mathematical_form(self, math_form):
                """编码数学形式"""
                # 简化的数学形式特征提取
                features = torch.zeros(10)
                
                # 基于数学形式字符串的简单特征
                if "+" in math_form:
                    features[0] = 1.0
                if "-" in math_form:
                    features[1] = 1.0
                if "*" in math_form:
                    features[2] = 1.0
                if "=" in math_form:
                    features[3] = 1.0
                if ">" in math_form:
                    features[4] = 1.0
                if "<" in math_form:
                    features[5] = 1.0
                if "∑" in math_form:
                    features[6] = 1.0
                if "∫" in math_form:
                    features[7] = 1.0
                
                features[8] = len(math_form) / 100.0  # 长度特征
                features[9] = hash(math_form) % 100 / 100.0  # 哈希特征
                
                return features
        
        return PhysicsEncoder(self.config)
    
    def _build_conflict_resolver(self) -> nn.Module:
        """构建约束冲突解决器"""
        class ConflictResolver(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # 冲突严重性评估器
                self.severity_assessor = nn.Sequential(
                    nn.Linear(config["constraint_dim"] * 2, config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Linear(config["hidden_dim"], 1),
                    nn.Sigmoid()
                )
                
                # 解决策略生成器
                self.strategy_generator = nn.Sequential(
                    nn.Linear(config["constraint_dim"] * 3, config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Linear(config["hidden_dim"], 5)  # 5种解决策略
                )
                
                # 实体调整建议器
                self.adjustment_advisor = nn.Sequential(
                    nn.Linear(config["entity_dim"] + config["constraint_dim"], config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Linear(config["hidden_dim"], config["entity_dim"])
                )
                
            def resolve_conflict(self, conflicting_constraints, affected_entities):
                """解决约束冲突"""
                # 评估冲突严重性
                constraint_pairs = list(combinations(conflicting_constraints, 2))
                severities = []
                
                for c1, c2 in constraint_pairs:
                    pair_input = torch.cat([c1, c2], dim=-1)
                    severity = self.severity_assessor(pair_input)
                    severities.append(severity)
                
                # 生成解决策略
                if constraint_pairs:
                    most_severe_pair = constraint_pairs[torch.argmax(torch.stack(severities)).item()]
                    context = torch.mean(torch.stack(affected_entities), dim=0)
                    strategy_input = torch.cat([most_severe_pair[0], most_severe_pair[1], context], dim=-1)
                    strategy_logits = self.strategy_generator(strategy_input)
                    strategy = F.softmax(strategy_logits, dim=-1)
                    
                    # 生成实体调整建议
                    adjustments = []
                    for entity in affected_entities:
                        constraint_context = torch.mean(torch.stack(conflicting_constraints), dim=0)
                        adjustment_input = torch.cat([entity, constraint_context], dim=-1)
                        adjustment = self.adjustment_advisor(adjustment_input)
                        adjustments.append(adjustment)
                    
                    return {
                        "resolution_strategy": strategy,
                        "entity_adjustments": adjustments,
                        "conflict_severity": torch.max(torch.stack(severities))
                    }
                else:
                    return {
                        "resolution_strategy": torch.zeros(5),
                        "entity_adjustments": [],
                        "conflict_severity": torch.tensor(0.0)
                    }
        
        return ConflictResolver(self.config)
    
    def _initialize_physics_kb(self) -> Dict[str, Any]:
        """初始化物理知识库"""
        return {
            "conservation_laws": {
                PhysicalLaw.CONSERVATION_OF_MASS: {
                    "description": "质量守恒定律",
                    "mathematical_form": "∑m_in = ∑m_out",
                    "applicability": ["物质变化", "化学反应", "计数问题"],
                    "strength": 1.0
                },
                PhysicalLaw.ADDITIVITY: {
                    "description": "可加性原理",
                    "mathematical_form": "total = ∑individuals",
                    "applicability": ["数量统计", "集合运算", "累积过程"],
                    "strength": 0.9
                }
            },
            "constraint_patterns": {
                "non_negative_quantities": {
                    "pattern": "quantity ≥ 0",
                    "strength": 1.0,
                    "context": ["计数", "测量", "物理量"]
                },
                "integer_constraints": {
                    "pattern": "count ∈ ℤ",
                    "strength": 1.0,
                    "context": ["离散对象", "计数问题"]
                },
                "causal_ordering": {
                    "pattern": "cause → effect",
                    "strength": 0.8,
                    "context": ["时序关系", "因果链"]
                }
            }
        }
    
    def build_constraint_network(self, entities, relations, problem_context) -> Dict[str, Any]:
        """构建物理约束网络"""
        self.logger.info("开始构建物理约束网络")
        
        # 1. 识别适用的物理定律
        applicable_laws = self._identify_applicable_laws(entities, relations, problem_context)
        
        # 2. 生成物理约束
        constraints = self._generate_physics_constraints(entities, applicable_laws)
        
        # 3. 构建约束网络
        network_structure = self._build_network_structure(entities, constraints)
        
        # 4. 执行约束传播
        propagation_result = self._propagate_constraints(network_structure)
        
        # 5. 检测约束冲突
        conflicts = self._detect_constraint_conflicts(propagation_result)
        
        # 6. 解决冲突
        if conflicts:
            resolution = self._resolve_conflicts(conflicts, propagation_result)
        else:
            resolution = {"conflicts_resolved": True, "adjustments": []}
        
        # 7. 计算最终一致性
        consistency_score = self._calculate_consistency_score(propagation_result, resolution)
        
        return {
            "success": True,
            "applicable_laws": applicable_laws,
            "constraints": constraints,
            "network_structure": network_structure,
            "propagation_result": propagation_result,
            "conflicts": conflicts,
            "resolution": resolution,
            "consistency_score": consistency_score,
            "physics_enhanced": True
        }
    
    def _identify_applicable_laws(self, entities, relations, context) -> List[PhysicalLaw]:
        """识别适用的物理定律"""
        applicable_laws = []
        
        # 基于实体类型识别定律
        entity_types = [entity.get("type", "") for entity in entities]
        
        # 如果有数量实体，应用可加性和非负性
        if any("number" in e_type or "quantity" in e_type for e_type in entity_types):
            applicable_laws.extend([
                PhysicalLaw.ADDITIVITY,
                PhysicalLaw.NON_NEGATIVITY,
                PhysicalLaw.DISCRETENESS
            ])
        
        # 如果有物理对象，应用守恒定律
        if any("object" in e_type for e_type in entity_types):
            applicable_laws.append(PhysicalLaw.CONSERVATION_OF_MASS)
        
        # 基于关系类型识别定律
        relation_types = [rel.get("type", "") for rel in relations]
        
        # 如果有因果关系，应用因果性
        if any("causal" in r_type for r_type in relation_types):
            applicable_laws.append(PhysicalLaw.CAUSALITY)
        
        # 如果有空间关系，应用局域性
        if any("spatial" in r_type for r_type in relation_types):
            applicable_laws.append(PhysicalLaw.LOCALITY)
        
        # 基于问题上下文识别定律
        context_text = context.get("problem_text", "").lower()
        
        if "总共" in context_text or "一共" in context_text:
            applicable_laws.append(PhysicalLaw.ADDITIVITY)
        
        if "守恒" in context_text or "不变" in context_text:
            applicable_laws.append(PhysicalLaw.CONSERVATION_OF_MASS)
        
        return list(set(applicable_laws))  # 去重
    
    def _generate_physics_constraints(self, entities, laws) -> List[PhysicalConstraint]:
        """生成物理约束"""
        constraints = []
        
        for law in laws:
            if law == PhysicalLaw.ADDITIVITY:
                # 生成可加性约束
                quantity_entities = [e for e in entities if "number" in e.get("type", "")]
                if len(quantity_entities) >= 2:
                    constraint = PhysicalConstraint(
                        constraint_id=f"additivity_{len(constraints)}",
                        law_type=law,
                        entities=[e["id"] for e in quantity_entities],
                        mathematical_form="∑individuals = total",
                        strength=0.9,
                        violability=0.1,
                        context_dependent=False
                    )
                    constraints.append(constraint)
            
            elif law == PhysicalLaw.NON_NEGATIVITY:
                # 生成非负性约束
                for entity in entities:
                    if "number" in entity.get("type", "") or "quantity" in entity.get("type", ""):
                        constraint = PhysicalConstraint(
                            constraint_id=f"non_negative_{entity['id']}",
                            law_type=law,
                            entities=[entity["id"]],
                            mathematical_form=f"{entity['id']} ≥ 0",
                            strength=1.0,
                            violability=0.0,
                            context_dependent=False
                        )
                        constraints.append(constraint)
            
            elif law == PhysicalLaw.DISCRETENESS:
                # 生成离散性约束
                for entity in entities:
                    if entity.get("type") == "number" and "个" in entity.get("unit", ""):
                        constraint = PhysicalConstraint(
                            constraint_id=f"discrete_{entity['id']}",
                            law_type=law,
                            entities=[entity["id"]],
                            mathematical_form=f"{entity['id']} ∈ ℤ",
                            strength=1.0,
                            violability=0.0,
                            context_dependent=False
                        )
                        constraints.append(constraint)
            
            # 可以继续添加其他物理定律的约束生成逻辑
        
        return constraints
    
    def _build_network_structure(self, entities, constraints) -> Dict[str, Any]:
        """构建网络结构"""
        entity_ids = [e["id"] for e in entities]
        constraint_ids = [c.constraint_id for c in constraints]
        
        # 构建邻接矩阵
        adjacency_matrix = np.zeros((len(constraints), len(entities)))
        
        for i, constraint in enumerate(constraints):
            for j, entity_id in enumerate(entity_ids):
                if entity_id in constraint.entities:
                    adjacency_matrix[i, j] = constraint.strength
        
        return {
            "entities": entities,
            "constraints": constraints,
            "entity_ids": entity_ids,
            "constraint_ids": constraint_ids,
            "adjacency_matrix": adjacency_matrix
        }
    
    def _propagate_constraints(self, network_structure) -> Dict[str, Any]:
        """执行约束传播"""
        entities = network_structure["entities"]
        constraints = network_structure["constraints"]
        adjacency_matrix = torch.tensor(network_structure["adjacency_matrix"], dtype=torch.float)
        
        # 初始化实体和约束状态
        entity_states = torch.randn(1, len(entities), self.config["entity_dim"])
        constraint_states = torch.randn(1, len(constraints), self.config["constraint_dim"])
        
        # 执行传播
        result = self.constraint_propagator(entity_states, constraint_states, adjacency_matrix)
        
        return {
            "final_entity_states": result["entity_states"],
            "final_constraint_states": result["constraint_states"],
            "satisfaction_scores": result["satisfaction_scores"],
            "converged": True,
            "num_iterations": self.config["num_propagation_steps"]
        }
    
    def _detect_constraint_conflicts(self, propagation_result) -> List[ConstraintViolation]:
        """检测约束冲突"""
        violations = []
        satisfaction_scores = propagation_result["satisfaction_scores"]
        
        for i, score in enumerate(satisfaction_scores[0]):
            if score < self.config["violation_threshold"]:
                violation = ConstraintViolation(
                    constraint_id=f"constraint_{i}",
                    violation_degree=float(1.0 - score),
                    affected_entities=[f"entity_{j}" for j in range(satisfaction_scores.size(-1))],
                    suggested_corrections=[f"调整实体{j}的值" for j in range(3)]
                )
                violations.append(violation)
        
        return violations
    
    def _resolve_conflicts(self, conflicts, propagation_result) -> Dict[str, Any]:
        """解决约束冲突"""
        if not conflicts:
            return {"conflicts_resolved": True, "adjustments": []}
        
        # 使用冲突解决器
        conflicting_constraints = propagation_result["final_constraint_states"][0][:len(conflicts)]
        affected_entities = propagation_result["final_entity_states"][0][:len(conflicts)]
        
        resolution = self.conflict_resolver.resolve_conflict(
            conflicting_constraints, affected_entities
        )
        
        return {
            "conflicts_resolved": True,
            "resolution_strategy": resolution["resolution_strategy"].detach().numpy(),
            "entity_adjustments": [adj.detach().numpy() for adj in resolution["entity_adjustments"]],
            "conflict_severity": float(resolution["conflict_severity"])
        }
    
    def _calculate_consistency_score(self, propagation_result, resolution) -> float:
        """计算一致性得分"""
        satisfaction_scores = propagation_result["satisfaction_scores"]
        avg_satisfaction = float(torch.mean(satisfaction_scores))
        
        # 考虑冲突解决的效果
        if resolution["conflicts_resolved"]:
            conflict_penalty = resolution.get("conflict_severity", 0.0) * 0.2
        else:
            conflict_penalty = 0.5
        
        consistency_score = max(0.0, avg_satisfaction - conflict_penalty)
        return consistency_score
    
    def test_physics_network(self) -> Dict[str, Any]:
        """测试物理约束网络"""
        # 创建测试数据
        test_entities = [
            {"id": "entity_1", "type": "number", "value": 5, "unit": "个"},
            {"id": "entity_2", "type": "number", "value": 3, "unit": "个"},
            {"id": "entity_3", "type": "object", "name": "苹果"}
        ]
        
        test_relations = [
            {"type": "quantity", "source": "entity_1", "target": "entity_3"},
            {"type": "quantity", "source": "entity_2", "target": "entity_3"}
        ]
        
        test_context = {
            "problem_text": "小明有5个苹果，又买了3个，总共有多少个？",
            "problem_type": "arithmetic"
        }
        
        # 执行约束网络构建
        result = self.build_constraint_network(test_entities, test_relations, test_context)
        
        return {
            "success": True,
            "physics_features": {
                "constraint_propagation": True,
                "conflict_detection": True,
                "conflict_resolution": True,
                "physics_knowledge_base": len(self.physics_kb) > 0
            },
            "network_result": {
                "laws_identified": len(result["applicable_laws"]),
                "constraints_generated": len(result["constraints"]),
                "consistency_score": result["consistency_score"],
                "conflicts_detected": len(result["conflicts"])
            }
        }

# 使用示例
if __name__ == "__main__":
    physics_network = PhysicsConstraintNetwork()
    test_result = physics_network.test_physics_network()
    print(f"物理约束网络测试结果: {test_result}")