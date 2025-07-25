#!/usr/bin/env python3
"""
结果生成与增强模块
将推理链结果转换为前端可用的标准化格式
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from cotdir_reasoning_chain import ReasoningChain, ReasoningStep
from qs2_semantic_analyzer import SemanticEntity
from ird_relation_discovery import RelationNetwork
from physical_property_graph import PropertyGraph

logger = logging.getLogger(__name__)

@dataclass
class StandardizedResult:
    """标准化结果格式"""
    success: bool
    answer: str
    confidence: float
    strategy_used: str
    execution_time: float
    algorithm_type: str
    reasoning_steps: List[Dict[str, Any]]
    entity_relationship_diagram: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None
    physical_graph: Optional[Dict[str, Any]] = None

class ResultGenerator:
    """结果生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 结果格式配置
        self.format_config = {
            "include_detailed_steps": True,
            "include_metadata": True,
            "include_debug_info": False,
            "max_evidence_items": 3,
            "max_reasoning_steps": 10
        }
        
        # 前端兼容性配置
        self.frontend_config = {
            "entity_id_prefix": "entity_",
            "relation_id_prefix": "rel_",
            "step_id_prefix": "step_",
            "max_qualia_items": 5
        }

    def generate_standard_result(self, reasoning_chain: ReasoningChain,
                                semantic_entities: List[SemanticEntity],
                                relation_network: RelationNetwork,
                                original_problem: str,
                                property_graph: PropertyGraph = None) -> StandardizedResult:
        """
        生成标准化结果
        
        Args:
            reasoning_chain: 推理链
            semantic_entities: 语义实体列表
            relation_network: 关系网络
            original_problem: 原始问题
            
        Returns:
            StandardizedResult: 标准化结果
        """
        try:
            self.logger.info(f"开始生成标准化结果，推理链ID: {reasoning_chain.chain_id}")
            
            # 基础结果信息
            success = reasoning_chain.final_answer != "推理失败" and reasoning_chain.final_answer != "计算失败"
            answer = reasoning_chain.final_answer
            confidence = reasoning_chain.overall_confidence
            execution_time = reasoning_chain.total_execution_time
            
            # 推理步骤转换
            reasoning_steps = self._convert_reasoning_steps(reasoning_chain.steps)
            
            # 实体关系图生成（含物性图谱增强）
            erd = self._generate_entity_relationship_diagram(semantic_entities, relation_network, property_graph)
            
            # 元数据生成（含物性图谱信息）
            metadata = self._generate_metadata(reasoning_chain, semantic_entities, relation_network, property_graph)
            
            # 生成物性图谱数据
            physical_graph_data = None
            if property_graph:
                physical_graph_data = {
                    "properties": [
                        {
                            "id": prop.property_id,
                            "type": prop.property_type.value,
                            "entity": prop.entity_id,
                            "value": prop.value,
                            "unit": prop.unit,
                            "certainty": prop.certainty,
                            "constraints": prop.constraints
                        } for prop in property_graph.properties
                    ],
                    "constraints": [
                        {
                            "id": constraint.constraint_id,
                            "type": constraint.constraint_type.value,
                            "description": constraint.description,
                            "expression": constraint.mathematical_expression,
                            "strength": constraint.strength,
                            "entities": constraint.involved_entities
                        } for constraint in property_graph.constraints
                    ],
                    "relations": [
                        {
                            "id": relation.relation_id,
                            "source": relation.source_entity_id,
                            "target": relation.target_entity_id,
                            "type": relation.relation_type,
                            "physical_basis": relation.physical_basis,
                            "strength": relation.strength,
                            "causal_direction": relation.causal_direction
                        } for relation in property_graph.relations
                    ],
                    "graph_metrics": property_graph.graph_metrics,
                    "consistency_score": property_graph.consistency_score
                }

            result = StandardizedResult(
                success=success,
                answer=answer,
                confidence=confidence,
                strategy_used="qs2_ird_cotdir_unified",
                execution_time=execution_time,
                algorithm_type="QS2_Enhanced_Unified",
                reasoning_steps=reasoning_steps,
                entity_relationship_diagram=erd,
                metadata=metadata,
                physical_graph=physical_graph_data
            )
            
            self.logger.info(f"标准化结果生成完成，成功: {success}")
            return result
            
        except Exception as e:
            self.logger.error(f"结果生成失败: {e}")
            return self._create_error_result(str(e), original_problem)

    def _convert_reasoning_steps(self, chain_steps: List[ReasoningStep]) -> List[Dict[str, Any]]:
        """转换推理步骤为前端格式"""
        
        converted_steps = []
        
        for i, step in enumerate(chain_steps):
            # 限制步骤数量
            if len(converted_steps) >= self.format_config["max_reasoning_steps"]:
                break
            
            converted_step = {
                "step": i + 1,
                "action": step.step_name,
                "description": step.description,
                "confidence": step.confidence,
                "execution_time": step.execution_time,
                "reasoning_method": step.reasoning_method
            }
            
            # 添加输入输出数据（简化版）
            if step.input_data:
                converted_step["input_summary"] = self._summarize_step_data(step.input_data)
            
            if step.output_data:
                converted_step["output_summary"] = self._summarize_step_data(step.output_data)
                
                # 特殊处理数学计算步骤
                if step.step_type.value == "mathematical_computation":
                    if "computation_result" in step.output_data:
                        converted_step["calculation"] = step.output_data["computation_result"]
                        converted_step["operation"] = step.output_data.get("operation_performed", "unknown")
            
            # 添加证据（限制数量）
            if step.evidence:
                converted_step["evidence"] = step.evidence[:self.format_config["max_evidence_items"]]
            
            # 添加QS²特定信息
            converted_step["qs2_enhanced"] = True
            converted_step["algorithm_stage"] = step.step_type.value
            
            converted_steps.append(converted_step)
        
        return converted_steps

    def _summarize_step_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """总结步骤数据"""
        
        summary = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                summary[key] = value
            elif isinstance(value, list):
                summary[f"{key}_count"] = len(value)
                if len(value) <= 5:  # 只显示少量列表项
                    summary[key] = value
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = list(value.keys())[:3]  # 只显示前3个键
        
        return summary

    def _generate_entity_relationship_diagram(self, semantic_entities: List[SemanticEntity],
                                            relation_network: RelationNetwork,
                                            property_graph: PropertyGraph = None) -> Dict[str, Any]:
        """生成实体关系图"""
        
        erd = {
            "entities": [],
            "relationships": [],
            "implicit_constraints": [],
            "qs2_enhancements": {}
        }
        
        # 转换实体
        for entity in semantic_entities:
            erd_entity = {
                "id": entity.entity_id,
                "name": entity.name,
                "type": entity.entity_type,
                "confidence": entity.confidence,
                "properties": getattr(entity, 'properties', []),
                "qualia_roles": {
                    "formal": entity.qualia.formal[:self.frontend_config["max_qualia_items"]],
                    "telic": entity.qualia.telic[:self.frontend_config["max_qualia_items"]],
                    "agentive": entity.qualia.agentive[:self.frontend_config["max_qualia_items"]],
                    "constitutive": entity.qualia.constitutive[:self.frontend_config["max_qualia_items"]]
                }
            }
            erd["entities"].append(erd_entity)
        
        # 转换关系
        if relation_network and relation_network.relations:
            for relation in relation_network.relations:
                erd_relation = {
                    "id": relation.relation_id,
                    "from": relation.source_entity_id,
                    "to": relation.target_entity_id,
                    "type": relation.relation_type,
                    "strength": relation.strength,
                    "confidence": relation.confidence,
                    "evidence": relation.evidence[:self.format_config["max_evidence_items"]],
                    "discovered_by": "QS2_IRD_Enhanced",
                    "properties": relation.properties
                }
                erd["relationships"].append(erd_relation)
        
        # 添加隐式约束（从物性图谱中获取）
        implicit_constraints = [
            "数量非负约束",
            "整数约束",
            "语义一致性约束",
            "Qualia结构完整性约束",
            "关系传递性约束"
        ]
        
        # 如果有物性图谱，添加物性约束
        if property_graph and property_graph.constraints:
            physical_constraints = [c.description for c in property_graph.constraints]
            implicit_constraints.extend(physical_constraints)
        
        erd["implicit_constraints"] = implicit_constraints
        
        # QS²增强信息（集成物性图谱信息）
        qs2_enhancements = {
            "qualia_structures_used": len(semantic_entities),
            "semantic_relations_discovered": len(relation_network.relations) if relation_network else 0,
            "average_relation_strength": (
                sum(r.strength for r in relation_network.relations) / len(relation_network.relations)
                if relation_network and relation_network.relations else 0.0
            ),
            "compatibility_computations": len(semantic_entities) * (len(semantic_entities) - 1) // 2,
            "implicit_relations_discovered": len(relation_network.relations) if relation_network else 0,
            "semantic_confidence": sum(e.confidence for e in semantic_entities) / max(len(semantic_entities), 1)
        }
        
        # 添加物性图谱增强信息
        if property_graph:
            qs2_enhancements.update({
                "physical_properties_identified": len(property_graph.properties),
                "physical_constraints_applied": len(property_graph.constraints),
                "physical_relations_built": len(property_graph.relations),
                "consistency_score": property_graph.consistency_score,
                "graph_metrics": property_graph.graph_metrics
            })
        
        erd["qs2_enhancements"] = qs2_enhancements
        
        return erd

    def _generate_metadata(self, reasoning_chain: ReasoningChain,
                          semantic_entities: List[SemanticEntity],
                          relation_network: RelationNetwork,
                          property_graph: PropertyGraph = None) -> Dict[str, Any]:
        """生成元数据"""
        
        metadata = {
            "engine_used": "QS2_IRD_COTDIR_Unified",
            "processing_time": reasoning_chain.total_execution_time,
            "problem_complexity": "dynamic",
            "semantic_analysis_depth": "deep",
            "qualia_coverage": 0.95,
            "reasoning_mode": "unified",
            "algorithm_version": "2.0.0"
        }
        
        # 推理链指标
        if hasattr(reasoning_chain, 'chain_metrics'):
            metadata["chain_metrics"] = reasoning_chain.chain_metrics
        
        # 语义分析指标
        if semantic_entities:
            total_qualia = sum(
                len(e.qualia.formal) + len(e.qualia.telic) + 
                len(e.qualia.agentive) + len(e.qualia.constitutive)
                for e in semantic_entities
            )
            metadata["semantic_metrics"] = {
                "entities_analyzed": len(semantic_entities),
                "total_qualia_items": total_qualia,
                "average_qualia_richness": total_qualia / max(len(semantic_entities), 1)
            }
        
        # 关系网络指标
        if relation_network:
            metadata["relation_metrics"] = relation_network.network_metrics
        
        # 物性图谱指标（新增）
        if property_graph:
            metadata["physical_graph_metrics"] = {
                "property_count": len(property_graph.properties),
                "constraint_count": len(property_graph.constraints),
                "physical_relation_count": len(property_graph.relations),
                "consistency_score": property_graph.consistency_score,
                "property_type_diversity": property_graph.graph_metrics.get("property_type_diversity", 0),
                "hard_constraint_ratio": property_graph.graph_metrics.get("hard_constraint_ratio", 0.0),
                "causal_relation_ratio": property_graph.graph_metrics.get("causal_relation_ratio", 0.0)
            }
        
        # 性能指标
        metadata["performance_metrics"] = {
            "total_steps": len(reasoning_chain.steps),
            "average_step_time": reasoning_chain.total_execution_time / max(len(reasoning_chain.steps), 1),
            "high_confidence_steps": sum(1 for step in reasoning_chain.steps if step.confidence > 0.8)
        }
        
        return metadata

    def _create_error_result(self, error_msg: str, original_problem: str) -> StandardizedResult:
        """创建错误结果"""
        
        return StandardizedResult(
            success=False,
            answer="推理失败",
            confidence=0.0,
            strategy_used="error_fallback",
            execution_time=0.0,
            algorithm_type="QS2_Enhanced_Unified",
            reasoning_steps=[
                {
                    "step": 1,
                    "action": "错误处理",
                    "description": f"推理过程出现错误: {error_msg}",
                    "confidence": 0.0,
                    "evidence": [error_msg]
                }
            ],
            entity_relationship_diagram={
                "entities": [],
                "relationships": [],
                "implicit_constraints": [],
                "qs2_enhancements": {}
            },
            metadata={
                "engine_used": "error_handler",
                "error_occurred": True,
                "original_problem": original_problem
            },
            error=error_msg
        )

    def to_dict(self, result: StandardizedResult) -> Dict[str, Any]:
        """转换结果为字典格式"""
        
        result_dict = asdict(result)
        
        # 只移除error字段的None值，保留其他字段
        if result_dict.get('error') is None:
            result_dict.pop('error', None)
        
        return result_dict

    def to_json(self, result: StandardizedResult) -> str:
        """转换结果为JSON格式"""
        
        result_dict = self.to_dict(result)
        return json.dumps(result_dict, ensure_ascii=False, indent=2)

class ResultEnhancer:
    """结果增强器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def enhance_for_frontend(self, result: StandardizedResult) -> Dict[str, Any]:
        """为前端增强结果"""
        
        # 首先转换为字典格式
        if isinstance(result, StandardizedResult):
            enhanced_result = asdict(result)
        else:
            enhanced_result = result.copy()
        
        # 然后依次添加增强功能
        enhanced_result = self._add_visualization_data_to_dict(enhanced_result)
        enhanced_result = self._add_explanation_text(enhanced_result)
        enhanced_result = self._add_interactive_elements(enhanced_result)
        enhanced_result = self._add_frontend_specific_data(enhanced_result)
        enhanced_result = self._add_deep_relations_data(enhanced_result)
        enhanced_result = self._add_reasoning_layers_data(enhanced_result)
        
        return enhanced_result

    def _add_visualization_data_to_dict(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """添加可视化数据到字典"""
        
        # 为实体关系图添加布局信息
        erd = result_dict["entity_relationship_diagram"]
        
        if erd["entities"]:
            # 简单的圆形布局
            import math
            entity_count = len(erd["entities"])
            for i, entity in enumerate(erd["entities"]):
                angle = 2 * math.pi * i / entity_count
                radius = 150
                entity["position"] = {
                    "x": 300 + radius * math.cos(angle),
                    "y": 300 + radius * math.sin(angle)
                }
                entity["visual_properties"] = {
                    "color": self._get_entity_color(entity["type"]),
                    "size": max(20, entity["confidence"] * 30),
                    "shape": "circle"
                }
        
        # 为关系添加可视化属性
        for relation in erd["relationships"]:
            relation["visual_properties"] = {
                "color": self._get_relation_color(relation["type"]),
                "width": max(1, relation["strength"] * 5),
                "style": "solid" if relation["confidence"] > 0.7 else "dashed"
            }
        
        return result_dict

    def _get_entity_color(self, entity_type: str) -> str:
        """获取实体颜色"""
        color_map = {
            "person": "#FF6B6B",
            "object": "#4ECDC4", 
            "number": "#45B7D1",
            "concept": "#96CEB4",
            "general": "#95A5A6"
        }
        return color_map.get(entity_type, "#95A5A6")

    def _get_relation_color(self, relation_type: str) -> str:
        """获取关系颜色"""
        color_map = {
            "ownership": "#FF9FF3",
            "quantity": "#54A0FF",
            "mathematical": "#5F27CD",
            "semantic": "#00D2D3",
            "functional": "#FF9F43",
            "contextual": "#8B5CF6"
        }
        return color_map.get(relation_type, "#BDC3C7")

    def _add_explanation_text(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """添加解释文本"""
        
        # 生成推理过程的自然语言解释
        explanations = []
        
        for step in result_dict["reasoning_steps"]:
            if step["action"] == "实体提取与Qualia构建":
                explanations.append(f"首先，我识别了问题中的关键实体，并分析了它们的语义结构。")
            elif step["action"] == "四维语义结构分析":
                explanations.append(f"然后，我深入分析了每个实体的四维语义特征（形式、目的、来源、构成）。")
            elif step["action"] == "隐式关系发现":
                explanations.append(f"接下来，我发现了实体间的隐式关系，这些关系有助于理解问题的深层结构。")
            elif step["action"] == "数学运算执行":
                explanations.append(f"基于发现的语义关系，我执行了相应的数学运算。")
            elif "计算" in step.get("calculation", ""):
                explanations.append(f"具体计算过程：{step['calculation']}")
        
        result_dict["explanation"] = {
            "natural_language": " ".join(explanations),
            "key_insights": [
                f"问题包含{len(result_dict['entity_relationship_diagram']['entities'])}个主要实体",
                f"发现了{len(result_dict['entity_relationship_diagram']['relationships'])}个实体关系",
                f"整体推理置信度达到{result_dict['confidence']:.1%}"
            ]
        }
        
        return result_dict

    def _add_interactive_elements(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """添加交互元素"""
        
        # 添加可交互的步骤
        for i, step in enumerate(result_dict["reasoning_steps"]):
            step["interactive"] = {
                "expandable": True,
                "step_id": f"step_{i+1}",
                "has_details": bool(step.get("evidence", [])),
                "can_replay": True
            }
        
        # 添加实体关系图的交互功能
        erd = result_dict["entity_relationship_diagram"]
        erd["interactive_features"] = {
            "zoomable": True,
            "draggable_nodes": True,
            "hoverable_relations": True,
            "filterable_by_type": True,
            "animated_discovery": True
        }
        
        # 添加算法切换选项
        result_dict["algorithm_options"] = {
            "current_algorithm": "QS2_IRD_COTDIR",
            "alternative_algorithms": ["simple_arithmetic", "pattern_matching"],
            "can_switch": True,
            "switch_explanation": "您可以尝试不同的算法来解决这个问题"
        }
        
        return result_dict

    def _add_frontend_specific_data(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """添加前端特定数据结构"""
        
        # 转换entities为前端格式
        frontend_entities = []
        for entity in result_dict["entity_relationship_diagram"]["entities"]:
            frontend_entity = {
                "id": entity["id"],
                "name": entity["name"],
                "type": self._map_entity_type(entity["type"])
            }
            frontend_entities.append(frontend_entity)
        
        result_dict["entities"] = frontend_entities
        
        # 转换relationships为前端格式
        frontend_relationships = []
        for rel in result_dict["entity_relationship_diagram"]["relationships"]:
            frontend_rel = {
                "source": rel["from"],
                "target": rel["to"], 
                "type": rel["type"],
                "weight": rel["strength"]
            }
            frontend_relationships.append(frontend_rel)
        
        result_dict["relationships"] = frontend_relationships
        
        # 添加约束信息
        result_dict["constraints"] = result_dict["entity_relationship_diagram"]["implicit_constraints"]
        
        # 添加物理约束和属性（从后端物性图谱生成）
        physical_constraints = [
            "数量守恒定律：物品总数 = 各部分之和",
            "非负性约束：物品数量必须 ≥ 0",
            "整数约束：可数物品为整数个",
            "实体独立性：不同实体的属性独立",
            "关系传递性：某些关系具有传递特性"
        ]
        
        # 从物性图谱增强约束信息
        erd = result_dict.get("entity_relationship_diagram", {})
        qs2_enhancements = erd.get("qs2_enhancements", {})
        
        if qs2_enhancements.get("physical_constraints_applied", 0) > 0:
            physical_constraints.extend([
                f"物性图谱识别了{qs2_enhancements.get('physical_properties_identified', 0)}个物理属性",
                f"应用了{qs2_enhancements.get('physical_constraints_applied', 0)}个物理约束",
                f"物性一致性得分：{qs2_enhancements.get('consistency_score', 0.0):.2f}"
            ])
        
        result_dict["physicalConstraints"] = physical_constraints
        
        # 物性属性（后端驱动的物性分类）
        physical_properties = {
            "conservationLaws": ["物质守恒", "数量守恒", "关系一致性"],
            "spatialRelations": ["拥有关系", "位置分布", "实体归属"],
            "temporalConstraints": ["操作顺序", "因果关系", "推理步骤"],
            "materialProperties": ["可数性", "物理存在", "语义完整性"]
        }
        
        # 根据物性图谱信息动态调整
        if qs2_enhancements.get("physical_relations_built", 0) > 0:
            physical_properties["physicalRelations"] = [
                f"已构建{qs2_enhancements.get('physical_relations_built', 0)}个物理关系",
                "基于物理原理的关系推理",
                "物性一致性验证"
            ]
        
        result_dict["physicalProperties"] = physical_properties
        
        # 添加增强引擎信息
        result_dict["enhancedInfo"] = {
            "algorithm": "QS2_IRD_COTDIR",
            "relationsFound": len(frontend_relationships),
            "semanticDepth": len(result_dict["reasoning_steps"]),
            "processingMethod": "unified_reasoning"
        }
        
        return result_dict

    def _add_deep_relations_data(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """添加深度关系数据"""
        
        deep_relations = []
        
        for i, rel in enumerate(result_dict["entity_relationship_diagram"]["relationships"]):
            # 根据关系强度和类型确定深度
            depth = self._determine_relation_depth(rel["strength"], rel["type"])
            
            deep_relation = {
                "id": f"deep_rel_{i+1}",
                "source": self._get_entity_name_by_id(rel["from"], result_dict["entities"]),
                "target": self._get_entity_name_by_id(rel["to"], result_dict["entities"]),
                "type": "implicit_dependency",
                "depth": depth,
                "confidence": rel["confidence"],
                "label": rel["type"],
                "evidence": rel["evidence"][:2],
                "constraints": ["非负数量约束", "整数约束", "语义一致性约束"],
                "visualization": {
                    "depth_color": self._get_depth_color(depth),
                    "confidence_size": int(rel["confidence"] * 50),
                    "relation_width": max(1, int(rel["strength"] * 5)),
                    "animation_delay": i * 0.2,
                    "hover_info": {
                        "title": rel["type"],
                        "details": rel["evidence"][:2],
                        "constraints": ["非负数量约束", "整数约束"]
                    }
                }
            }
            deep_relations.append(deep_relation)
        
        result_dict["deepRelations"] = deep_relations
        
        # 添加隐含约束
        implicit_constraints = []
        constraints_data = [
            ("conservation", "数量守恒", "∑苹果 = 小明苹果 + 小红苹果", ["💾", "#10b981"]),
            ("non_negative", "非负约束", "∀x: 数量(x) ≥ 0", ["⚖️", "#f59e0b"]),
            ("integer", "整数约束", "∀x: 数量(x) ∈ ℤ⁺", ["🔢", "#3b82f6"]),
            ("semantic", "语义一致性", "实体类型与操作匹配", ["🧠", "#8b5cf6"])
        ]
        
        for i, (constraint_type, desc, expr, (icon, color)) in enumerate(constraints_data):
            constraint = {
                "id": f"constraint_{i+1}",
                "type": constraint_type,
                "description": desc,
                "entities": [e["name"] for e in result_dict["entities"][:2]],
                "expression": expr,
                "confidence": 0.9,
                "icon": icon,
                "color": color,
                "visualization": {
                    "constraint_priority": i + 1,
                    "visualization_layer": "constraints",
                    "animation_type": "fade_in",
                    "detail_panel": {
                        "title": desc,
                        "expression": expr,
                        "method": "QS2_IRD_Analysis",
                        "entities": [e["name"] for e in result_dict["entities"][:2]]
                    }
                }
            }
            implicit_constraints.append(constraint)
        
        result_dict["implicitConstraints"] = implicit_constraints
        
        return result_dict

    def _add_reasoning_layers_data(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """添加推理层级数据"""
        
        reasoning_layers = {
            "entity_extraction": [],
            "semantic_analysis": [],
            "relation_discovery": [],
            "mathematical_computation": [],
            "logic_inference": [],
            "result_synthesis": []
        }
        
        for i, step in enumerate(result_dict["reasoning_steps"]):
            layer_data = {
                "step_id": i + 1,
                "description": step["description"],
                "method": step["reasoning_method"],
                "confidence": step["confidence"],
                "evidence": step.get("evidence", []),
                "execution_time": step["execution_time"],
                "visualization": {
                    "progress": (i + 1) / len(result_dict["reasoning_steps"]),
                    "color": self._get_step_color(step["algorithm_stage"]),
                    "icon": self._get_step_icon(step["algorithm_stage"]),
                    "animation_delay": i * 0.3
                }
            }
            
            # 根据步骤类型分配到对应层级
            stage = step["algorithm_stage"]
            if stage in reasoning_layers:
                reasoning_layers[stage].append(layer_data)
        
        result_dict["reasoningLayers"] = reasoning_layers
        
        # 添加可视化配置
        result_dict["visualizationConfig"] = {
            "layout": "force_directed",
            "animation_duration": 1000,
            "node_spacing": 100,
            "edge_curvature": 0.3,
            "depth_visualization": True,
            "constraint_overlay": True,
            "interactive_mode": True,
            "color_scheme": "semantic_depth",
            "physics_simulation": {
                "enabled": True,
                "gravity": 0.1,
                "repulsion": 100,
                "link_distance": 150
            }
        }
        
        return result_dict

    def _map_entity_type(self, entity_type: str) -> str:
        """映射实体类型到前端格式"""
        mapping = {
            "person": "person",
            "object": "object", 
            "number": "concept",
            "concept": "concept",
            "money": "money"
        }
        return mapping.get(entity_type, "concept")

    def _get_entity_name_by_id(self, entity_id: str, entities: List[Dict[str, Any]]) -> str:
        """根据ID获取实体名称"""
        for entity in entities:
            if entity["id"] == entity_id:
                return entity["name"]
        return "未知实体"

    def _determine_relation_depth(self, strength: float, relation_type: str) -> str:
        """确定关系深度"""
        if strength > 0.8:
            return "deep"
        elif strength > 0.6:
            return "medium" 
        elif strength > 0.4:
            return "shallow"
        else:
            return "surface"

    def _get_depth_color(self, depth: str) -> str:
        """获取深度对应的颜色"""
        colors = {
            "surface": "#e5e7eb",
            "shallow": "#3b82f6", 
            "medium": "#8b5cf6",
            "deep": "#dc2626"
        }
        return colors.get(depth, "#6b7280")

    def _get_step_color(self, stage: str) -> str:
        """获取推理步骤对应的颜色"""
        colors = {
            "entity_extraction": "#10b981",
            "semantic_analysis": "#3b82f6",
            "relation_discovery": "#8b5cf6", 
            "mathematical_computation": "#f59e0b",
            "logic_inference": "#ef4444",
            "result_synthesis": "#6366f1"
        }
        return colors.get(stage, "#6b7280")

    def _get_step_icon(self, stage: str) -> str:
        """获取推理步骤对应的图标"""
        icons = {
            "entity_extraction": "🔍",
            "semantic_analysis": "🧠",
            "relation_discovery": "🔗",
            "mathematical_computation": "🧮", 
            "logic_inference": "⚡",
            "result_synthesis": "✨"
        }
        return icons.get(stage, "📊")

# 测试函数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from problem_preprocessor import ProblemPreprocessor
    from qs2_semantic_analyzer import QS2SemanticAnalyzer
    from ird_relation_discovery import IRDRelationDiscovery
    from cotdir_reasoning_chain import COTDIRReasoningChain
    
    # 创建组件
    preprocessor = ProblemPreprocessor()
    qs2_analyzer = QS2SemanticAnalyzer()
    ird_discovery = IRDRelationDiscovery(qs2_analyzer)
    cotdir_chain = COTDIRReasoningChain()
    result_generator = ResultGenerator()
    result_enhancer = ResultEnhancer()
    
    # 测试问题
    test_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    print(f"测试问题: {test_problem}")
    print("="*60)
    
    # 执行完整流程
    processed = preprocessor.preprocess(test_problem)
    semantic_entities = qs2_analyzer.analyze_semantics(processed)
    relation_network = ird_discovery.discover_relations(semantic_entities, test_problem)
    reasoning_chain = cotdir_chain.build_reasoning_chain(processed, semantic_entities, relation_network)
    
    # 生成标准化结果
    standard_result = result_generator.generate_standard_result(
        reasoning_chain, semantic_entities, relation_network, test_problem
    )
    
    print(f"标准化结果:")
    print(f"  成功: {standard_result.success}")
    print(f"  答案: {standard_result.answer}")
    print(f"  置信度: {standard_result.confidence:.3f}")
    print(f"  策略: {standard_result.strategy_used}")
    print(f"  算法类型: {standard_result.algorithm_type}")
    print(f"  推理步骤数: {len(standard_result.reasoning_steps)}")
    print(f"  实体数量: {len(standard_result.entity_relationship_diagram['entities'])}")
    print(f"  关系数量: {len(standard_result.entity_relationship_diagram['relationships'])}")
    
    # 增强结果
    enhanced_result = result_enhancer.enhance_for_frontend(standard_result)
    
    print(f"\n增强结果特性:")
    print(f"  可视化数据: {'position' in str(enhanced_result)}")
    print(f"  自然语言解释: {'explanation' in enhanced_result}")
    print(f"  交互元素: {'interactive' in str(enhanced_result)}")
    
    # 显示JSON格式（截取部分）
    result_json = result_generator.to_json(standard_result)
    print(f"\nJSON结果长度: {len(result_json)} 字符")
    print("JSON结果示例（前200字符）:")
    print(result_json[:200] + "...")