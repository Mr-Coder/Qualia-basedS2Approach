"""
Graph Builder
=============

主要图构建器类，协调不同类型的图构建操作

用于从数学问题文本构建概念图、推理图和验证图
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .concept_graph_builder import ConceptGraphBuilder
from .reasoning_graph_builder import ReasoningGraphBuilder
from .verification_graph_builder import VerificationGraphBuilder

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    主要图构建器
    
    协调不同类型的图构建操作：
    - 概念图构建
    - 推理图构建
    - 验证图构建
    """
    
    def __init__(self):
        """初始化图构建器"""
        self.concept_builder = ConceptGraphBuilder()
        self.reasoning_builder = ReasoningGraphBuilder()
        self.verification_builder = VerificationGraphBuilder()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("GraphBuilder initialized")
    
    def build_all_graphs(self, problem_text: str, 
                        reasoning_steps: Optional[List[Dict[str, Any]]] = None,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        构建所有类型的图
        
        Args:
            problem_text: 问题文本
            reasoning_steps: 推理步骤（可选）
            context: 上下文信息（可选）
            
        Returns:
            包含所有图的字典
        """
        try:
            context = context or {}
            
            # 1. 构建概念图
            concept_graph = self.concept_builder.build_concept_graph(problem_text, context)
            
            # 2. 构建推理图（如果有推理步骤）
            reasoning_graph = None
            if reasoning_steps:
                reasoning_graph = self.reasoning_builder.build_reasoning_graph(reasoning_steps, context)
            
            # 3. 构建验证图（如果有推理步骤）
            verification_graph = None
            if reasoning_steps:
                verification_graph = self.verification_builder.build_verification_graph(reasoning_steps, context)
            
            # 4. 构建综合图信息
            integrated_graph = self._integrate_graphs(concept_graph, reasoning_graph, verification_graph)
            
            return {
                "concept_graph": concept_graph,
                "reasoning_graph": reasoning_graph,
                "verification_graph": verification_graph,
                "integrated_graph": integrated_graph,
                "problem_text": problem_text,
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Failed to build graphs: {e}")
            return {"error": str(e)}
    
    def build_concept_graph(self, problem_text: str, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建概念图"""
        return self.concept_builder.build_concept_graph(problem_text, context or {})
    
    def build_reasoning_graph(self, reasoning_steps: List[Dict[str, Any]], 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建推理图"""
        return self.reasoning_builder.build_reasoning_graph(reasoning_steps, context or {})
    
    def build_verification_graph(self, reasoning_steps: List[Dict[str, Any]], 
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建验证图"""
        return self.verification_builder.build_verification_graph(reasoning_steps, context or {})
    
    def _integrate_graphs(self, concept_graph: Optional[Dict[str, Any]],
                         reasoning_graph: Optional[Dict[str, Any]],
                         verification_graph: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """整合不同类型的图"""
        integrated = {
            "nodes": [],
            "edges": [],
            "graph_types": [],
            "statistics": {}
        }
        
        # 整合概念图
        if concept_graph and "concepts" in concept_graph:
            for concept in concept_graph["concepts"]:
                integrated["nodes"].append({
                    "id": len(integrated["nodes"]),
                    "type": "concept",
                    "content": concept,
                    "graph_source": "concept_graph"
                })
            integrated["graph_types"].append("concept")
        
        # 整合推理图
        if reasoning_graph and "steps" in reasoning_graph:
            for step in reasoning_graph["steps"]:
                integrated["nodes"].append({
                    "id": len(integrated["nodes"]),
                    "type": "reasoning_step",
                    "content": step,
                    "graph_source": "reasoning_graph"
                })
            integrated["graph_types"].append("reasoning")
        
        # 整合验证图
        if verification_graph and "verification_steps" in verification_graph:
            for step in verification_graph["verification_steps"]:
                integrated["nodes"].append({
                    "id": len(integrated["nodes"]),
                    "type": "verification_step",
                    "content": step,
                    "graph_source": "verification_graph"
                })
            integrated["graph_types"].append("verification")
        
        # 计算统计信息
        integrated["statistics"] = {
            "total_nodes": len(integrated["nodes"]),
            "concept_nodes": sum(1 for n in integrated["nodes"] if n["type"] == "concept"),
            "reasoning_nodes": sum(1 for n in integrated["nodes"] if n["type"] == "reasoning_step"),
            "verification_nodes": sum(1 for n in integrated["nodes"] if n["type"] == "verification_step"),
            "graph_types": integrated["graph_types"]
        }
        
        return integrated
    
    def get_graph_statistics(self, graphs: Dict[str, Any]) -> Dict[str, Any]:
        """获取图统计信息"""
        stats = {
            "concept_graph": {},
            "reasoning_graph": {},
            "verification_graph": {},
            "integrated_graph": {}
        }
        
        # 概念图统计
        if graphs.get("concept_graph"):
            concept_graph = graphs["concept_graph"]
            stats["concept_graph"] = {
                "num_concepts": len(concept_graph.get("concepts", [])),
                "num_relations": len(concept_graph.get("relations", [])),
                "has_graph_info": "graph_info" in concept_graph
            }
        
        # 推理图统计
        if graphs.get("reasoning_graph"):
            reasoning_graph = graphs["reasoning_graph"]
            stats["reasoning_graph"] = {
                "num_steps": len(reasoning_graph.get("steps", [])),
                "num_dependencies": len(reasoning_graph.get("dependencies", [])),
                "has_graph_info": "graph_info" in reasoning_graph
            }
        
        # 验证图统计
        if graphs.get("verification_graph"):
            verification_graph = graphs["verification_graph"]
            stats["verification_graph"] = {
                "num_verification_steps": len(verification_graph.get("verification_steps", [])),
                "num_dependencies": len(verification_graph.get("dependencies", [])),
                "has_graph_info": "graph_info" in verification_graph
            }
        
        # 综合图统计
        if graphs.get("integrated_graph"):
            integrated_graph = graphs["integrated_graph"]
            stats["integrated_graph"] = integrated_graph.get("statistics", {})
        
        return stats
    
    def validate_graphs(self, graphs: Dict[str, Any]) -> Dict[str, Any]:
        """验证图结构"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "graph_validations": {}
        }
        
        # 验证概念图
        if graphs.get("concept_graph"):
            concept_validation = self._validate_concept_graph(graphs["concept_graph"])
            validation_result["graph_validations"]["concept_graph"] = concept_validation
            if not concept_validation["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(concept_validation["errors"])
        
        # 验证推理图
        if graphs.get("reasoning_graph"):
            reasoning_validation = self._validate_reasoning_graph(graphs["reasoning_graph"])
            validation_result["graph_validations"]["reasoning_graph"] = reasoning_validation
            if not reasoning_validation["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(reasoning_validation["errors"])
        
        # 验证验证图
        if graphs.get("verification_graph"):
            verification_validation = self._validate_verification_graph(graphs["verification_graph"])
            validation_result["graph_validations"]["verification_graph"] = verification_validation
            if not verification_validation["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(verification_validation["errors"])
        
        return validation_result
    
    def _validate_concept_graph(self, concept_graph: Dict[str, Any]) -> Dict[str, Any]:
        """验证概念图"""
        validation = {"valid": True, "errors": [], "warnings": []}
        
        # 检查必要字段
        if "concepts" not in concept_graph:
            validation["valid"] = False
            validation["errors"].append("Missing 'concepts' field")
        
        if "relations" not in concept_graph:
            validation["warnings"].append("Missing 'relations' field")
        
        # 检查概念格式
        concepts = concept_graph.get("concepts", [])
        for i, concept in enumerate(concepts):
            if not isinstance(concept, dict):
                validation["errors"].append(f"Concept {i} is not a dictionary")
                validation["valid"] = False
            elif "text" not in concept:
                validation["errors"].append(f"Concept {i} missing 'text' field")
                validation["valid"] = False
        
        return validation
    
    def _validate_reasoning_graph(self, reasoning_graph: Dict[str, Any]) -> Dict[str, Any]:
        """验证推理图"""
        validation = {"valid": True, "errors": [], "warnings": []}
        
        # 检查必要字段
        if "steps" not in reasoning_graph:
            validation["valid"] = False
            validation["errors"].append("Missing 'steps' field")
        
        if "dependencies" not in reasoning_graph:
            validation["warnings"].append("Missing 'dependencies' field")
        
        # 检查步骤格式
        steps = reasoning_graph.get("steps", [])
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                validation["errors"].append(f"Step {i} is not a dictionary")
                validation["valid"] = False
            elif "id" not in step:
                validation["errors"].append(f"Step {i} missing 'id' field")
                validation["valid"] = False
        
        return validation
    
    def _validate_verification_graph(self, verification_graph: Dict[str, Any]) -> Dict[str, Any]:
        """验证验证图"""
        validation = {"valid": True, "errors": [], "warnings": []}
        
        # 检查必要字段
        if "verification_steps" not in verification_graph:
            validation["valid"] = False
            validation["errors"].append("Missing 'verification_steps' field")
        
        if "dependencies" not in verification_graph:
            validation["warnings"].append("Missing 'dependencies' field")
        
        # 检查验证步骤格式
        verification_steps = verification_graph.get("verification_steps", [])
        for i, step in enumerate(verification_steps):
            if not isinstance(step, dict):
                validation["errors"].append(f"Verification step {i} is not a dictionary")
                validation["valid"] = False
            elif "id" not in step:
                validation["errors"].append(f"Verification step {i} missing 'id' field")
                validation["valid"] = False
        
        return validation
    
    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            "name": "GraphBuilder",
            "version": "1.0.0",
            "components": {
                "concept_builder": self.concept_builder.__class__.__name__,
                "reasoning_builder": self.reasoning_builder.__class__.__name__,
                "verification_builder": self.verification_builder.__class__.__name__
            },
            "capabilities": [
                "concept_graph_building",
                "reasoning_graph_building",
                "verification_graph_building",
                "graph_integration",
                "graph_validation"
            ]
        } 