"""
GNN Integrator
==============

GNN集成器，将GNN功能集成到现有的COT-DIR1模块中

主要功能：
1. 与IRD模块集成，增强隐式关系发现
2. 与MLR模块集成，优化多层级推理
3. 与CV模块集成，增强链式验证
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加src路径
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

logger = logging.getLogger(__name__)


class GNNIntegrator:
    """
    GNN集成器
    
    负责将GNN功能集成到现有的COT-DIR1模块中
    """
    
    def __init__(self):
        """初始化GNN集成器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 延迟导入避免循环依赖
        self.math_concept_gnn = None
        self.reasoning_gnn = None
        self.verification_gnn = None
        self.graph_builder = None
        
        self._initialize_components()
        
        self.logger.info("GNNIntegrator initialized")
    
    def _initialize_components(self):
        """初始化GNN组件"""
        try:
            from ..core.concept_gnn import MathConceptGNN
            from ..core.reasoning_gnn import ReasoningGNN
            from ..core.verification_gnn import VerificationGNN
            from ..graph_builders import GraphBuilder
            
            self.math_concept_gnn = MathConceptGNN()
            self.reasoning_gnn = ReasoningGNN()
            self.verification_gnn = VerificationGNN()
            self.graph_builder = GraphBuilder()
            
            self.logger.info("GNN components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GNN components: {e}")
            raise
    
    def enhance_ird_module(self, problem_text: str, 
                          existing_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        增强IRD（隐式关系发现）模块
        
        Args:
            problem_text: 问题文本
            existing_relations: 现有关系列表
            
        Returns:
            增强后的关系发现结果
        """
        try:
            self.logger.info("Enhancing IRD module with GNN")
            
            # 使用MathConceptGNN增强隐式关系发现
            enhanced_relations = self.math_concept_gnn.enhance_implicit_relations(
                problem_text, existing_relations
            )
            
            # 构建概念图
            concept_graph = self.graph_builder.build_concept_graph(problem_text)
            
            # 整合结果
            result = {
                "enhanced_relations": enhanced_relations,
                "concept_graph": concept_graph,
                "original_relations": existing_relations,
                "enhancement_method": "MathConceptGNN",
                "num_original_relations": len(existing_relations),
                "num_enhanced_relations": len(enhanced_relations),
                "improvement_ratio": len(enhanced_relations) / max(len(existing_relations), 1)
            }
            
            self.logger.info(f"IRD enhancement completed: {len(existing_relations)} -> {len(enhanced_relations)} relations")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to enhance IRD module: {e}")
            return {"error": str(e)}
    
    def enhance_mlr_module(self, reasoning_steps: List[Dict[str, Any]], 
                          problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强MLR（多层级推理）模块
        
        Args:
            reasoning_steps: 推理步骤列表
            problem_context: 问题上下文
            
        Returns:
            增强后的推理结果
        """
        try:
            self.logger.info("Enhancing MLR module with GNN")
            
            # 使用ReasoningGNN优化推理路径
            optimized_steps = self.reasoning_gnn.optimize_reasoning_path(
                reasoning_steps, problem_context
            )
            
            # 构建推理图
            reasoning_graph = self.graph_builder.build_reasoning_graph(
                reasoning_steps, problem_context
            )
            
            # 计算推理质量分数
            quality_score = self.reasoning_gnn.get_reasoning_quality_score(
                reasoning_steps, problem_context
            )
            
            # 整合结果
            result = {
                "optimized_steps": optimized_steps,
                "reasoning_graph": reasoning_graph,
                "original_steps": reasoning_steps,
                "quality_score": quality_score,
                "enhancement_method": "ReasoningGNN",
                "num_original_steps": len(reasoning_steps),
                "num_optimized_steps": len(optimized_steps),
                "optimization_improvement": quality_score
            }
            
            self.logger.info(f"MLR enhancement completed: quality score = {quality_score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to enhance MLR module: {e}")
            return {"error": str(e)}
    
    def enhance_cv_module(self, reasoning_steps: List[Dict[str, Any]], 
                         existing_verification: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强CV（链式验证）模块
        
        Args:
            reasoning_steps: 推理步骤列表
            existing_verification: 现有验证结果
            
        Returns:
            增强后的验证结果
        """
        try:
            self.logger.info("Enhancing CV module with GNN")
            
            # 使用VerificationGNN增强验证准确性
            enhanced_verification = self.verification_gnn.enhance_verification_accuracy(
                reasoning_steps, existing_verification
            )
            
            # 构建验证图
            verification_graph = self.graph_builder.build_verification_graph(
                reasoning_steps, {}
            )
            
            # 整合结果
            result = {
                "enhanced_verification": enhanced_verification,
                "verification_graph": verification_graph,
                "original_verification": existing_verification,
                "enhancement_method": "VerificationGNN",
                "original_confidence": existing_verification.get("confidence_score", 0.5),
                "enhanced_confidence": enhanced_verification.get("confidence_score", 0.5),
                "confidence_improvement": enhanced_verification.get("confidence_score", 0.5) - existing_verification.get("confidence_score", 0.5)
            }
            
            self.logger.info(f"CV enhancement completed: confidence {existing_verification.get('confidence_score', 0.5):.3f} -> {enhanced_verification.get('confidence_score', 0.5):.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to enhance CV module: {e}")
            return {"error": str(e)}
    
    def integrate_with_processors(self, problem_text: str, 
                                 processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        与processors模块集成
        
        Args:
            problem_text: 问题文本
            processing_result: 处理结果
            
        Returns:
            GNN增强后的处理结果
        """
        try:
            self.logger.info("Integrating GNN with processors module")
            
            # 提取现有关系
            existing_relations = processing_result.get("relation_results", {}).get("relations", [])
            
            # 增强关系发现
            enhanced_ird = self.enhance_ird_module(problem_text, existing_relations)
            
            # 更新处理结果
            enhanced_result = processing_result.copy()
            enhanced_result["gnn_enhanced_relations"] = enhanced_ird
            enhanced_result["enhancement_status"] = "gnn_enhanced"
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with processors: {e}")
            return processing_result
    
    def integrate_with_reasoning(self, reasoning_steps: List[Dict[str, Any]], 
                                problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        与reasoning模块集成
        
        Args:
            reasoning_steps: 推理步骤
            problem_context: 问题上下文
            
        Returns:
            GNN增强后的推理结果
        """
        try:
            self.logger.info("Integrating GNN with reasoning module")
            
            # 增强推理过程
            enhanced_mlr = self.enhance_mlr_module(reasoning_steps, problem_context)
            
            # 构建综合结果
            result = {
                "original_reasoning": reasoning_steps,
                "enhanced_reasoning": enhanced_mlr,
                "integration_method": "ReasoningGNN",
                "quality_improvement": enhanced_mlr.get("quality_score", 0.5)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with reasoning: {e}")
            return {"error": str(e)}
    
    def integrate_with_evaluation(self, reasoning_steps: List[Dict[str, Any]], 
                                 evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        与evaluation模块集成
        
        Args:
            reasoning_steps: 推理步骤
            evaluation_result: 评估结果
            
        Returns:
            GNN增强后的评估结果
        """
        try:
            self.logger.info("Integrating GNN with evaluation module")
            
            # 增强验证过程
            enhanced_cv = self.enhance_cv_module(reasoning_steps, evaluation_result)
            
            # 更新评估结果
            enhanced_evaluation = evaluation_result.copy()
            enhanced_evaluation["gnn_enhanced_verification"] = enhanced_cv
            enhanced_evaluation["enhancement_status"] = "gnn_enhanced"
            
            return enhanced_evaluation
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with evaluation: {e}")
            return evaluation_result
    
    def comprehensive_integration(self, problem_text: str, 
                                reasoning_steps: List[Dict[str, Any]], 
                                processing_result: Dict[str, Any], 
                                evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合集成所有模块
        
        Args:
            problem_text: 问题文本
            reasoning_steps: 推理步骤
            processing_result: 处理结果
            evaluation_result: 评估结果
            
        Returns:
            综合GNN增强结果
        """
        try:
            self.logger.info("Performing comprehensive GNN integration")
            
            # 1. 增强IRD模块
            existing_relations = processing_result.get("relation_results", {}).get("relations", [])
            enhanced_ird = self.enhance_ird_module(problem_text, existing_relations)
            
            # 2. 增强MLR模块
            problem_context = {"problem_text": problem_text, "processing_result": processing_result}
            enhanced_mlr = self.enhance_mlr_module(reasoning_steps, problem_context)
            
            # 3. 增强CV模块
            enhanced_cv = self.enhance_cv_module(reasoning_steps, evaluation_result)
            
            # 4. 构建所有图
            all_graphs = self.graph_builder.build_all_graphs(
                problem_text, reasoning_steps, problem_context
            )
            
            # 5. 综合结果
            comprehensive_result = {
                "enhanced_ird": enhanced_ird,
                "enhanced_mlr": enhanced_mlr,
                "enhanced_cv": enhanced_cv,
                "all_graphs": all_graphs,
                "original_data": {
                    "problem_text": problem_text,
                    "reasoning_steps": reasoning_steps,
                    "processing_result": processing_result,
                    "evaluation_result": evaluation_result
                },
                "enhancement_summary": {
                    "ird_improvement": enhanced_ird.get("improvement_ratio", 1.0),
                    "mlr_quality": enhanced_mlr.get("quality_score", 0.5),
                    "cv_confidence": enhanced_cv.get("enhanced_confidence", 0.5),
                    "overall_enhancement": "comprehensive"
                }
            }
            
            self.logger.info("Comprehensive GNN integration completed successfully")
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Failed to perform comprehensive integration: {e}")
            return {"error": str(e)}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        return {
            "integrator_initialized": True,
            "components_status": {
                "math_concept_gnn": self.math_concept_gnn is not None,
                "reasoning_gnn": self.reasoning_gnn is not None,
                "verification_gnn": self.verification_gnn is not None,
                "graph_builder": self.graph_builder is not None
            },
            "integration_capabilities": [
                "ird_enhancement",
                "mlr_optimization",
                "cv_accuracy_improvement",
                "comprehensive_integration"
            ],
            "version": "1.0.0"
        }
    
    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            "name": "GNNIntegrator",
            "version": "1.0.0",
            "description": "Integrates GNN functionality with COT-DIR1 modules",
            "integration_targets": [
                "IRD (Implicit Relation Discovery)",
                "MLR (Multi-Level Reasoning)",
                "CV (Chain Verification)"
            ],
            "components": {
                "math_concept_gnn": "MathConceptGNN",
                "reasoning_gnn": "ReasoningGNN",
                "verification_gnn": "VerificationGNN",
                "graph_builder": "GraphBuilder"
            }
        } 