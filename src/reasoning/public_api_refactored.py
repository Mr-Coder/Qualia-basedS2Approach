"""
推理模块重构版公共API

整合IRD、MLR、CV三个核心组件，提供统一的推理接口。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..core.exceptions import APIError, handle_module_error
from ..core.interfaces import ModuleInfo, ModuleType, PublicAPI
from .private.ird_engine import ImplicitRelationDiscoveryEngine, IRDResult
from .private.mlr_processor import MultiLevelReasoningProcessor, MLRResult, ComplexityLevel
from .private.cv_validator import ChainVerificationValidator, ValidationResult


class ReasoningAPI(PublicAPI):
    """推理模块重构版公共API - 整合IRD+MLR+CV"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化推理API"""
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 核心组件
        self.ird_engine = None
        self.mlr_processor = None
        self.cv_validator = None
        
        # 状态管理
        self._initialized = False
        
        # 性能统计
        self.stats = {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "average_processing_time": 0.0,
            "component_stats": {}
        }
        
    def initialize(self) -> bool:
        """初始化推理模块"""
        try:
            self._logger.info("初始化重构版推理模块...")
            
            # 初始化IRD引擎
            ird_config = self.config.get("ird", {})
            self.ird_engine = ImplicitRelationDiscoveryEngine(ird_config)
            
            # 初始化MLR处理器
            mlr_config = self.config.get("mlr", {})
            self.mlr_processor = MultiLevelReasoningProcessor(mlr_config)
            
            # 初始化CV验证器
            cv_config = self.config.get("cv", {})
            self.cv_validator = ChainVerificationValidator(cv_config)
            
            self._initialized = True
            self._logger.info("推理模块初始化完成")
            return True
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "initialization")
            self._logger.error(f"推理模块初始化失败: {error}")
            raise error
    
    def get_module_info(self) -> ModuleInfo:
        """获取模块信息"""
        return ModuleInfo(
            name="reasoning",
            type=ModuleType.REASONING,
            version="2.0.0",  # 重构版本
            dependencies=[],
            public_api_class="ReasoningAPI",
            orchestrator_class="COTDIROrchestrator"
        )
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = {
                "status": "healthy" if self._initialized else "not_initialized",
                "initialized": self._initialized,
                "components": {
                    "ird_engine": self.ird_engine is not None,
                    "mlr_processor": self.mlr_processor is not None,
                    "cv_validator": self.cv_validator is not None
                }
            }
            
            if self._initialized:
                # 检查各组件统计信息
                status["component_stats"] = {
                    "ird": self.ird_engine.get_stats() if self.ird_engine else {},
                    "mlr": self.mlr_processor.get_stats() if self.mlr_processor else {},
                    "cv": self.cv_validator.get_stats() if self.cv_validator else {}
                }
            
            return status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        解决数学问题 - COT-DIR完整流程
        
        Args:
            problem: 问题数据，包含 'problem' 或 'cleaned_text' 字段
            
        Returns:
            包含推理结果的字典：
            - final_answer: 最终答案
            - confidence: 整体置信度 (0-1)
            - reasoning_steps: 推理步骤列表
            - complexity_level: 复杂度级别
            - relations_found: 发现的隐式关系
            - validation_result: 验证结果
            - processing_info: 处理信息
        """
        start_time = time.time()
        
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            self._validate_problem_input(problem)
            problem_text = problem.get("problem") or problem.get("cleaned_text")
            
            self._logger.info(f"开始COT-DIR推理: {problem_text[:50]}...")
            
            # Phase 1: 隐式关系发现 (IRD)
            ird_result = self._execute_ird_phase(problem_text, problem)
            
            # Phase 2: 多层级推理 (MLR) 
            mlr_result = self._execute_mlr_phase(problem_text, ird_result.relations, problem)
            
            # Phase 3: 链式验证 (CV)
            validation_result = self._execute_cv_phase(mlr_result.reasoning_steps, problem)
            
            # Phase 4: 结果整合
            final_result = self._integrate_results(
                ird_result, mlr_result, validation_result, problem_text
            )
            
            # 更新统计信息
            processing_time = time.time() - start_time
            final_result["processing_time"] = processing_time
            self._update_stats(final_result, processing_time)
            
            self._logger.info(f"COT-DIR推理完成: 答案={final_result.get('final_answer')}, "
                            f"置信度={final_result.get('confidence', 0):.3f}")
            
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats_failure(processing_time)
            
            error = handle_module_error(e, "reasoning", "solve_problem")
            self._logger.error(f"COT-DIR推理失败: {error}")
            
            # 返回错误结果
            return {
                "final_answer": "推理失败",
                "confidence": 0.0,
                "reasoning_steps": [],
                "complexity_level": "unknown",
                "relations_found": [],
                "validation_result": {"is_valid": False, "errors": [str(e)]},
                "processing_info": {"error": str(e), "processing_time": processing_time},
                "success": False
            }
    
    def _execute_ird_phase(self, problem_text: str, problem_context: Dict[str, Any]) -> IRDResult:
        """执行IRD阶段"""
        try:
            self._logger.debug("执行IRD阶段...")
            
            context = {
                "problem_type": problem_context.get("type"),
                "external_context": problem_context.get("context", {})
            }
            
            ird_result = self.ird_engine.discover_relations(problem_text, context)
            
            self._logger.debug(f"IRD完成: 发现{len(ird_result.relations)}个关系, "
                             f"置信度{ird_result.confidence_score:.3f}")
            
            return ird_result
            
        except Exception as e:
            self._logger.error(f"IRD阶段失败: {str(e)}")
            # 返回空结果继续流程
            return IRDResult(
                relations=[],
                confidence_score=0.0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def _execute_mlr_phase(
        self, 
        problem_text: str, 
        relations: List, 
        problem_context: Dict[str, Any]
    ) -> MLRResult:
        """执行MLR阶段"""
        try:
            self._logger.debug("执行MLR阶段...")
            
            context = {
                "problem_type": problem_context.get("type"),
                "template_info": problem_context.get("template_info"),
                "knowledge_context": problem_context.get("knowledge_context")
            }
            
            mlr_result = self.mlr_processor.execute_reasoning(
                problem_text, relations, context
            )
            
            self._logger.debug(f"MLR完成: 复杂度{mlr_result.complexity_level.value}, "
                             f"{len(mlr_result.reasoning_steps)}步, "
                             f"置信度{mlr_result.confidence_score:.3f}")
            
            return mlr_result
            
        except Exception as e:
            self._logger.error(f"MLR阶段失败: {str(e)}")
            # 返回失败结果
            return MLRResult(
                success=False,
                complexity_level=ComplexityLevel.L0_EXPLICIT,
                reasoning_steps=[],
                final_answer=None,
                confidence_score=0.0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def _execute_cv_phase(
        self, 
        reasoning_steps: List, 
        problem_context: Dict[str, Any]
    ) -> ValidationResult:
        """执行CV阶段"""
        try:
            self._logger.debug("执行CV阶段...")
            
            context = {
                "problem_text": problem_context.get("problem") or problem_context.get("cleaned_text"),
                "problem_type": problem_context.get("type")
            }
            
            validation_result = self.cv_validator.verify_reasoning_chain(
                reasoning_steps, context
            )
            
            self._logger.debug(f"CV完成: 有效={validation_result.is_valid}, "
                             f"一致性={validation_result.consistency_score:.3f}, "
                             f"错误={len(validation_result.errors)}个")
            
            return validation_result
            
        except Exception as e:
            self._logger.error(f"CV阶段失败: {str(e)}")
            # 返回失败结果
            return ValidationResult(
                is_valid=False,
                consistency_score=0.0,
                errors=[],
                warnings=[f"验证失败: {str(e)}"],
                suggestions=[],
                corrected_steps=reasoning_steps,
                validation_time=0.0,
                metadata={"error": str(e)}
            )
    
    def _integrate_results(
        self, 
        ird_result: IRDResult, 
        mlr_result: MLRResult, 
        validation_result: ValidationResult,
        problem_text: str
    ) -> Dict[str, Any]:
        """整合三个阶段的结果"""
        
        # 计算整体置信度（加权平均）
        ird_weight = 0.2
        mlr_weight = 0.6
        cv_weight = 0.2
        
        overall_confidence = (
            ird_result.confidence_score * ird_weight +
            mlr_result.confidence_score * mlr_weight +
            validation_result.consistency_score * cv_weight
        )
        
        # 确定最终答案
        final_answer = mlr_result.final_answer
        if validation_result.corrected_steps and validation_result.corrected_steps != mlr_result.reasoning_steps:
            # 如果有自动纠错，使用纠错后的结果
            final_answer = self._extract_answer_from_corrected_steps(validation_result.corrected_steps)
        
        # 成功判断
        success = (
            mlr_result.success and 
            validation_result.is_valid and 
            overall_confidence >= 0.5
        )
        
        return {
            "final_answer": final_answer,
            "confidence": overall_confidence,
            "success": success,
            "reasoning_steps": [step.to_dict() for step in mlr_result.reasoning_steps],
            "complexity_level": mlr_result.complexity_level.value,
            "relations_found": [rel.to_dict() for rel in ird_result.relations],
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "consistency_score": validation_result.consistency_score,
                "errors": [error.to_dict() for error in validation_result.errors],
                "warnings": validation_result.warnings,
                "suggestions": validation_result.suggestions,
                "has_corrections": len(validation_result.corrected_steps) != len(mlr_result.reasoning_steps)
            },
            "processing_info": {
                "ird_time": ird_result.processing_time,
                "mlr_time": mlr_result.processing_time,
                "cv_time": validation_result.validation_time,
                "total_relations": len(ird_result.relations),
                "total_steps": len(mlr_result.reasoning_steps),
                "component_versions": {
                    "ird": "1.0.0",
                    "mlr": "1.0.0", 
                    "cv": "1.0.0"
                }
            },
            "metadata": {
                "problem_text": problem_text,
                "ird_metadata": ird_result.metadata,
                "mlr_metadata": mlr_result.metadata,
                "cv_metadata": validation_result.metadata
            }
        }
    
    def _extract_answer_from_corrected_steps(self, corrected_steps: List) -> str:
        """从纠错后的步骤中提取答案"""
        if not corrected_steps:
            return "无法确定"
        
        # 查找最后一个有输出的步骤
        for step in reversed(corrected_steps):
            if hasattr(step, 'output_value') and step.output_value is not None:
                return str(step.output_value)
        
        return "无法确定"
    
    def batch_solve(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量解决数学问题"""
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            if not isinstance(problems, list):
                raise APIError("Problems must be a list", module_name="reasoning")
            
            self._logger.info(f"开始批量处理{len(problems)}个问题")
            
            results = []
            for i, problem in enumerate(problems):
                try:
                    result = self.solve_problem(problem)
                    result["problem_index"] = i
                    results.append(result)
                except Exception as e:
                    self._logger.warning(f"问题{i}处理失败: {e}")
                    results.append({
                        "problem_index": i,
                        "error": str(e),
                        "final_answer": "处理失败",
                        "confidence": 0.0,
                        "success": False
                    })
            
            self._logger.info(f"批量处理完成: {len(problems)}个问题, {len(results)}个结果")
            return results
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "batch_solve")
            self._logger.error(f"批量处理失败: {error}")
            raise error
    
    def get_reasoning_steps(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取详细推理步骤"""
        try:
            result = self.solve_problem(problem)
            return result.get("reasoning_steps", [])
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "get_reasoning_steps")
            self._logger.error(f"获取推理步骤失败: {error}")
            raise error
    
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证推理结果"""
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            # 提取推理步骤进行验证
            reasoning_steps = result.get("reasoning_steps", [])
            if not reasoning_steps:
                return {"is_valid": False, "error": "No reasoning steps found"}
            
            # 将字典格式转换为ReasoningStep对象（简化版本）
            # 这里需要根据实际的步骤格式进行转换
            
            validation_result = self.cv_validator.verify_reasoning_chain(reasoning_steps)
            
            return {
                "is_valid": validation_result.is_valid,
                "consistency_score": validation_result.consistency_score,
                "errors": [error.to_dict() for error in validation_result.errors],
                "warnings": validation_result.warnings,
                "suggestions": validation_result.suggestions
            }
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "validate_result")
            self._logger.error(f"结果验证失败: {error}")
            raise error
    
    def explain_reasoning(self, result: Dict[str, Any]) -> str:
        """生成推理过程的文本解释"""
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            explanation_parts = []
            
            # 问题复杂度说明
            complexity = result.get("complexity_level", "unknown")
            explanation_parts.append(f"问题复杂度级别: {complexity}")
            
            # 隐式关系说明
            relations = result.get("relations_found", [])
            if relations:
                explanation_parts.append(f"发现了{len(relations)}个隐式关系:")
                for i, rel in enumerate(relations[:3], 1):  # 只显示前3个
                    explanation_parts.append(f"  {i}. {rel.get('description', '未知关系')}")
            
            # 推理步骤说明
            steps = result.get("reasoning_steps", [])
            if steps:
                explanation_parts.append(f"推理过程包含{len(steps)}个步骤:")
                for i, step in enumerate(steps[:5], 1):  # 只显示前5步
                    desc = step.get("description", "未知操作")
                    explanation_parts.append(f"  步骤{i}: {desc}")
            
            # 验证结果说明
            validation = result.get("validation_result", {})
            if validation.get("is_valid"):
                explanation_parts.append("推理过程通过了一致性验证")
            else:
                errors = validation.get("errors", [])
                explanation_parts.append(f"验证发现{len(errors)}个问题，但推理仍然进行")
            
            # 最终结果
            final_answer = result.get("final_answer", "未知")
            confidence = result.get("confidence", 0)
            explanation_parts.append(f"最终答案: {final_answer} (置信度: {confidence:.2f})")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "explain_reasoning")
            self._logger.error(f"推理解释失败: {error}")
            raise error
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取推理模块统计信息"""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}
            
            stats = self.stats.copy()
            
            # 计算成功率
            if stats["total_problems"] > 0:
                stats["success_rate"] = stats["successful_problems"] / stats["total_problems"]
            else:
                stats["success_rate"] = 0.0
            
            # 组件统计
            stats["component_stats"] = {
                "ird": self.ird_engine.get_stats(),
                "mlr": self.mlr_processor.get_stats(),
                "cv": self.cv_validator.get_stats()
            }
            
            return stats
            
        except Exception as e:
            self._logger.warning(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def reset_statistics(self):
        """重置统计信息"""
        try:
            self.stats = {
                "total_problems": 0,
                "successful_problems": 0,
                "failed_problems": 0,
                "average_processing_time": 0.0,
                "component_stats": {}
            }
            
            if self._initialized:
                self.ird_engine.reset_stats()
                self.mlr_processor.reset_stats()
                self.cv_validator.reset_stats()
            
            self._logger.info("推理模块统计信息已重置")
            
        except Exception as e:
            self._logger.error(f"重置统计信息失败: {e}")
    
    def _validate_problem_input(self, problem: Dict[str, Any]) -> None:
        """验证问题输入"""
        if not isinstance(problem, dict):
            raise APIError("Problem must be a dictionary", module_name="reasoning")
        
        # 检查必要字段
        problem_text = problem.get("problem") or problem.get("cleaned_text")
        if not problem_text:
            raise APIError("Problem must contain 'problem' or 'cleaned_text' field", module_name="reasoning")
        
        if not isinstance(problem_text, str) or not problem_text.strip():
            raise APIError("Problem text must be a non-empty string", module_name="reasoning")
    
    def _update_stats(self, result: Dict[str, Any], processing_time: float):
        """更新统计信息"""
        self.stats["total_problems"] += 1
        
        if result.get("success", False):
            self.stats["successful_problems"] += 1
        else:
            self.stats["failed_problems"] += 1
        
        # 更新平均处理时间
        current_avg = self.stats["average_processing_time"]
        new_avg = ((current_avg * (self.stats["total_problems"] - 1) + processing_time) / 
                  self.stats["total_problems"])
        self.stats["average_processing_time"] = new_avg
    
    def _update_stats_failure(self, processing_time: float):
        """更新失败统计"""
        self.stats["total_problems"] += 1
        self.stats["failed_problems"] += 1
        
        # 更新平均处理时间
        current_avg = self.stats["average_processing_time"]
        new_avg = ((current_avg * (self.stats["total_problems"] - 1) + processing_time) / 
                  self.stats["total_problems"])
        self.stats["average_processing_time"] = new_avg
    
    def shutdown(self) -> bool:
        """关闭推理模块"""
        try:
            self._initialized = False
            self._logger.info("推理模块已关闭")
            return True
            
        except Exception as e:
            self._logger.error(f"推理模块关闭失败: {e}")
            return False