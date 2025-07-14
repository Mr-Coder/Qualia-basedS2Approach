"""
COT-DIR推理协调器

协调IRD、MLR、CV三个核心组件的工作流程。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .private.ird_engine import ImplicitRelationDiscoveryEngine, IRDResult
from .private.mlr_processor import MultiLevelReasoningProcessor, MLRResult
from .private.cv_validator import ChainVerificationValidator, ValidationResult


class COTDIROrchestrator:
    """COT-DIR推理协调器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化协调器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config or {}
        
        # 核心组件
        self.ird_engine = None
        self.mlr_processor = None  
        self.cv_validator = None
        
        # 工作流配置
        self.enable_ird = self.config.get("enable_ird", True)
        self.enable_mlr = self.config.get("enable_mlr", True)
        self.enable_cv = self.config.get("enable_cv", True)
        
        # 组件配置
        self.ird_config = self.config.get("ird", {})
        self.mlr_config = self.config.get("mlr", {})
        self.cv_config = self.config.get("cv", {})
        
        # 协调器统计
        self.orchestrator_stats = {
            "total_orchestrations": 0,
            "successful_orchestrations": 0,
            "failed_orchestrations": 0,
            "average_orchestration_time": 0.0,
            "component_usage": {
                "ird": 0,
                "mlr": 0,
                "cv": 0
            }
        }
        
        # 初始化标志
        self._initialized = False
        
    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            self.logger.info("初始化COT-DIR协调器...")
            
            # 初始化IRD引擎
            if self.enable_ird:
                self.ird_engine = ImplicitRelationDiscoveryEngine(self.ird_config)
                self.logger.info("IRD引擎初始化完成")
            
            # 初始化MLR处理器
            if self.enable_mlr:
                self.mlr_processor = MultiLevelReasoningProcessor(self.mlr_config)
                self.logger.info("MLR处理器初始化完成")
            
            # 初始化CV验证器
            if self.enable_cv:
                self.cv_validator = ChainVerificationValidator(self.cv_config)
                self.logger.info("CV验证器初始化完成")
            
            self._initialized = True
            self.logger.info("COT-DIR协调器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"COT-DIR协调器初始化失败: {str(e)}")
            return False
    
    def orchestrate_full_pipeline(
        self, 
        problem: Dict[str, Any], 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行完整的COT-DIR流水线
        
        Args:
            problem: 问题数据
            options: 执行选项
            
        Returns:
            完整的推理结果
        """
        start_time = time.time()
        options = options or {}
        
        try:
            if not self._initialized:
                raise RuntimeError("协调器未初始化")
            
            problem_text = problem.get("problem") or problem.get("cleaned_text")
            if not problem_text:
                raise ValueError("问题文本为空")
            
            self.logger.info(f"开始COT-DIR完整流水线: {problem_text[:50]}...")
            
            # 初始化结果容器
            pipeline_result = {
                "problem_text": problem_text,
                "success": False,
                "ird_result": None,
                "mlr_result": None,
                "cv_result": None,
                "final_answer": None,
                "confidence": 0.0,
                "processing_stages": []
            }
            
            # Stage 1: 隐式关系发现 (IRD)
            ird_result = self._execute_ird_stage(problem_text, problem, options)
            pipeline_result["ird_result"] = ird_result
            pipeline_result["processing_stages"].append("ird")
            
            # Stage 2: 多层级推理 (MLR)
            mlr_result = self._execute_mlr_stage(
                problem_text, ird_result.relations if ird_result else [], problem, options
            )
            pipeline_result["mlr_result"] = mlr_result
            pipeline_result["processing_stages"].append("mlr")
            
            # Stage 3: 链式验证 (CV)
            cv_result = self._execute_cv_stage(
                mlr_result.reasoning_steps if mlr_result else [], problem, options
            )
            pipeline_result["cv_result"] = cv_result
            pipeline_result["processing_stages"].append("cv")
            
            # Stage 4: 结果整合和最终判断
            final_result = self._integrate_pipeline_results(
                ird_result, mlr_result, cv_result, problem_text
            )
            pipeline_result.update(final_result)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            pipeline_result["processing_time"] = processing_time
            self._update_orchestrator_stats(pipeline_result, processing_time)
            
            self.logger.info(f"COT-DIR流水线完成: 成功={pipeline_result['success']}, "
                           f"置信度={pipeline_result['confidence']:.3f}")
            
            return pipeline_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"COT-DIR流水线执行失败: {str(e)}")
            
            # 更新失败统计
            self._update_orchestrator_stats_failure(processing_time)
            
            return {
                "problem_text": problem.get("problem", ""),
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "final_answer": "执行失败",
                "confidence": 0.0
            }
    
    def _execute_ird_stage(
        self, 
        problem_text: str, 
        problem_context: Dict[str, Any], 
        options: Dict[str, Any]
    ) -> Optional[IRDResult]:
        """执行IRD阶段"""
        if not self.enable_ird or not self.ird_engine:
            self.logger.info("IRD阶段被跳过")
            return None
        
        try:
            self.logger.debug("执行IRD阶段...")
            self.orchestrator_stats["component_usage"]["ird"] += 1
            
            # 准备IRD上下文
            ird_context = {
                "problem_type": problem_context.get("type"),
                "template_info": problem_context.get("template_info"),
                "external_context": problem_context.get("context", {}),
                "options": options.get("ird", {})
            }
            
            ird_result = self.ird_engine.discover_relations(problem_text, ird_context)
            
            self.logger.debug(f"IRD阶段完成: 发现{len(ird_result.relations)}个关系")
            return ird_result
            
        except Exception as e:
            self.logger.error(f"IRD阶段执行失败: {str(e)}")
            return None
    
    def _execute_mlr_stage(
        self, 
        problem_text: str, 
        relations: List, 
        problem_context: Dict[str, Any], 
        options: Dict[str, Any]
    ) -> Optional[MLRResult]:
        """执行MLR阶段"""
        if not self.enable_mlr or not self.mlr_processor:
            self.logger.info("MLR阶段被跳过")
            return None
        
        try:
            self.logger.debug("执行MLR阶段...")
            self.orchestrator_stats["component_usage"]["mlr"] += 1
            
            # 准备MLR上下文
            mlr_context = {
                "problem_type": problem_context.get("type"),
                "template_info": problem_context.get("template_info"),
                "knowledge_context": problem_context.get("knowledge_context"),
                "options": options.get("mlr", {})
            }
            
            mlr_result = self.mlr_processor.execute_reasoning(
                problem_text, relations, mlr_context
            )
            
            self.logger.debug(f"MLR阶段完成: 复杂度{mlr_result.complexity_level.value}, "
                            f"{len(mlr_result.reasoning_steps)}步")
            return mlr_result
            
        except Exception as e:
            self.logger.error(f"MLR阶段执行失败: {str(e)}")
            return None
    
    def _execute_cv_stage(
        self, 
        reasoning_steps: List, 
        problem_context: Dict[str, Any], 
        options: Dict[str, Any]
    ) -> Optional[ValidationResult]:
        """执行CV阶段"""
        if not self.enable_cv or not self.cv_validator or not reasoning_steps:
            self.logger.info("CV阶段被跳过")
            return None
        
        try:
            self.logger.debug("执行CV阶段...")
            self.orchestrator_stats["component_usage"]["cv"] += 1
            
            # 准备CV上下文
            cv_context = {
                "problem_text": problem_context.get("problem") or problem_context.get("cleaned_text"),
                "problem_type": problem_context.get("type"),
                "options": options.get("cv", {})
            }
            
            cv_result = self.cv_validator.verify_reasoning_chain(
                reasoning_steps, cv_context
            )
            
            self.logger.debug(f"CV阶段完成: 有效={cv_result.is_valid}, "
                            f"一致性={cv_result.consistency_score:.3f}")
            return cv_result
            
        except Exception as e:
            self.logger.error(f"CV阶段执行失败: {str(e)}")
            return None
    
    def _integrate_pipeline_results(
        self,
        ird_result: Optional[IRDResult],
        mlr_result: Optional[MLRResult], 
        cv_result: Optional[ValidationResult],
        problem_text: str
    ) -> Dict[str, Any]:
        """整合流水线结果"""
        
        # 基础结果结构
        integrated_result = {
            "success": False,
            "final_answer": "无法确定",
            "confidence": 0.0,
            "reasoning_steps": [],
            "relations_found": [],
            "validation_info": {},
            "processing_summary": {}
        }
        
        # 整合IRD结果
        if ird_result:
            integrated_result["relations_found"] = [rel.to_dict() for rel in ird_result.relations]
            integrated_result["processing_summary"]["ird"] = {
                "relations_count": len(ird_result.relations),
                "confidence": ird_result.confidence_score,
                "processing_time": ird_result.processing_time
            }
        
        # 整合MLR结果
        if mlr_result:
            integrated_result["reasoning_steps"] = [step.to_dict() for step in mlr_result.reasoning_steps]
            integrated_result["final_answer"] = mlr_result.final_answer or "无法确定"
            integrated_result["processing_summary"]["mlr"] = {
                "complexity_level": mlr_result.complexity_level.value,
                "steps_count": len(mlr_result.reasoning_steps),
                "confidence": mlr_result.confidence_score,
                "processing_time": mlr_result.processing_time,
                "success": mlr_result.success
            }
        
        # 整合CV结果
        if cv_result:
            integrated_result["validation_info"] = {
                "is_valid": cv_result.is_valid,
                "consistency_score": cv_result.consistency_score,
                "errors_count": len(cv_result.errors),
                "warnings_count": len(cv_result.warnings),
                "has_corrections": len(cv_result.corrected_steps) > 0
            }
            integrated_result["processing_summary"]["cv"] = {
                "validation_time": cv_result.validation_time,
                "errors_found": len(cv_result.errors),
                "auto_corrections": len(cv_result.corrected_steps)
            }
            
            # 如果有自动纠错，更新最终答案
            if cv_result.corrected_steps and cv_result.corrected_steps != (mlr_result.reasoning_steps if mlr_result else []):
                corrected_answer = self._extract_answer_from_steps(cv_result.corrected_steps)
                if corrected_answer:
                    integrated_result["final_answer"] = corrected_answer
                    integrated_result["processing_summary"]["final_answer_source"] = "cv_corrected"
        
        # 计算整体置信度
        integrated_result["confidence"] = self._calculate_integrated_confidence(
            ird_result, mlr_result, cv_result
        )
        
        # 判断整体成功
        integrated_result["success"] = self._determine_overall_success(
            ird_result, mlr_result, cv_result, integrated_result["confidence"]
        )
        
        return integrated_result
    
    def _calculate_integrated_confidence(
        self,
        ird_result: Optional[IRDResult],
        mlr_result: Optional[MLRResult],
        cv_result: Optional[ValidationResult]
    ) -> float:
        """计算整合置信度"""
        
        confidences = []
        weights = []
        
        # IRD置信度
        if ird_result:
            confidences.append(ird_result.confidence_score)
            weights.append(0.2)  # IRD权重20%
        
        # MLR置信度  
        if mlr_result:
            confidences.append(mlr_result.confidence_score)
            weights.append(0.6)  # MLR权重60%
        
        # CV置信度
        if cv_result:
            confidences.append(cv_result.consistency_score)
            weights.append(0.2)  # CV权重20%
        
        if not confidences:
            return 0.0
        
        # 加权平均
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_overall_success(
        self,
        ird_result: Optional[IRDResult],
        mlr_result: Optional[MLRResult], 
        cv_result: Optional[ValidationResult],
        overall_confidence: float
    ) -> bool:
        """判断整体成功"""
        
        # 基础成功条件
        if overall_confidence < 0.5:
            return False
        
        # MLR必须成功
        if mlr_result and not mlr_result.success:
            return False
        
        # 如果启用CV，验证必须通过或至少不能有严重错误
        if cv_result and not cv_result.is_valid:
            # 检查是否有严重错误
            severe_errors = cv_result.get_severe_errors(0.8)
            if severe_errors:
                return False
        
        return True
    
    def _extract_answer_from_steps(self, steps: List) -> Optional[str]:
        """从步骤中提取答案"""
        if not steps:
            return None
        
        # 查找最后一个有输出的步骤
        for step in reversed(steps):
            if hasattr(step, 'output_value') and step.output_value is not None:
                return str(step.output_value)
        
        return None
    
    def orchestrate_partial_pipeline(
        self, 
        stage: str, 
        input_data: Dict[str, Any], 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行部分流水线（单个阶段）
        
        Args:
            stage: 阶段名称 ("ird", "mlr", "cv")
            input_data: 输入数据
            options: 执行选项
            
        Returns:
            阶段执行结果
        """
        try:
            options = options or {}
            
            if stage == "ird":
                return self._orchestrate_ird_only(input_data, options)
            elif stage == "mlr":
                return self._orchestrate_mlr_only(input_data, options)
            elif stage == "cv":
                return self._orchestrate_cv_only(input_data, options)
            else:
                raise ValueError(f"未知阶段: {stage}")
                
        except Exception as e:
            self.logger.error(f"部分流水线执行失败 ({stage}): {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stage": stage
            }
    
    def _orchestrate_ird_only(self, input_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """仅执行IRD"""
        problem_text = input_data.get("problem_text", "")
        context = input_data.get("context", {})
        
        ird_result = self._execute_ird_stage(problem_text, context, options)
        
        return {
            "success": ird_result is not None,
            "stage": "ird",
            "result": ird_result.to_dict() if ird_result else None
        }
    
    def _orchestrate_mlr_only(self, input_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """仅执行MLR"""
        problem_text = input_data.get("problem_text", "")
        relations = input_data.get("relations", [])
        context = input_data.get("context", {})
        
        mlr_result = self._execute_mlr_stage(problem_text, relations, context, options)
        
        return {
            "success": mlr_result is not None and mlr_result.success,
            "stage": "mlr", 
            "result": mlr_result.__dict__ if mlr_result else None
        }
    
    def _orchestrate_cv_only(self, input_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """仅执行CV"""
        reasoning_steps = input_data.get("reasoning_steps", [])
        context = input_data.get("context", {})
        
        cv_result = self._execute_cv_stage(reasoning_steps, context, options)
        
        return {
            "success": cv_result is not None,
            "stage": "cv",
            "result": cv_result.__dict__ if cv_result else None
        }
    
    def get_component_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "initialized": self._initialized,
            "components": {
                "ird_engine": {
                    "enabled": self.enable_ird,
                    "available": self.ird_engine is not None,
                    "stats": self.ird_engine.get_stats() if self.ird_engine else {}
                },
                "mlr_processor": {
                    "enabled": self.enable_mlr,
                    "available": self.mlr_processor is not None,
                    "stats": self.mlr_processor.get_stats() if self.mlr_processor else {}
                },
                "cv_validator": {
                    "enabled": self.enable_cv,
                    "available": self.cv_validator is not None,
                    "stats": self.cv_validator.get_stats() if self.cv_validator else {}
                }
            },
            "orchestrator_stats": self.orchestrator_stats
        }
    
    def _update_orchestrator_stats(self, result: Dict[str, Any], processing_time: float):
        """更新协调器统计"""
        self.orchestrator_stats["total_orchestrations"] += 1
        
        if result.get("success", False):
            self.orchestrator_stats["successful_orchestrations"] += 1
        else:
            self.orchestrator_stats["failed_orchestrations"] += 1
        
        # 更新平均处理时间
        current_avg = self.orchestrator_stats["average_orchestration_time"]
        new_avg = ((current_avg * (self.orchestrator_stats["total_orchestrations"] - 1) + processing_time) / 
                  self.orchestrator_stats["total_orchestrations"])
        self.orchestrator_stats["average_orchestration_time"] = new_avg
    
    def _update_orchestrator_stats_failure(self, processing_time: float):
        """更新失败统计"""
        self.orchestrator_stats["total_orchestrations"] += 1
        self.orchestrator_stats["failed_orchestrations"] += 1
        
        # 更新平均处理时间
        current_avg = self.orchestrator_stats["average_orchestration_time"]
        new_avg = ((current_avg * (self.orchestrator_stats["total_orchestrations"] - 1) + processing_time) / 
                  self.orchestrator_stats["total_orchestrations"])
        self.orchestrator_stats["average_orchestration_time"] = new_avg
    
    def reset_stats(self):
        """重置所有统计信息"""
        self.orchestrator_stats = {
            "total_orchestrations": 0,
            "successful_orchestrations": 0,
            "failed_orchestrations": 0,
            "average_orchestration_time": 0.0,
            "component_usage": {
                "ird": 0,
                "mlr": 0,
                "cv": 0
            }
        }
        
        if self.ird_engine:
            self.ird_engine.reset_stats()
        if self.mlr_processor:
            self.mlr_processor.reset_stats()
        if self.cv_validator:
            self.cv_validator.reset_stats()
        
        self.logger.info("协调器统计信息已重置")
    
    def shutdown(self):
        """关闭协调器"""
        try:
            self._initialized = False
            self.logger.info("COT-DIR协调器已关闭")
        except Exception as e:
            self.logger.error(f"协调器关闭失败: {str(e)}")