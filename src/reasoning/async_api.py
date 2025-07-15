"""
推理模块异步版公共API

在原有功能基础上添加异步支持，提高并发处理能力。
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from ..core.exceptions import APIError, handle_module_error
from ..core.interfaces import ModuleInfo, ModuleType, PublicAPI
from .qs2_enhancement.enhanced_ird_engine import EnhancedIRDEngine, DiscoveryResult as IRDResult
from .private.mlr_processor import MultiLevelReasoningProcessor, MLRResult, ComplexityLevel
from .private.cv_validator import ChainVerificationValidator, ValidationResult


class AsyncReasoningAPI(PublicAPI):
    """推理模块异步版公共API - 支持并发处理的COT-DIR"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化异步推理API"""
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 核心组件
        self.ird_engine = None
        self.mlr_processor = None
        self.cv_validator = None
        
        # 异步执行器
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 4),
            thread_name_prefix="reasoning"
        )
        
        # 状态管理
        self._initialized = False
        self._semaphore = None
        
        # 性能统计
        self.stats = {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "concurrent_problems": 0,
            "average_processing_time": 0.0,
            "component_stats": {}
        }
        
        # 异步锁
        self._stats_lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """异步初始化推理模块"""
        try:
            self._logger.info("初始化异步推理模块...")
            
            # 创建信号量控制并发
            max_concurrent = self.config.get("max_concurrent_problems", 10)
            self._semaphore = asyncio.Semaphore(max_concurrent)
            
            # 在执行器中初始化组件（避免阻塞）
            loop = asyncio.get_event_loop()
            
            # 初始化IRD引擎
            ird_config = self.config.get("ird", {})
            self.ird_engine = await loop.run_in_executor(
                self.executor, 
                lambda: EnhancedIRDEngine(ird_config)
            )
            
            # 初始化MLR处理器
            mlr_config = self.config.get("mlr", {})
            self.mlr_processor = await loop.run_in_executor(
                self.executor,
                lambda: MultiLevelReasoningProcessor(mlr_config)
            )
            
            # 初始化CV验证器
            cv_config = self.config.get("cv", {})
            self.cv_validator = await loop.run_in_executor(
                self.executor,
                lambda: ChainVerificationValidator(cv_config)
            )
            
            self._initialized = True
            self._logger.info("异步推理模块初始化完成")
            return True
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "async_initialization")
            self._logger.error(f"异步推理模块初始化失败: {error}")
            raise error
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """同步初始化接口（保持兼容性）"""
        if config:
            self.config.update(config)
        
        # 运行异步初始化
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.initialize())
    
    async def solve_problem_async(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步解决数学问题 - COT-DIR完整流程
        
        Args:
            problem: 问题数据，包含 'problem' 或 'cleaned_text' 字段
            
        Returns:
            包含推理结果的字典
        """
        if not self._initialized:
            raise APIError("Async reasoning module not initialized", module_name="reasoning")
        
        # 使用信号量控制并发
        async with self._semaphore:
            start_time = time.time()
            
            try:
                await self._update_concurrent_stats(1)
                
                self._validate_problem_input(problem)
                problem_text = problem.get("problem") or problem.get("cleaned_text")
                
                self._logger.info(f"开始异步COT-DIR推理: {problem_text[:50]}...")
                
                # 并行执行三个阶段（如果可能）
                if self.config.get("enable_parallel_phases", False):
                    result = await self._execute_parallel_phases(problem_text, problem)
                else:
                    result = await self._execute_sequential_phases(problem_text, problem)
                
                # 更新统计信息
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                await self._update_stats_async(result, processing_time)
                
                self._logger.info(f"异步COT-DIR推理完成: 答案={result.get('final_answer')}, "
                                f"置信度={result.get('confidence', 0):.3f}")
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                await self._update_stats_failure_async(processing_time)
                
                error = handle_module_error(e, "reasoning", "async_solve_problem")
                self._logger.error(f"异步COT-DIR推理失败: {error}")
                
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
            finally:
                await self._update_concurrent_stats(-1)
    
    def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """同步解决问题接口（保持兼容性）"""
        if not self._initialized:
            raise APIError("Reasoning module not initialized", module_name="reasoning")
        
        # 运行异步版本
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.solve_problem_async(problem))
    
    async def _execute_sequential_phases(self, problem_text: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """顺序执行三个阶段"""
        
        # Phase 1: 隐式关系发现 (IRD)
        ird_result = await self._execute_ird_phase_async(problem_text, problem)
        
        # Phase 2: 多层级推理 (MLR) 
        mlr_result = await self._execute_mlr_phase_async(problem_text, ird_result.relations, problem)
        
        # Phase 3: 链式验证 (CV)
        validation_result = await self._execute_cv_phase_async(mlr_result.reasoning_steps, problem)
        
        # Phase 4: 结果整合
        return self._integrate_results(ird_result, mlr_result, validation_result, problem_text)
    
    async def _execute_parallel_phases(self, problem_text: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """并行执行部分阶段（实验性功能）"""
        
        # Phase 1: IRD必须先执行
        ird_result = await self._execute_ird_phase_async(problem_text, problem)
        
        # Phase 2 & 3: MLR和CV可以部分并行
        mlr_task = self._execute_mlr_phase_async(problem_text, ird_result.relations, problem)
        
        # 等待MLR完成
        mlr_result = await mlr_task
        
        # 现在可以执行CV
        validation_result = await self._execute_cv_phase_async(mlr_result.reasoning_steps, problem)
        
        # Phase 4: 结果整合
        return self._integrate_results(ird_result, mlr_result, validation_result, problem_text)
    
    async def _execute_ird_phase_async(self, problem_text: str, problem_context: Dict[str, Any]) -> IRDResult:
        """异步执行IRD阶段"""
        try:
            self._logger.debug("异步执行IRD阶段...")
            
            context = {
                "problem_type": problem_context.get("type"),
                "external_context": problem_context.get("context", {})
            }
            
            # 在执行器中运行IRD
            loop = asyncio.get_event_loop()
            ird_result = await loop.run_in_executor(
                self.executor,
                lambda: self.ird_engine.discover_relations(problem_text, context)
            )
            
            self._logger.debug(f"异步IRD完成: 发现{len(ird_result.relations)}个关系, "
                             f"置信度{ird_result.statistics.get('average_confidence', 0.0):.3f}")
            
            return ird_result
            
        except Exception as e:
            self._logger.error(f"异步IRD阶段失败: {str(e)}")
            # 返回空结果继续流程
            return IRDResult(
                relations=[],
                processing_time=0.0,
                entity_count=0,
                total_pairs_evaluated=0,
                high_strength_relations=0,
                statistics={"error": str(e)}
            )
    
    async def _execute_mlr_phase_async(
        self, 
        problem_text: str, 
        relations: List, 
        problem_context: Dict[str, Any]
    ) -> MLRResult:
        """异步执行MLR阶段"""
        try:
            self._logger.debug("异步执行MLR阶段...")
            
            context = {
                "problem_type": problem_context.get("type"),
                "template_info": problem_context.get("template_info"),
                "knowledge_context": problem_context.get("knowledge_context")
            }
            
            # 在执行器中运行MLR
            loop = asyncio.get_event_loop()
            mlr_result = await loop.run_in_executor(
                self.executor,
                lambda: self.mlr_processor.execute_reasoning(problem_text, relations, context)
            )
            
            self._logger.debug(f"异步MLR完成: 复杂度{mlr_result.complexity_level.value}, "
                             f"{len(mlr_result.reasoning_steps)}步, "
                             f"置信度{mlr_result.confidence_score:.3f}")
            
            return mlr_result
            
        except Exception as e:
            self._logger.error(f"异步MLR阶段失败: {str(e)}")
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
    
    async def _execute_cv_phase_async(
        self, 
        reasoning_steps: List, 
        problem_context: Dict[str, Any]
    ) -> ValidationResult:
        """异步执行CV阶段"""
        try:
            self._logger.debug("异步执行CV阶段...")
            
            context = {
                "problem_text": problem_context.get("problem") or problem_context.get("cleaned_text"),
                "problem_type": problem_context.get("type")
            }
            
            # 在执行器中运行CV
            loop = asyncio.get_event_loop()
            validation_result = await loop.run_in_executor(
                self.executor,
                lambda: self.cv_validator.verify_reasoning_chain(reasoning_steps, context)
            )
            
            self._logger.debug(f"异步CV完成: 有效={validation_result.is_valid}, "
                             f"一致性={validation_result.consistency_score:.3f}, "
                             f"错误={len(validation_result.errors)}个")
            
            return validation_result
            
        except Exception as e:
            self._logger.error(f"异步CV阶段失败: {str(e)}")
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
    
    async def batch_solve_async(self, problems: List[Dict[str, Any]], max_concurrent: Optional[int] = None) -> List[Dict[str, Any]]:
        """异步批量解决数学问题"""
        try:
            if not self._initialized:
                raise APIError("Async reasoning module not initialized", module_name="reasoning")
            
            if not isinstance(problems, list):
                raise APIError("Problems must be a list", module_name="reasoning")
            
            # 设置并发限制
            if max_concurrent is None:
                max_concurrent = self.config.get("batch_max_concurrent", 5)
            
            batch_semaphore = asyncio.Semaphore(max_concurrent)
            
            self._logger.info(f"开始异步批量处理{len(problems)}个问题，并发数: {max_concurrent}")
            
            async def solve_with_semaphore(problem, index):
                async with batch_semaphore:
                    try:
                        result = await self.solve_problem_async(problem)
                        result["problem_index"] = index
                        return result
                    except Exception as e:
                        self._logger.warning(f"问题{index}异步处理失败: {e}")
                        return {
                            "problem_index": index,
                            "error": str(e),
                            "final_answer": "处理失败",
                            "confidence": 0.0,
                            "success": False
                        }
            
            # 创建所有任务
            tasks = [solve_with_semaphore(problem, i) for i, problem in enumerate(problems)]
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "problem_index": i,
                        "error": str(result),
                        "final_answer": "处理失败",
                        "confidence": 0.0,
                        "success": False
                    })
                else:
                    processed_results.append(result)
            
            self._logger.info(f"异步批量处理完成: {len(problems)}个问题, {len(processed_results)}个结果")
            return processed_results
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "async_batch_solve")
            self._logger.error(f"异步批量处理失败: {error}")
            raise error
    
    def batch_solve(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """同步批量解决问题接口（保持兼容性）"""
        if not self._initialized:
            raise APIError("Reasoning module not initialized", module_name="reasoning")
        
        # 运行异步版本
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.batch_solve_async(problems))
    
    def _integrate_results(
        self, 
        ird_result: IRDResult, 
        mlr_result: MLRResult, 
        validation_result: ValidationResult,
        problem_text: str
    ) -> Dict[str, Any]:
        """整合三个阶段的结果（复用原有逻辑）"""
        
        # 计算整体置信度（加权平均）
        ird_weight = 0.2
        mlr_weight = 0.6
        cv_weight = 0.2
        
        overall_confidence = (
            ird_result.statistics.get("average_confidence", 0.0) * ird_weight +
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
                "async_mode": True,
                "component_versions": {
                    "ird": "2.0.0",  # Enhanced version
                    "mlr": "1.0.0", 
                    "cv": "1.0.0"
                }
            },
            "metadata": {
                "problem_text": problem_text,
                "ird_metadata": ird_result.statistics,
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
    
    async def _update_stats_async(self, result: Dict[str, Any], processing_time: float):
        """异步更新统计信息"""
        async with self._stats_lock:
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
    
    async def _update_stats_failure_async(self, processing_time: float):
        """异步更新失败统计"""
        async with self._stats_lock:
            self.stats["total_problems"] += 1
            self.stats["failed_problems"] += 1
            
            # 更新平均处理时间
            current_avg = self.stats["average_processing_time"]
            new_avg = ((current_avg * (self.stats["total_problems"] - 1) + processing_time) / 
                      self.stats["total_problems"])
            self.stats["average_processing_time"] = new_avg
    
    async def _update_concurrent_stats(self, delta: int):
        """更新并发统计"""
        async with self._stats_lock:
            self.stats["concurrent_problems"] += delta
    
    def get_module_info(self) -> ModuleInfo:
        """获取模块信息"""
        return ModuleInfo(
            name="reasoning",
            type=ModuleType.REASONING,
            version="2.1.0",  # 异步版本
            dependencies=[],
            public_api_class="AsyncReasoningAPI",
            orchestrator_class="COTDIROrchestrator"
        )
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = {
                "status": "healthy" if self._initialized else "not_initialized",
                "initialized": self._initialized,
                "async_enabled": True,
                "max_workers": self.executor._max_workers if self.executor else 0,
                "max_concurrent": self._semaphore._value if self._semaphore else 0,
                "current_concurrent": self.stats.get("concurrent_problems", 0),
                "components": {
                    "ird_engine": self.ird_engine is not None,
                    "mlr_processor": self.mlr_processor is not None,
                    "cv_validator": self.cv_validator is not None
                }
            }
            
            if self._initialized:
                # 检查各组件统计信息
                status["component_stats"] = {
                    "ird": self.ird_engine.get_global_stats() if self.ird_engine else {},
                    "mlr": self.mlr_processor.get_stats() if self.mlr_processor else {},
                    "cv": self.cv_validator.get_stats() if self.cv_validator else {}
                }
            
            return status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_statistics_async(self) -> Dict[str, Any]:
        """异步获取推理模块统计信息"""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}
            
            async with self._stats_lock:
                stats = self.stats.copy()
            
            # 计算成功率
            if stats["total_problems"] > 0:
                stats["success_rate"] = stats["successful_problems"] / stats["total_problems"]
            else:
                stats["success_rate"] = 0.0
            
            # 组件统计
            loop = asyncio.get_event_loop()
            component_stats = await loop.run_in_executor(
                self.executor,
                lambda: {
                    "ird": self.ird_engine.get_global_stats(),
                    "mlr": self.mlr_processor.get_stats(),
                    "cv": self.cv_validator.get_stats()
                }
            )
            stats["component_stats"] = component_stats
            
            return stats
            
        except Exception as e:
            self._logger.warning(f"获取异步统计信息失败: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """同步获取统计信息接口（保持兼容性）"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.get_statistics_async())
    
    async def shutdown_async(self) -> bool:
        """异步关闭推理模块"""
        try:
            self._initialized = False
            
            # 关闭执行器
            if self.executor:
                self.executor.shutdown(wait=True)
            
            self._logger.info("异步推理模块已关闭")
            return True
            
        except Exception as e:
            self._logger.error(f"异步推理模块关闭失败: {e}")
            return False
    
    def shutdown(self) -> None:
        """同步关闭接口（保持兼容性）"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self.shutdown_async())


# 创建全局实例
async_reasoning_api = AsyncReasoningAPI()

# 保持向后兼容
reasoning_api = async_reasoning_api