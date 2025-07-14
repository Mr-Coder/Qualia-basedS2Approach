"""
新版推理引擎
整合策略模式、多步推理执行器和置信度计算器的现代化推理引擎
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..config.config_manager import get_config
from ..core.exceptions import ReasoningError, handle_exceptions
from ..core.interfaces import (IReasoningEngine, ProcessingResult,
                               ReasoningContext)
from ..monitoring.performance_monitor import get_monitor, monitor_performance
from .confidence_calculator import BasicConfidenceCalculator, ConfidenceResult
from .multi_step_reasoner import ExecutionResult, StepExecutor
from .strategy_manager import (ChainOfThoughtStrategy, GraphOfThoughtsStrategy,
                               ReasoningStrategy, StrategyManager,
                               StrategyResult, TreeOfThoughtsStrategy)


class ModernReasoningEngine(IReasoningEngine):
    """现代化推理引擎"""
    
    def __init__(self):
        """初始化推理引擎"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 核心组件
        self.strategy_manager = StrategyManager()
        self.step_executor = StepExecutor()
        self.confidence_calculator = BasicConfidenceCalculator()
        
        # 配置管理
        try:
            self.config = get_config()
        except Exception:
            self.config = None
            self.logger.warning("配置管理器不可用，使用默认配置")
        
        # 性能监控
        self.monitor = get_monitor()
        
        # 推理统计
        self.reasoning_stats = {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "strategy_usage": {}
        }
        
        # 当前推理状态
        self.current_reasoning_steps = []
        self.current_strategy = None
        
        # 初始化策略
        self._initialize_strategies()
        
        self.logger.info("现代化推理引擎初始化完成")
    
    def _initialize_strategies(self):
        """初始化推理策略"""
        try:
            # 注册核心策略
            strategies = [
                ChainOfThoughtStrategy(),
                TreeOfThoughtsStrategy(), 
                GraphOfThoughtsStrategy()
            ]
            
            for strategy in strategies:
                success = self.strategy_manager.register_strategy(strategy)
                if success:
                    self.reasoning_stats["strategy_usage"][strategy.name] = 0
                    self.logger.info(f"策略 {strategy.name} 注册成功")
                else:
                    self.logger.warning(f"策略 {strategy.name} 注册失败")
            
            self.logger.info(f"已注册 {len(strategies)} 个推理策略")
            
        except Exception as e:
            self.logger.error(f"策略初始化失败: {str(e)}")
            raise ReasoningError(f"推理引擎策略初始化失败: {str(e)}")
    
    @monitor_performance("reasoning_engine_reason")
    @handle_exceptions(reraise_as=ReasoningError)
    def reason(self, problem: str, context: Optional[ReasoningContext] = None) -> ProcessingResult:
        """
        执行推理
        
        Args:
            problem: 问题文本
            context: 推理上下文
            
        Returns:
            ProcessingResult: 推理结果
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始推理问题: {problem[:100]}...")
            
            # 重置当前状态
            self.current_reasoning_steps = []
            self.current_strategy = None
            
            # 第一阶段：问题预处理和策略选择
            preprocessing_result = self._preprocess_problem(problem, context)
            
            # 第二阶段：执行推理
            reasoning_result = self._execute_reasoning(problem, context, preprocessing_result)
            
            # 第三阶段：后处理和验证
            final_result = self._postprocess_result(reasoning_result, problem, context)
            
            # 更新统计信息
            self._update_stats(final_result, time.time() - start_time)
            
            # 记录监控指标
            self.monitor.increment_counter("reasoning_problems_total")
            if final_result.success:
                self.monitor.increment_counter("reasoning_problems_success")
            
            self.logger.info(f"推理完成，成功: {final_result.success}, 置信度: {final_result.confidence:.2f}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"推理执行失败: {str(e)}")
            self._update_stats_failure(time.time() - start_time)
            
            # 返回错误结果
            from ..core.interfaces import ProcessingStatus
            return ProcessingResult(
                success=False,
                result="推理失败",
                confidence=0.0,
                processing_time=time.time() - start_time,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                metadata={
                    "error_type": type(e).__name__,
                    "reasoning_steps": self.current_reasoning_steps
                }
            )
    
    def _preprocess_problem(self, problem: str, context: Optional[ReasoningContext]) -> Dict[str, Any]:
        """预处理问题"""
        preprocessing_result = {
            "original_problem": problem,
            "cleaned_problem": problem.strip(),
            "problem_type": "unknown",
            "complexity": 0.5,
            "suggested_strategy": None
        }
        
        try:
            # 问题清理
            cleaned_problem = self._clean_problem_text(problem)
            preprocessing_result["cleaned_problem"] = cleaned_problem
            
            # 问题类型识别
            problem_type = self._identify_problem_type(cleaned_problem)
            preprocessing_result["problem_type"] = problem_type
            
            # 复杂度估计
            complexity = self._estimate_problem_complexity(cleaned_problem)
            preprocessing_result["complexity"] = complexity
            
            # 策略选择
            suggested_strategy = self.strategy_manager.select_strategy(cleaned_problem, context)
            preprocessing_result["suggested_strategy"] = suggested_strategy
            
            self.logger.debug(f"预处理完成: 类型={problem_type}, 复杂度={complexity:.2f}, 建议策略={suggested_strategy}")
            
        except Exception as e:
            self.logger.warning(f"预处理过程出错: {str(e)}")
        
        return preprocessing_result
    
    def _execute_reasoning(self, problem: str, context: Optional[ReasoningContext], 
                          preprocessing_result: Dict[str, Any]) -> StrategyResult:
        """执行推理过程"""
        try:
            # 获取建议的策略
            strategy_name = preprocessing_result.get("suggested_strategy")
            cleaned_problem = preprocessing_result.get("cleaned_problem", problem)
            
            # 执行推理
            reasoning_result = self.strategy_manager.execute_reasoning(
                cleaned_problem, 
                context, 
                strategy_name=strategy_name,
                enable_fallback=True
            )
            
            # 记录当前策略和步骤
            self.current_strategy = reasoning_result.strategy_used
            self.current_reasoning_steps = reasoning_result.reasoning_steps
            
            # 更新策略使用统计
            if reasoning_result.strategy_used in self.reasoning_stats["strategy_usage"]:
                self.reasoning_stats["strategy_usage"][reasoning_result.strategy_used] += 1
            
            return reasoning_result
            
        except Exception as e:
            self.logger.error(f"推理执行阶段失败: {str(e)}")
            raise ReasoningError(f"推理执行失败: {str(e)}", reasoning_step=0)
    
    def _postprocess_result(self, reasoning_result: StrategyResult, problem: str,
                           context: Optional[ReasoningContext]) -> ProcessingResult:
        """后处理推理结果"""
        try:
            # 计算置信度
            confidence_result = self.confidence_calculator.calculate_confidence(
                reasoning_result.reasoning_steps,
                reasoning_result.answer,
                {"problem": problem, "strategy": reasoning_result.strategy_used}
            )
            
            # 构建最终结果
            from ..core.interfaces import ProcessingStatus
            
            status = ProcessingStatus.COMPLETED if reasoning_result.success else ProcessingStatus.FAILED
            
            final_result = ProcessingResult(
                success=reasoning_result.success,
                result=reasoning_result.answer,
                confidence=confidence_result.overall_confidence,
                processing_time=reasoning_result.execution_time,
                status=status,
                error_message=reasoning_result.metadata.get("error") if not reasoning_result.success else None,
                metadata={
                    "reasoning_steps": reasoning_result.reasoning_steps,
                    "strategy_used": reasoning_result.strategy_used,
                    "confidence_details": confidence_result.to_dict(),
                    "preprocessing": {"problem_type": self._identify_problem_type(problem)},
                    "step_count": len(reasoning_result.reasoning_steps)
                }
            )
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"后处理阶段失败: {str(e)}")
            
            # 构建错误结果
            from ..core.interfaces import ProcessingStatus
            return ProcessingResult(
                success=False,
                result="后处理失败",
                confidence=0.0,
                processing_time=reasoning_result.execution_time,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                metadata={"postprocessing_error": True}
            )
    
    def get_reasoning_steps(self) -> List[Dict[str, Any]]:
        """获取当前的推理步骤"""
        return self.current_reasoning_steps.copy()
    
    def set_reasoning_strategy(self, strategy: str) -> None:
        """设置推理策略"""
        available_strategies = self.strategy_manager.get_available_strategies()
        
        if strategy not in available_strategies:
            raise ReasoningError(
                f"策略 {strategy} 不可用，可用策略: {available_strategies}",
                context={"requested_strategy": strategy, "available_strategies": available_strategies}
            )
        
        self.logger.info(f"推理策略设置为: {strategy}")
    
    def add_strategy(self, strategy: ReasoningStrategy) -> bool:
        """添加新的推理策略"""
        success = self.strategy_manager.register_strategy(strategy)
        
        if success:
            self.reasoning_stats["strategy_usage"][strategy.name] = 0
            self.logger.info(f"成功添加策略: {strategy.name}")
        else:
            self.logger.warning(f"添加策略失败: {strategy.name}")
        
        return success
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """移除推理策略"""
        success = self.strategy_manager.unregister_strategy(strategy_name)
        
        if success and strategy_name in self.reasoning_stats["strategy_usage"]:
            del self.reasoning_stats["strategy_usage"][strategy_name]
            self.logger.info(f"成功移除策略: {strategy_name}")
        
        return success
    
    def get_available_strategies(self) -> List[str]:
        """获取可用的推理策略"""
        return self.strategy_manager.get_available_strategies()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            "reasoning_stats": self.reasoning_stats.copy(),
            "strategy_manager_stats": self.strategy_manager.get_performance_report(),
            "step_executor_stats": self.step_executor.get_execution_stats()
        }
        
        # 计算成功率
        if self.reasoning_stats["total_problems"] > 0:
            report["success_rate"] = (self.reasoning_stats["successful_problems"] / 
                                    self.reasoning_stats["total_problems"])
        else:
            report["success_rate"] = 0.0
        
        return report
    
    def reset_stats(self):
        """重置统计信息"""
        self.reasoning_stats = {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "strategy_usage": {name: 0 for name in self.strategy_manager.get_available_strategies()}
        }
        
        self.strategy_manager.reset_performance_stats()
        self.step_executor.reset_stats()
        
        self.logger.info("统计信息已重置")
    
    # 辅助方法
    def _clean_problem_text(self, problem: str) -> str:
        """清理问题文本"""
        if not problem:
            return ""
        
        # 去除多余的空白字符
        cleaned = " ".join(problem.split())
        
        # 标准化标点符号
        replacements = {
            "？": "?",
            "。": ".",
            "，": ",",
            "；": ";",
            "：": ":",
            "（": "(",
            "）": ")"
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned.strip()
    
    def _identify_problem_type(self, problem: str) -> str:
        """识别问题类型"""
        problem_lower = problem.lower()
        
        # 几何问题
        if any(keyword in problem_lower for keyword in ["面积", "周长", "体积", "长度", "宽度", "高度"]):
            return "geometry"
        
        # 运动问题
        elif any(keyword in problem_lower for keyword in ["速度", "时间", "距离", "路程"]):
            return "motion"
        
        # 经济问题
        elif any(keyword in problem_lower for keyword in ["价格", "成本", "利润", "折扣", "元", "钱"]):
            return "economics"
        
        # 比例问题
        elif any(keyword in problem_lower for keyword in ["比例", "百分比", "%", "比率"]):
            return "proportion"
        
        # 算术问题
        elif any(keyword in problem_lower for keyword in ["加", "减", "乘", "除", "+", "-", "*", "/"]):
            return "arithmetic"
        
        else:
            return "general"
    
    def _estimate_problem_complexity(self, problem: str) -> float:
        """估计问题复杂度"""
        complexity = 0.0
        
        # 基于文本长度
        length_factor = min(len(problem) / 200, 1.0) * 0.2
        complexity += length_factor
        
        # 基于数字数量
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', problem)
        number_factor = min(len(numbers) / 5, 1.0) * 0.3
        complexity += number_factor
        
        # 基于操作词汇
        operation_words = ["加", "减", "乘", "除", "计算", "求", "比较", "分析"]
        operation_count = sum(1 for word in operation_words if word in problem)
        operation_factor = min(operation_count / 3, 1.0) * 0.3
        complexity += operation_factor
        
        # 基于复杂度关键词
        complex_keywords = ["如果", "假设", "当", "因为", "所以", "条件", "约束"]
        complex_count = sum(1 for keyword in complex_keywords if keyword in problem)
        complex_factor = min(complex_count / 2, 1.0) * 0.2
        complexity += complex_factor
        
        return min(1.0, complexity)
    
    def _update_stats(self, result: ProcessingResult, processing_time: float):
        """更新统计信息"""
        self.reasoning_stats["total_problems"] += 1
        
        if result.success:
            self.reasoning_stats["successful_problems"] += 1
        else:
            self.reasoning_stats["failed_problems"] += 1
        
        # 更新平均置信度
        total_confidence = (self.reasoning_stats["average_confidence"] * 
                           (self.reasoning_stats["total_problems"] - 1) + result.confidence)
        self.reasoning_stats["average_confidence"] = total_confidence / self.reasoning_stats["total_problems"]
        
        # 更新平均处理时间
        total_time = (self.reasoning_stats["average_processing_time"] * 
                     (self.reasoning_stats["total_problems"] - 1) + processing_time)
        self.reasoning_stats["average_processing_time"] = total_time / self.reasoning_stats["total_problems"]
    
    def _update_stats_failure(self, processing_time: float):
        """更新失败统计"""
        self.reasoning_stats["total_problems"] += 1
        self.reasoning_stats["failed_problems"] += 1
        
        # 更新平均处理时间
        total_time = (self.reasoning_stats["average_processing_time"] * 
                     (self.reasoning_stats["total_problems"] - 1) + processing_time)
        self.reasoning_stats["average_processing_time"] = total_time / self.reasoning_stats["total_problems"] 