"""
推理策略基类
定义所有推理策略必须实现的接口
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ...core.exceptions import ReasoningError, handle_exceptions
from ...core.interfaces import (ProcessingResult, ReasoningContext,
                                ReasoningStep)
from ...monitoring.performance_monitor import monitor_performance


class StrategyType(Enum):
    """推理策略类型"""
    CHAIN_OF_THOUGHT = "cot"  # 思维链
    TREE_OF_THOUGHTS = "tot"  # 思维树
    GRAPH_OF_THOUGHTS = "got"  # 思维图
    DIRECT_IMPLICIT = "dir"  # 直接隐式推理
    TEMPLATE_BASED = "tbr"  # 模板化推理
    MULTI_LAYER = "mlr"  # 多层推理

class StrategyComplexity(Enum):
    """策略复杂度级别"""
    SIMPLE = 1    # 简单策略 - 直接计算
    MODERATE = 2  # 中等策略 - 多步推理
    COMPLEX = 3   # 复杂策略 - 树形搜索
    ADVANCED = 4  # 高级策略 - 图形推理

@dataclass
class StrategyResult:
    """策略执行结果"""
    success: bool
    answer: str
    confidence: float
    reasoning_steps: List[Dict[str, Any]]
    strategy_used: str
    execution_time: float
    metadata: Dict[str, Any]
    
    def to_processing_result(self) -> ProcessingResult:
        """转换为标准处理结果"""
        from ...core.interfaces import ProcessingStatus
        
        status = ProcessingStatus.COMPLETED if self.success else ProcessingStatus.FAILED
        
        return ProcessingResult(
            success=self.success,
            result=self.answer,
            confidence=self.confidence,
            processing_time=self.execution_time,
            status=status,
            error_message=None if self.success else self.metadata.get("error", "Unknown error"),
            metadata={
                "reasoning_steps": self.reasoning_steps,
                "strategy_used": self.strategy_used,
                **self.metadata
            }
        )

class ReasoningStrategy(ABC):
    """推理策略基类"""
    
    def __init__(self, name: str, strategy_type: StrategyType, complexity: StrategyComplexity):
        """
        初始化推理策略
        
        Args:
            name: 策略名称
            strategy_type: 策略类型
            complexity: 策略复杂度
        """
        self.name = name
        self.strategy_type = strategy_type
        self.complexity = complexity
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 策略配置
        self.config = {
            "max_steps": 15,
            "timeout_seconds": 30.0,
            "confidence_threshold": 0.7,
            "enable_validation": True
        }
        
        # 策略统计
        self.stats = {
            "total_problems": 0,
            "successful_problems": 0,
            "average_confidence": 0.0,
            "average_execution_time": 0.0
        }
    
    @abstractmethod
    def can_handle(self, problem_text: str, context: Optional[ReasoningContext] = None) -> bool:
        """
        判断策略是否能处理给定的问题
        
        Args:
            problem_text: 问题文本
            context: 推理上下文
            
        Returns:
            bool: 是否能处理
        """
        pass
    
    @abstractmethod
    def estimate_complexity(self, problem_text: str, context: Optional[ReasoningContext] = None) -> float:
        """
        估计问题的复杂度
        
        Args:
            problem_text: 问题文本
            context: 推理上下文
            
        Returns:
            float: 复杂度分数 (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def _execute_reasoning(self, problem_text: str, context: Optional[ReasoningContext] = None) -> StrategyResult:
        """
        执行具体的推理逻辑 - 子类实现
        
        Args:
            problem_text: 问题文本
            context: 推理上下文
            
        Returns:
            StrategyResult: 推理结果
        """
        pass
    
    @monitor_performance("reasoning_strategy_execute")
    @handle_exceptions(reraise_as=ReasoningError)
    def execute(self, problem_text: str, context: Optional[ReasoningContext] = None) -> StrategyResult:
        """
        执行推理策略
        
        Args:
            problem_text: 问题文本
            context: 推理上下文
            
        Returns:
            StrategyResult: 推理结果
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始执行推理策略: {self.name}")
            
            # 验证输入
            if not self._validate_input(problem_text, context):
                return self._create_error_result(
                    "输入验证失败",
                    start_time,
                    {"validation_error": True}
                )
            
            # 检查是否能处理
            if not self.can_handle(problem_text, context):
                return self._create_error_result(
                    f"策略 {self.name} 无法处理此类问题",
                    start_time,
                    {"unsupported_problem": True}
                )
            
            # 执行推理
            result = self._execute_reasoning(problem_text, context)
            
            # 更新统计信息
            self._update_stats(result)
            
            self.logger.info(f"推理策略 {self.name} 执行完成，置信度: {result.confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"推理策略 {self.name} 执行失败: {str(e)}")
            return self._create_error_result(
                f"策略执行异常: {str(e)}",
                start_time,
                {"exception": str(e)}
            )
    
    def _validate_input(self, problem_text: str, context: Optional[ReasoningContext] = None) -> bool:
        """验证输入数据"""
        if not problem_text or not isinstance(problem_text, str):
            return False
        
        if len(problem_text.strip()) == 0:
            return False
        
        # 检查问题长度
        if len(problem_text) > 10000:  # 避免过长的问题
            return False
        
        return True
    
    def _create_error_result(self, error_message: str, start_time: float, metadata: Dict[str, Any]) -> StrategyResult:
        """创建错误结果"""
        return StrategyResult(
            success=False,
            answer="无法计算",
            confidence=0.0,
            reasoning_steps=[{
                "step": 1,
                "action": "error",
                "description": error_message,
                "confidence": 0.0
            }],
            strategy_used=self.name,
            execution_time=time.time() - start_time,
            metadata={"error": error_message, **metadata}
        )
    
    def _update_stats(self, result: StrategyResult):
        """更新策略统计信息"""
        self.stats["total_problems"] += 1
        
        if result.success:
            self.stats["successful_problems"] += 1
        
        # 更新平均置信度
        total_confidence = (self.stats["average_confidence"] * (self.stats["total_problems"] - 1) + 
                           result.confidence)
        self.stats["average_confidence"] = total_confidence / self.stats["total_problems"]
        
        # 更新平均执行时间
        total_time = (self.stats["average_execution_time"] * (self.stats["total_problems"] - 1) + 
                     result.execution_time)
        self.stats["average_execution_time"] = total_time / self.stats["total_problems"]
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "complexity": self.complexity.value,
            "config": self.config,
            "statistics": self.stats
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新策略配置"""
        self.config.update(new_config)
        self.logger.info(f"策略 {self.name} 配置已更新")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_problems": 0,
            "successful_problems": 0,
            "average_confidence": 0.0,
            "average_execution_time": 0.0
        }
        self.logger.info(f"策略 {self.name} 统计信息已重置") 