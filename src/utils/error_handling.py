#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
错误处理和异常管理模块
~~~~~~~~~~~~~~~~~~~

提供统一的错误处理、日志记录和异常恢复功能

Author: [Hao Meng]
Date: [2025-05-29]
"""

import functools
import logging
import sys
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别"""
    INITIALIZATION = "initialization"
    INPUT_VALIDATION = "input_validation"
    NLP_PROCESSING = "nlp_processing"
    CLASSIFICATION = "classification"
    RELATION_EXTRACTION = "relation_extraction"
    EQUATION_BUILDING = "equation_building"
    SOLVING = "solving"
    VISUALIZATION = "visualization"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class MathSolverBaseException(Exception):
    """数学求解器基础异常类"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = __import__('time').time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp,
            "original_exception": str(self.original_exception) if self.original_exception else None,
            "traceback": traceback.format_exc() if self.original_exception else None
        }


class InitializationError(MathSolverBaseException):
    """初始化错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message, 
            ErrorCategory.INITIALIZATION, 
            ErrorSeverity.CRITICAL,
            context,
            original_exception
        )


class InputValidationError(MathSolverBaseException):
    """输入验证错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message,
            ErrorCategory.INPUT_VALIDATION,
            ErrorSeverity.MEDIUM,
            context,
            original_exception
        )


class NLPProcessingError(MathSolverBaseException):
    """NLP处理错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message,
            ErrorCategory.NLP_PROCESSING,
            ErrorSeverity.HIGH,
            context,
            original_exception
        )


class ClassificationError(MathSolverBaseException):
    """分类错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message,
            ErrorCategory.CLASSIFICATION,
            ErrorSeverity.MEDIUM,
            context,
            original_exception
        )


class RelationExtractionError(MathSolverBaseException):
    """关系提取错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message,
            ErrorCategory.RELATION_EXTRACTION,
            ErrorSeverity.HIGH,
            context,
            original_exception
        )


class EquationBuildingError(MathSolverBaseException):
    """方程构建错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message,
            ErrorCategory.EQUATION_BUILDING,
            ErrorSeverity.HIGH,
            context,
            original_exception
        )


class SolvingError(MathSolverBaseException):
    """求解错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message,
            ErrorCategory.SOLVING,
            ErrorSeverity.HIGH,
            context,
            original_exception
        )


class VisualizationError(MathSolverBaseException):
    """可视化错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message,
            ErrorCategory.VISUALIZATION,
            ErrorSeverity.LOW,
            context,
            original_exception
        )


class ConfigurationError(MathSolverBaseException):
    """配置错误"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(
            message,
            ErrorCategory.CONFIGURATION,
            ErrorSeverity.CRITICAL,
            context,
            original_exception
        )


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None,
                 enable_recovery: bool = True,
                 max_retry_attempts: int = 3):
        self.logger = logger or logging.getLogger(__name__)
        self.enable_recovery = enable_recovery
        self.max_retry_attempts = max_retry_attempts
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
    
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """注册错误恢复策略"""
        self.recovery_strategies[category] = strategy
        self.logger.debug(f"注册恢复策略: {category.value}")
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """处理错误"""
        # 包装为自定义异常（如果需要）
        if not isinstance(error, MathSolverBaseException):
            error = self._wrap_exception(error, context)
        
        # 记录错误
        self._log_error(error)
        
        # 添加到错误历史
        self.error_history.append(error.to_dict())
        
        # 尝试恢复（如果启用）
        if self.enable_recovery and error.category in self.recovery_strategies:
            try:
                return self.recovery_strategies[error.category](error, context)
            except Exception as recovery_error:
                self.logger.error(f"错误恢复失败: {recovery_error}")
        
        # 如果无法恢复，重新抛出异常
        raise error
    
    def _wrap_exception(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> MathSolverBaseException:
        """包装原始异常为自定义异常"""
        error_type = type(error).__name__
        category = self._categorize_error(error)
        severity = self._assess_severity(error)
        
        return MathSolverBaseException(
            f"{error_type}: {str(error)}",
            category,
            severity,
            context,
            error
        )
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """根据异常类型分类错误"""
        error_type = type(error).__name__
        
        if error_type in ['ImportError', 'ModuleNotFoundError']:
            return ErrorCategory.INITIALIZATION
        elif error_type in ['ValueError', 'TypeError']:
            return ErrorCategory.INPUT_VALIDATION
        elif error_type in ['FileNotFoundError', 'PermissionError']:
            return ErrorCategory.SYSTEM
        elif error_type in ['KeyError', 'AttributeError']:
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """评估错误严重程度"""
        error_type = type(error).__name__
        
        if error_type in ['SystemExit', 'KeyboardInterrupt', 'MemoryError']:
            return ErrorSeverity.CRITICAL
        elif error_type in ['ImportError', 'ModuleNotFoundError']:
            return ErrorSeverity.HIGH
        elif error_type in ['ValueError', 'TypeError', 'KeyError']:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _log_error(self, error: MathSolverBaseException):
        """记录错误日志"""
        severity_map = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        
        log_level = severity_map.get(error.severity, logging.ERROR)
        
        log_message = (
            f"[{error.category.value.upper()}] {error.message}"
        )
        
        if error.context:
            log_message += f" | Context: {error.context}"
        
        self.logger.log(log_level, log_message)
        
        # 记录原始异常的堆栈跟踪
        if error.original_exception:
            self.logger.debug(
                f"原始异常堆栈: {traceback.format_exception(type(error.original_exception), error.original_exception, error.original_exception.__traceback__)}"
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_history:
            return {"total_errors": 0, "categories": {}, "severities": {}}
        
        category_counts = {}
        severity_counts = {}
        
        for error_info in self.error_history:
            category = error_info["category"]
            severity = error_info["severity"]
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "categories": category_counts,
            "severities": severity_counts,
            "latest_errors": self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        }
    
    def clear_error_history(self):
        """清空错误历史"""
        self.error_history.clear()


def error_handler_decorator(error_handler: ErrorHandler, 
                          context_provider: Optional[Callable] = None,
                          reraise: bool = True):
    """错误处理装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = context_provider(*args, **kwargs) if context_provider else {}
                context.update({
                    "function": func.__name__,
                    "args": str(args)[:200],  # 限制长度
                    "kwargs": str(kwargs)[:200]
                })
                
                try:
                    return error_handler.handle_error(e, context)
                except Exception:
                    if reraise:
                        raise
                    return None
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return: Any = None, 
                logger: Optional[logging.Logger] = None, **kwargs) -> Any:
    """安全执行函数"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"安全执行失败 {func.__name__}: {e}")
        return default_return


def retry_on_error(max_attempts: int = 3, delay: float = 1.0, 
                  exponential_backoff: bool = True,
                  exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # 不是最后一次尝试
                        import time
                        time.sleep(current_delay)
                        
                        if exponential_backoff:
                            current_delay *= 2
                    
                    logging.getLogger(__name__).warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}"
                    )
            
            # 所有尝试都失败，抛出最后的异常
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """断路器模式实现"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过断路器调用函数"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("断路器进入半开状态")
            else:
                raise Exception("断路器处于开启状态，拒绝调用")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """判断是否应该尝试重置断路器"""
        if self.last_failure_time is None:
            return True
        
        import time
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self):
        """成功时的处理"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info("断路器重置为关闭状态")
    
    def _on_failure(self):
        """失败时的处理"""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"断路器开启，失败次数: {self.failure_count}")


def circuit_breaker_decorator(failure_threshold: int = 5, timeout: float = 60.0):
    """断路器装饰器"""
    breaker = CircuitBreaker(failure_threshold, timeout)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


# 默认错误恢复策略
def default_nlp_recovery(error: MathSolverBaseException, context: Dict[str, Any]) -> Any:
    """NLP处理错误的默认恢复策略"""
    logging.getLogger(__name__).info("使用简化NLP处理进行恢复")
    
    # 返回简化的处理结果
    return {
        "segmentation": context.get("text", "").split(),
        "pos_tags": ["unknown"] * len(context.get("text", "").split()),
        "recovered": True
    }


def default_visualization_recovery(error: MathSolverBaseException, context: Dict[str, Any]) -> Any:
    """可视化错误的默认恢复策略"""
    logging.getLogger(__name__).info("跳过可视化生成")
    return None


def setup_default_error_handler() -> ErrorHandler:
    """设置默认错误处理器"""
    handler = ErrorHandler()
    
    # 注册默认恢复策略
    handler.register_recovery_strategy(ErrorCategory.NLP_PROCESSING, default_nlp_recovery)
    handler.register_recovery_strategy(ErrorCategory.VISUALIZATION, default_visualization_recovery)
    
    return handler


if __name__ == "__main__":
    # 演示错误处理功能
    
    # 创建错误处理器
    error_handler = setup_default_error_handler()
    
    # 演示自定义异常
    try:
        raise NLPProcessingError(
            "NLP模型加载失败",
            context={"model_path": "/path/to/model"},
            original_exception=FileNotFoundError("文件不存在")
        )
    except MathSolverBaseException as e:
        print("自定义异常演示:")
        print(f"消息: {e.message}")
        print(f"类别: {e.category.value}")
        print(f"严重程度: {e.severity.value}")
        print(f"上下文: {e.context}")
    
    # 演示错误处理器
    @error_handler_decorator(error_handler)
    def problematic_function(x: int) -> int:
        if x < 0:
            raise ValueError("输入值不能为负数")
        return x * 2
    
    try:
        result = problematic_function(-1)
    except Exception:
        print("\n错误处理演示完成")
    
    # 演示重试装饰器
    @retry_on_error(max_attempts=3, delay=0.1)
    def unreliable_function():
        import random
        if random.random() < 0.7:
            raise ConnectionError("连接失败")
        return "成功"
    
    try:
        result = unreliable_function()
        print(f"\n重试成功: {result}")
    except Exception as e:
        print(f"\n重试失败: {e}")
    
    # 显示错误摘要
    print("\n错误摘要:")
    print(error_handler.get_error_summary())

# 为了向后兼容，创建别名
MathProblemSolverError = MathSolverBaseException

# 导出主要类和函数
__all__ = [
    'ErrorSeverity',
    'ErrorCategory', 
    'MathSolverBaseException',
    'MathProblemSolverError',  # 别名
    'InitializationError',
    'InputValidationError',
    'NLPProcessingError',
    'ClassificationError',
    'RelationExtractionError',
    'EquationBuildingError',
    'SolvingError',
    'VisualizationError',
    'ConfigurationError',
    'SystemError',
    'ComponentInitializationError',
    'ErrorHandler',
    'error_recovery',
    'log_errors',
    'timeout_with_recovery',
    'retry_with_backoff'
]
