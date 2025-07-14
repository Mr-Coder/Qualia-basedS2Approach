"""
UI模块错误处理系统
提供全面的错误处理、异常管理和恢复机制
"""

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import json


class UIErrorType(Enum):
    """UI错误类型枚举"""
    VALIDATION_ERROR = "validation_error"
    COMPONENT_ERROR = "component_error"
    RENDER_ERROR = "render_error"
    EVENT_ERROR = "event_error"
    STATE_ERROR = "state_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    SYSTEM_ERROR = "system_error"


class UIErrorSeverity(Enum):
    """UI错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UIError:
    """UI错误数据结构"""
    error_id: str
    error_type: UIErrorType
    severity: UIErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    component_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "component_id": self.component_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
            "context": self.context
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class UIRecoveryAction:
    """UI恢复动作数据结构"""
    action_id: str
    action_type: str
    description: str
    handler: Callable
    timeout: float = 30.0
    retry_count: int = 3
    parameters: Dict[str, Any] = field(default_factory=dict)


class UIErrorHandler:
    """UI错误处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[UIError] = []
        self.recovery_actions: Dict[UIErrorType, List[UIRecoveryAction]] = {}
        self.error_callbacks: Dict[UIErrorType, List[Callable]] = {}
        self.max_history_size = 1000
        
        # 初始化默认恢复动作
        self._initialize_default_recovery_actions()
    
    def handle_error(self, error: Union[Exception, UIError], 
                    component_id: Optional[str] = None,
                    request_id: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> UIError:
        """处理错误"""
        try:
            # 如果是普通异常，转换为UIError
            if isinstance(error, Exception):
                ui_error = self._convert_exception_to_ui_error(
                    error, component_id, request_id, context
                )
            else:
                ui_error = error
            
            # 记录错误
            self._log_error(ui_error)
            
            # 添加到历史记录
            self._add_to_history(ui_error)
            
            # 执行错误回调
            self._execute_error_callbacks(ui_error)
            
            # 尝试恢复
            self._attempt_recovery(ui_error)
            
            return ui_error
            
        except Exception as e:
            # 处理错误处理器本身的错误
            self.logger.critical(f"Error in error handler: {e}")
            return UIError(
                error_id=self._generate_error_id(),
                error_type=UIErrorType.SYSTEM_ERROR,
                severity=UIErrorSeverity.CRITICAL,
                message=f"Error handler failure: {str(e)}"
            )
    
    def register_recovery_action(self, error_type: UIErrorType, action: UIRecoveryAction) -> None:
        """注册恢复动作"""
        if error_type not in self.recovery_actions:
            self.recovery_actions[error_type] = []
        self.recovery_actions[error_type].append(action)
    
    def register_error_callback(self, error_type: UIErrorType, callback: Callable) -> None:
        """注册错误回调"""
        if error_type not in self.error_callbacks:
            self.error_callbacks[error_type] = []
        self.error_callbacks[error_type].append(callback)
    
    def get_error_history(self, limit: Optional[int] = None) -> List[UIError]:
        """获取错误历史"""
        if limit:
            return self.error_history[-limit:]
        return self.error_history.copy()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        if not self.error_history:
            return {
                "total_errors": 0,
                "error_types": {},
                "severity_distribution": {},
                "recent_errors": []
            }
        
        # 按类型统计
        error_types = {}
        for error in self.error_history:
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # 按严重程度统计
        severity_distribution = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        # 最近的错误
        recent_errors = [error.to_dict() for error in self.error_history[-10:]]
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "severity_distribution": severity_distribution,
            "recent_errors": recent_errors
        }
    
    def clear_error_history(self) -> None:
        """清空错误历史"""
        self.error_history.clear()
        self.logger.info("Error history cleared")
    
    def _convert_exception_to_ui_error(self, exception: Exception,
                                     component_id: Optional[str] = None,
                                     request_id: Optional[str] = None,
                                     context: Optional[Dict[str, Any]] = None) -> UIError:
        """将异常转换为UI错误"""
        error_type = self._classify_exception(exception)
        severity = self._determine_severity(exception, error_type)
        
        return UIError(
            error_id=self._generate_error_id(),
            error_type=error_type,
            severity=severity,
            message=str(exception),
            details={"exception_type": type(exception).__name__},
            component_id=component_id,
            request_id=request_id,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
    
    def _classify_exception(self, exception: Exception) -> UIErrorType:
        """分类异常类型"""
        if isinstance(exception, ValueError):
            return UIErrorType.VALIDATION_ERROR
        elif isinstance(exception, KeyError):
            return UIErrorType.COMPONENT_ERROR
        elif isinstance(exception, TimeoutError):
            return UIErrorType.TIMEOUT_ERROR
        elif isinstance(exception, PermissionError):
            return UIErrorType.PERMISSION_ERROR
        elif isinstance(exception, MemoryError):
            return UIErrorType.RESOURCE_ERROR
        elif isinstance(exception, ConnectionError):
            return UIErrorType.NETWORK_ERROR
        else:
            return UIErrorType.SYSTEM_ERROR
    
    def _determine_severity(self, exception: Exception, error_type: UIErrorType) -> UIErrorSeverity:
        """确定错误严重程度"""
        if isinstance(exception, (MemoryError, SystemError)):
            return UIErrorSeverity.CRITICAL
        elif error_type in [UIErrorType.NETWORK_ERROR, UIErrorType.TIMEOUT_ERROR]:
            return UIErrorSeverity.HIGH
        elif error_type in [UIErrorType.VALIDATION_ERROR, UIErrorType.COMPONENT_ERROR]:
            return UIErrorSeverity.MEDIUM
        else:
            return UIErrorSeverity.LOW
    
    def _generate_error_id(self) -> str:
        """生成错误ID"""
        import uuid
        return f"ui_error_{uuid.uuid4().hex[:8]}"
    
    def _log_error(self, error: UIError) -> None:
        """记录错误"""
        log_level = {
            UIErrorSeverity.LOW: logging.WARNING,
            UIErrorSeverity.MEDIUM: logging.ERROR,
            UIErrorSeverity.HIGH: logging.ERROR,
            UIErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"UI Error [{error.error_id}] {error.error_type.value}: {error.message}"
        )
        
        if error.stack_trace:
            self.logger.debug(f"Stack trace for {error.error_id}:\n{error.stack_trace}")
    
    def _add_to_history(self, error: UIError) -> None:
        """添加到历史记录"""
        self.error_history.append(error)
        
        # 保持历史记录大小
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _execute_error_callbacks(self, error: UIError) -> None:
        """执行错误回调"""
        callbacks = self.error_callbacks.get(error.error_type, [])
        
        for callback in callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    def _attempt_recovery(self, error: UIError) -> None:
        """尝试恢复"""
        recovery_actions = self.recovery_actions.get(error.error_type, [])
        
        for action in recovery_actions:
            try:
                self.logger.info(f"Attempting recovery action: {action.description}")
                
                # 执行恢复动作
                success = action.handler(error, **action.parameters)
                
                if success:
                    self.logger.info(f"Recovery action succeeded: {action.description}")
                    break
                else:
                    self.logger.warning(f"Recovery action failed: {action.description}")
                    
            except Exception as e:
                self.logger.error(f"Error in recovery action {action.description}: {e}")
    
    def _initialize_default_recovery_actions(self) -> None:
        """初始化默认恢复动作"""
        # 组件错误恢复
        self.register_recovery_action(
            UIErrorType.COMPONENT_ERROR,
            UIRecoveryAction(
                action_id="reset_component",
                action_type="reset",
                description="Reset component state",
                handler=self._reset_component_handler
            )
        )
        
        # 渲染错误恢复
        self.register_recovery_action(
            UIErrorType.RENDER_ERROR,
            UIRecoveryAction(
                action_id="fallback_render",
                action_type="fallback",
                description="Use fallback renderer",
                handler=self._fallback_render_handler
            )
        )
        
        # 网络错误恢复
        self.register_recovery_action(
            UIErrorType.NETWORK_ERROR,
            UIRecoveryAction(
                action_id="retry_request",
                action_type="retry",
                description="Retry network request",
                handler=self._retry_request_handler,
                retry_count=3
            )
        )
        
        # 超时错误恢复
        self.register_recovery_action(
            UIErrorType.TIMEOUT_ERROR,
            UIRecoveryAction(
                action_id="extend_timeout",
                action_type="extend",
                description="Extend timeout and retry",
                handler=self._extend_timeout_handler
            )
        )
    
    def _reset_component_handler(self, error: UIError, **kwargs) -> bool:
        """重置组件处理器"""
        try:
            # 这里应该调用实际的组件重置逻辑
            self.logger.info(f"Resetting component: {error.component_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset component: {e}")
            return False
    
    def _fallback_render_handler(self, error: UIError, **kwargs) -> bool:
        """回退渲染处理器"""
        try:
            # 这里应该使用备用渲染器
            self.logger.info("Using fallback renderer")
            return True
        except Exception as e:
            self.logger.error(f"Failed to use fallback renderer: {e}")
            return False
    
    def _retry_request_handler(self, error: UIError, **kwargs) -> bool:
        """重试请求处理器"""
        try:
            # 这里应该重试网络请求
            self.logger.info(f"Retrying request: {error.request_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to retry request: {e}")
            return False
    
    def _extend_timeout_handler(self, error: UIError, **kwargs) -> bool:
        """延长超时处理器"""
        try:
            # 这里应该延长超时时间并重试
            self.logger.info("Extending timeout and retrying")
            return True
        except Exception as e:
            self.logger.error(f"Failed to extend timeout: {e}")
            return False


class UIErrorNotifier:
    """UI错误通知器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.notification_handlers: Dict[UIErrorSeverity, List[Callable]] = {}
        self.error_filters: List[Callable] = []
        self.rate_limiter = {}
    
    def register_notification_handler(self, severity: UIErrorSeverity, handler: Callable) -> None:
        """注册通知处理器"""
        if severity not in self.notification_handlers:
            self.notification_handlers[severity] = []
        self.notification_handlers[severity].append(handler)
    
    def add_error_filter(self, filter_func: Callable) -> None:
        """添加错误过滤器"""
        self.error_filters.append(filter_func)
    
    def notify_error(self, error: UIError) -> None:
        """发送错误通知"""
        try:
            # 应用过滤器
            if not self._should_notify(error):
                return
            
            # 检查速率限制
            if self._is_rate_limited(error):
                return
            
            # 发送通知
            handlers = self.notification_handlers.get(error.severity, [])
            
            for handler in handlers:
                try:
                    handler(error)
                except Exception as e:
                    self.logger.error(f"Error in notification handler: {e}")
            
            # 更新速率限制
            self._update_rate_limit(error)
            
        except Exception as e:
            self.logger.error(f"Error in error notifier: {e}")
    
    def _should_notify(self, error: UIError) -> bool:
        """判断是否应该发送通知"""
        for filter_func in self.error_filters:
            try:
                if not filter_func(error):
                    return False
            except Exception as e:
                self.logger.error(f"Error in error filter: {e}")
        
        return True
    
    def _is_rate_limited(self, error: UIError) -> bool:
        """检查速率限制"""
        # 简单的速率限制实现
        error_key = f"{error.error_type.value}_{error.component_id or 'global'}"
        current_time = datetime.now()
        
        if error_key in self.rate_limiter:
            last_time, count = self.rate_limiter[error_key]
            time_diff = (current_time - last_time).total_seconds()
            
            # 1分钟内不超过5个相同错误
            if time_diff < 60 and count >= 5:
                return True
        
        return False
    
    def _update_rate_limit(self, error: UIError) -> None:
        """更新速率限制"""
        error_key = f"{error.error_type.value}_{error.component_id or 'global'}"
        current_time = datetime.now()
        
        if error_key in self.rate_limiter:
            last_time, count = self.rate_limiter[error_key]
            time_diff = (current_time - last_time).total_seconds()
            
            if time_diff < 60:
                self.rate_limiter[error_key] = (current_time, count + 1)
            else:
                self.rate_limiter[error_key] = (current_time, 1)
        else:
            self.rate_limiter[error_key] = (current_time, 1)


class UIErrorRecoveryManager:
    """UI错误恢复管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.max_recovery_attempts = 3
        
        # 初始化默认恢复策略
        self._initialize_default_strategies()
    
    def register_recovery_strategy(self, strategy_name: str, strategy_func: Callable) -> None:
        """注册恢复策略"""
        self.recovery_strategies[strategy_name] = strategy_func
    
    def recover_from_error(self, error: UIError, strategy_name: Optional[str] = None) -> bool:
        """从错误中恢复"""
        try:
            # 选择恢复策略
            if strategy_name:
                strategy = self.recovery_strategies.get(strategy_name)
                if not strategy:
                    self.logger.error(f"Unknown recovery strategy: {strategy_name}")
                    return False
            else:
                strategy = self._select_recovery_strategy(error)
            
            if not strategy:
                self.logger.warning(f"No recovery strategy available for error: {error.error_id}")
                return False
            
            # 执行恢复
            recovery_result = strategy(error)
            
            # 记录恢复历史
            self._record_recovery_attempt(error, strategy_name or "auto", recovery_result)
            
            return recovery_result
            
        except Exception as e:
            self.logger.error(f"Error in recovery manager: {e}")
            return False
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """获取恢复历史"""
        return self.recovery_history.copy()
    
    def _select_recovery_strategy(self, error: UIError) -> Optional[Callable]:
        """选择恢复策略"""
        # 根据错误类型选择策略
        strategy_map = {
            UIErrorType.COMPONENT_ERROR: "component_reset",
            UIErrorType.RENDER_ERROR: "render_fallback",
            UIErrorType.NETWORK_ERROR: "network_retry",
            UIErrorType.TIMEOUT_ERROR: "timeout_extend",
            UIErrorType.STATE_ERROR: "state_restore"
        }
        
        strategy_name = strategy_map.get(error.error_type)
        return self.recovery_strategies.get(strategy_name) if strategy_name else None
    
    def _record_recovery_attempt(self, error: UIError, strategy_name: str, success: bool) -> None:
        """记录恢复尝试"""
        self.recovery_history.append({
            "error_id": error.error_id,
            "error_type": error.error_type.value,
            "strategy_name": strategy_name,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        # 限制历史记录大小
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]
    
    def _initialize_default_strategies(self) -> None:
        """初始化默认恢复策略"""
        self.register_recovery_strategy("component_reset", self._component_reset_strategy)
        self.register_recovery_strategy("render_fallback", self._render_fallback_strategy)
        self.register_recovery_strategy("network_retry", self._network_retry_strategy)
        self.register_recovery_strategy("timeout_extend", self._timeout_extend_strategy)
        self.register_recovery_strategy("state_restore", self._state_restore_strategy)
    
    def _component_reset_strategy(self, error: UIError) -> bool:
        """组件重置策略"""
        try:
            self.logger.info(f"Resetting component: {error.component_id}")
            # 这里应该实现实际的组件重置逻辑
            return True
        except Exception as e:
            self.logger.error(f"Component reset failed: {e}")
            return False
    
    def _render_fallback_strategy(self, error: UIError) -> bool:
        """渲染回退策略"""
        try:
            self.logger.info("Using fallback rendering")
            # 这里应该实现回退渲染逻辑
            return True
        except Exception as e:
            self.logger.error(f"Render fallback failed: {e}")
            return False
    
    def _network_retry_strategy(self, error: UIError) -> bool:
        """网络重试策略"""
        try:
            self.logger.info("Retrying network operation")
            # 这里应该实现网络重试逻辑
            return True
        except Exception as e:
            self.logger.error(f"Network retry failed: {e}")
            return False
    
    def _timeout_extend_strategy(self, error: UIError) -> bool:
        """超时延长策略"""
        try:
            self.logger.info("Extending timeout")
            # 这里应该实现超时延长逻辑
            return True
        except Exception as e:
            self.logger.error(f"Timeout extend failed: {e}")
            return False
    
    def _state_restore_strategy(self, error: UIError) -> bool:
        """状态恢复策略"""
        try:
            self.logger.info("Restoring component state")
            # 这里应该实现状态恢复逻辑
            return True
        except Exception as e:
            self.logger.error(f"State restore failed: {e}")
            return False


# 全局错误处理器实例
ui_error_handler = UIErrorHandler()
ui_error_notifier = UIErrorNotifier()
ui_error_recovery_manager = UIErrorRecoveryManager()


def handle_ui_error(error: Union[Exception, UIError], 
                   component_id: Optional[str] = None,
                   request_id: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None) -> UIError:
    """全局UI错误处理函数"""
    ui_error = ui_error_handler.handle_error(error, component_id, request_id, context)
    ui_error_notifier.notify_error(ui_error)
    return ui_error


def recover_from_ui_error(error: UIError, strategy_name: Optional[str] = None) -> bool:
    """全局UI错误恢复函数"""
    return ui_error_recovery_manager.recover_from_error(error, strategy_name)


def get_ui_error_statistics() -> Dict[str, Any]:
    """获取UI错误统计信息"""
    return ui_error_handler.get_error_statistics()


def clear_ui_error_history() -> None:
    """清空UI错误历史"""
    ui_error_handler.clear_error_history()