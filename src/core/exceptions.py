"""
统一异常处理系统
提供项目中所有模块使用的标准异常类
"""

import logging
import traceback
from typing import Any, Dict, Optional

# 配置日志
logger = logging.getLogger(__name__)

class COTBaseException(Exception):
    """COT项目基础异常类"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        self.traceback_info = traceback.format_exc()
        
        # 记录异常日志
        logger.error(
            f"[{error_code}] {message}",
            extra={
                "error_code": error_code,
                "context": context,
                "cause": str(cause) if cause else None
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_info
        }

# 验证相关异常
class ValidationError(COTBaseException):
    """数据验证异常"""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        context = kwargs.get('context', {})
        if field:
            context['field'] = field
        if value is not None:
            context['invalid_value'] = str(value)
        
        super().__init__(
            message, 
            error_code="VALIDATION_ERROR",
            context=context,
            **kwargs
        )

class InputValidationError(ValidationError):
    """输入验证异常"""
    
    def __init__(self, message: str, input_text: str = None, **kwargs):
        context = kwargs.get('context', {})
        if input_text:
            context['input_text'] = input_text[:100] + "..." if len(input_text) > 100 else input_text
        
        super().__init__(
            message,
            error_code="INPUT_VALIDATION_ERROR", 
            context=context,
            **kwargs
        )

# 处理相关异常
class ProcessingError(COTBaseException):
    """处理过程异常"""
    
    def __init__(self, message: str, stage: str = None, **kwargs):
        context = kwargs.get('context', {})
        if stage:
            context['processing_stage'] = stage
            
        super().__init__(
            message,
            error_code="PROCESSING_ERROR",
            context=context,
            **kwargs
        )

class ReasoningError(ProcessingError):
    """推理过程异常"""
    
    def __init__(self, message: str, reasoning_step: int = None, **kwargs):
        context = kwargs.get('context', {})
        if reasoning_step is not None:
            context['reasoning_step'] = reasoning_step
            
        super().__init__(
            message,
            stage="reasoning",
            error_code="REASONING_ERROR",
            context=context,
            **kwargs
        )

class TemplateMatchingError(ProcessingError):
    """模板匹配异常"""
    
    def __init__(self, message: str, template_type: str = None, **kwargs):
        context = kwargs.get('context', {})
        if template_type:
            context['template_type'] = template_type
            
        super().__init__(
            message,
            stage="template_matching",
            error_code="TEMPLATE_MATCHING_ERROR", 
            context=context,
            **kwargs
        )

class TemplateError(COTBaseException):
    """模板系统异常"""
    
    def __init__(self, message: str, template_id: str = None, **kwargs):
        context = kwargs.get('context', {})
        if template_id:
            context['template_id'] = template_id
            
        super().__init__(
            message,
            error_code="TEMPLATE_ERROR",
            context=context,
            **kwargs
        )

# 配置相关异常
class ConfigurationError(COTBaseException):
    """配置异常"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
            
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            context=context,
            **kwargs
        )

# 模块管理相关异常
class ModuleError(COTBaseException):
    """模块相关异常"""
    
    def __init__(self, message: str, module_name: str = None, **kwargs):
        context = kwargs.get('context', {})
        if module_name:
            context['module_name'] = module_name
            
        super().__init__(
            message,
            error_code="MODULE_ERROR",
            context=context,
            **kwargs
        )

class ModuleRegistrationError(ModuleError):
    """模块注册异常"""
    
    def __init__(self, message: str, module_name: str = None, **kwargs):
        super().__init__(
            message,
            module_name=module_name,
            error_code="MODULE_REGISTRATION_ERROR",
            **kwargs
        )

class ModuleNotFoundError(ModuleError):
    """模块未找到异常"""
    
    def __init__(self, message: str, module_name: str = None, **kwargs):
        super().__init__(
            message,
            module_name=module_name,
            error_code="MODULE_NOT_FOUND_ERROR",
            **kwargs
        )

class ModuleDependencyError(ModuleError):
    """模块依赖异常"""
    
    def __init__(self, message: str, module_name: str = None, **kwargs):
        super().__init__(
            message,
            module_name=module_name,
            error_code="MODULE_DEPENDENCY_ERROR",
            **kwargs
        )
class OrchestrationError(COTBaseException):
    """系统协调异常"""
    
    def __init__(self, message: str, module_name: str = None, **kwargs):
        context = kwargs.get('context', {})
        if module_name:
            context['module_name'] = module_name
            
        super().__init__(
            message,
            error_code="ORCHESTRATION_ERROR",
            context=context,
            **kwargs
        )

class APIError(COTBaseException):
    """API调用异常"""
    
    def __init__(self, message: str, module_name: str = None, **kwargs):
        context = kwargs.get('context', {})
        if module_name:
            context['module_name'] = module_name
            
        super().__init__(
            message,
            error_code="API_ERROR",
            context=context,
            **kwargs
        )

# 性能相关异常  
class PerformanceError(COTBaseException):
    """性能异常"""
    
    def __init__(self, message: str, operation: str = None, duration: float = None, **kwargs):
        context = kwargs.get('context', {})
        if operation:
            context['operation'] = operation
        if duration is not None:
            context['duration_seconds'] = duration
            
        super().__init__(
            message,
            error_code="PERFORMANCE_ERROR",
            context=context,
            **kwargs
        )

class TimeoutError(PerformanceError):
    """超时异常"""
    
    def __init__(self, message: str, timeout_seconds: float = None, **kwargs):
        context = kwargs.get('context', {})
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
            
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            context=context,
            **kwargs
        )

# 安全相关异常
class SecurityError(COTBaseException):
    """安全异常"""
    
    def __init__(self, message: str, security_check: str = None, **kwargs):
        context = kwargs.get('context', {})
        if security_check:
            context['security_check'] = security_check
            
        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            context=context,
            **kwargs
        )

class AuthenticationError(SecurityError):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败", **kwargs):
        super().__init__(
            message,
            security_check="authentication",
            error_code="AUTHENTICATION_ERROR",
            **kwargs
        )

class AuthorizationError(SecurityError):
    """授权异常"""
    
    def __init__(self, message: str = "权限不足", **kwargs):
        super().__init__(
            message,
            security_check="authorization", 
            error_code="AUTHORIZATION_ERROR",
            **kwargs
        )

# 异常处理装饰器
def handle_exceptions(
    default_return=None,
    log_errors=True,
    reraise_as=None
):
    """异常处理装饰器"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except COTBaseException:
                # COT异常直接重新抛出
                raise
            except Exception as e:
                if log_errors:
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                
                if reraise_as:
                    raise reraise_as(
                        f"Error in {func.__name__}: {str(e)}",
                        cause=e
                    )
                
                if default_return is not None:
                    return default_return
                    
                # 包装为COT异常
                raise ProcessingError(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    cause=e
                )
        
        return wrapper
    return decorator

# 异常恢复策略
class ExceptionRecoveryStrategy:
    """异常恢复策略"""
    
    @staticmethod
    def create_fallback_result(error: COTBaseException) -> Dict[str, Any]:
        """创建后备结果"""
        return {
            "success": False,
            "error": error.to_dict(),
            "fallback": True,
            "answer": "无法计算",
            "confidence": 0.0,
            "reasoning_steps": [],
            "recovery_strategy": "fallback_result"
        }
    
    @staticmethod
    def should_retry(error: COTBaseException, attempt: int, max_attempts: int = 3) -> bool:
        """判断是否应该重试"""
        if attempt >= max_attempts:
            return False
            
        # 某些错误类型可以重试
        retryable_errors = [
            "TIMEOUT_ERROR",
            "PERFORMANCE_ERROR", 
            "PROCESSING_ERROR"
        ]
        
        return error.error_code in retryable_errors 

def handle_module_error(error: Exception, module_name: str, operation: str = "") -> COTBaseException:
    """处理模块错误，统一错误格式"""
    if isinstance(error, COTBaseException):
        return error
    
    # 包装为模块特定的错误
    if module_name == "reasoning":
        return ReasoningError(f"{operation}: {str(error)}", cause=error)
    elif module_name == "models":
        return ProcessingError(f"Model {operation}: {str(error)}", stage="model", cause=error)
    elif module_name == "system":
        return OrchestrationError(f"System {operation}: {str(error)}", module_name=module_name, cause=error)
    else:
        return ProcessingError(f"{module_name} {operation}: {str(error)}", stage=module_name, cause=error) 