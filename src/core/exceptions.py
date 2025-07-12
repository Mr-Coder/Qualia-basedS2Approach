"""
系统异常定义

为模块化架构提供统一的异常处理机制。
"""


class ModularSystemError(Exception):
    """模块化系统基础异常"""
    
    def __init__(self, message: str, module_name: str = None, error_code: str = None):
        super().__init__(message)
        self.module_name = module_name
        self.error_code = error_code
        self.message = message


class ModuleInitializationError(ModularSystemError):
    """模块初始化异常"""
    pass


class ModuleRegistrationError(ModularSystemError):
    """模块注册异常"""
    pass


class ModuleNotFoundError(ModularSystemError):
    """模块未找到异常"""
    pass


class ModuleDependencyError(ModularSystemError):
    """模块依赖异常"""
    pass


class ValidationError(ModularSystemError):
    """数据验证异常"""
    pass


class ProcessingError(ModularSystemError):
    """数据处理异常"""
    pass


class OrchestrationError(ModularSystemError):
    """协调异常"""
    pass


class ConfigurationError(ModularSystemError):
    """配置异常"""
    pass


class APIError(ModularSystemError):
    """API调用异常"""
    pass


def handle_module_error(error: Exception, module_name: str, operation: str) -> ModularSystemError:
    """统一的错误处理函数"""
    if isinstance(error, ModularSystemError):
        return error
    
    error_message = f"Module '{module_name}' error during '{operation}': {str(error)}"
    return ModularSystemError(error_message, module_name=module_name) 