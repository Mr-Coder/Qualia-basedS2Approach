"""
核心模块初始化

导出模块化架构的核心组件和接口。
"""

from .exceptions import (APIError, ConfigurationError, ModularSystemError,
                         ModuleDependencyError, ModuleInitializationError,
                         ModuleNotFoundError, ModuleRegistrationError,
                         OrchestrationError, ProcessingError, ValidationError,
                         handle_module_error)
from .interfaces import (BaseOrchestrator, BaseProcessor, BaseValidator,
                         ModuleInfo, ModuleType, PublicAPI, SystemMessage)
from .module_registry import ModuleRegistryImpl, registry
from .system_orchestrator import SystemOrchestrator, system_orchestrator

__all__ = [
    # 接口和基础类
    "ModuleType",
    "ModuleInfo", 
    "BaseValidator",
    "BaseProcessor",
    "BaseOrchestrator",
    "PublicAPI",
    "SystemMessage",
    
    # 异常类
    "ModularSystemError",
    "ModuleInitializationError",
    "ModuleRegistrationError",
    "ModuleNotFoundError",
    "ModuleDependencyError",
    "ValidationError",
    "ProcessingError",
    "OrchestrationError",
    "ConfigurationError",
    "APIError",
    "handle_module_error",
    
    # 核心组件
    "ModuleRegistryImpl",
    "registry",
    "SystemOrchestrator",
    "system_orchestrator"
]

__version__ = "1.0.0" 