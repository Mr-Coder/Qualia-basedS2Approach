"""
核心模块初始化

导出核心组件和接口，包括重构后的统一架构。
"""

from .exceptions import (
    AuthenticationError, AuthorizationError, APIError, ConfigurationError, 
    COTBaseException, ExceptionRecoveryStrategy, InputValidationError,
    ModuleError, ModuleRegistrationError, ModuleNotFoundError, ModuleDependencyError,
    OrchestrationError, PerformanceError, ProcessingError, ReasoningError,
    SecurityError, TemplateMatchingError, TimeoutError,
    ValidationError, handle_exceptions, handle_module_error
)

from .interfaces import (
    ICacheManager, IConfidenceCalculator, IConfigManager,
    IGNNEnhancer, IMonitor, INumberExtractor, IProcessor,
    IReasoningEngine, IResultFormatter, ITemplateManager,
    IValidator, ProcessingResult, ProcessingStatus,
    ReasoningContext, ReasoningStep, ModuleType, ModuleInfo,
    PublicAPI, BaseProcessor, BaseOrchestrator
)

from .module_registry import registry
from .orchestrator import (
    unified_orchestrator, system_orchestrator, enhanced_system_orchestrator,
    UnifiedSystemOrchestrator
)

# 新的重构组件
from .orchestration_strategy import (
    OrchestrationStrategy, BaseOrchestratorStrategy, UnifiedStrategy,
    ReasoningStrategy, ProcessingStrategy, OrchestratorStrategyFactory,
    create_orchestrator_strategy
)

from .security_service import (
    SecurityService, get_security_service, get_secure_evaluator,
    safe_eval, get_secure_file_manager, get_secure_config_manager
)

from .problem_solver_interface import (
    ProblemType, SolutionStrategy, ProblemInput, ProblemOutput,
    BaseProblemSolver, ChainOfThoughtSolver, DirectReasoningSolver,
    ProblemSolverFactory, create_problem_solver, solve_problem_unified
)

__all__ = [
    # 异常类
    'COTBaseException', 'ValidationError', 'InputValidationError',
    'ProcessingError', 'ReasoningError', 'TemplateMatchingError',
    'ConfigurationError', 'PerformanceError', 'TimeoutError',
    'SecurityError', 'AuthenticationError', 'AuthorizationError',
    'ModuleError', 'ModuleRegistrationError', 'ModuleNotFoundError', 
    'ModuleDependencyError', 'OrchestrationError', 'APIError',
    'handle_exceptions', 'ExceptionRecoveryStrategy', 'handle_module_error',
    
    # 接口类
    'IProcessor', 'IValidator', 'IReasoningEngine', 'ITemplateManager',
    'INumberExtractor', 'IConfidenceCalculator', 'ICacheManager',
    'IMonitor', 'IGNNEnhancer', 'IConfigManager', 'IResultFormatter',
    'PublicAPI', 'BaseProcessor', 'BaseOrchestrator',
    
    # 数据类和枚举
    'ProcessingResult', 'ReasoningContext', 'ProcessingStatus', 'ReasoningStep',
    'ModuleType', 'ModuleInfo',
    
    # 核心组件
    'registry', 'unified_orchestrator', 'system_orchestrator', 'enhanced_system_orchestrator',
    'UnifiedSystemOrchestrator',
    
    # 重构后的组件
    'OrchestrationStrategy', 'BaseOrchestratorStrategy', 'UnifiedStrategy',
    'ReasoningStrategy', 'ProcessingStrategy', 'OrchestratorStrategyFactory',
    'create_orchestrator_strategy',
    
    # 安全服务
    'SecurityService', 'get_security_service', 'get_secure_evaluator',
    'safe_eval', 'get_secure_file_manager', 'get_secure_config_manager',
    
    # 统一问题求解接口
    'ProblemType', 'SolutionStrategy', 'ProblemInput', 'ProblemOutput',
    'BaseProblemSolver', 'ChainOfThoughtSolver', 'DirectReasoningSolver',
    'ProblemSolverFactory', 'create_problem_solver', 'solve_problem_unified'
]

__version__ = "2.1.0"  # 版本升级以反映重构改进 