"""
AI协作友好的标准化接口定义

这个模块提供了整个数学推理系统的标准化接口，
让AI助手能够轻松理解和扩展系统功能。

AI_CONTEXT: 这是系统的接口层，定义了所有组件必须遵循的协议
RESPONSIBILITY: 提供类型安全的接口定义和协议规范
"""

from .base_protocols import (DataProcessor, ExperimentRunner, Orchestrator,
                             PerformanceTracker, ReasoningStrategy, Validator)
from .data_structures import (ExperimentResult, MathProblem, OperationType,
                              PerformanceMetrics, ProblemComplexity,
                              ProblemType, ReasoningResult, ReasoningStep,
                              ValidationResult)
from .exceptions import (AICollaborativeError, ConfigurationError,
                         DataProcessingError, ReasoningError, ValidationError,
                         handle_ai_collaborative_error)

__all__ = [
    # 协议接口
    'ReasoningStrategy',
    'DataProcessor', 
    'Validator',
    'Orchestrator',
    'ExperimentRunner',
    'PerformanceTracker',
    
    # 数据结构
    'MathProblem',
    'ReasoningStep',
    'ReasoningResult',
    'ValidationResult',
    'ExperimentResult',
    'PerformanceMetrics',
    'ProblemComplexity',
    'ProblemType',
    'OperationType',
    
    # 异常类
    'AICollaborativeError',
    'ReasoningError',
    'ValidationError',
    'ConfigurationError',
    'DataProcessingError',
    'handle_ai_collaborative_error'
]

# AI_HINT: 这些是系统的核心接口，所有模块都应该依赖这些抽象 