"""
Models Package
=============

This package contains the core data structures and models for mathematical problem solving.

Main Components:
- ProcessedText: Represents processed text with NLP annotations
- Relations: Defines relationships between entities
- Equations: Mathematical equation representations
- ProblemStructure: Overall problem structure and components

Usage:
    from models import ProcessedText, Relations, Equations
    
    text = ProcessedText(...)
    relations = Relations(...)
    equations = Equations(...)

Version: 1.0.0
Author: Your Name
License: MIT
"""

import logging
from typing import List, Type

# 配置包级别的日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 如果需要ProcessedText，单独import
from .processed_text import ProcessedText
# 导入所有模型类
from .structures import (Attributes, Context, Entities, Equation, Equations,
                         ExtractionResult, FeatureSet, InferenceResult,
                         InferenceStep, MatchedModel, PatternMatch,
                         ProblemStructure, RelationCollection, RelationEntity,
                         Relations, Solution)

# 定义要添加日志记录器的类列表
LOGGED_CLASSES: List[Type] = [
    ProcessedText,
    Relations,
    RelationCollection,
    Equations,
    Solution,
    ProblemStructure
]

# 为每个主要类添加日志记录器
for cls in LOGGED_CLASSES:
    if not hasattr(cls, 'logger'):
        setattr(cls, 'logger', logging.getLogger(f"{__name__}.{cls.__name__}"))

# 导入新的模块化架构
try:
    from .orchestrator import ModelsOrchestrator, models_orchestrator
    from .public_api import ModelsAPI, models_api
    modular_architecture_available = True
except ImportError:
    modular_architecture_available = False

# 导入数学增强模块
try:
    from .advanced_math_engine import (
        AdvancedMathEngine, 
        MathematicalExpression, 
        MathResult,
        MathOperationType
    )
    from .physics_problem_solver import (
        PhysicsProblemSolver,
        PhysicsType,
        PhysicsQuantity,
        PhysicsProblem,
        PhysicsSolution
    )
    from .geometry_engine import (
        GeometryEngine,
        GeometryType,
        GeometricShape,
        GeometrySolution,
        Point2D,
        Point3D
    )
    from .mathematical_correctness_validator import (
        MathematicalCorrectnessValidator,
        ValidationResult,
        ValidationType
    )
    math_engine_available = True
except ImportError:
    math_engine_available = False

# 导出的类
__all__ = [
    'ProcessedText',
    'Relations',
    'RelationCollection',
    'Equations',
    'Solution',
    'ProblemStructure',
    'Entities',
    'Attributes',
    'Context',
    'InferenceResult',
    'InferenceStep',
    'Equation'
]

# 如果模块化架构可用，添加到导出列表
if modular_architecture_available:
    __all__.extend(['ModelsAPI', 'models_api', 'ModelsOrchestrator', 'models_orchestrator'])

# 如果数学引擎可用，添加到导出列表
if math_engine_available:
    __all__.extend([
        'AdvancedMathEngine',
        'MathematicalExpression',
        'MathResult',
        'MathOperationType',
        'PhysicsProblemSolver',
        'PhysicsType',
        'PhysicsQuantity',
        'PhysicsProblem',
        'PhysicsSolution',
        'GeometryEngine',
        'GeometryType',
        'GeometricShape',
        'GeometrySolution',
        'Point2D',
        'Point3D',
        'MathematicalCorrectnessValidator',
        'ValidationResult',
        'ValidationType'
    ])

# 包元数据
__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__license__ = 'MIT'

# 包初始化时的日志
logger.info(f"Initialized models package v{__version__}")

def get_version() -> str:
    """返回包的版本信息"""
    return __version__

def setup_logging(level: int = logging.INFO) -> None:
    """
    配置包的日志级别
    
    Args:
        level: 日志级别，默认为 INFO
    """
    logger.setLevel(level)
    for cls in LOGGED_CLASSES:
        if hasattr(cls, 'logger'):
            cls.logger.setLevel(level)
