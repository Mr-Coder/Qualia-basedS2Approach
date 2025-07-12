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
