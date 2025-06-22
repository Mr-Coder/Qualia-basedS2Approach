"""
评估器模块
~~~~~~~~~

这个模块包含了各种评估器类，用于评估数学问题求解系统的性能。

包含的评估器:
- PerformanceEvaluator: 性能评估器，评估整体准确率和按复杂度级别的性能
- RelationDiscoveryEvaluator: 关系发现评估器，评估隐式关系发现质量
- ReasoningChainEvaluator: 推理链评估器，评估推理链质量和错误传播

Author: Math Problem Solver Team
Version: 1.0.0
"""

import logging

# 配置日志
logger = logging.getLogger(__name__)

# 导入评估器类
from .performance_evaluator import PerformanceEvaluator
from .reasoning_chain_evaluator import ReasoningChainEvaluator
from .relation_discovery_evaluator import RelationDiscoveryEvaluator

# 为每个评估器类添加日志记录器
evaluator_classes = [
    PerformanceEvaluator,
    RelationDiscoveryEvaluator,
    ReasoningChainEvaluator
]

for cls in evaluator_classes:
    if not hasattr(cls, 'logger'):
        setattr(cls, 'logger', logging.getLogger(f"{__name__}.{cls.__name__}"))

# 导出的类
__all__ = [
    'PerformanceEvaluator',
    'RelationDiscoveryEvaluator',
    'ReasoningChainEvaluator'
]

# 包元数据
__version__ = '1.0.0'
__author__ = 'Math Problem Solver Team'

logger.info(f"Initialized evaluators package v{__version__}") 