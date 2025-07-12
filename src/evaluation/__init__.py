"""
Evaluation System
================

Comprehensive evaluation framework for mathematical reasoning systems.
Implements the paper's multi-dataset evaluation approach with SOTA comparison.
"""

# Benchmarks and reports will be implemented later
# from .benchmarks import BenchmarkSuite
# from .reports import EvaluationReporter

from .evaluator import ComprehensiveEvaluator
from .metrics import (AccuracyMetric, EfficiencyMetric, ExplainabilityMetric,
                      ReasoningQualityMetric, RobustnessMetric)
from .sota_benchmark import BenchmarkResult, DatasetInfo, SOTABenchmarkSuite

# 导入新的模块化架构
try:
    from .orchestrator import EvaluationOrchestrator, evaluation_orchestrator
    from .public_api import EvaluationAPI, evaluation_api
    modular_architecture_available = True
except ImportError:
    modular_architecture_available = False

__all__ = [
    'AccuracyMetric',
    'ReasoningQualityMetric', 
    'EfficiencyMetric',
    'RobustnessMetric',
    'ExplainabilityMetric',
    'ComprehensiveEvaluator',
    'SOTABenchmarkSuite',
    'BenchmarkResult',
    'DatasetInfo',
    # 'BenchmarkSuite',  # Will be implemented later
    # 'EvaluationReporter'  # Will be implemented later
]

# 如果模块化架构可用，添加到导出列表
if modular_architecture_available:
    __all__.extend(['EvaluationAPI', 'evaluation_api', 'EvaluationOrchestrator', 'evaluation_orchestrator']) 