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