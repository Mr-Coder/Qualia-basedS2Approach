"""
Data module for math problem solver.

This module contains dataset characteristics, performance analysis, and related utilities.
"""

from .dataset_characteristics import (DATASET_CHARACTERISTICS, DatasetInfo,
                                      calculate_weighted_complexity_score,
                                      export_to_json, get_all_datasets,
                                      get_complexity_distribution,
                                      get_dataset_info, get_dataset_statistics,
                                      get_datasets_by_domain,
                                      get_datasets_by_language,
                                      print_dataset_table)
from .performance_analysis import (ABLATION_DATA, COMPLEXITY_PERFORMANCE,
                                   COMPONENT_INTERACTION, EFFICIENCY_DATA,
                                   PERFORMANCE_DATA, REASONING_CHAIN_DATA,
                                   RELATION_DISCOVERY_DATA, AblationResults,
                                   ComplexityPerformance, ComponentInteraction,
                                   EfficiencyMetrics, MethodPerformance,
                                   ReasoningChainMetrics,
                                   RelationDiscoveryMetrics,
                                   analyze_component_contribution,
                                   calculate_average_performance,
                                   export_performance_data, get_all_methods,
                                   get_best_performing_method,
                                   get_efficiency_ranking,
                                   get_method_performance,
                                   get_robustness_ranking)

__all__ = [
    # Dataset characteristics
    'DatasetInfo',
    'DATASET_CHARACTERISTICS',
    'get_dataset_info',
    'get_all_datasets',
    'get_datasets_by_language',
    'get_datasets_by_domain',
    'get_complexity_distribution',
    'calculate_weighted_complexity_score',
    'get_dataset_statistics',
    'export_to_json',
    'print_dataset_table',
    
    # Performance analysis
    'MethodPerformance',
    'ComplexityPerformance',
    'EfficiencyMetrics',
    'AblationResults',
    'ComponentInteraction',
    'RelationDiscoveryMetrics',
    'ReasoningChainMetrics',
    'PERFORMANCE_DATA',
    'COMPLEXITY_PERFORMANCE',
    'EFFICIENCY_DATA',
    'ABLATION_DATA',
    'COMPONENT_INTERACTION',
    'RELATION_DISCOVERY_DATA',
    'REASONING_CHAIN_DATA',
    'get_method_performance',
    'get_all_methods',
    'get_best_performing_method',
    'calculate_average_performance',
    'get_efficiency_ranking',
    'get_robustness_ranking',
    'analyze_component_contribution',
    'export_performance_data'
]

# 导入新的模块化架构
try:
    from .orchestrator import DataOrchestrator, data_orchestrator
    from .public_api import DataAPI, data_api
    __all__.extend(['DataAPI', 'data_api', 'DataOrchestrator', 'data_orchestrator'])
    modular_architecture_available = True
except ImportError:
    modular_architecture_available = False 