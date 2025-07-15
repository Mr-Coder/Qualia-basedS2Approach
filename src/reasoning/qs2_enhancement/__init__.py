"""
QS²增强系统包初始化
==================

导出QS²增强隐式关系发现系统的主要组件和工具。
"""

from .qualia_constructor import (
    QualiaRole,
    QualiaStructure, 
    QualiaStructureConstructor
)

from .compatibility_engine import (
    CompatibilityType,
    CompatibilityResult,
    CompatibilityEngine
)

from .enhanced_ird_engine import (
    RelationStrength,
    RelationType,
    EnhancedRelation,
    DiscoveryResult,
    EnhancedIRDEngine
)

from .support_structures import (
    QS2Config,
    ProcessingCache,
    PerformanceMonitor,
    DataSerializer,
    BatchProcessor,
    ValidationUtils,
    create_default_config,
    create_performance_monitor,
    create_cache,
    setup_logging
)

__all__ = [
    # 语义构建组件
    "QualiaRole",
    "QualiaStructure",
    "QualiaStructureConstructor",
    
    # 兼容性计算组件
    "CompatibilityType",
    "CompatibilityResult", 
    "CompatibilityEngine",
    
    # 增强关系发现组件
    "RelationStrength",
    "RelationType",
    "EnhancedRelation",
    "DiscoveryResult",
    "EnhancedIRDEngine",
    
    # 支持结构和工具
    "QS2Config",
    "ProcessingCache",
    "PerformanceMonitor",
    "DataSerializer",
    "BatchProcessor",
    "ValidationUtils",
    "create_default_config",
    "create_performance_monitor",
    "create_cache",
    "setup_logging"
]

# 版本信息
__version__ = "1.0.0"
__author__ = "QS² Enhancement Team"
__description__ = "QS² Enhanced Implicit Relation Discovery System"