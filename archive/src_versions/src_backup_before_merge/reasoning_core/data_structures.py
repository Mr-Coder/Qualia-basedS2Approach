"""
Core Data Structures for Mathematical Reasoning
==============================================

Unified data structures used across all reasoning components.
Extracted from models and processors for better modularity.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ProblemComplexity(Enum):
    """Problem complexity levels"""
    L0_EXPLICIT = "L0_显式计算"
    L1_SHALLOW = "L1_浅层推理"
    L2_MEDIUM = "L2_中等推理"
    L3_DEEP = "L3_深层推理"


class EntityType(Enum):
    """Entity types in mathematical problems"""
    NUMERICAL = "numerical"
    OBJECT = "object"
    UNIT = "unit"
    VARIABLE = "variable"
    OPERATION = "operation"


class RelationType(Enum):
    """Relation types in mathematical reasoning"""
    ARITHMETIC = "arithmetic"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    LOGICAL = "logical"
    CAUSAL = "causal"


@dataclass
class ProblemInput:
    """Unified input data structure for mathematical problems."""
    problem_text: str
    problem_id: Optional[str] = None
    dataset: Optional[str] = None
    complexity: Optional[ProblemComplexity] = None
    expected_answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Entity:
    """Mathematical entity in a problem."""
    name: str
    entity_type: EntityType
    value: Union[int, float, str, None] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    position: Optional[int] = None
    
    def __post_init__(self):
        """Validate entity data"""
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            self.confidence = 1.0


@dataclass
class Relation:
    """Mathematical relation between entities."""
    relation_type: RelationType
    entities: List[str]
    expression: str
    mathematical_form: Optional[str] = None
    confidence: float = 0.8
    reasoning: str = ""
    implicit: bool = False
    
    def __post_init__(self):
        """Validate relation data"""
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            self.confidence = 0.8


@dataclass
class ReasoningOutput:
    """Unified output data structure for reasoning results."""
    answer: Union[str, int, float]
    reasoning_chain: List[str]
    confidence_score: float
    processing_time: float
    entities_found: List[Entity] = field(default_factory=list)
    relations_discovered: List[Relation] = field(default_factory=list)
    complexity_level: Optional[ProblemComplexity] = None
    memory_usage: Optional[float] = None
    intermediate_steps: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for reasoning systems."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_processing_time: float
    avg_memory_usage: float
    total_problems_solved: int
    error_rate: float
    confidence_correlation: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


__all__ = [
    'ProblemComplexity', 'EntityType', 'RelationType',
    'ProblemInput', 'Entity', 'Relation', 'ReasoningOutput', 'PerformanceMetrics'
] 