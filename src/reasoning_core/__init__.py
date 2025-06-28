"""
Reasoning Core Module
====================

Core reasoning components including strategies, tools, and data structures.
"""

# Import core data structures
from .data_structures import (Entity, EntityType, PerformanceMetrics,
                              ProblemComplexity, ProblemInput, ReasoningOutput,
                              Relation, RelationType)
# Import strategies
from .strategies.base_strategy import (BaseReasoningStrategy, ReasoningResult,
                                       ReasoningStep)
from .strategies.chain_of_thought import ChainOfThoughtStrategy
from .strategies.enhanced_cotdir_strategy import EnhancedCOTDIRStrategy
# Import tools
from .tools import (BaseTool, ComplexityAnalyzer, NumericalComputeTool,
                    RelationDiscoveryTool, SymbolicMathTool, VisualizationTool)

# Other strategies will be implemented later
# from .strategies import GraphOfThoughtsStrategy, MCTSStrategy, TreeOfThoughtsStrategy


# Validation module will be implemented later
# from .validation import ConfidenceEstimator, LogicValidator, MathValidator

__all__ = [
    # Data structures
    'ProblemComplexity', 'EntityType', 'RelationType',
    'ProblemInput', 'Entity', 'Relation', 'ReasoningOutput', 'PerformanceMetrics',
    
    # Strategies
    'BaseReasoningStrategy', 'ReasoningResult', 'ReasoningStep',
    'ChainOfThoughtStrategy', 'EnhancedCOTDIRStrategy',
    
    # Tools
    'BaseTool', 'SymbolicMathTool', 'VisualizationTool', 'NumericalComputeTool',
    'RelationDiscoveryTool', 'ComplexityAnalyzer',
    
    # 'ReasoningEngine',  # Will be implemented later
    # 'TreeOfThoughtsStrategy',  # Will be implemented later
    # 'GraphOfThoughtsStrategy',  # Will be implemented later
    # 'MCTSStrategy',  # Will be implemented later
    # 'LogicValidator',  # Will be implemented later
    # 'MathValidator',  # Will be implemented later
    # 'ConfidenceEstimator'  # Will be implemented later
] 