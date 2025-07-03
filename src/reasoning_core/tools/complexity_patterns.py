"""
Complexity Patterns and Utilities
================================

Defines complexity level patterns, keywords, and related utilities for mathematical problem analysis.
"""

from typing import Dict

from ..data_structures import ProblemComplexity

COMPLEXITY_INDICATORS: Dict[ProblemComplexity, Dict] = {
    ProblemComplexity.L0_EXPLICIT: {
        'patterns': [
            r'^\d+[\+\-\×\÷]\d+=\?$',  # Pure arithmetic
            r'计算\s*\d+[\+\-\×\÷]\d+',  # Calculate X op Y
        ],
        'keywords': ['计算', '等于', '是多少'],
        'max_operations': 1,
        'max_entities': 2
    },
    ProblemComplexity.L1_SHALLOW: {
        'patterns': [
            r'有.*个.*还有.*个',  # Simple word problems
            r'买了.*个.*卖了.*个',
            r'\d+.*\d+.*一共'
        ],
        'keywords': ['一共', '总共', '还有', '买了', '卖了'],
        'max_operations': 2,
        'max_entities': 4
    },
    ProblemComplexity.L2_MEDIUM: {
        'patterns': [
            r'每.*需要.*',  # Multi-step reasoning
            r'平均.*',
            r'比.*多.*',
            r'分成.*组'
        ],
        'keywords': ['每', '平均', '比', '多', '少', '分成', '需要'],
        'max_operations': 3,
        'max_entities': 6
    },
    ProblemComplexity.L3_DEEP: {
        'patterns': [
            r'首先.*然后.*最后',  # Multi-step with sequencing
            r'根据.*可以.*所以',
            r'如果.*那么.*'
        ],
        'keywords': ['首先', '然后', '最后', '根据', '如果', '那么', '因此', '所以'],
        'max_operations': 5,
        'max_entities': 8
    }
} 