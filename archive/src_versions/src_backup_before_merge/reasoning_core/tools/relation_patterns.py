"""
Relation Patterns and Utilities
==============================

Defines relation patterns and related utilities for mathematical relation discovery.
"""

from typing import Dict

from ..data_structures import RelationType

RELATION_PATTERNS: Dict[str, Dict] = {
    'arithmetic_addition': {
        'keywords': ['总共', '一共', '合计', '相加', '加起来', '总计', 'total', 'together'],
        'relation_type': RelationType.ARITHMETIC,
        'operation': 'addition',
        'confidence_base': 0.8
    },
    'arithmetic_subtraction': {
        'keywords': ['剩下', '还剩', '余下', '减去', '少了', 'remaining', 'left', 'subtract'],
        'relation_type': RelationType.ARITHMETIC,
        'operation': 'subtraction',
        'confidence_base': 0.8
    },
    'arithmetic_multiplication': {
        'keywords': ['每', '总计', '乘以', '倍', '共有', 'each', 'times', 'multiply'],
        'relation_type': RelationType.ARITHMETIC,
        'operation': 'multiplication',
        'confidence_base': 0.8
    },
    'arithmetic_division': {
        'keywords': ['平均分', '分成', '每组', '每人', '除以', 'divide', 'per', 'average'],
        'relation_type': RelationType.ARITHMETIC,
        'operation': 'division',
        'confidence_base': 0.8
    },
    'comparison': {
        'keywords': ['比', '多', '少', '大', '小', '更', 'more', 'less', 'than'],
        'relation_type': RelationType.COMPARISON,
        'operation': 'comparison',
        'confidence_base': 0.75
    },
    'temporal': {
        'keywords': ['小时', '分钟', '天', '从', '到', 'hour', 'minute', 'day', 'from', 'to'],
        'relation_type': RelationType.TEMPORAL,
        'operation': 'time_calculation',
        'confidence_base': 0.7
    }
} 