"""
Complexity Analyzer Tool
========================

Provides problem complexity classification for reasoning strategies.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..data_structures import Entity, ProblemComplexity
from .complexity_patterns import COMPLEXITY_INDICATORS


class ComplexityAnalyzer:
    """
    Problem complexity analyzer tool.
    Analyzes the complexity of a mathematical problem using patterns, keywords, and structure.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize complexity analyzer.
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.complexity_indicators = COMPLEXITY_INDICATORS

    def analyze_complexity(self, problem_text: str, entities: Optional[List[Entity]] = None) -> Dict[str, Any]:
        """
        Analyze the complexity of a mathematical problem.
        Args:
            problem_text: The problem text to analyze
            entities: Optional list of extracted entities
        Returns:
            Dict containing complexity analysis results
        """
        self.logger.debug(f"Analyzing complexity for problem: {problem_text[:50]}...")
        pattern_scores = self._analyze_patterns(problem_text)
        keyword_scores = self._analyze_keywords(problem_text)
        structure_scores = self._analyze_structure(problem_text, entities)
        complexity_scores = {}
        for level in ProblemComplexity:
            pattern_score = pattern_scores.get(level, 0)
            keyword_score = keyword_scores.get(level, 0)
            structure_score = structure_scores.get(level, 0)
            combined_score = (
                pattern_score * 0.4 +
                keyword_score * 0.3 +
                structure_score * 0.3
            )
            complexity_scores[level] = combined_score
        final_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0]
        text_length = len(problem_text)
        entity_count = len(entities) if entities else self._estimate_entity_count(problem_text)
        operation_count = self._count_operations(problem_text)
        return {
            'complexity_level': final_complexity,
            'confidence': complexity_scores[final_complexity],
            'complexity_scores': {level.value: score for level, score in complexity_scores.items()},
            'metrics': {
                'text_length': text_length,
                'entity_count': entity_count,
                'operation_count': operation_count,
                'reasoning_depth': self._estimate_reasoning_depth(problem_text)
            },
            'factors': {
                'pattern_scores': {level.value: score for level, score in pattern_scores.items()},
                'keyword_scores': {level.value: score for level, score in keyword_scores.items()},
                'structure_scores': {level.value: score for level, score in structure_scores.items()}
            }
        }

    def _analyze_patterns(self, problem_text: str) -> Dict[ProblemComplexity, float]:
        """Analyze complexity based on regex patterns."""
        scores = {}
        for level, indicators in self.complexity_indicators.items():
            pattern_score = 0
            pattern_count = 0
            for pattern in indicators['patterns']:
                if re.search(pattern, problem_text):
                    pattern_score += 1
                pattern_count += 1
            scores[level] = pattern_score / max(pattern_count, 1)
        return scores

    def _analyze_keywords(self, problem_text: str) -> Dict[ProblemComplexity, float]:
        """Analyze complexity based on keywords."""
        scores = {}
        for level, indicators in self.complexity_indicators.items():
            keyword_matches = sum(1 for keyword in indicators['keywords'] if keyword in problem_text)
            keyword_total = len(indicators['keywords'])
            scores[level] = keyword_matches / max(keyword_total, 1)
        return scores

    def _analyze_structure(self, problem_text: str, entities: Optional[List[Entity]]) -> Dict[ProblemComplexity, float]:
        """Analyze complexity based on structural features."""
        scores = {}
        entity_count = len(entities) if entities else self._estimate_entity_count(problem_text)
        operation_count = self._count_operations(problem_text)
        for level, indicators in self.complexity_indicators.items():
            max_operations = indicators['max_operations']
            max_entities = indicators['max_entities']
            operation_score = min(operation_count / max_operations, 1.0) if max_operations > 0 else 0
            entity_score = min(entity_count / max_entities, 1.0) if max_entities > 0 else 0
            if level in [ProblemComplexity.L2_MEDIUM, ProblemComplexity.L3_DEEP]:
                scores[level] = (operation_score + entity_score) / 2
            else:
                scores[level] = max(0, 1.0 - (operation_score + entity_score) / 2)
        return scores

    def _estimate_entity_count(self, problem_text: str) -> int:
        """Estimate number of entities in the problem."""
        numbers = re.findall(r'\d+', problem_text)
        object_patterns = [r'个', r'只', r'本', r'人', r'元', r'米', r'千克']
        objects = sum(1 for pattern in object_patterns if re.search(r'\d+.*?' + pattern, problem_text))
        return len(numbers) + objects

    def _count_operations(self, problem_text: str) -> int:
        """Count mathematical operations in the problem."""
        operation_indicators = [
            '加', '减', '乘', '除', '总共', '一共', '剩下', '比', '多', '少',
            '平均', '分', '每', '倍', '+', '-', '×', '÷', '*', '/'
        ]
        operation_count = sum(1 for indicator in operation_indicators if indicator in problem_text)
        return min(operation_count, 5)

    def _estimate_reasoning_depth(self, problem_text: str) -> int:
        """Estimate the depth of reasoning required."""
        depth_indicators = [
            ('首先.*然后.*最后', 3),
            ('如果.*那么', 2),
            ('因为.*所以', 2),
            ('根据.*可以', 2),
            ('比.*多.*少', 2),
            ('每.*需要', 1),
            ('一共', 1)
        ]
        max_depth = 0
        for pattern, depth in depth_indicators:
            if re.search(pattern, problem_text):
                max_depth = max(max_depth, depth)
        return max_depth if max_depth > 0 else 1

    def get_complexity_distribution(self, problems: List[str]) -> Dict[str, int]:
        """Get complexity distribution for a list of problems"""
        distribution = {level.value: 0 for level in ProblemComplexity}
        
        for problem in problems:
            result = self.analyze_complexity(problem)
            complexity = result['complexity_level']
            distribution[complexity.value] += 1
        
        return distribution
    
    def is_solvable_by_level(self, problem_complexity: ProblemComplexity, 
                           strategy_level: ProblemComplexity) -> bool:
        """Check if a strategy level can solve a problem of given complexity"""
        complexity_order = [
            ProblemComplexity.L0_EXPLICIT,
            ProblemComplexity.L1_SHALLOW,
            ProblemComplexity.L2_MEDIUM,
            ProblemComplexity.L3_DEEP
        ]
        
        return complexity_order.index(strategy_level) >= complexity_order.index(problem_complexity)


__all__ = ['ComplexityAnalyzer'] 