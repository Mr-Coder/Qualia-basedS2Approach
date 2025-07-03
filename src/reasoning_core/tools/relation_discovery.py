"""
Relation Discovery Tool
======================

Provides core relation extraction capabilities for reasoning strategies.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..data_structures import Entity, EntityType, Relation, RelationType
from .relation_patterns import RELATION_PATTERNS


class RelationDiscoveryTool:
    """
    Simplified relation discovery tool for mathematical reasoning.
    Extracts mathematical relations between entities in a given context.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize relation discovery tool.
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.relation_patterns = RELATION_PATTERNS

    def discover_relations(self, entities: List[Entity], context: str) -> List[Relation]:
        """
        Discover mathematical relations between entities in the given context.
        Args:
            entities: List of mathematical entities
            context: Problem text context
        Returns:
            List of discovered relations
        """
        self.logger.debug(f"Discovering relations for {len(entities)} entities")
        start_time = time.time()
        relations = []
        pattern_relations = self._discover_pattern_relations(entities, context)
        relations.extend(pattern_relations)
        implicit_relations = self._discover_implicit_relations(entities, context)
        relations.extend(implicit_relations)
        filtered_relations = [r for r in relations if r.confidence >= self.confidence_threshold]
        processing_time = time.time() - start_time
        self.logger.debug(f"Discovered {len(filtered_relations)} relations in {processing_time:.3f}s")
        return filtered_relations

    def _discover_pattern_relations(self, entities: List[Entity], context: str) -> List[Relation]:
        """Discover relations using predefined patterns."""
        relations = []
        numerical_entities = [e for e in entities if e.entity_type == EntityType.NUMERICAL]
        for pattern_name, pattern_data in self.relation_patterns.items():
            keywords_found = any(keyword in context for keyword in pattern_data['keywords'])
            if keywords_found and len(numerical_entities) >= 2:
                relation = self._create_relation_from_pattern(
                    pattern_name, pattern_data, numerical_entities, context
                )
                if relation:
                    relations.append(relation)
        return relations

    def _discover_implicit_relations(self, entities: List[Entity], context: str) -> List[Relation]:
        """Discover implicit relations through semantic analysis."""
        relations = []
        numerical_entities = [e for e in entities if e.entity_type == EntityType.NUMERICAL]
        if len(numerical_entities) >= 2:
            if self._implies_addition(context):
                relation = Relation(
                    relation_type=RelationType.ARITHMETIC,
                    entities=[e.name for e in numerical_entities[:2]],
                    expression=f"{numerical_entities[0].value} + {numerical_entities[1].value}",
                    mathematical_form="addition",
                    reasoning="Implicit addition inferred from context",
                    implicit=True
                )
                relation.confidence = 0.7
                relations.append(relation)
            elif self._implies_multiplication(context):
                relation = Relation(
                    relation_type=RelationType.ARITHMETIC,
                    entities=[e.name for e in numerical_entities[:2]],
                    expression=f"{numerical_entities[0].value} × {numerical_entities[1].value}",
                    mathematical_form="multiplication",
                    reasoning="Implicit multiplication inferred from context",
                    implicit=True
                )
                relation.confidence = 0.7
                relations.append(relation)
        return relations

    def _create_relation_from_pattern(self, pattern_name: str, pattern_data: Dict, entities: List[Entity], context: str) -> Optional[Relation]:
        """Create a relation from a matched pattern."""
        if len(entities) < 2:
            return None
        operation = pattern_data['operation']
        value1, value2 = entities[0].value, entities[1].value
        if operation == 'addition':
            expression = f"{value1} + {value2}"
        elif operation == 'subtraction':
            expression = f"{value1} - {value2}"
        elif operation == 'multiplication':
            expression = f"{value1} × {value2}"
        elif operation == 'division':
            expression = f"{value1} ÷ {value2}"
        else:
            expression = f"{value1} {operation} {value2}"
        confidence = self._calculate_pattern_confidence(pattern_data, context)
        relation = Relation(
            relation_type=pattern_data['relation_type'],
            entities=[e.name for e in entities[:2]],
            expression=expression,
            mathematical_form=operation,
            reasoning=f"Pattern match: {pattern_name}",
            implicit=False
        )
        relation.confidence = confidence
        return relation

    def _calculate_pattern_confidence(self, pattern_data: Dict, context: str) -> float:
        """Calculate confidence score for pattern match."""
        base_confidence = pattern_data['confidence_base']
        keyword_matches = sum(1 for keyword in pattern_data['keywords'] if keyword in context)
        keyword_bonus = min(keyword_matches * 0.05, 0.15)
        length_factor = max(0.9, 1.0 - len(context) / 1000)
        confidence = (base_confidence + keyword_bonus) * length_factor
        return min(confidence, 1.0)

    def _implies_addition(self, context: str) -> bool:
        """Check if context implies addition operation."""
        addition_indicators = ['和', '与', '还有', '以及', 'and', 'plus', 'with']
        return any(indicator in context for indicator in addition_indicators)

    def _implies_multiplication(self, context: str) -> bool:
        """Check if context implies multiplication operation."""
        multiplication_indicators = ['倍数', '乘', '组', 'times', 'groups of', 'sets of']
        return any(indicator in context for indicator in multiplication_indicators)

    def get_supported_operations(self) -> List[str]:
        """Get list of supported mathematical operations."""
        return list(set(pattern['operation'] for pattern in self.relation_patterns.values()))

    def get_pattern_info(self) -> Dict[str, Dict]:
        """Get information about available patterns."""
        return self.relation_patterns.copy()


__all__ = ['RelationDiscoveryTool'] 