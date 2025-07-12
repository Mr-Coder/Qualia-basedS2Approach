"""
Relation Discovery Tool
======================

Simplified relation discovery functionality extracted from processors.
Provides core relation extraction capabilities for reasoning strategies.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..data_structures import Entity, EntityType, Relation, RelationType


class RelationDiscoveryTool:
    """
    Simplified relation discovery tool for mathematical reasoning.
    
    Extracted from the complex relation_extractor.py for better modularity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize relation discovery tool"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Core relation patterns
        self.relation_patterns = {
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
        
        # 1. Pattern-based relation discovery
        pattern_relations = self._discover_pattern_relations(entities, context)
        relations.extend(pattern_relations)
        
        # 2. Implicit relation inference
        implicit_relations = self._discover_implicit_relations(entities, context)
        relations.extend(implicit_relations)
        
        # 3. Filter by confidence threshold
        filtered_relations = [r for r in relations if r.confidence >= self.confidence_threshold]
        
        processing_time = time.time() - start_time
        self.logger.debug(f"Discovered {len(filtered_relations)} relations in {processing_time:.3f}s")
        
        return filtered_relations
    
    def _discover_pattern_relations(self, entities: List[Entity], context: str) -> List[Relation]:
        """Discover relations using predefined patterns"""
        relations = []
        numerical_entities = [e for e in entities if e.entity_type == EntityType.NUMERICAL]
        
        for pattern_name, pattern_data in self.relation_patterns.items():
            # Check if pattern keywords exist in context
            keywords_found = any(keyword in context for keyword in pattern_data['keywords'])
            
            if keywords_found and len(numerical_entities) >= 2:
                relation = self._create_relation_from_pattern(
                    pattern_name, pattern_data, numerical_entities, context
                )
                if relation:
                    relations.append(relation)
        
        return relations
    
    def _discover_implicit_relations(self, entities: List[Entity], context: str) -> List[Relation]:
        """Discover implicit relations through semantic analysis"""
        relations = []
        numerical_entities = [e for e in entities if e.entity_type == EntityType.NUMERICAL]
        
        if len(numerical_entities) >= 2:
            # Implicit arithmetic relations based on context structure
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
    
    def _create_relation_from_pattern(self, pattern_name: str, pattern_data: Dict, 
                                    entities: List[Entity], context: str) -> Optional[Relation]:
        """Create a relation from a matched pattern"""
        if len(entities) < 2:
            return None
        
        # Generate mathematical expression
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
        
        # Calculate confidence based on pattern match strength
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
        """Calculate confidence score for pattern match"""
        base_confidence = pattern_data['confidence_base']
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in pattern_data['keywords'] if keyword in context)
        keyword_bonus = min(keyword_matches * 0.05, 0.15)  # Max 15% bonus
        
        # Context length factor (longer contexts get slight penalty)
        length_factor = max(0.9, 1.0 - len(context) / 1000)
        
        confidence = (base_confidence + keyword_bonus) * length_factor
        return min(confidence, 1.0)
    
    def _implies_addition(self, context: str) -> bool:
        """Check if context implies addition operation"""
        addition_indicators = ['和', '与', '还有', '以及', 'and', 'plus', 'with']
        return any(indicator in context for indicator in addition_indicators)
    
    def _implies_multiplication(self, context: str) -> bool:
        """Check if context implies multiplication operation"""
        multiplication_indicators = ['倍数', '乘', '组', 'times', 'groups of', 'sets of']
        return any(indicator in context for indicator in multiplication_indicators)
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported mathematical operations"""
        return list(set(pattern['operation'] for pattern in self.relation_patterns.values()))
    
    def get_pattern_info(self) -> Dict[str, Dict]:
        """Get information about available patterns"""
        return self.relation_patterns.copy()


__all__ = ['RelationDiscoveryTool'] 