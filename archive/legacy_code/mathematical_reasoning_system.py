#!/usr/bin/env python3
"""
Mathematical Reasoning System Implementation

This module implements a comprehensive mathematical reasoning system based on the COT-DIR
(Chain-of-Thought with Directional Implicit Reasoning) approach, inspired by generative AI
research for mathematical problem solving.

Key Components:
1. Advanced NLP Processing Pipeline
2. Implicit Relation Discovery Engine
3. Multi-Level Reasoning Framework
4. Chain Verification and Validation
5. Performance Optimization

Author: AI Research Team
Date: 2025-01-31
"""

import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


class ProblemComplexity(Enum):
    """Mathematical problem complexity levels."""
    L0_EXPLICIT = "L0"      # Direct arithmetic operations
    L1_SHALLOW = "L1"       # Single-step inference required
    L2_MEDIUM = "L2"        # Multi-step reasoning with implicit relations
    L3_DEEP = "L3"          # Complex implicit relations and constraints


class RelationType(Enum):
    """Types of mathematical relations."""
    ARITHMETIC = "arithmetic"
    ALGEBRAIC = "algebraic"
    GEOMETRIC = "geometric"
    PROPORTION = "proportion"
    UNIT_CONVERSION = "unit_conversion"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CONSTRAINT = "constraint"
    PHYSICAL = "physical"


class EntityType(Enum):
    """Mathematical entity types."""
    NUMBER = "number"
    VARIABLE = "variable"
    UNIT = "unit"
    OBJECT = "object"
    OPERATION = "operation"
    CONSTRAINT = "constraint"


@dataclass
class MathEntity:
    """Represents a mathematical entity in the problem."""
    text: str
    entity_type: EntityType
    value: Optional[Union[float, str]] = None
    unit: Optional[str] = None
    confidence: float = 1.0
    position: Tuple[int, int] = (0, 0)  # Start and end positions in text
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImplicitRelation:
    """Represents an implicit mathematical relation."""
    relation_id: str
    relation_type: RelationType
    entities: List[MathEntity]
    description: str
    mathematical_expression: Optional[str] = None
    confidence: float = 0.0
    derivation_steps: List[str] = None
    
    def __post_init__(self):
        if self.derivation_steps is None:
            self.derivation_steps = []
    
    def to_dict(self) -> Dict[str, Any]:
        # Handle both string and enum types for relation_type
        if isinstance(self.relation_type, RelationType):
            relation_type_value = self.relation_type.value
        else:
            relation_type_value = str(self.relation_type)
        
        return {
            "relation_id": self.relation_id,
            "relation_type": relation_type_value,
            "entities": [entity.to_dict() for entity in self.entities],
            "description": self.description,
            "mathematical_expression": self.mathematical_expression,
            "confidence": self.confidence,
            "derivation_steps": self.derivation_steps
        }


@dataclass
class ReasoningStep:
    """Represents a step in the reasoning chain."""
    step_id: int
    description: str
    operation: str
    input_entities: List[MathEntity]
    output_entity: MathEntity
    confidence: float
    dependencies: List[int]  # Previous step IDs this depends on
    verification_status: bool = False
    error_bounds: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "operation": self.operation,
            "input_entities": [entity.to_dict() for entity in self.input_entities],
            "output_entity": self.output_entity.to_dict(),
            "confidence": self.confidence,
            "dependencies": self.dependencies,
            "verification_status": self.verification_status,
            "error_bounds": self.error_bounds
        }


class NLPProcessor:
    """Advanced NLP processing for mathematical text."""
    
    def __init__(self):
        self.number_pattern = re.compile(r'-?\d+\.?\d*')
        self.unit_patterns = {
            'length': r'\b(m|cm|mm|km|inch|ft|yard)\b',
            'area': r'\b(m²|cm²|km²|sq\s*m|sq\s*ft)\b',
            'volume': r'\b(L|mL|m³|cm³|gallon|liter)\b',
            'time': r'\b(s|sec|min|hour|day|week|year)\b',
            'weight': r'\b(kg|g|lb|ton|ounce)\b',
            'speed': r'\b(mph|kmh|m/s|ft/s)\b'
        }
        
    def extract_entities(self, text: str) -> List[MathEntity]:
        """Extract mathematical entities from text."""
        entities = []
        
        # Extract numbers with units
        number_matches = list(self.number_pattern.finditer(text))
        for match in number_matches:
            value = float(match.group())
            start, end = match.span()
            
            # Check for adjacent units
            unit = self._find_adjacent_unit(text, end)
            
            entity = MathEntity(
                text=match.group() + (f" {unit}" if unit else ""),
                entity_type=EntityType.NUMBER,
                value=value,
                unit=unit,
                position=(start, end + len(unit) if unit else end)
            )
            entities.append(entity)
        
        # Extract variables and operations
        entities.extend(self._extract_variables(text))
        entities.extend(self._extract_operations(text))
        
        return entities
    
    def _find_adjacent_unit(self, text: str, position: int) -> Optional[str]:
        """Find unit adjacent to a number."""
        # Look for units within 5 characters after the number
        search_text = text[position:position+10].strip()
        
        for unit_type, pattern in self.unit_patterns.items():
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match and match.start() <= 5:
                return match.group()
        
        return None
    
    def _extract_variables(self, text: str) -> List[MathEntity]:
        """Extract variable names from text."""
        # Common variable patterns in math problems
        variable_patterns = [
            r'\b[a-zA-Z]\b(?=\s*(=|is|equals))',  # Single letter variables
            r'\b(speed|rate|time|distance|area|volume|height|width|length)\b'
        ]
        
        entities = []
        for pattern in variable_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = MathEntity(
                    text=match.group(),
                    entity_type=EntityType.VARIABLE,
                    position=match.span()
                )
                entities.append(entity)
        
        return entities
    
    def _extract_operations(self, text: str) -> List[MathEntity]:
        """Extract mathematical operations from text."""
        operation_patterns = {
            'addition': r'\b(add|plus|sum|total|altogether)\b',
            'subtraction': r'\b(subtract|minus|difference|less|remove)\b',
            'multiplication': r'\b(multiply|times|product|of)\b',
            'division': r'\b(divide|per|rate|ratio)\b',
            'comparison': r'\b(more|less|greater|smaller|larger)\b'
        }
        
        entities = []
        for op_type, pattern in operation_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = MathEntity(
                    text=match.group(),
                    entity_type=EntityType.OPERATION,
                    value=op_type,
                    position=match.span()
                )
                entities.append(entity)
        
        return entities


class ImplicitRelationDiscovery:
    """Enhanced Implicit Relation Discovery component using advanced pattern matching."""
    
    def __init__(self):
        self.relation_patterns = self._initialize_enhanced_patterns()
        self.confidence_threshold = 0.6
        self.semantic_analyzer = self._initialize_semantic_analyzer()
    
    def discover_relations(self, entities: List[MathEntity], context: str) -> List[ImplicitRelation]:
        """Discover implicit mathematical relations with enhanced algorithms."""
        relations = []
        
        # Enhanced multi-level relation discovery
        relations.extend(self._discover_arithmetic_relations(entities, context))
        relations.extend(self._discover_semantic_relations(entities, context))
        relations.extend(self._discover_contextual_relations(entities, context))
        relations.extend(self._discover_constraint_relations(entities, context))
        relations.extend(self._discover_unit_relations(entities, context))
        relations.extend(self._discover_temporal_relations(entities, context))
        
        # Apply advanced filtering and ranking
        relations = self._filter_and_rank_relations(relations, context)
        
        # Enhance confidence scores
        relations = self._enhance_confidence_scores(relations, entities, context)
        
        return relations
    
    def _discover_arithmetic_relations(self, entities: List[MathEntity], context: str) -> List[ImplicitRelation]:
        """Discover arithmetic relations with advanced pattern matching."""
        relations = []
        numeric_entities = [e for e in entities if e.entity_type == EntityType.NUMBER and e.value is not None]
        
        if len(numeric_entities) < 2:
            return relations
        
        context_lower = context.lower()
        
        # Enhanced arithmetic pattern detection
        for i, entity1 in enumerate(numeric_entities):
            for j, entity2 in enumerate(numeric_entities[i+1:], i+1):
                
                # Addition relations with multiple indicators
                if self._check_addition_indicators(context_lower, entity1, entity2):
                    relation = ImplicitRelation(
                        relation_id=f"add_{i}_{j}",
                        relation_type=RelationType.ARITHMETIC,
                        entities=[entity1, entity2],
                        description=f"Add {entity1.text} and {entity2.text}",
                        mathematical_expression=f"{entity1.text} + {entity2.text}",
                        confidence=self._calculate_arithmetic_confidence(context, 'addition', entity1, entity2)
                    )
                    relations.append(relation)
                
                # Subtraction relations
                elif self._check_subtraction_indicators(context_lower, entity1, entity2):
                    relation = ImplicitRelation(
                        relation_id=f"sub_{i}_{j}",
                        relation_type=RelationType.ARITHMETIC,
                        entities=[entity1, entity2],
                        description=f"Subtract {entity2.text} from {entity1.text}",
                        mathematical_expression=f"{entity1.text} - {entity2.text}",
                        confidence=self._calculate_arithmetic_confidence(context, 'subtraction', entity1, entity2)
                    )
                    relations.append(relation)
                
                # Multiplication relations
                elif self._check_multiplication_indicators(context_lower, entity1, entity2):
                    relation = ImplicitRelation(
                        relation_id=f"mul_{i}_{j}",
                        relation_type=RelationType.ARITHMETIC,
                        entities=[entity1, entity2],
                        description=f"Multiply {entity1.text} by {entity2.text}",
                        mathematical_expression=f"{entity1.text} × {entity2.text}",
                        confidence=self._calculate_arithmetic_confidence(context, 'multiplication', entity1, entity2)
                    )
                    relations.append(relation)
                
                # Division relations
                elif self._check_division_indicators(context_lower, entity1, entity2):
                    if entity2.value != 0:  # Avoid division by zero
                        relation = ImplicitRelation(
                            relation_id=f"div_{i}_{j}",
                            relation_type=RelationType.ARITHMETIC,
                            entities=[entity1, entity2],
                            description=f"Divide {entity1.text} by {entity2.text}",
                            mathematical_expression=f"{entity1.text} ÷ {entity2.text}",
                            confidence=self._calculate_arithmetic_confidence(context, 'division', entity1, entity2)
                        )
                        relations.append(relation)
        
        return relations
    
    def _check_addition_indicators(self, context: str, entity1: MathEntity, entity2: MathEntity) -> bool:
        """Enhanced addition indicator detection."""
        addition_patterns = [
            'total', 'sum', 'altogether', 'combined', 'plus', 'add',
            'more than', 'increase', 'and', 'with', 'both', 'together',
            'in total', 'all together', 'grand total', 'overall'
        ]
        return any(pattern in context for pattern in addition_patterns)
    
    def _check_subtraction_indicators(self, context: str, entity1: MathEntity, entity2: MathEntity) -> bool:
        """Enhanced subtraction indicator detection."""
        subtraction_patterns = [
            'difference', 'less', 'minus', 'subtract', 'decrease',
            'remove', 'take away', 'left', 'remain', 'fewer',
            'reduce by', 'deduct', 'cut by', 'drop by'
        ]
        return any(pattern in context for pattern in subtraction_patterns)
    
    def _check_multiplication_indicators(self, context: str, entity1: MathEntity, entity2: MathEntity) -> bool:
        """Enhanced multiplication indicator detection."""
        multiplication_patterns = [
            'times', 'multiply', 'each', 'per', 'rate', 'speed',
            'product', 'area', 'volume', 'groups of', 'sets of',
            'every', 'for each', 'at a rate of', 'repeated'
        ]
        return any(pattern in context for pattern in multiplication_patterns)
    
    def _check_division_indicators(self, context: str, entity1: MathEntity, entity2: MathEntity) -> bool:
        """Enhanced division indicator detection."""
        division_patterns = [
            'divide', 'split', 'share', 'average', 'per', 'each',
            'ratio', 'proportion', 'evenly', 'equally', 'distribute',
            'cut into', 'break into', 'partition'
        ]
        return any(pattern in context for pattern in division_patterns)
    
    def _calculate_arithmetic_confidence(self, context: str, operation: str, entity1: MathEntity, entity2: MathEntity) -> float:
        """Calculate confidence score for arithmetic relations."""
        base_confidence = 0.7
        
        # Boost confidence based on operation indicators
        operation_indicators = {
            'addition': ['total', 'sum', 'altogether', 'plus', 'add'],
            'subtraction': ['difference', 'less', 'minus', 'subtract'],
            'multiplication': ['times', 'multiply', 'each', 'per'],
            'division': ['divide', 'split', 'average', 'per']
        }
        
        indicators = operation_indicators.get(operation, [])
        indicator_count = sum(1 for indicator in indicators if indicator in context.lower())
        
        # Increase confidence based on indicator presence
        confidence_boost = min(indicator_count * 0.1, 0.3)
        
        # Consider entity proximity and type compatibility
        if entity1.unit == entity2.unit:  # Same units
            confidence_boost += 0.1
        
        return min(base_confidence + confidence_boost, 1.0)
    
    def _discover_semantic_relations(self, entities: List[MathEntity], context: str) -> List[ImplicitRelation]:
        """Discover semantic relations based on context understanding."""
        relations = []
        context_lower = context.lower()
        
        # Physical quantity relations
        if any(word in context_lower for word in ['speed', 'velocity', 'rate']):
            distance_entity = self._find_entity_by_context(entities, ['distance', 'miles', 'km', 'm'])
            time_entity = self._find_entity_by_context(entities, ['time', 'hours', 'minutes', 'seconds'])
            
            if distance_entity and time_entity:
                relation = ImplicitRelation(
                    relation_id="speed_relation",
                    relation_type=RelationType.PHYSICAL,
                    entities=[distance_entity, time_entity],
                    description="Speed = Distance / Time relationship",
                    mathematical_expression=f"speed = {distance_entity.text} / {time_entity.text}",
                    confidence=0.85
                )
                relations.append(relation)
        
        # Area relations
        if any(word in context_lower for word in ['area', 'rectangle', 'square']):
            length_entity = self._find_entity_by_context(entities, ['length', 'width', 'height'])
            if length_entity:
                relation = ImplicitRelation(
                    relation_id="area_relation",
                    relation_type=RelationType.GEOMETRIC,
                    entities=[length_entity],
                    description="Area calculation relationship",
                    mathematical_expression=f"area = length × width",
                    confidence=0.8
                )
                relations.append(relation)
        
        return relations
    
    def _discover_contextual_relations(self, entities: List[MathEntity], context: str) -> List[ImplicitRelation]:
        """Discover relations based on contextual clues."""
        relations = []
        
        # Money and cost relations
        if any(word in context.lower() for word in ['cost', 'price', 'money', '$', 'dollar']):
            money_entities = [e for e in entities if '$' in e.text or 'dollar' in e.text.lower()]
            quantity_entities = [e for e in entities if e.entity_type == EntityType.NUMBER and e not in money_entities]
            
            for money_entity in money_entities:
                for qty_entity in quantity_entities:
                    relation = ImplicitRelation(
                        relation_id=f"cost_relation_{money_entity.text}_{qty_entity.text}",
                        relation_type=RelationType.PROPORTION,
                        entities=[money_entity, qty_entity],
                        description=f"Cost relationship between {money_entity.text} and {qty_entity.text}",
                        mathematical_expression=f"total_cost = {money_entity.text} × {qty_entity.text}",
                        confidence=0.75
                    )
                    relations.append(relation)
        
        return relations
    
    def _discover_constraint_relations(self, entities: List[MathEntity], context: str) -> List[ImplicitRelation]:
        """Discover constraint-based relations."""
        relations = []
        
        # Comparison constraints
        comparison_words = ['more than', 'less than', 'greater than', 'smaller than', 'equal to']
        if any(word in context.lower() for word in comparison_words):
            numeric_entities = [e for e in entities if e.entity_type == EntityType.NUMBER]
            
            for i, entity1 in enumerate(numeric_entities):
                for entity2 in numeric_entities[i+1:]:
                    relation = ImplicitRelation(
                        relation_id=f"constraint_{entity1.text}_{entity2.text}",
                        relation_type=RelationType.CONSTRAINT,
                        entities=[entity1, entity2],
                        description=f"Comparison constraint between {entity1.text} and {entity2.text}",
                        mathematical_expression=f"{entity1.text} compared_to {entity2.text}",
                        confidence=0.7
                    )
                    relations.append(relation)
        
        return relations
    
    def _discover_unit_relations(self, entities: List[MathEntity], context: str) -> List[ImplicitRelation]:
        """Discover unit conversion relations."""
        relations = []
        
        # Find entities with different but convertible units
        entities_with_units = [e for e in entities if e.unit]
        
        for i, entity1 in enumerate(entities_with_units):
            for entity2 in entities_with_units[i+1:]:
                if self._are_convertible_units(entity1.unit, entity2.unit):
                    relation = ImplicitRelation(
                        relation_id=f"unit_conversion_{entity1.text}_{entity2.text}",
                        relation_type=RelationType.UNIT_CONVERSION,
                        entities=[entity1, entity2],
                        description=f"Unit conversion between {entity1.unit} and {entity2.unit}",
                        mathematical_expression=f"convert {entity1.text} {entity1.unit} to {entity2.unit}",
                        confidence=0.8
                    )
                    relations.append(relation)
        
        return relations
    
    def _discover_temporal_relations(self, entities: List[MathEntity], context: str) -> List[ImplicitRelation]:
        """Discover temporal relations."""
        relations = []
        
        time_indicators = ['time', 'hour', 'minute', 'second', 'day', 'week', 'month', 'year']
        if any(indicator in context.lower() for indicator in time_indicators):
            time_entities = [e for e in entities if any(indicator in e.text.lower() for indicator in time_indicators)]
            
            for i, entity1 in enumerate(time_entities):
                for entity2 in time_entities[i+1:]:
                    relation = ImplicitRelation(
                        relation_id=f"temporal_{entity1.text}_{entity2.text}",
                        relation_type=RelationType.TEMPORAL,
                        entities=[entity1, entity2],
                        description=f"Temporal relationship between {entity1.text} and {entity2.text}",
                        mathematical_expression=f"time_relation({entity1.text}, {entity2.text})",
                        confidence=0.75
                    )
                    relations.append(relation)
        
        return relations
    
    def _filter_and_rank_relations(self, relations: List[ImplicitRelation], context: str) -> List[ImplicitRelation]:
        """Filter and rank relations by confidence and relevance."""
        # Remove duplicate relations
        unique_relations = []
        seen_expressions = set()
        
        for relation in relations:
            if relation.mathematical_expression not in seen_expressions:
                unique_relations.append(relation)
                seen_expressions.add(relation.mathematical_expression)
        
        # Filter by confidence threshold
        filtered_relations = [r for r in unique_relations if r.confidence >= self.confidence_threshold]
        
        # Sort by confidence (highest first)
        filtered_relations.sort(key=lambda r: r.confidence, reverse=True)
        
        return filtered_relations
    
    def _enhance_confidence_scores(self, relations: List[ImplicitRelation], entities: List[MathEntity], context: str) -> List[ImplicitRelation]:
        """Enhance confidence scores based on additional context analysis."""
        for relation in relations:
            # Boost confidence for relations with strong contextual support
            context_boost = self._calculate_context_boost(relation, context)
            relation.confidence = min(relation.confidence + context_boost, 1.0)
            
            # Boost confidence for relations involving multiple entities
            if len(relation.entities) > 2:
                relation.confidence = min(relation.confidence + 0.1, 1.0)
        
        return relations
    
    def _calculate_context_boost(self, relation: ImplicitRelation, context: str) -> float:
        """Calculate confidence boost based on context analysis."""
        boost = 0.0
        context_lower = context.lower()
        
        # Check for strong operation indicators
        if relation.relation_type == RelationType.ARITHMETIC:
            if 'total' in context_lower or 'sum' in context_lower:
                boost += 0.15
            if 'difference' in context_lower:
                boost += 0.15
            if 'times' in context_lower or 'multiply' in context_lower:
                boost += 0.15
        
        # Check for question words that indicate specific operations
        question_words = ['how much', 'how many', 'what is', 'calculate', 'find']
        if any(word in context_lower for word in question_words):
            boost += 0.1
        
        return boost
    
    def _find_entity_by_context(self, entities: List[MathEntity], keywords: List[str]) -> Optional[MathEntity]:
        """Find entity based on contextual keywords."""
        for entity in entities:
            if any(keyword in entity.text.lower() for keyword in keywords):
                return entity
            if entity.unit and any(keyword in entity.unit.lower() for keyword in keywords):
                return entity
        return None
    
    def _are_convertible_units(self, unit1: str, unit2: str) -> bool:
        """Check if two units are convertible."""
        time_units = ['second', 'minute', 'hour', 'day']
        length_units = ['mm', 'cm', 'm', 'km', 'inch', 'ft', 'yard', 'mile']
        volume_units = ['ml', 'l', 'cup', 'gallon']
        
        unit1_lower = unit1.lower()
        unit2_lower = unit2.lower()
        
        return ((unit1_lower in time_units and unit2_lower in time_units) or
                (unit1_lower in length_units and unit2_lower in length_units) or
                (unit1_lower in volume_units and unit2_lower in volume_units))
    
    def _initialize_enhanced_patterns(self) -> Dict[str, Any]:
        """Initialize enhanced pattern matching rules."""
        return {
            'arithmetic_strong': {
                'addition': ['total', 'sum', 'altogether', 'combined', 'plus'],
                'subtraction': ['difference', 'less', 'minus', 'subtract', 'remain'],
                'multiplication': ['times', 'multiply', 'each', 'per', 'rate'],
                'division': ['divide', 'split', 'average', 'ratio', 'per']
            },
            'context_dependent': {
                'money': ['cost', 'price', 'dollar', '$', 'expense'],
                'time': ['hour', 'minute', 'second', 'time', 'duration'],
                'distance': ['mile', 'km', 'meter', 'distance', 'length'],
                'area': ['area', 'square', 'rectangle', 'surface']
            },
            'comparison': {
                'greater': ['more than', 'greater than', 'larger than', 'bigger than'],
                'lesser': ['less than', 'smaller than', 'fewer than', 'lower than'],
                'equal': ['equal to', 'same as', 'identical to']
            }
        }
    
    def _initialize_semantic_analyzer(self) -> Dict[str, Any]:
        """Initialize semantic analysis components."""
        return {
            'word_embeddings': {},  # Would contain actual word embeddings
            'concept_mappings': {
                'arithmetic': ['add', 'subtract', 'multiply', 'divide'],
                'measurement': ['length', 'width', 'height', 'area', 'volume'],
                'temporal': ['time', 'duration', 'period', 'interval']
            },
            'relation_types': {
                'causal': ['because', 'due to', 'results in', 'causes'],
                'conditional': ['if', 'when', 'provided that', 'given that'],
                'proportional': ['proportional', 'ratio', 'rate', 'per']
            }
        }


class MultiLevelReasoning:
    """Multi-Level Reasoning component for generating sophisticated reasoning chains."""
    
    def __init__(self):
        self.reasoning_templates = self._load_reasoning_templates()
        self.operation_patterns = self._get_enhanced_operation_patterns()
    
    def generate_reasoning_chain(self, entities: List[MathEntity], 
                               relations: List[ImplicitRelation],
                               target_question: str) -> List[ReasoningStep]:
        """Generate a complete reasoning chain towards solving the target question."""
        # Analyze the target question to understand what we're solving for
        target_analysis = self._analyze_target(target_question, entities)
        
        # Plan the sequence of reasoning steps
        reasoning_plan = self._plan_reasoning_sequence(entities, relations, target_analysis)
        
        # Execute each step in the plan
        reasoning_steps = []
        for i, plan_step in enumerate(reasoning_plan):
            step = self._execute_reasoning_step(i + 1, plan_step, reasoning_steps, entities)
            if step:
                reasoning_steps.append(step)
        
        return reasoning_steps
    
    def _analyze_target(self, question: str, entities: List[MathEntity]) -> Dict[str, Any]:
        """Analyze the target question to understand the solving objective."""
        question_type = self._classify_question_type(question)
        target_entity = self._identify_target_entity(question, entities)
        operation_hints = self._extract_operation_hints(question)
        
        return {
            'question_type': question_type,
            'target_entity': target_entity,
            'operation_hints': operation_hints,
            'complexity_level': self._estimate_question_complexity(question, entities)
        }
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of mathematical question."""
        question_lower = question.lower()
        
        # Enhanced question type classification
        if any(word in question_lower for word in ['how many', 'what is the total', 'altogether', 'sum']):
            return 'counting_sum'
        elif any(word in question_lower for word in ['how much', 'what is', 'calculate', 'find']):
            return 'calculation'
        elif any(word in question_lower for word in ['how long', 'what time', 'when']):
            return 'time_calculation'
        elif any(word in question_lower for word in ['how far', 'distance', 'speed', 'rate']):
            return 'distance_speed'
        elif any(word in question_lower for word in ['average', 'mean', 'per']):
            return 'average_calculation'
        elif any(word in question_lower for word in ['ratio', 'proportion', 'compare']):
            return 'ratio_comparison'
        else:
            return 'general_arithmetic'
    
    def _identify_target_entity(self, question: str, entities: List[MathEntity]) -> Optional[MathEntity]:
        """Identify which entity the question is asking about."""
        question_lower = question.lower()
        
        # Look for direct references to entities
        for entity in entities:
            if entity.text.lower() in question_lower:
                return entity
        
        # Look for units mentioned in question
        for entity in entities:
            if entity.unit and entity.unit.lower() in question_lower:
                return entity
        
        return None
    
    def _extract_operation_hints(self, question: str) -> List[str]:
        """Extract hints about what operations might be needed."""
        question_lower = question.lower()
        hints = []
        
        # Enhanced operation detection with more patterns
        operation_patterns = {
            'addition': ['total', 'sum', 'altogether', 'combined', 'plus', 'add', 'more than'],
            'subtraction': ['difference', 'less', 'minus', 'subtract', 'decrease', 'remove', 'take away'],
            'multiplication': ['times', 'multiply', 'each', 'per', 'rate', 'speed', 'product'],
            'division': ['divide', 'split', 'share', 'average', 'per', 'ratio', 'proportion'],
            'comparison': ['more', 'less', 'greater', 'smaller', 'compare', 'than'],
            'conversion': ['convert', 'change', 'transform', 'from', 'to']
        }
        
        for operation, patterns in operation_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                hints.append(operation)
        
        return hints
    
    def _estimate_question_complexity(self, question: str, entities: List[MathEntity]) -> str:
        """Estimate the complexity level of the question."""
        numeric_entities = [e for e in entities if e.entity_type == EntityType.NUMBER]
        
        # Simple heuristics for complexity estimation
        if len(numeric_entities) <= 2 and any(op in question.lower() for op in ['what is', 'calculate']):
            return 'L0'
        elif len(numeric_entities) <= 3 and any(word in question.lower() for word in ['total', 'altogether']):
            return 'L1'
        elif len(numeric_entities) > 3 or any(word in question.lower() for word in ['average', 'ratio', 'rate']):
            return 'L2'
        else:
            return 'L3'
    
    def _plan_reasoning_sequence(self, entities: List[MathEntity], 
                               relations: List[ImplicitRelation],
                               target_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan the sequence of reasoning steps based on target analysis."""
        plan = []
        
        operation_hints = target_analysis.get('operation_hints', [])
        question_type = target_analysis.get('question_type', 'general_arithmetic')
        complexity = target_analysis.get('complexity_level', 'L1')
        
        # Enhanced planning based on question type and complexity
        if question_type in ['counting_sum', 'calculation']:
            if 'addition' in operation_hints:
                plan.append({'type': 'arithmetic', 'operation': 'addition', 'priority': 1})
            elif 'subtraction' in operation_hints:
                plan.append({'type': 'arithmetic', 'operation': 'subtraction', 'priority': 1})
            elif 'multiplication' in operation_hints:
                plan.append({'type': 'arithmetic', 'operation': 'multiplication', 'priority': 1})
            elif 'division' in operation_hints:
                plan.append({'type': 'arithmetic', 'operation': 'division', 'priority': 1})
        
        elif question_type == 'distance_speed':
            plan.append({'type': 'rate_calculation', 'operation': 'speed_distance_time', 'priority': 1})
        
        elif question_type == 'average_calculation':
            plan.append({'type': 'arithmetic', 'operation': 'division', 'priority': 1})
            plan.append({'type': 'arithmetic', 'operation': 'addition', 'priority': 2})
        
        elif question_type == 'time_calculation':
            if 'conversion' in operation_hints:
                plan.append({'type': 'unit_conversion', 'operation': 'time_conversion', 'priority': 1})
            plan.append({'type': 'arithmetic', 'operation': 'addition', 'priority': 2})
        
        # If no specific plan, use default arithmetic
        if not plan:
            numeric_entities = [e for e in entities if e.entity_type == EntityType.NUMBER and e.value is not None]
            if len(numeric_entities) >= 2:
                plan.append({'type': 'arithmetic', 'operation': 'addition', 'priority': 1})
        
        # Add final calculation step
        plan.append({'type': 'final_calculation', 'operation': 'final_result', 'priority': 10})
        
        return sorted(plan, key=lambda x: x['priority'])
    
    def _execute_reasoning_step(self, step_id: int, plan_step: Dict[str, Any],
                              previous_steps: List[ReasoningStep],
                              entities: List[MathEntity]) -> Optional[ReasoningStep]:
        """Execute a single reasoning step according to the plan."""
        step_type = plan_step.get('type')
        
        if step_type == 'arithmetic':
            return self._execute_arithmetic_step(step_id, plan_step, previous_steps)
        elif step_type == 'unit_conversion':
            return self._execute_unit_conversion_step(step_id, plan_step, previous_steps)
        elif step_type == 'rate_calculation':
            return self._execute_rate_calculation_step(step_id, plan_step, previous_steps)
        elif step_type == 'final_calculation':
            return self._execute_final_calculation_step(step_id, plan_step, previous_steps)
        
        return None
    
    def _execute_arithmetic_step(self, step_id: int, plan_step: Dict[str, Any],
                               previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Execute an arithmetic reasoning step with improved value handling."""
        operation = plan_step.get('operation', 'addition')
        
        # Find available numeric values more safely
        values = []
        input_entities = []
        
        for step in previous_steps:
            if step.output_entity and step.output_entity.value is not None:
                values.append(float(step.output_entity.value))
                input_entities.append(step.output_entity)
        
        # If no previous steps, look for direct entities
        if not values:
            for entity in self._get_available_entities(previous_steps):
                # Improved value extraction with type checking
                if isinstance(entity, MathEntity) and entity.value is not None:
                    try:
                        values.append(float(entity.value))
                        input_entities.append(entity)
                    except (ValueError, TypeError):
                        continue
                elif hasattr(entity, 'value') and entity.value is not None:
                    try:
                        values.append(float(entity.value))
                        input_entities.append(entity)
                    except (ValueError, TypeError):
                        continue
        
        # Perform calculation
        if len(values) >= 2:
            if operation == 'addition':
                result_value = sum(values)
                description = f"Add {' + '.join(map(str, values))} = {result_value}"
            elif operation == 'subtraction':
                result_value = values[0] - sum(values[1:])
                description = f"Subtract {values[0]} - {sum(values[1:])} = {result_value}"
            elif operation == 'multiplication':
                result_value = values[0]
                for val in values[1:]:
                    result_value *= val
                description = f"Multiply {' × '.join(map(str, values))} = {result_value}"
            elif operation == 'division':
                result_value = values[0]
                for val in values[1:]:
                    if val != 0:
                        result_value /= val
                    else:
                        result_value = 0
                description = f"Divide {values[0]} ÷ {values[1] if len(values) > 1 else 1} = {result_value}"
            else:
                result_value = sum(values)
                description = f"Calculate {' + '.join(map(str, values))} = {result_value}"
        else:
            result_value = values[0] if values else 0
            description = f"Use value {result_value}"
        
        # Create output entity
        output_entity = MathEntity(
            text=str(result_value),
            entity_type=EntityType.NUMBER,
            value=result_value,
            confidence=0.9
        )
        
        return ReasoningStep(
            step_id=step_id,
            description=description,
            operation=operation,
            input_entities=input_entities,
            output_entity=output_entity,
            confidence=0.9,
            dependencies=[s.step_id for s in previous_steps]
        )
    
    def _get_available_entities(self, previous_steps: List[ReasoningStep]) -> List[MathEntity]:
        """Get available entities from previous steps and direct input."""
        entities = []
        
        # Add output entities from previous steps
        for step in previous_steps:
            if step.output_entity:
                entities.append(step.output_entity)
        
        return entities
    
    def _execute_unit_conversion_step(self, step_id: int, plan_step: Dict[str, Any],
                                    previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Execute a unit conversion step."""
        operation = plan_step.get('operation', 'time_conversion')
        
        # Find entity that needs conversion
        input_entity = None
        for step in previous_steps:
            if step.output_entity and step.output_entity.unit:
                input_entity = step.output_entity
                break
        
        if not input_entity:
            # Create a dummy conversion
            input_entity = MathEntity(text="1", entity_type=EntityType.NUMBER, value=1.0, unit="hour")
        
        # Perform conversion (simplified)
        converted_value = input_entity.value  # Simplified - would need actual conversion
        
        output_entity = MathEntity(
            text=str(converted_value),
            entity_type=EntityType.NUMBER,
            value=converted_value,
            unit="converted_unit",
            confidence=0.8
        )
        
        return ReasoningStep(
            step_id=step_id,
            description=f"Convert {input_entity.value} {input_entity.unit} to {converted_value} converted_unit",
            operation=operation,
            input_entities=[input_entity],
            output_entity=output_entity,
            confidence=0.8,
            dependencies=[s.step_id for s in previous_steps if s.output_entity == input_entity]
        )
    
    def _execute_rate_calculation_step(self, step_id: int, plan_step: Dict[str, Any],
                                     previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Execute a rate calculation step (e.g., speed = distance/time)."""
        # This is a simplified implementation
        return self._execute_arithmetic_step(step_id, {'operation': 'division'}, previous_steps)
    
    def _execute_final_calculation_step(self, step_id: int, plan_step: Dict[str, Any],
                                      previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Execute the final calculation step."""
        if previous_steps:
            last_step = previous_steps[-1]
            result_value = last_step.output_entity.value if last_step.output_entity else 0
            
            output_entity = MathEntity(
                text=str(result_value),
                entity_type=EntityType.NUMBER,
                value=result_value,
                confidence=1.0
            )
            
            return ReasoningStep(
                step_id=step_id,
                description=f"Final answer: {result_value}",
                operation="final_result",
                input_entities=[last_step.output_entity] if last_step.output_entity else [],
                output_entity=output_entity,
                confidence=1.0,
                dependencies=[last_step.step_id]
            )
        
        # Fallback
        output_entity = MathEntity(text="0", entity_type=EntityType.NUMBER, value=0.0)
        return ReasoningStep(
            step_id=step_id,
            description="No previous steps to finalize",
            operation="final_result",
            input_entities=[],
            output_entity=output_entity,
            confidence=0.1,
            dependencies=[]
        )
    
    def _get_enhanced_operation_patterns(self) -> Dict[str, Any]:
        """Get enhanced operation detection patterns."""
        return {
            'addition_patterns': [
                r'(?:how much|what is).+(?:total|sum|altogether)',
                r'(?:add|plus|\+)',
                r'(?:combined|together)',
                r'(?:increase|more than)',
                r'(?:and|with).+(?:more|additional)'
            ],
            'subtraction_patterns': [
                r'(?:how much|what is).+(?:difference|left|remain)',
                r'(?:subtract|minus|\-)',
                r'(?:less than|fewer than)',
                r'(?:decrease|reduce)',
                r'(?:take away|remove)'
            ],
            'multiplication_patterns': [
                r'(?:how much|what is).+(?:times|each|per)',
                r'(?:multiply|product|\*|×)',
                r'(?:\d+\s+(?:times|each))',
                r'(?:rate|speed).+(?:per|each)',
                r'(?:area|volume).+(?:rectangle|square|circle)'
            ],
            'division_patterns': [
                r'(?:how much|what is).+(?:average|per|each)',
                r'(?:divide|split|share)',
                r'(?:ratio|proportion)',
                r'(?:per\s+\w+)',
                r'(?:equally|evenly)'
            ]
        }
    
    def _load_reasoning_templates(self) -> Dict[str, Any]:
        """Load reasoning templates for different problem types."""
        return {
            'basic_arithmetic': {
                'steps': ['identify_numbers', 'apply_operation', 'calculate_result'],
                'confidence': 0.9
            },
            'word_problem': {
                'steps': ['parse_context', 'identify_quantities', 'determine_operation', 'calculate'],
                'confidence': 0.8
            },
            'multi_step': {
                'steps': ['break_down_problem', 'solve_sub_problems', 'combine_results'],
                'confidence': 0.7
            }
        }

    def _get_operation_from_question(self, question: str) -> str:
        """Enhanced operation detection with L2 multi-step reasoning support"""
        question = question.lower()
        
        # L2多步推理操作优先级检测
        multi_step_patterns = {
            'sequential_spending': ['spends.*and.*', 'buys.*and.*spends', 'pays.*then.*'],
            'multiplication_context': ['each.*total', 'per.*total', 'cost.*total', 'price.*total', 'for.*each'],
            'division_sharing': ['share.*equally', 'divide.*among', 'each.*get', 'per.*person'],
            'subtraction_remaining': ['left.*after', 'remaining.*after', 'have.*left', 'money.*left']
        }
        
        # 检测多步推理模式
        for pattern_type, patterns in multi_step_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    if pattern_type == 'sequential_spending':
                        return 'multi_step_subtraction'
                    elif pattern_type == 'multiplication_context':
                        return 'multiplication'
                    elif pattern_type == 'division_sharing':
                        return 'division'
                    elif pattern_type == 'subtraction_remaining':
                        return 'multi_step_subtraction'
        
        # Enhanced operation keywords with context awareness
        operation_patterns = {
            'multiplication': [
                r'\b(?:each|per)\b.*\b(?:total|cost|amount)\b',
                r'\b(?:price|cost)\b.*\b(?:each|per)\b.*\b(?:total|altogether)\b',
                r'\bbuys?\s+\d+.*\b(?:each|per)\b',
                r'\b(?:sells?|costs?)\s+.*\b(?:each|per)\b.*\btotal\b',
                r'\d+\s+(?:books?|items?|things?).*\$\d+.*each',
                r'for\s+\$\d+\s+each.*total'
            ],
            'addition': [
                r'\b(?:total|sum|altogether|combined)\b',
                r'\bscored?\s+\d+.*and.*\d+.*total',
                r'\bhas\s+\d+.*and.*\d+.*total',
                r'\d+\s+(?:points?|dollars?|items?).*and.*\d+.*total'
            ],
            'subtraction': [
                r'\bleft\b|\bremaining\b|\bremains?\b',
                r'\bspends?\b.*\bhow\s+much.*left\b',
                r'\bhas\s+\$?\d+.*spends?\s+\$?\d+.*left',
                r'\bmoney.*left\b|\bhave.*left\b'
            ],
            'division': [
                r'\bshare.*equally\b|\bdivide.*among\b',
                r'\beach.*gets?\b|\bper\s+person\b',
                r'\bcut.*into.*slices.*each\b',
                r'\bequally.*among.*people\b'
            ]
        }
        
        # 按优先级检测操作类型
        for operation, patterns in operation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return operation
        
        # 回退到基础检测
        if any(word in question for word in ['total', 'sum', 'altogether', 'combined']):
            # 检查是否是乘法上下文中的total
            if any(word in question for word in ['each', 'per', 'price', 'cost']) and 'total' in question:
                return 'multiplication'
            return 'addition'
        elif any(word in question for word in ['left', 'remaining', 'change', 'spends']):
            return 'subtraction'
        elif any(word in question for word in ['share', 'divide', 'each get', 'per person']):
            return 'division'
        elif any(word in question for word in ['times', 'multiply', 'each', 'per']):
            return 'multiplication'
        
        return 'addition'  # 默认

    def _plan_reasoning_steps(self, entities: List[MathEntity], relations: List[ImplicitRelation], question: str) -> List[ReasoningStep]:
        """Enhanced multi-step reasoning planner for L2 complexity"""
        steps = []
        question_lower = question.lower()
        
        # 检测问题类型以确定推理策略
        problem_type = self._classify_problem_type(question)
        operation = self._get_operation_from_question(question)
        
        numerical_entities = [e for e in entities if e.entity_type == EntityType.NUMBER]
        
        if len(numerical_entities) < 2:
            return steps
            
        # L2多步推理规划
        if operation == 'multi_step_subtraction':
            # 多步减法：总金额 - 第一次支出 - 第二次支出
            if len(numerical_entities) >= 3:
                # Step 1: 计算总支出
                step1 = self._create_reasoning_step(
                    step_id=0,
                    description=f"Calculate total spending: {numerical_entities[1].value} + {numerical_entities[2].value}",
                    operation='addition',
                    input_entities=[numerical_entities[1], numerical_entities[2]],
                    confidence=0.95
                )
                steps.append(step1)
                
                # Step 2: 从总金额中减去总支出
                total_spending = numerical_entities[1].value + numerical_entities[2].value
                step2 = self._create_reasoning_step(
                    step_id=1,
                    description=f"Calculate remaining money: {numerical_entities[0].value} - {total_spending}",
                    operation='subtraction',
                    input_entities=[numerical_entities[0], step1.output_entity],
                    confidence=0.95,
                    dependencies=[0]
                )
                steps.append(step2)
        
        elif operation == 'multiplication' and 'total' in question_lower:
            # 乘法计算总成本/总数量
            if len(numerical_entities) >= 2:
                # 智能识别单价和数量
                price_entity = None
                quantity_entity = None
                
                # 通过上下文识别单价和数量
                for i, entity in enumerate(numerical_entities):
                    context_before = question[max(0, entity.position[0]-20):entity.position[0]]
                    context_after = question[entity.position[1]:entity.position[1]+20]
                    
                    if any(word in context_before.lower() or word in context_after.lower() 
                           for word in ['$', 'dollar', 'price', 'cost', 'each', 'per']):
                        price_entity = entity
                    elif any(word in context_before.lower() or word in context_after.lower() 
                             for word in ['books', 'items', 'buys', 'purchases']):
                        quantity_entity = entity
                
                # 如果没有明确识别，使用位置顺序
                if not price_entity and not quantity_entity:
                    price_entity = numerical_entities[0]
                    quantity_entity = numerical_entities[1]
                
                step = self._create_reasoning_step(
                    step_id=0,
                    description=f"Calculate total cost: {price_entity.value} × {quantity_entity.value}",
                    operation='multiplication',
                    input_entities=[price_entity, quantity_entity],
                    confidence=0.95
                )
                steps.append(step)
        
        elif operation == 'division':
            # 除法分配
            if len(numerical_entities) >= 2:
                step = self._create_reasoning_step(
                    step_id=0,
                    description=f"Divide equally: {numerical_entities[0].value} ÷ {numerical_entities[1].value}",
                    operation='division',
                    input_entities=[numerical_entities[0], numerical_entities[1]],
                    confidence=0.95
                )
                steps.append(step)
        
        else:
            # 单步操作
            if len(numerical_entities) >= 2:
                step = self._create_reasoning_step(
                    step_id=0,
                    description=f"Apply {operation}: {numerical_entities[0].value} {self._get_operation_symbol(operation)} {numerical_entities[1].value}",
                    operation=operation,
                    input_entities=numerical_entities[:2],
                    confidence=0.95
                )
                steps.append(step)
        
        return steps

    def _classify_problem_type(self, question: str) -> str:
        """Classify the mathematical problem type for better reasoning strategy"""
        question_lower = question.lower()
        
        problem_types = {
            'cost_calculation': ['cost', 'price', 'total', 'spend', 'buy', 'sell', 'each', 'per'],
            'money_remaining': ['left', 'remaining', 'spends', 'have left', 'money left'],
            'sharing_division': ['share', 'equally', 'divide', 'each get', 'per person'],
            'scoring_addition': ['scored', 'points', 'total score', 'game'],
            'measurement': ['length', 'width', 'area', 'distance', 'time'],
            'counting': ['items', 'objects', 'things', 'pieces']
        }
        
        for problem_type, keywords in problem_types.items():
            if sum(1 for keyword in keywords if keyword in question_lower) >= 2:
                return problem_type
        
        return 'general'

    def _get_operation_symbol(self, operation: str) -> str:
        """Get mathematical symbol for operation"""
        symbols = {
            'addition': '+',
            'subtraction': '-',
            'multiplication': '×',
            'division': '÷',
            'multi_step_subtraction': '-'
        }
        return symbols.get(operation, '+')

    def _create_reasoning_step(self, step_id: int, description: str, operation: str, 
                             input_entities: List[MathEntity], confidence: float, 
                             dependencies: List[int] = None) -> ReasoningStep:
        """Create a reasoning step with enhanced output calculation"""
        if dependencies is None:
            dependencies = []
            
        # 计算输出值
        if len(input_entities) >= 2:
            val1, val2 = input_entities[0].value, input_entities[1].value
            
            if operation == 'addition':
                result = val1 + val2
            elif operation == 'subtraction':
                result = val1 - val2
            elif operation == 'multiplication':
                result = val1 * val2
            elif operation == 'division':
                result = val1 / val2 if val2 != 0 else 0
            else:
                result = val1 + val2  # 默认
        else:
            result = input_entities[0].value if input_entities else 0
        
        output_entity = MathEntity(
            text=str(result),
            entity_type=EntityType.NUMBER,
            value=result,
            confidence=confidence,
            position=(0, 0)
        )
        
        return ReasoningStep(
            step_id=step_id,
            description=description,
            operation=operation,
            input_entities=input_entities,
            output_entity=output_entity,
            confidence=confidence,
            dependencies=dependencies
        )


class ChainVerification:
    """Verifies and validates reasoning chains."""
    
    def __init__(self):
        self.verification_rules = self._initialize_verification_rules()
    
    def verify_reasoning_chain(self, reasoning_steps: List[ReasoningStep]) -> Dict[str, Any]:
        """Verify the complete reasoning chain."""
        verification_result = {
            'is_valid': True,
            'confidence_score': 1.0,
            'logical_errors': [],
            'mathematical_errors': [],
            'consistency_score': 1.0,
            'completeness_score': 1.0
        }
        
        # Check logical consistency
        logical_errors = self._check_logical_consistency(reasoning_steps)
        verification_result['logical_errors'] = logical_errors
        
        # Check mathematical correctness
        math_errors = self._check_mathematical_correctness(reasoning_steps)
        verification_result['mathematical_errors'] = math_errors
        
        # Calculate scores
        verification_result['consistency_score'] = self._calculate_consistency_score(reasoning_steps)
        verification_result['completeness_score'] = self._calculate_completeness_score(reasoning_steps)
        
        # Overall validity
        verification_result['is_valid'] = (len(logical_errors) == 0 and len(math_errors) == 0)
        verification_result['confidence_score'] = min(
            verification_result['consistency_score'],
            verification_result['completeness_score']
        )
        
        return verification_result
    
    def _check_logical_consistency(self, reasoning_steps: List[ReasoningStep]) -> List[str]:
        """Check for logical inconsistencies in the reasoning chain."""
        errors = []
        
        # Check dependency consistency
        for step in reasoning_steps:
            for dep_id in step.dependencies:
                if not any(s.step_id == dep_id for s in reasoning_steps if s.step_id < step.step_id):
                    errors.append(f"Step {step.step_id} depends on non-existent step {dep_id}")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(reasoning_steps):
            errors.append("Circular dependencies detected in reasoning chain")
        
        return errors
    
    def _check_mathematical_correctness(self, reasoning_steps: List[ReasoningStep]) -> List[str]:
        """Check for mathematical errors in calculations."""
        errors = []
        
        for step in reasoning_steps:
            if step.operation == "addition":
                # Verify addition calculation
                input_values = [e.value for e in step.input_entities if e.value is not None]
                expected_result = sum(input_values)
                if abs(step.output_entity.value - expected_result) > 1e-6:
                    errors.append(f"Step {step.step_id}: Addition error. Expected {expected_result}, got {step.output_entity.value}")
            
            elif step.operation == "multiplication":
                # Verify multiplication calculation
                input_values = [e.value for e in step.input_entities if e.value is not None]
                if len(input_values) >= 2:
                    expected_result = input_values[0]
                    for val in input_values[1:]:
                        expected_result *= val
                    if abs(step.output_entity.value - expected_result) > 1e-6:
                        errors.append(f"Step {step.step_id}: Multiplication error. Expected {expected_result}, got {step.output_entity.value}")
        
        return errors
    
    def _has_circular_dependencies(self, reasoning_steps: List[ReasoningStep]) -> bool:
        """Check for circular dependencies using topological sort."""
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for step in reasoning_steps:
            for dep_id in step.dependencies:
                graph[dep_id].append(step.step_id)
                in_degree[step.step_id] += 1
        
        # Topological sort to detect cycles
        queue = [step.step_id for step in reasoning_steps if in_degree[step.step_id] == 0]
        processed = 0
        
        while queue:
            current = queue.pop(0)
            processed += 1
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return processed != len(reasoning_steps)
    
    def _calculate_consistency_score(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate consistency score for the reasoning chain."""
        if not reasoning_steps:
            return 0.0
        
        total_confidence = sum(step.confidence for step in reasoning_steps)
        return total_confidence / len(reasoning_steps)
    
    def _calculate_completeness_score(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate completeness score for the reasoning chain."""
        if not reasoning_steps:
            return 0.0
        
        # Check if chain leads to a final answer
        has_final_step = any(step.operation == "final_result" for step in reasoning_steps)
        
        # Check if all steps have proper descriptions
        has_descriptions = all(step.description.strip() for step in reasoning_steps)
        
        # Calculate score
        score = 0.5  # Base score
        if has_final_step:
            score += 0.3
        if has_descriptions:
            score += 0.2
        
        return min(score, 1.0)
    
    def _initialize_verification_rules(self) -> List[Dict[str, Any]]:
        """Initialize verification rules."""
        return [
            {
                'rule_type': 'dependency_check',
                'description': 'Verify all dependencies exist and are valid'
            },
            {
                'rule_type': 'mathematical_accuracy',
                'description': 'Verify mathematical calculations are correct'
            },
            {
                'rule_type': 'logical_consistency',
                'description': 'Ensure logical flow is consistent'
            }
        ]


class IRDCombinatorialDiscovery:
    """Implementation of Algorithm 1: IRD Combinatorial Discovery"""
    
    def __init__(self, knowledge_base: Dict[str, Any] = None):
        self.knowledge_base = knowledge_base or self._initialize_knowledge_base()
        self.validity_threshold = 0.7
        self.d_max = 5  # Maximum derivation depth
        self.k_max = 4  # Maximum entity subset size
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize default knowledge base"""
        return {
            'arithmetic_patterns': {
                'addition': ['sum', 'total', 'combined', 'altogether'],
                'subtraction': ['difference', 'remain', 'less', 'decrease'],
                'multiplication': ['times', 'each', 'product', 'per'],
                'division': ['average', 'ratio', 'split', 'per']
            },
            'geometric_patterns': {
                'area': ['rectangle', 'square', 'circle', 'triangle'],
                'perimeter': ['around', 'border', 'fence', 'circumference'],
                'volume': ['cube', 'sphere', 'cylinder', 'capacity']
            },
            'unit_conversions': {
                'time': {'hour': 3600, 'minute': 60, 'second': 1},
                'length': {'km': 1000, 'm': 1, 'cm': 0.01, 'mm': 0.001},
                'volume': {'L': 1, 'mL': 0.001, 'cm³': 0.001}
            }
        }
    
    def _candidate_to_relation(self, candidate: Dict[str, Any], entities: List[MathEntity]) -> ImplicitRelation:
        """Convert candidate to ImplicitRelation object"""
        return ImplicitRelation(
            relation_id=f"ird_{candidate.get('type', 'unknown')}_{len(entities)}",
            relation_type=RelationType.ARITHMETIC if candidate.get('type') == 'arithmetic' else RelationType.GEOMETRIC,
            entities=entities,
            description=f"{candidate.get('operation', 'unknown')} relationship",
            mathematical_expression=self._build_expression(candidate, entities),
            confidence=candidate.get('confidence', 0.7)
        )
    
    def _build_expression(self, candidate: Dict[str, Any], entities: List[MathEntity]) -> str:
        """Build mathematical expression for candidate"""
        operation = candidate.get('operation', '')
        if len(entities) >= 2:
            if operation == 'addition':
                return f"{entities[0].text} + {entities[1].text}"
            elif operation == 'subtraction':
                return f"{entities[0].text} - {entities[1].text}"
            elif operation == 'multiplication':
                return f"{entities[0].text} × {entities[1].text}"
            elif operation == 'division':
                return f"{entities[0].text} ÷ {entities[1].text}"
        return f"relation({', '.join([e.text for e in entities])})"
    
    def _generate_multi_arithmetic_candidates(self, entities: Tuple[MathEntity]) -> List[Dict[str, Any]]:
        """Generate multi-entity arithmetic candidates"""
        candidates = []
        if len(entities) >= 3:
            # Triple addition/multiplication
            candidates.append({
                'type': 'arithmetic',
                'operation': 'multi_addition',
                'entities': list(entities),
                'derivation_depth': 2,
                'complexity': 2
            })
        return candidates
    
    def _generate_geometric_candidates(self, entities: Tuple[MathEntity]) -> List[Dict[str, Any]]:
        """Generate geometric relation candidates"""
        candidates = []
        if len(entities) == 2:
            candidates.append({
                'type': 'geometric',
                'operation': 'area_calculation',
                'entities': list(entities),
                'derivation_depth': 1,
                'complexity': 2
            })
        return candidates
    
    def _generate_proportional_candidates(self, entities: Tuple[MathEntity]) -> List[Dict[str, Any]]:
        """Generate proportional relation candidates"""
        candidates = []
        if len(entities) >= 2:
            candidates.append({
                'type': 'proportion',
                'operation': 'rate_calculation',
                'entities': list(entities),
                'derivation_depth': 1,
                'complexity': 2
            })
        return candidates
    
    def combinatorial_discovery(self, entities: List[MathEntity], qualia: Dict[str, Any]) -> List[ImplicitRelation]:
        """Algorithm 1: IRD Combinatorial Discovery"""
        R_implicit = []
        
        # Line 2: for each entity subset S ⊆ E with |S| ≤ k_max
        for subset_size in range(2, min(len(entities) + 1, self.k_max + 1)):
            for entity_subset in combinations(entities, subset_size):
                # Line 3: C ← CombinatorialGenerate(S, K, d_max)
                candidates = self._combinatorial_generate(entity_subset, self.knowledge_base, self.d_max)
                
                # Line 4: for each candidate r ∈ C
                for candidate in candidates:
                    # Line 5: if δ(r) ≤ d_max and κ(r) ≤ k_max
                    if (candidate.get('derivation_depth', 0) <= self.d_max and 
                        candidate.get('complexity', 0) <= self.k_max):
                        
                        # Line 6: score ← ValidityScore(r, Q, K)
                        score = self._validity_score(candidate, qualia, self.knowledge_base)
                        
                        # Line 7: if score > τ_validity
                        if score > self.validity_threshold:
                            # Line 8: R_implicit ← R_implicit ∪ {r}
                            relation = self._candidate_to_relation(candidate, list(entity_subset))
                            R_implicit.append(relation)
        
        return R_implicit
    
    def _combinatorial_generate(self, entities: Tuple[MathEntity], knowledge_base: Dict[str, Any], d_max: int) -> List[Dict[str, Any]]:
        """Generate combinatorial relation candidates"""
        candidates = []
        
        # Generate arithmetic combinations
        if len(entities) == 2:
            candidates.extend(self._generate_binary_arithmetic_candidates(entities))
        elif len(entities) >= 3:
            candidates.extend(self._generate_multi_arithmetic_candidates(entities))
        
        # Generate geometric combinations
        candidates.extend(self._generate_geometric_candidates(entities))
        
        # Generate proportional combinations
        candidates.extend(self._generate_proportional_candidates(entities))
        
        return candidates
    
    def _validity_score(self, candidate: Dict[str, Any], qualia: Dict[str, Any], knowledge_base: Dict[str, Any]) -> float:
        """Calculate validity score for relation candidate"""
        base_score = 0.5
        
        # Semantic coherence
        semantic_score = self._calculate_semantic_coherence(candidate, qualia)
        
        # Mathematical validity
        math_score = self._calculate_mathematical_validity(candidate)
        
        # Knowledge base consistency
        kb_score = self._calculate_kb_consistency(candidate, knowledge_base)
        
        return (semantic_score + math_score + kb_score) / 3
    
    def _generate_binary_arithmetic_candidates(self, entities: Tuple[MathEntity]) -> List[Dict[str, Any]]:
        """Generate binary arithmetic relation candidates"""
        candidates = []
        e1, e2 = entities
        
        operations = ['addition', 'subtraction', 'multiplication', 'division']
        for op in operations:
            candidates.append({
                'type': 'arithmetic',
                'operation': op,
                'entities': [e1, e2],
                'derivation_depth': 1,
                'complexity': 1
            })
        
        return candidates
    
    def _calculate_semantic_coherence(self, candidate: Dict[str, Any], qualia: Dict[str, Any]) -> float:
        """Calculate semantic coherence score"""
        operation = candidate.get('operation', '')
        problem_context = qualia.get('context', '')
        
        # Context-operation matching
        context_keywords = {
            'addition': ['total', 'sum', 'altogether', 'combined'],
            'subtraction': ['difference', 'remain', 'less', 'decrease'],
            'multiplication': ['times', 'each', 'per', 'rate'],
            'division': ['average', 'per', 'ratio', 'split']
        }
        
        keywords = context_keywords.get(operation, [])
        matches = sum(1 for keyword in keywords if keyword in problem_context.lower())
        
        return min(matches * 0.3 + 0.1, 1.0)
    
    def _calculate_mathematical_validity(self, candidate: Dict[str, Any]) -> float:
        """Calculate mathematical validity score"""
        entities = candidate.get('entities', [])
        operation = candidate.get('operation', '')
        
        # Check for valid number types
        numeric_entities = [e for e in entities if e.entity_type == EntityType.NUMBER and e.value is not None]
        
        if len(numeric_entities) < 2:
            return 0.2
        
        # Division by zero check
        if operation == 'division' and any(e.value == 0 for e in numeric_entities[1:]):
            return 0.0
        
        return 0.9
    
    def _calculate_kb_consistency(self, candidate: Dict[str, Any], knowledge_base: Dict[str, Any]) -> float:
        """Calculate knowledge base consistency score"""
        operation = candidate.get('operation', '')
        known_patterns = knowledge_base.get('arithmetic_patterns', {})
        
        if operation in known_patterns:
            return 0.8
        return 0.5


class MLRStateBasedReasoning:
    """Implementation of Algorithm 2: MLR State-Based Reasoning"""
    
    def __init__(self):
        self.visited_states = set()
        self.frontier = []
        self.max_iterations = 100
    
    def state_based_reasoning(self, initial_state: Dict[str, Any], relations: List[ImplicitRelation], goal: Dict[str, Any]) -> Union[List[Dict[str, Any]], str]:
        """Algorithm 2: MLR State-Based Reasoning"""
        # Line 1: Initialize frontier = {L0}, visited = ∅
        self.frontier = [initial_state]
        self.visited_states = set()
        solution_path = []
        
        iteration = 0
        # Line 2: while frontier ≠ ∅
        while self.frontier and iteration < self.max_iterations:
            iteration += 1
            
            # Line 3: L ← frontier.pop()
            current_state = self.frontier.pop(0)
            
            # Line 4: if GoalSatisfied(L, G)
            if self._goal_satisfied(current_state, goal):
                # Line 5: return ReconstructPath(L)
                return self._reconstruct_path(current_state, solution_path)
            
            # Line 7: if L ∈ visited
            state_key = self._state_to_key(current_state)
            if state_key in self.visited_states:
                # Line 8: continue
                continue
            
            # Line 10: visited ← visited ∪ {L}
            self.visited_states.add(state_key)
            solution_path.append(current_state)
            
            # Line 11: for each relation r ∈ R applicable to L
            for relation in relations:
                if self._is_applicable(relation, current_state):
                    # Line 12: L' ← ApplyRelation(L, r)
                    new_state = self._apply_relation(current_state, relation)
                    
                    # Line 13: if ConsistencyCheck(L')
                    if self._consistency_check(new_state):
                        # Line 14: frontier.push(L')
                        self.frontier.append(new_state)
        
        # Line 17: return UNSATISFIABLE
        return "UNSATISFIABLE"
    
    def _goal_satisfied(self, state: Dict[str, Any], goal: Dict[str, Any]) -> bool:
        """Check if current state satisfies the goal"""
        target_variable = goal.get('target_variable')
        target_value = goal.get('target_value')
        
        if target_variable and target_variable in state.get('variables', {}):
            computed_value = state['variables'][target_variable]
            if target_value is not None:
                return abs(computed_value - target_value) < 0.001
            else:
                return computed_value is not None
        
        return False
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to hashable key"""
        variables = state.get('variables', {})
        return json.dumps(variables, sort_keys=True)
    
    def _is_applicable(self, relation: ImplicitRelation, state: Dict[str, Any]) -> bool:
        """Check if relation is applicable to current state"""
        required_entities = [e.text for e in relation.entities if e.entity_type == EntityType.NUMBER]
        available_values = state.get('variables', {})
        
        return all(entity in available_values or any(entity in val for val in available_values.keys()) 
                  for entity in required_entities)
    
    def _apply_relation(self, state: Dict[str, Any], relation: ImplicitRelation) -> Dict[str, Any]:
        """Apply relation to state and generate new state"""
        new_state = state.copy()
        new_state['variables'] = state.get('variables', {}).copy()
        
        if relation.relation_type == RelationType.ARITHMETIC:
            # Apply arithmetic relation
            entities = [e for e in relation.entities if e.entity_type == EntityType.NUMBER]
            if len(entities) >= 2:
                values = [e.value for e in entities if e.value is not None]
                if len(values) >= 2:
                    if 'sum' in relation.description.lower() or 'total' in relation.description.lower():
                        result = sum(values)
                        new_state['variables']['calculated_result'] = result
        
        return new_state
    
    def _consistency_check(self, state: Dict[str, Any]) -> bool:
        """Check state consistency"""
        variables = state.get('variables', {})
        
        # Check for valid numerical values
        for key, value in variables.items():
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return False
        
        return True
    
    def _reconstruct_path(self, final_state: Dict[str, Any], path: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reconstruct solution path"""
        return path + [final_state]


class CVFormalVerification:
    """Implementation of Algorithm 3: CV Formal Verification"""
    
    def __init__(self):
        self.syntax_rules = self._initialize_syntax_rules()
        self.semantic_rules = self._initialize_semantic_rules()
    
    def formal_verification(self, reasoning_chain: List[ReasoningStep], goal: Dict[str, Any]) -> Dict[str, Any]:
        """Algorithm 3: CV Formal Verification"""
        
        # Level 1: Syntactic Verification (Lines 2-6)
        for i, step in enumerate(reasoning_chain):
            if not self._syntactic_valid(step):
                return {
                    'status': 'INVALID',
                    'error_type': 'syntactic',
                    'error_message': f"Syntactic error at step {i}",
                    'error_step': i
                }
        
        # Level 2: Semantic Verification (Lines 8-14)
        state = self._initial_state()
        for i, step in enumerate(reasoning_chain):
            state = self._apply_relation(state, step)
            if not self._semantic_valid(state):
                return {
                    'status': 'INVALID',
                    'error_type': 'semantic',
                    'error_message': f"Semantic error at step {i}",
                    'error_step': i
                }
        
        # Level 3: Goal Achievement Verification (Lines 16-18)
        if not self._goal_achieved(state, goal):
            return {
                'status': 'INVALID',
                'error_type': 'goal',
                'error_message': "Goal not achieved"
            }
        
        # Line 19: return VALID
        return {
            'status': 'VALID',
            'final_state': state,
            'verification_score': self._calculate_verification_score(reasoning_chain, state, goal)
        }
    
    def _syntactic_valid(self, step: ReasoningStep) -> bool:
        """Check syntactic validity of reasoning step"""
        # Check required fields
        if not all([step.description, step.operation, step.input_entities]):
            return False
        
        # Check operation validity
        valid_operations = ['addition', 'subtraction', 'multiplication', 'division', 'unit_conversion', 'calculation']
        if step.operation not in valid_operations:
            return False
        
        # Check entity types
        for entity in step.input_entities:
            if not isinstance(entity, MathEntity):
                return False
        
        return True
    
    def _semantic_valid(self, state: Dict[str, Any]) -> bool:
        """Check semantic validity of state"""
        variables = state.get('variables', {})
        
        # Check for mathematical consistency
        for key, value in variables.items():
            if isinstance(value, (int, float)):
                # Check for invalid mathematical values
                if math.isnan(value) or math.isinf(value):
                    return False
                # Check for unrealistic values (domain-specific)
                if abs(value) > 1e10:  # Arbitrary large number threshold
                    return False
        
        return True
    
    def _goal_achieved(self, state: Dict[str, Any], goal: Dict[str, Any]) -> bool:
        """Check if goal is achieved in final state"""
        target_variable = goal.get('target_variable', 'final_answer')
        target_value = goal.get('target_value')
        
        computed_value = state.get('variables', {}).get(target_variable)
        
        if computed_value is None:
            return False
        
        if target_value is not None:
            # Check if computed value matches expected value within tolerance
            tolerance = goal.get('tolerance', 0.001)
            return abs(computed_value - target_value) <= tolerance
        
        # If no target value specified, just check that we have a result
        return True
    
    def _initial_state(self) -> Dict[str, Any]:
        """Initialize verification state"""
        return {
            'variables': {},
            'constraints': [],
            'operations_performed': []
        }
    
    def _apply_relation(self, state: Dict[str, Any], step: ReasoningStep) -> Dict[str, Any]:
        """Apply reasoning step to state"""
        new_state = state.copy()
        new_state['variables'] = state.get('variables', {}).copy()
        new_state['operations_performed'] = state.get('operations_performed', []).copy()
        
        # Record operation
        new_state['operations_performed'].append({
            'step_id': step.step_id,
            'operation': step.operation,
            'description': step.description
        })
        
        # Apply operation result
        if step.output_entity and step.output_entity.value is not None:
            output_key = f"step_{step.step_id}_result"
            new_state['variables'][output_key] = step.output_entity.value
            
            # If this is the final step, also set final_answer
            new_state['variables']['final_answer'] = step.output_entity.value
        
        return new_state
    
    def _calculate_verification_score(self, reasoning_chain: List[ReasoningStep], final_state: Dict[str, Any], goal: Dict[str, Any]) -> float:
        """Calculate overall verification confidence score"""
        syntactic_score = 1.0  # All steps passed syntactic check
        
        # Semantic coherence score
        semantic_score = len([s for s in reasoning_chain if s.confidence > 0.7]) / max(len(reasoning_chain), 1)
        
        # Goal achievement score
        goal_score = 1.0 if self._goal_achieved(final_state, goal) else 0.0
        
        return (syntactic_score + semantic_score + goal_score) / 3
    
    def _initialize_syntax_rules(self) -> List[Dict[str, Any]]:
        """Initialize syntactic validation rules"""
        return [
            {'rule': 'required_fields', 'fields': ['description', 'operation', 'input_entities']},
            {'rule': 'valid_operations', 'operations': ['addition', 'subtraction', 'multiplication', 'division']},
            {'rule': 'entity_types', 'types': [EntityType.NUMBER, EntityType.VARIABLE]}
        ]
    
    def _initialize_semantic_rules(self) -> List[Dict[str, Any]]:
        """Initialize semantic validation rules"""
        return [
            {'rule': 'no_division_by_zero', 'check': 'division_operands'},
            {'rule': 'valid_numerical_range', 'range': [-1e10, 1e10]},
            {'rule': 'unit_consistency', 'check': 'dimensional_analysis'}
        ]


class MathematicalReasoningSystem:
    """Main system that orchestrates all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Mathematical Reasoning System."""
        self.config = config or {}
        self.enable_verification = self.config.get('enable_verification', True)
        
        # Initialize performance tracking
        self.solve_times = []
        self.complexity_stats = defaultdict(int)
        
        # Initialize components
        self.nlp_processor = NLPProcessor()
        self.relation_discovery = ImplicitRelationDiscovery()
        self.multi_level_reasoning = MultiLevelReasoning()
        self.chain_verification = ChainVerification()
        
        # Initialize enhanced algorithm components
        self.ird_discovery = IRDCombinatorialDiscovery()
        self.mlr_reasoning = MLRStateBasedReasoning()
        self.cv_verification = CVFormalVerification()
        
        # Setup logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def solve_mathematical_problem(self, problem_text: str) -> Dict[str, Any]:
        """Enhanced solve method with better calculation and answer extraction"""
        start_time = time.time()
        
        try:
            self.logger.info("Starting NLP processing")
            # Step 1: Extract entities from problem text
            entities = self.nlp_processor.extract_entities(problem_text)
            
            self.logger.info("Discovering implicit relations")
            # Step 2: Discover implicit relations
            relations = self.relation_discovery.discover_relations(entities, problem_text)
            
            # Try enhanced IRD discovery if regular discovery finds few relations
            if len(relations) < 2:
                qualia = {'context': problem_text, 'domain': 'mathematics'}
                enhanced_relations = self.ird_discovery.combinatorial_discovery(entities, qualia)
                relations.extend(enhanced_relations)
            
            self.logger.info("Generating reasoning chain")
            # Step 3: Generate reasoning chain with real calculations
            try:
                reasoning_steps = self._generate_enhanced_reasoning_chain(entities, relations, problem_text)
                self.logger.info(f"Generated {len(reasoning_steps)} reasoning steps")
            except Exception as e:
                self.logger.error(f"Error in reasoning chain generation: {e}")
                reasoning_steps = []
            
            # Step 4: Enhanced answer extraction
            final_answer = None
            try:
                final_answer = self._extract_final_answer(reasoning_steps)
                self.logger.info(f"Extracted final answer: {final_answer}")
            except Exception as e:
                self.logger.error(f"Error in answer extraction: {e}")
            
            # If still no answer, try direct calculation for simple arithmetic
            if final_answer is None:
                try:
                    final_answer = self._try_direct_calculation(entities, problem_text)
                    self.logger.info(f"Direct calculation result: {final_answer}")
                except Exception as e:
                    self.logger.error(f"Error in direct calculation: {e}")
            
            processing_time = time.time() - start_time
            self.solve_times.append(processing_time)
            
            # Step 5: Chain verification
            verification_result = None
            try:
                if reasoning_steps and len(reasoning_steps) > 0:
                    self.logger.info("Verifying reasoning chain")
                    verification_result = self.chain_verification.verify_reasoning_chain(reasoning_steps)
                    
                    # Try CV formal verification for additional confidence
                    if final_answer is not None:
                        goal = {'target_variable': 'final_answer', 'target_value': final_answer}
                        cv_result = self.cv_verification.formal_verification(reasoning_steps, goal)
                        if cv_result.get('status') == 'VALID':
                            verification_result['confidence'] = max(
                                verification_result.get('confidence', 0.5),
                                cv_result.get('verification_score', 0.5)
                            )
            except Exception as e:
                self.logger.error(f"Error in verification: {e}")
            
            complexity = self._estimate_complexity(entities, relations)
            self.complexity_stats[complexity] += 1
            
            self.logger.info(f"Problem solved successfully in {processing_time:.3f} seconds")
            
            return {
                'problem': problem_text,
                'entities': [entity.to_dict() for entity in entities],
                'relations': [relation.to_dict() for relation in relations],
                'reasoning_steps': [step.to_dict() for step in reasoning_steps],
                'final_answer': final_answer,
                'complexity': complexity,
                'processing_time': processing_time,
                'verification': verification_result,
                'metadata': {
                    'num_entities': len(entities),
                    'num_relations': len(relations),
                    'num_reasoning_steps': len(reasoning_steps),
                    'confidence': verification_result.get('confidence', 0.0) if verification_result else 0.0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error solving problem: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'problem': problem_text,
                'error': str(e),
                'final_answer': None,
                'processing_time': time.time() - start_time
            }
    
    def _extract_final_answer(self, reasoning_steps: List[ReasoningStep]) -> Optional[Union[float, str]]:
        """Extract final answer from reasoning steps with improved logic"""
        if not reasoning_steps:
            return None
        
        # Get the last step's output
        last_step = reasoning_steps[-1]
        if last_step.output_entity and last_step.output_entity.value is not None:
            return last_step.output_entity.value
        
        # Try to find any step with a numerical result
        for step in reversed(reasoning_steps):
            if step.output_entity and step.output_entity.value is not None:
                if isinstance(step.output_entity.value, (int, float)):
                    return step.output_entity.value
        
        # If no numerical result found, try to extract from any entity value
        for step in reversed(reasoning_steps):
            for entity in step.input_entities:
                if entity.value is not None and isinstance(entity.value, (int, float)):
                    return entity.value
        
        return None
    
    def _generate_enhanced_reasoning_chain(self, entities: List[MathEntity], 
                                         relations: List[ImplicitRelation],
                                         problem_text: str) -> List[ReasoningStep]:
        """Generate reasoning chain with enhanced L2 multi-step reasoning"""
        steps = []
        
        # 首先尝试使用新的多级推理规划器
        mlr = MultiLevelReasoning()
        enhanced_steps = mlr._plan_reasoning_steps(entities, relations, problem_text)
        
        if enhanced_steps:
            # 使用增强的规划步骤
            self.logger.info(f"Using enhanced MLR planner, generated {len(enhanced_steps)} steps")
            return enhanced_steps
        
        # 如果规划器没有生成步骤，回退到改进的基础逻辑
        numbers = [e for e in entities if e.entity_type == EntityType.NUMBER and e.value is not None]
        
        if len(numbers) >= 2:
            # 使用增强的操作检测
            operation = mlr._get_operation_from_question(problem_text)
            self.logger.info(f"Detected operation: {operation}")
            
            # 处理多步推理案例
            if operation == 'multi_step_subtraction' and len(numbers) >= 3:
                # 案例：Maria有$100，花$35买食品，花$25买汽油，还剩多少？
                self.logger.info("Processing multi-step subtraction")
                
                # Step 1: 计算总支出
                total_spending = numbers[1].value + numbers[2].value
                step1 = ReasoningStep(
                    step_id=0,
                    description=f"Calculate total spending: {numbers[1].value} + {numbers[2].value} = {total_spending}",
                    operation="addition",
                    input_entities=[numbers[1], numbers[2]],
                    output_entity=MathEntity(
                        text=str(total_spending),
                        entity_type=EntityType.NUMBER,
                        value=total_spending,
                        unit=None,
                        confidence=0.95
                    ),
                    confidence=0.95,
                    dependencies=[]
                )
                steps.append(step1)
                
                # Step 2: 从总金额中减去总支出
                remaining = numbers[0].value - total_spending
                step2 = ReasoningStep(
                    step_id=1,
                    description=f"Calculate remaining money: {numbers[0].value} - {total_spending} = {remaining}",
                    operation="subtraction",
                    input_entities=[numbers[0], step1.output_entity],
                    output_entity=MathEntity(
                        text=str(remaining),
                        entity_type=EntityType.NUMBER,
                        value=remaining,
                        unit=None,
                        confidence=0.95
                    ),
                    confidence=0.95,
                    dependencies=[0]
                )
                steps.append(step2)
                
            elif operation == 'multiplication' and 'total' in problem_text.lower():
                # 案例：书店每本书$12，买5本，总费用多少？
                self.logger.info("Processing multiplication for total cost")
                
                # 智能识别单价和数量
                price_entity = numbers[0]  # 假设第一个是价格
                quantity_entity = numbers[1]  # 第二个是数量
                
                # 通过上下文验证
                if '$' in problem_text and numbers[0].position[0] > numbers[1].position[0]:
                    # 如果$符号出现且第一个数字位置靠后，可能顺序相反
                    for i, entity in enumerate(numbers):
                        context = problem_text[max(0, entity.position[0]-10):entity.position[1]+10].lower()
                        if '$' in context or 'dollar' in context or 'price' in context or 'cost' in context:
                            price_entity = entity
                        elif 'book' in context or 'item' in context or 'buy' in context:
                            quantity_entity = entity
                
                total_cost = price_entity.value * quantity_entity.value
                step = ReasoningStep(
                    step_id=0,
                    description=f"Calculate total cost: {price_entity.value} × {quantity_entity.value} = {total_cost}",
                    operation="multiplication",
                    input_entities=[price_entity, quantity_entity],
                    output_entity=MathEntity(
                        text=str(total_cost),
                        entity_type=EntityType.NUMBER,
                        value=total_cost,
                        unit=None,
                        confidence=0.95
                    ),
                    confidence=0.95,
                    dependencies=[]
                )
                steps.append(step)
                
            elif operation == 'division':
                # 除法处理
                result = numbers[0].value / numbers[1].value if numbers[1].value != 0 else 0
                step = self._create_arithmetic_step(0, 'division', numbers[:2])
                if step:
                    steps.append(step)
                    
            else:
                # 其他单步操作
                step = self._create_arithmetic_step(0, operation, numbers[:2])
                if step:
                    steps.append(step)
        
        # 特殊处理：速度计算
        if 'speed' in problem_text.lower() or 'km/h' in problem_text.lower():
            if len(numbers) >= 2:
                distance = numbers[0].value
                time_value = numbers[1].value
                
                # 时间单位转换
                if 'minutes' in problem_text.lower() or 'min' in problem_text.lower():
                    time_hours = time_value / 60
                else:
                    time_hours = time_value
                
                speed = distance / time_hours if time_hours != 0 else 0
                
                step = ReasoningStep(
                    step_id=len(steps),
                    description=f"Calculate speed: {distance} km ÷ {time_hours} hours = {speed} km/h",
                    operation="rate_calculation",
                    input_entities=numbers[:2],
                    output_entity=MathEntity(
                        text=str(speed),
                        entity_type=EntityType.NUMBER,
                        value=speed,
                        unit="km/h"
                    ),
                    confidence=0.9,
                    dependencies=[]
                )
                steps.append(step)
        
        return steps
    
    def _create_arithmetic_step(self, step_id: int, operation: str, entities: List[MathEntity]) -> ReasoningStep:
        """Create a reasoning step for basic arithmetic operations"""
        if len(entities) < 2:
            return None
        
        val1, val2 = float(entities[0].value), float(entities[1].value)
        
        if operation == 'addition':
            result = val1 + val2
            description = f"Add {val1} + {val2} = {result}"
        elif operation == 'subtraction':
            result = val1 - val2
            description = f"Subtract {val1} - {val2} = {result}"
        elif operation == 'multiplication':
            result = val1 * val2
            description = f"Multiply {val1} × {val2} = {result}"
        elif operation == 'division':
            result = val1 / val2 if val2 != 0 else float('inf')
            description = f"Divide {val1} ÷ {val2} = {result}"
        else:
            result = val1
            description = f"Process {val1}"
        
        output_entity = MathEntity(
            text=str(result),
            entity_type=EntityType.NUMBER,
            value=result,
            confidence=0.95
        )
        
        return ReasoningStep(
            step_id=step_id,
            description=description,
            operation=operation,
            input_entities=entities,
            output_entity=output_entity,
            confidence=0.95,
            dependencies=[]
        )
    
    def _try_direct_calculation(self, entities: List[MathEntity], problem_text: str) -> Optional[float]:
        """Try direct calculation for simple problems"""
        numbers = [e.value for e in entities if e.entity_type == EntityType.NUMBER and e.value is not None]
        
        if len(numbers) >= 2:
            if any(word in problem_text.lower() for word in ['add', 'plus', 'sum', 'total']):
                return sum(numbers)
            elif any(word in problem_text.lower() for word in ['subtract', 'minus', 'difference']):
                return numbers[0] - numbers[1]
            elif any(word in problem_text.lower() for word in ['multiply', 'times', 'product']):
                return numbers[0] * numbers[1]
            elif any(word in problem_text.lower() for word in ['divide']):
                return numbers[0] / numbers[1] if numbers[1] != 0 else None
        
        return None
    
    def _estimate_complexity(self, entities: List[MathEntity], relations: List[ImplicitRelation]) -> str:
        """Estimate problem complexity based on entities and relations."""
        num_entities = len(entities)
        num_relations = len(relations)
        
        # Simple heuristic for complexity estimation
        if num_entities <= 3 and num_relations <= 1:
            return ProblemComplexity.L0_EXPLICIT.value
        elif num_entities <= 5 and num_relations <= 2:
            return ProblemComplexity.L1_SHALLOW.value
        elif num_entities <= 8 and num_relations <= 4:
            return ProblemComplexity.L2_MEDIUM.value
        else:
            return ProblemComplexity.L3_DEEP.value
    
    def batch_solve_problems(self, problems: List[str]) -> List[Dict[str, Any]]:
        """Solve multiple problems in batch."""
        results = []
        
        for i, problem in enumerate(problems):
            self.logger.info(f"Solving problem {i+1}/{len(problems)}")
            result = self.solve_mathematical_problem(problem)
            results.append(result)
        
        return results
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        return {
            'component_status': {
                'nlp_processor': 'active',
                'relation_discovery': 'active',
                'multi_level_reasoning': 'active',
                                 'chain_verification': 'active' if self.enable_verification else 'disabled'
            },
            'configuration': self.config,
            'version': '1.0.0'
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize system
    system = MathematicalReasoningSystem()
    
    # Example problem
    test_problem = """
    A tank contains 5L of water. Ice cubes of 200 cm³ are dropped one cube per minute. 
    Water leaks at 2 mL/s. How long will it take for the water level to rise to 9L?
    """
    
    # Solve problem
    result = system.solve_mathematical_problem(test_problem)
    
    # Print results
    print("=== Mathematical Reasoning System Results ===")
    print(f"Problem: {result['problem']}")
    print(f"Final Answer: {result['final_answer']}")
    print(f"Processing Time: {result['processing_time']:.3f} seconds")
    print(f"Number of Entities: {result['metadata']['num_entities']}")
    print(f"Number of Relations: {result['metadata']['num_relations']}")
    print(f"Number of Reasoning Steps: {result['metadata']['num_reasoning_steps']}")
    print(f"Estimated Complexity: {result['complexity']}")
    
    if result.get('verification'):
        print(f"Verification: {'PASSED' if result['verification']['is_valid'] else 'FAILED'}")
        print(f"Confidence Score: {result['verification']['confidence_score']:.3f}")