#!/usr/bin/env python3
"""
COT-DIR Method Implementation
============================

Chain-of-Thought with Deep Implicit Relations (COT-DIR) method implementation
based on the paper specifications.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ImplicitRelation:
    """Represents an implicit mathematical relation"""
    relation_type: str  # 'proportional', 'inverse', 'algebraic', 'geometric', etc.
    entities: List[str]  # Variables/quantities involved
    confidence: float  # Confidence score [0, 1]
    context: str  # Textual context where relation was found
    mathematical_form: Optional[str] = None  # Mathematical expression if applicable

@dataclass
class ReasoningStep:
    """Represents a step in the reasoning process"""
    step_id: int
    description: str
    operation: str
    input_values: List[Any]
    output_value: Any
    relations_used: List[ImplicitRelation]
    confidence: float

@dataclass
class COTDIRResult:
    """Result from COT-DIR processing"""
    answer: Any
    reasoning_steps: List[ReasoningStep]
    discovered_relations: List[ImplicitRelation]
    dir_score: float
    confidence: float
    efficiency_metrics: Dict[str, float]

class ImplicitRelationDetector:
    """Detects implicit mathematical relations in problem statements"""
    
    def __init__(self):
        self.relation_patterns = {
            'proportional': [
                r'(\w+)\s+is\s+(\d+)\s+times\s+(\w+)',
                r'for\s+every\s+(\w+),\s+there\s+are\s+(\d+)\s+(\w+)',
                r'(\w+)\s+costs?\s+\$?(\d+\.?\d*)\s+each',
                r'(\w+)\s+per\s+(\w+)'
            ],
            'inverse': [
                r'as\s+(\w+)\s+increases?,\s+(\w+)\s+decreases?',
                r'(\w+)\s+and\s+(\w+)\s+are\s+inversely\s+related'
            ],
            'algebraic': [
                r'(\w+)\s*=\s*(\w+)\s*[\+\-\*\/]\s*(\w+)',
                r'(\w+)\s+equals?\s+(\w+)\s+(plus|minus|times|divided by)\s+(\w+)',
                r'the\s+sum\s+of\s+(\w+)\s+and\s+(\w+)',
                r'the\s+difference\s+between\s+(\w+)\s+and\s+(\w+)'
            ],
            'geometric': [
                r'area\s+of\s+(\w+)',
                r'perimeter\s+of\s+(\w+)',
                r'volume\s+of\s+(\w+)',
                r'(\w+)\s+is\s+(\d+)\s+units\s+(long|wide|tall|high)'
            ],
            'temporal': [
                r'after\s+(\d+\.?\d*)\s+(minutes?|hours?|days?)',
                r'in\s+(\d+\.?\d*)\s+(minutes?|hours?|days?)',
                r'(\w+)\s+takes\s+(\d+\.?\d*)\s+(minutes?|hours?|days?)'
            ]
        }
    
    def detect_relations(self, text: str) -> List[ImplicitRelation]:
        """Detect implicit relations in problem text"""
        relations = []
        text_lower = text.lower()
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    entities = [group for group in match.groups() if group and not group.isdigit()]
                    if len(entities) >= 2:
                        relation = ImplicitRelation(
                            relation_type=relation_type,
                            entities=entities,
                            confidence=self._calculate_confidence(match.group(), pattern),
                            context=match.group()
                        )
                        relations.append(relation)
        
        # Additional semantic analysis for complex relations
        relations.extend(self._detect_semantic_relations(text))
        
        return self._deduplicate_relations(relations)
    
    def _calculate_confidence(self, match_text: str, pattern: str) -> float:
        """Calculate confidence score for a detected relation"""
        # Simple confidence based on pattern specificity and context
        base_confidence = 0.7
        
        # Boost confidence for specific mathematical terms
        math_terms = ['equals', 'sum', 'difference', 'product', 'quotient', 'times', 'per']
        if any(term in match_text.lower() for term in math_terms):
            base_confidence += 0.2
        
        # Boost confidence for numerical context
        if re.search(r'\d+', match_text):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _detect_semantic_relations(self, text: str) -> List[ImplicitRelation]:
        """Detect relations using semantic analysis"""
        relations = []
        
        # Look for comparative relations
        comparative_patterns = [
            r'(\w+)\s+is\s+(more|less|greater|smaller)\s+than\s+(\w+)',
            r'(\w+)\s+has\s+(more|fewer|less)\s+(\w+)\s+than\s+(\w+)'
        ]
        
        for pattern in comparative_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                groups = match.groups()
                if len(groups) >= 3:
                    relation = ImplicitRelation(
                        relation_type='comparative',
                        entities=[groups[0], groups[2]] if len(groups) == 3 else [groups[0], groups[3]],
                        confidence=0.8,
                        context=match.group()
                    )
                    relations.append(relation)
        
        return relations
    
    def _deduplicate_relations(self, relations: List[ImplicitRelation]) -> List[ImplicitRelation]:
        """Remove duplicate relations"""
        unique_relations = []
        seen_signatures = set()
        
        for relation in relations:
            # Create signature based on type and entities
            signature = (relation.relation_type, tuple(sorted(relation.entities)))
            if signature not in seen_signatures:
                unique_relations.append(relation)
                seen_signatures.add(signature)
        
        return unique_relations

class DeepRelationModeler:
    """Models relationships between detected implicit relations"""
    
    def __init__(self):
        self.relation_hierarchy = {
            'algebraic': ['proportional', 'comparative'],
            'geometric': ['proportional'],
            'temporal': ['proportional', 'inverse']
        }
    
    def model_deep_relations(self, relations: List[ImplicitRelation]) -> Dict[str, Any]:
        """Create deep models of relation interactions"""
        
        relation_graph = self._build_relation_graph(relations)
        hierarchical_structure = self._identify_hierarchy(relations)
        interaction_patterns = self._analyze_interactions(relations)
        
        return {
            'relation_graph': relation_graph,
            'hierarchical_structure': hierarchical_structure,
            'interaction_patterns': interaction_patterns,
            'complexity_score': self._calculate_complexity_score(relations)
        }
    
    def _build_relation_graph(self, relations: List[ImplicitRelation]) -> Dict[str, List[str]]:
        """Build a graph of entity relationships"""
        graph = {}
        
        for relation in relations:
            for entity in relation.entities:
                if entity not in graph:
                    graph[entity] = []
                
                # Connect to other entities in same relation
                for other_entity in relation.entities:
                    if other_entity != entity and other_entity not in graph[entity]:
                        graph[entity].append(other_entity)
        
        return graph
    
    def _identify_hierarchy(self, relations: List[ImplicitRelation]) -> Dict[str, int]:
        """Identify hierarchical levels of relations"""
        hierarchy = {}
        
        for relation in relations:
            level = 0
            relation_type = relation.relation_type
            
            # Check if this relation type depends on others
            for parent_type, children in self.relation_hierarchy.items():
                if relation_type in children:
                    level = max(level, 1)
                if relation_type == parent_type:
                    level = max(level, 2)
            
            hierarchy[f"{relation.relation_type}_{hash(tuple(relation.entities)) % 1000}"] = level
        
        return hierarchy
    
    def _analyze_interactions(self, relations: List[ImplicitRelation]) -> List[Dict]:
        """Analyze how relations interact with each other"""
        interactions = []
        
        for i, rel1 in enumerate(relations):
            for j, rel2 in enumerate(relations[i+1:], i+1):
                # Check for entity overlap
                overlap = set(rel1.entities) & set(rel2.entities)
                if overlap:
                    interaction = {
                        'relation_1': f"{rel1.relation_type}_{i}",
                        'relation_2': f"{rel2.relation_type}_{j}",
                        'shared_entities': list(overlap),
                        'interaction_type': self._classify_interaction(rel1, rel2),
                        'strength': len(overlap) / max(len(rel1.entities), len(rel2.entities))
                    }
                    interactions.append(interaction)
        
        return interactions
    
    def _classify_interaction(self, rel1: ImplicitRelation, rel2: ImplicitRelation) -> str:
        """Classify the type of interaction between two relations"""
        if rel1.relation_type == rel2.relation_type:
            return 'reinforcing'
        elif (rel1.relation_type, rel2.relation_type) in [('proportional', 'inverse'), ('inverse', 'proportional')]:
            return 'conflicting'
        else:
            return 'complementary'
    
    def _calculate_complexity_score(self, relations: List[ImplicitRelation]) -> float:
        """Calculate overall complexity score based on relations"""
        if not relations:
            return 0.0
        
        # Base score from number of relations
        base_score = min(len(relations) / 10.0, 0.5)
        
        # Bonus for relation diversity
        unique_types = len(set(rel.relation_type for rel in relations))
        diversity_bonus = min(unique_types / 5.0, 0.3)
        
        # Bonus for high-confidence relations
        confidence_bonus = np.mean([rel.confidence for rel in relations]) * 0.2
        
        return min(base_score + diversity_bonus + confidence_bonus, 1.0)

class AdaptiveReasoningPath:
    """Generates adaptive reasoning paths based on discovered relations"""
    
    def __init__(self):
        self.reasoning_strategies = {
            'direct': self._direct_computation,
            'step_by_step': self._step_by_step_reasoning,
            'relation_based': self._relation_based_reasoning,
            'hierarchical': self._hierarchical_reasoning
        }
    
    def generate_reasoning_path(self, problem: str, relations: List[ImplicitRelation], 
                              relation_model: Dict) -> List[ReasoningStep]:
        """Generate optimal reasoning path based on problem and relations"""
        
        # Select strategy based on problem complexity and relations
        strategy = self._select_strategy(problem, relations, relation_model)
        
        # Generate reasoning steps using selected strategy
        steps = self.reasoning_strategies[strategy](problem, relations, relation_model)
        
        return steps
    
    def _select_strategy(self, problem: str, relations: List[ImplicitRelation], 
                        relation_model: Dict) -> str:
        """Select optimal reasoning strategy"""
        
        complexity_score = relation_model.get('complexity_score', 0)
        num_relations = len(relations)
        
        if complexity_score < 0.3 and num_relations <= 2:
            return 'direct'
        elif complexity_score < 0.6 and num_relations <= 5:
            return 'step_by_step'
        elif len(relation_model.get('interaction_patterns', [])) > 0:
            return 'relation_based'
        else:
            return 'hierarchical'
    
    def _direct_computation(self, problem: str, relations: List[ImplicitRelation], 
                           relation_model: Dict) -> List[ReasoningStep]:
        """Direct computation for simple problems"""
        steps = []
        
        # Extract numbers from problem
        numbers = re.findall(r'\d+\.?\d*', problem)
        if len(numbers) >= 2:
            step = ReasoningStep(
                step_id=1,
                description="Direct computation",
                operation="arithmetic",
                input_values=numbers[:2],
                output_value=float(numbers[0]) + float(numbers[1]),  # Simple example
                relations_used=[],
                confidence=0.9
            )
            steps.append(step)
        
        return steps
    
    def _step_by_step_reasoning(self, problem: str, relations: List[ImplicitRelation], 
                               relation_model: Dict) -> List[ReasoningStep]:
        """Step-by-step reasoning for moderate complexity"""
        steps = []
        
        # Identify key quantities
        quantities = self._extract_quantities(problem)
        
        for i, (quantity, value) in enumerate(quantities.items()):
            step = ReasoningStep(
                step_id=i + 1,
                description=f"Identify {quantity}",
                operation="identification",
                input_values=[quantity],
                output_value=value,
                relations_used=[],
                confidence=0.8
            )
            steps.append(step)
        
        return steps
    
    def _relation_based_reasoning(self, problem: str, relations: List[ImplicitRelation], 
                                 relation_model: Dict) -> List[ReasoningStep]:
        """Reasoning based on discovered relations"""
        steps = []
        
        for i, relation in enumerate(relations):
            step = ReasoningStep(
                step_id=i + 1,
                description=f"Apply {relation.relation_type} relation",
                operation="relation_application",
                input_values=relation.entities,
                output_value=f"Using {relation.relation_type} relation between {', '.join(relation.entities)}",
                relations_used=[relation],
                confidence=relation.confidence
            )
            steps.append(step)
        
        return steps
    
    def _hierarchical_reasoning(self, problem: str, relations: List[ImplicitRelation], 
                               relation_model: Dict) -> List[ReasoningStep]:
        """Hierarchical reasoning for complex problems"""
        steps = []
        hierarchy = relation_model.get('hierarchical_structure', {})
        
        # Sort relations by hierarchy level
        sorted_relations = sorted(relations, 
                                key=lambda r: hierarchy.get(f"{r.relation_type}_{hash(tuple(r.entities)) % 1000}", 0))
        
        for i, relation in enumerate(sorted_relations):
            level = hierarchy.get(f"{relation.relation_type}_{hash(tuple(relation.entities)) % 1000}", 0)
            step = ReasoningStep(
                step_id=i + 1,
                description=f"Level {level}: Apply {relation.relation_type} relation",
                operation="hierarchical_application",
                input_values=relation.entities,
                output_value=f"Hierarchical application of {relation.relation_type}",
                relations_used=[relation],
                confidence=relation.confidence * (1 - level * 0.1)  # Confidence decreases with complexity
            )
            steps.append(step)
        
        return steps
    
    def _extract_quantities(self, problem: str) -> Dict[str, float]:
        """Extract numerical quantities from problem text"""
        quantities = {}
        
        # Simple pattern matching for quantities
        patterns = [
            r'(\w+)\s+(?:costs?|is|are|has)\s+\$?(\d+\.?\d*)',
            r'(\d+\.?\d*)\s+(\w+)',
            r'(\w+)\s*=\s*(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, problem.lower())
            for match in matches:
                groups = match.groups()
                if len(groups) == 2:
                    try:
                        if groups[0].replace('.', '').isdigit():
                            quantities[groups[1]] = float(groups[0])
                        elif groups[1].replace('.', '').isdigit():
                            quantities[groups[0]] = float(groups[1])
                    except ValueError:
                        continue
        
        return quantities

class RelationAwareAttention:
    """Attention mechanism that focuses on discovered relations"""
    
    def __init__(self):
        self.attention_weights = {}
    
    def compute_attention(self, reasoning_steps: List[ReasoningStep], 
                         relations: List[ImplicitRelation]) -> Dict[int, float]:
        """Compute attention weights for reasoning steps based on relations"""
        
        attention_scores = {}
        
        for step in reasoning_steps:
            score = self._base_attention_score(step)
            
            # Boost score based on relation usage
            for relation in step.relations_used:
                score += relation.confidence * 0.3
                
                # Extra boost for high-complexity relations
                if relation.relation_type in ['geometric', 'algebraic']:
                    score += 0.2
            
            # Normalize and store
            attention_scores[step.step_id] = min(score, 1.0)
        
        return attention_scores
    
    def _base_attention_score(self, step: ReasoningStep) -> float:
        """Calculate base attention score for a reasoning step"""
        base_score = 0.5
        
        # Boost for certain operations
        if step.operation in ['relation_application', 'hierarchical_application']:
            base_score += 0.3
        
        # Boost for high confidence
        base_score += step.confidence * 0.2
        
        return base_score
    
    def apply_attention(self, reasoning_steps: List[ReasoningStep]) -> List[ReasoningStep]:
        """Apply attention mechanism to enhance reasoning steps"""
        
        if not reasoning_steps:
            return reasoning_steps
        
        # Compute attention weights
        attention_weights = self.compute_attention(reasoning_steps, [])
        
        # Apply attention to enhance steps
        enhanced_steps = []
        for step in reasoning_steps:
            weight = attention_weights.get(step.step_id, 0.5)
            
            # Enhance confidence based on attention
            enhanced_step = ReasoningStep(
                step_id=step.step_id,
                description=step.description,
                operation=step.operation,
                input_values=step.input_values,
                output_value=step.output_value,
                relations_used=step.relations_used,
                confidence=min(step.confidence + weight * 0.1, 1.0)
            )
            enhanced_steps.append(enhanced_step)
        
        return enhanced_steps

class COTDIRMethod:
    """
    Main COT-DIR method implementation combining all components
    """
    
    def __init__(self):
        self.relation_detector = ImplicitRelationDetector()
        self.relation_modeler = DeepRelationModeler()
        self.reasoning_path_generator = AdaptiveReasoningPath()
        self.attention_mechanism = RelationAwareAttention()
        self.logger = logging.getLogger(__name__)
    
    def solve_problem(self, problem: Dict) -> COTDIRResult:
        """
        Solve a mathematical problem using COT-DIR method
        
        Args:
            problem: Dictionary containing 'problem' text and optionally 'answer'
            
        Returns:
            COTDIRResult with answer, reasoning steps, and discovered relations
        """
        start_time = time.time()
        
        problem_text = problem.get('problem', problem.get('question', ''))
        
        # Step 1: Detect implicit relations
        relations = self.relation_detector.detect_relations(problem_text)
        
        # Step 2: Model deep relationships
        relation_model = self.relation_modeler.model_deep_relations(relations)
        
        # Step 3: Generate adaptive reasoning path
        reasoning_steps = self.reasoning_path_generator.generate_reasoning_path(
            problem_text, relations, relation_model
        )
        
        # Step 4: Apply relation-aware attention
        enhanced_steps = self.attention_mechanism.apply_attention(reasoning_steps)
        
        # Step 5: Compute final answer (simplified)
        final_answer = self._compute_final_answer(problem_text, enhanced_steps)
        
        # Calculate metrics
        dir_score = relation_model.get('complexity_score', 0)
        overall_confidence = np.mean([step.confidence for step in enhanced_steps]) if enhanced_steps else 0
        
        processing_time = time.time() - start_time
        
        return COTDIRResult(
            answer=final_answer,
            reasoning_steps=enhanced_steps,
            discovered_relations=relations,
            dir_score=dir_score,
            confidence=overall_confidence,
            efficiency_metrics={
                'processing_time': processing_time,
                'relations_discovered': len(relations),
                'reasoning_steps': len(enhanced_steps),
                'attention_applied': True
            }
        )
    
    def _compute_final_answer(self, problem_text: str, reasoning_steps: List[ReasoningStep]) -> Any:
        """Compute final answer based on reasoning steps"""
        
        # Extract numbers from problem for simple computation
        numbers = re.findall(r'\d+\.?\d*', problem_text)
        
        if not numbers:
            return "Unable to extract numerical answer"
        
        # Simple heuristic: if it's asking for sum, add; if difference, subtract
        problem_lower = problem_text.lower()
        
        if 'total' in problem_lower or 'sum' in problem_lower or 'altogether' in problem_lower:
            return sum(float(n) for n in numbers)
        elif 'difference' in problem_lower or 'left' in problem_lower or 'remaining' in problem_lower:
            if len(numbers) >= 2:
                return float(numbers[0]) - float(numbers[1])
        elif 'each' in problem_lower or 'per' in problem_lower:
            if len(numbers) >= 2:
                return float(numbers[0]) / float(numbers[1])
        
        # Default: return first number found
        return float(numbers[0]) if numbers else 0
    
    def __call__(self, problem: Dict) -> Tuple[Any, List[ReasoningStep], List[ImplicitRelation]]:
        """Make the method callable for benchmark evaluation"""
        result = self.solve_problem(problem)
        return result.answer, result.reasoning_steps, result.discovered_relations 