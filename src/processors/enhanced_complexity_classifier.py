"""
Enhanced Complexity Classifier with Sub-level Classification
Implements fine-grained complexity classification for mathematical problems
Part of Story 6.1: Mathematical Reasoning Enhancement - Phase 2
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

# Import semantic components if available
try:
    from ..reasoning.semantic_ird_engine import SemanticIRDEngine
    from ..reasoning.mathematical_ontology import MathematicalOntology
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class ComplexityLevel(Enum):
    """Main complexity levels"""
    L0 = "L0"  # Explicit
    L1 = "L1"  # Shallow implicit
    L2 = "L2"  # Medium implicit
    L3 = "L3"  # Deep implicit

class SubLevel(Enum):
    """Sub-levels for fine-grained classification"""
    # L1 sub-levels
    L1_1 = "L1.1"  # Single inference step
    L1_2 = "L1.2"  # Two inference steps
    L1_3 = "L1.3"  # Three inference steps
    
    # L2 sub-levels
    L2_1 = "L2.1"  # Simple multi-step (3-5 steps)
    L2_2 = "L2.2"  # Moderate multi-step (5-7 steps)
    L2_3 = "L2.3"  # Complex multi-step (7-10 steps)
    
    # L3 sub-levels
    L3_1 = "L3.1"  # Deep reasoning (10-15 steps)
    L3_2 = "L3.2"  # Very deep reasoning (15-20 steps)
    L3_3 = "L3.3"  # Extremely deep reasoning (20+ steps)

class ReasoningType(Enum):
    """Types of reasoning required"""
    ALGEBRAIC = "algebraic"
    GEOMETRIC = "geometric"
    NUMERIC = "numeric"
    LOGICAL = "logical"
    ANALOGICAL = "analogical"
    PROBABILISTIC = "probabilistic"
    COMBINATORIAL = "combinatorial"
    OPTIMIZATION = "optimization"

@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics"""
    reasoning_depth: int
    knowledge_dependencies: int
    inference_steps: int
    variable_count: int
    equation_count: int
    constraint_count: int
    domain_switches: int
    abstraction_level: float
    semantic_complexity: float
    computational_complexity: float

@dataclass
class ComplexityClassification:
    """Complete complexity classification result"""
    main_level: ComplexityLevel
    sub_level: SubLevel
    reasoning_types: List[ReasoningType]
    metrics: ComplexityMetrics
    confidence: float
    explanation: str
    domain_specific_factors: Dict[str, Any]

class EnhancedComplexityClassifier:
    """
    Enhanced complexity classifier with sub-level classification
    and domain-specific complexity metrics
    """
    
    def __init__(self):
        """Initialize the enhanced complexity classifier"""
        self.semantic_engine = None
        self.ontology = None
        
        if SEMANTIC_AVAILABLE:
            self.semantic_engine = SemanticIRDEngine()
            self.ontology = MathematicalOntology()
        
        # Pattern libraries
        self.implicit_patterns = self._load_implicit_patterns()
        self.reasoning_patterns = self._load_reasoning_patterns()
        self.complexity_indicators = self._load_complexity_indicators()
        
        # Caches
        self.classification_cache = {}
        
        logger.info("Enhanced Complexity Classifier initialized")
    
    def classify_problem(self, problem: Dict[str, Any]) -> ComplexityClassification:
        """
        Classify problem complexity with sub-levels
        
        Args:
            problem: Mathematical problem representation
            
        Returns:
            ComplexityClassification with detailed metrics
        """
        # Check cache
        problem_id = problem.get('id', str(problem))
        if problem_id in self.classification_cache:
            return self.classification_cache[problem_id]
        
        # Extract metrics
        metrics = self._calculate_metrics(problem)
        
        # Determine reasoning types
        reasoning_types = self._identify_reasoning_types(problem)
        
        # Calculate main complexity level
        main_level = self._determine_main_level(metrics, reasoning_types)
        
        # Calculate sub-level
        sub_level = self._determine_sub_level(main_level, metrics, problem)
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics, problem)
        
        # Generate explanation
        explanation = self._generate_explanation(main_level, sub_level, metrics)
        
        # Extract domain-specific factors
        domain_factors = self._extract_domain_factors(problem, reasoning_types)
        
        classification = ComplexityClassification(
            main_level=main_level,
            sub_level=sub_level,
            reasoning_types=reasoning_types,
            metrics=metrics,
            confidence=confidence,
            explanation=explanation,
            domain_specific_factors=domain_factors
        )
        
        # Cache result
        self.classification_cache[problem_id] = classification
        
        return classification
    
    def _load_implicit_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for implicit relationship detection"""
        return {
            'temporal': [
                r'after\s+(\d+)\s+(hours?|minutes?|days?)',
                r'(\d+)\s+(hours?|minutes?|days?)\s+later',
                r'at\s+the\s+same\s+time',
                r'simultaneously'
            ],
            'proportional': [
                r'proportional\s+to',
                r'varies\s+(?:directly|inversely)\s+with',
                r'times\s+as\s+(?:much|many)',
                r'ratio\s+of'
            ],
            'comparative': [
                r'more\s+than',
                r'less\s+than',
                r'(?:twice|thrice|half)\s+(?:as|the)',
                r'compared\s+to'
            ],
            'conditional': [
                r'if\s+.*\s+then',
                r'provided\s+that',
                r'assuming\s+that',
                r'given\s+that'
            ],
            'recursive': [
                r'each\s+.*\s+has',
                r'every\s+.*\s+contains',
                r'nested',
                r'within\s+each'
            ]
        }
    
    def _load_reasoning_patterns(self) -> Dict[ReasoningType, List[str]]:
        """Load patterns for reasoning type identification"""
        return {
            ReasoningType.ALGEBRAIC: [
                r'solve\s+for',
                r'equation',
                r'variable',
                r'unknown',
                r'express\s+.*\s+in\s+terms'
            ],
            ReasoningType.GEOMETRIC: [
                r'angle',
                r'triangle',
                r'circle',
                r'area',
                r'perimeter',
                r'volume'
            ],
            ReasoningType.PROBABILISTIC: [
                r'probability',
                r'chance',
                r'likelihood',
                r'random',
                r'expected'
            ],
            ReasoningType.COMBINATORIAL: [
                r'combination',
                r'permutation',
                r'arrangement',
                r'selection',
                r'ways\s+to'
            ],
            ReasoningType.OPTIMIZATION: [
                r'maximi[zs]e',
                r'minimi[zs]e',
                r'optimal',
                r'best',
                r'most\s+efficient'
            ]
        }
    
    def _load_complexity_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Load indicators for complexity assessment"""
        return {
            'high_complexity': {
                'keywords': ['prove', 'demonstrate', 'derive', 'generalize'],
                'weight': 1.5
            },
            'medium_complexity': {
                'keywords': ['calculate', 'determine', 'find', 'compare'],
                'weight': 1.0
            },
            'low_complexity': {
                'keywords': ['identify', 'list', 'state', 'name'],
                'weight': 0.5
            },
            'abstraction': {
                'keywords': ['general', 'arbitrary', 'any', 'all'],
                'weight': 1.3
            },
            'constraints': {
                'keywords': ['constraint', 'condition', 'restriction', 'limitation'],
                'weight': 1.2
            }
        }
    
    def _calculate_metrics(self, problem: Dict[str, Any]) -> ComplexityMetrics:
        """Calculate detailed complexity metrics"""
        problem_text = problem.get('text', '')
        
        # Basic counts
        variable_count = self._count_variables(problem_text)
        equation_count = self._count_equations(problem_text)
        constraint_count = self._count_constraints(problem_text)
        
        # Reasoning metrics
        reasoning_depth = self._calculate_reasoning_depth(problem)
        knowledge_dependencies = self._count_knowledge_dependencies(problem)
        inference_steps = self._estimate_inference_steps(problem)
        
        # Domain switches
        domain_switches = self._count_domain_switches(problem)
        
        # Abstraction level
        abstraction_level = self._calculate_abstraction_level(problem_text)
        
        # Semantic complexity
        semantic_complexity = 0.5  # Default
        if self.semantic_engine and problem_text:
            semantic_analysis = self.semantic_engine.analyze_problem_semantics(problem_text)
            semantic_complexity = semantic_analysis['complexity']['score']
        
        # Computational complexity
        computational_complexity = self._estimate_computational_complexity(
            variable_count, equation_count, constraint_count
        )
        
        return ComplexityMetrics(
            reasoning_depth=reasoning_depth,
            knowledge_dependencies=knowledge_dependencies,
            inference_steps=inference_steps,
            variable_count=variable_count,
            equation_count=equation_count,
            constraint_count=constraint_count,
            domain_switches=domain_switches,
            abstraction_level=abstraction_level,
            semantic_complexity=semantic_complexity,
            computational_complexity=computational_complexity
        )
    
    def _identify_reasoning_types(self, problem: Dict[str, Any]) -> List[ReasoningType]:
        """Identify types of reasoning required"""
        problem_text = problem.get('text', '').lower()
        reasoning_types = []
        
        for reasoning_type, patterns in self.reasoning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_text):
                    reasoning_types.append(reasoning_type)
                    break
        
        # Default to algebraic if no specific type identified
        if not reasoning_types:
            reasoning_types.append(ReasoningType.ALGEBRAIC)
        
        return reasoning_types
    
    def _determine_main_level(self, metrics: ComplexityMetrics, 
                            reasoning_types: List[ReasoningType]) -> ComplexityLevel:
        """Determine main complexity level"""
        # L0: Explicit problems
        if metrics.reasoning_depth == 0 and metrics.inference_steps <= 1:
            return ComplexityLevel.L0
        
        # L1: Shallow implicit
        elif metrics.reasoning_depth <= 1 and metrics.inference_steps <= 3:
            return ComplexityLevel.L1
        
        # L2: Medium implicit
        elif metrics.reasoning_depth <= 3 and metrics.inference_steps <= 10:
            return ComplexityLevel.L2
        
        # L3: Deep implicit
        else:
            return ComplexityLevel.L3
    
    def _determine_sub_level(self, main_level: ComplexityLevel, 
                           metrics: ComplexityMetrics,
                           problem: Dict[str, Any]) -> SubLevel:
        """Determine sub-level classification"""
        if main_level == ComplexityLevel.L0:
            # L0 has no sub-levels
            return SubLevel.L1_1  # Default to simplest sub-level
        
        elif main_level == ComplexityLevel.L1:
            if metrics.inference_steps <= 1:
                return SubLevel.L1_1
            elif metrics.inference_steps <= 2:
                return SubLevel.L1_2
            else:
                return SubLevel.L1_3
        
        elif main_level == ComplexityLevel.L2:
            if metrics.inference_steps <= 5:
                return SubLevel.L2_1
            elif metrics.inference_steps <= 7:
                return SubLevel.L2_2
            else:
                return SubLevel.L2_3
        
        else:  # L3
            if metrics.inference_steps <= 15:
                return SubLevel.L3_1
            elif metrics.inference_steps <= 20:
                return SubLevel.L3_2
            else:
                return SubLevel.L3_3
    
    def _calculate_confidence(self, metrics: ComplexityMetrics, 
                            problem: Dict[str, Any]) -> float:
        """Calculate classification confidence"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on clarity of indicators
        if metrics.reasoning_depth > 0:
            confidence += 0.1
        
        if metrics.inference_steps > 5:
            confidence += 0.1
        
        # Adjust based on semantic analysis if available
        if metrics.semantic_complexity > 0.7:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _generate_explanation(self, main_level: ComplexityLevel, 
                            sub_level: SubLevel,
                            metrics: ComplexityMetrics) -> str:
        """Generate human-readable explanation"""
        explanation = f"Problem classified as {main_level.value} ({sub_level.value}) based on:\n"
        
        explanation += f"- Reasoning depth: {metrics.reasoning_depth} levels\n"
        explanation += f"- Estimated inference steps: {metrics.inference_steps}\n"
        explanation += f"- Variables: {metrics.variable_count}, "
        explanation += f"Equations: {metrics.equation_count}, "
        explanation += f"Constraints: {metrics.constraint_count}\n"
        
        if metrics.domain_switches > 0:
            explanation += f"- Cross-domain reasoning required ({metrics.domain_switches} switches)\n"
        
        if metrics.abstraction_level > 0.7:
            explanation += "- High level of abstraction detected\n"
        
        return explanation
    
    def _extract_domain_factors(self, problem: Dict[str, Any], 
                              reasoning_types: List[ReasoningType]) -> Dict[str, Any]:
        """Extract domain-specific complexity factors"""
        factors = {
            'primary_domain': self._identify_primary_domain(problem),
            'reasoning_types': [rt.value for rt in reasoning_types],
            'special_techniques': self._identify_special_techniques(problem)
        }
        
        # Add domain-specific metrics
        if ReasoningType.GEOMETRIC in reasoning_types:
            factors['geometric_complexity'] = self._assess_geometric_complexity(problem)
        
        if ReasoningType.PROBABILISTIC in reasoning_types:
            factors['probabilistic_complexity'] = self._assess_probabilistic_complexity(problem)
        
        return factors
    
    def _count_variables(self, text: str) -> int:
        """Count unique variables in problem"""
        # Simple pattern for single letter variables
        variables = set(re.findall(r'\b[a-zA-Z]\b(?!\w)', text))
        return len(variables)
    
    def _count_equations(self, text: str) -> int:
        """Count equations in problem"""
        # Pattern for equations (simplified)
        equation_patterns = [
            r'=',  # Equality
            r'[<>≤≥]',  # Inequalities
        ]
        
        count = 0
        for pattern in equation_patterns:
            count += len(re.findall(pattern, text))
        
        return count
    
    def _count_constraints(self, text: str) -> int:
        """Count constraints in problem"""
        constraint_keywords = [
            'constraint', 'condition', 'restriction',
            'must', 'should', 'cannot', 'limited'
        ]
        
        count = 0
        text_lower = text.lower()
        for keyword in constraint_keywords:
            count += text_lower.count(keyword)
        
        return count
    
    def _calculate_reasoning_depth(self, problem: Dict[str, Any]) -> int:
        """Calculate depth of reasoning required"""
        # Analyze implicit relationships
        implicit_count = 0
        problem_text = problem.get('text', '')
        
        for pattern_type, patterns in self.implicit_patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_text, re.IGNORECASE):
                    implicit_count += 1
        
        # Map to reasoning depth
        if implicit_count == 0:
            return 0
        elif implicit_count <= 2:
            return 1
        elif implicit_count <= 5:
            return 2
        else:
            return 3
    
    def _count_knowledge_dependencies(self, problem: Dict[str, Any]) -> int:
        """Count external knowledge dependencies"""
        # Check for references to theorems, formulas, etc.
        knowledge_indicators = [
            'theorem', 'formula', 'principle', 'law',
            'definition', 'property', 'axiom'
        ]
        
        count = 0
        problem_text = problem.get('text', '').lower()
        for indicator in knowledge_indicators:
            count += problem_text.count(indicator)
        
        return count
    
    def _estimate_inference_steps(self, problem: Dict[str, Any]) -> int:
        """Estimate number of inference steps required"""
        # Base estimate on problem structure
        problem_text = problem.get('text', '')
        
        # Count logical connectives
        connectives = ['therefore', 'thus', 'hence', 'so', 'then', 'implies']
        step_count = 1  # At least one step
        
        for connective in connectives:
            step_count += problem_text.lower().count(connective)
        
        # Add steps for each equation/constraint
        step_count += self._count_equations(problem_text) // 2
        step_count += self._count_constraints(problem_text) // 3
        
        return step_count
    
    def _count_domain_switches(self, problem: Dict[str, Any]) -> int:
        """Count switches between mathematical domains"""
        domains = {
            'algebra': ['equation', 'variable', 'solve', 'factor'],
            'geometry': ['angle', 'triangle', 'circle', 'area'],
            'calculus': ['derivative', 'integral', 'limit', 'rate'],
            'probability': ['probability', 'chance', 'random', 'expected']
        }
        
        problem_text = problem.get('text', '').lower()
        active_domains = []
        
        for domain, keywords in domains.items():
            for keyword in keywords:
                if keyword in problem_text:
                    active_domains.append(domain)
                    break
        
        # Count transitions between domains
        return max(0, len(set(active_domains)) - 1)
    
    def _calculate_abstraction_level(self, text: str) -> float:
        """Calculate level of abstraction (0-1)"""
        abstraction_indicators = [
            'general', 'arbitrary', 'any', 'all',
            'prove', 'show that', 'demonstrate'
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        for indicator in abstraction_indicators:
            if indicator in text_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _estimate_computational_complexity(self, variables: int, 
                                         equations: int, 
                                         constraints: int) -> float:
        """Estimate computational complexity (0-1)"""
        # Simple heuristic based on problem size
        complexity = (variables * 0.1 + equations * 0.2 + constraints * 0.15) / 3
        return min(complexity, 1.0)
    
    def _identify_primary_domain(self, problem: Dict[str, Any]) -> str:
        """Identify primary mathematical domain"""
        domains_count = defaultdict(int)
        problem_text = problem.get('text', '').lower()
        
        domain_keywords = {
            'algebra': ['equation', 'solve', 'variable', 'polynomial'],
            'geometry': ['triangle', 'circle', 'angle', 'area'],
            'calculus': ['derivative', 'integral', 'limit', 'function'],
            'number_theory': ['prime', 'divisor', 'factor', 'integer'],
            'probability': ['probability', 'random', 'expected', 'chance']
        }
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                domains_count[domain] += problem_text.count(keyword)
        
        if domains_count:
            return max(domains_count, key=domains_count.get)
        return 'general'
    
    def _identify_special_techniques(self, problem: Dict[str, Any]) -> List[str]:
        """Identify special techniques that might be needed"""
        techniques = []
        problem_text = problem.get('text', '').lower()
        
        technique_patterns = {
            'substitution': r'substitute|replace|let.*=',
            'elimination': r'eliminate|remove|cancel',
            'factorization': r'factor|factori[zs]e',
            'completing_square': r'complet.*square',
            'proof_by_induction': r'induction|base case',
            'proof_by_contradiction': r'contradiction|assume.*false'
        }
        
        for technique, pattern in technique_patterns.items():
            if re.search(pattern, problem_text):
                techniques.append(technique)
        
        return techniques
    
    def _assess_geometric_complexity(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Assess geometric-specific complexity"""
        problem_text = problem.get('text', '').lower()
        
        return {
            'dimensions': 2 if '2d' in problem_text or 'plane' in problem_text else 3,
            'shapes_count': sum(1 for shape in ['triangle', 'circle', 'square', 'polygon'] 
                              if shape in problem_text),
            'requires_construction': 'construct' in problem_text or 'draw' in problem_text
        }
    
    def _assess_probabilistic_complexity(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Assess probability-specific complexity"""
        problem_text = problem.get('text', '').lower()
        
        return {
            'distribution_type': 'discrete' if 'dice' in problem_text or 'coin' in problem_text else 'continuous',
            'conditional': 'given' in problem_text or 'if' in problem_text,
            'multiple_events': problem_text.count('and') + problem_text.count('or') > 2
        }
    
    def generate_complexity_report(self, problem: Dict[str, Any]) -> str:
        """Generate detailed complexity analysis report"""
        classification = self.classify_problem(problem)
        
        report = "=== Complexity Analysis Report ===\n\n"
        report += f"Classification: {classification.main_level.value} "
        report += f"({classification.sub_level.value})\n"
        report += f"Confidence: {classification.confidence:.2%}\n\n"
        
        report += "Reasoning Types:\n"
        for rt in classification.reasoning_types:
            report += f"  - {rt.value}\n"
        
        report += f"\n{classification.explanation}\n"
        
        report += "\nDomain-Specific Factors:\n"
        for factor, value in classification.domain_specific_factors.items():
            report += f"  - {factor}: {value}\n"
        
        return report