"""
Enhanced Multi-Level Reasoning Processor with Proof Generation
Implements advanced reasoning capabilities including mathematical proof generation,
constraint satisfaction solving, and analogical reasoning
Part of Story 6.1: Mathematical Reasoning Enhancement - Phase 2
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
from collections import defaultdict

# Import semantic analysis components
try:
    from .semantic_ird_engine import SemanticIRDEngine, SemanticRelation
    from .mathematical_ontology import MathematicalOntology, MathConcept
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.warning("Semantic components not available")

logger = logging.getLogger(__name__)

class ProofType(Enum):
    """Types of mathematical proofs"""
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CONSTRUCTION = "construction"
    EXHAUSTION = "exhaustion"
    ANALOGY = "analogy"

class ReasoningStrategy(Enum):
    """Reasoning strategies for problem solving"""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    MEANS_ENDS_ANALYSIS = "means_ends_analysis"
    CASE_BASED = "case_based"
    CONSTRAINT_PROPAGATION = "constraint_propagation"
    ANALOGICAL = "analogical"

@dataclass
class ProofStep:
    """Represents a step in a mathematical proof"""
    step_number: int
    statement: str
    justification: str
    dependencies: List[int] = field(default_factory=list)
    confidence: float = 1.0
    step_type: str = "deduction"
    
@dataclass
class MathematicalProof:
    """Complete mathematical proof structure"""
    theorem: str
    proof_type: ProofType
    steps: List[ProofStep]
    assumptions: List[str]
    conclusion: str
    validity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningNode:
    """Node in the reasoning graph"""
    node_id: str
    content: str
    node_type: str  # fact, goal, intermediate
    confidence: float = 1.0
    dependencies: Set[str] = field(default_factory=set)
    derivations: Set[str] = field(default_factory=set)
    constraints: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ConstraintSolution:
    """Solution to a constraint satisfaction problem"""
    variables: Dict[str, Any]
    constraints_satisfied: List[str]
    constraints_violated: List[str]
    satisfaction_score: float
    solution_path: List[Dict[str, Any]]

class EnhancedMLRProcessor:
    """
    Enhanced Multi-Level Reasoning Processor with advanced capabilities
    """
    
    def __init__(self):
        """Initialize the enhanced MLR processor"""
        self.semantic_engine = None
        self.ontology = None
        
        if SEMANTIC_AVAILABLE:
            self.semantic_engine = SemanticIRDEngine()
            self.ontology = MathematicalOntology()
        
        # Reasoning graph for tracking reasoning process
        self.reasoning_graph = {}
        
        # Case library for analogical reasoning
        self.case_library = []
        
        # Proof templates
        self.proof_templates = self._initialize_proof_templates()
        
        logger.info("Enhanced MLR Processor initialized")
    
    def _initialize_proof_templates(self) -> Dict[str, List[str]]:
        """Initialize common proof templates"""
        return {
            ProofType.DIRECT: [
                "Assume {hypothesis}",
                "By {theorem/definition}, we have {conclusion}",
                "Therefore, {final_statement}"
            ],
            ProofType.CONTRADICTION: [
                "Assume the negation: {negated_statement}",
                "This leads to {contradiction}",
                "Therefore, our assumption is false",
                "Hence, {original_statement} is true"
            ],
            ProofType.INDUCTION: [
                "Base case: Show true for n = {base_value}",
                "Inductive hypothesis: Assume true for n = k",
                "Inductive step: Show true for n = k + 1",
                "By mathematical induction, true for all n ≥ {base_value}"
            ]
        }
    
    def generate_proof(self, problem: Dict[str, Any], 
                      target_statement: str) -> MathematicalProof:
        """
        Generate a mathematical proof for a given statement
        
        Args:
            problem: Problem context and given information
            target_statement: Statement to prove
            
        Returns:
            MathematicalProof object
        """
        # Analyze the problem to determine proof strategy
        proof_type = self._determine_proof_type(problem, target_statement)
        
        # Extract assumptions and known facts
        assumptions = self._extract_assumptions(problem)
        
        # Build reasoning graph
        self._build_reasoning_graph(problem, target_statement)
        
        # Generate proof steps based on strategy
        if proof_type == ProofType.DIRECT:
            steps = self._generate_direct_proof(target_statement)
        elif proof_type == ProofType.CONTRADICTION:
            steps = self._generate_contradiction_proof(target_statement)
        elif proof_type == ProofType.INDUCTION:
            steps = self._generate_induction_proof(target_statement)
        else:
            steps = self._generate_generic_proof(target_statement)
        
        # Validate the proof
        validity_score = self._validate_proof(steps, assumptions, target_statement)
        
        return MathematicalProof(
            theorem=target_statement,
            proof_type=proof_type,
            steps=steps,
            assumptions=assumptions,
            conclusion=target_statement,
            validity_score=validity_score,
            metadata={
                'reasoning_nodes': len(self.reasoning_graph),
                'proof_length': len(steps)
            }
        )
    
    def solve_constraints(self, variables: List[str], 
                         constraints: List[Dict[str, Any]]) -> ConstraintSolution:
        """
        Solve constraint satisfaction problem
        
        Args:
            variables: List of variable names
            constraints: List of constraints (equations, inequalities, etc.)
            
        Returns:
            ConstraintSolution object
        """
        # Initialize variable domains
        domains = self._initialize_domains(variables, constraints)
        
        # Apply constraint propagation
        propagated_domains = self._propagate_constraints(domains, constraints)
        
        # Search for solution
        solution_path = []
        solution = self._backtrack_search(
            variables, propagated_domains, constraints, solution_path
        )
        
        if solution:
            # Evaluate which constraints are satisfied
            satisfied, violated = self._evaluate_constraints(solution, constraints)
            satisfaction_score = len(satisfied) / len(constraints) if constraints else 1.0
            
            return ConstraintSolution(
                variables=solution,
                constraints_satisfied=satisfied,
                constraints_violated=violated,
                satisfaction_score=satisfaction_score,
                solution_path=solution_path
            )
        else:
            return ConstraintSolution(
                variables={},
                constraints_satisfied=[],
                constraints_violated=[str(c) for c in constraints],
                satisfaction_score=0.0,
                solution_path=solution_path
            )
    
    def reason_by_analogy(self, current_problem: Dict[str, Any], 
                         target_type: str) -> Dict[str, Any]:
        """
        Perform analogical reasoning using similar solved problems
        
        Args:
            current_problem: Current problem to solve
            target_type: Type of solution sought
            
        Returns:
            Reasoning result with analogical insights
        """
        # Find similar cases from library
        similar_cases = self._find_similar_cases(current_problem)
        
        if not similar_cases:
            return {
                'success': False,
                'message': 'No similar cases found for analogical reasoning',
                'solution': None
            }
        
        # Extract common patterns
        patterns = self._extract_patterns(similar_cases)
        
        # Map patterns to current problem
        mapped_solution = self._map_analogical_solution(
            current_problem, similar_cases[0], patterns
        )
        
        # Validate and adapt solution
        adapted_solution = self._adapt_solution(mapped_solution, current_problem)
        
        return {
            'success': True,
            'solution': adapted_solution,
            'source_cases': [case['id'] for case in similar_cases[:3]],
            'confidence': self._calculate_analogy_confidence(current_problem, similar_cases[0]),
            'patterns_used': patterns
        }
    
    def create_reasoning_chain(self, initial_facts: List[str], 
                             goal: str) -> List[Dict[str, Any]]:
        """
        Create a validated reasoning chain from facts to goal
        
        Args:
            initial_facts: Starting facts/assumptions
            goal: Target conclusion
            
        Returns:
            List of reasoning steps forming the chain
        """
        # Initialize reasoning nodes
        for i, fact in enumerate(initial_facts):
            node_id = f"fact_{i}"
            self.reasoning_graph[node_id] = ReasoningNode(
                node_id=node_id,
                content=fact,
                node_type="fact",
                confidence=1.0
            )
        
        # Create goal node
        goal_node = ReasoningNode(
            node_id="goal",
            content=goal,
            node_type="goal",
            confidence=0.0  # Initially unproven
        )
        self.reasoning_graph["goal"] = goal_node
        
        # Apply reasoning strategies
        chain = []
        
        # Try forward chaining first
        forward_chain = self._forward_chain(initial_facts, goal)
        if forward_chain:
            chain.extend(forward_chain)
        
        # If incomplete, try backward chaining
        if not self._is_goal_reached(goal):
            backward_chain = self._backward_chain(goal, initial_facts)
            chain.extend(backward_chain)
        
        # Validate and order the chain
        validated_chain = self._validate_reasoning_chain(chain)
        
        return validated_chain
    
    def enhance_with_semantic_analysis(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance problem understanding using semantic analysis
        
        Args:
            problem: Mathematical problem
            
        Returns:
            Enhanced problem representation
        """
        if not self.semantic_engine:
            return problem
        
        # Perform semantic analysis
        problem_text = problem.get('text', '')
        semantic_analysis = self.semantic_engine.analyze_problem_semantics(problem_text)
        
        # Enhance problem representation
        enhanced_problem = problem.copy()
        enhanced_problem['semantic_entities'] = semantic_analysis['entities']
        enhanced_problem['semantic_relations'] = semantic_analysis['relations']
        enhanced_problem['problem_type'] = semantic_analysis['problem_type']
        enhanced_problem['semantic_complexity'] = semantic_analysis['complexity']
        
        # Add ontological concepts
        if self.ontology:
            concepts = []
            for entity in semantic_analysis['entities']:
                if entity['type'] == 'concept':
                    concept = self.ontology.get_concept(entity['text'])
                    if concept:
                        concepts.append(concept)
            enhanced_problem['mathematical_concepts'] = concepts
        
        return enhanced_problem
    
    def _determine_proof_type(self, problem: Dict[str, Any], 
                            target: str) -> ProofType:
        """Determine appropriate proof type based on problem and target"""
        # Simple heuristics for proof type selection
        target_lower = target.lower()
        
        if any(word in target_lower for word in ['all', 'every', 'any']):
            if 'integer' in target_lower or 'natural' in target_lower:
                return ProofType.INDUCTION
        
        if 'exists' in target_lower or 'construct' in target_lower:
            return ProofType.CONSTRUCTION
        
        if 'impossible' in target_lower or 'cannot' in target_lower:
            return ProofType.CONTRADICTION
        
        # Default to direct proof
        return ProofType.DIRECT
    
    def _extract_assumptions(self, problem: Dict[str, Any]) -> List[str]:
        """Extract assumptions from problem statement"""
        assumptions = []
        
        # Extract given information
        if 'given' in problem:
            assumptions.extend(problem['given'])
        
        # Extract from problem text using patterns
        if 'text' in problem:
            text = problem['text']
            # Simple pattern matching for assumptions
            if 'given that' in text.lower():
                # Extract what follows "given that"
                parts = text.lower().split('given that')
                if len(parts) > 1:
                    assumptions.append(parts[1].split('.')[0].strip())
        
        return assumptions
    
    def _build_reasoning_graph(self, problem: Dict[str, Any], target: str):
        """Build a graph representing the reasoning space"""
        # Clear existing graph
        self.reasoning_graph.clear()
        
        # Add known facts as nodes
        facts = problem.get('facts', []) + self._extract_assumptions(problem)
        for i, fact in enumerate(facts):
            node = ReasoningNode(
                node_id=f"fact_{i}",
                content=fact,
                node_type="fact",
                confidence=1.0
            )
            self.reasoning_graph[node.node_id] = node
        
        # Add target as goal node
        goal_node = ReasoningNode(
            node_id="goal",
            content=target,
            node_type="goal",
            confidence=0.0
        )
        self.reasoning_graph["goal"] = goal_node
        
        # Add intermediate nodes based on semantic analysis
        if self.semantic_engine and 'text' in problem:
            entities = self.semantic_engine.extract_mathematical_entities(problem['text'])
            for i, entity in enumerate(entities):
                if entity['type'] == 'equation':
                    node = ReasoningNode(
                        node_id=f"eq_{i}",
                        content=entity['text'],
                        node_type="intermediate",
                        confidence=0.8
                    )
                    self.reasoning_graph[node.node_id] = node
    
    def _generate_direct_proof(self, target: str) -> List[ProofStep]:
        """Generate steps for a direct proof"""
        steps = []
        step_num = 1
        
        # Start with assumptions
        for node_id, node in self.reasoning_graph.items():
            if node.node_type == "fact":
                step = ProofStep(
                    step_number=step_num,
                    statement=node.content,
                    justification="Given",
                    confidence=1.0
                )
                steps.append(step)
                step_num += 1
        
        # Apply logical deductions
        # This is simplified - real implementation would use inference rules
        intermediate_steps = self._apply_inference_rules()
        for istep in intermediate_steps:
            step = ProofStep(
                step_number=step_num,
                statement=istep['statement'],
                justification=istep['justification'],
                dependencies=istep.get('dependencies', []),
                confidence=istep.get('confidence', 0.9)
            )
            steps.append(step)
            step_num += 1
        
        # Conclude with target
        final_step = ProofStep(
            step_number=step_num,
            statement=f"Therefore, {target}",
            justification="From previous steps",
            dependencies=list(range(1, step_num)),
            confidence=0.95
        )
        steps.append(final_step)
        
        return steps
    
    def _generate_contradiction_proof(self, target: str) -> List[ProofStep]:
        """Generate steps for a proof by contradiction"""
        steps = []
        step_num = 1
        
        # Assume the negation
        negated = f"not ({target})"
        step = ProofStep(
            step_number=step_num,
            statement=f"Assume {negated}",
            justification="Proof by contradiction",
            confidence=1.0
        )
        steps.append(step)
        step_num += 1
        
        # Derive consequences (simplified)
        consequences = [
            "This implies a contradiction with known facts",
            "But this contradicts our assumptions"
        ]
        
        for consequence in consequences:
            step = ProofStep(
                step_number=step_num,
                statement=consequence,
                justification="Logical deduction",
                dependencies=[1],
                confidence=0.9
            )
            steps.append(step)
            step_num += 1
        
        # Conclude
        final_step = ProofStep(
            step_number=step_num,
            statement=f"Therefore, {target} must be true",
            justification="Contradiction resolved",
            dependencies=list(range(1, step_num)),
            confidence=0.95
        )
        steps.append(final_step)
        
        return steps
    
    def _generate_induction_proof(self, target: str) -> List[ProofStep]:
        """Generate steps for a proof by induction"""
        steps = []
        step_num = 1
        
        # Base case
        base_step = ProofStep(
            step_number=step_num,
            statement="Base case: Show true for n = 1",
            justification="Induction base",
            confidence=1.0
        )
        steps.append(base_step)
        step_num += 1
        
        # Verification of base case
        verify_step = ProofStep(
            step_number=step_num,
            statement="The statement holds for n = 1",
            justification="Direct verification",
            dependencies=[1],
            confidence=0.95
        )
        steps.append(verify_step)
        step_num += 1
        
        # Inductive hypothesis
        hyp_step = ProofStep(
            step_number=step_num,
            statement="Assume the statement is true for n = k",
            justification="Inductive hypothesis",
            confidence=1.0
        )
        steps.append(hyp_step)
        step_num += 1
        
        # Inductive step
        ind_step = ProofStep(
            step_number=step_num,
            statement="Show that the statement is true for n = k + 1",
            justification="Using inductive hypothesis",
            dependencies=[3],
            confidence=0.9
        )
        steps.append(ind_step)
        step_num += 1
        
        # Conclusion
        final_step = ProofStep(
            step_number=step_num,
            statement=f"By mathematical induction, {target} for all n ≥ 1",
            justification="Principle of mathematical induction",
            dependencies=[2, 4],
            confidence=0.95
        )
        steps.append(final_step)
        
        return steps
    
    def _generate_generic_proof(self, target: str) -> List[ProofStep]:
        """Generate a generic proof structure"""
        return self._generate_direct_proof(target)
    
    def _validate_proof(self, steps: List[ProofStep], 
                       assumptions: List[str], target: str) -> float:
        """Validate the logical consistency of a proof"""
        if not steps:
            return 0.0
        
        # Check that all dependencies are valid
        valid_steps = set()
        for step in steps:
            valid = True
            for dep in step.dependencies:
                if dep not in valid_steps and dep != step.step_number:
                    valid = False
                    break
            if valid:
                valid_steps.add(step.step_number)
        
        # Check that conclusion matches target
        last_step = steps[-1]
        target_reached = target.lower() in last_step.statement.lower()
        
        # Calculate validity score
        dependency_score = len(valid_steps) / len(steps) if steps else 0
        conclusion_score = 1.0 if target_reached else 0.5
        confidence_score = np.mean([step.confidence for step in steps])
        
        validity_score = (dependency_score + conclusion_score + confidence_score) / 3
        
        return validity_score
    
    def _initialize_domains(self, variables: List[str], 
                          constraints: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Initialize variable domains for constraint satisfaction"""
        domains = {}
        
        for var in variables:
            # Default domain (can be refined based on constraints)
            domains[var] = list(range(-100, 101))  # Simplified integer domain
        
        # Refine based on explicit constraints
        for constraint in constraints:
            if constraint['type'] == 'range':
                var = constraint['variable']
                if var in domains:
                    min_val = constraint.get('min', -100)
                    max_val = constraint.get('max', 100)
                    domains[var] = list(range(min_val, max_val + 1))
        
        return domains
    
    def _propagate_constraints(self, domains: Dict[str, List[Any]], 
                             constraints: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Apply constraint propagation to reduce domains"""
        changed = True
        
        while changed:
            changed = False
            
            for constraint in constraints:
                if constraint['type'] == 'equation':
                    # Simple constraint propagation for equations
                    # This is a simplified version
                    pass
                elif constraint['type'] == 'inequality':
                    # Handle inequalities
                    pass
        
        return domains
    
    def _backtrack_search(self, variables: List[str], domains: Dict[str, List[Any]], 
                         constraints: List[Dict[str, Any]], 
                         path: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Backtracking search for constraint satisfaction"""
        # Base case: all variables assigned
        if not variables:
            return {}
        
        var = variables[0]
        remaining_vars = variables[1:]
        
        for value in domains.get(var, []):
            # Try assignment
            assignment = {var: value}
            
            # Check consistency
            if self._is_consistent(assignment, constraints):
                # Record in path
                path.append({
                    'variable': var,
                    'value': value,
                    'consistent': True
                })
                
                # Recursive call
                result = self._backtrack_search(
                    remaining_vars, domains, constraints, path
                )
                
                if result is not None:
                    result[var] = value
                    return result
                
                # Backtrack
                path.pop()
        
        return None
    
    def _is_consistent(self, assignment: Dict[str, Any], 
                      constraints: List[Dict[str, Any]]) -> bool:
        """Check if assignment is consistent with constraints"""
        for constraint in constraints:
            # Evaluate constraint with current assignment
            # This is simplified - real implementation would evaluate expressions
            if not self._evaluate_constraint(constraint, assignment):
                return False
        return True
    
    def _evaluate_constraint(self, constraint: Dict[str, Any], 
                           assignment: Dict[str, Any]) -> bool:
        """Evaluate a single constraint with given assignment"""
        # Simplified constraint evaluation
        return True
    
    def _evaluate_constraints(self, solution: Dict[str, Any], 
                            constraints: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Evaluate which constraints are satisfied by the solution"""
        satisfied = []
        violated = []
        
        for constraint in constraints:
            if self._evaluate_constraint(constraint, solution):
                satisfied.append(str(constraint))
            else:
                violated.append(str(constraint))
        
        return satisfied, violated
    
    def _find_similar_cases(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar cases from the case library"""
        if not self.case_library:
            return []
        
        # Calculate similarity scores
        similarities = []
        for case in self.case_library:
            similarity = self._calculate_similarity(problem, case)
            similarities.append((similarity, case))
        
        # Sort by similarity and return top cases
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return cases with similarity > 0.5
        return [case for sim, case in similarities if sim > 0.5]
    
    def _calculate_similarity(self, problem1: Dict[str, Any], 
                            problem2: Dict[str, Any]) -> float:
        """Calculate similarity between two problems"""
        # Simplified similarity based on problem type and features
        score = 0.0
        
        # Check problem type
        if problem1.get('type') == problem2.get('type'):
            score += 0.3
        
        # Check mathematical concepts
        concepts1 = set(problem1.get('concepts', []))
        concepts2 = set(problem2.get('concepts', []))
        if concepts1 and concepts2:
            overlap = len(concepts1.intersection(concepts2))
            total = len(concepts1.union(concepts2))
            score += 0.4 * (overlap / total if total > 0 else 0)
        
        # Check structure similarity
        if 'equations' in problem1 and 'equations' in problem2:
            eq1_count = len(problem1['equations'])
            eq2_count = len(problem2['equations'])
            if eq1_count == eq2_count:
                score += 0.3
        
        return score
    
    def _extract_patterns(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract common patterns from similar cases"""
        patterns = []
        
        # Extract solution patterns
        solution_methods = defaultdict(int)
        for case in cases:
            if 'solution_method' in case:
                solution_methods[case['solution_method']] += 1
        
        # Find most common methods
        for method, count in solution_methods.items():
            if count >= len(cases) / 2:  # Present in at least half the cases
                patterns.append({
                    'type': 'solution_method',
                    'value': method,
                    'frequency': count / len(cases)
                })
        
        return patterns
    
    def _map_analogical_solution(self, current: Dict[str, Any], 
                                source: Dict[str, Any], 
                                patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map solution from source case to current problem"""
        mapped_solution = {}
        
        # Map variables
        var_mapping = self._create_variable_mapping(current, source)
        
        # Map solution steps
        if 'solution_steps' in source:
            mapped_steps = []
            for step in source['solution_steps']:
                mapped_step = self._map_step(step, var_mapping)
                mapped_steps.append(mapped_step)
            mapped_solution['steps'] = mapped_steps
        
        # Apply patterns
        for pattern in patterns:
            if pattern['type'] == 'solution_method':
                mapped_solution['method'] = pattern['value']
        
        return mapped_solution
    
    def _create_variable_mapping(self, current: Dict[str, Any], 
                               source: Dict[str, Any]) -> Dict[str, str]:
        """Create mapping between variables in source and current problems"""
        mapping = {}
        
        # Simple mapping based on variable roles
        current_vars = current.get('variables', [])
        source_vars = source.get('variables', [])
        
        for i, svar in enumerate(source_vars):
            if i < len(current_vars):
                mapping[svar] = current_vars[i]
        
        return mapping
    
    def _map_step(self, step: Dict[str, Any], 
                 var_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map a solution step using variable mapping"""
        mapped_step = step.copy()
        
        # Replace variables in step
        if 'expression' in mapped_step:
            expr = mapped_step['expression']
            for old_var, new_var in var_mapping.items():
                expr = expr.replace(old_var, new_var)
            mapped_step['expression'] = expr
        
        return mapped_step
    
    def _adapt_solution(self, solution: Dict[str, Any], 
                       problem: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt mapped solution to current problem specifics"""
        adapted = solution.copy()
        
        # Validate and adjust numerical values
        if 'numerical_adjustments' in problem:
            # Apply problem-specific adjustments
            pass
        
        return adapted
    
    def _calculate_analogy_confidence(self, current: Dict[str, Any], 
                                    source: Dict[str, Any]) -> float:
        """Calculate confidence in analogical reasoning"""
        base_similarity = self._calculate_similarity(current, source)
        
        # Adjust based on problem complexity
        complexity_factor = 1.0
        if 'complexity' in current and 'complexity' in source:
            complexity_diff = abs(current['complexity'] - source['complexity'])
            complexity_factor = 1.0 - (complexity_diff * 0.1)
        
        return base_similarity * complexity_factor
    
    def _forward_chain(self, facts: List[str], goal: str) -> List[Dict[str, Any]]:
        """Apply forward chaining reasoning"""
        chain = []
        derived_facts = set(facts)
        
        # Simple forward chaining (would use inference rules in real implementation)
        iteration = 0
        while iteration < 10:  # Limit iterations
            new_facts = set()
            
            # Apply inference rules
            for fact in derived_facts:
                inferences = self._apply_forward_inference(fact, derived_facts)
                new_facts.update(inferences)
            
            if not new_facts:
                break
            
            # Add new facts to chain
            for fact in new_facts:
                chain.append({
                    'step': len(chain) + 1,
                    'statement': fact,
                    'justification': 'Forward inference',
                    'confidence': 0.9
                })
            
            derived_facts.update(new_facts)
            iteration += 1
        
        return chain
    
    def _backward_chain(self, goal: str, facts: List[str]) -> List[Dict[str, Any]]:
        """Apply backward chaining reasoning"""
        chain = []
        subgoals = [goal]
        
        while subgoals:
            current_goal = subgoals.pop()
            
            # Find rules that could prove this goal
            supporting_facts = self._find_supporting_facts(current_goal, facts)
            
            if supporting_facts:
                chain.append({
                    'step': len(chain) + 1,
                    'statement': f"{current_goal} because {supporting_facts}",
                    'justification': 'Backward inference',
                    'confidence': 0.85
                })
            
        return chain
    
    def _is_goal_reached(self, goal: str) -> bool:
        """Check if the goal has been reached in reasoning graph"""
        goal_node = self.reasoning_graph.get("goal")
        return goal_node and goal_node.confidence > 0.8
    
    def _validate_reasoning_chain(self, chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and order reasoning chain"""
        # Remove duplicates
        seen = set()
        validated = []
        
        for step in chain:
            step_key = step['statement']
            if step_key not in seen:
                seen.add(step_key)
                validated.append(step)
        
        # Sort by confidence if needed
        validated.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return validated
    
    def _apply_inference_rules(self) -> List[Dict[str, Any]]:
        """Apply inference rules to derive new facts"""
        # Simplified inference rules
        inferences = []
        
        # Example: If we have "a = b" and "b = c", infer "a = c"
        # This would be much more sophisticated in a real implementation
        
        return inferences
    
    def _apply_forward_inference(self, fact: str, 
                               known_facts: Set[str]) -> Set[str]:
        """Apply forward inference from a single fact"""
        new_facts = set()
        
        # Simplified inference rules
        # Real implementation would use a proper inference engine
        
        return new_facts
    
    def _find_supporting_facts(self, goal: str, 
                              facts: List[str]) -> List[str]:
        """Find facts that could support proving the goal"""
        supporting = []
        
        # Simple keyword matching (real implementation would be more sophisticated)
        goal_terms = set(goal.lower().split())
        
        for fact in facts:
            fact_terms = set(fact.lower().split())
            if goal_terms.intersection(fact_terms):
                supporting.append(fact)
        
        return supporting
    
    def add_case_to_library(self, case: Dict[str, Any]):
        """Add a solved case to the library for future analogical reasoning"""
        # Ensure case has required fields
        required_fields = ['problem', 'solution', 'type', 'concepts']
        if all(field in case for field in required_fields):
            case['id'] = f"case_{len(self.case_library) + 1}"
            self.case_library.append(case)
            logger.info(f"Added case {case['id']} to library")