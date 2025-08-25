"""
Mathematical Concept Ontology and Knowledge Base
Comprehensive knowledge base for mathematical concepts, relationships, and patterns
Part of Story 6.1: Mathematical Reasoning Enhancement - Phase 2
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class ConceptCategory(Enum):
    """Categories of mathematical concepts"""
    # Core Mathematics
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    CALCULUS = "calculus"
    TRIGONOMETRY = "trigonometry"
    STATISTICS = "statistics"
    NUMBER_THEORY = "number_theory"
    
    # Applied Mathematics
    PHYSICS = "physics"
    ENGINEERING = "engineering"
    ECONOMICS = "economics"
    
    # Meta Categories
    OPERATION = "operation"
    RELATION = "relation"
    PROPERTY = "property"
    THEOREM = "theorem"
    METHOD = "method"

@dataclass
class ConceptRelationship:
    """Represents a relationship between mathematical concepts"""
    target: str
    relationship_type: str
    strength: float = 1.0
    bidirectional: bool = False
    conditions: Optional[List[str]] = None

@dataclass
class MathConcept:
    """Comprehensive mathematical concept representation"""
    name: str
    category: ConceptCategory
    subcategory: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[ConceptRelationship] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    complexity_level: int = 1  # 1-5 scale

class MathematicalOntology:
    """
    Comprehensive mathematical ontology and knowledge base
    """
    
    def __init__(self):
        """Initialize the mathematical ontology"""
        self.concepts: Dict[str, MathConcept] = {}
        self.concept_index: Dict[str, Set[str]] = {}  # Maps aliases to concept names
        self.relationship_types = self._define_relationship_types()
        
        # Build the ontology
        self._build_algebra_concepts()
        self._build_geometry_concepts()
        self._build_calculus_concepts()
        self._build_trigonometry_concepts()
        self._build_physics_concepts()
        self._build_operations_and_relations()
        
        # Create indices for efficient lookup
        self._build_indices()
        
        logger.info(f"Mathematical ontology initialized with {len(self.concepts)} concepts")
    
    def _define_relationship_types(self) -> Dict[str, str]:
        """Define types of relationships between concepts"""
        return {
            'is_a': 'Inheritance relationship',
            'part_of': 'Composition relationship',
            'used_in': 'Usage relationship',
            'derived_from': 'Derivation relationship',
            'equivalent_to': 'Equivalence relationship',
            'opposite_of': 'Opposition relationship',
            'generalizes': 'Generalization relationship',
            'specializes': 'Specialization relationship',
            'requires': 'Prerequisite relationship',
            'implies': 'Implication relationship',
            'transforms_to': 'Transformation relationship',
            'proportional_to': 'Proportionality relationship'
        }
    
    def _build_algebra_concepts(self):
        """Build algebraic concepts"""
        # Variables and unknowns
        self.add_concept(MathConcept(
            name="variable",
            category=ConceptCategory.ALGEBRA,
            subcategory="basic",
            aliases=["unknown", "parameter", "x", "y", "z"],
            definition="A symbol representing an unknown or changing quantity",
            properties={"can_be_solved": True, "symbolic": True},
            complexity_level=1
        ))
        
        # Equations
        self.add_concept(MathConcept(
            name="equation",
            category=ConceptCategory.ALGEBRA,
            subcategory="basic",
            aliases=["equality", "formula"],
            definition="A mathematical statement asserting equality between expressions",
            properties={"has_solution": True, "balance_required": True},
            relationships=[
                ConceptRelationship("variable", "used_in", 0.9),
                ConceptRelationship("solution", "produces", 1.0)
            ],
            complexity_level=1
        ))
        
        # Linear equations
        self.add_concept(MathConcept(
            name="linear_equation",
            category=ConceptCategory.ALGEBRA,
            subcategory="equations",
            aliases=["first_degree_equation", "ax+b=c"],
            definition="An equation of degree one",
            properties={"degree": 1, "graph": "straight_line"},
            relationships=[
                ConceptRelationship("equation", "is_a", 1.0),
                ConceptRelationship("line", "represents", 0.8)
            ],
            formulas=["ax + b = c", "y = mx + b"],
            complexity_level=2
        ))
        
        # Quadratic equations
        self.add_concept(MathConcept(
            name="quadratic_equation",
            category=ConceptCategory.ALGEBRA,
            subcategory="equations",
            aliases=["second_degree_equation", "ax²+bx+c=0"],
            definition="An equation of degree two",
            properties={"degree": 2, "graph": "parabola", "max_solutions": 2},
            relationships=[
                ConceptRelationship("equation", "is_a", 1.0),
                ConceptRelationship("parabola", "represents", 0.9),
                ConceptRelationship("discriminant", "uses", 0.8)
            ],
            formulas=["ax² + bx + c = 0", "x = (-b ± √(b²-4ac))/2a"],
            complexity_level=3
        ))
        
        # Systems of equations
        self.add_concept(MathConcept(
            name="system_of_equations",
            category=ConceptCategory.ALGEBRA,
            subcategory="equations",
            aliases=["simultaneous_equations", "equation_system"],
            definition="Multiple equations to be solved together",
            properties={"multiple_unknowns": True, "solution_methods": ["substitution", "elimination", "matrix"]},
            relationships=[
                ConceptRelationship("equation", "part_of", 1.0),
                ConceptRelationship("matrix", "used_in", 0.7)
            ],
            complexity_level=3
        ))
        
        # Polynomials
        self.add_concept(MathConcept(
            name="polynomial",
            category=ConceptCategory.ALGEBRA,
            subcategory="expressions",
            aliases=["polynomial_expression"],
            definition="An expression consisting of variables and coefficients with non-negative integer exponents",
            properties={"has_degree": True, "continuous": True},
            relationships=[
                ConceptRelationship("monomial", "part_of", 1.0),
                ConceptRelationship("coefficient", "uses", 1.0)
            ],
            formulas=["a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0"],
            complexity_level=2
        ))
        
        # Factorization
        self.add_concept(MathConcept(
            name="factorization",
            category=ConceptCategory.ALGEBRA,
            subcategory="operations",
            aliases=["factoring", "factor"],
            definition="Expressing an expression as a product of factors",
            properties={"inverse_of": "expansion", "simplifies": True},
            relationships=[
                ConceptRelationship("polynomial", "transforms_to", 0.9),
                ConceptRelationship("prime_factorization", "generalizes", 0.7)
            ],
            complexity_level=3
        ))
        
        # Inequalities
        self.add_concept(MathConcept(
            name="inequality",
            category=ConceptCategory.ALGEBRA,
            subcategory="relations",
            aliases=["inequation"],
            definition="A relation expressing that one quantity is greater or less than another",
            properties={"has_solution_set": True, "direction_matters": True},
            relationships=[
                ConceptRelationship("equation", "generalizes", 0.8),
                ConceptRelationship("interval", "produces", 0.9)
            ],
            formulas=["a < b", "a ≤ b", "a > b", "a ≥ b"],
            complexity_level=2
        ))
    
    def _build_geometry_concepts(self):
        """Build geometric concepts"""
        # Basic shapes - Triangle
        self.add_concept(MathConcept(
            name="triangle",
            category=ConceptCategory.GEOMETRY,
            subcategory="shapes",
            aliases=["triangular", "three-sided"],
            definition="A polygon with three edges and three vertices",
            properties={"sides": 3, "angles": 3, "angle_sum": 180},
            relationships=[
                ConceptRelationship("polygon", "is_a", 1.0),
                ConceptRelationship("angle", "part_of", 1.0),
                ConceptRelationship("side", "part_of", 1.0)
            ],
            formulas=["Area = 0.5 * base * height", "Perimeter = a + b + c"],
            complexity_level=1
        ))
        
        # Circle
        self.add_concept(MathConcept(
            name="circle",
            category=ConceptCategory.GEOMETRY,
            subcategory="shapes",
            aliases=["circular", "round"],
            definition="A shape consisting of all points equidistant from a center",
            properties={"constant_radius": True, "infinite_symmetry": True},
            relationships=[
                ConceptRelationship("radius", "part_of", 1.0),
                ConceptRelationship("diameter", "part_of", 1.0),
                ConceptRelationship("pi", "uses", 1.0)
            ],
            formulas=["Area = πr²", "Circumference = 2πr"],
            complexity_level=1
        ))
        
        # Rectangle
        self.add_concept(MathConcept(
            name="rectangle",
            category=ConceptCategory.GEOMETRY,
            subcategory="shapes",
            aliases=["rectangular", "oblong"],
            definition="A quadrilateral with four right angles",
            properties={"sides": 4, "right_angles": 4, "parallel_sides": 2},
            relationships=[
                ConceptRelationship("quadrilateral", "is_a", 1.0),
                ConceptRelationship("square", "generalizes", 0.9),
                ConceptRelationship("parallelogram", "is_a", 1.0)
            ],
            formulas=["Area = length × width", "Perimeter = 2(length + width)"],
            complexity_level=1
        ))
        
        # Angles
        self.add_concept(MathConcept(
            name="angle",
            category=ConceptCategory.GEOMETRY,
            subcategory="measurements",
            aliases=["angular_measure"],
            definition="The figure formed by two rays sharing a common endpoint",
            properties={"measured_in": ["degrees", "radians"], "has_vertex": True},
            relationships=[
                ConceptRelationship("vertex", "part_of", 1.0),
                ConceptRelationship("ray", "part_of", 1.0)
            ],
            complexity_level=1
        ))
        
        # Area
        self.add_concept(MathConcept(
            name="area",
            category=ConceptCategory.GEOMETRY,
            subcategory="measurements",
            aliases=["surface_area", "region"],
            definition="The amount of space inside a 2D shape",
            properties={"dimension": 2, "unit": "square_units"},
            relationships=[
                ConceptRelationship("shape", "property_of", 1.0),
                ConceptRelationship("integral", "calculated_by", 0.8)
            ],
            complexity_level=2
        ))
        
        # Volume
        self.add_concept(MathConcept(
            name="volume",
            category=ConceptCategory.GEOMETRY,
            subcategory="measurements",
            aliases=["capacity", "3d_space"],
            definition="The amount of space inside a 3D shape",
            properties={"dimension": 3, "unit": "cubic_units"},
            relationships=[
                ConceptRelationship("solid", "property_of", 1.0),
                ConceptRelationship("integral", "calculated_by", 0.9)
            ],
            complexity_level=2
        ))
        
        # Pythagorean theorem
        self.add_concept(MathConcept(
            name="pythagorean_theorem",
            category=ConceptCategory.GEOMETRY,
            subcategory="theorems",
            aliases=["pythagoras", "right_triangle_theorem"],
            definition="In a right triangle, the square of the hypotenuse equals the sum of squares of the other sides",
            properties={"applies_to": "right_triangles", "fundamental": True},
            relationships=[
                ConceptRelationship("right_triangle", "applies_to", 1.0),
                ConceptRelationship("hypotenuse", "calculates", 1.0)
            ],
            formulas=["a² + b² = c²"],
            complexity_level=2
        ))
    
    def _build_calculus_concepts(self):
        """Build calculus concepts"""
        # Derivative
        self.add_concept(MathConcept(
            name="derivative",
            category=ConceptCategory.CALCULUS,
            subcategory="differential",
            aliases=["differentiation", "rate_of_change", "d/dx"],
            definition="The instantaneous rate of change of a function",
            properties={"measures": "change", "linear_approximation": True},
            relationships=[
                ConceptRelationship("function", "applies_to", 1.0),
                ConceptRelationship("tangent", "represents", 0.9),
                ConceptRelationship("integral", "opposite_of", 1.0)
            ],
            formulas=["f'(x) = lim(h→0) [f(x+h) - f(x)]/h"],
            applications=["velocity", "acceleration", "optimization"],
            complexity_level=4
        ))
        
        # Integral
        self.add_concept(MathConcept(
            name="integral",
            category=ConceptCategory.CALCULUS,
            subcategory="integral",
            aliases=["integration", "antiderivative", "∫"],
            definition="The accumulation of quantities or area under a curve",
            properties={"accumulates": True, "inverse_of_derivative": True},
            relationships=[
                ConceptRelationship("function", "applies_to", 1.0),
                ConceptRelationship("area", "calculates", 0.9),
                ConceptRelationship("derivative", "opposite_of", 1.0)
            ],
            formulas=["∫f(x)dx = F(x) + C"],
            applications=["area", "volume", "work", "probability"],
            complexity_level=4
        ))
        
        # Limit
        self.add_concept(MathConcept(
            name="limit",
            category=ConceptCategory.CALCULUS,
            subcategory="basic",
            aliases=["limiting_value", "approaches", "lim"],
            definition="The value a function approaches as input approaches a value",
            properties={"fundamental": True, "defines_continuity": True},
            relationships=[
                ConceptRelationship("function", "applies_to", 1.0),
                ConceptRelationship("continuity", "defines", 0.9),
                ConceptRelationship("derivative", "used_in", 1.0)
            ],
            formulas=["lim(x→a) f(x) = L"],
            complexity_level=3
        ))
        
        # Function
        self.add_concept(MathConcept(
            name="function",
            category=ConceptCategory.CALCULUS,
            subcategory="basic",
            aliases=["mapping", "f(x)", "transformation"],
            definition="A relation that assigns exactly one output to each input",
            properties={"has_domain": True, "has_range": True, "deterministic": True},
            relationships=[
                ConceptRelationship("variable", "uses", 1.0),
                ConceptRelationship("graph", "represents", 0.9)
            ],
            complexity_level=2
        ))
        
        # Continuity
        self.add_concept(MathConcept(
            name="continuity",
            category=ConceptCategory.CALCULUS,
            subcategory="properties",
            aliases=["continuous", "unbroken"],
            definition="A function with no breaks, jumps, or holes",
            properties={"no_discontinuities": True, "limit_exists": True},
            relationships=[
                ConceptRelationship("limit", "defined_by", 1.0),
                ConceptRelationship("differentiability", "required_for", 0.9)
            ],
            complexity_level=3
        ))
    
    def _build_trigonometry_concepts(self):
        """Build trigonometry concepts"""
        # Sine
        self.add_concept(MathConcept(
            name="sine",
            category=ConceptCategory.TRIGONOMETRY,
            subcategory="functions",
            aliases=["sin", "opposite/hypotenuse"],
            definition="Ratio of opposite side to hypotenuse in a right triangle",
            properties={"periodic": True, "period": "2π", "range": [-1, 1]},
            relationships=[
                ConceptRelationship("right_triangle", "derived_from", 0.9),
                ConceptRelationship("unit_circle", "defined_on", 1.0),
                ConceptRelationship("cosine", "complement_of", 1.0)
            ],
            formulas=["sin(θ) = opposite/hypotenuse"],
            complexity_level=2
        ))
        
        # Cosine
        self.add_concept(MathConcept(
            name="cosine",
            category=ConceptCategory.TRIGONOMETRY,
            subcategory="functions",
            aliases=["cos", "adjacent/hypotenuse"],
            definition="Ratio of adjacent side to hypotenuse in a right triangle",
            properties={"periodic": True, "period": "2π", "range": [-1, 1]},
            relationships=[
                ConceptRelationship("right_triangle", "derived_from", 0.9),
                ConceptRelationship("unit_circle", "defined_on", 1.0),
                ConceptRelationship("sine", "complement_of", 1.0)
            ],
            formulas=["cos(θ) = adjacent/hypotenuse"],
            complexity_level=2
        ))
        
        # Tangent
        self.add_concept(MathConcept(
            name="tangent",
            category=ConceptCategory.TRIGONOMETRY,
            subcategory="functions",
            aliases=["tan", "opposite/adjacent"],
            definition="Ratio of opposite side to adjacent side in a right triangle",
            properties={"periodic": True, "period": "π", "unbounded": True},
            relationships=[
                ConceptRelationship("sine", "equals", 1.0, conditions=["tan = sin/cos"]),
                ConceptRelationship("cosine", "uses", 1.0)
            ],
            formulas=["tan(θ) = opposite/adjacent = sin(θ)/cos(θ)"],
            complexity_level=2
        ))
        
        # Trigonometric identities
        self.add_concept(MathConcept(
            name="trigonometric_identity",
            category=ConceptCategory.TRIGONOMETRY,
            subcategory="identities",
            aliases=["trig_identity"],
            definition="An equation involving trigonometric functions that is true for all values",
            properties={"always_true": True, "useful_for_simplification": True},
            relationships=[
                ConceptRelationship("sine", "involves", 1.0),
                ConceptRelationship("cosine", "involves", 1.0)
            ],
            formulas=["sin²θ + cos²θ = 1", "sin(2θ) = 2sin(θ)cos(θ)"],
            complexity_level=3
        ))
    
    def _build_physics_concepts(self):
        """Build physics-related mathematical concepts"""
        # Velocity
        self.add_concept(MathConcept(
            name="velocity",
            category=ConceptCategory.PHYSICS,
            subcategory="kinematics",
            aliases=["speed_with_direction", "v"],
            definition="Rate of change of position with respect to time",
            properties={"vector": True, "has_magnitude": True, "has_direction": True},
            relationships=[
                ConceptRelationship("derivative", "calculated_by", 1.0),
                ConceptRelationship("position", "derivative_of", 1.0),
                ConceptRelationship("acceleration", "integrated_to", 1.0)
            ],
            formulas=["v = dx/dt", "v = v₀ + at"],
            complexity_level=2
        ))
        
        # Acceleration
        self.add_concept(MathConcept(
            name="acceleration",
            category=ConceptCategory.PHYSICS,
            subcategory="kinematics",
            aliases=["a", "rate_of_velocity_change"],
            definition="Rate of change of velocity with respect to time",
            properties={"vector": True, "causes_force": True},
            relationships=[
                ConceptRelationship("velocity", "derivative_of", 1.0),
                ConceptRelationship("force", "proportional_to", 1.0)
            ],
            formulas=["a = dv/dt", "F = ma"],
            complexity_level=2
        ))
        
        # Force
        self.add_concept(MathConcept(
            name="force",
            category=ConceptCategory.PHYSICS,
            subcategory="dynamics",
            aliases=["F", "push_or_pull"],
            definition="An interaction that causes acceleration",
            properties={"vector": True, "measured_in": "Newtons"},
            relationships=[
                ConceptRelationship("mass", "proportional_to", 1.0),
                ConceptRelationship("acceleration", "proportional_to", 1.0)
            ],
            formulas=["F = ma", "F = -kx (Hooke's law)"],
            complexity_level=2
        ))
        
        # Energy
        self.add_concept(MathConcept(
            name="energy",
            category=ConceptCategory.PHYSICS,
            subcategory="dynamics",
            aliases=["E", "capacity_to_do_work"],
            definition="The capacity to do work or cause change",
            properties={"conserved": True, "scalar": True, "measured_in": "Joules"},
            relationships=[
                ConceptRelationship("work", "equivalent_to", 1.0),
                ConceptRelationship("mass", "related_to", 0.9)
            ],
            formulas=["KE = 0.5mv²", "PE = mgh", "E = mc²"],
            complexity_level=3
        ))
    
    def _build_operations_and_relations(self):
        """Build mathematical operations and relations"""
        # Addition
        self.add_concept(MathConcept(
            name="addition",
            category=ConceptCategory.OPERATION,
            subcategory="arithmetic",
            aliases=["+", "plus", "sum"],
            definition="Combining quantities to find their total",
            properties={"commutative": True, "associative": True, "identity": 0},
            relationships=[
                ConceptRelationship("subtraction", "opposite_of", 1.0),
                ConceptRelationship("sum", "produces", 1.0)
            ],
            complexity_level=1
        ))
        
        # Multiplication
        self.add_concept(MathConcept(
            name="multiplication",
            category=ConceptCategory.OPERATION,
            subcategory="arithmetic",
            aliases=["×", "*", "times", "product"],
            definition="Repeated addition or scaling operation",
            properties={"commutative": True, "associative": True, "identity": 1},
            relationships=[
                ConceptRelationship("division", "opposite_of", 1.0),
                ConceptRelationship("area", "calculates", 0.8)
            ],
            complexity_level=1
        ))
        
        # Equality
        self.add_concept(MathConcept(
            name="equality",
            category=ConceptCategory.RELATION,
            subcategory="comparison",
            aliases=["=", "equals", "is equal to"],
            definition="A relation stating two expressions have the same value",
            properties={"reflexive": True, "symmetric": True, "transitive": True},
            relationships=[
                ConceptRelationship("equation", "part_of", 1.0),
                ConceptRelationship("inequality", "opposite_of", 0.8)
            ],
            complexity_level=1
        ))
        
        # Proportionality
        self.add_concept(MathConcept(
            name="proportionality",
            category=ConceptCategory.RELATION,
            subcategory="comparison",
            aliases=["proportional", "∝", "varies as"],
            definition="A relationship where one quantity is a constant multiple of another",
            properties={"linear": True, "scalable": True},
            relationships=[
                ConceptRelationship("ratio", "uses", 1.0),
                ConceptRelationship("linear_equation", "represents", 0.9)
            ],
            formulas=["y = kx", "y/x = k (constant)"],
            complexity_level=2
        ))
    
    def add_concept(self, concept: MathConcept):
        """Add a concept to the ontology"""
        self.concepts[concept.name] = concept
        
        # Add to index
        self.concept_index[concept.name] = {concept.name}
        for alias in concept.aliases:
            if alias not in self.concept_index:
                self.concept_index[alias] = set()
            self.concept_index[alias].add(concept.name)
    
    def _build_indices(self):
        """Build additional indices for efficient lookup"""
        # Category index
        self.category_index = {}
        for name, concept in self.concepts.items():
            category = concept.category.value
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(name)
        
        # Complexity index
        self.complexity_index = {}
        for name, concept in self.concepts.items():
            level = concept.complexity_level
            if level not in self.complexity_index:
                self.complexity_index[level] = []
            self.complexity_index[level].append(name)
    
    def get_concept(self, name_or_alias: str) -> Optional[MathConcept]:
        """Get a concept by name or alias"""
        # Direct lookup
        if name_or_alias in self.concepts:
            return self.concepts[name_or_alias]
        
        # Alias lookup
        if name_or_alias in self.concept_index:
            concept_names = self.concept_index[name_or_alias]
            if concept_names:
                # Return the first matching concept
                return self.concepts[list(concept_names)[0]]
        
        return None
    
    def get_related_concepts(self, concept_name: str, 
                           relationship_type: Optional[str] = None) -> List[Tuple[str, ConceptRelationship]]:
        """Get concepts related to the given concept"""
        concept = self.get_concept(concept_name)
        if not concept:
            return []
        
        related = []
        for rel in concept.relationships:
            if relationship_type is None or rel.relationship_type == relationship_type:
                related.append((rel.target, rel))
        
        return related
    
    def get_concepts_by_category(self, category: ConceptCategory) -> List[MathConcept]:
        """Get all concepts in a category"""
        category_name = category.value
        if category_name in self.category_index:
            return [self.concepts[name] for name in self.category_index[category_name]]
        return []
    
    def get_concepts_by_complexity(self, min_level: int = 1, 
                                  max_level: int = 5) -> List[MathConcept]:
        """Get concepts within a complexity range"""
        concepts = []
        for level in range(min_level, max_level + 1):
            if level in self.complexity_index:
                concepts.extend([self.concepts[name] for name in self.complexity_index[level]])
        return concepts
    
    def find_path(self, start_concept: str, end_concept: str, 
                 max_depth: int = 5) -> Optional[List[str]]:
        """Find a path between two concepts through relationships"""
        start = self.get_concept(start_concept)
        end = self.get_concept(end_concept)
        
        if not start or not end:
            return None
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(start.name, [start.name])])
        visited = {start.name}
        
        while queue and max_depth > 0:
            current_name, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            current = self.concepts[current_name]
            
            for rel in current.relationships:
                if rel.target == end.name:
                    return path + [end.name]
                
                if rel.target not in visited and rel.target in self.concepts:
                    visited.add(rel.target)
                    queue.append((rel.target, path + [rel.target]))
        
        return None
    
    def export_to_json(self, filepath: str):
        """Export the ontology to a JSON file"""
        data = {
            'concepts': {},
            'relationship_types': self.relationship_types
        }
        
        for name, concept in self.concepts.items():
            data['concepts'][name] = {
                'category': concept.category.value,
                'subcategory': concept.subcategory,
                'aliases': concept.aliases,
                'definition': concept.definition,
                'properties': concept.properties,
                'relationships': [
                    {
                        'target': rel.target,
                        'type': rel.relationship_type,
                        'strength': rel.strength,
                        'bidirectional': rel.bidirectional,
                        'conditions': rel.conditions
                    }
                    for rel in concept.relationships
                ],
                'formulas': concept.formulas,
                'examples': concept.examples,
                'prerequisites': concept.prerequisites,
                'applications': concept.applications,
                'complexity_level': concept.complexity_level
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_concept_summary(self, concept_name: str) -> Dict[str, Any]:
        """Get a summary of a concept and its relationships"""
        concept = self.get_concept(concept_name)
        if not concept:
            return {}
        
        # Get related concepts
        related = self.get_related_concepts(concept_name)
        
        # Find concepts that reference this one
        referenced_by = []
        for name, other_concept in self.concepts.items():
            for rel in other_concept.relationships:
                if rel.target == concept.name:
                    referenced_by.append((name, rel))
        
        return {
            'concept': concept,
            'related_to': related,
            'referenced_by': referenced_by,
            'category_peers': [
                name for name in self.category_index.get(concept.category.value, [])
                if name != concept.name
            ]
        }