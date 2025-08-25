"""
Semantic IRD Engine with Transformer-based Analysis
Implements advanced semantic understanding for implicit relation discovery
Part of Story 6.1: Mathematical Reasoning Enhancement - Phase 2
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict

# For transformer integration (using sentence-transformers as lightweight option)
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)

@dataclass
class MathematicalConcept:
    """Represents a mathematical concept with semantic properties"""
    name: str
    category: str  # algebra, geometry, calculus, etc.
    subcategory: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

@dataclass
class SemanticRelation:
    """Represents a semantic relationship between mathematical entities"""
    source: str
    target: str
    relation_type: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    semantic_similarity: float = 0.0

class RelationType(Enum):
    """Types of mathematical relationships"""
    EQUALITY = "equality"
    PROPORTION = "proportion"
    TRANSFORMATION = "transformation"
    DEPENDENCY = "dependency"
    CONSTRAINT = "constraint"
    DERIVATION = "derivation"
    COMPOSITION = "composition"
    INVERSE = "inverse"
    ANALOGY = "analogy"

class SemanticIRDEngine:
    """
    Semantic Implicit Relation Discovery Engine
    Uses transformer models and mathematical ontology for enhanced understanding
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Semantic IRD Engine
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = None
        if TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded transformer model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load transformer model: {e}")
        
        # Initialize mathematical concept ontology
        self.concept_ontology = self._build_concept_ontology()
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        # Pattern library for mathematical expressions
        self.math_patterns = self._build_math_patterns()
        
    def _build_concept_ontology(self) -> Dict[str, MathematicalConcept]:
        """Build the mathematical concept ontology"""
        ontology = {}
        
        # Algebraic concepts
        algebra_concepts = [
            ("equation", "algebra", "basic", ["variable", "equality", "solution"]),
            ("variable", "algebra", "basic", ["unknown", "symbol", "parameter"]),
            ("coefficient", "algebra", "basic", ["multiplier", "factor", "constant"]),
            ("polynomial", "algebra", "advanced", ["degree", "term", "expression"]),
            ("quadratic", "algebra", "polynomial", ["parabola", "discriminant", "roots"]),
            ("linear", "algebra", "basic", ["proportional", "first-degree", "straight"]),
            ("system", "algebra", "advanced", ["simultaneous", "equations", "matrix"]),
        ]
        
        # Geometric concepts
        geometry_concepts = [
            ("triangle", "geometry", "shape", ["vertices", "sides", "angles"]),
            ("circle", "geometry", "shape", ["radius", "diameter", "circumference"]),
            ("angle", "geometry", "measurement", ["degrees", "radians", "vertex"]),
            ("area", "geometry", "measurement", ["surface", "square units", "region"]),
            ("perimeter", "geometry", "measurement", ["boundary", "length", "distance"]),
            ("parallel", "geometry", "relationship", ["lines", "never meet", "equidistant"]),
            ("perpendicular", "geometry", "relationship", ["right angle", "orthogonal", "90 degrees"]),
        ]
        
        # Calculus concepts
        calculus_concepts = [
            ("derivative", "calculus", "differential", ["rate of change", "slope", "tangent"]),
            ("integral", "calculus", "integral", ["accumulation", "area", "antiderivative"]),
            ("limit", "calculus", "basic", ["approach", "infinity", "continuity"]),
            ("function", "calculus", "basic", ["mapping", "domain", "range"]),
            ("continuous", "calculus", "property", ["unbroken", "smooth", "connected"]),
        ]
        
        # Physics concepts (for cross-domain problems)
        physics_concepts = [
            ("velocity", "physics", "kinematics", ["speed", "direction", "rate"]),
            ("acceleration", "physics", "kinematics", ["change", "velocity", "force"]),
            ("force", "physics", "dynamics", ["push", "pull", "Newton"]),
            ("energy", "physics", "dynamics", ["work", "power", "conservation"]),
            ("momentum", "physics", "dynamics", ["mass", "velocity", "collision"]),
        ]
        
        # Build ontology
        all_concepts = algebra_concepts + geometry_concepts + calculus_concepts + physics_concepts
        
        for name, category, subcategory, relationships in all_concepts:
            concept = MathematicalConcept(
                name=name,
                category=category,
                subcategory=subcategory,
                relationships=relationships
            )
            ontology[name] = concept
        
        return ontology
    
    def _build_math_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for mathematical expression recognition"""
        patterns = {
            'equation': re.compile(r'([a-zA-Z_]\w*)\s*=\s*(.+)'),
            'variable': re.compile(r'\b([a-zA-Z])\b(?!\w)'),
            'number': re.compile(r'\b(\d+\.?\d*)\b'),
            'operation': re.compile(r'[\+\-\*/\^]'),
            'function': re.compile(r'(sin|cos|tan|log|ln|sqrt|exp)\s*\('),
            'derivative': re.compile(r"(?:d/dx|d\w/d\w|'|derivative)"),
            'integral': re.compile(r'(?:∫|integral|integrate)'),
            'sum': re.compile(r'(?:∑|sum|total)'),
            'product': re.compile(r'(?:∏|product|multiply)'),
            'inequality': re.compile(r'[<>≤≥]'),
            'proportion': re.compile(r'(?::\s*|proportional|ratio)'),
        }
        return patterns
    
    def extract_mathematical_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical entities from text using semantic analysis
        
        Args:
            text: Input text containing mathematical content
            
        Returns:
            List of extracted entities with semantic properties
        """
        entities = []
        
        # Tokenize and analyze text
        sentences = text.split('.')
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Extract variables
            variables = self.math_patterns['variable'].findall(sentence)
            for var in variables:
                entities.append({
                    'text': var,
                    'type': 'variable',
                    'context': sentence.strip(),
                    'position': sentence.find(var)
                })
            
            # Extract numbers
            numbers = self.math_patterns['number'].findall(sentence)
            for num in numbers:
                entities.append({
                    'text': num,
                    'type': 'number',
                    'value': float(num),
                    'context': sentence.strip()
                })
            
            # Extract equations
            equations = self.math_patterns['equation'].findall(sentence)
            for lhs, rhs in equations:
                entities.append({
                    'text': f"{lhs} = {rhs}",
                    'type': 'equation',
                    'lhs': lhs.strip(),
                    'rhs': rhs.strip(),
                    'context': sentence.strip()
                })
            
            # Extract mathematical concepts from ontology
            for concept_name, concept in self.concept_ontology.items():
                if concept_name in sentence.lower():
                    entities.append({
                        'text': concept_name,
                        'type': 'concept',
                        'category': concept.category,
                        'subcategory': concept.subcategory,
                        'context': sentence.strip()
                    })
        
        # Add semantic embeddings if transformer is available
        if self.model and entities:
            self._add_semantic_embeddings(entities)
        
        return entities
    
    def discover_implicit_relations(self, entities: List[Dict[str, Any]], 
                                  problem_context: str) -> List[SemanticRelation]:
        """
        Discover implicit relationships between entities using semantic analysis
        
        Args:
            entities: List of mathematical entities
            problem_context: Overall problem context
            
        Returns:
            List of discovered semantic relations
        """
        relations = []
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity['type']].append(entity)
        
        # Discover variable-equation relationships
        for var_entity in entities_by_type.get('variable', []):
            for eq_entity in entities_by_type.get('equation', []):
                if var_entity['text'] in eq_entity['text']:
                    relation = SemanticRelation(
                        source=var_entity['text'],
                        target=eq_entity['text'],
                        relation_type=RelationType.DEPENDENCY.value,
                        confidence=0.9,
                        evidence=[f"Variable {var_entity['text']} appears in equation"]
                    )
                    relations.append(relation)
        
        # Discover concept relationships using ontology
        concept_entities = entities_by_type.get('concept', [])
        for i, concept1 in enumerate(concept_entities):
            for concept2 in concept_entities[i+1:]:
                relation = self._analyze_concept_relationship(concept1, concept2)
                if relation:
                    relations.append(relation)
        
        # Discover semantic similarities if transformer is available
        if self.model and len(entities) > 1:
            similarity_relations = self._discover_semantic_similarities(entities)
            relations.extend(similarity_relations)
        
        # Discover mathematical transformations
        equation_entities = entities_by_type.get('equation', [])
        if len(equation_entities) > 1:
            transformation_relations = self._discover_transformations(equation_entities)
            relations.extend(transformation_relations)
        
        # Filter and rank relations
        relations = self._rank_relations(relations, problem_context)
        
        return relations
    
    def _add_semantic_embeddings(self, entities: List[Dict[str, Any]]):
        """Add semantic embeddings to entities"""
        if not self.model:
            return
        
        # Prepare texts for embedding
        texts = []
        for entity in entities:
            # Use context for better semantic understanding
            text = entity.get('context', entity['text'])
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Add embeddings to entities
        for entity, embedding in zip(entities, embeddings):
            entity['embedding'] = embedding
            # Cache the embedding
            self.embedding_cache[entity['text']] = embedding
    
    def _analyze_concept_relationship(self, concept1: Dict[str, Any], 
                                    concept2: Dict[str, Any]) -> Optional[SemanticRelation]:
        """Analyze relationship between two mathematical concepts"""
        name1 = concept1['text']
        name2 = concept2['text']
        
        # Check if concepts are in the same category
        if concept1.get('category') == concept2.get('category'):
            confidence = 0.7
            relation_type = RelationType.COMPOSITION.value
            evidence = [f"Both concepts belong to {concept1.get('category')}"]
        else:
            # Check for known cross-domain relationships
            if (concept1.get('category') == 'geometry' and concept2.get('category') == 'algebra') or \
               (concept1.get('category') == 'algebra' and concept2.get('category') == 'geometry'):
                confidence = 0.6
                relation_type = RelationType.TRANSFORMATION.value
                evidence = ["Cross-domain relationship between algebra and geometry"]
            else:
                return None
        
        # Check ontology for specific relationships
        if name1 in self.concept_ontology and name2 in self.concept_ontology:
            concept1_obj = self.concept_ontology[name1]
            concept2_obj = self.concept_ontology[name2]
            
            # Check if one concept references the other
            if name2 in concept1_obj.relationships:
                confidence = 0.9
                evidence.append(f"{name1} has known relationship with {name2}")
            elif name1 in concept2_obj.relationships:
                confidence = 0.9
                evidence.append(f"{name2} has known relationship with {name1}")
        
        return SemanticRelation(
            source=name1,
            target=name2,
            relation_type=relation_type,
            confidence=confidence,
            evidence=evidence
        )
    
    def _discover_semantic_similarities(self, entities: List[Dict[str, Any]]) -> List[SemanticRelation]:
        """Discover relations based on semantic similarity"""
        relations = []
        
        # Only process entities with embeddings
        entities_with_embeddings = [e for e in entities if 'embedding' in e]
        
        for i, entity1 in enumerate(entities_with_embeddings):
            for entity2 in entities_with_embeddings[i+1:]:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(entity1['embedding'], entity2['embedding'])
                
                # High similarity suggests relationship
                if similarity > 0.7:
                    relation = SemanticRelation(
                        source=entity1['text'],
                        target=entity2['text'],
                        relation_type=RelationType.ANALOGY.value,
                        confidence=similarity,
                        evidence=[f"High semantic similarity: {similarity:.2f}"],
                        semantic_similarity=similarity
                    )
                    relations.append(relation)
        
        return relations
    
    def _discover_transformations(self, equations: List[Dict[str, Any]]) -> List[SemanticRelation]:
        """Discover transformation relationships between equations"""
        relations = []
        
        for i, eq1 in enumerate(equations):
            for eq2 in equations[i+1:]:
                # Check if equations share variables
                vars1 = set(self.math_patterns['variable'].findall(eq1['text']))
                vars2 = set(self.math_patterns['variable'].findall(eq2['text']))
                
                common_vars = vars1.intersection(vars2)
                if common_vars:
                    # Equations with common variables might be transformations
                    confidence = len(common_vars) / max(len(vars1), len(vars2))
                    
                    relation = SemanticRelation(
                        source=eq1['text'],
                        target=eq2['text'],
                        relation_type=RelationType.TRANSFORMATION.value,
                        confidence=confidence,
                        evidence=[f"Shares variables: {', '.join(common_vars)}"]
                    )
                    relations.append(relation)
        
        return relations
    
    def _rank_relations(self, relations: List[SemanticRelation], 
                       context: str) -> List[SemanticRelation]:
        """Rank and filter relations based on relevance to problem context"""
        # Calculate context embedding if available
        context_embedding = None
        if self.model:
            context_embedding = self.model.encode([context])[0]
        
        # Score each relation
        scored_relations = []
        for relation in relations:
            score = relation.confidence
            
            # Boost score based on relation type relevance
            if relation.relation_type == RelationType.DEPENDENCY.value:
                score *= 1.2  # Dependencies are usually important
            elif relation.relation_type == RelationType.TRANSFORMATION.value:
                score *= 1.1  # Transformations are valuable for solving
            
            # Consider semantic similarity to context if available
            if context_embedding is not None and relation.semantic_similarity > 0:
                score *= (1 + relation.semantic_similarity * 0.2)
            
            # Penalize very low confidence relations
            if relation.confidence < 0.3:
                score *= 0.5
            
            scored_relations.append((score, relation))
        
        # Sort by score and return top relations
        scored_relations.sort(key=lambda x: x[0], reverse=True)
        
        # Filter out very low scoring relations
        min_score = 0.2
        filtered_relations = [rel for score, rel in scored_relations if score >= min_score]
        
        return filtered_relations
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def build_knowledge_graph(self, entities: List[Dict[str, Any]], 
                            relations: List[SemanticRelation]) -> Dict[str, Any]:
        """
        Build a knowledge graph from entities and relations
        
        Args:
            entities: List of mathematical entities
            relations: List of semantic relations
            
        Returns:
            Knowledge graph representation
        """
        graph = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_entities': len(entities),
                'total_relations': len(relations),
                'entity_types': defaultdict(int),
                'relation_types': defaultdict(int)
            }
        }
        
        # Add nodes (entities)
        entity_map = {}
        for i, entity in enumerate(entities):
            node = {
                'id': f"entity_{i}",
                'label': entity['text'],
                'type': entity['type'],
                'properties': {
                    k: v for k, v in entity.items() 
                    if k not in ['text', 'type', 'embedding']
                }
            }
            graph['nodes'].append(node)
            entity_map[entity['text']] = f"entity_{i}"
            graph['metadata']['entity_types'][entity['type']] += 1
        
        # Add edges (relations)
        for i, relation in enumerate(relations):
            if relation.source in entity_map and relation.target in entity_map:
                edge = {
                    'id': f"relation_{i}",
                    'source': entity_map[relation.source],
                    'target': entity_map[relation.target],
                    'type': relation.relation_type,
                    'weight': relation.confidence,
                    'properties': {
                        'evidence': relation.evidence,
                        'semantic_similarity': relation.semantic_similarity
                    }
                }
                graph['edges'].append(edge)
                graph['metadata']['relation_types'][relation.relation_type] += 1
        
        return graph
    
    def analyze_problem_semantics(self, problem_text: str) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of a mathematical problem
        
        Args:
            problem_text: The mathematical problem text
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        # Extract entities
        entities = self.extract_mathematical_entities(problem_text)
        
        # Discover relations
        relations = self.discover_implicit_relations(entities, problem_text)
        
        # Build knowledge graph
        knowledge_graph = self.build_knowledge_graph(entities, relations)
        
        # Analyze problem type and complexity
        problem_type = self._classify_problem_type(entities, relations)
        complexity = self._assess_semantic_complexity(entities, relations)
        
        # Generate insights
        insights = self._generate_semantic_insights(entities, relations, problem_type)
        
        analysis_time = time.time() - start_time
        
        return {
            'entities': entities,
            'relations': relations,
            'knowledge_graph': knowledge_graph,
            'problem_type': problem_type,
            'complexity': complexity,
            'insights': insights,
            'analysis_time': analysis_time,
            'transformer_available': self.model is not None
        }
    
    def _classify_problem_type(self, entities: List[Dict[str, Any]], 
                              relations: List[SemanticRelation]) -> Dict[str, Any]:
        """Classify the type of mathematical problem"""
        # Count entity types
        type_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for entity in entities:
            type_counts[entity['type']] += 1
            if entity['type'] == 'concept':
                category_counts[entity.get('category', 'unknown')] += 1
        
        # Determine primary problem type
        if category_counts:
            primary_category = max(category_counts, key=category_counts.get)
        else:
            primary_category = 'general'
        
        # Check for specific problem patterns
        has_equations = type_counts.get('equation', 0) > 0
        has_variables = type_counts.get('variable', 0) > 0
        has_geometric_concepts = category_counts.get('geometry', 0) > 0
        has_calculus_concepts = category_counts.get('calculus', 0) > 0
        
        problem_type = {
            'primary_category': primary_category,
            'is_algebraic': has_equations and has_variables,
            'is_geometric': has_geometric_concepts,
            'is_calculus': has_calculus_concepts,
            'is_word_problem': len(entities) > 5,  # Heuristic
            'categories_involved': list(category_counts.keys())
        }
        
        return problem_type
    
    def _assess_semantic_complexity(self, entities: List[Dict[str, Any]], 
                                   relations: List[SemanticRelation]) -> Dict[str, Any]:
        """Assess the semantic complexity of the problem"""
        # Base complexity on various factors
        num_entities = len(entities)
        num_relations = len(relations)
        
        # Entity diversity
        unique_types = len(set(e['type'] for e in entities))
        
        # Relation complexity
        relation_types = [r.relation_type for r in relations]
        unique_relation_types = len(set(relation_types))
        
        # Calculate complexity score (0-1)
        entity_score = min(num_entities / 20, 1.0)  # Normalize to 20 entities
        relation_score = min(num_relations / 15, 1.0)  # Normalize to 15 relations
        diversity_score = min(unique_types / 5, 1.0)  # Normalize to 5 types
        relation_diversity_score = min(unique_relation_types / 4, 1.0)
        
        overall_score = (entity_score + relation_score + diversity_score + relation_diversity_score) / 4
        
        # Classify complexity level
        if overall_score < 0.3:
            level = "low"
        elif overall_score < 0.6:
            level = "medium"
        else:
            level = "high"
        
        return {
            'score': overall_score,
            'level': level,
            'factors': {
                'entity_complexity': entity_score,
                'relation_complexity': relation_score,
                'entity_diversity': diversity_score,
                'relation_diversity': relation_diversity_score
            },
            'metrics': {
                'num_entities': num_entities,
                'num_relations': num_relations,
                'unique_entity_types': unique_types,
                'unique_relation_types': unique_relation_types
            }
        }
    
    def _generate_semantic_insights(self, entities: List[Dict[str, Any]], 
                                   relations: List[SemanticRelation],
                                   problem_type: Dict[str, Any]) -> List[str]:
        """Generate semantic insights about the problem"""
        insights = []
        
        # Insight about problem domain
        if problem_type['is_algebraic'] and problem_type['is_geometric']:
            insights.append("This problem combines algebraic and geometric concepts, suggesting analytical geometry approach")
        
        # Insight about key relationships
        strong_relations = [r for r in relations if r.confidence > 0.8]
        if strong_relations:
            insights.append(f"Found {len(strong_relations)} strong relationships that are key to solving")
        
        # Insight about transformation opportunities
        transformation_relations = [r for r in relations if r.relation_type == RelationType.TRANSFORMATION.value]
        if transformation_relations:
            insights.append("Multiple equation transformations possible - consider algebraic manipulation")
        
        # Insight about variable dependencies
        dependency_relations = [r for r in relations if r.relation_type == RelationType.DEPENDENCY.value]
        if len(dependency_relations) > 3:
            insights.append("Complex variable dependencies detected - consider systematic substitution")
        
        # Insight about semantic clusters
        if self.model and entities:
            clusters = self._identify_semantic_clusters(entities)
            if len(clusters) > 1:
                insights.append(f"Problem contains {len(clusters)} distinct semantic clusters - consider divide-and-conquer approach")
        
        return insights
    
    def _identify_semantic_clusters(self, entities: List[Dict[str, Any]]) -> List[List[int]]:
        """Identify semantic clusters among entities"""
        if not self.model:
            return []
        
        entities_with_embeddings = [
            (i, e) for i, e in enumerate(entities) if 'embedding' in e
        ]
        
        if len(entities_with_embeddings) < 2:
            return [[i for i, _ in entities_with_embeddings]]
        
        # Simple clustering based on similarity threshold
        clusters = []
        processed = set()
        threshold = 0.7
        
        for i, (idx1, entity1) in enumerate(entities_with_embeddings):
            if idx1 in processed:
                continue
            
            cluster = [idx1]
            processed.add(idx1)
            
            for j, (idx2, entity2) in enumerate(entities_with_embeddings[i+1:], i+1):
                if idx2 not in processed:
                    similarity = self._cosine_similarity(
                        entity1['embedding'], 
                        entity2['embedding']
                    )
                    if similarity > threshold:
                        cluster.append(idx2)
                        processed.add(idx2)
            
            clusters.append(cluster)
        
        return clusters