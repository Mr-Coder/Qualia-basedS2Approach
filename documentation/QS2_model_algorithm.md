# QS² Model: Qualia-based Syntax-Semantic Model for Mathematical Problem Solving

## 1. Model Architecture Overview

The QS² (Qualia-based Syntax-Semantic) model is a three-layer architecture designed to discover deep implicit relations in mathematical word problems through qualia property mapping and semantic inference.

### 1.1 Architecture Layers

```
Layer 1: Syntax Layer (句法层)
├── POS Tagging & Dependency Parsing
├── Pattern Recognition & Matching
└── Variable Slot Identification

Layer 2: Semantic Layer (语义层)
├── Entity Recognition & Classification
├── Explicit Relation Extraction
└── Semantic Role Labeling

Layer 3: Qualia Layer (感质层)
├── Qualia Property Mapping
├── Implicit Relation Discovery
└── Deep Semantic Inference
```

## 2. Complete Algorithm Pseudocode

### 2.1 Main QS² Processing Algorithm

```python
ALGORITHM QS2_Process(input_text)
INPUT: Natural language mathematical problem text
OUTPUT: Mathematical relations with semantic dependencies

BEGIN
    // Phase 1: NLP Preprocessing
    processed_text ← NLP_Preprocess(input_text)
    
    // Phase 2: QS² Three-Layer Processing
    syntax_result ← Syntax_Layer_Process(processed_text)
    semantic_result ← Semantic_Layer_Process(processed_text, syntax_result)
    qualia_result ← Qualia_Layer_Process(processed_text, syntax_result, semantic_result)
    
    // Phase 3: Relation Integration
    final_relations ← Integrate_Relations(syntax_result, semantic_result, qualia_result)
    
    RETURN final_relations
END
```

### 2.2 Syntax Layer Algorithm

```python
ALGORITHM Syntax_Layer_Process(processed_text)
INPUT: Preprocessed text with tokens, POS tags, dependencies
OUTPUT: Pattern matches and variable slots

BEGIN
    patterns ← Load_Pattern_Repository()
    candidate_patterns ← []
    
    FOR each pattern P in patterns DO
        match_score ← Pattern_Match(P, processed_text)
        IF match_score > THRESHOLD THEN
            candidate_patterns.append((P, match_score))
        END IF
    END FOR
    
    // Sort by match score
    candidate_patterns ← Sort_Descending(candidate_patterns)
    
    // Extract variable slots
    variable_slots ← {}
    FOR each (pattern, score) in candidate_patterns DO
        slots ← Extract_Variable_Slots(pattern, processed_text)
        variable_slots[pattern.id] ← slots
    END FOR
    
    RETURN {
        'candidate_patterns': candidate_patterns,
        'variable_slots': variable_slots
    }
END

FUNCTION Pattern_Match(pattern, processed_text)
    // Multi-criteria pattern matching
    syntax_score ← Syntax_Structure_Match(pattern.structure, processed_text.dependencies)
    pos_score ← POS_Sequence_Match(pattern.pos_pattern, processed_text.pos_tags)
    keyword_score ← Keyword_Match(pattern.keywords, processed_text.tokens)
    
    total_score ← WEIGHT_SYNTAX * syntax_score + 
                  WEIGHT_POS * pos_score + 
                  WEIGHT_KEYWORD * keyword_score
    
    RETURN total_score
END
```

### 2.3 Semantic Layer Algorithm

```python
ALGORITHM Semantic_Layer_Process(processed_text, syntax_result)
INPUT: Processed text and syntax layer results
OUTPUT: Explicit relations and semantic dependencies

BEGIN
    explicit_relations ← []
    semantic_dependencies ← []
    
    FOR each pattern in syntax_result.candidate_patterns DO
        // Extract explicit relations
        relations ← Extract_Explicit_Relations(pattern, processed_text)
        explicit_relations.extend(relations)
        
        // Build semantic dependencies
        deps ← Build_Semantic_Dependencies(pattern, relations)
        semantic_dependencies.extend(deps)
    END FOR
    
    // Normalize and deduplicate
    explicit_relations ← Normalize_Relations(explicit_relations)
    semantic_dependencies ← Deduplicate_Dependencies(semantic_dependencies)
    
    RETURN {
        'explicit_relations': explicit_relations,
        'semantic_dependencies': semantic_dependencies
    }
END

FUNCTION Extract_Explicit_Relations(pattern, processed_text)
    var_entity ← Map_Variables_To_Entities(pattern, processed_text)
    relation_template ← pattern.relation_template
    
    // Substitute variables with actual entities
    FOR each (var, entity) in var_entity DO
        relation_template ← Replace(relation_template, var, entity)
    END FOR
    
    RETURN {
        'relation': relation_template,
        'source_pattern': pattern.id,
        'var_entity': var_entity,
        'type': 'explicit'
    }
END
```

### 2.4 Qualia Layer Algorithm (Core Innovation)

```python
ALGORITHM Qualia_Layer_Process(processed_text, syntax_result, semantic_result)
INPUT: All previous layer results
OUTPUT: Implicit relations discovered through qualia mapping

BEGIN
    qualia_mappings ← Build_Qualia_Mappings(processed_text)
    implicit_relations ← []
    deep_dependencies ← []
    
    // Phase 1: Qualia Property Mapping
    FOR each entity in processed_text.entities DO
        qualia_props ← Map_Qualia_Properties(entity)
        qualia_mappings[entity] ← qualia_props
    END FOR
    
    // Phase 2: Implicit Relation Discovery
    implicit_patterns ← Discover_Implicit_Patterns(qualia_mappings, semantic_result)
    
    FOR each implicit_pattern in implicit_patterns DO
        implicit_rel ← Generate_Implicit_Relation(implicit_pattern, qualia_mappings)
        implicit_relations.append(implicit_rel)
        
        // Generate deep semantic dependencies
        deep_deps ← Infer_Deep_Dependencies(implicit_rel, qualia_mappings)
        deep_dependencies.extend(deep_deps)
    END FOR
    
    // Phase 3: Recursive Dependency Resolution
    resolved_dependencies ← Resolve_Recursive_Dependencies(deep_dependencies)
    
    RETURN {
        'qualia_mappings': qualia_mappings,
        'implicit_relations': implicit_relations,
        'deep_dependencies': resolved_dependencies
    }
END

FUNCTION Map_Qualia_Properties(entity)
    // Four types of qualia properties (Pustejovsky's Qualia Structure)
    qualia ← {
        'formal': Determine_Formal_Role(entity),      // What it is
        'agentive': Determine_Agentive_Role(entity),  // How it came about
        'telic': Determine_Telic_Role(entity),        // Its purpose/function
        'constitutive': Determine_Constitutive_Role(entity) // What it's made of
    }
    
    RETURN qualia
END

FUNCTION Discover_Implicit_Patterns(qualia_mappings, semantic_result)
    implicit_patterns ← []
    
    // Pattern 1: Conservation Laws
    IF Detect_Conservation_Context(qualia_mappings) THEN
        pattern ← Create_Conservation_Pattern(qualia_mappings)
        implicit_patterns.append(pattern)
    END IF
    
    // Pattern 2: Rate-based Changes
    IF Detect_Rate_Context(qualia_mappings) THEN
        pattern ← Create_Rate_Pattern(qualia_mappings)
        implicit_patterns.append(pattern)
    END IF
    
    // Pattern 3: Equilibrium States
    IF Detect_Equilibrium_Context(qualia_mappings) THEN
        pattern ← Create_Equilibrium_Pattern(qualia_mappings)
        implicit_patterns.append(pattern)
    END IF
    
    // Pattern 4: Transformation Processes
    IF Detect_Transformation_Context(qualia_mappings) THEN
        pattern ← Create_Transformation_Pattern(qualia_mappings)
        implicit_patterns.append(pattern)
    END IF
    
    RETURN implicit_patterns
END
```

### 2.5 Deep Implicit Relation Discovery Algorithm

```python
ALGORITHM Discover_Deep_Implicit_Relations(problem_context, qualia_mappings)
INPUT: Problem context and qualia property mappings
OUTPUT: Deep implicit mathematical relations

BEGIN
    deep_relations ← []
    
    // Step 1: Analyze Physical/Mathematical Constraints
    constraints ← Analyze_Physical_Constraints(problem_context, qualia_mappings)
    
    // Step 2: Identify Hidden Variables
    hidden_vars ← Identify_Hidden_Variables(constraints, qualia_mappings)
    
    // Step 3: Infer Conservation Laws
    conservation_laws ← Infer_Conservation_Laws(problem_context, qualia_mappings)
    
    // Step 4: Discover Rate Relationships
    rate_relations ← Discover_Rate_Relations(problem_context, qualia_mappings)
    
    // Step 5: Build Implicit Equations
    FOR each constraint in constraints DO
        implicit_eq ← Build_Implicit_Equation(constraint, hidden_vars)
        deep_relations.append(implicit_eq)
    END FOR
    
    FOR each law in conservation_laws DO
        conservation_eq ← Build_Conservation_Equation(law, qualia_mappings)
        deep_relations.append(conservation_eq)
    END FOR
    
    FOR each rate_rel in rate_relations DO
        rate_eq ← Build_Rate_Equation(rate_rel, qualia_mappings)
        deep_relations.append(rate_eq)
    END FOR
    
    RETURN deep_relations
END

FUNCTION Analyze_Physical_Constraints(problem_context, qualia_mappings)
    constraints ← []
    
    // Tank problem constraints
    IF problem_context.type == "tank_problem" THEN
        constraints.append("volume_conservation")
        constraints.append("flow_rate_balance")
        constraints.append("time_dependency")
    END IF
    
    // Motion problem constraints
    IF problem_context.type == "motion_problem" THEN
        constraints.append("distance_time_relation")
        constraints.append("velocity_consistency")
    END IF
    
    // Work problem constraints
    IF problem_context.type == "work_problem" THEN
        constraints.append("efficiency_combination")
        constraints.append("time_work_relation")
    END IF
    
    RETURN constraints
END
```

### 2.6 Qualia Property Mapping Mechanism

```python
FUNCTION Map_Qualia_Properties_Detailed(entity, context)
    qualia ← {}
    
    // Formal Role: What the entity is categorically
    IF entity.type == "container" THEN
        qualia.formal ← "bounded_space"
    ELIF entity.type == "liquid" THEN
        qualia.formal ← "fluid_substance"
    ELIF entity.type == "solid" THEN
        qualia.formal ← "rigid_substance"
    END IF
    
    // Agentive Role: How the entity came to be
    IF entity.action == "added" THEN
        qualia.agentive ← "external_input"
    ELIF entity.action == "leaked" THEN
        qualia.agentive ← "natural_outflow"
    END IF
    
    // Telic Role: Purpose or function
    IF entity.role == "source" THEN
        qualia.telic ← "volume_provider"
    ELIF entity.role == "sink" THEN
        qualia.telic ← "volume_consumer"
    ELIF entity.role == "container" THEN
        qualia.telic ← "volume_holder"
    END IF
    
    // Constitutive Role: What it's made of or contains
    IF entity.contains THEN
        qualia.constitutive ← entity.contains
    END IF
    
    RETURN qualia
END
```

## 3. Key Innovation: Implicit Relation Discovery

### 3.1 Tank Problem Example

```python
ALGORITHM Discover_Tank_Implicit_Relations(tank_context, qualia_mappings)
BEGIN
    // Identify key entities and their qualia
    tank ← qualia_mappings["tank"]
    water ← qualia_mappings["water"]
    ice_cubes ← qualia_mappings["ice_cubes"]
    leak ← qualia_mappings["leak"]
    
    // Discover implicit conservation law
    IF tank.telic == "volume_holder" AND 
       ice_cubes.agentive == "external_input" AND
       leak.agentive == "natural_outflow" THEN
        
        conservation_relation ← "V(t) = V₀ + ∫(inflow_rate - outflow_rate)dt"
        
        // Discover unit conversion necessity
        IF ice_cubes.unit != water.unit THEN
            unit_conversion ← Discover_Unit_Conversion(ice_cubes, water)
        END IF
        
        // Discover net rate calculation
        net_rate_relation ← "net_rate = inflow_rate - outflow_rate"
        
        // Discover time calculation
        time_relation ← "time = (target_volume - initial_volume) / net_rate"
    END IF
    
    RETURN [conservation_relation, net_rate_relation, time_relation]
END
```

## 4. Semantic Dependency Chain Construction

```python
ALGORITHM Build_Semantic_Dependency_Chain(relations, qualia_mappings)
INPUT: Mathematical relations and qualia mappings
OUTPUT: Complete semantic dependency graph

BEGIN
    dependency_graph ← Create_Empty_Graph()
    
    FOR each relation in relations DO
        variables ← Extract_Variables(relation)
        
        FOR each var in variables DO
            qualia ← qualia_mappings[var]
            
            // Add dependency based on qualia roles
            IF qualia.telic == "volume_provider" THEN
                Add_Dependency(dependency_graph, "target_volume", var, "depends_on")
            ELIF qualia.telic == "volume_consumer" THEN
                Add_Dependency(dependency_graph, "target_volume", var, "inversely_depends_on")
            END IF
        END FOR
    END FOR
    
    // Resolve transitive dependencies
    transitive_deps ← Compute_Transitive_Closure(dependency_graph)
    
    RETURN transitive_deps
END
```

## 5. Performance Metrics and Validation

### 5.1 QS² Model Evaluation Metrics

```python
FUNCTION Evaluate_QS2_Performance(test_cases, model_output)
    metrics ← {}
    
    // Implicit relation discovery accuracy
    metrics.implicit_accuracy ← Count_Correct_Implicit_Relations(test_cases, model_output) / 
                                Total_Implicit_Relations(test_cases)
    
    // Semantic dependency precision
    metrics.dependency_precision ← Count_Correct_Dependencies(test_cases, model_output) / 
                                   Total_Predicted_Dependencies(model_output)
    
    // Qualia mapping consistency
    metrics.qualia_consistency ← Evaluate_Qualia_Consistency(test_cases, model_output)
    
    // Overall problem solving accuracy
    metrics.solving_accuracy ← Count_Correct_Solutions(test_cases, model_output) / 
                               Total_Test_Cases(test_cases)
    
    RETURN metrics
END
```

## 6. Complexity Analysis

- **Time Complexity**: O(n²m) where n is number of entities, m is number of patterns
- **Space Complexity**: O(nm + d²) where d is dependency graph size
- **Pattern Matching**: O(p×t) where p is patterns, t is tokens
- **Qualia Mapping**: O(e×q) where e is entities, q is qualia properties

## 7. Implementation Notes

1. **Pattern Repository**: Hierarchical structure with inheritance and composition
2. **Qualia Database**: Extensible knowledge base of entity properties
3. **Dependency Resolution**: Cycle detection and resolution algorithms
4. **Unit Conversion**: Automatic unit standardization system
5. **Error Handling**: Graceful degradation for incomplete information 