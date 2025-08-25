#!/usr/bin/env python3
"""
Standalone Import Test
Test core functionality imports without orchestrator dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_core_imports():
    """Test core component imports independently"""
    print("🧪 Testing Core Component Imports")
    print("=" * 50)
    
    # Test mathematical models - these should work
    try:
        print("\n1️⃣ Testing Mathematical Models:")
        
        from src.models.advanced_math_engine import AdvancedMathEngine
        print("   ✅ AdvancedMathEngine imported successfully")
        
        from src.models.physics_problem_solver import PhysicsProblemSolver
        print("   ✅ PhysicsProblemSolver imported successfully")
        
        from src.models.geometry_engine import GeometryEngine
        print("   ✅ GeometryEngine imported successfully")
        
        from src.models.mathematical_correctness_validator import MathematicalCorrectnessValidator
        print("   ✅ MathematicalCorrectnessValidator imported successfully")
        
        from src.models.processed_text import ProcessedText
        print("   ✅ ProcessedText imported successfully")
        
    except Exception as e:
        print(f"   ❌ Mathematical models import failed: {e}")
    
    # Test reasoning components
    try:
        print("\n2️⃣ Testing Reasoning Components:")
        
        from src.reasoning.mathematical_ontology import MathematicalOntology
        print("   ✅ MathematicalOntology imported successfully")
        
        # Test semantic IRD engine directly
        from src.reasoning.semantic_ird_engine import SemanticIRDEngine
        print("   ✅ SemanticIRDEngine imported successfully")
        
        from src.reasoning.enhanced_mlr_processor import EnhancedMLRProcessor
        print("   ✅ EnhancedMLRProcessor imported successfully")
        
    except Exception as e:
        print(f"   ❌ Reasoning components import failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test basic processor components without orchestrator
    try:
        print("\n3️⃣ Testing Basic Processor Components:")
        
        # Test processors that don't depend on orchestrator
        from src.processors.relation_matcher import RelationMatcher
        print("   ✅ RelationMatcher imported successfully")
        
    except Exception as e:
        print(f"   ❌ Processor components import failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test C++ accelerated components
    try:
        print("\n4️⃣ Testing C++ Accelerated Components:")
        
        from src.processors.cpp_accelerated_classifier import AcceleratedComplexityClassifier
        print("   ✅ AcceleratedComplexityClassifier imported successfully")
        
    except Exception as e:
        print(f"   ⚠️  C++ Accelerated components import failed (expected): {e}")
    
    print("\n" + "=" * 50)
    print("✅ Core import testing completed")

def test_basic_functionality():
    """Test basic functionality of imported components"""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        print("\n1️⃣ Testing Advanced Math Engine:")
        from src.models.advanced_math_engine import AdvancedMathEngine
        
        engine = AdvancedMathEngine()
        result = engine.solve_algebraic("2*x - 4", "x")
        print(f"   Solve 2*x - 4 = 0: x = {result.value}")
        print(f"   Success: {result.success}")
        
    except Exception as e:
        print(f"   ❌ Math Engine test failed: {e}")
    
    try:
        print("\n2️⃣ Testing Mathematical Ontology:")
        from src.reasoning.mathematical_ontology import MathematicalOntology
        
        ontology = MathematicalOntology()
        algebra_concept = ontology.get_concept('algebra')
        if algebra_concept:
            print(f"   ✅ Found algebra concept: {algebra_concept.get('description', 'No description')[:50]}...")
        else:
            print(f"   ⚠️  Algebra concept not found")
            
    except Exception as e:
        print(f"   ❌ Mathematical Ontology test failed: {e}")
    
    try:
        print("\n3️⃣ Testing Processed Text:")
        from src.models.processed_text import ProcessedText
        
        text = ProcessedText(raw_text="Solve x + 5 = 10")
        print(f"   ✅ ProcessedText created: {text}")
        
    except Exception as e:
        print(f"   ❌ ProcessedText test failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Basic functionality testing completed")

if __name__ == "__main__":
    test_core_imports()
    test_basic_functionality()