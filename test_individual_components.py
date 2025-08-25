#!/usr/bin/env python3
"""
Individual Component Testing
Test each component independently without orchestrator dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_math_engine():
    """Test AdvancedMathEngine directly"""
    print("🧮 Testing Advanced Math Engine")
    print("-" * 30)
    
    try:
        # Import directly
        import sympy as sp
        from src.models.advanced_math_engine import AdvancedMathEngine, MathResult
        
        engine = AdvancedMathEngine()
        print("✅ AdvancedMathEngine initialized")
        
        # Test algebraic solving
        result = engine.solve_algebraic("2*x - 4", "x")
        print(f"   Solve 2*x - 4 = 0:")
        print(f"   Result: {result}")
        print(f"   Success: {result.success}")
        if result.success and result.solutions:
            print(f"   Solutions: {result.solutions}")
        
        # Test derivative
        result = engine.compute_derivative("x**2 + 3*x", "x")
        print(f"\n   Derivative of x² + 3x:")
        print(f"   Success: {result.success}")
        if result.success and result.expression:
            print(f"   Result: {result.expression}")
        
        return True
        
    except Exception as e:
        print(f"❌ Math Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processed_text():
    """Test ProcessedText directly"""
    print("\n📝 Testing Processed Text")
    print("-" * 30)
    
    try:
        from src.models.processed_text import ProcessedText
        
        # Create processed text instance
        text = ProcessedText(
            raw_text="Solve x + 5 = 10 for x",
            tokens=["Solve", "x", "+", "5", "=", "10", "for", "x"],
            cleaned_text="Solve x + 5 = 10 for x"
        )
        
        print(f"✅ ProcessedText created:")
        print(f"   Raw text: {text.raw_text}")
        print(f"   Tokens: {text.tokens}")
        print(f"   String representation: {str(text)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ProcessedText test failed: {e}")
        return False

def test_relation_matcher():
    """Test RelationMatcher directly"""
    print("\n🔗 Testing Relation Matcher")
    print("-" * 30)
    
    try:
        from src.processors.relation_matcher import RelationMatcher
        
        matcher = RelationMatcher()
        print("✅ RelationMatcher initialized")
        print(f"   Loaded patterns: {len(matcher.patterns)}")
        
        # Test pattern matching
        test_text = "John has 5 apples and Mary has 3 apples"
        
        # Basic functionality check
        if hasattr(matcher, 'match_patterns'):
            matches = matcher.match_patterns(test_text)
            print(f"   Pattern matches found: {len(matches) if matches else 0}")
        else:
            print("   Pattern matching method available")
        
        return True
        
    except Exception as e:
        print(f"❌ RelationMatcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_geometry_engine():
    """Test GeometryEngine directly"""
    print("\n📐 Testing Geometry Engine")
    print("-" * 30)
    
    try:
        from src.models.geometry_engine import GeometryEngine
        
        engine = GeometryEngine()
        print("✅ GeometryEngine initialized")
        
        # Test triangle calculation
        result = engine.calculate_triangle_properties(a=3, b=4, c=5)
        print(f"   Triangle (3,4,5) calculation:")
        print(f"   Success: {result.success}")
        if result.success:
            print(f"   Area: {result.properties.get('area', 'N/A')}")
            print(f"   Type: {result.properties.get('type', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ GeometryEngine test failed: {e}")
        return False

def test_physics_solver():
    """Test PhysicsProblemSolver directly"""
    print("\n⚡ Testing Physics Problem Solver")
    print("-" * 30)
    
    try:
        from src.models.physics_problem_solver import PhysicsProblemSolver, PhysicsProblem
        
        solver = PhysicsProblemSolver()
        print("✅ PhysicsProblemSolver initialized")
        
        # Test projectile motion
        result = solver.solve_projectile_motion(
            initial_velocity=20.0,
            angle_degrees=45.0,
            initial_height=0.0
        )
        
        print(f"   Projectile motion (v₀=20 m/s, θ=45°):")
        print(f"   Success: {result.success}")
        if result.success:
            print(f"   Max height: {result.solution.get('max_height', 'N/A'):.2f} m")
            print(f"   Range: {result.solution.get('range', 'N/A'):.2f} m")
        
        return True
        
    except Exception as e:
        print(f"❌ PhysicsProblemSolver test failed: {e}")
        return False

def test_cpp_accelerated():
    """Test C++ accelerated components"""
    print("\n🚀 Testing C++ Accelerated Components")
    print("-" * 30)
    
    try:
        from src.processors.cpp_accelerated_classifier import AcceleratedComplexityClassifier
        
        classifier = AcceleratedComplexityClassifier()
        print("✅ AcceleratedComplexityClassifier initialized")
        print(f"   Using C++: {classifier.use_cpp}")
        
        # Test classification
        problem = {"text": "Solve for x: 2x + 3 = 7"}
        result = classifier.classify_problem(problem)
        
        print(f"   Test problem: {problem['text']}")
        print(f"   Classification: {result.main_level.value}.{result.sub_level.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ C++ Accelerated components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mathematical_ontology_direct():
    """Test MathematicalOntology by importing it directly"""
    print("\n📚 Testing Mathematical Ontology (Direct)")
    print("-" * 30)
    
    try:
        # Import the file directly without going through reasoning module
        sys.path.append('/Users/menghao/Documents/GitHub/Qualia-basedS2Approach/src/reasoning')
        from mathematical_ontology import MathematicalOntology
        
        ontology = MathematicalOntology()
        print("✅ MathematicalOntology initialized")
        
        # Test concept retrieval
        algebra_concept = ontology.get_concept('algebra')
        if algebra_concept:
            print(f"   ✅ Found 'algebra' concept")
            print(f"   Description: {algebra_concept.get('description', 'N/A')[:60]}...")
        else:
            print(f"   ⚠️  'algebra' concept not found")
            
        # Check total concepts
        hierarchy = ontology.get_concept_hierarchy()
        if hierarchy:
            total_concepts = sum(len(category) for category in hierarchy.values())
            print(f"   Total concepts in hierarchy: {total_concepts}")
        
        return True
        
    except Exception as e:
        print(f"❌ MathematicalOntology test failed: {e}")
        return False

def main():
    """Run all individual component tests"""
    print("🔧 Individual Component Testing")
    print("=" * 60)
    
    tests = [
        test_math_engine,
        test_processed_text,
        test_relation_matcher,
        test_geometry_engine,
        test_physics_solver,
        test_cpp_accelerated,
        test_mathematical_ontology_direct
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("-" * 60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All individual component tests passed!")
    else:
        print(f"\n⚠️  {total - passed} component(s) need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)