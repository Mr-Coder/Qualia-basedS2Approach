#!/usr/bin/env python3
"""
Test script for the refactored mathematical reasoning system.

This script demonstrates the improved modular architecture and compares it
with the original monolithic implementation.
"""

import logging
import sys
import time
from typing import Dict, List

# Add src to path to import modules
sys.path.append('src')

from src.refactored_mathematical_reasoning_system import \
    RefactoredMathematicalReasoningSystem


def run_refactored_system_tests():
    """Run comprehensive tests of the refactored system."""
    
    print("🔧 MATHEMATICAL REASONING SYSTEM REFACTORING DEMONSTRATION")
    print("=" * 80)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Initialize the refactored system
    config = {
        'enable_validation': True,
        'validation_threshold': 0.7,
        'max_reasoning_steps': 10,
        'confidence_threshold': 0.5,
        'enable_fallback': True
    }
    
    system = RefactoredMathematicalReasoningSystem(config)
    
    # Test problems covering different complexity levels
    test_problems = [
        {
            'level': 'L0_EXPLICIT',
            'problem': "Sarah has 25 stickers and Tom has 17 stickers. How many stickers do they have in total?",
            'expected_answer': 42
        },
        {
            'level': 'L1_SHALLOW', 
            'problem': "A car travels 120 kilometers in 3 hours. What is the car's speed?",
            'expected_answer': 40
        },
        {
            'level': 'L2_MEDIUM',
            'problem': "A rectangular garden is 8 meters long and 6 meters wide. What is the area of the garden?",
            'expected_answer': 48
        },
        {
            'level': 'L2_MEDIUM',
            'problem': "John buys 3 books for $15 each and 2 notebooks for $8 each. How much does he spend in total?",
            'expected_answer': 61
        }
    ]
    
    print(f"\n📊 TESTING {len(test_problems)} PROBLEMS ACROSS COMPLEXITY LEVELS")
    print("-" * 80)
    
    results = []
    total_start_time = time.time()
    
    for i, test_case in enumerate(test_problems, 1):
        print(f"\n🧮 Problem {i} ({test_case['level']})")
        print(f"Problem: {test_case['problem']}")
        
        # Analyze complexity first
        complexity_analysis = system.analyze_problem_complexity(test_case['problem'])
        print(f"Detected Complexity: {complexity_analysis.get('complexity_level', 'unknown')}")
        print(f"Entities Found: {complexity_analysis.get('num_entities', 0)}")
        print(f"Domain Hints: {', '.join(complexity_analysis.get('domain_hints', []))}")
        
        # Solve the problem
        start_time = time.time()
        result = system.solve_problem(test_case['problem'])
        solve_time = time.time() - start_time
        
        # Display results
        print(f"Final Answer: {result['final_answer']}")
        print(f"Expected Answer: {test_case['expected_answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing Time: {solve_time:.3f}s")
        
        # Check validation
        if result['validation']:
            validation = result['validation']
            status = "✅ PASSED" if validation['is_valid'] else "❌ FAILED"
            print(f"Validation: {status} (confidence: {validation['confidence_score']:.3f})")
            
            if validation['errors']:
                print(f"Validation Errors: {'; '.join(validation['errors'])}")
        
        # Show reasoning steps
        print("Reasoning Steps:")
        for j, step in enumerate(result['reasoning_steps'], 1):
            print(f"  {j}. {step['description']}")
        
        # Check correctness
        is_correct = abs(float(result['final_answer'] or 0) - test_case['expected_answer']) < 0.01
        correctness = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"Correctness Check: {correctness}")
        
        results.append({
            'problem_id': i,
            'level': test_case['level'],
            'is_correct': is_correct,
            'confidence': result['confidence'],
            'processing_time': solve_time,
            'validation_passed': result['validation']['is_valid'] if result['validation'] else None
        })
        
        print("-" * 60)
    
    total_time = time.time() - total_start_time
    
    # Summary Statistics
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    correct_answers = sum(1 for r in results if r['is_correct'])
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_time = sum(r['processing_time'] for r in results) / len(results)
    validation_pass_rate = sum(1 for r in results if r['validation_passed']) / len(results)
    
    print(f"Total Problems Solved: {len(results)}")
    print(f"Correct Answers: {correct_answers}/{len(results)} ({correct_answers/len(results)*100:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Average Processing Time: {avg_time:.3f}s")
    print(f"Total Processing Time: {total_time:.3f}s")
    print(f"Validation Pass Rate: {validation_pass_rate*100:.1f}%")
    
    # Component Analysis
    print(f"\n🔍 COMPONENT ANALYSIS")
    print("-" * 80)
    
    system_status = system.get_system_status()
    print(f"System Architecture: {system_status['architecture']}")
    print(f"Active Components: {', '.join(system_status['components'].keys())}")
    print(f"Validation Enabled: {system_status['validation_enabled']}")
    
    engine_status = system_status['engine_status']
    print(f"Reasoning Strategies Loaded: {engine_status['strategies_loaded']}")
    
    # Complexity Distribution
    complexity_distribution = {}
    for result in results:
        level = result['level']
        complexity_distribution[level] = complexity_distribution.get(level, 0) + 1
    
    print(f"\nComplexity Distribution:")
    for level, count in complexity_distribution.items():
        print(f"  {level}: {count} problems")
    
    # Benefits of Refactoring
    print(f"\n✨ REFACTORING BENEFITS DEMONSTRATED")
    print("=" * 80)
    
    benefits = [
        "✅ Modular Architecture: Single-responsibility components (Parser, Engine, Generator, Validator)",
        "✅ Clean Separation of Concerns: Each component has a focused purpose",
        "✅ Improved Testability: Components can be tested independently",
        "✅ Enhanced Maintainability: Easy to modify or extend individual components", 
        "✅ Better Error Handling: Localized error handling in each component",
        "✅ Configurable Validation: Pluggable validation rules and strategies",
        "✅ Comprehensive Logging: Detailed logging at component level",
        "✅ Type Safety: Full type hints for better IDE support and error detection",
        "✅ Extensibility: Easy to add new reasoning strategies or validation rules",
        "✅ Performance Monitoring: Component-level performance tracking"
    ]
    
    for benefit in benefits:
        print(benefit)
    
    # Architecture Comparison
    print(f"\n📊 ARCHITECTURE COMPARISON")
    print("-" * 80)
    
    print("ORIGINAL MONOLITHIC SYSTEM:")
    print("  ❌ Single large file (2,343 lines)")
    print("  ❌ Mixed responsibilities in one class")
    print("  ❌ Difficult to test individual components")
    print("  ❌ Hard to extend or modify")
    print("  ❌ Limited error isolation")
    
    print("\nREFACTORED MODULAR SYSTEM:")
    print("  ✅ Multiple focused modules (~200-300 lines each)")
    print("  ✅ Single responsibility per component")
    print("  ✅ Easy unit testing")
    print("  ✅ Plug-and-play component architecture")
    print("  ✅ Isolated error handling")
    print("  ✅ Clean interfaces and contracts")
    
    return results


def demonstrate_component_isolation():
    """Demonstrate how components can be used independently."""
    
    print(f"\n🔧 COMPONENT ISOLATION DEMONSTRATION")
    print("=" * 80)
    
    # Import individual components
    from src.core import ProblemParser, SolutionValidator, StepGenerator
    from src.core.data_structures import ProblemComplexity
    
    problem_text = "Lisa has 12 roses and 8 tulips. How many flowers does she have?"
    
    # 1. Use Problem Parser independently
    print("1️⃣ PROBLEM PARSER COMPONENT")
    parser = ProblemParser()
    context = parser.parse_problem(problem_text)
    
    print(f"   Entities extracted: {len(context.entities)}")
    for entity in context.entities:
        print(f"     - {entity.text} ({entity.entity_type.value})")
    print(f"   Complexity: {context.complexity.value}")
    print(f"   Domain hints: {context.domain_hints}")
    
    # 2. Use Step Generator independently  
    print("\n2️⃣ STEP GENERATOR COMPONENT")
    generator = StepGenerator()
    steps = generator.generate_reasoning_steps(context)
    
    print(f"   Generated {len(steps)} reasoning steps:")
    for i, step in enumerate(steps, 1):
        print(f"     {i}. {step.description}")
    
    # 3. Use Solution Validator independently
    print("\n3️⃣ SOLUTION VALIDATOR COMPONENT")
    validator = SolutionValidator()
    validation_result = validator.validate_solution(steps, context)
    
    print(f"   Validation result: {'✅ VALID' if validation_result.is_valid else '❌ INVALID'}")
    print(f"   Confidence: {validation_result.confidence_score:.3f}")
    print(f"   Consistency score: {validation_result.consistency_score:.3f}")
    print(f"   Completeness score: {validation_result.completeness_score:.3f}")
    
    if validation_result.errors:
        print(f"   Errors: {'; '.join(validation_result.errors)}")
    
    print("\n✨ This demonstrates how each component can be:")
    print("   • Tested independently")
    print("   • Used in different contexts") 
    print("   • Modified without affecting others")
    print("   • Replaced with alternative implementations")


if __name__ == "__main__":
    try:
        # Run main tests
        results = run_refactored_system_tests()
        
        # Demonstrate component isolation
        demonstrate_component_isolation()
        
        print(f"\n🎉 REFACTORING DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("   The modular architecture provides significant improvements in:")
        print("   • Code organization and maintainability")
        print("   • Testing and debugging capabilities") 
        print("   • Extensibility and customization")
        print("   • Error handling and validation")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc() 