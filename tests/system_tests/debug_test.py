#!/usr/bin/env python3
"""
Debug test script to identify the core issue
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mathematical_reasoning_system import MathematicalReasoningSystem


def simple_test():
    """Simple test to debug the issue"""
    system = MathematicalReasoningSystem()
    
    # Test a very simple problem
    problem = "What is 25 + 17?"
    print(f"Testing problem: {problem}")
    
    try:
        # Step by step debugging
        print("\n1. Testing NLP Processing...")
        entities = system.nlp_processor.extract_entities(problem)
        print(f"Extracted entities: {len(entities)}")
        for i, entity in enumerate(entities):
            print(f"  Entity {i}: {entity} (type: {type(entity)})")
            if hasattr(entity, 'value'):
                print(f"    Value: {entity.value}")
            else:
                print(f"    No value attribute!")
        
        print("\n2. Testing Relation Discovery...")
        relations = system.relation_discovery.discover_relations(entities, problem)
        print(f"Found relations: {len(relations)}")
        
        print("\n3. Testing Full Problem Solving...")
        try:
            result = system.solve_mathematical_problem(problem)
            print(f"Final answer: {result.get('final_answer')}")
            
            # Debug reasoning steps if available
            if 'reasoning_steps' in result:
                print(f"\nFound {len(result['reasoning_steps'])} reasoning steps:")
                for i, step in enumerate(result['reasoning_steps']):
                    print(f"  Step {i}:")
                    print(f"    Description: {step.get('description', 'N/A')}")
                    print(f"    Operation: {step.get('operation', 'N/A')}")
                    
                    # Debug input entities
                    input_entities = step.get('input_entities', [])
                    print(f"    Input entities ({len(input_entities)}):")
                    for j, entity in enumerate(input_entities):
                        print(f"      Entity {j}: {entity} (type: {type(entity)})")
                        if isinstance(entity, str):
                            print(f"        ❌ ERROR: Entity is a string, not MathEntity!")
                        elif hasattr(entity, 'value'):
                            print(f"        Value: {entity.value}")
                    
                    # Debug output entity
                    output_entity = step.get('output_entity')
                    if output_entity:
                        print(f"    Output entity: {output_entity} (type: {type(output_entity)})")
                        if hasattr(output_entity, 'value'):
                            print(f"      Value: {output_entity.value}")
            else:
                print("No reasoning steps found in result")
        except Exception as inner_e:
            print(f"Inner error during detailed analysis: {inner_e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test() 