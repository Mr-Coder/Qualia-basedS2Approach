"""
Refactored Mathematical Reasoning System

This module provides a clean, modular interface to the mathematical reasoning system
using the refactored core components. It demonstrates the improved architecture
with single-responsibility modules.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from .core import (MathEntity, ProblemContext, ProblemParser, ReasoningEngine,
                   SolutionResult, SolutionValidator, StepGenerator,
                   ValidationResult)


class RefactoredMathematicalReasoningSystem:
    """
    Main interface for the refactored mathematical reasoning system.
    
    This class coordinates the modular components to solve mathematical problems
    while maintaining clean separation of concerns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the refactored mathematical reasoning system.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize modular components
        self.problem_parser = ProblemParser()
        self.reasoning_engine = ReasoningEngine(config)
        self.step_generator = StepGenerator()
        self.solution_validator = SolutionValidator()
        
        # Set up logging
        self.logger = self._setup_logger()
        
        # Configuration options
        self.enable_validation = self.config.get('enable_validation', True)
        self.validation_threshold = self.config.get('validation_threshold', 0.7)
        
        self.logger.info("Refactored Mathematical Reasoning System initialized")
    
    def solve_problem(self, problem_text: str) -> Dict[str, Any]:
        """
        Solve a mathematical problem using the modular architecture.
        
        Args:
            problem_text: The mathematical problem statement
            
        Returns:
            Dictionary containing the solution and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Solving problem: {problem_text[:100]}...")
            
            # Step 1: Parse the problem using the problem parser
            problem_context = self.problem_parser.parse_problem(problem_text)
            self.logger.debug(f"Parsed {len(problem_context.entities)} entities and {len(problem_context.relations)} relations")
            
            # Step 2: Generate reasoning steps using the step generator
            reasoning_steps = self.step_generator.generate_reasoning_steps(problem_context)
            self.logger.debug(f"Generated {len(reasoning_steps)} reasoning steps")
            
            # Step 3: Use reasoning engine to coordinate and refine the solution
            solution_result = self.reasoning_engine.solve_problem(problem_context)
            
            # Step 4: Validate the solution if enabled
            validation_result = None
            if self.enable_validation and solution_result.reasoning_steps:
                validation_result = self.solution_validator.validate_solution(
                    solution_result.reasoning_steps, 
                    problem_context
                )
                self.logger.debug(f"Validation result: {validation_result.is_valid}, confidence: {validation_result.confidence_score:.3f}")
            
            # Step 5: Prepare the final result
            processing_time = time.time() - start_time
            
            result = {
                'problem': problem_text,
                'final_answer': solution_result.final_answer,
                'reasoning_steps': [step.to_dict() for step in solution_result.reasoning_steps],
                'confidence': solution_result.confidence,
                'processing_time': processing_time,
                'complexity': solution_result.complexity,
                'validation': validation_result.to_dict() if validation_result else None,
                'metadata': {
                    'num_entities': len(problem_context.entities),
                    'num_relations': len(problem_context.relations),
                    'num_reasoning_steps': len(solution_result.reasoning_steps),
                    'domain_hints': problem_context.domain_hints,
                    'architecture': 'refactored_modular',
                    'components_used': ['ProblemParser', 'ReasoningEngine', 'StepGenerator', 'SolutionValidator']
                }
            }
            
            self.logger.info(f"Problem solved successfully in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error solving problem: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                'problem': problem_text,
                'final_answer': None,
                'reasoning_steps': [],
                'confidence': 0.0,
                'processing_time': processing_time,
                'complexity': 'unknown',
                'validation': None,
                'metadata': {'error': str(e), 'architecture': 'refactored_modular'}
            }
    
    def solve_multiple_problems(self, problems: List[str]) -> List[Dict[str, Any]]:
        """
        Solve multiple mathematical problems in batch.
        
        Args:
            problems: List of problem text strings
            
        Returns:
            List of solution dictionaries
        """
        results = []
        
        for i, problem in enumerate(problems):
            self.logger.info(f"Solving problem {i+1}/{len(problems)}")
            result = self.solve_problem(problem)
            results.append(result)
        
        return results
    
    def analyze_problem_complexity(self, problem_text: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a mathematical problem without solving it.
        
        Args:
            problem_text: The mathematical problem statement
            
        Returns:
            Dictionary containing complexity analysis
        """
        try:
            # Parse the problem to extract entities and relations
            problem_context = self.problem_parser.parse_problem(problem_text)
            
            return {
                'complexity_level': problem_context.complexity.value,
                'num_entities': len(problem_context.entities),
                'num_relations': len(problem_context.relations),
                'domain_hints': problem_context.domain_hints,
                'target_question': problem_context.target_question,
                'entities': [entity.to_dict() for entity in problem_context.entities]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing problem complexity: {str(e)}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of all system components.
        
        Returns:
            Dictionary containing system status information
        """
        return {
            'system_name': 'Refactored Mathematical Reasoning System',
            'architecture': 'modular',
            'components': {
                'problem_parser': 'active',
                'reasoning_engine': 'active',
                'step_generator': 'active',
                'solution_validator': 'active' if self.enable_validation else 'disabled'
            },
            'configuration': self.config,
            'engine_status': self.reasoning_engine.get_engine_status(),
            'validation_enabled': self.enable_validation,
            'validation_threshold': self.validation_threshold
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """
        Update the system configuration.
        
        Args:
            new_config: New configuration settings
        """
        self.config.update(new_config)
        
        # Update component configurations
        self.reasoning_engine.update_config(new_config)
        
        # Update local settings
        self.enable_validation = self.config.get('enable_validation', True)
        self.validation_threshold = self.config.get('validation_threshold', 0.7)
        
        self.logger.info("System configuration updated")
    
    def add_validation_rule(self, rule):
        """
        Add a custom validation rule to the solution validator.
        
        Args:
            rule: ValidationRule instance
        """
        self.solution_validator.add_validation_rule(rule)
        self.logger.info(f"Added validation rule: {rule.get_rule_name()}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the system."""
        logger = logging.getLogger('RefactoredMathReasoningSystem')
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger


# Example usage and demonstration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the refactored system
    config = {
        'enable_validation': True,
        'validation_threshold': 0.7,
        'max_reasoning_steps': 10,
        'confidence_threshold': 0.5
    }
    
    system = RefactoredMathematicalReasoningSystem(config)
    
    # Test problems of different complexities
    test_problems = [
        # L0 - Simple arithmetic
        "John has 15 apples and Mary has 8 apples. How many apples do they have together?",
        
        # L1 - Basic inference
        "A car travels at 60 km/h for 2.5 hours. How far does it travel?",
        
        # L2 - Multi-step reasoning
        "A water tank can hold 500 liters. It currently has 200 liters. Water flows in at 15 liters per minute. How long will it take to fill the tank?",
    ]
    
    print("=== Refactored Mathematical Reasoning System Demo ===\n")
    
    # Solve each problem
    for i, problem in enumerate(test_problems, 1):
        print(f"Problem {i}: {problem}")
        
        # Analyze complexity first
        complexity_analysis = system.analyze_problem_complexity(problem)
        print(f"Complexity: {complexity_analysis.get('complexity_level', 'unknown')}")
        
        # Solve the problem
        result = system.solve_problem(problem)
        
        print(f"Answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing Time: {result['processing_time']:.3f}s")
        
        if result['validation']:
            validation = result['validation']
            print(f"Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
            if validation['errors']:
                print(f"Validation Errors: {', '.join(validation['errors'])}")
        
        print(f"Reasoning Steps:")
        for j, step in enumerate(result['reasoning_steps'], 1):
            print(f"  {j}. {step['description']}")
        
        print("-" * 80)
    
    # Show system status
    print("\n=== System Status ===")
    status = system.get_system_status()
    print(f"System: {status['system_name']}")
    print(f"Architecture: {status['architecture']}")
    print(f"Components: {', '.join(status['components'].keys())}")
    print(f"Validation Enabled: {status['validation_enabled']}")
    
    print("\n=== Refactoring Benefits Demonstrated ===")
    print("✅ Modular architecture with single-responsibility components")
    print("✅ Clean separation between parsing, reasoning, generation, and validation")
    print("✅ Easy to test, maintain, and extend individual components")
    print("✅ Configurable validation and reasoning strategies")
    print("✅ Comprehensive logging and error handling")
    print("✅ Type hints and documentation for better code quality") 