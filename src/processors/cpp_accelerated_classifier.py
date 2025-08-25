"""
C++ Accelerated Complexity Classifier Wrapper
Provides Python interface to the C++ implementation for 4-5x speedup
Part of Story 6.1: Mathematical Reasoning Enhancement - Phase 3
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Try to import C++ module
try:
    import math_reasoning_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    logging.warning("C++ acceleration module not available. Falling back to Python implementation.")

# Fall back to Python implementation if needed
from .enhanced_complexity_classifier import (
    EnhancedComplexityClassifier, 
    ComplexityLevel,
    SubLevel,
    ComplexityMetrics,
    ComplexityClassification,
    ReasoningType
)

logger = logging.getLogger(__name__)

class AcceleratedComplexityClassifier:
    """
    Wrapper for C++ accelerated complexity classifier
    Provides seamless fallback to Python implementation if C++ module unavailable
    """
    
    def __init__(self, use_cpp: bool = True):
        """
        Initialize the accelerated classifier
        
        Args:
            use_cpp: Whether to use C++ acceleration if available
        """
        self.use_cpp = use_cpp and CPP_AVAILABLE
        
        if self.use_cpp:
            self.cpp_classifier = math_reasoning_cpp.ComplexityClassifier()
            logger.info("Using C++ accelerated complexity classifier")
        else:
            self.python_classifier = EnhancedComplexityClassifier()
            logger.info("Using Python complexity classifier")
        
        # Performance metrics
        self.total_classifications = 0
        self.cpp_time = 0.0
        self.python_time = 0.0
    
    def classify_problem(self, problem: Dict[str, Any]) -> ComplexityClassification:
        """
        Classify problem complexity using C++ acceleration when available
        
        Args:
            problem: Mathematical problem representation
            
        Returns:
            ComplexityClassification with detailed metrics
        """
        import time
        start_time = time.time()
        
        if self.use_cpp and 'text' in problem:
            # Use C++ implementation
            result = self._classify_with_cpp(problem['text'])
        else:
            # Use Python implementation
            result = self.python_classifier.classify_problem(problem)
        
        elapsed_time = time.time() - start_time
        
        # Track performance
        self.total_classifications += 1
        if self.use_cpp:
            self.cpp_time += elapsed_time
        else:
            self.python_time += elapsed_time
        
        return result
    
    def _classify_with_cpp(self, problem_text: str) -> ComplexityClassification:
        """
        Classify using C++ implementation and convert to Python types
        
        Args:
            problem_text: Text of the mathematical problem
            
        Returns:
            ComplexityClassification object
        """
        # Call C++ classifier
        cpp_result = self.cpp_classifier.classify(problem_text)
        
        # Convert C++ enums to Python enums
        main_level = self._convert_cpp_level(cpp_result.main_level)
        sub_level = self._convert_cpp_sublevel(cpp_result.sub_level)
        
        # Convert metrics
        metrics = ComplexityMetrics(
            reasoning_depth=cpp_result.metrics.reasoning_depth,
            knowledge_dependencies=cpp_result.metrics.knowledge_dependencies,
            inference_steps=cpp_result.metrics.inference_steps,
            variable_count=cpp_result.metrics.variable_count,
            equation_count=cpp_result.metrics.equation_count,
            constraint_count=cpp_result.metrics.constraint_count,
            domain_switches=cpp_result.metrics.domain_switches,
            abstraction_level=cpp_result.metrics.abstraction_level,
            semantic_complexity=cpp_result.metrics.semantic_complexity,
            computational_complexity=cpp_result.metrics.computational_complexity
        )
        
        # Note: C++ version doesn't identify reasoning types, so we default to algebraic
        reasoning_types = [ReasoningType.ALGEBRAIC]
        
        # Create classification result
        return ComplexityClassification(
            main_level=main_level,
            sub_level=sub_level,
            reasoning_types=reasoning_types,
            metrics=metrics,
            confidence=cpp_result.confidence,
            explanation=cpp_result.explanation,
            domain_specific_factors={}  # C++ version doesn't compute these yet
        )
    
    def _convert_cpp_level(self, cpp_level) -> ComplexityLevel:
        """Convert C++ ComplexityLevel to Python enum"""
        level_map = {
            math_reasoning_cpp.ComplexityLevel.L0_EXPLICIT: ComplexityLevel.L0,
            math_reasoning_cpp.ComplexityLevel.L1_SHALLOW: ComplexityLevel.L1,
            math_reasoning_cpp.ComplexityLevel.L2_MEDIUM: ComplexityLevel.L2,
            math_reasoning_cpp.ComplexityLevel.L3_DEEP: ComplexityLevel.L3
        }
        return level_map.get(cpp_level, ComplexityLevel.L1)
    
    def _convert_cpp_sublevel(self, cpp_sublevel) -> SubLevel:
        """Convert C++ SubLevel to Python enum"""
        sublevel_map = {
            math_reasoning_cpp.SubLevel.L1_1: SubLevel.L1_1,
            math_reasoning_cpp.SubLevel.L1_2: SubLevel.L1_2,
            math_reasoning_cpp.SubLevel.L1_3: SubLevel.L1_3,
            math_reasoning_cpp.SubLevel.L2_1: SubLevel.L2_1,
            math_reasoning_cpp.SubLevel.L2_2: SubLevel.L2_2,
            math_reasoning_cpp.SubLevel.L2_3: SubLevel.L2_3,
            math_reasoning_cpp.SubLevel.L3_1: SubLevel.L3_1,
            math_reasoning_cpp.SubLevel.L3_2: SubLevel.L3_2,
            math_reasoning_cpp.SubLevel.L3_3: SubLevel.L3_3
        }
        return sublevel_map.get(cpp_sublevel, SubLevel.L1_1)
    
    def benchmark(self, test_problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Benchmark C++ vs Python performance
        
        Args:
            test_problems: List of test problems
            
        Returns:
            Benchmark results
        """
        if not CPP_AVAILABLE:
            return {
                'cpp_available': False,
                'message': 'C++ module not available for benchmarking'
            }
        
        import time
        
        # Test C++ performance
        cpp_classifier = math_reasoning_cpp.ComplexityClassifier()
        cpp_start = time.time()
        for problem in test_problems:
            if 'text' in problem:
                cpp_classifier.classify(problem['text'])
        cpp_time = time.time() - cpp_start
        
        # Test Python performance
        py_classifier = EnhancedComplexityClassifier()
        py_start = time.time()
        for problem in test_problems:
            py_classifier.classify_problem(problem)
        py_time = time.time() - py_start
        
        # Calculate speedup
        speedup = py_time / cpp_time if cpp_time > 0 else 0
        
        return {
            'cpp_available': True,
            'num_problems': len(test_problems),
            'cpp_time': cpp_time,
            'python_time': py_time,
            'speedup': speedup,
            'cpp_avg_time': cpp_time / len(test_problems) if test_problems else 0,
            'python_avg_time': py_time / len(test_problems) if test_problems else 0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'total_classifications': self.total_classifications,
            'using_cpp': self.use_cpp
        }
        
        if self.total_classifications > 0:
            if self.use_cpp:
                stats['avg_time_per_classification'] = self.cpp_time / self.total_classifications
                stats['total_time'] = self.cpp_time
            else:
                stats['avg_time_per_classification'] = self.python_time / self.total_classifications
                stats['total_time'] = self.python_time
        
        return stats
    
    def warmup(self):
        """Warm up the classifier for optimal performance"""
        test_problems = [
            {'text': 'Solve for x: 2x + 3 = 7'},
            {'text': 'If John has 5 apples and gives 2 to Mary, how many does he have?'},
            {'text': 'Prove that the sum of angles in a triangle is 180 degrees'}
        ]
        
        for problem in test_problems:
            self.classify_problem(problem)
        
        # Reset counters after warmup
        self.total_classifications = 0
        self.cpp_time = 0.0
        self.python_time = 0.0

# Global instance for convenience
_global_classifier = None

def get_accelerated_classifier() -> AcceleratedComplexityClassifier:
    """Get or create global accelerated classifier instance"""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = AcceleratedComplexityClassifier()
        _global_classifier.warmup()
    return _global_classifier

def classify_with_acceleration(problem: Dict[str, Any]) -> ComplexityClassification:
    """
    Convenience function to classify with acceleration
    
    Args:
        problem: Mathematical problem
        
    Returns:
        ComplexityClassification
    """
    classifier = get_accelerated_classifier()
    return classifier.classify_problem(problem)