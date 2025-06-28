"""
Unit Tests for Reasoning Strategies
==================================

Test individual reasoning strategy implementations.
"""

import pytest

from reasoning_core.strategies.base_strategy import (ReasoningResult,
                                                     ReasoningStep)
from reasoning_core.strategies.chain_of_thought import ChainOfThoughtStrategy


class TestChainOfThoughtStrategy:
    """Test cases for Chain of Thought strategy"""
    
    def test_strategy_initialization(self):
        """Test strategy can be initialized properly"""
        strategy = ChainOfThoughtStrategy()
        assert strategy.name == "chain_of_thought"
        assert strategy.max_steps == 10  # default value
        assert strategy.confidence_threshold == 0.8  # default value
    
    def test_strategy_with_custom_config(self):
        """Test strategy initialization with custom config"""
        config = {
            "max_steps": 5,
            "confidence_threshold": 0.9,
            "priority": 2
        }
        strategy = ChainOfThoughtStrategy(config)
        assert strategy.max_steps == 5
        assert strategy.confidence_threshold == 0.9
        assert strategy.get_priority() == 2
    
    def test_can_handle_method(self):
        """Test that CoT can handle various problem types"""
        strategy = ChainOfThoughtStrategy()
        
        # CoT should be able to handle any problem as a fallback
        assert strategy.can_handle("simple math problem") == True
        assert strategy.can_handle({"type": "algebra", "content": "2x + 3 = 7"}) == True
        assert strategy.can_handle(123) == True  # Should handle any input
    
    @pytest.mark.unit
    def test_solve_simple_problem(self, sample_math_problems):
        """Test solving a simple mathematical problem"""
        strategy = ChainOfThoughtStrategy()
        problem = sample_math_problems[0]  # Chinese addition problem
        
        result = strategy.solve(problem["problem"])
        
        # Verify result structure
        assert isinstance(result, ReasoningResult)
        assert result.success == True
        assert result.confidence > 0
        assert len(result.reasoning_steps) > 0
        assert result.final_answer is not None
        assert result.metadata is not None
        assert result.metadata["strategy"] == "chain_of_thought"
    
    def test_reasoning_steps_structure(self):
        """Test that reasoning steps have proper structure"""
        strategy = ChainOfThoughtStrategy()
        problem = "What is 2 + 2?"
        
        result = strategy.solve(problem)
        
        for step in result.reasoning_steps:
            assert isinstance(step, ReasoningStep)
            assert step.step_id > 0
            assert step.operation is not None
            assert step.explanation is not None
            assert step.confidence >= 0 and step.confidence <= 1
            assert step.metadata is not None
    
    def test_validate_step_method(self):
        """Test step validation logic"""
        strategy = ChainOfThoughtStrategy()
        
        # Valid step
        valid_step = ReasoningStep(
            step_id=1,
            operation="test_op",
            explanation="This is a test explanation",
            input_data="input",
            output_data="output",
            confidence=0.8,
            metadata={"test": True}
        )
        assert strategy.validate_step(valid_step) == True
        
        # Invalid step - low confidence
        invalid_step = ReasoningStep(
            step_id=1,
            operation="test_op", 
            explanation="This is a test explanation",
            input_data="input",
            output_data="output",
            confidence=0.05,  # Too low
            metadata={"test": True}
        )
        assert strategy.validate_step(invalid_step) == False
        
        # Invalid step - no explanation
        invalid_step_2 = ReasoningStep(
            step_id=1,
            operation="test_op",
            explanation="",  # Empty explanation
            input_data="input", 
            output_data="output",
            confidence=0.8,
            metadata={"test": True}
        )
        assert strategy.validate_step(invalid_step_2) == False
    
    def test_confidence_calculation(self):
        """Test overall confidence calculation from steps"""
        strategy = ChainOfThoughtStrategy()
        problem = "Test problem"
        
        result = strategy.solve(problem)
        
        # Verify confidence is reasonable
        assert 0 <= result.confidence <= 1
        
        # Confidence should be average of step confidences
        if result.reasoning_steps:
            expected_confidence = sum(
                step.confidence for step in result.reasoning_steps
            ) / len(result.reasoning_steps)
            assert abs(result.confidence - expected_confidence) < 0.01
    
    @pytest.mark.unit
    def test_max_steps_limit(self):
        """Test that strategy respects max_steps limit"""
        config = {"max_steps": 2}
        strategy = ChainOfThoughtStrategy(config)
        problem = "Complex problem requiring many steps"
        
        result = strategy.solve(problem)
        
        # Should not exceed max_steps
        assert len(result.reasoning_steps) <= 2
    
    def test_error_handling(self):
        """Test strategy handles errors gracefully"""
        strategy = ChainOfThoughtStrategy()
        
        # Test with None input
        result = strategy.solve(None)
        
        # Should still return a ReasoningResult structure
        assert isinstance(result, ReasoningResult)
        # Error cases should still attempt to provide some result
        
    @pytest.mark.smoke
    def test_strategy_basic_smoke_test(self):
        """Basic smoke test to ensure strategy doesn't crash"""
        strategy = ChainOfThoughtStrategy()
        
        test_problems = [
            "2 + 2 = ?",
            "What is the capital of France?",
            "Solve: x^2 = 16",
            123,
            {"complex": "problem structure"},
            None
        ]
        
        for problem in test_problems:
            try:
                result = strategy.solve(problem)
                assert isinstance(result, ReasoningResult)
                # Should not crash for any input
            except Exception as e:
                pytest.fail(f"Strategy crashed on problem {problem}: {str(e)}")


class TestReasoningDataStructures:
    """Test data structures used in reasoning"""
    
    def test_reasoning_step_creation(self):
        """Test ReasoningStep can be created with all fields"""
        step = ReasoningStep(
            step_id=1,
            operation="test",
            explanation="test explanation",
            input_data="input",
            output_data="output", 
            confidence=0.9,
            metadata={"key": "value"}
        )
        
        assert step.step_id == 1
        assert step.operation == "test"
        assert step.explanation == "test explanation"
        assert step.confidence == 0.9
        assert step.metadata["key"] == "value"
    
    def test_reasoning_result_creation(self):
        """Test ReasoningResult can be created with all fields"""
        steps = [
            ReasoningStep(1, "op1", "exp1", "in1", "out1", 0.8, {}),
            ReasoningStep(2, "op2", "exp2", "in2", "out2", 0.9, {})
        ]
        
        result = ReasoningResult(
            final_answer="42",
            reasoning_steps=steps,
            confidence=0.85,
            success=True,
            error_message=None,
            metadata={"strategy": "test"}
        )
        
        assert result.final_answer == "42"
        assert len(result.reasoning_steps) == 2
        assert result.confidence == 0.85
        assert result.success == True
        assert result.error_message is None
        assert result.metadata["strategy"] == "test" 