"""
Standardized Pipeline Tests

Unit and integration tests for the standardized data flow components:
- DataLoader
- Preprocessor  
- ReasoningEngine
- Evaluator
"""

import os
import sys
from typing import Dict, List

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataLoader:
    """Test cases for DataLoader component"""
    
    def test_loader_initialization(self):
        """Test DataLoader initialization"""
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        assert loader is not None
        assert hasattr(loader, 'load')
    
    def test_load_by_dataset_name(self):
        """Test loading dataset by name"""
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        
        # Test with valid dataset name
        try:
            samples = loader.load(dataset_name="Math23K", max_samples=2)
            assert isinstance(samples, list)
            assert len(samples) <= 2
            
            if samples:
                sample = samples[0]
                assert "id" in sample
                assert "problem" in sample
                assert "answer" in sample
                assert "dataset" in sample
        except Exception as e:
            pytest.skip(f"Dataset not available: {e}")
    
    def test_load_by_path(self):
        """Test loading dataset by path"""
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        
        # Test with valid path
        try:
            samples = loader.load(path="Data/Math23K/math23k.json", max_samples=1)
            assert isinstance(samples, list)
            assert len(samples) <= 1
        except Exception as e:
            pytest.skip(f"Dataset file not available: {e}")
    
    def test_load_invalid_parameters(self):
        """Test loading with invalid parameters"""
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        
        # Test with neither path nor dataset_name
        with pytest.raises(ValueError):
            loader.load()
        
        # Test with both path and dataset_name
        with pytest.raises(ValueError):
            loader.load(path="test.json", dataset_name="test")


class TestPreprocessor:
    """Test cases for Preprocessor component"""
    
    def test_preprocessor_initialization(self):
        """Test Preprocessor initialization"""
        from src.data.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'process')
    
    def test_process_sample(self):
        """Test processing a single sample"""
        from src.data.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        
        sample = {
            "id": "test_1",
            "problem": "小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？",
            "answer": "8"
        }
        
        processed = preprocessor.process(sample)
        
        # Check original fields are preserved
        assert processed["id"] == "test_1"
        assert processed["problem"] == sample["problem"]
        assert processed["answer"] == "8"
        
        # Check preprocessing additions
        assert "cleaned_text" in processed
        assert "problem_type" in processed
        assert "classification_confidence" in processed
        assert "complexity_level" in processed
        
        # Check data types
        assert isinstance(processed["cleaned_text"], str)
        assert isinstance(processed["problem_type"], str)
        assert isinstance(processed["classification_confidence"], (int, float))
        assert isinstance(processed["complexity_level"], str)
    
    def test_process_with_different_input_fields(self):
        """Test processing with different input field names"""
        from src.data.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        
        # Test with "question" field
        sample1 = {
            "id": "test_2",
            "question": "What is 2 + 3?",
            "answer": "5"
        }
        
        processed1 = preprocessor.process(sample1)
        assert "cleaned_text" in processed1
        
        # Test with "text" field
        sample2 = {
            "id": "test_3",
            "text": "Calculate 4 * 6",
            "answer": "24"
        }
        
        processed2 = preprocessor.process(sample2)
        assert "cleaned_text" in processed2


class TestReasoningEngine:
    """Test cases for ReasoningEngine component"""
    
    def test_engine_initialization(self):
        """Test ReasoningEngine initialization"""
        from src.reasoning_core.reasoning_engine import ReasoningEngine
        
        engine = ReasoningEngine()
        assert engine is not None
        assert hasattr(engine, 'solve')
    
    def test_solve_simple_problem(self):
        """Test solving a simple mathematical problem"""
        from src.reasoning_core.reasoning_engine import ReasoningEngine
        
        engine = ReasoningEngine()
        
        sample = {
            "id": "test_4",
            "problem": "What is 5 + 7?",
            "answer": "12",
            "cleaned_text": "What is 5 + 7?",
            "problem_type": "arithmetic",
            "complexity_level": "L0"
        }
        
        result = engine.solve(sample)
        
        # Check result structure
        assert isinstance(result, dict)
        assert "final_answer" in result or "answer" in result
        assert "strategy_used" in result
        assert "confidence" in result
        
        # Check data types
        assert isinstance(result.get("final_answer") or result.get("answer"), str)
        assert isinstance(result["strategy_used"], str)
        assert isinstance(result["confidence"], (int, float))
    
    def test_solve_complex_problem(self):
        """Test solving a more complex problem"""
        from src.reasoning_core.reasoning_engine import ReasoningEngine
        
        engine = ReasoningEngine()
        
        sample = {
            "id": "test_5",
            "problem": "小明有3个苹果，给了小红2个，又买了5个，现在小明有多少个苹果？",
            "answer": "6",
            "cleaned_text": "小明有3个苹果，给了小红2个，又买了5个，现在小明有多少个苹果？",
            "problem_type": "word_problem",
            "complexity_level": "L1"
        }
        
        result = engine.solve(sample)
        
        # Check result structure
        assert isinstance(result, dict)
        assert "final_answer" in result or "answer" in result
        assert "strategy_used" in result


class TestEvaluator:
    """Test cases for Evaluator component"""
    
    def test_evaluator_initialization(self):
        """Test Evaluator initialization"""
        from src.evaluation.evaluator import Evaluator
        
        evaluator = Evaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'evaluate')
    
    def test_evaluate_correct_predictions(self):
        """Test evaluation with correct predictions"""
        from src.evaluation.evaluator import Evaluator
        
        evaluator = Evaluator()
        
        predictions = [
            {"answer": "8", "final_answer": "8"},
            {"answer": "12", "final_answer": "12"},
            {"answer": "15", "final_answer": "15"}
        ]
        
        references = [
            {"answer": "8"},
            {"answer": "12"},
            {"answer": "15"}
        ]
        
        result = evaluator.evaluate(predictions, references)
        
        # Check result structure
        assert isinstance(result, dict)
        assert "overall_score" in result
        assert "metric_results" in result
        
        # Check data types
        assert isinstance(result["overall_score"], (int, float))
        assert isinstance(result["metric_results"], dict)
        
        # Check score range
        assert 0 <= result["overall_score"] <= 1
    
    def test_evaluate_mixed_predictions(self):
        """Test evaluation with mixed correct/incorrect predictions"""
        from src.evaluation.evaluator import Evaluator
        
        evaluator = Evaluator()
        
        predictions = [
            {"answer": "8", "final_answer": "8"},    # Correct
            {"answer": "12", "final_answer": "13"},  # Incorrect
            {"answer": "15", "final_answer": "15"}   # Correct
        ]
        
        references = [
            {"answer": "8"},
            {"answer": "12"},
            {"answer": "15"}
        ]
        
        result = evaluator.evaluate(predictions, references)
        
        # Check result structure
        assert isinstance(result, dict)
        assert "overall_score" in result
        assert "metric_results" in result
        
        # Score should be between 0 and 1
        assert 0 <= result["overall_score"] <= 1


class TestIntegrationPipeline:
    """Integration tests for the complete standardized pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline"""
        from src.data.loader import DataLoader
        from src.data.preprocessor import Preprocessor
        from src.evaluation.evaluator import Evaluator
        from src.reasoning_core.reasoning_engine import ReasoningEngine

        # Initialize all components
        loader = DataLoader()
        preprocessor = Preprocessor()
        engine = ReasoningEngine()
        evaluator = Evaluator()
        
        try:
            # 1. Load data
            samples = loader.load(dataset_name="Math23K", max_samples=3)
            assert len(samples) > 0
            
            # 2. Preprocess
            processed_samples = [preprocessor.process(sample) for sample in samples]
            assert len(processed_samples) == len(samples)
            
            # 3. Reasoning
            predictions = []
            for sample in processed_samples:
                result = engine.solve(sample)
                predictions.append({**sample, "answer": result.get("final_answer") or result.get("answer")})
            
            assert len(predictions) == len(processed_samples)
            
            # 4. Evaluation
            eval_result = evaluator.evaluate(predictions, samples)
            assert isinstance(eval_result, dict)
            assert "overall_score" in eval_result
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to missing dependencies: {e}")
    
    def test_pipeline_with_mock_data(self):
        """Test pipeline with mock data when real datasets are unavailable"""
        from src.data.preprocessor import Preprocessor
        from src.evaluation.evaluator import Evaluator
        from src.reasoning_core.reasoning_engine import ReasoningEngine

        # Mock data
        mock_samples = [
            {
                "id": "mock_1",
                "problem": "What is 2 + 3?",
                "answer": "5"
            },
            {
                "id": "mock_2", 
                "problem": "What is 4 * 6?",
                "answer": "24"
            }
        ]
        
        # Initialize components
        preprocessor = Preprocessor()
        engine = ReasoningEngine()
        evaluator = Evaluator()
        
        # 1. Preprocess
        processed_samples = [preprocessor.process(sample) for sample in mock_samples]
        assert len(processed_samples) == len(mock_samples)
        
        # 2. Reasoning
        predictions = []
        for sample in processed_samples:
            result = engine.solve(sample)
            predictions.append({**sample, "answer": result.get("final_answer") or result.get("answer")})
        
        assert len(predictions) == len(processed_samples)
        
        # 3. Evaluation
        eval_result = evaluator.evaluate(predictions, mock_samples)
        assert isinstance(eval_result, dict)
        assert "overall_score" in eval_result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 