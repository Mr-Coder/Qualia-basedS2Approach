#!/usr/bin/env python3
"""
Base Model Interface

This module defines the base interface for all mathematical reasoning models.
All models (baseline, LLM, and proposed) should inherit from these base classes.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModelInput:
    """Input data structure for models."""
    problem_text: str
    problem_id: Optional[str] = None
    dataset: Optional[str] = None
    complexity: Optional[str] = None
    expected_answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelOutput:
    """Output data structure for models."""
    answer: str
    reasoning_chain: List[str]
    confidence_score: float
    processing_time: float
    memory_usage: Optional[float] = None
    intermediate_steps: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelMetrics:
    """Performance metrics for model evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_processing_time: float
    avg_memory_usage: float
    total_problems_solved: int
    error_rate: float
    confidence_correlation: Optional[float] = None


class BaseModel(ABC):
    """Abstract base class for all mathematical reasoning models."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            model_name: Name identifier for the model
            config: Configuration parameters for the model
        """
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        self.is_initialized = False
        self.metrics = None
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the model. Must be implemented by subclasses.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """
        Solve a mathematical reasoning problem.
        
        Args:
            problem_input: Input problem data
            
        Returns:
            ModelOutput: Solution with reasoning chain and metadata
        """
        pass
    
    @abstractmethod
    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """
        Solve multiple problems in batch.
        
        Args:
            problems: List of input problems
            
        Returns:
            List[ModelOutput]: List of solutions
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "is_initialized": self.is_initialized,
            "metrics": self.metrics.__dict__ if self.metrics else None
        }
    
    def validate_input(self, problem_input: ModelInput) -> bool:
        """Validate input problem format."""
        if not problem_input.problem_text or not isinstance(problem_input.problem_text, str):
            self.logger.error("Invalid problem text")
            return False
        return True
    
    def update_metrics(self, new_metrics: ModelMetrics):
        """Update model performance metrics."""
        self.metrics = new_metrics


class BaselineModel(BaseModel):
    """Base class for baseline mathematical reasoning models."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.model_type = "baseline"
    
    @abstractmethod
    def extract_equations(self, problem_text: str) -> List[str]:
        """Extract mathematical equations from problem text."""
        pass
    
    @abstractmethod
    def solve_equations(self, equations: List[str]) -> Dict[str, Any]:
        """Solve extracted equations."""
        pass


class LLMModel(BaseModel):
    """Base class for Large Language Model implementations."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.model_type = "llm"
        self.api_client = None
        self.max_tokens = config.get("max_tokens", 2048) if config else 2048
        self.temperature = config.get("temperature", 0.7) if config else 0.7
    
    @abstractmethod
    def generate_prompt(self, problem_input: ModelInput) -> str:
        """Generate appropriate prompt for the LLM."""
        pass
    
    @abstractmethod
    def call_api(self, prompt: str) -> str:
        """Call the LLM API with the given prompt."""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> ModelOutput:
        """Parse LLM response into structured output."""
        pass
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API configuration information."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_client": type(self.api_client).__name__ if self.api_client else None
        }


class ProposedModel(BaseModel):
    """Base class for the proposed COT-DIR model."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.model_type = "proposed"
        self.components = {}
    
    @abstractmethod
    def implicit_relation_discovery(self, problem_input: ModelInput) -> Dict[str, Any]:
        """Discover implicit relations in the problem."""
        pass
    
    @abstractmethod
    def multi_level_reasoning(self, relations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform multi-level reasoning on discovered relations."""
        pass
    
    @abstractmethod
    def chain_verification(self, reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify the consistency and correctness of reasoning chain."""
        pass
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all model components."""
        return {
            "ird_enabled": self.components.get("ird", False),
            "mlr_enabled": self.components.get("mlr", False),
            "cv_enabled": self.components.get("cv", False)
        }


class ModelFactory:
    """Factory class for creating different types of models."""
    
    _registered_models = {}
    
    @classmethod
    def register_model(cls, model_class: type, model_type: str):
        """Register a model class with the factory."""
        cls._registered_models[model_type] = model_class
    
    @classmethod
    def create_model(cls, model_type: str, model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model (baseline, llm, proposed)
            model_name: Name of the specific model
            config: Configuration parameters
            
        Returns:
            BaseModel: Instance of the requested model
        """
        if model_type not in cls._registered_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._registered_models[model_type]
        return model_class(model_name, config)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all registered model types."""
        return list(cls._registered_models.keys())


class ModelEvaluator:
    """Utility class for evaluating model performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ModelEvaluator")
    
    def evaluate_model(self, model: BaseModel, test_problems: List[ModelInput]) -> ModelMetrics:
        """
        Evaluate a model on test problems.
        
        Args:
            model: Model to evaluate
            test_problems: List of test problems with expected answers
            
        Returns:
            ModelMetrics: Performance metrics
        """
        if not model.is_initialized:
            raise ValueError("Model must be initialized before evaluation")
        
        results = []
        total_time = 0
        total_memory = 0
        correct_answers = 0
        errors = 0
        
        self.logger.info(f"Evaluating {model.model_name} on {len(test_problems)} problems")
        
        for i, problem in enumerate(test_problems):
            try:
                start_time = time.time()
                result = model.solve_problem(problem)
                
                # Check correctness
                if problem.expected_answer and result.answer:
                    is_correct = self._check_answer_correctness(result.answer, problem.expected_answer)
                    if is_correct:
                        correct_answers += 1
                
                results.append(result)
                total_time += result.processing_time
                if result.memory_usage:
                    total_memory += result.memory_usage
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(test_problems)} problems")
                    
            except Exception as e:
                self.logger.error(f"Error processing problem {i}: {str(e)}")
                errors += 1
        
        # Calculate metrics
        accuracy = correct_answers / len(test_problems) if test_problems else 0
        error_rate = errors / len(test_problems) if test_problems else 0
        avg_time = total_time / len(test_problems) if test_problems else 0
        avg_memory = total_memory / len(test_problems) if test_problems else 0
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=accuracy,  # Simplified for now
            recall=accuracy,     # Simplified for now
            f1_score=accuracy,   # Simplified for now
            avg_processing_time=avg_time,
            avg_memory_usage=avg_memory,
            total_problems_solved=len(test_problems) - errors,
            error_rate=error_rate
        )
        
        model.update_metrics(metrics)
        return metrics
    
    def _check_answer_correctness(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected answer."""
        # Simplified comparison - can be enhanced with more sophisticated matching
        try:
            # Try numeric comparison first
            pred_num = float(predicted.strip())
            exp_num = float(expected.strip())
            return abs(pred_num - exp_num) < 1e-6
        except ValueError:
            # Fall back to string comparison
            return predicted.strip().lower() == expected.strip().lower()
    
    def compare_models(self, models: List[BaseModel], test_problems: List[ModelInput]) -> Dict[str, ModelMetrics]:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: List of models to compare
            test_problems: Test problems for evaluation
            
        Returns:
            Dict mapping model names to their metrics
        """
        comparison_results = {}
        
        for model in models:
            self.logger.info(f"Evaluating model: {model.model_name}")
            metrics = self.evaluate_model(model, test_problems)
            comparison_results[model.model_name] = metrics
        
        return comparison_results


# Export main classes
__all__ = [
    'BaseModel', 'BaselineModel', 'LLMModel', 'ProposedModel',
    'ModelInput', 'ModelOutput', 'ModelMetrics',
    'ModelFactory', 'ModelEvaluator'
] 