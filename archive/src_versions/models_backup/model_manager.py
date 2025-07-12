#!/usr/bin/env python3
"""
Model Manager

This module provides a unified interface for managing and using all mathematical reasoning models.
It includes model registry, configuration management, and batch processing capabilities.
"""

import concurrent.futures
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .base_model import (BaselineModel, BaseModel, LLMModel, ModelEvaluator,
                         ModelFactory, ModelInput, ModelMetrics, ModelOutput,
                         ProposedModel)
from .baseline_models import (EquationBasedModel, RuleBasedModel,
                              TemplateBasedModel)
from .llm_models import (ClaudeModel, DeepSeekMathModel, InternLMModel,
                         OpenAIGPTModel, QwenModel)
from .proposed_model import COTDIRModel
from .simple_pattern_model import SimplePatternModel


class ModelRegistry:
    """Registry for all available models."""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.logger = logging.getLogger(f"{__name__}.ModelRegistry")
        self._register_default_models()
    
    def _register_default_models(self):
        """Register all default models."""
        # Baseline models
        self.register_model("template_baseline", TemplateBasedModel)
        self.register_model("equation_baseline", EquationBasedModel)
        self.register_model("rule_baseline", RuleBasedModel)
        self.register_model("simple_pattern_solver", SimplePatternModel)
        
        # LLM models
        self.register_model("gpt4o", OpenAIGPTModel)
        self.register_model("claude", ClaudeModel)
        self.register_model("qwen", QwenModel)
        self.register_model("internlm", InternLMModel)
        self.register_model("deepseek", DeepSeekMathModel)
        
        # Proposed model
        self.register_model("cotdir", COTDIRModel)
        
        self.logger.info(f"Registered {len(self.models)} default models")
    
    def register_model(self, name: str, model_class: Type[BaseModel]):
        """Register a model class."""
        self.models[name] = model_class
        self.logger.debug(f"Registered model: {name}")
    
    def get_model_class(self, name: str) -> Optional[Type[BaseModel]]:
        """Get model class by name."""
        return self.models.get(name)
    
    def list_models(self) -> Dict[str, str]:
        """List all registered models."""
        return {name: cls.__name__ for name, cls in self.models.items()}
    
    def get_models_by_type(self, model_type: str) -> Dict[str, Type[BaseModel]]:
        """Get models by type (baseline, llm, proposed)."""
        filtered_models = {}
        for name, cls in self.models.items():
            if model_type == "baseline" and issubclass(cls, BaselineModel):
                filtered_models[name] = cls
            elif model_type == "llm" and issubclass(cls, LLMModel):
                filtered_models[name] = cls
            elif model_type == "proposed" and issubclass(cls, ProposedModel):
                filtered_models[name] = cls
        return filtered_models


class ModelManager:
    """Unified manager for all mathematical reasoning models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.registry = ModelRegistry()
        self.active_models = {}
        self.evaluator = ModelEvaluator()
        self.logger = logging.getLogger(f"{__name__}.ModelManager")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize logging
        self._setup_logging()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "models": {
                "gpt4o": {
                    "enabled": False,
                    "api_key": None,
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "claude": {
                    "enabled": False,
                    "api_key": None,
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "qwen": {
                    "enabled": True,
                    "is_local": True,
                    "base_url": "http://localhost:8000/v1"
                },
                "internlm": {
                    "enabled": True,
                    "is_local": True,
                    "base_url": "http://localhost:8001/v1"
                },
                "deepseek": {
                    "enabled": True,
                    "is_local": True,
                    "base_url": "http://localhost:8002/v1"
                },
                "cotdir": {
                    "enabled": True,
                    "enable_ird": True,
                    "enable_mlr": True,
                    "enable_cv": True,
                    "confidence_threshold": 0.7
                },
                "template_baseline": {"enabled": True},
                "equation_baseline": {"enabled": True},
                "rule_baseline": {"enabled": True},
                "simple_pattern_solver": {"enabled": True}
            },
            "evaluation": {
                "timeout": 30,
                "max_workers": 4,
                "retry_attempts": 3
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                # Merge configurations
                default_config.update(file_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        logging.basicConfig(level=level, format=format_str)
    
    def initialize_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a specific model.
        
        Args:
            model_name: Name of the model to initialize
            config: Optional configuration override
            
        Returns:
            bool: True if initialization successful
        """
        if model_name in self.active_models:
            self.logger.info(f"Model {model_name} already initialized")
            return True
        
        model_class = self.registry.get_model_class(model_name)
        if not model_class:
            self.logger.error(f"Unknown model: {model_name}")
            return False
        
        # Get model configuration
        model_config = self.config.get("models", {}).get(model_name, {})
        if config:
            model_config.update(config)
        
        if not model_config.get("enabled", True):
            self.logger.info(f"Model {model_name} is disabled in configuration")
            return False
        
        try:
            # Create and initialize model
            model_instance = model_class(config=model_config)
            if model_instance.initialize():
                self.active_models[model_name] = model_instance
                self.logger.info(f"Successfully initialized model: {model_name}")
                return True
            else:
                self.logger.error(f"Failed to initialize model: {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing model {model_name}: {e}")
            return False
    
    def initialize_all_models(self) -> Dict[str, bool]:
        """Initialize all enabled models."""
        results = {}
        model_configs = self.config.get("models", {})
        
        for model_name, config in model_configs.items():
            if config.get("enabled", True):
                results[model_name] = self.initialize_model(model_name)
            else:
                results[model_name] = False
                self.logger.info(f"Skipping disabled model: {model_name}")
        
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"Initialized {successful}/{len(results)} models")
        return results
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Get an initialized model."""
        return self.active_models.get(model_name)
    
    def solve_problem(self, model_name: str, problem: Union[str, ModelInput]) -> Optional[ModelOutput]:
        """
        Solve a problem using a specific model.
        
        Args:
            model_name: Name of the model to use
            problem: Problem text or ModelInput object
            
        Returns:
            ModelOutput or None if model not available
        """
        model = self.get_model(model_name)
        if not model:
            self.logger.error(f"Model {model_name} not available")
            return None
        
        # Convert string to ModelInput if needed
        if isinstance(problem, str):
            problem_input = ModelInput(problem_text=problem)
        else:
            problem_input = problem
        
        try:
            return model.solve_problem(problem_input)
        except Exception as e:
            self.logger.error(f"Error solving problem with {model_name}: {e}")
            return None
    
    def solve_with_multiple_models(self, models: List[str], problem: Union[str, ModelInput]) -> Dict[str, Optional[ModelOutput]]:
        """
        Solve a problem using multiple models.
        
        Args:
            models: List of model names to use
            problem: Problem text or ModelInput object
            
        Returns:
            Dictionary mapping model names to their outputs
        """
        results = {}
        
        # Convert string to ModelInput if needed
        if isinstance(problem, str):
            problem_input = ModelInput(problem_text=problem)
        else:
            problem_input = problem
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = self.config.get("evaluation", {}).get("max_workers", 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(self.solve_problem, model_name, problem_input): model_name
                for model_name in models
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=self.config.get("evaluation", {}).get("timeout", 30))
                    results[model_name] = result
                except Exception as e:
                    self.logger.error(f"Error with model {model_name}: {e}")
                    results[model_name] = None
        
        return results
    
    def batch_solve(self, model_name: str, problems: List[Union[str, ModelInput]]) -> List[Optional[ModelOutput]]:
        """
        Solve multiple problems using a single model.
        
        Args:
            model_name: Name of the model to use
            problems: List of problems
            
        Returns:
            List of model outputs
        """
        model = self.get_model(model_name)
        if not model:
            self.logger.error(f"Model {model_name} not available")
            return [None] * len(problems)
        
        # Convert strings to ModelInput objects
        problem_inputs = []
        for problem in problems:
            if isinstance(problem, str):
                problem_inputs.append(ModelInput(problem_text=problem))
            else:
                problem_inputs.append(problem)
        
        try:
            return model.batch_solve(problem_inputs)
        except Exception as e:
            self.logger.error(f"Error in batch solving with {model_name}: {e}")
            return [None] * len(problems)
    
    def evaluate_model(self, model_name: str, test_problems: List[ModelInput]) -> Optional[ModelMetrics]:
        """
        Evaluate a specific model.
        
        Args:
            model_name: Name of the model to evaluate
            test_problems: List of test problems with expected answers
            
        Returns:
            ModelMetrics or None if evaluation failed
        """
        model = self.get_model(model_name)
        if not model:
            self.logger.error(f"Model {model_name} not available for evaluation")
            return None
        
        try:
            return self.evaluator.evaluate_model(model, test_problems)
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {e}")
            return None
    
    def compare_models(self, model_names: List[str], test_problems: List[ModelInput]) -> Dict[str, Optional[ModelMetrics]]:
        """
        Compare multiple models on the same test set.
        
        Args:
            model_names: List of model names to compare
            test_problems: List of test problems
            
        Returns:
            Dictionary mapping model names to their metrics
        """
        results = {}
        
        for model_name in model_names:
            self.logger.info(f"Evaluating model: {model_name}")
            metrics = self.evaluate_model(model_name, test_problems)
            results[model_name] = metrics
        
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        model = self.get_model(model_name)
        if not model:
            return None
        
        info = model.get_model_info()
        
        # Add additional info for LLM models
        if isinstance(model, LLMModel):
            info.update(model.get_api_info())
        
        # Add component info for proposed models
        if isinstance(model, ProposedModel):
            info["components"] = model.get_component_status()
        
        return info
    
    def list_active_models(self) -> List[str]:
        """List all active (initialized) models."""
        return list(self.active_models.keys())
    
    def list_available_models(self) -> Dict[str, str]:
        """List all available models."""
        return self.registry.list_models()
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def load_test_problems(self, file_path: str) -> List[ModelInput]:
        """Load test problems from file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            problems = []
            for item in data:
                if isinstance(item, dict):
                    problems.append(ModelInput(**item))
                elif isinstance(item, str):
                    problems.append(ModelInput(problem_text=item))
            
            self.logger.info(f"Loaded {len(problems)} test problems from {file_path}")
            return problems
            
        except Exception as e:
            self.logger.error(f"Failed to load test problems: {e}")
            return []
    
    def create_model_comparison_report(self, comparison_results: Dict[str, Optional[ModelMetrics]]) -> Dict[str, Any]:
        """Create a comprehensive comparison report."""
        report = {
            "summary": {},
            "detailed_metrics": {},
            "rankings": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Extract metrics for comparison
        valid_results = {name: metrics for name, metrics in comparison_results.items() if metrics is not None}
        
        if not valid_results:
            report["summary"]["note"] = "No valid results to compare"
            return report
        
        # Calculate rankings
        metrics_to_rank = ["accuracy", "precision", "recall", "f1_score", "avg_processing_time"]
        
        for metric in metrics_to_rank:
            values = [(name, getattr(metrics, metric)) for name, metrics in valid_results.items()]
            
            if metric == "avg_processing_time":
                # Lower is better for processing time
                ranked = sorted(values, key=lambda x: x[1])
            else:
                # Higher is better for other metrics
                ranked = sorted(values, key=lambda x: x[1], reverse=True)
            
            report["rankings"][metric] = [{"model": name, "value": value} for name, value in ranked]
        
        # Summary statistics
        if valid_results:
            best_accuracy = max(valid_results.items(), key=lambda x: x[1].accuracy)
            fastest_model = min(valid_results.items(), key=lambda x: x[1].avg_processing_time)
            
            report["summary"]["best_accuracy"] = {
                "model": best_accuracy[0],
                "accuracy": best_accuracy[1].accuracy
            }
            report["summary"]["fastest_model"] = {
                "model": fastest_model[0],
                "avg_time": fastest_model[1].avg_processing_time
            }
            report["summary"]["total_models"] = len(valid_results)
        
        # Detailed metrics
        for model_name, metrics in valid_results.items():
            report["detailed_metrics"][model_name] = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "avg_processing_time": metrics.avg_processing_time,
                "avg_memory_usage": metrics.avg_memory_usage,
                "total_problems_solved": metrics.total_problems_solved,
                "error_rate": metrics.error_rate
            }
        
        return report
    
    def shutdown(self):
        """Shutdown all models and cleanup resources."""
        self.logger.info("Shutting down model manager")
        self.active_models.clear()


# Create a global instance for easy access
default_manager = ModelManager() 