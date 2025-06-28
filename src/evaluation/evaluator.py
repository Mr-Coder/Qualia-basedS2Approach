"""
Comprehensive Evaluator
=======================

Main evaluation engine that coordinates multiple metrics and generates reports.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .metrics import (AccuracyMetric, BaseMetric, EfficiencyMetric,
                      ExplainabilityMetric, MetricResult,
                      ReasoningQualityMetric, RobustnessMetric)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result containing all metrics"""
    dataset_name: str
    model_name: str
    metric_results: Dict[str, MetricResult]
    overall_score: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class ComprehensiveEvaluator:
    """Main evaluation engine for mathematical reasoning systems"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = self._initialize_metrics()
        
    def _initialize_metrics(self) -> Dict[str, BaseMetric]:
        """Initialize all evaluation metrics"""
        metric_configs = self.config.get('metrics', {})
        
        return {
            'accuracy': AccuracyMetric(metric_configs.get('accuracy', {})),
            'reasoning_quality': ReasoningQualityMetric(metric_configs.get('reasoning_quality', {})),
            'efficiency': EfficiencyMetric(metric_configs.get('efficiency', {})),
            'robustness': RobustnessMetric(metric_configs.get('robustness', {})),
            'explainability': ExplainabilityMetric(metric_configs.get('explainability', {}))
        }
    
    def evaluate(self, 
                 predictions: List[Any],
                 ground_truth: List[Any],
                 dataset_name: str = "unknown",
                 model_name: str = "unknown",
                 metadata: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Perform comprehensive evaluation using all metrics
        
        Args:
            predictions: List of model predictions
            ground_truth: List of ground truth answers
            dataset_name: Name of the dataset being evaluated
            model_name: Name of the model being evaluated
            metadata: Additional metadata (reasoning steps, processing times, etc.)
            
        Returns:
            EvaluationResult containing all metric scores
        """
        logger.info(f"Starting comprehensive evaluation: {model_name} on {dataset_name}")
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        metric_results = {}
        
        # Run each metric
        for metric_name, metric in self.metrics.items():
            try:
                logger.debug(f"Running metric: {metric_name}")
                result = metric.evaluate(predictions, ground_truth, metadata)
                metric_results[metric_name] = result
                logger.debug(f"Metric {metric_name} score: {result.score:.3f}")
                
            except Exception as e:
                logger.error(f"Error running metric {metric_name}: {str(e)}")
                # Create error result
                metric_results[metric_name] = MetricResult(
                    metric_name=metric_name,
                    score=0.0,
                    max_score=metric.get_max_score(),
                    details={'error': str(e)},
                    timestamp=time.time()
                )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metric_results)
        
        evaluation_result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=model_name,
            metric_results=metric_results,
            overall_score=overall_score,
            timestamp=time.time(),
            metadata={
                'total_predictions': len(predictions),
                'evaluation_config': self.config
            }
        )
        
        logger.info(f"Evaluation complete. Overall score: {overall_score:.3f}")
        return evaluation_result
    
    def _calculate_overall_score(self, metric_results: Dict[str, MetricResult]) -> float:
        """Calculate weighted overall score from individual metrics"""
        
        # Default weights
        default_weights = {
            'accuracy': 0.30,
            'reasoning_quality': 0.25,
            'efficiency': 0.15,
            'robustness': 0.15,
            'explainability': 0.15
        }
        
        # Use configured weights or defaults
        weights = self.config.get('metric_weights', default_weights)
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, result in metric_results.items():
            if metric_name in weights and 'error' not in result.details:
                weight = weights[metric_name]
                normalized_score = result.score / result.max_score
                weighted_sum += weight * normalized_score
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def evaluate_batch(self,
                       batch_predictions: List[List[Any]],
                       batch_ground_truth: List[List[Any]], 
                       dataset_names: List[str],
                       model_name: str = "unknown",
                       batch_metadata: Optional[List[Dict[str, Any]]] = None) -> List[EvaluationResult]:
        """
        Evaluate multiple datasets in batch
        
        Args:
            batch_predictions: List of prediction lists for each dataset
            batch_ground_truth: List of ground truth lists for each dataset
            dataset_names: Names of the datasets
            model_name: Name of the model being evaluated
            batch_metadata: Optional metadata for each dataset
            
        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"Starting batch evaluation for {len(dataset_names)} datasets")
        
        if not (len(batch_predictions) == len(batch_ground_truth) == len(dataset_names)):
            raise ValueError("All batch inputs must have same length")
        
        results = []
        
        for i, (predictions, ground_truth, dataset_name) in enumerate(
            zip(batch_predictions, batch_ground_truth, dataset_names)
        ):
            metadata = batch_metadata[i] if batch_metadata else None
            
            try:
                result = self.evaluate(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    metadata=metadata
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
                # Create error result
                error_result = EvaluationResult(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    metric_results={},
                    overall_score=0.0,
                    timestamp=time.time(),
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        logger.info(f"Batch evaluation complete. {len(results)} results generated")
        return results
    
    def compare_models(self,
                       model_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results
        
        Args:
            model_results: Dictionary mapping model names to their evaluation results
            
        Returns:
            Dictionary containing comparison analysis
        """
        logger.info(f"Comparing {len(model_results)} models")
        
        if len(model_results) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Extract scores for comparison
        model_scores = {}
        metric_comparisons = {}
        
        for model_name, result in model_results.items():
            model_scores[model_name] = result.overall_score
            
            for metric_name, metric_result in result.metric_results.items():
                if metric_name not in metric_comparisons:
                    metric_comparisons[metric_name] = {}
                metric_comparisons[metric_name][model_name] = metric_result.score
        
        # Find best performing model
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
        best_score = model_scores[best_model]
        
        # Calculate improvements/degradations
        comparisons = {}
        for model_name, score in model_scores.items():
            if model_name != best_model:
                improvement = best_score - score
                comparisons[model_name] = {
                    'overall_score': score,
                    'gap_from_best': improvement,
                    'relative_performance': score / best_score if best_score > 0 else 0
                }
        
        comparison_result = {
            'best_model': best_model,
            'best_overall_score': best_score,
            'model_rankings': sorted(model_scores.items(), key=lambda x: x[1], reverse=True),
            'metric_comparisons': metric_comparisons,
            'detailed_comparisons': comparisons,
            'summary': {
                'total_models': len(model_results),
                'score_range': {
                    'min': min(model_scores.values()),
                    'max': max(model_scores.values()),
                    'average': sum(model_scores.values()) / len(model_scores)
                }
            }
        }
        
        logger.info(f"Model comparison complete. Best model: {best_model} (score: {best_score:.3f})")
        return comparison_result
    
    def get_metric_weights(self) -> Dict[str, float]:
        """Get current metric weights"""
        default_weights = {
            'accuracy': 0.30,
            'reasoning_quality': 0.25,
            'efficiency': 0.15,
            'robustness': 0.15,
            'explainability': 0.15
        }
        return self.config.get('metric_weights', default_weights)
    
    def set_metric_weights(self, weights: Dict[str, float]) -> None:
        """Set new metric weights"""
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        if 'metric_weights' not in self.config:
            self.config['metric_weights'] = {}
        
        self.config['metric_weights'].update(weights)
        logger.info(f"Updated metric weights: {weights}")
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metric names"""
        return list(self.metrics.keys())
    
    def add_custom_metric(self, name: str, metric: BaseMetric) -> None:
        """Add a custom metric to the evaluator"""
        self.metrics[name] = metric
        logger.info(f"Added custom metric: {name}")
    
    def remove_metric(self, name: str) -> None:
        """Remove a metric from the evaluator"""
        if name in self.metrics:
            del self.metrics[name]
            logger.info(f"Removed metric: {name}")
        else:
            logger.warning(f"Metric {name} not found") 