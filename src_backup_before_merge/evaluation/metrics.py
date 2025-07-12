"""
Evaluation Metrics
=================

Various metrics for evaluating mathematical reasoning systems.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a metric evaluation"""
    metric_name: str
    score: float
    max_score: float
    details: Dict[str, Any]
    timestamp: float


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
    @abstractmethod
    def evaluate(self, predictions: List[Any], ground_truth: List[Any], 
                 metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Evaluate predictions against ground truth"""
        pass
        
    @abstractmethod
    def get_max_score(self) -> float:
        """Get maximum possible score for this metric"""
        pass


class AccuracyMetric(BaseMetric):
    """Accuracy metric for exact match evaluation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("accuracy", config)
        self.tolerance = self.config.get('tolerance', 0.0)
        
    def evaluate(self, predictions: List[Any], ground_truth: List[Any], 
                 metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Calculate accuracy score"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truth):
            if self._is_correct(pred, gt):
                correct += 1
        
        score = correct / total if total > 0 else 0.0
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            max_score=1.0,
            details={
                'correct': correct,
                'total': total,
                'tolerance': self.tolerance,
                'accuracy_percentage': score * 100
            },
            timestamp=time.time()
        )
    
    def get_max_score(self) -> float:
        return 1.0
    
    def _is_correct(self, prediction: Any, ground_truth: Any) -> bool:
        """Check if prediction matches ground truth within tolerance"""
        try:
            # Try numeric comparison with tolerance
            pred_num = float(prediction)
            gt_num = float(ground_truth)
            return abs(pred_num - gt_num) <= self.tolerance
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(prediction).strip().lower() == str(ground_truth).strip().lower()


class ReasoningQualityMetric(BaseMetric):
    """Metric for evaluating quality of reasoning steps"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("reasoning_quality", config)
        self.min_steps = self.config.get('min_steps', 1)
        self.max_steps = self.config.get('max_steps', 20)
        
    def evaluate(self, predictions: List[Any], ground_truth: List[Any], 
                 metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Evaluate reasoning quality based on step structure and coherence"""
        
        if not metadata or 'reasoning_steps' not in metadata:
            logger.warning("No reasoning steps provided for quality evaluation")
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                max_score=1.0,
                details={'error': 'No reasoning steps provided'},
                timestamp=time.time()
            )
        
        all_steps = metadata['reasoning_steps']
        quality_scores = []
        
        for steps in all_steps:
            quality_score = self._evaluate_reasoning_steps(steps)
            quality_scores.append(quality_score)
        
        overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return MetricResult(
            metric_name=self.name,
            score=overall_score,
            max_score=1.0,
            details={
                'individual_scores': quality_scores,
                'average_steps_count': sum(len(steps) for steps in all_steps) / len(all_steps),
                'min_steps_threshold': self.min_steps,
                'max_steps_threshold': self.max_steps
            },
            timestamp=time.time()
        )
    
    def get_max_score(self) -> float:
        return 1.0
    
    def _evaluate_reasoning_steps(self, steps: List[Any]) -> float:
        """Evaluate quality of individual reasoning steps"""
        if not steps:
            return 0.0
        
        # Check step count
        step_count_score = 1.0
        if len(steps) < self.min_steps:
            step_count_score = 0.5
        elif len(steps) > self.max_steps:
            step_count_score = 0.8
        
        # Check step coherence (simplified)
        coherence_score = 0.0
        confidence_sum = 0.0
        
        for step in steps:
            if hasattr(step, 'confidence'):
                confidence_sum += step.confidence
            if hasattr(step, 'explanation') and step.explanation:
                coherence_score += 0.5
        
        if steps:
            coherence_score /= len(steps)
            avg_confidence = confidence_sum / len(steps)
        else:
            avg_confidence = 0.0
        
        # Combine scores
        final_score = (step_count_score * 0.3 + 
                      coherence_score * 0.4 + 
                      avg_confidence * 0.3)
        
        return min(final_score, 1.0)


class EfficiencyMetric(BaseMetric):
    """Metric for evaluating computational efficiency"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("efficiency", config)
        self.target_time = self.config.get('target_time', 5.0)  # seconds
        self.max_time = self.config.get('max_time', 30.0)  # seconds
        
    def evaluate(self, predictions: List[Any], ground_truth: List[Any], 
                 metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Evaluate efficiency based on processing time"""
        
        if not metadata or 'processing_times' not in metadata:
            logger.warning("No processing times provided for efficiency evaluation")
            return MetricResult(
                metric_name=self.name,
                score=0.5,  # Default middle score
                max_score=1.0,
                details={'error': 'No processing times provided'},
                timestamp=time.time()
            )
        
        processing_times = metadata['processing_times']
        
        if len(processing_times) != len(predictions):
            logger.warning("Processing times count doesn't match predictions count")
        
        # Calculate efficiency scores
        efficiency_scores = []
        for time_taken in processing_times:
            if time_taken <= self.target_time:
                score = 1.0
            elif time_taken <= self.max_time:
                # Linear interpolation between target and max time
                score = 1.0 - (time_taken - self.target_time) / (self.max_time - self.target_time)
            else:
                score = 0.0
            efficiency_scores.append(score)
        
        overall_score = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
        
        return MetricResult(
            metric_name=self.name,
            score=overall_score,
            max_score=1.0,
            details={
                'individual_times': processing_times,
                'individual_scores': efficiency_scores,
                'average_time': sum(processing_times) / len(processing_times),
                'target_time': self.target_time,
                'max_time': self.max_time
            },
            timestamp=time.time()
        )
    
    def get_max_score(self) -> float:
        return 1.0


class RobustnessMetric(BaseMetric):
    """Metric for evaluating system robustness to various inputs"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("robustness", config)
        self.error_penalty = self.config.get('error_penalty', 0.1)
        
    def evaluate(self, predictions: List[Any], ground_truth: List[Any], 
                 metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Evaluate robustness based on error handling and consistency"""
        
        if not metadata:
            metadata = {}
        
        # Count successful vs failed predictions
        success_count = 0
        error_count = 0
        
        for pred in predictions:
            if pred is not None and pred != "ERROR":
                success_count += 1
            else:
                error_count += 1
        
        total = len(predictions)
        success_rate = success_count / total if total > 0 else 0.0
        
        # Check for consistency in confidence levels
        confidence_consistency = 1.0
        if 'confidence_scores' in metadata:
            confidences = metadata['confidence_scores']
            if confidences:
                confidence_std = self._calculate_std(confidences)
                # Lower standard deviation indicates more consistent confidence
                confidence_consistency = max(0.0, 1.0 - confidence_std)
        
        # Combine metrics
        robustness_score = (success_rate * 0.7 + 
                           confidence_consistency * 0.3 - 
                           error_count * self.error_penalty)
        
        robustness_score = max(0.0, min(1.0, robustness_score))
        
        return MetricResult(
            metric_name=self.name,
            score=robustness_score,
            max_score=1.0,
            details={
                'success_count': success_count,
                'error_count': error_count,
                'success_rate': success_rate,
                'confidence_consistency': confidence_consistency,
                'error_penalty_applied': error_count * self.error_penalty
            },
            timestamp=time.time()
        )
    
    def get_max_score(self) -> float:
        return 1.0
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


class ExplainabilityMetric(BaseMetric):
    """Metric for evaluating explainability of reasoning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("explainability", config)
        self.min_explanation_length = self.config.get('min_explanation_length', 10)
        
    def evaluate(self, predictions: List[Any], ground_truth: List[Any], 
                 metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Evaluate explainability based on explanation quality"""
        
        if not metadata or 'explanations' not in metadata:
            logger.warning("No explanations provided for explainability evaluation")
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                max_score=1.0,
                details={'error': 'No explanations provided'},
                timestamp=time.time()
            )
        
        explanations = metadata['explanations']
        explainability_scores = []
        
        for explanation in explanations:
            score = self._evaluate_explanation(explanation)
            explainability_scores.append(score)
        
        overall_score = sum(explainability_scores) / len(explainability_scores) if explainability_scores else 0.0
        
        return MetricResult(
            metric_name=self.name,
            score=overall_score,
            max_score=1.0,
            details={
                'individual_scores': explainability_scores,
                'average_explanation_length': sum(len(str(exp)) for exp in explanations) / len(explanations),
                'min_length_threshold': self.min_explanation_length
            },
            timestamp=time.time()
        )
    
    def get_max_score(self) -> float:
        return 1.0
    
    def _evaluate_explanation(self, explanation: Any) -> float:
        """Evaluate quality of individual explanation"""
        if not explanation:
            return 0.0
        
        explanation_text = str(explanation)
        
        # Check length
        length_score = 1.0 if len(explanation_text) >= self.min_explanation_length else 0.5
        
        # Check for mathematical terms (simplified heuristic)
        math_terms = ['calculate', 'solve', 'equation', 'add', 'subtract', 'multiply', 'divide', 'equals']
        math_score = sum(1 for term in math_terms if term in explanation_text.lower()) / len(math_terms)
        
        # Check for step indicators
        step_indicators = ['first', 'then', 'next', 'finally', 'step', 'therefore']
        step_score = sum(1 for indicator in step_indicators if indicator in explanation_text.lower()) / len(step_indicators)
        
        # Combine scores
        final_score = (length_score * 0.4 + 
                      math_score * 0.3 + 
                      step_score * 0.3)
        
        return min(final_score, 1.0) 