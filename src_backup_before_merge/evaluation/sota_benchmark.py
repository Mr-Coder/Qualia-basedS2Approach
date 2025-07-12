#!/usr/bin/env python3
"""
SOTA Benchmark Suite
===================

Implementation of the multi-dataset evaluation framework described in the paper.
Supports comparison with state-of-the-art methods on 11 mathematical reasoning datasets.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DatasetInfo:
    """Dataset information matching paper specifications"""
    name: str
    problems: int
    language: str
    level: str
    l0_percent: float
    l1_percent: float 
    l2_percent: float
    l3_percent: float
    dir_score: float

@dataclass
class BenchmarkResult:
    """Benchmark result structure"""
    method_name: str
    overall_accuracy: float
    l0_accuracy: float
    l1_accuracy: float
    l2_accuracy: float
    l3_accuracy: float
    relation_f1: float
    efficiency_seconds: float
    dataset_results: Dict[str, float]

class SOTABenchmarkSuite:
    """
    State-of-the-art benchmark suite implementing the paper's evaluation framework
    """
    
    def __init__(self, data_path: str = "Data"):
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)
        
        # Dataset specifications from paper
        self.datasets = {
            'AddSub': DatasetInfo('AddSub', 395, 'English', 'Elementary', 75.0, 20.0, 5.0, 0.0, 0.19),
            'MAWPS': DatasetInfo('MAWPS', 1200, 'English', 'Elementary', 90.0, 10.0, 0.0, 0.0, 0.13),
            'SingleEq': DatasetInfo('SingleEq', 508, 'English', 'Elementary', 85.0, 15.0, 0.0, 0.0, 0.14),
            'MultiArith': DatasetInfo('MultiArith', 600, 'English', 'Elementary', 60.0, 30.0, 10.0, 0.0, 0.25),
            'GSM8K': DatasetInfo('GSM8K', 1319, 'English', 'Grade 3-8', 50.0, 35.0, 15.0, 0.0, 0.30),
            'SVAMP': DatasetInfo('SVAMP', 1000, 'English', 'Grade 3-8', 45.0, 35.0, 20.0, 0.0, 0.33),
            'ASDiv': DatasetInfo('ASDiv', 1000, 'English', 'Grade 3-12', 50.0, 35.0, 15.0, 0.0, 0.30),
            'Math23K': DatasetInfo('Math23K', 3000, 'Chinese', 'Grade 3-9', 30.0, 40.0, 25.0, 5.0, 0.42),
            'MATH': DatasetInfo('MATH', 1500, 'English', 'Competition', 15.0, 25.0, 35.0, 25.0, 0.67),
            'GSM-Hard': DatasetInfo('GSM-Hard', 1319, 'English', 'Advanced', 25.0, 35.0, 30.0, 10.0, 0.52),
            'MathQA': DatasetInfo('MathQA', 2000, 'English', 'Competition', 20.0, 30.0, 35.0, 15.0, 0.58)
        }
        
        # SOTA baseline results from paper
        self.sota_baselines = {
            'GPT-4o': BenchmarkResult(
                'GPT-4o', 0.722, 0.892, 0.751, 0.634, 0.412, 0.681, 2.1, {}
            ),
            'Claude-3.5-Sonnet': BenchmarkResult(
                'Claude-3.5-Sonnet', 0.711, 0.885, 0.743, 0.618, 0.398, 0.672, 2.3, {}
            ),
            'Qwen2.5-Math-72B': BenchmarkResult(
                'Qwen2.5-Math-72B', 0.738, 0.903, 0.768, 0.651, 0.429, 0.695, 1.8, {}
            ),
            'DeepSeek-Math-7B': BenchmarkResult(
                'DeepSeek-Math-7B', 0.698, 0.876, 0.732, 0.598, 0.387, 0.663, 1.5, {}
            ),
            'Tree-of-Thought': BenchmarkResult(
                'Tree-of-Thought', 0.730, 0.901, 0.761, 0.641, 0.418, 0.692, 8.7, {}
            )
        }
    
    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """Load dataset problems from data directory"""
        dataset_path = self.data_path / dataset_name
        
        if not dataset_path.exists():
            self.logger.warning(f"Dataset {dataset_name} not found at {dataset_path}")
            return []
        
        # Try to load from various possible file formats
        problems = []
        for file_pattern in ['*.json', '*.jsonl', '*.txt']:
            for file_path in dataset_path.glob(file_pattern):
                try:
                    if file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                problems.extend(data)
                            else:
                                problems.append(data)
                    elif file_path.suffix == '.jsonl':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    problems.append(json.loads(line))
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
        
        return problems[:self.datasets[dataset_name].problems]  # Limit to expected size
    
    def classify_complexity(self, problem: Dict) -> str:
        """Classify problem complexity level (L0-L3)"""
        # This is a simplified classification
        # In practice, this would use the trained complexity classifier
        
        # Check if complexity is already annotated
        if 'complexity_level' in problem:
            return problem['complexity_level']
        
        # Simple heuristic based on problem characteristics
        text = problem.get('problem', problem.get('question', ''))
        
        # Count mathematical operations and keywords
        complexity_indicators = [
            'equation', 'system', 'quadratic', 'algebra', 'geometry',
            'calculus', 'derivative', 'integral', 'matrix', 'probability'
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in text.lower())
        step_count = len(text.split('.'))  # Rough estimate of reasoning steps
        
        if step_count <= 2 and indicator_count == 0:
            return 'L0'
        elif step_count <= 4 and indicator_count <= 1:
            return 'L1'
        elif step_count <= 8 and indicator_count <= 3:
            return 'L2'
        else:
            return 'L3'
    
    def evaluate_method(self, method_func, method_name: str, test_subset: Optional[int] = None) -> BenchmarkResult:
        """
        Evaluate a method against the benchmark suite
        
        Args:
            method_func: Function that takes a problem dict and returns (answer, reasoning_steps, relations)
            method_name: Name of the method being evaluated
            test_subset: If provided, only test on this many problems per dataset
        """
        self.logger.info(f"Evaluating method: {method_name}")
        
        total_correct = 0
        total_problems = 0
        complexity_stats = {'L0': {'correct': 0, 'total': 0},
                           'L1': {'correct': 0, 'total': 0},
                           'L2': {'correct': 0, 'total': 0},
                           'L3': {'correct': 0, 'total': 0}}
        
        dataset_results = {}
        relation_f1_scores = []
        efficiency_times = []
        
        for dataset_name, dataset_info in self.datasets.items():
            self.logger.info(f"Evaluating on {dataset_name}...")
            
            problems = self.load_dataset(dataset_name)
            if not problems:
                self.logger.warning(f"No problems loaded for {dataset_name}")
                continue
            
            if test_subset:
                problems = problems[:test_subset]
            
            dataset_correct = 0
            dataset_total = len(problems)
            
            for problem in problems:
                start_time = time.time()
                
                try:
                    # Get method prediction
                    result = method_func(problem)
                    if isinstance(result, tuple):
                        answer, reasoning_steps, relations = result
                    else:
                        answer = result
                        reasoning_steps = []
                        relations = []
                    
                    # Check correctness
                    correct_answer = problem.get('answer', problem.get('solution'))
                    is_correct = self._check_answer_correctness(answer, correct_answer)
                    
                    if is_correct:
                        dataset_correct += 1
                        total_correct += 1
                    
                    # Track complexity-wise performance
                    complexity = self.classify_complexity(problem)
                    complexity_stats[complexity]['total'] += 1
                    if is_correct:
                        complexity_stats[complexity]['correct'] += 1
                    
                    # Calculate relation F1 if relations are provided
                    if relations and 'relations' in problem:
                        relation_f1 = self._calculate_relation_f1(relations, problem['relations'])
                        relation_f1_scores.append(relation_f1)
                    
                    # Track efficiency
                    efficiency_times.append(time.time() - start_time)
                    
                except Exception as e:
                    self.logger.warning(f"Error evaluating problem in {dataset_name}: {e}")
                
                total_problems += 1
            
            dataset_accuracy = dataset_correct / dataset_total if dataset_total > 0 else 0
            dataset_results[dataset_name] = dataset_accuracy
            self.logger.info(f"{dataset_name}: {dataset_accuracy:.3f} accuracy")
        
        # Calculate final metrics
        overall_accuracy = total_correct / total_problems if total_problems > 0 else 0
        
        complexity_accuracies = {}
        for level in ['L0', 'L1', 'L2', 'L3']:
            stats = complexity_stats[level]
            complexity_accuracies[level] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        avg_relation_f1 = np.mean(relation_f1_scores) if relation_f1_scores else 0
        avg_efficiency = np.mean(efficiency_times) if efficiency_times else 0
        
        result = BenchmarkResult(
            method_name=method_name,
            overall_accuracy=overall_accuracy,
            l0_accuracy=complexity_accuracies['L0'],
            l1_accuracy=complexity_accuracies['L1'],
            l2_accuracy=complexity_accuracies['L2'],
            l3_accuracy=complexity_accuracies['L3'],
            relation_f1=avg_relation_f1,
            efficiency_seconds=avg_efficiency,
            dataset_results=dataset_results
        )
        
        self.logger.info(f"Evaluation complete. Overall accuracy: {overall_accuracy:.3f}")
        return result
    
    def _check_answer_correctness(self, predicted: Any, ground_truth: Any) -> bool:
        """Check if predicted answer matches ground truth"""
        if predicted is None or ground_truth is None:
            return False
        
        # Handle numeric answers
        try:
            pred_num = float(str(predicted).strip())
            gt_num = float(str(ground_truth).strip())
            return abs(pred_num - gt_num) < 1e-6
        except:
            pass
        
        # Handle string answers
        pred_str = str(predicted).strip().lower()
        gt_str = str(ground_truth).strip().lower()
        return pred_str == gt_str
    
    def _calculate_relation_f1(self, predicted_relations: List, ground_truth_relations: List) -> float:
        """Calculate F1 score for relation discovery"""
        if not predicted_relations or not ground_truth_relations:
            return 0.0
        
        pred_set = set(tuple(r) if isinstance(r, list) else r for r in predicted_relations)
        gt_set = set(tuple(r) if isinstance(r, list) else r for r in ground_truth_relations)
        
        if not pred_set and not gt_set:
            return 1.0
        
        intersection = len(pred_set & gt_set)
        precision = intersection / len(pred_set) if pred_set else 0
        recall = intersection / len(gt_set) if gt_set else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def compare_with_sota(self, result: BenchmarkResult) -> Dict[str, float]:
        """Compare result with SOTA baselines"""
        comparisons = {}
        
        for baseline_name, baseline_result in self.sota_baselines.items():
            improvement = result.overall_accuracy - baseline_result.overall_accuracy
            comparisons[baseline_name] = improvement
        
        return comparisons
    
    def generate_benchmark_report(self, result: BenchmarkResult, output_path: str = "benchmark_report.json"):
        """Generate comprehensive benchmark report"""
        
        comparisons = self.compare_with_sota(result)
        best_baseline = max(self.sota_baselines.values(), key=lambda x: x.overall_accuracy)
        
        report = {
            'method_name': result.method_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_performance': {
                'accuracy': result.overall_accuracy,
                'relation_f1': result.relation_f1,
                'efficiency_seconds': result.efficiency_seconds
            },
            'complexity_breakdown': {
                'L0_accuracy': result.l0_accuracy,
                'L1_accuracy': result.l1_accuracy,
                'L2_accuracy': result.l2_accuracy,
                'L3_accuracy': result.l3_accuracy
            },
            'dataset_results': result.dataset_results,
            'sota_comparisons': comparisons,
            'best_improvement': {
                'baseline': best_baseline.method_name,
                'improvement': result.overall_accuracy - best_baseline.overall_accuracy,
                'improvement_percent': (result.overall_accuracy - best_baseline.overall_accuracy) * 100
            },
            'statistical_significance': self._assess_statistical_significance(result, best_baseline),
            'dataset_statistics': {
                'total_datasets': len(self.datasets),
                'total_problems': sum(ds.problems for ds in self.datasets.values()),
                'language_distribution': self._get_language_distribution(),
                'complexity_distribution': self._get_complexity_distribution()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Benchmark report saved to {output_path}")
        return report
    
    def _assess_statistical_significance(self, result: BenchmarkResult, baseline: BenchmarkResult) -> Dict:
        """Assess statistical significance of improvement"""
        # Simplified significance assessment
        improvement = result.overall_accuracy - baseline.overall_accuracy
        
        # Rule of thumb: improvement > 0.01 (1%) is likely significant for large datasets
        total_problems = sum(ds.problems for ds in self.datasets.values())
        min_significant_improvement = 1.96 * np.sqrt(0.25 / total_problems)  # Conservative estimate
        
        return {
            'improvement': improvement,
            'min_significant_improvement': min_significant_improvement,
            'is_significant': improvement > min_significant_improvement,
            'confidence_level': 0.95,
            'note': 'Simplified significance test. Full bootstrap analysis recommended.'
        }
    
    def _get_language_distribution(self) -> Dict[str, int]:
        """Get language distribution of datasets"""
        distribution = {}
        for dataset in self.datasets.values():
            distribution[dataset.language] = distribution.get(dataset.language, 0) + dataset.problems
        return distribution
    
    def _get_complexity_distribution(self) -> Dict[str, float]:
        """Get overall complexity distribution"""
        total_problems = sum(ds.problems for ds in self.datasets.values())
        distribution = {'L0': 0, 'L1': 0, 'L2': 0, 'L3': 0}
        
        for dataset in self.datasets.values():
            distribution['L0'] += dataset.problems * dataset.l0_percent / 100
            distribution['L1'] += dataset.problems * dataset.l1_percent / 100
            distribution['L2'] += dataset.problems * dataset.l2_percent / 100
            distribution['L3'] += dataset.problems * dataset.l3_percent / 100
        
        # Convert to percentages
        for level in distribution:
            distribution[level] = (distribution[level] / total_problems) * 100
        
        return distribution
    
    def run_ablation_study(self, base_method, components: List[Tuple[str, callable]], test_subset: int = 100):
        """Run ablation study to measure component contributions"""
        self.logger.info("Running ablation study...")
        
        results = {}
        
        # Baseline
        baseline_result = self.evaluate_method(base_method, "Baseline", test_subset)
        results["Baseline"] = baseline_result
        
        # Progressive addition of components
        current_method = base_method
        for component_name, component_func in components:
            # Create enhanced method
            enhanced_method = lambda problem: component_func(current_method(problem), problem)
            
            result = self.evaluate_method(enhanced_method, f"+ {component_name}", test_subset)
            results[f"+ {component_name}"] = result
            
            current_method = enhanced_method
        
        return results 