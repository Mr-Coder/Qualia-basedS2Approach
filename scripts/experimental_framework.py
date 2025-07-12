#!/usr/bin/env python3
"""
Unified Experimental Framework for Mathematical Reasoning System
================================================================

Comprehensive experimental framework integrating all evaluation components:
- Automated ablation studies
- Failure case analysis  
- Computational complexity analysis
- Cross-linguistic validation
- Statistical significance testing
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from batch_complexity_classifier import BatchComplexityClassifier
from Data.dataset_loader import DatasetLoader
# Import experimental modules
from src.evaluation.ablation_study import AutomatedAblationStudy
from src.evaluation.computational_analysis import ComputationalAnalyzer
from src.evaluation.evaluator import ComprehensiveEvaluator
from src.evaluation.failure_analysis import FailureAnalyzer
from src.reasoning_core.strategies.enhanced_cotdir_strategy import \
    EnhancedCOTDIRStrategy

logger = logging.getLogger(__name__)


class UnifiedExperimentalFramework:
    """
    Unified experimental framework for comprehensive evaluation
    """
    
    def __init__(self, config_path: str = "config_files/experimental_config.json"):
        """
        Initialize experimental framework
        
        Args:
            config_path: Path to experimental configuration file
        """
        self.config = self._load_config(config_path)
        self.results_dir = Path(self.config.get("results_directory", "experimental_results"))
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.dataset_loader = DatasetLoader()
        self.complexity_classifier = BatchComplexityClassifier()
        self.cotdir_strategy = EnhancedCOTDIRStrategy()
        self.evaluator = ComprehensiveEvaluator()
        self.ablation_study = None
        self.failure_analyzer = FailureAnalyzer()
        self.computational_analyzer = ComputationalAnalyzer()
        
        # Experiment tracking
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_log = []
        
    def run_comprehensive_experiment(self, 
                                   datasets: List[str] = None,
                                   sample_sizes: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Run comprehensive experimental evaluation
        
        Args:
            datasets: List of dataset names to evaluate on
            sample_sizes: Dictionary mapping dataset names to sample sizes
            
        Returns:
            Complete experimental results
        """
        logger.info(f"Starting comprehensive experiment: {self.experiment_id}")
        
        # Default datasets if not specified
        if datasets is None:
            datasets = ["GSM8K", "Math23K", "SVAMP", "MAWPS", "ASDiv"]
        
        # Default sample sizes
        if sample_sizes is None:
            sample_sizes = {dataset: 100 for dataset in datasets}
        
        experiment_results = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "datasets_evaluated": datasets,
            "sample_sizes": sample_sizes,
            "results": {}
        }
        
        try:
            # Phase 1: Dataset Complexity Classification
            logger.info("Phase 1: Dataset Complexity Classification")
            complexity_results = self._run_complexity_classification(datasets, sample_sizes)
            experiment_results["results"]["complexity_classification"] = complexity_results
            self._log_phase_completion("complexity_classification", True)
            
            # Phase 2: Baseline Performance Evaluation
            logger.info("Phase 2: Baseline Performance Evaluation")
            baseline_results = self._run_baseline_evaluation(datasets, sample_sizes)
            experiment_results["results"]["baseline_performance"] = baseline_results
            self._log_phase_completion("baseline_performance", True)
            
            # Phase 3: Automated Ablation Study
            logger.info("Phase 3: Automated Ablation Study")
            ablation_results = self._run_ablation_study(datasets, sample_sizes)
            experiment_results["results"]["ablation_study"] = ablation_results
            self._log_phase_completion("ablation_study", True)
            
            # Phase 4: Failure Case Analysis
            logger.info("Phase 4: Failure Case Analysis")
            failure_results = self._run_failure_analysis(baseline_results)
            experiment_results["results"]["failure_analysis"] = failure_results
            self._log_phase_completion("failure_analysis", True)
            
            # Phase 5: Computational Complexity Analysis
            logger.info("Phase 5: Computational Complexity Analysis")
            computational_results = self._run_computational_analysis(datasets, sample_sizes)
            experiment_results["results"]["computational_analysis"] = computational_results
            self._log_phase_completion("computational_analysis", True)
            
            # Phase 6: Cross-linguistic Validation
            logger.info("Phase 6: Cross-linguistic Validation")
            cross_linguistic_results = self._run_cross_linguistic_validation(datasets)
            experiment_results["results"]["cross_linguistic"] = cross_linguistic_results
            self._log_phase_completion("cross_linguistic", True)
            
            # Phase 7: Statistical Analysis
            logger.info("Phase 7: Statistical Analysis")
            statistical_results = self._run_statistical_analysis(experiment_results["results"])
            experiment_results["results"]["statistical_analysis"] = statistical_results
            self._log_phase_completion("statistical_analysis", True)
            
            # Phase 8: Generate Final Report
            logger.info("Phase 8: Generating Final Report")
            final_report = self._generate_final_report(experiment_results)
            experiment_results["final_report"] = final_report
            self._log_phase_completion("final_report", True)
            
            # Save complete results
            self._save_experiment_results(experiment_results)
            
            logger.info(f"Comprehensive experiment completed: {self.experiment_id}")
            return experiment_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            experiment_results["error"] = str(e)
            experiment_results["experiment_log"] = self.experiment_log
            self._save_experiment_results(experiment_results)
            raise
    
    def _run_complexity_classification(self, 
                                     datasets: List[str], 
                                     sample_sizes: Dict[str, int]) -> Dict[str, Any]:
        """Run complexity classification on all datasets"""
        
        classification_results = {
            "timestamp": datetime.now().isoformat(),
            "datasets": {},
            "summary": {}
        }
        
        total_problems = 0
        total_distribution = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        
        for dataset_name in datasets:
            logger.info(f"Classifying complexity for {dataset_name}")
            
            try:
                # Run classification
                result = self.complexity_classifier.classify_dataset(
                    dataset_name, 
                    sample_sizes.get(dataset_name, 100)
                )
                
                classification_results["datasets"][dataset_name] = result
                
                # Update totals
                if "distribution" in result:
                    for level, count in result["distribution"].items():
                        total_distribution[level] += count
                    total_problems += result.get("total_problems", 0)
                
            except Exception as e:
                logger.error(f"Classification failed for {dataset_name}: {e}")
                classification_results["datasets"][dataset_name] = {"error": str(e)}
        
        # Calculate overall statistics
        if total_problems > 0:
            classification_results["summary"] = {
                "total_problems_classified": total_problems,
                "overall_distribution": {
                    level: count / total_problems * 100 
                    for level, count in total_distribution.items()
                },
                "complexity_trends": self._analyze_complexity_trends(classification_results["datasets"])
            }
        
        return classification_results
    
    def _run_baseline_evaluation(self, 
                               datasets: List[str], 
                               sample_sizes: Dict[str, int]) -> Dict[str, Any]:
        """Run baseline performance evaluation"""
        
        baseline_results = {
            "timestamp": datetime.now().isoformat(),
            "datasets": {},
            "summary": {}
        }
        
        overall_metrics = []
        
        for dataset_name in datasets:
            logger.info(f"Evaluating baseline performance on {dataset_name}")
            
            try:
                # Load test problems
                test_problems = self._load_test_problems(dataset_name, sample_sizes.get(dataset_name, 100))
                
                # Run evaluation
                dataset_results = self._evaluate_on_dataset(test_problems, dataset_name)
                baseline_results["datasets"][dataset_name] = dataset_results
                
                # Collect metrics
                if "performance_metrics" in dataset_results:
                    overall_metrics.append(dataset_results["performance_metrics"])
                
            except Exception as e:
                logger.error(f"Baseline evaluation failed for {dataset_name}: {e}")
                baseline_results["datasets"][dataset_name] = {"error": str(e)}
        
        # Calculate summary statistics
        if overall_metrics:
            baseline_results["summary"] = self._calculate_summary_metrics(overall_metrics)
        
        return baseline_results
    
    def _run_ablation_study(self, 
                          datasets: List[str], 
                          sample_sizes: Dict[str, int]) -> Dict[str, Any]:
        """Run automated ablation study"""
        
        # Prepare test problems for ablation study
        all_test_problems = []
        for dataset_name in datasets[:2]:  # Limit to 2 datasets for efficiency
            try:
                problems = self._load_test_problems(dataset_name, min(50, sample_sizes.get(dataset_name, 50)))
                all_test_problems.extend(problems)
            except Exception as e:
                logger.warning(f"Could not load {dataset_name} for ablation study: {e}")
        
        if not all_test_problems:
            return {"error": "No test problems available for ablation study"}
        
        # Run ablation study
        self.ablation_study = AutomatedAblationStudy(all_test_problems)
        return self.ablation_study.run_complete_ablation_study()
    
    def _run_failure_analysis(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run failure case analysis"""
        
        # Extract failure cases from baseline results
        all_test_results = []
        for dataset_name, dataset_results in baseline_results.get("datasets", {}).items():
            if "detailed_results" in dataset_results:
                all_test_results.extend(dataset_results["detailed_results"])
        
        if not all_test_results:
            return {"error": "No test results available for failure analysis"}
        
        # Run failure analysis
        return self.failure_analyzer.analyze_failures(all_test_results)
    
    def _run_computational_analysis(self, 
                                  datasets: List[str], 
                                  sample_sizes: Dict[str, int]) -> Dict[str, Any]:
        """Run computational complexity analysis"""
        
        # Prepare test problems
        test_problems = []
        for dataset_name in datasets[:2]:  # Limit for efficiency
            try:
                problems = self._load_test_problems(dataset_name, min(20, sample_sizes.get(dataset_name, 20)))
                test_problems.extend(problems)
            except Exception as e:
                logger.warning(f"Could not load {dataset_name} for computational analysis: {e}")
        
        if not test_problems:
            return {"error": "No test problems available for computational analysis"}
        
        # Define solve function
        def solve_function(problem_text: str) -> str:
            try:
                result = self.cotdir_strategy.solve_problem(problem_text)
                return result.get("final_answer", "")
            except Exception as e:
                raise Exception(f"Solve failed: {e}")
        
        # Run computational analysis
        return self.computational_analyzer.analyze_system_performance(solve_function, test_problems)
    
    def _run_cross_linguistic_validation(self, datasets: List[str]) -> Dict[str, Any]:
        """Run cross-linguistic validation"""
        
        # Identify English and Chinese datasets
        english_datasets = ["GSM8K", "SVAMP", "MAWPS", "ASDiv"]
        chinese_datasets = ["Math23K"]
        
        validation_results = {
            "english_performance": {},
            "chinese_performance": {},
            "cross_linguistic_comparison": {}
        }
        
        # Evaluate on English datasets
        for dataset in [d for d in datasets if d in english_datasets]:
            try:
                problems = self._load_test_problems(dataset, 50)
                results = self._evaluate_on_dataset(problems, dataset)
                validation_results["english_performance"][dataset] = results["performance_metrics"]
            except Exception as e:
                logger.warning(f"Cross-linguistic evaluation failed for {dataset}: {e}")
        
        # Evaluate on Chinese datasets
        for dataset in [d for d in datasets if d in chinese_datasets]:
            try:
                problems = self._load_test_problems(dataset, 50)
                results = self._evaluate_on_dataset(problems, dataset)
                validation_results["chinese_performance"][dataset] = results["performance_metrics"]
            except Exception as e:
                logger.warning(f"Cross-linguistic evaluation failed for {dataset}: {e}")
        
        # Compare performance
        validation_results["cross_linguistic_comparison"] = self._compare_linguistic_performance(
            validation_results["english_performance"],
            validation_results["chinese_performance"]
        )
        
        return validation_results
    
    def _run_statistical_analysis(self, experimental_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run statistical significance analysis"""
        
        statistical_results = {
            "significance_tests": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "summary": {}
        }
        
        # Extract performance data
        if "ablation_study" in experimental_results:
            ablation_data = experimental_results["ablation_study"]
            if "statistical_significance" in ablation_data:
                statistical_results["significance_tests"]["ablation"] = ablation_data["statistical_significance"]
        
        # Calculate effect sizes for key comparisons
        if "baseline_performance" in experimental_results:
            baseline_data = experimental_results["baseline_performance"]
            statistical_results["effect_sizes"]["baseline"] = self._calculate_effect_sizes(baseline_data)
        
        # Generate statistical summary
        statistical_results["summary"] = self._generate_statistical_summary(statistical_results)
        
        return statistical_results
    
    def _generate_final_report(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        report = {
            "experiment_overview": {
                "experiment_id": experiment_results["experiment_id"],
                "timestamp": experiment_results["timestamp"],
                "datasets_evaluated": experiment_results["datasets_evaluated"],
                "total_phases": len([p for p in self.experiment_log if p["success"]])
            },
            "key_findings": self._extract_key_findings(experiment_results),
            "performance_summary": self._generate_performance_summary(experiment_results),
            "recommendations": self._generate_recommendations(experiment_results),
            "limitations": self._identify_limitations(experiment_results),
            "future_work": self._suggest_future_work(experiment_results)
        }
        
        # Export report in multiple formats
        self._export_report(report)
        
        return report
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experimental configuration"""
        
        default_config = {
            "results_directory": "experimental_results",
            "enable_parallel_processing": True,
            "statistical_significance_threshold": 0.05,
            "effect_size_threshold": 0.3,
            "max_timeout_seconds": 30,
            "memory_limit_mb": 1000
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _load_test_problems(self, dataset_name: str, sample_size: int) -> List[Dict]:
        """Load test problems from dataset"""
        
        try:
            dataset = self.dataset_loader.load_dataset(dataset_name)
            if not dataset:
                raise ValueError(f"Could not load dataset: {dataset_name}")
            
            # Sample problems if needed
            if len(dataset) > sample_size:
                import random
                random.seed(42)  # For reproducibility
                dataset = random.sample(dataset, sample_size)
            
            # Convert to test format
            test_problems = []
            for i, item in enumerate(dataset):
                problem = {
                    "id": f"{dataset_name}_{i}",
                    "problem": item.get("problem", item.get("question", "")),
                    "expected_answer": item.get("answer", item.get("solution", "")),
                    "complexity": item.get("complexity", "L1"),
                    "dataset": dataset_name
                }
                test_problems.append(problem)
            
            return test_problems
            
        except Exception as e:
            logger.error(f"Failed to load test problems from {dataset_name}: {e}")
            return []
    
    def _evaluate_on_dataset(self, test_problems: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Evaluate COT-DIR strategy on dataset"""
        
        results = {
            "dataset_name": dataset_name,
            "total_problems": len(test_problems),
            "detailed_results": [],
            "performance_metrics": {}
        }
        
        correct_count = 0
        total_time = 0
        
        for problem in test_problems:
            start_time = time.time()
            
            try:
                # Solve problem
                solution = self.cotdir_strategy.solve_problem(problem["problem"])
                solve_time = time.time() - start_time
                
                # Evaluate correctness
                is_correct = self._evaluate_answer(
                    solution.get("final_answer"), 
                    problem["expected_answer"]
                )
                
                if is_correct:
                    correct_count += 1
                
                total_time += solve_time
                
                # Store detailed result
                detailed_result = {
                    "problem_id": problem["id"],
                    "problem": problem["problem"],
                    "expected_answer": problem["expected_answer"],
                    "predicted_answer": solution.get("final_answer"),
                    "is_correct": is_correct,
                    "solve_time": solve_time,
                    "complexity": problem["complexity"],
                    "confidence": solution.get("confidence", 0),
                    "reasoning_steps": solution.get("reasoning_steps", [])
                }
                results["detailed_results"].append(detailed_result)
                
            except Exception as e:
                logger.error(f"Failed to solve problem {problem['id']}: {e}")
                results["detailed_results"].append({
                    "problem_id": problem["id"],
                    "error": str(e),
                    "is_correct": False
                })
        
        # Calculate performance metrics
        results["performance_metrics"] = {
            "accuracy": correct_count / len(test_problems) if test_problems else 0,
            "average_time": total_time / len(test_problems) if test_problems else 0,
            "total_correct": correct_count,
            "total_problems": len(test_problems)
        }
        
        return results
    
    def _evaluate_answer(self, predicted: Any, expected: Any) -> bool:
        """Evaluate if predicted answer matches expected"""
        
        if predicted is None or expected is None:
            return False
        
        try:
            # Try numerical comparison
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 0.01
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(predicted).strip().lower() == str(expected).strip().lower()
    
    def _analyze_complexity_trends(self, classification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze complexity trends across datasets"""
        
        trends = {
            "dataset_difficulty_ranking": [],
            "language_complexity_differences": {},
            "size_complexity_correlation": {}
        }
        
        # Rank datasets by difficulty (higher L2+L3 percentage = more difficult)
        dataset_scores = []
        for dataset_name, data in classification_data.items():
            if "distribution" in data and not data.get("error"):
                dist = data["distribution"]
                difficulty_score = dist.get("L2", 0) + dist.get("L3", 0)
                dataset_scores.append((dataset_name, difficulty_score))
        
        trends["dataset_difficulty_ranking"] = sorted(dataset_scores, key=lambda x: x[1], reverse=True)
        
        return trends
    
    def _calculate_summary_metrics(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Calculate summary metrics across datasets"""
        
        if not metrics_list:
            return {}
        
        accuracies = [m["accuracy"] for m in metrics_list if "accuracy" in m]
        times = [m["average_time"] for m in metrics_list if "average_time" in m]
        
        return {
            "overall_accuracy": np.mean(accuracies) if accuracies else 0,
            "accuracy_std": np.std(accuracies) if accuracies else 0,
            "average_time": np.mean(times) if times else 0,
            "time_std": np.std(times) if times else 0,
            "datasets_evaluated": len(metrics_list)
        }
    
    def _compare_linguistic_performance(self, english_results: Dict, chinese_results: Dict) -> Dict[str, Any]:
        """Compare performance across languages"""
        
        comparison = {
            "english_avg_accuracy": 0,
            "chinese_avg_accuracy": 0,
            "language_advantage": "neutral",
            "performance_gap": 0
        }
        
        # Calculate English average
        if english_results:
            english_accuracies = [r["accuracy"] for r in english_results.values() if "accuracy" in r]
            comparison["english_avg_accuracy"] = np.mean(english_accuracies) if english_accuracies else 0
        
        # Calculate Chinese average
        if chinese_results:
            chinese_accuracies = [r["accuracy"] for r in chinese_results.values() if "accuracy" in r]
            comparison["chinese_avg_accuracy"] = np.mean(chinese_accuracies) if chinese_accuracies else 0
        
        # Determine language advantage
        gap = comparison["english_avg_accuracy"] - comparison["chinese_avg_accuracy"]
        comparison["performance_gap"] = abs(gap)
        
        if abs(gap) > 0.05:  # 5% threshold
            comparison["language_advantage"] = "english" if gap > 0 else "chinese"
        
        return comparison
    
    def _calculate_effect_sizes(self, baseline_data: Dict) -> Dict[str, float]:
        """Calculate effect sizes for baseline comparisons"""
        
        effect_sizes = {}
        
        # Extract accuracy data by complexity level
        accuracies_by_level = {"L0": [], "L1": [], "L2": [], "L3": []}
        
        for dataset_results in baseline_data.get("datasets", {}).values():
            if "detailed_results" in dataset_results:
                for result in dataset_results["detailed_results"]:
                    level = result.get("complexity", "L1")
                    is_correct = result.get("is_correct", False)
                    if level in accuracies_by_level:
                        accuracies_by_level[level].append(is_correct)
        
        # Calculate effect sizes between complexity levels
        for level in ["L1", "L2", "L3"]:
            if accuracies_by_level["L0"] and accuracies_by_level[level]:
                l0_acc = np.mean(accuracies_by_level["L0"])
                level_acc = np.mean(accuracies_by_level[level])
                
                # Cohen's d
                pooled_std = np.sqrt((np.var(accuracies_by_level["L0"]) + np.var(accuracies_by_level[level])) / 2)
                if pooled_std > 0:
                    effect_sizes[f"L0_vs_{level}"] = (l0_acc - level_acc) / pooled_std
        
        return effect_sizes
    
    def _generate_statistical_summary(self, statistical_results: Dict) -> Dict[str, Any]:
        """Generate statistical analysis summary"""
        
        summary = {
            "significant_findings": [],
            "effect_size_summary": {},
            "statistical_power": "adequate"  # Simplified
        }
        
        # Check for significant findings
        if "significance_tests" in statistical_results:
            for test_name, test_results in statistical_results["significance_tests"].items():
                if isinstance(test_results, dict):
                    for comparison, result in test_results.items():
                        if isinstance(result, dict) and result.get("is_significant", False):
                            summary["significant_findings"].append({
                                "comparison": comparison,
                                "p_value": result.get("p_value"),
                                "effect_size": result.get("effect_size")
                            })
        
        return summary
    
    def _extract_key_findings(self, experiment_results: Dict) -> List[str]:
        """Extract key findings from experimental results"""
        
        findings = []
        
        # Complexity classification findings
        if "complexity_classification" in experiment_results["results"]:
            complexity_data = experiment_results["results"]["complexity_classification"]
            if "summary" in complexity_data:
                total_problems = complexity_data["summary"].get("total_problems_classified", 0)
                findings.append(f"Classified {total_problems} problems across complexity levels")
        
        # Performance findings
        if "baseline_performance" in experiment_results["results"]:
            baseline_data = experiment_results["results"]["baseline_performance"]
            if "summary" in baseline_data:
                overall_acc = baseline_data["summary"].get("overall_accuracy", 0)
                findings.append(f"Overall system accuracy: {overall_acc:.1%}")
        
        # Ablation study findings
        if "ablation_study" in experiment_results["results"]:
            ablation_data = experiment_results["results"]["ablation_study"]
            if "component_analysis" in ablation_data:
                most_important = ablation_data["component_analysis"].get("most_important_component", {})
                if most_important:
                    findings.append(f"Most critical component: {most_important.get('name', 'unknown')}")
        
        return findings
    
    def _generate_performance_summary(self, experiment_results: Dict) -> Dict[str, Any]:
        """Generate performance summary"""
        
        summary = {
            "overall_performance": "good",  # Simplified
            "strength_areas": [],
            "weakness_areas": [],
            "performance_trends": {}
        }
        
        # Analyze baseline performance
        if "baseline_performance" in experiment_results["results"]:
            baseline_data = experiment_results["results"]["baseline_performance"]
            if "summary" in baseline_data:
                overall_acc = baseline_data["summary"].get("overall_accuracy", 0)
                
                if overall_acc > 0.8:
                    summary["overall_performance"] = "excellent"
                    summary["strength_areas"].append("High overall accuracy")
                elif overall_acc > 0.6:
                    summary["overall_performance"] = "good"
                else:
                    summary["overall_performance"] = "needs_improvement"
                    summary["weakness_areas"].append("Low overall accuracy")
        
        return summary
    
    def _generate_recommendations(self, experiment_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Failure analysis recommendations
        if "failure_analysis" in experiment_results["results"]:
            failure_data = experiment_results["results"]["failure_analysis"]
            if "recommendations" in failure_data:
                recommendations.extend(failure_data["recommendations"])
        
        # Computational analysis recommendations
        if "computational_analysis" in experiment_results["results"]:
            comp_data = experiment_results["results"]["computational_analysis"]
            if "recommendations" in comp_data:
                recommendations.extend(comp_data["recommendations"])
        
        # General recommendations
        recommendations.append("Continue regular experimental evaluation")
        recommendations.append("Focus on systematic improvement based on identified weaknesses")
        
        return recommendations
    
    def _identify_limitations(self, experiment_results: Dict) -> List[str]:
        """Identify experimental limitations"""
        
        limitations = [
            "Limited to available datasets",
            "Sample sizes may not represent full distribution",
            "Computational resources constrained analysis depth"
        ]
        
        # Add specific limitations based on results
        if "error" in experiment_results:
            limitations.append("Some experimental phases failed")
        
        return limitations
    
    def _suggest_future_work(self, experiment_results: Dict) -> List[str]:
        """Suggest future research directions"""
        
        future_work = [
            "Expand to additional mathematical domains",
            "Investigate advanced reasoning strategies",
            "Develop more sophisticated evaluation metrics",
            "Cross-cultural mathematical reasoning analysis"
        ]
        
        return future_work
    
    def _log_phase_completion(self, phase_name: str, success: bool):
        """Log completion of experimental phase"""
        
        self.experiment_log.append({
            "phase": phase_name,
            "timestamp": datetime.now().isoformat(),
            "success": success
        })
    
    def _save_experiment_results(self, results: Dict[str, Any]):
        """Save complete experimental results"""
        
        # Save main results
        results_file = self.results_dir / f"{self.experiment_id}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save experiment log
        log_file = self.results_dir / f"{self.experiment_id}_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Experimental results saved to {results_file}")
    
    def _export_report(self, report: Dict[str, Any]):
        """Export final report in multiple formats"""
        
        # JSON format
        report_file = self.results_dir / f"{self.experiment_id}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Markdown format
        markdown_file = self.results_dir / f"{self.experiment_id}_report.md"
        self._export_markdown_report(report, markdown_file)
        
        logger.info(f"Final report exported to {report_file} and {markdown_file}")
    
    def _export_markdown_report(self, report: Dict[str, Any], filepath: Path):
        """Export report in Markdown format"""
        
        markdown_content = f"""# Experimental Report: {report['experiment_overview']['experiment_id']}

## Overview
- **Experiment ID**: {report['experiment_overview']['experiment_id']}
- **Timestamp**: {report['experiment_overview']['timestamp']}
- **Datasets Evaluated**: {', '.join(report['experiment_overview']['datasets_evaluated'])}
- **Phases Completed**: {report['experiment_overview']['total_phases']}

## Key Findings
"""
        
        for finding in report['key_findings']:
            markdown_content += f"- {finding}\n"
        
        markdown_content += f"""

## Performance Summary
- **Overall Performance**: {report['performance_summary']['overall_performance']}

### Strengths
"""
        
        for strength in report['performance_summary']['strength_areas']:
            markdown_content += f"- {strength}\n"
        
        markdown_content += "\n### Areas for Improvement\n"
        
        for weakness in report['performance_summary']['weakness_areas']:
            markdown_content += f"- {weakness}\n"
        
        markdown_content += "\n## Recommendations\n"
        
        for rec in report['recommendations']:
            markdown_content += f"- {rec}\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)


def run_unified_experiment():
    """Run unified experimental framework demo"""
    
    # Initialize framework
    framework = UnifiedExperimentalFramework()
    
    # Run comprehensive experiment
    results = framework.run_comprehensive_experiment(
        datasets=["GSM8K", "SVAMP"],  # Limited for demo
        sample_sizes={"GSM8K": 50, "SVAMP": 50}
    )
    
    print(f"Experiment completed: {results['experiment_id']}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unified_experiment() 