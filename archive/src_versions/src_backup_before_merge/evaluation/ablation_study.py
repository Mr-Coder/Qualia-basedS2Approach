"""
Automated Ablation Study Framework
==================================

Comprehensive ablation study implementation for COT-DIR system.
"""

import json
import logging
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..reasoning_core.strategies.enhanced_cotdir_strategy import \
    EnhancedCOTDIRStrategy
from .evaluator import ComprehensiveEvaluator

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation study components"""
    enable_complexity_analysis: bool = True
    enable_relation_discovery: bool = True 
    enable_multilayer_reasoning: bool = True
    enable_five_dim_validation: bool = True
    enable_adaptive_depth: bool = True


@dataclass
class AblationResult:
    """Results from a single ablation configuration"""
    config_name: str
    config: AblationConfig
    accuracy_by_level: Dict[str, float]
    overall_accuracy: float
    relation_f1: float
    reasoning_quality: float
    processing_time: float
    confidence_score: float
    error_count: int
    test_cases_count: int


class AutomatedAblationStudy:
    """Automated ablation study framework"""
    
    def __init__(self, test_problems: List[Dict], config: Optional[Dict] = None):
        self.test_problems = test_problems
        self.config = config or {}
        self.evaluator = ComprehensiveEvaluator()
        self.results = {}
        
    def run_complete_ablation_study(self) -> Dict[str, Any]:
        """Run complete ablation study with all component combinations"""
        logger.info("Starting automated ablation study...")
        
        # Define ablation configurations
        ablation_configs = self._generate_ablation_configs()
        
        # Run each configuration
        for config_name, config in ablation_configs.items():
            logger.info(f"Testing configuration: {config_name}")
            result = self._test_configuration(config_name, config)
            self.results[config_name] = result
        
        # Analyze results
        analysis = self._analyze_ablation_results()
        
        # Statistical significance testing
        significance_results = self._test_statistical_significance()
        
        return {
            "individual_results": {name: self._result_to_dict(result) 
                                 for name, result in self.results.items()},
            "component_analysis": analysis,
            "statistical_significance": significance_results,
            "summary": self._generate_summary()
        }
    
    def _generate_ablation_configs(self) -> Dict[str, AblationConfig]:
        """Generate all ablation configurations"""
        return {
            "Full_System": AblationConfig(
                enable_complexity_analysis=True,
                enable_relation_discovery=True,
                enable_multilayer_reasoning=True,
                enable_five_dim_validation=True,
                enable_adaptive_depth=True
            ),
            "w/o_Complexity_Analysis": AblationConfig(
                enable_complexity_analysis=False,
                enable_relation_discovery=True,
                enable_multilayer_reasoning=True,
                enable_five_dim_validation=True,
                enable_adaptive_depth=True
            ),
            "w/o_Relation_Discovery": AblationConfig(
                enable_complexity_analysis=True,
                enable_relation_discovery=False,
                enable_multilayer_reasoning=True,
                enable_five_dim_validation=True,
                enable_adaptive_depth=True
            ),
            "w/o_Multilayer_Reasoning": AblationConfig(
                enable_complexity_analysis=True,
                enable_relation_discovery=True,
                enable_multilayer_reasoning=False,
                enable_five_dim_validation=True,
                enable_adaptive_depth=True
            ),
            "w/o_Five_Dim_Validation": AblationConfig(
                enable_complexity_analysis=True,
                enable_relation_discovery=True,
                enable_multilayer_reasoning=True,
                enable_five_dim_validation=False,
                enable_adaptive_depth=True
            ),
            "w/o_Adaptive_Depth": AblationConfig(
                enable_complexity_analysis=True,
                enable_relation_discovery=True,
                enable_multilayer_reasoning=True,
                enable_five_dim_validation=True,
                enable_adaptive_depth=False
            ),
            "Minimal_System": AblationConfig(
                enable_complexity_analysis=False,
                enable_relation_discovery=False,
                enable_multilayer_reasoning=False,
                enable_five_dim_validation=False,
                enable_adaptive_depth=False
            )
        }
    
    def _test_configuration(self, config_name: str, config: AblationConfig) -> AblationResult:
        """Test a specific ablation configuration"""
        
        # Initialize strategy with ablation config
        strategy = EnhancedCOTDIRStrategy(self._config_to_strategy_params(config))
        
        # Test on all problems
        correct_by_level = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        total_by_level = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        
        total_correct = 0
        total_problems = len(self.test_problems)
        total_time = 0
        confidence_scores = []
        error_count = 0
        relation_f1_scores = []
        reasoning_quality_scores = []
        
        for problem in self.test_problems:
            try:
                start_time = time.time()
                
                # Solve problem with current configuration
                result = strategy.solve_problem(
                    problem["problem"], 
                    problem.get("complexity", "L2")
                )
                
                solve_time = time.time() - start_time
                total_time += solve_time
                
                # Evaluate correctness
                is_correct = self._evaluate_answer(
                    result.get("final_answer"), 
                    problem["expected_answer"]
                )
                
                if is_correct:
                    total_correct += 1
                
                # Track by complexity level
                level = problem.get("complexity", "L2")
                total_by_level[level] += 1
                if is_correct:
                    correct_by_level[level] += 1
                
                # Collect metrics
                confidence_scores.append(result.get("confidence", 0))
                relation_f1_scores.append(result.get("relation_f1", 0))
                reasoning_quality_scores.append(result.get("reasoning_quality", 0))
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error in {config_name}: {e}")
        
        # Calculate final metrics
        accuracy_by_level = {
            level: (correct_by_level[level] / total_by_level[level] 
                   if total_by_level[level] > 0 else 0)
            for level in ["L0", "L1", "L2", "L3"]
        }
        
        overall_accuracy = total_correct / total_problems if total_problems > 0 else 0
        avg_relation_f1 = np.mean(relation_f1_scores) if relation_f1_scores else 0
        avg_reasoning_quality = np.mean(reasoning_quality_scores) if reasoning_quality_scores else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        avg_time = total_time / total_problems if total_problems > 0 else 0
        
        return AblationResult(
            config_name=config_name,
            config=config,
            accuracy_by_level=accuracy_by_level,
            overall_accuracy=overall_accuracy,
            relation_f1=avg_relation_f1,
            reasoning_quality=avg_reasoning_quality,
            processing_time=avg_time,
            confidence_score=avg_confidence,
            error_count=error_count,
            test_cases_count=total_problems
        )
    
    def _analyze_ablation_results(self) -> Dict[str, Any]:
        """Analyze component contributions"""
        
        if "Full_System" not in self.results:
            logger.error("Full system results not found")
            return {}
        
        full_result = self.results["Full_System"]
        component_contributions = {}
        
        # Calculate individual component contributions
        component_configs = [
            ("Complexity_Analysis", "w/o_Complexity_Analysis"),
            ("Relation_Discovery", "w/o_Relation_Discovery"),
            ("Multilayer_Reasoning", "w/o_Multilayer_Reasoning"),
            ("Five_Dim_Validation", "w/o_Five_Dim_Validation"),
            ("Adaptive_Depth", "w/o_Adaptive_Depth")
        ]
        
        for component_name, without_config in component_configs:
            if without_config in self.results:
                without_result = self.results[without_config]
                contribution = {
                    "accuracy_contribution": full_result.overall_accuracy - without_result.overall_accuracy,
                    "f1_contribution": full_result.relation_f1 - without_result.relation_f1,
                    "quality_contribution": full_result.reasoning_quality - without_result.reasoning_quality,
                    "time_impact": without_result.processing_time - full_result.processing_time,
                    "error_impact": without_result.error_count - full_result.error_count
                }
                component_contributions[component_name] = contribution
        
        # Find most important components
        most_important = max(component_contributions.items(), 
                           key=lambda x: x[1]["accuracy_contribution"])
        
        return {
            "component_contributions": component_contributions,
            "most_important_component": {
                "name": most_important[0],
                "accuracy_gain": most_important[1]["accuracy_contribution"]
            },
            "component_ranking": sorted(
                component_contributions.items(),
                key=lambda x: x[1]["accuracy_contribution"],
                reverse=True
            )
        }
    
    def _test_statistical_significance(self) -> Dict[str, Any]:
        """Test statistical significance of improvements"""
        
        if len(self.results) < 2:
            return {"error": "Need at least 2 configurations for significance testing"}
        
        # Compare full system vs best baseline
        full_system = self.results.get("Full_System")
        other_configs = {k: v for k, v in self.results.items() if k != "Full_System"}
        
        if not full_system or not other_configs:
            return {"error": "Missing full system or baseline results"}
        
        significance_results = {}
        
        for config_name, config_result in other_configs.items():
            # Perform paired t-test (simulated with bootstrap)
            full_accuracies = self._simulate_accuracy_distribution(full_system)
            config_accuracies = self._simulate_accuracy_distribution(config_result)
            
            t_stat, p_value = stats.ttest_ind(full_accuracies, config_accuracies)
            effect_size = self._calculate_cohens_d(full_accuracies, config_accuracies)
            
            significance_results[config_name] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "is_significant": p_value < 0.05,
                "improvement": full_system.overall_accuracy - config_result.overall_accuracy
            }
        
        return significance_results
    
    def _config_to_strategy_params(self, config: AblationConfig) -> Dict:
        """Convert ablation config to strategy parameters"""
        return {
            "enable_complexity_analysis": config.enable_complexity_analysis,
            "enable_relation_discovery": config.enable_relation_discovery,
            "enable_multilayer_reasoning": config.enable_multilayer_reasoning,
            "enable_five_dim_validation": config.enable_five_dim_validation,
            "enable_adaptive_depth": config.enable_adaptive_depth
        }
    
    def _evaluate_answer(self, predicted: Any, expected: Any) -> bool:
        """Evaluate if predicted answer matches expected"""
        try:
            # Try numerical comparison
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 0.01
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(predicted).strip().lower() == str(expected).strip().lower()
    
    def _simulate_accuracy_distribution(self, result: AblationResult) -> np.ndarray:
        """Simulate accuracy distribution for statistical testing"""
        # Bootstrap simulation based on binomial distribution
        n_trials = result.test_cases_count
        p_success = result.overall_accuracy
        
        # Generate bootstrap samples
        np.random.seed(42)  # For reproducibility
        return np.random.binomial(n_trials, p_success, 1000) / n_trials
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(group1) - 1) * var1 + (len(group2) - 1) * var2) / 
                           (len(group1) + len(group2) - 2))
        
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    def _result_to_dict(self, result: AblationResult) -> Dict:
        """Convert result to dictionary"""
        return {
            "config_name": result.config_name,
            "overall_accuracy": result.overall_accuracy,
            "accuracy_by_level": result.accuracy_by_level,
            "relation_f1": result.relation_f1,
            "reasoning_quality": result.reasoning_quality,
            "processing_time": result.processing_time,
            "confidence_score": result.confidence_score,
            "error_count": result.error_count,
            "test_cases_count": result.test_cases_count
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of ablation study"""
        
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Find best and worst configurations
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].overall_accuracy, reverse=True)
        
        best_config = sorted_results[0]
        worst_config = sorted_results[-1]
        
        # Calculate performance range
        accuracy_range = best_config[1].overall_accuracy - worst_config[1].overall_accuracy
        
        return {
            "total_configurations": len(self.results),
            "best_configuration": {
                "name": best_config[0],
                "accuracy": best_config[1].overall_accuracy
            },
            "worst_configuration": {
                "name": worst_config[0], 
                "accuracy": worst_config[1].overall_accuracy
            },
            "performance_range": accuracy_range,
            "average_accuracy": np.mean([r.overall_accuracy for r in self.results.values()]),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on ablation results"""
        recommendations = []
        
        if "Full_System" in self.results:
            full_acc = self.results["Full_System"].overall_accuracy
            
            if full_acc > 0.8:
                recommendations.append("System shows excellent performance with all components")
            elif full_acc > 0.6:
                recommendations.append("System shows good performance but may benefit from optimization")
            else:
                recommendations.append("System needs significant improvement")
        
        # Add component-specific recommendations
        analysis = self._analyze_ablation_results()
        if "most_important_component" in analysis:
            most_important = analysis["most_important_component"]
            recommendations.append(
                f"Focus on {most_important['name']} - provides {most_important['accuracy_gain']:.1%} improvement"
            )
        
        return recommendations


def run_ablation_study_demo():
    """Demo function to run ablation study"""
    
    # Sample test problems
    test_problems = [
        {
            "problem": "小明有15个苹果，小红有8个苹果。他们一共有多少个苹果？",
            "expected_answer": "23",
            "complexity": "L0"
        },
        {
            "problem": "一辆汽车每小时行驶60公里，行驶了2.5小时。这辆汽车总共行驶了多少公里？",
            "expected_answer": "150",
            "complexity": "L1"
        }
    ]
    
    # Run ablation study
    study = AutomatedAblationStudy(test_problems)
    results = study.run_complete_ablation_study()
    
    # Save results
    with open("ablation_study_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("Ablation study completed. Results saved to ablation_study_results.json")
    return results


if __name__ == "__main__":
    run_ablation_study_demo() 