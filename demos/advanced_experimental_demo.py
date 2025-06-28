#!/usr/bin/env python3
"""
Advanced Experimental Demonstration

This script demonstrates the complete mathematical reasoning system with
comprehensive testing, evaluation, and comparison capabilities.

Features:
1. Multi-problem testing suite
2. Performance benchmarking
3. Detailed analysis and visualization
4. Comparison with baseline models
5. Statistical evaluation

Author: AI Research Team
Date: 2025-01-31
"""

import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Import our mathematical reasoning system
from mathematical_reasoning_system import (MathematicalReasoningSystem,
                                           ProblemComplexity)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedExperimentalDemo:
    """Advanced demonstration and evaluation system."""
    
    def __init__(self):
        self.system = MathematicalReasoningSystem()
        self.test_problems = self._load_test_problems()
        self.results = []
        self.benchmarks = {}
        
    def _load_test_problems(self) -> List[Dict[str, Any]]:
        """Load comprehensive test problem suite."""
        return [
            {
                "id": "L0_001",
                "complexity": "L0",
                "problem": "What is 15 + 27?",
                "expected_answer": 42,
                "category": "arithmetic",
                "description": "Simple addition"
            },
            {
                "id": "L0_002", 
                "complexity": "L0",
                "problem": "Calculate 8 × 6.",
                "expected_answer": 48,
                "category": "arithmetic",
                "description": "Simple multiplication"
            },
            {
                "id": "L1_001",
                "complexity": "L1",
                "problem": "A box contains 24 apples. If 8 apples are eaten, how many apples remain?",
                "expected_answer": 16,
                "category": "word_problem",
                "description": "Simple subtraction word problem"
            },
            {
                "id": "L1_002",
                "complexity": "L1", 
                "problem": "Sarah has 5 bags with 7 candies each. How many candies does she have in total?",
                "expected_answer": 35,
                "category": "word_problem",
                "description": "Simple multiplication word problem"
            },
            {
                "id": "L2_001",
                "complexity": "L2",
                "problem": "A car travels 60 km in 45 minutes. What is the car's speed in km/h?",
                "expected_answer": 80,
                "category": "rate_calculation",
                "description": "Speed calculation with unit conversion"
            },
            {
                "id": "L2_002",
                "complexity": "L2",
                "problem": "A rectangular garden is 12 m long and 8 m wide. If fencing costs $15 per meter, how much will it cost to fence the entire perimeter?",
                "expected_answer": 600,
                "category": "geometry",
                "description": "Perimeter calculation with cost"
            },
            {
                "id": "L2_003",
                "complexity": "L2",
                "problem": "A tank contains 200L of water. Water flows out at 5L/min while water flows in at 3L/min. How long will it take to empty the tank?",
                "expected_answer": 100,
                "category": "rate_problem",
                "description": "Rate problem with inflow and outflow"
            },
            {
                "id": "L3_001",
                "complexity": "L3",
                "problem": "A tank contains 5L of water. Ice cubes of 200 cm³ are dropped one cube per minute. Water leaks at 2 mL/s. How long will it take for the water level to rise to 9L?",
                "expected_answer": 2000,  # Expected in minutes
                "category": "complex_rate",
                "description": "Multi-rate problem with unit conversion"
            },
            {
                "id": "L3_002",
                "complexity": "L3",
                "problem": "In a factory, Machine A produces 120 units/hour, Machine B produces 80 units/hour. They work together for 3 hours, then Machine A breaks down. Machine B continues alone and produces 200 more units. How many total units were produced?",
                "expected_answer": 800,
                "category": "complex_word",
                "description": "Multi-stage production problem"
            },
            {
                "id": "L3_003",
                "complexity": "L3",
                "problem": "A swimming pool can be filled by pipe A in 4 hours, by pipe B in 6 hours, and can be emptied by drain C in 12 hours. If all three are opened simultaneously, how long will it take to fill the pool?",
                "expected_answer": 4,  # Expected in hours
                "category": "work_rate",
                "description": "Complex work rate problem"
            }
        ]
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on all test problems."""
        logger.info("Starting comprehensive evaluation...")
        
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "total_problems": len(self.test_problems),
            "results_by_complexity": {},
            "results_by_category": {},
            "performance_metrics": {},
            "detailed_results": [],
            "system_statistics": {},
            "error_analysis": {}
        }
        
        # Initialize complexity and category tracking
        complexity_stats = {level: {"correct": 0, "total": 0, "times": [], "confidence": []} 
                          for level in ["L0", "L1", "L2", "L3"]}
        category_stats = {}
        
        # Process each test problem
        for i, test_case in enumerate(self.test_problems):
            logger.info(f"Processing problem {i+1}/{len(self.test_problems)}: {test_case['id']}")
            
            # Solve the problem
            start_time = time.time()
            result = self.system.solve_mathematical_problem(test_case["problem"])
            solve_time = time.time() - start_time
            
            # Evaluate result
            is_correct = self._evaluate_answer(result.get("final_answer"), test_case["expected_answer"])
            
            # Collect detailed information
            detailed_result = {
                "test_id": test_case["id"],
                "complexity": test_case["complexity"],
                "category": test_case["category"],
                "problem": test_case["problem"],
                "expected_answer": test_case["expected_answer"],
                "system_answer": result.get("final_answer"),
                "is_correct": is_correct,
                "solve_time": solve_time,
                "processing_time": result.get("processing_time", 0),
                "num_entities": result.get("system_metadata", {}).get("num_entities", 0),
                "num_relations": result.get("system_metadata", {}).get("num_relations", 0),
                "num_reasoning_steps": result.get("system_metadata", {}).get("num_reasoning_steps", 0),
                "verification_passed": result.get("verification_result", {}).get("is_valid", False) if result.get("verification_result") else None,
                "confidence_score": result.get("verification_result", {}).get("confidence_score", 0) if result.get("verification_result") else 0,
                "error_message": result.get("error")
            }
            
            evaluation_results["detailed_results"].append(detailed_result)
            
            # Update complexity statistics
            complexity = test_case["complexity"]
            complexity_stats[complexity]["total"] += 1
            complexity_stats[complexity]["times"].append(solve_time)
            complexity_stats[complexity]["confidence"].append(detailed_result["confidence_score"])
            if is_correct:
                complexity_stats[complexity]["correct"] += 1
            
            # Update category statistics
            category = test_case["category"]
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0, "times": []}
            
            category_stats[category]["total"] += 1
            category_stats[category]["times"].append(solve_time)
            if is_correct:
                category_stats[category]["correct"] += 1
        
        # Calculate final statistics
        evaluation_results["results_by_complexity"] = self._calculate_complexity_stats(complexity_stats)
        evaluation_results["results_by_category"] = self._calculate_category_stats(category_stats)
        evaluation_results["performance_metrics"] = self._calculate_performance_metrics(evaluation_results["detailed_results"])
        evaluation_results["system_statistics"] = self.system.get_system_statistics()
        evaluation_results["error_analysis"] = self._analyze_errors(evaluation_results["detailed_results"])
        
        # Save results
        self._save_evaluation_results(evaluation_results)
        
        logger.info("Comprehensive evaluation completed!")
        return evaluation_results
    
    def _evaluate_answer(self, system_answer: Any, expected_answer: Any) -> bool:
        """Evaluate if the system answer matches the expected answer."""
        if system_answer is None:
            return False
        
        try:
            # Convert to float for numerical comparison
            sys_val = float(system_answer)
            exp_val = float(expected_answer)
            
            # Allow small numerical errors (1% tolerance)
            return abs(sys_val - exp_val) <= abs(exp_val) * 0.01
        except (ValueError, TypeError):
            # Fallback to string comparison
            return str(system_answer).strip().lower() == str(expected_answer).strip().lower()
    
    def _calculate_complexity_stats(self, complexity_stats: Dict) -> Dict[str, Any]:
        """Calculate statistics by complexity level."""
        results = {}
        
        for level, stats in complexity_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                avg_time = statistics.mean(stats["times"]) if stats["times"] else 0
                avg_confidence = statistics.mean(stats["confidence"]) if stats["confidence"] else 0
                
                results[level] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "average_time": avg_time,
                    "average_confidence": avg_confidence,
                    "performance_grade": self._get_performance_grade(accuracy)
                }
        
        return results
    
    def _calculate_category_stats(self, category_stats: Dict) -> Dict[str, Any]:
        """Calculate statistics by problem category."""
        results = {}
        
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                avg_time = statistics.mean(stats["times"]) if stats["times"] else 0
                
                results[category] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "average_time": avg_time,
                    "performance_grade": self._get_performance_grade(accuracy)
                }
        
        return results
    
    def _calculate_performance_metrics(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        if not detailed_results:
            return {}
        
        correct_count = sum(1 for r in detailed_results if r["is_correct"])
        total_count = len(detailed_results)
        
        solve_times = [r["solve_time"] for r in detailed_results if r["solve_time"] > 0]
        confidence_scores = [r["confidence_score"] for r in detailed_results if r["confidence_score"] > 0]
        
        return {
            "overall_accuracy": correct_count / total_count,
            "total_problems": total_count,
            "correct_answers": correct_count,
            "average_solve_time": statistics.mean(solve_times) if solve_times else 0,
            "median_solve_time": statistics.median(solve_times) if solve_times else 0,
            "max_solve_time": max(solve_times) if solve_times else 0,
            "min_solve_time": min(solve_times) if solve_times else 0,
            "average_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
            "problems_with_errors": len([r for r in detailed_results if r.get("error_message")]),
            "verification_pass_rate": len([r for r in detailed_results if r["verification_passed"]]) / total_count
        }
    
    def _analyze_errors(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Analyze error patterns and types."""
        incorrect_results = [r for r in detailed_results if not r["is_correct"]]
        error_results = [r for r in detailed_results if r.get("error_message")]
        
        error_analysis = {
            "total_incorrect": len(incorrect_results),
            "total_errors": len(error_results),
            "incorrect_by_complexity": {},
            "incorrect_by_category": {},
            "common_error_patterns": [],
            "error_messages": [r["error_message"] for r in error_results if r["error_message"]]
        }
        
        # Analyze by complexity
        for result in incorrect_results:
            complexity = result["complexity"]
            if complexity not in error_analysis["incorrect_by_complexity"]:
                error_analysis["incorrect_by_complexity"][complexity] = 0
            error_analysis["incorrect_by_complexity"][complexity] += 1
        
        # Analyze by category
        for result in incorrect_results:
            category = result["category"]
            if category not in error_analysis["incorrect_by_category"]:
                error_analysis["incorrect_by_category"][category] = 0
            error_analysis["incorrect_by_category"][category] += 1
        
        return error_analysis
    
    def _get_performance_grade(self, accuracy: float) -> str:
        """Get performance grade based on accuracy."""
        if accuracy >= 0.9:
            return "A (Excellent)"
        elif accuracy >= 0.8:
            return "B (Good)"
        elif accuracy >= 0.7:
            return "C (Average)"
        elif accuracy >= 0.6:
            return "D (Below Average)"
        else:
            return "F (Poor)"
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = f"advanced_evaluation_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed CSV
        csv_path = f"detailed_results_{timestamp}.csv"
        self._save_csv_results(results["detailed_results"], csv_path)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def _save_csv_results(self, detailed_results: List[Dict], filepath: str) -> None:
        """Save detailed results to CSV file."""
        import csv
        
        if not detailed_results:
            return
        
        fieldnames = detailed_results[0].keys()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
    
    def generate_visualization_report(self, results: Dict[str, Any]) -> None:
        """Generate visualization report of evaluation results."""
        logger.info("Generating visualization report...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Mathematical Reasoning System - Comprehensive Evaluation Report', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by Complexity Level
        complexity_data = results["results_by_complexity"]
        complexities = list(complexity_data.keys())
        accuracies = [complexity_data[c]["accuracy"] for c in complexities]
        
        axes[0, 0].bar(complexities, accuracies, color=['#2E8B57', '#4682B4', '#DAA520', '#CD5C5C'])
        axes[0, 0].set_title('Accuracy by Complexity Level')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
        
        # 2. Processing Time by Complexity
        avg_times = [complexity_data[c]["average_time"] for c in complexities]
        axes[0, 1].bar(complexities, avg_times, color=['#2E8B57', '#4682B4', '#DAA520', '#CD5C5C'])
        axes[0, 1].set_title('Average Processing Time by Complexity')
        axes[0, 1].set_ylabel('Time (seconds)')
        for i, v in enumerate(avg_times):
            axes[0, 1].text(i, v + max(avg_times)*0.02, f'{v:.3f}s', ha='center', fontweight='bold')
        
        # 3. Accuracy by Category
        category_data = results["results_by_category"]
        categories = list(category_data.keys())
        cat_accuracies = [category_data[c]["accuracy"] for c in categories]
        
        axes[0, 2].barh(categories, cat_accuracies, color='skyblue')
        axes[0, 2].set_title('Accuracy by Problem Category')
        axes[0, 2].set_xlabel('Accuracy')
        axes[0, 2].set_xlim(0, 1)
        for i, v in enumerate(cat_accuracies):
            axes[0, 2].text(v + 0.02, i, f'{v:.2f}', va='center', fontweight='bold')
        
        # 4. Overall Performance Metrics
        metrics = results["performance_metrics"]
        metric_names = ['Overall\nAccuracy', 'Avg\nConfidence', 'Verification\nPass Rate']
        metric_values = [
            metrics["overall_accuracy"],
            metrics["average_confidence"],
            metrics["verification_pass_rate"]
        ]
        
        bars = axes[1, 0].bar(metric_names, metric_values, color=['#FF6347', '#32CD32', '#1E90FF'])
        axes[1, 0].set_title('Key Performance Indicators')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Error Analysis
        error_data = results["error_analysis"]
        if error_data["incorrect_by_complexity"]:
            error_complexities = list(error_data["incorrect_by_complexity"].keys())
            error_counts = list(error_data["incorrect_by_complexity"].values())
            
            axes[1, 1].pie(error_counts, labels=error_complexities, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Error Distribution by Complexity')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Errors!', ha='center', va='center', fontsize=16, fontweight='bold')
            axes[1, 1].set_title('Error Distribution by Complexity')
        
        # 6. Performance Trend
        detailed = results["detailed_results"]
        problem_indices = range(1, len(detailed) + 1)
        solve_times = [r["solve_time"] for r in detailed]
        
        axes[1, 2].plot(problem_indices, solve_times, marker='o', linestyle='-', color='purple')
        axes[1, 2].set_title('Processing Time Trend')
        axes[1, 2].set_xlabel('Problem Index')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'evaluation_report_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualization report saved as evaluation_report_{timestamp}.png")
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        logger.info("Running performance benchmarks...")
        
        # Test different system configurations
        configs = [
            {"name": "Standard", "config": {}},
            {"name": "No Verification", "config": {"enable_verification": False}},
            {"name": "Detailed Output", "config": {"output_format": "detailed"}}
        ]
        
        benchmark_results = {}
        
        for config_info in configs:
            logger.info(f"Testing configuration: {config_info['name']}")
            
            # Initialize system with configuration
            system = MathematicalReasoningSystem(config_info["config"])
            
            # Test on a subset of problems
            test_subset = self.test_problems[:5]  # Use first 5 problems for benchmarking
            times = []
            
            for problem in test_subset:
                start_time = time.time()
                result = system.solve_mathematical_problem(problem["problem"])
                end_time = time.time()
                times.append(end_time - start_time)
            
            benchmark_results[config_info["name"]] = {
                "average_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "min_time": min(times),
                "max_time": max(times),
                "total_time": sum(times),
                "configuration": config_info["config"]
            }
        
        return benchmark_results
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive text report."""
        results = self.run_comprehensive_evaluation()
        benchmark_results = self.run_performance_benchmark()
        
        report = []
        report.append("="*80)
        report.append("MATHEMATICAL REASONING SYSTEM - COMPREHENSIVE EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)
        metrics = results["performance_metrics"]
        report.append(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}")
        report.append(f"Total Problems Tested: {metrics['total_problems']}")
        report.append(f"Average Processing Time: {metrics['average_solve_time']:.3f} seconds")
        report.append(f"Average Confidence Score: {metrics['average_confidence']:.3f}")
        report.append("")
        
        # Performance by Complexity
        report.append("PERFORMANCE BY COMPLEXITY LEVEL")
        report.append("-" * 50)
        for level, stats in results["results_by_complexity"].items():
            report.append(f"{level}: {stats['accuracy']:.1%} accuracy ({stats['correct']}/{stats['total']}) - {stats['performance_grade']}")
        report.append("")
        
        # Performance by Category
        report.append("PERFORMANCE BY PROBLEM CATEGORY")
        report.append("-" * 50)
        for category, stats in results["results_by_category"].items():
            report.append(f"{category}: {stats['accuracy']:.1%} accuracy ({stats['correct']}/{stats['total']})")
        report.append("")
        
        # Benchmark Results
        report.append("PERFORMANCE BENCHMARKS")
        report.append("-" * 50)
        for config_name, bench_stats in benchmark_results.items():
            report.append(f"{config_name}: {bench_stats['average_time']:.3f}s average")
        report.append("")
        
        # Error Analysis
        error_analysis = results["error_analysis"]
        if error_analysis["total_incorrect"] > 0:
            report.append("ERROR ANALYSIS")
            report.append("-" * 50)
            report.append(f"Total Incorrect Answers: {error_analysis['total_incorrect']}")
            report.append(f"System Errors: {error_analysis['total_errors']}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 50)
        overall_acc = metrics['overall_accuracy']
        if overall_acc >= 0.9:
            report.append("✓ Excellent performance across all complexity levels")
        elif overall_acc >= 0.7:
            report.append("• Good performance with room for improvement in complex problems")
        else:
            report.append("• Significant improvements needed, especially for L2/L3 problems")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_text = "\n".join(report)
        
        with open(f"comprehensive_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text


def main():
    """Main execution function."""
    print("🚀 Starting Advanced Mathematical Reasoning System Demonstration")
    print("="*80)
    
    # Initialize demo system
    demo = AdvancedExperimentalDemo()
    
    try:
        # Run comprehensive evaluation
        print("📊 Running comprehensive evaluation...")
        results = demo.run_comprehensive_evaluation()
        
        # Generate visualizations
        print("📈 Generating visualization report...")
        demo.generate_visualization_report(results)
        
        # Generate comprehensive report
        print("📋 Generating comprehensive text report...")
        report = demo.generate_comprehensive_report()
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Overall Accuracy: {results['performance_metrics']['overall_accuracy']:.1%}")
        print(f"Total Problems: {results['performance_metrics']['total_problems']}")
        print(f"Average Time: {results['performance_metrics']['average_solve_time']:.3f}s")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        print(f"❌ Error: {e}")
    
    print("🎯 Advanced demonstration completed!")


if __name__ == "__main__":
    main()