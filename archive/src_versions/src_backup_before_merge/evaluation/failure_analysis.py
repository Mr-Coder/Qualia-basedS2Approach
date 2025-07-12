"""
Failure Case Analysis Framework
===============================

Systematic failure case analysis and error pattern recognition.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FailureCase:
    """Individual failure case data"""
    problem_id: str
    problem_text: str
    expected_answer: Any
    predicted_answer: Any
    complexity_level: str
    error_category: str
    error_message: Optional[str] = None
    processing_time: float = 0.0
    confidence_score: float = 0.0
    reasoning_steps: List[str] = None


@dataclass
class ErrorPattern:
    """Identified error pattern"""
    pattern_name: str
    frequency: int
    complexity_distribution: Dict[str, int]
    examples: List[str]
    proposed_solution: str


class FailureAnalyzer:
    """Systematic failure case analyzer"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.failure_cases = []
        self.error_categories = {
            "domain_knowledge_gap": "Missing specialized domain knowledge",
            "relation_discovery_failure": "Failed to identify implicit relations",
            "numerical_computation_error": "Arithmetic or symbolic computation mistake", 
            "reasoning_chain_break": "Logical inconsistency in reasoning steps",
            "parsing_error": "Failed to parse problem correctly",
            "timeout_error": "Processing timeout",
            "validation_error": "Answer validation failed"
        }
        
    def analyze_failures(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Analyze failure cases and identify patterns"""
        logger.info("Starting systematic failure analysis...")
        
        # Extract failure cases
        self.failure_cases = self._extract_failure_cases(test_results)
        
        # Categorize errors
        categorized_errors = self._categorize_errors()
        
        # Identify error patterns
        error_patterns = self._identify_error_patterns()
        
        # Analyze by complexity level
        complexity_analysis = self._analyze_by_complexity()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            "total_failures": len(self.failure_cases),
            "error_categories": categorized_errors,
            "error_patterns": error_patterns,
            "complexity_analysis": complexity_analysis,
            "critical_failure_modes": self._identify_critical_failures(),
            "robustness_assessment": self._assess_robustness(),
            "recommendations": recommendations,
            "detailed_cases": [self._case_to_dict(case) for case in self.failure_cases[:50]]  # Top 50
        }
    
    def _extract_failure_cases(self, test_results: List[Dict]) -> List[FailureCase]:
        """Extract failure cases from test results"""
        failures = []
        
        for i, result in enumerate(test_results):
            if not result.get("is_correct", False) or result.get("error_message"):
                
                # Determine error category
                error_category = self._classify_error(result)
                
                failure = FailureCase(
                    problem_id=result.get("test_id", f"problem_{i}"),
                    problem_text=result.get("problem", ""),
                    expected_answer=result.get("expected_answer"),
                    predicted_answer=result.get("system_answer"),
                    complexity_level=result.get("complexity", "unknown"),
                    error_category=error_category,
                    error_message=result.get("error_message"),
                    processing_time=result.get("solve_time", 0),
                    confidence_score=result.get("confidence_score", 0),
                    reasoning_steps=result.get("reasoning_steps", [])
                )
                failures.append(failure)
        
        logger.info(f"Extracted {len(failures)} failure cases")
        return failures
    
    def _classify_error(self, result: Dict) -> str:
        """Classify error into predefined categories"""
        
        error_msg = result.get("error_message", "").lower()
        problem_text = result.get("problem", "").lower()
        
        # Check for specific error patterns
        if "timeout" in error_msg or result.get("solve_time", 0) > 30:
            return "timeout_error"
        
        if "parse" in error_msg or "invalid format" in error_msg:
            return "parsing_error"
        
        if "validation" in error_msg or "verify" in error_msg:
            return "validation_error"
        
        if "domain" in error_msg or "knowledge" in error_msg:
            return "domain_knowledge_gap"
        
        if "relation" in error_msg or "dependency" in error_msg:
            return "relation_discovery_failure"
        
        # Check for numerical errors
        expected = result.get("expected_answer")
        predicted = result.get("system_answer")
        
        if expected and predicted:
            try:
                exp_num = float(expected)
                pred_num = float(predicted)
                
                # Check for computational errors
                if abs(exp_num - pred_num) > 0.1 and not any(keyword in problem_text 
                    for keyword in ["physics", "chemistry", "advanced"]):
                    return "numerical_computation_error"
            except (ValueError, TypeError):
                pass
        
        # Check for domain-specific problems
        if any(keyword in problem_text for keyword in 
               ["physics", "chemistry", "thermodynamics", "mechanics", "calculus"]):
            return "domain_knowledge_gap"
        
        # Default to reasoning chain break
        return "reasoning_chain_break"
    
    def _categorize_errors(self) -> Dict[str, Any]:
        """Categorize errors by type and complexity"""
        
        error_stats = defaultdict(lambda: {
            "count": 0,
            "percentage": 0,
            "by_complexity": {"L0": 0, "L1": 0, "L2": 0, "L3": 0},
            "avg_confidence": 0,
            "avg_time": 0
        })
        
        total_failures = len(self.failure_cases)
        
        for case in self.failure_cases:
            category = case.error_category
            error_stats[category]["count"] += 1
            error_stats[category]["by_complexity"][case.complexity_level] += 1
            error_stats[category]["avg_confidence"] += case.confidence_score
            error_stats[category]["avg_time"] += case.processing_time
        
        # Calculate averages and percentages
        for category, stats in error_stats.items():
            count = stats["count"]
            stats["percentage"] = (count / total_failures * 100) if total_failures > 0 else 0
            stats["avg_confidence"] = stats["avg_confidence"] / count if count > 0 else 0
            stats["avg_time"] = stats["avg_time"] / count if count > 0 else 0
        
        return dict(error_stats)
    
    def _identify_error_patterns(self) -> List[ErrorPattern]:
        """Identify common error patterns"""
        patterns = []
        
        # Group failures by error category
        by_category = defaultdict(list)
        for case in self.failure_cases:
            by_category[case.error_category].append(case)
        
        for category, cases in by_category.items():
            if len(cases) >= 3:  # Minimum frequency for pattern
                
                # Analyze complexity distribution
                complexity_dist = Counter(case.complexity_level for case in cases)
                
                # Extract example problems
                examples = [case.problem_text[:100] + "..." for case in cases[:3]]
                
                # Generate proposed solution
                solution = self._generate_solution_for_category(category, cases)
                
                pattern = ErrorPattern(
                    pattern_name=category,
                    frequency=len(cases),
                    complexity_distribution=dict(complexity_dist),
                    examples=examples,
                    proposed_solution=solution
                )
                patterns.append(pattern)
        
        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns
    
    def _analyze_by_complexity(self) -> Dict[str, Any]:
        """Analyze failures by complexity level"""
        
        complexity_stats = defaultdict(lambda: {
            "total_failures": 0,
            "error_categories": defaultdict(int),
            "avg_confidence": 0,
            "avg_processing_time": 0,
            "common_errors": []
        })
        
        # Group by complexity
        for case in self.failure_cases:
            level = case.complexity_level
            complexity_stats[level]["total_failures"] += 1
            complexity_stats[level]["error_categories"][case.error_category] += 1
            complexity_stats[level]["avg_confidence"] += case.confidence_score
            complexity_stats[level]["avg_processing_time"] += case.processing_time
        
        # Calculate averages and identify patterns
        for level, stats in complexity_stats.items():
            total = stats["total_failures"]
            if total > 0:
                stats["avg_confidence"] /= total
                stats["avg_processing_time"] /= total
                
                # Find most common errors for this level
                most_common = Counter(stats["error_categories"]).most_common(3)
                stats["common_errors"] = [{"category": cat, "count": count} 
                                        for cat, count in most_common]
        
        return dict(complexity_stats)
    
    def _identify_critical_failures(self) -> List[Dict[str, Any]]:
        """Identify critical failure modes that need immediate attention"""
        critical_failures = []
        
        # High-frequency error patterns
        category_counts = Counter(case.error_category for case in self.failure_cases)
        for category, count in category_counts.most_common(3):
            if count >= len(self.failure_cases) * 0.15:  # 15% threshold
                critical_failures.append({
                    "type": "high_frequency_error",
                    "category": category,
                    "frequency": count,
                    "severity": "critical",
                    "description": f"{category} accounts for {count} failures ({count/len(self.failure_cases)*100:.1f}%)"
                })
        
        # Complex problem failures (L3)
        l3_failures = [case for case in self.failure_cases if case.complexity_level == "L3"]
        if len(l3_failures) >= 5:
            critical_failures.append({
                "type": "complex_problem_handling",
                "category": "L3_failures",
                "frequency": len(l3_failures),
                "severity": "high",
                "description": f"Systematic failures on L3 problems: {len(l3_failures)} cases"
            })
        
        # Low confidence failures
        low_confidence = [case for case in self.failure_cases if case.confidence_score < 0.3]
        if len(low_confidence) >= len(self.failure_cases) * 0.2:
            critical_failures.append({
                "type": "confidence_calibration",
                "category": "low_confidence",
                "frequency": len(low_confidence),
                "severity": "medium",
                "description": f"Poor confidence calibration: {len(low_confidence)} low-confidence failures"
            })
        
        return critical_failures
    
    def _assess_robustness(self) -> Dict[str, Any]:
        """Assess system robustness across different dimensions"""
        
        robustness_metrics = {}
        
        # Complexity robustness
        complexity_failure_rates = {}
        complexity_counts = Counter(case.complexity_level for case in self.failure_cases)
        
        for level in ["L0", "L1", "L2", "L3"]:
            failure_count = complexity_counts.get(level, 0)
            # Note: This assumes total test count per level - should be passed as parameter
            failure_rate = failure_count / max(1, 100)  # Placeholder
            complexity_failure_rates[level] = failure_rate
        
        robustness_metrics["complexity_robustness"] = {
            "failure_rates": complexity_failure_rates,
            "robustness_score": 1.0 - np.mean(list(complexity_failure_rates.values()))
        }
        
        # Error diversity (more diverse errors = less robust)
        error_diversity = len(set(case.error_category for case in self.failure_cases))
        robustness_metrics["error_diversity"] = {
            "unique_error_types": error_diversity,
            "diversity_score": error_diversity / len(self.error_categories)
        }
        
        # Confidence calibration
        low_conf_failures = [case for case in self.failure_cases if case.confidence_score < 0.5]
        robustness_metrics["confidence_calibration"] = {
            "low_confidence_failures": len(low_conf_failures),
            "calibration_score": 1.0 - (len(low_conf_failures) / len(self.failure_cases))
        }
        
        # Overall robustness score
        overall_score = np.mean([
            robustness_metrics["complexity_robustness"]["robustness_score"],
            1.0 - robustness_metrics["error_diversity"]["diversity_score"],
            robustness_metrics["confidence_calibration"]["calibration_score"]
        ])
        
        robustness_metrics["overall_robustness_score"] = overall_score
        
        return robustness_metrics
    
    def _generate_solution_for_category(self, category: str, cases: List[FailureCase]) -> str:
        """Generate proposed solution for error category"""
        
        solutions = {
            "domain_knowledge_gap": "Integrate specialized domain knowledge bases for physics, chemistry, and advanced mathematics",
            "relation_discovery_failure": "Enhance implicit relation discovery with better pattern matching and context analysis",
            "numerical_computation_error": "Improve numerical precision and add computation verification steps",
            "reasoning_chain_break": "Strengthen logical consistency checking and step-by-step validation",
            "parsing_error": "Enhance problem text parsing with better error handling and format detection",
            "timeout_error": "Optimize algorithms for computational efficiency and add progressive timeout handling",
            "validation_error": "Improve answer validation with multiple verification strategies"
        }
        
        return solutions.get(category, "Investigate and address specific failure patterns")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not self.failure_cases:
            return ["No failures detected - system performing well"]
        
        # Most common error type
        most_common_error = Counter(case.error_category for case in self.failure_cases).most_common(1)[0]
        recommendations.append(
            f"Priority 1: Address {most_common_error[0]} (affects {most_common_error[1]} cases)"
        )
        
        # Complexity-specific recommendations
        l3_failures = [case for case in self.failure_cases if case.complexity_level == "L3"]
        if len(l3_failures) > len(self.failure_cases) * 0.3:
            recommendations.append("Priority 2: Focus on L3 problem handling - high failure rate detected")
        
        # Confidence calibration
        avg_confidence = np.mean([case.confidence_score for case in self.failure_cases])
        if avg_confidence > 0.5:
            recommendations.append("Priority 3: Improve confidence calibration - system overconfident on failures")
        
        # Performance optimization
        avg_time = np.mean([case.processing_time for case in self.failure_cases])
        if avg_time > 10:
            recommendations.append("Consider performance optimization - failures taking excessive time")
        
        return recommendations
    
    def _case_to_dict(self, case: FailureCase) -> Dict:
        """Convert failure case to dictionary"""
        return {
            "problem_id": case.problem_id,
            "problem_text": case.problem_text[:200] + "..." if len(case.problem_text) > 200 else case.problem_text,
            "expected_answer": case.expected_answer,
            "predicted_answer": case.predicted_answer,
            "complexity_level": case.complexity_level,
            "error_category": case.error_category,
            "error_message": case.error_message,
            "processing_time": case.processing_time,
            "confidence_score": case.confidence_score
        }
    
    def export_failure_report(self, results: Dict[str, Any], filename: str = "failure_analysis_report.json"):
        """Export detailed failure analysis report"""
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Failure analysis report saved to {filename}")


def run_failure_analysis_demo():
    """Demo function for failure analysis"""
    
    # Sample failure test results
    sample_results = [
        {
            "test_id": "fail_001",
            "problem": "在20°C的环境中，有一块重量为500克的冰块...",
            "expected_answer": "3340",
            "system_answer": "1670", 
            "is_correct": False,
            "complexity": "L3",
            "confidence_score": 0.8,
            "solve_time": 12.5,
            "error_message": "Domain knowledge gap in thermodynamics"
        },
        {
            "test_id": "fail_002", 
            "problem": "小明有15个苹果，给了小红3个...",
            "expected_answer": "12",
            "system_answer": "18",
            "is_correct": False,
            "complexity": "L0",
            "confidence_score": 0.9,
            "solve_time": 2.1,
            "error_message": "Arithmetic computation error"
        }
    ]
    
    # Run analysis
    analyzer = FailureAnalyzer()
    results = analyzer.analyze_failures(sample_results)
    
    # Export report
    analyzer.export_failure_report(results)
    
    return results


if __name__ == "__main__":
    run_failure_analysis_demo() 