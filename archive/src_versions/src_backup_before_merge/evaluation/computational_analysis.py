"""
Computational Complexity and Performance Analysis
================================================

Tools for analyzing computational performance, memory usage, and scalability.
"""

import gc
import logging
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test"""
    execution_time: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float
    complexity_level: str
    problem_size: int
    success: bool


@dataclass
class ComplexityAnalysis:
    """Computational complexity analysis results"""
    time_complexity: str
    space_complexity: str
    scaling_factor: float
    performance_metrics: List[PerformanceMetrics]
    bottlenecks: List[str]
    recommendations: List[str]


class ComputationalAnalyzer:
    """Computational performance and complexity analyzer"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.performance_history = []
        self.baseline_metrics = None
        
    def analyze_system_performance(self, 
                                 solve_function: Callable,
                                 test_problems: List[Dict],
                                 warmup_runs: int = 3) -> Dict[str, Any]:
        """Comprehensive system performance analysis"""
        
        logger.info("Starting computational performance analysis...")
        
        # Warmup runs
        self._warmup_system(solve_function, test_problems[:min(3, len(test_problems))], warmup_runs)
        
        # Performance testing
        performance_results = self._run_performance_tests(solve_function, test_problems)
        
        # Scalability analysis
        scalability_results = self._analyze_scalability(solve_function, test_problems)
        
        # Memory profiling
        memory_analysis = self._analyze_memory_usage(solve_function, test_problems)
        
        # Complexity analysis
        complexity_analysis = self._analyze_complexity_scaling(performance_results)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            performance_results, scalability_results, memory_analysis
        )
        
        return {
            "performance_metrics": performance_results,
            "scalability_analysis": scalability_results,
            "memory_analysis": memory_analysis,
            "complexity_analysis": complexity_analysis,
            "system_info": self._get_system_info(),
            "recommendations": recommendations,
            "summary": self._generate_performance_summary(performance_results)
        }
    
    def _warmup_system(self, solve_function: Callable, problems: List[Dict], runs: int):
        """Warmup the system to stabilize performance measurements"""
        logger.info(f"Warming up system with {runs} runs...")
        
        for _ in range(runs):
            for problem in problems:
                try:
                    solve_function(problem["problem"])
                except:
                    pass  # Ignore warmup errors
        
        # Force garbage collection
        gc.collect()
    
    def _run_performance_tests(self, solve_function: Callable, test_problems: List[Dict]) -> List[PerformanceMetrics]:
        """Run detailed performance tests"""
        
        results = []
        process = psutil.Process()
        
        for i, problem in enumerate(test_problems):
            logger.debug(f"Performance test {i+1}/{len(test_problems)}")
            
            # Prepare monitoring
            tracemalloc.start()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute and measure
            start_time = time.time()
            start_cpu = process.cpu_percent()
            
            try:
                result = solve_function(problem["problem"])
                success = True
            except Exception as e:
                logger.debug(f"Execution failed: {e}")
                success = False
                result = None
            
            end_time = time.time()
            end_cpu = process.cpu_percent()
            
            # Memory measurements
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024
            tracemalloc.stop()
            
            # Create metrics
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_peak_mb=peak_memory,
                memory_current_mb=current_memory - start_memory,
                cpu_percent=(start_cpu + end_cpu) / 2,
                complexity_level=problem.get("complexity", "unknown"),
                problem_size=len(problem["problem"]),
                success=success
            )
            
            results.append(metrics)
            self.performance_history.append(metrics)
        
        return results
    
    def _analyze_scalability(self, solve_function: Callable, test_problems: List[Dict]) -> Dict[str, Any]:
        """Analyze system scalability across different problem sizes and complexities"""
        
        # Group problems by complexity
        by_complexity = {}
        for problem in test_problems:
            complexity = problem.get("complexity", "unknown")
            if complexity not in by_complexity:
                by_complexity[complexity] = []
            by_complexity[complexity].append(problem)
        
        scalability_results = {}
        
        for complexity, problems in by_complexity.items():
            if len(problems) < 3:  # Need minimum problems for analysis
                continue
            
            # Sort by problem size (text length as proxy)
            problems.sort(key=lambda p: len(p["problem"]))
            
            # Sample different sizes
            size_samples = self._select_size_samples(problems)
            
            # Measure performance for each size
            size_performance = []
            for sample_problems in size_samples:
                avg_time = 0
                avg_memory = 0
                success_rate = 0
                
                for problem in sample_problems:
                    tracemalloc.start()
                    start_time = time.time()
                    
                    try:
                        solve_function(problem["problem"])
                        success = True
                    except:
                        success = False
                    
                    end_time = time.time()
                    memory_used = tracemalloc.get_traced_memory()[1] / 1024 / 1024
                    tracemalloc.stop()
                    
                    avg_time += end_time - start_time
                    avg_memory += memory_used
                    success_rate += success
                
                size_performance.append({
                    "problem_count": len(sample_problems),
                    "avg_problem_size": np.mean([len(p["problem"]) for p in sample_problems]),
                    "avg_time": avg_time / len(sample_problems),
                    "avg_memory": avg_memory / len(sample_problems),
                    "success_rate": success_rate / len(sample_problems)
                })
            
            # Calculate scaling factor
            if len(size_performance) >= 2:
                scaling_factor = self._calculate_scaling_factor(size_performance)
                scalability_results[complexity] = {
                    "size_performance": size_performance,
                    "scaling_factor": scaling_factor,
                    "is_scalable": scaling_factor < 2.0  # Linear or better
                }
        
        return scalability_results
    
    def _analyze_memory_usage(self, solve_function: Callable, test_problems: List[Dict]) -> Dict[str, Any]:
        """Detailed memory usage analysis"""
        
        memory_profiles = []
        
        # Sample problems for detailed memory analysis
        sample_problems = test_problems[:min(10, len(test_problems))]
        
        for problem in sample_problems:
            # Memory profiling
            tracemalloc.start()
            gc.collect()  # Clean start
            
            baseline_memory = tracemalloc.get_traced_memory()[0]
            
            try:
                result = solve_function(problem["problem"])
                success = True
            except:
                success = False
            
            peak_memory = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            memory_profiles.append({
                "problem_size": len(problem["problem"]),
                "complexity": problem.get("complexity", "unknown"),
                "baseline_mb": baseline_memory / 1024 / 1024,
                "peak_mb": peak_memory / 1024 / 1024,
                "memory_growth": (peak_memory - baseline_memory) / 1024 / 1024,
                "success": success
            })
        
        # Analyze memory patterns
        memory_by_complexity = {}
        for profile in memory_profiles:
            complexity = profile["complexity"]
            if complexity not in memory_by_complexity:
                memory_by_complexity[complexity] = []
            memory_by_complexity[complexity].append(profile)
        
        # Calculate statistics
        memory_stats = {}
        for complexity, profiles in memory_by_complexity.items():
            memory_stats[complexity] = {
                "avg_peak_memory": np.mean([p["peak_mb"] for p in profiles]),
                "max_peak_memory": np.max([p["peak_mb"] for p in profiles]),
                "avg_memory_growth": np.mean([p["memory_growth"] for p in profiles]),
                "memory_efficiency": np.mean([p["peak_mb"] / max(p["problem_size"], 1) * 1000 for p in profiles])
            }
        
        return {
            "memory_profiles": memory_profiles,
            "memory_by_complexity": memory_stats,
            "overall_memory_efficiency": np.mean([s["memory_efficiency"] for s in memory_stats.values()]),
            "memory_recommendations": self._generate_memory_recommendations(memory_stats)
        }
    
    def _analyze_complexity_scaling(self, performance_results: List[PerformanceMetrics]) -> ComplexityAnalysis:
        """Analyze computational complexity scaling"""
        
        # Group by complexity level
        by_complexity = {}
        for metrics in performance_results:
            level = metrics.complexity_level
            if level not in by_complexity:
                by_complexity[level] = []
            by_complexity[level].append(metrics)
        
        # Analyze scaling within each complexity level
        scaling_analysis = {}
        for level, metrics_list in by_complexity.items():
            if len(metrics_list) < 3:
                continue
            
            # Sort by problem size
            metrics_list.sort(key=lambda m: m.problem_size)
            
            # Calculate scaling factor
            sizes = [m.problem_size for m in metrics_list]
            times = [m.execution_time for m in metrics_list]
            
            scaling_factor = self._calculate_time_scaling(sizes, times)
            
            scaling_analysis[level] = {
                "scaling_factor": scaling_factor,
                "avg_time": np.mean(times),
                "avg_memory": np.mean([m.memory_peak_mb for m in metrics_list]),
                "success_rate": np.mean([m.success for m in metrics_list])
            }
        
        # Determine overall complexity
        overall_scaling = np.mean([analysis["scaling_factor"] for analysis in scaling_analysis.values()])
        
        if overall_scaling <= 1.2:
            time_complexity = "O(n) - Linear"
        elif overall_scaling <= 1.8:
            time_complexity = "O(n log n) - Log-linear" 
        elif overall_scaling <= 2.2:
            time_complexity = "O(n²) - Quadratic"
        else:
            time_complexity = "O(n³+) - Polynomial or worse"
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(performance_results, scaling_analysis)
        
        # Generate recommendations
        recommendations = self._generate_complexity_recommendations(time_complexity, bottlenecks)
        
        return ComplexityAnalysis(
            time_complexity=time_complexity,
            space_complexity="O(n) - Linear",  # Simplified assumption
            scaling_factor=overall_scaling,
            performance_metrics=performance_results,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )
    
    def _select_size_samples(self, problems: List[Dict]) -> List[List[Dict]]:
        """Select representative samples of different sizes"""
        
        if len(problems) <= 6:
            return [problems]
        
        # Create size-based bins
        small = problems[:len(problems)//3]
        medium = problems[len(problems)//3:2*len(problems)//3]
        large = problems[2*len(problems)//3:]
        
        return [small[:3], medium[:3], large[:3]]
    
    def _calculate_scaling_factor(self, size_performance: List[Dict]) -> float:
        """Calculate scaling factor from performance data"""
        
        if len(size_performance) < 2:
            return 1.0
        
        # Use time scaling as primary metric
        sizes = [p["avg_problem_size"] for p in size_performance]
        times = [p["avg_time"] for p in size_performance]
        
        return self._calculate_time_scaling(sizes, times)
    
    def _calculate_time_scaling(self, sizes: List[float], times: List[float]) -> float:
        """Calculate time scaling factor"""
        
        if len(sizes) < 2 or len(times) < 2:
            return 1.0
        
        # Calculate growth rate
        size_ratios = []
        time_ratios = []
        
        for i in range(1, len(sizes)):
            if sizes[i-1] > 0 and times[i-1] > 0:
                size_ratio = sizes[i] / sizes[i-1]
                time_ratio = times[i] / times[i-1]
                
                if size_ratio > 1.1:  # Significant size increase
                    size_ratios.append(size_ratio)
                    time_ratios.append(time_ratio)
        
        if not size_ratios:
            return 1.0
        
        # Calculate average scaling exponent
        scaling_factors = []
        for size_ratio, time_ratio in zip(size_ratios, time_ratios):
            if size_ratio > 1:
                scaling_factor = np.log(time_ratio) / np.log(size_ratio)
                scaling_factors.append(max(0, min(5, scaling_factor)))  # Clamp reasonable range
        
        return np.mean(scaling_factors) if scaling_factors else 1.0
    
    def _identify_bottlenecks(self, performance_results: List[PerformanceMetrics], 
                            scaling_analysis: Dict) -> List[str]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        # Check for high memory usage
        avg_memory = np.mean([m.memory_peak_mb for m in performance_results])
        if avg_memory > 100:  # MB
            bottlenecks.append(f"High memory usage: {avg_memory:.1f}MB average")
        
        # Check for slow complex problems
        for level, analysis in scaling_analysis.items():
            if level in ["L2", "L3"] and analysis["avg_time"] > 5:
                bottlenecks.append(f"Slow {level} processing: {analysis['avg_time']:.2f}s average")
        
        # Check for poor scaling
        for level, analysis in scaling_analysis.items():
            if analysis["scaling_factor"] > 2.5:
                bottlenecks.append(f"Poor scaling for {level}: factor {analysis['scaling_factor']:.1f}")
        
        # Check for low success rate
        for level, analysis in scaling_analysis.items():
            if analysis["success_rate"] < 0.8:
                bottlenecks.append(f"Low success rate for {level}: {analysis['success_rate']:.1%}")
        
        return bottlenecks
    
    def _generate_complexity_recommendations(self, time_complexity: str, bottlenecks: List[str]) -> List[str]:
        """Generate recommendations based on complexity analysis"""
        
        recommendations = []
        
        if "Quadratic" in time_complexity or "Polynomial" in time_complexity:
            recommendations.append("Consider algorithmic optimization - complexity worse than linear")
        
        if any("memory" in bottleneck.lower() for bottleneck in bottlenecks):
            recommendations.append("Implement memory optimization strategies")
        
        if any("slow" in bottleneck.lower() for bottleneck in bottlenecks):
            recommendations.append("Focus on optimizing complex problem processing")
        
        if any("scaling" in bottleneck.lower() for bottleneck in bottlenecks):
            recommendations.append("Redesign algorithms for better scalability")
        
        return recommendations
    
    def _generate_memory_recommendations(self, memory_stats: Dict) -> List[str]:
        """Generate memory-specific recommendations"""
        
        recommendations = []
        
        # Check overall memory efficiency
        avg_efficiency = np.mean([stats["memory_efficiency"] for stats in memory_stats.values()])
        if avg_efficiency > 10:  # MB per 1000 characters
            recommendations.append("Improve memory efficiency - high memory per problem size")
        
        # Check for memory growth patterns
        for complexity, stats in memory_stats.items():
            if stats["avg_memory_growth"] > 50:  # MB
                recommendations.append(f"Address memory growth in {complexity} problems")
        
        return recommendations
    
    def _generate_performance_recommendations(self, performance_results: List[PerformanceMetrics],
                                           scalability_results: Dict, memory_analysis: Dict) -> List[str]:
        """Generate comprehensive performance recommendations"""
        
        recommendations = []
        
        # Overall performance
        avg_time = np.mean([m.execution_time for m in performance_results if m.success])
        if avg_time > 5:
            recommendations.append("Consider overall performance optimization - average time > 5s")
        
        # Success rate
        success_rate = np.mean([m.success for m in performance_results])
        if success_rate < 0.9:
            recommendations.append(f"Improve system reliability - {success_rate:.1%} success rate")
        
        # Scalability issues
        for complexity, analysis in scalability_results.items():
            if not analysis.get("is_scalable", True):
                recommendations.append(f"Address scalability issues for {complexity} problems")
        
        # Memory recommendations
        recommendations.extend(memory_analysis.get("memory_recommendations", []))
        
        return recommendations
    
    def _generate_performance_summary(self, performance_results: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Generate performance summary statistics"""
        
        successful_results = [m for m in performance_results if m.success]
        
        if not successful_results:
            return {"error": "No successful executions for analysis"}
        
        return {
            "total_tests": len(performance_results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(performance_results),
            "avg_execution_time": np.mean([m.execution_time for m in successful_results]),
            "max_execution_time": np.max([m.execution_time for m in successful_results]),
            "avg_memory_usage": np.mean([m.memory_peak_mb for m in successful_results]),
            "max_memory_usage": np.max([m.memory_peak_mb for m in successful_results]),
            "performance_by_complexity": self._summarize_by_complexity(successful_results)
        }
    
    def _summarize_by_complexity(self, results: List[PerformanceMetrics]) -> Dict[str, Dict]:
        """Summarize performance by complexity level"""
        
        by_complexity = {}
        for result in results:
            level = result.complexity_level
            if level not in by_complexity:
                by_complexity[level] = []
            by_complexity[level].append(result)
        
        summary = {}
        for level, metrics_list in by_complexity.items():
            summary[level] = {
                "count": len(metrics_list),
                "avg_time": np.mean([m.execution_time for m in metrics_list]),
                "avg_memory": np.mean([m.memory_peak_mb for m in metrics_list]),
                "success_rate": np.mean([m.success for m in metrics_list])
            }
        
        return summary
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility"""
        
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "python_version": psutil.__version__,
            "platform": psutil.LINUX or psutil.WINDOWS or psutil.MACOS
        }


def run_computational_analysis_demo():
    """Demo function for computational analysis"""
    
    def dummy_solve_function(problem_text: str) -> str:
        """Dummy solve function for testing"""
        time.sleep(0.1 + len(problem_text) * 0.0001)  # Simulate work
        return "42"
    
    # Sample test problems
    test_problems = [
        {"problem": "Simple math problem", "complexity": "L0"},
        {"problem": "More complex word problem with multiple steps", "complexity": "L1"},
        {"problem": "Very complex multi-step reasoning problem requiring deep analysis", "complexity": "L2"}
    ]
    
    # Run analysis
    analyzer = ComputationalAnalyzer()
    results = analyzer.analyze_system_performance(dummy_solve_function, test_problems)
    
    # Save results
    import json
    with open("computational_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Computational analysis completed")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_computational_analysis_demo() 