"""
Performance Analysis Data Module

This module contains performance analysis data from multiple evaluation tables,
including computational efficiency, ablation studies, performance comparisons,
relation discovery, and reasoning chain quality assessments.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MethodPerformance:
    """Data class for method performance across datasets."""
    method_name: str
    math23k: float
    gsm8k: float
    mawps: float
    mathqa: float
    math: float
    svamp: float
    asdiv: float
    dir_test: float


@dataclass
class ComplexityPerformance:
    """Data class for performance by complexity level."""
    method_name: str
    l0_explicit: float
    l1_shallow: float
    l2_medium: float
    l3_deep: float
    robustness_score: float


@dataclass
class EfficiencyMetrics:
    """Data class for computational efficiency metrics."""
    method_name: str
    avg_runtime: float
    memory_mb: int
    l2_runtime: float
    l3_runtime: float
    efficiency_score: float


@dataclass
class AblationResults:
    """Data class for ablation study results."""
    configuration: str
    overall_acc: float
    l2_acc: float
    l3_acc: float
    relation_f1: float
    chain_quality: float
    efficiency: float


@dataclass
class ComponentInteraction:
    """Data class for component interaction analysis."""
    combination: str
    overall_acc: float
    relation_discovery: float
    reasoning_quality: float
    error_rate: float
    synergy_score: float


@dataclass
class RelationDiscoveryMetrics:
    """Data class for implicit relation discovery assessment."""
    method_name: str
    precision: float
    recall: float
    f1_score: float
    semantic_acc: float
    l2_f1: float
    l3_f1: float
    avg_relations: float


@dataclass
class ReasoningChainMetrics:
    """Data class for reasoning chain quality assessment."""
    method_name: str
    logical_correctness: float
    completeness: float
    coherence: float
    efficiency: float
    verifiability: float
    overall_score: float


# Table 4: Overall Performance Comparison Across Datasets
PERFORMANCE_DATA = {
    "Claude-3.5-Sonnet": MethodPerformance(
        method_name="Claude-3.5-Sonnet",
        math23k=82.4, gsm8k=85.7, mawps=89.3, mathqa=74.2,
        math=61.8, svamp=83.6, asdiv=87.1, dir_test=69.4
    ),
    "GPT-4o": MethodPerformance(
        method_name="GPT-4o",
        math23k=81.7, gsm8k=84.2, mawps=88.6, mathqa=73.1,
        math=59.7, svamp=82.3, asdiv=86.4, dir_test=68.2
    ),
    "Qwen2.5-Math-72B": MethodPerformance(
        method_name="Qwen2.5-Math-72B",
        math23k=84.1, gsm8k=87.9, mawps=91.2, mathqa=76.8,
        math=64.3, svamp=85.9, asdiv=89.7, dir_test=72.1
    ),
    "InternLM2.5-Math-7B": MethodPerformance(
        method_name="InternLM2.5-Math-7B",
        math23k=79.3, gsm8k=82.1, mawps=85.4, mathqa=70.6,
        math=56.2, svamp=80.7, asdiv=83.8, dir_test=65.3
    ),
    "DeepSeek-Math-7B": MethodPerformance(
        method_name="DeepSeek-Math-7B",
        math23k=80.8, gsm8k=83.6, mawps=87.1, mathqa=72.4,
        math=58.9, svamp=82.1, asdiv=85.2, dir_test=67.6
    ),
    "ToRA-13B": MethodPerformance(
        method_name="ToRA-13B",
        math23k=78.2, gsm8k=81.3, mawps=84.7, mathqa=69.8,
        math=54.6, svamp=79.4, asdiv=82.5, dir_test=63.9
    ),
    "COT-DIR": MethodPerformance(
        method_name="COT-DIR",
        math23k=87.3, gsm8k=91.2, mawps=94.1, mathqa=80.4,
        math=68.7, svamp=89.3, asdiv=92.8, dir_test=78.5
    )
}

# Table 5: Performance Analysis by Problem Complexity Level
COMPLEXITY_PERFORMANCE = {
    "Claude-3.5-Sonnet": ComplexityPerformance(
        method_name="Claude-3.5-Sonnet",
        l0_explicit=94.2, l1_shallow=87.6, l2_medium=78.4, l3_deep=65.7,
        robustness_score=0.74
    ),
    "GPT-4o": ComplexityPerformance(
        method_name="GPT-4o",
        l0_explicit=92.8, l1_shallow=85.3, l2_medium=76.1, l3_deep=63.2,
        robustness_score=0.71
    ),
    "Qwen2.5-Math-72B": ComplexityPerformance(
        method_name="Qwen2.5-Math-72B",
        l0_explicit=95.1, l1_shallow=89.3, l2_medium=80.7, l3_deep=68.4,
        robustness_score=0.77
    ),
    "InternLM2.5-Math-7B": ComplexityPerformance(
        method_name="InternLM2.5-Math-7B",
        l0_explicit=88.9, l1_shallow=80.4, l2_medium=70.1, l3_deep=57.2,
        robustness_score=0.66
    ),
    "DeepSeek-Math-7B": ComplexityPerformance(
        method_name="DeepSeek-Math-7B",
        l0_explicit=89.6, l1_shallow=81.8, l2_medium=71.9, l3_deep=59.1,
        robustness_score=0.68
    ),
    "Graph2Tree": ComplexityPerformance(
        method_name="Graph2Tree",
        l0_explicit=88.6, l1_shallow=79.2, l2_medium=68.5, l3_deep=54.3,
        robustness_score=0.62
    ),
    "COT-DIR": ComplexityPerformance(
        method_name="COT-DIR",
        l0_explicit=95.1, l1_shallow=90.7, l2_medium=83.4, l3_deep=73.2,
        robustness_score=0.82
    )
}

# Table 10: Computational Efficiency Analysis
EFFICIENCY_DATA = {
    "Claude-3.5-Sonnet": EfficiencyMetrics(
        method_name="Claude-3.5-Sonnet",
        avg_runtime=1.8, memory_mb=245, l2_runtime=2.1, l3_runtime=2.7,
        efficiency_score=0.73
    ),
    "GPT-4o": EfficiencyMetrics(
        method_name="GPT-4o",
        avg_runtime=2.1, memory_mb=268, l2_runtime=2.4, l3_runtime=3.1,
        efficiency_score=0.69
    ),
    "Qwen2.5-Math-72B": EfficiencyMetrics(
        method_name="Qwen2.5-Math-72B",
        avg_runtime=3.2, memory_mb=412, l2_runtime=3.8, l3_runtime=4.9,
        efficiency_score=0.61
    ),
    "InternLM2.5-Math-7B": EfficiencyMetrics(
        method_name="InternLM2.5-Math-7B",
        avg_runtime=1.6, memory_mb=198, l2_runtime=1.9, l3_runtime=2.4,
        efficiency_score=0.76
    ),
    "COT-DIR": EfficiencyMetrics(
        method_name="COT-DIR",
        avg_runtime=2.3, memory_mb=287, l2_runtime=2.8, l3_runtime=3.6,
        efficiency_score=0.71
    )
}

# Table 8: Ablation Study - Individual Component Contributions
ABLATION_DATA = {
    "COT-DIR (Full)": AblationResults(
        configuration="COT-DIR (Full)",
        overall_acc=80.4, l2_acc=83.4, l3_acc=73.2,
        relation_f1=0.80, chain_quality=0.92, efficiency=2.3
    ),
    "w/o IRD": AblationResults(
        configuration="w/o IRD",
        overall_acc=72.8, l2_acc=75.5, l3_acc=61.7,
        relation_f1=0.39, chain_quality=0.85, efficiency=1.8
    ),
    "w/o MLR": AblationResults(
        configuration="w/o MLR",
        overall_acc=74.9, l2_acc=77.4, l3_acc=66.7,
        relation_f1=0.77, chain_quality=0.73, efficiency=1.9
    ),
    "w/o CV": AblationResults(
        configuration="w/o CV",
        overall_acc=77.6, l2_acc=80.1, l3_acc=70.4,
        relation_f1=0.78, chain_quality=0.78, efficiency=1.7
    ),
    "IRD only": AblationResults(
        configuration="IRD only",
        overall_acc=65.2, l2_acc=67.8, l3_acc=55.1,
        relation_f1=0.74, chain_quality=0.64, efficiency=1.2
    ),
    "MLR only": AblationResults(
        configuration="MLR only",
        overall_acc=68.7, l2_acc=71.3, l3_acc=59.6,
        relation_f1=0.36, chain_quality=0.81, efficiency=1.4
    ),
    "CV only": AblationResults(
        configuration="CV only",
        overall_acc=62.9, l2_acc=64.7, l3_acc=52.8,
        relation_f1=0.33, chain_quality=0.89, efficiency=1.1
    )
}

# Table 9: Component Interaction Analysis
COMPONENT_INTERACTION = {
    "IRD + MLR": ComponentInteraction(
        combination="IRD + MLR",
        overall_acc=78.9, relation_discovery=0.79, reasoning_quality=0.84,
        error_rate=19.2, synergy_score=0.71
    ),
    "IRD + CV": ComponentInteraction(
        combination="IRD + CV",
        overall_acc=78.3, relation_discovery=0.78, reasoning_quality=0.87,
        error_rate=15.8, synergy_score=0.69
    ),
    "MLR + CV": ComponentInteraction(
        combination="MLR + CV",
        overall_acc=76.4, relation_discovery=0.40, reasoning_quality=0.86,
        error_rate=17.3, synergy_score=0.66
    ),
    "IRD + MLR + CV": ComponentInteraction(
        combination="IRD + MLR + CV",
        overall_acc=80.4, relation_discovery=0.80, reasoning_quality=0.92,
        error_rate=13.1, synergy_score=0.84
    )
}

# Table 6: Implicit Relation Discovery Quality Assessment
RELATION_DISCOVERY_DATA = {
    "Claude-3.5-Sonnet": RelationDiscoveryMetrics(
        method_name="Claude-3.5-Sonnet",
        precision=0.73, recall=0.68, f1_score=0.70, semantic_acc=0.81,
        l2_f1=0.67, l3_f1=0.58, avg_relations=2.3
    ),
    "GPT-4o": RelationDiscoveryMetrics(
        method_name="GPT-4o",
        precision=0.71, recall=0.65, f1_score=0.68, semantic_acc=0.79,
        l2_f1=0.64, l3_f1=0.55, avg_relations=2.1
    ),
    "Qwen2.5-Math-72B": RelationDiscoveryMetrics(
        method_name="Qwen2.5-Math-72B",
        precision=0.69, recall=0.72, f1_score=0.70, semantic_acc=0.76,
        l2_f1=0.68, l3_f1=0.61, avg_relations=2.7
    ),
    "InternLM2.5-Math-7B": RelationDiscoveryMetrics(
        method_name="InternLM2.5-Math-7B",
        precision=0.62, recall=0.59, f1_score=0.60, semantic_acc=0.69,
        l2_f1=0.57, l3_f1=0.46, avg_relations=1.7
    ),
    "DeepSeek-Math-7B": RelationDiscoveryMetrics(
        method_name="DeepSeek-Math-7B",
        precision=0.64, recall=0.61, f1_score=0.62, semantic_acc=0.71,
        l2_f1=0.59, l3_f1=0.48, avg_relations=1.8
    ),
    "Graph2Tree": RelationDiscoveryMetrics(
        method_name="Graph2Tree",
        precision=0.45, recall=0.38, f1_score=0.41, semantic_acc=0.52,
        l2_f1=0.35, l3_f1=0.21, avg_relations=1.2
    ),
    "COT-DIR": RelationDiscoveryMetrics(
        method_name="COT-DIR",
        precision=0.82, recall=0.79, f1_score=0.80, semantic_acc=0.87,
        l2_f1=0.77, l3_f1=0.71, avg_relations=2.9
    )
}

# Table 7: Reasoning Chain Quality Assessment
REASONING_CHAIN_DATA = {
    "Claude-3.5-Sonnet": ReasoningChainMetrics(
        method_name="Claude-3.5-Sonnet",
        logical_correctness=0.87, completeness=0.82, coherence=0.89,
        efficiency=0.76, verifiability=0.71, overall_score=0.81
    ),
    "GPT-4o": ReasoningChainMetrics(
        method_name="GPT-4o",
        logical_correctness=0.85, completeness=0.79, coherence=0.86,
        efficiency=0.73, verifiability=0.68, overall_score=0.78
    ),
    "Qwen2.5-Math-72B": ReasoningChainMetrics(
        method_name="Qwen2.5-Math-72B",
        logical_correctness=0.82, completeness=0.84, coherence=0.81,
        efficiency=0.79, verifiability=0.76, overall_score=0.80
    ),
    "InternLM2.5-Math-7B": ReasoningChainMetrics(
        method_name="InternLM2.5-Math-7B",
        logical_correctness=0.78, completeness=0.75, coherence=0.77,
        efficiency=0.74, verifiability=0.69, overall_score=0.75
    ),
    "DeepSeek-Math-7B": ReasoningChainMetrics(
        method_name="DeepSeek-Math-7B",
        logical_correctness=0.79, completeness=0.76, coherence=0.78,
        efficiency=0.75, verifiability=0.70, overall_score=0.76
    ),
    "Graph2Tree": ReasoningChainMetrics(
        method_name="Graph2Tree",
        logical_correctness=0.71, completeness=0.68, coherence=0.65,
        efficiency=0.82, verifiability=0.89, overall_score=0.75
    ),
    "COT-DIR": ReasoningChainMetrics(
        method_name="COT-DIR",
        logical_correctness=0.93, completeness=0.91, coherence=0.94,
        efficiency=0.88, verifiability=0.96, overall_score=0.92
    )
}


def get_method_performance(method_name: str) -> Optional[MethodPerformance]:
    """Get performance data for a specific method."""
    return PERFORMANCE_DATA.get(method_name)


def get_all_methods() -> List[str]:
    """Get list of all available methods."""
    return list(PERFORMANCE_DATA.keys())


def get_best_performing_method(dataset: str) -> Tuple[str, float]:
    """Get the best performing method for a specific dataset."""
    best_method = ""
    best_score = 0.0
    
    for method_name, performance in PERFORMANCE_DATA.items():
        score = getattr(performance, dataset.lower().replace('-', '_'))
        if score > best_score:
            best_score = score
            best_method = method_name
    
    return best_method, best_score


def calculate_average_performance(method_name: str) -> float:
    """Calculate average performance across all datasets for a method."""
    if method_name not in PERFORMANCE_DATA:
        return 0.0
    
    performance = PERFORMANCE_DATA[method_name]
    scores = [
        performance.math23k, performance.gsm8k, performance.mawps,
        performance.mathqa, performance.math, performance.svamp,
        performance.asdiv, performance.dir_test
    ]
    return sum(scores) / len(scores)


def get_efficiency_ranking() -> List[Tuple[str, float]]:
    """Get methods ranked by efficiency score."""
    efficiency_scores = [(name, metrics.efficiency_score) 
                        for name, metrics in EFFICIENCY_DATA.items()]
    return sorted(efficiency_scores, key=lambda x: x[1], reverse=True)


def get_robustness_ranking() -> List[Tuple[str, float]]:
    """Get methods ranked by robustness score."""
    robustness_scores = [(name, metrics.robustness_score) 
                        for name, metrics in COMPLEXITY_PERFORMANCE.items()]
    return sorted(robustness_scores, key=lambda x: x[1], reverse=True)


def analyze_component_contribution() -> Dict[str, Any]:
    """Analyze the contribution of each component in COT-DIR."""
    full_performance = ABLATION_DATA["COT-DIR (Full)"]
    
    # Calculate contribution by comparing w/o configurations
    ird_contribution = full_performance.overall_acc - ABLATION_DATA["w/o IRD"].overall_acc
    mlr_contribution = full_performance.overall_acc - ABLATION_DATA["w/o MLR"].overall_acc
    cv_contribution = full_performance.overall_acc - ABLATION_DATA["w/o CV"].overall_acc
    
    return {
        "IRD_contribution": ird_contribution,
        "MLR_contribution": mlr_contribution, 
        "CV_contribution": cv_contribution,
        "most_important": max([
            ("IRD", ird_contribution),
            ("MLR", mlr_contribution),
            ("CV", cv_contribution)
        ], key=lambda x: x[1])
    }


def export_performance_data(filename: str = "performance_analysis.json") -> None:
    """Export all performance data to JSON file."""
    export_data = {
        "overall_performance": {name: {
            "method_name": perf.method_name,
            "math23k": perf.math23k,
            "gsm8k": perf.gsm8k,
            "mawps": perf.mawps,
            "mathqa": perf.mathqa,
            "math": perf.math,
            "svamp": perf.svamp,
            "asdiv": perf.asdiv,
            "dir_test": perf.dir_test,
            "average": calculate_average_performance(name)
        } for name, perf in PERFORMANCE_DATA.items()},
        
        "complexity_performance": {name: {
            "method_name": perf.method_name,
            "l0_explicit": perf.l0_explicit,
            "l1_shallow": perf.l1_shallow,
            "l2_medium": perf.l2_medium,
            "l3_deep": perf.l3_deep,
            "robustness_score": perf.robustness_score
        } for name, perf in COMPLEXITY_PERFORMANCE.items()},
        
        "efficiency_metrics": {name: {
            "method_name": metrics.method_name,
            "avg_runtime": metrics.avg_runtime,
            "memory_mb": metrics.memory_mb,
            "l2_runtime": metrics.l2_runtime,
            "l3_runtime": metrics.l3_runtime,
            "efficiency_score": metrics.efficiency_score
        } for name, metrics in EFFICIENCY_DATA.items()},
        
        "ablation_results": {config: {
            "configuration": result.configuration,
            "overall_acc": result.overall_acc,
            "l2_acc": result.l2_acc,
            "l3_acc": result.l3_acc,
            "relation_f1": result.relation_f1,
            "chain_quality": result.chain_quality,
            "efficiency": result.efficiency
        } for config, result in ABLATION_DATA.items()},
        
        "component_interaction": {combo: {
            "combination": inter.combination,
            "overall_acc": inter.overall_acc,
            "relation_discovery": inter.relation_discovery,
            "reasoning_quality": inter.reasoning_quality,
            "error_rate": inter.error_rate,
            "synergy_score": inter.synergy_score
        } for combo, inter in COMPONENT_INTERACTION.items()},
        
        "relation_discovery": {name: {
            "method_name": metrics.method_name,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "semantic_acc": metrics.semantic_acc,
            "l2_f1": metrics.l2_f1,
            "l3_f1": metrics.l3_f1,
            "avg_relations": metrics.avg_relations
        } for name, metrics in RELATION_DISCOVERY_DATA.items()},
        
        "reasoning_chain": {name: {
            "method_name": metrics.method_name,
            "logical_correctness": metrics.logical_correctness,
            "completeness": metrics.completeness,
            "coherence": metrics.coherence,
            "efficiency": metrics.efficiency,
            "verifiability": metrics.verifiability,
            "overall_score": metrics.overall_score
        } for name, metrics in REASONING_CHAIN_DATA.items()},
        
        "analysis_summary": {
            "component_contributions": analyze_component_contribution(),
            "efficiency_ranking": get_efficiency_ranking(),
            "robustness_ranking": get_robustness_ranking(),
            "best_performers": {
                "math23k": get_best_performing_method("math23k"),
                "gsm8k": get_best_performing_method("gsm8k"),
                "mawps": get_best_performing_method("mawps"),
                "mathqa": get_best_performing_method("mathqa"),
                "math": get_best_performing_method("math"),
                "svamp": get_best_performing_method("svamp"),
                "asdiv": get_best_performing_method("asdiv"),
                "dir_test": get_best_performing_method("dir_test")
            }
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"Performance analysis data exported to {filename}")


if __name__ == "__main__":
    # Example usage
    print("=== Performance Analysis ===")
    print(f"Available methods: {get_all_methods()}")
    print(f"Best performer on MATH dataset: {get_best_performing_method('math')}")
    print(f"COT-DIR average performance: {calculate_average_performance('COT-DIR'):.1f}")
    
    # Export data
    export_performance_data() 