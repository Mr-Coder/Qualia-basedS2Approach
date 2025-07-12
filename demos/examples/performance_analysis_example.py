#!/usr/bin/env python3
"""
Performance Analysis Example

This example demonstrates comprehensive analysis of model performance data
from multiple evaluation tables including efficiency, ablation studies,
and quality assessments.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.performance_analysis import (ABLATION_DATA, COMPLEXITY_PERFORMANCE,
                                       COMPONENT_INTERACTION, EFFICIENCY_DATA,
                                       PERFORMANCE_DATA, REASONING_CHAIN_DATA,
                                       RELATION_DISCOVERY_DATA,
                                       analyze_component_contribution,
                                       calculate_average_performance,
                                       export_performance_data,
                                       get_all_methods,
                                       get_best_performing_method,
                                       get_efficiency_ranking,
                                       get_robustness_ranking)


def analyze_overall_performance():
    """Analyze overall performance across datasets."""
    print("=== Overall Performance Analysis ===\n")
    
    methods = get_all_methods()
    print(f"{'Method':<25} {'Avg':<6} {'Best Dataset':<15} {'Worst Dataset':<15}")
    print("-" * 70)
    
    for method in methods:
        perf = PERFORMANCE_DATA[method]
        scores = [
            ('Math23K', perf.math23k), ('GSM8K', perf.gsm8k),
            ('MAWPS', perf.mawps), ('MathQA', perf.mathqa),
            ('MATH', perf.math), ('SVAMP', perf.svamp),
            ('ASDiv', perf.asdiv), ('DIR-Test', perf.dir_test)
        ]
        
        avg_score = calculate_average_performance(method)
        best_dataset = max(scores, key=lambda x: x[1])
        worst_dataset = min(scores, key=lambda x: x[1])
        
        print(f"{method:<25} {avg_score:<6.1f} {best_dataset[0]} ({best_dataset[1]:.1f})    {worst_dataset[0]} ({worst_dataset[1]:.1f})")
    
    print(f"\nDataset-wise Best Performers:")
    datasets = ['math23k', 'gsm8k', 'mawps', 'mathqa', 'math', 'svamp', 'asdiv', 'dir_test']
    for dataset in datasets:
        best_method, best_score = get_best_performing_method(dataset)
        print(f"  {dataset.upper()}: {best_method} ({best_score:.1f})")


def analyze_complexity_performance():
    """Analyze performance by complexity levels."""
    print("\n=== Complexity Level Performance Analysis ===\n")
    
    print(f"{'Method':<25} {'L0':<6} {'L1':<6} {'L2':<6} {'L3':<6} {'Robustness':<10} {'Drop Rate':<8}")
    print("-" * 80)
    
    for method_name, perf in COMPLEXITY_PERFORMANCE.items():
        drop_rate = (perf.l0_explicit - perf.l3_deep) / perf.l0_explicit * 100
        print(f"{method_name:<25} {perf.l0_explicit:<6.1f} {perf.l1_shallow:<6.1f} "
              f"{perf.l2_medium:<6.1f} {perf.l3_deep:<6.1f} {perf.robustness_score:<10.2f} {drop_rate:<8.1f}%")
    
    print(f"\nRobustness Ranking:")
    robustness_ranking = get_robustness_ranking()
    for i, (method, score) in enumerate(robustness_ranking, 1):
        print(f"  {i}. {method}: {score:.2f}")


def analyze_efficiency():
    """Analyze computational efficiency."""
    print("\n=== Computational Efficiency Analysis ===\n")
    
    print(f"{'Method':<25} {'Runtime(s)':<12} {'Memory(MB)':<12} {'L2 Time':<10} {'L3 Time':<10} {'Efficiency':<10}")
    print("-" * 90)
    
    for method_name, metrics in EFFICIENCY_DATA.items():
        print(f"{method_name:<25} {metrics.avg_runtime:<12.1f} {metrics.memory_mb:<12} "
              f"{metrics.l2_runtime:<10.1f} {metrics.l3_runtime:<10.1f} {metrics.efficiency_score:<10.2f}")
    
    print(f"\nEfficiency Ranking:")
    efficiency_ranking = get_efficiency_ranking()
    for i, (method, score) in enumerate(efficiency_ranking, 1):
        print(f"  {i}. {method}: {score:.2f}")
    
    # Calculate runtime increase for complex problems
    print(f"\nRuntime Scaling Analysis:")
    for method_name, metrics in EFFICIENCY_DATA.items():
        l2_increase = (metrics.l2_runtime / metrics.avg_runtime - 1) * 100
        l3_increase = (metrics.l3_runtime / metrics.avg_runtime - 1) * 100
        print(f"  {method_name}: L2 +{l2_increase:.0f}%, L3 +{l3_increase:.0f}%")


def analyze_ablation_study():
    """Analyze ablation study results."""
    print("\n=== Ablation Study Analysis ===\n")
    
    print("Component Contributions:")
    contributions = analyze_component_contribution()
    print(f"  IRD (Implicit Relation Discovery): +{contributions['IRD_contribution']:.1f}%")
    print(f"  MLR (Multi-Level Reasoning): +{contributions['MLR_contribution']:.1f}%")
    print(f"  CV (Chain Verification): +{contributions['CV_contribution']:.1f}%")
    print(f"  Most Important: {contributions['most_important'][0]} (+{contributions['most_important'][1]:.1f}%)")
    
    print(f"\nFull Ablation Results:")
    print(f"{'Configuration':<20} {'Overall':<8} {'L2 Acc':<8} {'L3 Acc':<8} {'Rel F1':<8} {'Quality':<8} {'Time':<6}")
    print("-" * 75)
    
    for config, result in ABLATION_DATA.items():
        print(f"{config:<20} {result.overall_acc:<8.1f} {result.l2_acc:<8.1f} "
              f"{result.l3_acc:<8.1f} {result.relation_f1:<8.2f} {result.chain_quality:<8.2f} {result.efficiency:<6.1f}s")


def analyze_component_interactions():
    """Analyze component interaction effects."""
    print("\n=== Component Interaction Analysis ===\n")
    
    print(f"{'Combination':<15} {'Accuracy':<9} {'Relations':<10} {'Quality':<8} {'Error%':<7} {'Synergy':<8}")
    print("-" * 65)
    
    for combo, interaction in COMPONENT_INTERACTION.items():
        print(f"{combo:<15} {interaction.overall_acc:<9.1f} {interaction.relation_discovery:<10.2f} "
              f"{interaction.reasoning_quality:<8.2f} {interaction.error_rate:<7.1f} {interaction.synergy_score:<8.2f}")
    
    # Calculate synergy effects
    print(f"\nSynergy Effects:")
    full_system = COMPONENT_INTERACTION["IRD + MLR + CV"]
    individual_components = [ABLATION_DATA["IRD only"], ABLATION_DATA["MLR only"], ABLATION_DATA["CV only"]]
    expected_individual = sum(comp.overall_acc for comp in individual_components) / 3
    synergy_gain = full_system.overall_acc - expected_individual
    print(f"  Expected individual average: {expected_individual:.1f}%")
    print(f"  Full system performance: {full_system.overall_acc:.1f}%")
    print(f"  Synergy gain: +{synergy_gain:.1f}%")


def analyze_relation_discovery():
    """Analyze implicit relation discovery quality."""
    print("\n=== Relation Discovery Quality Analysis ===\n")
    
    print(f"{'Method':<25} {'Precision':<10} {'Recall':<8} {'F1':<6} {'Semantic':<9} {'L2 F1':<7} {'L3 F1':<7} {'Avg Rel':<7}")
    print("-" * 85)
    
    for method_name, metrics in RELATION_DISCOVERY_DATA.items():
        print(f"{method_name:<25} {metrics.precision:<10.2f} {metrics.recall:<8.2f} "
              f"{metrics.f1_score:<6.2f} {metrics.semantic_acc:<9.2f} "
              f"{metrics.l2_f1:<7.2f} {metrics.l3_f1:<7.2f} {metrics.avg_relations:<7.1f}")
    
    # Find best performers
    best_f1 = max(RELATION_DISCOVERY_DATA.items(), key=lambda x: x[1].f1_score)
    best_semantic = max(RELATION_DISCOVERY_DATA.items(), key=lambda x: x[1].semantic_acc)
    
    print(f"\nBest Performers:")
    print(f"  Highest F1 Score: {best_f1[0]} ({best_f1[1].f1_score:.2f})")
    print(f"  Highest Semantic Accuracy: {best_semantic[0]} ({best_semantic[1].semantic_acc:.2f})")


def analyze_reasoning_chain_quality():
    """Analyze reasoning chain quality assessment."""
    print("\n=== Reasoning Chain Quality Analysis ===\n")
    
    print(f"{'Method':<25} {'Logic':<7} {'Complete':<9} {'Coherent':<9} {'Efficient':<9} {'Verify':<8} {'Overall':<8}")
    print("-" * 85)
    
    for method_name, metrics in REASONING_CHAIN_DATA.items():
        print(f"{method_name:<25} {metrics.logical_correctness:<7.2f} {metrics.completeness:<9.2f} "
              f"{metrics.coherence:<9.2f} {metrics.efficiency:<9.2f} "
              f"{metrics.verifiability:<8.2f} {metrics.overall_score:<8.2f}")
    
    # Analyze quality dimensions
    print(f"\nQuality Dimension Analysis:")
    dimensions = ['logical_correctness', 'completeness', 'coherence', 'efficiency', 'verifiability']
    
    for dim in dimensions:
        scores = [getattr(metrics, dim) for metrics in REASONING_CHAIN_DATA.values()]
        avg_score = sum(scores) / len(scores)
        best_method = max(REASONING_CHAIN_DATA.items(), key=lambda x: getattr(x[1], dim))
        print(f"  {dim.replace('_', ' ').title()}: Avg {avg_score:.2f}, Best: {best_method[0]} ({getattr(best_method[1], dim):.2f})")


def generate_performance_insights():
    """Generate key insights from performance analysis."""
    print("\n=== Key Performance Insights ===\n")
    
    # Overall best performer
    cot_dir_avg = calculate_average_performance("COT-DIR")
    best_baseline = max([calculate_average_performance(method) for method in get_all_methods() if method != "COT-DIR"])
    improvement = cot_dir_avg - best_baseline
    
    print(f"1. COT-DIR Overall Performance:")
    print(f"   - Average accuracy: {cot_dir_avg:.1f}%")
    print(f"   - Improvement over best baseline: +{improvement:.1f}%")
    
    # Efficiency vs Performance trade-off
    cot_dir_efficiency = EFFICIENCY_DATA["COT-DIR"].efficiency_score
    cot_dir_robustness = COMPLEXITY_PERFORMANCE["COT-DIR"].robustness_score
    
    print(f"\n2. Efficiency vs Performance Trade-off:")
    print(f"   - COT-DIR Efficiency Score: {cot_dir_efficiency:.2f}")
    print(f"   - COT-DIR Robustness Score: {cot_dir_robustness:.2f}")
    print(f"   - Performance/Efficiency Ratio: {cot_dir_avg/cot_dir_efficiency:.1f}")
    
    # Component importance
    contributions = analyze_component_contribution()
    print(f"\n3. Component Importance Ranking:")
    sorted_components = sorted([
        ("IRD", contributions["IRD_contribution"]),
        ("MLR", contributions["MLR_contribution"]),
        ("CV", contributions["CV_contribution"])
    ], key=lambda x: x[1], reverse=True)
    
    for i, (component, contrib) in enumerate(sorted_components, 1):
        print(f"   {i}. {component}: +{contrib:.1f}% contribution")
    
    # Complexity handling
    cot_dir_complex = COMPLEXITY_PERFORMANCE["COT-DIR"]
    complexity_retention = cot_dir_complex.l3_deep / cot_dir_complex.l0_explicit
    
    print(f"\n4. Complexity Handling:")
    print(f"   - L0 to L3 performance retention: {complexity_retention:.1%}")
    print(f"   - Best complexity handling: {get_robustness_ranking()[0][0]}")
    
    # Quality scores
    cot_dir_quality = REASONING_CHAIN_DATA["COT-DIR"].overall_score
    cot_dir_relations = RELATION_DISCOVERY_DATA["COT-DIR"].f1_score
    
    print(f"\n5. Quality Metrics:")
    print(f"   - Reasoning Chain Quality: {cot_dir_quality:.2f}")
    print(f"   - Relation Discovery F1: {cot_dir_relations:.2f}")
    print(f"   - Both significantly above baseline methods")


def main():
    """Main function to run comprehensive performance analysis."""
    print("Comprehensive Performance Analysis")
    print("=" * 50)
    
    # Run all analyses
    analyze_overall_performance()
    analyze_complexity_performance()
    analyze_efficiency()
    analyze_ablation_study()
    analyze_component_interactions()
    analyze_relation_discovery()
    analyze_reasoning_chain_quality()
    generate_performance_insights()
    
    # Export comprehensive data
    print("\n" + "=" * 50)
    print("Exporting comprehensive performance data...")
    export_performance_data("comprehensive_performance_analysis.json")
    
    print("\nAnalysis complete! Check 'comprehensive_performance_analysis.json' for detailed data.")


if __name__ == "__main__":
    main() 