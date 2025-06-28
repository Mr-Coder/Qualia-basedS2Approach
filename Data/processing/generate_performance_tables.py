#!/usr/bin/env python3
"""
Generate Performance Tables

This script generates performance analysis table files in multiple formats
based on the performance analysis data.
"""

import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.performance_analysis import (ABLATION_DATA,
                                           COMPLEXITY_PERFORMANCE,
                                           COMPONENT_INTERACTION,
                                           EFFICIENCY_DATA, PERFORMANCE_DATA,
                                           REASONING_CHAIN_DATA,
                                           RELATION_DISCOVERY_DATA,
                                           export_performance_data)


def export_performance_comparison_csv():
    """Export Table 4: Overall Performance Comparison Across Datasets to CSV."""
    filename = "table4_performance_comparison.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Method', 'Math23K', 'GSM8K', 'MAWPS', 'MathQA', 'MATH', 'SVAMP', 'ASDiv', 'DIR-Test'])
        
        # Data rows
        for method_name, perf in PERFORMANCE_DATA.items():
            writer.writerow([
                method_name, perf.math23k, perf.gsm8k, perf.mawps,
                perf.mathqa, perf.math, perf.svamp, perf.asdiv, perf.dir_test
            ])
    
    print(f"Table 4 exported to {filename}")


def export_complexity_performance_csv():
    """Export Table 5: Performance Analysis by Problem Complexity Level to CSV."""
    filename = "table5_complexity_performance.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Method', 'L0 (Explicit)', 'L1 (Shallow)', 'L2 (Medium)', 'L3 (Deep)', 'Robustness Score'])
        
        # Data rows
        for method_name, perf in COMPLEXITY_PERFORMANCE.items():
            writer.writerow([
                method_name, perf.l0_explicit, perf.l1_shallow,
                perf.l2_medium, perf.l3_deep, perf.robustness_score
            ])
    
    print(f"Table 5 exported to {filename}")


def export_relation_discovery_csv():
    """Export Table 6: Implicit Relation Discovery Quality Assessment to CSV."""
    filename = "table6_relation_discovery.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Method', 'Precision', 'Recall', 'F1-Score', 'Semantic Acc.', 'L2 F1', 'L3 F1', 'Avg. Relations'])
        
        # Data rows
        for method_name, metrics in RELATION_DISCOVERY_DATA.items():
            writer.writerow([
                method_name, metrics.precision, metrics.recall, metrics.f1_score,
                metrics.semantic_acc, metrics.l2_f1, metrics.l3_f1, metrics.avg_relations
            ])
    
    print(f"Table 6 exported to {filename}")


def export_reasoning_chain_csv():
    """Export Table 7: Reasoning Chain Quality Assessment to CSV."""
    filename = "table7_reasoning_chain.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Method', 'Logical Correctness', 'Completeness', 'Coherence', 'Efficiency', 'Verifiability', 'Overall Score'])
        
        # Data rows
        for method_name, metrics in REASONING_CHAIN_DATA.items():
            writer.writerow([
                method_name, metrics.logical_correctness, metrics.completeness,
                metrics.coherence, metrics.efficiency, metrics.verifiability, metrics.overall_score
            ])
    
    print(f"Table 7 exported to {filename}")


def export_ablation_study_csv():
    """Export Table 8: Ablation Study - Individual Component Contributions to CSV."""
    filename = "table8_ablation_study.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Configuration', 'Overall Acc.', 'L2 Acc.', 'L3 Acc.', 'Relation F1', 'Chain Quality', 'Efficiency'])
        
        # Data rows
        for config, result in ABLATION_DATA.items():
            writer.writerow([
                config, result.overall_acc, result.l2_acc, result.l3_acc,
                result.relation_f1, result.chain_quality, result.efficiency
            ])
    
    print(f"Table 8 exported to {filename}")


def export_component_interaction_csv():
    """Export Table 9: Component Interaction Analysis to CSV."""
    filename = "table9_component_interaction.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Component Combination', 'Overall Acc.', 'Relation Discovery', 'Reasoning Quality', 'Error Rate', 'Synergy Score'])
        
        # Data rows
        for combo, interaction in COMPONENT_INTERACTION.items():
            writer.writerow([
                combo, interaction.overall_acc, interaction.relation_discovery,
                interaction.reasoning_quality, interaction.error_rate, interaction.synergy_score
            ])
    
    print(f"Table 9 exported to {filename}")


def export_efficiency_analysis_csv():
    """Export Table 10: Computational Efficiency Analysis to CSV."""
    filename = "table10_efficiency_analysis.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Method', 'Avg. Runtime (s)', 'Memory (MB)', 'L2 Runtime (s)', 'L3 Runtime (s)', 'Efficiency Score'])
        
        # Data rows
        for method_name, metrics in EFFICIENCY_DATA.items():
            writer.writerow([
                method_name, metrics.avg_runtime, metrics.memory_mb,
                metrics.l2_runtime, metrics.l3_runtime, metrics.efficiency_score
            ])
    
    print(f"Table 10 exported to {filename}")


def export_all_tables_markdown():
    """Export all tables in markdown format."""
    filename = "performance_tables.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Performance Analysis Tables\n\n")
        
        # Table 4: Overall Performance Comparison
        f.write("## Table 4: Overall Performance Comparison Across Datasets\n\n")
        f.write("| Method | Math23K | GSM8K | MAWPS | MathQA | MATH | SVAMP | ASDiv | DIR-Test |\n")
        f.write("|--------|---------|-------|-------|--------|------|-------|-------|----------|\n")
        
        for method_name, perf in PERFORMANCE_DATA.items():
            f.write(f"| {method_name} | {perf.math23k} | {perf.gsm8k} | {perf.mawps} | "
                   f"{perf.mathqa} | {perf.math} | {perf.svamp} | {perf.asdiv} | {perf.dir_test} |\n")
        
        # Table 5: Complexity Performance
        f.write("\n## Table 5: Performance Analysis by Problem Complexity Level\n\n")
        f.write("| Method | L0 (Explicit) | L1 (Shallow) | L2 (Medium) | L3 (Deep) | Robustness Score |\n")
        f.write("|--------|---------------|--------------|-------------|-----------|------------------|\n")
        
        for method_name, perf in COMPLEXITY_PERFORMANCE.items():
            f.write(f"| {method_name} | {perf.l0_explicit} | {perf.l1_shallow} | {perf.l2_medium} | "
                   f"{perf.l3_deep} | {perf.robustness_score} |\n")
        
        # Table 6: Relation Discovery
        f.write("\n## Table 6: Implicit Relation Discovery Quality Assessment\n\n")
        f.write("| Method | Precision | Recall | F1-Score | Semantic Acc. | L2 F1 | L3 F1 | Avg. Relations |\n")
        f.write("|--------|-----------|--------|----------|---------------|-------|-------|----------------|\n")
        
        for method_name, metrics in RELATION_DISCOVERY_DATA.items():
            f.write(f"| {method_name} | {metrics.precision:.2f} | {metrics.recall:.2f} | {metrics.f1_score:.2f} | "
                   f"{metrics.semantic_acc:.2f} | {metrics.l2_f1:.2f} | {metrics.l3_f1:.2f} | {metrics.avg_relations} |\n")
        
        # Table 7: Reasoning Chain Quality
        f.write("\n## Table 7: Reasoning Chain Quality Assessment\n\n")
        f.write("| Method | Logical Correctness | Completeness | Coherence | Efficiency | Verifiability | Overall Score |\n")
        f.write("|--------|---------------------|--------------|-----------|------------|---------------|---------------|\n")
        
        for method_name, metrics in REASONING_CHAIN_DATA.items():
            f.write(f"| {method_name} | {metrics.logical_correctness:.2f} | {metrics.completeness:.2f} | "
                   f"{metrics.coherence:.2f} | {metrics.efficiency:.2f} | {metrics.verifiability:.2f} | "
                   f"{metrics.overall_score:.2f} |\n")
        
        # Table 8: Ablation Study
        f.write("\n## Table 8: Ablation Study - Individual Component Contributions\n\n")
        f.write("| Configuration | Overall Acc. | L2 Acc. | L3 Acc. | Relation F1 | Chain Quality | Efficiency |\n")
        f.write("|---------------|--------------|---------|---------|-------------|---------------|------------|\n")
        
        for config, result in ABLATION_DATA.items():
            f.write(f"| {config} | {result.overall_acc} | {result.l2_acc} | {result.l3_acc} | "
                   f"{result.relation_f1:.2f} | {result.chain_quality:.2f} | {result.efficiency} |\n")
        
        # Table 9: Component Interaction
        f.write("\n## Table 9: Component Interaction Analysis\n\n")
        f.write("| Component Combination | Overall Acc. | Relation Discovery | Reasoning Quality | Error Rate | Synergy Score |\n")
        f.write("|-----------------------|--------------|--------------------|--------------------|------------|---------------|\n")
        
        for combo, interaction in COMPONENT_INTERACTION.items():
            f.write(f"| {combo} | {interaction.overall_acc} | {interaction.relation_discovery:.2f} | "
                   f"{interaction.reasoning_quality:.2f} | {interaction.error_rate}% | {interaction.synergy_score:.2f} |\n")
        
        # Table 10: Efficiency Analysis
        f.write("\n## Table 10: Computational Efficiency Analysis\n\n")
        f.write("| Method | Avg. Runtime (s) | Memory (MB) | L2 Runtime (s) | L3 Runtime (s) | Efficiency Score |\n")
        f.write("|--------|------------------|-------------|----------------|----------------|------------------|\n")
        
        for method_name, metrics in EFFICIENCY_DATA.items():
            f.write(f"| {method_name} | {metrics.avg_runtime} | {metrics.memory_mb} | "
                   f"{metrics.l2_runtime} | {metrics.l3_runtime} | {metrics.efficiency_score} |\n")
    
    print(f"All tables exported to {filename}")


def main():
    """Generate all performance table files."""
    print("Generating performance analysis table files...")
    print("=" * 50)
    
    # CSV exports for each table
    export_performance_comparison_csv()
    export_complexity_performance_csv()
    export_relation_discovery_csv()
    export_reasoning_chain_csv()
    export_ablation_study_csv()
    export_component_interaction_csv()
    export_efficiency_analysis_csv()
    
    # Markdown export with all tables
    export_all_tables_markdown()
    
    # Comprehensive JSON export
    export_performance_data("all_performance_tables.json")
    
    print("\n" + "=" * 50)
    print("All performance table files generated successfully!")
    print("\nGenerated files:")
    print("- table4_performance_comparison.csv")
    print("- table5_complexity_performance.csv")
    print("- table6_relation_discovery.csv")
    print("- table7_reasoning_chain.csv")
    print("- table8_ablation_study.csv")
    print("- table9_component_interaction.csv")
    print("- table10_efficiency_analysis.csv")
    print("- performance_tables.md (All tables in markdown)")
    print("- all_performance_tables.json (Comprehensive JSON)")


if __name__ == "__main__":
    main() 