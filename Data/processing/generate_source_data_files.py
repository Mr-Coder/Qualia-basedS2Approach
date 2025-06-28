#!/usr/bin/env python3
"""
Generate Source Data Files from src/data Module

This script extracts data from the src/data module and generates source data files
for all tables in multiple formats (JSON, CSV, Markdown).
"""

import csv
import json
import os
import sys
from typing import Any, Dict, List

# Add src to path to import data modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset_characteristics import (DATASET_CHARACTERISTICS,
                                          get_all_datasets,
                                          get_dataset_statistics)
from data.performance_analysis import (ABLATION_DATA, COMPLEXITY_PERFORMANCE,
                                       COMPONENT_INTERACTION, EFFICIENCY_DATA,
                                       PERFORMANCE_DATA, REASONING_CHAIN_DATA,
                                       RELATION_DISCOVERY_DATA,
                                       calculate_average_performance,
                                       get_all_methods)


def generate_table3_source_data():
    """Generate Table 3: Dataset Characteristics source data."""
    print("Generating Table 3: Dataset Characteristics source data...")
    
    # Generate JSON
    datasets_json = {}
    for name, info in DATASET_CHARACTERISTICS.items():
        datasets_json[name] = {
            'name': info.name,
            'size': info.size,
            'language': info.language,
            'domain': info.domain,
            'complexity_distribution': {
                'L0': info.l0_percent,
                'L1': info.l1_percent,
                'L2': info.l2_percent,
                'L3': info.l3_percent
            },
            'dir_score': info.dir_score
        }
    
    with open('table3_dataset_characteristics.json', 'w', encoding='utf-8') as f:
        json.dump(datasets_json, f, indent=2, ensure_ascii=False)
    
    # Generate CSV
    with open('table3_dataset_characteristics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Size', 'Language', 'Domain', 'L0 (%)', 'L1 (%)', 'L2 (%)', 'L3 (%)', 'DIR Score'])
        
        for info in DATASET_CHARACTERISTICS.values():
            writer.writerow([
                info.name, info.size, info.language, info.domain,
                info.l0_percent, info.l1_percent, info.l2_percent, info.l3_percent,
                info.dir_score
            ])
    
    # Generate Markdown
    with open('table3_dataset_characteristics.md', 'w', encoding='utf-8') as f:
        f.write("# Table 3: Dataset Characteristics with DIR-MWP Complexity Distribution\n\n")
        f.write("| Dataset | Size | Language | Domain | L0 (%) | L1 (%) | L2 (%) | L3 (%) | DIR Score |\n")
        f.write("|---------|------|----------|--------|--------|--------|--------|--------|-----------|\n")
        
        for info in DATASET_CHARACTERISTICS.values():
            f.write(f"| {info.name} | {info.size:,} | {info.language} | {info.domain} | "
                   f"{info.l0_percent} | {info.l1_percent} | {info.l2_percent} | {info.l3_percent} | "
                   f"{info.dir_score} |\n")
    
    print("✅ Table 3 source data files generated successfully")


def generate_table4_source_data():
    """Generate Table 4: Performance Comparison source data."""
    print("Generating Table 4: Performance Comparison source data...")
    
    # Generate JSON
    performance_json = {}
    for method, perf in PERFORMANCE_DATA.items():
        performance_json[method] = {
            'method_name': perf.method_name,
            'datasets': {
                'Math23K': perf.math23k,
                'GSM8K': perf.gsm8k,
                'MAWPS': perf.mawps,
                'MathQA': perf.mathqa,
                'MATH': perf.math,
                'SVAMP': perf.svamp,
                'ASDiv': perf.asdiv,
                'DIR-Test': perf.dir_test
            },
            'average': calculate_average_performance(method)
        }
    
    with open('table4_performance_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(performance_json, f, indent=2, ensure_ascii=False)
    
    # Generate CSV
    with open('table4_performance_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Math23K', 'GSM8K', 'MAWPS', 'MathQA', 'MATH', 'SVAMP', 'ASDiv', 'DIR-Test', 'Average'])
        
        for method, perf in PERFORMANCE_DATA.items():
            avg = calculate_average_performance(method)
            writer.writerow([
                perf.method_name, perf.math23k, perf.gsm8k, perf.mawps, perf.mathqa,
                perf.math, perf.svamp, perf.asdiv, perf.dir_test, f"{avg:.1f}"
            ])
    
    # Generate Markdown
    with open('table4_performance_comparison.md', 'w', encoding='utf-8') as f:
        f.write("# Table 4: Overall Performance Comparison Across Datasets\n\n")
        f.write("| Method | Math23K | GSM8K | MAWPS | MathQA | MATH | SVAMP | ASDiv | DIR-Test | Average |\n")
        f.write("|--------|---------|-------|-------|--------|------|-------|-------|----------|----------|\n")
        
        for method, perf in PERFORMANCE_DATA.items():
            avg = calculate_average_performance(method)
            f.write(f"| {perf.method_name} | {perf.math23k} | {perf.gsm8k} | {perf.mawps} | "
                   f"{perf.mathqa} | {perf.math} | {perf.svamp} | {perf.asdiv} | {perf.dir_test} | "
                   f"{avg:.1f} |\n")
    
    print("✅ Table 4 source data files generated successfully")


def generate_table5_source_data():
    """Generate Table 5: Complexity Performance source data."""
    print("Generating Table 5: Complexity Performance source data...")
    
    # Generate JSON
    complexity_json = {}
    for method, perf in COMPLEXITY_PERFORMANCE.items():
        complexity_json[method] = {
            'method_name': perf.method_name,
            'performance_by_level': {
                'L0_explicit': perf.l0_explicit,
                'L1_shallow': perf.l1_shallow,
                'L2_medium': perf.l2_medium,
                'L3_deep': perf.l3_deep
            },
            'robustness_score': perf.robustness_score,
            'complexity_drop': round(perf.l0_explicit - perf.l3_deep, 1)
        }
    
    with open('table5_complexity_performance.json', 'w', encoding='utf-8') as f:
        json.dump(complexity_json, f, indent=2, ensure_ascii=False)
    
    # Generate CSV
    with open('table5_complexity_performance.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'L0 (%)', 'L1 (%)', 'L2 (%)', 'L3 (%)', 'Robustness', 'Complexity Drop'])
        
        for method, perf in COMPLEXITY_PERFORMANCE.items():
            drop = round(perf.l0_explicit - perf.l3_deep, 1)
            writer.writerow([
                perf.method_name, perf.l0_explicit, perf.l1_shallow, perf.l2_medium,
                perf.l3_deep, perf.robustness_score, drop
            ])
    
    # Generate Markdown
    with open('table5_complexity_performance.md', 'w', encoding='utf-8') as f:
        f.write("# Table 5: Performance Analysis by Problem Complexity Level\n\n")
        f.write("| Method | L0 (%) | L1 (%) | L2 (%) | L3 (%) | Robustness | Complexity Drop |\n")
        f.write("|--------|--------|--------|--------|--------|------------|------------------|\n")
        
        for method, perf in COMPLEXITY_PERFORMANCE.items():
            drop = round(perf.l0_explicit - perf.l3_deep, 1)
            f.write(f"| {perf.method_name} | {perf.l0_explicit} | {perf.l1_shallow} | "
                   f"{perf.l2_medium} | {perf.l3_deep} | {perf.robustness_score} | {drop} |\n")
    
    print("✅ Table 5 source data files generated successfully")


def generate_table6_source_data():
    """Generate Table 6: Relation Discovery source data."""
    print("Generating Table 6: Relation Discovery source data...")
    
    # Generate JSON
    relation_json = {}
    for method, metrics in RELATION_DISCOVERY_DATA.items():
        relation_json[method] = {
            'method_name': metrics.method_name,
            'overall_metrics': {
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'semantic_accuracy': metrics.semantic_acc
            },
            'complexity_specific': {
                'L2_f1': metrics.l2_f1,
                'L3_f1': metrics.l3_f1
            },
            'avg_relations_discovered': metrics.avg_relations
        }
    
    with open('table6_relation_discovery.json', 'w', encoding='utf-8') as f:
        json.dump(relation_json, f, indent=2, ensure_ascii=False)
    
    # Generate CSV
    with open('table6_relation_discovery.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Precision', 'Recall', 'F1-Score', 'Semantic Acc', 'L2 F1', 'L3 F1', 'Avg Relations'])
        
        for method, metrics in RELATION_DISCOVERY_DATA.items():
            writer.writerow([
                metrics.method_name, metrics.precision, metrics.recall, metrics.f1_score,
                metrics.semantic_acc, metrics.l2_f1, metrics.l3_f1, metrics.avg_relations
            ])
    
    print("✅ Table 6 source data files generated successfully")


def generate_table7_source_data():
    """Generate Table 7: Reasoning Chain source data."""
    print("Generating Table 7: Reasoning Chain source data...")
    
    # Generate JSON
    reasoning_json = {}
    for method, metrics in REASONING_CHAIN_DATA.items():
        reasoning_json[method] = {
            'method_name': metrics.method_name,
            'quality_dimensions': {
                'logical_correctness': metrics.logical_correctness,
                'completeness': metrics.completeness,
                'coherence': metrics.coherence,
                'efficiency': metrics.efficiency,
                'verifiability': metrics.verifiability
            },
            'overall_score': metrics.overall_score
        }
    
    with open('table7_reasoning_chain.json', 'w', encoding='utf-8') as f:
        json.dump(reasoning_json, f, indent=2, ensure_ascii=False)
    
    # Generate CSV
    with open('table7_reasoning_chain.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Logical', 'Completeness', 'Coherence', 'Efficiency', 'Verifiability', 'Overall'])
        
        for method, metrics in REASONING_CHAIN_DATA.items():
            writer.writerow([
                metrics.method_name, metrics.logical_correctness, metrics.completeness,
                metrics.coherence, metrics.efficiency, metrics.verifiability, metrics.overall_score
            ])
    
    print("✅ Table 7 source data files generated successfully")


def generate_table8_source_data():
    """Generate Table 8: Ablation Study source data."""
    print("Generating Table 8: Ablation Study source data...")
    
    # Generate JSON
    ablation_json = {}
    for config, results in ABLATION_DATA.items():
        ablation_json[config] = {
            'configuration': results.configuration,
            'performance_metrics': {
                'overall_accuracy': results.overall_acc,
                'L2_accuracy': results.l2_acc,
                'L3_accuracy': results.l3_acc
            },
            'quality_metrics': {
                'relation_f1': results.relation_f1,
                'chain_quality': results.chain_quality
            },
            'efficiency': results.efficiency
        }
    
    with open('table8_ablation_study.json', 'w', encoding='utf-8') as f:
        json.dump(ablation_json, f, indent=2, ensure_ascii=False)
    
    # Generate CSV
    with open('table8_ablation_study.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuration', 'Overall Acc', 'L2 Acc', 'L3 Acc', 'Relation F1', 'Chain Quality', 'Efficiency'])
        
        for config, results in ABLATION_DATA.items():
            writer.writerow([
                results.configuration, results.overall_acc, results.l2_acc, results.l3_acc,
                results.relation_f1, results.chain_quality, results.efficiency
            ])
    
    print("✅ Table 8 source data files generated successfully")


def generate_table9_source_data():
    """Generate Table 9: Component Interaction source data."""
    print("Generating Table 9: Component Interaction source data...")
    
    # Generate JSON
    interaction_json = {}
    for combo, inter in COMPONENT_INTERACTION.items():
        interaction_json[combo] = {
            'combination': inter.combination,
            'performance_metrics': {
                'overall_accuracy': inter.overall_acc,
                'relation_discovery': inter.relation_discovery,
                'reasoning_quality': inter.reasoning_quality
            },
            'error_rate': inter.error_rate,
            'synergy_score': inter.synergy_score
        }
    
    with open('table9_component_interaction.json', 'w', encoding='utf-8') as f:
        json.dump(interaction_json, f, indent=2, ensure_ascii=False)
    
    # Generate CSV
    with open('table9_component_interaction.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Combination', 'Overall Acc', 'Relation Discovery', 'Reasoning Quality', 'Error Rate', 'Synergy Score'])
        
        for combo, inter in COMPONENT_INTERACTION.items():
            writer.writerow([
                inter.combination, inter.overall_acc, inter.relation_discovery,
                inter.reasoning_quality, inter.error_rate, inter.synergy_score
            ])
    
    print("✅ Table 9 source data files generated successfully")


def generate_table10_source_data():
    """Generate Table 10: Efficiency Analysis source data."""
    print("Generating Table 10: Efficiency Analysis source data...")
    
    # Generate JSON
    efficiency_json = {}
    for method, metrics in EFFICIENCY_DATA.items():
        efficiency_json[method] = {
            'method_name': metrics.method_name,
            'runtime_metrics': {
                'avg_runtime': metrics.avg_runtime,
                'L2_runtime': metrics.l2_runtime,
                'L3_runtime': metrics.l3_runtime
            },
            'memory_usage_mb': metrics.memory_mb,
            'efficiency_score': metrics.efficiency_score
        }
    
    with open('table10_efficiency_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(efficiency_json, f, indent=2, ensure_ascii=False)
    
    # Generate CSV
    with open('table10_efficiency_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Avg Runtime (s)', 'Memory (MB)', 'L2 Runtime (s)', 'L3 Runtime (s)', 'Efficiency Score'])
        
        for method, metrics in EFFICIENCY_DATA.items():
            writer.writerow([
                metrics.method_name, metrics.avg_runtime, metrics.memory_mb,
                metrics.l2_runtime, metrics.l3_runtime, metrics.efficiency_score
            ])
    
    print("✅ Table 10 source data files generated successfully")


def generate_summary_file():
    """Generate a summary of all generated source data files."""
    print("Generating summary file...")
    
    summary = {
        'generation_timestamp': '2025-01-31',
        'source_module': 'src/data',
        'generated_files': {
            'table3_dataset_characteristics': [
                'table3_dataset_characteristics.json',
                'table3_dataset_characteristics.csv',
                'table3_dataset_characteristics.md'
            ],
            'table4_performance_comparison': [
                'table4_performance_comparison.json',
                'table4_performance_comparison.csv',
                'table4_performance_comparison.md'
            ],
            'table5_complexity_performance': [
                'table5_complexity_performance.json',
                'table5_complexity_performance.csv',
                'table5_complexity_performance.md'
            ],
            'table6_relation_discovery': [
                'table6_relation_discovery.json',
                'table6_relation_discovery.csv'
            ],
            'table7_reasoning_chain': [
                'table7_reasoning_chain.json',
                'table7_reasoning_chain.csv'
            ],
            'table8_ablation_study': [
                'table8_ablation_study.json',
                'table8_ablation_study.csv'
            ],
            'table9_component_interaction': [
                'table9_component_interaction.json',
                'table9_component_interaction.csv'
            ],
            'table10_efficiency_analysis': [
                'table10_efficiency_analysis.json',
                'table10_efficiency_analysis.csv'
            ]
        },
        'statistics': {
            'total_datasets': len(DATASET_CHARACTERISTICS),
            'total_methods': len(PERFORMANCE_DATA),
            'total_ablation_configs': len(ABLATION_DATA),
            'total_component_interactions': len(COMPONENT_INTERACTION),
            'total_source_files': 20
        }
    }
    
    with open('source_data_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("✅ Summary file generated successfully")


def main():
    """Main function to generate all source data files."""
    print("🚀 Starting source data files generation from src/data module...\n")
    
    try:
        # Generate all table source data
        generate_table3_source_data()
        generate_table4_source_data()
        generate_table5_source_data()
        generate_table6_source_data()
        generate_table7_source_data()
        generate_table8_source_data()
        generate_table9_source_data()
        generate_table10_source_data()
        
        # Generate summary
        generate_summary_file()
        
        print("\n🎉 All source data files generated successfully!")
        print("\nGenerated files:")
        print("- Table 3: 3 files (JSON, CSV, Markdown)")
        print("- Table 4: 3 files (JSON, CSV, Markdown)")
        print("- Table 5: 3 files (JSON, CSV, Markdown)")
        print("- Table 6: 2 files (JSON, CSV)")
        print("- Table 7: 2 files (JSON, CSV)")
        print("- Table 8: 2 files (JSON, CSV)")
        print("- Table 9: 2 files (JSON, CSV)")
        print("- Table 10: 2 files (JSON, CSV)")
        print("- Summary: 1 file (JSON)")
        print(f"\nTotal: 21 source data files generated from src/data module")
        
    except Exception as e:
        print(f"❌ Error generating source data files: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 