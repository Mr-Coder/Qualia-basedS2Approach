#!/usr/bin/env python3
"""
Dataset Analysis Example

This example demonstrates how to use the dataset characteristics module
to analyze and compare different math problem datasets.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_characteristics import (calculate_weighted_complexity_score,
                                          export_to_json, get_all_datasets,
                                          get_complexity_distribution,
                                          get_dataset_info,
                                          get_dataset_statistics,
                                          get_datasets_by_domain,
                                          get_datasets_by_language,
                                          print_dataset_table)


def analyze_dataset_complexity():
    """Analyze complexity distribution across datasets."""
    print("=== Dataset Complexity Analysis ===\n")
    
    datasets = get_all_datasets()
    complexity_scores = []
    
    print(f"{'Dataset':<15} {'Weighted Score':<15} {'DIR Score':<10} {'Complexity Profile'}")
    print("-" * 80)
    
    for name, info in datasets.items():
        weighted_score = calculate_weighted_complexity_score(name)
        complexity_scores.append((name, weighted_score, info.dir_score))
        
        # Get complexity distribution
        dist = get_complexity_distribution(name)
        profile = f"L0:{dist['L0']:.1f}% L1:{dist['L1']:.1f}% L2:{dist['L2']:.1f}% L3:{dist['L3']:.1f}%"
        
        print(f"{name:<15} {weighted_score:<15.2f} {info.dir_score:<10.2f} {profile}")
    
    # Sort by complexity
    complexity_scores.sort(key=lambda x: x[1])
    
    print(f"\n{'Ranking by Weighted Complexity Score:'}")
    print("1. Easiest to Hardest:")
    for i, (name, score, dir_score) in enumerate(complexity_scores, 1):
        print(f"   {i}. {name} (Weighted: {score:.2f}, DIR: {dir_score})")


def analyze_by_language():
    """Analyze datasets by language."""
    print("\n=== Language Distribution Analysis ===\n")
    
    languages = ["English", "Chinese", "Mixed"]
    
    for lang in languages:
        datasets = get_datasets_by_language(lang)
        if datasets:
            print(f"{lang} Datasets:")
            total_size = sum(d.size for d in datasets)
            avg_dir = sum(d.dir_score for d in datasets) / len(datasets)
            
            for dataset in datasets:
                print(f"  - {dataset.name}: {dataset.size:,} problems, DIR: {dataset.dir_score}")
            
            print(f"  Total: {len(datasets)} datasets, {total_size:,} problems")
            print(f"  Average DIR Score: {avg_dir:.2f}\n")


def analyze_by_domain():
    """Analyze datasets by domain."""
    print("=== Domain Distribution Analysis ===\n")
    
    domains = ["Elementary", "Grade School", "Competition", "Multi-domain", "Specialized"]
    
    for domain in domains:
        datasets = get_datasets_by_domain(domain)
        if datasets:
            print(f"{domain} Datasets:")
            total_size = sum(d.size for d in datasets)
            avg_dir = sum(d.dir_score for d in datasets) / len(datasets)
            
            for dataset in datasets:
                print(f"  - {dataset.name}: {dataset.size:,} problems, DIR: {dataset.dir_score}")
            
            print(f"  Total: {len(datasets)} datasets, {total_size:,} problems")
            print(f"  Average DIR Score: {avg_dir:.2f}\n")


def compare_datasets():
    """Compare specific datasets."""
    print("=== Dataset Comparison ===\n")
    
    # Compare English competition datasets
    print("English Competition Datasets Comparison:")
    competition_datasets = get_datasets_by_domain("Competition")
    english_competition = [d for d in competition_datasets if d.language == "English"]
    
    for dataset in english_competition:
        dist = get_complexity_distribution(dataset.name)
        weighted = calculate_weighted_complexity_score(dataset.name)
        print(f"\n{dataset.name}:")
        print(f"  Size: {dataset.size:,} problems")
        print(f"  DIR Score: {dataset.dir_score}")
        print(f"  Weighted Complexity: {weighted:.2f}")
        print(f"  High Complexity (L2+L3): {dist['L2'] + dist['L3']:.1f}%")
    
    # Compare largest datasets
    print("\n\nLargest Datasets Comparison:")
    all_datasets = list(get_all_datasets().values())
    largest = sorted(all_datasets, key=lambda x: x.size, reverse=True)[:3]
    
    for dataset in largest:
        dist = get_complexity_distribution(dataset.name)
        weighted = calculate_weighted_complexity_score(dataset.name)
        print(f"\n{dataset.name}:")
        print(f"  Size: {dataset.size:,} problems")
        print(f"  Language: {dataset.language}")
        print(f"  Domain: {dataset.domain}")
        print(f"  DIR Score: {dataset.dir_score}")
        print(f"  Weighted Complexity: {weighted:.2f}")


def generate_insights():
    """Generate insights from the dataset analysis."""
    print("\n=== Key Insights ===\n")
    
    stats = get_dataset_statistics()
    all_datasets = get_all_datasets()
    
    # Find extremes
    largest = max(all_datasets.values(), key=lambda x: x.size)
    smallest = min(all_datasets.values(), key=lambda x: x.size)
    highest_dir = max(all_datasets.values(), key=lambda x: x.dir_score)
    lowest_dir = min(all_datasets.values(), key=lambda x: x.dir_score)
    
    # Calculate weighted complexity for all
    complexity_scores = [(name, calculate_weighted_complexity_score(name)) 
                        for name in all_datasets.keys()]
    most_complex = max(complexity_scores, key=lambda x: x[1])
    least_complex = min(complexity_scores, key=lambda x: x[1])
    
    print("Dataset Extremes:")
    print(f"  Largest: {largest.name} ({largest.size:,} problems)")
    print(f"  Smallest: {smallest.name} ({smallest.size:,} problems)")
    print(f"  Highest DIR Score: {highest_dir.name} ({highest_dir.dir_score})")
    print(f"  Lowest DIR Score: {lowest_dir.name} ({lowest_dir.dir_score})")
    print(f"  Most Complex: {most_complex[0]} (Weighted: {most_complex[1]:.2f})")
    print(f"  Least Complex: {least_complex[0]} (Weighted: {least_complex[1]:.2f})")
    
    print(f"\nOverall Statistics:")
    print(f"  Total Datasets: {stats['total_datasets']}")
    print(f"  Total Problems: {stats['total_problems']:,}")
    print(f"  Average DIR Score: {stats['average_dir_score']}")
    print(f"  Language Distribution: {stats['language_distribution']}")
    print(f"  Domain Distribution: {stats['domain_distribution']}")
    
    complexity_dist = stats['average_complexity_distribution']
    print(f"\nAverage Complexity Distribution:")
    print(f"  L0 (Basic): {complexity_dist['L0']}%")
    print(f"  L1 (Simple): {complexity_dist['L1']}%")
    print(f"  L2 (Medium): {complexity_dist['L2']}%")
    print(f"  L3 (Complex): {complexity_dist['L3']}%")


def main():
    """Main function to run all analyses."""
    print("Dataset Characteristics Analysis Tool")
    print("=" * 50)
    
    # Print the main table
    print_dataset_table()
    
    # Run various analyses
    analyze_dataset_complexity()
    analyze_by_language()
    analyze_by_domain()
    compare_datasets()
    generate_insights()
    
    # Export data
    print("\n=== Exporting Data ===")
    export_to_json("dataset_characteristics_analysis.json")
    
    print("\nAnalysis complete! Check 'dataset_characteristics_analysis.json' for detailed data.")


if __name__ == "__main__":
    main() 