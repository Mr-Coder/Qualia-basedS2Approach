#!/usr/bin/env python3
"""
Generate Dataset Files

This script generates dataset characteristic files in multiple formats
based on the table data provided.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import csv
import json

from src.data.dataset_characteristics import (
    calculate_weighted_complexity_score, export_to_json, get_all_datasets,
    get_dataset_statistics)


def export_to_csv(filename: str = "dataset_characteristics.csv") -> None:
    """Export dataset characteristics to CSV file."""
    datasets = get_all_datasets()
    
    fieldnames = [
        'Dataset', 'Size', 'Language', 'Domain', 
        'L0_Percent', 'L1_Percent', 'L2_Percent', 'L3_Percent', 
        'DIR_Score', 'Weighted_Complexity_Score'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for name, info in datasets.items():
            weighted_score = calculate_weighted_complexity_score(name)
            
            writer.writerow({
                'Dataset': info.name,
                'Size': info.size,
                'Language': info.language,
                'Domain': info.domain,
                'L0_Percent': info.l0_percent,
                'L1_Percent': info.l1_percent,
                'L2_Percent': info.l2_percent,
                'L3_Percent': info.l3_percent,
                'DIR_Score': info.dir_score,
                'Weighted_Complexity_Score': weighted_score
            })
    
    print(f"Dataset characteristics exported to {filename}")


def export_statistics_to_csv(filename: str = "dataset_statistics.csv") -> None:
    """Export overall statistics to CSV file."""
    stats = get_dataset_statistics()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Statistic', 'Value'])
        
        writer.writerow(['Total Datasets', stats['total_datasets']])
        writer.writerow(['Total Problems', stats['total_problems']])
        writer.writerow(['Average DIR Score', stats['average_dir_score']])
        
        # Language distribution
        writer.writerow(['', ''])  # Empty row for separation
        writer.writerow(['Language Distribution', ''])
        for lang, count in stats['language_distribution'].items():
            writer.writerow([f'  {lang}', count])
        
        # Domain distribution
        writer.writerow(['', ''])
        writer.writerow(['Domain Distribution', ''])
        for domain, count in stats['domain_distribution'].items():
            writer.writerow([f'  {domain}', count])
        
        # Complexity distribution
        writer.writerow(['', ''])
        writer.writerow(['Average Complexity Distribution', ''])
        for level, percent in stats['average_complexity_distribution'].items():
            writer.writerow([f'  {level}', f'{percent}%'])
    
    print(f"Dataset statistics exported to {filename}")


def export_complexity_matrix_to_csv(filename: str = "complexity_matrix.csv") -> None:
    """Export complexity distribution matrix to CSV file."""
    datasets = get_all_datasets()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Dataset', 'L0', 'L1', 'L2', 'L3', 'DIR_Score'])
        
        # Data rows
        for name, info in datasets.items():
            writer.writerow([
                info.name,
                info.l0_percent,
                info.l1_percent,
                info.l2_percent,
                info.l3_percent,
                info.dir_score
            ])
    
    print(f"Complexity matrix exported to {filename}")


def export_markdown_table(filename: str = "dataset_table.md") -> None:
    """Export dataset characteristics as a markdown table."""
    datasets = get_all_datasets()
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Dataset Characteristics with DIR-MWP Complexity Distribution\n\n")
        
        # Table header
        f.write("| Dataset | Size | Language | Domain | L0 (%) | L1 (%) | L2 (%) | L3 (%) | DIR Score |\n")
        f.write("|---------|------|----------|--------|--------|--------|--------|--------|----------|\n")
        
        # Table rows
        for info in datasets.values():
            f.write(f"| {info.name} | {info.size:,} | {info.language} | {info.domain} | "
                   f"{info.l0_percent} | {info.l1_percent} | {info.l2_percent} | "
                   f"{info.l3_percent} | {info.dir_score} |\n")
        
        # Add statistics section
        stats = get_dataset_statistics()
        f.write("\n## Overall Statistics\n\n")
        f.write(f"- **Total Datasets**: {stats['total_datasets']}\n")
        f.write(f"- **Total Problems**: {stats['total_problems']:,}\n")
        f.write(f"- **Average DIR Score**: {stats['average_dir_score']}\n\n")
        
        f.write("### Language Distribution\n\n")
        for lang, count in stats['language_distribution'].items():
            f.write(f"- {lang}: {count} datasets\n")
        
        f.write("\n### Domain Distribution\n\n")
        for domain, count in stats['domain_distribution'].items():
            f.write(f"- {domain}: {count} datasets\n")
        
        f.write("\n### Average Complexity Distribution\n\n")
        complexity = stats['average_complexity_distribution']
        f.write(f"- L0 (Basic): {complexity['L0']}%\n")
        f.write(f"- L1 (Simple): {complexity['L1']}%\n")
        f.write(f"- L2 (Medium): {complexity['L2']}%\n")
        f.write(f"- L3 (Complex): {complexity['L3']}%\n")
    
    print(f"Markdown table exported to {filename}")


def main():
    """Generate all dataset files."""
    print("Generating dataset files in multiple formats...")
    print("=" * 50)
    
    # JSON export
    export_to_json("dataset_characteristics.json")
    
    # CSV exports
    export_to_csv("dataset_characteristics.csv")
    export_statistics_to_csv("dataset_statistics.csv")
    export_complexity_matrix_to_csv("complexity_matrix.csv")
    
    # Markdown export
    export_markdown_table("dataset_table.md")
    
    print("\n" + "=" * 50)
    print("All dataset files generated successfully!")
    print("\nGenerated files:")
    print("- dataset_characteristics.json (JSON format)")
    print("- dataset_characteristics.csv (CSV format)")
    print("- dataset_statistics.csv (Statistics summary)")
    print("- complexity_matrix.csv (Complexity matrix)")
    print("- dataset_table.md (Markdown table)")


if __name__ == "__main__":
    main() 