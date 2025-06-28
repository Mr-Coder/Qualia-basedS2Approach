#!/usr/bin/env python3
"""
DIR-MWP Dataset Validation Script
Validates the generated dataset and displays sample problems from each complexity level
"""

import json
from collections import Counter

import pandas as pd


def load_dataset(file_path):
    """Load the DIR-MWP dataset from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
        return None

def validate_dataset_structure(dataset):
    """Validate the structure of the dataset"""
    print("=== Dataset Structure Validation ===")
    
    # Check basic structure
    required_keys = ['dataset_info', 'complexity_statistics', 'problems']
    for key in required_keys:
        if key not in dataset:
            print(f"❌ Missing required key: {key}")
            return False
        else:
            print(f"✅ Found key: {key}")
    
    # Check dataset info
    info = dataset['dataset_info']
    print(f"\nDataset Info:")
    print(f"  Name: {info.get('name', 'N/A')}")
    print(f"  Description: {info.get('description', 'N/A')}")
    print(f"  Total Problems: {info.get('total_problems', 'N/A')}")
    print(f"  Complexity Levels: {info.get('complexity_levels', 'N/A')}")
    print(f"  Creation Date: {info.get('creation_date', 'N/A')}")
    
    # Check complexity statistics
    stats = dataset['complexity_statistics']
    print(f"\nComplexity Statistics:")
    total_expected = 0
    for level, data in stats.items():
        count = data.get('count', 0)
        avg_relations = data.get('avg_relations', 0)
        inference_depth = data.get('inference_depth', 0)
        description = data.get('description', 'N/A')
        print(f"  {level}:")
        print(f"    Count: {count}")
        print(f"    Avg Relations: {avg_relations}")
        print(f"    Inference Depth: {inference_depth}")
        print(f"    Description: {description}")
        total_expected += count
    
    # Check problems
    problems = dataset['problems']
    actual_total = len(problems)
    print(f"\nActual vs Expected Problem Counts:")
    print(f"  Expected Total: {total_expected}")
    print(f"  Actual Total: {actual_total}")
    
    if actual_total == total_expected:
        print("✅ Problem count matches expectations")
    else:
        print("❌ Problem count mismatch")
    
    return True

def analyze_problem_distribution(dataset):
    """Analyze the distribution of problems across complexity levels"""
    print("\n=== Problem Distribution Analysis ===")
    
    problems = dataset['problems']
    
    # Count problems by complexity level
    complexity_counts = Counter(problem['complexity_level'] for problem in problems)
    print(f"\nActual distribution:")
    for level, count in sorted(complexity_counts.items()):
        print(f"  {level}: {count} problems")
    
    # Analyze inference depths and relation counts
    level_analysis = {}
    for problem in problems:
        level = problem['complexity_level']
        if level not in level_analysis:
            level_analysis[level] = {
                'inference_depths': [],
                'relation_counts': [],
                'domain_knowledge_counts': []
            }
        
        level_analysis[level]['inference_depths'].append(problem.get('inference_depth', 0))
        level_analysis[level]['relation_counts'].append(problem.get('relation_count', 0))
        level_analysis[level]['domain_knowledge_counts'].append(len(problem.get('domain_knowledge', [])))
    
    print(f"\nComplexity Analysis:")
    for level in sorted(level_analysis.keys()):
        data = level_analysis[level]
        avg_depth = sum(data['inference_depths']) / len(data['inference_depths'])
        avg_relations = sum(data['relation_counts']) / len(data['relation_counts'])
        avg_domain = sum(data['domain_knowledge_counts']) / len(data['domain_knowledge_counts'])
        
        print(f"  {level}:")
        print(f"    Avg Inference Depth: {avg_depth:.2f}")
        print(f"    Avg Relation Count: {avg_relations:.2f}")
        print(f"    Avg Domain Knowledge Items: {avg_domain:.2f}")

def show_sample_problems(dataset, samples_per_level=2):
    """Display sample problems from each complexity level"""
    print(f"\n=== Sample Problems ({samples_per_level} per level) ===")
    
    problems = dataset['problems']
    problems_by_level = {}
    
    # Group problems by complexity level
    for problem in problems:
        level = problem['complexity_level']
        if level not in problems_by_level:
            problems_by_level[level] = []
        problems_by_level[level].append(problem)
    
    # Display samples
    for level in sorted(problems_by_level.keys()):
        print(f"\n--- {level} ---")
        level_problems = problems_by_level[level]
        
        for i in range(min(samples_per_level, len(level_problems))):
            problem = level_problems[i]
            print(f"\nProblem ID: {problem['id']}")
            print(f"Problem: {problem['problem']}")
            print(f"Answer: {problem['answer']}")
            print(f"Solution Steps:")
            for j, step in enumerate(problem['solution_steps'], 1):
                print(f"  {j}. {step}")
            print(f"Explicit Relations: {', '.join(problem['explicit_relations'])}")
            print(f"Implicit Relations: {', '.join(problem['implicit_relations'])}")
            print(f"Domain Knowledge: {', '.join(problem['domain_knowledge'])}")
            print(f"Inference Depth: {problem['inference_depth']}")
            print(f"Relation Count: {problem['relation_count']}")
            print("-" * 50)

def generate_summary_report(dataset):
    """Generate a summary report of the dataset"""
    print("\n=== Dataset Summary Report ===")
    
    problems = dataset['problems']
    
    # Basic statistics
    total_problems = len(problems)
    
    # Complexity distribution
    complexity_counts = Counter(problem['complexity_level'] for problem in problems)
    
    # Domain knowledge analysis
    all_domains = []
    for problem in problems:
        all_domains.extend(problem.get('domain_knowledge', []))
    domain_counts = Counter(all_domains)
    
    # Inference depth analysis
    inference_depths = [problem.get('inference_depth', 0) for problem in problems]
    avg_inference_depth = sum(inference_depths) / len(inference_depths)
    
    print(f"Total Problems: {total_problems}")
    print(f"Average Inference Depth: {avg_inference_depth:.2f}")
    print(f"\nComplexity Level Distribution:")
    for level, count in sorted(complexity_counts.items()):
        percentage = (count / total_problems) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
    
    print(f"\nTop Domain Knowledge Areas:")
    for domain, count in domain_counts.most_common(10):
        print(f"  {domain}: {count} problems")
    
    # Export to CSV for further analysis
    export_to_csv(dataset)

def export_to_csv(dataset):
    """Export dataset to CSV format for analysis"""
    problems = dataset['problems']
    
    # Prepare data for CSV
    csv_data = []
    for problem in problems:
        csv_data.append({
            'id': problem['id'],
            'complexity_level': problem['complexity_level'],
            'problem': problem['problem'],
            'answer': problem['answer'],
            'inference_depth': problem['inference_depth'],
            'relation_count': problem['relation_count'],
            'explicit_relations_count': len(problem['explicit_relations']),
            'implicit_relations_count': len(problem['implicit_relations']),
            'domain_knowledge_count': len(problem['domain_knowledge']),
            'solution_steps_count': len(problem['solution_steps'])
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    csv_file = "Data/DIR-MWP/dir_mwp_analysis.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n✅ Analysis data exported to: {csv_file}")

def main():
    """Main validation function"""
    dataset_file = "Data/DIR-MWP/dir_mwp_complete_dataset.json"
    
    print("DIR-MWP Dataset Validation")
    print("=" * 50)
    
    # Load dataset
    dataset = load_dataset(dataset_file)
    if not dataset:
        return
    
    # Validate structure
    if not validate_dataset_structure(dataset):
        return
    
    # Analyze distribution
    analyze_problem_distribution(dataset)
    
    # Show sample problems
    show_sample_problems(dataset, samples_per_level=2)
    
    # Generate summary report
    generate_summary_report(dataset)
    
    print("\n✅ Dataset validation completed successfully!")

if __name__ == "__main__":
    main() 