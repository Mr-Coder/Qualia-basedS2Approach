"""
Dataset Characteristics with DIR-MWP Complexity Distribution

This module contains the dataset characteristics from Table 3, including
dataset size, language, domain, complexity distribution (L0-L3), and DIR scores.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DatasetInfo:
    """Data class for storing dataset characteristics."""
    name: str
    size: int
    language: str
    domain: str
    l0_percent: float
    l1_percent: float
    l2_percent: float
    l3_percent: float
    dir_score: float


# Dataset characteristics from Table 3
DATASET_CHARACTERISTICS = {
    "Math23K": DatasetInfo(
        name="Math23K",
        size=23162,
        language="Chinese",
        domain="Elementary",
        l0_percent=38.2,
        l1_percent=31.4,
        l2_percent=19.7,
        l3_percent=10.7,
        dir_score=2.03
    ),
    "GSM8K": DatasetInfo(
        name="GSM8K",
        size=8500,
        language="English",
        domain="Grade School",
        l0_percent=42.1,
        l1_percent=28.9,
        l2_percent=18.3,
        l3_percent=10.7,
        dir_score=1.98
    ),
    "MAWPS": DatasetInfo(
        name="MAWPS",
        size=2373,
        language="English",
        domain="Multi-domain",
        l0_percent=47.3,
        l1_percent=26.8,
        l2_percent=16.2,
        l3_percent=9.7,
        dir_score=1.88
    ),
    "MathQA": DatasetInfo(
        name="MathQA",
        size=37297,
        language="English",
        domain="Competition",
        l0_percent=35.7,
        l1_percent=32.1,
        l2_percent=21.4,
        l3_percent=10.8,
        dir_score=2.07
    ),
    "MATH": DatasetInfo(
        name="MATH",
        size=12500,
        language="English",
        domain="Competition",
        l0_percent=28.4,
        l1_percent=31.7,
        l2_percent=25.1,
        l3_percent=14.8,
        dir_score=2.26
    ),
    "SVAMP": DatasetInfo(
        name="SVAMP",
        size=1000,
        language="English",
        domain="Grade School",
        l0_percent=35.2,
        l1_percent=29.1,
        l2_percent=22.4,
        l3_percent=13.3,
        dir_score=2.14
    ),
    "ASDiv": DatasetInfo(
        name="ASDiv",
        size=2305,
        language="English",
        domain="Elementary",
        l0_percent=41.7,
        l1_percent=28.6,
        l2_percent=19.8,
        l3_percent=9.9,
        dir_score=1.98
    ),
    "DIR-MWP-Test": DatasetInfo(
        name="DIR-MWP-Test",
        size=1200,
        language="Mixed",
        domain="Specialized",
        l0_percent=15.0,
        l1_percent=25.0,
        l2_percent=35.0,
        l3_percent=25.0,
        dir_score=2.70
    )
}


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """
    Get dataset information by name.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        DatasetInfo object with dataset characteristics
        
    Raises:
        KeyError: If dataset name is not found
    """
    return DATASET_CHARACTERISTICS[dataset_name]


def get_all_datasets() -> Dict[str, DatasetInfo]:
    """
    Get all dataset characteristics.
    
    Returns:
        Dictionary mapping dataset names to DatasetInfo objects
    """
    return DATASET_CHARACTERISTICS.copy()


def get_datasets_by_language(language: str) -> List[DatasetInfo]:
    """
    Get datasets filtered by language.
    
    Args:
        language: Language to filter by (e.g., "English", "Chinese", "Mixed")
        
    Returns:
        List of DatasetInfo objects matching the language
    """
    return [info for info in DATASET_CHARACTERISTICS.values() 
            if info.language == language]


def get_datasets_by_domain(domain: str) -> List[DatasetInfo]:
    """
    Get datasets filtered by domain.
    
    Args:
        domain: Domain to filter by (e.g., "Elementary", "Grade School", "Competition")
        
    Returns:
        List of DatasetInfo objects matching the domain
    """
    return [info for info in DATASET_CHARACTERISTICS.values() 
            if info.domain == domain]


def get_complexity_distribution(dataset_name: str) -> Dict[str, float]:
    """
    Get complexity level distribution for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with complexity levels (L0-L3) and their percentages
    """
    info = get_dataset_info(dataset_name)
    return {
        "L0": info.l0_percent,
        "L1": info.l1_percent,
        "L2": info.l2_percent,
        "L3": info.l3_percent
    }


def calculate_weighted_complexity_score(dataset_name: str) -> float:
    """
    Calculate weighted complexity score based on distribution.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Weighted complexity score (0-3 scale)
    """
    info = get_dataset_info(dataset_name)
    return (0 * info.l0_percent + 
            1 * info.l1_percent + 
            2 * info.l2_percent + 
            3 * info.l3_percent) / 100


def get_dataset_statistics() -> Dict[str, Any]:
    """
    Get overall statistics across all datasets.
    
    Returns:
        Dictionary with various statistics
    """
    datasets = list(DATASET_CHARACTERISTICS.values())
    
    total_problems = sum(d.size for d in datasets)
    avg_dir_score = sum(d.dir_score for d in datasets) / len(datasets)
    
    # Language distribution
    languages = {}
    for dataset in datasets:
        languages[dataset.language] = languages.get(dataset.language, 0) + 1
    
    # Domain distribution
    domains = {}
    for dataset in datasets:
        domains[dataset.domain] = domains.get(dataset.domain, 0) + 1
    
    # Complexity level averages
    avg_l0 = sum(d.l0_percent for d in datasets) / len(datasets)
    avg_l1 = sum(d.l1_percent for d in datasets) / len(datasets)
    avg_l2 = sum(d.l2_percent for d in datasets) / len(datasets)
    avg_l3 = sum(d.l3_percent for d in datasets) / len(datasets)
    
    return {
        "total_datasets": len(datasets),
        "total_problems": total_problems,
        "average_dir_score": round(avg_dir_score, 2),
        "language_distribution": languages,
        "domain_distribution": domains,
        "average_complexity_distribution": {
            "L0": round(avg_l0, 1),
            "L1": round(avg_l1, 1),
            "L2": round(avg_l2, 1),
            "L3": round(avg_l3, 1)
        }
    }


def export_to_json(filename: str = "dataset_characteristics.json") -> None:
    """
    Export dataset characteristics to JSON file.
    
    Args:
        filename: Output filename
    """
    import json

    # Convert dataclasses to dictionaries
    export_data = {}
    for name, info in DATASET_CHARACTERISTICS.items():
        export_data[name] = {
            "name": info.name,
            "size": info.size,
            "language": info.language,
            "domain": info.domain,
            "complexity_distribution": {
                "L0": info.l0_percent,
                "L1": info.l1_percent,
                "L2": info.l2_percent,
                "L3": info.l3_percent
            },
            "dir_score": info.dir_score,
            "weighted_complexity_score": calculate_weighted_complexity_score(name)
        }
    
    # Add overall statistics
    export_data["_statistics"] = get_dataset_statistics()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset characteristics exported to {filename}")


def print_dataset_table() -> None:
    """Print a formatted table of all dataset characteristics."""
    print("Dataset Characteristics with DIR-MWP Complexity Distribution")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Size':<8} {'Language':<8} {'Domain':<12} {'L0(%)':<6} {'L1(%)':<6} {'L2(%)':<6} {'L3(%)':<6} {'DIR Score':<8}")
    print("-" * 90)
    
    for info in DATASET_CHARACTERISTICS.values():
        print(f"{info.name:<15} {info.size:<8} {info.language:<8} {info.domain:<12} "
              f"{info.l0_percent:<6} {info.l1_percent:<6} {info.l2_percent:<6} "
              f"{info.l3_percent:<6} {info.dir_score:<8}")


if __name__ == "__main__":
    # Example usage
    print_dataset_table()
    print("\n" + "=" * 50)
    print("Overall Statistics:")
    stats = get_dataset_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Export to JSON
    export_to_json() 