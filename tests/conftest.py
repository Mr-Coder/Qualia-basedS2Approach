"""
Pytest Configuration
====================

Global test configuration and fixtures.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Test data directory
TEST_DATA_DIR = project_root / "tests" / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture
def sample_math_problems():
    """Sample mathematical problems for testing"""
    return [
        {
            "id": "test_001",
            "problem": "小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？",
            "expected_answer": 8,
            "problem_type": "addition",
            "language": "chinese"
        },
        {
            "id": "test_002", 
            "problem": "A train travels 60 miles per hour for 2 hours. How far does it travel?",
            "expected_answer": 120,
            "problem_type": "multiplication",
            "language": "english"
        },
        {
            "id": "test_003",
            "problem": "Solve for x: 2x + 5 = 13",
            "expected_answer": 4,
            "problem_type": "algebra",
            "language": "english"
        }
    ]


@pytest.fixture
def reasoning_config():
    """Default reasoning configuration"""
    return {
        "max_steps": 10,
        "confidence_threshold": 0.7,
        "timeout_seconds": 30,
        "strategy_priorities": {
            "chain_of_thought": 1,
            "tree_of_thoughts": 2,
            "graph_of_thoughts": 3
        }
    }


@pytest.fixture
def test_datasets():
    """Test dataset configurations"""
    return {
        "small": {
            "name": "small_test_dataset",
            "size": 10,
            "difficulty": "easy"
        },
        "medium": {
            "name": "medium_test_dataset", 
            "size": 100,
            "difficulty": "medium"
        },
        "large": {
            "name": "large_test_dataset",
            "size": 1000, 
            "difficulty": "hard"
        }
    }


@pytest.fixture(scope="session")
def performance_baseline():
    """Baseline performance metrics for regression testing"""
    return {
        "accuracy": {
            "easy_problems": 0.95,
            "medium_problems": 0.85,
            "hard_problems": 0.70
        },
        "speed": {
            "max_time_per_problem": 10.0,  # seconds
            "average_time_per_problem": 3.0  # seconds
        },
        "confidence": {
            "min_confidence": 0.6,
            "target_confidence": 0.8
        }
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as smoke test"
    ) 