# COT-DIR API Documentation

This document provides detailed API documentation for the standardized data flow components in the COT-DIR mathematical reasoning system.

## Table of Contents

- [DataLoader](#dataloader)
- [Preprocessor](#preprocessor)
- [ReasoningEngine](#reasoningengine)
- [Evaluator](#evaluator)
- [Main Demo](#main-demo)

## DataLoader

The `DataLoader` class provides a standardized interface for loading mathematical datasets.

### Class Definition

```python
class DataLoader:
    def __init__(self, data_dir: str = "Data")
    def load(self, path: str = None, dataset_name: str = None, max_samples: int = None) -> List[Dict]
```

### Constructor

**Parameters:**
- `data_dir` (str, optional): Directory containing datasets. Defaults to "Data".

### Methods

#### load()

Loads a dataset and returns standardized problem samples.

**Parameters:**
- `path` (str, optional): File path to dataset. Mutually exclusive with `dataset_name`.
- `dataset_name` (str, optional): Name of the dataset to load. Mutually exclusive with `path`.
- `max_samples` (int, optional): Maximum number of samples to load. If None, loads all samples.

**Returns:**
- `List[Dict]`: List of standardized problem samples with the following structure:
  ```python
  {
      "id": "unique_identifier",
      "problem": "problem_text",
      "answer": "expected_answer",
      "dataset": "dataset_name",
      "metadata": {...}  # Original dataset fields
  }
  ```

**Raises:**
- `ValueError`: If neither `path` nor `dataset_name` is provided, or if dataset is not found.

**Example:**
```python
from src.data.loader import DataLoader

# Load by dataset name
loader = DataLoader()
samples = loader.load(dataset_name="Math23K", max_samples=10)

# Load by path
samples = loader.load(path="Data/Math23K/math23k.json", max_samples=5)
```

## Preprocessor

The `Preprocessor` class handles text cleaning, problem type classification, and complexity analysis.

### Class Definition

```python
class Preprocessor:
    def __init__(self)
    def process(self, sample: Dict) -> Dict
```

### Methods

#### process()

Processes a single sample to add preprocessing information.

**Parameters:**
- `sample` (Dict): Input sample with at least one of: "problem", "question", or "text" fields.

**Returns:**
- `Dict`: Enhanced sample with additional preprocessing fields:
  ```python
  {
      # Original fields
      "id": "unique_identifier",
      "problem": "problem_text",
      "answer": "expected_answer",
      
      # Preprocessing additions
      "cleaned_text": "cleaned_problem_text",
      "problem_type": "arithmetic|word_problem|equation|...",
      "classification_confidence": 0.85,
      "complexity_level": "L0|L1|L2|L3"
  }
  ```

**Example:**
```python
from src.data.preprocessor import Preprocessor

preprocessor = Preprocessor()
processed_sample = preprocessor.process({
    "problem": "小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？",
    "answer": "8"
})
```

## ReasoningEngine

The `ReasoningEngine` class provides a unified interface for mathematical reasoning with multiple strategies.

### Class Definition

```python
class ReasoningEngine:
    def __init__(self, config=None)
    def solve(self, sample: Dict) -> Dict
```

### Constructor

**Parameters:**
- `config` (Dict, optional): Configuration dictionary for reasoning strategies.

### Methods

#### solve()

Solves a mathematical problem using appropriate reasoning strategies.

**Parameters:**
- `sample` (Dict): Preprocessed sample containing problem information.

**Returns:**
- `Dict`: Reasoning result with the following structure:
  ```python
  {
      "final_answer": "computed_answer",
      "reasoning_steps": ["step1", "step2", ...],
      "confidence": 0.92,
      "strategy_used": "DIR|COT|DIR-COT|MLR",
      "processing_time": 1.23,
      "metadata": {...}  # Additional reasoning information
  }
  ```

**Example:**
```python
from src.reasoning_core.reasoning_engine import ReasoningEngine

engine = ReasoningEngine()
result = engine.solve(processed_sample)
print(f"Answer: {result['final_answer']}")
print(f"Strategy: {result['strategy_used']}")
```

## Evaluator

The `Evaluator` class provides comprehensive evaluation of reasoning results.

### Class Definition

```python
class Evaluator:
    def __init__(self, config=None)
    def evaluate(self, predictions: List[Dict], references: List[Dict]) -> Dict
```

### Constructor

**Parameters:**
- `config` (Dict, optional): Configuration dictionary for evaluation metrics.

### Methods

#### evaluate()

Evaluates a list of predictions against reference answers.

**Parameters:**
- `predictions` (List[Dict]): List of prediction results from reasoning engine.
- `references` (List[Dict]): List of reference samples with expected answers.

**Returns:**
- `Dict`: Evaluation results with the following structure:
  ```python
  {
      "overall_score": 0.85,
      "metric_results": {
          "accuracy": 0.90,
          "reasoning_quality": 0.82,
          "efficiency": 0.78,
          "robustness": 0.88,
          "explainability": 0.85
      },
      "details": {
          # Detailed metric information
      }
  }
  ```

**Example:**
```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate(predictions, references)
print(f"Overall Score: {results['overall_score']:.3f}")
```

## Main Demo

The main demo script demonstrates the complete standardized data flow.

### File: `demo_refactored_system.py`

**Main Function:** `demo_standardized_pipeline()`

Executes the complete end-to-end pipeline:
1. Data loading
2. Preprocessing
3. Reasoning
4. Evaluation

**Usage:**
```bash
PYTHONPATH=src python demo_refactored_system.py
```

**Output:**
- Loaded sample count
- Preprocessing examples
- Individual reasoning results
- Final evaluation metrics

### Example Output

```
============================================================
🚀 标准化数据流 End-to-End Demo
============================================================
加载样本数: 5
已完成预处理。示例: {'id': 'Math23K_0', 'problem': '...', 'cleaned_text': '...', 'problem_type': 'arithmetic', 'complexity_level': 'L1'}

--- 推理样本 1 ---
问题: 小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？
推理结果: {'final_answer': '8', 'strategy_used': 'DIR', 'confidence': 0.95}

评测结果: {'overall_score': 0.85, 'metric_results': {...}}

🎉 标准化数据流 demo completed successfully!
```

## Error Handling

All components include comprehensive error handling:

- **DataLoader**: Handles missing datasets and invalid file formats
- **Preprocessor**: Gracefully handles malformed input text
- **ReasoningEngine**: Provides fallback strategies for unsolvable problems
- **Evaluator**: Handles mismatched prediction/reference pairs

## Configuration

Each component supports configuration dictionaries for customization:

```python
# Example configurations
data_config = {"cache_enabled": True, "max_samples": 100}
preprocessing_config = {"text_cleaning": True, "complexity_analysis": True}
reasoning_config = {"default_strategy": "auto", "timeout": 30}
evaluation_config = {"metric_weights": {"accuracy": 0.4, "efficiency": 0.2}}
```

## Performance Considerations

- **DataLoader**: Implements caching for frequently accessed datasets
- **Preprocessor**: Optimized text processing with regex patterns
- **ReasoningEngine**: Supports parallel processing for batch inference
- **Evaluator**: Efficient metric computation with vectorized operations

## Extending the API

To extend the system with new components:

1. Implement the standard interface (e.g., `load()`, `process()`, `solve()`, `evaluate()`)
2. Add appropriate error handling and validation
3. Include configuration support
4. Add comprehensive tests
5. Update this documentation 