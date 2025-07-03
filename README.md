# COT-DIR: Chain-of-Thought with Deep Implicit Relations

A modular mathematical reasoning system that implements standardized data flow pipelines for mathematical problem solving, evaluation, and analysis.

## Project Overview

This project provides a streamlined, modular architecture for mathematical reasoning with the following key features:

- **Standardized Data Flow**: Unified interfaces for data loading, preprocessing, reasoning, and evaluation
- **Multiple Reasoning Strategies**: Support for DIR, COT, DIR-COT, MLR, and other reasoning approaches
- **Comprehensive Evaluation**: Multi-metric evaluation including accuracy, reasoning quality, efficiency, and robustness
- **Modular Design**: Clean separation of concerns with adapter patterns for easy extension

## Directory Structure

```
cot-dir1/
├── demo_refactored_system.py    # Main entry point with standardized pipeline demo
├── src/                         # Core source code (minimal, modular)
│   ├── data/                    # Data loading and preprocessing
│   ├── evaluation/              # Evaluation and benchmarking
│   └── reasoning_core/          # Reasoning strategies and pipelines
├── Data/                        # Mathematical datasets
├── tests/                       # Test suites
├── requirements.txt             # Python dependencies
└── pytest.ini                  # Test configuration
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cot-dir1

# Install dependencies
pip install -r requirements.txt
```

### Running the Main Demo

```bash
# Run the standardized end-to-end pipeline demo
PYTHONPATH=src python demo_refactored_system.py
```

This will execute the complete standardized data flow:
1. **Data Loading**: Load mathematical problems from datasets
2. **Preprocessing**: Clean text, classify problem types, analyze complexity
3. **Reasoning**: Apply appropriate reasoning strategies (DIR/COT/DIR-COT/MLR)
4. **Evaluation**: Assess performance across multiple metrics

## Standardized Data Flow API

### DataLoader
```python
from src.data.loader import DataLoader

loader = DataLoader()
samples = loader.load(dataset_name="Math23K", max_samples=5)
```

### Preprocessor
```python
from src.data.preprocessor import Preprocessor

preprocessor = Preprocessor()
processed_samples = [preprocessor.process(sample) for sample in samples]
```

### ReasoningEngine
```python
from src.reasoning_core.reasoning_engine import ReasoningEngine

engine = ReasoningEngine()
results = [engine.solve(sample) for sample in processed_samples]
```

### Evaluator
```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator()
evaluation_results = evaluator.evaluate(predictions, references)
```

## Supported Datasets

- **Math23K**: Chinese mathematical word problems
- **GSM8K**: English grade school math problems
- **DIR-MWP**: Domain-specific implicit relation problems
- **MATH**: Competition-level mathematics
- And more...

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit_tests/
pytest tests/integration_tests/
```

## Contributing

1. Follow the standardized data flow architecture
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure compatibility with existing adapters

## Architecture Principles

- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Adapter Pattern**: Standardized interfaces wrap existing implementations
- **End-to-End Pipeline**: Complete data flow from loading to evaluation
- **Modularity**: Easy to extend with new reasoning strategies or evaluation metrics

## License

[Add your license information here]

## Citation

If you use this project in your research, please cite:

```bibtex
@software{cot_dir_2025,
  title={COT-DIR: Chain-of-Thought with Deep Implicit Relations},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
``` 