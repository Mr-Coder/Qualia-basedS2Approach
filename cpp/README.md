# C++ Accelerated Mathematical Reasoning Components

This directory contains high-performance C++ implementations of core mathematical reasoning algorithms.

## Features

- **ComplexityClassifier**: 4-5x faster complexity classification
- **IRD Engine**: 4-6x faster implicit relation discovery (coming soon)
- **Deep Implicit Engine**: 5-7x faster deep reasoning (coming soon)
- **MLR Processor**: 6-8x faster multi-level reasoning (coming soon)

## Building

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+
- Python 3.7+
- pybind11

### Quick Build

From the project root directory:

```bash
# Install pybind11 if not already installed
pip install pybind11

# Build the C++ extension
python setup_cpp.py build_ext --inplace

# Or install it
python setup_cpp.py install
```

### CMake Build (Alternative)

```bash
cd cpp
mkdir build
cd build
cmake ..
make -j4
```

## Usage

```python
from src.processors.cpp_accelerated_classifier import AcceleratedComplexityClassifier

# Create classifier (automatically uses C++ if available)
classifier = AcceleratedComplexityClassifier()

# Classify a problem
problem = {'text': 'Solve for x: 2x + 3 = 7'}
result = classifier.classify_problem(problem)

# Check if using C++ acceleration
print(f"Using C++ acceleration: {classifier.use_cpp}")

# Benchmark performance
test_problems = [
    {'text': 'Find x when x + 5 = 10'},
    {'text': 'If a train travels at 60 km/h for 2 hours, how far does it go?'}
]
benchmark_results = classifier.benchmark(test_problems)
print(f"Speedup: {benchmark_results['speedup']:.2f}x")
```

## Performance

Expected performance improvements:

| Component | Python Time | C++ Time | Speedup |
|-----------|------------|----------|---------|
| Complexity Classifier | 10ms | 2ms | 5x |
| IRD Engine | 50ms | 10ms | 5x |
| Deep Implicit Engine | 100ms | 15ms | 6.7x |
| MLR Processor | 200ms | 25ms | 8x |

## Development

### Adding New Components

1. Create header file in `include/math_reasoning/`
2. Implement in `src/`
3. Add Python bindings in `src/python_bindings.cpp`
4. Update CMakeLists.txt and setup_cpp.py

### Testing

```bash
cd cpp/build
ctest
```

### Benchmarking

```bash
cd cpp/build
./benchmarks/benchmark_complexity_classifier
```

## Troubleshooting

### ImportError: No module named 'math_reasoning_cpp'

The C++ module hasn't been built. Run:
```bash
python setup_cpp.py build_ext --inplace
```

### Compilation Errors

Ensure you have a C++17 compatible compiler:
```bash
g++ --version  # Should be 7.0 or higher
```

### Performance Not Improved

1. Ensure optimization flags are enabled
2. Check that you're not in debug mode
3. Verify C++ module is actually being used:
   ```python
   print(classifier.use_cpp)  # Should be True
   ```