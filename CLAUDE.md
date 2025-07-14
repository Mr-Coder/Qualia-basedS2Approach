# Claude Instructions for COT-DIR Project

## Project Overview
This is a mathematical reasoning system implementing COT-DIR (Chain-of-Thought Directed Implicit Reasoning) for solving math word problems.

## Key Commands

### Testing
```bash
pytest tests/
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
```

### Linting & Type Checking
```bash
# Add specific commands when available - check requirements.txt for tools
```

### Running Demos
```bash
python demos/simple_reasoning_demo.py
python demos/template_system_demo.py
python demos/reasoning_refactor_demo.py
```

## Project Structure
- `src/` - Main source code
  - `reasoning/` - Core reasoning engine and strategies
  - `template_management/` - Template processing system
  - `processors/` - Data processing and NLP components
  - `models/` - Model definitions and interfaces
  - `evaluation/` - Evaluation and benchmarking tools
- `tests/` - Test suites (unit, integration, system)
- `demos/` - Example usage demonstrations
- `Data/` - Datasets and experimental data
- `config/` - Configuration files

## Important Notes
- This project handles mathematical word problems using implicit relation discovery
- The system supports multiple reasoning strategies (COT, GOT, TOT)
- Template management is a core component for problem processing
- GNN enhancement modules are available for graph-based reasoning

## Development Guidelines
- Follow existing code patterns in the modular architecture
- Use type hints and proper error handling
- Test changes with both unit and integration tests
- Check demos work after making changes