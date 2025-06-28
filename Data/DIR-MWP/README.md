# DIR-MWP Dataset

## Overview

The DIR-MWP (Domain-specific Implicit Relation Math Word Problems) dataset is a comprehensive collection of 200 math word problems designed to evaluate mathematical reasoning systems across different complexity levels. The dataset focuses on problems that require understanding of implicit relationships and domain-specific knowledge.

## Dataset Structure

### Complexity Levels

The dataset is organized into 4 complexity levels with specific characteristics:

| Level | Count | Avg Relations | Inference Depth | Description |
|-------|-------|---------------|-----------------|-------------|
| **L0_explicit** | 30 (15%) | 1.2 | 1.0 | 简单算术问题，所有关系都明确给出 |
| **L1_shallow** | 50 (25%) | 2.1 | 2.3 | 需要单步推理或基本单位转换 |
| **L2_medium** | 80 (40%) | 3.2 | 3.8 | 需要2-3步推理和基本领域知识 |
| **L3_deep** | 40 (20%) | 4.5 | 5.2 | 需要>3步推理和复杂领域知识 |

### File Structure

```
Data/DIR-MWP/
├── dir_mwp_complete_dataset.json    # Complete dataset with all 200 problems
├── dir_mwp_analysis.csv            # Analysis data in CSV format
├── README.md                       # This documentation file
└── (additional analysis files)
```

## Dataset Schema

Each problem in the dataset contains the following fields:

```json
{
  "id": "L0_001",                           // Unique identifier
  "complexity_level": "L0_explicit",        // Complexity level
  "problem": "问题描述...",                  // Problem statement in Chinese
  "answer": "23",                           // Correct answer
  "solution_steps": [                       // Step-by-step solution
    "步骤1: ...",
    "步骤2: ..."
  ],
  "explicit_relations": [                   // Explicitly stated relationships
    "关系1", "关系2"
  ],
  "implicit_relations": [                   // Implicitly required relationships
    "隐含关系1", "隐含关系2"
  ],
  "domain_knowledge": [                     // Required domain knowledge
    "领域知识1", "领域知识2"
  ],
  "inference_depth": 3,                     // Number of inference steps required
  "relation_count": 4                       // Total number of relations involved
}
```

## Complexity Level Details

### L0 - Explicit Problems
- **Characteristics**: Direct arithmetic operations with clearly stated relationships
- **Examples**: Simple addition, subtraction, multiplication, division
- **Domain Knowledge**: None required
- **Inference Steps**: 1 step
- **Sample**: "小明有15个苹果，小红有8个苹果。他们一共有多少个苹果？"

### L1 - Shallow Problems
- **Characteristics**: Single-step inference with basic formulas and unit conversions
- **Examples**: Speed-distance-time, area calculations, price calculations
- **Domain Knowledge**: Basic mathematical formulas
- **Inference Steps**: 2 steps
- **Sample**: "一辆汽车每小时行驶60公里，行驶了2.5小时。这辆汽车总共行驶了多少公里？"

### L2 - Medium Problems
- **Characteristics**: Multi-step reasoning requiring domain knowledge
- **Examples**: Fluid mechanics, age problems, production planning
- **Domain Knowledge**: Specific domain concepts
- **Inference Steps**: 3-4 steps
- **Sample**: "一个水箱可以装500升水。现在水箱里有水200升，水龙头每分钟流入15升水。需要多少分钟才能把水箱装满？"

### L3 - Deep Problems
- **Characteristics**: Complex multi-step inference with advanced concepts
- **Examples**: Thermodynamics, complex geometry, exponential decay
- **Domain Knowledge**: Advanced mathematical and scientific concepts
- **Inference Steps**: 5-6 steps
- **Sample**: "在20°C的环境中，有一块重量为500克的冰块。已知冰的融化潜热为334焦耳/克，环境每分钟向冰块传递热量50焦耳。冰块完全融化需要多少分钟？"

## Domain Knowledge Areas

The dataset covers various domain knowledge areas:

1. **Physics**: Thermodynamics, mechanics, heat transfer
2. **Geometry**: Area calculations, volume calculations, surface area
3. **Chemistry**: Reaction kinetics, exponential decay
4. **Economics**: Price calculations, cost analysis
5. **Production**: Manufacturing planning, inventory management
6. **Time**: Age problems, scheduling
7. **Unit Conversion**: Length, area, time units

## Usage

### Loading the Dataset

```python
import json

# Load the complete dataset
with open('Data/DIR-MWP/dir_mwp_complete_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Access problems
problems = dataset['problems']

# Filter by complexity level
l3_problems = [p for p in problems if p['complexity_level'] == 'L3_deep']
```

### Evaluation Metrics

When evaluating models on this dataset, consider:

1. **Accuracy by Complexity Level**: Track performance across L0-L3
2. **Inference Chain Quality**: Evaluate step-by-step reasoning
3. **Domain Knowledge Application**: Assess understanding of implicit relations
4. **Error Analysis**: Categorize errors by complexity and domain

### Validation

Use the provided validation script to verify dataset integrity:

```bash
python validate_dir_mwp_dataset.py
```

## Dataset Statistics

- **Total Problems**: 200
- **Average Inference Depth**: 3.13
- **Languages**: Chinese (问题文本), English (schema and documentation)
- **Domains Covered**: 10+ different knowledge areas
- **Problem Types**: Word problems requiring mathematical reasoning

## Research Applications

This dataset is suitable for:

1. **Mathematical Reasoning Evaluation**: Testing LLM reasoning capabilities
2. **Domain Knowledge Assessment**: Evaluating implicit relation understanding
3. **Complexity Analysis**: Studying performance degradation across difficulty levels
4. **Chain-of-Thought Research**: Analyzing step-by-step reasoning patterns
5. **Educational Research**: Understanding learning progression in math problems

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{dir_mwp_2025,
  title={DIR-MWP: Domain-specific Implicit Relation Math Word Problems Dataset},
  author={Generated for Research},
  year={2025},
  version={1.0},
  description={A comprehensive dataset of 200 math word problems across 4 complexity levels}
}
```

## License

This dataset is provided for research and educational purposes. Please ensure appropriate attribution when using or modifying the dataset.

## Contact

For questions about the dataset or to report issues, please refer to the dataset validation script or create an issue in the repository.

---

**Generated on**: 2025-01-31  
**Version**: 1.0  
**Total Problems**: 200  
**Complexity Levels**: 4 (L0-L3) 