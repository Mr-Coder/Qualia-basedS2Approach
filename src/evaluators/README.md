# 评估器模块 (Evaluators)

## 概述

评估器模块提供了一套完整的评估工具，用于评估数学问题求解系统的各个方面的性能。该模块包含三个主要的评估器类，每个都专注于系统的不同评估维度。

## 模块结构

```
src/evaluators/
├── __init__.py                          # 模块初始化文件
├── performance_evaluator.py            # 性能评估器
├── relation_discovery_evaluator.py     # 关系发现评估器
├── reasoning_chain_evaluator.py        # 推理链评估器
└── README.md                           # 本文档
```

## 评估器类介绍

### 1. PerformanceEvaluator (性能评估器)

**功能**: 评估数学问题求解系统的整体性能和准确率。

**主要特性**:
- 整体准确率计算
- 按复杂度级别(L0-L3)的性能分析
- 鲁棒性分数计算
- 错误分析和分类
- 详细性能报告生成

**使用示例**:
```python
from evaluators import PerformanceEvaluator

evaluator = PerformanceEvaluator()

# 评估整体准确率
predictions = [15, 23, 8, 12, 45]
ground_truth = [15, 25, 8, 10, 45]
accuracy = evaluator.evaluate_overall_accuracy(predictions, ground_truth)

# 按复杂度级别评估
complexity_labels = ["L0", "L1", "L0", "L2", "L1"]
level_results = evaluator.evaluate_by_complexity_level(
    predictions, ground_truth, complexity_labels
)

# 计算鲁棒性分数
robustness = evaluator.calculate_robustness_score(level_results)
```

### 2. RelationDiscoveryEvaluator (关系发现评估器)

**功能**: 评估系统发现隐式关系的质量和准确性。

**主要特性**:
- 精确率、召回率、F1分数计算
- 语义准确性评估
- 按关系类型的详细分析
- 覆盖度指标计算
- 发现模式分析

**支持的关系类型**:
- mathematical_operations (数学运算) - 权重: 1.0
- unit_conversions (单位转换) - 权重: 0.9
- physical_constraints (物理约束) - 权重: 0.8
- temporal_relations (时间关系) - 权重: 0.7
- geometric_properties (几何属性) - 权重: 0.8
- proportional_relations (比例关系) - 权重: 0.9

**使用示例**:
```python
from evaluators import RelationDiscoveryEvaluator

evaluator = RelationDiscoveryEvaluator()

# 定义发现的关系和真实关系
discovered_relations = [
    {"type": "mathematical_operations", "match": "addition", "position": (5, 15)},
    {"type": "unit_conversions", "match": "meters to km", "position": (20, 35)}
]

true_relations = [
    {"type": "mathematical_operations", "match": "addition", "position": (5, 15)},
    {"type": "proportional_relations", "match": "speed-time", "position": (30, 45)}
]

# 评估关系发现质量
result = evaluator.evaluate_relation_discovery(discovered_relations, true_relations)
print(f"精确率: {result['precision']:.3f}")
print(f"召回率: {result['recall']:.3f}")
print(f"F1分数: {result['f1']:.3f}")
```

### 3. ReasoningChainEvaluator (推理链评估器)

**功能**: 评估推理链的质量和逻辑正确性。

**主要特性**:
- 五个维度的质量评估：
  - 逻辑正确性 (logical_correctness)
  - 完整性 (completeness)
  - 连贯性 (coherence)
  - 效率性 (efficiency)
  - 可验证性 (verifiability)
- 错误传播模式分析
- 错误阶段识别

**使用示例**:
```python
from evaluators import ReasoningChainEvaluator

evaluator = ReasoningChainEvaluator()

# 定义推理链
reasoning_chain = [
    {
        "type": "analysis",
        "content": "理解题目：计算15-3",
        "input": "15, 3",
        "operation": "problem_analysis",
        "output": "减法运算"
    },
    {
        "type": "calculation",
        "content": "执行计算：15 - 3 = 12",
        "input": "15, 3",
        "operation": "subtraction",
        "output": "12"
    }
]

# 评估推理链质量
quality_scores = evaluator.evaluate_reasoning_chain_quality(reasoning_chain)
print(f"总体质量: {quality_scores['overall']:.3f}")
print(f"逻辑正确性: {quality_scores['logical_correctness']:.3f}")
```

## 综合评估流程

模块支持多维度的综合评估，可以同时使用三个评估器对系统进行全面评估：

```python
from evaluators import PerformanceEvaluator, RelationDiscoveryEvaluator, ReasoningChainEvaluator

# 创建评估器
perf_eval = PerformanceEvaluator()
rel_eval = RelationDiscoveryEvaluator()
reason_eval = ReasoningChainEvaluator()

# 进行综合评估
performance_score = perf_eval.evaluate_overall_accuracy(predictions, ground_truth)
relation_score = rel_eval.evaluate_relation_discovery(discovered, true_relations)
reasoning_score = reason_eval.evaluate_reasoning_chain_quality(chain)

# 计算综合分数
overall_score = (
    performance_score * 0.4 +
    relation_score['f1'] * 0.3 +
    reasoning_score['overall'] * 0.3
)
```

## 输出格式

### 性能评估结果
```json
{
  "overall_accuracy": 0.8000,
  "total_samples": 10,
  "correct_predictions": 8,
  "complexity_level_accuracy": {
    "L0": 0.9000,
    "L1": 0.7500,
    "L2": 0.6667,
    "L3": 0.5000
  },
  "robustness_score": 0.5556,
  "error_analysis": {
    "total_errors": 2,
    "error_types": {
      "minor_numerical_error": 1,
      "major_numerical_error": 1
    }
  }
}
```

### 关系发现评估结果
```json
{
  "precision": 0.7500,
  "recall": 0.6000,
  "f1": 0.6667,
  "semantic_accuracy": 0.8000,
  "weighted_f1": 0.7200,
  "avg_relations": 4,
  "true_relations_count": 5,
  "correct_relations": 3,
  "false_positives": 1,
  "false_negatives": 2
}
```

### 推理链评估结果
```json
{
  "logical_correctness": 0.9000,
  "completeness": 0.8000,
  "coherence": 0.8500,
  "efficiency": 0.7500,
  "verifiability": 0.8000,
  "overall": 0.8200
}
```

## 配置和自定义

### 关系类型权重配置
可以通过修改 `RelationDiscoveryEvaluator` 的 `relation_type_weights` 属性来调整不同关系类型的重要性：

```python
evaluator = RelationDiscoveryEvaluator()
evaluator.relation_type_weights = {
    "mathematical_operations": 1.0,
    "unit_conversions": 0.8,
    "physical_constraints": 0.9,
    # ... 其他类型
}
```

### 推理链步骤类型
推理链评估器支持以下步骤类型：
- `analysis`: 问题分析
- `method`: 方法选择
- `calculation`: 计算步骤
- `operation`: 操作步骤
- `inference`: 推理步骤
- `verification`: 验证步骤
- `conclusion`: 结论步骤

## 日志和调试

所有评估器都配置了详细的日志记录：

```python
import logging

# 配置日志级别
logging.basicConfig(level=logging.DEBUG)

# 运行评估器时会输出详细信息
evaluator = PerformanceEvaluator()
# 日志输出: "PerformanceEvaluator initialized"
# 日志输出: "Overall accuracy: 0.8000 (8/10)"
```

## 错误处理

所有评估器都包含完善的错误处理机制：

- 输入验证：检查输入数据的有效性和一致性
- 异常捕获：捕获并记录运行时错误
- 默认值：在出错时返回合理的默认值
- 日志记录：记录错误信息用于调试

## 扩展性

模块设计支持轻松扩展：

1. **添加新的评估维度**: 在相应评估器中添加新方法
2. **支持新的关系类型**: 在关系发现评估器中添加新类型
3. **自定义评估指标**: 继承基础评估器类并重写方法
4. **集成新的评估器**: 在 `__init__.py` 中添加新的评估器类

## 性能考虑

- 所有评估器都针对大规模数据进行了优化
- 支持批量处理以提高效率
- 内存使用经过优化，适合处理大型数据集
- 提供进度跟踪和性能监控功能

## 依赖项

模块依赖以下Python包：
- `json`: JSON数据处理
- `logging`: 日志记录
- `statistics`: 统计计算
- `collections`: 数据结构工具
- `typing`: 类型注解
- `re`: 正则表达式（关系发现评估器）

所有依赖项都是Python标准库的一部分，无需额外安装。 