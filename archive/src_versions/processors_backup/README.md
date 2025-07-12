# Processors 模块

这个模块包含了数学问题求解系统的核心处理器组件。

## 新增组件

### 1. DatasetLoader (数据集加载器)

**位置**: `src/processors/dataset_loader.py`

**功能**: 加载和标准化各种数学题数据集

**支持的数据集**:
- Math23K: 中文数学题数据集 (23,162题)
- GSM8K: 英文小学数学题 (8,500题) 
- MAWPS: 多领域数学题 (2,373题)
- MathQA: 竞赛数学题 (37,297题)
- MATH: 竞赛数学题 (12,500题)
- SVAMP: 小学数学题 (1,000题)
- ASDiv: 小学数学题 (2,305题)

**使用示例**:
```python
from processors.dataset_loader import DatasetLoader

loader = DatasetLoader()
data = loader.load_math23k("path/to/math23k.json")
stats = loader.get_dataset_stats()
```

### 2. ComplexityClassifier (复杂度分类器)

**位置**: `src/processors/complexity_classifier.py`

**功能**: 分析数学问题的复杂度并计算DIR分数

**复杂度级别**:
- L0: 显式问题 (δ=0, κ=0)
- L1: 浅层隐式 (δ=1, κ≤1)
- L2: 中等隐式 (1<δ≤3, κ≤2)
- L3: 深度隐式 (δ>3 或 κ>2)

**使用示例**:
```python
from processors.complexity_classifier import ComplexityClassifier

classifier = ComplexityClassifier()
level = classifier.classify_problem_complexity("问题文本")
dir_score, distribution = classifier.calculate_dir_score(dataset)
```

### 3. ImplicitRelationAnnotator (隐式关系标注器)

**位置**: `src/processors/implicit_relation_annotator.py`

**功能**: 识别和标注数学问题中的隐式关系

**关系类型**:
- 数学运算关系 (35.2%)
- 单位转换关系 (18.7%)
- 物理约束关系 (16.4%)
- 时间关系 (12.3%)
- 几何属性关系 (10.8%)
- 比例关系 (6.6%)

**使用示例**:
```python
from processors.implicit_relation_annotator import ImplicitRelationAnnotator

annotator = ImplicitRelationAnnotator()
relations = annotator.annotate_implicit_relations("问题文本")
annotated_problems = annotator.create_ground_truth_relations(problems)
```

## 完整使用流程

```python
# 1. 加载数据集
loader = DatasetLoader()
problems = loader.load_math23k("data/math23k.json")

# 2. 分析复杂度
classifier = ComplexityClassifier()
classified_problems = classifier.batch_classify_problems(problems)
complexity_analysis = classifier.analyze_dataset_complexity(classified_problems)

# 3. 标注隐式关系
annotator = ImplicitRelationAnnotator()
annotated_problems = annotator.create_ground_truth_relations(classified_problems)
relation_analysis = annotator.analyze_relation_distribution(annotated_problems)

# 4. 导出结果
classifier.export_complexity_analysis(complexity_analysis, "complexity_results.json")
annotator.export_annotations(annotated_problems, "annotations.json")
```

## 示例文件

查看 `src/examples/dataset_analysis_example.py` 获取完整的使用示例。

## 输出格式

### 复杂度分析结果
```json
{
  "dir_score": 1.25,
  "total_problems": 100,
  "level_distribution": {
    "L0": 20,
    "L1": 30,
    "L2": 35,
    "L3": 15
  },
  "level_percentages": {
    "L0": 20.0,
    "L1": 30.0,
    "L2": 35.0,
    "L3": 15.0
  },
  "complexity_summary": "中等复杂度数据集（隐式问题分布均匀）"
}
```

### 隐式关系标注结果
```json
{
  "type": "mathematical_operations",
  "pattern": "总共|一共|合计",
  "match": "一共",
  "position": [15, 17]
}
```

## 配置要求

确保安装了以下依赖:
- Python 3.8+
- json
- re
- pathlib
- logging
- typing
- collections

## 注意事项

1. 数据集文件路径需要根据实际情况调整
2. 支持 JSON 和 JSONL 格式的数据文件
3. 所有组件都包含详细的日志记录
4. 错误处理机制确保程序稳定运行 