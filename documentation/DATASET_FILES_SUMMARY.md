# Dataset Files Summary

基于您提供的表格"Table 3: Dataset Characteristics with DIR-MWP Complexity Distribution"，我已经为您生成了一个完整的数据集特征管理系统。

## 📊 生成的文件列表

### 🐍 Python模块文件
- **`src/data/dataset_characteristics.py`** - 核心数据模块，包含所有数据集特征信息和分析功能
- **`src/data/__init__.py`** - 数据模块初始化文件
- **`src/data/export_utils.py`** - 导出工具模块
- **`src/examples/dataset_analysis_example.py`** - 完整使用示例

### 📄 数据文件
- **`dataset_characteristics.json`** - JSON格式的完整数据集特征
- **`dataset_characteristics.csv`** - CSV格式的数据集特征表格
- **`dataset_statistics.csv`** - 统计摘要CSV文件
- **`complexity_matrix.csv`** - 复杂度分布矩阵CSV文件
- **`dataset_table.md`** - Markdown格式的美观表格

### 🛠️ 工具脚本
- **`generate_dataset_files.py`** - 生成所有格式数据文件的独立脚本

## 📋 包含的数据集

系统包含了表格中的所有8个数据集：

| 数据集 | 规模 | 语言 | 领域 | DIR评分 |
|--------|------|------|------|---------|
| Math23K | 23,162 | 中文 | 小学 | 2.03 |
| GSM8K | 8,500 | 英文 | 小学年级 | 1.98 |
| MAWPS | 2,373 | 英文 | 多领域 | 1.88 |
| MathQA | 37,297 | 英文 | 竞赛 | 2.07 |
| MATH | 12,500 | 英文 | 竞赛 | 2.26 |
| SVAMP | 1,000 | 英文 | 小学年级 | 2.14 |
| ASDiv | 2,305 | 英文 | 小学 | 1.98 |
| DIR-MWP-Test | 1,200 | 混合 | 专门化 | 2.70 |

## 🚀 主要功能

### 1. 数据查询和筛选
```python
from src.data import get_dataset_info, get_datasets_by_language

# 获取特定数据集信息
math23k = get_dataset_info("Math23K")

# 按语言筛选数据集
english_datasets = get_datasets_by_language("English")
```

### 2. 复杂度分析
```python
from src.data import get_complexity_distribution, calculate_weighted_complexity_score

# 获取复杂度分布
complexity = get_complexity_distribution("MATH")
# 结果: {"L0": 28.4, "L1": 31.7, "L2": 25.1, "L3": 14.8}

# 计算加权复杂度评分
score = calculate_weighted_complexity_score("MATH")
# 结果: 1.26
```

### 3. 统计分析
```python
from src.data import get_dataset_statistics

stats = get_dataset_statistics()
# 获取总体统计信息，包括：
# - 总数据集数量: 8
# - 总问题数量: 88,337
# - 平均DIR评分: 2.13
# - 语言分布
# - 领域分布
# - 平均复杂度分布
```

### 4. 数据导出
```python
from src.data import export_to_json

# 导出为JSON格式
export_to_json("my_dataset_analysis.json")
```

## 📈 分析洞察

### 复杂度排名 (从简单到困难)
1. **MAWPS** (加权: 0.88, DIR: 1.88) - 最简单
2. **GSM8K** (加权: 0.98, DIR: 1.98)
3. **ASDiv** (加权: 0.98, DIR: 1.98)
4. **Math23K** (加权: 1.03, DIR: 2.03)
5. **MathQA** (加权: 1.07, DIR: 2.07)
6. **SVAMP** (加权: 1.14, DIR: 2.14)
7. **MATH** (加权: 1.26, DIR: 2.26)
8. **DIR-MWP-Test** (加权: 1.70, DIR: 2.70) - 最困难

### 语言分布
- **英文数据集**: 6个 (63,975个问题)
- **中文数据集**: 1个 (23,162个问题)
- **混合语言**: 1个 (1,200个问题)

### 领域分布
- **竞赛类**: 2个数据集 (49,797个问题) - 平均DIR: 2.17
- **小学**: 2个数据集 (25,467个问题) - 平均DIR: 2.00
- **小学年级**: 2个数据集 (9,500个问题) - 平均DIR: 2.06
- **多领域**: 1个数据集 (2,373个问题) - 平均DIR: 1.88
- **专门化**: 1个数据集 (1,200个问题) - 平均DIR: 2.70

## 🔧 使用方法

### 运行分析示例
```bash
python src/examples/dataset_analysis_example.py
```

### 生成所有格式文件
```bash
python generate_dataset_files.py
```

### 在代码中使用
```python
# 导入模块
from src.data import DATASET_CHARACTERISTICS, get_all_datasets

# 查看所有数据集
datasets = get_all_datasets()
for name, info in datasets.items():
    print(f"{name}: {info.size} problems, DIR: {info.dir_score}")
```

## 📁 文件结构

```
newfile/
├── src/
│   ├── data/
│   │   ├── __init__.py                    # 模块初始化
│   │   ├── dataset_characteristics.py     # 核心数据模块
│   │   └── export_utils.py               # 导出工具
│   └── examples/
│       ├── dataset_analysis_example.py   # 使用示例
│       └── evaluator_usage_example.py    # 评估器示例
├── dataset_characteristics.json           # JSON数据
├── dataset_characteristics.csv            # CSV数据
├── dataset_statistics.csv                # 统计摘要
├── complexity_matrix.csv                 # 复杂度矩阵
├── dataset_table.md                      # Markdown表格
└── generate_dataset_files.py             # 生成脚本
```

## 🎯 应用场景

这个数据集特征系统可以用于：

1. **研究分析** - 比较不同数据集的特征和复杂度
2. **模型评估** - 根据数据集特征选择合适的测试集
3. **数据选择** - 为特定研究目的筛选合适的数据集
4. **性能分析** - 分析模型在不同复杂度级别上的表现
5. **报告生成** - 自动生成数据集特征报告

## 💡 扩展建议

- 可以轻松添加新的数据集到`DATASET_CHARACTERISTICS`字典中
- 支持添加新的分析维度和统计指标
- 可以扩展导出格式（如Excel、XML等）
- 支持可视化功能（图表生成）

这个系统为您的数学问题求解项目提供了完整的数据集管理和分析功能！ 