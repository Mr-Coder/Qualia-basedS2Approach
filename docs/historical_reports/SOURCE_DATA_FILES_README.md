# 源数据文件说明

## 概述

本目录包含从 `src/data` 模块自动生成的各个表格对应的源数据文件。这些文件是基于论文中实验数据生成的结构化数据，可用于进一步分析、可视化和报告生成。

## 生成脚本

- **生成脚本**: `generate_source_data_files.py`
- **源数据模块**: `src/data/dataset_characteristics.py` 和 `src/data/performance_analysis.py`
- **生成时间**: 2025-01-31

## 文件列表

### Table 3: 数据集特征分析
- `table3_dataset_characteristics.json` - 结构化JSON格式
- `table3_dataset_characteristics.csv` - CSV表格格式
- `table3_dataset_characteristics.md` - Markdown表格格式

**内容**: 8个数据集的详细特征，包括规模、语言、领域、复杂度分布(L0-L3)和DIR评分

### Table 4: 整体性能对比
- `table4_performance_comparison.json` - 结构化JSON格式
- `table4_performance_comparison.csv` - CSV表格格式  
- `table4_performance_comparison.md` - Markdown表格格式

**内容**: 7种方法在8个数据集上的性能表现，包括平均分

### Table 5: 复杂度性能分析
- `table5_complexity_performance.json` - 结构化JSON格式
- `table5_complexity_performance.csv` - CSV表格格式
- `table5_complexity_performance.md` - Markdown表格格式

**内容**: 各方法在不同复杂度级别(L0-L3)的性能，鲁棒性评分和复杂度下降幅度

### Table 6: 隐式关系发现质量评估
- `table6_relation_discovery.json` - 结构化JSON格式
- `table6_relation_discovery.csv` - CSV表格格式

**内容**: 各方法的关系发现精度、召回率、F1分数、语义准确率等指标

### Table 7: 推理链质量评估
- `table7_reasoning_chain.json` - 结构化JSON格式
- `table7_reasoning_chain.csv` - CSV表格格式

**内容**: 5个质量维度评估：逻辑正确性、完整性、连贯性、效率、可验证性

### Table 8: 消融研究
- `table8_ablation_study.json` - 结构化JSON格式
- `table8_ablation_study.csv` - CSV表格格式

**内容**: COT-DIR各组件的贡献分析，包括完整配置和各种简化配置的性能对比

### Table 9: 组件交互分析
- `table9_component_interaction.json` - 结构化JSON格式
- `table9_component_interaction.csv` - CSV表格格式

**内容**: 不同组件组合的协同效应分析，包括IRD、MLR、CV的各种组合

### Table 10: 计算效率分析
- `table10_efficiency_analysis.json` - 结构化JSON格式
- `table10_efficiency_analysis.csv` - CSV表格格式

**内容**: 各方法的运行时间、内存使用和效率评分

## 数据统计

- **数据集总数**: 8个
- **评估方法总数**: 7种
- **消融配置数**: 7种
- **组件交互组合数**: 4种
- **生成文件总数**: 21个

## 主要发现

### 最佳性能方法
- **整体最佳**: COT-DIR (85.3% 平均准确率)
- **最佳鲁棒性**: COT-DIR (0.82 鲁棒性评分)
- **最佳关系发现**: COT-DIR (0.80 F1分数)
- **最佳推理链质量**: COT-DIR (0.92 总分)

### 关键组件贡献
1. **IRD (隐式关系发现)**: +7.6% 性能提升
2. **MLR (多层推理)**: +5.5% 性能提升  
3. **CV (交叉验证)**: +2.8% 性能提升

### 数据集难度排序
1. **最简单**: MAWPS (DIR评分: 1.88)
2. **最困难**: DIR-MWP-Test (DIR评分: 2.70)

## 使用说明

1. **CSV文件**: 适用于Excel、数据分析软件直接导入
2. **JSON文件**: 适用于程序化处理、API数据交换
3. **Markdown文件**: 适用于文档展示、GitHub显示

## 文件格式示例

### CSV格式示例 (Table 4)
```csv
Method,Math23K,GSM8K,MAWPS,MathQA,MATH,SVAMP,ASDiv,DIR-Test,Average
COT-DIR,87.3,91.2,94.1,80.4,68.7,89.3,92.8,78.5,85.3
```

### JSON格式示例 (Table 6)
```json
{
  "COT-DIR": {
    "method_name": "COT-DIR",
    "overall_metrics": {
      "precision": 0.82,
      "recall": 0.79,
      "f1_score": 0.80
    }
  }
}
```

## 汇总文件

`source_data_summary.json` 包含了所有生成文件的索引和统计信息，便于整体管理和批量处理。 