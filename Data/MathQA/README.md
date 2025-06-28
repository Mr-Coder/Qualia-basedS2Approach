# MathQA Dataset

## 数据集概述
MathQA是一个大规模的数学问答数据集，包含37,000+个问题，涵盖多种数学主题和解题步骤。

## 数据格式
```json
{
  "id": "problem_id",
  "problem": "问题描述",
  "rationale": "解题步骤",
  "options": ["A) 选项1", "B) 选项2", "C) 选项3", "D) 选项4", "E) 选项5"],
  "correct": "正确答案选项",
  "category": "数学类别",
  "annotated_formula": "注释公式",
  "linear_formula": "线性公式"
}
```

## 统计信息
- **总问题数**: 37,297
- **语言**: 英文
- **问题类型**: 代数、几何、概率、统计等
- **难度级别**: 高中到大学水平

## 数学类别
- **Algebra**: 代数
- **Geometry**: 几何  
- **Probability**: 概率
- **Number Theory**: 数论
- **Statistics**: 统计
- **Arithmetic**: 算术
- **Physics**: 物理应用

## 数据特点
- 多步骤推理解题过程
- 多选题格式
- 详细的解题rationale
- 公式化表示
- 支持程序求解

## 使用说明
1. 数据文件: `mathqa.json`
2. 评估指标: 选择题准确率
3. 任务类型: 选择题问答、公式生成 