# MATH Dataset

## 数据集概述
MATH是一个高难度的竞赛数学数据集，包含12,500个竞赛级别的数学问题，涵盖7个主要数学领域。

## 数据格式
```json
{
  "problem": "问题描述",
  "solution": "详细解答步骤",
  "answer": "最终答案",
  "subject": "数学主题",
  "level": "难度等级",
  "unique_id": "唯一标识"
}
```

## 统计信息
- **总问题数**: 12,500
- **训练集**: 7,500
- **测试集**: 5,000  
- **语言**: 英文
- **难度级别**: 竞赛数学水平

## 数学主题
- **Prealgebra**: 代数预备 (938题)
- **Algebra**: 代数 (1,187题)
- **Number Theory**: 数论 (540题)
- **Counting & Probability**: 计数与概率 (474题)
- **Geometry**: 几何 (479题)
- **Intermediate Algebra**: 中级代数 (903题)
- **Precalculus**: 微积分预备 (546题)

## 难度等级
- **Level 1**: 最简单
- **Level 2**: 简单
- **Level 3**: 中等
- **Level 4**: 困难  
- **Level 5**: 最困难

## 数据特点
- 高质量的竞赛级数学问题
- 详细的step-by-step解答
- LaTeX格式的数学表达式
- 多样化的数学概念
- 适合评估高级数学推理能力

## 使用说明
1. 数据文件: `math_dataset.json`
2. 评估指标: 准确率(Accuracy)
3. 基准模型: GPT-3, Minerva等 