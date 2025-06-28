# MAWPS Dataset (Math Word Problem Solver)

## 数据集概述
MAWPS是一个综合性的数学文字题求解数据集，整合了多个现有数据集，包含2,373个数学文字题。

## 数据格式
```json
{
  "id": "problem_id",
  "body": "问题描述文本",
  "question": "具体问题",
  "answer": "数值答案",
  "equation": "求解方程",
  "numbers": ["问题中的数字"],
  "nouns": ["问题中的名词"],
  "grade": "年级水平"
}
```

## 统计信息
- **总问题数**: 2,373
- **语言**: 英文
- **来源数据集**: AddSub, MultiArith, SingleEq, SingleOp等
- **难度级别**: 小学水平

## 数据特点
- 标准化的数据格式
- 包含问题分解（body + question）
- 提供数字和名词标注
- 适合算法评估和比较

## 使用说明
1. 数据文件: `mawps.json`
2. 评估指标: 准确率(Accuracy)
3. 基准方法: Seq2Seq, GTS, DNS等 