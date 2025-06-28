# ASDiv Dataset (Academic Subjects Diversity)

## 数据集概述
ASDiv是一个多样化的学术数学文字题数据集，包含2,305个不同学科背景的数学问题，注重跨学科的数学应用。

## 数据格式
```json
{
  "id": "problem_id",
  "body": "问题背景描述",
  "question": "具体问题",
  "answer": "数值答案",
  "formula": "求解公式",
  "grade": "年级水平",
  "subject": "学科领域",
  "type": "问题类型"
}
```

## 统计信息
- **总问题数**: 2,305
- **语言**: 英文
- **年级跨度**: 3-12年级
- **学科覆盖**: 数学、科学、社会研究等

## 学科分布
- **Mathematics**: 纯数学问题
- **Science**: 科学应用数学
- **Social Studies**: 社会研究中的数学
- **Language Arts**: 语言艺术中的数学
- **Health**: 健康相关的数学
- **Arts**: 艺术中的数学应用

## 问题类型
- **Addition**: 加法
- **Subtraction**: 减法  
- **Multiplication**: 乘法
- **Division**: 除法
- **Mixed Operations**: 混合运算
- **Fractions**: 分数
- **Decimals**: 小数
- **Percentages**: 百分比

## 数据特点
- 跨学科数学应用
- 真实世界场景
- 多样化的问题背景
- 分级难度设计
- 支持学科交叉研究

## 使用说明
1. 数据文件: `asdiv.json`
2. 评估指标: 准确率(Accuracy)
3. 特色: 学科多样性分析 