# 精简API使用指南

## 🎯 精简完成！

经过精简，API从原来的30+个减少到**11个核心API**，去除了约80%的功能重复。

## 📊 精简结果总览

### ✅ 保留的核心API (11个)

```
核心API架构
├── 🏗️ 核心推理引擎 (3个)
│   ├── COTDIRIntegratedWorkflow     # ⭐⭐⭐⭐⭐ 完整工作流
│   ├── MLRMultiLayerReasoner       # ⭐⭐⭐⭐ 多层推理  
│   └── COTDIRModel                 # ⭐⭐⭐⭐ 模型接口
├── 🔧 核心工具类 (4个)
│   ├── ComplexityAnalyzer          # ⭐⭐⭐⭐ 复杂度分析
│   ├── RelationDiscoveryTool       # ⭐⭐⭐⭐ 关系发现
│   ├── BaseTool                    # ⭐⭐⭐ 工具基类
│   └── VisualizationTool           # ⭐⭐⭐ 可视化工具
├── 📊 核心数据结构 (2个)
│   ├── ReasoningResult & MathProblem
│   └── Entity & Relation & ReasoningStep
└── 🎨 精选演示 (2个)
    ├── single_question_demo.py     # ⭐⭐⭐⭐⭐ 算法演示
    └── experimental_framework.py   # ⭐⭐⭐⭐ 实验框架
```

### ❌ 已移除的重复API (19个)

- **重复演示程序** (8个): complete_cotdir_demo.py, interactive_demo.py 等
- **重复推理策略** (3个): ChainOfThoughtStrategy, EnhancedCOTDIRStrategy 等  
- **重复工具类** (2个): NumericalComputeTool, SymbolicMathTool
- **冗余文档** (6个): 各种重复的API概述文档

---

## 🚀 快速开始

### **基础使用 (推荐 - 80%场景)**

```python
# 1. 导入核心API
from src.reasoning_engine.cotdir_integration import COTDIRIntegratedWorkflow

# 2. 创建工作流实例
workflow = COTDIRIntegratedWorkflow()

# 3. 处理数学问题  
question = "小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？"
result = workflow.process(question)

# 4. 获取结果
print(f"答案: {result['answer']['value']}")
print(f"置信度: {result['overall_confidence']:.1%}")
print(f"推理步骤: {len(result['reasoning_process']['steps'])}")
```

## 🎯 总结

**精简成功！** 通过移除功能重复的API，系统现在：

- 🚀 **更易使用**: 从30+个API简化到11个核心API
- 🔧 **更易维护**: 减少代码冗余，提升开发效率  
- 📊 **更加专注**: 突出COT-DIR核心算法特色
- 💡 **更好扩展**: 清晰架构便于未来功能添加

**推荐开始使用**: `COTDIRIntegratedWorkflow` 是最佳入门选择！
