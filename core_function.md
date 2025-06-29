# DIR-COT 项目解答过程核心功能模块

本文件梳理了 DIR-COT（Directed Implicit Reasoning - Chain of Thought）数学推理系统在解答流程中的核心功能模块及其作用。

## 🏗️ 核心源码架构 (`src/`)

```
src/
├── reasoning_engine/          # 推理引擎核心
│   ├── cotdir_integration.py  # COT-DIR集成框架
│   ├── mlr_reasoner.py       # 多层推理器
│   └── strategies/           # 推理策略
├── reasoning_core/           # 推理算法核心
│   ├── cotdir_method.py     # COT-DIR方法实现
│   ├── ird_module.py        # 隐式关系发现
│   ├── mlr_module.py        # 多层推理
│   └── cv_module.py         # 置信度验证
├── processors/              # 数据处理器
│   ├── complexity_classifier.py  # 复杂度分类器
│   ├── nlp_processor.py     # 自然语言处理
│   └── relation_extractor.py # 关系提取器
├── models/                  # 数据模型
│   ├── structures.py        # 数据结构定义
│   └── data_types.py        # 数据类型
├── evaluation/              # 评估模块
│   ├── sota_benchmark.py    # SOTA基准测试
│   ├── metrics.py           # 评估指标
│   └── dir_focused_benchmark.py # DIR专项评估
├── ai_core/                 # AI协作接口
│   └── interfaces/          # 接口定义
└── utilities/               # 工具函数
    └── helpers.py           # 辅助函数
```
---

## 1. reasoning_core/（推理算法核心）

- **cotdir_method.py**  
  COT-DIR方法实现，负责将显式与隐式关系链条化，生成推理链（Chain of Thought）。
  > **地位**：DIR-COT推理流程的"大脑"，最核心的算法模块。

- **ird_module.py**  
  隐式关系发现（Implicit Relation Discovery），自动挖掘题目中的L1/L2/L3等隐含关系。
  > **地位**：DIR思想的关键实现，补全显式信息无法覆盖的推理链路。

---

## 2. reasoning_engine/（推理引擎核心）

- **cotdir_integration.py**  
  COT-DIR集成框架，负责调度推理方法、整合多层推理、输出最终解答。
  > **地位**：将各推理模块有机整合，形成端到端的解答流程。

---

## 3. processors/（数据处理器）

- **relation_extractor.py**  
  关系提取器，从原始题目文本中抽取显式关系，为后续隐式推理提供基础。

- **complexity_classifier.py**  
  复杂度分类器，辅助推理引擎选择合适的推理深度和策略。

---

## 4. models/（数据模型）

- **structures.py**  
  定义"关系"、"推理链"、"解答"等核心数据结构，保证各模块数据交互一致。

- **data_types.py**  
  类型定义，提升系统健壮性。

---

## 5. src/models/pattern.json（关系模式配置）

- **pattern.json**  
  定义各种关系模式、正则表达式、推理模板，是隐式/显式关系发现的"知识库"。
  > **地位**：算法与数据之间的桥梁，保证推理的可扩展性和灵活性。

---

## 6. 主流程调度与演示（根目录/演示文件）

- **single_question_demo.py**、**detailed_case_results_generator.py** 等  
  调用上述核心模块，完成单题/批量推理演示与结果输出。

---

> 以上模块共同构成了DIR-COT项目解答过程的"核心功能的核心模块"。 