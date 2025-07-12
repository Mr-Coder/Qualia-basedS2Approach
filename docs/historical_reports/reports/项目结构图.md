# COT-DIR数学推理系统 - 项目结构图

## 📋 项目概览

```
newfile/                                    # 项目根目录
├── 🏗️ 核心源码                           # src/ 目录
├── 📊 数据集管理                          # Data/ 目录  
├── 🎯 演示程序                           # 根目录演示文件
├── 📝 文档系统                           # documentation/ + 根目录文档
├── 🧪 测试套件                           # tests/ 目录
├── ⚙️ 配置管理                           # config_files/ 目录
├── 📈 实验分析                           # 实验相关文件
└── 🗂️ 辅助工具                           # 其他工具文件
```

---

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

## 📊 数据集管理系统 (`Data/`)

```
Data/
├── 🎯 核心数据集 (15个)
│   ├── GSM8K/              # 小学数学应用题
│   ├── Math23K/            # 中文数学应用题
│   ├── MATH/               # 高中竞赛数学
│   ├── MathQA/             # 数学问答
│   ├── GSM-hard/           # 困难版GSM8K
│   ├── SVAMP/              # 变体数学问题
│   ├── ASDiv/              # 学术分部数据集
│   ├── MultiArith/         # 多步算术
│   ├── SingleEq/           # 单方程
│   ├── MAWPS/              # 数学应用题
│   ├── AddSub/             # 加减法问题
│   ├── AQuA/               # 代数问题
│   └── DIR-MWP/            # 定向推理数据集
├── 📋 管理文件
│   ├── dataset_loader.py           # 数据集加载器
│   ├── DATASETS_OVERVIEW.md       # 数据集概览
│   ├── DATASET_STATISTICS.md      # 数据集统计
│   ├── DATA_SCREENING_DECLARATION.md # 数据筛选声明
│   └── quality_validation_report.json # 质量验证报告
└── 🔧 处理工具
    ├── processing/         # 数据处理工具
    ├── management/         # 数据管理工具
    └── validation/         # 数据验证工具
```

---

## 🎯 演示程序系统

### 核心演示程序
```
根目录演示文件/
├── single_question_demo.py           # 单问题详细推理演示
├── simplified_cases_demo.py          # 简化批量演示
├── cases_results_demo.py             # 原版案例演示
├── detailed_case_results_generator.py # 详细结果生成器
├── experimental_framework.py         # 完整实验评估框架
└── batch_complexity_classifier.py    # 批量复杂度分类器
```

### 演示结果文件
```
结果文件/
├── simplified_case_results.json          # 简化案例结果
├── detailed_case_results.json            # 详细案例结果（含完整推理流程）
└── classification_results/               # 分类结果目录
```

---

## 📝 文档系统分类

### 分析报告类
```
分析报告/
├── PAPER_CODE_COMPARISON.md              # 论文代码对比
├── PAPER_VS_CODE_ANALYSIS.md             # 论文与代码分析
├── API_PAPER_IMPLEMENTATION_COMPARISON.md # API论文实现对比
├── CASE_RESULTS_ANALYSIS_REPORT.md       # 案例结果分析报告
├── DETAILED_RESULTS_COMPARISON.md        # 详细结果对比分析
├── PROJECT_STRUCTURE_ANALYSIS.md         # 项目结构分析
├── PROJECT_MAIN_FUNCTIONS_AND_ENTRIES.md # 主要功能和入口
├── MODULES_COLLABORATION_ANALYSIS.md     # 模块协作分析
├── sota_data_credibility_analysis.md     # SOTA数据可信度分析
└── 数据可靠性准确性检查报告.md          # 数据可靠性检查
```

### 优化重构类
```
优化重构/
├── CLEANUP_COMPLETION_REPORT.md          # 清理完成报告
├── SECOND_ROUND_CLEANUP_REPORT.md        # 二轮清理报告
├── ROOTDIR_FILES_ANALYSIS.md             # 根目录文件分析
├── API_STREAMLINED_CORE.md               # API精简核心
├── STREAMLINED_API_USAGE_GUIDE.md        # 精简API使用指南
└── FUNCTIONAL_MODULE_REFACTORING_REPORT.md # 功能模块重构报告
```

### 总结文档类
```
总结文档/
├── FINAL_PROJECT_SUMMARY.md              # 最终项目总结
└── PROJECT_STRUCTURE_DIAGRAM.md          # 项目结构图
```

---

## 🧪 测试系统 (`tests/`)

```
tests/
├── unit_tests/              # 单元测试
├── integration_tests/       # 集成测试
├── system_tests/            # 系统测试
├── performance_tests/       # 性能测试
```

---

## 📈 实验分析文件

### LaTeX实验报告
```
LaTeX文件/
├── performance_analysis_section.tex      # 性能分析章节
├── credible_sota_performance_table.tex   # 可信SOTA性能表
├── ablation_study_table.tex              # 消融研究表
└── FINAL_CORRECTED_EXPERIMENTAL_SECTION.tex # 最终实验章节
```

### 论文文档
```
论文文档/
└── CE_AI__Generative_AI__October_30__2024 (2).pdf # COT-DIR论文PDF
```

---

## ⚙️ 配置管理 (`config_files/`)

```
config_files/
├── 实验配置
├── 模型配置  
├── 评估配置
└── 数据配置
```

---

## 🗂️ 辅助目录

### 历史版本管理
```
legacy/                     # 历史版本代码
__pycache__/               # Python缓存文件
.github/                   # GitHub配置
```

### 演示扩展
```
demos/                     # 扩展演示程序
└── examples/              # 示例程序
```

---

> 本结构图涵盖了COT-DIR系统的全部核心与辅助模块，便于快速理解与导航。 