# COT-DIR: Chain-of-Thought with Deep Implicit Relations

一个先进的模块化数学推理系统，集成了元知识、策略推荐和多层推理能力。

## 🎯 项目概述

COT-DIR 是基于链式思维和深度隐式关系的数学推理系统，具有以下核心特性：

- **🧠 智能推理引擎**: 集成元知识系统的多层推理能力
- **🎯 策略推荐**: 18种解题策略的智能推荐系统
- **📊 元知识增强**: 10个数学概念的深度知识库
- **⚡ 高性能**: 平均响应时间 < 1ms，优秀级别性能
- **🔧 模块化设计**: 清晰的分层架构，易于扩展和维护

## 🚀 5分钟快速开始

### 1. 环境准备

```bash
# 确保 Python 3.8+
python --version

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行核心演示

```bash
# 基础功能演示
python demos/basic_demo.py

# 增强功能演示（元知识+策略）
python demos/enhanced_demo.py

# 系统验证和性能测试
python demos/validation_demo.py
```

## 📁 项目架构

### 重构后的清晰结构
```
cot-dir1/
├── demos/                       # 🎯 精选演示（3个核心演示）
│   ├── basic_demo.py           # 基础功能演示
│   ├── enhanced_demo.py        # 增强功能演示
│   └── validation_demo.py      # 验证和性能测试
├── src/                         # 🏗️ 核心源码
│   ├── reasoning_core/          # 推理核心模块
│   │   ├── meta_knowledge.py   # 元知识系统
│   │   ├── reasoning_engine.py # 推理引擎
│   │   └── tools/              # 推理工具
│   ├── data/                   # 数据处理模块
│   │   ├── loader.py           # 数据加载器
│   │   └── preprocessor.py     # 数据预处理器
│   └── evaluation/             # 评估模块
├── tests/                      # 🧪 统一测试目录
│   ├── unit/                   # 单元测试
│   ├── integration/            # 集成测试
│   └── system/                 # 系统测试
├── docs/                       # 📚 统一文档目录
│   ├── api/                    # API文档
│   ├── user_guide/             # 用户指南
│   └── technical/              # 技术文档
├── config/                     # ⚙️ 统一配置目录
│   └── default.yaml           # 标准化配置
├── Data/                       # 📊 数学数据集（15个数据集）
└── archive/                    # 📦 归档目录
```

## 🛠️ 核心功能

### 元知识系统
- **10个数学概念**: 分数、百分比、面积、体积、速度、折扣、利润、平均数、比例、方程
- **18种解题策略**: 从基础到高级的完整策略库
- **智能推荐**: 基于问题特征的策略优先级排序

### 推理引擎
- **多层推理**: L0-L3四个复杂度级别
- **模板匹配**: 常见题型自动识别
- **错误检测**: 智能错误预防和纠正

### 数据处理
- **15个数据集**: 覆盖各种数学问题类型
- **多语言支持**: 中英文问题处理
- **标准化格式**: 统一的数据输入输出

## 💡 使用示例

### 基础推理

```python
from src.reasoning_core.reasoning_engine import ReasoningEngine
from src.data.preprocessor import Preprocessor

# 初始化组件
preprocessor = Preprocessor()
engine = ReasoningEngine()

# 处理问题
problem = "小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？"
sample = {"problem": problem, "id": "example_1"}

# 预处理和推理
processed = preprocessor.process(sample)
result = engine.solve(processed)

# 查看结果
print(f"答案: {result['final_answer']}")
print(f"置信度: {result['confidence']:.2f}")
```

### 元知识增强

```python
from src.reasoning_core.meta_knowledge import MetaKnowledge

# 初始化元知识系统
meta_knowledge = MetaKnowledge()

# 概念识别
concepts = meta_knowledge.identify_concepts_in_text("计算分数 1/2 + 1/3")
print(f"识别概念: {concepts}")

# 策略推荐
strategies = meta_knowledge.suggest_strategies("解方程 x + 5 = 10")
print(f"推荐策略: {strategies}")
```

## 📊 系统性能

- **⚡ 响应时间**: 平均 0.7ms (优秀级别)
- **🎯 系统健康度**: 100% (全部组件正常)
- **🧠 元知识**: 正常工作 (概念识别+策略推荐)
- **📦 数据处理**: 支持多种格式
- **🔧 可扩展性**: 模块化设计易于扩展

## 🧪 测试和验证

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/unit/
pytest tests/integration/
pytest tests/system/

# 系统验证
python demos/validation_demo.py
```

## 📚 文档资源

- **快速开始**: [`docs/user_guide/GETTING_STARTED.md`](docs/user_guide/GETTING_STARTED.md)
- **API文档**: [`docs/api/`](docs/api/)
- **重构计划**: [`REFACTORING_PLAN.md`](REFACTORING_PLAN.md)
- **完成报告**: [`REFACTORING_COMPLETION_REPORT.md`](REFACTORING_COMPLETION_REPORT.md)

## ⚙️ 配置

系统使用标准化的YAML配置文件：

```yaml
# config/default.yaml
system:
  name: "COT-DIR"
  version: "1.0.0"

reasoning_engine:
  enable_meta_knowledge: true
  strategy_threshold: 0.8
  max_reasoning_steps: 15

meta_knowledge:
  concepts_count: 10
  strategies_count: 18
```

## 🔧 开发指南

### 添加新功能
1. 在适当的模块中添加代码
2. 编写对应的测试
3. 更新相关文档
4. 运行完整测试验证

### 自定义配置
```python
# 自定义推理引擎配置
config = {
    "enable_meta_knowledge": True,
    "strategy_threshold": 0.8,
    "max_reasoning_steps": 10
}

engine = ReasoningEngine(config=config)
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 遵循新的项目结构
4. 添加测试和文档
5. 提交 Pull Request

## 📈 路线图

### 近期目标
- 🎯 提升复杂推理准确率 (目标: 80%+)
- 🧠 扩展元知识库
- 📊 增加更多评估指标

### 长期规划
- 🌍 多模态输入支持
- 🚀 并发处理优化
- 🎓 教育应用扩展

## 📄 许可证

[请添加您的许可证信息]

## 🙏 致谢

感谢所有贡献者对COT-DIR项目的支持和贡献。

---

**🎉 现在就开始探索COT-DIR的强大功能吧！**

```bash
python demos/basic_demo.py
``` 