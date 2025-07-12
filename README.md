# COT-DIR1 数学推理系统

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)](https://github.com/menghao/cot-dir1)
[![Tests](https://img.shields.io/badge/Tests-Passing-success.svg)](tests/)

*基于隐式推理的智能数学问题求解系统*

[快速开始](#🚀-快速开始) • [架构文档](#🏗️-技术架构选型) • [API文档](#📖-api文档) • [贡献指南](#🤝-贡献指南)

</div>

---

## 🎯 项目概述和核心功能

### 项目简介

COT-DIR1 是一个基于**链式思维与方向性隐式推理（Chain-of-Thought with Directional Implicit Reasoning）**的先进数学推理系统。该系统结合了深度学习、符号计算和启发式推理，能够解决各种复杂的数学问题。

### 🌟 核心特性

#### 智能推理引擎
- **隐式关系发现（IRD）**：自动识别题目中的隐式数学关系
- **多层级推理（MLR）**：支持L0-L3四个复杂度级别的推理
- **链式验证（CV）**：确保推理过程的逻辑一致性和正确性

#### 多模型融合
- **本地大语言模型**：Qwen2.5-Math、InternLM2.5-Math、DeepSeek-Math
- **云端API支持**：OpenAI GPT-4o、Anthropic Claude 3.5 Sonnet
- **基线模型**：模板匹配、规则推理、符号计算

#### 全面评估体系
- **8个标准数据集**：GSM8K、SVAMP、MultiArith、Math23K等
- **多维度评估**：准确率、推理质量、效率、鲁棒性、可解释性
- **消融研究**：组件贡献度分析和性能优化

### 🔍 应用场景

- **教育领域**：数学学习辅助、自动阅卷、个性化教学
- **科研应用**：数学建模、公式推导、算法验证
- **工业应用**：工程计算、数据分析、智能决策
- **竞赛训练**：数学竞赛、算法竞赛、模型评测

---

## 🏗️ 技术架构选型

### 🔧 核心技术栈

#### 编程语言与框架
```python
# 主要技术栈
Python 3.8+          # 主要开发语言
NumPy 2.3.1          # 数值计算
SciPy 1.16.0         # 科学计算
SymPy 1.14.0         # 符号数学
Matplotlib 3.10.3    # 可视化
```

#### 机器学习与AI
```python
# ML/AI 技术栈
Scikit-learn 1.7.0   # 机器学习
Pydantic 2.11.7      # 数据验证
Requests 2.32.4      # API通信
TQDM 4.67.1          # 进度显示
```

### 🏛️ 系统架构设计

#### 模块化架构
```
COT-DIR1 系统架构
├── 🧠 核心推理层 (Core Reasoning)
│   ├── 隐式关系发现 (IRD Engine)
│   ├── 多层级推理 (MLR Engine)
│   └── 链式验证 (CV Engine)
├── 🔄 模型融合层 (Model Integration)
│   ├── 本地模型管理
│   ├── 云端API接口
│   └── 基线模型集成
├── 📊 数据处理层 (Data Processing)
│   ├── 问题解析与标准化
│   ├── 复杂度分类
│   └── 数据增强
├── 🎯 评估验证层 (Evaluation)
│   ├── 多数据集评估
│   ├── 性能指标计算
│   └── 消融研究
└── 🔧 基础服务层 (Infrastructure)
    ├── 配置管理
    ├── 日志系统
    └── 异常处理
```

#### 关键设计原则

1. **模块化设计**：松耦合、高内聚的模块架构
2. **可扩展性**：支持新模型和算法的快速集成
3. **可观测性**：完整的日志、监控和诊断系统
4. **鲁棒性**：异常处理和容错机制
5. **性能优化**：并行处理和缓存机制

---

## 📁 详细的项目结构

### 🗂️ 目录结构

```
COT-DIR1/
├── 📁 src/                          # 源代码
│   ├── 🧠 core/                     # 核心架构
│   │   ├── interfaces.py            # 标准接口定义
│   │   ├── system_orchestrator.py   # 系统协调器
│   │   ├── module_registry.py       # 模块注册中心
│   │   └── exceptions.py            # 异常处理
│   ├── 🔮 reasoning/                # 推理模块
│   │   ├── public_api.py            # 公共API
│   │   ├── orchestrator.py          # 推理协调器
│   │   └── private/                 # 私有实现
│   │       ├── processor.py         # 推理处理器
│   │       ├── step_builder.py      # 推理步骤构建
│   │       ├── confidence_calc.py   # 置信度计算
│   │       └── validator.py         # 推理验证
│   ├── 🤖 models/                   # 模型管理
│   │   ├── model_manager.py         # 模型管理器
│   │   ├── proposed_model.py        # COT-DIR模型
│   │   ├── baseline_models.py       # 基线模型
│   │   ├── llm_models.py            # 大语言模型
│   │   └── structures.py            # 数据结构
│   ├── ⚙️ processors/               # 数据处理
│   │   ├── complexity_classifier.py # 复杂度分类
│   │   ├── dataset_loader.py        # 数据集加载
│   │   ├── relation_extractor.py    # 关系提取
│   │   └── scalable_architecture.py # 可扩展架构
│   ├── 📊 evaluation/               # 评估系统
│   │   ├── evaluator.py             # 综合评估器
│   │   ├── metrics.py               # 评估指标
│   │   └── sota_benchmark.py        # SOTA基准测试
│   ├── 📈 data/                     # 数据管理
│   │   ├── dataset_info.py          # 数据集信息
│   │   └── performance_analysis.py  # 性能分析
│   ├── 🌉 bridge/                   # 桥接层
│   │   └── reasoning_bridge.py      # 推理桥接
│   └── 🤝 ai_core/                  # AI核心接口
│       └── interfaces/              # AI协作接口
├── 📁 config/                       # 配置文件
│   ├── config.json                  # 主配置
│   ├── model_config.json            # 模型配置
│   ├── logging.yaml                 # 日志配置
│   └── advanced/                    # 高级配置
├── 📁 tests/                        # 测试套件
│   ├── unit/                        # 单元测试
│   ├── integration/                 # 集成测试
│   ├── performance/                 # 性能测试
│   └── system_tests/                # 系统测试
├── 📁 demos/                        # 演示示例
│   ├── basic_usage.py               # 基础使用
│   ├── advanced_features.py         # 高级功能
│   └── benchmarks/                  # 基准测试
├── 📁 docs/                         # 文档
│   ├── api/                         # API文档
│   ├── user_guide/                  # 用户指南
│   └── historical_reports/          # 历史报告
├── 📁 Data/                         # 数据集
│   ├── GSM8K/                       # GSM8K数据集
│   ├── SVAMP/                       # SVAMP数据集
│   └── Math23K/                     # Math23K数据集
├── 📁 results/                      # 结果输出
├── 📁 logs/                         # 日志文件
└── 📁 scripts/                      # 工具脚本
```

### 🔍 核心模块说明

#### 1. 核心架构模块 (`src/core/`)
- **系统协调器**：管理模块间协作和系统级操作
- **模块注册中心**：动态模块发现和生命周期管理
- **接口标准化**：统一的API协议和数据格式
- **异常处理**：全局异常捕获和错误恢复

#### 2. 推理引擎模块 (`src/reasoning/`)
- **隐式关系发现**：自动识别题目中的隐含关系
- **多层级推理**：支持不同复杂度的推理策略
- **链式验证**：确保推理过程的逻辑一致性
- **置信度计算**：量化推理结果的可信度

#### 3. 模型管理模块 (`src/models/`)
- **模型注册表**：统一管理所有推理模型
- **COT-DIR模型**：核心提出的推理模型
- **基线模型**：传统的数学求解方法
- **LLM集成**：大语言模型的标准化接口

#### 4. 数据处理模块 (`src/processors/`)
- **复杂度分类**：自动分析问题复杂度等级
- **数据集加载**：支持多种数学题数据集
- **关系提取**：从文本中提取数学关系
- **可扩展架构**：插件式的处理器管理

#### 5. 评估系统模块 (`src/evaluation/`)
- **多维度评估**：准确率、效率、鲁棒性评估
- **SOTA基准**：与最新方法的对比评估
- **消融研究**：组件贡献度分析
- **性能监控**：实时性能指标追踪

---

## 📅 开发计划和里程碑

### 🚀 项目阶段规划

#### 🎯 第一阶段：基础架构（已完成）
- ✅ **模块化架构设计**：完成核心架构和接口标准化
- ✅ **基础推理引擎**：实现COT-DIR核心算法
- ✅ **数据处理流水线**：构建完整的数据处理管道
- ✅ **基线模型集成**：集成传统求解方法

#### 🔄 第二阶段：模型优化（进行中）
- 🔄 **LLM集成优化**：提升大语言模型的集成效果
- 🔄 **推理算法优化**：改进隐式关系发现算法
- 🔄 **性能调优**：系统性能优化和并发处理
- 🔄 **测试覆盖**：完善单元测试和集成测试

#### 🎯 第三阶段：生产部署（计划中）
- 🔮 **API服务化**：构建RESTful API服务
- 🔮 **Web界面**：开发用户友好的Web界面
- 🔮 **容器化部署**：Docker容器化和K8s部署
- 🔮 **监控告警**：生产环境监控和告警系统

#### 🎯 第四阶段：扩展应用（规划中）
- 🔮 **多语言支持**：支持中英文以外的其他语言
- 🔮 **领域扩展**：扩展到物理、化学等其他学科
- 🔮 **移动端适配**：开发移动端应用
- 🔮 **教育集成**：与在线教育平台集成

### 📊 里程碑时间表

| 里程碑 | 时间 | 主要交付物 | 状态 |
|--------|------|------------|------|
| **M1: 架构设计** | 2024-Q1 | 核心架构、接口标准 | ✅ 完成 |
| **M2: 核心算法** | 2024-Q2 | COT-DIR算法、推理引擎 | ✅ 完成 |
| **M3: 模型集成** | 2024-Q3 | 多模型融合、基准测试 | ✅ 完成 |
| **M4: 性能优化** | 2024-Q4 | 性能调优、测试完善 | 🔄 进行中 |
| **M5: 生产部署** | 2025-Q1 | API服务、Web界面 | 🔮 计划中 |
| **M6: 扩展应用** | 2025-Q2 | 多语言、领域扩展 | 🔮 规划中 |

### 🎯 当前开发重点

1. **模型性能优化**
   - 改进隐式关系发现算法的准确率
   - 优化多层级推理的效率
   - 提升链式验证的可解释性

2. **系统稳定性**
   - 完善异常处理和错误恢复机制
   - 增强系统的容错能力
   - 优化内存使用和性能

3. **用户体验**
   - 简化API接口设计
   - 改进错误信息的可读性
   - 增强系统的可观测性

---

## 🔧 开发环境配置

### 🐍 环境要求

#### 系统要求
- **操作系统**：Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python版本**：3.8+ (推荐3.10+)
- **内存**：8GB+ (推荐16GB+)
- **存储**：10GB+ 可用空间

#### 硬件要求
- **CPU**：4核+ (推荐8核+)
- **GPU**：可选，支持CUDA 11.0+
- **网络**：稳定的网络连接（用于模型下载）

### 🚀 快速开始

#### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/menghao/cot-dir1.git
cd cot-dir1

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 2. 基础配置

```bash
# 复制配置文件
cp config/config.json.example config/config.json
cp config/model_config.json.example config/model_config.json

# 配置环境变量（可选）
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

#### 3. 快速验证

```python
# 运行基础示例
python demos/basic_usage.py

# 运行模块化系统演示
python src/demo_modular_system.py

# 运行测试套件
pytest tests/ -v
```

### ⚙️ 详细配置

#### 1. 主要配置文件

```json
// config/config.json - 系统主配置
{
    "version": "1.0.0",
    "log_level": "INFO",
    "solver": {
        "max_iterations": 1000,
        "tolerance": 1.0
    },
    "nlp": {
        "device": "mps",  // 设备：cpu/cuda/mps
        "model_path": "models/nlp"
    }
}
```

```json
// config/model_config.json - 模型配置
{
    "models": {
        "cotdir": {
            "enabled": true,
            "enable_ird": true,
            "enable_mlr": true,
            "enable_cv": true,
            "confidence_threshold": 0.7
        },
        "qwen": {
            "enabled": true,
            "base_url": "http://localhost:8000/v1"
        }
    }
}
```

#### 2. 日志配置

```yaml
# config/logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: default
    filename: logs/cot-dir1.log
loggers:
  root:
    level: INFO
    handlers: [console, file]
```

#### 3. 开发工具配置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 配置pre-commit hooks
pre-commit install

# 代码格式化
black src/ tests/
isort src/ tests/

# 代码检查
flake8 src/ tests/
pylint src/ tests/
```

### 🔧 高级配置

#### 1. 模型部署配置

```bash
# 本地模型部署
# 1. 下载模型权重
wget https://huggingface.co/Qwen/Qwen2.5-Math-72B-Instruct
wget https://huggingface.co/internlm/internlm2_5-7b-chat

# 2. 启动模型服务
python scripts/deploy_local_models.py

# 3. 验证模型服务
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}]}'
```

#### 2. 数据集配置

```bash
# 下载标准数据集
python scripts/download_datasets.py

# 数据集结构
Data/
├── GSM8K/
│   ├── train.jsonl
│   └── test.jsonl
├── SVAMP/
│   └── SVAMP.json
└── Math23K/
    ├── train.json
    └── test.json
```

#### 3. 性能优化配置

```json
// config/advanced/performance.json
{
    "parallel_processing": {
        "max_workers": 4,
        "enable_multiprocessing": true,
        "chunk_size": 100
    },
    "caching": {
        "enable_result_cache": true,
        "cache_size": 1000,
        "ttl": 3600
    },
    "memory_optimization": {
        "gc_threshold": 0.8,
        "memory_limit_mb": 4096
    }
}
```

### 🧪 测试配置

#### 1. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/unit/test_reasoning.py -v
pytest tests/integration/test_system.py -v

# 生成测试报告
pytest tests/ --cov=src --cov-report=html
```

#### 2. 性能测试

```bash
# 运行性能测试
python tests/performance_tests/test_benchmark.py

# 内存使用测试
python tests/performance_tests/test_memory.py

# 并发测试
python tests/performance_tests/test_concurrent.py
```

### 🐛 故障排除

#### 常见问题解决

1. **依赖安装失败**
   ```bash
   # 升级pip
   python -m pip install --upgrade pip
   
   # 清理缓存
   pip cache purge
   
   # 重新安装
   pip install -r requirements.txt --force-reinstall
   ```

2. **模型加载失败**
   ```bash
   # 检查模型配置
   python -c "import json; print(json.load(open('config/model_config.json')))"
   
   # 验证模型路径
   ls -la models/
   ```

3. **内存不足**
   ```bash
   # 减少并发数
   export MAX_WORKERS=2
   
   # 启用内存优化
   export ENABLE_MEMORY_OPTIMIZATION=true
   ```

### 📞 支持与帮助

- **问题反馈**：[GitHub Issues](https://github.com/menghao/cot-dir1/issues)
- **功能请求**：[GitHub Discussions](https://github.com/menghao/cot-dir1/discussions)
- **文档中心**：[在线文档](https://cot-dir1.readthedocs.io/)
- **社区交流**：[Discord频道](https://discord.gg/cot-dir1)

---

## 📖 API文档

### 🔌 核心API接口

#### 系统初始化
```python
from src.core.system_orchestrator import system_orchestrator

# 初始化系统
system_orchestrator.initialize_system()

# 解决数学问题
result = system_orchestrator.solve_math_problem({
    "problem": "小明有10个苹果，给了小红3个，还剩多少个？",
    "type": "arithmetic"
})
```

#### 模型调用
```python
from src.models.model_manager import ModelManager

# 创建模型管理器
manager = ModelManager()

# 使用COT-DIR模型
result = manager.solve_with_model(
    model_name="cotdir",
    problem="A train travels 120 km in 2 hours. What is its speed?",
    config={"enable_ird": True, "enable_mlr": True}
)
```

### 📚 更多文档

- [API参考手册](docs/api/)
- [用户指南](docs/user_guide/)
- [开发指南](docs/developer_guide/)

---

## 🤝 贡献指南

### 🌟 参与贡献

我们欢迎各种形式的贡献！

#### 贡献方式
- 🐛 **Bug报告**：发现问题请提交Issue
- 💡 **功能建议**：提出新功能想法
- 📝 **文档改进**：完善文档和示例
- 🔧 **代码贡献**：提交Pull Request

#### 开发流程
1. Fork项目到您的GitHub账号
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建Pull Request

### 📋 开发规范

#### 代码风格
- 使用Black进行代码格式化
- 遵循PEP 8代码规范
- 编写清晰的函数和类文档
- 添加类型注解

#### 测试要求
- 新功能必须包含单元测试
- 测试覆盖率不低于80%
- 通过所有现有测试

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢

感谢所有为此项目做出贡献的开发者和研究者！

- **开源社区**：感谢Python、NumPy、SciPy等开源项目
- **研究机构**：感谢提供数据集和基准测试的研究机构
- **用户反馈**：感谢用户的宝贵建议和bug报告

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请考虑给我们一个星标！**

[回到顶部](#cot-dir1-数学推理系统)

</div>


