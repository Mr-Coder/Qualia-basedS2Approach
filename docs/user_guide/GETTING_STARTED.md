# COT-DIR 用户快速开始指南

## 🎯 欢迎使用 COT-DIR 数学推理系统

COT-DIR 是一个先进的数学推理系统，集成了元知识、策略推荐和多层推理能力。

## 🚀 快速开始（5分钟上手）

### 1. 环境准备

```bash
# 确保Python 3.8+
python --version

# 安装依赖
pip install -r requirements.txt
```

### 2. 基础功能演示

```bash
# 运行基础功能演示
python demos/basic_demo.py
```

这将展示：
- ✅ 系统组件初始化
- 🧠 基础推理能力
- 📦 数据加载功能
- 💡 简单问题求解

### 3. 增强功能演示

```bash
# 运行增强功能演示
python demos/enhanced_demo.py
```

这将展示：
- 🧠 元知识系统
- 🎯 策略推荐
- 🔧 复杂推理能力
- 📊 性能分析

### 4. 系统验证

```bash
# 运行系统验证
python demos/validation_demo.py
```

这将进行：
- 🔍 功能完整性验证
- ⚡ 性能基准测试
- 🎯 准确率评估
- 📋 验证报告生成

## 📖 核心功能

### 元知识系统
- **10个数学概念**：分数、百分比、面积、体积、速度、折扣、利润、平均数、比例、方程
- **18种解题策略**：从基础到高级的完整策略库
- **智能推荐**：基于问题特征的策略优先级排序

### 推理引擎
- **多层推理**：L0-L3四个复杂度级别
- **模板匹配**：常见题型自动识别
- **错误检测**：智能错误预防和纠正

### 数据处理
- **15个数据集**：覆盖各种数学问题类型
- **多语言支持**：中英文问题处理
- **标准化格式**：统一的数据输入输出

## 🛠️ API 使用示例

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

# 预处理
processed = preprocessor.process(sample)

# 推理求解
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

- **响应时间**：平均 < 100ms
- **准确率**：基础问题 > 95%
- **支持复杂度**：L0-L3全覆盖
- **并发处理**：支持批量处理

## 🔧 高级配置

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

### 数据集扩展

```python
# 加载自定义数据集
from src.data.loader import DataLoader

loader = DataLoader()
samples = loader.load(path="your_dataset.json", max_samples=100)
```

## 🆘 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保设置正确的 Python 路径
   export PYTHONPATH="${PYTHONPATH}:src"
   ```

2. **数据集加载失败**
   ```python
   # 检查数据集路径和格式
   loader = DataLoader(data_dir="your_data_directory")
   ```

3. **推理结果异常**
   ```python
   # 启用详细日志
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### 获取帮助

- 📖 查看 API 文档：`docs/api/`
- 🔧 查看技术文档：`docs/technical/`
- 🧪 运行测试：`pytest tests/`
- 📊 查看验证结果：`validation_results.json`

## 🎯 下一步

1. **深入了解**：阅读 `docs/technical/` 中的技术文档
2. **定制开发**：参考 `docs/api/` 中的 API 文档
3. **性能优化**：运行 `demos/validation_demo.py` 了解系统性能
4. **扩展功能**：基于现有模块开发新功能

---

**🎉 恭喜！你已经掌握了 COT-DIR 系统的基础使用方法。开始探索更多强大功能吧！** 