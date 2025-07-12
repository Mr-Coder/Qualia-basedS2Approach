# 数学推理模型接口文档

## 概述

本项目提供了统一的数学推理模型接口，支持三种类型的模型：
1. **基线模型（Baseline Models）** - 传统的数学问题求解方法
2. **大语言模型（LLM Models）** - 基于大语言模型的求解方法
3. **提出模型（Proposed Model）** - COT-DIR（Chain-of-Thought with Directional Implicit Reasoning）方法

## 项目结构

```
src/models/
├── base_model.py           # 基础模型接口定义
├── baseline_models.py      # 基线模型实现
├── llm_models.py          # 大语言模型实现
├── proposed_model.py      # 提出的COT-DIR模型实现
└── model_manager.py       # 模型管理器（可选）

config/
└── model_config.json      # 模型配置文件

examples/
└── model_interface_demo.py # 使用示例
```

## 快速开始

### 1. 基本使用示例

```python
from src.models.base_model import ModelInput
from src.models.baseline_models import TemplateBasedModel
from src.models.llm_models import QwenModel
from src.models.proposed_model import COTDIRModel

# 创建问题输入
problem = ModelInput(
    problem_text="John has 15 apples. He gives 7 apples to Mary. How many apples does John have left?",
    expected_answer="8"
)

# 使用基线模型
baseline_model = TemplateBasedModel()
baseline_model.initialize()
result = baseline_model.solve_problem(problem)
print(f"基线模型答案: {result.answer}")

# 使用LLM模型
llm_model = QwenModel(config={"is_local": True})
llm_model.initialize()
result = llm_model.solve_problem(problem)
print(f"LLM模型答案: {result.answer}")

# 使用提出的COT-DIR模型
cotdir_model = COTDIRModel()
cotdir_model.initialize()
result = cotdir_model.solve_problem(problem)
print(f"COT-DIR模型答案: {result.answer}")
```

### 2. 运行演示脚本

```bash
cd /Users/menghao/Desktop/newfile
python examples/model_interface_demo.py
```

## 模型详细说明

### 基线模型（Baseline Models）

#### 1. 模板基础模型（TemplateBasedModel）
- **功能**: 使用预定义模板匹配数学问题
- **适用场景**: 简单的算术运算问题
- **特点**: 快速、轻量级、但覆盖范围有限

```python
from src.models.baseline_models import TemplateBasedModel

model = TemplateBasedModel()
model.initialize()
```

#### 2. 方程基础模型（EquationBasedModel）
- **功能**: 通过提取和求解数学方程来解决问题
- **适用场景**: 涉及方程的数学问题
- **特点**: 使用SymPy进行符号计算

```python
from src.models.baseline_models import EquationBasedModel

model = EquationBasedModel()
model.initialize()
```

#### 3. 规则基础模型（RuleBasedModel）
- **功能**: 使用启发式规则匹配问题类型
- **适用场景**: 常见的应用题类型
- **特点**: 包含多种问题类型的处理规则

```python
from src.models.baseline_models import RuleBasedModel

model = RuleBasedModel()
model.initialize()
```

### 大语言模型（LLM Models）

#### 1. OpenAI GPT-4o
```python
from src.models.llm_models import OpenAIGPTModel

model = OpenAIGPTModel(config={
    "api_key": "your-openai-api-key",
    "temperature": 0.7,
    "max_tokens": 2048
})
model.initialize()
```

#### 2. Anthropic Claude
```python
from src.models.llm_models import ClaudeModel

model = ClaudeModel(config={
    "api_key": "your-anthropic-api-key",
    "temperature": 0.7,
    "max_tokens": 2048
})
model.initialize()
```

#### 3. Qwen2.5-Math-72B
```python
from src.models.llm_models import QwenModel

# 本地部署
model = QwenModel(config={
    "is_local": True,
    "base_url": "http://localhost:8000/v1"
})

# 或使用API
model = QwenModel(config={
    "is_local": False,
    "api_key": "your-qwen-api-key"
})
```

#### 4. InternLM2.5-Math-7B
```python
from src.models.llm_models import InternLMModel

model = InternLMModel(config={
    "is_local": True,
    "base_url": "http://localhost:8001/v1"
})
```

#### 5. DeepSeek-Math-7B
```python
from src.models.llm_models import DeepSeekMathModel

model = DeepSeekMathModel(config={
    "is_local": True,
    "base_url": "http://localhost:8002/v1"
})
```

### 提出的COT-DIR模型

COT-DIR模型包含三个核心组件：

1. **IRD (Implicit Relation Discovery)** - 隐式关系发现
2. **MLR (Multi-Level Reasoning)** - 多层推理
3. **CV (Chain Verification)** - 链式验证

#### 完整配置使用
```python
from src.models.proposed_model import COTDIRModel

model = COTDIRModel(config={
    "enable_ird": True,
    "enable_mlr": True, 
    "enable_cv": True,
    "confidence_threshold": 0.7,
    "max_reasoning_depth": 5,
    "relation_threshold": 0.6,
    "ird_weight": 0.3,
    "mlr_weight": 0.5,
    "cv_weight": 0.2
})
model.initialize()
```

#### 消融研究配置
```python
# 无IRD组件
model_wo_ird = COTDIRModel(config={
    "enable_ird": False,
    "enable_mlr": True,
    "enable_cv": True
})

# 无MLR组件
model_wo_mlr = COTDIRModel(config={
    "enable_ird": True,
    "enable_mlr": False,
    "enable_cv": True
})

# 无CV组件
model_wo_cv = COTDIRModel(config={
    "enable_ird": True,
    "enable_mlr": True,
    "enable_cv": False
})

# 仅IRD组件
model_ird_only = COTDIRModel(config={
    "enable_ird": True,
    "enable_mlr": False,
    "enable_cv": False
})
```

## 配置文件使用

### 1. 环境变量设置
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export QWEN_API_KEY="your-qwen-api-key"
```

### 2. 配置文件修改
编辑 `config/model_config.json` 文件：

```json
{
  "models": {
    "gpt4o": {
      "enabled": true,
      "api_key": "your-api-key-here",
      "temperature": 0.7
    },
    "cotdir": {
      "enabled": true,
      "enable_ird": true,
      "enable_mlr": true,
      "enable_cv": true,
      "confidence_threshold": 0.7
    }
  }
}
```

## 输出格式

所有模型都返回统一的`ModelOutput`对象：

```python
@dataclass
class ModelOutput:
    answer: str                                    # 最终答案
    reasoning_chain: List[str]                     # 推理链
    confidence_score: float                        # 置信度分数
    processing_time: float                         # 处理时间
    memory_usage: Optional[float] = None           # 内存使用量
    intermediate_steps: Optional[List[Dict]] = None # 中间步骤
    error_message: Optional[str] = None            # 错误信息
    metadata: Optional[Dict[str, Any]] = None      # 元数据
```

### 示例输出
```python
result = model.solve_problem(problem)
print(f"答案: {result.answer}")
print(f"置信度: {result.confidence_score:.3f}")
print(f"处理时间: {result.processing_time:.3f}s")
print("推理步骤:")
for step in result.reasoning_chain:
    print(f"  - {step}")
```

## 批量处理

### 1. 单模型批量处理
```python
problems = [
    ModelInput(problem_text="Calculate 15 + 27"),
    ModelInput(problem_text="What is 8 × 9?"),
    ModelInput(problem_text="Find 100 - 37")
]

model = COTDIRModel()
model.initialize()
results = model.batch_solve(problems)

for problem, result in zip(problems, results):
    print(f"{problem.problem_text} = {result.answer}")
```

### 2. 多模型比较
```python
from src.models.base_model import ModelEvaluator

models = {
    "baseline": TemplateBasedModel(),
    "llm": QwenModel(config={"is_local": True}),
    "cotdir": COTDIRModel()
}

# 初始化所有模型
for model in models.values():
    model.initialize()

# 评估比较
evaluator = ModelEvaluator()
test_problems = [...]  # 测试问题列表

comparison_results = evaluator.compare_models(
    list(models.values()), 
    test_problems
)

for model_name, metrics in comparison_results.items():
    print(f"{model_name}: 准确率 = {metrics.accuracy:.3f}")
```

## 性能评估

### 1. 单模型评估
```python
from src.models.base_model import ModelEvaluator

evaluator = ModelEvaluator()
model = COTDIRModel()
model.initialize()

test_problems = [
    ModelInput(problem_text="...", expected_answer="10"),
    # 更多测试问题...
]

metrics = evaluator.evaluate_model(model, test_problems)
print(f"准确率: {metrics.accuracy:.3f}")
print(f"平均处理时间: {metrics.avg_processing_time:.3f}s")
print(f"错误率: {metrics.error_rate:.3f}")
```

### 2. 复杂度分析
COT-DIR模型会自动分析问题复杂度：

- **L0**: 显式问题（直接计算）
- **L1**: 浅层问题（简单推理）
- **L2**: 中等问题（多步推理）
- **L3**: 深层问题（复杂推理）

```python
result = cotdir_model.solve_problem(problem)
complexity = result.metadata.get("complexity", "Unknown")
print(f"问题复杂度: {complexity}")
```

## 故障排除

### 1. 常见错误

#### API密钥错误
```python
# 错误: OpenAI API key not provided
# 解决: 设置环境变量或在配置中提供API密钥
export OPENAI_API_KEY="your-key"
```

#### 本地服务器连接失败
```python
# 错误: Local Qwen server not available
# 解决: 启动本地模型服务器或使用mock模式
config = {"is_local": True, "use_mock": True}
```

#### 内存不足
```python
# 错误: Memory limit exceeded
# 解决: 减少batch_size或启用内存优化
config = {"batch_size": 10, "enable_memory_optimization": True}
```

### 2. 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
model = COTDIRModel(config={"debug": True})
```

### 3. 性能优化

```python
# 启用缓存
config = {
    "enable_caching": True,
    "cache_size": 1000
}

# 并行处理
config = {
    "enable_parallel": True,
    "max_workers": 4
}
```

## 扩展开发

### 1. 添加新的基线模型
```python
from src.models.base_model import BaselineModel

class CustomBaselineModel(BaselineModel):
    def __init__(self, config=None):
        super().__init__("Custom-Baseline", config)
    
    def initialize(self) -> bool:
        # 初始化逻辑
        return True
    
    def solve_problem(self, problem_input) -> ModelOutput:
        # 求解逻辑
        pass
    
    def extract_equations(self, problem_text):
        # 提取方程逻辑
        pass
    
    def solve_equations(self, equations):
        # 求解方程逻辑
        pass
```

### 2. 添加新的LLM模型
```python
from src.models.base_model import LLMModel

class CustomLLMModel(LLMModel):
    def __init__(self, config=None):
        super().__init__("Custom-LLM", config)
    
    def initialize(self) -> bool:
        # API初始化
        return True
    
    def generate_prompt(self, problem_input) -> str:
        # 生成提示词
        pass
    
    def call_api(self, prompt) -> str:
        # 调用API
        pass
    
    def parse_response(self, response) -> ModelOutput:
        # 解析响应
        pass
```

## 联系信息


## 更新日志

### v1.0.0 (2025-01-31)
- 初始发布
- 支持3种类型的模型接口
- 提供完整的COT-DIR模型实现
- 包含演示脚本和配置文件
- 支持批量处理和性能评估 