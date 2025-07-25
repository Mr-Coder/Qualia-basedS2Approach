# QS²+IRD+COT-DIR 统一推理系统

基于量子语义学(QS²)和隐式关系发现(IRD)的数学推理系统，实现了完整的COT-DIR推理链。

## 系统架构

### 核心模块

1. **问题预处理模块** (`problem_preprocessor.py`)
   - 文本清理和标准化
   - 实体提取和数字识别
   - 复杂度评估和问题分类

2. **QS²语义分析模块** (`qs2_semantic_analyzer.py`)
   - 基于Qualia理论的四维语义分析
   - Formal, Telic, Agentive, Constitutive语义角色构建
   - 语义兼容性计算

3. **IRD隐式关系发现模块** (`ird_relation_discovery.py`)
   - 三层关系发现：直接语义、上下文、传递性
   - 关系强度计算和置信度评估
   - 关系网络构建和优化

4. **推理引擎选择模块** (`reasoning_engine_selector.py`)
   - 智能引擎选择(简单/高级/混合/自动)
   - 简单引擎：基础算术运算
   - 高级引擎：QS²+IRD+COT-DIR完整推理

5. **COT-DIR推理链模块** (`cotdir_reasoning_chain.py`)
   - 六步推理流程：实体提取→语义分析→关系发现→数学计算→逻辑推理→结果综合
   - 数学运算处理器和逻辑推理处理器
   - 推理链指标计算和置信度评估

6. **结果生成与增强模块** (`result_generator.py`)
   - 标准化结果格式转换
   - 前端可视化数据生成
   - 交互元素和自然语言解释

### 统一后端服务

**统一后端服务器** (`unified_backend_server.py`)
- FastAPI REST API服务
- 完整推理流程编排
- 错误处理和降级机制
- 性能监控和状态管理

## API接口

### 核心接口

#### 健康检查
```bash
GET /api/health
```

#### 问题求解
```bash
POST /api/solve
Content-Type: application/json

{
  "problem": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
  "mode": "auto",  // auto | simple | advanced
  "preferences": {}
}
```

#### QS²算法演示
```bash
GET /api/qs2/demo
```

#### 算法执行状态
```bash
GET /api/algorithm/execution
GET /api/algorithm/execution/history?limit=20
```

#### 系统状态
```bash
GET /api/status
```

## 快速开始

### 启动服务器

```bash
cd refactored_backend
python unified_backend_server.py
```

服务将运行在 `http://127.0.0.1:8000`

### 测试系统

```bash
python test_unified_system.py
```

### API测试

```bash
# 健康检查
curl -X GET "http://127.0.0.1:8000/api/health"

# 求解问题
curl -X POST "http://127.0.0.1:8000/api/solve" \
  -H "Content-Type: application/json" \
  -d '{"problem": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？", "mode": "advanced"}'

# QS²演示
curl -X GET "http://127.0.0.1:8000/api/qs2/demo"
```

## 算法特性

### QS²语义分析
- **Qualia四维结构**：基于Qualia理论的深度语义分析
- **语义兼容性计算**：精确的实体关系建模
- **上下文增强**：动态上下文语义提升

### IRD关系发现
- **多层次关系发现**：
  - Layer 1: 直接语义关系
  - Layer 2: 基于上下文的关系增强
  - Layer 3: 传递性关系推理
- **领域知识整合**：数学问题特定的关系模式
- **置信度量化**：关系强度和可信度评估

### COT-DIR推理链
- **六步推理流程**：
  1. 实体提取与Qualia构建
  2. 四维语义结构分析
  3. 隐式关系发现
  4. 数学运算执行
  5. 逻辑推理验证
  6. 结果综合
- **多推理器协作**：数学处理器 + 逻辑推理器
- **动态置信度评估**：步骤级和整体置信度计算

### 智能引擎选择
- **自适应模式选择**：根据问题复杂度自动选择引擎
- **多引擎支持**：简单引擎、高级引擎、混合模式
- **Fallback机制**：引擎失败时的降级处理
- **性能监控**：引擎状态跟踪和性能优化

## 技术实现

### 数据结构
- 使用Python `dataclass` 确保类型安全
- 标准化的实体、关系、推理步骤表示
- 丰富的元数据和指标信息

### 错误处理
- 分层错误处理机制
- 优雅降级和Fallback逻辑
- 详细的错误日志和诊断信息

### 性能优化
- 异步处理支持
- 模块化架构减少耦合
- 缓存和状态管理

### 前端集成
- 标准化JSON响应格式
- 实体关系图可视化数据
- 交互元素和动画支持
- 自然语言解释生成

## 系统指标

### 性能指标
- 平均响应时间：1-3ms
- 推理步骤：6步完整流程
- 支持并发：多用户同时访问
- 置信度：通常70%+

### 功能指标
- **实体识别**：人物、物体、数字、概念
- **关系类型**：拥有、数量、功能、语义、上下文、传递、数学、因果
- **运算支持**：加法、减法、乘法、面积计算等
- **问题类型**：算术、几何、应用题等

## 技术栈

- **后端框架**：FastAPI
- **异步支持**：asyncio
- **数据验证**：Pydantic
- **日志系统**：Python logging
- **类型检查**：Python typing
- **数学计算**：原生Python + math
- **HTTP服务**：Uvicorn

## 目录结构

```
refactored_backend/
├── problem_preprocessor.py          # 问题预处理模块
├── qs2_semantic_analyzer.py         # QS²语义分析模块
├── ird_relation_discovery.py        # IRD关系发现模块
├── reasoning_engine_selector.py     # 推理引擎选择模块
├── cotdir_reasoning_chain.py        # COT-DIR推理链模块
├── result_generator.py              # 结果生成与增强模块
├── unified_backend_server.py        # 统一后端服务器
├── test_unified_system.py           # 系统集成测试
└── README.md                        # 本文档
```

## 扩展性

### 新增推理引擎
1. 继承基础引擎接口
2. 实现 `solve()` 方法
3. 在引擎选择器中注册
4. 配置选择策略

### 新增关系类型
1. 在IRD模块中定义新关系类型
2. 添加关系发现规则
3. 配置关系强度计算
4. 更新可视化映射

### 新增问题类型
1. 扩展预处理器的问题分类
2. 添加特定领域的语义规则
3. 配置数学运算器支持
4. 更新结果格式化

## 部署说明

### 开发环境
```bash
python unified_backend_server.py
```

### 生产环境
```bash
uvicorn unified_backend_server:app --host 0.0.0.0 --port 8000
```

### Docker部署
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install fastapi uvicorn
CMD ["uvicorn", "unified_backend_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 监控和维护

### 日志监控
- 结构化日志输出
- 不同级别的日志信息
- 性能指标记录

### 健康检查
- `/api/health` 端点
- 模块状态监控
- 性能指标统计

### 错误诊断
- 详细的错误堆栈
- 请求追踪ID
- 失败原因分析

## 版本信息

- **版本**：2.0.0
- **最后更新**：2025-07-24
- **算法**：QS²+IRD+COT-DIR统一推理
- **兼容性**：Python 3.8+

## 许可证

MIT License

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系方式

- 技术支持：开发团队
- 问题反馈：GitHub Issues
- 文档更新：Wiki页面