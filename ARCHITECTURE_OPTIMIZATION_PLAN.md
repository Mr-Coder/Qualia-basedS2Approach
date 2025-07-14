# COT-DIR 架构优化方案

## 🎯 优化目标
- 减少模块数量从100+到60以内
- 消除功能重叠和代码重复
- 简化目录层次结构
- 提高模块内聚性和降低耦合度

## 📊 当前架构问题
1. **冗余模块**：processors/ 和 reasoning/ 功能重叠
2. **分散工具**：parsing/, validation/, template_management/ 可合并
3. **重复API**：多个public_api.py文件
4. **层次过深**：某些目录嵌套过深

## 🏗️ 优化后目录结构

```
src/
├── core/                           # 核心框架（保持）
│   ├── __init__.py
│   ├── interfaces.py              # 统一接口定义
│   ├── exceptions.py              # 统一异常处理
│   ├── orchestrator.py           # 合并system_orchestrator + enhanced_system_orchestrator
│   └── registry.py               # 模块注册管理
│
├── reasoning/                      # 推理引擎（合并processors功能）
│   ├── __init__.py
│   ├── engines/                   # 推理引擎
│   │   ├── ird_engine.py         # 隐式关系发现
│   │   ├── mlr_processor.py      # 多层级推理
│   │   └── cv_validator.py       # 链式验证
│   ├── strategies/               # 推理策略
│   │   ├── cot_strategy.py       # Chain-of-Thought
│   │   ├── got_strategy.py       # Graph-of-Thought  
│   │   └── tot_strategy.py       # Tree-of-Thought
│   ├── processing/               # 数据处理（合并processors）
│   │   ├── text_processor.py     # 文本处理
│   │   ├── complexity_classifier.py
│   │   ├── relation_extractor.py
│   │   └── template_matcher.py   # 合并template_management
│   └── api.py                    # 统一推理API
│
├── models/                        # 模型管理（保持优化）
│   ├── __init__.py
│   ├── factory.py                # 合并model_factory
│   ├── cache.py                  # 模型缓存
│   ├── tracker.py                # 性能跟踪
│   ├── baseline/                 # 基线模型
│   ├── llm/                      # 大语言模型
│   └── api.py                    # 模型API
│
├── data/                          # 数据管理（保持）
│   ├── __init__.py
│   ├── loader.py
│   ├── preprocessor.py
│   ├── analyzer.py               # 合并performance_analysis
│   └── api.py
│
├── evaluation/                    # 评估框架（保持）
│   ├── __init__.py
│   ├── metrics.py
│   ├── benchmark.py              # 合并sota_benchmark + dir_focused_benchmark
│   ├── analysis.py               # 合并computational_analysis + failure_analysis
│   └── api.py
│
├── utils/                         # 工具模块（新建，合并分散工具）
│   ├── __init__.py
│   ├── validation.py             # 合并validation/
│   ├── parsing.py                # 合并parsing/
│   ├── security.py               # 安全工具
│   └── monitoring.py             # 合并monitoring/
│
├── extensions/                    # 可选扩展（重构）
│   ├── __init__.py
│   ├── gnn/                      # 重构gnn_enhancement
│   │   ├── models.py
│   │   ├── builders.py
│   │   └── integration.py
│   └── bridge/                   # 保持bridge
│
├── config/                        # 配置管理（优化）
│   ├── __init__.py
│   ├── manager.py                # 合并config_manager + advanced_config
│   ├── security.py               # 配置加密
│   └── environments/             # 环境配置
│
└── api/                          # 统一API层（新建）
    ├── __init__.py
    ├── public.py                 # 统一公共API
    ├── internal.py               # 内部API
    └── middleware.py             # API中间件
```

## 📋 实施计划

### Phase 1: 核心模块整合
1. 合并 core/ 中的重复文件
2. 统一接口定义和异常处理
3. 合并系统协调器

### Phase 2: 推理模块重构  
1. 将 processors/ 功能合并到 reasoning/processing/
2. 整合 template_management/ 到 reasoning/processing/
3. 统一推理API

### Phase 3: 工具模块整合
1. 创建 utils/ 模块
2. 合并 parsing/, validation/, monitoring/
3. 统一工具API

### Phase 4: 扩展模块优化
1. 重构 gnn_enhancement/ 为 extensions/gnn/
2. 简化目录结构
3. 优化模块接口

### Phase 5: API层统一
1. 创建统一API层
2. 消除重复的public_api.py文件
3. 标准化API接口

## 📈 预期收益
- **文件数量减少**: 100+ → 60以内
- **目录层次简化**: 最深4层 → 最深3层  
- **代码重复消除**: 估计减少30%重复代码
- **维护复杂度降低**: 模块间依赖关系更清晰
- **新功能开发效率提升**: 标准化接口和统一架构

## ⚠️ 风险控制
1. **渐进式重构**: 分阶段实施，每阶段验证功能
2. **向后兼容**: 保留关键API的兼容性
3. **测试覆盖**: 每个重构步骤都进行充分测试
4. **文档更新**: 同步更新相关文档

## 🎯 成功指标
- [ ] 文件数量减少至60以内
- [ ] 所有测试通过
- [ ] 性能不降低
- [ ] API接口保持稳定
- [ ] 文档完整更新