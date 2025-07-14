# COT-DIR API Reference\n\nThis document provides a comprehensive reference for the COT-DIR API.\n\n## Table of Contents\n\n- [ai_core.interfaces.base_protocols](#ai_core-interfaces-base_protocols)\n- [ai_core.interfaces.data_structures](#ai_core-interfaces-data_structures)\n- [ai_core.interfaces.exceptions](#ai_core-interfaces-exceptions)\n- [bridge.reasoning_bridge](#bridge-reasoning_bridge)\n- [config.advanced_config](#config-advanced_config)\n- [config.config_manager](#config-config_manager)\n- [core.enhanced_system_orchestrator](#core-enhanced_system_orchestrator)\n- [core.exceptions](#core-exceptions)\n- [core.interfaces](#core-interfaces)\n- [core.module_registry](#core-module_registry)\n- [core.orchestration_strategy](#core-orchestration_strategy)\n- [core.orchestrator](#core-orchestrator)\n- [core.problem_solver_interface](#core-problem_solver_interface)\n- [core.security_service](#core-security_service)\n- [core.system_orchestrator](#core-system_orchestrator)\n- [data.dataset_characteristics](#data-dataset_characteristics)\n- [data.export_utils](#data-export_utils)\n- [data.loader](#data-loader)\n- [data.orchestrator](#data-orchestrator)\n- [data.performance_analysis](#data-performance_analysis)\n- [data.preprocessor](#data-preprocessor)\n- [data.public_api](#data-public_api)\n- [demo_modular_system](#demo_modular_system)\n- [evaluation.ablation_study](#evaluation-ablation_study)\n- [evaluation.computational_analysis](#evaluation-computational_analysis)\n- [evaluation.dir_focused_benchmark](#evaluation-dir_focused_benchmark)\n- [evaluation.evaluator](#evaluation-evaluator)\n- [evaluation.failure_analysis](#evaluation-failure_analysis)\n- [evaluation.metrics](#evaluation-metrics)\n- [evaluation.orchestrator](#evaluation-orchestrator)\n- [evaluation.public_api](#evaluation-public_api)\n- [evaluation.sota_benchmark](#evaluation-sota_benchmark)\n- [gnn_enhancement.core.concept_gnn.math_concept_gnn](#gnn_enhancement-core-concept_gnn-math_concept_gnn)\n- [gnn_enhancement.core.reasoning_gnn.reasoning_gnn](#gnn_enhancement-core-reasoning_gnn-reasoning_gnn)\n- [gnn_enhancement.core.verification_gnn.verification_gnn](#gnn_enhancement-core-verification_gnn-verification_gnn)\n- [gnn_enhancement.graph_builders.concept_graph_builder](#gnn_enhancement-graph_builders-concept_graph_builder)\n- [gnn_enhancement.graph_builders.graph_builder](#gnn_enhancement-graph_builders-graph_builder)\n- [gnn_enhancement.graph_builders.reasoning_graph_builder](#gnn_enhancement-graph_builders-reasoning_graph_builder)\n- [gnn_enhancement.graph_builders.verification_graph_builder](#gnn_enhancement-graph_builders-verification_graph_builder)\n- [gnn_enhancement.integration.gnn_integrator](#gnn_enhancement-integration-gnn_integrator)\n- [gnn_enhancement.utils.gnn_utils](#gnn_enhancement-utils-gnn_utils)\n- [models.async_api](#models-async_api)\n- [models.base_model](#models-base_model)\n- [models.baseline_models](#models-baseline_models)\n- [models.data_types](#models-data_types)\n- [models.equation](#models-equation)\n- [models.equations](#models-equations)\n- [models.llm_models](#models-llm_models)\n- [models.model_manager](#models-model_manager)\n- [models.orchestrator](#models-orchestrator)\n- [models.pattern_loader](#models-pattern_loader)\n- [models.private.model_cache](#models-private-model_cache)\n- [models.private.model_factory](#models-private-model_factory)\n- [models.private.performance_tracker](#models-private-performance_tracker)\n- [models.private.processor](#models-private-processor)\n- [models.private.utils](#models-private-utils)\n- [models.private.validator](#models-private-validator)\n- [models.processed_text](#models-processed_text)\n- [models.proposed_model](#models-proposed_model)\n- [models.public_api](#models-public_api)\n- [models.public_api_refactored](#models-public_api_refactored)\n- [models.relation](#models-relation)\n- [models.secure_components](#models-secure_components)\n- [models.simple_pattern_model](#models-simple_pattern_model)\n- [models.structures](#models-structures)\n- [monitoring.performance_monitor](#monitoring-performance_monitor)\n- [processors.MWP_process](#processors-MWP_process)\n- [processors.batch_processor](#processors-batch_processor)\n- [processors.complexity_classifier](#processors-complexity_classifier)\n- [processors.dataset_loader](#processors-dataset_loader)\n- [processors.dynamic_dataset_manager](#processors-dynamic_dataset_manager)\n- [processors.equation_builder](#processors-equation_builder)\n- [processors.implicit_relation_annotator](#processors-implicit_relation_annotator)\n- [processors.inference_tracker](#processors-inference_tracker)\n- [processors.intelligent_classifier](#processors-intelligent_classifier)\n- [processors.nlp_processor](#processors-nlp_processor)\n- [processors.orchestrator](#processors-orchestrator)\n- [processors.private.processor](#processors-private-processor)\n- [processors.private.utils](#processors-private-utils)\n- [processors.private.validator](#processors-private-validator)\n- [processors.public_api](#processors-public_api)\n- [processors.relation_extractor](#processors-relation_extractor)\n- [processors.relation_matcher](#processors-relation_matcher)\n- [processors.scalable_architecture](#processors-scalable_architecture)\n- [processors.secure_components](#processors-secure_components)\n- [processors.visualization](#processors-visualization)\n- [reasoning.async_api](#reasoning-async_api)\n- [reasoning.confidence_calculator.confidence_base](#reasoning-confidence_calculator-confidence_base)\n- [reasoning.cotdir_orchestrator](#reasoning-cotdir_orchestrator)\n- [reasoning.multi_step_reasoner.step_executor](#reasoning-multi_step_reasoner-step_executor)\n- [reasoning.new_reasoning_engine](#reasoning-new_reasoning_engine)\n- [reasoning.orchestrator](#reasoning-orchestrator)\n- [reasoning.private.confidence_calc](#reasoning-private-confidence_calc)\n- [reasoning.private.cv_validator](#reasoning-private-cv_validator)\n- [reasoning.private.ird_engine](#reasoning-private-ird_engine)\n- [reasoning.private.mlr_processor](#reasoning-private-mlr_processor)\n- [reasoning.private.processor](#reasoning-private-processor)\n- [reasoning.private.step_builder](#reasoning-private-step_builder)\n- [reasoning.private.utils](#reasoning-private-utils)\n- [reasoning.private.validator](#reasoning-private-validator)\n- [reasoning.public_api](#reasoning-public_api)\n- [reasoning.public_api_refactored](#reasoning-public_api_refactored)\n- [reasoning.strategy_manager.cot_strategy](#reasoning-strategy_manager-cot_strategy)\n- [reasoning.strategy_manager.got_strategy](#reasoning-strategy_manager-got_strategy)\n- [reasoning.strategy_manager.strategy_base](#reasoning-strategy_manager-strategy_base)\n- [reasoning.strategy_manager.strategy_manager](#reasoning-strategy_manager-strategy_manager)\n- [reasoning.strategy_manager.tot_strategy](#reasoning-strategy_manager-tot_strategy)\n- [template_management.template_loader](#template_management-template_loader)\n- [template_management.template_manager](#template_management-template_manager)\n- [template_management.template_matcher](#template_management-template_matcher)\n- [template_management.template_registry](#template_management-template_registry)\n- [template_management.template_validator](#template_management-template_validator)\n- [validation.input_validator](#validation-input_validator)\n\n## ai_core.interfaces.base_protocols\n\n基础协议定义 - AI协作友好的接口规范

这个模块定义了系统中所有组件必须遵循的协议接口。
AI助手可以通过这些协议理解如何实现新的组件。

AI_CONTEXT: 协议定义了组件的行为契约
RESPONSIBILITY: 定义标准化的接口规范\n\n### Classes\n\n#### ReasoningStrategy\n\n推理策略协议 - AI可以通过实现这个协议来创建新的推理策略

AI_INSTRUCTION: 要创建新的推理策略，实现以下方法：
1. can_handle() - 判断能否处理特定问题
2. solve() - 解决问题并返回结构化结果
3. get_confidence() - 返回策略对问题的置信度\n\n**Inherits from:** Protocol\n\n**Methods:**\n\n##### can_handle\n```python\ndef can_handle(self: Any, problem: MathProblem) -> bool\n```\n\n判断此策略是否能处理给定的数学问题

Args:
    problem: 待处理的数学问题
    
Returns:
    bool: True if can handle, False otherwise
    
AI_HINT: 实现这个方法来决定策略的适用范围\n\n##### solve\n```python\ndef solve(self: Any, problem: MathProblem) -> ReasoningResult\n```\n\n解决数学问题

Args:
    problem: 待解决的数学问题
    
Returns:
    ReasoningResult: 包含推理步骤和最终答案的结果
    
AI_HINT: 这是策略的核心方法，返回详细的推理过程\n\n##### get_confidence\n```python\ndef get_confidence(self: Any, problem: MathProblem) -> float\n```\n\n获取策略对问题的置信度

Args:
    problem: 待评估的数学问题
    
Returns:
    float: 置信度 [0.0, 1.0]
    
AI_HINT: 用于策略选择，高置信度的策略优先使用\n\n---\n\n#### DataProcessor\n\n数据处理器协议 - AI可以实现这个协议来创建数据处理组件

AI_INSTRUCTION: 数据处理器负责：
1. process() - 处理输入数据
2. validate_input() - 验证输入数据有效性
3. get_output_schema() - 返回输出数据模式\n\n**Inherits from:** Protocol\n\n**Methods:**\n\n##### process\n```python\ndef process(self: Any, data: Any) -> Any\n```\n\n处理输入数据

Args:
    data: 待处理的数据
    
Returns:
    Any: 处理后的数据
    
AI_HINT: 实现具体的数据转换逻辑\n\n##### validate_input\n```python\ndef validate_input(self: Any, data: Any) -> bool\n```\n\n验证输入数据的有效性

Args:
    data: 待验证的数据
    
Returns:
    bool: 数据是否有效
    
AI_HINT: 在处理前检查数据完整性\n\n##### get_output_schema\n```python\ndef get_output_schema(self: Any) -> Dict[Unknown]\n```\n\n获取输出数据的模式定义

Returns:
    Dict: 输出数据的结构描述
    
AI_HINT: 用于下游组件理解数据格式\n\n---\n\n#### Validator\n\n验证器协议 - AI可以实现这个协议来创建验证组件

AI_INSTRUCTION: 验证器用于：
1. validate() - 执行验证逻辑
2. get_error_details() - 获取详细错误信息
3. suggest_fixes() - 提供修复建议\n\n**Inherits from:** Protocol\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any, target: Any) -> ValidationResult\n```\n\n执行验证逻辑

Args:
    target: 待验证的对象
    
Returns:
    ValidationResult: 验证结果
    
AI_HINT: 返回详细的验证信息，包括错误和建议\n\n##### get_error_details\n```python\ndef get_error_details(self: Any, target: Any) -> List[str]\n```\n\n获取详细的错误信息

Args:
    target: 待检查的对象
    
Returns:
    List[str]: 错误信息列表
    
AI_HINT: 提供人类可读的错误描述\n\n##### suggest_fixes\n```python\ndef suggest_fixes(self: Any, target: Any) -> List[str]\n```\n\n提供修复建议

Args:
    target: 需要修复的对象
    
Returns:
    List[str]: 修复建议列表
    
AI_HINT: 帮助用户理解如何修复问题\n\n---\n\n#### Orchestrator\n\n协调器协议 - AI可以实现这个协议来创建流程协调组件

AI_INSTRUCTION: 协调器负责：
1. orchestrate() - 协调整个处理流程
2. register_component() - 注册组件
3. get_execution_plan() - 获取执行计划\n\n**Inherits from:** Protocol\n\n**Methods:**\n\n##### orchestrate\n```python\ndef orchestrate(self: Any, input_data: Any) -> Any\n```\n\n协调整个处理流程

Args:
    input_data: 输入数据
    
Returns:
    Any: 处理结果
    
AI_HINT: 管理多个组件的协作执行\n\n##### register_component\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件到协调器

Args:
    name: 组件名称
    component: 组件实例
    
AI_HINT: 动态注册可插拔组件\n\n##### get_execution_plan\n```python\ndef get_execution_plan(self: Any, input_data: Any) -> List[str]\n```\n\n获取执行计划

Args:
    input_data: 输入数据
    
Returns:
    List[str]: 执行步骤列表
    
AI_HINT: 帮助理解处理流程\n\n---\n\n#### ExperimentRunner\n\n实验运行器协议 - AI可以实现这个协议来创建实验组件

AI_INSTRUCTION: 实验运行器用于：
1. run_experiment() - 运行实验
2. setup_experiment() - 设置实验环境
3. analyze_results() - 分析结果\n\n**Inherits from:** Protocol\n\n**Methods:**\n\n##### run_experiment\n```python\ndef run_experiment(self: Any, config: Dict[Unknown]) -> ExperimentResult\n```\n\n运行实验

Args:
    config: 实验配置
    
Returns:
    ExperimentResult: 实验结果
    
AI_HINT: 执行完整的实验流程\n\n##### setup_experiment\n```python\ndef setup_experiment(self: Any, config: Dict[Unknown]) -> Any\n```\n\n设置实验环境

Args:
    config: 实验配置
    
AI_HINT: 准备实验所需的环境和资源\n\n##### analyze_results\n```python\ndef analyze_results(self: Any, results: List[Any]) -> Dict[Unknown]\n```\n\n分析实验结果

Args:
    results: 实验结果列表
    
Returns:
    Dict: 分析报告
    
AI_HINT: 提供统计分析和洞察\n\n---\n\n#### PerformanceTracker\n\n性能跟踪器协议 - AI可以实现这个协议来创建监控组件

AI_INSTRUCTION: 性能跟踪器用于：
1. track() - 跟踪性能指标
2. get_metrics() - 获取性能指标
3. generate_report() - 生成性能报告\n\n**Inherits from:** Protocol\n\n**Methods:**\n\n##### track\n```python\ndef track(self: Any, operation: str, duration: float, metadata: Dict[Unknown]) -> Any\n```\n\n跟踪性能指标

Args:
    operation: 操作名称
    duration: 执行时长
    metadata: 附加元数据
    
AI_HINT: 记录系统性能数据\n\n##### get_metrics\n```python\ndef get_metrics(self: Any, operation: Optional[str]) -> PerformanceMetrics\n```\n\n获取性能指标

Args:
    operation: 可选的操作名称过滤
    
Returns:
    PerformanceMetrics: 性能指标数据
    
AI_HINT: 提供性能统计信息\n\n##### generate_report\n```python\ndef generate_report(self: Any, format: str) -> str\n```\n\n生成性能报告

Args:
    format: 报告格式 ("json", "html", "csv")
    
Returns:
    str: 格式化的报告内容
    
AI_HINT: 生成人类可读的性能报告\n\n---\n\n### Functions\n\n#### can_handle\n\n```python\ndef can_handle(self: Any, problem: MathProblem) -> bool\n```\n\n判断此策略是否能处理给定的数学问题

Args:
    problem: 待处理的数学问题
    
Returns:
    bool: True if can handle, False otherwise
    
AI_HINT: 实现这个方法来决定策略的适用范围\n\n---\n\n#### solve\n\n```python\ndef solve(self: Any, problem: MathProblem) -> ReasoningResult\n```\n\n解决数学问题

Args:
    problem: 待解决的数学问题
    
Returns:
    ReasoningResult: 包含推理步骤和最终答案的结果
    
AI_HINT: 这是策略的核心方法，返回详细的推理过程\n\n---\n\n#### get_confidence\n\n```python\ndef get_confidence(self: Any, problem: MathProblem) -> float\n```\n\n获取策略对问题的置信度

Args:
    problem: 待评估的数学问题
    
Returns:
    float: 置信度 [0.0, 1.0]
    
AI_HINT: 用于策略选择，高置信度的策略优先使用\n\n---\n\n#### process\n\n```python\ndef process(self: Any, data: Any) -> Any\n```\n\n处理输入数据

Args:
    data: 待处理的数据
    
Returns:
    Any: 处理后的数据
    
AI_HINT: 实现具体的数据转换逻辑\n\n---\n\n#### validate_input\n\n```python\ndef validate_input(self: Any, data: Any) -> bool\n```\n\n验证输入数据的有效性

Args:
    data: 待验证的数据
    
Returns:
    bool: 数据是否有效
    
AI_HINT: 在处理前检查数据完整性\n\n---\n\n#### get_output_schema\n\n```python\ndef get_output_schema(self: Any) -> Dict[Unknown]\n```\n\n获取输出数据的模式定义

Returns:
    Dict: 输出数据的结构描述
    
AI_HINT: 用于下游组件理解数据格式\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any, target: Any) -> ValidationResult\n```\n\n执行验证逻辑

Args:
    target: 待验证的对象
    
Returns:
    ValidationResult: 验证结果
    
AI_HINT: 返回详细的验证信息，包括错误和建议\n\n---\n\n#### get_error_details\n\n```python\ndef get_error_details(self: Any, target: Any) -> List[str]\n```\n\n获取详细的错误信息

Args:
    target: 待检查的对象
    
Returns:
    List[str]: 错误信息列表
    
AI_HINT: 提供人类可读的错误描述\n\n---\n\n#### suggest_fixes\n\n```python\ndef suggest_fixes(self: Any, target: Any) -> List[str]\n```\n\n提供修复建议

Args:
    target: 需要修复的对象
    
Returns:
    List[str]: 修复建议列表
    
AI_HINT: 帮助用户理解如何修复问题\n\n---\n\n#### orchestrate\n\n```python\ndef orchestrate(self: Any, input_data: Any) -> Any\n```\n\n协调整个处理流程

Args:
    input_data: 输入数据
    
Returns:
    Any: 处理结果
    
AI_HINT: 管理多个组件的协作执行\n\n---\n\n#### register_component\n\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件到协调器

Args:
    name: 组件名称
    component: 组件实例
    
AI_HINT: 动态注册可插拔组件\n\n---\n\n#### get_execution_plan\n\n```python\ndef get_execution_plan(self: Any, input_data: Any) -> List[str]\n```\n\n获取执行计划

Args:
    input_data: 输入数据
    
Returns:
    List[str]: 执行步骤列表
    
AI_HINT: 帮助理解处理流程\n\n---\n\n#### run_experiment\n\n```python\ndef run_experiment(self: Any, config: Dict[Unknown]) -> ExperimentResult\n```\n\n运行实验

Args:
    config: 实验配置
    
Returns:
    ExperimentResult: 实验结果
    
AI_HINT: 执行完整的实验流程\n\n---\n\n#### setup_experiment\n\n```python\ndef setup_experiment(self: Any, config: Dict[Unknown]) -> Any\n```\n\n设置实验环境

Args:
    config: 实验配置
    
AI_HINT: 准备实验所需的环境和资源\n\n---\n\n#### analyze_results\n\n```python\ndef analyze_results(self: Any, results: List[Any]) -> Dict[Unknown]\n```\n\n分析实验结果

Args:
    results: 实验结果列表
    
Returns:
    Dict: 分析报告
    
AI_HINT: 提供统计分析和洞察\n\n---\n\n#### track\n\n```python\ndef track(self: Any, operation: str, duration: float, metadata: Dict[Unknown]) -> Any\n```\n\n跟踪性能指标

Args:
    operation: 操作名称
    duration: 执行时长
    metadata: 附加元数据
    
AI_HINT: 记录系统性能数据\n\n---\n\n#### get_metrics\n\n```python\ndef get_metrics(self: Any, operation: Optional[str]) -> PerformanceMetrics\n```\n\n获取性能指标

Args:
    operation: 可选的操作名称过滤
    
Returns:
    PerformanceMetrics: 性能指标数据
    
AI_HINT: 提供性能统计信息\n\n---\n\n#### generate_report\n\n```python\ndef generate_report(self: Any, format: str) -> str\n```\n\n生成性能报告

Args:
    format: 报告格式 ("json", "html", "csv")
    
Returns:
    str: 格式化的报告内容
    
AI_HINT: 生成人类可读的性能报告\n\n---\n\n\n## ai_core.interfaces.data_structures\n\nAI协作友好的数据结构定义

这个模块定义了系统中使用的所有核心数据结构。
AI助手可以通过这些定义理解数据的结构和含义。

AI_CONTEXT: 标准化的数据模型，确保类型安全和清晰的数据流
RESPONSIBILITY: 定义系统中所有核心数据类型\n\n### Classes\n\n#### ProblemComplexity\n\n问题复杂度枚举

AI_HINT: 用于分类数学问题的难度级别\n\n**Inherits from:** Enum\n\n---\n\n#### ProblemType\n\n问题类型枚举

AI_HINT: 用于分类数学问题的具体类型\n\n**Inherits from:** Enum\n\n---\n\n#### OperationType\n\n操作类型枚举

AI_HINT: 推理步骤中的操作类型\n\n**Inherits from:** Enum\n\n---\n\n#### MathProblem\n\n数学问题数据结构 - AI协作友好设计

AI_CONTEXT: 表示一个完整的数学问题
RESPONSIBILITY: 包含问题的所有必要信息\n\n---\n\n#### ReasoningStep\n\n推理步骤数据结构

AI_CONTEXT: 表示推理过程中的一个步骤
RESPONSIBILITY: 记录单个推理操作的详细信息\n\n---\n\n#### ReasoningResult\n\n推理结果数据结构

AI_CONTEXT: 表示完整的推理过程和结果
RESPONSIBILITY: 包含推理的所有步骤和最终答案\n\n---\n\n#### ValidationResult\n\n验证结果数据结构

AI_CONTEXT: 表示验证过程的结果
RESPONSIBILITY: 记录验证的详细信息和建议\n\n---\n\n#### ExperimentResult\n\n实验结果数据结构

AI_CONTEXT: 表示实验的执行结果
RESPONSIBILITY: 记录实验的完整信息和分析\n\n---\n\n#### PerformanceMetrics\n\n性能指标数据结构

AI_CONTEXT: 表示系统性能的详细指标
RESPONSIBILITY: 记录和分析性能数据\n\n---\n\n\n## ai_core.interfaces.exceptions\n\nAI协作友好的异常类定义

这个模块定义了系统中使用的所有异常类型。
AI助手可以通过这些异常理解错误情况并提供相应的解决方案。

AI_CONTEXT: 结构化的错误处理，提供清晰的错误信息和修复建议
RESPONSIBILITY: 定义系统中所有可能的异常情况\n\n### Classes\n\n#### AICollaborativeError\n\nAI协作系统基础异常类

AI_CONTEXT: 所有系统异常的基类，提供统一的错误处理接口
RESPONSIBILITY: 提供丰富的错误上下文和修复建议\n\n**Inherits from:** Exception\n\n**Methods:**\n\n##### get_ai_friendly_description\n```python\ndef get_ai_friendly_description(self: Any) -> Dict[Unknown]\n```\n\n获取AI友好的错误描述

Returns:
    Dict: 包含错误详情、上下文和建议的结构化信息
    
AI_HINT: 使用这个方法获取结构化的错误信息\n\n---\n\n#### ReasoningError\n\n推理过程异常

AI_CONTEXT: 推理引擎执行过程中出现的错误
AI_INSTRUCTION: 当推理策略无法处理问题或推理过程失败时抛出\n\n**Inherits from:** AICollaborativeError\n\n---\n\n#### ValidationError\n\n验证过程异常

AI_CONTEXT: 数据验证或结果验证过程中出现的错误
AI_INSTRUCTION: 当验证规则不通过或验证过程失败时抛出\n\n**Inherits from:** AICollaborativeError\n\n---\n\n#### ConfigurationError\n\n配置错误异常

AI_CONTEXT: 系统配置相关的错误
AI_INSTRUCTION: 当配置文件无效、缺少必需配置或配置冲突时抛出\n\n**Inherits from:** AICollaborativeError\n\n---\n\n#### DataProcessingError\n\n数据处理异常

AI_CONTEXT: 数据加载、处理、转换过程中的错误
AI_INSTRUCTION: 当数据处理失败或数据格式不兼容时抛出\n\n**Inherits from:** AICollaborativeError\n\n---\n\n#### ExperimentError\n\n实验执行异常

AI_CONTEXT: 实验运行过程中的错误
AI_INSTRUCTION: 当实验设置无效或执行失败时抛出\n\n**Inherits from:** AICollaborativeError\n\n---\n\n#### PerformanceError\n\n性能相关异常

AI_CONTEXT: 系统性能监控和分析过程中的错误
AI_INSTRUCTION: 当性能指标异常或监控失败时抛出\n\n**Inherits from:** AICollaborativeError\n\n---\n\n### Functions\n\n#### handle_ai_collaborative_error\n\n```python\ndef handle_ai_collaborative_error(error: AICollaborativeError) -> Dict[Unknown]\n```\n\n处理AI协作异常的标准方法

Args:
    error: AI协作异常实例
    
Returns:
    Dict: 结构化的错误处理信息
    
AI_HINT: 使用这个函数获取标准化的错误处理信息\n\n---\n\n#### create_error_from_exception\n\n```python\ndef create_error_from_exception(exception: Exception, context: Optional[Unknown]) -> AICollaborativeError\n```\n\n从标准异常创建AI协作友好的异常

Args:
    exception: 原始异常
    context: 额外的上下文信息
    
Returns:
    AICollaborativeError: AI协作友好的异常
    
AI_HINT: 用于包装第三方库或系统异常\n\n---\n\n#### get_ai_friendly_description\n\n```python\ndef get_ai_friendly_description(self: Any) -> Dict[Unknown]\n```\n\n获取AI友好的错误描述

Returns:
    Dict: 包含错误详情、上下文和建议的结构化信息
    
AI_HINT: 使用这个方法获取结构化的错误信息\n\n---\n\n\n## bridge.reasoning_bridge\n\n推理引擎桥接层 - 激活重构后的代码
将旧版本ReasoningEngine接口桥接到新版本ReasoningAPI\n\n### Classes\n\n#### ReasoningEngineBridge\n\n桥接旧版本ReasoningEngine到新版本ReasoningAPI\n\n**Methods:**\n\n##### solve\n```python\ndef solve(self: Any, sample: Dict) -> Dict\n```\n\n兼容旧版本的solve方法\n\n---\n\n### Functions\n\n#### solve\n\n```python\ndef solve(self: Any, sample: Dict) -> Dict\n```\n\n兼容旧版本的solve方法\n\n---\n\n\n## config.advanced_config\n\nAdvanced Configuration System

This module provides comprehensive configuration management for the mathematical
reasoning system, supporting different experimental setups, model parameters,
and evaluation metrics.

Features:
1. Hierarchical configuration structure
2. Validation and type checking
3. Environment-specific configurations
4. Dynamic parameter adjustment
5. Configuration versioning

Author: AI Research Team
Date: 2025-01-31\n\n### Classes\n\n#### ConfigurationError\n\nCustom exception for configuration errors.\n\n**Inherits from:** Exception\n\n---\n\n#### ExperimentMode\n\nExperiment execution modes.\n\n**Inherits from:** Enum\n\n---\n\n#### ModelType\n\nMathematical reasoning model types.\n\n**Inherits from:** Enum\n\n---\n\n#### LoggingConfig\n\nLogging configuration.\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate logging configuration.\n\n---\n\n#### NLPConfig\n\nNLP processing configuration.\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate NLP configuration.\n\n---\n\n#### RelationDiscoveryConfig\n\nImplicit relation discovery configuration.\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate relation discovery configuration.\n\n---\n\n#### ReasoningConfig\n\nMulti-level reasoning configuration.\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate reasoning configuration.\n\n---\n\n#### VerificationConfig\n\nChain verification configuration.\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate verification configuration.\n\n---\n\n#### EvaluationConfig\n\nEvaluation and testing configuration.\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate evaluation configuration.\n\n---\n\n#### ExperimentConfig\n\nExperiment-specific configuration.\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate experiment configuration.\n\n---\n\n#### AdvancedConfiguration\n\nMain configuration class containing all sub-configurations.\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate entire configuration.\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\nConvert configuration to dictionary.\n\n##### to_json\n```python\ndef to_json(self: Any, filepath: Optional[str]) -> str\n```\n\nExport configuration to JSON.\n\n##### to_yaml\n```python\ndef to_yaml(self: Any, filepath: Optional[str]) -> str\n```\n\nExport configuration to YAML.\n\n##### from_dict\n```python\ndef from_dict(cls: Any, config_dict: Dict[Unknown]) -> Any\n```\n\nCreate configuration from dictionary.\n\n##### from_json\n```python\ndef from_json(cls: Any, filepath: str) -> Any\n```\n\nLoad configuration from JSON file.\n\n##### from_yaml\n```python\ndef from_yaml(cls: Any, filepath: str) -> Any\n```\n\nLoad configuration from YAML file.\n\n---\n\n#### ConfigurationManager\n\nManages loading and saving configurations for different environments.\n\n**Methods:**\n\n##### get_config\n```python\ndef get_config(self: Any, environment: str) -> AdvancedConfiguration\n```\n\nGet configuration for a specific environment.\n\n##### save_config\n```python\ndef save_config(self: Any, config: AdvancedConfiguration, environment: str, format: str) -> str\n```\n\nSave configuration for a specific environment.\n\n##### create_custom_config\n```python\ndef create_custom_config(self: Any, base_environment: str) -> AdvancedConfiguration\n```\n\nCreate a custom configuration by overriding a base configuration.\n\n##### validate_all_configs\n```python\ndef validate_all_configs(self: Any) -> Dict[Unknown]\n```\n\nValidate all existing configuration files.\n\n---\n\n### Functions\n\n#### main\n\n```python\ndef main()\n```\n\nDemonstrates the usage of the advanced configuration system.\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate logging configuration.\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate NLP configuration.\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate relation discovery configuration.\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate reasoning configuration.\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate verification configuration.\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate evaluation configuration.\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate experiment configuration.\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any) -> bool\n```\n\nValidate entire configuration.\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\nConvert configuration to dictionary.\n\n---\n\n#### to_json\n\n```python\ndef to_json(self: Any, filepath: Optional[str]) -> str\n```\n\nExport configuration to JSON.\n\n---\n\n#### to_yaml\n\n```python\ndef to_yaml(self: Any, filepath: Optional[str]) -> str\n```\n\nExport configuration to YAML.\n\n---\n\n#### from_dict\n\n```python\ndef from_dict(cls: Any, config_dict: Dict[Unknown]) -> Any\n```\n\nCreate configuration from dictionary.\n\n---\n\n#### from_json\n\n```python\ndef from_json(cls: Any, filepath: str) -> Any\n```\n\nLoad configuration from JSON file.\n\n---\n\n#### from_yaml\n\n```python\ndef from_yaml(cls: Any, filepath: str) -> Any\n```\n\nLoad configuration from YAML file.\n\n---\n\n#### get_config\n\n```python\ndef get_config(self: Any, environment: str) -> AdvancedConfiguration\n```\n\nGet configuration for a specific environment.\n\n---\n\n#### save_config\n\n```python\ndef save_config(self: Any, config: AdvancedConfiguration, environment: str, format: str) -> str\n```\n\nSave configuration for a specific environment.\n\n---\n\n#### create_custom_config\n\n```python\ndef create_custom_config(self: Any, base_environment: str) -> AdvancedConfiguration\n```\n\nCreate a custom configuration by overriding a base configuration.\n\n---\n\n#### validate_all_configs\n\n```python\ndef validate_all_configs(self: Any) -> Dict[Unknown]\n```\n\nValidate all existing configuration files.\n\n---\n\n#### update_nested_dict\n\n```python\ndef update_nested_dict(d: Any, overrides: Any)\n```\n\n---\n\n\n## config.config_manager\n\nCOT-DIR 增强配置管理系统

提供分层配置管理、环境隔离、安全加密、动态重载和配置监听功能。
整合原有功能并添加高级特性。\n\n### Classes\n\n#### ConfigLevel\n\n配置级别枚举\n\n**Inherits from:** Enum\n\n---\n\n#### ConfigSchema\n\n配置模式定义\n\n---\n\n#### ConfigSource\n\n配置源定义\n\n---\n\n#### EnhancedConfigurationManager\n\n增强配置管理器\n\n**Methods:**\n\n##### register_config_source\n```python\ndef register_config_source(self: Any, source: ConfigSource)\n```\n\n注册配置源\n\n##### get\n```python\ndef get(self: Any, key: str, default: Any, level: Optional[ConfigLevel]) -> Any\n```\n\n获取配置值\n\n##### set\n```python\ndef set(self: Any, key: str, value: Any, level: ConfigLevel, persist: bool)\n```\n\n设置配置值\n\n##### get_all\n```python\ndef get_all(self: Any, level: Optional[ConfigLevel]) -> Dict[Unknown]\n```\n\n获取所有配置\n\n##### reload_config\n```python\ndef reload_config(self: Any)\n```\n\n重新加载所有配置\n\n##### add_listener\n```python\ndef add_listener(self: Any, callback: Callable[Unknown])\n```\n\n添加配置变更监听器\n\n##### remove_listener\n```python\ndef remove_listener(self: Any, callback: Callable[Unknown])\n```\n\n移除配置变更监听器\n\n##### add_change_callback\n```python\ndef add_change_callback(self: Any, key: str, callback: Callable[Unknown])\n```\n\n添加特定键的变更回调\n\n##### remove_change_callback\n```python\ndef remove_change_callback(self: Any, key: str, callback: Callable[Unknown])\n```\n\n移除特定键的变更回调\n\n##### override\n```python\ndef override(self: Any, overrides: Dict[Unknown])\n```\n\n临时覆盖配置\n\n##### validate_config\n```python\ndef validate_config(self: Any, schema: Dict[Unknown]) -> bool\n```\n\n验证配置\n\n##### encrypt_and_save_secure_config\n```python\ndef encrypt_and_save_secure_config(self: Any, config: Dict[Unknown])\n```\n\n加密并保存敏感配置\n\n##### get_config_summary\n```python\ndef get_config_summary(self: Any) -> Dict[Unknown]\n```\n\n获取配置摘要（屏蔽敏感信息）\n\n##### create_default_configs\n```python\ndef create_default_configs(self: Any)\n```\n\n创建默认配置文件\n\n---\n\n#### ConfigurationManager\n\n配置管理器（兼容性类）\n\n**Inherits from:** EnhancedConfigurationManager\n\n---\n\n### Functions\n\n#### get_config\n\n```python\ndef get_config() -> EnhancedConfigurationManager\n```\n\n获取全局配置实例\n\n---\n\n#### init_config\n\n```python\ndef init_config(env: str, config_dir: str) -> EnhancedConfigurationManager\n```\n\n初始化全局配置\n\n---\n\n#### get_config_value\n\n```python\ndef get_config_value(key: str, default: Any) -> Any\n```\n\n获取配置值的便利函数\n\n---\n\n#### set_config_value\n\n```python\ndef set_config_value(key: str, value: Any, persist: bool)\n```\n\n设置配置值的便利函数\n\n---\n\n#### config_override\n\n```python\ndef config_override()\n```\n\n配置覆盖上下文管理器\n\n---\n\n#### register_config_source\n\n```python\ndef register_config_source(self: Any, source: ConfigSource)\n```\n\n注册配置源\n\n---\n\n#### get\n\n```python\ndef get(self: Any, key: str, default: Any, level: Optional[ConfigLevel]) -> Any\n```\n\n获取配置值\n\n---\n\n#### set\n\n```python\ndef set(self: Any, key: str, value: Any, level: ConfigLevel, persist: bool)\n```\n\n设置配置值\n\n---\n\n#### get_all\n\n```python\ndef get_all(self: Any, level: Optional[ConfigLevel]) -> Dict[Unknown]\n```\n\n获取所有配置\n\n---\n\n#### reload_config\n\n```python\ndef reload_config(self: Any)\n```\n\n重新加载所有配置\n\n---\n\n#### add_listener\n\n```python\ndef add_listener(self: Any, callback: Callable[Unknown])\n```\n\n添加配置变更监听器\n\n---\n\n#### remove_listener\n\n```python\ndef remove_listener(self: Any, callback: Callable[Unknown])\n```\n\n移除配置变更监听器\n\n---\n\n#### add_change_callback\n\n```python\ndef add_change_callback(self: Any, key: str, callback: Callable[Unknown])\n```\n\n添加特定键的变更回调\n\n---\n\n#### remove_change_callback\n\n```python\ndef remove_change_callback(self: Any, key: str, callback: Callable[Unknown])\n```\n\n移除特定键的变更回调\n\n---\n\n#### override\n\n```python\ndef override(self: Any, overrides: Dict[Unknown])\n```\n\n临时覆盖配置\n\n---\n\n#### validate_config\n\n```python\ndef validate_config(self: Any, schema: Dict[Unknown]) -> bool\n```\n\n验证配置\n\n---\n\n#### encrypt_and_save_secure_config\n\n```python\ndef encrypt_and_save_secure_config(self: Any, config: Dict[Unknown])\n```\n\n加密并保存敏感配置\n\n---\n\n#### get_config_summary\n\n```python\ndef get_config_summary(self: Any) -> Dict[Unknown]\n```\n\n获取配置摘要（屏蔽敏感信息）\n\n---\n\n#### create_default_configs\n\n```python\ndef create_default_configs(self: Any)\n```\n\n创建默认配置文件\n\n---\n\n#### reload_worker\n\n```python\ndef reload_worker()\n```\n\n---\n\n#### mask_sensitive\n\n```python\ndef mask_sensitive(obj: Any, path: Any)\n```\n\n递归屏蔽敏感信息\n\n---\n\n\n## core.enhanced_system_orchestrator\n\n增强系统协调器

支持重构后的模块架构，提供依赖图管理、并发处理和错误恢复。\n\n### Classes\n\n#### DependencyGraph\n\n模块依赖图管理器\n\n**Methods:**\n\n##### add_dependency\n```python\ndef add_dependency(self: Any, module: str, depends_on: str)\n```\n\n添加依赖关系\n\n##### remove_dependency\n```python\ndef remove_dependency(self: Any, module: str, depends_on: str)\n```\n\n移除依赖关系\n\n##### get_shutdown_order\n```python\ndef get_shutdown_order(self: Any) -> List[str]\n```\n\n获取关闭顺序（拓扑排序）\n\n##### has_cycle\n```python\ndef has_cycle(self: Any) -> bool\n```\n\n检查是否存在循环依赖\n\n---\n\n#### ConcurrentExecutor\n\n并发执行器\n\n**Methods:**\n\n##### execute_parallel\n```python\ndef execute_parallel(self: Any, tasks: List[Unknown]) -> Dict[Unknown]\n```\n\n并行执行任务\n\n##### shutdown\n```python\ndef shutdown(self: Any)\n```\n\n关闭执行器\n\n---\n\n#### ErrorRecoveryManager\n\n错误恢复管理器\n\n**Methods:**\n\n##### register_recovery_strategy\n```python\ndef register_recovery_strategy(self: Any, error_type: str, strategy: callable)\n```\n\n注册恢复策略\n\n##### handle_error\n```python\ndef handle_error(self: Any, error: Exception, context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n处理错误\n\n---\n\n#### EnhancedSystemOrchestrator\n\n增强系统协调器\n\n**Methods:**\n\n##### solve_math_problem\n```python\ndef solve_math_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n同步问题解决（保持向后兼容）\n\n##### batch_solve_problems\n```python\ndef batch_solve_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n同步批量解决问题\n\n##### shutdown_system_ordered\n```python\ndef shutdown_system_ordered(self: Any) -> bool\n```\n\n基于依赖图的有序关闭\n\n##### get_system_status\n```python\ndef get_system_status(self: Any) -> Dict[Unknown]\n```\n\n获取增强系统状态\n\n##### get_performance_report\n```python\ndef get_performance_report(self: Any) -> Dict[Unknown]\n```\n\n获取性能报告\n\n---\n\n### Functions\n\n#### add_dependency\n\n```python\ndef add_dependency(self: Any, module: str, depends_on: str)\n```\n\n添加依赖关系\n\n---\n\n#### remove_dependency\n\n```python\ndef remove_dependency(self: Any, module: str, depends_on: str)\n```\n\n移除依赖关系\n\n---\n\n#### get_shutdown_order\n\n```python\ndef get_shutdown_order(self: Any) -> List[str]\n```\n\n获取关闭顺序（拓扑排序）\n\n---\n\n#### has_cycle\n\n```python\ndef has_cycle(self: Any) -> bool\n```\n\n检查是否存在循环依赖\n\n---\n\n#### execute_parallel\n\n```python\ndef execute_parallel(self: Any, tasks: List[Unknown]) -> Dict[Unknown]\n```\n\n并行执行任务\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any)\n```\n\n关闭执行器\n\n---\n\n#### register_recovery_strategy\n\n```python\ndef register_recovery_strategy(self: Any, error_type: str, strategy: callable)\n```\n\n注册恢复策略\n\n---\n\n#### handle_error\n\n```python\ndef handle_error(self: Any, error: Exception, context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n处理错误\n\n---\n\n#### solve_math_problem\n\n```python\ndef solve_math_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n同步问题解决（保持向后兼容）\n\n---\n\n#### batch_solve_problems\n\n```python\ndef batch_solve_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n同步批量解决问题\n\n---\n\n#### shutdown_system_ordered\n\n```python\ndef shutdown_system_ordered(self: Any) -> bool\n```\n\n基于依赖图的有序关闭\n\n---\n\n#### get_system_status\n\n```python\ndef get_system_status(self: Any) -> Dict[Unknown]\n```\n\n获取增强系统状态\n\n---\n\n#### get_performance_report\n\n```python\ndef get_performance_report(self: Any) -> Dict[Unknown]\n```\n\n获取性能报告\n\n---\n\n#### dfs\n\n```python\ndef dfs(module: Any)\n```\n\n---\n\n\n## core.exceptions\n\n统一异常处理系统
提供项目中所有模块使用的标准异常类\n\n### Classes\n\n#### COTBaseException\n\nCOT项目基础异常类\n\n**Inherits from:** Exception\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### ValidationError\n\n数据验证异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### InputValidationError\n\n输入验证异常\n\n**Inherits from:** ValidationError\n\n---\n\n#### ProcessingError\n\n处理过程异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### ReasoningError\n\n推理过程异常\n\n**Inherits from:** ProcessingError\n\n---\n\n#### TemplateMatchingError\n\n模板匹配异常\n\n**Inherits from:** ProcessingError\n\n---\n\n#### TemplateError\n\n模板系统异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### ConfigurationError\n\n配置异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### ModuleError\n\n模块相关异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### ModuleRegistrationError\n\n模块注册异常\n\n**Inherits from:** ModuleError\n\n---\n\n#### ModuleNotFoundError\n\n模块未找到异常\n\n**Inherits from:** ModuleError\n\n---\n\n#### ModuleDependencyError\n\n模块依赖异常\n\n**Inherits from:** ModuleError\n\n---\n\n#### OrchestrationError\n\n系统协调异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### APIError\n\nAPI调用异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### PerformanceError\n\n性能异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### TimeoutError\n\n超时异常\n\n**Inherits from:** PerformanceError\n\n---\n\n#### SecurityError\n\n安全异常\n\n**Inherits from:** COTBaseException\n\n---\n\n#### AuthenticationError\n\n认证异常\n\n**Inherits from:** SecurityError\n\n---\n\n#### AuthorizationError\n\n授权异常\n\n**Inherits from:** SecurityError\n\n---\n\n#### ExceptionRecoveryStrategy\n\n异常恢复策略\n\n**Methods:**\n\n##### create_fallback_result\n```python\ndef create_fallback_result(error: COTBaseException) -> Dict[Unknown]\n```\n\n创建后备结果\n\n##### should_retry\n```python\ndef should_retry(error: COTBaseException, attempt: int, max_attempts: int) -> bool\n```\n\n判断是否应该重试\n\n---\n\n### Functions\n\n#### handle_exceptions\n\n```python\ndef handle_exceptions(default_return: Any, log_errors: Any, reraise_as: Any)\n```\n\n异常处理装饰器\n\n---\n\n#### handle_module_error\n\n```python\ndef handle_module_error(error: Exception, module_name: str, operation: str) -> COTBaseException\n```\n\n处理模块错误，统一错误格式\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### decorator\n\n```python\ndef decorator(func: Any)\n```\n\n---\n\n#### create_fallback_result\n\n```python\ndef create_fallback_result(error: COTBaseException) -> Dict[Unknown]\n```\n\n创建后备结果\n\n---\n\n#### should_retry\n\n```python\ndef should_retry(error: COTBaseException, attempt: int, max_attempts: int) -> bool\n```\n\n判断是否应该重试\n\n---\n\n#### wrapper\n\n```python\ndef wrapper()\n```\n\n---\n\n\n## core.interfaces\n\n核心接口定义
提供系统各组件的标准接口\n\n### Classes\n\n#### ProcessingStatus\n\n处理状态枚举\n\n**Inherits from:** Enum\n\n---\n\n#### ModuleType\n\n模块类型枚举\n\n**Inherits from:** Enum\n\n---\n\n#### ReasoningStep\n\n推理步骤类型\n\n**Inherits from:** Enum\n\n---\n\n#### ProcessingResult\n\n处理结果数据类\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n---\n\n#### ReasoningContext\n\n推理上下文\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n---\n\n#### ModuleInfo\n\n模块信息数据类\n\n---\n\n#### IProcessor\n\n处理器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### process\n```python\ndef process(self: Any, input_data: Any, context: Optional[ReasoningContext]) -> ProcessingResult\n```\n\n处理输入数据\n\n##### validate_input\n```python\ndef validate_input(self: Any, input_data: Any) -> bool\n```\n\n验证输入数据\n\n##### get_capabilities\n```python\ndef get_capabilities(self: Any) -> Dict[Unknown]\n```\n\n获取处理器能力描述\n\n---\n\n#### BaseProcessor\n\n基础处理器类\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### process\n```python\ndef process(self: Any, input_data: Any, context: Optional[ReasoningContext]) -> ProcessingResult\n```\n\n处理输入数据\n\n##### validate_input\n```python\ndef validate_input(self: Any, input_data: Any) -> bool\n```\n\n验证输入数据（默认实现）\n\n##### get_capabilities\n```python\ndef get_capabilities(self: Any) -> Dict[Unknown]\n```\n\n获取处理器能力描述（默认实现）\n\n---\n\n#### IValidator\n\n验证器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any, data: Any) -> Dict[Unknown]\n```\n\n验证数据\n\n##### get_validation_rules\n```python\ndef get_validation_rules(self: Any) -> Dict[Unknown]\n```\n\n获取验证规则\n\n---\n\n#### IReasoningEngine\n\n推理引擎接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### reason\n```python\ndef reason(self: Any, problem: str, context: Optional[ReasoningContext]) -> ProcessingResult\n```\n\n执行推理\n\n##### get_reasoning_steps\n```python\ndef get_reasoning_steps(self: Any) -> List[Unknown]\n```\n\n获取推理步骤\n\n##### set_reasoning_strategy\n```python\ndef set_reasoning_strategy(self: Any, strategy: str) -> Any\n```\n\n设置推理策略\n\n---\n\n#### ITemplateManager\n\n模板管理器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### match_template\n```python\ndef match_template(self: Any, text: str) -> Optional[Unknown]\n```\n\n匹配模板\n\n##### get_templates\n```python\ndef get_templates(self: Any) -> List[Unknown]\n```\n\n获取所有模板\n\n##### add_template\n```python\ndef add_template(self: Any, template: Dict[Unknown]) -> bool\n```\n\n添加模板\n\n---\n\n#### INumberExtractor\n\n数字提取器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### extract_numbers\n```python\ndef extract_numbers(self: Any, text: str) -> List[Unknown]\n```\n\n提取数字\n\n##### identify_number_patterns\n```python\ndef identify_number_patterns(self: Any, text: str) -> List[str]\n```\n\n识别数字模式\n\n---\n\n#### IConfidenceCalculator\n\n置信度计算器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### calculate_confidence\n```python\ndef calculate_confidence(self: Any, reasoning_steps: List[Unknown], result: Any) -> float\n```\n\n计算置信度\n\n##### get_confidence_factors\n```python\ndef get_confidence_factors(self: Any) -> List[str]\n```\n\n获取置信度影响因素\n\n---\n\n#### ICacheManager\n\n缓存管理器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### get\n```python\ndef get(self: Any, key: str) -> Optional[Any]\n```\n\n获取缓存\n\n##### set\n```python\ndef set(self: Any, key: str, value: Any, ttl: Optional[int]) -> bool\n```\n\n设置缓存\n\n##### delete\n```python\ndef delete(self: Any, key: str) -> bool\n```\n\n删除缓存\n\n##### clear\n```python\ndef clear(self: Any) -> bool\n```\n\n清空缓存\n\n---\n\n#### IMonitor\n\n监控器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### start_timer\n```python\ndef start_timer(self: Any, name: str) -> str\n```\n\n开始计时\n\n##### stop_timer\n```python\ndef stop_timer(self: Any, timer_id: str) -> Optional[float]\n```\n\n停止计时\n\n##### record_metric\n```python\ndef record_metric(self: Any, name: str, value: float, tags: Optional[Unknown]) -> Any\n```\n\n记录指标\n\n##### get_metrics_summary\n```python\ndef get_metrics_summary(self: Any) -> Dict[Unknown]\n```\n\n获取指标摘要\n\n---\n\n#### IGNNEnhancer\n\nGNN增强器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### enhance_reasoning\n```python\ndef enhance_reasoning(self: Any, reasoning_data: Dict[Unknown]) -> Dict[Unknown]\n```\n\n增强推理\n\n##### build_graph\n```python\ndef build_graph(self: Any, input_data: Any) -> Any\n```\n\n构建图\n\n##### predict\n```python\ndef predict(self: Any, graph_data: Any) -> Dict[Unknown]\n```\n\n执行预测\n\n---\n\n#### IConfigManager\n\n配置管理器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### get\n```python\ndef get(self: Any, key: str, default: Any) -> Any\n```\n\n获取配置\n\n##### set\n```python\ndef set(self: Any, key: str, value: Any, persist: bool) -> Any\n```\n\n设置配置\n\n##### reload_config\n```python\ndef reload_config(self: Any) -> Any\n```\n\n重新加载配置\n\n---\n\n#### IResultFormatter\n\n结果格式化器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### format_result\n```python\ndef format_result(self: Any, result: ProcessingResult, format_type: str) -> str\n```\n\n格式化结果\n\n##### get_supported_formats\n```python\ndef get_supported_formats(self: Any) -> List[str]\n```\n\n获取支持的格式\n\n---\n\n#### IComponentRegistry\n\n组件注册表接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### register_component\n```python\ndef register_component(self: Any, name: str, component: Any, component_type: str) -> bool\n```\n\n注册组件\n\n##### get_component\n```python\ndef get_component(self: Any, name: str) -> Optional[Any]\n```\n\n获取组件\n\n##### list_components\n```python\ndef list_components(self: Any, component_type: Optional[str]) -> List[str]\n```\n\n列出组件\n\n##### unregister_component\n```python\ndef unregister_component(self: Any, name: str) -> bool\n```\n\n注销组件\n\n---\n\n#### IEventHandler\n\n事件处理器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### handle_event\n```python\ndef handle_event(self: Any, event_type: str, event_data: Dict[Unknown]) -> Any\n```\n\n处理事件\n\n##### get_supported_events\n```python\ndef get_supported_events(self: Any) -> List[str]\n```\n\n获取支持的事件类型\n\n---\n\n#### IEventBus\n\n事件总线接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### subscribe\n```python\ndef subscribe(self: Any, event_type: str, handler: IEventHandler) -> Any\n```\n\n订阅事件\n\n##### unsubscribe\n```python\ndef unsubscribe(self: Any, event_type: str, handler: IEventHandler) -> Any\n```\n\n取消订阅\n\n##### publish\n```python\ndef publish(self: Any, event_type: str, event_data: Dict[Unknown]) -> Any\n```\n\n发布事件\n\n---\n\n#### IPlugin\n\n插件接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any, config: Dict[Unknown]) -> bool\n```\n\n初始化插件\n\n##### get_plugin_info\n```python\ndef get_plugin_info(self: Any) -> Dict[Unknown]\n```\n\n获取插件信息\n\n##### cleanup\n```python\ndef cleanup(self: Any) -> Any\n```\n\n清理插件资源\n\n---\n\n#### IPluginManager\n\n插件管理器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### load_plugin\n```python\ndef load_plugin(self: Any, plugin_path: str) -> bool\n```\n\n加载插件\n\n##### unload_plugin\n```python\ndef unload_plugin(self: Any, plugin_name: str) -> bool\n```\n\n卸载插件\n\n##### get_loaded_plugins\n```python\ndef get_loaded_plugins(self: Any) -> List[str]\n```\n\n获取已加载的插件\n\n##### get_plugin\n```python\ndef get_plugin(self: Any, plugin_name: str) -> Optional[IPlugin]\n```\n\n获取插件实例\n\n---\n\n#### IHealthChecker\n\n健康检查器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### check_health\n```python\ndef check_health(self: Any) -> Dict[Unknown]\n```\n\n检查系统健康状态\n\n##### get_health_status\n```python\ndef get_health_status(self: Any) -> str\n```\n\n获取健康状态\n\n##### register_health_check\n```python\ndef register_health_check(self: Any, name: str, check_func: callable) -> Any\n```\n\n注册健康检查函数\n\n---\n\n#### PublicAPI\n\n公共API基类\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any, config: Optional[Unknown]) -> bool\n```\n\n初始化API\n\n##### process_request\n```python\ndef process_request(self: Any, request: Dict[Unknown]) -> Dict[Unknown]\n```\n\n处理请求\n\n##### get_api_info\n```python\ndef get_api_info(self: Any) -> Dict[Unknown]\n```\n\n获取API信息\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> Any\n```\n\n关闭API\n\n---\n\n#### BaseOrchestrator\n\n基础系统编排器接口\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### initialize_system\n```python\ndef initialize_system(self: Any, config: Optional[Unknown]) -> bool\n```\n\n初始化系统\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决问题\n\n##### batch_solve_problems\n```python\ndef batch_solve_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决问题\n\n##### get_system_status\n```python\ndef get_system_status(self: Any) -> Dict[Unknown]\n```\n\n获取系统状态\n\n##### shutdown_system\n```python\ndef shutdown_system(self: Any) -> bool\n```\n\n关闭系统\n\n---\n\n### Functions\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n---\n\n#### process\n\n```python\ndef process(self: Any, input_data: Any, context: Optional[ReasoningContext]) -> ProcessingResult\n```\n\n处理输入数据\n\n---\n\n#### validate_input\n\n```python\ndef validate_input(self: Any, input_data: Any) -> bool\n```\n\n验证输入数据\n\n---\n\n#### get_capabilities\n\n```python\ndef get_capabilities(self: Any) -> Dict[Unknown]\n```\n\n获取处理器能力描述\n\n---\n\n#### process\n\n```python\ndef process(self: Any, input_data: Any, context: Optional[ReasoningContext]) -> ProcessingResult\n```\n\n处理输入数据\n\n---\n\n#### validate_input\n\n```python\ndef validate_input(self: Any, input_data: Any) -> bool\n```\n\n验证输入数据（默认实现）\n\n---\n\n#### get_capabilities\n\n```python\ndef get_capabilities(self: Any) -> Dict[Unknown]\n```\n\n获取处理器能力描述（默认实现）\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any, data: Any) -> Dict[Unknown]\n```\n\n验证数据\n\n---\n\n#### get_validation_rules\n\n```python\ndef get_validation_rules(self: Any) -> Dict[Unknown]\n```\n\n获取验证规则\n\n---\n\n#### reason\n\n```python\ndef reason(self: Any, problem: str, context: Optional[ReasoningContext]) -> ProcessingResult\n```\n\n执行推理\n\n---\n\n#### get_reasoning_steps\n\n```python\ndef get_reasoning_steps(self: Any) -> List[Unknown]\n```\n\n获取推理步骤\n\n---\n\n#### set_reasoning_strategy\n\n```python\ndef set_reasoning_strategy(self: Any, strategy: str) -> Any\n```\n\n设置推理策略\n\n---\n\n#### match_template\n\n```python\ndef match_template(self: Any, text: str) -> Optional[Unknown]\n```\n\n匹配模板\n\n---\n\n#### get_templates\n\n```python\ndef get_templates(self: Any) -> List[Unknown]\n```\n\n获取所有模板\n\n---\n\n#### add_template\n\n```python\ndef add_template(self: Any, template: Dict[Unknown]) -> bool\n```\n\n添加模板\n\n---\n\n#### extract_numbers\n\n```python\ndef extract_numbers(self: Any, text: str) -> List[Unknown]\n```\n\n提取数字\n\n---\n\n#### identify_number_patterns\n\n```python\ndef identify_number_patterns(self: Any, text: str) -> List[str]\n```\n\n识别数字模式\n\n---\n\n#### calculate_confidence\n\n```python\ndef calculate_confidence(self: Any, reasoning_steps: List[Unknown], result: Any) -> float\n```\n\n计算置信度\n\n---\n\n#### get_confidence_factors\n\n```python\ndef get_confidence_factors(self: Any) -> List[str]\n```\n\n获取置信度影响因素\n\n---\n\n#### get\n\n```python\ndef get(self: Any, key: str) -> Optional[Any]\n```\n\n获取缓存\n\n---\n\n#### set\n\n```python\ndef set(self: Any, key: str, value: Any, ttl: Optional[int]) -> bool\n```\n\n设置缓存\n\n---\n\n#### delete\n\n```python\ndef delete(self: Any, key: str) -> bool\n```\n\n删除缓存\n\n---\n\n#### clear\n\n```python\ndef clear(self: Any) -> bool\n```\n\n清空缓存\n\n---\n\n#### start_timer\n\n```python\ndef start_timer(self: Any, name: str) -> str\n```\n\n开始计时\n\n---\n\n#### stop_timer\n\n```python\ndef stop_timer(self: Any, timer_id: str) -> Optional[float]\n```\n\n停止计时\n\n---\n\n#### record_metric\n\n```python\ndef record_metric(self: Any, name: str, value: float, tags: Optional[Unknown]) -> Any\n```\n\n记录指标\n\n---\n\n#### get_metrics_summary\n\n```python\ndef get_metrics_summary(self: Any) -> Dict[Unknown]\n```\n\n获取指标摘要\n\n---\n\n#### enhance_reasoning\n\n```python\ndef enhance_reasoning(self: Any, reasoning_data: Dict[Unknown]) -> Dict[Unknown]\n```\n\n增强推理\n\n---\n\n#### build_graph\n\n```python\ndef build_graph(self: Any, input_data: Any) -> Any\n```\n\n构建图\n\n---\n\n#### predict\n\n```python\ndef predict(self: Any, graph_data: Any) -> Dict[Unknown]\n```\n\n执行预测\n\n---\n\n#### get\n\n```python\ndef get(self: Any, key: str, default: Any) -> Any\n```\n\n获取配置\n\n---\n\n#### set\n\n```python\ndef set(self: Any, key: str, value: Any, persist: bool) -> Any\n```\n\n设置配置\n\n---\n\n#### reload_config\n\n```python\ndef reload_config(self: Any) -> Any\n```\n\n重新加载配置\n\n---\n\n#### format_result\n\n```python\ndef format_result(self: Any, result: ProcessingResult, format_type: str) -> str\n```\n\n格式化结果\n\n---\n\n#### get_supported_formats\n\n```python\ndef get_supported_formats(self: Any) -> List[str]\n```\n\n获取支持的格式\n\n---\n\n#### register_component\n\n```python\ndef register_component(self: Any, name: str, component: Any, component_type: str) -> bool\n```\n\n注册组件\n\n---\n\n#### get_component\n\n```python\ndef get_component(self: Any, name: str) -> Optional[Any]\n```\n\n获取组件\n\n---\n\n#### list_components\n\n```python\ndef list_components(self: Any, component_type: Optional[str]) -> List[str]\n```\n\n列出组件\n\n---\n\n#### unregister_component\n\n```python\ndef unregister_component(self: Any, name: str) -> bool\n```\n\n注销组件\n\n---\n\n#### handle_event\n\n```python\ndef handle_event(self: Any, event_type: str, event_data: Dict[Unknown]) -> Any\n```\n\n处理事件\n\n---\n\n#### get_supported_events\n\n```python\ndef get_supported_events(self: Any) -> List[str]\n```\n\n获取支持的事件类型\n\n---\n\n#### subscribe\n\n```python\ndef subscribe(self: Any, event_type: str, handler: IEventHandler) -> Any\n```\n\n订阅事件\n\n---\n\n#### unsubscribe\n\n```python\ndef unsubscribe(self: Any, event_type: str, handler: IEventHandler) -> Any\n```\n\n取消订阅\n\n---\n\n#### publish\n\n```python\ndef publish(self: Any, event_type: str, event_data: Dict[Unknown]) -> Any\n```\n\n发布事件\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any, config: Dict[Unknown]) -> bool\n```\n\n初始化插件\n\n---\n\n#### get_plugin_info\n\n```python\ndef get_plugin_info(self: Any) -> Dict[Unknown]\n```\n\n获取插件信息\n\n---\n\n#### cleanup\n\n```python\ndef cleanup(self: Any) -> Any\n```\n\n清理插件资源\n\n---\n\n#### load_plugin\n\n```python\ndef load_plugin(self: Any, plugin_path: str) -> bool\n```\n\n加载插件\n\n---\n\n#### unload_plugin\n\n```python\ndef unload_plugin(self: Any, plugin_name: str) -> bool\n```\n\n卸载插件\n\n---\n\n#### get_loaded_plugins\n\n```python\ndef get_loaded_plugins(self: Any) -> List[str]\n```\n\n获取已加载的插件\n\n---\n\n#### get_plugin\n\n```python\ndef get_plugin(self: Any, plugin_name: str) -> Optional[IPlugin]\n```\n\n获取插件实例\n\n---\n\n#### check_health\n\n```python\ndef check_health(self: Any) -> Dict[Unknown]\n```\n\n检查系统健康状态\n\n---\n\n#### get_health_status\n\n```python\ndef get_health_status(self: Any) -> str\n```\n\n获取健康状态\n\n---\n\n#### register_health_check\n\n```python\ndef register_health_check(self: Any, name: str, check_func: callable) -> Any\n```\n\n注册健康检查函数\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any, config: Optional[Unknown]) -> bool\n```\n\n初始化API\n\n---\n\n#### process_request\n\n```python\ndef process_request(self: Any, request: Dict[Unknown]) -> Dict[Unknown]\n```\n\n处理请求\n\n---\n\n#### get_api_info\n\n```python\ndef get_api_info(self: Any) -> Dict[Unknown]\n```\n\n获取API信息\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> Any\n```\n\n关闭API\n\n---\n\n#### initialize_system\n\n```python\ndef initialize_system(self: Any, config: Optional[Unknown]) -> bool\n```\n\n初始化系统\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决问题\n\n---\n\n#### batch_solve_problems\n\n```python\ndef batch_solve_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决问题\n\n---\n\n#### get_system_status\n\n```python\ndef get_system_status(self: Any) -> Dict[Unknown]\n```\n\n获取系统状态\n\n---\n\n#### shutdown_system\n\n```python\ndef shutdown_system(self: Any) -> bool\n```\n\n关闭系统\n\n---\n\n\n## core.module_registry\n\n模块注册表实现

管理所有模块的注册、发现和生命周期。\n\n### Classes\n\n#### ModuleRegistryImpl\n\n模块注册表实现\n\n**Methods:**\n\n##### register_module\n```python\ndef register_module(self: Any, module_info: ModuleInfo, api_instance: PublicAPI) -> bool\n```\n\n注册模块\n\n##### get_module\n```python\ndef get_module(self: Any, module_name: str) -> Optional[PublicAPI]\n```\n\n获取模块实例\n\n##### list_modules\n```python\ndef list_modules(self: Any) -> List[ModuleInfo]\n```\n\n列出所有已注册模块\n\n##### unregister_module\n```python\ndef unregister_module(self: Any, module_name: str) -> bool\n```\n\n注销模块\n\n##### get_module_info\n```python\ndef get_module_info(self: Any, module_name: str) -> Optional[ModuleInfo]\n```\n\n获取模块信息\n\n##### is_module_registered\n```python\ndef is_module_registered(self: Any, module_name: str) -> bool\n```\n\n检查模块是否已注册\n\n##### get_modules_by_type\n```python\ndef get_modules_by_type(self: Any, module_type: ModuleType) -> List[PublicAPI]\n```\n\n根据类型获取模块\n\n##### health_check_all\n```python\ndef health_check_all(self: Any) -> Dict[Unknown]\n```\n\n对所有模块进行健康检查\n\n---\n\n### Functions\n\n#### register_module\n\n```python\ndef register_module(self: Any, module_info: ModuleInfo, api_instance: PublicAPI) -> bool\n```\n\n注册模块\n\n---\n\n#### get_module\n\n```python\ndef get_module(self: Any, module_name: str) -> Optional[PublicAPI]\n```\n\n获取模块实例\n\n---\n\n#### list_modules\n\n```python\ndef list_modules(self: Any) -> List[ModuleInfo]\n```\n\n列出所有已注册模块\n\n---\n\n#### unregister_module\n\n```python\ndef unregister_module(self: Any, module_name: str) -> bool\n```\n\n注销模块\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any, module_name: str) -> Optional[ModuleInfo]\n```\n\n获取模块信息\n\n---\n\n#### is_module_registered\n\n```python\ndef is_module_registered(self: Any, module_name: str) -> bool\n```\n\n检查模块是否已注册\n\n---\n\n#### get_modules_by_type\n\n```python\ndef get_modules_by_type(self: Any, module_type: ModuleType) -> List[PublicAPI]\n```\n\n根据类型获取模块\n\n---\n\n#### health_check_all\n\n```python\ndef health_check_all(self: Any) -> Dict[Unknown]\n```\n\n对所有模块进行健康检查\n\n---\n\n\n## core.orchestration_strategy\n\n统一协调器策略模式

创建可配置的协调器架构，消除多个重复的协调器类。\n\n### Classes\n\n#### OrchestrationStrategy\n\n协调器策略类型\n\n**Inherits from:** Enum\n\n---\n\n#### BaseOrchestratorStrategy\n\n协调器策略基类\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化策略\n\n##### execute_operation\n```python\ndef execute_operation(self: Any, operation: str) -> Any\n```\n\n执行操作\n\n##### get_capabilities\n```python\ndef get_capabilities(self: Any) -> List[str]\n```\n\n获取策略能力列表\n\n##### validate_operation\n```python\ndef validate_operation(self: Any, operation: str) -> bool\n```\n\n验证操作有效性\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭策略\n\n---\n\n#### UnifiedStrategy\n\n统一协调器策略 - 支持所有模块协调\n\n**Inherits from:** BaseOrchestratorStrategy\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化统一策略\n\n##### execute_operation\n```python\ndef execute_operation(self: Any, operation: str) -> Any\n```\n\n执行统一操作\n\n##### get_capabilities\n```python\ndef get_capabilities(self: Any) -> List[str]\n```\n\n获取统一策略能力\n\n---\n\n#### ReasoningStrategy\n\n推理专用协调器策略\n\n**Inherits from:** BaseOrchestratorStrategy\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化推理策略\n\n##### execute_operation\n```python\ndef execute_operation(self: Any, operation: str) -> Any\n```\n\n执行推理操作\n\n##### get_capabilities\n```python\ndef get_capabilities(self: Any) -> List[str]\n```\n\n获取推理策略能力\n\n---\n\n#### ProcessingStrategy\n\n处理专用协调器策略\n\n**Inherits from:** BaseOrchestratorStrategy\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化处理策略\n\n##### execute_operation\n```python\ndef execute_operation(self: Any, operation: str) -> Any\n```\n\n执行处理操作\n\n##### get_capabilities\n```python\ndef get_capabilities(self: Any) -> List[str]\n```\n\n获取处理策略能力\n\n---\n\n#### OrchestratorStrategyFactory\n\n协调器策略工厂\n\n**Methods:**\n\n##### create_strategy\n```python\ndef create_strategy(cls: Any, strategy_type: OrchestrationStrategy, config: Optional[Unknown]) -> BaseOrchestratorStrategy\n```\n\n创建协调器策略\n\n##### get_available_strategies\n```python\ndef get_available_strategies(cls: Any) -> List[OrchestrationStrategy]\n```\n\n获取可用策略列表\n\n---\n\n#### MockReasoningProcessor\n\n**Methods:**\n\n##### process_reasoning\n```python\ndef process_reasoning(self: Any, problem: Any)\n```\n\n---\n\n#### MockConfidenceCalculator\n\n**Methods:**\n\n##### calculate\n```python\ndef calculate(self: Any, solution: Any)\n```\n\n---\n\n#### MockReasoningValidator\n\n**Methods:**\n\n##### validate_input\n```python\ndef validate_input(self: Any, problem: Any)\n```\n\n##### validate_chain\n```python\ndef validate_chain(self: Any, reasoning_chain: Any)\n```\n\n---\n\n#### MockProcessor\n\n**Methods:**\n\n##### process\n```python\ndef process(self: Any, data: Any)\n```\n\n---\n\n#### MockValidator\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any, data: Any)\n```\n\n---\n\n#### MockUtils\n\n**Methods:**\n\n##### format_data\n```python\ndef format_data(data: Any)\n```\n\n---\n\n#### MockReasoningProcessor\n\n**Methods:**\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem: Any)\n```\n\n---\n\n#### MockCoreProcessor\n\n**Methods:**\n\n##### process\n```python\ndef process(self: Any, data: Any)\n```\n\n---\n\n#### MockModelFactory\n\n**Methods:**\n\n##### create_model\n```python\ndef create_model(self: Any, name: Any, config: Any)\n```\n\n---\n\n#### MockModel\n\n**Methods:**\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem: Any)\n```\n\n---\n\n### Functions\n\n#### create_orchestrator_strategy\n\n```python\ndef create_orchestrator_strategy(strategy: Union[Unknown], config: Optional[Unknown]) -> BaseOrchestratorStrategy\n```\n\n创建协调器策略的便利函数\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化策略\n\n---\n\n#### execute_operation\n\n```python\ndef execute_operation(self: Any, operation: str) -> Any\n```\n\n执行操作\n\n---\n\n#### get_capabilities\n\n```python\ndef get_capabilities(self: Any) -> List[str]\n```\n\n获取策略能力列表\n\n---\n\n#### validate_operation\n\n```python\ndef validate_operation(self: Any, operation: str) -> bool\n```\n\n验证操作有效性\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭策略\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化统一策略\n\n---\n\n#### execute_operation\n\n```python\ndef execute_operation(self: Any, operation: str) -> Any\n```\n\n执行统一操作\n\n---\n\n#### get_capabilities\n\n```python\ndef get_capabilities(self: Any) -> List[str]\n```\n\n获取统一策略能力\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化推理策略\n\n---\n\n#### execute_operation\n\n```python\ndef execute_operation(self: Any, operation: str) -> Any\n```\n\n执行推理操作\n\n---\n\n#### get_capabilities\n\n```python\ndef get_capabilities(self: Any) -> List[str]\n```\n\n获取推理策略能力\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化处理策略\n\n---\n\n#### execute_operation\n\n```python\ndef execute_operation(self: Any, operation: str) -> Any\n```\n\n执行处理操作\n\n---\n\n#### get_capabilities\n\n```python\ndef get_capabilities(self: Any) -> List[str]\n```\n\n获取处理策略能力\n\n---\n\n#### create_strategy\n\n```python\ndef create_strategy(cls: Any, strategy_type: OrchestrationStrategy, config: Optional[Unknown]) -> BaseOrchestratorStrategy\n```\n\n创建协调器策略\n\n---\n\n#### get_available_strategies\n\n```python\ndef get_available_strategies(cls: Any) -> List[OrchestrationStrategy]\n```\n\n获取可用策略列表\n\n---\n\n#### process_reasoning\n\n```python\ndef process_reasoning(self: Any, problem: Any)\n```\n\n---\n\n#### calculate\n\n```python\ndef calculate(self: Any, solution: Any)\n```\n\n---\n\n#### validate_input\n\n```python\ndef validate_input(self: Any, problem: Any)\n```\n\n---\n\n#### validate_chain\n\n```python\ndef validate_chain(self: Any, reasoning_chain: Any)\n```\n\n---\n\n#### process\n\n```python\ndef process(self: Any, data: Any)\n```\n\n---\n\n#### validate\n\n```python\ndef validate(self: Any, data: Any)\n```\n\n---\n\n#### format_data\n\n```python\ndef format_data(data: Any)\n```\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem: Any)\n```\n\n---\n\n#### process\n\n```python\ndef process(self: Any, data: Any)\n```\n\n---\n\n#### create_model\n\n```python\ndef create_model(self: Any, name: Any, config: Any)\n```\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem: Any)\n```\n\n---\n\n\n## core.orchestrator\n\n统一系统协调器

整合基础协调器和增强协调器功能，提供完整的系统管理和问题解决能力。
采用策略模式支持不同类型的协调需求。\n\n### Classes\n\n#### DependencyGraph\n\n模块依赖图管理器\n\n**Methods:**\n\n##### add_dependency\n```python\ndef add_dependency(self: Any, module: str, depends_on: str)\n```\n\n添加依赖关系\n\n##### remove_dependency\n```python\ndef remove_dependency(self: Any, module: str, depends_on: str)\n```\n\n移除依赖关系\n\n##### get_shutdown_order\n```python\ndef get_shutdown_order(self: Any) -> List[str]\n```\n\n获取关闭顺序（拓扑排序）\n\n##### has_cycle\n```python\ndef has_cycle(self: Any) -> bool\n```\n\n检查是否存在循环依赖\n\n---\n\n#### ConcurrentExecutor\n\n并发执行器\n\n**Methods:**\n\n##### execute_parallel\n```python\ndef execute_parallel(self: Any, tasks: List[Unknown]) -> Dict[Unknown]\n```\n\n并行执行任务\n\n##### shutdown\n```python\ndef shutdown(self: Any)\n```\n\n关闭执行器\n\n---\n\n#### ErrorRecoveryManager\n\n错误恢复管理器\n\n**Methods:**\n\n##### register_recovery_strategy\n```python\ndef register_recovery_strategy(self: Any, error_type: str, strategy: callable)\n```\n\n注册恢复策略\n\n##### handle_error\n```python\ndef handle_error(self: Any, error: Exception, context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n处理错误\n\n---\n\n#### UnifiedSystemOrchestrator\n\n统一系统协调器 - 整合基础和增强协调器功能\n\n**Methods:**\n\n##### solve_math_problem\n```python\ndef solve_math_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决数学问题的系统级流程协调

使用策略模式进行问题求解，支持不同的协调策略\n\n##### batch_solve_problems\n```python\ndef batch_solve_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决数学问题 - 使用策略模式\n\n##### initialize_system\n```python\ndef initialize_system(self: Any, config: Optional[Unknown]) -> bool\n```\n\n初始化整个系统\n\n##### get_system_status\n```python\ndef get_system_status(self: Any) -> Dict[Unknown]\n```\n\n获取系统状态\n\n##### shutdown_system\n```python\ndef shutdown_system(self: Any) -> bool\n```\n\n有序关闭系统\n\n##### get_performance_report\n```python\ndef get_performance_report(self: Any) -> Dict[Unknown]\n```\n\n获取性能报告\n\n---\n\n### Functions\n\n#### add_dependency\n\n```python\ndef add_dependency(self: Any, module: str, depends_on: str)\n```\n\n添加依赖关系\n\n---\n\n#### remove_dependency\n\n```python\ndef remove_dependency(self: Any, module: str, depends_on: str)\n```\n\n移除依赖关系\n\n---\n\n#### get_shutdown_order\n\n```python\ndef get_shutdown_order(self: Any) -> List[str]\n```\n\n获取关闭顺序（拓扑排序）\n\n---\n\n#### has_cycle\n\n```python\ndef has_cycle(self: Any) -> bool\n```\n\n检查是否存在循环依赖\n\n---\n\n#### execute_parallel\n\n```python\ndef execute_parallel(self: Any, tasks: List[Unknown]) -> Dict[Unknown]\n```\n\n并行执行任务\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any)\n```\n\n关闭执行器\n\n---\n\n#### register_recovery_strategy\n\n```python\ndef register_recovery_strategy(self: Any, error_type: str, strategy: callable)\n```\n\n注册恢复策略\n\n---\n\n#### handle_error\n\n```python\ndef handle_error(self: Any, error: Exception, context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n处理错误\n\n---\n\n#### solve_math_problem\n\n```python\ndef solve_math_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决数学问题的系统级流程协调

使用策略模式进行问题求解，支持不同的协调策略\n\n---\n\n#### batch_solve_problems\n\n```python\ndef batch_solve_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决数学问题 - 使用策略模式\n\n---\n\n#### initialize_system\n\n```python\ndef initialize_system(self: Any, config: Optional[Unknown]) -> bool\n```\n\n初始化整个系统\n\n---\n\n#### get_system_status\n\n```python\ndef get_system_status(self: Any) -> Dict[Unknown]\n```\n\n获取系统状态\n\n---\n\n#### shutdown_system\n\n```python\ndef shutdown_system(self: Any) -> bool\n```\n\n有序关闭系统\n\n---\n\n#### get_performance_report\n\n```python\ndef get_performance_report(self: Any) -> Dict[Unknown]\n```\n\n获取性能报告\n\n---\n\n#### dfs\n\n```python\ndef dfs(module: Any)\n```\n\n---\n\n\n## core.problem_solver_interface\n\n统一问题求解接口

创建标准化的问题求解接口，使用模板方法模式消除代码重复。\n\n### Classes\n\n#### ProblemType\n\n问题类型枚举\n\n**Inherits from:** Enum\n\n---\n\n#### SolutionStrategy\n\n解决策略枚举\n\n**Inherits from:** Enum\n\n---\n\n#### ProblemInput\n\n标准化问题输入\n\n**Methods:**\n\n##### get_text\n```python\ndef get_text(self: Any) -> str\n```\n\n获取问题文本\n\n##### get_data\n```python\ndef get_data(self: Any) -> Dict[Unknown]\n```\n\n获取问题数据\n\n##### get_type\n```python\ndef get_type(self: Any) -> ProblemType\n```\n\n获取问题类型\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### ProblemOutput\n\n标准化问题输出\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n##### from_dict\n```python\ndef from_dict(cls: Any, data: Dict[Unknown]) -> Any\n```\n\n从字典创建输出对象\n\n---\n\n#### BaseProblemSolver\n\n问题求解器基类 - 使用模板方法模式\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem: Union[Unknown]) -> ProblemOutput\n```\n\n解决问题的模板方法

定义了问题求解的标准流程：
1. 预处理
2. 核心求解
3. 后处理\n\n##### preprocess\n```python\ndef preprocess(self: Any, problem_input: ProblemInput) -> ProblemInput\n```\n\n预处理步骤（可被子类重写）

默认实现：基本的文本清理和问题类型识别\n\n##### core_solve\n```python\ndef core_solve(self: Any, problem_input: ProblemInput) -> ProblemOutput\n```\n\n核心求解逻辑（子类必须实现）

Args:
    problem_input: 预处理后的问题输入
    
Returns:
    原始求解结果\n\n##### postprocess\n```python\ndef postprocess(self: Any, raw_output: ProblemOutput, problem_input: ProblemInput) -> ProblemOutput\n```\n\n后处理步骤（可被子类重写）

默认实现：基本的结果格式化\n\n##### validate_output\n```python\ndef validate_output(self: Any, output: ProblemOutput) -> ProblemOutput\n```\n\n验证输出结果（可被子类重写）

默认实现：基本的有效性检查\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[Unknown]) -> List[ProblemOutput]\n```\n\n批量解决问题\n\n##### get_statistics\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取求解器统计信息\n\n##### set_strategy\n```python\ndef set_strategy(self: Any, strategy: SolutionStrategy)\n```\n\n设置求解策略\n\n##### get_strategy\n```python\ndef get_strategy(self: Any) -> SolutionStrategy\n```\n\n获取当前求解策略\n\n---\n\n#### ChainOfThoughtSolver\n\n链式思维求解器实现示例\n\n**Inherits from:** BaseProblemSolver\n\n**Methods:**\n\n##### core_solve\n```python\ndef core_solve(self: Any, problem_input: ProblemInput) -> ProblemOutput\n```\n\n使用链式思维进行求解\n\n---\n\n#### DirectReasoningSolver\n\n直接推理求解器实现示例\n\n**Inherits from:** BaseProblemSolver\n\n**Methods:**\n\n##### core_solve\n```python\ndef core_solve(self: Any, problem_input: ProblemInput) -> ProblemOutput\n```\n\n使用直接推理进行求解\n\n---\n\n#### ProblemSolverFactory\n\n问题求解器工厂\n\n**Methods:**\n\n##### create_solver\n```python\ndef create_solver(cls: Any, strategy: SolutionStrategy, config: Optional[Unknown]) -> BaseProblemSolver\n```\n\n创建问题求解器\n\n##### register_solver\n```python\ndef register_solver(cls: Any, strategy: SolutionStrategy, solver_class: type)\n```\n\n注册新的求解器\n\n##### get_available_strategies\n```python\ndef get_available_strategies(cls: Any) -> List[SolutionStrategy]\n```\n\n获取可用策略列表\n\n---\n\n### Functions\n\n#### create_problem_solver\n\n```python\ndef create_problem_solver(strategy: Union[Unknown], config: Optional[Unknown]) -> BaseProblemSolver\n```\n\n创建问题求解器的便利函数\n\n---\n\n#### solve_problem_unified\n\n```python\ndef solve_problem_unified(problem: Union[Unknown], strategy: Union[Unknown], config: Optional[Unknown]) -> Dict[Unknown]\n```\n\n统一问题求解函数\n\n---\n\n#### get_text\n\n```python\ndef get_text(self: Any) -> str\n```\n\n获取问题文本\n\n---\n\n#### get_data\n\n```python\ndef get_data(self: Any) -> Dict[Unknown]\n```\n\n获取问题数据\n\n---\n\n#### get_type\n\n```python\ndef get_type(self: Any) -> ProblemType\n```\n\n获取问题类型\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### from_dict\n\n```python\ndef from_dict(cls: Any, data: Dict[Unknown]) -> Any\n```\n\n从字典创建输出对象\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem: Union[Unknown]) -> ProblemOutput\n```\n\n解决问题的模板方法

定义了问题求解的标准流程：
1. 预处理
2. 核心求解
3. 后处理\n\n---\n\n#### preprocess\n\n```python\ndef preprocess(self: Any, problem_input: ProblemInput) -> ProblemInput\n```\n\n预处理步骤（可被子类重写）

默认实现：基本的文本清理和问题类型识别\n\n---\n\n#### core_solve\n\n```python\ndef core_solve(self: Any, problem_input: ProblemInput) -> ProblemOutput\n```\n\n核心求解逻辑（子类必须实现）

Args:
    problem_input: 预处理后的问题输入
    
Returns:
    原始求解结果\n\n---\n\n#### postprocess\n\n```python\ndef postprocess(self: Any, raw_output: ProblemOutput, problem_input: ProblemInput) -> ProblemOutput\n```\n\n后处理步骤（可被子类重写）

默认实现：基本的结果格式化\n\n---\n\n#### validate_output\n\n```python\ndef validate_output(self: Any, output: ProblemOutput) -> ProblemOutput\n```\n\n验证输出结果（可被子类重写）

默认实现：基本的有效性检查\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[Unknown]) -> List[ProblemOutput]\n```\n\n批量解决问题\n\n---\n\n#### get_statistics\n\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取求解器统计信息\n\n---\n\n#### set_strategy\n\n```python\ndef set_strategy(self: Any, strategy: SolutionStrategy)\n```\n\n设置求解策略\n\n---\n\n#### get_strategy\n\n```python\ndef get_strategy(self: Any) -> SolutionStrategy\n```\n\n获取当前求解策略\n\n---\n\n#### core_solve\n\n```python\ndef core_solve(self: Any, problem_input: ProblemInput) -> ProblemOutput\n```\n\n使用链式思维进行求解\n\n---\n\n#### core_solve\n\n```python\ndef core_solve(self: Any, problem_input: ProblemInput) -> ProblemOutput\n```\n\n使用直接推理进行求解\n\n---\n\n#### create_solver\n\n```python\ndef create_solver(cls: Any, strategy: SolutionStrategy, config: Optional[Unknown]) -> BaseProblemSolver\n```\n\n创建问题求解器\n\n---\n\n#### register_solver\n\n```python\ndef register_solver(cls: Any, strategy: SolutionStrategy, solver_class: type)\n```\n\n注册新的求解器\n\n---\n\n#### get_available_strategies\n\n```python\ndef get_available_strategies(cls: Any) -> List[SolutionStrategy]\n```\n\n获取可用策略列表\n\n---\n\n\n## core.security_service\n\n共享安全服务

提供单例的安全计算器和其他安全工具，消除代码重复。\n\n### Classes\n\n#### SecurityService\n\n安全服务单例类\n\n**Methods:**\n\n##### get_secure_evaluator\n```python\ndef get_secure_evaluator(self: Any)\n```\n\n获取安全数学计算器\n\n##### get_secure_file_manager\n```python\ndef get_secure_file_manager(self: Any)\n```\n\n获取安全文件管理器\n\n##### get_secure_config_manager\n```python\ndef get_secure_config_manager(self: Any, config_dir: Optional[str])\n```\n\n获取安全配置管理器\n\n##### safe_eval\n```python\ndef safe_eval(self: Any, expression: str, allowed_names: Optional[Unknown]) -> Union[Unknown]\n```\n\n安全计算数学表达式\n\n---\n\n#### FallbackSecureEvaluator\n\n**Methods:**\n\n##### safe_eval\n```python\ndef safe_eval(self: Any, expression: str, allowed_names: Optional[Unknown]) -> Union[Unknown]\n```\n\n---\n\n### Functions\n\n#### get_security_service\n\n```python\ndef get_security_service() -> SecurityService\n```\n\n获取全局安全服务实例\n\n---\n\n#### get_secure_evaluator\n\n```python\ndef get_secure_evaluator()\n```\n\n便利函数：获取安全计算器\n\n---\n\n#### safe_eval\n\n```python\ndef safe_eval(expression: str, allowed_names: Optional[Unknown]) -> Union[Unknown]\n```\n\n便利函数：安全计算数学表达式\n\n---\n\n#### get_secure_file_manager\n\n```python\ndef get_secure_file_manager()\n```\n\n便利函数：获取安全文件管理器\n\n---\n\n#### get_secure_config_manager\n\n```python\ndef get_secure_config_manager(config_dir: Optional[str])\n```\n\n便利函数：获取安全配置管理器\n\n---\n\n#### get_secure_evaluator\n\n```python\ndef get_secure_evaluator(self: Any)\n```\n\n获取安全数学计算器\n\n---\n\n#### get_secure_file_manager\n\n```python\ndef get_secure_file_manager(self: Any)\n```\n\n获取安全文件管理器\n\n---\n\n#### get_secure_config_manager\n\n```python\ndef get_secure_config_manager(self: Any, config_dir: Optional[str])\n```\n\n获取安全配置管理器\n\n---\n\n#### safe_eval\n\n```python\ndef safe_eval(self: Any, expression: str, allowed_names: Optional[Unknown]) -> Union[Unknown]\n```\n\n安全计算数学表达式\n\n---\n\n#### safe_eval\n\n```python\ndef safe_eval(self: Any, expression: str, allowed_names: Optional[Unknown]) -> Union[Unknown]\n```\n\n---\n\n\n## core.system_orchestrator\n\n系统级协调器

管理所有模块间的协作和系统级操作。\n\n### Classes\n\n#### SystemOrchestrator\n\n系统级协调器\n\n**Methods:**\n\n##### solve_math_problem\n```python\ndef solve_math_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决数学问题的系统级流程协调

这是系统的主要入口点，协调多个模块来解决数学问题。\n\n##### batch_solve_problems\n```python\ndef batch_solve_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决数学问题\n\n##### initialize_system\n```python\ndef initialize_system(self: Any) -> bool\n```\n\n初始化整个系统\n\n##### get_system_status\n```python\ndef get_system_status(self: Any) -> Dict[Unknown]\n```\n\n获取系统状态\n\n##### shutdown_system\n```python\ndef shutdown_system(self: Any) -> bool\n```\n\n关闭系统\n\n---\n\n### Functions\n\n#### solve_math_problem\n\n```python\ndef solve_math_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决数学问题的系统级流程协调

这是系统的主要入口点，协调多个模块来解决数学问题。\n\n---\n\n#### batch_solve_problems\n\n```python\ndef batch_solve_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决数学问题\n\n---\n\n#### initialize_system\n\n```python\ndef initialize_system(self: Any) -> bool\n```\n\n初始化整个系统\n\n---\n\n#### get_system_status\n\n```python\ndef get_system_status(self: Any) -> Dict[Unknown]\n```\n\n获取系统状态\n\n---\n\n#### shutdown_system\n\n```python\ndef shutdown_system(self: Any) -> bool\n```\n\n关闭系统\n\n---\n\n\n## data.dataset_characteristics\n\nDataset Characteristics with DIR-MWP Complexity Distribution

This module contains the dataset characteristics from Table 3, including
dataset size, language, domain, complexity distribution (L0-L3), and DIR scores.\n\n### Classes\n\n#### DatasetInfo\n\nData class for storing dataset characteristics.\n\n---\n\n### Functions\n\n#### get_dataset_info\n\n```python\ndef get_dataset_info(dataset_name: str) -> DatasetInfo\n```\n\nGet dataset information by name.

Args:
    dataset_name: Name of the dataset
    
Returns:
    DatasetInfo object with dataset characteristics
    
Raises:
    KeyError: If dataset name is not found\n\n---\n\n#### get_all_datasets\n\n```python\ndef get_all_datasets() -> Dict[Unknown]\n```\n\nGet all dataset characteristics.

Returns:
    Dictionary mapping dataset names to DatasetInfo objects\n\n---\n\n#### get_datasets_by_language\n\n```python\ndef get_datasets_by_language(language: str) -> List[DatasetInfo]\n```\n\nGet datasets filtered by language.

Args:
    language: Language to filter by (e.g., "English", "Chinese", "Mixed")
    
Returns:
    List of DatasetInfo objects matching the language\n\n---\n\n#### get_datasets_by_domain\n\n```python\ndef get_datasets_by_domain(domain: str) -> List[DatasetInfo]\n```\n\nGet datasets filtered by domain.

Args:
    domain: Domain to filter by (e.g., "Elementary", "Grade School", "Competition")
    
Returns:
    List of DatasetInfo objects matching the domain\n\n---\n\n#### get_complexity_distribution\n\n```python\ndef get_complexity_distribution(dataset_name: str) -> Dict[Unknown]\n```\n\nGet complexity level distribution for a dataset.

Args:
    dataset_name: Name of the dataset
    
Returns:
    Dictionary with complexity levels (L0-L3) and their percentages\n\n---\n\n#### calculate_weighted_complexity_score\n\n```python\ndef calculate_weighted_complexity_score(dataset_name: str) -> float\n```\n\nCalculate weighted complexity score based on distribution.

Args:
    dataset_name: Name of the dataset
    
Returns:
    Weighted complexity score (0-3 scale)\n\n---\n\n#### get_dataset_statistics\n\n```python\ndef get_dataset_statistics() -> Dict[Unknown]\n```\n\nGet overall statistics across all datasets.

Returns:
    Dictionary with various statistics\n\n---\n\n#### export_to_json\n\n```python\ndef export_to_json(filename: str) -> Any\n```\n\nExport dataset characteristics to JSON file.

Args:
    filename: Output filename\n\n---\n\n#### print_dataset_table\n\n```python\ndef print_dataset_table() -> Any\n```\n\nPrint a formatted table of all dataset characteristics.\n\n---\n\n\n## data.export_utils\n\nExport utilities for dataset characteristics.

This module provides functions to export dataset characteristics to various formats.\n\n### Functions\n\n#### export_to_csv\n\n```python\ndef export_to_csv(filename: str) -> Any\n```\n\nExport dataset characteristics to CSV file.

Args:
    filename: Output filename\n\n---\n\n#### export_statistics_to_csv\n\n```python\ndef export_statistics_to_csv(filename: str) -> Any\n```\n\nExport overall statistics to CSV file.

Args:
    filename: Output filename\n\n---\n\n#### export_complexity_matrix_to_csv\n\n```python\ndef export_complexity_matrix_to_csv(filename: str) -> Any\n```\n\nExport complexity distribution matrix to CSV file.

Args:
    filename: Output filename\n\n---\n\n#### export_markdown_table\n\n```python\ndef export_markdown_table(filename: str) -> Any\n```\n\nExport dataset characteristics as a markdown table.

Args:
    filename: Output filename\n\n---\n\n#### export_all_formats\n\n```python\ndef export_all_formats(base_filename: str) -> Any\n```\n\nExport dataset characteristics to all supported formats.

Args:
    base_filename: Base filename (extensions will be added automatically)\n\n---\n\n\n## data.loader\n\n### Classes\n\n#### DataLoader\n\nMinimal DataLoader for loading math datasets from Data/ directory.\n\n**Methods:**\n\n##### load\n```python\ndef load(self: Any, dataset_name: Optional[str], path: Optional[str], max_samples: Optional[int]) -> List[Dict]\n```\n\nLoad dataset by name (from Data/ directory) or by path. Supports .json and .jsonl files.
Args:
    dataset_name: Name of the dataset (e.g., 'Math23K')
    path: Direct path to dataset file
    max_samples: Maximum number of samples to load
Returns:
    List of dict samples\n\n---\n\n### Functions\n\n#### load\n\n```python\ndef load(self: Any, dataset_name: Optional[str], path: Optional[str], max_samples: Optional[int]) -> List[Dict]\n```\n\nLoad dataset by name (from Data/ directory) or by path. Supports .json and .jsonl files.
Args:
    dataset_name: Name of the dataset (e.g., 'Math23K')
    path: Direct path to dataset file
    max_samples: Maximum number of samples to load
Returns:
    List of dict samples\n\n---\n\n\n## data.orchestrator\n\nData Module - Orchestrator
==========================

数据模块协调器

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### DataOrchestrator\n\n数据模块协调器\n\n**Methods:**\n\n##### initialize_orchestrator\n```python\ndef initialize_orchestrator(self: Any) -> bool\n```\n\n初始化协调器\n\n##### orchestrate\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行\n\n##### register_component\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件\n\n##### get_component\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件\n\n##### get_operation_history\n```python\ndef get_operation_history(self: Any) -> List[Unknown]\n```\n\n获取操作历史\n\n##### clear_operation_history\n```python\ndef clear_operation_history(self: Any) -> Any\n```\n\n清空操作历史\n\n---\n\n### Functions\n\n#### initialize_orchestrator\n\n```python\ndef initialize_orchestrator(self: Any) -> bool\n```\n\n初始化协调器\n\n---\n\n#### orchestrate\n\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行\n\n---\n\n#### register_component\n\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件\n\n---\n\n#### get_component\n\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件\n\n---\n\n#### get_operation_history\n\n```python\ndef get_operation_history(self: Any) -> List[Unknown]\n```\n\n获取操作历史\n\n---\n\n#### clear_operation_history\n\n```python\ndef clear_operation_history(self: Any) -> Any\n```\n\n清空操作历史\n\n---\n\n\n## data.performance_analysis\n\nPerformance Analysis Data Module

This module contains performance analysis data from multiple evaluation tables,
including computational efficiency, ablation studies, performance comparisons,
relation discovery, and reasoning chain quality assessments.\n\n### Classes\n\n#### MethodPerformance\n\nData class for method performance across datasets.\n\n---\n\n#### ComplexityPerformance\n\nData class for performance by complexity level.\n\n---\n\n#### EfficiencyMetrics\n\nData class for computational efficiency metrics.\n\n---\n\n#### AblationResults\n\nData class for ablation study results.\n\n---\n\n#### ComponentInteraction\n\nData class for component interaction analysis.\n\n---\n\n#### RelationDiscoveryMetrics\n\nData class for implicit relation discovery assessment.\n\n---\n\n#### ReasoningChainMetrics\n\nData class for reasoning chain quality assessment.\n\n---\n\n### Functions\n\n#### get_method_performance\n\n```python\ndef get_method_performance(method_name: str) -> Optional[MethodPerformance]\n```\n\nGet performance data for a specific method.\n\n---\n\n#### get_all_methods\n\n```python\ndef get_all_methods() -> List[str]\n```\n\nGet list of all available methods.\n\n---\n\n#### get_best_performing_method\n\n```python\ndef get_best_performing_method(dataset: str) -> Tuple[Unknown]\n```\n\nGet the best performing method for a specific dataset.\n\n---\n\n#### calculate_average_performance\n\n```python\ndef calculate_average_performance(method_name: str) -> float\n```\n\nCalculate average performance across all datasets for a method.\n\n---\n\n#### get_efficiency_ranking\n\n```python\ndef get_efficiency_ranking() -> List[Unknown]\n```\n\nGet methods ranked by efficiency score.\n\n---\n\n#### get_robustness_ranking\n\n```python\ndef get_robustness_ranking() -> List[Unknown]\n```\n\nGet methods ranked by robustness score.\n\n---\n\n#### analyze_component_contribution\n\n```python\ndef analyze_component_contribution() -> Dict[Unknown]\n```\n\nAnalyze the contribution of each component in COT-DIR.\n\n---\n\n#### export_performance_data\n\n```python\ndef export_performance_data(filename: str) -> Any\n```\n\nExport all performance data to JSON file.\n\n---\n\n\n## data.preprocessor\n\n### Classes\n\n#### Preprocessor\n\nMinimal Preprocessor for math problem samples.
Adds cleaned_text, problem_type, classification_confidence, complexity_level.\n\n**Methods:**\n\n##### process\n```python\ndef process(self: Any, sample: Dict) -> Dict\n```\n\n---\n\n### Functions\n\n#### process\n\n```python\ndef process(self: Any, sample: Dict) -> Dict\n```\n\n---\n\n\n## data.public_api\n\nData Module - Public API
========================

数据模块公共API：提供统一的数据接口

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### DataAPI\n\n数据模块公共API\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化数据模块\n\n##### get_dataset_information\n```python\ndef get_dataset_information(self: Any, dataset_name: Any) -> Dict[Unknown]\n```\n\n获取数据集信息\n\n##### get_performance_data\n```python\ndef get_performance_data(self: Any, method_name: Any) -> Dict[Unknown]\n```\n\n获取性能数据\n\n##### get_module_status\n```python\ndef get_module_status(self: Any) -> Dict[Unknown]\n```\n\n获取模块状态\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭数据模块\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化数据模块\n\n---\n\n#### get_dataset_information\n\n```python\ndef get_dataset_information(self: Any, dataset_name: Any) -> Dict[Unknown]\n```\n\n获取数据集信息\n\n---\n\n#### get_performance_data\n\n```python\ndef get_performance_data(self: Any, method_name: Any) -> Dict[Unknown]\n```\n\n获取性能数据\n\n---\n\n#### get_module_status\n\n```python\ndef get_module_status(self: Any) -> Dict[Unknown]\n```\n\n获取模块状态\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭数据模块\n\n---\n\n\n## demo_modular_system\n\n模块化数学推理系统演示

展示新模块化架构的使用方式和功能。\n\n### Functions\n\n#### setup_logging\n\n```python\ndef setup_logging()\n```\n\n设置日志\n\n---\n\n#### register_modules\n\n```python\ndef register_modules()\n```\n\n注册系统模块\n\n---\n\n#### test_basic_reasoning\n\n```python\ndef test_basic_reasoning()\n```\n\n测试基础推理功能\n\n---\n\n#### test_batch_processing\n\n```python\ndef test_batch_processing()\n```\n\n测试批量处理功能\n\n---\n\n#### test_system_status\n\n```python\ndef test_system_status()\n```\n\n测试系统状态监控\n\n---\n\n#### generate_report\n\n```python\ndef generate_report(basic_results: List[Dict], batch_results: List[Dict], system_status: Dict) -> Any\n```\n\n生成测试报告\n\n---\n\n#### main\n\n```python\ndef main()\n```\n\n主函数\n\n---\n\n\n## evaluation.ablation_study\n\nAutomated Ablation Study Framework
==================================

Comprehensive ablation study implementation for COT-DIR system.\n\n### Classes\n\n#### AblationConfig\n\nConfiguration for ablation study components\n\n---\n\n#### AblationResult\n\nResults from a single ablation configuration\n\n---\n\n#### AutomatedAblationStudy\n\nAutomated ablation study framework\n\n**Methods:**\n\n##### run_complete_ablation_study\n```python\ndef run_complete_ablation_study(self: Any) -> Dict[Unknown]\n```\n\nRun complete ablation study with all component combinations\n\n---\n\n### Functions\n\n#### run_ablation_study_demo\n\n```python\ndef run_ablation_study_demo()\n```\n\nDemo function to run ablation study\n\n---\n\n#### run_complete_ablation_study\n\n```python\ndef run_complete_ablation_study(self: Any) -> Dict[Unknown]\n```\n\nRun complete ablation study with all component combinations\n\n---\n\n\n## evaluation.computational_analysis\n\nComputational Complexity and Performance Analysis
================================================

Tools for analyzing computational performance, memory usage, and scalability.\n\n### Classes\n\n#### PerformanceMetrics\n\nPerformance metrics for a single test\n\n---\n\n#### ComplexityAnalysis\n\nComputational complexity analysis results\n\n---\n\n#### ComputationalAnalyzer\n\nComputational performance and complexity analyzer\n\n**Methods:**\n\n##### analyze_system_performance\n```python\ndef analyze_system_performance(self: Any, solve_function: Callable, test_problems: List[Dict], warmup_runs: int) -> Dict[Unknown]\n```\n\nComprehensive system performance analysis\n\n---\n\n### Functions\n\n#### run_computational_analysis_demo\n\n```python\ndef run_computational_analysis_demo()\n```\n\nDemo function for computational analysis\n\n---\n\n#### analyze_system_performance\n\n```python\ndef analyze_system_performance(self: Any, solve_function: Callable, test_problems: List[Dict], warmup_runs: int) -> Dict[Unknown]\n```\n\nComprehensive system performance analysis\n\n---\n\n#### dummy_solve_function\n\n```python\ndef dummy_solve_function(problem_text: str) -> str\n```\n\nDummy solve function for testing\n\n---\n\n\n## evaluation.dir_focused_benchmark\n\nDIR-Focused Benchmark Suite
===========================

Targeted evaluation framework focusing on problems with deep implicit relations (DIR ≥ 0.25).
This implements the strategic problem selection methodology described in the paper.\n\n### Classes\n\n#### DIRProblem\n\nProblem with DIR score and complexity classification\n\n---\n\n#### DIRBenchmarkResult\n\nResults from DIR-focused evaluation\n\n---\n\n#### DIRProblemSelector\n\nSelects problems based on DIR score and complexity thresholds\n\n**Methods:**\n\n##### select_problems\n```python\ndef select_problems(self: Any, all_problems: List[DIRProblem]) -> List[DIRProblem]\n```\n\nSelect problems meeting DIR and complexity criteria

Args:
    all_problems: Complete list of classified problems
    
Returns:
    Filtered list of problems with DIR ≥ threshold and complexity ≥ min_level\n\n##### analyze_selection\n```python\ndef analyze_selection(self: Any, all_problems: List[DIRProblem], selected_problems: List[DIRProblem]) -> Dict[Unknown]\n```\n\nAnalyze the impact of problem selection\n\n---\n\n#### DIRFocusedBenchmarkSuite\n\nBenchmark suite focused on deep implicit relations problems\n\n**Methods:**\n\n##### load_and_classify_problems\n```python\ndef load_and_classify_problems(self: Any) -> List[DIRProblem]\n```\n\nLoad problems and apply classification\n\n##### evaluate_on_dir_subset\n```python\ndef evaluate_on_dir_subset(self: Any, method_func: Any, method_name: str) -> DIRBenchmarkResult\n```\n\nEvaluate method on DIR-filtered problem subset

Args:
    method_func: Function that takes problem dict and returns answer
    method_name: Name of the method being evaluated\n\n##### generate_dir_focused_report\n```python\ndef generate_dir_focused_report(self: Any, result: DIRBenchmarkResult, output_path: str)\n```\n\nGenerate comprehensive DIR-focused evaluation report\n\n---\n\n### Functions\n\n#### select_problems\n\n```python\ndef select_problems(self: Any, all_problems: List[DIRProblem]) -> List[DIRProblem]\n```\n\nSelect problems meeting DIR and complexity criteria

Args:
    all_problems: Complete list of classified problems
    
Returns:
    Filtered list of problems with DIR ≥ threshold and complexity ≥ min_level\n\n---\n\n#### analyze_selection\n\n```python\ndef analyze_selection(self: Any, all_problems: List[DIRProblem], selected_problems: List[DIRProblem]) -> Dict[Unknown]\n```\n\nAnalyze the impact of problem selection\n\n---\n\n#### load_and_classify_problems\n\n```python\ndef load_and_classify_problems(self: Any) -> List[DIRProblem]\n```\n\nLoad problems and apply classification\n\n---\n\n#### evaluate_on_dir_subset\n\n```python\ndef evaluate_on_dir_subset(self: Any, method_func: Any, method_name: str) -> DIRBenchmarkResult\n```\n\nEvaluate method on DIR-filtered problem subset

Args:
    method_func: Function that takes problem dict and returns answer
    method_name: Name of the method being evaluated\n\n---\n\n#### generate_dir_focused_report\n\n```python\ndef generate_dir_focused_report(self: Any, result: DIRBenchmarkResult, output_path: str)\n```\n\nGenerate comprehensive DIR-focused evaluation report\n\n---\n\n#### get_stats\n\n```python\ndef get_stats(problems: Any)\n```\n\n---\n\n\n## evaluation.evaluator\n\nComprehensive Evaluator
=======================

Main evaluation engine that coordinates multiple metrics and generates reports.\n\n### Classes\n\n#### EvaluationResult\n\nComplete evaluation result containing all metrics\n\n---\n\n#### ComprehensiveEvaluator\n\nMain evaluation engine for mathematical reasoning systems\n\n**Methods:**\n\n##### evaluate\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], dataset_name: str, model_name: str, metadata: Optional[Unknown]) -> EvaluationResult\n```\n\nPerform comprehensive evaluation using all metrics

Args:
    predictions: List of model predictions
    ground_truth: List of ground truth answers
    dataset_name: Name of the dataset being evaluated
    model_name: Name of the model being evaluated
    metadata: Additional metadata (reasoning steps, processing times, etc.)
    
Returns:
    EvaluationResult containing all metric scores\n\n##### evaluate_batch\n```python\ndef evaluate_batch(self: Any, batch_predictions: List[Unknown], batch_ground_truth: List[Unknown], dataset_names: List[str], model_name: str, batch_metadata: Optional[Unknown]) -> List[EvaluationResult]\n```\n\nEvaluate multiple datasets in batch

Args:
    batch_predictions: List of prediction lists for each dataset
    batch_ground_truth: List of ground truth lists for each dataset
    dataset_names: Names of the datasets
    model_name: Name of the model being evaluated
    batch_metadata: Optional metadata for each dataset
    
Returns:
    List of EvaluationResult objects\n\n##### compare_models\n```python\ndef compare_models(self: Any, model_results: Dict[Unknown]) -> Dict[Unknown]\n```\n\nCompare multiple model evaluation results

Args:
    model_results: Dictionary mapping model names to their evaluation results
    
Returns:
    Dictionary containing comparison analysis\n\n##### get_metric_weights\n```python\ndef get_metric_weights(self: Any) -> Dict[Unknown]\n```\n\nGet current metric weights\n\n##### set_metric_weights\n```python\ndef set_metric_weights(self: Any, weights: Dict[Unknown]) -> Any\n```\n\nSet new metric weights\n\n##### get_available_metrics\n```python\ndef get_available_metrics(self: Any) -> List[str]\n```\n\nGet list of available metric names\n\n##### add_custom_metric\n```python\ndef add_custom_metric(self: Any, name: str, metric: BaseMetric) -> Any\n```\n\nAdd a custom metric to the evaluator\n\n##### remove_metric\n```python\ndef remove_metric(self: Any, name: str) -> Any\n```\n\nRemove a metric from the evaluator\n\n---\n\n### Functions\n\n#### evaluate\n\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], dataset_name: str, model_name: str, metadata: Optional[Unknown]) -> EvaluationResult\n```\n\nPerform comprehensive evaluation using all metrics

Args:
    predictions: List of model predictions
    ground_truth: List of ground truth answers
    dataset_name: Name of the dataset being evaluated
    model_name: Name of the model being evaluated
    metadata: Additional metadata (reasoning steps, processing times, etc.)
    
Returns:
    EvaluationResult containing all metric scores\n\n---\n\n#### evaluate_batch\n\n```python\ndef evaluate_batch(self: Any, batch_predictions: List[Unknown], batch_ground_truth: List[Unknown], dataset_names: List[str], model_name: str, batch_metadata: Optional[Unknown]) -> List[EvaluationResult]\n```\n\nEvaluate multiple datasets in batch

Args:
    batch_predictions: List of prediction lists for each dataset
    batch_ground_truth: List of ground truth lists for each dataset
    dataset_names: Names of the datasets
    model_name: Name of the model being evaluated
    batch_metadata: Optional metadata for each dataset
    
Returns:
    List of EvaluationResult objects\n\n---\n\n#### compare_models\n\n```python\ndef compare_models(self: Any, model_results: Dict[Unknown]) -> Dict[Unknown]\n```\n\nCompare multiple model evaluation results

Args:
    model_results: Dictionary mapping model names to their evaluation results
    
Returns:
    Dictionary containing comparison analysis\n\n---\n\n#### get_metric_weights\n\n```python\ndef get_metric_weights(self: Any) -> Dict[Unknown]\n```\n\nGet current metric weights\n\n---\n\n#### set_metric_weights\n\n```python\ndef set_metric_weights(self: Any, weights: Dict[Unknown]) -> Any\n```\n\nSet new metric weights\n\n---\n\n#### get_available_metrics\n\n```python\ndef get_available_metrics(self: Any) -> List[str]\n```\n\nGet list of available metric names\n\n---\n\n#### add_custom_metric\n\n```python\ndef add_custom_metric(self: Any, name: str, metric: BaseMetric) -> Any\n```\n\nAdd a custom metric to the evaluator\n\n---\n\n#### remove_metric\n\n```python\ndef remove_metric(self: Any, name: str) -> Any\n```\n\nRemove a metric from the evaluator\n\n---\n\n\n## evaluation.failure_analysis\n\nFailure Case Analysis Framework
===============================

Systematic failure case analysis and error pattern recognition.\n\n### Classes\n\n#### FailureCase\n\nIndividual failure case data\n\n---\n\n#### ErrorPattern\n\nIdentified error pattern\n\n---\n\n#### FailureAnalyzer\n\nSystematic failure case analyzer\n\n**Methods:**\n\n##### analyze_failures\n```python\ndef analyze_failures(self: Any, test_results: List[Dict]) -> Dict[Unknown]\n```\n\nAnalyze failure cases and identify patterns\n\n##### export_failure_report\n```python\ndef export_failure_report(self: Any, results: Dict[Unknown], filename: str)\n```\n\nExport detailed failure analysis report\n\n---\n\n### Functions\n\n#### run_failure_analysis_demo\n\n```python\ndef run_failure_analysis_demo()\n```\n\nDemo function for failure analysis\n\n---\n\n#### analyze_failures\n\n```python\ndef analyze_failures(self: Any, test_results: List[Dict]) -> Dict[Unknown]\n```\n\nAnalyze failure cases and identify patterns\n\n---\n\n#### export_failure_report\n\n```python\ndef export_failure_report(self: Any, results: Dict[Unknown], filename: str)\n```\n\nExport detailed failure analysis report\n\n---\n\n\n## evaluation.metrics\n\nEvaluation Metrics
=================

Various metrics for evaluating mathematical reasoning systems.\n\n### Classes\n\n#### MetricResult\n\nResult of a metric evaluation\n\n---\n\n#### BaseMetric\n\nAbstract base class for evaluation metrics\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### evaluate\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate predictions against ground truth\n\n##### get_max_score\n```python\ndef get_max_score(self: Any) -> float\n```\n\nGet maximum possible score for this metric\n\n---\n\n#### AccuracyMetric\n\nAccuracy metric for exact match evaluation\n\n**Inherits from:** BaseMetric\n\n**Methods:**\n\n##### evaluate\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nCalculate accuracy score\n\n##### get_max_score\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n#### ReasoningQualityMetric\n\nMetric for evaluating quality of reasoning steps\n\n**Inherits from:** BaseMetric\n\n**Methods:**\n\n##### evaluate\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate reasoning quality based on step structure and coherence\n\n##### get_max_score\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n#### EfficiencyMetric\n\nMetric for evaluating computational efficiency\n\n**Inherits from:** BaseMetric\n\n**Methods:**\n\n##### evaluate\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate efficiency based on processing time\n\n##### get_max_score\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n#### RobustnessMetric\n\nMetric for evaluating system robustness to various inputs\n\n**Inherits from:** BaseMetric\n\n**Methods:**\n\n##### evaluate\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate robustness based on error handling and consistency\n\n##### get_max_score\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n#### ExplainabilityMetric\n\nMetric for evaluating explainability of reasoning\n\n**Inherits from:** BaseMetric\n\n**Methods:**\n\n##### evaluate\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate explainability based on explanation quality\n\n##### get_max_score\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n### Functions\n\n#### evaluate\n\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate predictions against ground truth\n\n---\n\n#### get_max_score\n\n```python\ndef get_max_score(self: Any) -> float\n```\n\nGet maximum possible score for this metric\n\n---\n\n#### evaluate\n\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nCalculate accuracy score\n\n---\n\n#### get_max_score\n\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n#### evaluate\n\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate reasoning quality based on step structure and coherence\n\n---\n\n#### get_max_score\n\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n#### evaluate\n\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate efficiency based on processing time\n\n---\n\n#### get_max_score\n\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n#### evaluate\n\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate robustness based on error handling and consistency\n\n---\n\n#### get_max_score\n\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n#### evaluate\n\n```python\ndef evaluate(self: Any, predictions: List[Any], ground_truth: List[Any], metadata: Optional[Unknown]) -> MetricResult\n```\n\nEvaluate explainability based on explanation quality\n\n---\n\n#### get_max_score\n\n```python\ndef get_max_score(self: Any) -> float\n```\n\n---\n\n\n## evaluation.orchestrator\n\nEvaluation Module - Orchestrator
================================

评估模块协调器

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### EvaluationOrchestrator\n\n评估模块协调器\n\n**Methods:**\n\n##### initialize_orchestrator\n```python\ndef initialize_orchestrator(self: Any) -> bool\n```\n\n初始化协调器\n\n##### orchestrate\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行\n\n##### register_component\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件\n\n##### get_component\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件\n\n##### get_operation_history\n```python\ndef get_operation_history(self: Any) -> List[Unknown]\n```\n\n获取操作历史\n\n##### clear_operation_history\n```python\ndef clear_operation_history(self: Any) -> Any\n```\n\n清空操作历史\n\n---\n\n### Functions\n\n#### initialize_orchestrator\n\n```python\ndef initialize_orchestrator(self: Any) -> bool\n```\n\n初始化协调器\n\n---\n\n#### orchestrate\n\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行\n\n---\n\n#### register_component\n\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件\n\n---\n\n#### get_component\n\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件\n\n---\n\n#### get_operation_history\n\n```python\ndef get_operation_history(self: Any) -> List[Unknown]\n```\n\n获取操作历史\n\n---\n\n#### clear_operation_history\n\n```python\ndef clear_operation_history(self: Any) -> Any\n```\n\n清空操作历史\n\n---\n\n\n## evaluation.public_api\n\nEvaluation Module - Public API
==============================

评估模块公共API：提供统一的评估接口

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### EvaluationAPI\n\n评估模块公共API\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化评估模块\n\n##### evaluate_performance\n```python\ndef evaluate_performance(self: Any, results: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n评估性能\n\n##### get_module_status\n```python\ndef get_module_status(self: Any) -> Dict[Unknown]\n```\n\n获取模块状态\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭评估模块\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化评估模块\n\n---\n\n#### evaluate_performance\n\n```python\ndef evaluate_performance(self: Any, results: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n评估性能\n\n---\n\n#### get_module_status\n\n```python\ndef get_module_status(self: Any) -> Dict[Unknown]\n```\n\n获取模块状态\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭评估模块\n\n---\n\n\n## evaluation.sota_benchmark\n\nSOTA Benchmark Suite
===================

Implementation of the multi-dataset evaluation framework described in the paper.
Supports comparison with state-of-the-art methods on 11 mathematical reasoning datasets.\n\n### Classes\n\n#### DatasetInfo\n\nDataset information matching paper specifications\n\n---\n\n#### BenchmarkResult\n\nBenchmark result structure\n\n---\n\n#### SOTABenchmarkSuite\n\nState-of-the-art benchmark suite implementing the paper's evaluation framework\n\n**Methods:**\n\n##### load_dataset\n```python\ndef load_dataset(self: Any, dataset_name: str) -> List[Dict]\n```\n\nLoad dataset problems from data directory\n\n##### classify_complexity\n```python\ndef classify_complexity(self: Any, problem: Dict) -> str\n```\n\nClassify problem complexity level (L0-L3)\n\n##### evaluate_method\n```python\ndef evaluate_method(self: Any, method_func: Any, method_name: str, test_subset: Optional[int]) -> BenchmarkResult\n```\n\nEvaluate a method against the benchmark suite

Args:
    method_func: Function that takes a problem dict and returns (answer, reasoning_steps, relations)
    method_name: Name of the method being evaluated
    test_subset: If provided, only test on this many problems per dataset\n\n##### compare_with_sota\n```python\ndef compare_with_sota(self: Any, result: BenchmarkResult) -> Dict[Unknown]\n```\n\nCompare result with SOTA baselines\n\n##### generate_benchmark_report\n```python\ndef generate_benchmark_report(self: Any, result: BenchmarkResult, output_path: str)\n```\n\nGenerate comprehensive benchmark report\n\n##### run_ablation_study\n```python\ndef run_ablation_study(self: Any, base_method: Any, components: List[Unknown], test_subset: int)\n```\n\nRun ablation study to measure component contributions\n\n---\n\n### Functions\n\n#### load_dataset\n\n```python\ndef load_dataset(self: Any, dataset_name: str) -> List[Dict]\n```\n\nLoad dataset problems from data directory\n\n---\n\n#### classify_complexity\n\n```python\ndef classify_complexity(self: Any, problem: Dict) -> str\n```\n\nClassify problem complexity level (L0-L3)\n\n---\n\n#### evaluate_method\n\n```python\ndef evaluate_method(self: Any, method_func: Any, method_name: str, test_subset: Optional[int]) -> BenchmarkResult\n```\n\nEvaluate a method against the benchmark suite

Args:
    method_func: Function that takes a problem dict and returns (answer, reasoning_steps, relations)
    method_name: Name of the method being evaluated
    test_subset: If provided, only test on this many problems per dataset\n\n---\n\n#### compare_with_sota\n\n```python\ndef compare_with_sota(self: Any, result: BenchmarkResult) -> Dict[Unknown]\n```\n\nCompare result with SOTA baselines\n\n---\n\n#### generate_benchmark_report\n\n```python\ndef generate_benchmark_report(self: Any, result: BenchmarkResult, output_path: str)\n```\n\nGenerate comprehensive benchmark report\n\n---\n\n#### run_ablation_study\n\n```python\ndef run_ablation_study(self: Any, base_method: Any, components: List[Unknown], test_subset: int)\n```\n\nRun ablation study to measure component contributions\n\n---\n\n\n## gnn_enhancement.core.concept_gnn.math_concept_gnn\n\nMath Concept Graph Neural Network
=================================

数学概念图神经网络实现

用于构建数学概念之间的关系图，学习概念间的隐式关系，
增强IRD（隐式关系发现）模块的能力。

主要功能:
1. 构建数学概念图
2. 学习概念间的隐式关系
3. 提供概念相似度计算
4. 支持动态概念图更新

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### MathConceptGNN\n\n数学概念图神经网络

核心功能：
- 构建数学概念之间的关系图
- 学习概念间的隐式关系
- 增强关系发现能力
- 提供概念嵌入和相似度计算\n\n**Methods:**\n\n##### build_concept_graph\n```python\ndef build_concept_graph(self: Any, problem_text: str, entities: List[str]) -> Dict[Unknown]\n```\n\n从问题文本构建概念图

Args:
    problem_text: 问题文本
    entities: 识别出的实体列表
    
Returns:
    概念图信息\n\n##### enhance_implicit_relations\n```python\ndef enhance_implicit_relations(self: Any, problem_text: str, existing_relations: List[Unknown]) -> List[Unknown]\n```\n\n增强隐式关系发现

Args:
    problem_text: 问题文本
    existing_relations: 现有关系列表
    
Returns:
    增强后的关系列表\n\n##### get_concept_similarity\n```python\ndef get_concept_similarity(self: Any, concept1: str, concept2: str) -> float\n```\n\n计算概念相似度\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n#### ConceptGNNModel\n\n**Inherits from:** nn.Module\n\n**Methods:**\n\n##### forward\n```python\ndef forward(self: Any, graph: Any, features: Any)\n```\n\n---\n\n### Functions\n\n#### build_concept_graph\n\n```python\ndef build_concept_graph(self: Any, problem_text: str, entities: List[str]) -> Dict[Unknown]\n```\n\n从问题文本构建概念图

Args:
    problem_text: 问题文本
    entities: 识别出的实体列表
    
Returns:
    概念图信息\n\n---\n\n#### enhance_implicit_relations\n\n```python\ndef enhance_implicit_relations(self: Any, problem_text: str, existing_relations: List[Unknown]) -> List[Unknown]\n```\n\n增强隐式关系发现

Args:
    problem_text: 问题文本
    existing_relations: 现有关系列表
    
Returns:
    增强后的关系列表\n\n---\n\n#### get_concept_similarity\n\n```python\ndef get_concept_similarity(self: Any, concept1: str, concept2: str) -> float\n```\n\n计算概念相似度\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n#### forward\n\n```python\ndef forward(self: Any, graph: Any, features: Any)\n```\n\n---\n\n\n## gnn_enhancement.core.reasoning_gnn.reasoning_gnn\n\nReasoning Graph Neural Network
==============================

推理过程图神经网络实现

用于优化多层级推理（MLR）过程，构建推理步骤之间的依赖图，
学习最优推理路径，提高推理效率和准确性。

主要功能:
1. 构建推理步骤依赖图
2. 学习推理路径优化
3. 提供推理步骤排序
4. 支持动态推理调整

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### ReasoningGNN\n\n推理过程图神经网络

核心功能：
- 构建推理步骤之间的依赖图
- 学习最优推理路径
- 提供推理步骤排序和优化
- 支持动态推理调整\n\n**Methods:**\n\n##### build_reasoning_graph\n```python\ndef build_reasoning_graph(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建推理步骤依赖图

Args:
    reasoning_steps: 推理步骤列表
    problem_context: 问题上下文
    
Returns:
    推理图信息\n\n##### optimize_reasoning_path\n```python\ndef optimize_reasoning_path(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> List[Unknown]\n```\n\n优化推理路径

Args:
    reasoning_steps: 原始推理步骤
    problem_context: 问题上下文
    
Returns:
    优化后的推理步骤\n\n##### get_reasoning_quality_score\n```python\ndef get_reasoning_quality_score(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> float\n```\n\n计算推理质量分数

Args:
    reasoning_steps: 推理步骤
    problem_context: 问题上下文
    
Returns:
    推理质量分数 (0-1)\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n#### ReasoningGNNModel\n\n**Inherits from:** nn.Module\n\n**Methods:**\n\n##### forward\n```python\ndef forward(self: Any, graph: Any, features: Any)\n```\n\n---\n\n### Functions\n\n#### build_reasoning_graph\n\n```python\ndef build_reasoning_graph(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建推理步骤依赖图

Args:
    reasoning_steps: 推理步骤列表
    problem_context: 问题上下文
    
Returns:
    推理图信息\n\n---\n\n#### optimize_reasoning_path\n\n```python\ndef optimize_reasoning_path(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> List[Unknown]\n```\n\n优化推理路径

Args:
    reasoning_steps: 原始推理步骤
    problem_context: 问题上下文
    
Returns:
    优化后的推理步骤\n\n---\n\n#### get_reasoning_quality_score\n\n```python\ndef get_reasoning_quality_score(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> float\n```\n\n计算推理质量分数

Args:
    reasoning_steps: 推理步骤
    problem_context: 问题上下文
    
Returns:
    推理质量分数 (0-1)\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n#### forward\n\n```python\ndef forward(self: Any, graph: Any, features: Any)\n```\n\n---\n\n\n## gnn_enhancement.core.verification_gnn.verification_gnn\n\nVerification Graph Neural Network
=================================

验证图神经网络实现

用于增强链式验证（CV）准确性，构建验证步骤之间的关系图，
学习验证模式，提高验证的准确性和可靠性。

主要功能:
1. 构建验证步骤关系图
2. 学习验证模式
3. 提供验证结果评估
4. 支持多层验证策略

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### VerificationGNN\n\n验证图神经网络

核心功能：
- 构建验证步骤之间的关系图
- 学习验证模式和规律
- 提供验证结果评估和置信度计算
- 支持多层验证策略\n\n**Methods:**\n\n##### build_verification_graph\n```python\ndef build_verification_graph(self: Any, reasoning_steps: List[Unknown], verification_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建验证图

Args:
    reasoning_steps: 推理步骤列表
    verification_context: 验证上下文
    
Returns:
    验证图信息\n\n##### perform_verification\n```python\ndef perform_verification(self: Any, reasoning_steps: List[Unknown], verification_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n执行验证

Args:
    reasoning_steps: 推理步骤列表
    verification_context: 验证上下文
    
Returns:
    验证结果\n\n##### enhance_verification_accuracy\n```python\ndef enhance_verification_accuracy(self: Any, reasoning_steps: List[Unknown], existing_verification: Dict[Unknown]) -> Dict[Unknown]\n```\n\n增强验证准确性

Args:
    reasoning_steps: 推理步骤
    existing_verification: 现有验证结果
    
Returns:
    增强后的验证结果\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n#### VerificationGNNModel\n\n**Inherits from:** nn.Module\n\n**Methods:**\n\n##### forward\n```python\ndef forward(self: Any, graph: Any, features: Any)\n```\n\n---\n\n### Functions\n\n#### build_verification_graph\n\n```python\ndef build_verification_graph(self: Any, reasoning_steps: List[Unknown], verification_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建验证图

Args:
    reasoning_steps: 推理步骤列表
    verification_context: 验证上下文
    
Returns:
    验证图信息\n\n---\n\n#### perform_verification\n\n```python\ndef perform_verification(self: Any, reasoning_steps: List[Unknown], verification_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n执行验证

Args:
    reasoning_steps: 推理步骤列表
    verification_context: 验证上下文
    
Returns:
    验证结果\n\n---\n\n#### enhance_verification_accuracy\n\n```python\ndef enhance_verification_accuracy(self: Any, reasoning_steps: List[Unknown], existing_verification: Dict[Unknown]) -> Dict[Unknown]\n```\n\n增强验证准确性

Args:
    reasoning_steps: 推理步骤
    existing_verification: 现有验证结果
    
Returns:
    增强后的验证结果\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n#### forward\n\n```python\ndef forward(self: Any, graph: Any, features: Any)\n```\n\n---\n\n\n## gnn_enhancement.graph_builders.concept_graph_builder\n\nConcept Graph Builder
====================

概念图构建器，封装MathConceptGNN的功能\n\n### Classes\n\n#### ConceptGraphBuilder\n\n概念图构建器\n\n**Methods:**\n\n##### build_concept_graph\n```python\ndef build_concept_graph(self: Any, problem_text: str, context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建概念图

Args:
    problem_text: 问题文本
    context: 上下文信息
    
Returns:
    概念图信息\n\n---\n\n### Functions\n\n#### build_concept_graph\n\n```python\ndef build_concept_graph(self: Any, problem_text: str, context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建概念图

Args:
    problem_text: 问题文本
    context: 上下文信息
    
Returns:
    概念图信息\n\n---\n\n\n## gnn_enhancement.graph_builders.graph_builder\n\nGraph Builder
=============

主要图构建器类，协调不同类型的图构建操作

用于从数学问题文本构建概念图、推理图和验证图\n\n### Classes\n\n#### GraphBuilder\n\n主要图构建器

协调不同类型的图构建操作：
- 概念图构建
- 推理图构建
- 验证图构建\n\n**Methods:**\n\n##### build_all_graphs\n```python\ndef build_all_graphs(self: Any, problem_text: str, reasoning_steps: Optional[Unknown], context: Optional[Unknown]) -> Dict[Unknown]\n```\n\n构建所有类型的图

Args:
    problem_text: 问题文本
    reasoning_steps: 推理步骤（可选）
    context: 上下文信息（可选）
    
Returns:
    包含所有图的字典\n\n##### build_concept_graph\n```python\ndef build_concept_graph(self: Any, problem_text: str, context: Optional[Unknown]) -> Dict[Unknown]\n```\n\n构建概念图\n\n##### build_reasoning_graph\n```python\ndef build_reasoning_graph(self: Any, reasoning_steps: List[Unknown], context: Optional[Unknown]) -> Dict[Unknown]\n```\n\n构建推理图\n\n##### build_verification_graph\n```python\ndef build_verification_graph(self: Any, reasoning_steps: List[Unknown], context: Optional[Unknown]) -> Dict[Unknown]\n```\n\n构建验证图\n\n##### get_graph_statistics\n```python\ndef get_graph_statistics(self: Any, graphs: Dict[Unknown]) -> Dict[Unknown]\n```\n\n获取图统计信息\n\n##### validate_graphs\n```python\ndef validate_graphs(self: Any, graphs: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证图结构\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n### Functions\n\n#### build_all_graphs\n\n```python\ndef build_all_graphs(self: Any, problem_text: str, reasoning_steps: Optional[Unknown], context: Optional[Unknown]) -> Dict[Unknown]\n```\n\n构建所有类型的图

Args:
    problem_text: 问题文本
    reasoning_steps: 推理步骤（可选）
    context: 上下文信息（可选）
    
Returns:
    包含所有图的字典\n\n---\n\n#### build_concept_graph\n\n```python\ndef build_concept_graph(self: Any, problem_text: str, context: Optional[Unknown]) -> Dict[Unknown]\n```\n\n构建概念图\n\n---\n\n#### build_reasoning_graph\n\n```python\ndef build_reasoning_graph(self: Any, reasoning_steps: List[Unknown], context: Optional[Unknown]) -> Dict[Unknown]\n```\n\n构建推理图\n\n---\n\n#### build_verification_graph\n\n```python\ndef build_verification_graph(self: Any, reasoning_steps: List[Unknown], context: Optional[Unknown]) -> Dict[Unknown]\n```\n\n构建验证图\n\n---\n\n#### get_graph_statistics\n\n```python\ndef get_graph_statistics(self: Any, graphs: Dict[Unknown]) -> Dict[Unknown]\n```\n\n获取图统计信息\n\n---\n\n#### validate_graphs\n\n```python\ndef validate_graphs(self: Any, graphs: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证图结构\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n\n## gnn_enhancement.graph_builders.reasoning_graph_builder\n\nReasoning Graph Builder
======================

推理图构建器，封装ReasoningGNN的功能\n\n### Classes\n\n#### ReasoningGraphBuilder\n\n推理图构建器\n\n**Methods:**\n\n##### build_reasoning_graph\n```python\ndef build_reasoning_graph(self: Any, reasoning_steps: List[Unknown], context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建推理图

Args:
    reasoning_steps: 推理步骤列表
    context: 上下文信息
    
Returns:
    推理图信息\n\n---\n\n### Functions\n\n#### build_reasoning_graph\n\n```python\ndef build_reasoning_graph(self: Any, reasoning_steps: List[Unknown], context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建推理图

Args:
    reasoning_steps: 推理步骤列表
    context: 上下文信息
    
Returns:
    推理图信息\n\n---\n\n\n## gnn_enhancement.graph_builders.verification_graph_builder\n\nVerification Graph Builder
=========================

验证图构建器，封装VerificationGNN的功能\n\n### Classes\n\n#### VerificationGraphBuilder\n\n验证图构建器\n\n**Methods:**\n\n##### build_verification_graph\n```python\ndef build_verification_graph(self: Any, reasoning_steps: List[Unknown], context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建验证图

Args:
    reasoning_steps: 推理步骤列表
    context: 上下文信息
    
Returns:
    验证图信息\n\n---\n\n### Functions\n\n#### build_verification_graph\n\n```python\ndef build_verification_graph(self: Any, reasoning_steps: List[Unknown], context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n构建验证图

Args:
    reasoning_steps: 推理步骤列表
    context: 上下文信息
    
Returns:
    验证图信息\n\n---\n\n\n## gnn_enhancement.integration.gnn_integrator\n\nGNN Integrator
==============

GNN集成器，将GNN功能集成到现有的COT-DIR1模块中

主要功能：
1. 与IRD模块集成，增强隐式关系发现
2. 与MLR模块集成，优化多层级推理
3. 与CV模块集成，增强链式验证\n\n### Classes\n\n#### GNNIntegrator\n\nGNN集成器

负责将GNN功能集成到现有的COT-DIR1模块中\n\n**Methods:**\n\n##### enhance_ird_module\n```python\ndef enhance_ird_module(self: Any, problem_text: str, existing_relations: List[Unknown]) -> Dict[Unknown]\n```\n\n增强IRD（隐式关系发现）模块

Args:
    problem_text: 问题文本
    existing_relations: 现有关系列表
    
Returns:
    增强后的关系发现结果\n\n##### enhance_mlr_module\n```python\ndef enhance_mlr_module(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n增强MLR（多层级推理）模块

Args:
    reasoning_steps: 推理步骤列表
    problem_context: 问题上下文
    
Returns:
    增强后的推理结果\n\n##### enhance_cv_module\n```python\ndef enhance_cv_module(self: Any, reasoning_steps: List[Unknown], existing_verification: Dict[Unknown]) -> Dict[Unknown]\n```\n\n增强CV（链式验证）模块

Args:
    reasoning_steps: 推理步骤列表
    existing_verification: 现有验证结果
    
Returns:
    增强后的验证结果\n\n##### integrate_with_processors\n```python\ndef integrate_with_processors(self: Any, problem_text: str, processing_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n与processors模块集成

Args:
    problem_text: 问题文本
    processing_result: 处理结果
    
Returns:
    GNN增强后的处理结果\n\n##### integrate_with_reasoning\n```python\ndef integrate_with_reasoning(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n与reasoning模块集成

Args:
    reasoning_steps: 推理步骤
    problem_context: 问题上下文
    
Returns:
    GNN增强后的推理结果\n\n##### integrate_with_evaluation\n```python\ndef integrate_with_evaluation(self: Any, reasoning_steps: List[Unknown], evaluation_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n与evaluation模块集成

Args:
    reasoning_steps: 推理步骤
    evaluation_result: 评估结果
    
Returns:
    GNN增强后的评估结果\n\n##### comprehensive_integration\n```python\ndef comprehensive_integration(self: Any, problem_text: str, reasoning_steps: List[Unknown], processing_result: Dict[Unknown], evaluation_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n综合集成所有模块

Args:
    problem_text: 问题文本
    reasoning_steps: 推理步骤
    processing_result: 处理结果
    evaluation_result: 评估结果
    
Returns:
    综合GNN增强结果\n\n##### get_integration_status\n```python\ndef get_integration_status(self: Any) -> Dict[Unknown]\n```\n\n获取集成状态\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n### Functions\n\n#### enhance_ird_module\n\n```python\ndef enhance_ird_module(self: Any, problem_text: str, existing_relations: List[Unknown]) -> Dict[Unknown]\n```\n\n增强IRD（隐式关系发现）模块

Args:
    problem_text: 问题文本
    existing_relations: 现有关系列表
    
Returns:
    增强后的关系发现结果\n\n---\n\n#### enhance_mlr_module\n\n```python\ndef enhance_mlr_module(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n增强MLR（多层级推理）模块

Args:
    reasoning_steps: 推理步骤列表
    problem_context: 问题上下文
    
Returns:
    增强后的推理结果\n\n---\n\n#### enhance_cv_module\n\n```python\ndef enhance_cv_module(self: Any, reasoning_steps: List[Unknown], existing_verification: Dict[Unknown]) -> Dict[Unknown]\n```\n\n增强CV（链式验证）模块

Args:
    reasoning_steps: 推理步骤列表
    existing_verification: 现有验证结果
    
Returns:
    增强后的验证结果\n\n---\n\n#### integrate_with_processors\n\n```python\ndef integrate_with_processors(self: Any, problem_text: str, processing_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n与processors模块集成

Args:
    problem_text: 问题文本
    processing_result: 处理结果
    
Returns:
    GNN增强后的处理结果\n\n---\n\n#### integrate_with_reasoning\n\n```python\ndef integrate_with_reasoning(self: Any, reasoning_steps: List[Unknown], problem_context: Dict[Unknown]) -> Dict[Unknown]\n```\n\n与reasoning模块集成

Args:
    reasoning_steps: 推理步骤
    problem_context: 问题上下文
    
Returns:
    GNN增强后的推理结果\n\n---\n\n#### integrate_with_evaluation\n\n```python\ndef integrate_with_evaluation(self: Any, reasoning_steps: List[Unknown], evaluation_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n与evaluation模块集成

Args:
    reasoning_steps: 推理步骤
    evaluation_result: 评估结果
    
Returns:
    GNN增强后的评估结果\n\n---\n\n#### comprehensive_integration\n\n```python\ndef comprehensive_integration(self: Any, problem_text: str, reasoning_steps: List[Unknown], processing_result: Dict[Unknown], evaluation_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n综合集成所有模块

Args:
    problem_text: 问题文本
    reasoning_steps: 推理步骤
    processing_result: 处理结果
    evaluation_result: 评估结果
    
Returns:
    综合GNN增强结果\n\n---\n\n#### get_integration_status\n\n```python\ndef get_integration_status(self: Any) -> Dict[Unknown]\n```\n\n获取集成状态\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> Dict[Unknown]\n```\n\n获取模块信息\n\n---\n\n\n## gnn_enhancement.utils.gnn_utils\n\nGNN Utils
=========

GNN工具类，提供通用的工具函数和辅助方法\n\n### Classes\n\n#### GNNUtils\n\nGNN工具类\n\n**Methods:**\n\n##### normalize_embeddings\n```python\ndef normalize_embeddings(embeddings: np.ndarray) -> np.ndarray\n```\n\n归一化嵌入向量\n\n##### calculate_cosine_similarity\n```python\ndef calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float\n```\n\n计算余弦相似度\n\n##### build_adjacency_matrix\n```python\ndef build_adjacency_matrix(nodes: List[Unknown], edges: List[Unknown]) -> np.ndarray\n```\n\n构建邻接矩阵\n\n##### extract_graph_features\n```python\ndef extract_graph_features(graph_info: Dict[Unknown]) -> Dict[Unknown]\n```\n\n提取图特征\n\n##### validate_graph_structure\n```python\ndef validate_graph_structure(graph_info: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证图结构\n\n##### merge_graphs\n```python\ndef merge_graphs(graph1: Dict[Unknown], graph2: Dict[Unknown]) -> Dict[Unknown]\n```\n\n合并两个图\n\n##### calculate_graph_metrics\n```python\ndef calculate_graph_metrics(graph_info: Dict[Unknown]) -> Dict[Unknown]\n```\n\n计算图度量指标\n\n##### format_graph_for_visualization\n```python\ndef format_graph_for_visualization(graph_info: Dict[Unknown]) -> Dict[Unknown]\n```\n\n格式化图数据用于可视化\n\n##### get_utils_info\n```python\ndef get_utils_info() -> Dict[Unknown]\n```\n\n获取工具信息\n\n---\n\n### Functions\n\n#### normalize_embeddings\n\n```python\ndef normalize_embeddings(embeddings: np.ndarray) -> np.ndarray\n```\n\n归一化嵌入向量\n\n---\n\n#### calculate_cosine_similarity\n\n```python\ndef calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float\n```\n\n计算余弦相似度\n\n---\n\n#### build_adjacency_matrix\n\n```python\ndef build_adjacency_matrix(nodes: List[Unknown], edges: List[Unknown]) -> np.ndarray\n```\n\n构建邻接矩阵\n\n---\n\n#### extract_graph_features\n\n```python\ndef extract_graph_features(graph_info: Dict[Unknown]) -> Dict[Unknown]\n```\n\n提取图特征\n\n---\n\n#### validate_graph_structure\n\n```python\ndef validate_graph_structure(graph_info: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证图结构\n\n---\n\n#### merge_graphs\n\n```python\ndef merge_graphs(graph1: Dict[Unknown], graph2: Dict[Unknown]) -> Dict[Unknown]\n```\n\n合并两个图\n\n---\n\n#### calculate_graph_metrics\n\n```python\ndef calculate_graph_metrics(graph_info: Dict[Unknown]) -> Dict[Unknown]\n```\n\n计算图度量指标\n\n---\n\n#### format_graph_for_visualization\n\n```python\ndef format_graph_for_visualization(graph_info: Dict[Unknown]) -> Dict[Unknown]\n```\n\n格式化图数据用于可视化\n\n---\n\n#### get_utils_info\n\n```python\ndef get_utils_info() -> Dict[Unknown]\n```\n\n获取工具信息\n\n---\n\n#### dfs\n\n```python\ndef dfs(node: Any)\n```\n\n---\n\n\n## models.async_api\n\n模型管理异步版公共API

在原有功能基础上添加异步支持，提高模型调用并发性能。\n\n### Classes\n\n#### AsyncModelAPI\n\n模型管理异步版公共API\n\n**Inherits from:** PublicAPI\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any, config: Optional[Unknown]) -> bool\n```\n\n同步初始化接口（保持兼容性）\n\n##### solve_with_model\n```python\ndef solve_with_model(self: Any, model_name: str, problem: Dict[Unknown], model_config: Optional[Unknown], use_cache: bool) -> Dict[Unknown]\n```\n\n同步解决问题接口（保持兼容性）\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, model_name: str, problems: List[Unknown], model_config: Optional[Unknown], use_cache: bool, max_workers: int) -> List[Unknown]\n```\n\n同步批量求解接口（保持兼容性）\n\n##### compare_models\n```python\ndef compare_models(self: Any, model_names: List[str], problems: List[Unknown], model_configs: Optional[Unknown]) -> Dict[Unknown]\n```\n\n同步模型比较接口（保持兼容性）\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### get_statistics\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n同步获取统计信息接口（保持兼容性）\n\n##### get_available_models\n```python\ndef get_available_models(self: Any) -> Dict[Unknown]\n```\n\n获取可用模型列表\n\n##### get_model_info\n```python\ndef get_model_info(self: Any, model_name: str) -> Dict[Unknown]\n```\n\n获取模型详细信息\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> Any\n```\n\n同步关闭接口（保持兼容性）\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any, config: Optional[Unknown]) -> bool\n```\n\n同步初始化接口（保持兼容性）\n\n---\n\n#### solve_with_model\n\n```python\ndef solve_with_model(self: Any, model_name: str, problem: Dict[Unknown], model_config: Optional[Unknown], use_cache: bool) -> Dict[Unknown]\n```\n\n同步解决问题接口（保持兼容性）\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, model_name: str, problems: List[Unknown], model_config: Optional[Unknown], use_cache: bool, max_workers: int) -> List[Unknown]\n```\n\n同步批量求解接口（保持兼容性）\n\n---\n\n#### compare_models\n\n```python\ndef compare_models(self: Any, model_names: List[str], problems: List[Unknown], model_configs: Optional[Unknown]) -> Dict[Unknown]\n```\n\n同步模型比较接口（保持兼容性）\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### get_statistics\n\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n同步获取统计信息接口（保持兼容性）\n\n---\n\n#### get_available_models\n\n```python\ndef get_available_models(self: Any) -> Dict[Unknown]\n```\n\n获取可用模型列表\n\n---\n\n#### get_model_info\n\n```python\ndef get_model_info(self: Any, model_name: str) -> Dict[Unknown]\n```\n\n获取模型详细信息\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> Any\n```\n\n同步关闭接口（保持兼容性）\n\n---\n\n#### solve_in_executor\n\n```python\ndef solve_in_executor()\n```\n\n---\n\n\n## models.base_model\n\nBase Model Interface

This module defines the base interface for all mathematical reasoning models.
All models (baseline, LLM, and proposed) should inherit from these base classes.\n\n### Classes\n\n#### ModelInput\n\nInput data structure for models.\n\n---\n\n#### ModelOutput\n\nOutput data structure for models.\n\n---\n\n#### ModelMetrics\n\nPerformance metrics for model evaluation.\n\n---\n\n#### BaseModel\n\nAbstract base class for all mathematical reasoning models.\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the model. Must be implemented by subclasses.

Returns:
    bool: True if initialization successful, False otherwise\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve a mathematical reasoning problem.

Args:
    problem_input: Input problem data
    
Returns:
    ModelOutput: Solution with reasoning chain and metadata\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems in batch.

Args:
    problems: List of input problems
    
Returns:
    List[ModelOutput]: List of solutions\n\n##### get_model_info\n```python\ndef get_model_info(self: Any) -> Dict[Unknown]\n```\n\nGet model information and configuration.\n\n##### validate_input\n```python\ndef validate_input(self: Any, problem_input: ModelInput) -> bool\n```\n\nValidate input problem format.\n\n##### update_metrics\n```python\ndef update_metrics(self: Any, new_metrics: ModelMetrics)\n```\n\nUpdate model performance metrics.\n\n---\n\n#### BaselineModel\n\nBase class for baseline mathematical reasoning models.\n\n**Inherits from:** BaseModel\n\n**Methods:**\n\n##### extract_equations\n```python\ndef extract_equations(self: Any, problem_text: str) -> List[str]\n```\n\nExtract mathematical equations from problem text.\n\n##### solve_equations\n```python\ndef solve_equations(self: Any, equations: List[str]) -> Dict[Unknown]\n```\n\nSolve extracted equations.\n\n---\n\n#### LLMModel\n\nBase class for Large Language Model implementations.\n\n**Inherits from:** BaseModel\n\n**Methods:**\n\n##### generate_prompt\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for the LLM.\n\n##### call_api\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall the LLM API with the given prompt.\n\n##### parse_response\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse LLM response into structured output.\n\n##### get_api_info\n```python\ndef get_api_info(self: Any) -> Dict[Unknown]\n```\n\nGet API configuration information.\n\n---\n\n#### ProposedModel\n\nBase class for the proposed COT-DIR model.\n\n**Inherits from:** BaseModel\n\n**Methods:**\n\n##### implicit_relation_discovery\n```python\ndef implicit_relation_discovery(self: Any, problem_input: ModelInput) -> Dict[Unknown]\n```\n\nDiscover implicit relations in the problem.\n\n##### multi_level_reasoning\n```python\ndef multi_level_reasoning(self: Any, relations: Dict[Unknown]) -> List[Unknown]\n```\n\nPerform multi-level reasoning on discovered relations.\n\n##### chain_verification\n```python\ndef chain_verification(self: Any, reasoning_chain: List[Unknown]) -> Dict[Unknown]\n```\n\nVerify the consistency and correctness of reasoning chain.\n\n##### get_component_status\n```python\ndef get_component_status(self: Any) -> Dict[Unknown]\n```\n\nGet status of all model components.\n\n---\n\n#### ModelFactory\n\nFactory class for creating different types of models.\n\n**Methods:**\n\n##### register_model\n```python\ndef register_model(cls: Any, model_class: type, model_type: str)\n```\n\nRegister a model class with the factory.\n\n##### create_model\n```python\ndef create_model(cls: Any, model_type: str, model_name: str, config: Optional[Unknown]) -> BaseModel\n```\n\nCreate a model instance.

Args:
    model_type: Type of model (baseline, llm, proposed)
    model_name: Name of the specific model
    config: Configuration parameters
    
Returns:
    BaseModel: Instance of the requested model\n\n##### list_available_models\n```python\ndef list_available_models(cls: Any) -> List[str]\n```\n\nList all registered model types.\n\n---\n\n#### ModelEvaluator\n\nUtility class for evaluating model performance.\n\n**Methods:**\n\n##### evaluate_model\n```python\ndef evaluate_model(self: Any, model: BaseModel, test_problems: List[ModelInput]) -> ModelMetrics\n```\n\nEvaluate a model on test problems.

Args:
    model: Model to evaluate
    test_problems: List of test problems with expected answers
    
Returns:
    ModelMetrics: Performance metrics\n\n##### compare_models\n```python\ndef compare_models(self: Any, models: List[BaseModel], test_problems: List[ModelInput]) -> Dict[Unknown]\n```\n\nCompare multiple models on the same test set.

Args:
    models: List of models to compare
    test_problems: Test problems for evaluation
    
Returns:
    Dict mapping model names to their metrics\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the model. Must be implemented by subclasses.

Returns:
    bool: True if initialization successful, False otherwise\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve a mathematical reasoning problem.

Args:
    problem_input: Input problem data
    
Returns:
    ModelOutput: Solution with reasoning chain and metadata\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems in batch.

Args:
    problems: List of input problems
    
Returns:
    List[ModelOutput]: List of solutions\n\n---\n\n#### get_model_info\n\n```python\ndef get_model_info(self: Any) -> Dict[Unknown]\n```\n\nGet model information and configuration.\n\n---\n\n#### validate_input\n\n```python\ndef validate_input(self: Any, problem_input: ModelInput) -> bool\n```\n\nValidate input problem format.\n\n---\n\n#### update_metrics\n\n```python\ndef update_metrics(self: Any, new_metrics: ModelMetrics)\n```\n\nUpdate model performance metrics.\n\n---\n\n#### extract_equations\n\n```python\ndef extract_equations(self: Any, problem_text: str) -> List[str]\n```\n\nExtract mathematical equations from problem text.\n\n---\n\n#### solve_equations\n\n```python\ndef solve_equations(self: Any, equations: List[str]) -> Dict[Unknown]\n```\n\nSolve extracted equations.\n\n---\n\n#### generate_prompt\n\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for the LLM.\n\n---\n\n#### call_api\n\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall the LLM API with the given prompt.\n\n---\n\n#### parse_response\n\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse LLM response into structured output.\n\n---\n\n#### get_api_info\n\n```python\ndef get_api_info(self: Any) -> Dict[Unknown]\n```\n\nGet API configuration information.\n\n---\n\n#### implicit_relation_discovery\n\n```python\ndef implicit_relation_discovery(self: Any, problem_input: ModelInput) -> Dict[Unknown]\n```\n\nDiscover implicit relations in the problem.\n\n---\n\n#### multi_level_reasoning\n\n```python\ndef multi_level_reasoning(self: Any, relations: Dict[Unknown]) -> List[Unknown]\n```\n\nPerform multi-level reasoning on discovered relations.\n\n---\n\n#### chain_verification\n\n```python\ndef chain_verification(self: Any, reasoning_chain: List[Unknown]) -> Dict[Unknown]\n```\n\nVerify the consistency and correctness of reasoning chain.\n\n---\n\n#### get_component_status\n\n```python\ndef get_component_status(self: Any) -> Dict[Unknown]\n```\n\nGet status of all model components.\n\n---\n\n#### register_model\n\n```python\ndef register_model(cls: Any, model_class: type, model_type: str)\n```\n\nRegister a model class with the factory.\n\n---\n\n#### create_model\n\n```python\ndef create_model(cls: Any, model_type: str, model_name: str, config: Optional[Unknown]) -> BaseModel\n```\n\nCreate a model instance.

Args:
    model_type: Type of model (baseline, llm, proposed)
    model_name: Name of the specific model
    config: Configuration parameters
    
Returns:
    BaseModel: Instance of the requested model\n\n---\n\n#### list_available_models\n\n```python\ndef list_available_models(cls: Any) -> List[str]\n```\n\nList all registered model types.\n\n---\n\n#### evaluate_model\n\n```python\ndef evaluate_model(self: Any, model: BaseModel, test_problems: List[ModelInput]) -> ModelMetrics\n```\n\nEvaluate a model on test problems.

Args:
    model: Model to evaluate
    test_problems: List of test problems with expected answers
    
Returns:
    ModelMetrics: Performance metrics\n\n---\n\n#### compare_models\n\n```python\ndef compare_models(self: Any, models: List[BaseModel], test_problems: List[ModelInput]) -> Dict[Unknown]\n```\n\nCompare multiple models on the same test set.

Args:
    models: List of models to compare
    test_problems: Test problems for evaluation
    
Returns:
    Dict mapping model names to their metrics\n\n---\n\n\n## models.baseline_models\n\n### Classes\n\n#### TemplateBasedModel\n\nTemplate-based baseline model using predefined patterns.\n\n**Inherits from:** BaselineModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the template-based model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using template matching.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using template matching.\n\n##### extract_equations\n```python\ndef extract_equations(self: Any, problem_text: str) -> List[str]\n```\n\nExtract equations using template patterns.\n\n##### solve_equations\n```python\ndef solve_equations(self: Any, equations: List[str]) -> Dict[Unknown]\n```\n\nSolve extracted equations.\n\n---\n\n#### EquationBasedModel\n\nEquation-based model using symbolic math.\n\n**Inherits from:** BaselineModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the equation-based model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem by extracting and solving equations.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using equation-based approach.\n\n##### extract_equations\n```python\ndef extract_equations(self: Any, problem_text: str) -> List[str]\n```\n\nExtract mathematical equations from problem text.\n\n##### solve_equations\n```python\ndef solve_equations(self: Any, equations: List[str]) -> Dict[Unknown]\n```\n\nSolve extracted equations using symbolic math.\n\n---\n\n#### RuleBasedModel\n\nRule-based model using heuristic rules.\n\n**Inherits from:** BaselineModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the rule-based model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using rule-based approach.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using rule-based approach.\n\n##### extract_equations\n```python\ndef extract_equations(self: Any, problem_text: str) -> List[str]\n```\n\nExtract equations using rules.\n\n##### solve_equations\n```python\ndef solve_equations(self: Any, equations: List[str]) -> Dict[Unknown]\n```\n\nSolve equations using rule-based evaluation.\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the template-based model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using template matching.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using template matching.\n\n---\n\n#### extract_equations\n\n```python\ndef extract_equations(self: Any, problem_text: str) -> List[str]\n```\n\nExtract equations using template patterns.\n\n---\n\n#### solve_equations\n\n```python\ndef solve_equations(self: Any, equations: List[str]) -> Dict[Unknown]\n```\n\nSolve extracted equations.\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the equation-based model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem by extracting and solving equations.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using equation-based approach.\n\n---\n\n#### extract_equations\n\n```python\ndef extract_equations(self: Any, problem_text: str) -> List[str]\n```\n\nExtract mathematical equations from problem text.\n\n---\n\n#### solve_equations\n\n```python\ndef solve_equations(self: Any, equations: List[str]) -> Dict[Unknown]\n```\n\nSolve extracted equations using symbolic math.\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the rule-based model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using rule-based approach.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using rule-based approach.\n\n---\n\n#### extract_equations\n\n```python\ndef extract_equations(self: Any, problem_text: str) -> List[str]\n```\n\nExtract equations using rules.\n\n---\n\n#### solve_equations\n\n```python\ndef solve_equations(self: Any, equations: List[str]) -> Dict[Unknown]\n```\n\nSolve equations using rule-based evaluation.\n\n---\n\n\n## models.data_types\n\n定义共享的数据类型\n\n### Classes\n\n#### ExtractionResult\n\n提取结果\n\n---\n\n\n## models.equation\n\n### Classes\n\n#### Equation\n\n---\n\n#### Equations\n\n**Methods:**\n\n##### add_equation\n```python\ndef add_equation(self: Any, equation: Equation)\n```\n\n##### add_variable\n```python\ndef add_variable(self: Any, name: str, value: Any)\n```\n\n---\n\n### Functions\n\n#### add_equation\n\n```python\ndef add_equation(self: Any, equation: Equation)\n```\n\n---\n\n#### add_variable\n\n```python\ndef add_variable(self: Any, name: str, value: Any)\n```\n\n---\n\n\n## models.equations\n\n### Classes\n\n#### Equation\n\n方程类

用于表示数学方程，支持：
- 变量和常数项提取
- 方程化简和标准化
- 方程求解

Attributes:
    expression: 方程表达式
    variables: 变量集合
    constants: 常数集合
    var_entity: 变量实体映射\n\n**Methods:**\n\n##### replace\n```python\ndef replace(self: Any, old: str, new: str) -> str\n```\n\n替换方程中的字符串

Args:
    old: 要替换的字符串
    new: 新的字符串
    
Returns:
    str: 替换后的字符串\n\n##### from_relation\n```python\ndef from_relation(cls: Any, relation: Dict) -> Optional[Equation]\n```\n\n从关系字典创建方程式对象

Args:
    relation: 关系字典
    
Returns:
    Optional[Equation]: 方程式对象\n\n---\n\n#### RelationType\n\n**Inherits from:** Enum\n\n---\n\n#### EquationOperator\n\n**Inherits from:** Enum\n\n---\n\n### Functions\n\n#### replace\n\n```python\ndef replace(self: Any, old: str, new: str) -> str\n```\n\n替换方程中的字符串

Args:
    old: 要替换的字符串
    new: 新的字符串
    
Returns:
    str: 替换后的字符串\n\n---\n\n#### from_relation\n\n```python\ndef from_relation(cls: Any, relation: Dict) -> Optional[Equation]\n```\n\n从关系字典创建方程式对象

Args:
    relation: 关系字典
    
Returns:
    Optional[Equation]: 方程式对象\n\n---\n\n\n## models.llm_models\n\nLarge Language Model (LLM) Implementations

This module implements various LLM models for mathematical word problem solving.
Supports OpenAI GPT, Claude, Qwen, InternLM, and other popular LLMs.\n\n### Classes\n\n#### OpenAIGPTModel\n\nOpenAI GPT model implementation (GPT-3.5, GPT-4, GPT-4o).\n\n**Inherits from:** LLMModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize OpenAI GPT model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using OpenAI GPT.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using OpenAI GPT.\n\n##### generate_prompt\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for OpenAI GPT.\n\n##### call_api\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall OpenAI API with the given prompt.\n\n##### parse_response\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse OpenAI response into structured output.\n\n---\n\n#### ClaudeModel\n\nAnthropic Claude model implementation.\n\n**Inherits from:** LLMModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize Claude model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using Claude.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using Claude.\n\n##### generate_prompt\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for Claude.\n\n##### call_api\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall Claude API with the given prompt.\n\n##### parse_response\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse Claude response into structured output.\n\n---\n\n#### QwenModel\n\nQwen model implementation (local or API).\n\n**Inherits from:** LLMModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize Qwen model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using Qwen.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using Qwen.\n\n##### generate_prompt\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for Qwen.\n\n##### call_api\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall Qwen API or local server.\n\n##### parse_response\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse Qwen response into structured output.\n\n---\n\n#### InternLMModel\n\nInternLM model implementation.\n\n**Inherits from:** LLMModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize InternLM model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using InternLM.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using InternLM.\n\n##### generate_prompt\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for InternLM.\n\n##### call_api\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall InternLM API or generate mock response.\n\n##### parse_response\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse InternLM response into structured output.\n\n---\n\n#### DeepSeekMathModel\n\nDeepSeek-Math model implementation.\n\n**Inherits from:** LLMModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize DeepSeek-Math model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using DeepSeek-Math.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using DeepSeek-Math.\n\n##### generate_prompt\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for DeepSeek-Math.\n\n##### call_api\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall DeepSeek-Math API or generate mock response.\n\n##### parse_response\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse DeepSeek-Math response into structured output.\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize OpenAI GPT model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using OpenAI GPT.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using OpenAI GPT.\n\n---\n\n#### generate_prompt\n\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for OpenAI GPT.\n\n---\n\n#### call_api\n\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall OpenAI API with the given prompt.\n\n---\n\n#### parse_response\n\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse OpenAI response into structured output.\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize Claude model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using Claude.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using Claude.\n\n---\n\n#### generate_prompt\n\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for Claude.\n\n---\n\n#### call_api\n\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall Claude API with the given prompt.\n\n---\n\n#### parse_response\n\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse Claude response into structured output.\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize Qwen model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using Qwen.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using Qwen.\n\n---\n\n#### generate_prompt\n\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for Qwen.\n\n---\n\n#### call_api\n\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall Qwen API or local server.\n\n---\n\n#### parse_response\n\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse Qwen response into structured output.\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize InternLM model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using InternLM.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using InternLM.\n\n---\n\n#### generate_prompt\n\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for InternLM.\n\n---\n\n#### call_api\n\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall InternLM API or generate mock response.\n\n---\n\n#### parse_response\n\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse InternLM response into structured output.\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize DeepSeek-Math model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using DeepSeek-Math.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using DeepSeek-Math.\n\n---\n\n#### generate_prompt\n\n```python\ndef generate_prompt(self: Any, problem_input: ModelInput) -> str\n```\n\nGenerate appropriate prompt for DeepSeek-Math.\n\n---\n\n#### call_api\n\n```python\ndef call_api(self: Any, prompt: str) -> str\n```\n\nCall DeepSeek-Math API or generate mock response.\n\n---\n\n#### parse_response\n\n```python\ndef parse_response(self: Any, response: str) -> ModelOutput\n```\n\nParse DeepSeek-Math response into structured output.\n\n---\n\n\n## models.model_manager\n\nModel Manager

This module provides a unified interface for managing and using all mathematical reasoning models.
It includes model registry, configuration management, and batch processing capabilities.\n\n### Classes\n\n#### ModelRegistry\n\nRegistry for all available models.\n\n**Methods:**\n\n##### register_model\n```python\ndef register_model(self: Any, name: str, model_class: Type[BaseModel])\n```\n\nRegister a model class.\n\n##### get_model_class\n```python\ndef get_model_class(self: Any, name: str) -> Optional[Unknown]\n```\n\nGet model class by name.\n\n##### list_models\n```python\ndef list_models(self: Any) -> Dict[Unknown]\n```\n\nList all registered models.\n\n##### get_models_by_type\n```python\ndef get_models_by_type(self: Any, model_type: str) -> Dict[Unknown]\n```\n\nGet models by type (baseline, llm, proposed).\n\n---\n\n#### ModelManager\n\nUnified manager for all mathematical reasoning models.\n\n**Methods:**\n\n##### initialize_model\n```python\ndef initialize_model(self: Any, model_name: str, config: Optional[Unknown]) -> bool\n```\n\nInitialize a specific model.

Args:
    model_name: Name of the model to initialize
    config: Optional configuration override
    
Returns:
    bool: True if initialization successful\n\n##### initialize_all_models\n```python\ndef initialize_all_models(self: Any) -> Dict[Unknown]\n```\n\nInitialize all enabled models.\n\n##### get_model\n```python\ndef get_model(self: Any, model_name: str) -> Optional[BaseModel]\n```\n\nGet an initialized model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, model_name: str, problem: Union[Unknown]) -> Optional[ModelOutput]\n```\n\nSolve a problem using a specific model.

Args:
    model_name: Name of the model to use
    problem: Problem text or ModelInput object
    
Returns:
    ModelOutput or None if model not available\n\n##### solve_with_multiple_models\n```python\ndef solve_with_multiple_models(self: Any, models: List[str], problem: Union[Unknown]) -> Dict[Unknown]\n```\n\nSolve a problem using multiple models.

Args:
    models: List of model names to use
    problem: Problem text or ModelInput object
    
Returns:
    Dictionary mapping model names to their outputs\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, model_name: str, problems: List[Unknown]) -> List[Unknown]\n```\n\nSolve multiple problems using a single model.

Args:
    model_name: Name of the model to use
    problems: List of problems
    
Returns:
    List of model outputs\n\n##### evaluate_model\n```python\ndef evaluate_model(self: Any, model_name: str, test_problems: List[ModelInput]) -> Optional[ModelMetrics]\n```\n\nEvaluate a specific model.

Args:
    model_name: Name of the model to evaluate
    test_problems: List of test problems with expected answers
    
Returns:
    ModelMetrics or None if evaluation failed\n\n##### compare_models\n```python\ndef compare_models(self: Any, model_names: List[str], test_problems: List[ModelInput]) -> Dict[Unknown]\n```\n\nCompare multiple models on the same test set.

Args:
    model_names: List of model names to compare
    test_problems: List of test problems
    
Returns:
    Dictionary mapping model names to their metrics\n\n##### get_model_info\n```python\ndef get_model_info(self: Any, model_name: str) -> Optional[Unknown]\n```\n\nGet information about a specific model.\n\n##### list_active_models\n```python\ndef list_active_models(self: Any) -> List[str]\n```\n\nList all active (initialized) models.\n\n##### list_available_models\n```python\ndef list_available_models(self: Any) -> Dict[Unknown]\n```\n\nList all available models.\n\n##### save_results\n```python\ndef save_results(self: Any, results: Dict[Unknown], output_path: str)\n```\n\nSave evaluation results to file.\n\n##### load_test_problems\n```python\ndef load_test_problems(self: Any, file_path: str) -> List[ModelInput]\n```\n\nLoad test problems from file.\n\n##### create_model_comparison_report\n```python\ndef create_model_comparison_report(self: Any, comparison_results: Dict[Unknown]) -> Dict[Unknown]\n```\n\nCreate a comprehensive comparison report.\n\n##### shutdown\n```python\ndef shutdown(self: Any)\n```\n\nShutdown all models and cleanup resources.\n\n---\n\n### Functions\n\n#### register_model\n\n```python\ndef register_model(self: Any, name: str, model_class: Type[BaseModel])\n```\n\nRegister a model class.\n\n---\n\n#### get_model_class\n\n```python\ndef get_model_class(self: Any, name: str) -> Optional[Unknown]\n```\n\nGet model class by name.\n\n---\n\n#### list_models\n\n```python\ndef list_models(self: Any) -> Dict[Unknown]\n```\n\nList all registered models.\n\n---\n\n#### get_models_by_type\n\n```python\ndef get_models_by_type(self: Any, model_type: str) -> Dict[Unknown]\n```\n\nGet models by type (baseline, llm, proposed).\n\n---\n\n#### initialize_model\n\n```python\ndef initialize_model(self: Any, model_name: str, config: Optional[Unknown]) -> bool\n```\n\nInitialize a specific model.

Args:
    model_name: Name of the model to initialize
    config: Optional configuration override
    
Returns:
    bool: True if initialization successful\n\n---\n\n#### initialize_all_models\n\n```python\ndef initialize_all_models(self: Any) -> Dict[Unknown]\n```\n\nInitialize all enabled models.\n\n---\n\n#### get_model\n\n```python\ndef get_model(self: Any, model_name: str) -> Optional[BaseModel]\n```\n\nGet an initialized model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, model_name: str, problem: Union[Unknown]) -> Optional[ModelOutput]\n```\n\nSolve a problem using a specific model.

Args:
    model_name: Name of the model to use
    problem: Problem text or ModelInput object
    
Returns:
    ModelOutput or None if model not available\n\n---\n\n#### solve_with_multiple_models\n\n```python\ndef solve_with_multiple_models(self: Any, models: List[str], problem: Union[Unknown]) -> Dict[Unknown]\n```\n\nSolve a problem using multiple models.

Args:
    models: List of model names to use
    problem: Problem text or ModelInput object
    
Returns:
    Dictionary mapping model names to their outputs\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, model_name: str, problems: List[Unknown]) -> List[Unknown]\n```\n\nSolve multiple problems using a single model.

Args:
    model_name: Name of the model to use
    problems: List of problems
    
Returns:
    List of model outputs\n\n---\n\n#### evaluate_model\n\n```python\ndef evaluate_model(self: Any, model_name: str, test_problems: List[ModelInput]) -> Optional[ModelMetrics]\n```\n\nEvaluate a specific model.

Args:
    model_name: Name of the model to evaluate
    test_problems: List of test problems with expected answers
    
Returns:
    ModelMetrics or None if evaluation failed\n\n---\n\n#### compare_models\n\n```python\ndef compare_models(self: Any, model_names: List[str], test_problems: List[ModelInput]) -> Dict[Unknown]\n```\n\nCompare multiple models on the same test set.

Args:
    model_names: List of model names to compare
    test_problems: List of test problems
    
Returns:
    Dictionary mapping model names to their metrics\n\n---\n\n#### get_model_info\n\n```python\ndef get_model_info(self: Any, model_name: str) -> Optional[Unknown]\n```\n\nGet information about a specific model.\n\n---\n\n#### list_active_models\n\n```python\ndef list_active_models(self: Any) -> List[str]\n```\n\nList all active (initialized) models.\n\n---\n\n#### list_available_models\n\n```python\ndef list_available_models(self: Any) -> Dict[Unknown]\n```\n\nList all available models.\n\n---\n\n#### save_results\n\n```python\ndef save_results(self: Any, results: Dict[Unknown], output_path: str)\n```\n\nSave evaluation results to file.\n\n---\n\n#### load_test_problems\n\n```python\ndef load_test_problems(self: Any, file_path: str) -> List[ModelInput]\n```\n\nLoad test problems from file.\n\n---\n\n#### create_model_comparison_report\n\n```python\ndef create_model_comparison_report(self: Any, comparison_results: Dict[Unknown]) -> Dict[Unknown]\n```\n\nCreate a comprehensive comparison report.\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any)\n```\n\nShutdown all models and cleanup resources.\n\n---\n\n\n## models.orchestrator\n\nModels Module - Orchestrator
============================

模型模块协调器：负责协调模型相关操作

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### ModelsOrchestrator\n\n模型模块协调器\n\n**Methods:**\n\n##### initialize_orchestrator\n```python\ndef initialize_orchestrator(self: Any) -> bool\n```\n\n初始化协调器\n\n##### orchestrate\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行\n\n##### register_component\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件\n\n##### get_component\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件\n\n##### get_operation_history\n```python\ndef get_operation_history(self: Any) -> List[Unknown]\n```\n\n获取操作历史\n\n##### clear_operation_history\n```python\ndef clear_operation_history(self: Any) -> Any\n```\n\n清空操作历史\n\n---\n\n### Functions\n\n#### initialize_orchestrator\n\n```python\ndef initialize_orchestrator(self: Any) -> bool\n```\n\n初始化协调器\n\n---\n\n#### orchestrate\n\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行\n\n---\n\n#### register_component\n\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件\n\n---\n\n#### get_component\n\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件\n\n---\n\n#### get_operation_history\n\n```python\ndef get_operation_history(self: Any) -> List[Unknown]\n```\n\n获取操作历史\n\n---\n\n#### clear_operation_history\n\n```python\ndef clear_operation_history(self: Any) -> Any\n```\n\n清空操作历史\n\n---\n\n\n## models.pattern_loader\n\n模式加载器
~~~~~~~~

这个模块负责加载和管理模式定义，提供统一的模式访问接口。\n\n### Classes\n\n#### PatternLoader\n\n模式加载器类\n\n**Methods:**\n\n##### get_pattern\n```python\ndef get_pattern(self: Any, problem_type: str, subtype: Optional[str]) -> Dict[Unknown]\n```\n\n获取模式定义

Args:
    problem_type: 问题类型
    subtype: 子类型，如果为None，则返回所有子类型
    
Returns:
    Dict[str, Any]: 模式定义字典\n\n##### get_compiled_pattern\n```python\ndef get_compiled_pattern(self: Any, problem_type: str, subtype: str, key: str) -> List[Pattern]\n```\n\n获取编译后的正则表达式模式

Args:
    problem_type: 问题类型
    subtype: 子类型
    key: 模式键名
    
Returns:
    List[Pattern]: 编译后的正则表达式模式列表\n\n##### get_keywords\n```python\ndef get_keywords(self: Any, problem_type: str, subtype: Optional[str]) -> List[str]\n```\n\n获取关键词

Args:
    problem_type: 问题类型
    subtype: 子类型，如果为None，则返回所有子类型的关键词
    
Returns:
    List[str]: 关键词列表\n\n##### get_equation_template\n```python\ndef get_equation_template(self: Any, problem_type: str, subtype: str) -> str\n```\n\n获取方程模板

Args:
    problem_type: 问题类型
    subtype: 子类型
    
Returns:
    str: 方程模板\n\n##### get_subtypes\n```python\ndef get_subtypes(self: Any, problem_type: str) -> List[str]\n```\n\n获取子类型

Args:
    problem_type: 问题类型
    
Returns:
    List[str]: 子类型列表\n\n##### match_pattern\n```python\ndef match_pattern(self: Any, text: str, problem_type: str, subtype: str, key: str) -> List[Unknown]\n```\n\n匹配模式

Args:
    text: 文本
    problem_type: 问题类型
    subtype: 子类型
    key: 模式键名
    
Returns:
    List[Dict[str, Any]]: 匹配结果列表，每个元素包含匹配的值和单位\n\n##### identify_subtype\n```python\ndef identify_subtype(self: Any, text: str, problem_type: str) -> str\n```\n\n识别子类型

Args:
    text: 文本
    problem_type: 问题类型
    
Returns:
    str: 子类型\n\n##### extract_variables\n```python\ndef extract_variables(self: Any, text: str, problem_type: str, subtype: Optional[str]) -> Dict[Unknown]\n```\n\n提取变量

Args:
    text: 文本
    problem_type: 问题类型
    subtype: 子类型，如果为None，则自动识别
    
Returns:
    Dict[str, Any]: 提取的变量、单位和子类型\n\n##### reload\n```python\ndef reload(self: Any) -> Any\n```\n\n重新加载模式定义\n\n---\n\n### Functions\n\n#### get_pattern_loader\n\n```python\ndef get_pattern_loader() -> PatternLoader\n```\n\n获取模式加载器实例

Returns:
    PatternLoader: 模式加载器实例\n\n---\n\n#### get_pattern\n\n```python\ndef get_pattern(self: Any, problem_type: str, subtype: Optional[str]) -> Dict[Unknown]\n```\n\n获取模式定义

Args:
    problem_type: 问题类型
    subtype: 子类型，如果为None，则返回所有子类型
    
Returns:
    Dict[str, Any]: 模式定义字典\n\n---\n\n#### get_compiled_pattern\n\n```python\ndef get_compiled_pattern(self: Any, problem_type: str, subtype: str, key: str) -> List[Pattern]\n```\n\n获取编译后的正则表达式模式

Args:
    problem_type: 问题类型
    subtype: 子类型
    key: 模式键名
    
Returns:
    List[Pattern]: 编译后的正则表达式模式列表\n\n---\n\n#### get_keywords\n\n```python\ndef get_keywords(self: Any, problem_type: str, subtype: Optional[str]) -> List[str]\n```\n\n获取关键词

Args:
    problem_type: 问题类型
    subtype: 子类型，如果为None，则返回所有子类型的关键词
    
Returns:
    List[str]: 关键词列表\n\n---\n\n#### get_equation_template\n\n```python\ndef get_equation_template(self: Any, problem_type: str, subtype: str) -> str\n```\n\n获取方程模板

Args:
    problem_type: 问题类型
    subtype: 子类型
    
Returns:
    str: 方程模板\n\n---\n\n#### get_subtypes\n\n```python\ndef get_subtypes(self: Any, problem_type: str) -> List[str]\n```\n\n获取子类型

Args:
    problem_type: 问题类型
    
Returns:
    List[str]: 子类型列表\n\n---\n\n#### match_pattern\n\n```python\ndef match_pattern(self: Any, text: str, problem_type: str, subtype: str, key: str) -> List[Unknown]\n```\n\n匹配模式

Args:
    text: 文本
    problem_type: 问题类型
    subtype: 子类型
    key: 模式键名
    
Returns:
    List[Dict[str, Any]]: 匹配结果列表，每个元素包含匹配的值和单位\n\n---\n\n#### identify_subtype\n\n```python\ndef identify_subtype(self: Any, text: str, problem_type: str) -> str\n```\n\n识别子类型

Args:
    text: 文本
    problem_type: 问题类型
    
Returns:
    str: 子类型\n\n---\n\n#### extract_variables\n\n```python\ndef extract_variables(self: Any, text: str, problem_type: str, subtype: Optional[str]) -> Dict[Unknown]\n```\n\n提取变量

Args:
    text: 文本
    problem_type: 问题类型
    subtype: 子类型，如果为None，则自动识别
    
Returns:
    Dict[str, Any]: 提取的变量、单位和子类型\n\n---\n\n#### reload\n\n```python\ndef reload(self: Any) -> Any\n```\n\n重新加载模式定义\n\n---\n\n\n## models.private.model_cache\n\n模型缓存管理器 (Model Cache Manager)

专注于模型结果的缓存、性能优化和内存管理。\n\n### Classes\n\n#### CacheEntry\n\n缓存条目\n\n**Methods:**\n\n##### is_expired\n```python\ndef is_expired(self: Any) -> bool\n```\n\n检查是否过期\n\n##### touch\n```python\ndef touch(self: Any)\n```\n\n更新访问时间和计数\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典\n\n---\n\n#### ModelCacheManager\n\n模型缓存管理器\n\n**Methods:**\n\n##### get\n```python\ndef get(self: Any, key: str) -> Optional[Any]\n```\n\n获取缓存值

Args:
    key: 缓存键
    
Returns:
    缓存的值，如果不存在或过期则返回None\n\n##### put\n```python\ndef put(self: Any, key: str, value: Any, ttl: Optional[float]) -> bool\n```\n\n存储缓存值

Args:
    key: 缓存键
    value: 要缓存的值
    ttl: 生存时间（秒），None使用默认值
    
Returns:
    是否成功存储\n\n##### remove\n```python\ndef remove(self: Any, key: str) -> bool\n```\n\n删除缓存条目\n\n##### clear\n```python\ndef clear(self: Any)\n```\n\n清空所有缓存\n\n##### get_problem_hash\n```python\ndef get_problem_hash(self: Any, problem: Dict[Unknown], model_name: str, config: Dict[Unknown]) -> str\n```\n\n生成问题的哈希键

Args:
    problem: 问题数据
    model_name: 模型名称
    config: 模型配置
    
Returns:
    哈希键\n\n##### cache_model_result\n```python\ndef cache_model_result(self: Any, problem: Dict[Unknown], model_name: str, result: Dict[Unknown], model_config: Optional[Unknown], ttl: Optional[float]) -> bool\n```\n\n缓存模型结果

Args:
    problem: 问题数据
    model_name: 模型名称
    result: 模型结果
    model_config: 模型配置
    ttl: 生存时间
    
Returns:
    是否成功缓存\n\n##### get_cached_model_result\n```python\ndef get_cached_model_result(self: Any, problem: Dict[Unknown], model_name: str, model_config: Optional[Unknown]) -> Optional[Unknown]\n```\n\n获取缓存的模型结果

Args:
    problem: 问题数据
    model_name: 模型名称
    model_config: 模型配置
    
Returns:
    缓存的结果，如果不存在则返回None\n\n##### get_cache_stats\n```python\ndef get_cache_stats(self: Any) -> Dict[Unknown]\n```\n\n获取缓存统计信息\n\n##### get_cache_entries_info\n```python\ndef get_cache_entries_info(self: Any, limit: int) -> List[Unknown]\n```\n\n获取缓存条目信息\n\n##### optimize_cache\n```python\ndef optimize_cache(self: Any)\n```\n\n优化缓存\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n##### shutdown\n```python\ndef shutdown(self: Any)\n```\n\n关闭缓存管理器\n\n---\n\n### Functions\n\n#### is_expired\n\n```python\ndef is_expired(self: Any) -> bool\n```\n\n检查是否过期\n\n---\n\n#### touch\n\n```python\ndef touch(self: Any)\n```\n\n更新访问时间和计数\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典\n\n---\n\n#### get\n\n```python\ndef get(self: Any, key: str) -> Optional[Any]\n```\n\n获取缓存值

Args:
    key: 缓存键
    
Returns:
    缓存的值，如果不存在或过期则返回None\n\n---\n\n#### put\n\n```python\ndef put(self: Any, key: str, value: Any, ttl: Optional[float]) -> bool\n```\n\n存储缓存值

Args:
    key: 缓存键
    value: 要缓存的值
    ttl: 生存时间（秒），None使用默认值
    
Returns:
    是否成功存储\n\n---\n\n#### remove\n\n```python\ndef remove(self: Any, key: str) -> bool\n```\n\n删除缓存条目\n\n---\n\n#### clear\n\n```python\ndef clear(self: Any)\n```\n\n清空所有缓存\n\n---\n\n#### get_problem_hash\n\n```python\ndef get_problem_hash(self: Any, problem: Dict[Unknown], model_name: str, config: Dict[Unknown]) -> str\n```\n\n生成问题的哈希键

Args:
    problem: 问题数据
    model_name: 模型名称
    config: 模型配置
    
Returns:
    哈希键\n\n---\n\n#### cache_model_result\n\n```python\ndef cache_model_result(self: Any, problem: Dict[Unknown], model_name: str, result: Dict[Unknown], model_config: Optional[Unknown], ttl: Optional[float]) -> bool\n```\n\n缓存模型结果

Args:
    problem: 问题数据
    model_name: 模型名称
    result: 模型结果
    model_config: 模型配置
    ttl: 生存时间
    
Returns:
    是否成功缓存\n\n---\n\n#### get_cached_model_result\n\n```python\ndef get_cached_model_result(self: Any, problem: Dict[Unknown], model_name: str, model_config: Optional[Unknown]) -> Optional[Unknown]\n```\n\n获取缓存的模型结果

Args:
    problem: 问题数据
    model_name: 模型名称
    model_config: 模型配置
    
Returns:
    缓存的结果，如果不存在则返回None\n\n---\n\n#### get_cache_stats\n\n```python\ndef get_cache_stats(self: Any) -> Dict[Unknown]\n```\n\n获取缓存统计信息\n\n---\n\n#### get_cache_entries_info\n\n```python\ndef get_cache_entries_info(self: Any, limit: int) -> List[Unknown]\n```\n\n获取缓存条目信息\n\n---\n\n#### optimize_cache\n\n```python\ndef optimize_cache(self: Any)\n```\n\n优化缓存\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any)\n```\n\n关闭缓存管理器\n\n---\n\n\n## models.private.model_factory\n\n模型工厂 (Model Factory)

专注于模型的创建、配置和初始化。\n\n### Classes\n\n#### ModelCreationError\n\n模型创建错误\n\n**Inherits from:** Exception\n\n---\n\n#### ModelConfigurationError\n\n模型配置错误\n\n**Inherits from:** Exception\n\n---\n\n#### BaseModel\n\n基础模型类\n\n**Methods:**\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决问题的基础方法\n\n---\n\n#### BaselineModel\n\n基线模型基类\n\n**Inherits from:** BaseModel\n\n---\n\n#### LLMModel\n\n大语言模型基类\n\n**Inherits from:** BaseModel\n\n---\n\n#### ProposedModel\n\n提出模型基类\n\n**Inherits from:** BaseModel\n\n---\n\n#### TemplateBasedModel\n\n模板基线模型\n\n**Inherits from:** BaselineModel\n\n---\n\n#### EquationBasedModel\n\n方程基线模型\n\n**Inherits from:** BaselineModel\n\n---\n\n#### RuleBasedModel\n\n规则基线模型\n\n**Inherits from:** BaselineModel\n\n---\n\n#### SimplePatternModel\n\n简单模式模型\n\n**Inherits from:** BaselineModel\n\n---\n\n#### OpenAIGPTModel\n\nOpenAI GPT模型\n\n**Inherits from:** LLMModel\n\n---\n\n#### ClaudeModel\n\nClaude模型\n\n**Inherits from:** LLMModel\n\n---\n\n#### QwenModel\n\nQwen模型\n\n**Inherits from:** LLMModel\n\n---\n\n#### InternLMModel\n\nInternLM模型\n\n**Inherits from:** LLMModel\n\n---\n\n#### DeepSeekMathModel\n\nDeepSeek数学模型\n\n**Inherits from:** LLMModel\n\n---\n\n#### COTDIRModel\n\nCOT-DIR模型\n\n**Inherits from:** ProposedModel\n\n---\n\n#### ModelFactory\n\n模型工厂 - 负责创建和配置模型实例\n\n**Methods:**\n\n##### register_model_class\n```python\ndef register_model_class(self: Any, name: str, model_class: Type[BaseModel], model_type: str)\n```\n\n注册模型类

Args:
    name: 模型名称
    model_class: 模型类
    model_type: 模型类型 (baseline, llm, proposed)\n\n##### create_model\n```python\ndef create_model(self: Any, model_name: str, config: Optional[Unknown]) -> BaseModel\n```\n\n创建模型实例

Args:
    model_name: 模型名称
    config: 模型配置
    
Returns:
    创建的模型实例\n\n##### batch_create_models\n```python\ndef batch_create_models(self: Any, model_configs: Dict[Unknown]) -> Dict[Unknown]\n```\n\n批量创建模型

Args:
    model_configs: 模型名称到配置的映射
    
Returns:
    模型名称到模型实例的映射\n\n##### get_available_models\n```python\ndef get_available_models(self: Any) -> Dict[Unknown]\n```\n\n获取可用模型列表\n\n##### get_models_by_type\n```python\ndef get_models_by_type(self: Any, model_type: str) -> Dict[Unknown]\n```\n\n按类型获取模型\n\n##### get_model_info\n```python\ndef get_model_info(self: Any, model_name: str) -> Optional[Unknown]\n```\n\n获取模型信息\n\n##### get_creation_stats\n```python\ndef get_creation_stats(self: Any) -> Dict[Unknown]\n```\n\n获取创建统计\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n### Functions\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决问题的基础方法\n\n---\n\n#### register_model_class\n\n```python\ndef register_model_class(self: Any, name: str, model_class: Type[BaseModel], model_type: str)\n```\n\n注册模型类

Args:
    name: 模型名称
    model_class: 模型类
    model_type: 模型类型 (baseline, llm, proposed)\n\n---\n\n#### create_model\n\n```python\ndef create_model(self: Any, model_name: str, config: Optional[Unknown]) -> BaseModel\n```\n\n创建模型实例

Args:
    model_name: 模型名称
    config: 模型配置
    
Returns:
    创建的模型实例\n\n---\n\n#### batch_create_models\n\n```python\ndef batch_create_models(self: Any, model_configs: Dict[Unknown]) -> Dict[Unknown]\n```\n\n批量创建模型

Args:
    model_configs: 模型名称到配置的映射
    
Returns:
    模型名称到模型实例的映射\n\n---\n\n#### get_available_models\n\n```python\ndef get_available_models(self: Any) -> Dict[Unknown]\n```\n\n获取可用模型列表\n\n---\n\n#### get_models_by_type\n\n```python\ndef get_models_by_type(self: Any, model_type: str) -> Dict[Unknown]\n```\n\n按类型获取模型\n\n---\n\n#### get_model_info\n\n```python\ndef get_model_info(self: Any, model_name: str) -> Optional[Unknown]\n```\n\n获取模型信息\n\n---\n\n#### get_creation_stats\n\n```python\ndef get_creation_stats(self: Any) -> Dict[Unknown]\n```\n\n获取创建统计\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n\n## models.private.performance_tracker\n\n性能监控器 (Performance Monitor)

专注于模型性能的监控、分析和报告。\n\n### Classes\n\n#### PerformanceMetric\n\n性能指标\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典\n\n---\n\n#### PerformanceTracker\n\n性能跟踪器\n\n**Methods:**\n\n##### record_metric\n```python\ndef record_metric(self: Any, metric: PerformanceMetric)\n```\n\n记录性能指标\n\n##### get_model_performance\n```python\ndef get_model_performance(self: Any, model_name: str) -> Dict[Unknown]\n```\n\n获取模型性能统计\n\n##### get_operation_performance\n```python\ndef get_operation_performance(self: Any, operation: str) -> Dict[Unknown]\n```\n\n获取操作性能统计\n\n##### get_performance_trends\n```python\ndef get_performance_trends(self: Any, time_window: int) -> Dict[Unknown]\n```\n\n获取性能趋势（时间窗口内）\n\n---\n\n#### PerformanceMonitor\n\n性能监控器\n\n**Methods:**\n\n##### monitor_model_call\n```python\ndef monitor_model_call(self: Any, model_name: str, operation: str, start_time: float, end_time: float, success: bool, input_size: int, output_size: int, error_message: Optional[str])\n```\n\n监控模型调用\n\n##### get_system_overview\n```python\ndef get_system_overview(self: Any) -> Dict[Unknown]\n```\n\n获取系统概览\n\n##### get_model_ranking\n```python\ndef get_model_ranking(self: Any, metric: str) -> List[Unknown]\n```\n\n获取模型性能排名\n\n##### get_performance_report\n```python\ndef get_performance_report(self: Any, model_name: Optional[str]) -> Dict[Unknown]\n```\n\n获取性能报告\n\n##### get_health_status\n```python\ndef get_health_status(self: Any) -> Dict[Unknown]\n```\n\n获取健康状态\n\n##### clear_alerts\n```python\ndef clear_alerts(self: Any, alert_type: Optional[str])\n```\n\n清除告警\n\n##### export_metrics\n```python\ndef export_metrics(self: Any, format: str) -> Union[Unknown]\n```\n\n导出性能指标\n\n##### reset_metrics\n```python\ndef reset_metrics(self: Any)\n```\n\n重置所有性能指标\n\n##### shutdown\n```python\ndef shutdown(self: Any)\n```\n\n关闭性能监控器\n\n---\n\n### Functions\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典\n\n---\n\n#### record_metric\n\n```python\ndef record_metric(self: Any, metric: PerformanceMetric)\n```\n\n记录性能指标\n\n---\n\n#### get_model_performance\n\n```python\ndef get_model_performance(self: Any, model_name: str) -> Dict[Unknown]\n```\n\n获取模型性能统计\n\n---\n\n#### get_operation_performance\n\n```python\ndef get_operation_performance(self: Any, operation: str) -> Dict[Unknown]\n```\n\n获取操作性能统计\n\n---\n\n#### get_performance_trends\n\n```python\ndef get_performance_trends(self: Any, time_window: int) -> Dict[Unknown]\n```\n\n获取性能趋势（时间窗口内）\n\n---\n\n#### monitor_model_call\n\n```python\ndef monitor_model_call(self: Any, model_name: str, operation: str, start_time: float, end_time: float, success: bool, input_size: int, output_size: int, error_message: Optional[str])\n```\n\n监控模型调用\n\n---\n\n#### get_system_overview\n\n```python\ndef get_system_overview(self: Any) -> Dict[Unknown]\n```\n\n获取系统概览\n\n---\n\n#### get_model_ranking\n\n```python\ndef get_model_ranking(self: Any, metric: str) -> List[Unknown]\n```\n\n获取模型性能排名\n\n---\n\n#### get_performance_report\n\n```python\ndef get_performance_report(self: Any, model_name: Optional[str]) -> Dict[Unknown]\n```\n\n获取性能报告\n\n---\n\n#### get_health_status\n\n```python\ndef get_health_status(self: Any) -> Dict[Unknown]\n```\n\n获取健康状态\n\n---\n\n#### clear_alerts\n\n```python\ndef clear_alerts(self: Any, alert_type: Optional[str])\n```\n\n清除告警\n\n---\n\n#### export_metrics\n\n```python\ndef export_metrics(self: Any, format: str) -> Union[Unknown]\n```\n\n导出性能指标\n\n---\n\n#### reset_metrics\n\n```python\ndef reset_metrics(self: Any)\n```\n\n重置所有性能指标\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any)\n```\n\n关闭性能监控器\n\n---\n\n\n## models.private.processor\n\nModels Module - Core Processor
==============================

核心处理器：整合模型相关的处理功能

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### ModelCoreProcessor\n\n模型核心处理器\n\n**Methods:**\n\n##### process_model_data\n```python\ndef process_model_data(self: Any, model_data: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理模型数据\n\n---\n\n### Functions\n\n#### process_model_data\n\n```python\ndef process_model_data(self: Any, model_data: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理模型数据\n\n---\n\n\n## models.private.utils\n\nModels Module - Utility Functions
=================================

工具函数：提供模型相关的辅助功能

Author: AI Assistant
Date: 2024-07-13\n\n### Functions\n\n#### validate_model_structure\n\n```python\ndef validate_model_structure(model: Dict[Unknown]) -> bool\n```\n\n验证模型结构\n\n---\n\n#### format_model_output\n\n```python\ndef format_model_output(model_data: Dict[Unknown]) -> Dict[Unknown]\n```\n\n格式化模型输出\n\n---\n\n#### merge_model_results\n\n```python\ndef merge_model_results(results: List[Unknown]) -> Dict[Unknown]\n```\n\n合并模型结果\n\n---\n\n\n## models.private.validator\n\nModels Module - Data Validator
==============================

数据验证器：负责验证模型相关数据的有效性

Author: AI Assistant  
Date: 2024-07-13\n\n### Classes\n\n#### ModelValidator\n\n模型数据验证器\n\n**Methods:**\n\n##### validate_model_input\n```python\ndef validate_model_input(self: Any, model_data: Any) -> Dict[Unknown]\n```\n\n验证模型输入数据\n\n##### validate_equation_input\n```python\ndef validate_equation_input(self: Any, equation: Any) -> Dict[Unknown]\n```\n\n验证方程输入\n\n---\n\n### Functions\n\n#### validate_model_input\n\n```python\ndef validate_model_input(self: Any, model_data: Any) -> Dict[Unknown]\n```\n\n验证模型输入数据\n\n---\n\n#### validate_equation_input\n\n```python\ndef validate_equation_input(self: Any, equation: Any) -> Dict[Unknown]\n```\n\n验证方程输入\n\n---\n\n\n## models.processed_text\n\n处理后的文本类\n\n### Classes\n\n#### ProcessedText\n\n处理后的文本类

Attributes:
    raw_text: 原始文本
    segmentation: 分词结果
    pos_tags: 词性标注结果
    dependencies: 依存句法结果
    semantic_roles: 语义角色结果
    cleaned_text: 清理后的文本
    tokens: 分词结果
    ner_tags: 命名实体识别结果
    features: 额外特征字典
    values_and_units: 数值和单位信息\n\n---\n\n\n## models.proposed_model\n\n### Classes\n\n#### ComplexityLevel\n\nComplexity levels for mathematical problems.\n\n**Inherits from:** Enum\n\n---\n\n#### RelationType\n\nTypes of implicit relations.\n\n**Inherits from:** Enum\n\n---\n\n#### ImplicitRelation\n\nRepresents an implicit relation discovered in the problem.\n\n---\n\n#### ReasoningStep\n\nRepresents a step in the reasoning chain.\n\n---\n\n#### VerificationResult\n\nResults from chain verification.\n\n---\n\n#### COTDIRModel\n\nChain-of-Thought with Directional Implicit Reasoning model.\n\n**Inherits from:** ProposedModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the COT-DIR model.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using COT-DIR approach.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using COT-DIR.\n\n##### implicit_relation_discovery\n```python\ndef implicit_relation_discovery(self: Any, problem_input: ModelInput) -> List[ImplicitRelation]\n```\n\nDiscover implicit relations in the problem using IRD component.\n\n##### multi_level_reasoning\n```python\ndef multi_level_reasoning(self: Any, context: Dict[Unknown]) -> List[ReasoningStep]\n```\n\nPerform multi-level reasoning using MLR component.\n\n##### chain_verification\n```python\ndef chain_verification(self: Any, reasoning_chain: List[ReasoningStep]) -> VerificationResult\n```\n\nVerify the consistency and correctness of reasoning chain using CV component.\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitialize the COT-DIR model.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolve problem using COT-DIR approach.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolve multiple problems using COT-DIR.\n\n---\n\n#### implicit_relation_discovery\n\n```python\ndef implicit_relation_discovery(self: Any, problem_input: ModelInput) -> List[ImplicitRelation]\n```\n\nDiscover implicit relations in the problem using IRD component.\n\n---\n\n#### multi_level_reasoning\n\n```python\ndef multi_level_reasoning(self: Any, context: Dict[Unknown]) -> List[ReasoningStep]\n```\n\nPerform multi-level reasoning using MLR component.\n\n---\n\n#### chain_verification\n\n```python\ndef chain_verification(self: Any, reasoning_chain: List[ReasoningStep]) -> VerificationResult\n```\n\nVerify the consistency and correctness of reasoning chain using CV component.\n\n---\n\n\n## models.public_api\n\nModels Module - Public API
==========================

模型模块公共API：提供统一的模型接口

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### ModelsAPI\n\n模型模块公共API\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化模型模块\n\n##### create_model\n```python\ndef create_model(self: Any, model_data: Union[Unknown], model_type: str, config: Optional[Dict]) -> Dict[Unknown]\n```\n\n创建模型\n\n##### process_equations\n```python\ndef process_equations(self: Any, equations: Any, config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理方程\n\n##### process_relations\n```python\ndef process_relations(self: Any, relations: Any, config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理关系\n\n##### get_module_status\n```python\ndef get_module_status(self: Any) -> Dict[Unknown]\n```\n\n获取模块状态\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭模型模块\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化模型模块\n\n---\n\n#### create_model\n\n```python\ndef create_model(self: Any, model_data: Union[Unknown], model_type: str, config: Optional[Dict]) -> Dict[Unknown]\n```\n\n创建模型\n\n---\n\n#### process_equations\n\n```python\ndef process_equations(self: Any, equations: Any, config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理方程\n\n---\n\n#### process_relations\n\n```python\ndef process_relations(self: Any, relations: Any, config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理关系\n\n---\n\n#### get_module_status\n\n```python\ndef get_module_status(self: Any) -> Dict[Unknown]\n```\n\n获取模块状态\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭模型模块\n\n---\n\n\n## models.public_api_refactored\n\n模型管理重构版公共API

整合模型工厂、缓存管理和性能监控，提供统一的模型管理接口。\n\n### Classes\n\n#### ModelAPI\n\n模型管理重构版公共API\n\n**Inherits from:** PublicAPI\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化模型管理模块\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### solve_with_model\n```python\ndef solve_with_model(self: Any, model_name: str, problem: Dict[Unknown], model_config: Optional[Unknown], use_cache: bool) -> Dict[Unknown]\n```\n\n使用指定模型解决问题

Args:
    model_name: 模型名称
    problem: 问题数据
    model_config: 模型配置
    use_cache: 是否使用缓存
    
Returns:
    模型求解结果\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, model_name: str, problems: List[Unknown], model_config: Optional[Unknown], use_cache: bool, max_workers: int) -> List[Unknown]\n```\n\n批量求解问题

Args:
    model_name: 模型名称
    problems: 问题列表
    model_config: 模型配置
    use_cache: 是否使用缓存
    max_workers: 最大并发数
    
Returns:
    结果列表\n\n##### compare_models\n```python\ndef compare_models(self: Any, model_names: List[str], problems: List[Unknown], model_configs: Optional[Unknown]) -> Dict[Unknown]\n```\n\n模型性能比较

Args:
    model_names: 模型名称列表
    problems: 测试问题列表
    model_configs: 各模型配置
    
Returns:
    比较结果\n\n##### get_available_models\n```python\ndef get_available_models(self: Any) -> Dict[Unknown]\n```\n\n获取可用模型列表\n\n##### get_model_info\n```python\ndef get_model_info(self: Any, model_name: str) -> Dict[Unknown]\n```\n\n获取模型详细信息\n\n##### get_performance_report\n```python\ndef get_performance_report(self: Any, model_name: Optional[str]) -> Dict[Unknown]\n```\n\n获取性能报告\n\n##### get_cache_stats\n```python\ndef get_cache_stats(self: Any) -> Dict[Unknown]\n```\n\n获取缓存统计\n\n##### clear_cache\n```python\ndef clear_cache(self: Any, model_name: Optional[str])\n```\n\n清空缓存\n\n##### optimize_cache\n```python\ndef optimize_cache(self: Any)\n```\n\n优化缓存\n\n##### get_statistics\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取模块统计信息\n\n##### reset_statistics\n```python\ndef reset_statistics(self: Any)\n```\n\n重置统计信息\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭模型管理模块\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化模型管理模块\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### solve_with_model\n\n```python\ndef solve_with_model(self: Any, model_name: str, problem: Dict[Unknown], model_config: Optional[Unknown], use_cache: bool) -> Dict[Unknown]\n```\n\n使用指定模型解决问题

Args:
    model_name: 模型名称
    problem: 问题数据
    model_config: 模型配置
    use_cache: 是否使用缓存
    
Returns:
    模型求解结果\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, model_name: str, problems: List[Unknown], model_config: Optional[Unknown], use_cache: bool, max_workers: int) -> List[Unknown]\n```\n\n批量求解问题

Args:
    model_name: 模型名称
    problems: 问题列表
    model_config: 模型配置
    use_cache: 是否使用缓存
    max_workers: 最大并发数
    
Returns:
    结果列表\n\n---\n\n#### compare_models\n\n```python\ndef compare_models(self: Any, model_names: List[str], problems: List[Unknown], model_configs: Optional[Unknown]) -> Dict[Unknown]\n```\n\n模型性能比较

Args:
    model_names: 模型名称列表
    problems: 测试问题列表
    model_configs: 各模型配置
    
Returns:
    比较结果\n\n---\n\n#### get_available_models\n\n```python\ndef get_available_models(self: Any) -> Dict[Unknown]\n```\n\n获取可用模型列表\n\n---\n\n#### get_model_info\n\n```python\ndef get_model_info(self: Any, model_name: str) -> Dict[Unknown]\n```\n\n获取模型详细信息\n\n---\n\n#### get_performance_report\n\n```python\ndef get_performance_report(self: Any, model_name: Optional[str]) -> Dict[Unknown]\n```\n\n获取性能报告\n\n---\n\n#### get_cache_stats\n\n```python\ndef get_cache_stats(self: Any) -> Dict[Unknown]\n```\n\n获取缓存统计\n\n---\n\n#### clear_cache\n\n```python\ndef clear_cache(self: Any, model_name: Optional[str])\n```\n\n清空缓存\n\n---\n\n#### optimize_cache\n\n```python\ndef optimize_cache(self: Any)\n```\n\n优化缓存\n\n---\n\n#### get_statistics\n\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取模块统计信息\n\n---\n\n#### reset_statistics\n\n```python\ndef reset_statistics(self: Any)\n```\n\n重置统计信息\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭模型管理模块\n\n---\n\n\n## models.relation\n\n### Classes\n\n#### RelationType\n\n**Inherits from:** Enum\n\n---\n\n#### Entity\n\n实体类\n\n---\n\n#### Relation\n\n关系类\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n##### from_dict\n```python\ndef from_dict(cls: Any, data: Dict[Unknown]) -> Any\n```\n\n从字典创建关系实例\n\n---\n\n#### Relations\n\n**Methods:**\n\n##### relations\n```python\ndef relations(self: Any) -> List[Relation]\n```\n\n##### add_relation\n```python\ndef add_relation(self: Any, relation: Relation, is_explicit: bool)\n```\n\n##### get_relations\n```python\ndef get_relations(self: Any, relation_type: str) -> List[Relation]\n```\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n##### from_dict\n```python\ndef from_dict(cls: Any, data: Dict[Unknown]) -> Any\n```\n\n---\n\n### Functions\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### from_dict\n\n```python\ndef from_dict(cls: Any, data: Dict[Unknown]) -> Any\n```\n\n从字典创建关系实例\n\n---\n\n#### relations\n\n```python\ndef relations(self: Any) -> List[Relation]\n```\n\n---\n\n#### add_relation\n\n```python\ndef add_relation(self: Any, relation: Relation, is_explicit: bool)\n```\n\n---\n\n#### get_relations\n\n```python\ndef get_relations(self: Any, relation_type: str) -> List[Relation]\n```\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n---\n\n#### from_dict\n\n```python\ndef from_dict(cls: Any, data: Dict[Unknown]) -> Any\n```\n\n---\n\n\n## models.secure_components\n\nCOT-DIR 安全组件

提供安全的数学计算、文件操作等功能，替代不安全的操作。\n\n### Classes\n\n#### SecurityError\n\n安全异常\n\n**Inherits from:** Exception\n\n---\n\n#### SecureMathEvaluator\n\n安全的数学表达式计算器\n\n**Methods:**\n\n##### safe_eval\n```python\ndef safe_eval(self: Any, expression: str, allowed_names: Dict[Unknown]) -> Union[Unknown]\n```\n\n安全地计算数学表达式\n\n---\n\n### Functions\n\n#### safe_eval\n\n```python\ndef safe_eval(self: Any, expression: str, allowed_names: Dict[Unknown]) -> Union[Unknown]\n```\n\n安全地计算数学表达式\n\n---\n\n\n## models.simple_pattern_model\n\nSimple Pattern-Based Model

This model uses the simple, regex-based pattern solver from the reasoning_engine.\n\n### Classes\n\n#### SimplePatternModel\n\nA simple model that wraps the PatternBasedSolver.\n\n**Inherits from:** BaseModel\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitializes the pattern-based solver.\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolves a single problem using the PatternBasedSolver.\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolves a batch of problems.\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\nInitializes the pattern-based solver.\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem_input: ModelInput) -> ModelOutput\n```\n\nSolves a single problem using the PatternBasedSolver.\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[ModelInput]) -> List[ModelOutput]\n```\n\nSolves a batch of problems.\n\n---\n\n\n## models.structures\n\n数据结构定义模块

This module contains all the data structure definitions used in the project,
including text processing, relation extraction, equation solving and inference models.\n\n### Classes\n\n#### MatchedModel\n\n**Inherits from:** BaseModel\n\n---\n\n#### RelationEntity\n\n**Inherits from:** BaseModel\n\n---\n\n#### Relations\n\n**Inherits from:** BaseModel\n\n**Methods:**\n\n##### explicit\n```python\ndef explicit(self: Any)\n```\n\n##### implicit\n```python\ndef implicit(self: Any)\n```\n\n---\n\n#### ProcessedTextData\n\n处理后的文本结构

属性:
    raw_text: 原始文本
    segmentation: 分词结果
    pos_tags: 词性标注
    dependencies: 依存关系\n\n---\n\n#### Relations\n\n关系提取结果的集合\n\n**Inherits from:** BaseModel\n\n**Methods:**\n\n##### explicit\n```python\ndef explicit(self: Any) -> List[RelationEntity]\n```\n\n获取显式关系列表\n\n##### implicit\n```python\ndef implicit(self: Any) -> List[RelationEntity]\n```\n\n获取隐式关系列表\n\n---\n\n#### ExtractionResult\n\n提取结果类

属性:
    status: 提取状态
    explicit_relations: 显式关系列表
    implicit_relations: 隐式关系列表
    message: 错误信息（如果有）\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典

Returns:
    Dict[str, Any]: 字典形式的提取结果\n\n---\n\n#### RelationCollection\n\n关系集合，用于存储和组织提取的关系\n\n**Inherits from:** BaseModel\n\n---\n\n#### Context\n\n方程式上下文信息\n\n**Inherits from:** BaseModel\n\n---\n\n#### Equation\n\n数学方程式表示\n\n**Inherits from:** BaseModel\n\n---\n\n#### Equations\n\n方程组系统\n\n**Inherits from:** BaseModel\n\n---\n\n#### Solution\n\n问题求解结果\n\n**Inherits from:** BaseModel\n\n---\n\n#### Entities\n\n实体关系\n\n**Inherits from:** BaseModel\n\n---\n\n#### Attributes\n\n属性信息\n\n**Inherits from:** BaseModel\n\n---\n\n#### InferenceResult\n\n推理结果\n\n**Inherits from:** BaseModel\n\n---\n\n#### InferenceStep\n\n推理步骤\n\n**Inherits from:** BaseModel\n\n---\n\n#### ProblemStructure\n\n问题整体结构\n\n**Inherits from:** BaseModel\n\n---\n\n#### FeatureSet\n\n问题特征集

属性:
    math_complexity: 数学复杂度特征
    linguistic_structure: 语言结构特征
    relation_type: 关系类型特征
    domain_indicators: 问题领域特征
    question_target: 问题目标特征\n\n---\n\n#### PatternMatch\n\n模式匹配结果

属性:
    pattern_id: 模式ID
    matched_text: 匹配到的文本
    score: 匹配分数
    variables: 提取的变量\n\n---\n\n### Functions\n\n#### explicit\n\n```python\ndef explicit(self: Any)\n```\n\n---\n\n#### implicit\n\n```python\ndef implicit(self: Any)\n```\n\n---\n\n#### explicit\n\n```python\ndef explicit(self: Any) -> List[RelationEntity]\n```\n\n获取显式关系列表\n\n---\n\n#### implicit\n\n```python\ndef implicit(self: Any) -> List[RelationEntity]\n```\n\n获取隐式关系列表\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典

Returns:
    Dict[str, Any]: 字典形式的提取结果\n\n---\n\n\n## monitoring.performance_monitor\n\n性能监控系统
提供系统性能指标收集和监控功能\n\n### Classes\n\n#### PerformanceMetric\n\n性能指标数据类\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n---\n\n#### PerformanceMonitor\n\n性能监控器\n\n**Methods:**\n\n##### start_timer\n```python\ndef start_timer(self: Any, name: str, tags: Dict[Unknown]) -> str\n```\n\n开始计时器\n\n##### stop_timer\n```python\ndef stop_timer(self: Any, timer_id: str) -> Optional[float]\n```\n\n停止计时器并记录性能指标\n\n##### record_metric\n```python\ndef record_metric(self: Any, metric: PerformanceMetric)\n```\n\n记录性能指标\n\n##### increment_counter\n```python\ndef increment_counter(self: Any, name: str, value: int, tags: Dict[Unknown])\n```\n\n增加计数器\n\n##### set_gauge\n```python\ndef set_gauge(self: Any, name: str, value: float, tags: Dict[Unknown])\n```\n\n设置仪表值\n\n##### get_metrics_summary\n```python\ndef get_metrics_summary(self: Any, time_window: timedelta) -> Dict[Unknown]\n```\n\n获取性能指标摘要\n\n##### start_system_monitoring\n```python\ndef start_system_monitoring(self: Any)\n```\n\n开始系统监控线程\n\n##### stop_system_monitoring\n```python\ndef stop_system_monitoring(self: Any)\n```\n\n停止系统监控\n\n##### export_metrics\n```python\ndef export_metrics(self: Any, format_type: str) -> str\n```\n\n导出性能指标\n\n---\n\n### Functions\n\n#### monitor_performance\n\n```python\ndef monitor_performance(operation_name: str, tags: Dict[Unknown])\n```\n\n性能监控装饰器\n\n---\n\n#### timeout_monitor\n\n```python\ndef timeout_monitor(timeout_seconds: float, operation_name: str)\n```\n\n超时监控装饰器\n\n---\n\n#### get_monitor\n\n```python\ndef get_monitor() -> PerformanceMonitor\n```\n\n获取全局性能监控器实例\n\n---\n\n#### init_monitor\n\n```python\ndef init_monitor() -> PerformanceMonitor\n```\n\n初始化全局性能监控器\n\n---\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n---\n\n#### start_timer\n\n```python\ndef start_timer(self: Any, name: str, tags: Dict[Unknown]) -> str\n```\n\n开始计时器\n\n---\n\n#### stop_timer\n\n```python\ndef stop_timer(self: Any, timer_id: str) -> Optional[float]\n```\n\n停止计时器并记录性能指标\n\n---\n\n#### record_metric\n\n```python\ndef record_metric(self: Any, metric: PerformanceMetric)\n```\n\n记录性能指标\n\n---\n\n#### increment_counter\n\n```python\ndef increment_counter(self: Any, name: str, value: int, tags: Dict[Unknown])\n```\n\n增加计数器\n\n---\n\n#### set_gauge\n\n```python\ndef set_gauge(self: Any, name: str, value: float, tags: Dict[Unknown])\n```\n\n设置仪表值\n\n---\n\n#### get_metrics_summary\n\n```python\ndef get_metrics_summary(self: Any, time_window: timedelta) -> Dict[Unknown]\n```\n\n获取性能指标摘要\n\n---\n\n#### start_system_monitoring\n\n```python\ndef start_system_monitoring(self: Any)\n```\n\n开始系统监控线程\n\n---\n\n#### stop_system_monitoring\n\n```python\ndef stop_system_monitoring(self: Any)\n```\n\n停止系统监控\n\n---\n\n#### export_metrics\n\n```python\ndef export_metrics(self: Any, format_type: str) -> str\n```\n\n导出性能指标\n\n---\n\n#### decorator\n\n```python\ndef decorator(func: Any)\n```\n\n---\n\n#### decorator\n\n```python\ndef decorator(func: Any)\n```\n\n---\n\n#### wrapper\n\n```python\ndef wrapper()\n```\n\n---\n\n#### wrapper\n\n```python\ndef wrapper()\n```\n\n---\n\n\n## processors.MWP_process\n\n数学应用题粗粒度分类器模块

用于对数学应用题进行初步分类，识别问题类型和特征。\n\n### Classes\n\n#### MWPCoarseClassifier\n\n数学应用题粗粒度分类器\n\n**Methods:**\n\n##### classify\n```python\ndef classify(self: Any, processed_text: Any) -> Dict[Unknown]\n```\n\n对处理后的文本进行粗粒度分类

Args:
    processed_text: 处理后的文本对象
    
Returns:
    Dict 包含分类结果，包括：
    - pattern_categories: 匹配的模式类别列表
    - features: 问题特征字典\n\n---\n\n### Functions\n\n#### classify\n\n```python\ndef classify(self: Any, processed_text: Any) -> Dict[Unknown]\n```\n\n对处理后的文本进行粗粒度分类

Args:
    processed_text: 处理后的文本对象
    
Returns:
    Dict 包含分类结果，包括：
    - pattern_categories: 匹配的模式类别列表
    - features: 问题特征字典\n\n---\n\n\n## processors.batch_processor\n\n### Classes\n\n#### ProcessingStatus\n\n处理状态\n\n**Inherits from:** Enum\n\n---\n\n#### QualityLevel\n\n质量等级\n\n**Inherits from:** Enum\n\n---\n\n#### BatchJob\n\n批处理任务\n\n---\n\n#### QualityMetrics\n\n质量指标\n\n---\n\n#### ProcessingReport\n\n处理报告\n\n---\n\n#### BatchProcessor\n\n📊 批量处理器\n\n**Methods:**\n\n##### submit_job\n```python\ndef submit_job(self: Any, name: str, input_data: List[Any], processor_func: Callable, processor_config: Optional[Dict], quality_evaluator: Optional[str]) -> str\n```\n\n📤 提交批处理任务

Args:
    name: 任务名称
    input_data: 输入数据
    processor_func: 处理函数
    processor_config: 处理器配置
    quality_evaluator: 质量评估器名称
    
Returns:
    任务ID\n\n##### process_job\n```python\ndef process_job(self: Any, job_id: str) -> ProcessingReport\n```\n\n🔄 同步处理任务\n\n##### get_job_status\n```python\ndef get_job_status(self: Any, job_id: str) -> Dict[Unknown]\n```\n\n📋 获取任务状态\n\n##### get_performance_dashboard\n```python\ndef get_performance_dashboard(self: Any) -> Dict[Unknown]\n```\n\n📊 获取性能仪表板\n\n##### export_report\n```python\ndef export_report(self: Any, job_id: str, output_path: str)\n```\n\n💾 导出报告\n\n##### cleanup_completed_jobs\n```python\ndef cleanup_completed_jobs(self: Any, keep_recent: int)\n```\n\n🧹 清理已完成的任务\n\n---\n\n### Functions\n\n#### demo_batch_processor\n\n```python\ndef demo_batch_processor()\n```\n\n演示批量处理器\n\n---\n\n#### submit_job\n\n```python\ndef submit_job(self: Any, name: str, input_data: List[Any], processor_func: Callable, processor_config: Optional[Dict], quality_evaluator: Optional[str]) -> str\n```\n\n📤 提交批处理任务

Args:
    name: 任务名称
    input_data: 输入数据
    processor_func: 处理函数
    processor_config: 处理器配置
    quality_evaluator: 质量评估器名称
    
Returns:
    任务ID\n\n---\n\n#### process_job\n\n```python\ndef process_job(self: Any, job_id: str) -> ProcessingReport\n```\n\n🔄 同步处理任务\n\n---\n\n#### get_job_status\n\n```python\ndef get_job_status(self: Any, job_id: str) -> Dict[Unknown]\n```\n\n📋 获取任务状态\n\n---\n\n#### get_performance_dashboard\n\n```python\ndef get_performance_dashboard(self: Any) -> Dict[Unknown]\n```\n\n📊 获取性能仪表板\n\n---\n\n#### export_report\n\n```python\ndef export_report(self: Any, job_id: str, output_path: str)\n```\n\n💾 导出报告\n\n---\n\n#### cleanup_completed_jobs\n\n```python\ndef cleanup_completed_jobs(self: Any, keep_recent: int)\n```\n\n🧹 清理已完成的任务\n\n---\n\n#### simple_math_processor\n\n```python\ndef simple_math_processor(problem: Any)\n```\n\n简单数学处理器示例\n\n---\n\n\n## processors.complexity_classifier\n\nComplexity Classifier Module
===========================

This module provides functionality to classify mathematical problem complexity
and calculate DIR (Depth of Implicit Reasoning) scores.

The classifier categorizes problems into four levels:
- L0: Explicit problems (no implicit reasoning required)
- L1: Shallow implicit problems (minimal inference needed)
- L2: Medium implicit problems (moderate inference required)
- L3: Deep implicit problems (complex reasoning chains)

Author: Math Problem Solver Team
Version: 1.0.0\n\n### Classes\n\n#### ComplexityClassifier\n\n数学问题复杂度分类器

根据推理深度(δ)和知识依赖(κ)对数学问题进行复杂度分级：
- L0: 显式问题 (δ=0, κ=0)
- L1: 浅层隐式 (δ=1, κ≤1)
- L2: 中等隐式 (1<δ≤3, κ≤2)
- L3: 深度隐式 (δ>3 或 κ>2)\n\n**Methods:**\n\n##### load_implicit_patterns\n```python\ndef load_implicit_patterns(self: Any) -> Dict[Unknown]\n```\n\n加载隐式关系模式

Returns:
    Dict: 隐式关系模式字典\n\n##### load_domain_knowledge\n```python\ndef load_domain_knowledge(self: Any) -> Dict[Unknown]\n```\n\n加载领域知识库

Returns:
    Dict: 领域知识字典\n\n##### classify_problem_complexity\n```python\ndef classify_problem_complexity(self: Any, problem_text: str, solution_steps: Optional[Unknown]) -> str\n```\n\n分类问题复杂度 (L0-L3)

Args:
    problem_text: 问题文本
    solution_steps: 解题步骤（可选）
    
Returns:
    str: 复杂度级别 ("L0", "L1", "L2", "L3")\n\n##### calculate_inference_depth\n```python\ndef calculate_inference_depth(self: Any, problem_text: str, solution_steps: Optional[Unknown]) -> int\n```\n\n计算推理深度 δ

Args:
    problem_text: 问题文本
    solution_steps: 解题步骤
    
Returns:
    int: 推理深度\n\n##### calculate_knowledge_dependency\n```python\ndef calculate_knowledge_dependency(self: Any, problem_text: str) -> int\n```\n\n计算知识依赖 κ

Args:
    problem_text: 问题文本
    
Returns:
    int: 知识依赖级别\n\n##### identify_implicit_relations\n```python\ndef identify_implicit_relations(self: Any, problem_text: str) -> List[Unknown]\n```\n\n识别隐式关系

Args:
    problem_text: 问题文本
    
Returns:
    List[Dict]: 识别到的隐式关系列表\n\n##### calculate_dir_score\n```python\ndef calculate_dir_score(self: Any, dataset: List[Unknown]) -> Tuple[Unknown]\n```\n\n计算数据集的DIR分数

Args:
    dataset: 数据集
    
Returns:
    Tuple[float, Dict]: DIR分数和各级别统计\n\n##### analyze_dataset_complexity\n```python\ndef analyze_dataset_complexity(self: Any, dataset: List[Unknown]) -> Dict[Unknown]\n```\n\n分析数据集复杂度分布

Args:
    dataset: 数据集
    
Returns:
    Dict: 复杂度分析结果\n\n##### batch_classify_problems\n```python\ndef batch_classify_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量分类问题复杂度

Args:
    problems: 问题列表
    
Returns:
    List[Dict]: 带有复杂度标签的问题列表\n\n##### export_complexity_analysis\n```python\ndef export_complexity_analysis(self: Any, analysis: Dict[Unknown], output_path: str) -> Any\n```\n\n导出复杂度分析结果

Args:
    analysis: 分析结果
    output_path: 输出文件路径\n\n---\n\n### Functions\n\n#### load_implicit_patterns\n\n```python\ndef load_implicit_patterns(self: Any) -> Dict[Unknown]\n```\n\n加载隐式关系模式

Returns:
    Dict: 隐式关系模式字典\n\n---\n\n#### load_domain_knowledge\n\n```python\ndef load_domain_knowledge(self: Any) -> Dict[Unknown]\n```\n\n加载领域知识库

Returns:
    Dict: 领域知识字典\n\n---\n\n#### classify_problem_complexity\n\n```python\ndef classify_problem_complexity(self: Any, problem_text: str, solution_steps: Optional[Unknown]) -> str\n```\n\n分类问题复杂度 (L0-L3)

Args:
    problem_text: 问题文本
    solution_steps: 解题步骤（可选）
    
Returns:
    str: 复杂度级别 ("L0", "L1", "L2", "L3")\n\n---\n\n#### calculate_inference_depth\n\n```python\ndef calculate_inference_depth(self: Any, problem_text: str, solution_steps: Optional[Unknown]) -> int\n```\n\n计算推理深度 δ

Args:
    problem_text: 问题文本
    solution_steps: 解题步骤
    
Returns:
    int: 推理深度\n\n---\n\n#### calculate_knowledge_dependency\n\n```python\ndef calculate_knowledge_dependency(self: Any, problem_text: str) -> int\n```\n\n计算知识依赖 κ

Args:
    problem_text: 问题文本
    
Returns:
    int: 知识依赖级别\n\n---\n\n#### identify_implicit_relations\n\n```python\ndef identify_implicit_relations(self: Any, problem_text: str) -> List[Unknown]\n```\n\n识别隐式关系

Args:
    problem_text: 问题文本
    
Returns:
    List[Dict]: 识别到的隐式关系列表\n\n---\n\n#### calculate_dir_score\n\n```python\ndef calculate_dir_score(self: Any, dataset: List[Unknown]) -> Tuple[Unknown]\n```\n\n计算数据集的DIR分数

Args:
    dataset: 数据集
    
Returns:
    Tuple[float, Dict]: DIR分数和各级别统计\n\n---\n\n#### analyze_dataset_complexity\n\n```python\ndef analyze_dataset_complexity(self: Any, dataset: List[Unknown]) -> Dict[Unknown]\n```\n\n分析数据集复杂度分布

Args:
    dataset: 数据集
    
Returns:
    Dict: 复杂度分析结果\n\n---\n\n#### batch_classify_problems\n\n```python\ndef batch_classify_problems(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量分类问题复杂度

Args:
    problems: 问题列表
    
Returns:
    List[Dict]: 带有复杂度标签的问题列表\n\n---\n\n#### export_complexity_analysis\n\n```python\ndef export_complexity_analysis(self: Any, analysis: Dict[Unknown], output_path: str) -> Any\n```\n\n导出复杂度分析结果

Args:
    analysis: 分析结果
    output_path: 输出文件路径\n\n---\n\n\n## processors.dataset_loader\n\nDataset Loader Module
====================

This module provides functionality to load various mathematical problem datasets
including Math23K, GSM8K, MAWPS, MathQA, MATH, SVAMP, and ASDiv.

Author: Math Problem Solver Team
Version: 1.0.0\n\n### Classes\n\n#### DatasetLoader\n\n数据集加载器，支持加载多种数学题数据集

支持的数据集:
- Math23K: 中文数学题数据集 (23,162题)
- GSM8K: 英文小学数学题 (8,500题)
- MAWPS: 多领域数学题 (2,373题)
- MathQA: 竞赛数学题 (37,297题)
- MATH: 竞赛数学题 (12,500题)
- SVAMP: 小学数学题 (1,000题)
- ASDiv: 小学数学题 (2,305题)\n\n**Methods:**\n\n##### load_math23k\n```python\ndef load_math23k(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载Math23K中文数学题数据集 (23,162题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 包含问题、方程和答案的字典列表
    数据格式: {"question": "...", "equation": "...", "answer": "..."}\n\n##### load_gsm8k\n```python\ndef load_gsm8k(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载GSM8K英文小学数学题 (8,500题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n##### load_mawps\n```python\ndef load_mawps(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载MAWPS多领域数学题 (2,373题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n##### load_mathqa\n```python\ndef load_mathqa(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载MathQA竞赛数学题 (37,297题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n##### load_math_dataset\n```python\ndef load_math_dataset(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载MATH竞赛数学题 (12,500题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n##### load_svamp\n```python\ndef load_svamp(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载SVAMP小学数学题 (1,000题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n##### load_asdiv\n```python\ndef load_asdiv(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载ASDiv小学数学题 (2,305题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n##### get_dataset\n```python\ndef get_dataset(self: Any, dataset_name: str) -> Optional[Unknown]\n```\n\n获取已加载的数据集

Args:
    dataset_name: 数据集名称
    
Returns:
    数据集数据或None\n\n##### get_all_datasets\n```python\ndef get_all_datasets(self: Any) -> Dict[Unknown]\n```\n\n获取所有已加载的数据集

Returns:
    所有数据集的字典\n\n##### get_dataset_stats\n```python\ndef get_dataset_stats(self: Any) -> Dict[Unknown]\n```\n\n获取数据集统计信息

Returns:
    每个数据集的统计信息\n\n---\n\n### Functions\n\n#### load_math23k\n\n```python\ndef load_math23k(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载Math23K中文数学题数据集 (23,162题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 包含问题、方程和答案的字典列表
    数据格式: {"question": "...", "equation": "...", "answer": "..."}\n\n---\n\n#### load_gsm8k\n\n```python\ndef load_gsm8k(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载GSM8K英文小学数学题 (8,500题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n---\n\n#### load_mawps\n\n```python\ndef load_mawps(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载MAWPS多领域数学题 (2,373题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n---\n\n#### load_mathqa\n\n```python\ndef load_mathqa(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载MathQA竞赛数学题 (37,297题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n---\n\n#### load_math_dataset\n\n```python\ndef load_math_dataset(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载MATH竞赛数学题 (12,500题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n---\n\n#### load_svamp\n\n```python\ndef load_svamp(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载SVAMP小学数学题 (1,000题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n---\n\n#### load_asdiv\n\n```python\ndef load_asdiv(self: Any, file_path: str) -> List[Unknown]\n```\n\n加载ASDiv小学数学题 (2,305题)

Args:
    file_path: 数据集文件路径
    
Returns:
    List[Dict]: 标准化的问题数据\n\n---\n\n#### get_dataset\n\n```python\ndef get_dataset(self: Any, dataset_name: str) -> Optional[Unknown]\n```\n\n获取已加载的数据集

Args:
    dataset_name: 数据集名称
    
Returns:
    数据集数据或None\n\n---\n\n#### get_all_datasets\n\n```python\ndef get_all_datasets(self: Any) -> Dict[Unknown]\n```\n\n获取所有已加载的数据集

Returns:
    所有数据集的字典\n\n---\n\n#### get_dataset_stats\n\n```python\ndef get_dataset_stats(self: Any) -> Dict[Unknown]\n```\n\n获取数据集统计信息

Returns:
    每个数据集的统计信息\n\n---\n\n\n## processors.dynamic_dataset_manager\n\n🚀 Dynamic Dataset Manager - 零代码添加新题目
动态从数据集加载，支持自动发现和热加载\n\n### Classes\n\n#### DatasetMetadata\n\n数据集元数据\n\n---\n\n#### ProblemBatch\n\n问题批次\n\n---\n\n#### DynamicDatasetManager\n\n🚀 零代码动态数据集管理器\n\n**Methods:**\n\n##### discover_datasets\n```python\ndef discover_datasets(self: Any) -> int\n```\n\n🔍 自动发现数据集\n\n##### load_dataset\n```python\ndef load_dataset(self: Any, dataset_name: str, max_samples: Optional[int], complexity_filter: Optional[str]) -> List[Dict]\n```\n\n🔄 加载数据集（支持缓存和过滤）\n\n##### get_dynamic_batch\n```python\ndef get_dynamic_batch(self: Any, batch_size: int, datasets: Optional[Unknown], complexity_mix: Optional[Unknown]) -> ProblemBatch\n```\n\n📦 获取动态问题批次\n\n##### start_watching\n```python\ndef start_watching(self: Any)\n```\n\n🔍 启动文件监控\n\n##### stop_watching\n```python\ndef stop_watching(self: Any)\n```\n\n⏹️ 停止文件监控\n\n##### add_change_callback\n```python\ndef add_change_callback(self: Any, callback: Callable)\n```\n\n添加变更回调\n\n##### get_stats\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n📊 获取统计信息\n\n##### list_datasets\n```python\ndef list_datasets(self: Any, detailed: bool) -> List[Unknown]\n```\n\n📋 列出所有数据集\n\n##### export_config\n```python\ndef export_config(self: Any, config_path: str)\n```\n\n💾 导出配置\n\n---\n\n### Functions\n\n#### demo_dynamic_dataset_manager\n\n```python\ndef demo_dynamic_dataset_manager()\n```\n\n演示动态数据集管理器\n\n---\n\n#### discover_datasets\n\n```python\ndef discover_datasets(self: Any) -> int\n```\n\n🔍 自动发现数据集\n\n---\n\n#### load_dataset\n\n```python\ndef load_dataset(self: Any, dataset_name: str, max_samples: Optional[int], complexity_filter: Optional[str]) -> List[Dict]\n```\n\n🔄 加载数据集（支持缓存和过滤）\n\n---\n\n#### get_dynamic_batch\n\n```python\ndef get_dynamic_batch(self: Any, batch_size: int, datasets: Optional[Unknown], complexity_mix: Optional[Unknown]) -> ProblemBatch\n```\n\n📦 获取动态问题批次\n\n---\n\n#### start_watching\n\n```python\ndef start_watching(self: Any)\n```\n\n🔍 启动文件监控\n\n---\n\n#### stop_watching\n\n```python\ndef stop_watching(self: Any)\n```\n\n⏹️ 停止文件监控\n\n---\n\n#### add_change_callback\n\n```python\ndef add_change_callback(self: Any, callback: Callable)\n```\n\n添加变更回调\n\n---\n\n#### get_stats\n\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n📊 获取统计信息\n\n---\n\n#### list_datasets\n\n```python\ndef list_datasets(self: Any, detailed: bool) -> List[Unknown]\n```\n\n📋 列出所有数据集\n\n---\n\n#### export_config\n\n```python\ndef export_config(self: Any, config_path: str)\n```\n\n💾 导出配置\n\n---\n\n\n## processors.equation_builder\n\n### Classes\n\n#### EquationBuilder\n\n**Methods:**\n\n##### solve_equation_system\n```python\ndef solve_equation_system(self: Any, equations: Any, target_vars: Any, values_and_units: Any)\n```\n\n自动单位换算、优先求解目标变量、支持多目标/多步递推\n\n##### build_equations\n```python\ndef build_equations(self: Any, extraction_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n根据关系提取结果构建方程组
Args:
    extraction_result: 关系提取结果
Returns:
    Dict 包含方程组和变量信息\n\n---\n\n### Functions\n\n#### solve_equation_system\n\n```python\ndef solve_equation_system(self: Any, equations: Any, target_vars: Any, values_and_units: Any)\n```\n\n自动单位换算、优先求解目标变量、支持多目标/多步递推\n\n---\n\n#### build_equations\n\n```python\ndef build_equations(self: Any, extraction_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n根据关系提取结果构建方程组
Args:
    extraction_result: 关系提取结果
Returns:
    Dict 包含方程组和变量信息\n\n---\n\n#### get_equation_score\n\n```python\ndef get_equation_score(eq_data: Any)\n```\n\n---\n\n\n## processors.implicit_relation_annotator\n\nImplicit Relation Annotator Module
==================================

This module provides functionality to annotate implicit relations in mathematical problems.
It identifies and categorizes different types of implicit relationships that are not
explicitly stated in the problem text but are necessary for solving the problem.

Relation Types:
- Mathematical Operations (35.2%)
- Unit Conversions (18.7%)
- Physical Constraints (16.4%)
- Temporal Relations (12.3%)
- Geometric Properties (10.8%)
- Proportional Relations (6.6%)

Author: Math Problem Solver Team
Version: 1.0.0\n\n### Classes\n\n#### ImplicitRelationAnnotator\n\n隐式关系标注器

用于识别和标注数学问题中的隐式关系，包括：
- 数学运算关系 (35.2%)
- 单位转换关系 (18.7%)
- 物理约束关系 (16.4%)
- 时间关系 (12.3%)
- 几何属性关系 (10.8%)
- 比例关系 (6.6%)\n\n**Methods:**\n\n##### annotate_implicit_relations\n```python\ndef annotate_implicit_relations(self: Any, problem_text: str) -> List[Unknown]\n```\n\n标注问题中的隐式关系

Args:
    problem_text: 问题文本
    
Returns:
    List[Dict]: 标注的隐式关系列表\n\n##### extract_mathematical_operations\n```python\ndef extract_mathematical_operations(self: Any, text: str) -> List[Unknown]\n```\n\n提取数学运算关系\n\n##### extract_unit_conversions\n```python\ndef extract_unit_conversions(self: Any, text: str) -> List[Unknown]\n```\n\n提取单位转换关系\n\n##### extract_physical_constraints\n```python\ndef extract_physical_constraints(self: Any, text: str) -> List[Unknown]\n```\n\n提取物理约束关系\n\n##### extract_temporal_relations\n```python\ndef extract_temporal_relations(self: Any, text: str) -> List[Unknown]\n```\n\n提取时间关系

Args:
    text: 文本
    
Returns:
    List[Dict]: 时间关系列表\n\n##### extract_geometric_properties\n```python\ndef extract_geometric_properties(self: Any, text: str) -> List[Unknown]\n```\n\n提取几何属性关系

Args:
    text: 文本
    
Returns:
    List[Dict]: 几何属性关系列表\n\n##### extract_proportional_relations\n```python\ndef extract_proportional_relations(self: Any, text: str) -> List[Unknown]\n```\n\n提取比例关系

Args:
    text: 文本
    
Returns:
    List[Dict]: 比例关系列表\n\n##### create_ground_truth_relations\n```python\ndef create_ground_truth_relations(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n为问题创建隐式关系的真值标注

Args:
    problems: 问题列表
    
Returns:
    List[Dict]: 带有隐式关系真值标注的问题列表\n\n##### analyze_relation_distribution\n```python\ndef analyze_relation_distribution(self: Any, annotated_problems: List[Unknown]) -> Dict[Unknown]\n```\n\n分析隐式关系分布

Args:
    annotated_problems: 标注的问题列表
    
Returns:
    Dict: 关系分布分析结果\n\n##### validate_annotations\n```python\ndef validate_annotations(self: Any, annotated_problems: List[Unknown], sample_size: int) -> Dict[Unknown]\n```\n\n验证标注质量

Args:
    annotated_problems: 标注的问题列表
    sample_size: 验证样本大小
    
Returns:
    Dict: 验证结果\n\n##### export_annotations\n```python\ndef export_annotations(self: Any, annotated_problems: List[Unknown], output_path: str, include_analysis: bool) -> Any\n```\n\n导出标注结果

Args:
    annotated_problems: 标注的问题列表
    output_path: 输出文件路径
    include_analysis: 是否包含分析结果\n\n---\n\n### Functions\n\n#### annotate_implicit_relations\n\n```python\ndef annotate_implicit_relations(self: Any, problem_text: str) -> List[Unknown]\n```\n\n标注问题中的隐式关系

Args:
    problem_text: 问题文本
    
Returns:
    List[Dict]: 标注的隐式关系列表\n\n---\n\n#### extract_mathematical_operations\n\n```python\ndef extract_mathematical_operations(self: Any, text: str) -> List[Unknown]\n```\n\n提取数学运算关系\n\n---\n\n#### extract_unit_conversions\n\n```python\ndef extract_unit_conversions(self: Any, text: str) -> List[Unknown]\n```\n\n提取单位转换关系\n\n---\n\n#### extract_physical_constraints\n\n```python\ndef extract_physical_constraints(self: Any, text: str) -> List[Unknown]\n```\n\n提取物理约束关系\n\n---\n\n#### extract_temporal_relations\n\n```python\ndef extract_temporal_relations(self: Any, text: str) -> List[Unknown]\n```\n\n提取时间关系

Args:
    text: 文本
    
Returns:
    List[Dict]: 时间关系列表\n\n---\n\n#### extract_geometric_properties\n\n```python\ndef extract_geometric_properties(self: Any, text: str) -> List[Unknown]\n```\n\n提取几何属性关系

Args:
    text: 文本
    
Returns:
    List[Dict]: 几何属性关系列表\n\n---\n\n#### extract_proportional_relations\n\n```python\ndef extract_proportional_relations(self: Any, text: str) -> List[Unknown]\n```\n\n提取比例关系

Args:
    text: 文本
    
Returns:
    List[Dict]: 比例关系列表\n\n---\n\n#### create_ground_truth_relations\n\n```python\ndef create_ground_truth_relations(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n为问题创建隐式关系的真值标注

Args:
    problems: 问题列表
    
Returns:
    List[Dict]: 带有隐式关系真值标注的问题列表\n\n---\n\n#### analyze_relation_distribution\n\n```python\ndef analyze_relation_distribution(self: Any, annotated_problems: List[Unknown]) -> Dict[Unknown]\n```\n\n分析隐式关系分布

Args:
    annotated_problems: 标注的问题列表
    
Returns:
    Dict: 关系分布分析结果\n\n---\n\n#### validate_annotations\n\n```python\ndef validate_annotations(self: Any, annotated_problems: List[Unknown], sample_size: int) -> Dict[Unknown]\n```\n\n验证标注质量

Args:
    annotated_problems: 标注的问题列表
    sample_size: 验证样本大小
    
Returns:
    Dict: 验证结果\n\n---\n\n#### export_annotations\n\n```python\ndef export_annotations(self: Any, annotated_problems: List[Unknown], output_path: str, include_analysis: bool) -> Any\n```\n\n导出标注结果

Args:
    annotated_problems: 标注的问题列表
    output_path: 输出文件路径
    include_analysis: 是否包含分析结果\n\n---\n\n\n## processors.inference_tracker\n\n### Classes\n\n#### InferenceTracker\n\n推理过程跟踪器，记录每一步推理的输入输出和历史\n\n**Methods:**\n\n##### start_tracking\n```python\ndef start_tracking(self: Any)\n```\n\n开始跟踪推理过程\n\n##### add_inference\n```python\ndef add_inference(self: Any, step_name: str, input_data: Any, output_data: Any)\n```\n\n添加一步推理记录

Args:
    step_name: 步骤名称
    input_data: 输入数据
    output_data: 输出数据\n\n##### get_inference_history\n```python\ndef get_inference_history(self: Any) -> List[Unknown]\n```\n\n获取推理历史记录

Returns:
    List[Dict]: 推理历史记录列表\n\n##### get_inference_summary\n```python\ndef get_inference_summary(self: Any) -> str\n```\n\n获取推理摘要

Returns:
    str: 推理摘要文本\n\n##### end_tracking\n```python\ndef end_tracking(self: Any)\n```\n\n结束推理跟踪\n\n##### export_history\n```python\ndef export_history(self: Any, filepath: str)\n```\n\n导出推理历史到文件

Args:
    filepath: 导出文件路径\n\n---\n\n### Functions\n\n#### start_tracking\n\n```python\ndef start_tracking(self: Any)\n```\n\n开始跟踪推理过程\n\n---\n\n#### add_inference\n\n```python\ndef add_inference(self: Any, step_name: str, input_data: Any, output_data: Any)\n```\n\n添加一步推理记录

Args:
    step_name: 步骤名称
    input_data: 输入数据
    output_data: 输出数据\n\n---\n\n#### get_inference_history\n\n```python\ndef get_inference_history(self: Any) -> List[Unknown]\n```\n\n获取推理历史记录

Returns:
    List[Dict]: 推理历史记录列表\n\n---\n\n#### get_inference_summary\n\n```python\ndef get_inference_summary(self: Any) -> str\n```\n\n获取推理摘要

Returns:
    str: 推理摘要文本\n\n---\n\n#### end_tracking\n\n```python\ndef end_tracking(self: Any)\n```\n\n结束推理跟踪\n\n---\n\n#### export_history\n\n```python\ndef export_history(self: Any, filepath: str)\n```\n\n导出推理历史到文件

Args:
    filepath: 导出文件路径\n\n---\n\n\n## processors.intelligent_classifier\n\n🧠 Intelligent Problem Classifier - 智能分类和模板匹配
10种题型自动识别，智能模板匹配系统\n\n### Classes\n\n#### ProblemType\n\n数学问题类型枚举\n\n**Inherits from:** Enum\n\n---\n\n#### ProblemPattern\n\n问题模式\n\n---\n\n#### ClassificationResult\n\n分类结果\n\n---\n\n#### IntelligentClassifier\n\n🧠 智能问题分类器\n\n**Methods:**\n\n##### classify\n```python\ndef classify(self: Any, problem_text: str) -> ClassificationResult\n```\n\n🎯 对问题进行智能分类

Args:
    problem_text: 问题文本
    
Returns:
    分类结果\n\n##### batch_classify\n```python\ndef batch_classify(self: Any, problems: List[str]) -> List[ClassificationResult]\n```\n\n📦 批量分类\n\n##### get_statistics\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n📊 获取分类统计\n\n##### save_model\n```python\ndef save_model(self: Any, model_path: str)\n```\n\n💾 保存分类模型\n\n##### add_pattern\n```python\ndef add_pattern(self: Any, pattern: ProblemPattern)\n```\n\n➕ 添加新模式\n\n##### analyze_classification_accuracy\n```python\ndef analyze_classification_accuracy(self: Any, test_data: List[Unknown]) -> Dict[Unknown]\n```\n\n🎯 分析分类准确度\n\n---\n\n### Functions\n\n#### demo_intelligent_classifier\n\n```python\ndef demo_intelligent_classifier()\n```\n\n演示智能分类器\n\n---\n\n#### classify\n\n```python\ndef classify(self: Any, problem_text: str) -> ClassificationResult\n```\n\n🎯 对问题进行智能分类

Args:
    problem_text: 问题文本
    
Returns:
    分类结果\n\n---\n\n#### batch_classify\n\n```python\ndef batch_classify(self: Any, problems: List[str]) -> List[ClassificationResult]\n```\n\n📦 批量分类\n\n---\n\n#### get_statistics\n\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n📊 获取分类统计\n\n---\n\n#### save_model\n\n```python\ndef save_model(self: Any, model_path: str)\n```\n\n💾 保存分类模型\n\n---\n\n#### add_pattern\n\n```python\ndef add_pattern(self: Any, pattern: ProblemPattern)\n```\n\n➕ 添加新模式\n\n---\n\n#### analyze_classification_accuracy\n\n```python\ndef analyze_classification_accuracy(self: Any, test_data: List[Unknown]) -> Dict[Unknown]\n```\n\n🎯 分析分类准确度\n\n---\n\n\n## processors.nlp_processor\n\n自然语言处理器
~~~~~~~~~~

这个模块负责对输入文本进行自然语言处理，包括分词、词性标注等。

Author: [Your Name]
Date: [Current Date]\n\n### Classes\n\n#### NLPProcessor\n\n自然语言处理器\n\n**Methods:**\n\n##### analyze\n```python\ndef analyze(self: Any, problem_text: str) -> Dict\n```\n\n分析问题文本，提取关键信息和问题类型\n\n##### process_text\n```python\ndef process_text(self: Any, text: str) -> Dict\n```\n\n处理文本

Args:
    text: 输入文本
    
Returns:
    ProcessedText: 处理后的文本结构
    
Raises:
    ValueError: 当输入文本为空或None时\n\n##### load_examples_from_json\n```python\ndef load_examples_from_json(self: Any, json_path: str)\n```\n\n批量读取examples/problems.json中的问题并NLP处理，返回ProcessedText对象列表\n\n##### save_processed_examples_to_file\n```python\ndef save_processed_examples_to_file(self: Any, output_path: str, json_path: str)\n```\n\n批量处理examples/problems.json中的问题，并将所有ProcessedText对象的结构化结果保存到指定文件（JSON格式）。
Args:
    output_path: 输出文件路径
    json_path: 输入的problems.json路径（可选）\n\n---\n\n### Functions\n\n#### analyze_problem\n\n```python\ndef analyze_problem(problem_text: Any)\n```\n\n---\n\n#### analyze\n\n```python\ndef analyze(self: Any, problem_text: str) -> Dict\n```\n\n分析问题文本，提取关键信息和问题类型\n\n---\n\n#### process_text\n\n```python\ndef process_text(self: Any, text: str) -> Dict\n```\n\n处理文本

Args:
    text: 输入文本
    
Returns:
    ProcessedText: 处理后的文本结构
    
Raises:
    ValueError: 当输入文本为空或None时\n\n---\n\n#### load_examples_from_json\n\n```python\ndef load_examples_from_json(self: Any, json_path: str)\n```\n\n批量读取examples/problems.json中的问题并NLP处理，返回ProcessedText对象列表\n\n---\n\n#### save_processed_examples_to_file\n\n```python\ndef save_processed_examples_to_file(self: Any, output_path: str, json_path: str)\n```\n\n批量处理examples/problems.json中的问题，并将所有ProcessedText对象的结构化结果保存到指定文件（JSON格式）。
Args:
    output_path: 输出文件路径
    json_path: 输入的problems.json路径（可选）\n\n---\n\n\n## processors.orchestrator\n\nProcessors Module - Orchestrator
===============================

处理器模块协调器：负责协调各种处理操作

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### ProcessorsOrchestrator\n\n处理器模块协调器\n\n**Methods:**\n\n##### orchestrate\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行

Args:
    operation: 操作名称
    **kwargs: 操作参数
    
Returns:
    操作结果\n\n##### register_component\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件

Args:
    name: 组件名称
    component: 组件实例\n\n##### get_component\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件

Args:
    name: 组件名称
    
Returns:
    组件实例\n\n##### get_operation_history\n```python\ndef get_operation_history(self: Any) -> List[Unknown]\n```\n\n获取操作历史\n\n##### clear_operation_history\n```python\ndef clear_operation_history(self: Any) -> Any\n```\n\n清空操作历史\n\n##### get_orchestrator_status\n```python\ndef get_orchestrator_status(self: Any) -> Dict[Unknown]\n```\n\n获取协调器状态\n\n##### initialize_orchestrator\n```python\ndef initialize_orchestrator(self: Any) -> bool\n```\n\n初始化协调器\n\n##### shutdown_orchestrator\n```python\ndef shutdown_orchestrator(self: Any) -> bool\n```\n\n关闭协调器\n\n---\n\n### Functions\n\n#### orchestrate\n\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行

Args:
    operation: 操作名称
    **kwargs: 操作参数
    
Returns:
    操作结果\n\n---\n\n#### register_component\n\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件

Args:
    name: 组件名称
    component: 组件实例\n\n---\n\n#### get_component\n\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件

Args:
    name: 组件名称
    
Returns:
    组件实例\n\n---\n\n#### get_operation_history\n\n```python\ndef get_operation_history(self: Any) -> List[Unknown]\n```\n\n获取操作历史\n\n---\n\n#### clear_operation_history\n\n```python\ndef clear_operation_history(self: Any) -> Any\n```\n\n清空操作历史\n\n---\n\n#### get_orchestrator_status\n\n```python\ndef get_orchestrator_status(self: Any) -> Dict[Unknown]\n```\n\n获取协调器状态\n\n---\n\n#### initialize_orchestrator\n\n```python\ndef initialize_orchestrator(self: Any) -> bool\n```\n\n初始化协调器\n\n---\n\n#### shutdown_orchestrator\n\n```python\ndef shutdown_orchestrator(self: Any) -> bool\n```\n\n关闭协调器\n\n---\n\n\n## processors.private.processor\n\nProcessors Module - Core Processor
=================================

核心处理器：整合各种处理功能的核心逻辑

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### CoreProcessor\n\n核心处理器：整合各种处理功能\n\n**Methods:**\n\n##### process_text\n```python\ndef process_text(self: Any, text: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理文本数据

Args:
    text: 输入文本
    config: 处理配置
    
Returns:
    处理结果\n\n##### process_dataset\n```python\ndef process_dataset(self: Any, dataset: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理数据集

Args:
    dataset: 数据集
    config: 处理配置
    
Returns:
    处理结果\n\n##### get_processing_statistics\n```python\ndef get_processing_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取处理统计信息\n\n---\n\n### Functions\n\n#### process_text\n\n```python\ndef process_text(self: Any, text: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理文本数据

Args:
    text: 输入文本
    config: 处理配置
    
Returns:
    处理结果\n\n---\n\n#### process_dataset\n\n```python\ndef process_dataset(self: Any, dataset: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n处理数据集

Args:
    dataset: 数据集
    config: 处理配置
    
Returns:
    处理结果\n\n---\n\n#### get_processing_statistics\n\n```python\ndef get_processing_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取处理统计信息\n\n---\n\n\n## processors.private.utils\n\nProcessors Module - Utility Functions
====================================

工具函数：提供各种辅助功能和工具方法

Author: AI Assistant
Date: 2024-07-13\n\n### Functions\n\n#### clean_text\n\n```python\ndef clean_text(text: str) -> str\n```\n\n清理文本数据

Args:
    text: 输入文本
    
Returns:
    清理后的文本\n\n---\n\n#### extract_numbers\n\n```python\ndef extract_numbers(text: str) -> List[float]\n```\n\n从文本中提取数字

Args:
    text: 输入文本
    
Returns:
    提取的数字列表\n\n---\n\n#### extract_mathematical_expressions\n\n```python\ndef extract_mathematical_expressions(text: str) -> List[str]\n```\n\n从文本中提取数学表达式

Args:
    text: 输入文本
    
Returns:
    数学表达式列表\n\n---\n\n#### normalize_text\n\n```python\ndef normalize_text(text: str) -> str\n```\n\n标准化文本

Args:
    text: 输入文本
    
Returns:
    标准化后的文本\n\n---\n\n#### format_processing_result\n\n```python\ndef format_processing_result(result: Dict[Unknown], include_metadata: bool) -> Dict[Unknown]\n```\n\n格式化处理结果

Args:
    result: 处理结果
    include_metadata: 是否包含元数据
    
Returns:
    格式化后的结果\n\n---\n\n#### validate_and_clean_input\n\n```python\ndef validate_and_clean_input(input_data: Any) -> Dict[Unknown]\n```\n\n验证和清理输入数据

Args:
    input_data: 输入数据
    
Returns:
    清理后的数据和验证结果\n\n---\n\n#### merge_processing_results\n\n```python\ndef merge_processing_results(results: List[Unknown]) -> Dict[Unknown]\n```\n\n合并多个处理结果

Args:
    results: 结果列表
    
Returns:
    合并后的结果\n\n---\n\n#### save_processing_results\n\n```python\ndef save_processing_results(results: Dict[Unknown], filename: str) -> bool\n```\n\n保存处理结果到文件

Args:
    results: 处理结果
    filename: 文件名
    
Returns:
    是否保存成功\n\n---\n\n#### load_processing_results\n\n```python\ndef load_processing_results(filename: str) -> Optional[Unknown]\n```\n\n从文件加载处理结果

Args:
    filename: 文件名
    
Returns:
    处理结果或None\n\n---\n\n#### get_processing_summary\n\n```python\ndef get_processing_summary(results: Dict[Unknown]) -> Dict[Unknown]\n```\n\n获取处理结果摘要

Args:
    results: 处理结果
    
Returns:
    结果摘要\n\n---\n\n\n## processors.private.validator\n\nProcessors Module - Data Validator
=================================

数据验证器：负责验证输入数据的有效性和完整性

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### ProcessorValidator\n\n处理器数据验证器\n\n**Methods:**\n\n##### validate_text_input\n```python\ndef validate_text_input(self: Any, text: Union[Unknown]) -> Dict[Unknown]\n```\n\n验证文本输入数据

Args:
    text: 输入的文本数据
    
Returns:
    验证结果，包含is_valid和error_messages\n\n##### validate_dataset_input\n```python\ndef validate_dataset_input(self: Any, dataset: Any) -> Dict[Unknown]\n```\n\n验证数据集输入

Args:
    dataset: 数据集数据
    
Returns:
    验证结果\n\n##### validate_relation_input\n```python\ndef validate_relation_input(self: Any, relations: Any) -> Dict[Unknown]\n```\n\n验证关系数据输入

Args:
    relations: 关系数据
    
Returns:
    验证结果\n\n##### validate_processing_config\n```python\ndef validate_processing_config(self: Any, config: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证处理配置

Args:
    config: 处理配置
    
Returns:
    验证结果\n\n---\n\n### Functions\n\n#### validate_text_input\n\n```python\ndef validate_text_input(self: Any, text: Union[Unknown]) -> Dict[Unknown]\n```\n\n验证文本输入数据

Args:
    text: 输入的文本数据
    
Returns:
    验证结果，包含is_valid和error_messages\n\n---\n\n#### validate_dataset_input\n\n```python\ndef validate_dataset_input(self: Any, dataset: Any) -> Dict[Unknown]\n```\n\n验证数据集输入

Args:
    dataset: 数据集数据
    
Returns:
    验证结果\n\n---\n\n#### validate_relation_input\n\n```python\ndef validate_relation_input(self: Any, relations: Any) -> Dict[Unknown]\n```\n\n验证关系数据输入

Args:
    relations: 关系数据
    
Returns:
    验证结果\n\n---\n\n#### validate_processing_config\n\n```python\ndef validate_processing_config(self: Any, config: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证处理配置

Args:
    config: 处理配置
    
Returns:
    验证结果\n\n---\n\n\n## processors.public_api\n\nProcessors Module - Public API
==============================

处理器模块公共API：提供统一的处理器接口

Author: AI Assistant
Date: 2024-07-13\n\n### Classes\n\n#### ProcessorsAPI\n\n处理器模块公共API\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化处理器模块\n\n##### process_text\n```python\ndef process_text(self: Any, text: Union[Unknown], config: Optional[Dict], validate_input: bool) -> Dict[Unknown]\n```\n\n处理文本数据的主要接口

Args:
    text: 输入文本或字典
    config: 处理配置
    validate_input: 是否验证输入
    
Returns:
    处理结果\n\n##### process_dataset\n```python\ndef process_dataset(self: Any, dataset: Union[Unknown], config: Optional[Dict], validate_input: bool) -> Dict[Unknown]\n```\n\n处理数据集的主要接口

Args:
    dataset: 数据集
    config: 处理配置
    validate_input: 是否验证输入
    
Returns:
    处理结果\n\n##### extract_relations\n```python\ndef extract_relations(self: Any, text: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n关系提取接口

Args:
    text: 输入文本
    config: 处理配置
    
Returns:
    关系提取结果\n\n##### classify_complexity\n```python\ndef classify_complexity(self: Any, text: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n复杂度分类接口

Args:
    text: 输入文本
    config: 处理配置
    
Returns:
    分类结果\n\n##### process_nlp\n```python\ndef process_nlp(self: Any, text: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\nNLP处理接口

Args:
    text: 输入文本
    config: 处理配置
    
Returns:
    NLP处理结果\n\n##### batch_process\n```python\ndef batch_process(self: Any, inputs: List[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n批量处理接口

Args:
    inputs: 输入列表
    config: 处理配置
    
Returns:
    批量处理结果\n\n##### get_module_status\n```python\ndef get_module_status(self: Any) -> Dict[Unknown]\n```\n\n获取模块状态\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭处理器模块\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化处理器模块\n\n---\n\n#### process_text\n\n```python\ndef process_text(self: Any, text: Union[Unknown], config: Optional[Dict], validate_input: bool) -> Dict[Unknown]\n```\n\n处理文本数据的主要接口

Args:
    text: 输入文本或字典
    config: 处理配置
    validate_input: 是否验证输入
    
Returns:
    处理结果\n\n---\n\n#### process_dataset\n\n```python\ndef process_dataset(self: Any, dataset: Union[Unknown], config: Optional[Dict], validate_input: bool) -> Dict[Unknown]\n```\n\n处理数据集的主要接口

Args:
    dataset: 数据集
    config: 处理配置
    validate_input: 是否验证输入
    
Returns:
    处理结果\n\n---\n\n#### extract_relations\n\n```python\ndef extract_relations(self: Any, text: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n关系提取接口

Args:
    text: 输入文本
    config: 处理配置
    
Returns:
    关系提取结果\n\n---\n\n#### classify_complexity\n\n```python\ndef classify_complexity(self: Any, text: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n复杂度分类接口

Args:
    text: 输入文本
    config: 处理配置
    
Returns:
    分类结果\n\n---\n\n#### process_nlp\n\n```python\ndef process_nlp(self: Any, text: Union[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\nNLP处理接口

Args:
    text: 输入文本
    config: 处理配置
    
Returns:
    NLP处理结果\n\n---\n\n#### batch_process\n\n```python\ndef batch_process(self: Any, inputs: List[Unknown], config: Optional[Dict]) -> Dict[Unknown]\n```\n\n批量处理接口

Args:
    inputs: 输入列表
    config: 处理配置
    
Returns:
    批量处理结果\n\n---\n\n#### get_module_status\n\n```python\ndef get_module_status(self: Any) -> Dict[Unknown]\n```\n\n获取模块状态\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭处理器模块\n\n---\n\n\n## processors.relation_extractor\n\n关系提取器模块

从处理后的文本中提取数学关系，利用粗粒度分类结果进行精细模式匹配。\n\n### Classes\n\n#### RelationExtractor\n\n关系提取器 - 支持递归、复合、变量对齐、依赖链自动生成、兜底机制\n\n**Methods:**\n\n##### extract_relations\n```python\ndef extract_relations(self: Any, processed_text: ProcessedText, classification_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n从处理后的文本中提取关系

Args:
    processed_text: 处理后的文本对象
    classification_result: 分类结果
    
Returns:
    Dict 包含提取的关系\n\n---\n\n### Functions\n\n#### extract_relations\n\n```python\ndef extract_relations(self: Any, processed_text: ProcessedText, classification_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n从处理后的文本中提取关系

Args:
    processed_text: 处理后的文本对象
    classification_result: 分类结果
    
Returns:
    Dict 包含提取的关系\n\n---\n\n#### map_var\n\n```python\ndef map_var(name: Any)\n```\n\n---\n\n#### to_dict_dep\n\n```python\ndef to_dict_dep(dep: Any)\n```\n\n---\n\n#### map_var\n\n```python\ndef map_var(name: Any)\n```\n\n---\n\n\n## processors.relation_matcher\n\n### Classes\n\n#### MatchedPattern\n\n匹配到的模式\n\n---\n\n#### RelationMatcher\n\n关系匹配器\n\n**Methods:**\n\n##### match_patterns\n```python\ndef match_patterns(self: Any, tokens: Any, pos_tags: Any, text: Any, problem_type: Any, dependencies: Any, scene: Any, reasoning_type: Any) -> list\n```\n\n增强：支持 scene/reasoning_type 优先筛选\n\n##### match_problem_patterns\n```python\ndef match_problem_patterns(self: Any, problem_analysis: Any, tokens: Any, pos_tags: Any, text: Any)\n```\n\n匹配问题模式

Args:
    problem_analysis: 问题分析结果
    tokens: 分词结果
    pos_tags: 词性标注
    text: 原始文本
    
Returns:
    List[Tuple[Dict, float]]: 匹配到的模式及其分数\n\n---\n\n### Functions\n\n#### match_patterns\n\n```python\ndef match_patterns(self: Any, tokens: Any, pos_tags: Any, text: Any, problem_type: Any, dependencies: Any, scene: Any, reasoning_type: Any) -> list\n```\n\n增强：支持 scene/reasoning_type 优先筛选\n\n---\n\n#### match_problem_patterns\n\n```python\ndef match_problem_patterns(self: Any, problem_analysis: Any, tokens: Any, pos_tags: Any, text: Any)\n```\n\n匹配问题模式

Args:
    problem_analysis: 问题分析结果
    tokens: 分词结果
    pos_tags: 词性标注
    text: 原始文本
    
Returns:
    List[Tuple[Dict, float]]: 匹配到的模式及其分数\n\n---\n\n#### collect\n\n```python\ndef collect(group: Any)\n```\n\n---\n\n\n## processors.scalable_architecture\n\n### Classes\n\n#### ModuleType\n\n模块类型\n\n**Inherits from:** Enum\n\n---\n\n#### PluginStatus\n\n插件状态\n\n**Inherits from:** Enum\n\n---\n\n#### PluginInfo\n\n插件信息\n\n---\n\n#### ProcessorProtocol\n\n处理器协议\n\n**Inherits from:** Protocol\n\n**Methods:**\n\n##### process\n```python\ndef process(self: Any, input_data: Any, config: Dict[Unknown]) -> Any\n```\n\n处理数据\n\n##### get_info\n```python\ndef get_info(self: Any) -> PluginInfo\n```\n\n获取插件信息\n\n---\n\n#### EvaluatorProtocol\n\n评估器协议\n\n**Inherits from:** Protocol\n\n**Methods:**\n\n##### evaluate\n```python\ndef evaluate(self: Any, input_data: Any, expected_output: Any) -> Dict[Unknown]\n```\n\n评估结果\n\n---\n\n#### BasePlugin\n\n插件基类\n\n**Inherits from:** abc.ABC\n\n**Methods:**\n\n##### get_info\n```python\ndef get_info(self: Any) -> PluginInfo\n```\n\n获取插件信息\n\n##### initialize\n```python\ndef initialize(self: Any, config: Dict[Unknown])\n```\n\n初始化插件\n\n##### cleanup\n```python\ndef cleanup(self: Any)\n```\n\n清理资源\n\n---\n\n#### PluginManager\n\n🔧 插件管理器\n\n**Methods:**\n\n##### register_plugin\n```python\ndef register_plugin(self: Any, plugin_class: Type[BasePlugin], config: Dict[Unknown]) -> bool\n```\n\n📋 注册插件

Args:
    plugin_class: 插件类
    config: 插件配置
    
Returns:
    是否注册成功\n\n##### load_plugin\n```python\ndef load_plugin(self: Any, plugin_id: str, config: Dict[Unknown]) -> bool\n```\n\n🔄 加载插件

Args:
    plugin_id: 插件ID
    config: 插件配置
    
Returns:
    是否加载成功\n\n##### unload_plugin\n```python\ndef unload_plugin(self: Any, plugin_id: str) -> bool\n```\n\n⏹️ 卸载插件

Args:
    plugin_id: 插件ID
    
Returns:
    是否卸载成功\n\n##### get_plugins_by_type\n```python\ndef get_plugins_by_type(self: Any, module_type: ModuleType) -> List[str]\n```\n\n📋 按类型获取插件\n\n##### get_plugin\n```python\ndef get_plugin(self: Any, plugin_id: str) -> Optional[BasePlugin]\n```\n\n🔍 获取插件实例\n\n##### execute_plugin\n```python\ndef execute_plugin(self: Any, plugin_id: str, method: str) -> Any\n```\n\n▶️ 执行插件方法\n\n##### add_event_handler\n```python\ndef add_event_handler(self: Any, event_name: str, handler: callable)\n```\n\n添加事件处理器\n\n##### get_registry_info\n```python\ndef get_registry_info(self: Any) -> Dict[Unknown]\n```\n\n📊 获取注册表信息\n\n---\n\n#### ModularFramework\n\n🏗️ 模块化框架\n\n**Methods:**\n\n##### create_pipeline\n```python\ndef create_pipeline(self: Any, pipeline_name: str, plugin_sequence: List[str]) -> bool\n```\n\n🔗 创建处理管道

Args:
    pipeline_name: 管道名称
    plugin_sequence: 插件序列
    
Returns:
    是否创建成功\n\n##### execute_pipeline\n```python\ndef execute_pipeline(self: Any, pipeline_name: str, input_data: Any, config: Dict[Unknown]) -> Any\n```\n\n▶️ 执行处理管道

Args:
    pipeline_name: 管道名称
    input_data: 输入数据
    config: 配置
    
Returns:
    处理结果\n\n##### register_processor\n```python\ndef register_processor(self: Any, processor_class: Type[BasePlugin])\n```\n\n注册处理器\n\n##### get_available_processors\n```python\ndef get_available_processors(self: Any) -> List[str]\n```\n\n获取可用处理器\n\n##### save_configuration\n```python\ndef save_configuration(self: Any, config_path: str)\n```\n\n💾 保存配置\n\n##### load_configuration\n```python\ndef load_configuration(self: Any, config_path: str)\n```\n\n📂 加载配置\n\n---\n\n#### SimpleArithmeticProcessor\n\n简单算术处理器插件示例\n\n**Inherits from:** BasePlugin\n\n**Methods:**\n\n##### get_info\n```python\ndef get_info(self: Any) -> PluginInfo\n```\n\n##### process\n```python\ndef process(self: Any, input_data: Any, config: Dict[Unknown]) -> Any\n```\n\n处理算术表达式\n\n---\n\n#### ProblemClassifierProcessor\n\n问题分类处理器插件示例\n\n**Inherits from:** BasePlugin\n\n**Methods:**\n\n##### get_info\n```python\ndef get_info(self: Any) -> PluginInfo\n```\n\n##### process\n```python\ndef process(self: Any, input_data: Any, config: Dict[Unknown]) -> Any\n```\n\n分类数学问题\n\n---\n\n### Functions\n\n#### demo_scalable_architecture\n\n```python\ndef demo_scalable_architecture()\n```\n\n演示可扩展架构\n\n---\n\n#### process\n\n```python\ndef process(self: Any, input_data: Any, config: Dict[Unknown]) -> Any\n```\n\n处理数据\n\n---\n\n#### get_info\n\n```python\ndef get_info(self: Any) -> PluginInfo\n```\n\n获取插件信息\n\n---\n\n#### evaluate\n\n```python\ndef evaluate(self: Any, input_data: Any, expected_output: Any) -> Dict[Unknown]\n```\n\n评估结果\n\n---\n\n#### get_info\n\n```python\ndef get_info(self: Any) -> PluginInfo\n```\n\n获取插件信息\n\n---\n\n#### initialize\n\n```python\ndef initialize(self: Any, config: Dict[Unknown])\n```\n\n初始化插件\n\n---\n\n#### cleanup\n\n```python\ndef cleanup(self: Any)\n```\n\n清理资源\n\n---\n\n#### register_plugin\n\n```python\ndef register_plugin(self: Any, plugin_class: Type[BasePlugin], config: Dict[Unknown]) -> bool\n```\n\n📋 注册插件

Args:
    plugin_class: 插件类
    config: 插件配置
    
Returns:
    是否注册成功\n\n---\n\n#### load_plugin\n\n```python\ndef load_plugin(self: Any, plugin_id: str, config: Dict[Unknown]) -> bool\n```\n\n🔄 加载插件

Args:
    plugin_id: 插件ID
    config: 插件配置
    
Returns:
    是否加载成功\n\n---\n\n#### unload_plugin\n\n```python\ndef unload_plugin(self: Any, plugin_id: str) -> bool\n```\n\n⏹️ 卸载插件

Args:
    plugin_id: 插件ID
    
Returns:
    是否卸载成功\n\n---\n\n#### get_plugins_by_type\n\n```python\ndef get_plugins_by_type(self: Any, module_type: ModuleType) -> List[str]\n```\n\n📋 按类型获取插件\n\n---\n\n#### get_plugin\n\n```python\ndef get_plugin(self: Any, plugin_id: str) -> Optional[BasePlugin]\n```\n\n🔍 获取插件实例\n\n---\n\n#### execute_plugin\n\n```python\ndef execute_plugin(self: Any, plugin_id: str, method: str) -> Any\n```\n\n▶️ 执行插件方法\n\n---\n\n#### add_event_handler\n\n```python\ndef add_event_handler(self: Any, event_name: str, handler: callable)\n```\n\n添加事件处理器\n\n---\n\n#### get_registry_info\n\n```python\ndef get_registry_info(self: Any) -> Dict[Unknown]\n```\n\n📊 获取注册表信息\n\n---\n\n#### create_pipeline\n\n```python\ndef create_pipeline(self: Any, pipeline_name: str, plugin_sequence: List[str]) -> bool\n```\n\n🔗 创建处理管道

Args:
    pipeline_name: 管道名称
    plugin_sequence: 插件序列
    
Returns:
    是否创建成功\n\n---\n\n#### execute_pipeline\n\n```python\ndef execute_pipeline(self: Any, pipeline_name: str, input_data: Any, config: Dict[Unknown]) -> Any\n```\n\n▶️ 执行处理管道

Args:
    pipeline_name: 管道名称
    input_data: 输入数据
    config: 配置
    
Returns:
    处理结果\n\n---\n\n#### register_processor\n\n```python\ndef register_processor(self: Any, processor_class: Type[BasePlugin])\n```\n\n注册处理器\n\n---\n\n#### get_available_processors\n\n```python\ndef get_available_processors(self: Any) -> List[str]\n```\n\n获取可用处理器\n\n---\n\n#### save_configuration\n\n```python\ndef save_configuration(self: Any, config_path: str)\n```\n\n💾 保存配置\n\n---\n\n#### load_configuration\n\n```python\ndef load_configuration(self: Any, config_path: str)\n```\n\n📂 加载配置\n\n---\n\n#### get_info\n\n```python\ndef get_info(self: Any) -> PluginInfo\n```\n\n---\n\n#### process\n\n```python\ndef process(self: Any, input_data: Any, config: Dict[Unknown]) -> Any\n```\n\n处理算术表达式\n\n---\n\n#### get_info\n\n```python\ndef get_info(self: Any) -> PluginInfo\n```\n\n---\n\n#### process\n\n```python\ndef process(self: Any, input_data: Any, config: Dict[Unknown]) -> Any\n```\n\n分类数学问题\n\n---\n\n\n## processors.secure_components\n\nCOT-DIR 安全组件

提供安全的数学计算、文件操作等功能，替代不安全的操作。\n\n### Classes\n\n#### SecurityError\n\n安全异常\n\n**Inherits from:** Exception\n\n---\n\n#### SecureMathEvaluator\n\n安全的数学表达式计算器\n\n**Methods:**\n\n##### safe_eval\n```python\ndef safe_eval(self: Any, expression: str, allowed_names: Dict[Unknown]) -> Union[Unknown]\n```\n\n安全地计算数学表达式\n\n---\n\n### Functions\n\n#### safe_eval\n\n```python\ndef safe_eval(self: Any, expression: str, allowed_names: Dict[Unknown]) -> Union[Unknown]\n```\n\n安全地计算数学表达式\n\n---\n\n\n## processors.visualization\n\n### Functions\n\n#### build_reasoning_graph\n\n```python\ndef build_reasoning_graph(semantic_dependencies_list: Any, relation_types: Any)\n```\n\n构建 reasoning chain 的有向图，支持多 pattern 合并。
:param semantic_dependencies_list: List[List[str|dict]] 或 List[str|dict]
:param relation_types: 可选，List[str]，与每组 semantic_dependencies 对应（如 'explicit', 'implicit'）
:return: networkx.DiGraph, node_type_map\n\n---\n\n#### group_nodes_by_relation_type\n\n```python\ndef group_nodes_by_relation_type(node_type_map: Any)\n```\n\n将节点按类型分组，返回 {type: [nodes]} 字典\n\n---\n\n#### visualize_reasoning_chain\n\n```python\ndef visualize_reasoning_chain(G: Any, node_type_map: Any, title: Any, save_path: Any, reasoning_paths: Any, known_vars: Any, target_vars: Any, show: Any)\n```\n\n增强版可视化推理链，支持多目标变量高亮和路径显示

Args:
    G: networkx.DiGraph 推理链图
    node_type_map: 节点类型映射 {node: type}
    title: 图表标题
    save_path: 保存路径
    reasoning_paths: 推理路径列表，每个路径是节点列表
    known_vars: 已知变量列表
    target_vars: 目标变量列表
    show: 是否显示图表\n\n---\n\n#### visualize_reasoning_chain_interactive\n\n```python\ndef visualize_reasoning_chain_interactive(G: Any, node_type_map: Any, title: Any, save_path: Any, reasoning_paths: Any, known_vars: Any, target_vars: Any)\n```\n\n交互式可视化推理链

Args:
    G: networkx.DiGraph 推理链图
    node_type_map: 节点类型映射 {node: type}
    title: 图表标题
    save_path: 保存路径
    reasoning_paths: 推理路径列表，每个路径是节点列表
    known_vars: 已知变量列表
    target_vars: 目标变量列表
    
Returns:
    Network: pyvis.network.Network 对象\n\n---\n\n#### export_graph_to_dot\n\n```python\ndef export_graph_to_dot(G: Any, dot_path: Any)\n```\n\n导出 networkx.DiGraph 为 graphviz dot 文件。\n\n---\n\n#### annotate_reasoning_steps\n\n```python\ndef annotate_reasoning_steps(G: Any, reasoning_path: Any, equations: Any)\n```\n\n为推理路径上的每一步添加方程标注

Args:
    G: networkx.DiGraph 推理链图
    reasoning_path: 推理路径，节点列表
    equations: 方程列表
    
Returns:
    List[Dict]: 推理步骤列表，每个步骤包含 step, from, to, equation\n\n---\n\n#### detect_cycles\n\n```python\ndef detect_cycles(G: Any)\n```\n\n检测图中的环路

Args:
    G: networkx.DiGraph
    
Returns:
    List[List]: 环路列表，每个环路是节点列表\n\n---\n\n#### find_all_reasoning_paths\n\n```python\ndef find_all_reasoning_paths(G: Any, known_vars: Any, target_vars: Any, max_paths: Any, max_length: Any)\n```\n\n查找从已知变量到目标变量的所有可能推理路径

Args:
    G: networkx.DiGraph 推理链图
    known_vars: 已知变量列表
    target_vars: 目标变量列表
    max_paths: 每个起点-终点对返回的最大路径数
    max_length: 路径的最大长度限制
    
Returns:
    List[List]: 所有推理路径列表\n\n---\n\n#### rank_reasoning_paths\n\n```python\ndef rank_reasoning_paths(G: Any, paths: Any, known_vars: Any, target_vars: Any)\n```\n\n对推理路径进行排序，选择最优路径

Args:
    G: networkx.DiGraph 推理链图
    paths: 推理路径列表
    known_vars: 已知变量列表
    target_vars: 目标变量列表
    
Returns:
    List[Dict]: 排序后的路径及其评分\n\n---\n\n#### select_optimal_reasoning_paths\n\n```python\ndef select_optimal_reasoning_paths(G: Any, known_vars: Any, target_vars: Any, max_paths: Any)\n```\n\n选择最优的推理路径集合

Args:
    G: networkx.DiGraph 推理链图
    known_vars: 已知变量列表
    target_vars: 目标变量列表
    max_paths: 返回的最大路径数
    
Returns:
    List[List]: 最优推理路径列表\n\n---\n\n#### evaluate_reasoning_chain\n\n```python\ndef evaluate_reasoning_chain(reasoning_path: Any, G: Any)\n```\n\n评估推理链的质量

Args:
    reasoning_path: 推理路径，节点列表
    G: networkx.DiGraph 推理链图
    
Returns:
    Dict: 评估结果，包含 score, path_length, is_direct, missing_nodes, completeness\n\n---\n\n\n## reasoning.async_api\n\n推理模块异步版公共API

在原有功能基础上添加异步支持，提高并发处理能力。\n\n### Classes\n\n#### AsyncReasoningAPI\n\n推理模块异步版公共API - 支持并发处理的COT-DIR\n\n**Inherits from:** PublicAPI\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any, config: Optional[Unknown]) -> bool\n```\n\n同步初始化接口（保持兼容性）\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n同步解决问题接口（保持兼容性）\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n同步批量解决问题接口（保持兼容性）\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### get_statistics\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n同步获取统计信息接口（保持兼容性）\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> Any\n```\n\n同步关闭接口（保持兼容性）\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any, config: Optional[Unknown]) -> bool\n```\n\n同步初始化接口（保持兼容性）\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n同步解决问题接口（保持兼容性）\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n同步批量解决问题接口（保持兼容性）\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### get_statistics\n\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n同步获取统计信息接口（保持兼容性）\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> Any\n```\n\n同步关闭接口（保持兼容性）\n\n---\n\n\n## reasoning.confidence_calculator.confidence_base\n\n置信度计算器基类
定义置信度计算的通用接口和基础实现\n\n### Classes\n\n#### ConfidenceResult\n\n置信度计算结果\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### ConfidenceCalculator\n\n置信度计算器基类\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### calculate_confidence\n```python\ndef calculate_confidence(self: Any, reasoning_steps: List[Unknown], final_result: Any, context: Optional[Unknown]) -> ConfidenceResult\n```\n\n计算整体置信度

Args:
    reasoning_steps: 推理步骤列表
    final_result: 最终结果
    context: 上下文信息
    
Returns:
    ConfidenceResult: 置信度计算结果\n\n##### calculate_step_confidence\n```python\ndef calculate_step_confidence(self: Any, step: Dict[Unknown]) -> float\n```\n\n计算单个步骤的置信度

Args:
    step: 推理步骤
    
Returns:
    float: 步骤置信度\n\n##### calculate_logical_consistency\n```python\ndef calculate_logical_consistency(self: Any, reasoning_steps: List[Unknown]) -> float\n```\n\n计算推理步骤的逻辑一致性

Args:
    reasoning_steps: 推理步骤列表
    
Returns:
    float: 逻辑一致性分数\n\n##### calculate_numerical_accuracy\n```python\ndef calculate_numerical_accuracy(self: Any, reasoning_steps: List[Unknown]) -> float\n```\n\n计算数值计算的准确性

Args:
    reasoning_steps: 推理步骤列表
    
Returns:
    float: 数值准确性分数\n\n##### calculate_validation_confidence\n```python\ndef calculate_validation_confidence(self: Any, reasoning_steps: List[Unknown]) -> float\n```\n\n基于验证结果计算置信度

Args:
    reasoning_steps: 推理步骤列表
    
Returns:
    float: 验证置信度\n\n##### calculate_complexity_penalty\n```python\ndef calculate_complexity_penalty(self: Any, reasoning_steps: List[Unknown]) -> float\n```\n\n基于复杂度计算惩罚因子

Args:
    reasoning_steps: 推理步骤列表
    
Returns:
    float: 复杂度惩罚（越复杂惩罚越大，返回值越小）\n\n##### get_confidence_level\n```python\ndef get_confidence_level(self: Any, confidence: float) -> str\n```\n\n获取置信度等级

Args:
    confidence: 置信度值
    
Returns:
    str: 置信度等级\n\n##### update_weights\n```python\ndef update_weights(self: Any, new_weights: Dict[Unknown])\n```\n\n更新置信度权重

Args:
    new_weights: 新的权重配置\n\n---\n\n#### BasicConfidenceCalculator\n\n基础置信度计算器\n\n**Inherits from:** ConfidenceCalculator\n\n**Methods:**\n\n##### calculate_confidence\n```python\ndef calculate_confidence(self: Any, reasoning_steps: List[Unknown], final_result: Any, context: Optional[Unknown]) -> ConfidenceResult\n```\n\n计算基础置信度

使用简单的加权平均方法\n\n---\n\n### Functions\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### calculate_confidence\n\n```python\ndef calculate_confidence(self: Any, reasoning_steps: List[Unknown], final_result: Any, context: Optional[Unknown]) -> ConfidenceResult\n```\n\n计算整体置信度

Args:
    reasoning_steps: 推理步骤列表
    final_result: 最终结果
    context: 上下文信息
    
Returns:
    ConfidenceResult: 置信度计算结果\n\n---\n\n#### calculate_step_confidence\n\n```python\ndef calculate_step_confidence(self: Any, step: Dict[Unknown]) -> float\n```\n\n计算单个步骤的置信度

Args:
    step: 推理步骤
    
Returns:
    float: 步骤置信度\n\n---\n\n#### calculate_logical_consistency\n\n```python\ndef calculate_logical_consistency(self: Any, reasoning_steps: List[Unknown]) -> float\n```\n\n计算推理步骤的逻辑一致性

Args:
    reasoning_steps: 推理步骤列表
    
Returns:
    float: 逻辑一致性分数\n\n---\n\n#### calculate_numerical_accuracy\n\n```python\ndef calculate_numerical_accuracy(self: Any, reasoning_steps: List[Unknown]) -> float\n```\n\n计算数值计算的准确性

Args:
    reasoning_steps: 推理步骤列表
    
Returns:
    float: 数值准确性分数\n\n---\n\n#### calculate_validation_confidence\n\n```python\ndef calculate_validation_confidence(self: Any, reasoning_steps: List[Unknown]) -> float\n```\n\n基于验证结果计算置信度

Args:
    reasoning_steps: 推理步骤列表
    
Returns:
    float: 验证置信度\n\n---\n\n#### calculate_complexity_penalty\n\n```python\ndef calculate_complexity_penalty(self: Any, reasoning_steps: List[Unknown]) -> float\n```\n\n基于复杂度计算惩罚因子

Args:
    reasoning_steps: 推理步骤列表
    
Returns:
    float: 复杂度惩罚（越复杂惩罚越大，返回值越小）\n\n---\n\n#### get_confidence_level\n\n```python\ndef get_confidence_level(self: Any, confidence: float) -> str\n```\n\n获取置信度等级

Args:
    confidence: 置信度值
    
Returns:
    str: 置信度等级\n\n---\n\n#### update_weights\n\n```python\ndef update_weights(self: Any, new_weights: Dict[Unknown])\n```\n\n更新置信度权重

Args:
    new_weights: 新的权重配置\n\n---\n\n#### calculate_confidence\n\n```python\ndef calculate_confidence(self: Any, reasoning_steps: List[Unknown], final_result: Any, context: Optional[Unknown]) -> ConfidenceResult\n```\n\n计算基础置信度

使用简单的加权平均方法\n\n---\n\n\n## reasoning.cotdir_orchestrator\n\nCOT-DIR推理协调器

协调IRD、MLR、CV三个核心组件的工作流程。\n\n### Classes\n\n#### COTDIROrchestrator\n\nCOT-DIR推理协调器\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化所有组件\n\n##### orchestrate_full_pipeline\n```python\ndef orchestrate_full_pipeline(self: Any, problem: Dict[Unknown], options: Optional[Unknown]) -> Dict[Unknown]\n```\n\n执行完整的COT-DIR流水线

Args:
    problem: 问题数据
    options: 执行选项
    
Returns:
    完整的推理结果\n\n##### orchestrate_partial_pipeline\n```python\ndef orchestrate_partial_pipeline(self: Any, stage: str, input_data: Dict[Unknown], options: Optional[Unknown]) -> Dict[Unknown]\n```\n\n执行部分流水线（单个阶段）

Args:
    stage: 阶段名称 ("ird", "mlr", "cv")
    input_data: 输入数据
    options: 执行选项
    
Returns:
    阶段执行结果\n\n##### get_component_status\n```python\ndef get_component_status(self: Any) -> Dict[Unknown]\n```\n\n获取组件状态\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置所有统计信息\n\n##### shutdown\n```python\ndef shutdown(self: Any)\n```\n\n关闭协调器\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化所有组件\n\n---\n\n#### orchestrate_full_pipeline\n\n```python\ndef orchestrate_full_pipeline(self: Any, problem: Dict[Unknown], options: Optional[Unknown]) -> Dict[Unknown]\n```\n\n执行完整的COT-DIR流水线

Args:
    problem: 问题数据
    options: 执行选项
    
Returns:
    完整的推理结果\n\n---\n\n#### orchestrate_partial_pipeline\n\n```python\ndef orchestrate_partial_pipeline(self: Any, stage: str, input_data: Dict[Unknown], options: Optional[Unknown]) -> Dict[Unknown]\n```\n\n执行部分流水线（单个阶段）

Args:
    stage: 阶段名称 ("ird", "mlr", "cv")
    input_data: 输入数据
    options: 执行选项
    
Returns:
    阶段执行结果\n\n---\n\n#### get_component_status\n\n```python\ndef get_component_status(self: Any) -> Dict[Unknown]\n```\n\n获取组件状态\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置所有统计信息\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any)\n```\n\n关闭协调器\n\n---\n\n\n## reasoning.multi_step_reasoner.step_executor\n\n推理步骤执行器
负责执行具体的推理步骤和操作\n\n### Classes\n\n#### StepType\n\n推理步骤类型\n\n**Inherits from:** Enum\n\n---\n\n#### OperationType\n\n操作类型\n\n**Inherits from:** Enum\n\n---\n\n#### ExecutionResult\n\n执行结果\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典\n\n---\n\n#### StepExecutor\n\n推理步骤执行器\n\n**Methods:**\n\n##### execute_step\n```python\ndef execute_step(self: Any, step_data: Dict[Unknown], context: Optional[ReasoningContext]) -> ExecutionResult\n```\n\n执行推理步骤

Args:
    step_data: 步骤数据
    context: 推理上下文
    
Returns:
    ExecutionResult: 执行结果\n\n##### get_execution_stats\n```python\ndef get_execution_stats(self: Any) -> Dict[Unknown]\n```\n\n获取执行统计信息\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n### Functions\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典\n\n---\n\n#### execute_step\n\n```python\ndef execute_step(self: Any, step_data: Dict[Unknown], context: Optional[ReasoningContext]) -> ExecutionResult\n```\n\n执行推理步骤

Args:
    step_data: 步骤数据
    context: 推理上下文
    
Returns:
    ExecutionResult: 执行结果\n\n---\n\n#### get_execution_stats\n\n```python\ndef get_execution_stats(self: Any) -> Dict[Unknown]\n```\n\n获取执行统计信息\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n\n## reasoning.new_reasoning_engine\n\n新版推理引擎
整合策略模式、多步推理执行器和置信度计算器的现代化推理引擎\n\n### Classes\n\n#### ModernReasoningEngine\n\n现代化推理引擎\n\n**Inherits from:** IReasoningEngine\n\n**Methods:**\n\n##### reason\n```python\ndef reason(self: Any, problem: str, context: Optional[ReasoningContext]) -> ProcessingResult\n```\n\n执行推理

Args:
    problem: 问题文本
    context: 推理上下文
    
Returns:
    ProcessingResult: 推理结果\n\n##### get_reasoning_steps\n```python\ndef get_reasoning_steps(self: Any) -> List[Unknown]\n```\n\n获取当前的推理步骤\n\n##### set_reasoning_strategy\n```python\ndef set_reasoning_strategy(self: Any, strategy: str) -> Any\n```\n\n设置推理策略\n\n##### add_strategy\n```python\ndef add_strategy(self: Any, strategy: ReasoningStrategy) -> bool\n```\n\n添加新的推理策略\n\n##### remove_strategy\n```python\ndef remove_strategy(self: Any, strategy_name: str) -> bool\n```\n\n移除推理策略\n\n##### get_available_strategies\n```python\ndef get_available_strategies(self: Any) -> List[str]\n```\n\n获取可用的推理策略\n\n##### get_performance_report\n```python\ndef get_performance_report(self: Any) -> Dict[Unknown]\n```\n\n获取性能报告\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n### Functions\n\n#### reason\n\n```python\ndef reason(self: Any, problem: str, context: Optional[ReasoningContext]) -> ProcessingResult\n```\n\n执行推理

Args:
    problem: 问题文本
    context: 推理上下文
    
Returns:
    ProcessingResult: 推理结果\n\n---\n\n#### get_reasoning_steps\n\n```python\ndef get_reasoning_steps(self: Any) -> List[Unknown]\n```\n\n获取当前的推理步骤\n\n---\n\n#### set_reasoning_strategy\n\n```python\ndef set_reasoning_strategy(self: Any, strategy: str) -> Any\n```\n\n设置推理策略\n\n---\n\n#### add_strategy\n\n```python\ndef add_strategy(self: Any, strategy: ReasoningStrategy) -> bool\n```\n\n添加新的推理策略\n\n---\n\n#### remove_strategy\n\n```python\ndef remove_strategy(self: Any, strategy_name: str) -> bool\n```\n\n移除推理策略\n\n---\n\n#### get_available_strategies\n\n```python\ndef get_available_strategies(self: Any) -> List[str]\n```\n\n获取可用的推理策略\n\n---\n\n#### get_performance_report\n\n```python\ndef get_performance_report(self: Any) -> Dict[Unknown]\n```\n\n获取性能报告\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n\n## reasoning.orchestrator\n\n推理模块协调器

管理推理模块内部组件的协调和流程控制。\n\n### Classes\n\n#### ReasoningOrchestrator\n\n推理模块协调器\n\n**Inherits from:** BaseOrchestrator\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化协调器和所有组件\n\n##### orchestrate\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行\n\n##### register_component\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件\n\n##### get_component\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件\n\n##### get_component_status\n```python\ndef get_component_status(self: Any) -> Dict[Unknown]\n```\n\n获取所有组件状态\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭协调器\n\n##### get_performance_metrics\n```python\ndef get_performance_metrics(self: Any) -> Dict[Unknown]\n```\n\n获取性能指标\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化协调器和所有组件\n\n---\n\n#### orchestrate\n\n```python\ndef orchestrate(self: Any, operation: str) -> Any\n```\n\n协调指定操作的执行\n\n---\n\n#### register_component\n\n```python\ndef register_component(self: Any, name: str, component: Any) -> Any\n```\n\n注册组件\n\n---\n\n#### get_component\n\n```python\ndef get_component(self: Any, name: str) -> Any\n```\n\n获取组件\n\n---\n\n#### get_component_status\n\n```python\ndef get_component_status(self: Any) -> Dict[Unknown]\n```\n\n获取所有组件状态\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭协调器\n\n---\n\n#### get_performance_metrics\n\n```python\ndef get_performance_metrics(self: Any) -> Dict[Unknown]\n```\n\n获取性能指标\n\n---\n\n\n## reasoning.private.confidence_calc\n\n置信度计算器

负责计算推理过程的置信度分数，提供多维度的可信度评估。\n\n### Classes\n\n#### ConfidenceCalculator\n\n置信度计算器\n\n**Methods:**\n\n##### calculate_overall_confidence\n```python\ndef calculate_overall_confidence(self: Any, reasoning_steps: List[Unknown], validation_result: Optional[Dict], knowledge_context: Optional[Dict]) -> float\n```\n\n计算整体置信度\n\n##### analyze_confidence_distribution\n```python\ndef analyze_confidence_distribution(self: Any, reasoning_steps: List[Unknown]) -> Dict[Unknown]\n```\n\n分析置信度分布\n\n##### get_confidence_explanation\n```python\ndef get_confidence_explanation(self: Any, overall_confidence: float, components: Dict[Unknown]) -> str\n```\n\n生成置信度解释\n\n---\n\n### Functions\n\n#### calculate_overall_confidence\n\n```python\ndef calculate_overall_confidence(self: Any, reasoning_steps: List[Unknown], validation_result: Optional[Dict], knowledge_context: Optional[Dict]) -> float\n```\n\n计算整体置信度\n\n---\n\n#### analyze_confidence_distribution\n\n```python\ndef analyze_confidence_distribution(self: Any, reasoning_steps: List[Unknown]) -> Dict[Unknown]\n```\n\n分析置信度分布\n\n---\n\n#### get_confidence_explanation\n\n```python\ndef get_confidence_explanation(self: Any, overall_confidence: float, components: Dict[Unknown]) -> str\n```\n\n生成置信度解释\n\n---\n\n\n## reasoning.private.cv_validator\n\n链式验证器 (Chain Verification Validator)

专注于验证推理链的逻辑一致性和数学正确性。
这是COT-DIR算法的第三个核心组件。\n\n### Classes\n\n#### ValidationLevel\n\n验证级别\n\n**Inherits from:** Enum\n\n---\n\n#### ErrorType\n\n错误类型\n\n**Inherits from:** Enum\n\n---\n\n#### ValidationError\n\n验证错误\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### ValidationResult\n\n验证结果\n\n**Methods:**\n\n##### get_errors_by_type\n```python\ndef get_errors_by_type(self: Any, error_type: ErrorType) -> List[ValidationError]\n```\n\n按类型获取错误\n\n##### get_severe_errors\n```python\ndef get_severe_errors(self: Any, threshold: float) -> List[ValidationError]\n```\n\n获取严重错误\n\n##### has_critical_errors\n```python\ndef has_critical_errors(self: Any) -> bool\n```\n\n是否存在关键错误\n\n---\n\n#### ChainVerificationValidator\n\n链式验证器\n\n**Methods:**\n\n##### verify_reasoning_chain\n```python\ndef verify_reasoning_chain(self: Any, reasoning_steps: List[ReasoningStep], problem_context: Optional[Unknown]) -> ValidationResult\n```\n\n验证推理链

Args:
    reasoning_steps: 推理步骤列表
    problem_context: 问题上下文
    
Returns:
    ValidationResult: 验证结果\n\n##### get_stats\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n获取统计信息\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n### Functions\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### get_errors_by_type\n\n```python\ndef get_errors_by_type(self: Any, error_type: ErrorType) -> List[ValidationError]\n```\n\n按类型获取错误\n\n---\n\n#### get_severe_errors\n\n```python\ndef get_severe_errors(self: Any, threshold: float) -> List[ValidationError]\n```\n\n获取严重错误\n\n---\n\n#### has_critical_errors\n\n```python\ndef has_critical_errors(self: Any) -> bool\n```\n\n是否存在关键错误\n\n---\n\n#### verify_reasoning_chain\n\n```python\ndef verify_reasoning_chain(self: Any, reasoning_steps: List[ReasoningStep], problem_context: Optional[Unknown]) -> ValidationResult\n```\n\n验证推理链

Args:
    reasoning_steps: 推理步骤列表
    problem_context: 问题上下文
    
Returns:
    ValidationResult: 验证结果\n\n---\n\n#### get_stats\n\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n获取统计信息\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n\n## reasoning.private.ird_engine\n\n隐式关系发现引擎 (Implicit Relation Discovery Engine)

专注于从数学问题文本中发现隐式的数学关系。
这是COT-DIR算法的第一个核心组件。\n\n### Classes\n\n#### RelationType\n\n隐式关系类型\n\n**Inherits from:** Enum\n\n---\n\n#### ImplicitRelation\n\n隐式关系数据结构\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### IRDResult\n\nIRD处理结果\n\n**Methods:**\n\n##### get_relations_by_type\n```python\ndef get_relations_by_type(self: Any, relation_type: RelationType) -> List[ImplicitRelation]\n```\n\n按类型获取关系\n\n##### get_high_confidence_relations\n```python\ndef get_high_confidence_relations(self: Any, threshold: float) -> List[ImplicitRelation]\n```\n\n获取高置信度关系\n\n---\n\n#### ImplicitRelationDiscoveryEngine\n\n隐式关系发现引擎\n\n**Methods:**\n\n##### discover_relations\n```python\ndef discover_relations(self: Any, problem_text: str, context: Optional[Unknown]) -> IRDResult\n```\n\n发现问题文本中的隐式关系

Args:
    problem_text: 问题文本
    context: 可选的上下文信息
    
Returns:
    IRDResult: 发现的关系列表及相关信息\n\n##### get_stats\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n获取统计信息\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n### Functions\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### get_relations_by_type\n\n```python\ndef get_relations_by_type(self: Any, relation_type: RelationType) -> List[ImplicitRelation]\n```\n\n按类型获取关系\n\n---\n\n#### get_high_confidence_relations\n\n```python\ndef get_high_confidence_relations(self: Any, threshold: float) -> List[ImplicitRelation]\n```\n\n获取高置信度关系\n\n---\n\n#### discover_relations\n\n```python\ndef discover_relations(self: Any, problem_text: str, context: Optional[Unknown]) -> IRDResult\n```\n\n发现问题文本中的隐式关系

Args:
    problem_text: 问题文本
    context: 可选的上下文信息
    
Returns:
    IRDResult: 发现的关系列表及相关信息\n\n---\n\n#### get_stats\n\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n获取统计信息\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n\n## reasoning.private.mlr_processor\n\n多层级推理处理器 (Multi-Level Reasoning Processor)

专注于执行L0-L3不同复杂度级别的推理。
这是COT-DIR算法的第二个核心组件。\n\n### Classes\n\n#### ComplexityLevel\n\n推理复杂度级别\n\n**Inherits from:** Enum\n\n---\n\n#### ReasoningStepType\n\n推理步骤类型\n\n**Inherits from:** Enum\n\n---\n\n#### ReasoningStep\n\n推理步骤\n\n**Methods:**\n\n##### to_dict\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### MLRResult\n\n多层级推理结果\n\n**Methods:**\n\n##### get_steps_by_type\n```python\ndef get_steps_by_type(self: Any, step_type: ReasoningStepType) -> List[ReasoningStep]\n```\n\n按类型获取步骤\n\n##### get_calculation_chain\n```python\ndef get_calculation_chain(self: Any) -> List[ReasoningStep]\n```\n\n获取计算链\n\n---\n\n#### MultiLevelReasoningProcessor\n\n多层级推理处理器\n\n**Methods:**\n\n##### execute_reasoning\n```python\ndef execute_reasoning(self: Any, problem_text: str, relations: List[ImplicitRelation], context: Optional[Unknown]) -> MLRResult\n```\n\n执行多层级推理

Args:
    problem_text: 问题文本
    relations: 隐式关系列表
    context: 可选的上下文信息
    
Returns:
    MLRResult: 推理结果\n\n##### get_stats\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n获取统计信息\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n### Functions\n\n#### to_dict\n\n```python\ndef to_dict(self: Any) -> Dict[Unknown]\n```\n\n转换为字典格式\n\n---\n\n#### get_steps_by_type\n\n```python\ndef get_steps_by_type(self: Any, step_type: ReasoningStepType) -> List[ReasoningStep]\n```\n\n按类型获取步骤\n\n---\n\n#### get_calculation_chain\n\n```python\ndef get_calculation_chain(self: Any) -> List[ReasoningStep]\n```\n\n获取计算链\n\n---\n\n#### execute_reasoning\n\n```python\ndef execute_reasoning(self: Any, problem_text: str, relations: List[ImplicitRelation], context: Optional[Unknown]) -> MLRResult\n```\n\n执行多层级推理

Args:
    problem_text: 问题文本
    relations: 隐式关系列表
    context: 可选的上下文信息
    
Returns:
    MLRResult: 推理结果\n\n---\n\n#### get_stats\n\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n获取统计信息\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n\n## reasoning.private.processor\n\n推理处理器

实现核心的数学推理逻辑，从原有的ReasoningEngine重构而来。\n\n### Classes\n\n#### ReasoningProcessor\n\n推理处理器 - 核心推理逻辑\n\n**Inherits from:** BaseProcessor\n\n**Methods:**\n\n##### process\n```python\ndef process(self: Any, input_data: Any) -> Any\n```\n\n处理推理请求\n\n---\n\n### Functions\n\n#### process\n\n```python\ndef process(self: Any, input_data: Any) -> Any\n```\n\n处理推理请求\n\n---\n\n\n## reasoning.private.step_builder\n\n推理步骤构建器

负责构建结构化的推理步骤，为推理过程提供清晰的步骤记录。\n\n### Classes\n\n#### StepBuilder\n\n推理步骤构建器\n\n**Methods:**\n\n##### reset\n```python\ndef reset(self: Any)\n```\n\n重置步骤计数器\n\n##### create_step\n```python\ndef create_step(self: Any, action: str, description: str) -> Dict[Unknown]\n```\n\n创建一个推理步骤\n\n##### create_number_extraction_step\n```python\ndef create_number_extraction_step(self: Any, numbers: List[float]) -> Dict[Unknown]\n```\n\n创建数字提取步骤\n\n##### create_expression_parsing_step\n```python\ndef create_expression_parsing_step(self: Any, expression: str, result: Any) -> Dict[Unknown]\n```\n\n创建表达式解析步骤\n\n##### create_template_identification_step\n```python\ndef create_template_identification_step(self: Any, template_type: str, template: str) -> Dict[Unknown]\n```\n\n创建模板识别步骤\n\n##### create_calculation_step\n```python\ndef create_calculation_step(self: Any, operation: str, operands: List[Any], result: Any, calculation_details: str) -> Dict[Unknown]\n```\n\n创建计算步骤\n\n##### create_validation_step\n```python\ndef create_validation_step(self: Any, validation_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n创建验证步骤\n\n##### create_knowledge_enhancement_step\n```python\ndef create_knowledge_enhancement_step(self: Any, concepts: List[str], strategies: List[str]) -> Dict[Unknown]\n```\n\n创建知识增强步骤\n\n##### create_fallback_step\n```python\ndef create_fallback_step(self: Any, reason: str, fallback_action: str) -> Dict[Unknown]\n```\n\n创建回退步骤\n\n##### create_error_step\n```python\ndef create_error_step(self: Any, error_message: str, error_type: str) -> Dict[Unknown]\n```\n\n创建错误步骤\n\n##### build_step_summary\n```python\ndef build_step_summary(self: Any, steps: List[Unknown]) -> Dict[Unknown]\n```\n\n构建步骤摘要\n\n##### validate_step_sequence\n```python\ndef validate_step_sequence(self: Any, steps: List[Unknown]) -> Dict[Unknown]\n```\n\n验证步骤序列的合理性\n\n---\n\n### Functions\n\n#### reset\n\n```python\ndef reset(self: Any)\n```\n\n重置步骤计数器\n\n---\n\n#### create_step\n\n```python\ndef create_step(self: Any, action: str, description: str) -> Dict[Unknown]\n```\n\n创建一个推理步骤\n\n---\n\n#### create_number_extraction_step\n\n```python\ndef create_number_extraction_step(self: Any, numbers: List[float]) -> Dict[Unknown]\n```\n\n创建数字提取步骤\n\n---\n\n#### create_expression_parsing_step\n\n```python\ndef create_expression_parsing_step(self: Any, expression: str, result: Any) -> Dict[Unknown]\n```\n\n创建表达式解析步骤\n\n---\n\n#### create_template_identification_step\n\n```python\ndef create_template_identification_step(self: Any, template_type: str, template: str) -> Dict[Unknown]\n```\n\n创建模板识别步骤\n\n---\n\n#### create_calculation_step\n\n```python\ndef create_calculation_step(self: Any, operation: str, operands: List[Any], result: Any, calculation_details: str) -> Dict[Unknown]\n```\n\n创建计算步骤\n\n---\n\n#### create_validation_step\n\n```python\ndef create_validation_step(self: Any, validation_result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n创建验证步骤\n\n---\n\n#### create_knowledge_enhancement_step\n\n```python\ndef create_knowledge_enhancement_step(self: Any, concepts: List[str], strategies: List[str]) -> Dict[Unknown]\n```\n\n创建知识增强步骤\n\n---\n\n#### create_fallback_step\n\n```python\ndef create_fallback_step(self: Any, reason: str, fallback_action: str) -> Dict[Unknown]\n```\n\n创建回退步骤\n\n---\n\n#### create_error_step\n\n```python\ndef create_error_step(self: Any, error_message: str, error_type: str) -> Dict[Unknown]\n```\n\n创建错误步骤\n\n---\n\n#### build_step_summary\n\n```python\ndef build_step_summary(self: Any, steps: List[Unknown]) -> Dict[Unknown]\n```\n\n构建步骤摘要\n\n---\n\n#### validate_step_sequence\n\n```python\ndef validate_step_sequence(self: Any, steps: List[Unknown]) -> Dict[Unknown]\n```\n\n验证步骤序列的合理性\n\n---\n\n\n## reasoning.private.utils\n\n推理工具函数

提供推理模块通用的辅助功能和工具函数。\n\n### Classes\n\n#### ReasoningUtils\n\n推理工具类\n\n**Methods:**\n\n##### clean_text\n```python\ndef clean_text(text: str) -> str\n```\n\n清理文本，标准化输入\n\n##### extract_numbers_advanced\n```python\ndef extract_numbers_advanced(text: str) -> List[Unknown]\n```\n\n高级数字提取，包含位置和上下文信息\n\n##### identify_mathematical_operations\n```python\ndef identify_mathematical_operations(text: str) -> List[str]\n```\n\n识别文本中的数学运算指示词\n\n##### parse_mathematical_expression\n```python\ndef parse_mathematical_expression(text: str) -> Optional[Unknown]\n```\n\n解析数学表达式\n\n##### detect_problem_complexity\n```python\ndef detect_problem_complexity(text: str, numbers: List[Any]) -> str\n```\n\n检测问题复杂度\n\n##### validate_numerical_result\n```python\ndef validate_numerical_result(result: Union[Unknown], context: str, expected_range: Optional[Unknown]) -> Dict[Unknown]\n```\n\n验证数值结果的合理性\n\n##### format_reasoning_output\n```python\ndef format_reasoning_output(result: Dict[Unknown]) -> str\n```\n\n格式化推理输出为可读文本\n\n##### merge_reasoning_results\n```python\ndef merge_reasoning_results(results: List[Unknown], strategy: str) -> Dict[Unknown]\n```\n\n合并多个推理结果\n\n---\n\n### Functions\n\n#### clean_text\n\n```python\ndef clean_text(text: str) -> str\n```\n\n清理文本，标准化输入\n\n---\n\n#### extract_numbers_advanced\n\n```python\ndef extract_numbers_advanced(text: str) -> List[Unknown]\n```\n\n高级数字提取，包含位置和上下文信息\n\n---\n\n#### identify_mathematical_operations\n\n```python\ndef identify_mathematical_operations(text: str) -> List[str]\n```\n\n识别文本中的数学运算指示词\n\n---\n\n#### parse_mathematical_expression\n\n```python\ndef parse_mathematical_expression(text: str) -> Optional[Unknown]\n```\n\n解析数学表达式\n\n---\n\n#### detect_problem_complexity\n\n```python\ndef detect_problem_complexity(text: str, numbers: List[Any]) -> str\n```\n\n检测问题复杂度\n\n---\n\n#### validate_numerical_result\n\n```python\ndef validate_numerical_result(result: Union[Unknown], context: str, expected_range: Optional[Unknown]) -> Dict[Unknown]\n```\n\n验证数值结果的合理性\n\n---\n\n#### format_reasoning_output\n\n```python\ndef format_reasoning_output(result: Dict[Unknown]) -> str\n```\n\n格式化推理输出为可读文本\n\n---\n\n#### merge_reasoning_results\n\n```python\ndef merge_reasoning_results(results: List[Unknown], strategy: str) -> Dict[Unknown]\n```\n\n合并多个推理结果\n\n---\n\n\n## reasoning.private.validator\n\n推理结果验证器

负责验证推理结果的合理性和正确性。\n\n### Classes\n\n#### ReasoningValidator\n\n推理结果验证器\n\n**Inherits from:** BaseValidator\n\n**Methods:**\n\n##### validate\n```python\ndef validate(self: Any, data: Any) -> Dict[Unknown]\n```\n\n验证推理结果\n\n##### validate_template_match\n```python\ndef validate_template_match(self: Any, text: str, template_info: Optional[Dict]) -> Dict[Unknown]\n```\n\n验证模板匹配结果\n\n---\n\n### Functions\n\n#### validate\n\n```python\ndef validate(self: Any, data: Any) -> Dict[Unknown]\n```\n\n验证推理结果\n\n---\n\n#### validate_template_match\n\n```python\ndef validate_template_match(self: Any, text: str, template_info: Optional[Dict]) -> Dict[Unknown]\n```\n\n验证模板匹配结果\n\n---\n\n\n## reasoning.public_api\n\n推理模块公共API

提供标准化的推理接口，是外部访问推理功能的唯一入口。\n\n### Classes\n\n#### ReasoningAPI\n\n推理模块公共API\n\n**Inherits from:** PublicAPI\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化推理模块\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决数学问题

Args:
    problem: 问题数据，包含 'problem' 或 'cleaned_text' 字段
    
Returns:
    包含推理结果的字典，包含：
    - final_answer: 最终答案
    - confidence: 置信度 (0-1)
    - reasoning_steps: 推理步骤列表
    - strategy_used: 使用的推理策略\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决数学问题

Args:
    problems: 问题列表
    
Returns:
    结果列表，每个元素包含单个问题的推理结果\n\n##### get_reasoning_steps\n```python\ndef get_reasoning_steps(self: Any, problem: Dict[Unknown]) -> List[Unknown]\n```\n\n获取详细推理步骤（不包含最终答案）

Args:
    problem: 问题数据
    
Returns:
    推理步骤列表\n\n##### validate_result\n```python\ndef validate_result(self: Any, result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证推理结果

Args:
    result: 推理结果
    
Returns:
    验证结果，包含 is_valid, confidence, issues 等字段\n\n##### explain_reasoning\n```python\ndef explain_reasoning(self: Any, result: Dict[Unknown]) -> str\n```\n\n生成推理过程的文本解释

Args:
    result: 推理结果
    
Returns:
    推理过程的文本描述\n\n##### set_configuration\n```python\ndef set_configuration(self: Any, config: Dict[Unknown]) -> bool\n```\n\n设置推理模块配置

Args:
    config: 配置字典
    
Returns:
    设置是否成功\n\n##### get_configuration\n```python\ndef get_configuration(self: Any) -> Dict[Unknown]\n```\n\n获取当前配置

Returns:
    当前配置字典\n\n##### get_statistics\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取推理模块统计信息

Returns:
    统计信息字典\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭推理模块\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化推理模块\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决数学问题

Args:
    problem: 问题数据，包含 'problem' 或 'cleaned_text' 字段
    
Returns:
    包含推理结果的字典，包含：
    - final_answer: 最终答案
    - confidence: 置信度 (0-1)
    - reasoning_steps: 推理步骤列表
    - strategy_used: 使用的推理策略\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决数学问题

Args:
    problems: 问题列表
    
Returns:
    结果列表，每个元素包含单个问题的推理结果\n\n---\n\n#### get_reasoning_steps\n\n```python\ndef get_reasoning_steps(self: Any, problem: Dict[Unknown]) -> List[Unknown]\n```\n\n获取详细推理步骤（不包含最终答案）

Args:
    problem: 问题数据
    
Returns:
    推理步骤列表\n\n---\n\n#### validate_result\n\n```python\ndef validate_result(self: Any, result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证推理结果

Args:
    result: 推理结果
    
Returns:
    验证结果，包含 is_valid, confidence, issues 等字段\n\n---\n\n#### explain_reasoning\n\n```python\ndef explain_reasoning(self: Any, result: Dict[Unknown]) -> str\n```\n\n生成推理过程的文本解释

Args:
    result: 推理结果
    
Returns:
    推理过程的文本描述\n\n---\n\n#### set_configuration\n\n```python\ndef set_configuration(self: Any, config: Dict[Unknown]) -> bool\n```\n\n设置推理模块配置

Args:
    config: 配置字典
    
Returns:
    设置是否成功\n\n---\n\n#### get_configuration\n\n```python\ndef get_configuration(self: Any) -> Dict[Unknown]\n```\n\n获取当前配置

Returns:
    当前配置字典\n\n---\n\n#### get_statistics\n\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取推理模块统计信息

Returns:
    统计信息字典\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭推理模块\n\n---\n\n\n## reasoning.public_api_refactored\n\n推理模块重构版公共API

整合IRD、MLR、CV三个核心组件，提供统一的推理接口。\n\n### Classes\n\n#### ReasoningAPI\n\n推理模块重构版公共API - 整合IRD+MLR+CV\n\n**Inherits from:** PublicAPI\n\n**Methods:**\n\n##### initialize\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化推理模块\n\n##### get_module_info\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n##### health_check\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n##### solve_problem\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决数学问题 - COT-DIR完整流程

Args:
    problem: 问题数据，包含 'problem' 或 'cleaned_text' 字段
    
Returns:
    包含推理结果的字典：
    - final_answer: 最终答案
    - confidence: 整体置信度 (0-1)
    - reasoning_steps: 推理步骤列表
    - complexity_level: 复杂度级别
    - relations_found: 发现的隐式关系
    - validation_result: 验证结果
    - processing_info: 处理信息\n\n##### batch_solve\n```python\ndef batch_solve(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决数学问题\n\n##### get_reasoning_steps\n```python\ndef get_reasoning_steps(self: Any, problem: Dict[Unknown]) -> List[Unknown]\n```\n\n获取详细推理步骤\n\n##### validate_result\n```python\ndef validate_result(self: Any, result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证推理结果\n\n##### explain_reasoning\n```python\ndef explain_reasoning(self: Any, result: Dict[Unknown]) -> str\n```\n\n生成推理过程的文本解释\n\n##### get_statistics\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取推理模块统计信息\n\n##### reset_statistics\n```python\ndef reset_statistics(self: Any)\n```\n\n重置统计信息\n\n##### shutdown\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭推理模块\n\n---\n\n### Functions\n\n#### initialize\n\n```python\ndef initialize(self: Any) -> bool\n```\n\n初始化推理模块\n\n---\n\n#### get_module_info\n\n```python\ndef get_module_info(self: Any) -> ModuleInfo\n```\n\n获取模块信息\n\n---\n\n#### health_check\n\n```python\ndef health_check(self: Any) -> Dict[Unknown]\n```\n\n健康检查\n\n---\n\n#### solve_problem\n\n```python\ndef solve_problem(self: Any, problem: Dict[Unknown]) -> Dict[Unknown]\n```\n\n解决数学问题 - COT-DIR完整流程

Args:
    problem: 问题数据，包含 'problem' 或 'cleaned_text' 字段
    
Returns:
    包含推理结果的字典：
    - final_answer: 最终答案
    - confidence: 整体置信度 (0-1)
    - reasoning_steps: 推理步骤列表
    - complexity_level: 复杂度级别
    - relations_found: 发现的隐式关系
    - validation_result: 验证结果
    - processing_info: 处理信息\n\n---\n\n#### batch_solve\n\n```python\ndef batch_solve(self: Any, problems: List[Unknown]) -> List[Unknown]\n```\n\n批量解决数学问题\n\n---\n\n#### get_reasoning_steps\n\n```python\ndef get_reasoning_steps(self: Any, problem: Dict[Unknown]) -> List[Unknown]\n```\n\n获取详细推理步骤\n\n---\n\n#### validate_result\n\n```python\ndef validate_result(self: Any, result: Dict[Unknown]) -> Dict[Unknown]\n```\n\n验证推理结果\n\n---\n\n#### explain_reasoning\n\n```python\ndef explain_reasoning(self: Any, result: Dict[Unknown]) -> str\n```\n\n生成推理过程的文本解释\n\n---\n\n#### get_statistics\n\n```python\ndef get_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取推理模块统计信息\n\n---\n\n#### reset_statistics\n\n```python\ndef reset_statistics(self: Any)\n```\n\n重置统计信息\n\n---\n\n#### shutdown\n\n```python\ndef shutdown(self: Any) -> bool\n```\n\n关闭推理模块\n\n---\n\n\n## reasoning.strategy_manager.cot_strategy\n\n思维链推理策略 (Chain of Thought)
实现逐步推理的策略，适合中等复杂度的数学问题\n\n### Classes\n\n#### ChainOfThoughtStrategy\n\n思维链推理策略\n\n**Inherits from:** ReasoningStrategy\n\n**Methods:**\n\n##### can_handle\n```python\ndef can_handle(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> bool\n```\n\n判断是否能处理该问题

思维链策略适合：
- 包含数学关键词的问题
- 有明确数字的问题
- 中等复杂度的推理问题\n\n##### estimate_complexity\n```python\ndef estimate_complexity(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> float\n```\n\n估计问题复杂度

复杂度因素：
- 数字数量
- 运算类型数量
- 文本长度
- 关键词多样性\n\n---\n\n### Functions\n\n#### can_handle\n\n```python\ndef can_handle(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> bool\n```\n\n判断是否能处理该问题

思维链策略适合：
- 包含数学关键词的问题
- 有明确数字的问题
- 中等复杂度的推理问题\n\n---\n\n#### estimate_complexity\n\n```python\ndef estimate_complexity(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> float\n```\n\n估计问题复杂度

复杂度因素：
- 数字数量
- 运算类型数量
- 文本长度
- 关键词多样性\n\n---\n\n\n## reasoning.strategy_manager.got_strategy\n\n思维图推理策略 (Graph of Thoughts)
实现图状结构的推理策略，适合最复杂的数学问题\n\n### Classes\n\n#### ConceptNode\n\n概念节点\n\n---\n\n#### ReasoningEdge\n\n推理边\n\n---\n\n#### GraphOfThoughtsStrategy\n\n思维图推理策略\n\n**Inherits from:** ReasoningStrategy\n\n**Methods:**\n\n##### can_handle\n```python\ndef can_handle(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> bool\n```\n\n判断是否能处理该问题

思维图策略适合：
- 极其复杂的多步骤问题
- 包含多种概念关系的问题
- 需要综合推理的问题
- 有循环依赖或复杂约束的问题\n\n##### estimate_complexity\n```python\ndef estimate_complexity(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> float\n```\n\n估计问题复杂度\n\n---\n\n### Functions\n\n#### can_handle\n\n```python\ndef can_handle(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> bool\n```\n\n判断是否能处理该问题

思维图策略适合：
- 极其复杂的多步骤问题
- 包含多种概念关系的问题
- 需要综合推理的问题
- 有循环依赖或复杂约束的问题\n\n---\n\n#### estimate_complexity\n\n```python\ndef estimate_complexity(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> float\n```\n\n估计问题复杂度\n\n---\n\n\n## reasoning.strategy_manager.strategy_base\n\n推理策略基类
定义所有推理策略必须实现的接口\n\n### Classes\n\n#### StrategyType\n\n推理策略类型\n\n**Inherits from:** Enum\n\n---\n\n#### StrategyComplexity\n\n策略复杂度级别\n\n**Inherits from:** Enum\n\n---\n\n#### StrategyResult\n\n策略执行结果\n\n**Methods:**\n\n##### to_processing_result\n```python\ndef to_processing_result(self: Any) -> ProcessingResult\n```\n\n转换为标准处理结果\n\n---\n\n#### ReasoningStrategy\n\n推理策略基类\n\n**Inherits from:** ABC\n\n**Methods:**\n\n##### can_handle\n```python\ndef can_handle(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> bool\n```\n\n判断策略是否能处理给定的问题

Args:
    problem_text: 问题文本
    context: 推理上下文
    
Returns:
    bool: 是否能处理\n\n##### estimate_complexity\n```python\ndef estimate_complexity(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> float\n```\n\n估计问题的复杂度

Args:
    problem_text: 问题文本
    context: 推理上下文
    
Returns:
    float: 复杂度分数 (0.0-1.0)\n\n##### execute\n```python\ndef execute(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> StrategyResult\n```\n\n执行推理策略

Args:
    problem_text: 问题文本
    context: 推理上下文
    
Returns:
    StrategyResult: 推理结果\n\n##### get_strategy_info\n```python\ndef get_strategy_info(self: Any) -> Dict[Unknown]\n```\n\n获取策略信息\n\n##### update_config\n```python\ndef update_config(self: Any, new_config: Dict[Unknown])\n```\n\n更新策略配置\n\n##### reset_stats\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n### Functions\n\n#### to_processing_result\n\n```python\ndef to_processing_result(self: Any) -> ProcessingResult\n```\n\n转换为标准处理结果\n\n---\n\n#### can_handle\n\n```python\ndef can_handle(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> bool\n```\n\n判断策略是否能处理给定的问题

Args:
    problem_text: 问题文本
    context: 推理上下文
    
Returns:
    bool: 是否能处理\n\n---\n\n#### estimate_complexity\n\n```python\ndef estimate_complexity(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> float\n```\n\n估计问题的复杂度

Args:
    problem_text: 问题文本
    context: 推理上下文
    
Returns:
    float: 复杂度分数 (0.0-1.0)\n\n---\n\n#### execute\n\n```python\ndef execute(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> StrategyResult\n```\n\n执行推理策略

Args:
    problem_text: 问题文本
    context: 推理上下文
    
Returns:
    StrategyResult: 推理结果\n\n---\n\n#### get_strategy_info\n\n```python\ndef get_strategy_info(self: Any) -> Dict[Unknown]\n```\n\n获取策略信息\n\n---\n\n#### update_config\n\n```python\ndef update_config(self: Any, new_config: Dict[Unknown])\n```\n\n更新策略配置\n\n---\n\n#### reset_stats\n\n```python\ndef reset_stats(self: Any)\n```\n\n重置统计信息\n\n---\n\n\n## reasoning.strategy_manager.strategy_manager\n\n推理策略管理器
负责推理策略的选择、调度和管理\n\n### Classes\n\n#### StrategyManager\n\n推理策略管理器\n\n**Methods:**\n\n##### register_strategy\n```python\ndef register_strategy(self: Any, strategy: ReasoningStrategy) -> bool\n```\n\n注册推理策略

Args:
    strategy: 推理策略实例
    
Returns:
    bool: 注册是否成功\n\n##### unregister_strategy\n```python\ndef unregister_strategy(self: Any, strategy_name: str) -> bool\n```\n\n注销推理策略

Args:
    strategy_name: 策略名称
    
Returns:
    bool: 注销是否成功\n\n##### get_available_strategies\n```python\ndef get_available_strategies(self: Any) -> List[str]\n```\n\n获取可用策略列表\n\n##### get_strategies_by_type\n```python\ndef get_strategies_by_type(self: Any, strategy_type: StrategyType) -> List[str]\n```\n\n根据类型获取策略列表\n\n##### select_strategy\n```python\ndef select_strategy(self: Any, problem_text: str, context: Optional[ReasoningContext], preferred_strategy: Optional[str]) -> Optional[str]\n```\n\n智能选择推理策略

Args:
    problem_text: 问题文本
    context: 推理上下文
    preferred_strategy: 首选策略名称
    
Returns:
    Optional[str]: 选中的策略名称\n\n##### execute_reasoning\n```python\ndef execute_reasoning(self: Any, problem_text: str, context: Optional[ReasoningContext], strategy_name: Optional[str], enable_fallback: bool) -> StrategyResult\n```\n\n执行推理

Args:
    problem_text: 问题文本
    context: 推理上下文
    strategy_name: 指定策略名称
    enable_fallback: 是否启用回退机制
    
Returns:
    StrategyResult: 推理结果\n\n##### get_performance_report\n```python\ndef get_performance_report(self: Any) -> Dict[Unknown]\n```\n\n获取性能报告\n\n##### update_selection_rules\n```python\ndef update_selection_rules(self: Any, new_rules: Dict[Unknown])\n```\n\n更新策略选择规则\n\n##### reset_performance_stats\n```python\ndef reset_performance_stats(self: Any)\n```\n\n重置性能统计\n\n---\n\n### Functions\n\n#### register_strategy\n\n```python\ndef register_strategy(self: Any, strategy: ReasoningStrategy) -> bool\n```\n\n注册推理策略

Args:
    strategy: 推理策略实例
    
Returns:
    bool: 注册是否成功\n\n---\n\n#### unregister_strategy\n\n```python\ndef unregister_strategy(self: Any, strategy_name: str) -> bool\n```\n\n注销推理策略

Args:
    strategy_name: 策略名称
    
Returns:
    bool: 注销是否成功\n\n---\n\n#### get_available_strategies\n\n```python\ndef get_available_strategies(self: Any) -> List[str]\n```\n\n获取可用策略列表\n\n---\n\n#### get_strategies_by_type\n\n```python\ndef get_strategies_by_type(self: Any, strategy_type: StrategyType) -> List[str]\n```\n\n根据类型获取策略列表\n\n---\n\n#### select_strategy\n\n```python\ndef select_strategy(self: Any, problem_text: str, context: Optional[ReasoningContext], preferred_strategy: Optional[str]) -> Optional[str]\n```\n\n智能选择推理策略

Args:
    problem_text: 问题文本
    context: 推理上下文
    preferred_strategy: 首选策略名称
    
Returns:
    Optional[str]: 选中的策略名称\n\n---\n\n#### execute_reasoning\n\n```python\ndef execute_reasoning(self: Any, problem_text: str, context: Optional[ReasoningContext], strategy_name: Optional[str], enable_fallback: bool) -> StrategyResult\n```\n\n执行推理

Args:
    problem_text: 问题文本
    context: 推理上下文
    strategy_name: 指定策略名称
    enable_fallback: 是否启用回退机制
    
Returns:
    StrategyResult: 推理结果\n\n---\n\n#### get_performance_report\n\n```python\ndef get_performance_report(self: Any) -> Dict[Unknown]\n```\n\n获取性能报告\n\n---\n\n#### update_selection_rules\n\n```python\ndef update_selection_rules(self: Any, new_rules: Dict[Unknown])\n```\n\n更新策略选择规则\n\n---\n\n#### reset_performance_stats\n\n```python\ndef reset_performance_stats(self: Any)\n```\n\n重置性能统计\n\n---\n\n\n## reasoning.strategy_manager.tot_strategy\n\n思维树推理策略 (Tree of Thoughts)
实现多路径探索的推理策略，适合复杂数学问题\n\n### Classes\n\n#### ThoughtNode\n\n思维节点\n\n---\n\n#### TreeOfThoughtsStrategy\n\n思维树推理策略\n\n**Inherits from:** ReasoningStrategy\n\n**Methods:**\n\n##### can_handle\n```python\ndef can_handle(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> bool\n```\n\n判断是否能处理该问题

思维树策略适合：
- 复杂的多步骤问题
- 需要探索多种解法的问题
- 包含条件分支的问题\n\n##### estimate_complexity\n```python\ndef estimate_complexity(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> float\n```\n\n估计问题复杂度\n\n---\n\n### Functions\n\n#### can_handle\n\n```python\ndef can_handle(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> bool\n```\n\n判断是否能处理该问题

思维树策略适合：
- 复杂的多步骤问题
- 需要探索多种解法的问题
- 包含条件分支的问题\n\n---\n\n#### estimate_complexity\n\n```python\ndef estimate_complexity(self: Any, problem_text: str, context: Optional[ReasoningContext]) -> float\n```\n\n估计问题复杂度\n\n---\n\n\n## template_management.template_loader\n\n模板加载器
从外部文件加载模板，支持热重载\n\n### Classes\n\n#### TemplateLoader\n\n模板加载器\n\n**Methods:**\n\n##### load_external_templates\n```python\ndef load_external_templates(self: Any) -> int\n```\n\n加载外部模板文件

Returns:
    加载的模板数量\n\n##### load_templates_from_file\n```python\ndef load_templates_from_file(self: Any, file_path: str) -> int\n```\n\n从指定文件加载模板

Args:
    file_path: 文件路径
    
Returns:
    加载的模板数量\n\n##### watch_for_changes\n```python\ndef watch_for_changes(self: Any) -> bool\n```\n\n监控文件变更

Returns:
    是否有变更\n\n##### get_load_statistics\n```python\ndef get_load_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取加载统计信息

Returns:
    统计信息字典\n\n##### create_template_file\n```python\ndef create_template_file(self: Any, file_path: str, templates: List[Unknown]) -> bool\n```\n\n创建模板文件

Args:
    file_path: 文件路径
    templates: 模板列表
    
Returns:
    是否创建成功\n\n##### backup_templates\n```python\ndef backup_templates(self: Any, backup_dir: str) -> bool\n```\n\n备份模板

Args:
    backup_dir: 备份目录
    
Returns:
    是否备份成功\n\n---\n\n### Functions\n\n#### load_external_templates\n\n```python\ndef load_external_templates(self: Any) -> int\n```\n\n加载外部模板文件

Returns:
    加载的模板数量\n\n---\n\n#### load_templates_from_file\n\n```python\ndef load_templates_from_file(self: Any, file_path: str) -> int\n```\n\n从指定文件加载模板

Args:
    file_path: 文件路径
    
Returns:
    加载的模板数量\n\n---\n\n#### watch_for_changes\n\n```python\ndef watch_for_changes(self: Any) -> bool\n```\n\n监控文件变更

Returns:
    是否有变更\n\n---\n\n#### get_load_statistics\n\n```python\ndef get_load_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取加载统计信息

Returns:
    统计信息字典\n\n---\n\n#### create_template_file\n\n```python\ndef create_template_file(self: Any, file_path: str, templates: List[Unknown]) -> bool\n```\n\n创建模板文件

Args:
    file_path: 文件路径
    templates: 模板列表
    
Returns:
    是否创建成功\n\n---\n\n#### backup_templates\n\n```python\ndef backup_templates(self: Any, backup_dir: str) -> bool\n```\n\n备份模板

Args:
    backup_dir: 备份目录
    
Returns:
    是否备份成功\n\n---\n\n\n## template_management.template_manager\n\n模板管理器
实现ITemplateManager接口，协调模板注册表、匹配器和验证器\n\n### Classes\n\n#### TemplateManager\n\n模板管理器\n\n**Inherits from:** ITemplateManager\n\n**Methods:**\n\n##### match_template\n```python\ndef match_template(self: Any, text: str) -> Optional[Unknown]\n```\n\n匹配模板

Args:
    text: 待匹配文本
    
Returns:
    匹配结果字典\n\n##### get_templates\n```python\ndef get_templates(self: Any) -> List[Unknown]\n```\n\n获取所有模板

Returns:
    模板列表\n\n##### add_template\n```python\ndef add_template(self: Any, template: Dict[Unknown]) -> bool\n```\n\n添加模板

Args:
    template: 模板定义字典
    
Returns:
    是否添加成功\n\n##### remove_template\n```python\ndef remove_template(self: Any, template_id: str) -> bool\n```\n\n移除模板

Args:
    template_id: 模板ID
    
Returns:
    是否移除成功\n\n##### update_template\n```python\ndef update_template(self: Any, template_id: str, updates: Dict[Unknown]) -> bool\n```\n\n更新模板

Args:
    template_id: 模板ID
    updates: 更新内容
    
Returns:
    是否更新成功\n\n##### get_template_statistics\n```python\ndef get_template_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取模板统计信息

Returns:
    统计信息字典\n\n##### search_templates\n```python\ndef search_templates(self: Any, query: str, categories: Optional[Unknown]) -> List[Unknown]\n```\n\n搜索模板

Args:
    query: 搜索查询
    categories: 分类限制
    
Returns:
    匹配的模板列表\n\n##### export_templates\n```python\ndef export_templates(self: Any, file_path: str, categories: Optional[Unknown]) -> bool\n```\n\n导出模板

Args:
    file_path: 导出文件路径
    categories: 分类限制
    
Returns:
    是否导出成功\n\n##### import_templates\n```python\ndef import_templates(self: Any, file_path: str, overwrite: bool) -> int\n```\n\n导入模板

Args:
    file_path: 导入文件路径
    overwrite: 是否覆盖现有模板
    
Returns:
    导入的模板数量\n\n##### reload_templates\n```python\ndef reload_templates(self: Any) -> bool\n```\n\n重新加载模板

Returns:
    是否重新加载成功\n\n---\n\n#### TemplateError\n\n**Inherits from:** Exception\n\n---\n\n#### ITemplateManager\n\n---\n\n### Functions\n\n#### match_template\n\n```python\ndef match_template(self: Any, text: str) -> Optional[Unknown]\n```\n\n匹配模板

Args:
    text: 待匹配文本
    
Returns:
    匹配结果字典\n\n---\n\n#### get_templates\n\n```python\ndef get_templates(self: Any) -> List[Unknown]\n```\n\n获取所有模板

Returns:
    模板列表\n\n---\n\n#### add_template\n\n```python\ndef add_template(self: Any, template: Dict[Unknown]) -> bool\n```\n\n添加模板

Args:
    template: 模板定义字典
    
Returns:
    是否添加成功\n\n---\n\n#### remove_template\n\n```python\ndef remove_template(self: Any, template_id: str) -> bool\n```\n\n移除模板

Args:
    template_id: 模板ID
    
Returns:
    是否移除成功\n\n---\n\n#### update_template\n\n```python\ndef update_template(self: Any, template_id: str, updates: Dict[Unknown]) -> bool\n```\n\n更新模板

Args:
    template_id: 模板ID
    updates: 更新内容
    
Returns:
    是否更新成功\n\n---\n\n#### get_template_statistics\n\n```python\ndef get_template_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取模板统计信息

Returns:
    统计信息字典\n\n---\n\n#### search_templates\n\n```python\ndef search_templates(self: Any, query: str, categories: Optional[Unknown]) -> List[Unknown]\n```\n\n搜索模板

Args:
    query: 搜索查询
    categories: 分类限制
    
Returns:
    匹配的模板列表\n\n---\n\n#### export_templates\n\n```python\ndef export_templates(self: Any, file_path: str, categories: Optional[Unknown]) -> bool\n```\n\n导出模板

Args:
    file_path: 导出文件路径
    categories: 分类限制
    
Returns:
    是否导出成功\n\n---\n\n#### import_templates\n\n```python\ndef import_templates(self: Any, file_path: str, overwrite: bool) -> int\n```\n\n导入模板

Args:
    file_path: 导入文件路径
    overwrite: 是否覆盖现有模板
    
Returns:
    导入的模板数量\n\n---\n\n#### reload_templates\n\n```python\ndef reload_templates(self: Any) -> bool\n```\n\n重新加载模板

Returns:
    是否重新加载成功\n\n---\n\n#### monitor_performance\n\n```python\ndef monitor_performance(func: Any)\n```\n\n---\n\n\n## template_management.template_matcher\n\n模板匹配器
动态匹配文本与模板，支持多模式匹配和置信度计算\n\n### Classes\n\n#### MatchResult\n\n匹配结果\n\n---\n\n#### TemplateMatcher\n\n模板匹配器\n\n**Methods:**\n\n##### match_text\n```python\ndef match_text(self: Any, text: str, categories: Optional[Unknown]) -> List[MatchResult]\n```\n\n匹配文本与模板

Args:
    text: 待匹配文本
    categories: 限制匹配的分类，None表示匹配所有分类
    
Returns:
    匹配结果列表，按置信度降序排列\n\n##### match_text_best\n```python\ndef match_text_best(self: Any, text: str, categories: Optional[Unknown]) -> Optional[MatchResult]\n```\n\n获取最佳匹配结果

Args:
    text: 待匹配文本
    categories: 限制匹配的分类
    
Returns:
    最佳匹配结果，如果没有匹配则返回None\n\n##### extract_numbers\n```python\ndef extract_numbers(self: Any, text: str) -> List[float]\n```\n\n提取文本中的数字

Args:
    text: 文本
    
Returns:
    数字列表\n\n##### extract_variables\n```python\ndef extract_variables(self: Any, text: str, template: TemplateDefinition) -> Dict[Unknown]\n```\n\n根据模板提取变量

Args:
    text: 文本
    template: 模板定义
    
Returns:
    变量字典\n\n##### get_match_statistics\n```python\ndef get_match_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取匹配统计信息

Returns:
    统计信息字典\n\n---\n\n### Functions\n\n#### match_text\n\n```python\ndef match_text(self: Any, text: str, categories: Optional[Unknown]) -> List[MatchResult]\n```\n\n匹配文本与模板

Args:
    text: 待匹配文本
    categories: 限制匹配的分类，None表示匹配所有分类
    
Returns:
    匹配结果列表，按置信度降序排列\n\n---\n\n#### match_text_best\n\n```python\ndef match_text_best(self: Any, text: str, categories: Optional[Unknown]) -> Optional[MatchResult]\n```\n\n获取最佳匹配结果

Args:
    text: 待匹配文本
    categories: 限制匹配的分类
    
Returns:
    最佳匹配结果，如果没有匹配则返回None\n\n---\n\n#### extract_numbers\n\n```python\ndef extract_numbers(self: Any, text: str) -> List[float]\n```\n\n提取文本中的数字

Args:
    text: 文本
    
Returns:
    数字列表\n\n---\n\n#### extract_variables\n\n```python\ndef extract_variables(self: Any, text: str, template: TemplateDefinition) -> Dict[Unknown]\n```\n\n根据模板提取变量

Args:
    text: 文本
    template: 模板定义
    
Returns:
    变量字典\n\n---\n\n#### get_match_statistics\n\n```python\ndef get_match_statistics(self: Any) -> Dict[Unknown]\n```\n\n获取匹配统计信息

Returns:
    统计信息字典\n\n---\n\n\n## template_management.template_registry\n\n模板注册表
动态管理所有模板，支持模板的注册、查询、更新和删除\n\n### Classes\n\n#### TemplateMetadata\n\n模板元数据\n\n---\n\n#### TemplatePattern\n\n模板模式\n\n---\n\n#### TemplateDefinition\n\n模板定义\n\n---\n\n#### TemplateRegistry\n\n模板注册表\n\n**Methods:**\n\n##### register_template\n```python\ndef register_template(self: Any, template: TemplateDefinition) -> bool\n```\n\n注册模板

Args:
    template: 模板定义
    
Returns:
    是否注册成功\n\n##### unregister_template\n```python\ndef unregister_template(self: Any, template_id: str) -> bool\n```\n\n注销模板

Args:
    template_id: 模板ID
    
Returns:
    是否注销成功\n\n##### get_template\n```python\ndef get_template(self: Any, template_id: str) -> Optional[TemplateDefinition]\n```\n\n获取模板

Args:
    template_id: 模板ID
    
Returns:
    模板定义\n\n##### get_all_templates\n```python\ndef get_all_templates(self: Any) -> List[TemplateDefinition]\n```\n\n获取所有模板

Returns:
    模板列表\n\n##### get_active_templates\n```python\ndef get_active_templates(self: Any) -> List[TemplateDefinition]\n```\n\n获取活跃模板

Returns:
    活跃模板列表\n\n##### get_templates_by_category\n```python\ndef get_templates_by_category(self: Any, category: str) -> List[TemplateDefinition]\n```\n\n根据分类获取模板

Args:
    category: 分类名称
    
Returns:
    模板列表\n\n##### search_templates\n```python\ndef search_templates(self: Any, query: str) -> List[TemplateDefinition]\n```\n\n搜索模板

Args:
    query: 搜索查询
    
Returns:
    匹配的模板列表\n\n##### update_template_usage\n```python\ndef update_template_usage(self: Any, template_id: str, success: bool)\n```\n\n更新模板使用统计

Args:
    template_id: 模板ID
    success: 是否成功匹配\n\n##### get_stats\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n获取统计信息

Returns:
    统计信息字典\n\n##### export_templates\n```python\ndef export_templates(self: Any, file_path: str) -> bool\n```\n\n导出模板

Args:
    file_path: 导出文件路径
    
Returns:
    是否导出成功\n\n##### import_templates\n```python\ndef import_templates(self: Any, file_path: str) -> int\n```\n\n导入模板

Args:
    file_path: 导入文件路径
    
Returns:
    导入的模板数量\n\n---\n\n#### TemplateError\n\n**Inherits from:** Exception\n\n---\n\n#### ITemplateManager\n\n---\n\n### Functions\n\n#### register_template\n\n```python\ndef register_template(self: Any, template: TemplateDefinition) -> bool\n```\n\n注册模板

Args:
    template: 模板定义
    
Returns:
    是否注册成功\n\n---\n\n#### unregister_template\n\n```python\ndef unregister_template(self: Any, template_id: str) -> bool\n```\n\n注销模板

Args:
    template_id: 模板ID
    
Returns:
    是否注销成功\n\n---\n\n#### get_template\n\n```python\ndef get_template(self: Any, template_id: str) -> Optional[TemplateDefinition]\n```\n\n获取模板

Args:
    template_id: 模板ID
    
Returns:
    模板定义\n\n---\n\n#### get_all_templates\n\n```python\ndef get_all_templates(self: Any) -> List[TemplateDefinition]\n```\n\n获取所有模板

Returns:
    模板列表\n\n---\n\n#### get_active_templates\n\n```python\ndef get_active_templates(self: Any) -> List[TemplateDefinition]\n```\n\n获取活跃模板

Returns:
    活跃模板列表\n\n---\n\n#### get_templates_by_category\n\n```python\ndef get_templates_by_category(self: Any, category: str) -> List[TemplateDefinition]\n```\n\n根据分类获取模板

Args:
    category: 分类名称
    
Returns:
    模板列表\n\n---\n\n#### search_templates\n\n```python\ndef search_templates(self: Any, query: str) -> List[TemplateDefinition]\n```\n\n搜索模板

Args:
    query: 搜索查询
    
Returns:
    匹配的模板列表\n\n---\n\n#### update_template_usage\n\n```python\ndef update_template_usage(self: Any, template_id: str, success: bool)\n```\n\n更新模板使用统计

Args:
    template_id: 模板ID
    success: 是否成功匹配\n\n---\n\n#### get_stats\n\n```python\ndef get_stats(self: Any) -> Dict[Unknown]\n```\n\n获取统计信息

Returns:
    统计信息字典\n\n---\n\n#### export_templates\n\n```python\ndef export_templates(self: Any, file_path: str) -> bool\n```\n\n导出模板

Args:
    file_path: 导出文件路径
    
Returns:
    是否导出成功\n\n---\n\n#### import_templates\n\n```python\ndef import_templates(self: Any, file_path: str) -> int\n```\n\n导入模板

Args:
    file_path: 导入文件路径
    
Returns:
    导入的模板数量\n\n---\n\n\n## template_management.template_validator\n\n模板验证器
验证模板定义的有效性和质量\n\n### Classes\n\n#### TemplateValidator\n\n模板验证器\n\n**Methods:**\n\n##### validate_template\n```python\ndef validate_template(self: Any, template: TemplateDefinition) -> bool\n```\n\n验证模板定义

Args:
    template: 模板定义
    
Returns:
    是否验证通过\n\n##### validate_template_dict\n```python\ndef validate_template_dict(self: Any, template_dict: Dict[Unknown]) -> bool\n```\n\n验证模板字典

Args:
    template_dict: 模板字典
    
Returns:
    是否验证通过\n\n##### validate_pattern\n```python\ndef validate_pattern(self: Any, pattern: TemplatePattern) -> bool\n```\n\n验证单个模式

Args:
    pattern: 模式定义
    
Returns:
    是否验证通过\n\n##### get_validation_errors\n```python\ndef get_validation_errors(self: Any, template: TemplateDefinition) -> List[str]\n```\n\n获取验证错误列表

Args:
    template: 模板定义
    
Returns:
    错误信息列表\n\n---\n\n### Functions\n\n#### validate_template\n\n```python\ndef validate_template(self: Any, template: TemplateDefinition) -> bool\n```\n\n验证模板定义

Args:
    template: 模板定义
    
Returns:
    是否验证通过\n\n---\n\n#### validate_template_dict\n\n```python\ndef validate_template_dict(self: Any, template_dict: Dict[Unknown]) -> bool\n```\n\n验证模板字典

Args:
    template_dict: 模板字典
    
Returns:
    是否验证通过\n\n---\n\n#### validate_pattern\n\n```python\ndef validate_pattern(self: Any, pattern: TemplatePattern) -> bool\n```\n\n验证单个模式

Args:
    pattern: 模式定义
    
Returns:
    是否验证通过\n\n---\n\n#### get_validation_errors\n\n```python\ndef get_validation_errors(self: Any, template: TemplateDefinition) -> List[str]\n```\n\n获取验证错误列表

Args:
    template: 模板定义
    
Returns:
    错误信息列表\n\n---\n\n\n## validation.input_validator\n\n输入验证系统
提供全面的输入安全检查和数据验证\n\n### Classes\n\n#### InputValidator\n\n输入验证器\n\n**Methods:**\n\n##### validate_math_problem\n```python\ndef validate_math_problem(self: Any, text: str) -> Dict[Unknown]\n```\n\n验证数学问题输入\n\n##### validate_file_path\n```python\ndef validate_file_path(self: Any, path: str) -> Dict[Unknown]\n```\n\n验证文件路径安全性\n\n##### validate_numeric_input\n```python\ndef validate_numeric_input(self: Any, value: Any) -> Dict[Unknown]\n```\n\n验证数值输入\n\n##### batch_validate\n```python\ndef batch_validate(self: Any, inputs: List[Unknown]) -> Dict[Unknown]\n```\n\n批量验证输入\n\n---\n\n### Functions\n\n#### get_validator\n\n```python\ndef get_validator() -> InputValidator\n```\n\n获取全局验证器实例\n\n---\n\n#### validate_input\n\n```python\ndef validate_input(text: str, input_type: str) -> Dict[Unknown]\n```\n\n便捷的输入验证函数\n\n---\n\n#### validate_math_problem\n\n```python\ndef validate_math_problem(self: Any, text: str) -> Dict[Unknown]\n```\n\n验证数学问题输入\n\n---\n\n#### validate_file_path\n\n```python\ndef validate_file_path(self: Any, path: str) -> Dict[Unknown]\n```\n\n验证文件路径安全性\n\n---\n\n#### validate_numeric_input\n\n```python\ndef validate_numeric_input(self: Any, value: Any) -> Dict[Unknown]\n```\n\n验证数值输入\n\n---\n\n#### batch_validate\n\n```python\ndef batch_validate(self: Any, inputs: List[Unknown]) -> Dict[Unknown]\n```\n\n批量验证输入\n\n---\n\n