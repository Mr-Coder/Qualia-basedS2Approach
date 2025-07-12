"""
AI协作友好的异常类定义

这个模块定义了系统中使用的所有异常类型。
AI助手可以通过这些异常理解错误情况并提供相应的解决方案。

AI_CONTEXT: 结构化的错误处理，提供清晰的错误信息和修复建议
RESPONSIBILITY: 定义系统中所有可能的异常情况
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class AICollaborativeError(Exception):
    """
    AI协作系统基础异常类
    
    AI_CONTEXT: 所有系统异常的基类，提供统一的错误处理接口
    RESPONSIBILITY: 提供丰富的错误上下文和修复建议
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        fix_recommendations: Optional[List[str]] = None
    ):
        """
        初始化AI协作友好的异常
        
        Args:
            message: 错误描述信息
            error_code: 错误代码，便于分类和处理
            context: 错误发生的上下文信息
            suggestions: AI助手可以参考的建议
            fix_recommendations: 具体的修复建议
            
        AI_HINT: 提供尽可能详细的错误信息，便于AI理解和处理
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.suggestions = suggestions or []
        self.fix_recommendations = fix_recommendations or []
    
    def get_ai_friendly_description(self) -> Dict[str, Any]:
        """
        获取AI友好的错误描述
        
        Returns:
            Dict: 包含错误详情、上下文和建议的结构化信息
            
        AI_HINT: 使用这个方法获取结构化的错误信息
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "suggestions": self.suggestions,
            "fix_recommendations": self.fix_recommendations,
            "severity": self._get_severity_level()
        }
    
    def _get_severity_level(self) -> str:
        """获取错误严重程度"""
        return "medium"  # 子类可以重写此方法


class ReasoningError(AICollaborativeError):
    """
    推理过程异常
    
    AI_CONTEXT: 推理引擎执行过程中出现的错误
    AI_INSTRUCTION: 当推理策略无法处理问题或推理过程失败时抛出
    """
    
    def __init__(
        self, 
        message: str, 
        strategy_name: str = "",
        problem_id: str = "",
        reasoning_step: int = -1,
        **kwargs
    ):
        """
        推理错误初始化
        
        Args:
            message: 错误描述
            strategy_name: 失败的推理策略名称
            problem_id: 相关问题ID
            reasoning_step: 失败的推理步骤
        """
        context = kwargs.get('context', {})
        context.update({
            "strategy_name": strategy_name,
            "problem_id": problem_id,
            "reasoning_step": reasoning_step
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "检查问题是否符合策略的处理范围",
            "验证输入数据的完整性和格式",
            "考虑使用其他推理策略",
            "检查推理步骤的逻辑一致性"
        ])
        
        fix_recommendations = kwargs.get('fix_recommendations', [])
        fix_recommendations.extend([
            f"使用不同的推理策略重新处理问题 {problem_id}",
            "增加数据预处理步骤",
            "调整策略参数配置",
            "添加问题复杂度预判断"
        ])
        
        # 移除可能重复的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['context', 'suggestions', 'fix_recommendations']}
        
        super().__init__(
            message, 
            error_code="REASONING_ERROR",
            context=context,
            suggestions=suggestions,
            fix_recommendations=fix_recommendations,
            **filtered_kwargs
        )
    
    def _get_severity_level(self) -> str:
        return "high"  # 推理错误通常较严重


class ValidationError(AICollaborativeError):
    """
    验证过程异常
    
    AI_CONTEXT: 数据验证或结果验证过程中出现的错误
    AI_INSTRUCTION: 当验证规则不通过或验证过程失败时抛出
    """
    
    def __init__(
        self, 
        message: str, 
        validation_type: str = "",
        target_object: str = "",
        failed_rules: Optional[List[str]] = None,
        **kwargs
    ):
        """
        验证错误初始化
        
        Args:
            message: 错误描述
            validation_type: 验证类型
            target_object: 验证目标对象
            failed_rules: 失败的验证规则列表
        """
        context = kwargs.get('context', {})
        context.update({
            "validation_type": validation_type,
            "target_object": target_object,
            "failed_rules": failed_rules or []
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "检查输入数据是否符合预期格式",
            "验证必需字段是否完整",
            "确认数据类型和取值范围",
            "检查业务逻辑约束"
        ])
        
        fix_recommendations = kwargs.get('fix_recommendations', [])
        if failed_rules:
            fix_recommendations.extend([
                f"修复验证规则: {', '.join(failed_rules)}",
                "调整数据格式以符合验证要求",
                "更新验证规则配置"
            ])
        
        # 移除可能重复的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['context', 'suggestions', 'fix_recommendations']}
        
        super().__init__(
            message, 
            error_code="VALIDATION_ERROR",
            context=context,
            suggestions=suggestions,
            fix_recommendations=fix_recommendations,
            **filtered_kwargs
        )
    
    def _get_severity_level(self) -> str:
        return "medium"


class ConfigurationError(AICollaborativeError):
    """
    配置错误异常
    
    AI_CONTEXT: 系统配置相关的错误
    AI_INSTRUCTION: 当配置文件无效、缺少必需配置或配置冲突时抛出
    """
    
    def __init__(
        self, 
        message: str, 
        config_file: str = "",
        config_key: str = "",
        expected_type: str = "",
        actual_value: Any = None,
        **kwargs
    ):
        """
        配置错误初始化
        
        Args:
            message: 错误描述
            config_file: 配置文件路径
            config_key: 错误的配置键
            expected_type: 期望的配置类型
            actual_value: 实际的配置值
        """
        context = kwargs.get('context', {})
        context.update({
            "config_file": config_file,
            "config_key": config_key,
            "expected_type": expected_type,
            "actual_value": str(actual_value) if actual_value is not None else None
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "检查配置文件的语法和格式",
            "确认必需的配置项已设置",
            "验证配置值的类型和范围",
            "查看配置文件的示例和文档"
        ])
        
        fix_recommendations = kwargs.get('fix_recommendations', [])
        if config_key:
            fix_recommendations.extend([
                f"修正配置项 '{config_key}' 的值",
                f"确保 '{config_key}' 的类型为 {expected_type}",
                f"检查配置文件 {config_file} 的格式"
            ])
        
        # 移除可能重复的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['context', 'suggestions', 'fix_recommendations']}
        
        super().__init__(
            message, 
            error_code="CONFIG_ERROR",
            context=context,
            suggestions=suggestions,
            fix_recommendations=fix_recommendations,
            **filtered_kwargs
        )
    
    def _get_severity_level(self) -> str:
        return "high"  # 配置错误通常会影响系统启动


class DataProcessingError(AICollaborativeError):
    """
    数据处理异常
    
    AI_CONTEXT: 数据加载、处理、转换过程中的错误
    AI_INSTRUCTION: 当数据处理失败或数据格式不兼容时抛出
    """
    
    def __init__(
        self, 
        message: str, 
        processor_name: str = "",
        data_source: str = "",
        processing_stage: str = "",
        data_sample: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        数据处理错误初始化
        
        Args:
            message: 错误描述
            processor_name: 处理器名称
            data_source: 数据源
            processing_stage: 处理阶段
            data_sample: 问题数据样本
        """
        context = kwargs.get('context', {})
        context.update({
            "processor_name": processor_name,
            "data_source": data_source,
            "processing_stage": processing_stage,
            "data_sample": data_sample
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "检查数据源的可用性和格式",
            "验证数据处理器的配置",
            "确认数据结构与预期一致",
            "检查数据编码和字符集"
        ])
        
        fix_recommendations = kwargs.get('fix_recommendations', [])
        fix_recommendations.extend([
            f"重新配置数据处理器 {processor_name}",
            f"检查数据源 {data_source} 的格式",
            "添加数据预处理步骤",
            "更新数据处理逻辑"
        ])
        
        super().__init__(
            message, 
            error_code="DATA_PROCESSING_ERROR",
            context=context,
            suggestions=suggestions,
            fix_recommendations=fix_recommendations,
            **kwargs
        )
    
    def _get_severity_level(self) -> str:
        return "medium"


class ExperimentError(AICollaborativeError):
    """
    实验执行异常
    
    AI_CONTEXT: 实验运行过程中的错误
    AI_INSTRUCTION: 当实验设置无效或执行失败时抛出
    """
    
    def __init__(
        self, 
        message: str, 
        experiment_id: str = "",
        experiment_name: str = "",
        stage: str = "",
        **kwargs
    ):
        """
        实验错误初始化
        
        Args:
            message: 错误描述
            experiment_id: 实验ID
            experiment_name: 实验名称
            stage: 失败阶段
        """
        context = kwargs.get('context', {})
        context.update({
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "stage": stage
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "检查实验配置的完整性",
            "验证实验环境的准备情况",
            "确认实验数据的可用性",
            "检查资源分配和权限"
        ])
        
        super().__init__(
            message, 
            error_code="EXPERIMENT_ERROR",
            context=context,
            suggestions=suggestions,
            **kwargs
        )


class PerformanceError(AICollaborativeError):
    """
    性能相关异常
    
    AI_CONTEXT: 系统性能监控和分析过程中的错误
    AI_INSTRUCTION: 当性能指标异常或监控失败时抛出
    """
    
    def __init__(
        self, 
        message: str, 
        operation: str = "",
        threshold_exceeded: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        性能错误初始化
        
        Args:
            message: 错误描述
            operation: 相关操作
            threshold_exceeded: 超出的阈值信息
        """
        context = kwargs.get('context', {})
        context.update({
            "operation": operation,
            "threshold_exceeded": threshold_exceeded or {}
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "检查系统资源使用情况",
            "分析性能瓶颈原因",
            "考虑优化算法或数据结构",
            "调整性能监控阈值"
        ])
        
        super().__init__(
            message, 
            error_code="PERFORMANCE_ERROR",
            context=context,
            suggestions=suggestions,
            **kwargs
        )


# AI_HELPER: 异常处理工具函数
def handle_ai_collaborative_error(error: AICollaborativeError) -> Dict[str, Any]:
    """
    处理AI协作异常的标准方法
    
    Args:
        error: AI协作异常实例
        
    Returns:
        Dict: 结构化的错误处理信息
        
    AI_HINT: 使用这个函数获取标准化的错误处理信息
    """
    error_info = error.get_ai_friendly_description()
    
    # 添加处理建议
    error_info["handling_steps"] = [
        "1. 分析错误上下文和原因",
        "2. 参考建议列表进行初步诊断",
        "3. 按照修复建议进行问题解决",
        "4. 验证修复效果",
        "5. 记录处理过程以便后续参考"
    ]
    
    return error_info


def create_error_from_exception(exception: Exception, context: Optional[Dict[str, Any]] = None) -> AICollaborativeError:
    """
    从标准异常创建AI协作友好的异常
    
    Args:
        exception: 原始异常
        context: 额外的上下文信息
        
    Returns:
        AICollaborativeError: AI协作友好的异常
        
    AI_HINT: 用于包装第三方库或系统异常
    """
    return AICollaborativeError(
        message=str(exception),
        error_code=f"WRAPPED_{exception.__class__.__name__.upper()}",
        context=context or {"original_exception": exception.__class__.__name__},
        suggestions=[
            "检查原始异常的具体信息",
            "查看相关的系统日志",
            "验证输入参数和环境配置"
        ],
        fix_recommendations=[
            "根据原始异常类型查找解决方案",
            "检查系统依赖和环境配置",
            "添加适当的错误处理逻辑"
        ]
    )


# AI_HINT: 异常使用指南
"""
AI_USAGE_GUIDE:

1. 抛出异常时提供尽可能详细的上下文信息
2. 使用适当的异常类型来表示不同的错误情况
3. 在异常处理中使用 get_ai_friendly_description() 获取结构化信息
4. 利用建议和修复建议帮助用户解决问题
5. 记录异常信息用于系统改进和学习

示例用法:
try:
    # 执行推理操作
    result = strategy.solve(problem)
except Exception as e:
    raise ReasoningError(
        "推理策略执行失败",
        strategy_name=strategy.__class__.__name__,
        problem_id=problem.id,
        context={"original_error": str(e)}
    )
""" 