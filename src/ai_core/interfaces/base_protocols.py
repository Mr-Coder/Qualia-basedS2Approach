"""
基础协议定义 - AI协作友好的接口规范

这个模块定义了系统中所有组件必须遵循的协议接口。
AI助手可以通过这些协议理解如何实现新的组件。

AI_CONTEXT: 协议定义了组件的行为契约
RESPONSIBILITY: 定义标准化的接口规范
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union

from .data_structures import (ExperimentResult, MathProblem,
                              PerformanceMetrics, ReasoningResult,
                              ReasoningStep, ValidationResult)


class ReasoningStrategy(Protocol):
    """
    推理策略协议 - AI可以通过实现这个协议来创建新的推理策略
    
    AI_INSTRUCTION: 要创建新的推理策略，实现以下方法：
    1. can_handle() - 判断能否处理特定问题
    2. solve() - 解决问题并返回结构化结果
    3. get_confidence() - 返回策略对问题的置信度
    """
    
    @abstractmethod
    def can_handle(self, problem: MathProblem) -> bool:
        """
        判断此策略是否能处理给定的数学问题
        
        Args:
            problem: 待处理的数学问题
            
        Returns:
            bool: True if can handle, False otherwise
            
        AI_HINT: 实现这个方法来决定策略的适用范围
        """
        ...
    
    @abstractmethod
    def solve(self, problem: MathProblem) -> ReasoningResult:
        """
        解决数学问题
        
        Args:
            problem: 待解决的数学问题
            
        Returns:
            ReasoningResult: 包含推理步骤和最终答案的结果
            
        AI_HINT: 这是策略的核心方法，返回详细的推理过程
        """
        ...
    
    @abstractmethod
    def get_confidence(self, problem: MathProblem) -> float:
        """
        获取策略对问题的置信度
        
        Args:
            problem: 待评估的数学问题
            
        Returns:
            float: 置信度 [0.0, 1.0]
            
        AI_HINT: 用于策略选择，高置信度的策略优先使用
        """
        ...


class DataProcessor(Protocol):
    """
    数据处理器协议 - AI可以实现这个协议来创建数据处理组件
    
    AI_INSTRUCTION: 数据处理器负责：
    1. process() - 处理输入数据
    2. validate_input() - 验证输入数据有效性
    3. get_output_schema() - 返回输出数据模式
    """
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        处理输入数据
        
        Args:
            data: 待处理的数据
            
        Returns:
            Any: 处理后的数据
            
        AI_HINT: 实现具体的数据转换逻辑
        """
        ...
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """
        验证输入数据的有效性
        
        Args:
            data: 待验证的数据
            
        Returns:
            bool: 数据是否有效
            
        AI_HINT: 在处理前检查数据完整性
        """
        ...
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        获取输出数据的模式定义
        
        Returns:
            Dict: 输出数据的结构描述
            
        AI_HINT: 用于下游组件理解数据格式
        """
        ...


class Validator(Protocol):
    """
    验证器协议 - AI可以实现这个协议来创建验证组件
    
    AI_INSTRUCTION: 验证器用于：
    1. validate() - 执行验证逻辑
    2. get_error_details() - 获取详细错误信息
    3. suggest_fixes() - 提供修复建议
    """
    
    @abstractmethod
    def validate(self, target: Any) -> ValidationResult:
        """
        执行验证逻辑
        
        Args:
            target: 待验证的对象
            
        Returns:
            ValidationResult: 验证结果
            
        AI_HINT: 返回详细的验证信息，包括错误和建议
        """
        ...
    
    @abstractmethod
    def get_error_details(self, target: Any) -> List[str]:
        """
        获取详细的错误信息
        
        Args:
            target: 待检查的对象
            
        Returns:
            List[str]: 错误信息列表
            
        AI_HINT: 提供人类可读的错误描述
        """
        ...
    
    @abstractmethod
    def suggest_fixes(self, target: Any) -> List[str]:
        """
        提供修复建议
        
        Args:
            target: 需要修复的对象
            
        Returns:
            List[str]: 修复建议列表
            
        AI_HINT: 帮助用户理解如何修复问题
        """
        ...


class Orchestrator(Protocol):
    """
    协调器协议 - AI可以实现这个协议来创建流程协调组件
    
    AI_INSTRUCTION: 协调器负责：
    1. orchestrate() - 协调整个处理流程
    2. register_component() - 注册组件
    3. get_execution_plan() - 获取执行计划
    """
    
    @abstractmethod
    def orchestrate(self, input_data: Any) -> Any:
        """
        协调整个处理流程
        
        Args:
            input_data: 输入数据
            
        Returns:
            Any: 处理结果
            
        AI_HINT: 管理多个组件的协作执行
        """
        ...
    
    @abstractmethod
    def register_component(self, name: str, component: Any) -> None:
        """
        注册组件到协调器
        
        Args:
            name: 组件名称
            component: 组件实例
            
        AI_HINT: 动态注册可插拔组件
        """
        ...
    
    @abstractmethod
    def get_execution_plan(self, input_data: Any) -> List[str]:
        """
        获取执行计划
        
        Args:
            input_data: 输入数据
            
        Returns:
            List[str]: 执行步骤列表
            
        AI_HINT: 帮助理解处理流程
        """
        ...


class ExperimentRunner(Protocol):
    """
    实验运行器协议 - AI可以实现这个协议来创建实验组件
    
    AI_INSTRUCTION: 实验运行器用于：
    1. run_experiment() - 运行实验
    2. setup_experiment() - 设置实验环境
    3. analyze_results() - 分析结果
    """
    
    @abstractmethod
    def run_experiment(self, config: Dict[str, Any]) -> ExperimentResult:
        """
        运行实验
        
        Args:
            config: 实验配置
            
        Returns:
            ExperimentResult: 实验结果
            
        AI_HINT: 执行完整的实验流程
        """
        ...
    
    @abstractmethod
    def setup_experiment(self, config: Dict[str, Any]) -> None:
        """
        设置实验环境
        
        Args:
            config: 实验配置
            
        AI_HINT: 准备实验所需的环境和资源
        """
        ...
    
    @abstractmethod
    def analyze_results(self, results: List[Any]) -> Dict[str, Any]:
        """
        分析实验结果
        
        Args:
            results: 实验结果列表
            
        Returns:
            Dict: 分析报告
            
        AI_HINT: 提供统计分析和洞察
        """
        ...


class PerformanceTracker(Protocol):
    """
    性能跟踪器协议 - AI可以实现这个协议来创建监控组件
    
    AI_INSTRUCTION: 性能跟踪器用于：
    1. track() - 跟踪性能指标
    2. get_metrics() - 获取性能指标
    3. generate_report() - 生成性能报告
    """
    
    @abstractmethod
    def track(self, operation: str, duration: float, metadata: Dict[str, Any]) -> None:
        """
        跟踪性能指标
        
        Args:
            operation: 操作名称
            duration: 执行时长
            metadata: 附加元数据
            
        AI_HINT: 记录系统性能数据
        """
        ...
    
    @abstractmethod
    def get_metrics(self, operation: Optional[str] = None) -> PerformanceMetrics:
        """
        获取性能指标
        
        Args:
            operation: 可选的操作名称过滤
            
        Returns:
            PerformanceMetrics: 性能指标数据
            
        AI_HINT: 提供性能统计信息
        """
        ...
    
    @abstractmethod
    def generate_report(self, format: str = "json") -> str:
        """
        生成性能报告
        
        Args:
            format: 报告格式 ("json", "html", "csv")
            
        Returns:
            str: 格式化的报告内容
            
        AI_HINT: 生成人类可读的性能报告
        """
        ... 