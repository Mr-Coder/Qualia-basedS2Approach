"""
AI协作友好的数据结构定义

这个模块定义了系统中使用的所有核心数据结构。
AI助手可以通过这些定义理解数据的结构和含义。

AI_CONTEXT: 标准化的数据模型，确保类型安全和清晰的数据流
RESPONSIBILITY: 定义系统中所有核心数据类型
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class ProblemComplexity(Enum):
    """
    问题复杂度枚举
    
    AI_HINT: 用于分类数学问题的难度级别
    """
    L0 = "basic"           # 基础级：简单算术
    L1 = "intermediate"    # 中级：基本代数
    L2 = "advanced"        # 高级：复杂推理
    L3 = "expert"          # 专家级：多步推理


class ProblemType(Enum):
    """
    问题类型枚举
    
    AI_HINT: 用于分类数学问题的具体类型
    """
    ARITHMETIC = "arithmetic"           # 算术问题
    ALGEBRA = "algebra"                # 代数问题
    GEOMETRY = "geometry"              # 几何问题
    WORD_PROBLEM = "word_problem"      # 应用题
    MULTI_STEP = "multi_step"          # 多步骤问题


class OperationType(Enum):
    """
    操作类型枚举
    
    AI_HINT: 推理步骤中的操作类型
    """
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    MULTIPLICATION = "multiplication"
    DIVISION = "division"
    EQUATION_SOLVING = "equation_solving"
    UNIT_CONVERSION = "unit_conversion"
    LOGICAL_REASONING = "logical_reasoning"


@dataclass
class MathProblem:
    """
    数学问题数据结构 - AI协作友好设计
    
    AI_CONTEXT: 表示一个完整的数学问题
    RESPONSIBILITY: 包含问题的所有必要信息
    """
    
    # 基本信息
    id: str = field(metadata={"ai_hint": "问题唯一标识符"})
    text: str = field(metadata={"ai_hint": "问题的文本描述"})
    answer: Optional[Union[str, float, int]] = field(
        default=None, 
        metadata={"ai_hint": "标准答案，可能是数字或文本"}
    )
    
    # 分类信息
    complexity: ProblemComplexity = field(
        default=ProblemComplexity.L0,
        metadata={"ai_hint": "问题复杂度级别"}
    )
    problem_type: ProblemType = field(
        default=ProblemType.ARITHMETIC,
        metadata={"ai_hint": "问题类型分类"}
    )
    
    # 结构化信息
    entities: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "从问题中提取的实体信息"}
    )
    constraints: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "问题的约束条件"}
    )
    target_variable: Optional[str] = field(
        default=None,
        metadata={"ai_hint": "需要求解的目标变量"}
    )
    
    # 元数据
    source: Optional[str] = field(
        default=None,
        metadata={"ai_hint": "问题来源数据集"}
    )
    difficulty_score: Optional[float] = field(
        default=None,
        metadata={"ai_hint": "难度评分 [0.0, 1.0]"}
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "额外的元数据信息"}
    )
    
    def __post_init__(self):
        """AI_HINT: 数据验证和标准化"""
        if isinstance(self.complexity, str):
            self.complexity = ProblemComplexity(self.complexity)
        if isinstance(self.problem_type, str):
            self.problem_type = ProblemType(self.problem_type)


@dataclass
class ReasoningStep:
    """
    推理步骤数据结构
    
    AI_CONTEXT: 表示推理过程中的一个步骤
    RESPONSIBILITY: 记录单个推理操作的详细信息
    """
    
    step_id: int = field(metadata={"ai_hint": "步骤序号"})
    operation: OperationType = field(metadata={"ai_hint": "执行的操作类型"})
    description: str = field(metadata={"ai_hint": "步骤的文字描述"})
    
    # 输入输出
    inputs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "步骤的输入数据"}
    )
    outputs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "步骤的输出结果"}
    )
    
    # 质量指标
    confidence: float = field(
        default=1.0,
        metadata={"ai_hint": "步骤置信度 [0.0, 1.0]"}
    )
    reasoning: str = field(
        default="",
        metadata={"ai_hint": "推理逻辑解释"}
    )
    
    # 验证信息
    is_verified: bool = field(
        default=False,
        metadata={"ai_hint": "步骤是否已验证"}
    )
    verification_method: Optional[str] = field(
        default=None,
        metadata={"ai_hint": "验证方法"}
    )
    
    def __post_init__(self):
        """AI_HINT: 数据验证"""
        if isinstance(self.operation, str):
            self.operation = OperationType(self.operation)
        assert 0.0 <= self.confidence <= 1.0, "置信度必须在 [0.0, 1.0] 范围内"


@dataclass
class ReasoningResult:
    """
    推理结果数据结构
    
    AI_CONTEXT: 表示完整的推理过程和结果
    RESPONSIBILITY: 包含推理的所有步骤和最终答案
    """
    
    problem_id: str = field(metadata={"ai_hint": "关联的问题ID"})
    final_answer: Union[str, float, int] = field(metadata={"ai_hint": "最终答案"})
    
    # 推理过程
    reasoning_steps: List[ReasoningStep] = field(
        default_factory=list,
        metadata={"ai_hint": "完整的推理步骤序列"}
    )
    
    # 质量指标
    overall_confidence: float = field(
        default=1.0,
        metadata={"ai_hint": "整体置信度"}
    )
    execution_time: float = field(
        default=0.0,
        metadata={"ai_hint": "推理耗时（秒）"}
    )
    
    # 策略信息
    strategy_used: str = field(
        default="",
        metadata={"ai_hint": "使用的推理策略"}
    )
    alternative_strategies: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "可选的其他策略"}
    )
    
    # 验证结果
    is_correct: Optional[bool] = field(
        default=None,
        metadata={"ai_hint": "答案是否正确（如果有标准答案）"}
    )
    validation_details: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "详细的验证信息"}
    )
    
    # 元数据
    timestamp: datetime = field(
        default_factory=datetime.now,
        metadata={"ai_hint": "推理完成时间"}
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "额外的元数据"}
    )


@dataclass
class ValidationResult:
    """
    验证结果数据结构
    
    AI_CONTEXT: 表示验证过程的结果
    RESPONSIBILITY: 记录验证的详细信息和建议
    """
    
    is_valid: bool = field(metadata={"ai_hint": "验证是否通过"})
    target_type: str = field(metadata={"ai_hint": "验证目标的类型"})
    
    # 错误信息
    errors: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "发现的错误列表"}
    )
    warnings: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "警告信息列表"}
    )
    
    # 建议信息
    suggestions: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "改进建议列表"}
    )
    fix_recommendations: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "修复建议列表"}
    )
    
    # 验证详情
    validation_method: str = field(
        default="",
        metadata={"ai_hint": "使用的验证方法"}
    )
    confidence_score: float = field(
        default=1.0,
        metadata={"ai_hint": "验证置信度"}
    )
    
    # 元数据
    timestamp: datetime = field(
        default_factory=datetime.now,
        metadata={"ai_hint": "验证时间"}
    )
    details: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "详细的验证信息"}
    )


@dataclass
class ExperimentResult:
    """
    实验结果数据结构
    
    AI_CONTEXT: 表示实验的执行结果
    RESPONSIBILITY: 记录实验的完整信息和分析
    """
    
    experiment_id: str = field(metadata={"ai_hint": "实验唯一标识"})
    experiment_name: str = field(metadata={"ai_hint": "实验名称"})
    start_time: datetime = field(metadata={"ai_hint": "实验开始时间"})
    end_time: datetime = field(metadata={"ai_hint": "实验结束时间"})
    duration: float = field(metadata={"ai_hint": "实验持续时间（秒）"})
    
    # 配置信息
    config: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "实验配置参数"}
    )
    
    # 结果数据
    results: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={"ai_hint": "实验结果数据"}
    )
    metrics: Dict[str, float] = field(
        default_factory=dict,
        metadata={"ai_hint": "性能指标"}
    )
    
    # 分析结果
    analysis: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "结果分析"}
    )
    conclusions: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "实验结论"}
    )
    
    # 状态信息
    status: Literal["completed", "failed", "cancelled"] = field(
        default="completed",
        metadata={"ai_hint": "实验状态"}
    )
    error_message: Optional[str] = field(
        default=None,
        metadata={"ai_hint": "错误信息（如果失败）"}
    )


@dataclass
class PerformanceMetrics:
    """
    性能指标数据结构
    
    AI_CONTEXT: 表示系统性能的详细指标
    RESPONSIBILITY: 记录和分析性能数据
    """
    
    # 基本指标
    operation_count: int = field(
        default=0,
        metadata={"ai_hint": "操作总数"}
    )
    total_duration: float = field(
        default=0.0,
        metadata={"ai_hint": "总耗时（秒）"}
    )
    average_duration: float = field(
        default=0.0,
        metadata={"ai_hint": "平均耗时（秒）"}
    )
    
    # 统计指标
    min_duration: float = field(
        default=0.0,
        metadata={"ai_hint": "最短耗时"}
    )
    max_duration: float = field(
        default=0.0,
        metadata={"ai_hint": "最长耗时"}
    )
    std_duration: float = field(
        default=0.0,
        metadata={"ai_hint": "耗时标准差"}
    )
    
    # 成功率指标
    success_count: int = field(
        default=0,
        metadata={"ai_hint": "成功操作数"}
    )
    failure_count: int = field(
        default=0,
        metadata={"ai_hint": "失败操作数"}
    )
    success_rate: float = field(
        default=1.0,
        metadata={"ai_hint": "成功率 [0.0, 1.0]"}
    )
    
    # 资源使用
    memory_usage: Dict[str, float] = field(
        default_factory=dict,
        metadata={"ai_hint": "内存使用情况（MB）"}
    )
    cpu_usage: Dict[str, float] = field(
        default_factory=dict,
        metadata={"ai_hint": "CPU使用情况（%）"}
    )
    
    # 时间窗口
    measurement_period: Dict[str, datetime] = field(
        default_factory=dict,
        metadata={"ai_hint": "测量时间窗口"}
    )
    
    # 分类指标
    operation_metrics: Dict[str, Dict[str, float]] = field(
        default_factory=dict,
        metadata={"ai_hint": "按操作类型分类的指标"}
    )


# AI_HELPER: 类型别名定义，便于AI理解常用的类型组合
ProblemInput = Union[str, MathProblem]
StrategyOutput = Union[ReasoningResult, None]
ValidationTarget = Union[MathProblem, ReasoningResult, ReasoningStep]
MetricsData = Dict[str, Union[int, float, str]]

# AI_HINT: 这些数据结构设计遵循以下原则：
# 1. 类型安全：使用严格的类型注解
# 2. 自文档化：丰富的字段说明和元数据
# 3. 可扩展性：支持额外的元数据字段
# 4. 验证友好：包含数据验证逻辑
# 5. AI友好：明确的字段含义和使用提示 