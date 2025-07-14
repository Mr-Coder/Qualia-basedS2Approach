"""
统一问题求解接口

创建标准化的问题求解接口，使用模板方法模式消除代码重复。
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .exceptions import APIError


class ProblemType(Enum):
    """问题类型枚举"""
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    TEXTUAL = "textual"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SolutionStrategy(Enum):
    """解决策略枚举"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    DIRECT_REASONING = "direct_reasoning"
    TEMPLATE_BASED = "template_based"
    HYBRID = "hybrid"


class ProblemInput:
    """标准化问题输入"""
    
    def __init__(
        self, 
        problem: Union[str, Dict[str, Any]],
        problem_type: Optional[ProblemType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        if isinstance(problem, str):
            self.problem_text = problem
            self.problem_data = {"problem": problem}
        elif isinstance(problem, dict):
            self.problem_data = problem
            self.problem_text = problem.get("problem") or problem.get("cleaned_text", "")
        else:
            raise APIError(f"Unsupported problem input type: {type(problem)}")
        
        self.problem_type = problem_type or ProblemType.UNKNOWN
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def get_text(self) -> str:
        """获取问题文本"""
        return self.problem_text
    
    def get_data(self) -> Dict[str, Any]:
        """获取问题数据"""
        return self.problem_data
    
    def get_type(self) -> ProblemType:
        """获取问题类型"""
        return self.problem_type
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "problem_text": self.problem_text,
            "problem_data": self.problem_data,
            "problem_type": self.problem_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class ProblemOutput:
    """标准化问题输出"""
    
    def __init__(
        self,
        final_answer: str,
        confidence: float = 0.0,
        success: bool = True,
        reasoning_steps: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        processing_time: Optional[float] = None
    ):
        self.final_answer = final_answer
        self.confidence = max(0.0, min(1.0, confidence))  # 确保在[0,1]范围内
        self.success = success
        self.reasoning_steps = reasoning_steps or []
        self.metadata = metadata or {}
        self.error = error
        self.processing_time = processing_time
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "success": self.success,
            "reasoning_steps": self.reasoning_steps,
            "metadata": self.metadata,
            "error": self.error,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProblemOutput':
        """从字典创建输出对象"""
        return cls(
            final_answer=data.get("final_answer", ""),
            confidence=data.get("confidence", 0.0),
            success=data.get("success", True),
            reasoning_steps=data.get("reasoning_steps", []),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
            processing_time=data.get("processing_time")
        )


class BaseProblemSolver(ABC):
    """问题求解器基类 - 使用模板方法模式"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._strategy = SolutionStrategy.DIRECT_REASONING
        self._stats = {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "total_processing_time": 0.0
        }
    
    def solve_problem(self, problem: Union[str, Dict[str, Any], ProblemInput]) -> ProblemOutput:
        """
        解决问题的模板方法
        
        定义了问题求解的标准流程：
        1. 预处理
        2. 核心求解
        3. 后处理
        """
        start_time = time.time()
        
        try:
            # 统计更新
            self._stats["total_problems"] += 1
            
            # 步骤1：标准化输入
            standardized_input = self._standardize_input(problem)
            
            # 步骤2：预处理
            preprocessed_input = self.preprocess(standardized_input)
            
            # 步骤3：核心求解（子类实现）
            raw_solution = self.core_solve(preprocessed_input)
            
            # 步骤4：后处理
            final_output = self.postprocess(raw_solution, preprocessed_input)
            
            # 步骤5：验证结果
            validated_output = self.validate_output(final_output)
            
            # 更新处理时间
            processing_time = time.time() - start_time
            validated_output.processing_time = processing_time
            
            # 统计更新
            self._stats["successful_problems"] += 1
            self._stats["total_processing_time"] += processing_time
            
            self.logger.debug(f"问题求解成功，耗时: {processing_time:.3f}秒")
            return validated_output
            
        except Exception as e:
            # 统计更新
            processing_time = time.time() - start_time
            self._stats["failed_problems"] += 1
            self._stats["total_processing_time"] += processing_time
            
            self.logger.error(f"问题求解失败: {e}")
            
            # 返回错误结果
            return ProblemOutput(
                final_answer="求解失败",
                confidence=0.0,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def _standardize_input(self, problem: Union[str, Dict[str, Any], ProblemInput]) -> ProblemInput:
        """标准化输入格式"""
        if isinstance(problem, ProblemInput):
            return problem
        else:
            return ProblemInput(problem)
    
    def preprocess(self, problem_input: ProblemInput) -> ProblemInput:
        """
        预处理步骤（可被子类重写）
        
        默认实现：基本的文本清理和问题类型识别
        """
        # 基本文本清理
        cleaned_text = problem_input.get_text().strip()
        
        # 简单的问题类型识别
        problem_type = self._identify_problem_type(cleaned_text)
        
        # 更新问题输入
        problem_input.problem_type = problem_type
        problem_input.problem_text = cleaned_text
        
        return problem_input
    
    @abstractmethod
    def core_solve(self, problem_input: ProblemInput) -> ProblemOutput:
        """
        核心求解逻辑（子类必须实现）
        
        Args:
            problem_input: 预处理后的问题输入
            
        Returns:
            原始求解结果
        """
        pass
    
    def postprocess(self, raw_output: ProblemOutput, problem_input: ProblemInput) -> ProblemOutput:
        """
        后处理步骤（可被子类重写）
        
        默认实现：基本的结果格式化
        """
        # 添加问题类型信息到元数据
        raw_output.metadata["problem_type"] = problem_input.get_type().value
        raw_output.metadata["solver_class"] = self.__class__.__name__
        
        return raw_output
    
    def validate_output(self, output: ProblemOutput) -> ProblemOutput:
        """
        验证输出结果（可被子类重写）
        
        默认实现：基本的有效性检查
        """
        # 确保必要字段存在
        if not output.final_answer:
            output.final_answer = "无法得出答案"
            output.success = False
        
        # 确保置信度在合理范围内
        output.confidence = max(0.0, min(1.0, output.confidence))
        
        return output
    
    def _identify_problem_type(self, problem_text: str) -> ProblemType:
        """识别问题类型"""
        text_lower = problem_text.lower()
        
        # 简单的规则基础识别
        math_keywords = ["计算", "求", "等于", "+", "-", "*", "/", "数学", "方程"]
        logic_keywords = ["逻辑", "推理", "如果", "那么", "因为", "所以"]
        
        math_score = sum(1 for keyword in math_keywords if keyword in text_lower)
        logic_score = sum(1 for keyword in logic_keywords if keyword in text_lower)
        
        if math_score > logic_score and math_score > 0:
            return ProblemType.MATHEMATICAL
        elif logic_score > 0:
            return ProblemType.LOGICAL
        else:
            return ProblemType.TEXTUAL
    
    def batch_solve(self, problems: List[Union[str, Dict[str, Any], ProblemInput]]) -> List[ProblemOutput]:
        """批量解决问题"""
        results = []
        for problem in problems:
            result = self.solve_problem(problem)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取求解器统计信息"""
        total = self._stats["total_problems"]
        avg_time = self._stats["total_processing_time"] / total if total > 0 else 0.0
        
        return {
            **self._stats,
            "success_rate": self._stats["successful_problems"] / total if total > 0 else 0.0,
            "average_processing_time": avg_time
        }
    
    def set_strategy(self, strategy: SolutionStrategy):
        """设置求解策略"""
        self._strategy = strategy
        self.logger.info(f"求解策略设置为: {strategy.value}")
    
    def get_strategy(self) -> SolutionStrategy:
        """获取当前求解策略"""
        return self._strategy


class ChainOfThoughtSolver(BaseProblemSolver):
    """链式思维求解器实现示例"""
    
    def core_solve(self, problem_input: ProblemInput) -> ProblemOutput:
        """使用链式思维进行求解"""
        problem_text = problem_input.get_text()
        
        # 模拟链式思维步骤
        reasoning_steps = [
            f"分析问题: {problem_text}",
            "识别关键信息",
            "制定求解步骤", 
            "执行计算",
            "验证答案"
        ]
        
        # 简单的示例求解逻辑
        final_answer = f"基于链式思维的答案: {problem_text}"
        
        return ProblemOutput(
            final_answer=final_answer,
            confidence=0.8,
            success=True,
            reasoning_steps=reasoning_steps,
            metadata={"strategy": "chain_of_thought"}
        )


class DirectReasoningSolver(BaseProblemSolver):
    """直接推理求解器实现示例"""
    
    def core_solve(self, problem_input: ProblemInput) -> ProblemOutput:
        """使用直接推理进行求解"""
        problem_text = problem_input.get_text()
        
        # 直接推理逻辑
        final_answer = f"直接推理答案: {problem_text}"
        
        return ProblemOutput(
            final_answer=final_answer,
            confidence=0.7,
            success=True,
            reasoning_steps=["直接分析", "得出结论"],
            metadata={"strategy": "direct_reasoning"}
        )


class ProblemSolverFactory:
    """问题求解器工厂"""
    
    _solvers = {
        SolutionStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtSolver,
        SolutionStrategy.DIRECT_REASONING: DirectReasoningSolver,
    }
    
    @classmethod
    def create_solver(
        cls, 
        strategy: SolutionStrategy,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseProblemSolver:
        """创建问题求解器"""
        if strategy not in cls._solvers:
            raise APIError(f"不支持的求解策略: {strategy}")
        
        solver_class = cls._solvers[strategy]
        return solver_class(config)
    
    @classmethod
    def register_solver(cls, strategy: SolutionStrategy, solver_class: type):
        """注册新的求解器"""
        if not issubclass(solver_class, BaseProblemSolver):
            raise APIError("求解器必须继承自BaseProblemSolver")
        
        cls._solvers[strategy] = solver_class
    
    @classmethod
    def get_available_strategies(cls) -> List[SolutionStrategy]:
        """获取可用策略列表"""
        return list(cls._solvers.keys())


# 便利函数
def create_problem_solver(
    strategy: Union[str, SolutionStrategy] = SolutionStrategy.DIRECT_REASONING,
    config: Optional[Dict[str, Any]] = None
) -> BaseProblemSolver:
    """创建问题求解器的便利函数"""
    if isinstance(strategy, str):
        try:
            strategy = SolutionStrategy(strategy)
        except ValueError:
            raise APIError(f"无效的策略名称: {strategy}")
    
    return ProblemSolverFactory.create_solver(strategy, config)


def solve_problem_unified(
    problem: Union[str, Dict[str, Any]],
    strategy: Union[str, SolutionStrategy] = SolutionStrategy.DIRECT_REASONING,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """统一问题求解函数"""
    solver = create_problem_solver(strategy, config)
    result = solver.solve_problem(problem)
    return result.to_dict()