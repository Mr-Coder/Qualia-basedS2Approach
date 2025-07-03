#!/usr/bin/env python3
"""
Intelligent Math Tutor System
Integrating Chain of Responsibility, State Machine, Strategy Composite, and Observer patterns
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# ==================== 基础数据结构 ====================

@dataclass
class StudentState:
    """学生状态数据"""
    student_id: str
    current_level: int = 1
    attempts: int = 0
    correct_answers: int = 0
    total_problems: int = 0
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    current_concept: str = ""
    time_spent: float = 0.0
    frustration_level: float = 0.0  # 0.0-1.0
    
    @property
    def accuracy_rate(self) -> float:
        """计算准确率"""
        return self.correct_answers / max(self.total_problems, 1)
    
    @property
    def needs_encouragement(self) -> bool:
        """是否需要鼓励"""
        return self.frustration_level > 0.7 or self.accuracy_rate < 0.3


@dataclass
class ProblemContext:
    """问题上下文"""
    problem_text: str
    problem_id: str
    difficulty_level: int
    concept_tags: List[str]
    expected_answer: str
    solution_steps: List[str] = field(default_factory=list)
    hints_available: List[str] = field(default_factory=list)
    similar_problems: List[str] = field(default_factory=list)
    
    def get_step_solution(self, step_index: int = 0) -> str:
        """获取部分解答"""
        if step_index < len(self.solution_steps):
            return self.solution_steps[step_index]
        return "暂无步骤解答"
    
    def get_full_solution(self) -> str:
        """获取完整解答"""
        return "\n".join(self.solution_steps)


@dataclass
class TutorResponse:
    """辅导响应"""
    message: str
    response_type: str  # "hint", "partial", "full", "encouragement", "explanation"
    confidence_level: float
    next_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 1. 责任链模式 - 渐进式解题辅导 ====================

class Handler(ABC):
    """责任链处理器基类"""
    
    def __init__(self):
        self.next_handler = None
    
    def set_next(self, handler: 'Handler') -> 'Handler':
        """设置下一个处理器"""
        self.next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, student_state: StudentState, problem: ProblemContext) -> Optional[TutorResponse]:
        """处理请求"""
        pass
    
    def _pass_to_next(self, student_state: StudentState, problem: ProblemContext) -> Optional[TutorResponse]:
        """传递给下一个处理器"""
        if self.next_handler:
            return self.next_handler.handle(student_state, problem)
        return None


class EncouragementHandler(Handler):
    """鼓励处理器 - 处理学生挫折情绪"""
    
    def handle(self, student_state: StudentState, problem: ProblemContext) -> Optional[TutorResponse]:
        if student_state.needs_encouragement:
            encouragement_messages = [
                "别担心，每个人学习新概念都需要时间！",
                "你已经很努力了，让我们一步一步来解决这个问题。",
                "记住，错误是学习的一部分。让我们看看哪里可以改进。",
                "你之前解决过类似的问题，相信这次也能成功！"
            ]
            import random
            message = random.choice(encouragement_messages)
            return TutorResponse(
                message=message,
                response_type="encouragement",
                confidence_level=0.9
            )
        return self._pass_to_next(student_state, problem)


class HintHandler(Handler):
    """提示处理器 - 提供概念提示"""
    
    def handle(self, student_state: StudentState, problem: ProblemContext) -> Optional[TutorResponse]:
        if student_state.attempts == 0 and student_state.frustration_level < 0.5:
            # 首次尝试，提供概念提示
            hint_message = f"💡 提示：这个问题涉及 {', '.join(problem.concept_tags)} 概念。"
            if problem.hints_available:
                hint_message += f"\n思考提示：{problem.hints_available[0]}"
            
            return TutorResponse(
                message=hint_message,
                response_type="hint",
                confidence_level=0.8,
                next_action="try_solve"
            )
        return self._pass_to_next(student_state, problem)


class PartialSolutionHandler(Handler):
    """部分解答处理器 - 提供步骤指导"""
    
    def handle(self, student_state: StudentState, problem: ProblemContext) -> Optional[TutorResponse]:
        if student_state.attempts <= 2:
            step_index = min(student_state.attempts, len(problem.solution_steps) - 1)
            partial_solution = problem.get_step_solution(step_index)
            
            message = f"📝 让我们看看第 {step_index + 1} 步：\n{partial_solution}"
            if step_index < len(problem.solution_steps) - 1:
                message += "\n\n现在尝试完成下一步！"
            
            return TutorResponse(
                message=message,
                response_type="partial",
                confidence_level=0.7,
                next_action="continue_solving"
            )
        return self._pass_to_next(student_state, problem)


class FullSolutionHandler(Handler):
    """完整解答处理器 - 提供完整解答"""
    
    def handle(self, student_state: StudentState, problem: ProblemContext) -> Optional[TutorResponse]:
        full_solution = problem.get_full_solution()
        message = f"🎯 完整解答：\n{full_solution}\n\n现在你明白了吗？"
        
        return TutorResponse(
            message=message,
            response_type="full",
            confidence_level=0.9,
            next_action="explain_concept"
        )


# ==================== 2. 状态机模式 - 学习状态管理 ====================

class LearningState(ABC):
    """学习状态基类"""
    
    @abstractmethod
    def study(self, context: 'LearningContext', problem: ProblemContext) -> TutorResponse:
        pass
    
    @abstractmethod
    def check_answer(self, context: 'LearningContext', answer: str, expected: str) -> TutorResponse:
        pass


class ExplorationState(LearningState):
    """探索状态 - 学生自主尝试"""
    
    def study(self, context: 'LearningContext', problem: ProblemContext) -> TutorResponse:
        if problem.difficulty_level > context.student_state.current_level + 1:
            context.state = GuidedLearningState()
            return TutorResponse(
                message="这个问题可能有点挑战，让我来指导你。",
                response_type="state_transition",
                confidence_level=0.8
            )
        return TutorResponse(
            message="让我们先试着解决这个问题。记住，错误是学习的一部分！",
            response_type="exploration",
            confidence_level=0.6
        )
    
    def check_answer(self, context: 'LearningContext', answer: str, expected: str) -> TutorResponse:
        is_correct = answer.strip() == expected.strip()
        context.student_state.attempts += 1
        context.student_state.total_problems += 1
        
        if is_correct:
            context.student_state.correct_answers += 1
            context.state = MasteryState()
            return TutorResponse(
                message="🎉 太棒了！你已经掌握了这个概念！",
                response_type="success",
                confidence_level=0.9
            )
        else:
            context.student_state.frustration_level += 0.2
            if context.student_state.attempts >= 3:
                context.state = GuidedLearningState()
                return TutorResponse(
                    message="让我们换个方式来解决这个问题。",
                    response_type="state_transition",
                    confidence_level=0.7
                )
            return TutorResponse(
                message="再试一次，仔细检查你的计算。",
                response_type="retry",
                confidence_level=0.5
            )


class GuidedLearningState(LearningState):
    """指导学习状态 - 提供详细指导"""
    
    def study(self, context: 'LearningContext', problem: ProblemContext) -> TutorResponse:
        steps = problem.get_full_solution()
        message = f"📚 让我来指导你解决这个问题：\n\n{steps}\n\n现在你明白了吗？"
        
        return TutorResponse(
            message=message,
            response_type="guided_learning",
            confidence_level=0.8
        )
    
    def check_answer(self, context: 'LearningContext', answer: str, expected: str) -> TutorResponse:
        is_correct = answer.strip() == expected.strip()
        context.student_state.attempts += 1
        context.student_state.total_problems += 1
        
        if is_correct:
            context.student_state.correct_answers += 1
            context.student_state.frustration_level = max(0, context.student_state.frustration_level - 0.3)
            context.state = MasteryState()
            return TutorResponse(
                message="很好！你已经理解了这个问题。",
                response_type="success",
                confidence_level=0.8
            )
        else:
            context.student_state.frustration_level += 0.1
            return TutorResponse(
                message="让我们再仔细看看解答步骤。",
                response_type="retry",
                confidence_level=0.6
            )


class MasteryState(LearningState):
    """掌握状态 - 学生已掌握概念"""
    
    def study(self, context: 'LearningContext', problem: ProblemContext) -> TutorResponse:
        return TutorResponse(
            message="你已经掌握了这个概念！让我们尝试一个更有挑战性的问题。",
            response_type="mastery",
            confidence_level=0.9
        )
    
    def check_answer(self, context: 'LearningContext', answer: str, expected: str) -> TutorResponse:
        is_correct = answer.strip() == expected.strip()
        context.student_state.total_problems += 1
        
        if is_correct:
            context.student_state.correct_answers += 1
            return TutorResponse(
                message="继续保持！你的理解很准确。",
                response_type="success",
                confidence_level=0.9
            )
        else:
            context.student_state.frustration_level += 0.1
            context.state = GuidedLearningState()
            return TutorResponse(
                message="看来这个问题有点不同，让我重新指导你。",
                response_type="state_transition",
                confidence_level=0.7
            )


class LearningContext:
    """学习上下文 - 管理状态转换"""
    
    def __init__(self, student_state: StudentState):
        self.student_state = student_state
        self.state = ExplorationState()
        self.logger = logging.getLogger(__name__)
    
    def study(self, problem: ProblemContext) -> TutorResponse:
        """学习问题"""
        self.logger.debug(f"Student {self.student_state.student_id} studying problem {problem.problem_id}")
        return self.state.study(self, problem)
    
    def check_answer(self, answer: str, expected: str) -> TutorResponse:
        """检查答案"""
        self.logger.debug(f"Student {self.student_state.student_id} checking answer")
        return self.state.check_answer(self, answer, expected)
    
    def transition_to(self, new_state: LearningState):
        """状态转换"""
        old_state = type(self.state).__name__
        self.state = new_state
        self.logger.info(f"Student {self.student_state.student_id} transitioned from {old_state} to {type(new_state).__name__}")


# ==================== 3. 策略组合模式 - 灵活教学方法 ====================

class TeachingStrategy(ABC):
    """教学策略基类"""
    
    @abstractmethod
    def apply(self, student_state: StudentState, problem: ProblemContext) -> str:
        pass


class ConceptExplanationStrategy(TeachingStrategy):
    """概念解释策略"""
    
    def apply(self, student_state: StudentState, problem: ProblemContext) -> str:
        concept_explanations = {
            "addition": "加法是将两个或多个数合并在一起。",
            "subtraction": "减法是从一个数中减去另一个数。",
            "multiplication": "乘法是重复加法的快捷方式。",
            "division": "除法是将一个数分成相等的部分。",
            "fractions": "分数表示整体的一部分。",
            "percentages": "百分比是分数的一种表示方式，以100为基数。"
        }
        
        explanations = []
        for concept in problem.concept_tags:
            if concept in concept_explanations:
                explanations.append(f"📖 {concept}: {concept_explanations[concept]}")
        
        return "\n".join(explanations) if explanations else ""


class ExampleStrategy(TeachingStrategy):
    """例题策略"""
    
    def apply(self, student_state: StudentState, problem: ProblemContext) -> str:
        if problem.similar_problems:
            return f"📋 相关例题：\n{problem.similar_problems[0]}"
        return ""


class VisualAidStrategy(TeachingStrategy):
    """视觉辅助策略"""
    
    def apply(self, student_state: StudentState, problem: ProblemContext) -> str:
        visual_aids = {
            "addition": "🔢 想象你有一些苹果，再拿来一些苹果，现在总共有多少个？",
            "subtraction": "🍎 你有10个苹果，吃了3个，还剩几个？",
            "multiplication": "📦 你有3个盒子，每个盒子有4个糖果，总共有多少个糖果？",
            "division": "🍪 你有12个饼干，要分给3个朋友，每人分几个？"
        }
        
        for concept in problem.concept_tags:
            if concept in visual_aids:
                return f"🎨 {visual_aids[concept]}"
        return ""


class CompositeStrategy(TeachingStrategy):
    """组合策略"""
    
    def __init__(self):
        self.strategies: List[TeachingStrategy] = []
    
    def add_strategy(self, strategy: TeachingStrategy):
        """添加策略"""
        self.strategies.append(strategy)
    
    def apply(self, student_state: StudentState, problem: ProblemContext) -> str:
        """应用所有策略"""
        results = []
        for strategy in self.strategies:
            result = strategy.apply(student_state, problem)
            if result:
                results.append(result)
        return "\n\n".join(results)


# ==================== 4. 观察者模式 - 实时反馈系统 ====================

class LearningObserver(ABC):
    """学习观察者基类"""
    
    @abstractmethod
    def update(self, event_type: str, data: Dict[str, Any]) -> Optional[str]:
        pass


class ProgressTracker(LearningObserver):
    """进度跟踪器"""
    
    def __init__(self):
        self.student_progress = {}
    
    def update(self, event_type: str, data: Dict[str, Any]) -> Optional[str]:
        if event_type == "answer_submitted":
            student_id = data.get("student_id")
            is_correct = data.get("is_correct", False)
            
            if student_id not in self.student_progress:
                self.student_progress[student_id] = {"correct": 0, "total": 0}
            
            self.student_progress[student_id]["total"] += 1
            if is_correct:
                self.student_progress[student_id]["correct"] += 1
            
            accuracy = self.student_progress[student_id]["correct"] / self.student_progress[student_id]["total"]
            
            if accuracy >= 0.8:
                return "🎯 你的进步非常显著！继续保持！"
            elif accuracy >= 0.6:
                return "📈 你正在稳步进步，加油！"
            else:
                return "💪 需要更多练习，但你已经很努力了！"
        
        return None


class AdaptiveDifficultyAdjuster(LearningObserver):
    """自适应难度调整器"""
    
    def update(self, event_type: str, data: Dict[str, Any]) -> Optional[str]:
        if event_type == "consecutive_correct":
            student_state = data.get("student_state")
            if student_state and data.get("count", 0) >= 3:
                return "🌟 连续答对3题！建议尝试更有挑战性的问题。"
        
        elif event_type == "consecutive_incorrect":
            student_state = data.get("student_state")
            if student_state and data.get("count", 0) >= 3:
                return "📚 连续答错3题，建议复习基础概念。"
        
        return None


class EmotionalSupportProvider(LearningObserver):
    """情感支持提供者"""
    
    def update(self, event_type: str, data: Dict[str, Any]) -> Optional[str]:
        if event_type == "frustration_detected":
            frustration_level = data.get("frustration_level", 0)
            if frustration_level > 0.8:
                return "😊 深呼吸，休息一下。学习是一个过程，不要给自己太大压力！"
            elif frustration_level > 0.5:
                return "💪 遇到困难是正常的，让我们一步一步来解决。"
        
        elif event_type == "breakthrough":
            return "🎉 太棒了！你突破了自己的极限！"
        
        return None


class LearningSession:
    """学习会话 - 管理观察者"""
    
    def __init__(self):
        self.observers: List[LearningObserver] = []
        self.logger = logging.getLogger(__name__)
    
    def add_observer(self, observer: LearningObserver):
        """添加观察者"""
        self.observers.append(observer)
    
    def remove_observer(self, observer: LearningObserver):
        """移除观察者"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify(self, event_type: str, data: Dict[str, Any]) -> List[str]:
        """通知所有观察者"""
        responses = []
        for observer in self.observers:
            try:
                response = observer.update(event_type, data)
                if response:
                    responses.append(response)
            except Exception as e:
                self.logger.error(f"Observer {type(observer).__name__} failed: {e}")
        
        return responses


# ==================== 5. 智能辅导系统主类 ====================

class IntelligentTutor:
    """智能数学辅导系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化责任链
        self.solution_chain = self._create_solution_chain()
        
        # 初始化学习会话
        self.learning_session = LearningSession()
        self.learning_session.add_observer(ProgressTracker())
        self.learning_session.add_observer(AdaptiveDifficultyAdjuster())
        self.learning_session.add_observer(EmotionalSupportProvider())
        
        # 学生上下文映射
        self.student_contexts: Dict[str, LearningContext] = {}
        
        # 教学策略
        self.teaching_strategies = self._create_teaching_strategies()
    
    def _create_solution_chain(self) -> Handler:
        """创建责任链"""
        encouragement = EncouragementHandler()
        hint = HintHandler()
        partial = PartialSolutionHandler()
        full = FullSolutionHandler()
        
        encouragement.set_next(hint).set_next(partial).set_next(full)
        return encouragement
    
    def _create_teaching_strategies(self) -> CompositeStrategy:
        """创建教学策略组合"""
        composite = CompositeStrategy()
        composite.add_strategy(ConceptExplanationStrategy())
        composite.add_strategy(ExampleStrategy())
        composite.add_strategy(VisualAidStrategy())
        return composite
    
    def get_or_create_student_context(self, student_id: str) -> LearningContext:
        """获取或创建学生上下文"""
        if student_id not in self.student_contexts:
            student_state = StudentState(student_id=student_id)
            self.student_contexts[student_id] = LearningContext(student_state)
        return self.student_contexts[student_id]
    
    def solve_problem(self, student_id: str, problem: ProblemContext, student_answer: str = "") -> TutorResponse:
        """解决数学问题"""
        context = self.get_or_create_student_context(student_id)
        
        # 1. 使用责任链获取基础响应
        base_response = self.solution_chain.handle(context.student_state, problem)
        
        # 2. 使用状态机处理学习状态
        if student_answer:
            state_response = context.check_answer(student_answer, problem.expected_answer)
        else:
            state_response = context.study(problem)
        
        # 3. 使用策略组合提供教学支持
        teaching_support = self.teaching_strategies.apply(context.student_state, problem)
        
        # 4. 通知观察者并获取反馈
        feedback_data = {
            "student_id": student_id,
            "student_state": context.student_state,
            "problem": problem,
            "is_correct": student_answer == problem.expected_answer if student_answer else None
        }
        
        observer_feedback = self.learning_session.notify("answer_submitted", feedback_data)
        
        # 5. 组合所有响应
        combined_message = base_response.message
        if teaching_support:
            combined_message += f"\n\n{teaching_support}"
        if observer_feedback:
            combined_message += f"\n\n" + "\n".join(observer_feedback)
        
        return TutorResponse(
            message=combined_message,
            response_type=base_response.response_type,
            confidence_level=base_response.confidence_level,
            next_action=base_response.next_action,
            metadata={
                "state_type": type(context.state).__name__,
                "student_level": context.student_state.current_level,
                "accuracy_rate": context.student_state.accuracy_rate,
                "frustration_level": context.student_state.frustration_level
            }
        )
    
    def get_student_progress(self, student_id: str) -> Dict[str, Any]:
        """获取学生进度"""
        context = self.get_or_create_student_context(student_id)
        return {
            "student_id": student_id,
            "current_level": context.student_state.current_level,
            "accuracy_rate": context.student_state.accuracy_rate,
            "total_problems": context.student_state.total_problems,
            "correct_answers": context.student_state.correct_answers,
            "current_state": type(context.state).__name__,
            "frustration_level": context.student_state.frustration_level
        }
    
    def reset_student_progress(self, student_id: str):
        """重置学生进度"""
        if student_id in self.student_contexts:
            del self.student_contexts[student_id]
            self.logger.info(f"Reset progress for student {student_id}")


# ==================== 使用示例 ====================

def create_sample_problem() -> ProblemContext:
    """创建示例问题"""
    return ProblemContext(
        problem_text="小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
        problem_id="addition_001",
        difficulty_level=1,
        concept_tags=["addition", "counting"],
        expected_answer="8",
        solution_steps=[
            "1. 识别问题：这是一个加法问题",
            "2. 提取数字：小明有5个苹果，小红有3个苹果",
            "3. 计算：5 + 3 = 8",
            "4. 答案：他们一共有8个苹果"
        ],
        hints_available=["想想你有几个苹果，再拿来几个苹果，现在总共有多少个？"],
        similar_problems=["小华有4个橘子，小李有2个橘子，他们一共有多少个橘子？"]
    )


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建智能辅导系统
    tutor = IntelligentTutor()
    
    # 创建示例问题
    problem = create_sample_problem()
    
    # 模拟学生学习过程
    student_id = "student_001"
    
    print("🎓 智能数学辅导系统演示")
    print("=" * 50)
    
    # 第一次尝试
    print("\n📝 第一次尝试（无答案）：")
    response1 = tutor.solve_problem(student_id, problem)
    print(f"响应：{response1.message}")
    
    # 提交错误答案
    print("\n❌ 提交错误答案 '6'：")
    response2 = tutor.solve_problem(student_id, problem, "6")
    print(f"响应：{response2.message}")
    
    # 提交正确答案
    print("\n✅ 提交正确答案 '8'：")
    response3 = tutor.solve_problem(student_id, problem, "8")
    print(f"响应：{response3.message}")
    
    # 查看学生进度
    print("\n📊 学生进度：")
    progress = tutor.get_student_progress(student_id)
    for key, value in progress.items():
        print(f"  {key}: {value}") 