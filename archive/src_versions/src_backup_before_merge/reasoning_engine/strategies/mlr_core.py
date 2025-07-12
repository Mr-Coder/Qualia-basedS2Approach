"""
MLR多层推理核心组件

实现MLR推理的基础数据结构和核心组件。

AI_CONTEXT: MLR推理的核心定义和数据结构
RESPONSIBILITY: 提供推理状态、动作和路径的基本定义
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReasoningLevel(Enum):
    """推理层次枚举"""
    L1_DIRECT = "direct_computation"      # L1: 直接计算层
    L2_RELATIONAL = "relational_apply"    # L2: 关系应用层  
    L3_GOAL_ORIENTED = "goal_oriented"    # L3: 目标导向层


class StateType(Enum):
    """状态类型枚举"""
    INITIAL = "initial"           # 初始状态
    INTERMEDIATE = "intermediate" # 中间状态
    GOAL = "goal"                # 目标状态
    DEAD_END = "dead_end"        # 死路状态


@dataclass
class ReasoningState:
    """
    推理状态数据结构
    
    AI_CONTEXT: 表示推理过程中的一个状态
    RESPONSIBILITY: 记录状态的所有信息和转换路径
    """
    
    state_id: str = field(metadata={"ai_hint": "状态唯一标识"})
    state_type: StateType = field(metadata={"ai_hint": "状态类型"})
    
    # 状态数据
    variables: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "状态变量字典"}
    )
    constraints: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "状态约束条件"}
    )
    
    # 路径信息
    parent_state: Optional[str] = field(
        default=None,
        metadata={"ai_hint": "父状态ID"}
    )
    path_cost: float = field(
        default=0.0,
        metadata={"ai_hint": "从初始状态到当前状态的代价"}
    )
    heuristic_value: float = field(
        default=0.0,
        metadata={"ai_hint": "启发式估值"}
    )
    
    # 元数据
    level: ReasoningLevel = field(
        default=ReasoningLevel.L1_DIRECT,
        metadata={"ai_hint": "推理层次"}
    )
    confidence: float = field(
        default=1.0,
        metadata={"ai_hint": "状态置信度"}
    )
    timestamp: float = field(
        default_factory=time.time,
        metadata={"ai_hint": "创建时间戳"}
    )


@dataclass 
class MLRConfig:
    """
    MLR配置数据结构
    
    AI_CONTEXT: MLR推理的配置参数
    RESPONSIBILITY: 控制推理行为的各种参数
    """
    
    # 推理控制
    max_iterations: int = field(
        default=100,
        metadata={"ai_hint": "最大迭代次数"}
    )
    max_depth: int = field(
        default=10,
        metadata={"ai_hint": "最大搜索深度"}
    )
    timeout: float = field(
        default=30.0,
        metadata={"ai_hint": "超时时间（秒）"}
    )
    
    # 置信度阈值
    min_confidence: float = field(
        default=0.1,
        metadata={"ai_hint": "最小置信度阈值"}
    )
    goal_confidence: float = field(
        default=0.8,
        metadata={"ai_hint": "目标达成置信度阈值"}
    )
    
    # 搜索策略
    search_strategy: str = field(
        default="a_star",
        metadata={"ai_hint": "搜索策略名称"}
    )
    use_heuristic: bool = field(
        default=True,
        metadata={"ai_hint": "是否使用启发式搜索"}
    )
    
    # 优化选项
    enable_caching: bool = field(
        default=True,
        metadata={"ai_hint": "是否启用状态缓存"}
    )
    enable_pruning: bool = field(
        default=True,
        metadata={"ai_hint": "是否启用分支剪枝"}
    )


def create_initial_state(problem_data: Dict[str, Any], state_id: str) -> ReasoningState:
    """
    创建初始推理状态
    
    Args:
        problem_data: 问题数据字典
        state_id: 状态ID
        
    Returns:
        ReasoningState: 初始状态
        
    AI_HINT: 从问题数据中提取初始变量和约束
    """
    # 提取问题中的实体作为初始变量
    initial_variables = {}
    entities = problem_data.get("entities", {})
    
    for entity_name, entity_data in entities.items():
        if isinstance(entity_data, dict) and "value" in entity_data:
            initial_variables[entity_name] = entity_data["value"]
        elif isinstance(entity_data, (int, float)):
            initial_variables[entity_name] = entity_data
    
    # 创建初始状态
    initial_state = ReasoningState(
        state_id=state_id,
        state_type=StateType.INITIAL,
        variables=initial_variables,
        constraints=problem_data.get("constraints", []),
        level=ReasoningLevel.L1_DIRECT,
        confidence=1.0
    )
    
    return initial_state


def check_goal_condition(state: ReasoningState, target_variable: str) -> bool:
    """
    检查是否达到目标条件
    
    Args:
        state: 当前状态
        target_variable: 目标变量名
        
    Returns:
        bool: 是否达到目标
        
    AI_HINT: 检查目标变量是否已被求解
    """
    # 检查目标变量是否已求解
    if target_variable in state.variables and state.variables[target_variable] is not None:
        return True
    
    # 检查常见的答案变量
    answer_candidates = ["answer", "result", "total", "final_answer", "solution"]
    for candidate in answer_candidates:
        if candidate in state.variables and state.variables[candidate] is not None:
            return True
    
    return False


def calculate_state_heuristic(state: ReasoningState, target_variable: str) -> float:
    """
    计算状态的启发式值
    
    Args:
        state: 当前状态
        target_variable: 目标变量
        
    Returns:
        float: 启发式值（越小表示越接近目标）
        
    AI_HINT: 基于已知变量数量和目标距离估算启发式值
    """
    variables = state.variables
    
    # 如果已达到目标，启发式值为0
    if check_goal_condition(state, target_variable):
        return 0.0
    
    # 计算已知变量的数量
    known_vars = sum(1 for v in variables.values() if v is not None)
    
    # 估算还需要的步数（简单启发式）
    estimated_steps = max(0, 3 - known_vars)
    
    # 考虑推理层次（层次越高，说明推理越深入）
    level_bonus = {
        ReasoningLevel.L1_DIRECT: 2.0,
        ReasoningLevel.L2_RELATIONAL: 1.0,
        ReasoningLevel.L3_GOAL_ORIENTED: 0.0
    }.get(state.level, 1.0)
    
    return estimated_steps + level_bonus


def validate_state_transition(from_state: ReasoningState, 
                            to_state: ReasoningState) -> bool:
    """
    验证状态转换的有效性
    
    Args:
        from_state: 源状态
        to_state: 目标状态
        
    Returns:
        bool: 转换是否有效
        
    AI_HINT: 检查状态转换是否符合推理逻辑
    """
    # 检查基本约束
    if to_state.path_cost < from_state.path_cost:
        return False  # 代价不能减少
    
    if to_state.confidence > from_state.confidence * 1.1:
        return False  # 置信度不能大幅增加
    
    # 检查变量一致性
    from_vars = from_state.variables
    to_vars = to_state.variables
    
    # 已知变量的值不应该改变
    for var, value in from_vars.items():
        if value is not None and var in to_vars and to_vars[var] != value:
            return False
    
    # 新状态应该有进展（增加了新的变量或修改了未知变量）
    has_progress = False
    for var, value in to_vars.items():
        if var not in from_vars or (from_vars[var] is None and value is not None):
            has_progress = True
            break
    
    return has_progress


def extract_reasoning_operations(state_path: List[ReasoningState]) -> List[str]:
    """
    从状态路径中提取推理操作
    
    Args:
        state_path: 状态路径
        
    Returns:
        List[str]: 推理操作列表
        
    AI_HINT: 分析状态变化以推断执行的操作
    """
    operations = []
    
    for i in range(1, len(state_path)):
        prev_state = state_path[i-1]
        curr_state = state_path[i]
        
        # 分析变量变化
        prev_vars = prev_state.variables
        curr_vars = curr_state.variables
        
        # 找出新增或修改的变量
        changed_vars = []
        for var, value in curr_vars.items():
            if var not in prev_vars or prev_vars[var] != value:
                changed_vars.append((var, value))
        
        if changed_vars:
            # 根据变化推断操作类型
            if len(changed_vars) == 1:
                var_name, var_value = changed_vars[0]
                if isinstance(var_value, (int, float)):
                    # 数值计算
                    operations.append(f"calculate_{var_name}")
                else:
                    # 逻辑推理
                    operations.append(f"infer_{var_name}")
            else:
                # 多变量操作
                operations.append("multi_variable_operation")
        else:
            # 状态更新但变量未变化
            operations.append("state_update")
    
    return operations 