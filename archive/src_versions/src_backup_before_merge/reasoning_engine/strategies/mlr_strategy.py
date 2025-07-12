"""
MLR多层推理策略 - 优化实现

基于工作流程第3阶段的详细规范，实现符合AI协作标准的多层推理引擎。

AI_CONTEXT: 多层推理的核心实现，包含状态空间搜索、推理链构建和目标导向推理
RESPONSIBILITY: 构建分层推理链，执行逐步求解
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ...ai_core.interfaces.base_protocols import ReasoningStrategy
from ...ai_core.interfaces.data_structures import (MathProblem, OperationType,
                                                   ProblemComplexity,
                                                   ProblemType,
                                                   ReasoningResult,
                                                   ReasoningStep,
                                                   ValidationResult)
from ...ai_core.interfaces.exceptions import ReasoningError


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
class ReasoningAction:
    """
    推理动作数据结构
    
    AI_CONTEXT: 表示状态转换的操作
    RESPONSIBILITY: 定义如何从一个状态转换到另一个状态
    """
    
    action_id: str = field(metadata={"ai_hint": "动作唯一标识"})
    operation: OperationType = field(metadata={"ai_hint": "操作类型"})
    description: str = field(metadata={"ai_hint": "动作描述"})
    
    # 动作参数
    inputs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "动作输入参数"}
    )
    expected_outputs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "期望输出"}
    )
    
    # 适用条件
    preconditions: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "前置条件"}
    )
    effects: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "动作效果"}
    )
    
    # 评估指标
    cost: float = field(
        default=1.0,
        metadata={"ai_hint": "动作代价"}
    )
    confidence: float = field(
        default=1.0,
        metadata={"ai_hint": "动作置信度"}
    )


class MLRMultiLayerReasoner:
    """
    MLR多层推理器主类
    
    AI_CONTEXT: 实现多层推理的核心引擎
    RESPONSIBILITY: 状态空间搜索、推理链构建、目标导向推理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多层推理器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 推理控制参数
        self.max_iterations = self.config.get('max_iterations', 100)
        self.max_depth = self.config.get('max_depth', 10)
        self.timeout = self.config.get('timeout', 30.0)
        
        # 状态管理
        self.state_counter = 0
        self.action_counter = 0
        self.visited_states: Set[str] = set()
        self.state_cache: Dict[str, ReasoningState] = {}
        
        # 推理路径
        self.frontier: List[ReasoningState] = []
        self.solution_path: List[ReasoningState] = []
        
        self.logger.info("MLR多层推理器初始化完成")
    
    def reason(self, problem: MathProblem, relations: List[Dict[str, Any]]) -> ReasoningResult:
        """
        执行多层推理主流程
        
        Args:
            problem: 数学问题
            relations: 关系列表
            
        Returns:
            ReasoningResult: 推理结果
            
        AI_INSTRUCTION: 这是推理的主入口，实现完整的3层推理架构
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始MLR推理: {problem.id}")
            
            # 步骤1: 目标分解
            target_analysis = self._analyze_target(problem)
            
            # 步骤2: 初始化状态空间
            initial_state = self._create_initial_state(problem, relations)
            goal_state = self._define_goal_state(problem, target_analysis)
            
            # 步骤3: 多层推理执行
            reasoning_chain = self._execute_multi_layer_reasoning(
                initial_state, goal_state, relations, target_analysis
            )
            
            # 步骤4: 构建推理结果
            final_answer = self._extract_final_answer(reasoning_chain, goal_state)
            execution_time = time.time() - start_time
            
            result = ReasoningResult(
                problem_id=problem.id,
                final_answer=final_answer,
                reasoning_steps=reasoning_chain,
                overall_confidence=self._calculate_overall_confidence(reasoning_chain),
                execution_time=execution_time,
                strategy_used="MLR_Multi_Layer_Reasoning",
                metadata={
                    "target_analysis": target_analysis,
                    "states_explored": len(self.visited_states),
                    "reasoning_levels": [step.metadata.get('level', 'L1') for step in reasoning_chain]
                }
            )
            
            self.logger.info(f"MLR推理完成: {execution_time:.3f}s, {len(reasoning_chain)}步")
            return result
            
        except Exception as e:
            self.logger.error(f"MLR推理失败: {e}")
            raise ReasoningError(f"多层推理执行失败: {e}")
    
    def _analyze_target(self, problem: MathProblem) -> Dict[str, Any]:
        """
        目标分析
        
        Args:
            problem: 数学问题
            
        Returns:
            Dict: 目标分析结果
            
        AI_HINT: 分析问题的求解目标和约束条件
        """
        target_analysis = {
            "question_type": self._classify_question_type(problem.text),
            "target_variable": problem.target_variable or self._identify_target_variable(problem.text),
            "complexity_level": problem.complexity,
            "operation_hints": self._extract_operation_hints(problem.text),
            "constraints": problem.constraints,
            "success_criteria": self._define_success_criteria(problem)
        }
        
        self.logger.debug(f"目标分析完成: {target_analysis}")
        return target_analysis
    
    def _create_initial_state(self, problem: MathProblem, relations: List[Dict[str, Any]]) -> ReasoningState:
        """
        创建初始状态
        
        Args:
            problem: 数学问题
            relations: 关系列表
            
        Returns:
            ReasoningState: 初始推理状态
        """
        initial_variables = {}
        
        # 从问题实体中提取已知量
        for entity_name, entity_data in problem.entities.items():
            if isinstance(entity_data, dict) and 'value' in entity_data:
                initial_variables[entity_name] = entity_data['value']
            elif isinstance(entity_data, (int, float)):
                initial_variables[entity_name] = entity_data
        
        initial_state = ReasoningState(
            state_id=f"state_{self.state_counter}",
            state_type=StateType.INITIAL,
            variables=initial_variables,
            constraints=problem.constraints.copy(),
            level=ReasoningLevel.L1_DIRECT,
            confidence=1.0
        )
        
        self.state_counter += 1
        self.state_cache[initial_state.state_id] = initial_state
        
        self.logger.debug(f"初始状态创建: {initial_state.state_id}, 变量: {len(initial_variables)}")
        return initial_state
    
    def _define_goal_state(self, problem: MathProblem, target_analysis: Dict[str, Any]) -> ReasoningState:
        """
        定义目标状态
        
        Args:
            problem: 数学问题
            target_analysis: 目标分析结果
            
        Returns:
            ReasoningState: 目标状态
        """
        target_var = target_analysis["target_variable"]
        goal_variables = {target_var: None}  # None表示待求解
        
        goal_state = ReasoningState(
            state_id="goal_state",
            state_type=StateType.GOAL,
            variables=goal_variables,
            level=ReasoningLevel.L3_GOAL_ORIENTED,
            confidence=1.0
        )
        
        self.logger.debug(f"目标状态定义: {goal_state.state_id}, 目标变量: {target_var}")
        return goal_state
    
    def _execute_multi_layer_reasoning(self, initial_state: ReasoningState, 
                                     goal_state: ReasoningState,
                                     relations: List[Dict[str, Any]],
                                     target_analysis: Dict[str, Any]) -> List[ReasoningStep]:
        """
        执行多层推理
        
        Args:
            initial_state: 初始状态
            goal_state: 目标状态
            relations: 关系列表
            target_analysis: 目标分析
            
        Returns:
            List[ReasoningStep]: 推理步骤序列
            
        AI_INSTRUCTION: 实现三层推理架构的核心逻辑
        """
        reasoning_steps = []
        current_state = initial_state
        step_id = 0
        
        # L1层: 直接计算层
        l1_steps, l1_state = self._execute_l1_direct_computation(
            current_state, relations, step_id
        )
        reasoning_steps.extend(l1_steps)
        step_id += len(l1_steps)
        current_state = l1_state
        
        # L2层: 关系应用层
        l2_steps, l2_state = self._execute_l2_relational_apply(
            current_state, relations, target_analysis, step_id
        )
        reasoning_steps.extend(l2_steps)
        step_id += len(l2_steps)
        current_state = l2_state
        
        # L3层: 目标导向层
        l3_steps, l3_state = self._execute_l3_goal_oriented(
            current_state, goal_state, target_analysis, step_id
        )
        reasoning_steps.extend(l3_steps)
        
        self.logger.info(f"多层推理完成: L1({len(l1_steps)}) + L2({len(l2_steps)}) + L3({len(l3_steps)}) = {len(reasoning_steps)}步")
        return reasoning_steps
    
    def _execute_l1_direct_computation(self, state: ReasoningState, 
                                     relations: List[Dict[str, Any]], 
                                     start_step_id: int) -> Tuple[List[ReasoningStep], ReasoningState]:
        """
        L1层: 直接计算层
        
        处理显式信息和直接计算
        
        Args:
            state: 当前状态
            relations: 关系列表
            start_step_id: 起始步骤ID
            
        Returns:
            Tuple[List[ReasoningStep], ReasoningState]: L1推理步骤和结果状态
        """
        steps = []
        current_state = state
        step_id = start_step_id
        
        self.logger.debug("执行L1层: 直接计算层")
        
        # 处理所有显式数值计算
        explicit_relations = [r for r in relations if r.get('type') == 'explicit']
        
        for relation in explicit_relations:
            if self._is_relation_applicable(relation, current_state):
                step, new_state = self._apply_relation_as_step(
                    relation, current_state, step_id, ReasoningLevel.L1_DIRECT
                )
                
                if step and new_state:
                    steps.append(step)
                    current_state = new_state
                    step_id += 1
        
        # 如果没有显式关系，尝试基础算术操作
        if not steps:
            basic_step = self._create_basic_arithmetic_step(current_state, step_id)
            if basic_step:
                steps.append(basic_step)
                current_state = self._apply_step_to_state(basic_step, current_state)
        
        self.logger.debug(f"L1层完成: {len(steps)}步")
        return steps, current_state
    
    def _execute_l2_relational_apply(self, state: ReasoningState,
                                   relations: List[Dict[str, Any]],
                                   target_analysis: Dict[str, Any],
                                   start_step_id: int) -> Tuple[List[ReasoningStep], ReasoningState]:
        """
        L2层: 关系应用层
        
        应用发现的隐式关系
        
        Args:
            state: 当前状态
            relations: 关系列表
            target_analysis: 目标分析
            start_step_id: 起始步骤ID
            
        Returns:
            Tuple[List[ReasoningStep], ReasoningState]: L2推理步骤和结果状态
        """
        steps = []
        current_state = state
        step_id = start_step_id
        
        self.logger.debug("执行L2层: 关系应用层")
        
        # 应用隐式关系
        implicit_relations = [r for r in relations if r.get('type') == 'implicit']
        
        for relation in implicit_relations:
            if self._is_relation_applicable(relation, current_state):
                step, new_state = self._apply_relation_as_step(
                    relation, current_state, step_id, ReasoningLevel.L2_RELATIONAL
                )
                
                if step and new_state:
                    steps.append(step)
                    current_state = new_state
                    step_id += 1
        
        # 组合中间结果
        if len(current_state.variables) > 1:
            combination_step = self._create_combination_step(
                current_state, target_analysis, step_id
            )
            if combination_step:
                steps.append(combination_step)
                current_state = self._apply_step_to_state(combination_step, current_state)
        
        self.logger.debug(f"L2层完成: {len(steps)}步")
        return steps, current_state
    
    def _execute_l3_goal_oriented(self, state: ReasoningState,
                                goal_state: ReasoningState,
                                target_analysis: Dict[str, Any],
                                start_step_id: int) -> Tuple[List[ReasoningStep], ReasoningState]:
        """
        L3层: 目标导向层
        
        面向最终目标的高阶推理
        
        Args:
            state: 当前状态
            goal_state: 目标状态
            target_analysis: 目标分析
            start_step_id: 起始步骤ID
            
        Returns:
            Tuple[List[ReasoningStep], ReasoningState]: L3推理步骤和结果状态
        """
        steps = []
        current_state = state
        step_id = start_step_id
        
        self.logger.debug("执行L3层: 目标导向层")
        
        target_var = target_analysis["target_variable"]
        
        # 检查是否已达到目标
        if target_var in current_state.variables and current_state.variables[target_var] is not None:
            # 目标已达成，创建结果确认步骤
            confirmation_step = ReasoningStep(
                step_id=step_id,
                operation=OperationType.LOGICAL_REASONING,
                description=f"确认最终答案: {target_var} = {current_state.variables[target_var]}",
                inputs={"target_variable": target_var, "value": current_state.variables[target_var]},
                outputs={"final_answer": current_state.variables[target_var]},
                confidence=0.95,
                reasoning="目标变量已求解，确认为最终答案",
                metadata={
                    "level": "L3",
                    "reasoning_type": "goal_confirmation",
                    "target_achieved": True
                }
            )
            steps.append(confirmation_step)
        else:
            # 需要进一步推理达到目标
            goal_step = self._create_goal_directed_step(
                current_state, target_analysis, step_id
            )
            if goal_step:
                steps.append(goal_step)
                current_state = self._apply_step_to_state(goal_step, current_state)
        
        self.logger.debug(f"L3层完成: {len(steps)}步")
        return steps, current_state
    
    def _is_relation_applicable(self, relation: Dict[str, Any], state: ReasoningState) -> bool:
        """
        检查关系是否适用于当前状态
        
        Args:
            relation: 关系字典
            state: 当前状态
            
        Returns:
            bool: 是否适用
        """
        # 检查关系所需的变量是否在状态中
        required_vars = relation.get('var_entity', {}).values()
        available_vars = set(state.variables.keys())
        
        return all(var in available_vars for var in required_vars if var)
    
    def _apply_relation_as_step(self, relation: Dict[str, Any], 
                              state: ReasoningState,
                              step_id: int,
                              level: ReasoningLevel) -> Tuple[Optional[ReasoningStep], Optional[ReasoningState]]:
        """
        将关系应用转换为推理步骤
        
        Args:
            relation: 关系字典
            state: 当前状态
            step_id: 步骤ID
            level: 推理层次
            
        Returns:
            Tuple[Optional[ReasoningStep], Optional[ReasoningState]]: 推理步骤和新状态
        """
        try:
            # 提取关系信息
            relation_expr = relation.get('relation', '')
            var_mapping = relation.get('var_entity', {})
            
            # 解析数学表达式
            result_var, operation, operands = self._parse_mathematical_expression(
                relation_expr, var_mapping, state.variables
            )
            
            if result_var and operation and operands:
                # 计算结果
                result_value = self._evaluate_operation(operation, operands)
                
                # 创建推理步骤
                step = ReasoningStep(
                    step_id=step_id,
                    operation=operation,
                    description=f"应用关系: {relation_expr} = {result_value}",
                    inputs={f"operand_{i}": val for i, val in enumerate(operands)},
                    outputs={result_var: result_value},
                    confidence=relation.get('confidence', 0.8),
                    reasoning=f"根据关系 {relation_expr} 计算得出",
                    metadata={
                        "level": level.value,
                        "relation_source": relation.get('source_pattern', 'unknown'),
                        "mathematical_expression": relation_expr
                    }
                )
                
                # 创建新状态
                new_variables = state.variables.copy()
                new_variables[result_var] = result_value
                
                new_state = ReasoningState(
                    state_id=f"state_{self.state_counter}",
                    state_type=StateType.INTERMEDIATE,
                    variables=new_variables,
                    constraints=state.constraints.copy(),
                    parent_state=state.state_id,
                    level=level,
                    confidence=state.confidence * step.confidence
                )
                
                self.state_counter += 1
                self.state_cache[new_state.state_id] = new_state
                
                return step, new_state
            
        except Exception as e:
            self.logger.warning(f"关系应用失败: {e}")
        
        return None, None
    
    def _create_basic_arithmetic_step(self, state: ReasoningState, step_id: int) -> Optional[ReasoningStep]:
        """
        创建基础算术步骤
        
        Args:
            state: 当前状态
            step_id: 步骤ID
            
        Returns:
            Optional[ReasoningStep]: 基础算术步骤
        """
        variables = state.variables
        numeric_vars = {k: v for k, v in variables.items() 
                       if isinstance(v, (int, float)) and not math.isnan(v)}
        
        if len(numeric_vars) >= 2:
            var_names = list(numeric_vars.keys())
            values = list(numeric_vars.values())
            
            # 简单加法示例
            result = sum(values)
            
            step = ReasoningStep(
                step_id=step_id,
                operation=OperationType.ADDITION,
                description=f"计算总和: {' + '.join(f'{k}({v})' for k, v in numeric_vars.items())} = {result}",
                inputs=numeric_vars,
                outputs={"total": result},
                confidence=0.9,
                reasoning="对所有数值变量求和",
                metadata={"level": "L1", "reasoning_type": "basic_arithmetic"}
            )
            
            return step
        
        return None
    
    def _create_combination_step(self, state: ReasoningState, 
                               target_analysis: Dict[str, Any], 
                               step_id: int) -> Optional[ReasoningStep]:
        """
        创建组合步骤
        
        Args:
            state: 当前状态
            target_analysis: 目标分析
            step_id: 步骤ID
            
        Returns:
            Optional[ReasoningStep]: 组合步骤
        """
        # 根据目标分析确定组合策略
        operation_hints = target_analysis.get("operation_hints", [])
        
        if "加法" in operation_hints or "总和" in operation_hints:
            return self._create_basic_arithmetic_step(state, step_id)
        
        # 其他组合策略可以在这里添加
        return None
    
    def _create_goal_directed_step(self, state: ReasoningState,
                                 target_analysis: Dict[str, Any],
                                 step_id: int) -> Optional[ReasoningStep]:
        """
        创建目标导向步骤
        
        Args:
            state: 当前状态
            target_analysis: 目标分析
            step_id: 步骤ID
            
        Returns:
            Optional[ReasoningStep]: 目标导向步骤
        """
        target_var = target_analysis["target_variable"]
        
        # 尝试从现有变量推导目标变量
        available_vars = state.variables
        
        # 示例：如果有total变量且目标是最终答案，直接赋值
        if "total" in available_vars and target_var in ["answer", "result", "最终答案"]:
            step = ReasoningStep(
                step_id=step_id,
                operation=OperationType.LOGICAL_REASONING,
                description=f"确定最终答案: {target_var} = {available_vars['total']}",
                inputs={"total": available_vars["total"]},
                outputs={target_var: available_vars["total"]},
                confidence=0.95,
                reasoning="将计算得到的总和作为最终答案",
                metadata={
                    "level": "L3",
                    "reasoning_type": "goal_assignment",
                    "target_variable": target_var
                }
            )
            return step
        
        return None
    
    def _apply_step_to_state(self, step: ReasoningStep, state: ReasoningState) -> ReasoningState:
        """
        将推理步骤应用到状态
        
        Args:
            step: 推理步骤
            state: 当前状态
            
        Returns:
            ReasoningState: 新状态
        """
        new_variables = state.variables.copy()
        new_variables.update(step.outputs)
        
        new_state = ReasoningState(
            state_id=f"state_{self.state_counter}",
            state_type=StateType.INTERMEDIATE,
            variables=new_variables,
            constraints=state.constraints.copy(),
            parent_state=state.state_id,
            level=ReasoningLevel.L2_RELATIONAL,
            confidence=state.confidence * step.confidence
        )
        
        self.state_counter += 1
        self.state_cache[new_state.state_id] = new_state
        
        return new_state
    
    def _parse_mathematical_expression(self, expression: str, 
                                     var_mapping: Dict[str, str],
                                     variables: Dict[str, Any]) -> Tuple[Optional[str], Optional[OperationType], Optional[List[float]]]:
        """
        解析数学表达式
        
        Args:
            expression: 数学表达式
            var_mapping: 变量映射
            variables: 状态变量
            
        Returns:
            Tuple: (结果变量, 操作类型, 操作数列表)
        """
        try:
            # 简单的表达式解析（a = b + c 格式）
            if '=' in expression and '+' in expression:
                left, right = expression.split('=', 1)
                result_var = left.strip()
                
                if '+' in right:
                    operand_vars = [v.strip() for v in right.split('+')]
                    operands = []
                    
                    for var in operand_vars:
                        mapped_var = var_mapping.get(var, var)
                        if mapped_var in variables:
                            operands.append(float(variables[mapped_var]))
                    
                    if len(operands) == len(operand_vars):
                        return result_var, OperationType.ADDITION, operands
            
            elif '=' in expression and '-' in expression:
                left, right = expression.split('=', 1)
                result_var = left.strip()
                
                if '-' in right:
                    operand_vars = [v.strip() for v in right.split('-')]
                    operands = []
                    
                    for i, var in enumerate(operand_vars):
                        mapped_var = var_mapping.get(var, var)
                        if mapped_var in variables:
                            value = float(variables[mapped_var])
                            if i == 0:
                                operands.append(value)
                            else:
                                operands.append(-value)
                    
                    if len(operands) == len(operand_vars):
                        return result_var, OperationType.SUBTRACTION, operands
            
        except Exception as e:
            self.logger.warning(f"表达式解析失败: {expression}, {e}")
        
        return None, None, None
    
    def _evaluate_operation(self, operation: OperationType, operands: List[float]) -> float:
        """
        评估数学操作
        
        Args:
            operation: 操作类型
            operands: 操作数列表
            
        Returns:
            float: 计算结果
        """
        if operation == OperationType.ADDITION:
            return sum(operands)
        elif operation == OperationType.SUBTRACTION:
            return operands[0] - sum(operands[1:]) if len(operands) > 1 else operands[0]
        elif operation == OperationType.MULTIPLICATION:
            result = 1.0
            for operand in operands:
                result *= operand
            return result
        elif operation == OperationType.DIVISION:
            if len(operands) >= 2 and operands[1] != 0:
                return operands[0] / operands[1]
        
        return 0.0
    
    def _classify_question_type(self, text: str) -> str:
        """分类问题类型"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['多少', 'how many', 'how much', 'total']):
            return 'quantity_question'
        elif any(word in text_lower for word in ['时间', 'time', '多长时间', 'how long']):
            return 'time_question'
        elif any(word in text_lower for word in ['比例', 'percentage', '%', '百分之']):
            return 'percentage_question'
        else:
            return 'general_question'
    
    def _identify_target_variable(self, text: str) -> str:
        """识别目标变量"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['时间', 'time', '多长时间', 'how long']):
            return 'time'
        elif any(word in text_lower for word in ['多少', 'how many', 'how much']):
            return 'answer'
        else:
            return 'result'
    
    def _extract_operation_hints(self, text: str) -> List[str]:
        """提取操作提示"""
        hints = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['一共', '总共', 'total', '加', '+']):
            hints.append('加法')
        if any(word in text_lower for word in ['剩下', '还剩', 'remaining', '减', '-']):
            hints.append('减法')
        if any(word in text_lower for word in ['倍', 'times', '乘', '×']):
            hints.append('乘法')
        if any(word in text_lower for word in ['平均', 'average', '除', '÷']):
            hints.append('除法')
        
        return hints
    
    def _define_success_criteria(self, problem: MathProblem) -> Dict[str, Any]:
        """定义成功标准"""
        return {
            "target_found": True,
            "answer_type": "numeric" if problem.problem_type in [ProblemType.ARITHMETIC, ProblemType.ALGEBRA] else "text",
            "confidence_threshold": 0.8,
            "reasoning_steps_min": 1,
            "reasoning_steps_max": 10
        }
    
    def _extract_final_answer(self, reasoning_chain: List[ReasoningStep], 
                            goal_state: ReasoningState) -> Union[str, float, int]:
        """提取最终答案"""
        if reasoning_chain:
            last_step = reasoning_chain[-1]
            
            # 从最后一步的输出中提取答案
            for key, value in last_step.outputs.items():
                if key in ['final_answer', 'answer', 'result', 'total']:
                    return value
            
            # 如果没有明确的答案字段，返回最后一步的第一个输出值
            if last_step.outputs:
                return list(last_step.outputs.values())[0]
        
        return "未找到答案"
    
    def _calculate_overall_confidence(self, reasoning_chain: List[ReasoningStep]) -> float:
        """计算整体置信度"""
        if not reasoning_chain:
            return 0.0
        
        # 使用几何平均数计算整体置信度
        total_confidence = 1.0
        for step in reasoning_chain:
            total_confidence *= step.confidence
        
        return total_confidence ** (1.0 / len(reasoning_chain))


class MLRReasoningStrategy:
    """
    MLR推理策略实现类
    
    实现ReasoningStrategy协议，提供标准化的AI协作接口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化MLR推理策略"""
        self.reasoner = MLRMultiLayerReasoner(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def can_handle(self, problem: MathProblem) -> bool:
        """
        判断此策略是否能处理给定的数学问题
        
        Args:
            problem: 待处理的数学问题
            
        Returns:
            bool: True if can handle, False otherwise
        """
        # MLR策略可以处理大多数类型的数学问题
        supported_types = [
            ProblemType.ARITHMETIC,
            ProblemType.ALGEBRA,
            ProblemType.WORD_PROBLEM,
            ProblemType.MULTI_STEP
        ]
        
        supported_complexities = [
            ProblemComplexity.L0,
            ProblemComplexity.L1,
            ProblemComplexity.L2,
            ProblemComplexity.L3
        ]
        
        return (problem.problem_type in supported_types and 
                problem.complexity in supported_complexities)
    
    def solve(self, problem: MathProblem) -> ReasoningResult:
        """
        解决数学问题
        
        Args:
            problem: 待解决的数学问题
            
        Returns:
            ReasoningResult: 包含推理步骤和最终答案的结果
        """
        # 模拟从IRD阶段获取的关系列表
        relations = self._extract_relations_from_problem(problem)
        
        return self.reasoner.reason(problem, relations)
    
    def get_confidence(self, problem: MathProblem) -> float:
        """
        获取策略对问题的置信度
        
        Args:
            problem: 待评估的数学问题
            
        Returns:
            float: 置信度 [0.0, 1.0]
        """
        base_confidence = 0.8
        
        # 根据问题复杂度调整置信度
        complexity_adjustment = {
            ProblemComplexity.L0: 0.1,
            ProblemComplexity.L1: 0.05,
            ProblemComplexity.L2: 0.0,
            ProblemComplexity.L3: -0.1
        }
        
        # 根据问题类型调整置信度
        type_adjustment = {
            ProblemType.ARITHMETIC: 0.1,
            ProblemType.WORD_PROBLEM: 0.05,
            ProblemType.MULTI_STEP: 0.0,
            ProblemType.ALGEBRA: -0.05
        }
        
        final_confidence = (base_confidence + 
                          complexity_adjustment.get(problem.complexity, 0) +
                          type_adjustment.get(problem.problem_type, 0))
        
        return max(0.0, min(1.0, final_confidence))
    
    def _extract_relations_from_problem(self, problem: MathProblem) -> List[Dict[str, Any]]:
        """从问题中提取关系（模拟IRD阶段的输出）"""
        relations = []
        
        # 基于问题实体创建基础关系
        entities = problem.entities
        if len(entities) >= 2:
            # 创建加法关系示例
            entity_names = list(entities.keys())
            relations.append({
                'type': 'explicit',
                'relation': 'total = a + b',
                'var_entity': {
                    'total': 'total',
                    'a': entity_names[0],
                    'b': entity_names[1] if len(entity_names) > 1 else entity_names[0]
                },
                'confidence': 0.9,
                'source_pattern': 'addition_pattern'
            })
        
        return relations 