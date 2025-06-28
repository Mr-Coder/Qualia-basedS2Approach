"""
MLR多层推理处理器 - 优化实现

基于工作流程第3阶段的详细规范，实现完整的MLR推理处理器。

AI_CONTEXT: MLR推理的处理器组件，负责工作流程的具体实现
RESPONSIBILITY: 执行5步MLR工作流程，实现状态空间搜索和推理链构建
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ...ai_core.interfaces.data_structures import (MathProblem, OperationType,
                                                   ReasoningResult,
                                                   ReasoningStep)
from ...ai_core.interfaces.exceptions import ReasoningError
from ..strategies.mlr_core import (MLRConfig, ReasoningLevel, ReasoningState,
                                   StateType, calculate_state_heuristic,
                                   check_goal_condition, create_initial_state)


class SearchStrategy(Enum):
    """搜索策略枚举"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    A_STAR = "a_star"
    BEST_FIRST = "best_first"


@dataclass
class MLRWorkflowResult:
    """MLR工作流程结果"""
    
    # 核心输出
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_answer: Any = None
    
    # 性能指标
    overall_confidence: float = 0.0
    execution_time: float = 0.0
    state_path_length: int = 0
    total_states_explored: int = 0
    
    # 工作流程详情
    workflow_stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimization_metrics: Dict[str, float] = field(default_factory=dict)


class MLRProcessor:
    """
    MLR多层推理处理器
    
    AI_CONTEXT: 实现工作流程第3阶段的完整MLR处理流程
    RESPONSIBILITY: 执行5步工作流程，提供高效的推理处理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化MLR处理器"""
        self.config = MLRConfig(**config) if config else MLRConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 状态管理
        self.state_counter = 0
        self.action_counter = 0
        self.visited_states: Set[str] = set()
        self.state_cache: Dict[str, ReasoningState] = {}
        
        # 搜索控制
        self.search_frontier: deque = deque()
        self.explored_nodes: List[ReasoningState] = []
        
        self.logger.info("MLR处理器初始化完成")
    
    def process_mlr_workflow(self, problem_data: Dict[str, Any], 
                           relations: List[Dict[str, Any]]) -> MLRWorkflowResult:
        """
        执行完整的MLR工作流程
        
        Args:
            problem_data: 问题数据（结构化实体列表 + 问题类型）
            relations: 关系列表（从IRD阶段输出）
            
        Returns:
            MLRWorkflowResult: 包含推理步骤序列和中间结果的工作流程结果
            
        AI_INSTRUCTION: 实现工作流程第3阶段的5个核心步骤
        """
        start_time = time.time()
        workflow_stages = {}
        
        try:
            self.logger.info("🚀 开始MLR工作流程处理")
            
            # 阶段1: 目标分解 (Target Decomposition)
            stage1_start = time.time()
            target_analysis = self._stage1_target_decomposition(problem_data)
            workflow_stages["stage1_target_decomposition"] = {
                "result": target_analysis,
                "execution_time": time.time() - stage1_start,
                "success": True
            }
            
            # 阶段2: 推理路径规划 (Reasoning Path Planning)
            stage2_start = time.time()
            reasoning_plan = self._stage2_reasoning_planning(
                problem_data, relations, target_analysis
            )
            workflow_stages["stage2_reasoning_planning"] = {
                "result": reasoning_plan,
                "execution_time": time.time() - stage2_start,
                "success": True
            }
            
            # 阶段3: 状态空间搜索 (State Space Search)
            stage3_start = time.time()
            state_path = self._stage3_state_space_search(
                problem_data, reasoning_plan, target_analysis
            )
            workflow_stages["stage3_state_space_search"] = {
                "result": {"path_length": len(state_path), "states_explored": len(self.visited_states)},
                "execution_time": time.time() - stage3_start,
                "success": len(state_path) > 0
            }
            
            # 阶段4: 逐步推理执行 (Step-by-Step Reasoning)
            stage4_start = time.time()
            reasoning_steps = self._stage4_step_by_step_reasoning(
                state_path, relations, target_analysis
            )
            workflow_stages["stage4_step_by_step_reasoning"] = {
                "result": {"steps_count": len(reasoning_steps)},
                "execution_time": time.time() - stage4_start,
                "success": len(reasoning_steps) > 0
            }
            
            # 阶段5: 中间结果验证 (Intermediate Verification)
            stage5_start = time.time()
            verified_steps, intermediate_results = self._stage5_intermediate_verification(
                reasoning_steps
            )
            workflow_stages["stage5_intermediate_verification"] = {
                "result": {"verified_steps": len(verified_steps), "verification_rate": len(verified_steps) / max(len(reasoning_steps), 1)},
                "execution_time": time.time() - stage5_start,
                "success": len(verified_steps) > 0
            }
            
            # 构建最终结果
            final_answer = self._extract_final_answer(verified_steps, target_analysis)
            execution_time = time.time() - start_time
            
            # 计算优化指标
            optimization_metrics = self._calculate_optimization_metrics(
                workflow_stages, len(state_path), len(self.visited_states)
            )
            
            result = MLRWorkflowResult(
                reasoning_steps=verified_steps,
                intermediate_results=intermediate_results,
                final_answer=final_answer,
                overall_confidence=self._calculate_overall_confidence(verified_steps),
                execution_time=execution_time,
                state_path_length=len(state_path),
                total_states_explored=len(self.visited_states),
                workflow_stages=workflow_stages,
                optimization_metrics=optimization_metrics
            )
            
            self.logger.info(f"✅ MLR工作流程完成: {execution_time:.3f}s, {len(verified_steps)}步")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ MLR工作流程失败: {e}")
            raise ReasoningError(f"MLR工作流程处理失败: {e}")
    
    def _stage1_target_decomposition(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        阶段1: 目标分解
        
        分析问题文本，识别求解目标和操作提示，制定分解策略
        
        Args:
            problem_data: 问题数据
            
        Returns:
            Dict: 目标分析结果
        """
        self.logger.info("🎯 阶段1: 目标分解")
        
        problem_text = problem_data.get("text", "").lower()
        entities = problem_data.get("entities", {})
        problem_type = problem_data.get("type", "arithmetic")
        
        # 目标变量识别
        target_variable = self._identify_target_variable(problem_text, problem_type)
        
        # 操作提示提取
        operation_hints = self._extract_operation_hints(problem_text)
        
        # 分解策略制定
        decomposition_strategy = self._determine_decomposition_strategy(
            entities, operation_hints, problem_type
        )
        
        # 成功标准定义
        success_criteria = {
            "target_found": True,
            "confidence_threshold": 0.8,
            "max_reasoning_steps": min(len(entities) + 3, 10),
            "required_operations": operation_hints
        }
        
        target_analysis = {
            "target_variable": target_variable,
            "operation_hints": operation_hints,
            "decomposition_strategy": decomposition_strategy,
            "success_criteria": success_criteria,
            "problem_complexity": self._assess_problem_complexity(entities, operation_hints),
            "entity_dependencies": self._analyze_entity_dependencies(entities, problem_text)
        }
        
        self.logger.debug(f"目标分解完成: {target_analysis}")
        return target_analysis
    
    def _stage2_reasoning_planning(self, problem_data: Dict[str, Any],
                                 relations: List[Dict[str, Any]],
                                 target_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        阶段2: 推理路径规划
        
        制定分层推理策略，规划L1/L2/L3层的具体操作
        
        Args:
            problem_data: 问题数据
            relations: 关系列表
            target_analysis: 目标分析结果
            
        Returns:
            Dict: 推理规划结果
        """
        self.logger.info("📋 阶段2: 推理路径规划")
        
        entities = problem_data.get("entities", {})
        operation_hints = target_analysis["operation_hints"]
        
        # L1层规划: 直接计算层
        l1_operations = []
        for entity_name, entity_info in entities.items():
            if isinstance(entity_info, dict) and "value" in entity_info:
                l1_operations.append({
                    "operation": "extract_value",
                    "entity": entity_name,
                    "value": entity_info["value"],
                    "confidence": 0.95,
                    "level": "L1"
                })
        
        # L2层规划: 关系应用层
        l2_operations = []
        for relation in relations:
            if relation.get("type") in ["explicit", "implicit"]:
                l2_operations.append({
                    "operation": "apply_relation",
                    "relation": relation,
                    "confidence": relation.get("confidence", 0.8),
                    "level": "L2"
                })
        
        # L3层规划: 目标导向层
        l3_operations = []
        target_var = target_analysis["target_variable"]
        l3_operations.append({
            "operation": "goal_achievement",
            "target_variable": target_var,
            "strategy": target_analysis["decomposition_strategy"],
            "confidence": 0.9,
            "level": "L3"
        })
        
        reasoning_plan = {
            "l1_direct_computation": l1_operations,
            "l2_relational_apply": l2_operations,
            "l3_goal_oriented": l3_operations,
            "overall_strategy": self._determine_overall_strategy(operation_hints),
            "estimated_complexity": target_analysis["problem_complexity"],
            "planning_metadata": {
                "total_operations": len(l1_operations) + len(l2_operations) + len(l3_operations),
                "primary_level": self._determine_primary_level(l1_operations, l2_operations, l3_operations)
            }
        }
        
        self.logger.debug(f"推理规划完成: {len(l1_operations)}L1 + {len(l2_operations)}L2 + {len(l3_operations)}L3")
        return reasoning_plan
    
    def _stage3_state_space_search(self, problem_data: Dict[str, Any],
                                 reasoning_plan: Dict[str, Any],
                                 target_analysis: Dict[str, Any]) -> List[ReasoningState]:
        """
        阶段3: 状态空间搜索
        
        使用A*算法执行状态空间搜索，找到从初始状态到目标状态的最优路径
        
        Args:
            problem_data: 问题数据
            reasoning_plan: 推理规划
            target_analysis: 目标分析
            
        Returns:
            List[ReasoningState]: 状态路径序列
        """
        self.logger.info("🔍 阶段3: 状态空间搜索")
        
        # 创建初始状态
        initial_state = create_initial_state(
            problem_data, f"state_{self.state_counter}"
        )
        self.state_counter += 1
        self.state_cache[initial_state.state_id] = initial_state
        
        # 初始化搜索
        self.search_frontier.clear()
        self.visited_states.clear()
        self.explored_nodes.clear()
        
        self.search_frontier.append(initial_state)
        target_variable = target_analysis["target_variable"]
        
        # A*搜索算法
        max_iterations = self.config.max_iterations
        iteration = 0
        
        while self.search_frontier and iteration < max_iterations:
            iteration += 1
            
            # 选择最优节点（A*策略）
            current_state = self._select_best_state_a_star(target_variable)
            if not current_state:
                break
            
            self.visited_states.add(current_state.state_id)
            self.explored_nodes.append(current_state)
            
            # 检查目标条件
            if check_goal_condition(current_state, target_variable):
                path = self._reconstruct_path(current_state)
                self.logger.debug(f"找到解决方案: {len(path)}步, 探索{len(self.visited_states)}状态")
                return path
            
            # 生成后继状态
            successors = self._generate_successor_states(
                current_state, reasoning_plan, target_analysis
            )
            
            for successor in successors:
                if successor.state_id not in self.visited_states:
                    successor.heuristic_value = calculate_state_heuristic(
                        successor, target_variable
                    )
                    self.search_frontier.append(successor)
                    self.state_cache[successor.state_id] = successor
        
        # 如果没有找到完整解决方案，返回最佳路径
        if self.explored_nodes:
            best_state = min(self.explored_nodes, 
                           key=lambda s: calculate_state_heuristic(s, target_variable))
            path = self._reconstruct_path(best_state)
            self.logger.warning(f"未找到完整解决方案，返回最佳路径: {len(path)}步")
            return path
        
        return [initial_state]
    
    def _stage4_step_by_step_reasoning(self, state_path: List[ReasoningState],
                                     relations: List[Dict[str, Any]],
                                     target_analysis: Dict[str, Any]) -> List[ReasoningStep]:
        """
        阶段4: 逐步推理执行
        
        将状态路径转换为详细的推理步骤序列
        
        Args:
            state_path: 状态路径
            relations: 关系列表
            target_analysis: 目标分析
            
        Returns:
            List[ReasoningStep]: 推理步骤序列
        """
        self.logger.info("🔄 阶段4: 逐步推理执行")
        
        reasoning_steps = []
        
        for i in range(len(state_path) - 1):
            current_state = state_path[i]
            next_state = state_path[i + 1]
            
            # 分析状态变化
            step = self._create_reasoning_step_from_transition(
                i + 1, current_state, next_state, relations
            )
            
            if step:
                reasoning_steps.append(step)
        
        # 如果没有生成步骤，创建默认步骤
        if not reasoning_steps and len(state_path) > 0:
            final_state = state_path[-1]
            target_var = target_analysis["target_variable"]
            
            if target_var in final_state.variables:
                step = ReasoningStep(
                    step_id=1,
                    operation=OperationType.LOGICAL_REASONING,
                    description=f"确定最终答案: {final_state.variables[target_var]}",
                    inputs=final_state.variables,
                    outputs={target_var: final_state.variables[target_var]},
                    confidence=0.9,
                    reasoning="直接从已知信息得出答案",
                    metadata={"level": "L1", "reasoning_type": "direct_inference"}
                )
                reasoning_steps.append(step)
        
        self.logger.debug(f"生成推理步骤: {len(reasoning_steps)}步")
        return reasoning_steps
    
    def _stage5_intermediate_verification(self, reasoning_steps: List[ReasoningStep]) -> Tuple[List[ReasoningStep], Dict[str, Any]]:
        """
        阶段5: 中间结果验证
        
        验证每个推理步骤的正确性，更新置信度
        
        Args:
            reasoning_steps: 原始推理步骤
            
        Returns:
            Tuple[List[ReasoningStep], Dict]: 验证后的步骤和中间结果
        """
        self.logger.info("🔍 阶段5: 中间结果验证")
        
        verified_steps = []
        intermediate_results = {}
        verification_count = 0
        
        for step in reasoning_steps:
            # 验证步骤正确性
            is_valid, verification_details = self._verify_reasoning_step(step)
            
            if is_valid:
                verification_count += 1
                # 更新置信度
                step.confidence = min(step.confidence * 1.05, 1.0)
                step.metadata = step.metadata or {}
                step.metadata["verification_status"] = "verified"
                step.metadata["verification_details"] = verification_details
            else:
                # 降低置信度但保留步骤
                step.confidence = max(step.confidence * 0.8, 0.1)
                step.metadata = step.metadata or {}
                step.metadata["verification_status"] = "unverified"
                step.metadata["verification_details"] = verification_details
            
            verified_steps.append(step)
            
            # 收集中间结果
            for key, value in step.outputs.items():
                intermediate_results[f"step_{step.step_id}_{key}"] = value
        
        # 计算验证统计
        intermediate_results["verification_summary"] = {
            "total_steps": len(reasoning_steps),
            "verified_steps": verification_count,
            "verification_rate": verification_count / max(len(reasoning_steps), 1),
            "average_confidence": sum(s.confidence for s in verified_steps) / max(len(verified_steps), 1)
        }
        
        self.logger.debug(f"验证完成: {verification_count}/{len(reasoning_steps)}步通过验证")
        return verified_steps, intermediate_results
    
    # 辅助方法实现
    
    def _identify_target_variable(self, problem_text: str, problem_type: str) -> str:
        """识别目标变量"""
        if "总" in problem_text or "一共" in problem_text or "total" in problem_text:
            return "total"
        elif "剩" in problem_text or "remaining" in problem_text:
            return "remaining"
        elif "多少" in problem_text or "how many" in problem_text:
            return "answer"
        else:
            return "result"
    
    def _extract_operation_hints(self, problem_text: str) -> List[str]:
        """提取操作提示"""
        hints = []
        if any(word in problem_text for word in ["一共", "总共", "加", "plus", "和"]):
            hints.append("addition")
        if any(word in problem_text for word in ["剩下", "还剩", "减", "minus"]):
            hints.append("subtraction")
        if any(word in problem_text for word in ["倍", "times", "乘", "multiply"]):
            hints.append("multiplication")
        if any(word in problem_text for word in ["平均", "分", "divide", "每"]):
            hints.append("division")
        return hints
    
    def _determine_decomposition_strategy(self, entities: Dict, 
                                        operation_hints: List[str], 
                                        problem_type: str) -> str:
        """确定分解策略"""
        if len(entities) <= 2 and "addition" in operation_hints:
            return "sequential"
        elif len(entities) > 2:
            return "hierarchical"
        else:
            return "direct"
    
    def _assess_problem_complexity(self, entities: Dict, operation_hints: List[str]) -> str:
        """评估问题复杂度"""
        complexity_score = len(entities) + len(operation_hints)
        if complexity_score <= 3:
            return "low"
        elif complexity_score <= 6:
            return "medium"
        else:
            return "high"
    
    def _analyze_entity_dependencies(self, entities: Dict, problem_text: str) -> Dict[str, List[str]]:
        """分析实体依赖关系"""
        dependencies = {}
        entity_names = list(entities.keys())
        
        for entity in entity_names:
            dependencies[entity] = []
            # 简单的依赖分析：如果两个实体在同一句子中提到，可能有依赖关系
            for other_entity in entity_names:
                if entity != other_entity and entity in problem_text and other_entity in problem_text:
                    dependencies[entity].append(other_entity)
        
        return dependencies
    
    def _determine_overall_strategy(self, operation_hints: List[str]) -> str:
        """确定整体策略"""
        if "addition" in operation_hints:
            return "additive"
        elif "multiplication" in operation_hints:
            return "multiplicative"
        else:
            return "composite"
    
    def _determine_primary_level(self, l1_ops: List, l2_ops: List, l3_ops: List) -> str:
        """确定主要推理层次"""
        if len(l2_ops) > len(l1_ops) and len(l2_ops) > len(l3_ops):
            return "L2"
        elif len(l3_ops) > len(l1_ops):
            return "L3"
        else:
            return "L1"
    
    def _select_best_state_a_star(self, target_variable: str) -> Optional[ReasoningState]:
        """A*算法选择最优状态"""
        if not self.search_frontier:
            return None
        
        # 计算f(n) = g(n) + h(n)
        best_state = None
        best_score = float('inf')
        best_index = -1
        
        for i, state in enumerate(self.search_frontier):
            if state.state_id in self.visited_states:
                continue
            
            g_score = state.path_cost  # 已知代价
            h_score = calculate_state_heuristic(state, target_variable)  # 启发式估值
            f_score = g_score + h_score
            
            if f_score < best_score:
                best_score = f_score
                best_state = state
                best_index = i
        
        if best_state and best_index >= 0:
            del self.search_frontier[best_index]
            return best_state
        
        return None
    
    def _generate_successor_states(self, current_state: ReasoningState,
                                 reasoning_plan: Dict[str, Any],
                                 target_analysis: Dict[str, Any]) -> List[ReasoningState]:
        """生成后继状态"""
        successors = []
        
        # 尝试应用L1操作
        for operation in reasoning_plan.get("l1_direct_computation", []):
            successor = self._apply_l1_operation(current_state, operation)
            if successor:
                successors.append(successor)
        
        # 尝试应用L2操作
        for operation in reasoning_plan.get("l2_relational_apply", []):
            successor = self._apply_l2_operation(current_state, operation)
            if successor:
                successors.append(successor)
        
        # 尝试应用L3操作
        for operation in reasoning_plan.get("l3_goal_oriented", []):
            successor = self._apply_l3_operation(current_state, operation, target_analysis)
            if successor:
                successors.append(successor)
        
        return successors
    
    def _apply_l1_operation(self, state: ReasoningState, operation: Dict) -> Optional[ReasoningState]:
        """应用L1层操作"""
        if operation["operation"] == "extract_value":
            entity = operation["entity"]
            value = operation["value"]
            
            new_variables = state.variables.copy()
            new_variables[entity] = value
            
            successor = ReasoningState(
                state_id=f"state_{self.state_counter}",
                state_type=StateType.INTERMEDIATE,
                variables=new_variables,
                constraints=state.constraints.copy(),
                parent_state=state.state_id,
                path_cost=state.path_cost + 0.5,
                level=ReasoningLevel.L1_DIRECT,
                confidence=state.confidence * operation["confidence"]
            )
            
            self.state_counter += 1
            return successor
        
        return None
    
    def _apply_l2_operation(self, state: ReasoningState, operation: Dict) -> Optional[ReasoningState]:
        """应用L2层操作"""
        if operation["operation"] == "apply_relation":
            relation = operation["relation"]
            
            # 检查关系是否适用
            if self._is_relation_applicable(relation, state):
                result = self._evaluate_relation(relation, state)
                if result:
                    new_variables = state.variables.copy()
                    new_variables.update(result)
                    
                    successor = ReasoningState(
                        state_id=f"state_{self.state_counter}",
                        state_type=StateType.INTERMEDIATE,
                        variables=new_variables,
                        constraints=state.constraints.copy(),
                        parent_state=state.state_id,
                        path_cost=state.path_cost + 1.0,
                        level=ReasoningLevel.L2_RELATIONAL,
                        confidence=state.confidence * operation["confidence"]
                    )
                    
                    self.state_counter += 1
                    return successor
        
        return None
    
    def _apply_l3_operation(self, state: ReasoningState, operation: Dict, 
                          target_analysis: Dict) -> Optional[ReasoningState]:
        """应用L3层操作"""
        if operation["operation"] == "goal_achievement":
            target_var = operation["target_variable"]
            
            # 尝试从现有变量推导目标
            if self._can_derive_target(state, target_var):
                derived_value = self._derive_target_value(state, target_var)
                
                new_variables = state.variables.copy()
                new_variables[target_var] = derived_value
                
                successor = ReasoningState(
                    state_id=f"state_{self.state_counter}",
                    state_type=StateType.GOAL,
                    variables=new_variables,
                    constraints=state.constraints.copy(),
                    parent_state=state.state_id,
                    path_cost=state.path_cost + 0.5,
                    level=ReasoningLevel.L3_GOAL_ORIENTED,
                    confidence=state.confidence * operation["confidence"]
                )
                
                self.state_counter += 1
                return successor
        
        return None
    
    def _is_relation_applicable(self, relation: Dict, state: ReasoningState) -> bool:
        """检查关系是否适用"""
        var_mapping = relation.get("var_entity", {})
        required_vars = var_mapping.values()
        available_vars = set(state.variables.keys())
        
        return all(var in available_vars for var in required_vars if var)
    
    def _evaluate_relation(self, relation: Dict, state: ReasoningState) -> Optional[Dict[str, Any]]:
        """评估关系"""
        try:
            relation_expr = relation.get("relation", "")
            var_mapping = relation.get("var_entity", {})
            
            # 简单的数学表达式解析和计算
            if "=" in relation_expr and "+" in relation_expr:
                # 处理加法关系: total = a + b
                parts = relation_expr.split("=")
                if len(parts) == 2:
                    result_var = parts[0].strip()
                    expr = parts[1].strip()
                    
                    if "+" in expr:
                        operands = expr.split("+")
                        values = []
                        for operand in operands:
                            operand = operand.strip()
                            if operand in var_mapping:
                                entity_name = var_mapping[operand]
                                if entity_name in state.variables:
                                    values.append(state.variables[entity_name])
                        
                        if len(values) >= 2:
                            result = sum(values)
                            return {result_var: result}
            
        except Exception as e:
            self.logger.warning(f"关系评估失败: {e}")
        
        return None
    
    def _can_derive_target(self, state: ReasoningState, target_var: str) -> bool:
        """检查是否可以推导目标变量"""
        # 如果有total变量且目标是answer/result，可以推导
        if "total" in state.variables and target_var in ["answer", "result"]:
            return True
        
        # 如果已有足够的数值变量，可以进行计算
        numeric_vars = [v for v in state.variables.values() 
                       if isinstance(v, (int, float))]
        
        return len(numeric_vars) >= 2
    
    def _derive_target_value(self, state: ReasoningState, target_var: str) -> Any:
        """推导目标变量值"""
        # 如果有total变量，直接使用
        if "total" in state.variables:
            return state.variables["total"]
        
        # 否则对数值变量求和
        numeric_vars = [v for v in state.variables.values() 
                       if isinstance(v, (int, float))]
        
        if numeric_vars:
            return sum(numeric_vars)
        
        return None
    
    def _reconstruct_path(self, goal_state: ReasoningState) -> List[ReasoningState]:
        """重构状态路径"""
        path = []
        current = goal_state
        
        while current:
            path.append(current)
            if current.parent_state:
                current = self.state_cache.get(current.parent_state)
            else:
                break
        
        return list(reversed(path))
    
    def _create_reasoning_step_from_transition(self, step_id: int,
                                             current_state: ReasoningState,
                                             next_state: ReasoningState,
                                             relations: List[Dict]) -> Optional[ReasoningStep]:
        """从状态转换创建推理步骤"""
        # 分析状态变化
        current_vars = current_state.variables
        next_vars = next_state.variables
        
        changed_vars = {}
        for key, value in next_vars.items():
            if key not in current_vars or current_vars[key] != value:
                changed_vars[key] = value
        
        if changed_vars:
            # 确定操作类型
            operation, description = self._analyze_variable_changes(
                current_vars, changed_vars
            )
            
            step = ReasoningStep(
                step_id=step_id,
                operation=operation,
                description=description,
                inputs=current_vars,
                outputs=changed_vars,
                confidence=next_state.confidence,
                reasoning=f"状态转换: {current_state.state_id} -> {next_state.state_id}",
                metadata={
                    "level": next_state.level.value,
                    "reasoning_type": "state_transition",
                    "path_cost": next_state.path_cost
                }
            )
            
            return step
        
        return None
    
    def _analyze_variable_changes(self, current_vars: Dict, 
                                changed_vars: Dict) -> Tuple[OperationType, str]:
        """分析变量变化，确定操作类型"""
        if "total" in changed_vars:
            # 如果新增了total变量，可能是加法操作
            numeric_values = [v for v in current_vars.values() 
                            if isinstance(v, (int, float))]
            if len(numeric_values) >= 2:
                var_names = [k for k, v in current_vars.items() 
                           if isinstance(v, (int, float))]
                description = f"计算总和: {' + '.join(f'{k}({current_vars[k]})' for k in var_names)} = {changed_vars['total']}"
                return OperationType.ADDITION, description
        
        # 默认为逻辑推理
        var_descriptions = [f"{k}={v}" for k, v in changed_vars.items()]
        description = f"推导结果: {', '.join(var_descriptions)}"
        return OperationType.LOGICAL_REASONING, description
    
    def _verify_reasoning_step(self, step: ReasoningStep) -> Tuple[bool, Dict[str, Any]]:
        """验证推理步骤"""
        verification_details = {
            "mathematical_correctness": True,
            "logical_consistency": True,
            "completeness": True,
            "verification_method": "basic_validation"
        }
        
        try:
            # 基本数学正确性检查
            if step.operation == OperationType.ADDITION:
                inputs = step.inputs
                outputs = step.outputs
                
                if "total" in outputs:
                    expected_total = sum(v for v in inputs.values() 
                                       if isinstance(v, (int, float)))
                    actual_total = outputs["total"]
                    
                    if abs(expected_total - actual_total) > 1e-6:
                        verification_details["mathematical_correctness"] = False
                        verification_details["error"] = f"期望{expected_total}, 实际{actual_total}"
            
            # 逻辑一致性检查
            if step.confidence < 0.0 or step.confidence > 1.0:
                verification_details["logical_consistency"] = False
                verification_details["error"] = f"置信度超出范围: {step.confidence}"
            
            # 完整性检查
            if not step.description or not step.outputs:
                verification_details["completeness"] = False
                verification_details["error"] = "步骤描述或输出缺失"
            
        except Exception as e:
            verification_details["mathematical_correctness"] = False
            verification_details["error"] = f"验证异常: {e}"
        
        is_valid = (verification_details["mathematical_correctness"] and 
                   verification_details["logical_consistency"] and 
                   verification_details["completeness"])
        
        return is_valid, verification_details
    
    def _extract_final_answer(self, verified_steps: List[ReasoningStep], 
                            target_analysis: Dict) -> Any:
        """提取最终答案"""
        target_var = target_analysis["target_variable"]
        
        # 从最后一步的输出中查找答案
        for step in reversed(verified_steps):
            if target_var in step.outputs:
                return step.outputs[target_var]
            
            # 检查常见答案字段
            for key in ["total", "answer", "result", "final_answer"]:
                if key in step.outputs:
                    return step.outputs[key]
        
        return "未找到答案"
    
    def _calculate_overall_confidence(self, verified_steps: List[ReasoningStep]) -> float:
        """计算整体置信度"""
        if not verified_steps:
            return 0.0
        
        # 使用几何平均数
        total_confidence = 1.0
        for step in verified_steps:
            total_confidence *= step.confidence
        
        return total_confidence ** (1.0 / len(verified_steps))
    
    def _calculate_optimization_metrics(self, workflow_stages: Dict,
                                      path_length: int,
                                      states_explored: int) -> Dict[str, float]:
        """计算优化指标"""
        total_time = sum(stage["execution_time"] for stage in workflow_stages.values())
        
        return {
            "search_efficiency": path_length / max(states_explored, 1),
            "average_stage_time": total_time / len(workflow_stages),
            "workflow_success_rate": sum(1 for stage in workflow_stages.values() 
                                       if stage["success"]) / len(workflow_stages),
            "state_space_utilization": states_explored / max(path_length * 10, 1)
        } 