"""
MLR多层推理优化演示

基于工作流程第3阶段规范，展示完整的多层推理实现。

AI_CONTEXT: 演示MLR多层推理的完整工作流程
RESPONSIBILITY: 展示状态空间搜索、推理链构建、目标导向推理
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..ai_core.interfaces.data_structures import (MathProblem, OperationType,
                                                  ProblemComplexity,
                                                  ProblemType, ReasoningResult,
                                                  ReasoningStep)
from .strategies.mlr_core import (MLRConfig, ReasoningLevel, ReasoningState,
                                  StateType, calculate_state_heuristic,
                                  check_goal_condition, create_initial_state)


@dataclass
class MLRWorkflowStep:
    """MLR工作流程步骤"""
    
    step_id: int = field(metadata={"ai_hint": "步骤ID"})
    stage: str = field(metadata={"ai_hint": "工作流程阶段"})
    operation: OperationType = field(metadata={"ai_hint": "执行的操作"})
    description: str = field(metadata={"ai_hint": "步骤描述"})
    
    # 状态信息
    input_state: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "输入状态"}
    )
    output_state: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"ai_hint": "输出状态"}
    )
    
    # 推理信息
    reasoning_level: ReasoningLevel = field(
        default=ReasoningLevel.L1_DIRECT,
        metadata={"ai_hint": "推理层次"}
    )
    confidence: float = field(
        default=1.0,
        metadata={"ai_hint": "置信度"}
    )
    execution_time: float = field(
        default=0.0,
        metadata={"ai_hint": "执行耗时"}
    )


class MLREnhancedReasoner:
    """
    MLR增强推理器
    
    实现工作流程第3阶段的多层推理引擎
    """
    
    def __init__(self, config: Optional[MLRConfig] = None):
        """初始化增强推理器"""
        self.config = config or MLRConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 状态管理
        self.states: Dict[str, ReasoningState] = {}
        self.state_counter = 0
        self.step_counter = 0
        
        # 推理历史
        self.reasoning_history: List[MLRWorkflowStep] = []
        
        self.logger.info("MLR增强推理器初始化完成")
    
    def execute_mlr_workflow(self, problem: MathProblem, 
                           relations: List[Dict[str, Any]]) -> ReasoningResult:
        """
        执行MLR工作流程
        
        实现工作流程第3阶段：多层推理
        - 目标分解
        - 推理路径规划  
        - 状态空间搜索
        - 逐步执行推理
        - 中间结果验证
        
        Args:
            problem: 数学问题
            relations: 来自IRD阶段的关系列表
            
        Returns:
            ReasoningResult: 推理结果
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始MLR工作流程: {problem.id}")
            
            # 步骤1: 目标分解
            target_analysis = self._execute_target_decomposition(problem)
            
            # 步骤2: 推理路径规划
            reasoning_plan = self._execute_reasoning_planning(
                problem, relations, target_analysis
            )
            
            # 步骤3: 状态空间搜索
            state_path = self._execute_state_space_search(
                problem, reasoning_plan, target_analysis
            )
            
            # 步骤4: 逐步执行推理
            reasoning_steps = self._execute_step_by_step_reasoning(
                state_path, relations, target_analysis
            )
            
            # 步骤5: 中间结果验证
            verified_steps = self._execute_intermediate_verification(reasoning_steps)
            
            # 构建结果
            final_answer = self._extract_final_answer(verified_steps, target_analysis)
            execution_time = time.time() - start_time
            
            result = ReasoningResult(
                problem_id=problem.id,
                final_answer=final_answer,
                reasoning_steps=verified_steps,
                overall_confidence=self._calculate_overall_confidence(verified_steps),
                execution_time=execution_time,
                strategy_used="MLR_Enhanced_Multi_Layer_Reasoning",
                metadata={
                    "target_analysis": target_analysis,
                    "reasoning_plan": reasoning_plan,
                    "state_path_length": len(state_path),
                    "workflow_steps": len(self.reasoning_history),
                    "reasoning_levels_used": list(set(step.reasoning_level.value for step in self.reasoning_history))
                }
            )
            
            self.logger.info(f"MLR工作流程完成: {execution_time:.3f}s, {len(verified_steps)}步")
            return result
            
        except Exception as e:
            self.logger.error(f"MLR工作流程失败: {e}")
            raise
    
    def _execute_target_decomposition(self, problem: MathProblem) -> Dict[str, Any]:
        """
        步骤1: 目标分解
        
        分析问题目标，确定求解策略
        """
        start_time = time.time()
        
        # 分析问题文本
        problem_text = problem.text.lower()
        
        # 识别求解目标
        target_variable = "answer"
        if "时间" in problem_text or "time" in problem_text:
            target_variable = "time"
        elif "总数" in problem_text or "total" in problem_text:
            target_variable = "total"
        elif "剩余" in problem_text or "remaining" in problem_text:
            target_variable = "remaining"
        
        # 识别操作提示
        operation_hints = []
        if any(word in problem_text for word in ["一共", "总共", "total", "加", "plus"]):
            operation_hints.append("addition")
        if any(word in problem_text for word in ["剩下", "还剩", "remaining", "减", "minus"]):
            operation_hints.append("subtraction")
        if any(word in problem_text for word in ["倍", "times", "乘", "multiply"]):
            operation_hints.append("multiplication")
        
        # 确定复杂度级别
        complexity_score = 0
        if len(problem.entities) > 2:
            complexity_score += 1
        if len(operation_hints) > 1:
            complexity_score += 1
        if problem.complexity in [ProblemComplexity.L2, ProblemComplexity.L3]:
            complexity_score += 1
        
        target_analysis = {
            "target_variable": target_variable,
            "operation_hints": operation_hints,
            "complexity_score": complexity_score,
            "decomposition_strategy": "sequential" if complexity_score <= 1 else "hierarchical",
            "success_criteria": {
                "target_found": True,
                "confidence_threshold": 0.8,
                "step_count_limit": 10
            }
        }
        
        # 记录工作流程步骤
        workflow_step = MLRWorkflowStep(
            step_id=self.step_counter,
            stage="target_decomposition",
            operation=OperationType.LOGICAL_REASONING,
            description="分析问题目标并制定求解策略",
            input_state={"problem": problem.text},
            output_state=target_analysis,
            reasoning_level=ReasoningLevel.L3_GOAL_ORIENTED,
            confidence=0.95,
            execution_time=time.time() - start_time
        )
        self.reasoning_history.append(workflow_step)
        self.step_counter += 1
        
        self.logger.debug(f"目标分解完成: {target_analysis}")
        return target_analysis
    
    def _execute_reasoning_planning(self, problem: MathProblem,
                                  relations: List[Dict[str, Any]],
                                  target_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        步骤2: 推理路径规划
        
        规划从初始状态到目标状态的推理路径
        """
        start_time = time.time()
        
        # 分析可用资源
        available_entities = list(problem.entities.keys())
        available_relations = [r.get('relation', '') for r in relations]
        
        # 规划推理层次
        reasoning_layers = []
        
        # L1层: 直接计算
        l1_operations = []
        for entity_name, entity_data in problem.entities.items():
            if isinstance(entity_data, dict) and "value" in entity_data:
                l1_operations.append(f"extract_{entity_name}")
        reasoning_layers.append({"level": "L1", "operations": l1_operations})
        
        # L2层: 关系应用
        l2_operations = []
        for relation in relations:
            if relation.get('type') == 'arithmetic':
                l2_operations.append(f"apply_{relation.get('operation', 'unknown')}")
        reasoning_layers.append({"level": "L2", "operations": l2_operations})
        
        # L3层: 目标导向
        l3_operations = [f"solve_{target_analysis['target_variable']}"]
        reasoning_layers.append({"level": "L3", "operations": l3_operations})
        
        reasoning_plan = {
            "strategy": target_analysis["decomposition_strategy"],
            "layers": reasoning_layers,
            "estimated_steps": sum(len(layer["operations"]) for layer in reasoning_layers),
            "critical_path": [
                "entity_extraction",
                "relation_application", 
                "target_resolution"
            ]
        }
        
        # 记录工作流程步骤
        workflow_step = MLRWorkflowStep(
            step_id=self.step_counter,
            stage="reasoning_planning",
            operation=OperationType.LOGICAL_REASONING,
            description="规划多层推理路径",
            input_state={"entities": available_entities, "relations": len(relations)},
            output_state=reasoning_plan,
            reasoning_level=ReasoningLevel.L2_RELATIONAL,
            confidence=0.9,
            execution_time=time.time() - start_time
        )
        self.reasoning_history.append(workflow_step)
        self.step_counter += 1
        
        self.logger.debug(f"推理规划完成: {reasoning_plan}")
        return reasoning_plan
    
    def _execute_state_space_search(self, problem: MathProblem,
                                   reasoning_plan: Dict[str, Any],
                                   target_analysis: Dict[str, Any]) -> List[ReasoningState]:
        """
        步骤3: 状态空间搜索
        
        使用A*算法搜索最优推理路径
        """
        start_time = time.time()
        
        # 创建初始状态
        initial_state = create_initial_state(
            problem.__dict__, f"state_{self.state_counter}"
        )
        self.states[initial_state.state_id] = initial_state
        self.state_counter += 1
        
        # 定义目标条件
        target_variable = target_analysis["target_variable"]
        
        def goal_test(state: ReasoningState) -> bool:
            return check_goal_condition(state, target_variable)
        
        # A*搜索
        frontier = [(0, initial_state)]  # (f_score, state)
        visited = set()
        came_from = {}
        g_score = {initial_state.state_id: 0}
        
        while frontier and len(visited) < self.config.max_iterations:
            current_f, current_state = frontier.pop(0)
            
            if current_state.state_id in visited:
                continue
            
            visited.add(current_state.state_id)
            
            # 检查目标
            if goal_test(current_state):
                # 重构路径
                path = []
                state = current_state
                while state:
                    path.append(state)
                    state_id = came_from.get(state.state_id)
                    state = self.states.get(state_id) if state_id else None
                
                path.reverse()
                
                # 记录工作流程步骤
                workflow_step = MLRWorkflowStep(
                    step_id=self.step_counter,
                    stage="state_space_search",
                    operation=OperationType.LOGICAL_REASONING,
                    description="完成状态空间搜索，找到最优路径",
                    input_state={"initial_state": initial_state.state_id},
                    output_state={"path_length": len(path), "goal_reached": True},
                    reasoning_level=ReasoningLevel.L2_RELATIONAL,
                    confidence=0.85,
                    execution_time=time.time() - start_time
                )
                self.reasoning_history.append(workflow_step)
                self.step_counter += 1
                
                self.logger.debug(f"状态空间搜索完成: 路径长度 {len(path)}")
                return path
            
            # 生成后继状态
            for next_state in self._generate_successor_states(current_state, reasoning_plan):
                if next_state.state_id in visited:
                    continue
                
                tentative_g = g_score[current_state.state_id] + 1
                
                if next_state.state_id not in g_score or tentative_g < g_score[next_state.state_id]:
                    came_from[next_state.state_id] = current_state.state_id
                    g_score[next_state.state_id] = tentative_g
                    f_score = tentative_g + calculate_state_heuristic(next_state, target_variable)
                    
                    # 插入到frontier中（保持排序）
                    inserted = False
                    for i, (f, _) in enumerate(frontier):
                        if f_score < f:
                            frontier.insert(i, (f_score, next_state))
                            inserted = True
                            break
                    if not inserted:
                        frontier.append((f_score, next_state))
        
        # 搜索失败，返回空路径
        self.logger.warning("状态空间搜索失败")
        return [initial_state]
    
    def _generate_successor_states(self, current_state: ReasoningState,
                                 reasoning_plan: Dict[str, Any]) -> List[ReasoningState]:
        """生成后继状态"""
        successors = []
        
        # 基于当前状态的变量尝试推理操作
        variables = current_state.variables
        numeric_vars = {k: v for k, v in variables.items() 
                       if isinstance(v, (int, float)) and v is not None}
        
        # 如果有2个或以上数值变量，尝试算术操作
        if len(numeric_vars) >= 2:
            var_names = list(numeric_vars.keys())
            values = list(numeric_vars.values())
            
            # 加法操作
            if "addition" in str(reasoning_plan):
                new_variables = variables.copy()
                new_variables["sum"] = sum(values)
                
                new_state = ReasoningState(
                    state_id=f"state_{self.state_counter}",
                    state_type=StateType.INTERMEDIATE,
                    variables=new_variables,
                    constraints=current_state.constraints,
                    parent_state=current_state.state_id,
                    path_cost=current_state.path_cost + 1,
                    level=ReasoningLevel.L2_RELATIONAL,
                    confidence=current_state.confidence * 0.9
                )
                
                self.states[new_state.state_id] = new_state
                self.state_counter += 1
                successors.append(new_state)
            
            # 减法操作
            if "subtraction" in str(reasoning_plan) and len(values) >= 2:
                new_variables = variables.copy()
                new_variables["difference"] = values[0] - sum(values[1:])
                
                new_state = ReasoningState(
                    state_id=f"state_{self.state_counter}",
                    state_type=StateType.INTERMEDIATE,
                    variables=new_variables,
                    constraints=current_state.constraints,
                    parent_state=current_state.state_id,
                    path_cost=current_state.path_cost + 1,
                    level=ReasoningLevel.L2_RELATIONAL,
                    confidence=current_state.confidence * 0.9
                )
                
                self.states[new_state.state_id] = new_state
                self.state_counter += 1
                successors.append(new_state)
        
        # 目标解析状态
        if any(key in variables for key in ["sum", "difference", "product", "quotient"]):
            new_variables = variables.copy()
            
            # 选择最合适的值作为答案
            if "sum" in variables:
                new_variables["answer"] = variables["sum"]
            elif "difference" in variables:
                new_variables["answer"] = variables["difference"]
            elif "product" in variables:
                new_variables["answer"] = variables["product"]
            elif "quotient" in variables:
                new_variables["answer"] = variables["quotient"]
            
            new_state = ReasoningState(
                state_id=f"state_{self.state_counter}",
                state_type=StateType.GOAL,
                variables=new_variables,
                constraints=current_state.constraints,
                parent_state=current_state.state_id,
                path_cost=current_state.path_cost + 1,
                level=ReasoningLevel.L3_GOAL_ORIENTED,
                confidence=current_state.confidence * 0.95
            )
            
            self.states[new_state.state_id] = new_state
            self.state_counter += 1
            successors.append(new_state)
        
        return successors
    
    def _execute_step_by_step_reasoning(self, state_path: List[ReasoningState],
                                      relations: List[Dict[str, Any]],
                                      target_analysis: Dict[str, Any]) -> List[ReasoningStep]:
        """
        步骤4: 逐步执行推理
        
        将状态路径转换为详细的推理步骤
        """
        start_time = time.time()
        reasoning_steps = []
        
        for i in range(len(state_path) - 1):
            current_state = state_path[i]
            next_state = state_path[i + 1]
            
            step = self._create_reasoning_step_from_states(
                i, current_state, next_state, relations
            )
            
            if step:
                reasoning_steps.append(step)
        
        # 记录工作流程步骤
        workflow_step = MLRWorkflowStep(
            step_id=self.step_counter,
            stage="step_by_step_reasoning",
            operation=OperationType.LOGICAL_REASONING,
            description="执行逐步推理，生成详细推理步骤",
            input_state={"state_path_length": len(state_path)},
            output_state={"reasoning_steps": len(reasoning_steps)},
            reasoning_level=ReasoningLevel.L2_RELATIONAL,
            confidence=0.9,
            execution_time=time.time() - start_time
        )
        self.reasoning_history.append(workflow_step)
        self.step_counter += 1
        
        self.logger.debug(f"逐步推理完成: {len(reasoning_steps)}步")
        return reasoning_steps
    
    def _create_reasoning_step_from_states(self, step_id: int,
                                         current_state: ReasoningState,
                                         next_state: ReasoningState,
                                         relations: List[Dict[str, Any]]) -> Optional[ReasoningStep]:
        """从状态转换创建推理步骤"""
        current_vars = current_state.variables
        next_vars = next_state.variables
        
        # 找出变化的变量
        new_vars = {}
        for var, value in next_vars.items():
            if var not in current_vars or current_vars[var] != value:
                new_vars[var] = value
        
        if not new_vars:
            return None
        
        # 确定操作类型和描述
        if "sum" in new_vars:
            operation = OperationType.ADDITION
            var_names = [k for k, v in current_vars.items() if isinstance(v, (int, float))]
            description = f"计算总和: {' + '.join(f'{k}({current_vars[k]})' for k in var_names)} = {new_vars['sum']}"
        elif "difference" in new_vars:
            operation = OperationType.SUBTRACTION
            description = f"计算差值: 结果 = {new_vars['difference']}"
        elif "answer" in new_vars:
            operation = OperationType.LOGICAL_REASONING
            description = f"确定最终答案: {new_vars['answer']}"
        else:
            operation = OperationType.LOGICAL_REASONING
            description = f"推理步骤: 更新变量 {list(new_vars.keys())}"
        
        step = ReasoningStep(
            step_id=step_id,
            operation=operation,
            description=description,
            inputs=current_vars,
            outputs=new_vars,
            confidence=next_state.confidence,
            reasoning=f"从状态{current_state.state_id}转换到{next_state.state_id}",
            metadata={
                "reasoning_level": next_state.level.value,
                "state_transition": f"{current_state.state_id} -> {next_state.state_id}",
                "variable_changes": new_vars
            }
        )
        
        return step
    
    def _execute_intermediate_verification(self, reasoning_steps: List[ReasoningStep]) -> List[ReasoningStep]:
        """
        步骤5: 中间结果验证
        
        验证每个推理步骤的正确性
        """
        start_time = time.time()
        verified_steps = []
        
        for step in reasoning_steps:
            # 验证步骤
            verification_passed = self._verify_reasoning_step(step)
            
            # 更新步骤的验证状态
            step.is_verified = verification_passed
            step.verification_method = "intermediate_verification"
            
            # 如果验证失败，降低置信度
            if not verification_passed:
                step.confidence *= 0.7
                self.logger.warning(f"步骤{step.step_id}验证失败，置信度降低")
            
            verified_steps.append(step)
        
        # 记录工作流程步骤
        workflow_step = MLRWorkflowStep(
            step_id=self.step_counter,
            stage="intermediate_verification",
            operation=OperationType.LOGICAL_REASONING,
            description="验证中间推理结果",
            input_state={"steps_to_verify": len(reasoning_steps)},
            output_state={"verified_steps": len(verified_steps)},
            reasoning_level=ReasoningLevel.L3_GOAL_ORIENTED,
            confidence=0.95,
            execution_time=time.time() - start_time
        )
        self.reasoning_history.append(workflow_step)
        self.step_counter += 1
        
        self.logger.debug(f"中间结果验证完成: {len(verified_steps)}步")
        return verified_steps
    
    def _verify_reasoning_step(self, step: ReasoningStep) -> bool:
        """验证单个推理步骤"""
        try:
            if step.operation == OperationType.ADDITION:
                # 验证加法
                numeric_inputs = [v for v in step.inputs.values() if isinstance(v, (int, float))]
                if len(numeric_inputs) >= 2:
                    expected_sum = sum(numeric_inputs)
                    actual_sum = list(step.outputs.values())[0] if step.outputs else None
                    return abs(expected_sum - actual_sum) < 1e-10 if actual_sum is not None else False
            
            elif step.operation == OperationType.SUBTRACTION:
                # 验证减法
                numeric_inputs = [v for v in step.inputs.values() if isinstance(v, (int, float))]
                if len(numeric_inputs) >= 2:
                    expected_diff = numeric_inputs[0] - sum(numeric_inputs[1:])
                    actual_diff = list(step.outputs.values())[0] if step.outputs else None
                    return abs(expected_diff - actual_diff) < 1e-10 if actual_diff is not None else False
            
            # 其他操作类型的验证
            return True
            
        except Exception as e:
            self.logger.warning(f"验证步骤{step.step_id}时出错: {e}")
            return False
    
    def _extract_final_answer(self, verified_steps: List[ReasoningStep],
                            target_analysis: Dict[str, Any]) -> Any:
        """提取最终答案"""
        target_variable = target_analysis["target_variable"]
        
        # 从最后一步中提取答案
        if verified_steps:
            last_step = verified_steps[-1]
            
            # 优先查找目标变量
            if target_variable in last_step.outputs:
                return last_step.outputs[target_variable]
            
            # 查找常见答案变量
            for key in ["answer", "result", "sum", "total", "difference"]:
                if key in last_step.outputs:
                    return last_step.outputs[key]
            
            # 返回第一个输出值
            if last_step.outputs:
                return list(last_step.outputs.values())[0]
        
        return "未找到答案"
    
    def _calculate_overall_confidence(self, reasoning_steps: List[ReasoningStep]) -> float:
        """计算整体置信度"""
        if not reasoning_steps:
            return 0.0
        
        # 使用几何平均数
        total_confidence = 1.0
        for step in reasoning_steps:
            total_confidence *= step.confidence
        
        return total_confidence ** (1.0 / len(reasoning_steps))


# 演示程序
def demonstrate_mlr_enhanced_reasoning():
    """演示MLR增强推理功能"""
    
    print("\n" + "="*60)
    print("🚀 MLR多层推理增强演示")
    print("="*60)
    
    # 创建测试问题
    test_problem = MathProblem(
        id="demo_mlr_001",
        text="小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？",
        answer=8,
        complexity=ProblemComplexity.L1,
        problem_type=ProblemType.ARITHMETIC,
        entities={
            "小明苹果": {"value": 3, "type": "number"},
            "小红苹果": {"value": 5, "type": "number"}
        },
        target_variable="answer"
    )
    
    # 模拟IRD阶段的关系
    test_relations = [
        {
            "type": "arithmetic",
            "operation": "addition",
            "relation": "total = a + b",
            "confidence": 0.95,
            "entities": ["小明苹果", "小红苹果"]
        }
    ]
    
    # 创建MLR推理器
    config = MLRConfig(
        max_iterations=50,
        max_depth=8,
        timeout=10.0
    )
    
    reasoner = MLREnhancedReasoner(config)
    
    print(f"\n📋 测试问题: {test_problem.text}")
    print(f"🎯 目标答案: {test_problem.answer}")
    print(f"🔧 关系数量: {len(test_relations)}")
    
    # 执行MLR推理
    try:
        result = reasoner.execute_mlr_workflow(test_problem, test_relations)
        
        print(f"\n✅ 推理完成!")
        print(f"📊 最终答案: {result.final_answer}")
        print(f"🎯 答案正确: {'✓' if result.final_answer == test_problem.answer else '✗'}")
        print(f"📈 整体置信度: {result.overall_confidence:.3f}")
        print(f"⏱️ 执行时间: {result.execution_time:.3f}秒")
        print(f"🔄 推理步数: {len(result.reasoning_steps)}")
        
        print(f"\n📝 推理步骤详情:")
        for i, step in enumerate(result.reasoning_steps, 1):
            print(f"  {i}. {step.description} (置信度: {step.confidence:.2f})")
        
        print(f"\n🔍 工作流程统计:")
        metadata = result.metadata
        print(f"  • 状态路径长度: {metadata.get('state_path_length', 0)}")
        print(f"  • 工作流程步数: {metadata.get('workflow_steps', 0)}")
        print(f"  • 使用的推理层次: {', '.join(metadata.get('reasoning_levels_used', []))}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        return False


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行演示
    demonstrate_mlr_enhanced_reasoning() 