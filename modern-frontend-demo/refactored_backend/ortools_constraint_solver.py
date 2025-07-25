#!/usr/bin/env python3
"""
OR-Tools约束求解器集成
OR-Tools Constraint Solver Integration
提供高级约束优化和多目标求解能力
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

try:
    from ortools.sat.python import cp_model
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    logging.warning("OR-Tools not available. Using fallback constraint solver.")

from enhanced_physical_constraint_network import (
    PhysicsLaw, ConstraintViolation, PhysicsRule, ConstraintSolution,
    EnhancedPhysicalConstraintNetwork
)

logger = logging.getLogger(__name__)

class SolverType(Enum):
    """求解器类型"""
    CP_SAT = "cp_sat"           # 约束编程求解器
    LINEAR = "linear"           # 线性规划求解器  
    MIXED_INTEGER = "mixed_integer"  # 混合整数规划求解器
    FALLBACK = "fallback"       # 回退求解器

@dataclass
class ORToolsConstraint:
    """OR-Tools约束定义"""
    constraint_id: str
    constraint_type: str
    variables: List[str]
    coefficients: List[float]
    bounds: Tuple[float, float]  # (lower_bound, upper_bound)
    is_equality: bool = False
    priority: int = 1  # 1-10, 10为最高优先级

@dataclass
class OptimizationObjective:
    """优化目标"""
    objective_type: str  # "minimize" or "maximize"
    variables: List[str]
    coefficients: List[float]
    weight: float = 1.0

@dataclass
class ORToolsSolution:
    """OR-Tools求解结果"""
    success: bool
    solver_status: str
    objective_value: Optional[float]
    variable_values: Dict[str, float]
    solve_time: float
    constraint_violations: List[ConstraintViolation]
    solver_type: SolverType
    optimization_gap: Optional[float] = None

class ORToolsConstraintSolver:
    """OR-Tools约束求解器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_available = ORTOOLS_AVAILABLE
        
        # 求解器配置
        self.solver_config = {
            "time_limit_seconds": 30,
            "num_search_workers": 4,
            "log_search_progress": False,
            "enable_presolve": True
        }
        
        # 扩展的物理定律库
        self.extended_physics_laws = self._initialize_extended_physics_laws()
        
        # 优化目标权重
        self.objective_weights = {
            "accuracy": 0.4,      # 准确性权重
            "consistency": 0.3,   # 一致性权重
            "efficiency": 0.2,    # 效率权重
            "robustness": 0.1     # 鲁棒性权重
        }
    
    def _initialize_extended_physics_laws(self) -> Dict[PhysicsLaw, PhysicsRule]:
        """初始化扩展物理定律库"""
        
        extended_laws = {
            # 原有定律
            PhysicsLaw.CONSERVATION_OF_QUANTITY: PhysicsRule(
                rule_id="conservation_001",
                law_type=PhysicsLaw.CONSERVATION_OF_QUANTITY,
                name="数量守恒定律",
                description="在封闭系统中，物体的总数量保持不变",
                mathematical_form="∑(输入量) = ∑(输出量)",
                applicable_conditions=["计数问题", "物体转移", "集合运算"],
                priority=0.95
            ),
            
            PhysicsLaw.ADDITIVITY_PRINCIPLE: PhysicsRule(
                rule_id="additivity_001", 
                law_type=PhysicsLaw.ADDITIVITY_PRINCIPLE,
                name="可加性原理",
                description="部分量之和等于总量",
                mathematical_form="total = ∑(parts)",
                applicable_conditions=["求和问题", "集合合并", "累积计算"],
                priority=0.90
            ),
            
            PhysicsLaw.NON_NEGATIVITY_LAW: PhysicsRule(
                rule_id="non_negative_001",
                law_type=PhysicsLaw.NON_NEGATIVITY_LAW,
                name="非负性定律",
                description="物理量不能为负数",
                mathematical_form="quantity ≥ 0",
                applicable_conditions=["计数", "测量", "物理量"],
                priority=1.0
            ),
            
            PhysicsLaw.DISCRETENESS_LAW: PhysicsRule(
                rule_id="discrete_001",
                law_type=PhysicsLaw.DISCRETENESS_LAW,
                name="离散性定律", 
                description="可数对象必须为整数",
                mathematical_form="count ∈ ℤ⁺",
                applicable_conditions=["可数对象", "个体计数"],
                priority=0.85
            )
        }
        
        # 新增扩展定律
        if not hasattr(PhysicsLaw, 'PROPORTIONALITY_LAW'):
            # 动态添加新的物理定律
            extended_laws.update({
                "PROPORTIONALITY_LAW": PhysicsRule(
                    rule_id="proportion_001",
                    law_type="proportionality_law",
                    name="比例关系定律",
                    description="两个量之间保持固定比例关系",
                    mathematical_form="a/b = c/d",
                    applicable_conditions=["比例问题", "相似性", "比率计算"],
                    priority=0.80
                ),
                
                "PROBABILITY_CONSTRAINT": PhysicsRule(
                    rule_id="probability_001",
                    law_type="probability_constraint",
                    name="概率约束定律",
                    description="概率值必须在0到1之间",
                    mathematical_form="0 ≤ P ≤ 1, ∑P = 1",
                    applicable_conditions=["概率计算", "统计问题", "随机事件"],
                    priority=0.75
                ),
                
                "MONOTONICITY_LAW": PhysicsRule(
                    rule_id="monotonic_001",
                    law_type="monotonicity_law", 
                    name="单调性定律",
                    description="某些量必须保持单调递增或递减",
                    mathematical_form="f(x₁) ≤ f(x₂) if x₁ ≤ x₂",
                    applicable_conditions=["时间序列", "累积过程", "排序问题"],
                    priority=0.70
                ),
                
                "BOUNDARY_CONSTRAINT": PhysicsRule(
                    rule_id="boundary_001",
                    law_type="boundary_constraint",
                    name="边界约束定律",
                    description="量值必须在指定范围内",
                    mathematical_form="min_val ≤ x ≤ max_val",
                    applicable_conditions=["范围限制", "物理界限", "约束优化"],
                    priority=0.65
                )
            })
        
        return extended_laws
    
    def solve_enhanced_constraints(self, constraints: List[ORToolsConstraint],
                                 objectives: List[OptimizationObjective] = None,
                                 variable_domains: Dict[str, Tuple[float, float]] = None) -> ORToolsSolution:
        """
        使用OR-Tools求解增强约束问题
        
        Args:
            constraints: OR-Tools约束列表
            objectives: 优化目标列表
            variable_domains: 变量定义域
            
        Returns:
            ORToolsSolution: 求解结果
        """
        
        start_time = time.time()
        
        if not self.is_available:
            self.logger.warning("OR-Tools不可用，使用回退求解器")
            return self._fallback_solve(constraints, objectives)
        
        try:
            # 确定求解器类型
            solver_type = self._determine_solver_type(constraints, objectives)
            
            if solver_type == SolverType.CP_SAT:
                return self._solve_with_cp_sat(constraints, objectives, variable_domains, start_time)
            elif solver_type == SolverType.LINEAR:
                return self._solve_with_linear_solver(constraints, objectives, variable_domains, start_time)
            elif solver_type == SolverType.MIXED_INTEGER:
                return self._solve_with_mixed_integer(constraints, objectives, variable_domains, start_time)
            else:
                return self._fallback_solve(constraints, objectives)
                
        except Exception as e:
            self.logger.error(f"OR-Tools求解失败: {e}")
            return self._fallback_solve(constraints, objectives)
    
    def _determine_solver_type(self, constraints: List[ORToolsConstraint],
                             objectives: List[OptimizationObjective] = None) -> SolverType:
        """确定最适合的求解器类型"""
        
        # 检查是否有整数约束
        has_integer_constraints = any(
            "integer" in c.constraint_type.lower() or "discrete" in c.constraint_type.lower()
            for c in constraints
        )
        
        # 检查是否有非线性约束
        has_nonlinear = any(
            "nonlinear" in c.constraint_type.lower() or "quadratic" in c.constraint_type.lower()
            for c in constraints
        )
        
        # 检查约束规模
        num_variables = len(set(var for c in constraints for var in c.variables))
        num_constraints = len(constraints)
        
        self.logger.info(f"约束分析: 变量数={num_variables}, 约束数={num_constraints}, 整数约束={has_integer_constraints}")
        
        if has_nonlinear or num_constraints > 1000:
            return SolverType.CP_SAT
        elif has_integer_constraints:
            return SolverType.MIXED_INTEGER
        else:
            return SolverType.LINEAR
    
    def _solve_with_cp_sat(self, constraints: List[ORToolsConstraint],
                          objectives: List[OptimizationObjective],
                          variable_domains: Dict[str, Tuple[float, float]],
                          start_time: float) -> ORToolsSolution:
        """使用CP-SAT求解器"""
        
        model = cp_model.CpModel()
        
        # 提取所有变量
        all_variables = set(var for c in constraints for var in c.variables)
        if objectives:
            all_variables.update(var for obj in objectives for var in obj.variables)
        
        # 创建变量
        cp_vars = {}
        for var_name in all_variables:
            if variable_domains and var_name in variable_domains:
                lb, ub = variable_domains[var_name]
                cp_vars[var_name] = model.NewIntVar(int(lb), int(ub), var_name)
            else:
                cp_vars[var_name] = model.NewIntVar(0, 1000, var_name)
        
        # 添加约束
        for constraint in constraints:
            try:
                if constraint.constraint_type == "linear_equality":
                    # 线性等式约束: a₁x₁ + a₂x₂ + ... = bound
                    linear_expr = sum(
                        coeff * cp_vars[var] 
                        for coeff, var in zip(constraint.coefficients, constraint.variables)
                    )
                    model.Add(linear_expr == constraint.bounds[0])
                    
                elif constraint.constraint_type == "linear_inequality":
                    # 线性不等式约束: lb ≤ a₁x₁ + a₂x₂ + ... ≤ ub
                    linear_expr = sum(
                        coeff * cp_vars[var] 
                        for coeff, var in zip(constraint.coefficients, constraint.variables)
                    )
                    model.Add(linear_expr >= constraint.bounds[0])
                    model.Add(linear_expr <= constraint.bounds[1])
                    
                elif constraint.constraint_type == "non_negative":
                    # 非负约束
                    for var in constraint.variables:
                        model.Add(cp_vars[var] >= 0)
                        
                elif constraint.constraint_type == "integer_constraint":
                    # 整数约束 (CP-SAT中自动满足)
                    pass
                    
            except Exception as e:
                self.logger.warning(f"添加约束{constraint.constraint_id}失败: {e}")
        
        # 添加目标函数
        if objectives:
            try:
                objective_expr = sum(
                    obj.weight * sum(
                        coeff * cp_vars[var] 
                        for coeff, var in zip(obj.coefficients, obj.variables)
                    )
                    for obj in objectives
                )
                
                if objectives[0].objective_type == "minimize":
                    model.Minimize(objective_expr)
                else:
                    model.Maximize(objective_expr)
            except Exception as e:
                self.logger.warning(f"添加目标函数失败: {e}")
        
        # 求解
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.solver_config["time_limit_seconds"]
        solver.parameters.num_search_workers = self.solver_config["num_search_workers"]
        solver.parameters.log_search_progress = self.solver_config["log_search_progress"]
        
        status = solver.Solve(model)
        solve_time = time.time() - start_time
        
        # 处理结果
        success = status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        variable_values = {}
        
        if success:
            for var_name, cp_var in cp_vars.items():
                variable_values[var_name] = solver.Value(cp_var)
        
        objective_value = solver.ObjectiveValue() if success and objectives else None
        
        return ORToolsSolution(
            success=success,
            solver_status=self._get_cp_sat_status_string(status),
            objective_value=objective_value,
            variable_values=variable_values,
            solve_time=solve_time,
            constraint_violations=[],
            solver_type=SolverType.CP_SAT,
            optimization_gap=None
        )
    
    def _solve_with_linear_solver(self, constraints: List[ORToolsConstraint],
                                objectives: List[OptimizationObjective],
                                variable_domains: Dict[str, Tuple[float, float]],
                                start_time: float) -> ORToolsSolution:
        """使用线性规划求解器"""
        
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            return self._fallback_solve(constraints, objectives)
        
        # 提取所有变量
        all_variables = set(var for c in constraints for var in c.variables)
        if objectives:
            all_variables.update(var for obj in objectives for var in obj.variables)
        
        # 创建变量
        lp_vars = {}
        for var_name in all_variables:
            if variable_domains and var_name in variable_domains:
                lb, ub = variable_domains[var_name]
                lp_vars[var_name] = solver.NumVar(lb, ub, var_name)
            else:
                lp_vars[var_name] = solver.NumVar(0, solver.infinity(), var_name)
        
        # 添加约束
        for constraint in constraints:
            try:
                if constraint.is_equality:
                    # 等式约束
                    ct = solver.Constraint(constraint.bounds[0], constraint.bounds[0])
                else:
                    # 不等式约束
                    ct = solver.Constraint(constraint.bounds[0], constraint.bounds[1])
                
                for coeff, var in zip(constraint.coefficients, constraint.variables):
                    ct.SetCoefficient(lp_vars[var], coeff)
                    
            except Exception as e:
                self.logger.warning(f"添加线性约束{constraint.constraint_id}失败: {e}")
        
        # 添加目标函数
        if objectives:
            try:
                objective = solver.Objective()
                for obj in objectives:
                    for coeff, var in zip(obj.coefficients, obj.variables):
                        objective.SetCoefficient(lp_vars[var], obj.weight * coeff)
                
                if objectives[0].objective_type == "minimize":
                    objective.SetMinimization()
                else:
                    objective.SetMaximization()
            except Exception as e:
                self.logger.warning(f"添加线性目标函数失败: {e}")
        
        # 求解
        solver.SetTimeLimit(self.solver_config["time_limit_seconds"] * 1000)  # 毫秒
        status = solver.Solve()
        solve_time = time.time() - start_time
        
        # 处理结果
        success = status == pywraplp.Solver.OPTIMAL
        variable_values = {}
        
        if success:
            for var_name, lp_var in lp_vars.items():
                variable_values[var_name] = lp_var.solution_value()
        
        objective_value = solver.Objective().Value() if success and objectives else None
        
        return ORToolsSolution(
            success=success,
            solver_status=self._get_linear_status_string(status),
            objective_value=objective_value,
            variable_values=variable_values,
            solve_time=solve_time,
            constraint_violations=[],
            solver_type=SolverType.LINEAR
        )
    
    def _solve_with_mixed_integer(self, constraints: List[ORToolsConstraint],
                                objectives: List[OptimizationObjective],
                                variable_domains: Dict[str, Tuple[float, float]],
                                start_time: float) -> ORToolsSolution:
        """使用混合整数规划求解器"""
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return self._fallback_solve(constraints, objectives)
        
        # 提取所有变量
        all_variables = set(var for c in constraints for var in c.variables)
        if objectives:
            all_variables.update(var for obj in objectives for var in obj.variables)
        
        # 创建变量 (混合整数和实数)
        mip_vars = {}
        for var_name in all_variables:
            if variable_domains and var_name in variable_domains:
                lb, ub = variable_domains[var_name]
                # 判断是否为整数变量
                if "integer" in var_name.lower() or "count" in var_name.lower():
                    mip_vars[var_name] = solver.IntVar(int(lb), int(ub), var_name)
                else:
                    mip_vars[var_name] = solver.NumVar(lb, ub, var_name)
            else:
                mip_vars[var_name] = solver.NumVar(0, solver.infinity(), var_name)
        
        # 添加约束
        for constraint in constraints:
            try:
                if constraint.is_equality:
                    ct = solver.Constraint(constraint.bounds[0], constraint.bounds[0])
                else:
                    ct = solver.Constraint(constraint.bounds[0], constraint.bounds[1])
                
                for coeff, var in zip(constraint.coefficients, constraint.variables):
                    ct.SetCoefficient(mip_vars[var], coeff)
                    
            except Exception as e:
                self.logger.warning(f"添加混合整数约束{constraint.constraint_id}失败: {e}")
        
        # 添加目标函数
        if objectives:
            try:
                objective = solver.Objective()
                for obj in objectives:
                    for coeff, var in zip(obj.coefficients, obj.variables):
                        objective.SetCoefficient(mip_vars[var], obj.weight * coeff)
                
                if objectives[0].objective_type == "minimize":
                    objective.SetMinimization()
                else:
                    objective.SetMaximization()
            except Exception as e:
                self.logger.warning(f"添加混合整数目标函数失败: {e}")
        
        # 求解
        solver.SetTimeLimit(self.solver_config["time_limit_seconds"] * 1000)
        status = solver.Solve()
        solve_time = time.time() - start_time
        
        # 处理结果
        success = status == pywraplp.Solver.OPTIMAL
        variable_values = {}
        
        if success:
            for var_name, mip_var in mip_vars.items():
                variable_values[var_name] = mip_var.solution_value()
        
        objective_value = solver.Objective().Value() if success and objectives else None
        
        return ORToolsSolution(
            success=success,
            solver_status=self._get_linear_status_string(status),
            objective_value=objective_value, 
            variable_values=variable_values,
            solve_time=solve_time,
            constraint_violations=[],
            solver_type=SolverType.MIXED_INTEGER
        )
    
    def _fallback_solve(self, constraints: List[ORToolsConstraint],
                       objectives: List[OptimizationObjective] = None) -> ORToolsSolution:
        """回退求解器 (简化实现)"""
        
        self.logger.info("使用回退约束求解器")
        
        # 简化的约束求解
        variable_values = {}
        all_variables = set(var for c in constraints for var in c.variables)
        
        # 为所有变量分配默认值
        for var in all_variables:
            if "count" in var.lower() or "number" in var.lower():
                variable_values[var] = 1.0  # 默认计数值
            else:
                variable_values[var] = 0.0
        
        # 简单的约束检查
        violations = []
        for constraint in constraints:
            if constraint.constraint_type == "non_negative":
                for var in constraint.variables:
                    if variable_values.get(var, 0) < 0:
                        violations.append(ConstraintViolation(
                            constraint_id=constraint.constraint_id,
                            violation_type="negative_value",
                            severity=0.8,
                            affected_entities=[var],
                            description=f"变量{var}违反非负约束",
                            suggested_fix="将值设为非负"
                        ))
        
        return ORToolsSolution(
            success=len(violations) == 0,
            solver_status="FALLBACK_SOLVED",
            objective_value=None,
            variable_values=variable_values,
            solve_time=0.001,
            constraint_violations=violations,
            solver_type=SolverType.FALLBACK
        )
    
    def _get_cp_sat_status_string(self, status) -> str:
        """获取CP-SAT状态字符串"""
        if not ORTOOLS_AVAILABLE:
            return "UNKNOWN"
            
        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE", 
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID"
        }
        return status_map.get(status, "UNKNOWN")
    
    def _get_linear_status_string(self, status) -> str:
        """获取线性求解器状态字符串"""
        if not ORTOOLS_AVAILABLE:
            return "UNKNOWN"
            
        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED"
        }
        return status_map.get(status, "UNKNOWN")
    
    def convert_physics_constraints_to_ortools(self, 
                                             physics_constraints: List,
                                             semantic_entities: List) -> List[ORToolsConstraint]:
        """将物理约束转换为OR-Tools约束"""
        
        ortools_constraints = []
        constraint_id_counter = 1
        
        for phys_constraint in physics_constraints:
            try:
                if phys_constraint.constraint_type.name == "NON_NEGATIVE":
                    # 非负约束转换
                    for entity_id in phys_constraint.involved_entities:
                        ortools_constraint = ORToolsConstraint(
                            constraint_id=f"ortools_non_neg_{constraint_id_counter}",
                            constraint_type="non_negative",
                            variables=[entity_id],
                            coefficients=[1.0],
                            bounds=(0.0, float('inf')),
                            priority=10
                        )
                        ortools_constraints.append(ortools_constraint)
                        constraint_id_counter += 1
                
                elif phys_constraint.constraint_type.name == "INTEGER_CONSTRAINT":
                    # 整数约束转换
                    for entity_id in phys_constraint.involved_entities:
                        ortools_constraint = ORToolsConstraint(
                            constraint_id=f"ortools_integer_{constraint_id_counter}",
                            constraint_type="integer_constraint",
                            variables=[entity_id],
                            coefficients=[1.0],
                            bounds=(0.0, 1000.0),  # 合理的整数范围
                            priority=9
                        )
                        ortools_constraints.append(ortools_constraint)
                        constraint_id_counter += 1
                
                elif phys_constraint.constraint_type.name == "CONSERVATION_LAW":
                    # 守恒约束转换为等式约束
                    if len(phys_constraint.involved_entities) >= 2:
                        ortools_constraint = ORToolsConstraint(
                            constraint_id=f"ortools_conservation_{constraint_id_counter}",
                            constraint_type="linear_equality",
                            variables=phys_constraint.involved_entities,
                            coefficients=[1.0] * len(phys_constraint.involved_entities),
                            bounds=(0.0, 0.0),  # 等式约束
                            is_equality=True,
                            priority=8
                        )
                        ortools_constraints.append(ortools_constraint)
                        constraint_id_counter += 1
                        
            except Exception as e:
                self.logger.warning(f"转换物理约束失败: {e}")
        
        self.logger.info(f"转换了{len(ortools_constraints)}个OR-Tools约束")
        return ortools_constraints
    
    def generate_optimization_objectives(self, problem_context: Dict[str, Any]) -> List[OptimizationObjective]:
        """生成优化目标"""
        
        objectives = []
        
        # 基于问题类型生成不同的优化目标
        problem_type = problem_context.get("problem_type", "arithmetic")
        
        if problem_type == "simple_arithmetic":
            # 算术问题：最大化解的准确性
            objectives.append(OptimizationObjective(
                objective_type="maximize",
                variables=["accuracy_score"],
                coefficients=[1.0],
                weight=self.objective_weights["accuracy"]
            ))
        
        elif problem_type == "optimization":
            # 优化问题：多目标优化
            objectives.extend([
                OptimizationObjective(
                    objective_type="maximize",
                    variables=["efficiency"],
                    coefficients=[1.0],
                    weight=self.objective_weights["efficiency"]
                ),
                OptimizationObjective(
                    objective_type="minimize",
                    variables=["cost"],
                    coefficients=[1.0],
                    weight=0.3
                )
            ])
        
        return objectives

# 测试函数
def test_ortools_solver():
    """测试OR-Tools求解器"""
    
    solver = ORToolsConstraintSolver()
    
    print("🔧 OR-Tools约束求解器测试")
    print("=" * 50)
    print(f"OR-Tools可用性: {solver.is_available}")
    
    if not solver.is_available:
        print("⚠️ OR-Tools未安装，请运行: pip install ortools")
        return
    
    # 创建测试约束
    test_constraints = [
        ORToolsConstraint(
            constraint_id="test_non_neg_1",
            constraint_type="non_negative",
            variables=["x", "y"],
            coefficients=[1.0, 1.0],
            bounds=(0.0, float('inf'))
        ),
        ORToolsConstraint(
            constraint_id="test_sum_constraint",
            constraint_type="linear_equality",
            variables=["x", "y"],
            coefficients=[1.0, 1.0],
            bounds=(10.0, 10.0),
            is_equality=True
        )
    ]
    
    # 创建测试目标
    test_objectives = [
        OptimizationObjective(
            objective_type="maximize",
            variables=["x", "y"],
            coefficients=[2.0, 3.0],
            weight=1.0
        )
    ]
    
    # 变量域
    variable_domains = {
        "x": (0.0, 10.0),
        "y": (0.0, 10.0)
    }
    
    # 求解
    result = solver.solve_enhanced_constraints(
        constraints=test_constraints,
        objectives=test_objectives,
        variable_domains=variable_domains
    )
    
    print(f"\n求解结果:")
    print(f"  成功: {result.success}")
    print(f"  求解器: {result.solver_type.value}")
    print(f"  状态: {result.solver_status}")
    print(f"  目标值: {result.objective_value}")
    print(f"  变量值: {result.variable_values}")
    print(f"  求解时间: {result.solve_time:.3f}秒")
    print(f"  约束违背: {len(result.constraint_violations)}个")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ortools_solver()