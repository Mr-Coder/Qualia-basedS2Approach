#!/usr/bin/env python3
"""
简化约束系统 - 基于真实解题需求的约束验证
Simplified Constraint System - Based on actual problem-solving needs
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from core_constraint_validator import CoreConstraintValidator, ValidationResult
from constraint_guided_reasoner import ConstraintGuidedReasoner, ConstraintGuidedPath

logger = logging.getLogger(__name__)

@dataclass
class SimplifiedConstraintResult:
    """简化约束系统结果"""
    success: bool
    validation_result: ValidationResult
    reasoning_path: ConstraintGuidedPath
    constraint_guidance: List[str]
    verification_steps: List[str]
    execution_time: float
    confidence_adjustment: float

class SimplifiedConstraintSystem:
    """
    简化约束系统 - 专注于真正帮助解题的约束功能
    
    这个系统的核心理念是：
    1. 约束不是为了展示复杂性，而是为了帮助验证和引导推理
    2. 只保留对数学问题求解真正有用的约束类型
    3. 约束系统应该与IRD和QS²引擎协作，而不是独立运行
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 核心组件
        self.constraint_validator = CoreConstraintValidator()
        self.constraint_reasoner = ConstraintGuidedReasoner()
        
        # 系统统计
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "constraint_violations": 0,
            "average_confidence_boost": 0.0
        }
        
        self.logger.info("简化约束系统初始化完成")
    
    def process_problem_with_constraints(self, 
                                       problem_text: str,
                                       entities: List[Dict],
                                       solution_value: Any,
                                       reasoning_steps: List[Dict] = None) -> SimplifiedConstraintResult:
        """
        使用约束系统处理问题 - 这是约束系统的主要入口点
        
        Args:
            problem_text: 问题文本
            entities: 实体列表
            solution_value: 解答值
            reasoning_steps: 推理步骤
            
        Returns:
            SimplifiedConstraintResult: 约束处理结果
        """
        
        start_time = time.time()
        
        try:
            self.logger.info("开始约束系统处理")
            
            # Step 1: 生成约束引导的推理路径
            self.logger.debug("生成约束引导推理路径")
            reasoning_path = self.constraint_reasoner.generate_reasoning_path(
                problem_text, entities
            )
            
            # Step 2: 验证解答是否满足基本约束
            self.logger.debug("验证解答约束")
            validation_result = self.constraint_validator.validate_solution(
                problem_text, entities, solution_value, reasoning_steps or []
            )
            
            # Step 3: 生成约束指导建议
            constraint_guidance = self.constraint_validator.generate_constraint_guided_reasoning_paths(
                problem_text, entities
            )
            
            # Step 4: 生成解答验证步骤
            verification_steps = self.constraint_validator.suggest_solution_verification_steps(
                problem_text, solution_value
            )
            
            # Step 5: 计算执行时间和总体成功状态
            execution_time = time.time() - start_time
            overall_success = validation_result.is_valid and reasoning_path.confidence_score > 0.3
            
            # Step 6: 更新统计信息
            self._update_stats(validation_result, reasoning_path)
            
            result = SimplifiedConstraintResult(
                success=overall_success,
                validation_result=validation_result,
                reasoning_path=reasoning_path,
                constraint_guidance=constraint_guidance,
                verification_steps=verification_steps,
                execution_time=execution_time,
                confidence_adjustment=validation_result.confidence_adjustment
            )
            
            self.logger.info(f"约束系统处理完成，成功: {overall_success}，耗时: {execution_time:.3f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"约束系统处理失败: {e}")
            return self._create_fallback_result(problem_text, entities, time.time() - start_time)
    
    def validate_reasoning_step(self, step: Dict, problem_context: str) -> Dict[str, Any]:
        """
        验证单个推理步骤是否符合约束
        这个功能展示了约束如何在推理过程中发挥作用
        """
        
        step_validation = {
            "step_id": step.get("step_id", "unknown"),
            "is_valid": True,
            "constraint_checks": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # 检查步骤中的数值是否合理
            if "values" in step:
                for value in step["values"]:
                    try:
                        numeric_value = float(value)
                        
                        # 非负性检查
                        if "个" in problem_context or "只" in problem_context:
                            if numeric_value < 0:
                                step_validation["is_valid"] = False
                                step_validation["constraint_checks"].append({
                                    "constraint": "非负性约束",
                                    "passed": False,
                                    "message": f"计数值不能为负: {value}"
                                })
                            else:
                                step_validation["constraint_checks"].append({
                                    "constraint": "非负性约束",
                                    "passed": True,
                                    "message": "数值为非负，符合物理常识"
                                })
                        
                        # 整数性检查
                        if any(word in problem_context for word in ["几个", "多少个"]):
                            if not numeric_value.is_integer():
                                step_validation["warnings"].append(
                                    f"计数问题的中间值{value}不是整数，需要检查计算过程"
                                )
                    
                    except ValueError:
                        step_validation["warnings"].append(f"无法解析数值: {value}")
            
            # 检查运算类型是否合理
            if "operation" in step:
                operation = step["operation"]
                if operation == "subtraction":
                    step_validation["suggestions"].append(
                        "减法运算：确保结果不为负数，符合实际情况"
                    )
                elif operation == "addition":
                    step_validation["suggestions"].append(
                        "加法运算：验证各部分之和等于总和"
                    )
        
        except Exception as e:
            self.logger.warning(f"推理步骤验证失败: {e}")
            step_validation["warnings"].append(f"验证过程出错: {str(e)}")
        
        return step_validation
    
    def explain_constraint_role_in_reasoning(self, problem_text: str) -> Dict[str, Any]:
        """
        解释约束在推理中的作用 - 这回答了用户关于"约束如何帮助构建物理推理路径"的问题
        """
        
        explanation = {
            "problem_analysis": self._analyze_problem_for_constraints(problem_text),
            "constraint_roles": [],
            "reasoning_guidance": [],
            "verification_checkpoints": []
        }
        
        # 分析约束在不同推理阶段的作用
        if "一共" in problem_text or "总共" in problem_text:
            explanation["constraint_roles"].append({
                "stage": "问题理解阶段",
                "constraint_type": "守恒约束",
                "role": "识别这是一个加法运算，需要验证部分之和等于总和",
                "guidance": "引导学生思考：所有部分加起来应该等于总数"
            })
            
            explanation["reasoning_guidance"].append(
                "步骤1: 识别所有需要相加的部分"
            )
            explanation["reasoning_guidance"].append(
                "步骤2: 执行加法运算"
            )
            explanation["reasoning_guidance"].append(
                "步骤3: 验证结果是否符合守恒定律"
            )
            
            explanation["verification_checkpoints"].append({
                "checkpoint": "加法验证",
                "check": "各部分之和 = 最终总和",
                "purpose": "确保计算正确性"
            })
        
        if "剩下" in problem_text or "还有" in problem_text:
            explanation["constraint_roles"].append({
                "stage": "推理执行阶段",
                "constraint_type": "非负约束",
                "role": "确保减法结果不为负数，符合物理现实",
                "guidance": "提醒学生：剩余的东西不能是负数"
            })
            
            explanation["verification_checkpoints"].append({
                "checkpoint": "非负验证",
                "check": "剩余数量 ≥ 0",
                "purpose": "确保答案符合物理常识"
            })
        
        if any(word in problem_text for word in ["几个", "多少个"]):
            explanation["constraint_roles"].append({
                "stage": "答案验证阶段", 
                "constraint_type": "整数约束",
                "role": "确保计数结果为整数",
                "guidance": "提醒学生：数苹果的结果必须是整数"
            })
        
        return explanation
    
    def _analyze_problem_for_constraints(self, problem_text: str) -> Dict[str, Any]:
        """分析问题文本，识别需要应用的约束类型"""
        
        analysis = {
            "problem_type": "unknown",
            "key_constraints": [],
            "potential_violations": [],
            "reasoning_checkpoints": []
        }
        
        # 问题类型识别
        if "一共" in problem_text or "总共" in problem_text:
            analysis["problem_type"] = "aggregation"
            analysis["key_constraints"].append("守恒约束：总和 = 各部分之和")
            analysis["reasoning_checkpoints"].append("验证加法计算")
            
        elif "剩下" in problem_text or "还有" in problem_text:
            analysis["problem_type"] = "reduction" 
            analysis["key_constraints"].append("非负约束：剩余量 ≥ 0")
            analysis["reasoning_checkpoints"].append("检查减法结果")
            
        elif "平均" in problem_text or "每" in problem_text:
            analysis["problem_type"] = "distribution"
            analysis["key_constraints"].append("均匀分配约束")
            
        # 识别计数约束
        if any(word in problem_text for word in ["几个", "多少个", "多少只"]):
            analysis["key_constraints"].append("整数约束：计数结果必须为整数")
            analysis["reasoning_checkpoints"].append("验证答案为整数")
            
        # 识别潜在违规
        if "借" in problem_text or "欠" in problem_text:
            analysis["potential_violations"].append("可能出现负数结果，需要特别注意")
            
        return analysis
    
    def _update_stats(self, validation_result: ValidationResult, reasoning_path: ConstraintGuidedPath):
        """更新系统统计信息"""
        
        self.validation_stats["total_validations"] += 1
        
        if validation_result.is_valid:
            self.validation_stats["successful_validations"] += 1
        
        if validation_result.violations:
            self.validation_stats["constraint_violations"] += len(validation_result.violations)
        
        # 更新平均置信度提升
        current_avg = self.validation_stats["average_confidence_boost"]
        new_boost = validation_result.confidence_adjustment
        total_validations = self.validation_stats["total_validations"]
        
        self.validation_stats["average_confidence_boost"] = (
            (current_avg * (total_validations - 1) + new_boost) / total_validations
        )
    
    def _create_fallback_result(self, problem_text: str, entities: List[Dict], execution_time: float) -> SimplifiedConstraintResult:
        """创建回退结果"""
        
        from core_constraint_validator import ValidationResult, ConstraintViolation
        from constraint_guided_reasoner import ConstraintGuidedPath, ReasoningStep, ReasoningConstraint
        
        fallback_validation = ValidationResult(
            is_valid=False,
            violations=[ConstraintViolation(
                constraint_id="system_error",
                violation_message="约束系统处理出错",
                severity=0.5,
                entities_affected=[]
            )],
            confidence_adjustment=-0.2,
            validation_notes=["系统异常，使用基础验证"]
        )
        
        fallback_path = ConstraintGuidedPath(
            path_id="fallback",
            reasoning_steps=[
                ReasoningStep(
                    step_id=1,
                    operation_type="fallback",
                    description="基础问题分析",
                    entities_involved=[],
                    constraints_applied=[ReasoningConstraint.ENTITY_PERSISTENCE],
                    confidence=0.3,
                    rationale="系统异常时的基础处理"
                )
            ],
            confidence_score=0.3,
            constraint_satisfaction_rate=0.1,
            path_rationale="系统异常，采用简化处理"
        )
        
        return SimplifiedConstraintResult(
            success=False,
            validation_result=fallback_validation,
            reasoning_path=fallback_path,
            constraint_guidance=["基础约束验证"],
            verification_steps=["检查答案合理性"],
            execution_time=execution_time,
            confidence_adjustment=-0.2
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            **self.validation_stats,
            "success_rate": (
                self.validation_stats["successful_validations"] / 
                max(self.validation_stats["total_validations"], 1)
            )
        }

# 工厂函数
def create_simplified_constraint_system() -> SimplifiedConstraintSystem:
    """创建简化约束系统实例"""
    return SimplifiedConstraintSystem()

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    system = SimplifiedConstraintSystem()
    
    # 测试用例：展示约束如何真正帮助解题
    test_cases = [
        {
            "problem": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
            "entities": [
                {"name": "小明", "type": "person"},
                {"name": "小红", "type": "person"}, 
                {"name": "苹果", "type": "object"}
            ],
            "solution": 8,
            "expected_constraints": ["守恒约束", "整数约束"]
        },
        {
            "problem": "小明有10元，买东西花了12元，还剩多少钱？",
            "entities": [
                {"name": "小明", "type": "person"},
                {"name": "钱", "type": "money"}
            ],
            "solution": -2,  # 这应该违背非负约束
            "expected_constraints": ["非负约束"]
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n=== 测试用例 {i+1} ===")
        print(f"问题: {test['problem']}")
        print(f"答案: {test['solution']}")
        
        # 处理问题
        result = system.process_problem_with_constraints(
            test['problem'],
            test['entities'], 
            test['solution']
        )
        
        print(f"约束验证: {'通过' if result.success else '失败'}")
        print(f"置信度调整: {result.confidence_adjustment:+.2f}")
        
        # 显示约束指导
        print("约束如何指导推理:")
        for guidance in result.constraint_guidance:
            print(f"  • {guidance}")
        
        # 显示推理路径
        print("约束引导的推理步骤:")
        for step in result.reasoning_path.reasoning_steps:
            print(f"  {step.step_id}. {step.description}")
            print(f"     约束作用: {step.rationale}")
        
        # 解释约束在推理中的作用
        explanation = system.explain_constraint_role_in_reasoning(test['problem'])
        print("约束在推理中的作用:")
        for role in explanation["constraint_roles"]:
            print(f"  - {role['stage']}: {role['role']}")