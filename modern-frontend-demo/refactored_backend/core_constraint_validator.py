#!/usr/bin/env python3
"""
核心约束验证器 - 只保留真正帮助解题的约束功能
Core Constraint Validator - Only constraint functions that actually help solving problems
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """真正有用的约束类型"""
    NON_NEGATIVE = "non_negative"           # 数量非负约束
    INTEGER_ONLY = "integer_only"           # 计数必须整数
    CONSERVATION = "conservation"           # 守恒约束 (input = output)
    TYPE_CONSISTENCY = "type_consistency"   # 类型一致性
    ARITHMETIC_VALIDITY = "arithmetic_validity"  # 算术有效性

@dataclass
class BasicConstraint:
    """基础约束"""
    constraint_id: str
    constraint_type: ConstraintType
    entities_involved: List[str]
    rule_description: str
    violation_threshold: float = 0.0

@dataclass
class ConstraintViolation:
    """约束违背"""
    constraint_id: str
    violation_message: str
    severity: float  # 0.0-1.0
    entities_affected: List[str]

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    violations: List[ConstraintViolation]
    confidence_adjustment: float  # 对最终答案置信度的调整 (-0.5 to +0.2)
    validation_notes: List[str]

class CoreConstraintValidator:
    """核心约束验证器 - 专注于真正有助于解题的约束"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def validate_solution(self, 
                         problem_text: str,
                         entities: List[Dict],
                         solution_value: Any,
                         reasoning_steps: List[Dict]) -> ValidationResult:
        """
        验证解答是否满足基本数学约束
        这是约束系统对解题的真正贡献
        """
        
        violations = []
        confidence_adjustment = 0.0
        validation_notes = []
        
        try:
            # 1. 非负性验证 - 对计数和数量类问题
            non_neg_violation = self._check_non_negativity(problem_text, solution_value)
            if non_neg_violation:
                violations.append(non_neg_violation)
                confidence_adjustment -= 0.3
            
            # 2. 整数性验证 - 对计数类问题
            integer_violation = self._check_integer_constraint(problem_text, solution_value)
            if integer_violation:
                violations.append(integer_violation)
                confidence_adjustment -= 0.2
                
            # 3. 守恒性验证 - 对加减法问题
            conservation_violation = self._check_conservation(problem_text, entities, solution_value)
            if conservation_violation:
                violations.append(conservation_violation)
                confidence_adjustment -= 0.4
                
            # 4. 类型一致性验证 - 答案类型是否合理
            type_violation = self._check_type_consistency(problem_text, solution_value)
            if type_violation:
                violations.append(type_violation)
                confidence_adjustment -= 0.2
                
            # 5. 数值合理性验证 - 答案是否在合理范围
            range_violation = self._check_value_range(problem_text, entities, solution_value)
            if range_violation:
                violations.append(range_violation)
                confidence_adjustment -= 0.1
            
            # 如果没有违背，给予小幅置信度提升
            if not violations:
                confidence_adjustment = 0.1
                validation_notes.append("所有基础约束验证通过")
            
            is_valid = len(violations) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                violations=violations,
                confidence_adjustment=confidence_adjustment,
                validation_notes=validation_notes
            )
            
        except Exception as e:
            self.logger.error(f"约束验证失败: {e}")
            return ValidationResult(
                is_valid=False,
                violations=[ConstraintViolation(
                    constraint_id="validation_error",
                    violation_message=f"验证过程出错: {str(e)}",
                    severity=0.3,
                    entities_affected=[]
                )],
                confidence_adjustment=-0.1,
                validation_notes=["验证过程出现异常"]
            )
    
    def _check_non_negativity(self, problem_text: str, solution_value: Any) -> Optional[ConstraintViolation]:
        """检查非负性约束 - 这是最基本的物理约束"""
        
        # 检查问题是否涉及物理数量
        quantity_indicators = ['个', '只', '本', '支', '张', '元', '米', '公斤', '分钟', '小时']
        involves_quantity = any(indicator in problem_text for indicator in quantity_indicators)
        
        if involves_quantity:
            try:
                numeric_value = float(solution_value)
                if numeric_value < 0:
                    return ConstraintViolation(
                        constraint_id="non_negative_001",
                        violation_message=f"物理数量不能为负数，当前答案: {solution_value}",
                        severity=0.8,
                        entities_affected=["solution"]
                    )
            except (ValueError, TypeError):
                # 如果不能转换为数字，说明答案类型可能有问题，但这不是非负性问题
                pass
        
        return None
    
    def _check_integer_constraint(self, problem_text: str, solution_value: Any) -> Optional[ConstraintViolation]:
        """检查整数约束 - 计数类问题的答案必须是整数"""
        
        # 检查是否是计数类问题
        counting_indicators = ['几个', '多少个', '多少只', '多少本', '多少支', '多少张']
        is_counting = any(indicator in problem_text for indicator in counting_indicators)
        
        if is_counting:
            try:
                numeric_value = float(solution_value)
                if not numeric_value.is_integer():
                    return ConstraintViolation(
                        constraint_id="integer_001", 
                        violation_message=f"计数结果必须是整数，当前答案: {solution_value}",
                        severity=0.7,
                        entities_affected=["solution"]
                    )
            except (ValueError, TypeError):
                return ConstraintViolation(
                    constraint_id="integer_002",
                    violation_message=f"计数问题的答案应该是数字，当前答案: {solution_value}",
                    severity=0.6,
                    entities_affected=["solution"]
                )
        
        return None
    
    def _check_conservation(self, problem_text: str, entities: List[Dict], solution_value: Any) -> Optional[ConstraintViolation]:
        """检查守恒约束 - 加减法问题的数量守恒"""
        
        # 检查是否是简单的加法问题
        if '一共' in problem_text or '总共' in problem_text:
            # 提取所有数字
            numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
            
            if len(numbers) >= 2:
                try:
                    # 简单的加法守恒检查
                    input_numbers = [float(n) for n in numbers[:-1] if float(n) > 0]  # 排除0
                    expected_sum = sum(input_numbers)
                    actual_solution = float(solution_value)
                    
                    if abs(expected_sum - actual_solution) > 0.001:  # 允许小的浮点误差
                        return ConstraintViolation(
                            constraint_id="conservation_001",
                            violation_message=f"数量守恒违背: 预期总和{expected_sum}，实际答案{actual_solution}",
                            severity=0.9,
                            entities_affected=["solution", "input_numbers"]
                        )
                except (ValueError, TypeError):
                    pass
        
        # 检查减法问题的守恒
        if '剩下' in problem_text or '还有' in problem_text:
            numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
            if len(numbers) >= 2:
                try:
                    initial_amount = float(numbers[0])
                    subtracted_amount = float(numbers[1])
                    expected_remainder = initial_amount - subtracted_amount
                    actual_solution = float(solution_value)
                    
                    if abs(expected_remainder - actual_solution) > 0.001:
                        return ConstraintViolation(
                            constraint_id="conservation_002", 
                            violation_message=f"减法守恒违背: 预期剩余{expected_remainder}，实际答案{actual_solution}",
                            severity=0.9,
                            entities_affected=["solution"]
                        )
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def _check_type_consistency(self, problem_text: str, solution_value: Any) -> Optional[ConstraintViolation]:
        """检查类型一致性 - 答案类型是否符合问题预期"""
        
        # 数值类问题应该有数值答案
        if any(word in problem_text for word in ['多少', '几', '计算', '求']):
            try:
                float(solution_value)  # 尝试转换为数字
            except (ValueError, TypeError):
                return ConstraintViolation(
                    constraint_id="type_001",
                    violation_message=f"数值问题的答案应该是数字，当前答案: {solution_value}",
                    severity=0.6,
                    entities_affected=["solution"]
                )
        
        return None
    
    def _check_value_range(self, problem_text: str, entities: List[Dict], solution_value: Any) -> Optional[ConstraintViolation]:
        """检查数值合理性 - 答案是否在合理范围内"""
        
        try:
            numeric_value = float(solution_value)
            
            # 提取问题中的数字来判断合理范围
            numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', problem_text)]
            
            if numbers:
                max_input = max(numbers)
                
                # 简单的合理性检查
                if numeric_value > max_input * 100:  # 答案不应该比输入大100倍以上
                    return ConstraintViolation(
                        constraint_id="range_001",
                        violation_message=f"答案数值可能过大: {solution_value}，输入最大值: {max_input}",
                        severity=0.3,
                        entities_affected=["solution"]
                    )
                
        except (ValueError, TypeError):
            pass
        
        return None

    def generate_constraint_guided_reasoning_paths(self, problem_text: str, entities: List[Dict]) -> List[str]:
        """
        基于约束生成推理路径提示 - 这是约束系统真正帮助解题的方式
        Generate constraint-guided reasoning paths to help problem solving
        """
        
        reasoning_paths = []
        
        # 1. 基于约束的解题步骤提示
        if '一共' in problem_text or '总共' in problem_text:
            reasoning_paths.append("识别加法运算：需要将所有部分数量相加")
            reasoning_paths.append("验证守恒约束：确保总和等于各部分之和")
            
        if '剩下' in problem_text or '还有' in problem_text:
            reasoning_paths.append("识别减法运算：从初始数量中减去消耗量")
            reasoning_paths.append("验证非负约束：确保剩余数量不为负数")
            
        # 2. 基于实体类型的约束提示
        if any(indicator in problem_text for indicator in ['几个', '多少个']):
            reasoning_paths.append("计数问题：答案必须是正整数")
            reasoning_paths.append("检查整数约束：验证最终答案为整数")
            
        if any(indicator in problem_text for indicator in ['元', '钱', '价格']):
            reasoning_paths.append("货币问题：答案应为非负数值")
            reasoning_paths.append("检查非负约束：确保金额不为负数")
            
        # 3. 基于物理常识的推理引导
        if any(indicator in problem_text for indicator in ['速度', '时间', '距离']):
            reasoning_paths.append("物理量关系：考虑速度=距离/时间的基本关系")
            reasoning_paths.append("单位一致性：确保所有物理量单位匹配")
            
        return reasoning_paths

    def suggest_solution_verification_steps(self, problem_text: str, solution_value: Any) -> List[str]:
        """
        建议解答验证步骤 - 引导学生自我验证
        """
        
        verification_steps = []
        
        # 基于问题类型提供验证建议
        if '一共' in problem_text:
            verification_steps.append("验证步骤1: 将各个部分数量相加，检查是否等于答案")
            verification_steps.append("验证步骤2: 确认答案为正数且符合实际情况")
            
        if '剩下' in problem_text:
            verification_steps.append("验证步骤1: 用初始数量减去使用量，检查是否等于答案")
            verification_steps.append("验证步骤2: 确认剩余数量不为负数")
            
        # 通用验证步骤
        verification_steps.append("验证步骤3: 检查答案的单位和数值是否合理")
        verification_steps.append("验证步骤4: 重新阅读题目，确认理解正确")
        
        return verification_steps

# 工厂函数
def create_constraint_validator() -> CoreConstraintValidator:
    """创建约束验证器实例"""
    return CoreConstraintValidator()

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    validator = CoreConstraintValidator()
    
    # 测试用例
    test_cases = [
        {
            "problem": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
            "solution": 8,
            "entities": [{"name": "小明", "type": "person"}, {"name": "苹果", "type": "object"}]
        },
        {
            "problem": "小明有10元钱，买东西花了12元，还剩多少钱？",
            "solution": -2,  # 这应该违背非负约束
            "entities": [{"name": "小明", "type": "person"}, {"name": "钱", "type": "money"}]
        },
        {
            "problem": "班级有30个学生，今天来了25.5个，问多少个学生？",
            "solution": 25.5,  # 这应该违背整数约束
            "entities": [{"name": "学生", "type": "person"}]
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {test['problem']}")
        print(f"给定答案: {test['solution']}")
        
        result = validator.validate_solution(
            test['problem'], 
            test['entities'], 
            test['solution'], 
            []
        )
        
        print(f"验证结果: {'通过' if result.is_valid else '失败'}")
        print(f"置信度调整: {result.confidence_adjustment:+.2f}")
        
        if result.violations:
            print("约束违背:")
            for violation in result.violations:
                print(f"  - {violation.violation_message} (严重度: {violation.severity:.2f})")
        
        print("推理路径建议:")
        for path in validator.generate_constraint_guided_reasoning_paths(test['problem'], test['entities']):
            print(f"  • {path}")