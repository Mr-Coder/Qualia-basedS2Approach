"""
链式验证器 (Chain Verification Validator)

专注于验证推理链的逻辑一致性和数学正确性。
这是COT-DIR算法的第三个核心组件。
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .mlr_processor import ReasoningStep, ReasoningStepType


class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"           # 基础验证
    INTERMEDIATE = "intermediate"  # 中级验证
    COMPREHENSIVE = "comprehensive"  # 全面验证


class ErrorType(Enum):
    """错误类型"""
    LOGICAL_ERROR = "logical_error"      # 逻辑错误
    MATHEMATICAL_ERROR = "mathematical_error"  # 数学错误
    CONSISTENCY_ERROR = "consistency_error"    # 一致性错误
    DEPENDENCY_ERROR = "dependency_error"      # 依赖关系错误
    VALUE_ERROR = "value_error"               # 数值错误


@dataclass
class ValidationError:
    """验证错误"""
    error_type: ErrorType
    step_id: int
    description: str
    severity: float  # 0.0-1.0, 1.0为最严重
    suggestion: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.error_type.value,
            "step_id": self.step_id,
            "description": self.description,
            "severity": self.severity,
            "suggestion": self.suggestion,
            "metadata": self.metadata
        }


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    consistency_score: float
    errors: List[ValidationError]
    warnings: List[str]
    suggestions: List[str]
    corrected_steps: List[ReasoningStep]
    validation_time: float
    metadata: Dict[str, Any]
    
    def get_errors_by_type(self, error_type: ErrorType) -> List[ValidationError]:
        """按类型获取错误"""
        return [error for error in self.errors if error.error_type == error_type]
    
    def get_severe_errors(self, threshold: float = 0.7) -> List[ValidationError]:
        """获取严重错误"""
        return [error for error in self.errors if error.severity >= threshold]
    
    def has_critical_errors(self) -> bool:
        """是否存在关键错误"""
        return any(error.severity >= 0.8 for error in self.errors)


class ChainVerificationValidator:
    """链式验证器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化CV验证器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 配置参数
        self.config = config or {}
        self.validation_level = ValidationLevel(self.config.get("validation_level", "intermediate"))
        self.error_threshold = self.config.get("error_threshold", 0.7)
        self.enable_auto_correction = self.config.get("enable_auto_correction", True)
        self.max_correction_attempts = self.config.get("max_correction_attempts", 3)
        
        # 验证规则配置
        self.validation_rules = self._initialize_validation_rules()
        
        # 统计信息
        self.stats = {
            "total_validations": 0,
            "valid_chains": 0,
            "invalid_chains": 0,
            "errors_found": 0,
            "corrections_made": 0,
            "average_consistency_score": 0.0,
            "error_type_counts": {et.value: 0 for et in ErrorType}
        }
        
        self.logger.info(f"链式验证器初始化完成，验证级别: {self.validation_level.value}")
    
    def verify_reasoning_chain(
        self, 
        reasoning_steps: List[ReasoningStep], 
        problem_context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        验证推理链
        
        Args:
            reasoning_steps: 推理步骤列表
            problem_context: 问题上下文
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始验证推理链，共{len(reasoning_steps)}个步骤")
            
            # 初始化验证上下文
            validation_context = {
                "steps": reasoning_steps,
                "problem_context": problem_context or {},
                "step_values": {},
                "dependencies": {},
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
            
            # 第一阶段：结构验证
            structural_errors = self._validate_structure(validation_context)
            validation_context["errors"].extend(structural_errors)
            
            # 第二阶段：逻辑验证
            logical_errors = self._validate_logic(validation_context)
            validation_context["errors"].extend(logical_errors)
            
            # 第三阶段：数学验证
            mathematical_errors = self._validate_mathematics(validation_context)
            validation_context["errors"].extend(mathematical_errors)
            
            # 第四阶段：一致性验证
            consistency_errors = self._validate_consistency(validation_context)
            validation_context["errors"].extend(consistency_errors)
            
            # 计算一致性分数
            consistency_score = self._calculate_consistency_score(validation_context)
            
            # 自动纠错（如果启用）
            corrected_steps = reasoning_steps.copy()
            if self.enable_auto_correction and validation_context["errors"]:
                corrected_steps = self._auto_correct_errors(validation_context)
            
            # 生成建议
            suggestions = self._generate_suggestions(validation_context)
            validation_context["suggestions"].extend(suggestions)
            
            # 判断是否有效
            is_valid = self._determine_validity(validation_context, consistency_score)
            
            # 更新统计信息
            self._update_stats(validation_context, consistency_score, is_valid)
            
            validation_time = time.time() - start_time
            
            result = ValidationResult(
                is_valid=is_valid,
                consistency_score=consistency_score,
                errors=validation_context["errors"],
                warnings=validation_context["warnings"],
                suggestions=validation_context["suggestions"],
                corrected_steps=corrected_steps,
                validation_time=validation_time,
                metadata={
                    "validation_level": self.validation_level.value,
                    "step_count": len(reasoning_steps),
                    "error_count": len(validation_context["errors"]),
                    "correction_attempts": self.max_correction_attempts,
                    "auto_correction_enabled": self.enable_auto_correction
                }
            )
            
            self.logger.info(f"验证完成: 有效={is_valid}, 一致性={consistency_score:.3f}, 错误={len(validation_context['errors'])}个")
            return result
            
        except Exception as e:
            self.logger.error(f"链式验证失败: {str(e)}")
            return ValidationResult(
                is_valid=False,
                consistency_score=0.0,
                errors=[ValidationError(
                    error_type=ErrorType.LOGICAL_ERROR,
                    step_id=-1,
                    description=f"验证过程出错: {str(e)}",
                    severity=1.0,
                    suggestion="请检查输入数据和验证配置",
                    metadata={"exception": str(e)}
                )],
                warnings=[],
                suggestions=[],
                corrected_steps=reasoning_steps,
                validation_time=time.time() - start_time,
                metadata={"validation_failed": True}
            )
    
    def _validate_structure(self, context: Dict[str, Any]) -> List[ValidationError]:
        """验证结构"""
        errors = []
        steps = context["steps"]
        
        # 检查步骤ID的连续性
        expected_ids = set(range(len(steps)))
        actual_ids = {step.step_id for step in steps}
        
        if expected_ids != actual_ids:
            errors.append(ValidationError(
                error_type=ErrorType.DEPENDENCY_ERROR,
                step_id=-1,
                description="步骤ID不连续或重复",
                severity=0.8,
                suggestion="确保步骤ID从0开始连续编号",
                metadata={"expected_ids": list(expected_ids), "actual_ids": list(actual_ids)}
            ))
        
        # 检查依赖关系
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id >= step.step_id:
                    errors.append(ValidationError(
                        error_type=ErrorType.DEPENDENCY_ERROR,
                        step_id=step.step_id,
                        description=f"步骤{step.step_id}依赖于后续步骤{dep_id}",
                        severity=0.9,
                        suggestion="确保依赖关系指向前序步骤",
                        metadata={"invalid_dependency": dep_id}
                    ))
                
                if dep_id not in actual_ids:
                    errors.append(ValidationError(
                        error_type=ErrorType.DEPENDENCY_ERROR,
                        step_id=step.step_id,
                        description=f"步骤{step.step_id}依赖于不存在的步骤{dep_id}",
                        severity=0.9,
                        suggestion="检查依赖关系的正确性",
                        metadata={"missing_dependency": dep_id}
                    ))
        
        return errors
    
    def _validate_logic(self, context: Dict[str, Any]) -> List[ValidationError]:
        """验证逻辑"""
        errors = []
        steps = context["steps"]
        
        # 检查推理步骤的逻辑顺序
        step_types_order = [step.step_type for step in steps]
        
        # 验证初始化步骤
        if (ReasoningStepType.INITIALIZATION not in step_types_order and 
            len(steps) > 1):
            errors.append(ValidationError(
                error_type=ErrorType.LOGICAL_ERROR,
                step_id=0,
                description="缺少初始化步骤",
                severity=0.6,
                suggestion="在推理开始时添加初始化步骤",
                metadata={"missing_step_type": "initialization"}
            ))
        
        # 检查每个步骤的逻辑
        for step in steps:
            # 验证输入输出的逻辑性
            if step.step_type == ReasoningStepType.CALCULATION:
                if not step.input_values or step.output_value is None:
                    errors.append(ValidationError(
                        error_type=ErrorType.LOGICAL_ERROR,
                        step_id=step.step_id,
                        description="计算步骤缺少必要的输入或输出",
                        severity=0.8,
                        suggestion="确保计算步骤有明确的输入和输出",
                        metadata={"has_input": bool(step.input_values), "has_output": step.output_value is not None}
                    ))
            
            # 验证置信度合理性
            if not (0.0 <= step.confidence <= 1.0):
                errors.append(ValidationError(
                    error_type=ErrorType.VALUE_ERROR,
                    step_id=step.step_id,
                    description=f"置信度{step.confidence}超出合理范围[0,1]",
                    severity=0.5,
                    suggestion="确保置信度在0到1之间",
                    metadata={"invalid_confidence": step.confidence}
                ))
        
        return errors
    
    def _validate_mathematics(self, context: Dict[str, Any]) -> List[ValidationError]:
        """验证数学"""
        errors = []
        steps = context["steps"]
        
        for step in steps:
            if step.step_type == ReasoningStepType.CALCULATION:
                # 验证数学运算
                math_error = self._check_mathematical_operation(step)
                if math_error:
                    errors.append(math_error)
                
                # 验证数值合理性
                value_error = self._check_value_reasonableness(step)
                if value_error:
                    errors.append(value_error)
        
        return errors
    
    def _check_mathematical_operation(self, step: ReasoningStep) -> Optional[ValidationError]:
        """检查数学运算"""
        try:
            operation = step.operation.lower()
            inputs = step.input_values
            output = step.output_value
            
            if not inputs or output is None:
                return None
            
            # 检查除零错误
            if operation in ["division", "divide"] and len(inputs) >= 2:
                try:
                    divisor = float(inputs[1])
                    if abs(divisor) < 1e-10:
                        return ValidationError(
                            error_type=ErrorType.MATHEMATICAL_ERROR,
                            step_id=step.step_id,
                            description="除零错误",
                            severity=0.9,
                            suggestion="检查除数是否为零",
                            metadata={"operation": operation, "divisor": divisor}
                        )
                except (ValueError, IndexError):
                    pass
            
            # 验证简单运算结果
            if operation == "addition" and len(inputs) >= 2:
                try:
                    nums = [float(x) for x in inputs[:2]]
                    expected = nums[0] + nums[1]
                    actual = float(output)
                    
                    if abs(expected - actual) > 1e-6:
                        return ValidationError(
                            error_type=ErrorType.MATHEMATICAL_ERROR,
                            step_id=step.step_id,
                            description=f"加法计算错误: {nums[0]} + {nums[1]} = {expected}, 但得到{actual}",
                            severity=0.8,
                            suggestion="重新检查加法计算",
                            metadata={"expected": expected, "actual": actual}
                        )
                except (ValueError, TypeError):
                    pass
            
            elif operation == "subtraction" and len(inputs) >= 2:
                try:
                    nums = [float(x) for x in inputs[:2]]
                    expected = nums[0] - nums[1]
                    actual = float(output)
                    
                    if abs(expected - actual) > 1e-6:
                        return ValidationError(
                            error_type=ErrorType.MATHEMATICAL_ERROR,
                            step_id=step.step_id,
                            description=f"减法计算错误: {nums[0]} - {nums[1]} = {expected}, 但得到{actual}",
                            severity=0.8,
                            suggestion="重新检查减法计算",
                            metadata={"expected": expected, "actual": actual}
                        )
                except (ValueError, TypeError):
                    pass
            
        except Exception:
            # 如果无法验证，不报错，因为可能是复杂运算
            pass
        
        return None
    
    def _check_value_reasonableness(self, step: ReasoningStep) -> Optional[ValidationError]:
        """检查数值合理性"""
        try:
            if isinstance(step.output_value, (int, float)):
                value = float(step.output_value)
                
                # 检查极端值
                if abs(value) > 1e10:
                    return ValidationError(
                        error_type=ErrorType.VALUE_ERROR,
                        step_id=step.step_id,
                        description=f"输出值{value}过大，可能不合理",
                        severity=0.4,
                        suggestion="检查计算过程是否存在错误",
                        metadata={"large_value": value}
                    )
                
                # 检查负值的合理性（根据上下文）
                if value < 0 and step.operation in ["distance", "area", "volume", "count"]:
                    return ValidationError(
                        error_type=ErrorType.VALUE_ERROR,
                        step_id=step.step_id,
                        description=f"在{step.operation}运算中得到负值{value}",
                        severity=0.6,
                        suggestion="检查是否应该使用绝对值",
                        metadata={"negative_value": value, "operation": step.operation}
                    )
        
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _validate_consistency(self, context: Dict[str, Any]) -> List[ValidationError]:
        """验证一致性"""
        errors = []
        steps = context["steps"]
        
        # 构建值传递图
        value_map = {}
        for step in steps:
            if step.output_value is not None:
                value_map[step.step_id] = step.output_value
        
        # 检查依赖关系的一致性
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id in value_map:
                    dep_value = value_map[dep_id]
                    
                    # 检查是否在当前步骤的输入中使用了依赖步骤的输出
                    if (isinstance(dep_value, (int, float)) and 
                        step.input_values and 
                        str(dep_value) not in [str(v) for v in step.input_values]):
                        
                        # 这可能是正常的，不一定是错误，所以只给出警告
                        context["warnings"].append(
                            f"步骤{step.step_id}依赖步骤{dep_id}但未使用其输出值{dep_value}"
                        )
        
        # 检查数值的传递一致性
        for i, step in enumerate(steps[1:], 1):
            if step.step_type == ReasoningStepType.CALCULATION and step.input_values:
                # 检查输入值是否来自前序步骤
                for input_val in step.input_values:
                    if isinstance(input_val, (int, float)):
                        found_source = False
                        for prev_step in steps[:i]:
                            if (isinstance(prev_step.output_value, (int, float)) and 
                                abs(float(input_val) - float(prev_step.output_value)) < 1e-6):
                                found_source = True
                                break
                        
                        if not found_source and abs(float(input_val)) > 1e-6:
                            # 检查是否是原始输入（来自问题文本）
                            problem_context = context.get("problem_context", {})
                            problem_text = problem_context.get("problem_text", "")
                            
                            if str(input_val) not in problem_text:
                                errors.append(ValidationError(
                                    error_type=ErrorType.CONSISTENCY_ERROR,
                                    step_id=step.step_id,
                                    description=f"输入值{input_val}无明确来源",
                                    severity=0.3,
                                    suggestion="确保所有输入值都有明确的来源",
                                    metadata={"orphan_value": input_val}
                                ))
        
        return errors
    
    def _calculate_consistency_score(self, context: Dict[str, Any]) -> float:
        """计算一致性分数"""
        errors = context["errors"]
        steps = context["steps"]
        
        if not steps:
            return 0.0
        
        # 基础分数
        base_score = 1.0
        
        # 根据错误严重程度扣分
        total_penalty = 0.0
        for error in errors:
            penalty = error.severity * 0.1  # 每个错误最多扣0.1分
            total_penalty += penalty
        
        # 应用惩罚
        score = max(0.0, base_score - total_penalty)
        
        # 根据警告数量微调
        warning_penalty = len(context.get("warnings", [])) * 0.01
        score = max(0.0, score - warning_penalty)
        
        return score
    
    def _auto_correct_errors(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """自动纠错"""
        corrected_steps = context["steps"].copy()
        errors = context["errors"]
        
        correction_count = 0
        
        for error in errors:
            if correction_count >= self.max_correction_attempts:
                break
            
            if error.error_type == ErrorType.MATHEMATICAL_ERROR:
                corrected = self._correct_mathematical_error(corrected_steps, error)
                if corrected:
                    correction_count += 1
                    self.stats["corrections_made"] += 1
            
            elif error.error_type == ErrorType.VALUE_ERROR:
                corrected = self._correct_value_error(corrected_steps, error)
                if corrected:
                    correction_count += 1
                    self.stats["corrections_made"] += 1
        
        return corrected_steps
    
    def _correct_mathematical_error(self, steps: List[ReasoningStep], error: ValidationError) -> bool:
        """纠正数学错误"""
        try:
            step_id = error.step_id
            if 0 <= step_id < len(steps):
                step = steps[step_id]
                
                # 重新计算简单运算
                if step.operation == "addition" and len(step.input_values) >= 2:
                    nums = [float(x) for x in step.input_values[:2]]
                    corrected_output = nums[0] + nums[1]
                    step.output_value = corrected_output
                    step.metadata["auto_corrected"] = True
                    return True
                
                elif step.operation == "subtraction" and len(step.input_values) >= 2:
                    nums = [float(x) for x in step.input_values[:2]]
                    corrected_output = nums[0] - nums[1]
                    step.output_value = corrected_output
                    step.metadata["auto_corrected"] = True
                    return True
        
        except (ValueError, IndexError, TypeError):
            pass
        
        return False
    
    def _correct_value_error(self, steps: List[ReasoningStep], error: ValidationError) -> bool:
        """纠正数值错误"""
        try:
            step_id = error.step_id
            if 0 <= step_id < len(steps):
                step = steps[step_id]
                
                # 纠正负值问题
                if (isinstance(step.output_value, (int, float)) and 
                    float(step.output_value) < 0 and
                    step.operation in ["distance", "area", "volume", "count"]):
                    step.output_value = abs(float(step.output_value))
                    step.metadata["auto_corrected"] = True
                    step.metadata["correction_type"] = "absolute_value"
                    return True
        
        except (ValueError, TypeError):
            pass
        
        return False
    
    def _generate_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """生成建议"""
        suggestions = []
        errors = context["errors"]
        
        # 基于错误类型生成建议
        error_types = {error.error_type for error in errors}
        
        if ErrorType.MATHEMATICAL_ERROR in error_types:
            suggestions.append("建议重新检查数学计算，特别是基本运算")
        
        if ErrorType.LOGICAL_ERROR in error_types:
            suggestions.append("建议重新审查推理逻辑，确保步骤顺序合理")
        
        if ErrorType.DEPENDENCY_ERROR in error_types:
            suggestions.append("建议检查步骤间的依赖关系，确保引用的是正确的前序步骤")
        
        if ErrorType.CONSISTENCY_ERROR in error_types:
            suggestions.append("建议检查数值的传递和使用，确保一致性")
        
        # 基于错误数量生成建议
        if len(errors) > 5:
            suggestions.append("错误较多，建议重新设计推理流程")
        elif len(errors) > 2:
            suggestions.append("建议逐步检查每个推理步骤的正确性")
        
        return suggestions
    
    def _determine_validity(self, context: Dict[str, Any], consistency_score: float) -> bool:
        """判断有效性"""
        errors = context["errors"]
        
        # 检查是否有关键错误
        critical_errors = [e for e in errors if e.severity >= 0.8]
        if critical_errors:
            return False
        
        # 检查一致性分数
        if consistency_score < self.error_threshold:
            return False
        
        return True
    
    def _update_stats(self, context: Dict[str, Any], consistency_score: float, is_valid: bool):
        """更新统计信息"""
        self.stats["total_validations"] += 1
        
        if is_valid:
            self.stats["valid_chains"] += 1
        else:
            self.stats["invalid_chains"] += 1
        
        # 更新错误统计
        errors = context["errors"]
        self.stats["errors_found"] += len(errors)
        
        for error in errors:
            self.stats["error_type_counts"][error.error_type.value] += 1
        
        # 更新平均一致性分数
        current_avg = self.stats["average_consistency_score"]
        new_avg = ((current_avg * (self.stats["total_validations"] - 1) + consistency_score) / 
                  self.stats["total_validations"])
        self.stats["average_consistency_score"] = new_avg
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """初始化验证规则"""
        return {
            "structural_rules": [],
            "logical_rules": [],
            "mathematical_rules": [],
            "consistency_rules": []
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats["total_validations"] > 0:
            stats["validation_success_rate"] = stats["valid_chains"] / stats["total_validations"]
        else:
            stats["validation_success_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_validations": 0,
            "valid_chains": 0,
            "invalid_chains": 0,
            "errors_found": 0,
            "corrections_made": 0,
            "average_consistency_score": 0.0,
            "error_type_counts": {et.value: 0 for et in ErrorType}
        }
        self.logger.info("CV验证器统计信息已重置")