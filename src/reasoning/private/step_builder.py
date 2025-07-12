"""
推理步骤构建器

负责构建结构化的推理步骤，为推理过程提供清晰的步骤记录。
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional


class StepBuilder:
    """推理步骤构建器"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._current_step = 0
        
    def reset(self):
        """重置步骤计数器"""
        self._current_step = 0
    
    def create_step(self, action: str, description: str, **kwargs) -> Dict[str, Any]:
        """创建一个推理步骤"""
        self._current_step += 1
        
        step = {
            "step": self._current_step,
            "action": action,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "confidence": kwargs.get("confidence", 0.8)
        }
        
        # 添加其他可选字段
        for key, value in kwargs.items():
            if key not in step:
                step[key] = value
        
        return step
    
    def create_number_extraction_step(self, numbers: List[float]) -> Dict[str, Any]:
        """创建数字提取步骤"""
        return self.create_step(
            action="number_extraction",
            description=f"从文本中提取数字: {numbers}",
            numbers=numbers,
            confidence=0.9
        )
    
    def create_expression_parsing_step(self, expression: str, result: Any) -> Dict[str, Any]:
        """创建表达式解析步骤"""
        return self.create_step(
            action="expression_parsing",
            description=f"解析数学表达式: {expression} = {result}",
            expression=expression,
            result=result,
            confidence=0.95
        )
    
    def create_template_identification_step(self, template_type: str, template: str) -> Dict[str, Any]:
        """创建模板识别步骤"""
        return self.create_step(
            action="template_identification",
            description=f"识别问题类型: {template_type}",
            template_type=template_type,
            template=template,
            confidence=0.8
        )
    
    def create_calculation_step(self, operation: str, operands: List[Any], 
                               result: Any, calculation_details: str = None) -> Dict[str, Any]:
        """创建计算步骤"""
        description = f"执行{operation}计算"
        if calculation_details:
            description += f": {calculation_details}"
        
        return self.create_step(
            action=f"{operation}_calculation",
            description=description,
            operation=operation,
            operands=operands,
            result=result,
            calculation=calculation_details,
            confidence=0.9
        )
    
    def create_validation_step(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建验证步骤"""
        status = "通过" if validation_result.get("is_valid", False) else "失败"
        issues = validation_result.get("issues", [])
        
        description = f"答案验证{status}"
        if issues:
            description += f", 发现问题: {', '.join(issues[:2])}"
            if len(issues) > 2:
                description += f" 等{len(issues)}个问题"
        
        return self.create_step(
            action="answer_validation",
            description=description,
            validation_result=validation_result,
            confidence=validation_result.get("confidence", 0.5)
        )
    
    def create_knowledge_enhancement_step(self, concepts: List[str], 
                                        strategies: List[str]) -> Dict[str, Any]:
        """创建知识增强步骤"""
        return self.create_step(
            action="knowledge_enhancement",
            description=f"应用元知识增强: 概念[{', '.join(concepts)}], 策略[{', '.join(strategies)}]",
            concepts=concepts,
            strategies=strategies,
            confidence=0.7
        )
    
    def create_fallback_step(self, reason: str, fallback_action: str) -> Dict[str, Any]:
        """创建回退步骤"""
        return self.create_step(
            action="fallback_reasoning",
            description=f"回退推理: {reason}, 采用{fallback_action}",
            reason=reason,
            fallback_action=fallback_action,
            confidence=0.3
        )
    
    def create_error_step(self, error_message: str, error_type: str = "unknown") -> Dict[str, Any]:
        """创建错误步骤"""
        return self.create_step(
            action="error_handling",
            description=f"处理错误: {error_message}",
            error_message=error_message,
            error_type=error_type,
            confidence=0.1
        )
    
    def build_step_summary(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建步骤摘要"""
        if not steps:
            return {
                "total_steps": 0,
                "average_confidence": 0.0,
                "actions_performed": [],
                "has_errors": False
            }
        
        actions = [step.get("action", "unknown") for step in steps]
        confidences = [step.get("confidence", 0.5) for step in steps]
        has_errors = any(step.get("action") == "error_handling" for step in steps)
        
        return {
            "total_steps": len(steps),
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "actions_performed": list(set(actions)),
            "has_errors": has_errors,
            "confidence_range": {
                "min": min(confidences) if confidences else 0.0,
                "max": max(confidences) if confidences else 0.0
            }
        }
    
    def validate_step_sequence(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证步骤序列的合理性"""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }
        
        if not steps:
            validation_result["issues"].append("No steps provided")
            validation_result["is_valid"] = False
            return validation_result
        
        # 检查步骤编号连续性
        step_numbers = [step.get("step", 0) for step in steps]
        if step_numbers != list(range(1, len(steps) + 1)):
            validation_result["issues"].append("Step numbers are not consecutive")
        
        # 检查必要字段
        required_fields = ["step", "action", "description"]
        for i, step in enumerate(steps):
            for field in required_fields:
                if field not in step:
                    validation_result["issues"].append(f"Step {i+1} missing required field: {field}")
        
        # 检查逻辑顺序
        actions = [step.get("action", "") for step in steps]
        if "number_extraction" in actions and "expression_parsing" in actions:
            extraction_index = actions.index("number_extraction")
            parsing_index = actions.index("expression_parsing")
            if extraction_index > parsing_index:
                validation_result["suggestions"].append("Number extraction should come before expression parsing")
        
        if validation_result["issues"]:
            validation_result["is_valid"] = False
        
        return validation_result 