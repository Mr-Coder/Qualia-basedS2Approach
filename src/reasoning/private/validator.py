"""
推理结果验证器

负责验证推理结果的合理性和正确性。
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加src_new到路径
src_new_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_new_path))

from core.exceptions import ValidationError
from core.interfaces import BaseValidator


class ReasoningValidator(BaseValidator):
    """推理结果验证器"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """获取验证规则"""
        return {
            "required_fields": ["final_answer", "confidence", "reasoning_steps"],
            "answer_constraints": {
                "max_length": 100,
                "min_confidence": 0.1,
                "numeric_range": [-1000000, 1000000]
            },
            "step_constraints": {
                "required_step_fields": ["step", "action", "description"],
                "max_steps": 50
            }
        }
        
    def validate(self, data: Any) -> Dict[str, Any]:
        """验证推理结果"""
        try:
            if isinstance(data, dict) and "final_answer" in data:
                return self._validate_reasoning_result(data)
            elif isinstance(data, str):
                return self._validate_answer_string(data)
            else:
                return self._validate_input_problem(data)
                
        except Exception as e:
            self._logger.error(f"Validation failed: {e}")
            raise ValidationError(f"Validation failed: {e}", module_name="reasoning")
    
    def _validate_reasoning_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证完整的推理结果"""
        validation_result = {
            "is_valid": True,
            "confidence": 0.0,
            "issues": [],
            "warnings": []
        }
        
        # 验证必要字段
        required_fields = ["final_answer", "confidence", "reasoning_steps"]
        for field in required_fields:
            if field not in result:
                validation_result["issues"].append(f"Missing required field: {field}")
                validation_result["is_valid"] = False
        
        if not validation_result["is_valid"]:
            return validation_result
        
        # 验证答案合理性
        answer_validation = self._validate_answer_string(result["final_answer"])
        if not answer_validation["is_valid"]:
            validation_result["issues"].extend(answer_validation["issues"])
            validation_result["is_valid"] = False
        
        # 验证置信度
        confidence_validation = self._validate_confidence(result["confidence"])
        if not confidence_validation["is_valid"]:
            validation_result["issues"].extend(confidence_validation["issues"])
            validation_result["is_valid"] = False
        
        # 验证推理步骤
        steps_validation = self._validate_reasoning_steps(result["reasoning_steps"])
        if not steps_validation["is_valid"]:
            validation_result["issues"].extend(steps_validation["issues"])
            validation_result["is_valid"] = False
        
        # 计算总体置信度
        if validation_result["is_valid"]:
            validation_result["confidence"] = min(
                answer_validation["confidence"],
                confidence_validation["confidence"],
                steps_validation["confidence"]
            )
        
        return validation_result
    
    def _validate_answer_string(self, answer: str) -> Dict[str, Any]:
        """验证答案字符串的合理性"""
        validation_result = {
            "is_valid": True,
            "confidence": 0.0,
            "issues": []
        }
        
        if not answer or answer.strip() == "":
            validation_result["issues"].append("Answer is empty")
            validation_result["is_valid"] = False
            return validation_result
        
        # 检查是否为 "unknown"
        if answer.lower() in ["unknown", "无法确定", "不知道"]:
            validation_result["confidence"] = 0.1
            validation_result["issues"].append("Answer is unknown")
            return validation_result
        
        # 尝试解析为数字
        try:
            answer_num = float(answer)
            
            # 检查数字范围合理性
            if answer_num < 0:
                validation_result["issues"].append("Negative answer may be unusual")
                validation_result["confidence"] = 0.6
            elif answer_num > 1000000:
                validation_result["issues"].append("Very large answer may be unrealistic")
                validation_result["confidence"] = 0.5
            else:
                validation_result["confidence"] = 0.9
                
            # 检查是否为整数但表示为小数
            if answer_num.is_integer() and "." in answer:
                validation_result["issues"].append("Integer represented as decimal")
                
        except ValueError:
            # 非数字答案，检查是否为合理的文本答案
            if len(answer) > 100:
                validation_result["issues"].append("Answer text is too long")
                validation_result["confidence"] = 0.4
            else:
                validation_result["confidence"] = 0.7
        
        return validation_result
    
    def _validate_confidence(self, confidence: float) -> Dict[str, Any]:
        """验证置信度值"""
        validation_result = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": []
        }
        
        if not isinstance(confidence, (int, float)):
            validation_result["issues"].append("Confidence must be a number")
            validation_result["is_valid"] = False
            return validation_result
        
        if confidence < 0 or confidence > 1:
            validation_result["issues"].append("Confidence must be between 0 and 1")
            validation_result["is_valid"] = False
            return validation_result
        
        # 低置信度警告
        if confidence < 0.3:
            validation_result["issues"].append("Very low confidence")
            validation_result["confidence"] = 0.3
        elif confidence < 0.5:
            validation_result["issues"].append("Low confidence")
            validation_result["confidence"] = 0.6
        
        return validation_result
    
    def _validate_reasoning_steps(self, steps: List[Dict]) -> Dict[str, Any]:
        """验证推理步骤"""
        validation_result = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": []
        }
        
        if not isinstance(steps, list):
            validation_result["issues"].append("Reasoning steps must be a list")
            validation_result["is_valid"] = False
            return validation_result
        
        if len(steps) == 0:
            validation_result["issues"].append("No reasoning steps provided")
            validation_result["confidence"] = 0.3
        
        # 验证每个步骤
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                validation_result["issues"].append(f"Step {i} is not a dictionary")
                validation_result["is_valid"] = False
                continue
            
            # 检查必要字段
            required_step_fields = ["step", "action", "description"]
            for field in required_step_fields:
                if field not in step:
                    validation_result["issues"].append(f"Step {i} missing field: {field}")
                    validation_result["confidence"] = min(validation_result["confidence"], 0.7)
        
        return validation_result
    
    def _validate_input_problem(self, problem: Any) -> Dict[str, Any]:
        """验证输入问题"""
        validation_result = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": []
        }
        
        if not isinstance(problem, dict):
            validation_result["issues"].append("Problem must be a dictionary")
            validation_result["is_valid"] = False
            return validation_result
        
        # 检查问题文本
        problem_text = problem.get("problem") or problem.get("cleaned_text") or ""
        if not problem_text or problem_text.strip() == "":
            validation_result["issues"].append("Problem text is empty")
            validation_result["is_valid"] = False
        
        # 检查问题长度
        if len(problem_text) > 1000:
            validation_result["issues"].append("Problem text is very long")
            validation_result["confidence"] = 0.8
        elif len(problem_text) < 10:
            validation_result["issues"].append("Problem text is very short")
            validation_result["confidence"] = 0.7
        
        return validation_result
    
    def validate_template_match(self, text: str, template_info: Optional[Dict]) -> Dict[str, Any]:
        """验证模板匹配结果"""
        validation_result = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": []
        }
        
        if template_info is None:
            validation_result["issues"].append("No template matched")
            validation_result["confidence"] = 0.5
            return validation_result
        
        # 验证模板信息完整性
        required_fields = ["type", "template", "pattern_matched"]
        for field in required_fields:
            if field not in template_info:
                validation_result["issues"].append(f"Template missing field: {field}")
                validation_result["confidence"] = min(validation_result["confidence"], 0.7)
        
        # 验证模式匹配
        if "pattern_matched" in template_info:
            pattern = template_info["pattern_matched"]
            try:
                if not re.search(pattern, text):
                    validation_result["issues"].append("Pattern does not match text")
                    validation_result["confidence"] = 0.3
            except re.error:
                validation_result["issues"].append("Invalid regex pattern")
                validation_result["confidence"] = 0.4
        
        return validation_result 