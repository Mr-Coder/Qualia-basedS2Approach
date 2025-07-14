"""
置信度计算器基类
定义置信度计算的通用接口和基础实现
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ConfidenceResult:
    """置信度计算结果"""
    overall_confidence: float
    component_confidences: Dict[str, float]
    confidence_factors: List[str]
    calculation_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "overall_confidence": self.overall_confidence,
            "component_confidences": self.component_confidences,
            "confidence_factors": self.confidence_factors,
            "calculation_details": self.calculation_details
        }

class ConfidenceCalculator(ABC):
    """置信度计算器基类"""
    
    def __init__(self, name: str):
        """
        初始化置信度计算器
        
        Args:
            name: 计算器名称
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 置信度权重配置
        self.confidence_weights = {
            "step_confidence": 0.4,      # 单步置信度
            "logical_consistency": 0.2,  # 逻辑一致性
            "numerical_accuracy": 0.2,   # 数值准确性
            "validation_result": 0.1,    # 验证结果
            "complexity_penalty": 0.1    # 复杂度惩罚
        }
        
        # 置信度阈值
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        self.logger.info(f"置信度计算器 {name} 初始化完成")
    
    @abstractmethod
    def calculate_confidence(self, reasoning_steps: List[Dict[str, Any]], 
                           final_result: Any, 
                           context: Optional[Dict[str, Any]] = None) -> ConfidenceResult:
        """
        计算整体置信度
        
        Args:
            reasoning_steps: 推理步骤列表
            final_result: 最终结果
            context: 上下文信息
            
        Returns:
            ConfidenceResult: 置信度计算结果
        """
        pass
    
    def calculate_step_confidence(self, step: Dict[str, Any]) -> float:
        """
        计算单个步骤的置信度
        
        Args:
            step: 推理步骤
            
        Returns:
            float: 步骤置信度
        """
        # 基础置信度（来自步骤本身）
        base_confidence = step.get("confidence", 0.5)
        
        # 调整因子
        adjustments = []
        
        # 1. 操作类型调整
        action = step.get("action", "").lower()
        if action in ["number_extraction", "calculation", "addition", "subtraction"]:
            adjustments.append(0.1)  # 数学操作通常更可靠
        elif action in ["reasoning", "analysis"]:
            adjustments.append(-0.05)  # 推理操作不确定性更高
        
        # 2. 数据完整性调整
        if "result" in step and step["result"] is not None:
            adjustments.append(0.05)  # 有具体结果的步骤更可信
        
        if "numbers" in step and len(step.get("numbers", [])) > 0:
            adjustments.append(0.05)  # 有数字支撑的步骤更可信
        
        # 3. 描述完整性调整
        description = step.get("description", "")
        if len(description) > 10:
            adjustments.append(0.02)  # 详细描述的步骤更可信
        
        # 应用调整
        adjusted_confidence = base_confidence + sum(adjustments)
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def calculate_logical_consistency(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """
        计算推理步骤的逻辑一致性
        
        Args:
            reasoning_steps: 推理步骤列表
            
        Returns:
            float: 逻辑一致性分数
        """
        if not reasoning_steps:
            return 0.0
        
        consistency_score = 1.0
        
        # 检查步骤间的连接性
        for i in range(len(reasoning_steps) - 1):
            current_step = reasoning_steps[i]
            next_step = reasoning_steps[i + 1]
            
            # 检查输出是否能作为下一步的输入
            current_result = current_step.get("result")
            next_inputs = next_step.get("inputs", next_step.get("numbers", []))
            
            if current_result is not None and next_inputs:
                if isinstance(next_inputs, list) and current_result not in next_inputs:
                    consistency_score -= 0.1  # 步骤间缺乏连接
        
        # 检查操作的合理性
        for step in reasoning_steps:
            action = step.get("action", "").lower()
            result = step.get("result")
            
            # 数学操作的合理性检查
            if action in ["addition", "subtraction", "multiplication", "division"]:
                numbers = step.get("numbers", step.get("operands", []))
                if numbers and result is not None:
                    # 简单的合理性检查
                    if action == "addition" and result < max(numbers):
                        consistency_score -= 0.1
                    elif action == "multiplication" and len(numbers) == 2 and result < max(numbers):
                        consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def calculate_numerical_accuracy(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """
        计算数值计算的准确性
        
        Args:
            reasoning_steps: 推理步骤列表
            
        Returns:
            float: 数值准确性分数
        """
        accuracy_score = 1.0
        calculation_steps = 0
        
        for step in reasoning_steps:
            action = step.get("action", "").lower()
            
            if action in ["addition", "subtraction", "multiplication", "division", "calculation"]:
                calculation_steps += 1
                
                # 检查计算是否有数字支撑
                numbers = step.get("numbers", step.get("operands", []))
                result = step.get("result")
                
                if not numbers:
                    accuracy_score -= 0.2  # 缺少数字的计算步骤
                elif result is None:
                    accuracy_score -= 0.3  # 没有计算结果
                else:
                    # 尝试验证计算结果
                    if self._verify_calculation(numbers, action, result):
                        accuracy_score += 0.1  # 计算正确
                    else:
                        accuracy_score -= 0.3  # 计算错误
        
        # 如果没有计算步骤，给中等分数
        if calculation_steps == 0:
            accuracy_score = 0.7
        
        return max(0.0, min(1.0, accuracy_score))
    
    def _verify_calculation(self, numbers: List, action: str, result: Any) -> bool:
        """
        验证计算结果
        
        Args:
            numbers: 操作数
            action: 操作类型
            result: 计算结果
            
        Returns:
            bool: 计算是否正确
        """
        try:
            numbers = [float(n) for n in numbers if isinstance(n, (int, float, str)) and 
                      str(n).replace('.', '').replace('-', '').isdigit()]
            
            if not numbers or result is None:
                return False
            
            result = float(result)
            
            if action in ["addition", "add"]:
                expected = sum(numbers)
            elif action in ["subtraction", "subtract"]:
                expected = numbers[0] - sum(numbers[1:]) if len(numbers) > 1 else numbers[0]
            elif action in ["multiplication", "multiply"]:
                expected = 1
                for n in numbers:
                    expected *= n
            elif action in ["division", "divide"]:
                if len(numbers) >= 2 and numbers[1] != 0:
                    expected = numbers[0] / numbers[1]
                else:
                    return False
            else:
                return True  # 无法验证的操作默认为正确
            
            # 允许小的浮点误差
            return abs(result - expected) < 1e-6
            
        except (ValueError, TypeError, ZeroDivisionError):
            return False
    
    def calculate_validation_confidence(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """
        基于验证结果计算置信度
        
        Args:
            reasoning_steps: 推理步骤列表
            
        Returns:
            float: 验证置信度
        """
        validation_score = 0.7  # 默认中等置信度
        validation_found = False
        
        for step in reasoning_steps:
            action = step.get("action", "").lower()
            
            if "validation" in action or "verify" in action:
                validation_found = True
                validation_result = step.get("validation", step.get("result", {}))
                
                if isinstance(validation_result, dict):
                    if validation_result.get("valid", False):
                        validation_score = max(validation_score, validation_result.get("confidence", 0.8))
                    else:
                        validation_score = min(validation_score, 0.3)
        
        # 如果没有找到验证步骤，给较低的分数
        if not validation_found:
            validation_score = 0.5
        
        return validation_score
    
    def calculate_complexity_penalty(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """
        基于复杂度计算惩罚因子
        
        Args:
            reasoning_steps: 推理步骤列表
            
        Returns:
            float: 复杂度惩罚（越复杂惩罚越大，返回值越小）
        """
        if not reasoning_steps:
            return 1.0
        
        # 基于步骤数量的惩罚
        step_count = len(reasoning_steps)
        step_penalty = 1.0 - (step_count - 3) * 0.05 if step_count > 3 else 1.0
        
        # 基于步骤复杂度的惩罚
        complexity_levels = []
        for step in reasoning_steps:
            action = step.get("action", "").lower()
            
            if action in ["number_extraction", "addition", "subtraction"]:
                complexity_levels.append(1)  # 简单操作
            elif action in ["multiplication", "division", "calculation"]:
                complexity_levels.append(2)  # 中等操作
            elif action in ["reasoning", "analysis", "synthesis"]:
                complexity_levels.append(3)  # 复杂操作
            else:
                complexity_levels.append(2)  # 默认中等
        
        avg_complexity = sum(complexity_levels) / len(complexity_levels)
        complexity_penalty = 1.0 - (avg_complexity - 1.5) * 0.1
        
        return max(0.3, min(1.0, step_penalty * complexity_penalty))
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        获取置信度等级
        
        Args:
            confidence: 置信度值
            
        Returns:
            str: 置信度等级
        """
        if confidence >= self.confidence_thresholds["high"]:
            return "high"
        elif confidence >= self.confidence_thresholds["medium"]:
            return "medium"
        elif confidence >= self.confidence_thresholds["low"]:
            return "low"
        else:
            return "very_low"
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        更新置信度权重
        
        Args:
            new_weights: 新的权重配置
        """
        self.confidence_weights.update(new_weights)
        
        # 确保权重总和为1
        total_weight = sum(self.confidence_weights.values())
        if total_weight != 1.0:
            for key in self.confidence_weights:
                self.confidence_weights[key] /= total_weight
        
        self.logger.info(f"置信度权重已更新: {self.confidence_weights}")

class BasicConfidenceCalculator(ConfidenceCalculator):
    """基础置信度计算器"""
    
    def __init__(self):
        super().__init__("basic_confidence_calculator")
    
    def calculate_confidence(self, reasoning_steps: List[Dict[str, Any]], 
                           final_result: Any, 
                           context: Optional[Dict[str, Any]] = None) -> ConfidenceResult:
        """
        计算基础置信度
        
        使用简单的加权平均方法
        """
        try:
            # 计算各个组件的置信度
            step_conf = self._calculate_average_step_confidence(reasoning_steps)
            logical_conf = self.calculate_logical_consistency(reasoning_steps)
            numerical_conf = self.calculate_numerical_accuracy(reasoning_steps)
            validation_conf = self.calculate_validation_confidence(reasoning_steps)
            complexity_penalty = self.calculate_complexity_penalty(reasoning_steps)
            
            component_confidences = {
                "step_confidence": step_conf,
                "logical_consistency": logical_conf,
                "numerical_accuracy": numerical_conf,
                "validation_result": validation_conf,
                "complexity_penalty": complexity_penalty
            }
            
            # 计算加权总和
            overall_confidence = (
                step_conf * self.confidence_weights["step_confidence"] +
                logical_conf * self.confidence_weights["logical_consistency"] +
                numerical_conf * self.confidence_weights["numerical_accuracy"] +
                validation_conf * self.confidence_weights["validation_result"]
            ) * complexity_penalty
            
            # 确保在有效范围内
            overall_confidence = max(0.0, min(1.0, overall_confidence))
            
            # 识别主要置信度因子
            confidence_factors = self._identify_confidence_factors(component_confidences)
            
            calculation_details = {
                "method": "weighted_average",
                "weights": self.confidence_weights.copy(),
                "step_count": len(reasoning_steps),
                "confidence_level": self.get_confidence_level(overall_confidence)
            }
            
            return ConfidenceResult(
                overall_confidence=overall_confidence,
                component_confidences=component_confidences,
                confidence_factors=confidence_factors,
                calculation_details=calculation_details
            )
            
        except Exception as e:
            self.logger.error(f"置信度计算失败: {str(e)}")
            
            return ConfidenceResult(
                overall_confidence=0.5,  # 默认中等置信度
                component_confidences={},
                confidence_factors=["calculation_error"],
                calculation_details={"error": str(e)}
            )
    
    def _calculate_average_step_confidence(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """计算平均步骤置信度"""
        if not reasoning_steps:
            return 0.0
        
        step_confidences = [self.calculate_step_confidence(step) for step in reasoning_steps]
        return sum(step_confidences) / len(step_confidences)
    
    def _identify_confidence_factors(self, component_confidences: Dict[str, float]) -> List[str]:
        """识别主要的置信度影响因子"""
        factors = []
        
        for component, confidence in component_confidences.items():
            if confidence >= 0.8:
                factors.append(f"高{component}")
            elif confidence <= 0.3:
                factors.append(f"低{component}")
        
        if not factors:
            factors.append("中等置信度")
        
        return factors 