"""
置信度计算器

负责计算推理过程的置信度分数，提供多维度的可信度评估。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple


class ConfidenceCalculator:
    """置信度计算器"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        # 置信度权重配置
        self.weights = {
            "step_confidence": 0.4,      # 单步置信度权重
            "sequence_consistency": 0.3,  # 序列一致性权重
            "result_validation": 0.2,     # 结果验证权重
            "knowledge_support": 0.1      # 知识支持权重
        }
    
    def calculate_overall_confidence(self, reasoning_steps: List[Dict[str, Any]], 
                                   validation_result: Optional[Dict] = None,
                                   knowledge_context: Optional[Dict] = None) -> float:
        """计算整体置信度"""
        try:
            if not reasoning_steps:
                return 0.1
            
            # 计算各个维度的置信度
            step_conf = self._calculate_step_confidence(reasoning_steps)
            sequence_conf = self._calculate_sequence_consistency(reasoning_steps)
            validation_conf = self._calculate_validation_confidence(validation_result)
            knowledge_conf = self._calculate_knowledge_confidence(knowledge_context)
            
            # 加权平均
            overall_confidence = (
                self.weights["step_confidence"] * step_conf +
                self.weights["sequence_consistency"] * sequence_conf +
                self.weights["result_validation"] * validation_conf +
                self.weights["knowledge_support"] * knowledge_conf
            )
            
            # 确保在[0, 1]范围内
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            self._logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # 默认中等置信度
    
    def _calculate_step_confidence(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """计算步骤置信度"""
        if not reasoning_steps:
            return 0.1
        
        confidences = []
        for step in reasoning_steps:
            step_conf = step.get("confidence", 0.5)
            
            # 根据步骤类型调整置信度
            action = step.get("action", "")
            if action == "expression_parsing":
                step_conf *= 1.1  # 表达式解析通常更可靠
            elif action == "template_identification":
                step_conf *= 1.05  # 模板识别相对可靠
            elif action == "fallback_reasoning":
                step_conf *= 0.5  # 回退推理降低置信度
            elif action == "error_handling":
                step_conf *= 0.2  # 错误处理大幅降低置信度
            
            confidences.append(min(1.0, step_conf))
        
        # 计算加权平均（较后的步骤权重稍高）
        weights = [1 + 0.1 * i for i in range(len(confidences))]
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.5
    
    def _calculate_sequence_consistency(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """计算推理序列一致性"""
        if len(reasoning_steps) <= 1:
            return 0.9  # 单步或无步骤认为一致性很高
        
        consistency_score = 1.0
        
        # 检查步骤编号连续性
        step_numbers = [step.get("step", 0) for step in reasoning_steps]
        expected_numbers = list(range(1, len(reasoning_steps) + 1))
        if step_numbers != expected_numbers:
            consistency_score -= 0.2
        
        # 检查逻辑顺序
        actions = [step.get("action", "") for step in reasoning_steps]
        
        # 数字提取应该在表达式解析之前
        if "number_extraction" in actions and "expression_parsing" in actions:
            extraction_idx = actions.index("number_extraction")
            parsing_idx = actions.index("expression_parsing")
            if extraction_idx > parsing_idx:
                consistency_score -= 0.15
        
        # 模板识别应该在模板应用之前
        template_actions = [action for action in actions if "template" in action]
        calculation_actions = [action for action in actions if "calculation" in action]
        if template_actions and calculation_actions:
            first_template = min(i for i, action in enumerate(actions) if "template" in action)
            first_calc = min(i for i, action in enumerate(actions) if "calculation" in action)
            if first_template > first_calc:
                consistency_score -= 0.1
        
        # 检查置信度递减趋势（通常推理越深入置信度应该略有下降）
        confidences = [step.get("confidence", 0.5) for step in reasoning_steps]
        if len(confidences) > 2:
            # 允许小幅波动，但不应该有大幅上升
            for i in range(1, len(confidences)):
                if confidences[i] > confidences[i-1] + 0.3:
                    consistency_score -= 0.05
        
        return max(0.0, consistency_score)
    
    def _calculate_validation_confidence(self, validation_result: Optional[Dict]) -> float:
        """计算验证置信度"""
        if not validation_result:
            return 0.5  # 无验证结果时返回中等置信度
        
        is_valid = validation_result.get("is_valid", False)
        if not is_valid:
            return 0.2  # 验证失败时置信度很低
        
        # 基于验证结果的置信度
        base_confidence = validation_result.get("confidence", 0.8)
        
        # 根据问题数量调整
        issues = validation_result.get("issues", [])
        warnings = validation_result.get("warnings", [])
        
        confidence_penalty = len(issues) * 0.1 + len(warnings) * 0.05
        adjusted_confidence = base_confidence - confidence_penalty
        
        return max(0.1, min(1.0, adjusted_confidence))
    
    def _calculate_knowledge_confidence(self, knowledge_context: Optional[Dict]) -> float:
        """计算知识支持置信度"""
        if not knowledge_context:
            return 0.5  # 无知识上下文时返回中等置信度
        
        confidence = 0.5
        
        # 有概念支持
        concepts = knowledge_context.get("concepts", [])
        if concepts:
            confidence += 0.2 * min(len(concepts) / 3, 1.0)  # 最多增加0.2
        
        # 有策略支持
        strategies = knowledge_context.get("strategies", [])
        if strategies:
            confidence += 0.2 * min(len(strategies) / 2, 1.0)  # 最多增加0.2
        
        # 有验证支持
        if knowledge_context.get("validation_support"):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_confidence_distribution(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析置信度分布"""
        if not reasoning_steps:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "distribution": "empty"
            }
        
        confidences = [step.get("confidence", 0.5) for step in reasoning_steps]
        
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        std_conf = math.sqrt(variance)
        
        # 分析分布模式
        distribution_type = "stable"
        if std_conf > 0.3:
            distribution_type = "highly_variable"
        elif std_conf > 0.15:
            distribution_type = "moderately_variable"
        
        # 分析趋势
        if len(confidences) > 2:
            increasing = sum(1 for i in range(1, len(confidences)) 
                           if confidences[i] > confidences[i-1])
            decreasing = sum(1 for i in range(1, len(confidences)) 
                           if confidences[i] < confidences[i-1])
            
            if increasing > decreasing * 1.5:
                distribution_type += "_increasing"
            elif decreasing > increasing * 1.5:
                distribution_type += "_decreasing"
        
        return {
            "mean": mean_conf,
            "std": std_conf,
            "min": min(confidences),
            "max": max(confidences),
            "distribution": distribution_type,
            "count": len(confidences)
        }
    
    def get_confidence_explanation(self, overall_confidence: float, 
                                 components: Dict[str, float]) -> str:
        """生成置信度解释"""
        if overall_confidence >= 0.8:
            level = "高"
        elif overall_confidence >= 0.6:
            level = "中高"
        elif overall_confidence >= 0.4:
            level = "中等"
        elif overall_confidence >= 0.2:
            level = "较低"
        else:
            level = "很低"
        
        explanation = f"整体置信度: {level} ({overall_confidence:.2f})"
        
        # 分析主要影响因素
        max_component = max(components.items(), key=lambda x: x[1])
        min_component = min(components.items(), key=lambda x: x[1])
        
        explanation += f"\n主要优势: {max_component[0]} ({max_component[1]:.2f})"
        explanation += f"\n主要弱项: {min_component[0]} ({min_component[1]:.2f})"
        
        return explanation 