"""
思维链推理策略 (Chain of Thought)
实现逐步推理的策略，适合中等复杂度的数学问题
"""

import re
import time
from typing import Any, Dict, List, Optional

from ...core.exceptions import ReasoningError
from ...core.interfaces import ReasoningContext
from .strategy_base import (ReasoningStrategy, StrategyComplexity,
                            StrategyResult, StrategyType)


class ChainOfThoughtStrategy(ReasoningStrategy):
    """思维链推理策略"""
    
    def __init__(self):
        super().__init__(
            name="chain_of_thought",
            strategy_type=StrategyType.CHAIN_OF_THOUGHT,
            complexity=StrategyComplexity.MODERATE
        )
        
        # 数学关键词模式
        self.math_keywords = {
            "arithmetic": ["加", "减", "乘", "除", "计算", "+", "-", "*", "/", "×", "÷"],
            "comparison": ["比较", "大于", "小于", "等于", "多", "少", "相等"],
            "quantity": ["总共", "一共", "合计", "总和", "剩余", "还有"],
            "rate": ["每", "平均", "速度", "效率", "比率"],
            "geometry": ["面积", "周长", "体积", "长度", "宽度", "高度", "直径", "半径"],
            "time": ["小时", "分钟", "天", "周", "月", "年", "秒"],
            "money": ["元", "角", "分", "价格", "成本", "利润", "折扣"]
        }
        
        # 运算模式
        self.operation_patterns = {
            "addition": [
                r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*加\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*和\s*(\d+(?:\.\d+)?)\s*的?\s*和",
            ],
            "subtraction": [
                r"(\d+(?:\.\d+)?)\s*\-\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*减\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*减去\s*(\d+(?:\.\d+)?)",
            ],
            "multiplication": [
                r"(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*×\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*乘\s*(\d+(?:\.\d+)?)",
            ],
            "division": [
                r"(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*÷\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*除以\s*(\d+(?:\.\d+)?)",
            ]
        }
    
    def can_handle(self, problem_text: str, context: Optional[ReasoningContext] = None) -> bool:
        """
        判断是否能处理该问题
        
        思维链策略适合：
        - 包含数学关键词的问题
        - 有明确数字的问题
        - 中等复杂度的推理问题
        """
        try:
            # 检查是否包含数字
            numbers = self._extract_numbers(problem_text)
            if len(numbers) < 1:
                return False
            
            # 检查是否包含数学关键词
            text_lower = problem_text.lower()
            for category, keywords in self.math_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    return True
            
            # 检查是否包含运算表达式
            for operation, patterns in self.operation_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, problem_text):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"判断能力检查失败: {str(e)}")
            return False
    
    def estimate_complexity(self, problem_text: str, context: Optional[ReasoningContext] = None) -> float:
        """
        估计问题复杂度
        
        复杂度因素：
        - 数字数量
        - 运算类型数量
        - 文本长度
        - 关键词多样性
        """
        try:
            complexity = 0.0
            
            # 基于数字数量 (0.0-0.3)
            numbers = self._extract_numbers(problem_text)
            num_count = len(numbers)
            if num_count <= 2:
                complexity += 0.1
            elif num_count <= 4:
                complexity += 0.2
            else:
                complexity += 0.3
            
            # 基于运算类型 (0.0-0.3)
            operation_types = set()
            for operation, patterns in self.operation_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, problem_text):
                        operation_types.add(operation)
            
            complexity += len(operation_types) * 0.075  # 最多4种运算
            
            # 基于文本长度 (0.0-0.2)
            text_length = len(problem_text)
            if text_length > 200:
                complexity += 0.2
            elif text_length > 100:
                complexity += 0.1
            else:
                complexity += 0.05
            
            # 基于关键词类别数量 (0.0-0.2)
            keyword_categories = 0
            text_lower = problem_text.lower()
            for category, keywords in self.math_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    keyword_categories += 1
            
            complexity += min(keyword_categories * 0.04, 0.2)
            
            return min(1.0, complexity)
            
        except Exception as e:
            self.logger.error(f"复杂度估计失败: {str(e)}")
            return 0.5  # 默认中等复杂度
    
    def _execute_reasoning(self, problem_text: str, context: Optional[ReasoningContext] = None) -> StrategyResult:
        """执行思维链推理"""
        start_time = time.time()
        reasoning_steps = []
        step_count = 0
        
        try:
            # 步骤1: 问题理解和分析
            step_count += 1
            understanding = self._analyze_problem(problem_text)
            reasoning_steps.append({
                "step": step_count,
                "action": "problem_analysis",
                "description": f"分析问题类型: {understanding['type']}",
                "details": understanding,
                "confidence": 0.9
            })
            
            # 步骤2: 数字和关键信息提取
            step_count += 1
            numbers = self._extract_numbers(problem_text)
            key_info = self._extract_key_information(problem_text)
            reasoning_steps.append({
                "step": step_count,
                "action": "information_extraction",
                "description": f"提取数字: {numbers}, 关键信息: {key_info}",
                "numbers": numbers,
                "key_info": key_info,
                "confidence": 0.95
            })
            
            # 步骤3: 推理路径规划
            step_count += 1
            reasoning_path = self._plan_reasoning_path(problem_text, numbers, understanding)
            reasoning_steps.append({
                "step": step_count,
                "action": "path_planning",
                "description": f"规划推理路径: {reasoning_path['strategy']}",
                "path": reasoning_path,
                "confidence": 0.8
            })
            
            # 步骤4: 逐步计算
            calculation_steps, final_answer = self._execute_calculations(
                problem_text, numbers, reasoning_path
            )
            reasoning_steps.extend(calculation_steps)
            
            # 步骤5: 答案验证
            step_count = len(reasoning_steps) + 1
            validation = self._validate_answer(final_answer, problem_text, numbers)
            reasoning_steps.append({
                "step": step_count,
                "action": "answer_validation",
                "description": f"验证答案: {validation['status']}",
                "validation": validation,
                "confidence": validation.get("confidence", 0.7)
            })
            
            # 计算整体置信度
            overall_confidence = self._calculate_overall_confidence(reasoning_steps)
            
            execution_time = time.time() - start_time
            
            return StrategyResult(
                success=True,
                answer=str(final_answer),
                confidence=overall_confidence,
                reasoning_steps=reasoning_steps,
                strategy_used=self.name,
                execution_time=execution_time,
                metadata={
                    "problem_type": understanding.get("type", "unknown"),
                    "numbers_found": len(numbers),
                    "calculation_steps": len(calculation_steps),
                    "validation_passed": validation.get("passed", False)
                }
            )
            
        except Exception as e:
            self.logger.error(f"思维链推理执行失败: {str(e)}")
            
            reasoning_steps.append({
                "step": step_count + 1,
                "action": "error",
                "description": f"推理过程出错: {str(e)}",
                "confidence": 0.0
            })
            
            return StrategyResult(
                success=False,
                answer="计算失败",
                confidence=0.0,
                reasoning_steps=reasoning_steps,
                strategy_used=self.name,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _analyze_problem(self, problem_text: str) -> Dict[str, Any]:
        """分析问题类型和结构"""
        analysis = {
            "type": "unknown",
            "operations": [],
            "keywords": [],
            "structure": "simple"
        }
        
        text_lower = problem_text.lower()
        
        # 识别问题类型
        if any(kw in text_lower for kw in self.math_keywords["arithmetic"]):
            analysis["type"] = "arithmetic"
        elif any(kw in text_lower for kw in self.math_keywords["geometry"]):
            analysis["type"] = "geometry"
        elif any(kw in text_lower for kw in self.math_keywords["rate"]):
            analysis["type"] = "rate_problem"
        elif any(kw in text_lower for kw in self.math_keywords["comparison"]):
            analysis["type"] = "comparison"
        
        # 识别运算类型
        for operation, patterns in self.operation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_text):
                    analysis["operations"].append(operation)
        
        # 提取关键词
        for category, keywords in self.math_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                analysis["keywords"].extend(found_keywords)
        
        # 判断结构复杂度
        if len(analysis["operations"]) > 2 or len(self._extract_numbers(problem_text)) > 4:
            analysis["structure"] = "complex"
        elif len(analysis["operations"]) > 1 or len(self._extract_numbers(problem_text)) > 2:
            analysis["structure"] = "moderate"
        
        return analysis
    
    def _extract_numbers(self, text: str) -> List[float]:
        """提取文本中的数字"""
        # 匹配整数和小数
        number_pattern = r'\d+(?:\.\d+)?'
        matches = re.findall(number_pattern, text)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def _extract_key_information(self, problem_text: str) -> Dict[str, Any]:
        """提取关键信息"""
        key_info = {
            "units": [],
            "conditions": [],
            "objectives": []
        }
        
        # 提取单位
        unit_patterns = [
            r'(\d+(?:\.\d+)?)\s*(元|角|分|米|厘米|千米|公里|小时|分钟|秒|天|个|只|本|张|斤|公斤)',
            r'(\d+(?:\.\d+)?)\s*(平方米|立方米|平方厘米|立方厘米)'
        ]
        
        for pattern in unit_patterns:
            matches = re.findall(pattern, problem_text)
            for number, unit in matches:
                key_info["units"].append({"number": float(number), "unit": unit})
        
        # 提取条件（包含"如果"、"假设"等）
        condition_keywords = ["如果", "假设", "已知", "条件"]
        for keyword in condition_keywords:
            if keyword in problem_text:
                key_info["conditions"].append(f"包含条件关键词: {keyword}")
        
        # 提取目标（包含"求"、"计算"等）
        objective_keywords = ["求", "计算", "多少", "几", "什么"]
        for keyword in objective_keywords:
            if keyword in problem_text:
                key_info["objectives"].append(f"目标关键词: {keyword}")
        
        return key_info
    
    def _plan_reasoning_path(self, problem_text: str, numbers: List[float], 
                            understanding: Dict[str, Any]) -> Dict[str, Any]:
        """规划推理路径"""
        path = {
            "strategy": "step_by_step",
            "steps": [],
            "approach": "linear"
        }
        
        problem_type = understanding.get("type", "unknown")
        operations = understanding.get("operations", [])
        
        if problem_type == "arithmetic":
            if "addition" in operations:
                path["steps"].append("execute_addition")
            if "subtraction" in operations:
                path["steps"].append("execute_subtraction")
            if "multiplication" in operations:
                path["steps"].append("execute_multiplication")
            if "division" in operations:
                path["steps"].append("execute_division")
        
        elif problem_type == "geometry":
            path["steps"].extend(["identify_shape", "apply_formula", "calculate_result"])
        
        elif problem_type == "rate_problem":
            path["steps"].extend(["identify_rate", "calculate_total", "derive_answer"])
        
        else:
            # 通用推理路径
            path["steps"].extend(["analyze_relationships", "perform_calculations", "verify_result"])
        
        return path
    
    def _execute_calculations(self, problem_text: str, numbers: List[float], 
                            reasoning_path: Dict[str, Any]) -> tuple:
        """执行计算步骤"""
        calculation_steps = []
        step_count = 3  # 从第4步开始（前面已有3步）
        
        # 简单的计算逻辑
        if len(numbers) >= 2:
            # 基于问题文本判断运算类型
            text_lower = problem_text.lower()
            
            if any(kw in text_lower for kw in ["加", "和", "总共", "一共", "+"]):
                result = sum(numbers)
                step_count += 1
                calculation_steps.append({
                    "step": step_count,
                    "action": "addition",
                    "description": f"计算总和: {' + '.join(map(str, numbers))} = {result}",
                    "calculation": f"{' + '.join(map(str, numbers))} = {result}",
                    "result": result,
                    "confidence": 0.95
                })
                final_answer = result
                
            elif any(kw in text_lower for kw in ["减", "剩", "还有", "少", "-"]):
                result = numbers[0] - sum(numbers[1:])
                step_count += 1
                calculation_steps.append({
                    "step": step_count,
                    "action": "subtraction",
                    "description": f"计算差值: {numbers[0]} - {sum(numbers[1:])} = {result}",
                    "calculation": f"{numbers[0]} - {sum(numbers[1:])} = {result}",
                    "result": result,
                    "confidence": 0.95
                })
                final_answer = result
                
            elif any(kw in text_lower for kw in ["乘", "倍", "×", "*"]):
                result = numbers[0] * numbers[1]
                step_count += 1
                calculation_steps.append({
                    "step": step_count,
                    "action": "multiplication",
                    "description": f"计算乘积: {numbers[0]} × {numbers[1]} = {result}",
                    "calculation": f"{numbers[0]} × {numbers[1]} = {result}",
                    "result": result,
                    "confidence": 0.95
                })
                final_answer = result
                
            elif any(kw in text_lower for kw in ["除", "平均", "每", "÷", "/"]):
                if numbers[1] != 0:
                    result = numbers[0] / numbers[1]
                    step_count += 1
                    calculation_steps.append({
                        "step": step_count,
                        "action": "division",
                        "description": f"计算商: {numbers[0]} ÷ {numbers[1]} = {result}",
                        "calculation": f"{numbers[0]} ÷ {numbers[1]} = {result}",
                        "result": result,
                        "confidence": 0.95
                    })
                    final_answer = result
                else:
                    raise ValueError("除数不能为零")
            else:
                # 默认尝试加法
                result = sum(numbers)
                final_answer = result
        else:
            final_answer = numbers[0] if numbers else 0
        
        return calculation_steps, final_answer
    
    def _validate_answer(self, answer: float, problem_text: str, numbers: List[float]) -> Dict[str, Any]:
        """验证答案的合理性"""
        validation = {
            "passed": True,
            "confidence": 0.8,
            "status": "valid",
            "checks": []
        }
        
        try:
            # 检查答案是否为数字
            if not isinstance(answer, (int, float)):
                validation["passed"] = False
                validation["status"] = "non_numeric"
                validation["checks"].append("答案不是数字")
                return validation
            
            # 检查答案是否合理（不能是极端值）
            if abs(answer) > 1e10:
                validation["passed"] = False
                validation["status"] = "too_large"
                validation["checks"].append("答案过大，可能不合理")
                return validation
            
            if answer < 0 and any(kw in problem_text.lower() for kw in ["个数", "数量", "长度", "面积"]):
                validation["passed"] = False
                validation["status"] = "negative_quantity"
                validation["checks"].append("数量类问题答案不应为负数")
                return validation
            
            # 检查答案是否在合理范围内
            if numbers:
                max_input = max(numbers)
                min_input = min(numbers)
                
                # 对于加法，结果应该比最大输入大
                if "加" in problem_text or "总" in problem_text:
                    if answer < max_input:
                        validation["confidence"] = 0.6
                        validation["checks"].append("加法结果可能偏小")
                
                # 对于减法，结果应该比第一个数小
                if "减" in problem_text or "剩" in problem_text:
                    if len(numbers) >= 2 and answer > numbers[0]:
                        validation["confidence"] = 0.6
                        validation["checks"].append("减法结果可能偏大")
            
            validation["checks"].append("基本合理性检查通过")
            
        except Exception as e:
            validation["passed"] = False
            validation["status"] = "validation_error"
            validation["checks"].append(f"验证过程出错: {str(e)}")
        
        return validation
    
    def _calculate_overall_confidence(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """计算整体置信度"""
        if not reasoning_steps:
            return 0.0
        
        confidences = [step.get("confidence", 0.5) for step in reasoning_steps]
        
        # 使用加权平均，给后面的步骤更高权重
        weights = [i + 1 for i in range(len(confidences))]
        weighted_sum = sum(conf * weight for conf, weight in zip(confidences, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0 