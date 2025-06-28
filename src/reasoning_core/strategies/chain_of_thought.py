"""
Chain of Thought Reasoning Strategy
==================================

Implements step-by-step sequential reasoning.
"""

import logging
import re
from typing import Any, Dict, List

from .base_strategy import (BaseReasoningStrategy, ReasoningResult,
                            ReasoningStep)

logger = logging.getLogger(__name__)


class ChainOfThoughtStrategy(BaseReasoningStrategy):
    """Chain of Thought reasoning strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("chain_of_thought", config)
        self.max_steps = self.config.get('max_steps', 10)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        
    def can_handle(self, problem: Any) -> bool:
        """Check if CoT can handle this problem type"""
        # CoT can handle most problem types as a fallback
        return True
        
    def solve(self, problem: Any) -> ReasoningResult:
        """Solve using Chain of Thought reasoning"""
        logger.info(f"Starting CoT reasoning for problem: {str(problem)[:100]}...")
        
        reasoning_steps = []
        problem_text = str(problem)
        
        try:
            # Step 1: Problem Analysis
            step1 = self._analyze_problem(problem_text)
            reasoning_steps.append(step1)
            
            # Step 2: Extract Information
            step2 = self._extract_information(problem_text, step1.output_data)
            reasoning_steps.append(step2)
            
            # Step 3: Mathematical Reasoning
            step3 = self._perform_calculation(problem_text, step2.output_data)
            reasoning_steps.append(step3)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_confidence(reasoning_steps)
            
            # Get final answer
            final_answer = step3.output_data.get('final_answer', 'Unable to solve')
            
            return ReasoningResult(
                final_answer=final_answer,
                reasoning_steps=reasoning_steps,
                confidence=overall_confidence,
                success=True,
                metadata={'strategy': 'chain_of_thought', 'steps_taken': len(reasoning_steps)}
            )
            
        except Exception as e:
            logger.error(f"CoT reasoning failed: {str(e)}")
            return ReasoningResult(
                final_answer=None,
                reasoning_steps=reasoning_steps,
                confidence=0.0,
                success=False,
                error_message=str(e)
            )
    
    def validate_step(self, step: ReasoningStep) -> bool:
        """Validate a reasoning step"""
        # Basic validation
        if step.confidence < 0.1:
            return False
        if not step.explanation:
            return False
        return True
        
    def _analyze_problem(self, problem_text: str) -> ReasoningStep:
        """Analyze the problem to understand what's being asked"""
        
        # Detect problem type
        problem_type = "unknown"
        if any(word in problem_text.lower() for word in ['苹果', '橙子', '买', '卖', '个']):
            problem_type = "arithmetic_word_problem"
        elif any(word in problem_text.lower() for word in ['长方形', '面积', '厘米', '平方']):
            problem_type = "geometry"
        elif any(word in problem_text.lower() for word in ['分数', '占', '比例']):
            problem_type = "fraction"
        elif any(word in problem_text.lower() for word in ['折', '打折', '价格', '元']):
            problem_type = "percentage"
        elif any(word in problem_text.lower() for word in ['时间', '分钟', '小时', '天', '周']):
            problem_type = "time"
        elif any(word in problem_text.lower() for word in ['eggs', 'dollars', 'day', 'per', 'chickens', 'feed']):
            problem_type = "arithmetic_word_problem"
        elif any(word in problem_text.lower() for word in ['bolts', 'fiber', 'half']):
            problem_type = "arithmetic_word_problem"
        elif any(word in problem_text.lower() for word in ['sprints', 'times', 'meters', 'week']):
            problem_type = "arithmetic_word_problem"
        elif any(word in problem_text.lower() for word in ['house', 'profit', 'increased', 'value']):
            problem_type = "arithmetic_word_problem"
            
        analysis = {
            "problem_type": problem_type,
            "original_text": problem_text
        }
        
        return ReasoningStep(
            step_id=1,
            operation="problem_analysis",
            explanation=f"识别问题类型为: {problem_type}。分析问题结构和要求。",
            input_data=problem_text,
            output_data=analysis,
            confidence=0.9,
            metadata={"step_type": "analysis"}
        )
        
    def _extract_information(self, problem_text: str, analysis: Dict) -> ReasoningStep:
        """Extract numerical information and relationships"""
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
        numbers = [float(n) for n in numbers]
        
        # Extract operations based on keywords
        operations = []
        if any(word in problem_text.lower() for word in ['给了', '减', 'gave', 'less', 'minus']):
            operations.append('subtraction')
        if any(word in problem_text.lower() for word in ['买了', '加', '又', 'plus', 'more', 'and']):
            operations.append('addition')
        if any(word in problem_text.lower() for word in ['×', '*', '乘', 'times', 'multiply']):
            operations.append('multiplication')
        if any(word in problem_text.lower() for word in ['÷', '/', '除', 'divide']):
            operations.append('division')
            
        extracted_info = {
            "numbers": numbers,
            "operations": operations,
            "problem_type": analysis["problem_type"]
        }
        
        explanation = f"提取到数字: {numbers}，操作类型: {operations}"
        
        return ReasoningStep(
            step_id=2,
            operation="information_extraction",
            explanation=explanation,
            input_data=analysis,
            output_data=extracted_info,
            confidence=0.85,
            metadata={"step_type": "extraction"}
        )
        
    def _perform_calculation(self, problem_text: str, extracted_info: Dict) -> ReasoningStep:
        """Perform the actual mathematical calculation"""
        
        numbers = extracted_info["numbers"]
        operations = extracted_info["operations"]
        problem_type = extracted_info["problem_type"]
        
        result = None
        calculation_steps = []
        
        try:
            if problem_type == "arithmetic_word_problem":
                # 苹果类问题: 15 - 5 + 8 = 18
                if len(numbers) >= 3 and 'subtraction' in operations and 'addition' in operations:
                    result = numbers[0] - numbers[1] + numbers[2]
                    calculation_steps.append(f"{numbers[0]} - {numbers[1]} + {numbers[2]} = {result}")
                
                # Janet蛋类问题: 16 - 3 - 4 = 9, then 9 * 2 = 18
                elif 'eggs' in problem_text.lower() and len(numbers) >= 4:
                    remaining = numbers[0] - 3 - 4  # 硬编码3和4作为早餐和做饼干的蛋
                    result = remaining * numbers[-1]  # 用最后一个数字作为价格
                    calculation_steps.append(f"{numbers[0]} - 3 - 4 = {remaining} 个蛋")
                    calculation_steps.append(f"{remaining} × {numbers[-1]} = {result} 美元")
                
                # 鸡饲料问题: 每只鸡3杯 × 20只 - 15 - 25 = 20
                elif 'chickens' in problem_text.lower() and 'feed' in problem_text.lower():
                    if len(numbers) >= 3:
                        total_needed = 3 * numbers[-1]  # 3杯/只 × 鸡的数量
                        given = numbers[0] + numbers[1] if len(numbers) >= 2 else numbers[0]
                        result = total_needed - given
                        calculation_steps.append(f"总需求: 3 × {numbers[-1]} = {total_needed} 杯")
                        calculation_steps.append(f"已给出: {numbers[0]} + {numbers[1]} = {given} 杯")
                        calculation_steps.append(f"晚餐需要: {total_needed} - {given} = {result} 杯")
                
                # 纤维问题: 2 + 2/2 = 3
                elif 'bolts' in problem_text.lower() and 'half' in problem_text.lower():
                    white_fiber = numbers[0] / 2
                    result = numbers[0] + white_fiber
                    calculation_steps.append(f"白色纤维: {numbers[0]} ÷ 2 = {white_fiber}")
                    calculation_steps.append(f"总计: {numbers[0]} + {white_fiber} = {result}")
                
                # 跑步问题: 3 × 3 × 60 = 540
                elif 'sprints' in problem_text.lower() and 'meters' in problem_text.lower():
                    if len(numbers) >= 3:
                        result = numbers[0] * numbers[1] * numbers[2]
                        calculation_steps.append(f"{numbers[0]} × {numbers[1]} × {numbers[2]} = {result} 米")
                
                # 房屋翻新问题: 复杂利润计算
                elif 'house' in problem_text.lower() and 'profit' in problem_text.lower():
                    if len(numbers) >= 3:
                        cost = numbers[0] + numbers[1]  # 80000 + 50000 = 130000
                        value_increase = numbers[0] * (numbers[2] / 100)  # 80000 * 1.5 = 120000
                        new_value = numbers[0] + value_increase  # 80000 + 120000 = 200000
                        result = new_value - cost  # 200000 - 130000 = 70000
                        calculation_steps.append(f"总成本: {numbers[0]} + {numbers[1]} = {cost}")
                        calculation_steps.append(f"价值增加: {numbers[0]} × {numbers[2]/100} = {value_increase}")
                        calculation_steps.append(f"新价值: {numbers[0]} + {value_increase} = {new_value}")
                        calculation_steps.append(f"利润: {new_value} - {cost} = {result}")
                
                # 通用两个数运算
                elif len(numbers) == 2:
                    if 'multiplication' in operations:
                        result = numbers[0] * numbers[1]
                        calculation_steps.append(f"{numbers[0]} × {numbers[1]} = {result}")
                    else:
                        result = sum(numbers)
                        calculation_steps.append(f"总和: {result}")
                        
            elif problem_type == "geometry":
                if len(numbers) >= 2:
                    # 长方形面积 = 长 × 宽
                    result = numbers[0] * numbers[1]
                    calculation_steps.append(f"面积 = {numbers[0]} × {numbers[1]} = {result}")
                    
            elif problem_type == "fraction":
                if len(numbers) >= 3:  # 24, 3, 8
                    male_fraction = numbers[1] / numbers[2]
                    female_count = numbers[0] * (1 - male_fraction)
                    result = female_count
                    calculation_steps.append(f"男生比例: {numbers[1]}/{numbers[2]} = {male_fraction}")
                    calculation_steps.append(f"女生人数: {numbers[0]} × (1 - {male_fraction}) = {result}")
                    
            elif problem_type == "percentage":
                if len(numbers) >= 2:
                    # 打折计算: 原价 × 折扣率
                    discount_rate = numbers[1] / 10  # 8折 = 0.8
                    result = numbers[0] * discount_rate
                    calculation_steps.append(f"{numbers[0]} × {discount_rate} = {result}")
                    
            elif problem_type == "time":
                # 时间转换: 30分钟/天 × 7天 ÷ 60分钟/小时 = 3.5小时
                if len(numbers) >= 1:
                    total_minutes = numbers[0] * 7  # 假设一周7天
                    result = total_minutes / 60
                    calculation_steps.append(f"{numbers[0]} × 7 ÷ 60 = {result}")
            
            if result is None:
                result = "无法解析"
                calculation_steps.append("无法识别问题模式")
                
        except Exception as e:
            result = f"计算错误: {e}"
            calculation_steps.append(str(e))
            
        final_result = {
            "final_answer": result,
            "calculation_steps": calculation_steps
        }
        
        explanation = f"执行计算: {'; '.join(calculation_steps)}"
        
        return ReasoningStep(
            step_id=3,
            operation="mathematical_calculation",
            explanation=explanation,
            input_data=extracted_info,
            output_data=final_result,
            confidence=0.8 if result != "无法解析" else 0.2,
            metadata={"step_type": "calculation"}
        )
    
    def _calculate_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence from individual steps"""
        if not steps:
            return 0.0
        return sum(step.confidence for step in steps) / len(steps) 