"""
推理处理器

实现核心的数学推理逻辑，从原有的ReasoningEngine重构而来。
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加src_new到路径
src_new_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_new_path))

from core.exceptions import ProcessingError
from core.interfaces import BaseProcessor


class ReasoningProcessor(BaseProcessor):
    """推理处理器 - 核心推理逻辑"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._logger = logging.getLogger(__name__)
        self.config = config or {}
        
    def process(self, input_data: Any) -> Any:
        """处理推理请求"""
        try:
            if not isinstance(input_data, dict):
                raise ProcessingError("Input must be a dictionary", module_name="reasoning")
            
            return self._execute_reasoning(input_data)
            
        except Exception as e:
            self._logger.error(f"Reasoning processing failed: {e}")
            raise ProcessingError(f"Reasoning processing failed: {e}", module_name="reasoning")
    
    def _execute_reasoning(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理过程"""
        text = problem_data.get("cleaned_text") or problem_data.get("problem") or ""
        template_info = problem_data.get("template_info")
        knowledge_context = problem_data.get("knowledge_context")
        
        reasoning_steps = []
        intermediate_vars = {}
        
        # Step 1: 数字提取
        numbers = self._extract_numbers(text)
        reasoning_steps.append({
            "step": 1,
            "action": "number_extraction",
            "description": f"提取数字: {numbers}",
            "numbers": numbers,
            "confidence": 0.9
        })
        
        # Step 2: 表达式解析
        expression_result = self._parse_expression(text)
        if expression_result:
            reasoning_steps.append({
                "step": 2,
                "action": "expression_parsing",
                "description": f"解析表达式: {expression_result['expression']}",
                "expression": expression_result['expression'],
                "result": expression_result['result'],
                "confidence": 0.9
            })
            final_answer = str(expression_result['result'])
            strategy = "DIR"  # Direct Implicit Reasoning
        else:
            # Step 3: 模板化推理
            template_result = self._template_based_reasoning(text, numbers, template_info)
            if template_result["answer"] != "unknown":
                reasoning_steps.extend(template_result["steps"])
                final_answer = template_result["answer"]
                strategy = "TBR"  # Template-Based Reasoning
            else:
                # Step 4: 通用推理
                general_result = self._general_reasoning(text, numbers)
                reasoning_steps.extend(general_result["steps"])
                final_answer = general_result["answer"]
                strategy = "COT"  # Chain of Thought
        
        # Step 5: 知识增强（如果有元知识上下文）
        if knowledge_context:
            knowledge_step = self._apply_knowledge_context(
                text, final_answer, knowledge_context
            )
            reasoning_steps.append(knowledge_step)
        
        return {
            "final_answer": final_answer,
            "strategy_used": strategy,
            "confidence": self._calculate_overall_confidence(reasoning_steps),
            "reasoning_steps": reasoning_steps,
            "intermediate_variables": intermediate_vars
        }
    
    def _extract_numbers(self, text: str) -> List[float]:
        """提取文本中的数字"""
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return [float(n) for n in numbers]
    
    def _parse_expression(self, text: str) -> Optional[Dict]:
        """解析简单数学表达式"""
        # 简单表达式模式
        patterns = [
            (r'(\d+)\s*[+加]\s*(\d+)', lambda m: int(m.group(1)) + int(m.group(2))),
            (r'(\d+)\s*[-减]\s*(\d+)', lambda m: int(m.group(1)) - int(m.group(2))),
            (r'(\d+)\s*[×*乘]\s*(\d+)', lambda m: int(m.group(1)) * int(m.group(2))),
            (r'(\d+)\s*[÷/除]\s*(\d+)', lambda m: int(m.group(1)) // int(m.group(2))),
            (r'(\d+)\s*\*\s*(\d+)\s*/\s*(\d+)', lambda m: int(m.group(1)) * int(m.group(2)) // int(m.group(3))),
        ]
        
        for pattern, func in patterns:
            if match := re.search(pattern, text):
                try:
                    result = func(match)
                    return {
                        "expression": match.group(0),
                        "result": result
                    }
                except (ValueError, ZeroDivisionError):
                    continue
        return None
    
    def _template_based_reasoning(self, text: str, numbers: List[float], 
                                 template_info: Optional[Dict]) -> Dict[str, Any]:
        """基于模板的推理"""
        steps = []
        answer = "unknown"
        
        if not template_info:
            return {"answer": answer, "steps": steps}
        
        template_type = template_info.get('type')
        
        if template_type == "discount" and len(numbers) >= 2:
            # 打折问题: 原价 * (折扣/10) = 现价
            original_price = numbers[0]
            discount = numbers[1]
            if discount > 10:  # 如果是百分比形式
                discount = discount / 10
            final_price = original_price * discount
            
            steps.append({
                "step": len(steps) + 1,
                "action": "discount_calculation",
                "description": f"计算折扣价格: {original_price} * {discount}",
                "calculation": f"{original_price} * {discount} = {final_price}",
                "result": final_price,
                "confidence": 0.9
            })
            answer = str(int(final_price))
            
        elif template_type == "area" and len(numbers) >= 2:
            # 面积问题: 长 * 宽 = 面积
            length = numbers[0]
            width = numbers[1]
            area = length * width
            steps.append({
                "step": len(steps) + 1,
                "action": "area_calculation",
                "description": f"计算面积: {length} * {width}",
                "calculation": f"{length} * {width} = {area}",
                "result": area,
                "confidence": 0.95
            })
            answer = str(int(area))
            
        elif template_type == "percentage" and len(numbers) >= 2:
            # 百分比问题: 总数 * (百分比/100) = 部分
            total = numbers[0]
            percentage = numbers[1]
            part = total * (percentage / 100)
            steps.append({
                "step": len(steps) + 1,
                "action": "percentage_calculation",
                "description": f"计算百分比: {total} * {percentage}%",
                "calculation": f"{total} * {percentage/100} = {part}",
                "result": part,
                "confidence": 0.9
            })
            answer = str(int(part))
            
        elif template_type == "time" and len(numbers) >= 2:
            # 时间问题处理
            answer, step = self._handle_time_problem(text, numbers)
            if step:
                steps.append(step)
        
        return {"answer": answer, "steps": steps}
    
    def _general_reasoning(self, text: str, numbers: List[float]) -> Dict[str, Any]:
        """通用推理方法"""
        steps = []
        answer = "unknown"
        
        if len(numbers) >= 2:
            # 基于关键词推断操作
            if any(keyword in text for keyword in ["买", "花", "用"]) and \
               any(keyword in text for keyword in ["剩", "还有", "余"]):
                # 减法操作
                initial = numbers[0]
                spent = sum(numbers[1:])
                result = initial - spent
                steps.append({
                    "step": 1,
                    "action": "subtraction_reasoning",
                    "description": f"减法推理: {initial} - {spent}",
                    "calculation": f"{initial} - {spent} = {result}",
                    "result": result,
                    "confidence": 0.8
                })
                answer = str(int(result))
                
            elif any(keyword in text for keyword in ["总", "一共", "加起来"]):
                # 加法操作
                total = sum(numbers)
                steps.append({
                    "step": 1,
                    "action": "addition_reasoning",
                    "description": f"加法推理: {' + '.join(map(str, numbers))}",
                    "calculation": f"{' + '.join(map(str, numbers))} = {total}",
                    "result": total,
                    "confidence": 0.8
                })
                answer = str(int(total))
                
            elif any(keyword in text for keyword in ["每", "平均"]):
                # 除法/平均操作
                if len(numbers) >= 2:
                    total = numbers[0]
                    count = numbers[1]
                    average = total / count if count != 0 else 0
                    steps.append({
                        "step": 1,
                        "action": "division_reasoning",
                        "description": f"除法推理: {total} ÷ {count}",
                        "calculation": f"{total} ÷ {count} = {average}",
                        "result": average,
                        "confidence": 0.8
                    })
                    answer = str(int(average))
        
        # 如果没有找到合适的推理方式
        if answer == "unknown" and numbers:
            steps.append({
                "step": 1,
                "action": "fallback_reasoning",
                "description": "无法确定具体操作，返回第一个数字",
                "result": numbers[0],
                "confidence": 0.3
            })
            answer = str(int(numbers[0]))
        
        return {"answer": answer, "steps": steps}
    
    def _handle_time_problem(self, text: str, numbers: List[float]) -> Tuple[str, Optional[Dict]]:
        """处理时间相关问题"""
        if "小时" in text and "分钟" in text:
            # 时间转换
            if len(numbers) >= 2:
                hours = numbers[0]
                minutes = numbers[1]
                total_minutes = hours * 60 + minutes
                return str(int(total_minutes)), {
                    "step": 1,
                    "action": "time_conversion",
                    "description": f"时间转换: {hours}小时{minutes}分钟",
                    "calculation": f"{hours} * 60 + {minutes} = {total_minutes}分钟",
                    "result": total_minutes,
                    "confidence": 0.9
                }
        return "unknown", None
    
    def _apply_knowledge_context(self, text: str, answer: str, 
                                knowledge_context: Dict) -> Dict[str, Any]:
        """应用知识上下文增强推理"""
        concepts = knowledge_context.get("concepts", [])
        strategies = knowledge_context.get("strategies", [])
        
        return {
            "step": 99,
            "action": "knowledge_enhancement",
            "description": f"应用知识增强: 概念{concepts}, 策略{strategies}",
            "concepts": concepts,
            "strategies": strategies,
            "confidence": 0.7
        }
    
    def _calculate_overall_confidence(self, reasoning_steps: List[Dict]) -> float:
        """计算整体置信度"""
        if not reasoning_steps:
            return 0.5
        
        total_confidence = sum(step.get('confidence', 0.5) for step in reasoning_steps)
        return min(total_confidence / len(reasoning_steps), 1.0) 