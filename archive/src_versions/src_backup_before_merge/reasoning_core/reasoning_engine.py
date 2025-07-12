import re
from typing import Dict, List, Optional, Tuple

import sympy

from .meta_knowledge import MetaKnowledge, MetaKnowledgeReasoning


class ReasoningEngine:
    """
    Enhanced ReasoningEngine for math problems.
    Supports multi-step reasoning, expression parsing, and template recognition.
    """
    def __init__(self, config=None):
        self.config = config or {}
        
        # 初始化元知识
        self.meta_knowledge = MetaKnowledge()
        self.meta_reasoning = MetaKnowledgeReasoning(self.meta_knowledge)
        
        # 常见题型模板
        self.templates = {
            "discount": {
                "patterns": [r"打(\d+)折", r"(\d+)%折扣", r"(\d+)折"],
                "template": "原价 * (折扣/10) = 现价"
            },
            "area": {
                "patterns": [r"面积", r"平方", r"长.*宽"],
                "template": "长 * 宽 = 面积"
            },
            "percentage": {
                "patterns": [r"(\d+)%", r"百分之(\d+)"],
                "template": "总数 * (百分比/100) = 部分"
            },
            "average": {
                "patterns": [r"平均", r"每", r"per"],
                "template": "总和 / 数量 = 平均值"
            },
            "time": {
                "patterns": [r"小时", r"分钟", r"天", r"周"],
                "template": "时间单位转换"
            }
        }

    def solve(self, sample: Dict) -> Dict:
        """
        Enhanced solve method with multi-step reasoning.
        Returns detailed reasoning steps and intermediate variables.
        """
        text = sample.get("cleaned_text") or sample.get("problem") or ""
        reasoning_steps = []
        intermediate_vars = {}
        
        # Step 1: 识别题型模板
        template_info = self._identify_template(text)
        if template_info:
            reasoning_steps.append({
                "step": 1,
                "action": "template_identification",
                "description": f"识别题型: {template_info['type']}",
                "template": template_info['template'],
                "confidence": 0.8
            })
        
        # Step 2: 提取数字和变量
        numbers = self._extract_numbers(text)
        reasoning_steps.append({
            "step": 2,
            "action": "number_extraction",
            "description": f"提取数字: {numbers}",
            "numbers": numbers,
            "confidence": 0.9
        })
        
        # Step 3: 尝试直接表达式解析
        expression_result = self._parse_expression(text)
        if expression_result:
            reasoning_steps.append({
                "step": 3,
                "action": "expression_parsing",
                "description": f"解析表达式: {expression_result['expression']}",
                "expression": expression_result['expression'],
                "result": expression_result['result'],
                "confidence": 0.9
            })
            final_answer = str(expression_result['result'])
            strategy = "DIR"
        else:
            # Step 4: 多步推理
            multi_step_result = self._multi_step_reasoning(text, numbers, template_info)
            reasoning_steps.extend(multi_step_result['steps'])
            final_answer = multi_step_result['answer']
            strategy = "COT"
        
        # Step 5: 验证结果
        validation = self._validate_answer(final_answer, text)
        reasoning_steps.append({
            "step": len(reasoning_steps) + 1,
            "action": "answer_validation",
            "description": f"验证答案: {final_answer}",
            "is_valid": validation['valid'],
            "confidence": validation['confidence']
        })
        
        # Step 6: 元知识增强
        meta_enhancement = self.meta_reasoning.enhance_reasoning(text, reasoning_steps)
        
        # Step 7: 解决方案验证
        solution_validation = self.meta_reasoning.validate_solution(
            text, final_answer, 
            [step.get('calculation', '') for step in reasoning_steps if 'calculation' in step]
        )
        
        return {
            "final_answer": final_answer,
            "strategy_used": strategy,
            "confidence": self._calculate_overall_confidence(reasoning_steps),
            "reasoning_steps": reasoning_steps,
            "intermediate_variables": intermediate_vars,
            "template_used": template_info['type'] if template_info else None,
            "meta_knowledge_enhancement": meta_enhancement,
            "solution_validation": solution_validation
        }

    def _identify_template(self, text: str) -> Optional[Dict]:
        """识别题型模板"""
        for template_type, template_data in self.templates.items():
            for pattern in template_data['patterns']:
                if re.search(pattern, text):
                    return {
                        "type": template_type,
                        "template": template_data['template'],
                        "pattern_matched": pattern
                    }
        return None

    def _extract_numbers(self, text: str) -> List[float]:
        """提取文本中的数字"""
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return [float(n) for n in numbers]

    def _parse_expression(self, text: str) -> Optional[Dict]:
        """解析数学表达式"""
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

    def _multi_step_reasoning(self, text: str, numbers: List[float], template_info: Optional[Dict]) -> Dict:
        """多步推理 - 增强版，集成元知识"""
        steps = []
        answer = "unknown"
        
        # 使用元知识分析问题
        concepts = self.meta_knowledge.identify_concepts_in_text(text)
        strategies = self.meta_knowledge.suggest_strategies(text)
        
        # 添加元知识分析步骤
        if concepts:
            steps.append({
                "step": 1,
                "action": "concept_analysis",
                "description": f"识别数学概念: {', '.join(concepts)}",
                "concepts": concepts,
                "confidence": 0.9
            })
        
        if strategies:
            steps.append({
                "step": 2,
                "action": "strategy_suggestion",
                "description": f"推荐解题策略: {', '.join(strategies)}",
                "strategies": strategies,
                "confidence": 0.8
            })
        
        if template_info:
            template_type = template_info['type']
            
            if template_type == "discount":
                # 打折问题: 原价 * (折扣/10) = 现价
                if len(numbers) >= 2:
                    original_price = numbers[0]
                    discount = numbers[1]
                    if discount > 10:  # 如果是百分比形式
                        discount = discount / 10
                    final_price = original_price * discount
                    
                    # 使用元知识验证计算
                    validation = self.meta_knowledge.validate_mathematical_expression(f"{original_price} * {discount}")
                    
                    steps.append({
                        "step": len(steps) + 1,
                        "action": "discount_calculation",
                        "description": f"计算折扣价格: {original_price} * {discount}",
                        "calculation": f"{original_price} * {discount} = {final_price}",
                        "result": final_price,
                        "validation": validation
                    })
                    answer = str(int(final_price))
            
            elif template_type == "area":
                # 面积问题: 长 * 宽 = 面积
                if len(numbers) >= 2:
                    length = numbers[0]
                    width = numbers[1]
                    area = length * width
                    steps.append({
                        "step": 1,
                        "action": "area_calculation",
                        "description": f"计算面积: {length} * {width}",
                        "calculation": f"{length} * {width} = {area}",
                        "result": area
                    })
                    answer = str(int(area))
            
            elif template_type == "percentage":
                # 百分比问题: 总数 * (百分比/100) = 部分
                if len(numbers) >= 2:
                    total = numbers[0]
                    percentage = numbers[1]
                    part = total * (percentage / 100)
                    steps.append({
                        "step": 1,
                        "action": "percentage_calculation",
                        "description": f"计算百分比: {total} * {percentage}%",
                        "calculation": f"{total} * {percentage/100} = {part}",
                        "result": part
                    })
                    answer = str(int(part))
        
        # 如果没有模板或模板解析失败，尝试简单算术
        if answer == "unknown" and len(numbers) >= 2:
            # 尝试简单的加减法
            if "给" in text and "买" in text:
                # 初始 - 给出 + 新增
                initial = numbers[0]
                given = numbers[1]
                bought = numbers[2] if len(numbers) > 2 else 0
                result = initial - given + bought
                steps.append({
                    "step": 1,
                    "action": "simple_arithmetic",
                    "description": f"计算: {initial} - {given} + {bought}",
                    "calculation": f"{initial} - {given} + {bought} = {result}",
                    "result": result
                })
                answer = str(int(result))
        
        return {
            "answer": answer,
            "steps": steps
        }

    def _validate_answer(self, answer: str, text: str) -> Dict:
        """验证答案的合理性"""
        try:
            answer_num = float(answer)
            # 简单验证：答案应该是正数且合理范围
            is_valid = answer_num >= 0 and answer_num < 10000
            confidence = 0.8 if is_valid else 0.3
        except ValueError:
            is_valid = False
            confidence = 0.1
        
        return {
            "valid": is_valid,
            "confidence": confidence
        }

    def _calculate_overall_confidence(self, reasoning_steps: List[Dict]) -> float:
        """计算整体置信度"""
        if not reasoning_steps:
            return 0.5
        
        total_confidence = sum(step.get('confidence', 0.5) for step in reasoning_steps)
        return total_confidence / len(reasoning_steps) 