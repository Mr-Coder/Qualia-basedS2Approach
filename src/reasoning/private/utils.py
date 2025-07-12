"""
推理工具函数

提供推理模块通用的辅助功能和工具函数。
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union


class ReasoningUtils:
    """推理工具类"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本，标准化输入"""
        if not text:
            return ""
        
        # 标准化Unicode字符
        text = unicodedata.normalize('NFKC', text)
        
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 标准化中文数字和符号
        text = text.replace('，', ',').replace('。', '.').replace('：', ':')
        text = text.replace('（', '(').replace('）', ')')
        
        return text
    
    @staticmethod
    def extract_numbers_advanced(text: str) -> List[Dict[str, Any]]:
        """高级数字提取，包含位置和上下文信息"""
        numbers = []
        
        # 匹配模式：整数、小数、分数、百分比
        patterns = [
            (r'(\d+\.\d+)', 'decimal'),
            (r'(\d+/\d+)', 'fraction'),
            (r'(\d+)%', 'percentage'),
            (r'(\d+)', 'integer')
        ]
        
        for pattern, num_type in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                value_str = match.group(1)
                
                # 计算数值
                if num_type == 'decimal':
                    value = float(value_str)
                elif num_type == 'fraction':
                    parts = value_str.split('/')
                    value = float(parts[0]) / float(parts[1]) if len(parts) == 2 else 0
                elif num_type == 'percentage':
                    value = float(value_str)
                else:  # integer
                    value = float(value_str)
                
                # 提取上下文
                context_start = max(0, start - 10)
                context_end = min(len(text), end + 10)
                context = text[context_start:context_end]
                
                numbers.append({
                    'value': value,
                    'original': match.group(0),
                    'type': num_type,
                    'position': (start, end),
                    'context': context
                })
        
        # 按位置排序
        numbers.sort(key=lambda x: x['position'][0])
        return numbers
    
    @staticmethod
    def identify_mathematical_operations(text: str) -> List[str]:
        """识别文本中的数学运算指示词"""
        operations = []
        
        operation_keywords = {
            'addition': ['加', '和', '总共', '一共', '合计', '相加', '+'],
            'subtraction': ['减', '少', '差', '剩', '余', '还有', '-'],
            'multiplication': ['乘', '倍', '×', '*', '每'],
            'division': ['除', '分', '平均', '÷', '/'],
            'percentage': ['百分比', '%', '折扣', '打折'],
            'area': ['面积', '平方', '长方形', '正方形'],
            'time': ['小时', '分钟', '秒', '天', '时间']
        }
        
        text_lower = text.lower()
        for operation, keywords in operation_keywords.items():
            for keyword in keywords:
                if keyword in text or keyword.lower() in text_lower:
                    operations.append(operation)
                    break
        
        return list(set(operations))  # 去重
    
    @staticmethod
    def parse_mathematical_expression(text: str) -> Optional[Dict[str, Any]]:
        """解析数学表达式"""
        # 简化的表达式解析
        expression_patterns = [
            (r'(\d+)\s*\+\s*(\d+)', lambda a, b: int(a) + int(b), 'addition'),
            (r'(\d+)\s*-\s*(\d+)', lambda a, b: int(a) - int(b), 'subtraction'),
            (r'(\d+)\s*×\s*(\d+)', lambda a, b: int(a) * int(b), 'multiplication'),
            (r'(\d+)\s*\*\s*(\d+)', lambda a, b: int(a) * int(b), 'multiplication'),
            (r'(\d+)\s*÷\s*(\d+)', lambda a, b: int(a) // int(b) if int(b) != 0 else 0, 'division'),
            (r'(\d+)\s*/\s*(\d+)', lambda a, b: int(a) // int(b) if int(b) != 0 else 0, 'division'),
        ]
        
        for pattern, func, op_type in expression_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    result = func(match.group(1), match.group(2))
                    return {
                        'expression': match.group(0),
                        'operands': [int(match.group(1)), int(match.group(2))],
                        'operation': op_type,
                        'result': result,
                        'position': match.span()
                    }
                except (ValueError, ZeroDivisionError):
                    continue
        
        return None
    
    @staticmethod
    def detect_problem_complexity(text: str, numbers: List[Any]) -> str:
        """检测问题复杂度"""
        complexity_score = 0
        
        # 基于数字数量
        num_count = len(numbers) if isinstance(numbers, list) else 0
        if num_count <= 2:
            complexity_score += 1
        elif num_count <= 4:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # 基于文本长度
        if len(text) < 50:
            complexity_score += 1
        elif len(text) < 150:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # 基于运算类型
        operations = ReasoningUtils.identify_mathematical_operations(text)
        if len(operations) == 1:
            complexity_score += 1
        elif len(operations) <= 3:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # 基于关键词复杂度
        complex_keywords = ['比较', '如果', '假设', '条件', '至少', '最多', '最少']
        for keyword in complex_keywords:
            if keyword in text:
                complexity_score += 1
        
        # 映射到复杂度级别
        if complexity_score <= 4:
            return "L0"  # 简单
        elif complexity_score <= 7:
            return "L1"  # 中等
        elif complexity_score <= 10:
            return "L2"  # 复杂
        else:
            return "L3"  # 非常复杂
    
    @staticmethod
    def validate_numerical_result(result: Union[int, float], 
                                context: str = "",
                                expected_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """验证数值结果的合理性"""
        validation = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": [],
            "warnings": []
        }
        
        try:
            result_num = float(result)
            
            # 检查是否为有效数字
            if not isinstance(result, (int, float)) or math.isnan(result_num) or math.isinf(result_num):
                validation["is_valid"] = False
                validation["issues"].append("Result is not a valid number")
                return validation
            
            # 检查范围合理性
            if expected_range:
                min_val, max_val = expected_range
                if result_num < min_val or result_num > max_val:
                    validation["warnings"].append(f"Result {result_num} outside expected range [{min_val}, {max_val}]")
                    validation["confidence"] *= 0.7
            
            # 基于上下文的合理性检查
            if context:
                # 金额相关
                if any(word in context for word in ['元', '钱', '价格', '费用']):
                    if result_num < 0:
                        validation["issues"].append("Negative amount in financial context")
                        validation["confidence"] *= 0.3
                    elif result_num > 100000:
                        validation["warnings"].append("Very large amount")
                        validation["confidence"] *= 0.8
                
                # 时间相关
                if any(word in context for word in ['小时', '分钟', '天']):
                    if result_num < 0:
                        validation["issues"].append("Negative time")
                        validation["confidence"] *= 0.3
                    elif result_num > 24 and '小时' in context:
                        validation["warnings"].append("Time exceeds 24 hours")
                        validation["confidence"] *= 0.9
                
                # 数量相关
                if any(word in context for word in ['个', '件', '只', '本']):
                    if result_num < 0:
                        validation["issues"].append("Negative quantity")
                        validation["confidence"] *= 0.3
                    elif not result_num.is_integer():
                        validation["warnings"].append("Non-integer quantity")
                        validation["confidence"] *= 0.9
            
            # 一般合理性检查
            if abs(result_num) > 1e6:
                validation["warnings"].append("Result is very large")
                validation["confidence"] *= 0.8
            
            if validation["issues"]:
                validation["is_valid"] = False
            
        except (ValueError, TypeError) as e:
            validation["is_valid"] = False
            validation["issues"].append(f"Cannot validate result: {e}")
            validation["confidence"] = 0.1
        
        return validation
    
    @staticmethod
    def format_reasoning_output(result: Dict[str, Any]) -> str:
        """格式化推理输出为可读文本"""
        if not result:
            return "无推理结果"
        
        output = []
        
        # 基本信息
        final_answer = result.get("final_answer", "未知")
        strategy = result.get("strategy_used", "未知")
        confidence = result.get("confidence", 0.0)
        
        output.append(f"最终答案: {final_answer}")
        output.append(f"推理策略: {strategy}")
        output.append(f"置信度: {confidence:.2f}")
        
        # 推理步骤
        steps = result.get("reasoning_steps", [])
        if steps:
            output.append("\n推理步骤:")
            for i, step in enumerate(steps, 1):
                description = step.get("description", "")
                step_conf = step.get("confidence", 0.0)
                output.append(f"  {i}. {description} (置信度: {step_conf:.2f})")
        
        # 模板信息
        template_used = result.get("template_used")
        if template_used:
            output.append(f"\n使用模板: {template_used}")
        
        return "\n".join(output)
    
    @staticmethod
    def merge_reasoning_results(results: List[Dict[str, Any]], 
                              strategy: str = "confidence_weighted") -> Dict[str, Any]:
        """合并多个推理结果"""
        if not results:
            return {}
        
        if len(results) == 1:
            return results[0]
        
        if strategy == "confidence_weighted":
            # 基于置信度加权合并
            total_weight = sum(r.get("confidence", 0.5) for r in results)
            if total_weight == 0:
                return results[0]
            
            # 选择置信度最高的答案
            best_result = max(results, key=lambda x: x.get("confidence", 0))
            
            # 合并推理步骤
            all_steps = []
            for result in results:
                steps = result.get("reasoning_steps", [])
                all_steps.extend(steps)
            
            merged_result = best_result.copy()
            merged_result["reasoning_steps"] = all_steps
            merged_result["merged_from"] = len(results)
            
            return merged_result
        
        else:
            # 简单选择第一个结果
            return results[0]


# 导入math模块用于数值验证
import math
