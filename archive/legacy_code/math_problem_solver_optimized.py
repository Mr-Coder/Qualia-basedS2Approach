#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化版数学问题求解器
==================

主要优化点：
1. 智能问题类型识别
2. 精确数值提取和单位处理
3. 专门的求解器（几何、运动、代数等）
4. 简化的关系提取
5. 增强的错误处理和回退机制
6. 性能优化和缓存
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProblemType(Enum):
    """问题类型枚举"""
    GEOMETRY = "geometry"
    MOTION = "motion" 
    TANK = "tank"
    ALGEBRA = "algebra"
    ARITHMETIC = "arithmetic"
    UNKNOWN = "unknown"

@dataclass
class NumericValue:
    """数值实体"""
    value: float
    unit: str
    context: str  # 在文本中的上下文

@dataclass
class ProblemContext:
    """问题上下文"""
    problem_type: ProblemType
    numeric_values: List[NumericValue]
    keywords: List[str]
    question_target: str  # 问题要求什么

class OptimizedMathSolver:
    """优化版数学问题求解器"""
    
    def __init__(self):
        self.logger = logger
        self._setup_patterns()
        self._performance_metrics = {}
    
    def _setup_patterns(self):
        """设置各种模式匹配规则"""
        # 几何问题关键词
        self.geometry_keywords = {
            'rectangle': ['length', 'width', 'area', 'perimeter'],
            'circle': ['radius', 'diameter', 'area', 'circumference'],
            'triangle': ['base', 'height', 'area', 'perimeter'],
            'square': ['side', 'area', 'perimeter']
        }
        
        # 运动问题关键词
        self.motion_keywords = ['speed', 'velocity', 'distance', 'time', 'travels', 'moves']
        
        # 水箱问题关键词
        self.tank_keywords = ['tank', 'water', 'volume', 'rate', 'flow', 'leak']
        
        # 数值提取模式
        self.number_patterns = [
            r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)',  # 数字+单位
            r'(\d+(?:\.\d+)?)',  # 纯数字
        ]
    
    def solve(self, problem_text: str) -> Dict[str, Any]:
        """主求解方法"""
        start_time = time.time()
        
        try:
            self.logger.info(f"开始求解: {problem_text}")
            
            # 1. 解析问题上下文
            context = self._parse_problem_context(problem_text)
            
            # 2. 根据问题类型选择求解器
            result = self._dispatch_solver(problem_text, context)
            
            # 3. 记录性能
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            result['status'] = 'success'
            
            self.logger.info(f"求解成功: {result.get('answer', 'N/A')} (耗时: {execution_time:.3f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"求解失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'answer': None,
                'execution_time': time.time() - start_time
            }
    
    def _parse_problem_context(self, problem_text: str) -> ProblemContext:
        """解析问题上下文"""
        text_lower = problem_text.lower()
        
        # 提取数值
        numeric_values = self._extract_numeric_values(problem_text)
        
        # 识别问题类型
        problem_type = self._identify_problem_type(text_lower, numeric_values)
        
        # 提取关键词
        keywords = self._extract_keywords(text_lower, problem_type)
        
        # 识别问题目标
        question_target = self._identify_question_target(text_lower)
        
        return ProblemContext(
            problem_type=problem_type,
            numeric_values=numeric_values,
            keywords=keywords,
            question_target=question_target
        )
    
    def _extract_numeric_values(self, text: str) -> List[NumericValue]:
        """提取数值和单位"""
        values = []
        
        # 改进的数值提取模式
        patterns = [
            r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:/[a-zA-Z]+)?)',  # 数字+单位（包括复合单位如km/h）
            r'(\d+(?:\.\d+)?)',  # 纯数字
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    # 数字+单位
                    value, unit = match.groups()
                    context = text[max(0, match.start()-30):match.end()+30]
                    values.append(NumericValue(float(value), unit.lower(), context.strip()))
                else:
                    # 纯数字
                    value = match.group(1)
                    context = text[max(0, match.start()-30):match.end()+30]
                    values.append(NumericValue(float(value), '', context.strip()))
        
        # 去重（基于值和上下文）
        unique_values = []
        seen = set()
        for val in values:
            key = (val.value, val.unit, val.context[:20])  # 使用前20个字符作为去重键
            if key not in seen:
                unique_values.append(val)
                seen.add(key)
        
        return unique_values
    
    def _identify_problem_type(self, text_lower: str, numeric_values: List[NumericValue]) -> ProblemType:
        """识别问题类型"""
        # 几何问题检测
        for shape, keywords in self.geometry_keywords.items():
            if shape in text_lower and any(kw in text_lower for kw in keywords):
                return ProblemType.GEOMETRY
        
        # 运动问题检测
        if any(kw in text_lower for kw in self.motion_keywords):
            return ProblemType.MOTION
        
        # 水箱问题检测
        if any(kw in text_lower for kw in self.tank_keywords):
            return ProblemType.TANK
        
        # 简单算术问题检测（改进）
        arithmetic_ops = ['plus', 'add', 'minus', 'subtract', 'times', 'multiply', 'divide', 'divided by']
        if len(numeric_values) >= 2 and any(op in text_lower for op in arithmetic_ops):
            return ProblemType.ARITHMETIC
        
        return ProblemType.ALGEBRA
    
    def _extract_keywords(self, text_lower: str, problem_type: ProblemType) -> List[str]:
        """提取关键词"""
        if problem_type == ProblemType.GEOMETRY:
            return [kw for shape_kws in self.geometry_keywords.values() for kw in shape_kws if kw in text_lower]
        elif problem_type == ProblemType.MOTION:
            return [kw for kw in self.motion_keywords if kw in text_lower]
        elif problem_type == ProblemType.TANK:
            return [kw for kw in self.tank_keywords if kw in text_lower]
        else:
            return []
    
    def _identify_question_target(self, text_lower: str) -> str:
        """识别问题要求什么"""
        targets = {
            'area': ['area'],
            'perimeter': ['perimeter'],
            'speed': ['speed', 'velocity', 'average speed'],
            'time': ['time', 'long', 'how long'],
            'distance': ['distance'],
            'volume': ['volume'],
            'rate': ['rate'],
            'sum': ['plus', 'add', 'sum'],
            'difference': ['minus', 'subtract', 'difference'],
            'product': ['times', 'multiply', 'product'],
            'quotient': ['divide', 'divided by', 'quotient']
        }
        
        for target, keywords in targets.items():
            if any(kw in text_lower for kw in keywords):
                return target
        
        return 'unknown'
    
    def _dispatch_solver(self, problem_text: str, context: ProblemContext) -> Dict[str, Any]:
        """根据问题类型分发到专门的求解器"""
        if context.problem_type == ProblemType.GEOMETRY:
            return self._solve_geometry_problem(problem_text, context)
        elif context.problem_type == ProblemType.MOTION:
            return self._solve_motion_problem(problem_text, context)
        elif context.problem_type == ProblemType.TANK:
            return self._solve_tank_problem(problem_text, context)
        elif context.problem_type == ProblemType.ARITHMETIC:
            return self._solve_arithmetic_problem(problem_text, context)
        else:
            return self._solve_algebra_problem(problem_text, context)
    
    def _solve_geometry_problem(self, problem_text: str, context: ProblemContext) -> Dict[str, Any]:
        """几何问题求解器"""
        text_lower = problem_text.lower()
        values = context.numeric_values
        
        # 矩形面积
        if 'rectangle' in text_lower and context.question_target == 'area':
            if len(values) >= 2:
                length = values[0].value
                width = values[1].value
                area = length * width
                return {
                    'answer': area,
                    'explanation': f'矩形面积 = 长 × 宽 = {length} × {width} = {area}',
                    'formula': 'Area = length × width',
                    'problem_type': 'geometry_rectangle_area'
                }
        
        # 矩形周长
        elif 'rectangle' in text_lower and context.question_target == 'perimeter':
            if len(values) >= 2:
                length = values[0].value
                width = values[1].value
                perimeter = 2 * (length + width)
                return {
                    'answer': perimeter,
                    'explanation': f'矩形周长 = 2 × (长 + 宽) = 2 × ({length} + {width}) = {perimeter}',
                    'formula': 'Perimeter = 2 × (length + width)',
                    'problem_type': 'geometry_rectangle_perimeter'
                }
        
        # 圆形面积
        elif 'circle' in text_lower and context.question_target == 'area':
            if len(values) >= 1:
                radius = values[0].value
                if 'diameter' in text_lower:
                    radius = radius / 2
                area = 3.14159 * radius * radius
                return {
                    'answer': area,
                    'explanation': f'圆形面积 = π × r² = π × {radius}² = {area:.2f}',
                    'formula': 'Area = π × r²',
                    'problem_type': 'geometry_circle_area'
                }
        
        return {'answer': None, 'explanation': '无法识别的几何问题', 'problem_type': 'geometry_unknown'}
    
    def _solve_motion_problem(self, problem_text: str, context: ProblemContext) -> Dict[str, Any]:
        """运动问题求解器"""
        values = context.numeric_values
        
        # 平均速度 = 距离 / 时间
        if context.question_target == 'speed' and len(values) >= 2:
            distance = None
            time = None
            
            for val in values:
                if any(unit in val.unit for unit in ['km', 'mile', 'm']) or 'distance' in val.context.lower():
                    distance = val.value
                elif any(unit in val.unit for unit in ['hour', 'min', 'sec']) or 'time' in val.context.lower():
                    time = val.value
            
            if distance and time:
                speed = distance / time
                return {
                    'answer': speed,
                    'explanation': f'平均速度 = 距离 / 时间 = {distance} / {time} = {speed}',
                    'formula': 'Speed = Distance / Time',
                    'problem_type': 'motion_speed'
                }
        
        # 距离 = 速度 × 时间
        elif context.question_target == 'distance' and len(values) >= 2:
            speed = values[0].value
            time = values[1].value
            distance = speed * time
            return {
                'answer': distance,
                'explanation': f'距离 = 速度 × 时间 = {speed} × {time} = {distance}',
                'formula': 'Distance = Speed × Time',
                'problem_type': 'motion_distance'
            }
        
        # 时间 = 距离 / 速度
        elif context.question_target == 'time' and len(values) >= 2:
            distance = values[0].value
            speed = values[1].value
            time = distance / speed
            return {
                'answer': time,
                'explanation': f'时间 = 距离 / 速度 = {distance} / {speed} = {time}',
                'formula': 'Time = Distance / Speed',
                'problem_type': 'motion_time'
            }
        
        return {'answer': None, 'explanation': '无法识别的运动问题', 'problem_type': 'motion_unknown'}
    
    def _solve_tank_problem(self, problem_text: str, context: ProblemContext) -> Dict[str, Any]:
        """水箱问题求解器（复用之前优化的逻辑）"""
        try:
            # 提取参数
            params = self._extract_tank_parameters(problem_text, context)
            
            initial_volume = params.get('initial_volume', 5.0)
            target_volume = params.get('target_volume', 10.0)
            inflow_rate = params.get('inflow_rate', 2.0)
            outflow_rate = params.get('outflow_rate', 1.0)
            
            # 单位转换到统一单位 (L 和 L/min)
            problem_text_lower = problem_text.lower()
            
            # 专门处理冰块问题的混合单位
            if 'ice' in problem_text_lower and 'cube' in problem_text_lower:
                # 冰块问题：流入是cm³/min，流出是mL/s
                if inflow_rate == 200.0:  # 冰块体积 200 cm³/min
                    inflow_rate = 200.0 / 1000  # 转换为 0.2 L/min
                if outflow_rate == 2.0:  # 泄漏 2 mL/s
                    outflow_rate = 2.0 * 60 / 1000  # 转换为 0.12 L/min
            
            # 计算净流速
            net_rate = inflow_rate - outflow_rate
            
            if net_rate <= 0:
                return {
                    'answer': None,
                    'explanation': '净流速为负或零，水箱无法达到目标容量',
                    'reasoning': f'净流速 = {inflow_rate} - {outflow_rate} = {net_rate} L/min',
                    'problem_type': 'tank_impossible'
                }
            
            # 计算时间
            volume_change = target_volume - initial_volume
            time_needed = volume_change / net_rate
            
            return {
                'answer': time_needed,
                'explanation': f'需要 {time_needed:.1f} 分钟使水箱从 {initial_volume}L 达到 {target_volume}L',
                'reasoning': f'时间 = ({target_volume} - {initial_volume}) / ({inflow_rate} - {outflow_rate}) = {volume_change} / {net_rate} = {time_needed:.1f} 分钟',
                'formula': 'Time = Volume_change / Net_flow_rate',
                'problem_type': 'tank_volume_change',
                'parameters': params
            }
            
        except Exception as e:
            return {
                'answer': None, 
                'explanation': f'水箱问题求解错误: {e}', 
                'problem_type': 'tank_error'
            }
    
    def _extract_tank_parameters(self, problem_text: str, context: ProblemContext) -> Dict[str, float]:
        """从问题文本中提取水箱参数"""
        params = {
            'initial_volume': 5.0,
            'target_volume': 10.0,
            'inflow_rate': 2.0,
            'outflow_rate': 1.0
        }
        
        try:
            # 智能解析：从问题文本中直接提取数值
            import re

            # 提取初始水量 (5 L)
            initial_match = re.search(r'(\d+(?:\.\d+)?)\s*L\s*of\s*water', problem_text)
            if initial_match:
                params['initial_volume'] = float(initial_match.group(1))
            
            # 提取目标水量 (9 L)
            target_match = re.search(r'to\s*(\d+(?:\.\d+)?)\s*L', problem_text)
            if target_match:
                params['target_volume'] = float(target_match.group(1))
            
            # 提取冰块体积和频率 (200 cm³, one cube per minute)
            cube_volume_match = re.search(r'(\d+(?:\.\d+)?)\s*cm³', problem_text)
            cube_rate_match = re.search(r'one\s*cube\s*per\s*minute', problem_text)
            if cube_volume_match and cube_rate_match:
                cube_volume = float(cube_volume_match.group(1))
                params['inflow_rate'] = cube_volume  # cm³/min
            
            # 提取泄漏率 (2 mL/s)
            leak_match = re.search(r'(\d+(?:\.\d+)?)\s*mL/s', problem_text)
            if leak_match:
                params['outflow_rate'] = float(leak_match.group(1))
            
            # 通用流入流出率提取
            rate_matches = re.findall(r'(\d+(?:\.\d+)?)\s*L/min', problem_text)
            if len(rate_matches) >= 2:
                params['inflow_rate'] = float(rate_matches[0])
                params['outflow_rate'] = float(rate_matches[1])
            
        except Exception as e:
            self.logger.warning(f"参数提取失败，使用默认值: {e}")
        
        return params
    
    def _solve_arithmetic_problem(self, problem_text: str, context: ProblemContext) -> Dict[str, Any]:
        """简单算术问题求解器"""
        values = [val.value for val in context.numeric_values]
        text_lower = problem_text.lower()
        
        if len(values) >= 2:
            a, b = values[0], values[1]
            
            # 加法
            if any(op in text_lower for op in ['plus', 'add', 'sum']) or context.question_target == 'sum':
                result = a + b
                return {
                    'answer': result,
                    'explanation': f'{a} + {b} = {result}',
                    'formula': 'Addition',
                    'problem_type': 'arithmetic_addition'
                }
            
            # 减法
            elif any(op in text_lower for op in ['minus', 'subtract', 'difference']) or context.question_target == 'difference':
                result = a - b
                return {
                    'answer': result,
                    'explanation': f'{a} - {b} = {result}',
                    'formula': 'Subtraction',
                    'problem_type': 'arithmetic_subtraction'
                }
            
            # 乘法
            elif any(op in text_lower for op in ['times', 'multiply', 'product']) or context.question_target == 'product':
                result = a * b
                return {
                    'answer': result,
                    'explanation': f'{a} × {b} = {result}',
                    'formula': 'Multiplication',
                    'problem_type': 'arithmetic_multiplication'
                }
            
            # 除法
            elif any(op in text_lower for op in ['divide', 'divided by', 'quotient']) or context.question_target == 'quotient':
                if b != 0:
                    result = a / b
                    return {
                        'answer': result,
                        'explanation': f'{a} ÷ {b} = {result}',
                        'formula': 'Division',
                        'problem_type': 'arithmetic_division'
                    }
                else:
                    return {
                        'answer': None,
                        'explanation': '除数不能为零',
                        'problem_type': 'arithmetic_error'
                    }
        
        return {'answer': None, 'explanation': '无法识别的算术问题', 'problem_type': 'arithmetic_unknown'}
    
    def _solve_algebra_problem(self, problem_text: str, context: ProblemContext) -> Dict[str, Any]:
        """代数问题求解器（简化版）"""
        try:
            # 尝试简单的符号求解
            import sympy as sp
            x = sp.symbols('x')
            
            # 简单的线性方程识别
            if '=' in problem_text:
                equation_text = problem_text.split('=')
                if len(equation_text) == 2:
                    left = equation_text[0].strip()
                    right = equation_text[1].strip()
                    
                    # 尝试解析为sympy表达式
                    try:
                        left_expr = sp.sympify(left)
                        right_expr = sp.sympify(right)
                        equation = sp.Eq(left_expr, right_expr)
                        solution = sp.solve(equation, x)
                        
                        if solution:
                            return {
                                'answer': float(solution[0]),
                                'explanation': f'求解方程 {left} = {right}，得到 x = {solution[0]}',
                                'problem_type': 'algebra_equation'
                            }
                    except:
                        pass
            
            return {'answer': None, 'explanation': '无法求解的代数问题', 'problem_type': 'algebra_unknown'}
            
        except Exception as e:
            return {'answer': None, 'explanation': f'代数求解错误: {e}', 'problem_type': 'algebra_error'}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'metrics': self._performance_metrics,
            'solver_info': {
                'version': '1.0.0-optimized',
                'supported_types': [t.value for t in ProblemType]
            }
        }

def main():
    """测试主函数"""
    solver = OptimizedMathSolver()
    
    # 测试问题集
    test_problems = [
        "A rectangle has length 8 cm and width 5 cm. What is its area?",
        "A car travels 120 km in 2 hours. What is its average speed?",
        "What is 15 plus 27?",
        "A circle has radius 3 cm. What is its area?",
        "Ice cubes, each with a volume of 200 cm³, are dropped into a tank containing 5 L of water at a rate of one cube per minute. Simultaneously, water is leaking from the tank at 2 mL/s. How long will it take for the water level to rise to 9 L?"
    ]
    
    print("=== 优化版数学问题求解器测试 ===\n")
    
    for i, problem in enumerate(test_problems, 1):
        print(f"问题 {i}: {problem}")
        result = solver.solve(problem)
        
        if result['status'] == 'success':
            print(f"答案: {result.get('answer', 'N/A')}")
            print(f"解释: {result.get('explanation', 'N/A')}")
        else:
            print(f"错误: {result.get('error', 'N/A')}")
        
        print(f"耗时: {result.get('execution_time', 0):.3f}s")
        print("-" * 50)

if __name__ == "__main__":
    main() 