#!/usr/bin/env python3
"""
增强数学求解器 - 真正能解题的数学推理引擎
Enhanced Math Solver - A real mathematical reasoning engine that can solve problems
"""

import re
import logging
import sympy as sp
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import ast
import operator

logger = logging.getLogger(__name__)

class ProblemType(Enum):
    """问题类型"""
    ARITHMETIC = "arithmetic"          # 基本算术
    WORD_PROBLEM = "word_problem"      # 应用题
    EQUATION = "equation"              # 方程求解
    PERCENTAGE = "percentage"          # 百分比问题
    RATIO = "ratio"                    # 比例问题
    GEOMETRY = "geometry"              # 几何问题
    UNKNOWN = "unknown"

@dataclass
class MathEntity:
    """数学实体"""
    value: Union[int, float, str]
    unit: Optional[str] = None
    entity_type: str = "number"  # number, person, object, operation
    context: Optional[str] = None

@dataclass
class MathRelation:
    """数学关系"""
    relation_type: str  # has, total, equal, more_than, less_than
    entities: List[str]
    mathematical_expression: Optional[str] = None

@dataclass
class SolutionStep:
    """求解步骤"""
    step_number: int
    description: str
    mathematical_expression: str
    result: Union[int, float, str]
    confidence: float

class EnhancedMathSolver:
    """增强数学求解器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 数学运算映射
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow
        }
        
        # 关键词模式
        self.keyword_patterns = {
            'addition': ['一共', '总共', '合计', '总数', '总计', '加上', '和是', '共有'],
            'subtraction': ['剩下', '剩余', '还有', '减去', '少了', '差是', '相差', '卖了', '用了', '花了'],
            'multiplication': ['倍', '乘以', '每个', '共需要', '总共需要', '每包', '每箱', '每支'],
            'division': ['平均', '每份', '分成', '分给', '每人', '分配', '每组'],
            'comparison': ['比', '多', '少', '大', '小', '高', '低'],
            'percentage': ['百分比', '%', '折扣', '利率', '增长率'],
            'equal': ['等于', '是', '为', '相等']
        }
    
    def solve_problem(self, problem_text: str) -> Dict[str, Any]:
        """主求解方法"""
        try:
            self.logger.info(f"开始求解问题: {problem_text}")
            
            # Step 1: 分析问题类型
            problem_type = self._classify_problem(problem_text)
            self.logger.debug(f"问题类型: {problem_type}")
            
            # Step 2: 提取数学实体
            entities = self._extract_math_entities(problem_text)
            self.logger.debug(f"提取实体: {entities}")
            
            # Step 3: 识别数学关系
            relations = self._identify_math_relations(problem_text, entities)
            self.logger.debug(f"识别关系: {relations}")
            
            # Step 4: 构建数学表达式
            expressions = self._build_mathematical_expressions(problem_text, entities, relations)
            self.logger.debug(f"数学表达式: {expressions}")
            
            # Step 5: 求解计算
            solution_steps = self._solve_expressions(expressions, problem_text)
            self.logger.debug(f"求解步骤: {solution_steps}")
            
            # Step 6: 生成最终答案
            final_answer = self._generate_final_answer(solution_steps, problem_text)
            
            return {
                "success": True,
                "answer": final_answer["answer"],
                "confidence": final_answer["confidence"],
                "strategy_used": "enhanced_math_solver",
                "problem_type": problem_type.value,
                "entities": [{"id": f"entity_{i}", "name": str(e.value), "type": e.entity_type, "unit": e.unit} for i, e in enumerate(entities)],
                "relations": [{"type": r.relation_type, "entities": r.entities, "expression": r.mathematical_expression} for r in relations],
                "solution_steps": [
                    {
                        "step": step.step_number,
                        "description": step.description,
                        "expression": step.mathematical_expression,
                        "result": step.result,
                        "confidence": step.confidence
                    } for step in solution_steps
                ],
                "reasoning_steps": [
                    {
                        "step": step.step_number,
                        "action": "mathematical_reasoning",
                        "description": step.description,
                        "confidence": step.confidence
                    } for step in solution_steps
                ]
            }
            
        except Exception as e:
            self.logger.error(f"求解失败: {e}")
            return {
                "success": False,
                "answer": "求解失败",
                "confidence": 0.0,
                "error": str(e),
                "strategy_used": "enhanced_math_solver",
                "problem_type": "unknown"
            }
    
    def _classify_problem(self, problem_text: str) -> ProblemType:
        """分类问题类型"""
        text = problem_text.lower()
        
        # 检查是否包含方程
        if '=' in text or 'x' in text or '未知数' in text:
            return ProblemType.EQUATION
        
        # 检查百分比问题
        if '%' in text or '百分' in text or '折扣' in text:
            return ProblemType.PERCENTAGE
        
        # 检查比例问题
        if '比' in text and ('：' in text or ':' in text):
            return ProblemType.RATIO
        
        # 检查几何问题
        if any(word in text for word in ['面积', '周长', '体积', '长方形', '正方形', '圆形', '三角形']):
            return ProblemType.GEOMETRY
        
        # 检查是否是应用题
        if any(word in text for word in ['小明', '小红', '小华', '同学', '老师', '妈妈', '爸爸']):
            return ProblemType.WORD_PROBLEM
        
        # 默认为算术问题
        return ProblemType.ARITHMETIC
    
    def _extract_math_entities(self, problem_text: str) -> List[MathEntity]:
        """提取数学实体"""
        entities = []
        
        # 提取数字（包括小数和分数）
        number_pattern = r'(\d+\.?\d*|\d+\/\d+)'
        numbers = re.findall(number_pattern, problem_text)
        
        for i, num_str in enumerate(numbers):
            try:
                if '/' in num_str:
                    # 处理分数
                    parts = num_str.split('/')
                    value = float(parts[0]) / float(parts[1])
                else:
                    value = float(num_str) if '.' in num_str else int(num_str)
                
                # 尝试提取单位
                unit = self._extract_unit_for_number(problem_text, num_str)
                
                entities.append(MathEntity(
                    value=value,
                    unit=unit,
                    entity_type="number",
                    context=f"number_{i}"
                ))
            except ValueError:
                continue
        
        # 提取人物
        person_pattern = r'(小\w+|老师|同学|妈妈|爸爸|学生)'
        persons = re.findall(person_pattern, problem_text)
        for person in set(persons):
            entities.append(MathEntity(
                value=person,
                entity_type="person",
                context=f"person_{person}"
            ))
        
        # 提取物品
        object_pattern = r'(苹果|橘子|书|笔|糖果|玩具|元|公斤|米|厘米)'
        objects = re.findall(object_pattern, problem_text)
        for obj in set(objects):
            entities.append(MathEntity(
                value=obj,
                entity_type="object",
                context=f"object_{obj}"
            ))
        
        return entities
    
    def _extract_unit_for_number(self, text: str, number: str) -> Optional[str]:
        """为数字提取单位"""
        # 在数字后查找单位
        pattern = f'{re.escape(number)}([个只本支元公斤米厘米平方米立方米]{{1,3}})'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None
    
    def _identify_math_relations(self, problem_text: str, entities: List[MathEntity]) -> List[MathRelation]:
        """识别数学关系"""
        relations = []
        numbers = [e for e in entities if e.entity_type == "number"]
        
        # 识别几何面积问题 (长 × 宽)
        if any(word in problem_text for word in ['面积', '长方形']) and '长' in problem_text and '宽' in problem_text:
            if len(numbers) >= 2:
                relations.append(MathRelation(
                    relation_type="area",
                    entities=[str(n.value) for n in numbers[:2]],
                    mathematical_expression=f"{numbers[0].value} * {numbers[1].value}"
                ))
                return relations
        
        # 识别除法关系 (优先级高)
        if any(kw in problem_text for kw in self.keyword_patterns['division']):
            if len(numbers) >= 2:
                relations.append(MathRelation(
                    relation_type="quotient",
                    entities=[str(n.value) for n in numbers[:2]],
                    mathematical_expression=f"{numbers[0].value} / {numbers[1].value}"
                ))
                return relations
        
        # 识别乘法关系
        if any(kw in problem_text for kw in self.keyword_patterns['multiplication']) or ('包' in problem_text and '个' in problem_text):
            if len(numbers) >= 2:
                relations.append(MathRelation(
                    relation_type="product",
                    entities=[str(n.value) for n in numbers[:2]],
                    mathematical_expression=f"{numbers[0].value} * {numbers[1].value}"
                ))
                return relations
        
        # 识别差值关系 (减法)
        if any(kw in problem_text for kw in self.keyword_patterns['subtraction']):
            if len(numbers) >= 2:
                relations.append(MathRelation(
                    relation_type="difference",
                    entities=[str(n.value) for n in numbers[:2]],
                    mathematical_expression=f"{numbers[0].value} - {numbers[1].value}"
                ))
                return relations
        
        # 识别总数关系 (加法) - 默认情况
        if any(kw in problem_text for kw in self.keyword_patterns['addition']) or len(numbers) >= 2:
            relations.append(MathRelation(
                relation_type="total",
                entities=[str(n.value) for n in numbers],
                mathematical_expression=f"{' + '.join(str(n.value) for n in numbers)}"
            ))
        
        return relations
    
    def _build_mathematical_expressions(self, problem_text: str, entities: List[MathEntity], relations: List[MathRelation]) -> List[str]:
        """构建数学表达式"""
        expressions = []
        
        # 从关系中提取表达式
        for relation in relations:
            if relation.mathematical_expression:
                expressions.append(relation.mathematical_expression)
        
        # 如果没有明确关系，尝试推断
        if not expressions:
            numbers = [e.value for e in entities if e.entity_type == "number" and isinstance(e.value, (int, float))]
            if len(numbers) >= 2:
                # 默认推断为加法（最常见的情况）
                if any(kw in problem_text for kw in ['多少', '几个', '总共', '一共']):
                    expressions.append(f"{' + '.join(map(str, numbers))}")
        
        return expressions
    
    def _solve_expressions(self, expressions: List[str], problem_text: str) -> List[SolutionStep]:
        """求解数学表达式"""
        solution_steps = []
        
        for i, expr in enumerate(expressions):
            try:
                # 使用sympy求解
                result = sp.sympify(expr)
                result_value = float(result.evalf())
                
                # 生成描述
                description = self._generate_step_description(expr, result_value, problem_text)
                
                solution_steps.append(SolutionStep(
                    step_number=i + 1,
                    description=description,
                    mathematical_expression=expr,
                    result=result_value,
                    confidence=0.95
                ))
                
            except Exception as e:
                self.logger.error(f"表达式求解失败 {expr}: {e}")
                # 尝试简单的eval（安全版本）
                try:
                    result_value = self._safe_eval(expr)
                    description = f"计算 {expr} = {result_value}"
                    
                    solution_steps.append(SolutionStep(
                        step_number=i + 1,
                        description=description,
                        mathematical_expression=expr,
                        result=result_value,
                        confidence=0.8
                    ))
                except:
                    continue
        
        return solution_steps
    
    def _safe_eval(self, expr: str) -> float:
        """安全的表达式求值"""
        # 只允许安全的数学运算
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expr):
            raise ValueError("不安全的表达式")
        
        try:
            # 使用ast.literal_eval的改进版本
            node = ast.parse(expr, mode='eval')
            return self._eval_node(node.body)
        except:
            raise ValueError("表达式求值失败")
    
    def _eval_node(self, node):
        """递归求值AST节点"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8 兼容
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = type(node.op)
            if op == ast.Add:
                return left + right
            elif op == ast.Sub:
                return left - right
            elif op == ast.Mult:
                return left * right
            elif op == ast.Div:
                return left / right
            elif op == ast.Pow:
                return left ** right
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
        
        raise ValueError(f"不支持的节点类型: {type(node)}")
    
    def _generate_step_description(self, expr: str, result: float, problem_text: str) -> str:
        """生成步骤描述"""
        if '+' in expr:
            return f"将数字相加: {expr} = {result}"
        elif '-' in expr:
            return f"计算差值: {expr} = {result}"
        elif '*' in expr:
            return f"相乘得到: {expr} = {result}"
        elif '/' in expr:
            return f"除法计算: {expr} = {result}"
        else:
            return f"计算结果: {expr} = {result}"
    
    def _generate_final_answer(self, solution_steps: List[SolutionStep], problem_text: str) -> Dict[str, Any]:
        """生成最终答案"""
        if not solution_steps:
            return {
                "answer": "无法求解",
                "confidence": 0.0
            }
        
        # 取最后一步的结果
        final_step = solution_steps[-1]
        result = final_step.result
        
        # 根据问题文本推断单位
        unit = self._infer_answer_unit(problem_text)
        
        # 格式化答案
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        
        answer = f"{result}{unit}" if unit else str(result)
        
        # 计算总体置信度
        avg_confidence = sum(step.confidence for step in solution_steps) / len(solution_steps)
        
        return {
            "answer": answer,
            "confidence": avg_confidence
        }
    
    def _infer_answer_unit(self, problem_text: str) -> str:
        """推断答案单位"""
        # 常见单位模式
        units = ['个', '只', '本', '支', '元', '公斤', '米', '厘米', '平方米', '立方米']
        
        for unit in units:
            if unit in problem_text:
                return unit
        
        return ""

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    solver = EnhancedMathSolver()
    
    # 测试不同类型的问题
    test_problems = [
        "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
        "书店有30本书，卖了12本，还剩多少本？",
        "一个长方形的长是8米，宽是5米，面积是多少平方米？",
        "小华买了3包糖，每包有15个，一共有多少个糖？",
        "100元的商品打8折，现在要多少元？"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n=== 测试问题 {i} ===")
        print(f"问题: {problem}")
        
        result = solver.solve_problem(problem)
        
        if result["success"]:
            print(f"答案: {result['answer']}")
            print(f"置信度: {result['confidence']:.2f}")
            print(f"问题类型: {result['problem_type']}")
            print("求解步骤:")
            for step in result["solution_steps"]:
                print(f"  步骤{step['step']}: {step['description']}")
        else:
            print(f"求解失败: {result['error']}")