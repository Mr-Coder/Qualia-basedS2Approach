from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union
import logging

from pydantic import BaseModel


class Equation:
    """方程类
    
    用于表示数学方程，支持：
    - 变量和常数项提取
    - 方程化简和标准化
    - 方程求解
    
    Attributes:
        expression: 方程表达式
        variables: 变量集合
        constants: 常数集合
        var_entity: 变量实体映射
    """
    
    def __init__(self, expression: str, var_entity: Optional[Dict[str, str]] = None):
        """初始化方程
        
        Args:
            expression: 方程表达式
            var_entity: 变量实体映射
        """
        self.expression = expression
        self.variables = set()
        self.constants = set()
        self.var_entity = var_entity or {}
        self._parse_expression()
        
    def _parse_expression(self):
        """解析方程表达式，提取变量和常数"""
        try:
            # 分割等号两边
            if '=' in self.expression:
                left, right = self.expression.split('=')
                
                # 提取变量（假设变量名由字母开头）
                import re
                var_pattern = r'[a-zA-Z][a-zA-Z0-9_]*'
                self.variables.update(re.findall(var_pattern, left))
                self.variables.update(re.findall(var_pattern, right))
                
                # 提取常数（假设常数为数字）
                const_pattern = r'\d+(?:\.\d+)?'
                self.constants.update(map(float, re.findall(const_pattern, left)))
                self.constants.update(map(float, re.findall(const_pattern, right)))
                
        except Exception as e:
            logging.error(f"解析方程式时出错: {str(e)}")
            raise ValueError(f"无效的方程式: {self.expression}")
            
    def __str__(self) -> str:
        """返回方程的字符串表示"""
        return self.expression
        
    def __repr__(self) -> str:
        """返回方程的详细字符串表示"""
        return f"Equation(expression='{self.expression}', variables={self.variables}, constants={self.constants})"
        
    def replace(self, old: str, new: str) -> str:
        """替换方程中的字符串
        
        Args:
            old: 要替换的字符串
            new: 新的字符串
            
        Returns:
            str: 替换后的字符串
        """
        return self.expression.replace(old, new)
        
    @classmethod
    def from_relation(cls, relation: Dict) -> Optional['Equation']:
        """从关系字典创建方程式对象
        
        Args:
            relation: 关系字典
            
        Returns:
            Optional[Equation]: 方程式对象
        """
        try:
            expression = relation.get('relation')
            if not expression:
                return None
                
            var_entity = relation.get('var_entity', {})
            
            return cls(expression=expression, var_entity=var_entity)
            
        except Exception as e:
            return None


class RelationType(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"

class EquationOperator(Enum):
    EQUAL = "="
    LESS_THAN = "<"
    GREATER_THAN = ">"
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="