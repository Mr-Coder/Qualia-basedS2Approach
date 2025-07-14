"""
COT-DIR 安全组件

提供安全的数学计算、文件操作等功能，替代不安全的操作。
"""

import ast
import json
import os
import math
import logging
from pathlib import Path
from typing import Any, Dict, Union
from cryptography.fernet import Fernet


class SecurityError(Exception):
    """安全异常"""
    pass


class SecureMathEvaluator:
    """安全的数学表达式计算器"""
    
    ALLOWED_OPERATORS = {
        ast.Add: '+',
        ast.Sub: '-', 
        ast.Mult: '*',
        ast.Div: '/',
        ast.Pow: '**',
        ast.USub: '-',
        ast.UAdd: '+'
    }
    
    ALLOWED_FUNCTIONS = {
        'abs', 'round', 'min', 'max', 'sum',
        'sqrt', 'pow', 'exp', 'log', 'sin', 'cos', 'tan'
    }
    
    def __init__(self):
        self.logger = logging.getLogger("security.math_evaluator")
        
    def safe_eval(self, expression: str, allowed_names: Dict[str, Any] = None) -> Union[float, int]:
        """安全地计算数学表达式"""
        try:
            # 简单的数字直接返回
            try:
                return float(expression.strip())
            except ValueError:
                pass
            
            # 解析表达式为AST
            tree = ast.parse(expression, mode='eval')
            
            # 验证AST安全性
            self._validate_ast_security(tree)
            
            # 准备安全的执行环境
            safe_dict = self._create_safe_environment(allowed_names or {})
            
            # 执行表达式
            result = eval(compile(tree, '<string>', 'eval'), safe_dict)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"安全计算失败，使用默认值: {expression} - {e}")
            return 0.0  # 返回安全的默认值
    
    def _validate_ast_security(self, tree: ast.AST):
        """验证AST节点的安全性"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.ALLOWED_FUNCTIONS:
                        raise SecurityError(f"不允许的函数调用: {func_name}")
                else:
                    raise SecurityError("不允许的复杂函数调用")
            
            if isinstance(node, ast.Attribute):
                raise SecurityError("不允许访问对象属性")
            
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SecurityError("不允许导入模块")
    
    def _create_safe_environment(self, allowed_names: Dict[str, Any]) -> Dict[str, Any]:
        """创建安全的执行环境"""
        safe_env = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'sqrt': math.sqrt,
            'pow': pow,
            'exp': math.exp,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e
        }
        
        # 添加允许的变量
        for name, value in allowed_names.items():
            if isinstance(value, (int, float, complex)):
                safe_env[name] = value
        
        return safe_env


# 全局安全计算器实例
_secure_evaluator = SecureMathEvaluator()
