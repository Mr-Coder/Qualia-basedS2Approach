"""
Symbolic Math Tool
=================

Tool for symbolic mathematical computations using SymPy.
"""

import logging
from typing import Any, List

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SymbolicMathTool(BaseTool):
    """Tool for symbolic math operations using SymPy"""
    
    def __init__(self, config: dict = None):
        super().__init__("symbolic_math", config)
        self.sympy = None
        if self.is_available:
            import sympy as sp
            self.sympy = sp
    
    def _check_availability(self) -> bool:
        """Check if SymPy is available"""
        try:
            import sympy
            return True
        except ImportError:
            logger.warning("SymPy not available - symbolic math tools disabled")
            return False
    
    def execute(self, operation: str, *args, **kwargs) -> ToolResult:
        """Execute symbolic math operation"""
        if not self.is_available:
            return ToolResult(
                success=False,
                result=None,
                error_message="SymPy not available"
            )
        
        try:
            if operation == "solve_equation":
                return self._solve_equation(*args, **kwargs)
            elif operation == "simplify":
                return self._simplify(*args, **kwargs)
            elif operation == "differentiate":
                return self._differentiate(*args, **kwargs)
            elif operation == "integrate":
                return self._integrate(*args, **kwargs)
            elif operation == "factor":
                return self._factor(*args, **kwargs)
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error_message=f"Unsupported operation: {operation}"
                )
                
        except Exception as e:
            logger.error(f"SymPy operation failed: {str(e)}")
            return ToolResult(
                success=False,
                result=None,
                error_message=str(e)
            )
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations"""
        return [
            "solve_equation",
            "simplify", 
            "differentiate",
            "integrate",
            "factor"
        ]
    
    def _solve_equation(self, equation: str, variable: str = None) -> ToolResult:
        """Solve an equation symbolically"""
        if variable:
            var = self.sympy.Symbol(variable)
        else:
            var = self.sympy.Symbol('x')  # Default variable
            
        eq = self.sympy.sympify(equation)
        solution = self.sympy.solve(eq, var)
        
        return ToolResult(
            success=True,
            result=solution,
            metadata={
                'equation': equation,
                'variable': str(var),
                'solution_type': 'symbolic'
            }
        )
    
    def _simplify(self, expression: str) -> ToolResult:
        """Simplify a mathematical expression"""
        expr = self.sympy.sympify(expression)
        simplified = self.sympy.simplify(expr)
        
        return ToolResult(
            success=True,
            result=simplified,
            metadata={
                'original': expression,
                'simplified': str(simplified)
            }
        )
    
    def _differentiate(self, expression: str, variable: str = 'x') -> ToolResult:
        """Differentiate an expression"""
        expr = self.sympy.sympify(expression)
        var = self.sympy.Symbol(variable)
        derivative = self.sympy.diff(expr, var)
        
        return ToolResult(
            success=True,
            result=derivative,
            metadata={
                'expression': expression,
                'variable': variable,
                'derivative': str(derivative)
            }
        )
    
    def _integrate(self, expression: str, variable: str = 'x') -> ToolResult:
        """Integrate an expression"""
        expr = self.sympy.sympify(expression)
        var = self.sympy.Symbol(variable)
        integral = self.sympy.integrate(expr, var)
        
        return ToolResult(
            success=True,
            result=integral,
            metadata={
                'expression': expression,
                'variable': variable,
                'integral': str(integral)
            }
        )
    
    def _factor(self, expression: str) -> ToolResult:
        """Factor a mathematical expression"""
        expr = self.sympy.sympify(expression)
        factored = self.sympy.factor(expr)
        
        return ToolResult(
            success=True,
            result=factored,
            metadata={
                'original': expression,
                'factored': str(factored)
            }
        ) 