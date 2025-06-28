"""
Numerical Compute Tool
=====================

Tool for numerical computations using NumPy.
"""

import logging
from typing import Any, List

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class NumericalComputeTool(BaseTool):
    """Tool for numerical computation operations using NumPy"""
    
    def __init__(self, config: dict = None):
        super().__init__("numerical_compute", config)
        self.numpy = None
        if self.is_available:
            import numpy as np
            self.numpy = np
    
    def _check_availability(self) -> bool:
        """Check if NumPy is available"""
        try:
            import numpy
            return True
        except ImportError:
            logger.warning("NumPy not available - numerical compute tools disabled")
            return False
    
    def execute(self, operation: str, *args, **kwargs) -> ToolResult:
        """Execute numerical computation operation"""
        if not self.is_available:
            return ToolResult(
                success=False,
                result=None,
                error_message="NumPy not available"
            )
        
        try:
            if operation == "calculate":
                return self._calculate(*args, **kwargs)
            elif operation == "solve_linear":
                return self._solve_linear(*args, **kwargs)
            elif operation == "statistics":
                return self._statistics(*args, **kwargs)
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error_message=f"Unsupported operation: {operation}"
                )
                
        except Exception as e:
            logger.error(f"Numerical computation failed: {str(e)}")
            return ToolResult(
                success=False,
                result=None,
                error_message=str(e)
            )
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations"""
        return [
            "calculate",
            "solve_linear",
            "statistics"
        ]
    
    def _calculate(self, expression: str) -> ToolResult:
        """Evaluate a numerical expression"""
        # Simple evaluation - in production, use safer evaluation
        try:
            result = eval(expression, {"__builtins__": {}}, {
                "abs": abs, "min": min, "max": max, "sum": sum,
                "pow": pow, "round": round
            })
            
            return ToolResult(
                success=True,
                result=result,
                metadata={'expression': expression}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error_message=f"Calculation error: {str(e)}"
            )
    
    def _solve_linear(self, coefficients: List[List[float]], constants: List[float]) -> ToolResult:
        """Solve linear system Ax = b"""
        A = self.numpy.array(coefficients)
        b = self.numpy.array(constants)
        
        try:
            solution = self.numpy.linalg.solve(A, b)
            return ToolResult(
                success=True,
                result=solution.tolist(),
                metadata={
                    'coefficients': coefficients,
                    'constants': constants,
                    'method': 'numpy.linalg.solve'
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error_message=f"Linear solve error: {str(e)}"
            )
    
    def _statistics(self, data: List[float]) -> ToolResult:
        """Calculate basic statistics"""
        arr = self.numpy.array(data)
        
        stats = {
            'mean': float(self.numpy.mean(arr)),
            'median': float(self.numpy.median(arr)),
            'std': float(self.numpy.std(arr)),
            'min': float(self.numpy.min(arr)),
            'max': float(self.numpy.max(arr))
        }
        
        return ToolResult(
            success=True,
            result=stats,
            metadata={'data_size': len(data)}
        ) 