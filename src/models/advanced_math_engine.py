"""
Advanced Mathematical Engine with Symbolic Operations
Implements symbolic mathematics capabilities using SymPy for enhanced mathematical reasoning.
Part of Story 6.1: Mathematical Reasoning Enhancement
"""

import sympy as sp
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum

class MathOperationType(Enum):
    """Types of mathematical operations supported"""
    ALGEBRAIC = "algebraic"
    CALCULUS = "calculus"
    GEOMETRIC = "geometric"
    TRIGONOMETRIC = "trigonometric"
    STATISTICAL = "statistical"
    PHYSICS = "physics"

@dataclass
class MathematicalExpression:
    """Represents a mathematical expression with metadata"""
    expression: sp.Basic
    operation_type: MathOperationType
    variables: List[str]
    constraints: Optional[Dict[str, Any]] = None
    domain: Optional[str] = None

@dataclass
class MathResult:
    """Result of mathematical computation"""
    solution: Any
    steps: List[Dict[str, Any]]
    validation_status: bool
    operation_type: MathOperationType
    confidence: float = 1.0
    warnings: List[str] = None

class AdvancedMathEngine:
    """
    Advanced Mathematical Engine for symbolic and numerical operations.
    Provides enhanced mathematical reasoning capabilities across multiple domains.
    """
    
    def __init__(self):
        """Initialize the Advanced Math Engine"""
        self.symbol_registry = {}  # Store defined symbols
        self.expression_cache = {}  # Cache parsed expressions
        self.operation_history = []  # Track operations for debugging
        
    def parse_expression(self, expr_str: str, variables: Optional[List[str]] = None) -> MathematicalExpression:
        """
        Parse a string expression into a symbolic mathematical expression
        
        Args:
            expr_str: String representation of mathematical expression
            variables: Optional list of variable names
            
        Returns:
            MathematicalExpression object
        """
        # Check cache first
        cache_key = f"{expr_str}:{','.join(variables or [])}"
        if cache_key in self.expression_cache:
            return self.expression_cache[cache_key]
        
        # Auto-detect variables if not provided
        if variables is None:
            variables = self._detect_variables(expr_str)
        
        # Create symbols
        symbols = {}
        for var in variables:
            if var not in self.symbol_registry:
                self.symbol_registry[var] = sp.Symbol(var, real=True)
            symbols[var] = self.symbol_registry[var]
        
        # Parse expression
        try:
            expression = sp.sympify(expr_str, locals=symbols)
            operation_type = self._detect_operation_type(expression, expr_str)
            
            math_expr = MathematicalExpression(
                expression=expression,
                operation_type=operation_type,
                variables=variables
            )
            
            # Cache the result
            self.expression_cache[cache_key] = math_expr
            
            return math_expr
            
        except Exception as e:
            raise ValueError(f"Failed to parse expression '{expr_str}': {str(e)}")
    
    def solve_algebraic(self, equation_str: str, target_var: str, 
                       constraints: Optional[Dict[str, Any]] = None) -> MathResult:
        """
        Solve algebraic equations symbolically
        
        Args:
            equation_str: Equation string (e.g., "x^2 + 2*x - 3 = 0")
            target_var: Variable to solve for
            constraints: Optional constraints on the solution
            
        Returns:
            MathResult with solution and steps
        """
        steps = []
        warnings = []
        
        # Parse equation
        if "=" in equation_str:
            lhs, rhs = equation_str.split("=")
            lhs_expr = self.parse_expression(lhs.strip())
            rhs_expr = self.parse_expression(rhs.strip())
            equation = sp.Eq(lhs_expr.expression, rhs_expr.expression)
            steps.append({
                "action": "parse_equation",
                "equation": str(equation),
                "description": f"Parsed equation: {equation}"
            })
        else:
            # Assume equation = 0
            expr = self.parse_expression(equation_str)
            equation = sp.Eq(expr.expression, 0)
            steps.append({
                "action": "assume_zero",
                "equation": str(equation),
                "description": f"Assumed equation equals zero: {equation}"
            })
        
        # Get target symbol
        if target_var not in self.symbol_registry:
            self.symbol_registry[target_var] = sp.Symbol(target_var, real=True)
        target_symbol = self.symbol_registry[target_var]
        
        # Solve equation
        try:
            solutions = sp.solve(equation, target_symbol)
            steps.append({
                "action": "solve",
                "solutions": [str(sol) for sol in solutions],
                "description": f"Found {len(solutions)} solution(s)"
            })
            
            # Apply constraints if provided
            if constraints:
                filtered_solutions = self._apply_constraints(solutions, target_var, constraints)
                if len(filtered_solutions) < len(solutions):
                    steps.append({
                        "action": "apply_constraints",
                        "original_count": len(solutions),
                        "filtered_count": len(filtered_solutions),
                        "description": "Applied constraints to filter solutions"
                    })
                    warnings.append(f"Filtered out {len(solutions) - len(filtered_solutions)} solutions due to constraints")
                solutions = filtered_solutions
            
            # Validate solutions
            validation_status = self._validate_algebraic_solutions(equation, target_symbol, solutions)
            
            # Record operation
            self.operation_history.append({
                "operation": "solve_algebraic",
                "equation": equation_str,
                "target": target_var,
                "result": solutions
            })
            
            return MathResult(
                solution=solutions,
                steps=steps,
                validation_status=validation_status,
                operation_type=MathOperationType.ALGEBRAIC,
                confidence=0.95 if validation_status else 0.7,
                warnings=warnings if warnings else None
            )
            
        except Exception as e:
            steps.append({
                "action": "error",
                "error": str(e),
                "description": f"Failed to solve equation: {str(e)}"
            })
            return MathResult(
                solution=None,
                steps=steps,
                validation_status=False,
                operation_type=MathOperationType.ALGEBRAIC,
                confidence=0.0,
                warnings=[f"Solving failed: {str(e)}"]
            )
    
    def compute_derivative(self, expr_str: str, var: str, order: int = 1) -> MathResult:
        """
        Compute derivative of an expression
        
        Args:
            expr_str: Expression to differentiate
            var: Variable to differentiate with respect to
            order: Order of derivative (default 1)
            
        Returns:
            MathResult with derivative
        """
        steps = []
        
        # Parse expression
        expr = self.parse_expression(expr_str)
        steps.append({
            "action": "parse",
            "expression": str(expr.expression),
            "description": f"Parsed expression: {expr.expression}"
        })
        
        # Get variable symbol
        if var not in self.symbol_registry:
            self.symbol_registry[var] = sp.Symbol(var, real=True)
        var_symbol = self.symbol_registry[var]
        
        # Compute derivative
        try:
            derivative = sp.diff(expr.expression, var_symbol, order)
            steps.append({
                "action": "differentiate",
                "order": order,
                "result": str(derivative),
                "description": f"Computed {order}-order derivative with respect to {var}"
            })
            
            # Simplify result
            simplified = sp.simplify(derivative)
            if simplified != derivative:
                steps.append({
                    "action": "simplify",
                    "original": str(derivative),
                    "simplified": str(simplified),
                    "description": "Simplified derivative expression"
                })
                derivative = simplified
            
            # Record operation
            self.operation_history.append({
                "operation": "derivative",
                "expression": expr_str,
                "variable": var,
                "order": order,
                "result": str(derivative)
            })
            
            return MathResult(
                solution=derivative,
                steps=steps,
                validation_status=True,
                operation_type=MathOperationType.CALCULUS,
                confidence=1.0
            )
            
        except Exception as e:
            steps.append({
                "action": "error",
                "error": str(e),
                "description": f"Failed to compute derivative: {str(e)}"
            })
            return MathResult(
                solution=None,
                steps=steps,
                validation_status=False,
                operation_type=MathOperationType.CALCULUS,
                confidence=0.0,
                warnings=[f"Differentiation failed: {str(e)}"]
            )
    
    def compute_integral(self, expr_str: str, var: str, 
                        bounds: Optional[Tuple[float, float]] = None) -> MathResult:
        """
        Compute integral of an expression
        
        Args:
            expr_str: Expression to integrate
            var: Variable to integrate with respect to
            bounds: Optional tuple of (lower, upper) bounds for definite integral
            
        Returns:
            MathResult with integral
        """
        steps = []
        
        # Parse expression
        expr = self.parse_expression(expr_str)
        steps.append({
            "action": "parse",
            "expression": str(expr.expression),
            "description": f"Parsed expression: {expr.expression}"
        })
        
        # Get variable symbol
        if var not in self.symbol_registry:
            self.symbol_registry[var] = sp.Symbol(var, real=True)
        var_symbol = self.symbol_registry[var]
        
        # Compute integral
        try:
            if bounds:
                # Definite integral
                integral = sp.integrate(expr.expression, (var_symbol, bounds[0], bounds[1]))
                steps.append({
                    "action": "integrate_definite",
                    "bounds": bounds,
                    "result": str(integral),
                    "description": f"Computed definite integral from {bounds[0]} to {bounds[1]}"
                })
            else:
                # Indefinite integral
                integral = sp.integrate(expr.expression, var_symbol)
                steps.append({
                    "action": "integrate_indefinite",
                    "result": str(integral),
                    "description": f"Computed indefinite integral with respect to {var}"
                })
                
                # Add constant of integration for indefinite integrals
                C = sp.Symbol('C')
                integral = integral + C
                steps.append({
                    "action": "add_constant",
                    "description": "Added constant of integration C"
                })
            
            # Simplify result if possible
            try:
                simplified = sp.simplify(integral)
                if simplified != integral:
                    steps.append({
                        "action": "simplify",
                        "original": str(integral),
                        "simplified": str(simplified),
                        "description": "Simplified integral expression"
                    })
                    integral = simplified
            except:
                pass  # Some integrals can't be simplified
            
            # Record operation
            self.operation_history.append({
                "operation": "integral",
                "expression": expr_str,
                "variable": var,
                "bounds": bounds,
                "result": str(integral)
            })
            
            return MathResult(
                solution=integral,
                steps=steps,
                validation_status=True,
                operation_type=MathOperationType.CALCULUS,
                confidence=0.95 if bounds else 1.0
            )
            
        except Exception as e:
            steps.append({
                "action": "error",
                "error": str(e),
                "description": f"Failed to compute integral: {str(e)}"
            })
            return MathResult(
                solution=None,
                steps=steps,
                validation_status=False,
                operation_type=MathOperationType.CALCULUS,
                confidence=0.0,
                warnings=[f"Integration failed: {str(e)}"]
            )
    
    def compute_limit(self, expr_str: str, var: str, point: Union[float, str], 
                     direction: Optional[str] = None) -> MathResult:
        """
        Compute limit of an expression
        
        Args:
            expr_str: Expression to find limit of
            var: Variable approaching the point
            point: Point to approach (can be number or 'oo' for infinity)
            direction: Optional '+' or '-' for one-sided limits
            
        Returns:
            MathResult with limit
        """
        steps = []
        
        # Parse expression
        expr = self.parse_expression(expr_str)
        steps.append({
            "action": "parse",
            "expression": str(expr.expression),
            "description": f"Parsed expression: {expr.expression}"
        })
        
        # Get variable symbol
        if var not in self.symbol_registry:
            self.symbol_registry[var] = sp.Symbol(var, real=True)
        var_symbol = self.symbol_registry[var]
        
        # Convert point to SymPy format
        if isinstance(point, str) and point in ['oo', 'inf', 'infinity']:
            point_value = sp.oo
        elif isinstance(point, str) and point in ['-oo', '-inf', '-infinity']:
            point_value = -sp.oo
        else:
            point_value = sp.sympify(point)
        
        # Compute limit
        try:
            if direction == '+':
                limit = sp.limit(expr.expression, var_symbol, point_value, '+')
                steps.append({
                    "action": "limit_right",
                    "point": str(point_value),
                    "result": str(limit),
                    "description": f"Computed right-hand limit as {var} → {point_value}+"
                })
            elif direction == '-':
                limit = sp.limit(expr.expression, var_symbol, point_value, '-')
                steps.append({
                    "action": "limit_left",
                    "point": str(point_value),
                    "result": str(limit),
                    "description": f"Computed left-hand limit as {var} → {point_value}-"
                })
            else:
                limit = sp.limit(expr.expression, var_symbol, point_value)
                steps.append({
                    "action": "limit",
                    "point": str(point_value),
                    "result": str(limit),
                    "description": f"Computed limit as {var} → {point_value}"
                })
            
            # Check if limit exists
            if limit == sp.zoo:  # Complex infinity
                steps.append({
                    "action": "undefined",
                    "description": "Limit is undefined (complex infinity)"
                })
                warnings = ["Limit does not exist (undefined)"]
            else:
                warnings = None
            
            # Record operation
            self.operation_history.append({
                "operation": "limit",
                "expression": expr_str,
                "variable": var,
                "point": str(point_value),
                "direction": direction,
                "result": str(limit)
            })
            
            return MathResult(
                solution=limit,
                steps=steps,
                validation_status=(limit != sp.zoo),
                operation_type=MathOperationType.CALCULUS,
                confidence=0.95 if limit != sp.zoo else 0.5,
                warnings=warnings
            )
            
        except Exception as e:
            steps.append({
                "action": "error",
                "error": str(e),
                "description": f"Failed to compute limit: {str(e)}"
            })
            return MathResult(
                solution=None,
                steps=steps,
                validation_status=False,
                operation_type=MathOperationType.CALCULUS,
                confidence=0.0,
                warnings=[f"Limit computation failed: {str(e)}"]
            )
    
    def _detect_variables(self, expr_str: str) -> List[str]:
        """Auto-detect variables in an expression"""
        # Parse as a generic expression
        expr = sp.sympify(expr_str)
        # Extract all free symbols
        return sorted([str(symbol) for symbol in expr.free_symbols])
    
    def _detect_operation_type(self, expression: sp.Basic, expr_str: str) -> MathOperationType:
        """Detect the type of mathematical operation"""
        # Check for trigonometric functions
        trig_funcs = [sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan]
        if any(expression.has(func) for func in trig_funcs):
            return MathOperationType.TRIGONOMETRIC
        
        # Check for calculus operations (derivatives, integrals in expression)
        if expression.has(sp.Derivative) or expression.has(sp.Integral):
            return MathOperationType.CALCULUS
        
        # Check for statistical functions (would need more comprehensive check)
        # For now, default to algebraic
        return MathOperationType.ALGEBRAIC
    
    def _apply_constraints(self, solutions: List, var: str, 
                          constraints: Dict[str, Any]) -> List:
        """Apply constraints to filter solutions"""
        filtered = []
        
        for sol in solutions:
            valid = True
            
            # Check domain constraints
            if 'domain' in constraints:
                domain = constraints['domain']
                if domain == 'real' and not sol.is_real:
                    valid = False
                elif domain == 'positive' and not (sol.is_real and sol > 0):
                    valid = False
                elif domain == 'integer' and not sol.is_integer:
                    valid = False
            
            # Check range constraints
            if 'min' in constraints and sol.is_real:
                if sol < constraints['min']:
                    valid = False
            
            if 'max' in constraints and sol.is_real:
                if sol > constraints['max']:
                    valid = False
            
            if valid:
                filtered.append(sol)
        
        return filtered
    
    def _validate_algebraic_solutions(self, equation: sp.Eq, var: sp.Symbol, 
                                    solutions: List) -> bool:
        """Validate algebraic solutions by substitution"""
        if not solutions:
            return False
        
        for sol in solutions:
            try:
                # Substitute solution back into equation
                lhs = equation.lhs.subs(var, sol)
                rhs = equation.rhs.subs(var, sol)
                
                # Check if equation is satisfied (within numerical tolerance)
                diff = sp.simplify(lhs - rhs)
                if diff != 0 and not sp.Abs(diff) < 1e-10:
                    return False
            except:
                # If substitution fails, validation fails
                return False
        
        return True
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of all operations performed"""
        return self.operation_history.copy()
    
    def clear_cache(self):
        """Clear expression cache"""
        self.expression_cache.clear()
    
    def reset(self):
        """Reset engine to initial state"""
        self.symbol_registry.clear()
        self.expression_cache.clear()
        self.operation_history.clear()