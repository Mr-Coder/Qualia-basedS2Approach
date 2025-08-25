"""
Mathematical Correctness Validator
Validates mathematical solutions through step-by-step verification, algebraic manipulation checks,
geometric proof validation, and numerical accuracy verification
Part of Story 6.1: Mathematical Reasoning Enhancement
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import re

@dataclass
class ValidationResult:
    """Result of mathematical validation"""
    is_valid: bool
    confidence: float
    validation_type: str
    details: Dict[str, Any]
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None

class ValidationType(Enum):
    """Types of mathematical validation"""
    ALGEBRAIC_STEPS = "algebraic_steps"
    NUMERIC_ACCURACY = "numeric_accuracy"
    GEOMETRIC_PROOF = "geometric_proof"
    EQUATION_BALANCE = "equation_balance"
    DOMAIN_CONSTRAINTS = "domain_constraints"
    UNIT_CONSISTENCY = "unit_consistency"
    LOGICAL_CONSISTENCY = "logical_consistency"

class MathematicalCorrectnessValidator:
    """
    Validates mathematical solutions and reasoning steps.
    Ensures correctness through multiple validation strategies.
    """
    
    # Tolerance for numerical comparisons
    NUMERIC_TOLERANCE = 1e-10
    
    # Common mathematical constraints
    DOMAIN_RULES = {
        'sqrt': lambda x: x >= 0,
        'log': lambda x: x > 0,
        'asin': lambda x: -1 <= x <= 1,
        'acos': lambda x: -1 <= x <= 1,
        'division': lambda x: x != 0
    }
    
    def __init__(self):
        """Initialize the Mathematical Correctness Validator"""
        self.validation_history = []
        self.error_patterns = {}
        
    def validate_algebraic_steps(self, steps: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate a sequence of algebraic manipulation steps
        
        Args:
            steps: List of algebraic steps, each containing 'from', 'to', and 'operation'
            
        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        details = {
            'total_steps': len(steps),
            'valid_steps': 0,
            'invalid_steps': [],
            'step_details': []
        }
        
        for i, step in enumerate(steps):
            step_valid = True
            step_detail = {
                'step_number': i + 1,
                'operation': step.get('operation', 'unknown'),
                'valid': True,
                'issues': []
            }
            
            try:
                from_expr = sp.sympify(step.get('from', ''))
                to_expr = sp.sympify(step.get('to', ''))
                operation = step.get('operation', '')
                
                # Validate based on operation type
                if operation == 'simplify':
                    # Check if expressions are equivalent
                    if not self._expressions_equivalent(from_expr, to_expr):
                        step_valid = False
                        step_detail['issues'].append("Simplification is incorrect")
                        
                elif operation == 'expand':
                    # Check if expansion is correct
                    expanded = sp.expand(from_expr)
                    if expanded != to_expr:
                        step_valid = False
                        step_detail['issues'].append("Expansion is incorrect")
                        
                elif operation == 'factor':
                    # Check if factorization is correct
                    expanded_to = sp.expand(to_expr)
                    if not self._expressions_equivalent(from_expr, expanded_to):
                        step_valid = False
                        step_detail['issues'].append("Factorization is incorrect")
                        
                elif operation == 'substitute':
                    # Validate substitution
                    substitutions = step.get('substitutions', {})
                    result = from_expr
                    for var, val in substitutions.items():
                        result = result.subs(sp.Symbol(var), val)
                    if result != to_expr:
                        step_valid = False
                        step_detail['issues'].append("Substitution is incorrect")
                        
                elif operation in ['add', 'subtract', 'multiply', 'divide']:
                    # Validate arithmetic operations
                    operand = step.get('operand')
                    if operand is not None:
                        operand_expr = sp.sympify(operand)
                        if operation == 'add':
                            expected = from_expr + operand_expr
                        elif operation == 'subtract':
                            expected = from_expr - operand_expr
                        elif operation == 'multiply':
                            expected = from_expr * operand_expr
                        elif operation == 'divide':
                            if operand_expr == 0:
                                step_valid = False
                                step_detail['issues'].append("Division by zero")
                            else:
                                expected = from_expr / operand_expr
                        
                        if step_valid and not self._expressions_equivalent(expected, to_expr):
                            step_valid = False
                            step_detail['issues'].append(f"{operation} operation is incorrect")
                
                if step_valid:
                    details['valid_steps'] += 1
                else:
                    details['invalid_steps'].append(i + 1)
                    errors.append(f"Step {i + 1}: {', '.join(step_detail['issues'])}")
                
            except Exception as e:
                step_valid = False
                step_detail['issues'].append(f"Error processing step: {str(e)}")
                errors.append(f"Step {i + 1}: Failed to process - {str(e)}")
            
            step_detail['valid'] = step_valid
            details['step_details'].append(step_detail)
        
        # Calculate overall validity and confidence
        is_valid = len(details['invalid_steps']) == 0
        confidence = details['valid_steps'] / details['total_steps'] if details['total_steps'] > 0 else 0
        
        # Add suggestions for common errors
        if not is_valid:
            suggestions = self._generate_algebraic_suggestions(details['step_details'])
        else:
            suggestions = None
        
        # Record validation
        self._record_validation('algebraic_steps', is_valid, confidence)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            validation_type=ValidationType.ALGEBRAIC_STEPS.value,
            details=details,
            errors=errors if errors else None,
            warnings=warnings if warnings else None,
            suggestions=suggestions
        )
    
    def validate_equation_solution(self, equation: str, variable: str, 
                                 solution: Union[float, List[float]]) -> ValidationResult:
        """
        Validate that a solution satisfies an equation
        
        Args:
            equation: String representation of equation (e.g., "x^2 - 4 = 0")
            variable: Variable being solved for
            solution: Proposed solution(s)
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        details = {
            'equation': equation,
            'variable': variable,
            'solutions_tested': [],
            'valid_solutions': [],
            'invalid_solutions': []
        }
        
        try:
            # Parse equation
            if '=' in equation:
                lhs, rhs = equation.split('=')
                lhs_expr = sp.sympify(lhs.strip())
                rhs_expr = sp.sympify(rhs.strip())
            else:
                lhs_expr = sp.sympify(equation)
                rhs_expr = 0
            
            var_symbol = sp.Symbol(variable)
            
            # Convert single solution to list
            solutions = solution if isinstance(solution, list) else [solution]
            
            # Test each solution
            for sol in solutions:
                details['solutions_tested'].append(sol)
                
                # Substitute solution into equation
                lhs_val = lhs_expr.subs(var_symbol, sol)
                rhs_val = rhs_expr.subs(var_symbol, sol)
                
                # Evaluate if possible
                try:
                    lhs_numeric = float(lhs_val)
                    rhs_numeric = float(rhs_val)
                    
                    # Check if equation is satisfied
                    if abs(lhs_numeric - rhs_numeric) < self.NUMERIC_TOLERANCE:
                        details['valid_solutions'].append(sol)
                    else:
                        details['invalid_solutions'].append(sol)
                        errors.append(f"Solution {sol} does not satisfy equation: "
                                    f"{lhs_numeric} ≠ {rhs_numeric}")
                except:
                    # Handle symbolic results
                    diff = sp.simplify(lhs_val - rhs_val)
                    if diff == 0:
                        details['valid_solutions'].append(sol)
                    else:
                        details['invalid_solutions'].append(sol)
                        errors.append(f"Solution {sol} does not satisfy equation symbolically")
            
            # Validate completeness (for polynomial equations)
            if isinstance(lhs_expr - rhs_expr, sp.Poly):
                poly = sp.Poly(lhs_expr - rhs_expr, var_symbol)
                expected_solutions = poly.degree()
                if len(details['valid_solutions']) < expected_solutions:
                    warnings.append(f"Expected {expected_solutions} solutions for degree {poly.degree()} "
                                  f"polynomial, found {len(details['valid_solutions'])}")
            
        except Exception as e:
            errors.append(f"Failed to validate equation: {str(e)}")
            details['error'] = str(e)
        
        is_valid = len(details['invalid_solutions']) == 0 and len(errors) == 0
        confidence = len(details['valid_solutions']) / len(solutions) if solutions else 0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            validation_type=ValidationType.EQUATION_BALANCE.value,
            details=details,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )
    
    def validate_numeric_accuracy(self, calculated: float, expected: float, 
                                tolerance: Optional[float] = None) -> ValidationResult:
        """
        Validate numerical accuracy of a calculation
        
        Args:
            calculated: Calculated value
            expected: Expected value
            tolerance: Acceptable tolerance (default: NUMERIC_TOLERANCE)
            
        Returns:
            ValidationResult
        """
        if tolerance is None:
            tolerance = self.NUMERIC_TOLERANCE
        
        absolute_error = abs(calculated - expected)
        relative_error = absolute_error / abs(expected) if expected != 0 else absolute_error
        
        is_valid = absolute_error <= tolerance
        
        details = {
            'calculated': calculated,
            'expected': expected,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'tolerance': tolerance,
            'percentage_error': relative_error * 100
        }
        
        errors = []
        warnings = []
        
        if not is_valid:
            errors.append(f"Numerical error {absolute_error:.2e} exceeds tolerance {tolerance:.2e}")
        
        if relative_error > 0.01:  # More than 1% error
            warnings.append(f"Relative error is {relative_error*100:.2f}%")
        
        # Confidence based on how close we are to tolerance
        confidence = max(0, 1 - (absolute_error / tolerance)) if tolerance > 0 else 0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            validation_type=ValidationType.NUMERIC_ACCURACY.value,
            details=details,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )
    
    def validate_geometric_proof(self, proof_steps: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate a geometric proof
        
        Args:
            proof_steps: List of proof steps with statements and justifications
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        details = {
            'total_steps': len(proof_steps),
            'valid_steps': 0,
            'logical_flow': True,
            'step_validation': []
        }
        
        # Common geometric theorems and postulates
        valid_justifications = {
            'given', 'definition', 'postulate', 'theorem',
            'reflexive', 'symmetric', 'transitive',
            'substitution', 'addition', 'subtraction',
            'multiplication', 'division', 'pythagorean',
            'parallel', 'perpendicular', 'congruent', 'similar'
        }
        
        previous_statements = set()
        
        for i, step in enumerate(proof_steps):
            step_detail = {
                'step': i + 1,
                'valid': True,
                'issues': []
            }
            
            statement = step.get('statement', '')
            justification = step.get('justification', '').lower()
            
            # Check justification validity
            if not any(valid in justification for valid in valid_justifications):
                step_detail['valid'] = False
                step_detail['issues'].append(f"Invalid justification: {justification}")
                errors.append(f"Step {i + 1}: Invalid justification")
            
            # Check logical dependencies
            dependencies = step.get('depends_on', [])
            for dep in dependencies:
                if dep > i + 1:
                    step_detail['valid'] = False
                    step_detail['issues'].append(f"Cannot depend on future step {dep}")
                    details['logical_flow'] = False
            
            # Validate specific geometric relationships
            if 'angle' in statement.lower():
                # Check angle sum properties
                if 'triangle' in statement.lower() and '180' not in statement:
                    warnings.append(f"Step {i + 1}: Triangle angle sum should equal 180°")
            
            if 'parallel' in statement.lower() and 'angle' in statement.lower():
                # Check parallel line angle relationships
                if not any(term in justification for term in ['alternate', 'corresponding', 'co-interior']):
                    warnings.append(f"Step {i + 1}: Parallel line angle relationship not specified")
            
            if step_detail['valid']:
                details['valid_steps'] += 1
                previous_statements.add(statement)
            
            details['step_validation'].append(step_detail)
        
        is_valid = details['valid_steps'] == details['total_steps'] and details['logical_flow']
        confidence = details['valid_steps'] / details['total_steps'] if details['total_steps'] > 0 else 0
        
        # Generate suggestions for improving the proof
        suggestions = []
        if not details['logical_flow']:
            suggestions.append("Ensure all steps reference only previous statements")
        if confidence < 1.0:
            suggestions.append("Review and correct invalid justifications")
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            validation_type=ValidationType.GEOMETRIC_PROOF.value,
            details=details,
            errors=errors if errors else None,
            warnings=warnings if warnings else None,
            suggestions=suggestions if suggestions else None
        )
    
    def validate_domain_constraints(self, expression: str, 
                                  variables: Dict[str, float]) -> ValidationResult:
        """
        Validate that variable values satisfy mathematical domain constraints
        
        Args:
            expression: Mathematical expression
            variables: Dictionary of variable values
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        details = {
            'expression': expression,
            'variables': variables,
            'constraints_checked': [],
            'violations': []
        }
        
        try:
            expr = sp.sympify(expression)
            
            # Check for square roots
            for sqrt_expr in expr.find(sp.sqrt):
                arg = sqrt_expr.args[0]
                value = arg.subs([(sp.Symbol(k), v) for k, v in variables.items()])
                
                constraint = {
                    'type': 'sqrt',
                    'expression': str(arg),
                    'value': float(value) if value.is_number else None
                }
                
                if value.is_number and value < 0:
                    errors.append(f"Square root of negative number: sqrt({arg}) = sqrt({value})")
                    constraint['violated'] = True
                    details['violations'].append(constraint)
                else:
                    constraint['violated'] = False
                
                details['constraints_checked'].append(constraint)
            
            # Check for logarithms
            for log_expr in expr.find(sp.log):
                arg = log_expr.args[0]
                value = arg.subs([(sp.Symbol(k), v) for k, v in variables.items()])
                
                constraint = {
                    'type': 'log',
                    'expression': str(arg),
                    'value': float(value) if value.is_number else None
                }
                
                if value.is_number and value <= 0:
                    errors.append(f"Logarithm of non-positive number: log({arg}) = log({value})")
                    constraint['violated'] = True
                    details['violations'].append(constraint)
                else:
                    constraint['violated'] = False
                
                details['constraints_checked'].append(constraint)
            
            # Check for division by zero
            denominators = []
            
            def find_denominators(expr):
                if expr.is_Pow and expr.exp < 0:
                    denominators.append(expr.base)
                elif expr.is_Mul:
                    for arg in expr.args:
                        find_denominators(arg)
                elif hasattr(expr, 'as_numer_denom'):
                    _, denom = expr.as_numer_denom()
                    if denom != 1:
                        denominators.append(denom)
            
            find_denominators(expr)
            
            for denom in denominators:
                value = denom.subs([(sp.Symbol(k), v) for k, v in variables.items()])
                
                constraint = {
                    'type': 'division',
                    'expression': str(denom),
                    'value': float(value) if value.is_number else None
                }
                
                if value.is_number and value == 0:
                    errors.append(f"Division by zero: denominator {denom} = {value}")
                    constraint['violated'] = True
                    details['violations'].append(constraint)
                else:
                    constraint['violated'] = False
                
                details['constraints_checked'].append(constraint)
            
            # Check inverse trig functions
            for func, check in [(sp.asin, lambda x: -1 <= x <= 1), 
                               (sp.acos, lambda x: -1 <= x <= 1)]:
                for trig_expr in expr.find(func):
                    arg = trig_expr.args[0]
                    value = arg.subs([(sp.Symbol(k), v) for k, v in variables.items()])
                    
                    if value.is_number:
                        constraint = {
                            'type': func.__name__,
                            'expression': str(arg),
                            'value': float(value)
                        }
                        
                        if not check(float(value)):
                            errors.append(f"{func.__name__} domain violation: "
                                        f"{func.__name__}({value}) is undefined")
                            constraint['violated'] = True
                            details['violations'].append(constraint)
                        else:
                            constraint['violated'] = False
                        
                        details['constraints_checked'].append(constraint)
            
        except Exception as e:
            errors.append(f"Failed to check domain constraints: {str(e)}")
            details['error'] = str(e)
        
        is_valid = len(details['violations']) == 0
        confidence = 1.0 - (len(details['violations']) / len(details['constraints_checked']) 
                          if details['constraints_checked'] else 1.0)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            validation_type=ValidationType.DOMAIN_CONSTRAINTS.value,
            details=details,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )
    
    def validate_unit_consistency(self, calculations: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate unit consistency in calculations
        
        Args:
            calculations: List of calculations with values and units
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        details = {
            'calculations': len(calculations),
            'unit_errors': [],
            'unit_conversions': []
        }
        
        # Common unit conversions
        unit_conversions = {
            ('m', 'cm'): 100,
            ('cm', 'm'): 0.01,
            ('kg', 'g'): 1000,
            ('g', 'kg'): 0.001,
            ('s', 'ms'): 1000,
            ('ms', 's'): 0.001,
            ('m/s', 'km/h'): 3.6,
            ('km/h', 'm/s'): 1/3.6
        }
        
        # Unit compatibility for operations
        compatible_units = {
            'length': ['m', 'cm', 'mm', 'km'],
            'mass': ['kg', 'g', 'mg'],
            'time': ['s', 'ms', 'min', 'h'],
            'velocity': ['m/s', 'km/h'],
            'acceleration': ['m/s^2', 'm/s²'],
            'force': ['N', 'kN'],
            'energy': ['J', 'kJ']
        }
        
        for calc in calculations:
            operation = calc.get('operation')
            operands = calc.get('operands', [])
            result = calc.get('result')
            
            if operation in ['add', 'subtract']:
                # Check if all operands have compatible units
                units = [op.get('unit') for op in operands if 'unit' in op]
                
                if len(set(units)) > 1:
                    # Check if units are compatible
                    compatible = False
                    for unit_group in compatible_units.values():
                        if all(unit in unit_group for unit in units):
                            compatible = True
                            warnings.append(f"Mixed units in {operation}: {units}. "
                                          "Consider converting to same unit.")
                            break
                    
                    if not compatible:
                        errors.append(f"Incompatible units in {operation}: {units}")
                        details['unit_errors'].append({
                            'operation': operation,
                            'units': units,
                            'error': 'incompatible units'
                        })
            
            elif operation == 'multiply':
                # Check unit multiplication rules
                if len(operands) == 2:
                    unit1 = operands[0].get('unit', '')
                    unit2 = operands[1].get('unit', '')
                    
                    # Simple unit multiplication rules
                    expected_unit = None
                    if unit1 == 'm' and unit2 == 'm':
                        expected_unit = 'm^2'
                    elif unit1 == 'm/s' and unit2 == 's':
                        expected_unit = 'm'
                    elif unit1 == 'kg' and unit2 == 'm/s^2':
                        expected_unit = 'N'
                    
                    if expected_unit and result.get('unit') != expected_unit:
                        warnings.append(f"Expected unit {expected_unit}, got {result.get('unit')}")
            
            elif operation == 'divide':
                # Check unit division rules
                if len(operands) == 2:
                    unit1 = operands[0].get('unit', '')
                    unit2 = operands[1].get('unit', '')
                    
                    # Simple unit division rules
                    expected_unit = None
                    if unit1 == 'm' and unit2 == 's':
                        expected_unit = 'm/s'
                    elif unit1 == unit2:
                        expected_unit = 'dimensionless'
                    
                    if expected_unit and result.get('unit') != expected_unit:
                        warnings.append(f"Expected unit {expected_unit}, got {result.get('unit')}")
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(details['unit_errors']) / len(calculations) if calculations else 0)
        
        suggestions = []
        if warnings:
            suggestions.append("Consider standardizing units before calculations")
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            validation_type=ValidationType.UNIT_CONSISTENCY.value,
            details=details,
            errors=errors if errors else None,
            warnings=warnings if warnings else None,
            suggestions=suggestions if suggestions else None
        )
    
    def validate_complete_solution(self, problem: Dict[str, Any], 
                                 solution: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive validation of a complete mathematical solution
        
        Args:
            problem: Problem description with constraints
            solution: Complete solution with steps and results
            
        Returns:
            ValidationResult
        """
        sub_validations = []
        overall_errors = []
        overall_warnings = []
        
        # Validate algebraic steps if present
        if 'steps' in solution:
            steps_result = self.validate_algebraic_steps(solution['steps'])
            sub_validations.append(steps_result)
            if steps_result.errors:
                overall_errors.extend(steps_result.errors)
        
        # Validate final answer
        if 'equation' in problem and 'answer' in solution:
            equation_result = self.validate_equation_solution(
                problem['equation'],
                problem.get('variable', 'x'),
                solution['answer']
            )
            sub_validations.append(equation_result)
            if equation_result.errors:
                overall_errors.extend(equation_result.errors)
        
        # Validate domain constraints
        if 'constraints' in problem and 'variables' in solution:
            domain_result = self.validate_domain_constraints(
                problem.get('expression', ''),
                solution['variables']
            )
            sub_validations.append(domain_result)
            if domain_result.errors:
                overall_errors.extend(domain_result.errors)
        
        # Validate unit consistency if applicable
        if 'calculations' in solution:
            unit_result = self.validate_unit_consistency(solution['calculations'])
            sub_validations.append(unit_result)
            if unit_result.warnings:
                overall_warnings.extend(unit_result.warnings)
        
        # Calculate overall validity and confidence
        is_valid = all(v.is_valid for v in sub_validations) and len(overall_errors) == 0
        confidence = np.mean([v.confidence for v in sub_validations]) if sub_validations else 0
        
        details = {
            'sub_validations': len(sub_validations),
            'all_valid': is_valid,
            'validation_summary': {
                v.validation_type: {
                    'valid': v.is_valid,
                    'confidence': v.confidence
                }
                for v in sub_validations
            }
        }
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            validation_type=ValidationType.LOGICAL_CONSISTENCY.value,
            details=details,
            errors=overall_errors if overall_errors else None,
            warnings=overall_warnings if overall_warnings else None
        )
    
    def _expressions_equivalent(self, expr1: sp.Basic, expr2: sp.Basic) -> bool:
        """Check if two expressions are mathematically equivalent"""
        try:
            # Try direct equality first
            if expr1 == expr2:
                return True
            
            # Try simplification
            diff = sp.simplify(expr1 - expr2)
            if diff == 0:
                return True
            
            # Try expansion and simplification
            expanded_diff = sp.expand(expr1 - expr2)
            if sp.simplify(expanded_diff) == 0:
                return True
            
            # Try numerical evaluation for specific values
            free_symbols = list(expr1.free_symbols.union(expr2.free_symbols))
            if free_symbols:
                # Test with random values
                test_values = {sym: np.random.randn() for sym in free_symbols}
                val1 = float(expr1.subs(test_values))
                val2 = float(expr2.subs(test_values))
                if abs(val1 - val2) < self.NUMERIC_TOLERANCE:
                    # Test with more values to be sure
                    for _ in range(5):
                        test_values = {sym: np.random.randn() for sym in free_symbols}
                        val1 = float(expr1.subs(test_values))
                        val2 = float(expr2.subs(test_values))
                        if abs(val1 - val2) >= self.NUMERIC_TOLERANCE:
                            return False
                    return True
            
            return False
            
        except:
            return False
    
    def _generate_algebraic_suggestions(self, step_details: List[Dict]) -> List[str]:
        """Generate suggestions for fixing algebraic errors"""
        suggestions = []
        
        for detail in step_details:
            if not detail['valid']:
                for issue in detail['issues']:
                    if 'incorrect' in issue.lower():
                        suggestions.append(f"Step {detail['step_number']}: Double-check the {issue}")
                    elif 'division by zero' in issue.lower():
                        suggestions.append(f"Step {detail['step_number']}: Ensure denominator is non-zero")
        
        if not suggestions:
            suggestions.append("Review algebraic manipulation rules")
        
        return suggestions
    
    def _record_validation(self, validation_type: str, is_valid: bool, confidence: float):
        """Record validation for pattern analysis"""
        self.validation_history.append({
            'type': validation_type,
            'valid': is_valid,
            'confidence': confidence,
            'timestamp': np.datetime64('now')
        })
        
        # Track error patterns
        if not is_valid:
            if validation_type not in self.error_patterns:
                self.error_patterns[validation_type] = 0
            self.error_patterns[validation_type] += 1
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about validations performed"""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'success_rate': 0,
                'average_confidence': 0,
                'error_patterns': {}
            }
        
        total = len(self.validation_history)
        successful = sum(1 for v in self.validation_history if v['valid'])
        avg_confidence = np.mean([v['confidence'] for v in self.validation_history])
        
        return {
            'total_validations': total,
            'success_rate': successful / total,
            'average_confidence': avg_confidence,
            'error_patterns': self.error_patterns,
            'validation_types': {
                vtype: sum(1 for v in self.validation_history if v['type'] == vtype)
                for vtype in set(v['type'] for v in self.validation_history)
            }
        }