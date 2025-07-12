# src/models/equation.py
from typing import Any, Dict, List


class Equation:
    def __init__(self, left_side: str, right_side: str):
        self.left_side = left_side
        self.right_side = right_side

class Equations:
    def __init__(self):
        self.equations: List[Equation] = []
        self.variables: Dict[str, Any] = {}

    def add_equation(self, equation: Equation):
        self.equations.append(equation)

    def add_variable(self, name: str, value: Any):
        self.variables[name] = value
