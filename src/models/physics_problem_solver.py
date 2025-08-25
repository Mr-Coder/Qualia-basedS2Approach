"""
Physics Problem Solver for kinematics and dynamics
Implements domain-specific physics problem solving capabilities
Part of Story 6.1: Mathematical Reasoning Enhancement
"""

import sympy as sp
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .advanced_math_engine import AdvancedMathEngine, MathResult, MathOperationType

class PhysicsType(Enum):
    """Types of physics problems supported"""
    KINEMATICS_1D = "kinematics_1d"
    KINEMATICS_2D = "kinematics_2d"
    DYNAMICS = "dynamics"
    ENERGY = "energy"
    MOMENTUM = "momentum"
    CIRCULAR_MOTION = "circular_motion"
    WORK_POWER = "work_power"

@dataclass
class PhysicsQuantity:
    """Represents a physical quantity with units"""
    value: Union[float, sp.Basic]
    unit: str
    name: str
    symbol: Optional[str] = None

@dataclass
class PhysicsProblem:
    """Represents a physics problem with given data and unknowns"""
    problem_type: PhysicsType
    given_quantities: Dict[str, PhysicsQuantity]
    find_quantities: List[str]
    constraints: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

@dataclass
class PhysicsSolution:
    """Solution to a physics problem"""
    results: Dict[str, PhysicsQuantity]
    equations_used: List[str]
    solution_steps: List[Dict[str, Any]]
    validation_status: bool
    warnings: Optional[List[str]] = None

class PhysicsProblemSolver:
    """
    Physics Problem Solver for kinematics, dynamics, and related problems.
    Uses domain-specific knowledge and equations to solve physics problems.
    """
    
    # Standard physics constants
    GRAVITY = 9.81  # m/s^2
    
    # Physics equations database
    EQUATIONS = {
        'kinematics_1d': {
            'position': 'x = x0 + v0*t + 0.5*a*t^2',
            'velocity': 'v = v0 + a*t',
            'velocity_squared': 'v^2 = v0^2 + 2*a*(x - x0)',
            'average_velocity': 'v_avg = (v + v0)/2',
            'displacement': 'delta_x = v_avg * t'
        },
        'kinematics_2d': {
            'position_x': 'x = x0 + vx0*t',
            'position_y': 'y = y0 + vy0*t - 0.5*g*t^2',
            'velocity_x': 'vx = vx0',
            'velocity_y': 'vy = vy0 - g*t',
            'range': 'R = (v0^2 * sin(2*theta))/g',
            'max_height': 'H = (v0^2 * sin(theta)^2)/(2*g)',
            'time_of_flight': 'T = (2*v0*sin(theta))/g'
        },
        'dynamics': {
            'newton_second': 'F = m*a',
            'weight': 'W = m*g',
            'friction': 'f = mu*N',
            'normal_force': 'N = m*g*cos(theta)',
            'net_force': 'F_net = F_applied - f'
        },
        'energy': {
            'kinetic': 'KE = 0.5*m*v^2',
            'potential_gravity': 'PE = m*g*h',
            'potential_spring': 'PE_spring = 0.5*k*x^2',
            'conservation': 'KE_i + PE_i = KE_f + PE_f',
            'work': 'W = F*d*cos(theta)'
        },
        'momentum': {
            'linear': 'p = m*v',
            'conservation': 'p_i = p_f',
            'impulse': 'J = F*t = delta_p',
            'collision_elastic': 'm1*v1i + m2*v2i = m1*v1f + m2*v2f'
        }
    }
    
    def __init__(self):
        """Initialize the Physics Problem Solver"""
        self.math_engine = AdvancedMathEngine()
        self.symbol_registry = {}
        self.solution_history = []
    
    def solve_kinematics_1d(self, problem: PhysicsProblem) -> PhysicsSolution:
        """
        Solve 1D kinematics problems
        
        Args:
            problem: PhysicsProblem with kinematics data
            
        Returns:
            PhysicsSolution with results
        """
        steps = []
        equations_used = []
        warnings = []
        results = {}
        
        # Extract given quantities
        given = problem.given_quantities
        find = problem.find_quantities
        
        steps.append({
            "action": "identify_problem",
            "type": "1D Kinematics",
            "given": list(given.keys()),
            "find": find,
            "description": "Identified 1D kinematics problem"
        })
        
        # Common kinematics variables
        variables = {
            'x': 'position',
            'x0': 'initial_position', 
            'v': 'velocity',
            'v0': 'initial_velocity',
            'a': 'acceleration',
            't': 'time',
            'delta_x': 'displacement'
        }
        
        # Create symbols for all variables
        symbols = {}
        for var, name in variables.items():
            symbols[var] = sp.Symbol(var, real=True)
        
        # Assign known values
        known_values = {}
        for key, quantity in given.items():
            if key in variables:
                known_values[symbols[key]] = quantity.value
        
        # Default initial position to 0 if not given
        if 'x0' not in given and symbols['x0'] not in known_values:
            known_values[symbols['x0']] = 0
            steps.append({
                "action": "assume_default",
                "variable": "x0",
                "value": 0,
                "description": "Assumed initial position x0 = 0"
            })
        
        # Solve for each requested quantity
        for target in find:
            if target not in variables:
                warnings.append(f"Unknown quantity requested: {target}")
                continue
            
            target_symbol = symbols[target]
            
            # Try different equations to solve for target
            solved = False
            
            # Strategy 1: Direct calculation using kinematic equations
            if target == 'x' and all(k in known_values for k in [symbols['x0'], symbols['v0'], symbols['a'], symbols['t']]):
                # x = x0 + v0*t + 0.5*a*t^2
                x0_val = known_values[symbols['x0']]
                v0_val = known_values[symbols['v0']]
                a_val = known_values[symbols['a']]
                t_val = known_values[symbols['t']]
                
                result = x0_val + v0_val * t_val + 0.5 * a_val * t_val**2
                results[target] = PhysicsQuantity(result, 'm', 'position', 'x')
                equations_used.append(self.EQUATIONS['kinematics_1d']['position'])
                solved = True
                
                steps.append({
                    "action": "calculate",
                    "equation": self.EQUATIONS['kinematics_1d']['position'],
                    "target": target,
                    "result": float(result) if isinstance(result, (int, float)) else str(result),
                    "description": f"Calculated {target} using position equation"
                })
                
            elif target == 'v' and all(k in known_values for k in [symbols['v0'], symbols['a'], symbols['t']]):
                # v = v0 + a*t
                v0_val = known_values[symbols['v0']]
                a_val = known_values[symbols['a']]
                t_val = known_values[symbols['t']]
                
                result = v0_val + a_val * t_val
                results[target] = PhysicsQuantity(result, 'm/s', 'velocity', 'v')
                equations_used.append(self.EQUATIONS['kinematics_1d']['velocity'])
                solved = True
                
                steps.append({
                    "action": "calculate",
                    "equation": self.EQUATIONS['kinematics_1d']['velocity'],
                    "target": target,
                    "result": float(result) if isinstance(result, (int, float)) else str(result),
                    "description": f"Calculated {target} using velocity equation"
                })
                
            elif target == 't' and 'v' in given and 'v0' in given and 'a' in given:
                # Solve v = v0 + a*t for t
                equation_str = f"{given['v'].value} = {given['v0'].value} + {given['a'].value}*t"
                math_result = self.math_engine.solve_algebraic(equation_str, 't')
                
                if math_result.validation_status and math_result.solution:
                    # Take positive time value
                    time_solutions = [sol for sol in math_result.solution if sol >= 0]
                    if time_solutions:
                        result = time_solutions[0]
                        results[target] = PhysicsQuantity(result, 's', 'time', 't')
                        equations_used.append(self.EQUATIONS['kinematics_1d']['velocity'])
                        solved = True
                        
                        steps.append({
                            "action": "solve_equation",
                            "equation": self.EQUATIONS['kinematics_1d']['velocity'],
                            "target": target,
                            "result": float(result) if isinstance(result, (int, float)) else str(result),
                            "description": f"Solved for {target} using velocity equation"
                        })
            
            # Strategy 2: Use velocity-squared equation if needed
            if not solved and target == 'v' and all(k in given for k in ['v0', 'a', 'x', 'x0']):
                # v^2 = v0^2 + 2*a*(x - x0)
                v0_val = given['v0'].value
                a_val = given['a'].value
                x_val = given['x'].value
                x0_val = given.get('x0', PhysicsQuantity(0, 'm', 'initial_position')).value
                
                v_squared = v0_val**2 + 2*a_val*(x_val - x0_val)
                if v_squared >= 0:
                    result = sp.sqrt(v_squared)
                    results[target] = PhysicsQuantity(result, 'm/s', 'velocity', 'v')
                    equations_used.append(self.EQUATIONS['kinematics_1d']['velocity_squared'])
                    solved = True
                    
                    steps.append({
                        "action": "calculate",
                        "equation": self.EQUATIONS['kinematics_1d']['velocity_squared'],
                        "target": target,
                        "result": float(result) if isinstance(result, (int, float)) else str(result),
                        "description": f"Calculated {target} using velocity-squared equation"
                    })
                else:
                    warnings.append(f"Negative value under square root for velocity calculation")
            
            if not solved:
                warnings.append(f"Unable to solve for {target} with given information")
        
        # Validate results
        validation_status = len(results) == len(find) and len(warnings) == 0
        
        # Record solution
        self.solution_history.append({
            "problem_type": "kinematics_1d",
            "given": list(given.keys()),
            "found": list(results.keys()),
            "success": validation_status
        })
        
        return PhysicsSolution(
            results=results,
            equations_used=equations_used,
            solution_steps=steps,
            validation_status=validation_status,
            warnings=warnings if warnings else None
        )
    
    def solve_projectile_motion(self, initial_velocity: float, angle_degrees: float,
                               initial_height: float = 0) -> PhysicsSolution:
        """
        Solve 2D projectile motion problems
        
        Args:
            initial_velocity: Initial velocity magnitude (m/s)
            angle_degrees: Launch angle in degrees
            initial_height: Initial height (m), default 0
            
        Returns:
            PhysicsSolution with trajectory information
        """
        steps = []
        equations_used = []
        results = {}
        
        # Convert angle to radians
        angle_rad = np.radians(angle_degrees)
        
        steps.append({
            "action": "setup",
            "initial_velocity": initial_velocity,
            "angle_degrees": angle_degrees,
            "angle_radians": angle_rad,
            "initial_height": initial_height,
            "description": "Set up projectile motion problem"
        })
        
        # Calculate velocity components
        vx0 = initial_velocity * np.cos(angle_rad)
        vy0 = initial_velocity * np.sin(angle_rad)
        
        results['vx0'] = PhysicsQuantity(vx0, 'm/s', 'initial_x_velocity', 'vx0')
        results['vy0'] = PhysicsQuantity(vy0, 'm/s', 'initial_y_velocity', 'vy0')
        
        steps.append({
            "action": "decompose_velocity",
            "vx0": vx0,
            "vy0": vy0,
            "description": "Decomposed initial velocity into components"
        })
        
        # Calculate time of flight
        # Using quadratic formula for: y = y0 + vy0*t - 0.5*g*t^2 = 0
        a = -0.5 * self.GRAVITY
        b = vy0
        c = initial_height
        
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
            time_of_flight = max(t1, t2)  # Take positive time
            
            results['time_of_flight'] = PhysicsQuantity(
                time_of_flight, 's', 'time_of_flight', 'T'
            )
            equations_used.append(self.EQUATIONS['kinematics_2d']['position_y'])
            
            steps.append({
                "action": "calculate_time",
                "equation": "Quadratic formula on y-position equation",
                "time": time_of_flight,
                "description": "Calculated time of flight"
            })
            
            # Calculate range
            range_x = vx0 * time_of_flight
            results['range'] = PhysicsQuantity(range_x, 'm', 'range', 'R')
            equations_used.append(self.EQUATIONS['kinematics_2d']['position_x'])
            
            steps.append({
                "action": "calculate_range",
                "range": range_x,
                "description": "Calculated horizontal range"
            })
            
            # Calculate maximum height
            # At max height, vy = 0
            time_to_max = vy0 / self.GRAVITY
            max_height = initial_height + vy0 * time_to_max - 0.5 * self.GRAVITY * time_to_max**2
            
            results['max_height'] = PhysicsQuantity(max_height, 'm', 'maximum_height', 'H')
            equations_used.append(self.EQUATIONS['kinematics_2d']['position_y'])
            
            steps.append({
                "action": "calculate_max_height",
                "time_to_max": time_to_max,
                "max_height": max_height,
                "description": "Calculated maximum height"
            })
            
            # Calculate impact velocity
            vx_impact = vx0  # Constant in projectile motion
            vy_impact = vy0 - self.GRAVITY * time_of_flight
            v_impact = np.sqrt(vx_impact**2 + vy_impact**2)
            
            results['impact_velocity'] = PhysicsQuantity(
                v_impact, 'm/s', 'impact_velocity', 'v_f'
            )
            results['impact_angle'] = PhysicsQuantity(
                np.degrees(np.arctan2(abs(vy_impact), vx_impact)), 
                'degrees', 'impact_angle', 'theta_f'
            )
            
            steps.append({
                "action": "calculate_impact",
                "vx_impact": vx_impact,
                "vy_impact": vy_impact,
                "v_impact": v_impact,
                "description": "Calculated impact velocity and angle"
            })
            
            validation_status = True
        else:
            validation_status = False
            warnings = ["Invalid projectile motion parameters"]
        
        return PhysicsSolution(
            results=results,
            equations_used=equations_used,
            solution_steps=steps,
            validation_status=validation_status,
            warnings=None if validation_status else warnings
        )
    
    def solve_dynamics_problem(self, problem: PhysicsProblem) -> PhysicsSolution:
        """
        Solve dynamics problems involving forces and Newton's laws
        
        Args:
            problem: PhysicsProblem with dynamics data
            
        Returns:
            PhysicsSolution with forces and accelerations
        """
        steps = []
        equations_used = []
        warnings = []
        results = {}
        
        given = problem.given_quantities
        find = problem.find_quantities
        
        steps.append({
            "action": "identify_problem",
            "type": "Dynamics",
            "given": list(given.keys()),
            "find": find,
            "description": "Identified dynamics problem"
        })
        
        # Common dynamics calculations
        for target in find:
            if target == 'acceleration' and 'force' in given and 'mass' in given:
                # F = ma → a = F/m
                force = given['force'].value
                mass = given['mass'].value
                
                acceleration = force / mass
                results['acceleration'] = PhysicsQuantity(
                    acceleration, 'm/s^2', 'acceleration', 'a'
                )
                equations_used.append(self.EQUATIONS['dynamics']['newton_second'])
                
                steps.append({
                    "action": "calculate",
                    "equation": "a = F/m",
                    "result": acceleration,
                    "description": "Calculated acceleration using Newton's second law"
                })
                
            elif target == 'force' and 'mass' in given and 'acceleration' in given:
                # F = ma
                mass = given['mass'].value
                acceleration = given['acceleration'].value
                
                force = mass * acceleration
                results['force'] = PhysicsQuantity(force, 'N', 'force', 'F')
                equations_used.append(self.EQUATIONS['dynamics']['newton_second'])
                
                steps.append({
                    "action": "calculate",
                    "equation": "F = ma",
                    "result": force,
                    "description": "Calculated force using Newton's second law"
                })
                
            elif target == 'weight' and 'mass' in given:
                # W = mg
                mass = given['mass'].value
                weight = mass * self.GRAVITY
                
                results['weight'] = PhysicsQuantity(weight, 'N', 'weight', 'W')
                equations_used.append(self.EQUATIONS['dynamics']['weight'])
                
                steps.append({
                    "action": "calculate",
                    "equation": "W = mg",
                    "result": weight,
                    "description": "Calculated weight"
                })
                
            elif target == 'friction' and 'coefficient' in given and 'normal_force' in given:
                # f = μN
                mu = given['coefficient'].value
                normal = given['normal_force'].value
                
                friction = mu * normal
                results['friction'] = PhysicsQuantity(friction, 'N', 'friction', 'f')
                equations_used.append(self.EQUATIONS['dynamics']['friction'])
                
                steps.append({
                    "action": "calculate",
                    "equation": "f = μN",
                    "result": friction,
                    "description": "Calculated friction force"
                })
            else:
                warnings.append(f"Unable to calculate {target} with given information")
        
        validation_status = len(warnings) == 0 and len(results) > 0
        
        return PhysicsSolution(
            results=results,
            equations_used=equations_used,
            solution_steps=steps,
            validation_status=validation_status,
            warnings=warnings if warnings else None
        )
    
    def solve_energy_problem(self, problem: PhysicsProblem) -> PhysicsSolution:
        """
        Solve energy and work problems
        
        Args:
            problem: PhysicsProblem with energy data
            
        Returns:
            PhysicsSolution with energy calculations
        """
        steps = []
        equations_used = []
        warnings = []
        results = {}
        
        given = problem.given_quantities
        find = problem.find_quantities
        
        steps.append({
            "action": "identify_problem",
            "type": "Energy",
            "given": list(given.keys()),
            "find": find,
            "description": "Identified energy problem"
        })
        
        # Energy calculations
        for target in find:
            if target == 'kinetic_energy' and 'mass' in given and 'velocity' in given:
                # KE = 0.5*m*v^2
                mass = given['mass'].value
                velocity = given['velocity'].value
                
                ke = 0.5 * mass * velocity**2
                results['kinetic_energy'] = PhysicsQuantity(ke, 'J', 'kinetic_energy', 'KE')
                equations_used.append(self.EQUATIONS['energy']['kinetic'])
                
                steps.append({
                    "action": "calculate",
                    "equation": "KE = 0.5*m*v^2",
                    "result": ke,
                    "description": "Calculated kinetic energy"
                })
                
            elif target == 'potential_energy' and 'mass' in given and 'height' in given:
                # PE = mgh
                mass = given['mass'].value
                height = given['height'].value
                
                pe = mass * self.GRAVITY * height
                results['potential_energy'] = PhysicsQuantity(
                    pe, 'J', 'potential_energy', 'PE'
                )
                equations_used.append(self.EQUATIONS['energy']['potential_gravity'])
                
                steps.append({
                    "action": "calculate",
                    "equation": "PE = mgh",
                    "result": pe,
                    "description": "Calculated gravitational potential energy"
                })
                
            elif target == 'work' and 'force' in given and 'distance' in given:
                # W = F*d*cos(theta)
                force = given['force'].value
                distance = given['distance'].value
                angle = given.get('angle', PhysicsQuantity(0, 'degrees', 'angle')).value
                
                # Convert angle to radians if needed
                if given.get('angle') and given['angle'].unit == 'degrees':
                    angle = np.radians(angle)
                
                work = force * distance * np.cos(angle)
                results['work'] = PhysicsQuantity(work, 'J', 'work', 'W')
                equations_used.append(self.EQUATIONS['energy']['work'])
                
                steps.append({
                    "action": "calculate",
                    "equation": "W = F*d*cos(θ)",
                    "result": work,
                    "description": "Calculated work done"
                })
            else:
                warnings.append(f"Unable to calculate {target} with given information")
        
        # Check for energy conservation problems
        if 'initial_ke' in given and 'initial_pe' in given and 'final_ke' in find:
            # Use conservation of energy
            initial_total = given['initial_ke'].value + given['initial_pe'].value
            final_pe = given.get('final_pe', PhysicsQuantity(0, 'J', 'final_pe')).value
            
            final_ke = initial_total - final_pe
            results['final_ke'] = PhysicsQuantity(final_ke, 'J', 'final_kinetic_energy', 'KE_f')
            equations_used.append(self.EQUATIONS['energy']['conservation'])
            
            steps.append({
                "action": "apply_conservation",
                "equation": "KE_i + PE_i = KE_f + PE_f",
                "result": final_ke,
                "description": "Applied conservation of energy"
            })
        
        validation_status = len(warnings) == 0 and len(results) > 0
        
        return PhysicsSolution(
            results=results,
            equations_used=equations_used,
            solution_steps=steps,
            validation_status=validation_status,
            warnings=warnings if warnings else None
        )
    
    def create_problem(self, problem_type: str, **kwargs) -> PhysicsProblem:
        """
        Helper method to create a PhysicsProblem object
        
        Args:
            problem_type: Type of physics problem
            **kwargs: Problem parameters
            
        Returns:
            PhysicsProblem object
        """
        # Map string to enum
        type_map = {
            'kinematics_1d': PhysicsType.KINEMATICS_1D,
            'kinematics_2d': PhysicsType.KINEMATICS_2D,
            'dynamics': PhysicsType.DYNAMICS,
            'energy': PhysicsType.ENERGY,
            'momentum': PhysicsType.MOMENTUM
        }
        
        physics_type = type_map.get(problem_type.lower())
        if not physics_type:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        given_quantities = kwargs.get('given', {})
        find_quantities = kwargs.get('find', [])
        constraints = kwargs.get('constraints', None)
        description = kwargs.get('description', None)
        
        return PhysicsProblem(
            problem_type=physics_type,
            given_quantities=given_quantities,
            find_quantities=find_quantities,
            constraints=constraints,
            description=description
        )
    
    def get_solution_history(self) -> List[Dict[str, Any]]:
        """Get history of solved problems"""
        return self.solution_history.copy()