"""
Geometry Engine for 2D and 3D calculations
Implements geometric problem solving capabilities for shapes, angles, and spatial relationships
Part of Story 6.1: Mathematical Reasoning Enhancement
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from .advanced_math_engine import AdvancedMathEngine, MathResult

class GeometryType(Enum):
    """Types of geometric objects and problems"""
    # 2D Shapes
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    CIRCLE = "circle"
    POLYGON = "polygon"
    
    # 3D Shapes
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CONE = "cone"
    PYRAMID = "pyramid"
    PRISM = "prism"
    
    # Problems
    DISTANCE = "distance"
    ANGLE = "angle"
    TRANSFORMATION = "transformation"
    INTERSECTION = "intersection"

@dataclass
class Point2D:
    """Represents a 2D point"""
    x: float
    y: float
    
    def distance_to(self, other: 'Point2D') -> float:
        """Calculate distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Point3D:
    """Represents a 3D point"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        """Calculate distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class GeometricShape:
    """Base class for geometric shapes"""
    shape_type: GeometryType
    dimensions: Dict[str, float]
    properties: Dict[str, Any]

@dataclass
class GeometrySolution:
    """Solution to a geometry problem"""
    results: Dict[str, Any]
    formulas_used: List[str]
    solution_steps: List[Dict[str, Any]]
    validation_status: bool
    visualizations: Optional[List[Any]] = None
    warnings: Optional[List[str]] = None

class GeometryEngine:
    """
    Geometry Engine for 2D and 3D calculations.
    Handles geometric shapes, transformations, and spatial relationships.
    """
    
    # Common geometry formulas
    FORMULAS = {
        'triangle': {
            'area_base_height': 'A = 0.5 * base * height',
            'area_heron': 'A = sqrt(s*(s-a)*(s-b)*(s-c)), s = (a+b+c)/2',
            'perimeter': 'P = a + b + c',
            'pythagorean': 'c^2 = a^2 + b^2',
            'sine_rule': 'a/sin(A) = b/sin(B) = c/sin(C)',
            'cosine_rule': 'c^2 = a^2 + b^2 - 2*a*b*cos(C)'
        },
        'circle': {
            'area': 'A = π * r^2',
            'circumference': 'C = 2 * π * r',
            'arc_length': 'L = r * θ',
            'sector_area': 'A = 0.5 * r^2 * θ'
        },
        'rectangle': {
            'area': 'A = length * width',
            'perimeter': 'P = 2 * (length + width)',
            'diagonal': 'd = sqrt(length^2 + width^2)'
        },
        'sphere': {
            'volume': 'V = (4/3) * π * r^3',
            'surface_area': 'A = 4 * π * r^2'
        },
        'cylinder': {
            'volume': 'V = π * r^2 * h',
            'surface_area': 'A = 2*π*r*h + 2*π*r^2',
            'lateral_area': 'A = 2*π*r*h'
        },
        'cone': {
            'volume': 'V = (1/3) * π * r^2 * h',
            'surface_area': 'A = π*r*l + π*r^2',
            'slant_height': 'l = sqrt(r^2 + h^2)'
        }
    }
    
    def __init__(self):
        """Initialize the Geometry Engine"""
        self.math_engine = AdvancedMathEngine()
        self.calculation_history = []
    
    def calculate_triangle_properties(self, **kwargs) -> GeometrySolution:
        """
        Calculate properties of a triangle given various inputs
        
        Args:
            **kwargs: Can include sides (a, b, c), angles (A, B, C), 
                     base, height, vertices, etc.
                     
        Returns:
            GeometrySolution with calculated properties
        """
        steps = []
        formulas_used = []
        results = {}
        warnings = []
        
        # Extract given information
        sides = {k: v for k, v in kwargs.items() if k in ['a', 'b', 'c']}
        angles = {k: v for k, v in kwargs.items() if k in ['A', 'B', 'C']}
        
        steps.append({
            "action": "analyze_input",
            "given_sides": list(sides.keys()),
            "given_angles": list(angles.keys()),
            "description": "Analyzed triangle input parameters"
        })
        
        # Case 1: Three sides given (SSS) - use Heron's formula
        if len(sides) == 3:
            a, b, c = sides.get('a', 0), sides.get('b', 0), sides.get('c', 0)
            
            # Check triangle inequality
            if a + b > c and b + c > a and a + c > b:
                # Calculate perimeter
                perimeter = a + b + c
                results['perimeter'] = perimeter
                formulas_used.append(self.FORMULAS['triangle']['perimeter'])
                
                # Calculate area using Heron's formula
                s = perimeter / 2
                area = math.sqrt(s * (s - a) * (s - b) * (s - c))
                results['area'] = area
                formulas_used.append(self.FORMULAS['triangle']['area_heron'])
                
                # Calculate angles using cosine rule
                angle_A = math.degrees(math.acos((b**2 + c**2 - a**2) / (2 * b * c)))
                angle_B = math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))
                angle_C = 180 - angle_A - angle_B
                
                results['angle_A'] = angle_A
                results['angle_B'] = angle_B
                results['angle_C'] = angle_C
                formulas_used.append(self.FORMULAS['triangle']['cosine_rule'])
                
                steps.append({
                    "action": "calculate_sss",
                    "perimeter": perimeter,
                    "area": area,
                    "angles": [angle_A, angle_B, angle_C],
                    "description": "Calculated triangle properties using SSS method"
                })
            else:
                warnings.append("Invalid triangle: sides do not satisfy triangle inequality")
                
        # Case 2: Base and height given
        elif 'base' in kwargs and 'height' in kwargs:
            base = kwargs['base']
            height = kwargs['height']
            
            area = 0.5 * base * height
            results['area'] = area
            formulas_used.append(self.FORMULAS['triangle']['area_base_height'])
            
            steps.append({
                "action": "calculate_area",
                "base": base,
                "height": height,
                "area": area,
                "description": "Calculated area using base and height"
            })
            
        # Case 3: Right triangle with two sides
        elif 'right_angle' in kwargs and len(sides) == 2:
            # Apply Pythagorean theorem
            known_sides = list(sides.items())
            if 'c' not in sides:  # c is hypotenuse
                a = sides.get('a', 0)
                b = sides.get('b', 0)
                c = math.sqrt(a**2 + b**2)
                results['c'] = c
                results['hypotenuse'] = c
            elif 'a' not in sides:
                b = sides.get('b', 0)
                c = sides.get('c', 0)
                if c > b:
                    a = math.sqrt(c**2 - b**2)
                    results['a'] = a
                else:
                    warnings.append("Invalid right triangle: hypotenuse must be longest side")
            
            if not warnings:
                formulas_used.append(self.FORMULAS['triangle']['pythagorean'])
                
                # Calculate area
                if 'a' in sides or 'a' in results:
                    a = sides.get('a', results.get('a', 0))
                if 'b' in sides or 'b' in results:
                    b = sides.get('b', results.get('b', 0))
                
                area = 0.5 * a * b
                results['area'] = area
                
                steps.append({
                    "action": "calculate_right_triangle",
                    "description": "Calculated right triangle properties using Pythagorean theorem"
                })
        
        # Case 4: Vertices given
        elif 'vertices' in kwargs:
            vertices = kwargs['vertices']
            if len(vertices) == 3:
                # Calculate area using cross product formula
                x1, y1 = vertices[0]
                x2, y2 = vertices[1]
                x3, y3 = vertices[2]
                
                area = abs(0.5 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))
                results['area'] = area
                
                # Calculate side lengths
                a = math.sqrt((x2-x3)**2 + (y2-y3)**2)
                b = math.sqrt((x1-x3)**2 + (y1-y3)**2)
                c = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                results['a'] = a
                results['b'] = b
                results['c'] = c
                results['perimeter'] = a + b + c
                
                steps.append({
                    "action": "calculate_from_vertices",
                    "area": area,
                    "sides": [a, b, c],
                    "description": "Calculated triangle properties from vertices"
                })
            else:
                warnings.append("Triangle requires exactly 3 vertices")
        
        validation_status = len(warnings) == 0 and len(results) > 0
        
        # Record calculation
        self.calculation_history.append({
            "shape": "triangle",
            "input": kwargs,
            "results": results,
            "success": validation_status
        })
        
        return GeometrySolution(
            results=results,
            formulas_used=formulas_used,
            solution_steps=steps,
            validation_status=validation_status,
            warnings=warnings if warnings else None
        )
    
    def calculate_circle_properties(self, **kwargs) -> GeometrySolution:
        """
        Calculate properties of a circle
        
        Args:
            **kwargs: Can include radius, diameter, area, circumference
            
        Returns:
            GeometrySolution with calculated properties
        """
        steps = []
        formulas_used = []
        results = {}
        
        # Determine what we know
        radius = kwargs.get('radius')
        diameter = kwargs.get('diameter')
        area = kwargs.get('area')
        circumference = kwargs.get('circumference')
        
        steps.append({
            "action": "analyze_input",
            "given": list(kwargs.keys()),
            "description": "Analyzed circle input parameters"
        })
        
        # Find radius first
        if radius is not None:
            r = radius
        elif diameter is not None:
            r = diameter / 2
            results['radius'] = r
        elif area is not None:
            r = math.sqrt(area / math.pi)
            results['radius'] = r
        elif circumference is not None:
            r = circumference / (2 * math.pi)
            results['radius'] = r
        else:
            return GeometrySolution(
                results={},
                formulas_used=[],
                solution_steps=steps,
                validation_status=False,
                warnings=["Insufficient information to calculate circle properties"]
            )
        
        # Calculate all properties
        results['radius'] = r
        results['diameter'] = 2 * r
        results['area'] = math.pi * r**2
        results['circumference'] = 2 * math.pi * r
        
        formulas_used.extend([
            self.FORMULAS['circle']['area'],
            self.FORMULAS['circle']['circumference']
        ])
        
        steps.append({
            "action": "calculate_properties",
            "radius": r,
            "area": results['area'],
            "circumference": results['circumference'],
            "description": "Calculated all circle properties"
        })
        
        # Arc and sector calculations if angle provided
        if 'angle' in kwargs:
            angle_rad = math.radians(kwargs['angle'])
            arc_length = r * angle_rad
            sector_area = 0.5 * r**2 * angle_rad
            
            results['arc_length'] = arc_length
            results['sector_area'] = sector_area
            
            formulas_used.extend([
                self.FORMULAS['circle']['arc_length'],
                self.FORMULAS['circle']['sector_area']
            ])
            
            steps.append({
                "action": "calculate_arc_sector",
                "angle_degrees": kwargs['angle'],
                "arc_length": arc_length,
                "sector_area": sector_area,
                "description": "Calculated arc and sector properties"
            })
        
        return GeometrySolution(
            results=results,
            formulas_used=formulas_used,
            solution_steps=steps,
            validation_status=True
        )
    
    def calculate_3d_shape_properties(self, shape_type: str, **kwargs) -> GeometrySolution:
        """
        Calculate properties of 3D shapes
        
        Args:
            shape_type: Type of 3D shape (sphere, cylinder, cone, etc.)
            **kwargs: Shape-specific parameters
            
        Returns:
            GeometrySolution with calculated properties
        """
        steps = []
        formulas_used = []
        results = {}
        warnings = []
        
        shape_type = shape_type.lower()
        
        steps.append({
            "action": "identify_shape",
            "shape": shape_type,
            "parameters": list(kwargs.keys()),
            "description": f"Identified {shape_type} calculation"
        })
        
        if shape_type == 'sphere':
            radius = kwargs.get('radius')
            if radius:
                volume = (4/3) * math.pi * radius**3
                surface_area = 4 * math.pi * radius**2
                
                results['volume'] = volume
                results['surface_area'] = surface_area
                results['diameter'] = 2 * radius
                
                formulas_used.extend([
                    self.FORMULAS['sphere']['volume'],
                    self.FORMULAS['sphere']['surface_area']
                ])
                
                steps.append({
                    "action": "calculate_sphere",
                    "radius": radius,
                    "volume": volume,
                    "surface_area": surface_area,
                    "description": "Calculated sphere properties"
                })
            else:
                warnings.append("Sphere calculation requires radius")
                
        elif shape_type == 'cylinder':
            radius = kwargs.get('radius')
            height = kwargs.get('height')
            
            if radius and height:
                volume = math.pi * radius**2 * height
                lateral_area = 2 * math.pi * radius * height
                surface_area = lateral_area + 2 * math.pi * radius**2
                
                results['volume'] = volume
                results['lateral_area'] = lateral_area
                results['surface_area'] = surface_area
                
                formulas_used.extend([
                    self.FORMULAS['cylinder']['volume'],
                    self.FORMULAS['cylinder']['lateral_area'],
                    self.FORMULAS['cylinder']['surface_area']
                ])
                
                steps.append({
                    "action": "calculate_cylinder",
                    "radius": radius,
                    "height": height,
                    "volume": volume,
                    "surface_area": surface_area,
                    "description": "Calculated cylinder properties"
                })
            else:
                warnings.append("Cylinder calculation requires radius and height")
                
        elif shape_type == 'cone':
            radius = kwargs.get('radius')
            height = kwargs.get('height')
            
            if radius and height:
                volume = (1/3) * math.pi * radius**2 * height
                slant_height = math.sqrt(radius**2 + height**2)
                lateral_area = math.pi * radius * slant_height
                surface_area = lateral_area + math.pi * radius**2
                
                results['volume'] = volume
                results['slant_height'] = slant_height
                results['lateral_area'] = lateral_area
                results['surface_area'] = surface_area
                
                formulas_used.extend([
                    self.FORMULAS['cone']['volume'],
                    self.FORMULAS['cone']['slant_height'],
                    self.FORMULAS['cone']['surface_area']
                ])
                
                steps.append({
                    "action": "calculate_cone",
                    "radius": radius,
                    "height": height,
                    "volume": volume,
                    "slant_height": slant_height,
                    "description": "Calculated cone properties"
                })
            else:
                warnings.append("Cone calculation requires radius and height")
                
        elif shape_type == 'cube':
            side = kwargs.get('side')
            if side:
                volume = side**3
                surface_area = 6 * side**2
                diagonal = side * math.sqrt(3)
                
                results['volume'] = volume
                results['surface_area'] = surface_area
                results['diagonal'] = diagonal
                
                steps.append({
                    "action": "calculate_cube",
                    "side": side,
                    "volume": volume,
                    "surface_area": surface_area,
                    "diagonal": diagonal,
                    "description": "Calculated cube properties"
                })
            else:
                warnings.append("Cube calculation requires side length")
        
        elif shape_type == 'pyramid':
            base_area = kwargs.get('base_area')
            height = kwargs.get('height')
            
            if base_area and height:
                volume = (1/3) * base_area * height
                results['volume'] = volume
                
                steps.append({
                    "action": "calculate_pyramid",
                    "base_area": base_area,
                    "height": height,
                    "volume": volume,
                    "description": "Calculated pyramid volume"
                })
            else:
                warnings.append("Pyramid calculation requires base area and height")
        
        validation_status = len(warnings) == 0 and len(results) > 0
        
        return GeometrySolution(
            results=results,
            formulas_used=formulas_used,
            solution_steps=steps,
            validation_status=validation_status,
            warnings=warnings if warnings else None
        )
    
    def calculate_distance(self, point1: Union[Point2D, Point3D, List], 
                          point2: Union[Point2D, Point3D, List]) -> GeometrySolution:
        """
        Calculate distance between two points (2D or 3D)
        
        Args:
            point1: First point
            point2: Second point
            
        Returns:
            GeometrySolution with distance
        """
        steps = []
        results = {}
        
        # Convert lists to Point objects if needed
        if isinstance(point1, list):
            if len(point1) == 2:
                point1 = Point2D(point1[0], point1[1])
            elif len(point1) == 3:
                point1 = Point3D(point1[0], point1[1], point1[2])
        
        if isinstance(point2, list):
            if len(point2) == 2:
                point2 = Point2D(point2[0], point2[1])
            elif len(point2) == 3:
                point2 = Point3D(point2[0], point2[1], point2[2])
        
        # Calculate distance
        if isinstance(point1, Point2D) and isinstance(point2, Point2D):
            distance = point1.distance_to(point2)
            formula = "d = sqrt((x2-x1)^2 + (y2-y1)^2)"
            dimension = "2D"
        elif isinstance(point1, Point3D) and isinstance(point2, Point3D):
            distance = point1.distance_to(point2)
            formula = "d = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)"
            dimension = "3D"
        else:
            return GeometrySolution(
                results={},
                formulas_used=[],
                solution_steps=[],
                validation_status=False,
                warnings=["Points must be of same dimension (2D or 3D)"]
            )
        
        results['distance'] = distance
        results['dimension'] = dimension
        
        steps.append({
            "action": "calculate_distance",
            "point1": str(point1),
            "point2": str(point2),
            "distance": distance,
            "description": f"Calculated {dimension} distance between points"
        })
        
        return GeometrySolution(
            results=results,
            formulas_used=[formula],
            solution_steps=steps,
            validation_status=True
        )
    
    def calculate_angle(self, point1: List[float], vertex: List[float], 
                       point2: List[float], unit: str = 'degrees') -> GeometrySolution:
        """
        Calculate angle between three points (vertex is the middle point)
        
        Args:
            point1: First point
            vertex: Vertex point (where angle is measured)
            point2: Second point
            unit: 'degrees' or 'radians'
            
        Returns:
            GeometrySolution with angle
        """
        steps = []
        results = {}
        
        # Create vectors from vertex to each point
        v1 = np.array(point1) - np.array(vertex)
        v2 = np.array(point2) - np.array(vertex)
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        # Handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = math.acos(cos_angle)
        
        if unit == 'degrees':
            angle = math.degrees(angle_rad)
        else:
            angle = angle_rad
        
        results['angle'] = angle
        results['unit'] = unit
        results['cos_angle'] = cos_angle
        
        steps.append({
            "action": "calculate_angle",
            "vertex": vertex,
            "angle": angle,
            "unit": unit,
            "description": f"Calculated angle at vertex using dot product"
        })
        
        return GeometrySolution(
            results=results,
            formulas_used=["cos(θ) = (v1 · v2) / (|v1| × |v2|)"],
            solution_steps=steps,
            validation_status=True
        )
    
    def solve_trigonometry(self, **kwargs) -> GeometrySolution:
        """
        Solve trigonometry problems
        
        Args:
            **kwargs: Can include angles, sides, and trigonometric values
            
        Returns:
            GeometrySolution with results
        """
        steps = []
        formulas_used = []
        results = {}
        
        # Basic trigonometric calculations
        if 'angle' in kwargs:
            angle_deg = kwargs['angle']
            angle_rad = math.radians(angle_deg)
            
            results['sin'] = math.sin(angle_rad)
            results['cos'] = math.cos(angle_rad)
            results['tan'] = math.tan(angle_rad)
            
            steps.append({
                "action": "calculate_trig_functions",
                "angle_degrees": angle_deg,
                "sin": results['sin'],
                "cos": results['cos'],
                "tan": results['tan'],
                "description": "Calculated trigonometric functions"
            })
            
        # Solve triangle using sine or cosine rule
        if 'triangle_sides' in kwargs or 'triangle_angles' in kwargs:
            # Implementation for sine/cosine rule solving
            pass
        
        return GeometrySolution(
            results=results,
            formulas_used=formulas_used,
            solution_steps=steps,
            validation_status=True
        )
    
    def get_calculation_history(self) -> List[Dict[str, Any]]:
        """Get history of calculations"""
        return self.calculation_history.copy()