"""
Math Problem Solver Project
===========================

A comprehensive mathematical problem solving system with advanced evaluation capabilities.

Author: Math Problem Solver Team
Version: 1.0.0
"""

# Import evaluators module
from . import evaluators

# Import processors module (if exists)
try:
    from . import processors
except ImportError:
    pass

# Import models module (if exists)
try:
    from . import models
except ImportError:
    pass

# Version information
__version__ = '1.0.0'
__author__ = 'Math Problem Solver Team'

# Export main modules
__all__ = [
    'evaluators',
    'processors',
    'models'
] 