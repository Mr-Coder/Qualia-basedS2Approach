"""
Math Problem Solver Project
===========================

A streamlined mathematical problem solving system focused on core functionality.

Author: Math Problem Solver Team  
Version: 3.0.0 (Streamlined)
"""

# Import core modules
try:
    from . import reasoning_core
except ImportError:
    pass

try:
    from . import evaluation
except ImportError:
    pass

# Import supporting modules  
try:
    from . import ai_core
except ImportError:
    pass

try:
    from . import processors
except ImportError:
    pass

try:
    from . import data
except ImportError:
    pass

# utilities module removed - functionality moved to new architecture

# Import specialized modules
try:
    from . import models
except ImportError:
    pass

try:
    from . import reasoning_engine
except ImportError:
    pass

# Version information
__version__ = '3.0.0'
__author__ = 'Math Problem Solver Team'

# Export main modules
__all__ = [
    'reasoning_core',    # ✅ Core reasoning engine
    'evaluation',        # ✅ Evaluation system  
    'ai_core',          # 🟡 AI interfaces
    'processors',       # 🟡 Data processing
    'data',             # 🟡 Data analysis
    'models',           # 🟡 Model management
    'reasoning_engine', # 🟡 Advanced reasoning
    'core',             # ✅ New modular architecture
    'reasoning',        # ✅ New reasoning module
] 