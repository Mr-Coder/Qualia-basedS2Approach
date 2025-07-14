"""
COT-DIR UI模块
为数学推理系统提供用户界面支持
"""

from .interfaces import UIRequest, UIResponse, IUIComponent
from .core import UIManager, UIRenderer, UIEventHandler
from .components import BaseProblemInputComponent, BaseReasoningDisplayComponent, BaseResultDisplayComponent
from .error_handling import UIError, UIErrorHandler, handle_ui_error

__all__ = [
    "UIRequest",
    "UIResponse", 
    "IUIComponent",
    "UIManager",
    "UIRenderer",
    "UIEventHandler",
    "BaseProblemInputComponent",
    "BaseReasoningDisplayComponent",
    "BaseResultDisplayComponent",
    "UIError",
    "UIErrorHandler",
    "handle_ui_error"
]

__version__ = "1.0.0"