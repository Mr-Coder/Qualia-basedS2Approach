"""
Base Tool Class
==============

Abstract base class for all external tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    result: Any
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    """Abstract base class for tools"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_available = self._check_availability()
        
    @abstractmethod
    def _check_availability(self) -> bool:
        """Check if this tool is available/installed"""
        pass
        
    @abstractmethod
    def execute(self, operation: str, *args, **kwargs) -> ToolResult:
        """Execute a tool operation"""
        pass
        
    @abstractmethod
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations"""
        pass
        
    def can_execute(self, operation: str) -> bool:
        """Check if tool can execute the given operation"""
        return self.is_available and operation in self.get_supported_operations() 