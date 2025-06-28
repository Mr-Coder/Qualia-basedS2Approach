"""
Base Reasoning Strategy
======================

Abstract base class for all reasoning strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningStep:
    """Single step in a reasoning process"""
    step_id: int
    operation: str
    explanation: str
    input_data: Any
    output_data: Any
    confidence: float
    metadata: Dict[str, Any]


@dataclass 
class ReasoningResult:
    """Result of a reasoning process"""
    final_answer: Any
    reasoning_steps: List[ReasoningStep]
    confidence: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
    @abstractmethod
    def can_handle(self, problem: Any) -> bool:
        """Check if this strategy can handle the given problem"""
        pass
        
    @abstractmethod
    def solve(self, problem: Any) -> ReasoningResult:
        """Solve the problem using this strategy"""
        pass
        
    @abstractmethod
    def validate_step(self, step: ReasoningStep) -> bool:
        """Validate a single reasoning step"""
        pass
        
    def get_priority(self) -> int:
        """Get strategy priority (higher number = higher priority)"""
        return self.config.get('priority', 0) 