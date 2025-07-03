"""
Simple Pattern-Based Model

This model uses the simple, regex-based pattern solver from the reasoning_engine.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..reasoning_engine.pattern_based_solver import PatternBasedSolver
from .base_model import BaseModel, ModelInput, ModelOutput


class SimplePatternModel(BaseModel):
    """
    A simple model that wraps the PatternBasedSolver.
    """
    def __init__(self, model_name: str = "simple_pattern_solver", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.solver = None
        self.model_type = "baseline"

    def initialize(self) -> bool:
        """
        Initializes the pattern-based solver.
        """
        self.logger.info(f"Initializing {self.model_name}...")
        try:
            # The patterns.json for this solver is in the reasoning_engine directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            patterns_path = os.path.join(current_dir, '..', 'reasoning_engine', 'patterns.json')
            
            if not os.path.exists(patterns_path):
                self.logger.error(f"Patterns file not found at {patterns_path}")
                self.is_initialized = False
                return False
            
            self.solver = PatternBasedSolver(patterns_path)
            self.is_initialized = True
            self.logger.info(f"{self.model_name} initialized successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.model_name}: {e}", exc_info=True)
            self.is_initialized = False
            return False

    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """
        使用基于模式的求解器处理数学问题
        
        Args:
            problem_input: 包含问题文本的输入对象
        
        Returns:
            ModelOutput: 包含答案和推理步骤的输出对象
        """
        start_time = time.time()
        
        try:
            # Call the pattern-based solver
            answer, reasoning_steps = self.solver.solve(problem_input.problem_text)
            
            # Convert result to ModelOutput format
            if answer is not None:
                predicted_answer = str(answer)
                final_reasoning_steps = reasoning_steps
                confidence = 0.8
            else:
                predicted_answer = ""
                final_reasoning_steps = reasoning_steps + ["Unable to solve this problem with current patterns"]
                confidence = 0.0
            
            return ModelOutput(
                answer=predicted_answer,
                reasoning_chain=final_reasoning_steps,
                confidence_score=confidence,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ModelOutput(
                answer="",
                reasoning_chain=[f"Error in pattern-based solving: {str(e)}"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """
        Solves a batch of problems.
        """
        return [self.solve_problem(p) for p in problems] 