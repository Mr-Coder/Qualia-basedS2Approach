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
        Solves a single problem using the PatternBasedSolver.
        """
        start_time = time.time()
        if not self.is_initialized or not self.solver:
            return ModelOutput(
                answer="",
                reasoning_chain=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message="Model is not initialized."
            )

        solution = self.solver.solve(problem_input.problem_text)
        
        final_answer = solution.get("answer")
        if final_answer is not None:
            answer_str = str(final_answer)
            confidence = 1.0 # It's a deterministic model
        else:
            answer_str = ""
            confidence = 0.0

        return ModelOutput(
            answer=answer_str,
            reasoning_chain=solution.get("reasoning_steps", []),
            confidence_score=confidence,
            processing_time=time.time() - start_time
        )

    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """
        Solves a batch of problems.
        """
        return [self.solve_problem(p) for p in problems] 