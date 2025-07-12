"""
COT-DIR-like workflow orchestrator.
This module defines the main workflow for processing problems.
"""
import logging
import os
from typing import Any, Dict, List, Tuple

from ..config.advanced_config import AdvancedConfiguration
from .pattern_based_solver import PatternBasedSolver


class COTDIRIntegratedWorkflow:
    """
    Orchestrates the problem-solving process using a pattern-based solver.
    """
    def __init__(self, config: AdvancedConfiguration):
        """
        Initializes the workflow.

        Args:
            config: An AdvancedConfiguration object containing configuration settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        patterns_path = os.path.join(current_dir, 'patterns.json')
        
        self.logger.info(f"Loading patterns from: {patterns_path}")
        if not os.path.exists(patterns_path):
            self.logger.error(f"Patterns file not found at {patterns_path}")
            raise FileNotFoundError(f"Patterns file not found at {patterns_path}")
            
        self.solver = PatternBasedSolver(patterns_path)

    def process(self, problem_data: Dict) -> Dict:
        """
        Processes a given problem.

        Args:
            problem_data: A dictionary containing the problem text.

        Returns:
            A dictionary containing the answer and reasoning steps.
        """
        problem_text = problem_data.get("problem", "")
        if not problem_text:
            self.logger.warning("No problem text provided.")
            return {"problem": "", "answer": "Error: No problem text.", "reasoning_steps": []}

        self.logger.info(f"Processing problem: {problem_text}")

        try:
            solution = self.solver.solve(problem_text)
            final_answer = solution.get("answer")
            reasoning_steps = solution.get("reasoning_steps", [])
            
            self.logger.info(f"Final answer: {final_answer}")

            return {
                "problem": problem_text,
                "answer": final_answer,
                "reasoning_steps": reasoning_steps
            }
        except Exception as e:
            self.logger.exception(f"An error occurred during processing: {e}")
            return {
                "problem": problem_text,
                "answer": f"Error: {e}",
                "reasoning_steps": []
            }
