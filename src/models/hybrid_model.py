#!/usr/bin/env python3
"""
Hybrid Model Implementation

This module implements a hybrid approach that combines pattern-based solving
with LLM-based reasoning for mathematical word problems.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base_model import BaseModel, ModelInput, ModelOutput
from .simple_pattern_model import SimplePatternModel


@dataclass
class HybridConfig:
    """Configuration for HybridModel."""
    pattern_confidence_threshold: float = 0.5
    enable_llm_fallback: bool = True
    llm_model_name: str = "gpt-3.5-turbo"
    max_llm_tokens: int = 2048
    llm_temperature: float = 0.1
    enable_debug_logging: bool = False


class HybridModel(BaseModel):
    """
    Hybrid model that combines pattern-based solving with LLM reasoning.
    
    Strategy:
    1. First, try pattern-based solver (fast, deterministic)
    2. If pattern solver fails or has low confidence, fallback to LLM
    3. Combine results and provide unified output
    """
    
    def __init__(self, model_name: str = "hybrid_solver", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid model.
        
        Args:
            model_name: Name identifier for the model
            config: Configuration parameters
        """
        super().__init__(model_name, config or {})
        
        # Filter config to only include HybridConfig fields
        hybrid_config_dict = {}
        if config:
            # Only include fields that HybridConfig expects
            expected_fields = {
                'pattern_confidence_threshold', 'enable_llm_fallback', 
                'llm_model_name', 'max_llm_tokens', 'llm_temperature', 
                'enable_debug_logging'
            }
            hybrid_config_dict = {k: v for k, v in config.items() if k in expected_fields}
        
        self.hybrid_config = HybridConfig(**hybrid_config_dict)
        self.pattern_model = None
        self.llm_model = None
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
        # Performance tracking
        self.pattern_success_count = 0
        self.llm_fallback_count = 0
        self.total_problems = 0
        
    def initialize(self) -> bool:
        """
        Initialize both pattern model and LLM model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize pattern model
            self.pattern_model = SimplePatternModel("hybrid_pattern_solver")
            pattern_success = self.pattern_model.initialize()
            
            if not pattern_success:
                self.logger.error("Failed to initialize pattern model")
                return False
                
            # Initialize LLM model (placeholder for now)
            # In a real implementation, this would initialize GPT/Claude/etc.
            self.llm_model = self._create_llm_model()
            llm_success = self.llm_model is not None
            
            self.is_initialized = pattern_success and llm_success
            
            if self.is_initialized:
                self.logger.info(f"Hybrid model initialized successfully. "
                               f"Pattern model: {pattern_success}, LLM model: {llm_success}")
            else:
                self.logger.error("Failed to initialize hybrid model")
                
            return self.is_initialized
            
        except Exception as e:
            self.logger.error(f"Error initializing hybrid model: {e}")
            return False
    
    def _create_llm_model(self):
        """
        Create LLM model instance.
        
        Returns:
            LLM model instance or None if not available
        """
        # Placeholder implementation
        # In a real system, this would create GPT/Claude/etc. client
        try:
            # For now, return a mock LLM model
            return MockLLMModel("hybrid_llm_solver", self.hybrid_config)
        except Exception as e:
            self.logger.warning(f"LLM model not available: {e}")
            return None
    
    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """
        Solve problem using hybrid approach.
        
        Args:
            problem_input: Input problem data
            
        Returns:
            ModelOutput: Solution with reasoning chain and metadata
        """
        start_time = time.time()
        self.total_problems += 1
        
        if not self.is_initialized:
            return ModelOutput(
                answer="",
                reasoning_chain=["Hybrid model not initialized"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message="Model not initialized"
            )
        
        try:
            # Step 1: Try pattern-based solving
            pattern_output = self.pattern_model.solve_problem(problem_input)
            pattern_processing_time = time.time() - start_time
            
            # Check if pattern solver succeeded
            if (pattern_output.answer and 
                pattern_output.confidence_score >= self.hybrid_config.pattern_confidence_threshold):
                
                self.pattern_success_count += 1
                self.logger.debug(f"Pattern solver succeeded with confidence {pattern_output.confidence_score}")
                
                return ModelOutput(
                    answer=pattern_output.answer,
                    reasoning_chain=pattern_output.reasoning_chain + ["[Pattern-based solution]"],
                    confidence_score=pattern_output.confidence_score,
                    processing_time=pattern_processing_time,
                    metadata={
                        "solver_type": "pattern",
                        "pattern_confidence": pattern_output.confidence_score,
                        "llm_fallback_used": False
                    }
                )
            
            # Step 2: Fallback to LLM if pattern solver failed or has low confidence
            if self.hybrid_config.enable_llm_fallback and self.llm_model:
                self.llm_fallback_count += 1
                self.logger.debug(f"Pattern solver failed (confidence: {pattern_output.confidence_score}), "
                                f"falling back to LLM")
                
                llm_output = self.llm_model.solve_problem(problem_input)
                llm_processing_time = time.time() - start_time
                
                # Combine reasoning chains
                combined_reasoning = [
                    "[Pattern solver attempt]",
                    *pattern_output.reasoning_chain,
                    "[LLM fallback]",
                    *llm_output.reasoning_chain
                ]
                
                return ModelOutput(
                    answer=llm_output.answer,
                    reasoning_chain=combined_reasoning,
                    confidence_score=llm_output.confidence_score,
                    processing_time=llm_processing_time,
                    metadata={
                        "solver_type": "llm_fallback",
                        "pattern_confidence": pattern_output.confidence_score,
                        "llm_confidence": llm_output.confidence_score,
                        "llm_fallback_used": True
                    }
                )
            
            # Step 3: If no LLM fallback available, return pattern result
            self.logger.warning("Pattern solver failed and no LLM fallback available")
            return ModelOutput(
                answer=pattern_output.answer or "",
                reasoning_chain=pattern_output.reasoning_chain + ["[No LLM fallback available]"],
                confidence_score=pattern_output.confidence_score,
                processing_time=pattern_processing_time,
                metadata={
                    "solver_type": "pattern_only",
                    "pattern_confidence": pattern_output.confidence_score,
                    "llm_fallback_used": False
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in hybrid solving: {e}")
            return ModelOutput(
                answer="",
                reasoning_chain=[f"Error in hybrid solving: {str(e)}"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """
        Solve multiple problems using hybrid approach.
        
        Args:
            problems: List of input problems
            
        Returns:
            List[ModelOutput]: List of solutions
        """
        return [self.solve_problem(problem) for problem in problems]
    
    def get_hybrid_stats(self) -> Dict[str, Any]:
        """
        Get hybrid model performance statistics.
        
        Returns:
            Dict containing performance metrics
        """
        if self.total_problems == 0:
            return {
                "total_problems": 0,
                "pattern_success_rate": 0.0,
                "llm_fallback_rate": 0.0
            }
        
        return {
            "total_problems": self.total_problems,
            "pattern_success_count": self.pattern_success_count,
            "llm_fallback_count": self.llm_fallback_count,
            "pattern_success_rate": self.pattern_success_count / self.total_problems,
            "llm_fallback_rate": self.llm_fallback_count / self.total_problems,
            "pattern_confidence_threshold": self.hybrid_config.pattern_confidence_threshold,
            "enable_llm_fallback": self.hybrid_config.enable_llm_fallback
        }
    
    def update_confidence_threshold(self, new_threshold: float):
        """
        Update the confidence threshold for pattern solver.
        
        Args:
            new_threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.hybrid_config.pattern_confidence_threshold = new_threshold
            self.logger.info(f"Updated confidence threshold to {new_threshold}")
        else:
            self.logger.error(f"Invalid confidence threshold: {new_threshold}")


class MockLLMModel(BaseModel):
    """
    Mock LLM model for testing purposes.
    In a real implementation, this would be replaced with actual LLM clients.
    """
    
    def __init__(self, model_name: str, config: HybridConfig):
        super().__init__(model_name, config.__dict__)
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
    def initialize(self) -> bool:
        """Mock initialization."""
        self.is_initialized = True
        return True
    
    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """
        Mock LLM solving - returns a placeholder response.
        
        In a real implementation, this would:
        1. Generate a prompt for the LLM
        2. Call the LLM API
        3. Parse the response
        4. Extract the answer and reasoning
        """
        start_time = time.time()
        
        # Mock response - in reality this would be the LLM's actual response
        mock_answer = "42"  # Placeholder answer
        mock_reasoning = [
            f"[LLM Analysis] Problem: {problem_input.problem_text[:50]}...",
            "[LLM] This appears to be a mathematical word problem.",
            "[LLM] Let me break it down step by step...",
            "[LLM] Based on my analysis, the answer is 42.",
            "[Note: This is a mock LLM response for testing]"
        ]
        
        return ModelOutput(
            answer=mock_answer,
            reasoning_chain=mock_reasoning,
            confidence_score=0.7,  # Mock confidence
            processing_time=time.time() - start_time,
            metadata={
                "llm_model": self.config.llm_model_name,
                "mock_response": True
            }
        )
    
    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """Mock batch solving."""
        return [self.solve_problem(problem) for problem in problems] 