#!/usr/bin/env python3
"""
Enhanced Hybrid Tutor System
Integrating Intelligent Tutor with COT-DIR Method

This system combines:
1. Intelligent Tutor (Chain of Responsibility + State Machine + Strategy + Observer)
2. COT-DIR Method (Chain of Thought + Directed Implicit Reasoning)
3. Pattern-based Solver (Fast, rule-based solving)
4. LLM Fallback (Intelligent reasoning when needed)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base_model import ModelInput, ModelOutput
from .hybrid_model import HybridConfig, HybridModel
from .intelligent_tutor import (IntelligentTutor, ProblemContext, StudentState,
                                TutorResponse)
from .proposed_model import COTDIRModel, ImplicitRelation, ReasoningStep


@dataclass
class EnhancedTutorResponse:
    """Enhanced response combining tutor and COT-DIR results"""
    message: str
    response_type: str
    confidence_level: float
    next_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # COT-DIR specific fields
    cotdir_answer: Optional[str] = None
    reasoning_chain: List[str] = field(default_factory=list)
    discovered_relations: List[ImplicitRelation] = field(default_factory=list)
    complexity_level: str = ""
    verification_score: float = 0.0


class EnhancedHybridTutor:
    """
    Enhanced Hybrid Tutor combining Intelligent Tutor with COT-DIR Method
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.intelligent_tutor = IntelligentTutor()
        self.cotdir_model = COTDIRModel(config.get("cotdir_config", {}))
        self.hybrid_model = HybridModel("enhanced_hybrid", config.get("hybrid_config", {}))
        
        # Configuration
        self.enable_cotdir = config.get("enable_cotdir", True) if config else True
        self.enable_intelligent_tutoring = config.get("enable_intelligent_tutoring", True) if config else True
        self.cotdir_confidence_threshold = config.get("cotdir_confidence_threshold", 0.7) if config else 0.7
        
        # Performance tracking
        self.total_problems = 0
        self.cotdir_used_count = 0
        self.tutor_used_count = 0
        self.hybrid_used_count = 0
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            # Initialize COT-DIR model
            if self.enable_cotdir:
                success = self.cotdir_model.initialize()
                if not success:
                    self.logger.warning("COT-DIR model initialization failed")
                    self.enable_cotdir = False
            
            # Initialize hybrid model
            success = self.hybrid_model.initialize()
            if not success:
                self.logger.warning("Hybrid model initialization failed")
            
            self.logger.info("Enhanced Hybrid Tutor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Hybrid Tutor: {e}")
    
    def solve_problem(self, student_id: str, problem_text: str, student_answer: str = "") -> EnhancedTutorResponse:
        """
        Solve problem using enhanced hybrid approach
        
        Args:
            student_id: Student identifier
            problem_text: Problem text
            student_answer: Student's answer (optional)
            
        Returns:
            EnhancedTutorResponse with combined results
        """
        start_time = time.time()
        self.total_problems += 1
        
        try:
            # Step 1: Create problem context
            problem_context = self._create_problem_context(problem_text)
            
            # Step 2: Analyze problem complexity and choose approach
            approach_decision = self._choose_solving_approach(problem_context, student_id)
            
            # Step 3: Execute chosen approach
            if approach_decision["use_cotdir"]:
                return self._solve_with_cotdir(student_id, problem_context, student_answer)
            elif approach_decision["use_intelligent_tutor"]:
                return self._solve_with_intelligent_tutor(student_id, problem_context, student_answer)
            else:
                return self._solve_with_hybrid(student_id, problem_context, student_answer)
                
        except Exception as e:
            self.logger.error(f"Error in enhanced solving: {e}")
            return EnhancedTutorResponse(
                message=f"Error in solving: {str(e)}",
                response_type="error",
                confidence_level=0.0,
                metadata={"error": str(e)}
            )
    
    def _create_problem_context(self, problem_text: str) -> ProblemContext:
        """Create problem context from text"""
        # Extract basic information
        numbers = self._extract_numbers(problem_text)
        expected_answer = str(sum(numbers)) if numbers else "0"  # Simple fallback
        
        # Determine concept tags based on problem content
        concept_tags = self._identify_concepts(problem_text)
        
        # Determine difficulty level
        difficulty_level = self._assess_difficulty(problem_text, concept_tags)
        
        return ProblemContext(
            problem_text=problem_text,
            problem_id=f"enhanced_{int(time.time())}",
            difficulty_level=difficulty_level,
            concept_tags=concept_tags,
            expected_answer=expected_answer,
            solution_steps=[],  # Will be filled by solving approach
            hints_available=[],
            similar_problems=[]
        )
    
    def _choose_solving_approach(self, problem_context: ProblemContext, student_id: str) -> Dict[str, bool]:
        """Choose the best solving approach based on problem and student state"""
        
        # Get student state
        student_context = self.intelligent_tutor.get_or_create_student_context(student_id)
        student_state = student_context.student_state
        
        # Decision factors
        problem_complexity = problem_context.difficulty_level
        student_level = student_state.current_level
        student_accuracy = student_state.accuracy_rate
        student_frustration = student_state.frustration_level
        
        # Decision logic
        use_cotdir = False
        use_intelligent_tutor = False
        use_hybrid = False
        
        # Use COT-DIR for complex problems or advanced students
        if (problem_complexity >= 4 or 
            (student_level >= 3 and student_accuracy >= 0.7) or
            len(problem_context.concept_tags) >= 3):
            use_cotdir = True
            self.cotdir_used_count += 1
        
        # Use intelligent tutor for struggling students or simple problems
        elif (student_frustration > 0.5 or 
              student_accuracy < 0.4 or 
              problem_complexity <= 2):
            use_intelligent_tutor = True
            self.tutor_used_count += 1
        
        # Use hybrid approach as default
        else:
            use_hybrid = True
            self.hybrid_used_count += 1
        
        return {
            "use_cotdir": use_cotdir,
            "use_intelligent_tutor": use_intelligent_tutor,
            "use_hybrid": use_hybrid
        }
    
    def _solve_with_cotdir(self, student_id: str, problem_context: ProblemContext, student_answer: str) -> EnhancedTutorResponse:
        """Solve using COT-DIR method"""
        
        # Create model input
        model_input = ModelInput(
            problem_text=problem_context.problem_text,
            problem_id=problem_context.problem_id
        )
        
        # Solve with COT-DIR
        cotdir_result = self.cotdir_model.solve_problem(model_input)
        
        # Create enhanced response
        message = self._format_cotdir_response(cotdir_result, student_answer)
        
        return EnhancedTutorResponse(
            message=message,
            response_type="cotdir_solution",
            confidence_level=cotdir_result.confidence_score,
            next_action="explain_reasoning",
            cotdir_answer=cotdir_result.answer,
            reasoning_chain=cotdir_result.reasoning_chain,
            discovered_relations=cotdir_result.metadata.get("discovered_relations", []),
            complexity_level=cotdir_result.metadata.get("complexity", "unknown"),
            verification_score=cotdir_result.metadata.get("verification_score", 0.0),
            metadata={
                "solver_type": "cotdir",
                "relations_count": len(cotdir_result.metadata.get("discovered_relations", [])),
                "reasoning_steps": len(cotdir_result.metadata.get("intermediate_steps", [])),
                "processing_time": cotdir_result.processing_time
            }
        )
    
    def _solve_with_intelligent_tutor(self, student_id: str, problem_context: ProblemContext, student_answer: str) -> EnhancedTutorResponse:
        """Solve using intelligent tutor"""
        
        # Get tutor response
        tutor_response = self.intelligent_tutor.solve_problem(student_id, problem_context, student_answer)
        
        # Create enhanced response
        return EnhancedTutorResponse(
            message=tutor_response.message,
            response_type=tutor_response.response_type,
            confidence_level=tutor_response.confidence_level,
            next_action=tutor_response.next_action,
            metadata={
                "solver_type": "intelligent_tutor",
                "student_state": tutor_response.metadata.get("state_type", "unknown"),
                "student_level": tutor_response.metadata.get("student_level", 1),
                "accuracy_rate": tutor_response.metadata.get("accuracy_rate", 0.0),
                "frustration_level": tutor_response.metadata.get("frustration_level", 0.0)
            }
        )
    
    def _solve_with_hybrid(self, student_id: str, problem_context: ProblemContext, student_answer: str) -> EnhancedTutorResponse:
        """Solve using hybrid approach"""
        
        # Create model input
        model_input = ModelInput(
            problem_text=problem_context.problem_text,
            problem_id=problem_context.problem_id
        )
        
        # Solve with hybrid model
        hybrid_result = self.hybrid_model.solve_problem(model_input)
        
        # Create enhanced response
        message = self._format_hybrid_response(hybrid_result, student_answer)
        
        return EnhancedTutorResponse(
            message=message,
            response_type="hybrid_solution",
            confidence_level=hybrid_result.confidence_score,
            next_action="continue_learning",
            metadata={
                "solver_type": "hybrid",
                "pattern_confidence": hybrid_result.metadata.get("pattern_confidence", 0.0),
                "llm_fallback_used": hybrid_result.metadata.get("llm_fallback_used", False),
                "processing_time": hybrid_result.processing_time
            }
        )
    
    def _format_cotdir_response(self, cotdir_result: ModelOutput, student_answer: str) -> str:
        """Format COT-DIR response for student"""
        
        message_parts = []
        
        # Add encouragement if student provided answer
        if student_answer:
            if student_answer == cotdir_result.answer:
                message_parts.append("🎉 太棒了！你的答案是正确的！")
            else:
                message_parts.append("💪 让我们用COT-DIR方法来分析这个问题。")
        
        # Add COT-DIR explanation
        message_parts.append("🧠 COT-DIR推理过程：")
        
        # Add key reasoning steps
        key_steps = [step for step in cotdir_result.reasoning_chain if ":" in step][:3]
        for step in key_steps:
            message_parts.append(f"  • {step}")
        
        # Add final answer
        message_parts.append(f"\n🎯 最终答案：{cotdir_result.answer}")
        
        # Add confidence information
        if cotdir_result.confidence_score >= 0.8:
            message_parts.append("✅ 推理置信度很高")
        elif cotdir_result.confidence_score >= 0.6:
            message_parts.append("⚠️ 推理置信度中等，建议仔细检查")
        else:
            message_parts.append("❓ 推理置信度较低，可能需要其他方法")
        
        return "\n".join(message_parts)
    
    def _format_hybrid_response(self, hybrid_result: ModelOutput, student_answer: str) -> str:
        """Format hybrid response for student"""
        
        message_parts = []
        
        # Add encouragement if student provided answer
        if student_answer:
            if student_answer == hybrid_result.answer:
                message_parts.append("🎉 很好！你的答案是正确的！")
            else:
                message_parts.append("💪 让我们用混合方法来分析这个问题。")
        
        # Add hybrid explanation
        solver_type = hybrid_result.metadata.get("solver_type", "unknown")
        if solver_type == "pattern":
            message_parts.append("🔍 使用模式匹配方法：")
        elif solver_type == "llm_fallback":
            message_parts.append("🤖 使用智能推理方法：")
        else:
            message_parts.append("🔄 使用混合方法：")
        
        # Add reasoning chain
        for step in hybrid_result.reasoning_chain[:3]:
            message_parts.append(f"  • {step}")
        
        # Add final answer
        message_parts.append(f"\n🎯 最终答案：{hybrid_result.answer}")
        
        return "\n".join(message_parts)
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(n) for n in numbers]
    
    def _identify_concepts(self, text: str) -> List[str]:
        """Identify mathematical concepts in text"""
        concepts = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["add", "sum", "total", "altogether"]):
            concepts.append("addition")
        if any(word in text_lower for word in ["subtract", "difference", "left", "remaining"]):
            concepts.append("subtraction")
        if any(word in text_lower for word in ["multiply", "product", "times"]):
            concepts.append("multiplication")
        if any(word in text_lower for word in ["divide", "quotient", "each", "per"]):
            concepts.append("division")
        if any(word in text_lower for word in ["percent", "%"]):
            concepts.append("percentages")
        if any(word in text_lower for word in ["fraction", "half", "third"]):
            concepts.append("fractions")
        
        return concepts if concepts else ["general_math"]
    
    def _assess_difficulty(self, text: str, concepts: List[str]) -> int:
        """Assess problem difficulty level (1-5)"""
        difficulty = 1
        
        # Base difficulty from concepts
        concept_difficulty = {
            "addition": 1, "subtraction": 1, "multiplication": 2,
            "division": 2, "percentages": 3, "fractions": 3
        }
        
        for concept in concepts:
            difficulty = max(difficulty, concept_difficulty.get(concept, 1))
        
        # Adjust based on text complexity
        words = text.split()
        if len(words) > 30:
            difficulty += 1
        if len(self._extract_numbers(text)) > 4:
            difficulty += 1
        
        return min(difficulty, 5)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.total_problems == 0:
            return {"total_problems": 0}
        
        return {
            "total_problems": self.total_problems,
            "cotdir_usage_rate": self.cotdir_used_count / self.total_problems,
            "tutor_usage_rate": self.tutor_used_count / self.total_problems,
            "hybrid_usage_rate": self.hybrid_used_count / self.total_problems,
            "cotdir_used_count": self.cotdir_used_count,
            "tutor_used_count": self.tutor_used_count,
            "hybrid_used_count": self.hybrid_used_count
        }
    
    def get_student_progress(self, student_id: str) -> Dict[str, Any]:
        """Get student progress from intelligent tutor"""
        return self.intelligent_tutor.get_student_progress(student_id)


# ==================== 使用示例 ====================

def create_enhanced_demo():
    """Create enhanced hybrid tutor demo"""
    
    # Configuration
    config = {
        "enable_cotdir": True,
        "enable_intelligent_tutoring": True,
        "cotdir_confidence_threshold": 0.7,
        "cotdir_config": {
            "enable_ird": True,
            "enable_mlr": True,
            "enable_cv": True
        },
        "hybrid_config": {
            "pattern_confidence_threshold": 0.5,
            "enable_llm_fallback": True
        }
    }
    
    # Create enhanced tutor
    tutor = EnhancedHybridTutor(config)
    
    # Test problems
    test_problems = [
        {
            "text": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
            "expected": "8",
            "type": "simple_addition"
        },
        {
            "text": "一个复杂的数学问题，涉及多个变量和关系，需要深入推理和分析。",
            "expected": "unknown",
            "type": "complex_reasoning"
        },
        {
            "text": "小华有10个糖果，他给了小明3个，还剩多少个？",
            "expected": "7",
            "type": "subtraction"
        }
    ]
    
    student_id = "demo_student"
    
    print("🚀 Enhanced Hybrid Tutor Demo")
    print("=" * 60)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n📝 问题 {i}: {problem['text']}")
        print(f"   类型: {problem['type']}")
        
        # Solve with enhanced tutor
        response = tutor.solve_problem(student_id, problem['text'])
        
        print(f"   方法: {response.metadata.get('solver_type', 'unknown')}")
        print(f"   置信度: {response.confidence_level:.2f}")
        print(f"   响应类型: {response.response_type}")
        print(f"   答案: {response.cotdir_answer or 'N/A'}")
        print(f"   消息: {response.message[:100]}...")
    
    # Show performance stats
    print(f"\n📊 性能统计:")
    stats = tutor.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2%}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Enhanced Hybrid Tutor demo completed!")


if __name__ == "__main__":
    create_enhanced_demo() 