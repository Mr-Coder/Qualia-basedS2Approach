#!/usr/bin/env python3
"""
GSM8K Dataset Performance Test
使用GSM8K标准数据集测试数学推理系统性能
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gsm8k_performance_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mathematical_reasoning_system import MathematicalReasoningSystem


class GSM8KPerformanceEvaluator:
    """GSM8K dataset performance evaluator for mathematical reasoning system."""
    
    def __init__(self, max_problems: int = 50):
        """Initialize the evaluator.
        
        Args:
            max_problems: Maximum number of problems to test (default 50 for quick testing)
        """
        self.system = MathematicalReasoningSystem()
        self.max_problems = max_problems
        self.test_data = self._load_gsm8k_data()
        
    def _load_gsm8k_data(self) -> List[Dict[str, Any]]:
        """Load GSM8K test data from JSONL file."""
        data = []
        gsm8k_file = os.path.join("Data", "GSM8K", "test.jsonl")
        
        try:
            with open(gsm8k_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= self.max_problems:
                        break
                    
                    item = json.loads(line.strip())
                    
                    # Extract the final answer from the answer field
                    answer_text = item.get('answer', '')
                    final_answer = self._extract_final_answer_from_solution(answer_text)
                    
                    if final_answer is not None:
                        data.append({
                            'question': item.get('question', ''),
                            'expected_answer': final_answer,
                            'solution': answer_text,
                            'problem_id': i + 1
                        })
                        
            logger.info(f"Successfully loaded {len(data)} problems from GSM8K dataset")
            return data
            
        except FileNotFoundError:
            logger.error(f"GSM8K file not found: {gsm8k_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading GSM8K data: {e}")
            return []
    
    def _extract_final_answer_from_solution(self, solution: str) -> Optional[float]:
        """Extract the final numerical answer from the GSM8K solution text.
        
        Args:
            solution: The solution text containing #### answer format
            
        Returns:
            The final numerical answer or None if not found
        """
        # GSM8K format: "#### 18" at the end
        pattern = r'####\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, solution)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                logger.warning(f"Could not parse answer: {match.group(1)}")
                return None
        
        # Fallback: look for last number in the solution
        numbers = re.findall(r'\d+(?:\.\d+)?', solution)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
                
        return None
    
    def _classify_problem_complexity(self, question: str) -> str:
        """Classify GSM8K problems by complexity."""
        question_lower = question.lower()
        
        # Count mathematical operations and concepts
        complexity_indicators = {
            'multi_step': ['then', 'after', 'next', 'finally', 'later'],
            'percentages': ['%', 'percent', 'percentage'],
            'fractions': ['half', 'third', 'quarter', 'fraction'],
            'time': ['hour', 'day', 'week', 'month', 'year'],
            'money': ['$', 'dollar', 'cost', 'price', 'pay'],
            'ratios': ['ratio', 'times as', 'times more', 'times less'],
            'rates': ['per', 'each', 'every'],
            'multiple_actors': ['and', 'both', 'total', 'altogether']
        }
        
        complexity_score = 0
        for category, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                complexity_score += 1
        
        # Simple classification based on complexity score
        if complexity_score <= 2:
            return "L1"  # Simple word problems
        elif complexity_score <= 4:
            return "L2"  # Multi-step problems 
        else:
            return "L3"  # Complex problems
    
    def run_gsm8k_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on GSM8K dataset."""
        logger.info(f"🚀 Starting GSM8K evaluation on {len(self.test_data)} problems...")
        
        results = {
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "GSM8K",
            "total_problems": len(self.test_data),
            "overall_performance": {},
            "complexity_breakdown": {},
            "detailed_results": []
        }
        
        correct_answers = 0
        total_processing_time = 0
        complexity_stats = {"L1": [], "L2": [], "L3": []}
        
        for i, problem in enumerate(self.test_data, 1):
            logger.info(f"📝 Problem {i}/{len(self.test_data)}: {problem['question'][:50]}...")
            
            start_time = time.time()
            try:
                # Solve the problem using our system
                result = self.system.solve_mathematical_problem(problem['question'])
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Extract computed answer
                computed_answer = self._safe_extract_answer(result)
                expected_answer = problem['expected_answer']
                
                # Check correctness
                is_correct = self._check_answer_correctness(computed_answer, expected_answer)
                
                if is_correct:
                    correct_answers += 1
                    logger.info(f"  ✅ CORRECT: Expected {expected_answer}, Got {computed_answer}")
                else:
                    logger.info(f"  ❌ INCORRECT: Expected {expected_answer}, Got {computed_answer}")
                
                # Classify problem complexity
                complexity = self._classify_problem_complexity(problem['question'])
                
                # Store detailed results
                detailed_result = {
                    "problem_id": problem['problem_id'],
                    "question": problem['question'],
                    "expected_answer": expected_answer,
                    "computed_answer": computed_answer,
                    "is_correct": is_correct,
                    "complexity": complexity,
                    "processing_time": processing_time,
                    "reasoning_steps": len(result.get("reasoning_steps", [])),
                    "entities_found": len(result.get("entities", [])),
                    "relations_found": len(result.get("relations", []))
                }
                
                results["detailed_results"].append(detailed_result)
                complexity_stats[complexity].append(is_correct)
                
            except Exception as e:
                logger.error(f"❌ Error solving problem {i}: {e}")
                detailed_result = {
                    "problem_id": problem['problem_id'],
                    "question": problem['question'],
                    "expected_answer": problem['expected_answer'],
                    "computed_answer": None,
                    "is_correct": False,
                    "complexity": "Unknown",
                    "processing_time": 0,
                    "error": str(e)
                }
                results["detailed_results"].append(detailed_result)
        
        # Calculate overall performance
        total_problems = len(self.test_data)
        accuracy = (correct_answers / total_problems) * 100
        avg_processing_time = total_processing_time / total_problems
        
        results["overall_performance"] = {
            "accuracy": accuracy,
            "correct_answers": correct_answers,
            "total_problems": total_problems,
            "average_processing_time": avg_processing_time,
            "total_processing_time": total_processing_time
        }
        
        # Calculate complexity breakdown
        for complexity, correct_list in complexity_stats.items():
            if correct_list:
                complexity_accuracy = (sum(correct_list) / len(correct_list)) * 100
                results["complexity_breakdown"][complexity] = {
                    "accuracy": complexity_accuracy,
                    "correct": sum(correct_list),
                    "total": len(correct_list),
                    "percentage_of_dataset": (len(correct_list) / total_problems) * 100
                }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"gsm8k_evaluation_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _safe_extract_answer(self, result: Dict[str, Any]) -> Optional[float]:
        """Safely extract numerical answer from system result."""
        if not result:
            return None
            
        # Try to get the final answer
        if 'final_answer' in result and result['final_answer'] is not None:
            try:
                return float(result['final_answer'])
            except (ValueError, TypeError):
                pass
        
        # Try reasoning steps
        reasoning_steps = result.get('reasoning_steps', [])
        if reasoning_steps:
            last_step = reasoning_steps[-1]
            if isinstance(last_step, dict) and 'output' in last_step:
                try:
                    return float(last_step['output'])
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def _check_answer_correctness(self, computed: Optional[float], expected: float, tolerance: float = 0.01) -> bool:
        """Check if computed answer matches expected with tolerance."""
        if computed is None:
            return False
        
        return abs(computed - expected) <= tolerance or abs(computed - expected) / expected <= tolerance
    
    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of evaluation results."""
        performance = results["overall_performance"]
        breakdown = results["complexity_breakdown"]
        
        print("\n" + "="*80)
        print("🎯 GSM8K DATASET EVALUATION SUMMARY")
        print("="*80)
        print(f"📊 Dataset: {results['dataset']}")
        print(f"🔢 Total Problems: {performance['total_problems']}")
        print(f"✅ Correct Answers: {performance['correct_answers']}")
        print(f"📈 Overall Accuracy: {performance['accuracy']:.1f}%")
        print(f"⏱️ Average Processing Time: {performance['average_processing_time']:.3f}s")
        print(f"⏰ Total Processing Time: {performance['total_processing_time']:.1f}s")
        
        print("\n🔍 COMPLEXITY BREAKDOWN:")
        for complexity, stats in breakdown.items():
            print(f"  {complexity}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']}) - {stats['percentage_of_dataset']:.1f}% of dataset")
        
        print("\n🏆 PERFORMANCE RATING:")
        accuracy = performance['accuracy']
        if accuracy >= 90:
            rating = "🌟 EXCELLENT"
        elif accuracy >= 75:
            rating = "🔥 GOOD"
        elif accuracy >= 60:
            rating = "⚡ FAIR"
        elif accuracy >= 40:
            rating = "⚠️ NEEDS IMPROVEMENT"
        else:
            rating = "❌ POOR"
        
        print(f"  {rating} ({accuracy:.1f}%)")
        print("="*80)

def main():
    """Main function to run GSM8K evaluation."""
    # You can adjust the number of problems to test
    evaluator = GSM8KPerformanceEvaluator(max_problems=50)  # Start with 50 problems
    
    if not evaluator.test_data:
        logger.error("No test data loaded. Please check GSM8K dataset file.")
        return
    
    results = evaluator.run_gsm8k_evaluation()
    
    print(f"\n🎉 GSM8K evaluation completed!")
    print(f"📊 Final Accuracy: {results['overall_performance']['accuracy']:.1f}%")

if __name__ == "__main__":
    main() 