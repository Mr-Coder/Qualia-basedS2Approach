#!/usr/bin/env python3
"""
Comprehensive GSM8K Testing for Fixed Reasoning System
Tests the improved mathematical reasoning system on a diverse set of problems
"""

import json
import logging
import time
from datetime import datetime

from fixed_reasoning_system import FixedMathematicalReasoningSystem


def load_gsm8k_problems(limit=30):
    """Load diverse GSM8K problems for comprehensive testing"""
    problems = []
    try:
        with open('Data/GSM8K/test.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Select diverse problems by index to get variety
        indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
                  32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]
        
        for i, idx in enumerate(indices[:limit]):
            if idx < len(lines):
                line = lines[idx]
                data = json.loads(line.strip())
                
                # Extract expected answer
                answer_text = data['answer']
                expected = float(answer_text.split('#### ')[-1])
                
                problems.append({
                    'id': f'gsm8k_{idx}',
                    'question': data['question'],
                    'expected_answer': expected,
                    'full_solution': answer_text
                })
                
        return problems
        
    except Exception as e:
        print(f"Error loading GSM8K problems: {e}")
        return []

def run_comprehensive_test():
    """Run comprehensive testing on GSM8K problems"""
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"comprehensive_gsm8k_test_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive GSM8K testing...")
    
    # Initialize system
    system = FixedMathematicalReasoningSystem()
    
    # Load test problems
    problems = load_gsm8k_problems(30)
    if not problems:
        logger.error("Failed to load GSM8K problems")
        return
    
    logger.info(f"Loaded {len(problems)} problems for testing")
    
    # Test results
    results = []
    correct_count = 0
    total_time = 0
    
    # Track performance by complexity/type
    complexity_stats = {
        'L1_simple': {'correct': 0, 'total': 0},
        'L2_compound': {'correct': 0, 'total': 0},
        'L3_complex': {'correct': 0, 'total': 0}
    }
    
    strategy_stats = {
        'fraction_calculation': {'correct': 0, 'total': 0},
        'compound_expression': {'correct': 0, 'total': 0},
        'time_reasoning': {'correct': 0, 'total': 0},
        'rate_calculation': {'correct': 0, 'total': 0},
        'multi_step': {'correct': 0, 'total': 0}
    }
    
    for i, problem in enumerate(problems, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Problem {i}/{len(problems)}: {problem['id']}")
        logger.info(f"Question: {problem['question']}")
        logger.info(f"Expected Answer: {problem['expected_answer']}")
        
        try:
            start_time = time.time()
            
            # Solve problem
            result = system.solve_mathematical_problem(problem['question'])
            
            end_time = time.time()
            processing_time = end_time - start_time
            total_time += processing_time
            
            # Extract answer
            predicted_answer = result.get('final_answer', 0.0)
            is_correct = abs(predicted_answer - problem['expected_answer']) < 0.01
            
            if is_correct:
                correct_count += 1
                logger.info(f"✅ CORRECT - Got {predicted_answer}")
            else:
                logger.info(f"❌ WRONG - Got {predicted_answer}, Expected {problem['expected_answer']}")
            
            # Track complexity
            complexity = result.get('complexity_level', 'unknown')
            if complexity in complexity_stats:
                complexity_stats[complexity]['total'] += 1
                if is_correct:
                    complexity_stats[complexity]['correct'] += 1
            
            # Track strategy
            strategy = result.get('strategy_used', 'unknown')
            if strategy in strategy_stats:
                strategy_stats[strategy]['total'] += 1
                if is_correct:
                    strategy_stats[strategy]['correct'] += 1
            
            # Log detailed results
            logger.info(f"Strategy Used: {strategy}")
            logger.info(f"Complexity Level: {complexity}")
            logger.info(f"Processing Time: {processing_time:.4f}s")
            logger.info(f"Verification Score: {result.get('verification_score', 0.0):.3f}")
            
            if result.get('reasoning_chain'):
                logger.info("Reasoning Chain:")
                for step in result['reasoning_chain']:
                    logger.info(f"  - {step}")
            
            # Store result
            results.append({
                'problem_id': problem['id'],
                'question': problem['question'],
                'expected_answer': problem['expected_answer'],
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'strategy_used': strategy,
                'complexity_level': complexity,
                'processing_time': processing_time,
                'verification_score': result.get('verification_score', 0.0),
                'reasoning_chain': result.get('reasoning_chain', []),
                'calculations': result.get('calculations', {}),
                'semantic_analysis': result.get('semantic_analysis', {})
            })
            
        except Exception as e:
            logger.error(f"Error processing problem {problem['id']}: {e}")
            results.append({
                'problem_id': problem['id'],
                'question': problem['question'],
                'expected_answer': problem['expected_answer'],
                'predicted_answer': 0.0,
                'is_correct': False,
                'error': str(e),
                'processing_time': 0.0
            })
    
    # Calculate final statistics
    accuracy = (correct_count / len(problems)) * 100
    avg_time = total_time / len(problems)
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE TEST RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Problems: {len(problems)}")
    logger.info(f"Correct Solutions: {correct_count}")
    logger.info(f"Overall Accuracy: {accuracy:.1f}%")
    logger.info(f"Average Processing Time: {avg_time:.4f}s")
    logger.info(f"Total Processing Time: {total_time:.2f}s")
    
    # Complexity breakdown
    logger.info(f"\n📊 COMPLEXITY ANALYSIS:")
    for complexity, stats in complexity_stats.items():
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            logger.info(f"  {complexity}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # Strategy breakdown
    logger.info(f"\n🎯 STRATEGY PERFORMANCE:")
    for strategy, stats in strategy_stats.items():
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            logger.info(f"  {strategy}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # Identify problem areas
    logger.info(f"\n🔍 PROBLEM ANALYSIS:")
    incorrect_problems = [r for r in results if not r['is_correct']]
    if incorrect_problems:
        logger.info(f"Failed Problems ({len(incorrect_problems)}):")
        for problem in incorrect_problems:
            logger.info(f"  - {problem['problem_id']}: Expected {problem['expected_answer']}, Got {problem['predicted_answer']}")
            if 'error' in problem:
                logger.info(f"    Error: {problem['error']}")
    
    # Performance insights
    logger.info(f"\n💡 PERFORMANCE INSIGHTS:")
    if accuracy >= 80:
        logger.info("🎉 EXCELLENT: System showing strong mathematical reasoning capabilities")
    elif accuracy >= 60:
        logger.info("👍 GOOD: System performing well with room for improvement")
    elif accuracy >= 40:
        logger.info("⚠️  MODERATE: System needs significant improvements")
    else:
        logger.info("❌ POOR: System requires major fixes")
    
    # Save detailed results
    results_filename = f"comprehensive_gsm8k_results_{timestamp}.json"
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': {
                'timestamp': timestamp,
                'total_problems': len(problems),
                'correct_count': correct_count,
                'accuracy': accuracy,
                'avg_processing_time': avg_time,
                'total_time': total_time
            },
            'complexity_stats': complexity_stats,
            'strategy_stats': strategy_stats,
            'detailed_results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 Detailed results saved to: {results_filename}")
    logger.info(f"📁 Test log saved to: {log_filename}")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting Comprehensive GSM8K Testing...")
    print("Testing fixed mathematical reasoning system on 30 diverse problems")
    
    results = run_comprehensive_test()
    
    print("\n✅ Testing complete! Check the log files for detailed results.") 