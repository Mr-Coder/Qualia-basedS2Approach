"""
Improved vs Robust System Comparison Test
========================================

比较改进的推理系统与鲁棒系统在以下方面的性能：
1. 语义理解准确性
2. 推理逻辑连贯性  
3. 歧义解决能力
4. 多步推理准确性
5. 答案准确性

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from improved_reasoning_system import ImprovedMathematicalReasoningSystem
from robust_reasoning_system import RobustMathematicalReasoningSystem

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_vs_robust_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemComparisonTester:
    """系统比较测试器"""
    
    def __init__(self):
        self.improved_system = ImprovedMathematicalReasoningSystem()
        self.robust_system = RobustMathematicalReasoningSystem()
        
    def get_test_problems(self) -> List[Dict[str, Any]]:
        """获取测试问题"""
        return [
            {
                'id': 0,
                'question': "John has 25 apples. He gives 8 apples to Mary and then buys 12 more apples. How many apples does John have now?",
                'expected_numeric': 29,
                'focus': 'multi_step_sequential'
            },
            {
                'id': 1,
                'question': "A store sells notebooks for $3 each. If Sarah buys 4 notebooks and pays with a $20 bill, how much change does she receive?",
                'expected_numeric': 8,
                'focus': 'calculation_with_change'
            },
            {
                'id': 2,
                'question': "There are 15 students in a class. 3 students are absent today. If the remaining students are divided into groups of 4, how many complete groups can be formed?",
                'expected_numeric': 3,
                'focus': 'division_with_remainder'
            },
            {
                'id': 3,
                'question': "Mike saves $5 every week. After 6 weeks, he spends $18 on a book. How much money does he have left?",
                'expected_numeric': 12,
                'focus': 'savings_and_spending'
            },
            {
                'id': 4,
                'question': "A recipe calls for 2 cups of flour. If Lisa wants to make 3 times the recipe, how many cups of flour does she need in total?",
                'expected_numeric': 6,
                'focus': 'scaling_recipe'
            }
        ]
    
    def compare_systems_on_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """比较两个系统在单个问题上的表现"""
        
        # 测试改进系统
        improved_start = time.time()
        try:
            improved_result = self.improved_system.solve_mathematical_problem(problem['question'])
            improved_time = time.time() - improved_start
            improved_success = True
        except Exception as e:
            logger.error(f"Improved system failed on problem {problem['id']}: {e}")
            improved_result = {'error': str(e)}
            improved_time = time.time() - improved_start
            improved_success = False
        
        # 测试鲁棒系统
        robust_start = time.time()
        try:
            robust_result = self.robust_system.solve_mathematical_problem(problem['question'])
            robust_time = time.time() - robust_start
            robust_success = True
        except Exception as e:
            logger.error(f"Robust system failed on problem {problem['id']}: {e}")
            robust_result = {'error': str(e)}
            robust_time = time.time() - robust_start
            robust_success = False
        
        # 分析比较结果
        comparison = self._analyze_comparison(
            problem, improved_result, robust_result, improved_success, robust_success
        )
        
        return {
            'problem_id': problem['id'],
            'question': problem['question'],
            'expected_answer': problem['expected_numeric'],
            'improved_system': {
                'result': improved_result,
                'processing_time': improved_time,
                'success': improved_success
            },
            'robust_system': {
                'result': robust_result,
                'processing_time': robust_time,
                'success': robust_success
            },
            'comparison': comparison
        }
    
    def _analyze_comparison(self, problem: Dict[str, Any], 
                          improved_result: Dict[str, Any], 
                          robust_result: Dict[str, Any],
                          improved_success: bool,
                          robust_success: bool) -> Dict[str, Any]:
        """分析比较结果"""
        
        comparison = {
            'accuracy_comparison': {},
            'semantic_analysis_comparison': {},
            'reasoning_quality_comparison': {},
            'winner': 'tie'
        }
        
        expected = problem['expected_numeric']
        
        # 准确性比较
        if improved_success and robust_success:
            improved_answer = improved_result.get('final_answer', 0)
            robust_answer = robust_result.get('final_answer', 0)
            
            improved_correct = abs(improved_answer - expected) < 0.01 if isinstance(improved_answer, (int, float)) else False
            robust_correct = abs(robust_answer - expected) < 0.01 if isinstance(robust_answer, (int, float)) else False
            
            comparison['accuracy_comparison'] = {
                'improved_correct': improved_correct,
                'robust_correct': robust_correct,
                'improved_answer': improved_answer,
                'robust_answer': robust_answer,
                'expected_answer': expected
            }
            
            # 语义分析比较
            improved_semantic = improved_result.get('semantic_context', {})
            
            comparison['semantic_analysis_comparison'] = {
                'improved_has_intent_analysis': bool(improved_semantic.get('problem_intent')),
                'improved_has_ambiguity_resolution': bool(improved_semantic.get('ambiguity_resolution')),
                'improved_has_implicit_operations': bool(improved_semantic.get('implicit_operations')),
                'improved_reasoning_steps_count': len(improved_result.get('reasoning_steps', [])),
                'robust_reasoning_steps_count': len(robust_result.get('reasoning_steps', []))
            }
            
            # 推理质量比较
            improved_confidence = improved_result.get('confidence', 0)
            improved_verification = improved_result.get('verification_summary', {})
            
            comparison['reasoning_quality_comparison'] = {
                'improved_confidence': improved_confidence,
                'improved_verification_passed': improved_verification.get('verification_passed', False),
                'improved_has_semantic_justification': any(
                    step.get('semantic_justification') for step in improved_result.get('reasoning_steps', [])
                )
            }
            
            # 确定获胜者
            if improved_correct and not robust_correct:
                comparison['winner'] = 'improved'
            elif robust_correct and not improved_correct:
                comparison['winner'] = 'robust'
            elif improved_correct and robust_correct:
                # 两者都正确，比较其他因素
                if improved_confidence > 0.8 and improved_semantic.get('problem_intent'):
                    comparison['winner'] = 'improved'
                else:
                    comparison['winner'] = 'tie'
            else:
                comparison['winner'] = 'both_failed'
        
        elif improved_success and not robust_success:
            comparison['winner'] = 'improved'
        elif robust_success and not improved_success:
            comparison['winner'] = 'robust'
        else:
            comparison['winner'] = 'both_failed'
        
        return comparison
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """运行综合比较测试"""
        
        logger.info("Starting comprehensive comparison test")
        
        # 获取测试问题
        problems = self.get_test_problems()
        
        # 比较每个问题
        comparison_results = []
        for problem in problems:
            logger.info(f"Testing problem {problem['id']}: {problem['question'][:50]}...")
            result = self.compare_systems_on_problem(problem)
            comparison_results.append(result)
        
        # 计算总体统计
        overall_stats = self._calculate_overall_statistics(comparison_results)
        
        # 生成比较报告
        comparison_report = self._generate_comparison_report(comparison_results, overall_stats)
        
        return {
            'comparison_results': comparison_results,
            'overall_statistics': overall_stats,
            'comparison_report': comparison_report
        }
    
    def _calculate_overall_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算总体统计"""
        
        total_problems = len(results)
        improved_wins = sum(1 for r in results if r['comparison']['winner'] == 'improved')
        robust_wins = sum(1 for r in results if r['comparison']['winner'] == 'robust')
        ties = sum(1 for r in results if r['comparison']['winner'] == 'tie')
        
        # 准确性统计
        improved_correct = sum(1 for r in results 
                             if r['comparison']['accuracy_comparison'].get('improved_correct', False))
        robust_correct = sum(1 for r in results 
                           if r['comparison']['accuracy_comparison'].get('robust_correct', False))
        
        # 语义分析统计
        problems_with_intent = sum(1 for r in results 
                                 if r['comparison']['semantic_analysis_comparison'].get('improved_has_intent_analysis', False))
        problems_with_ambiguity_resolution = sum(1 for r in results 
                                               if r['comparison']['semantic_analysis_comparison'].get('improved_has_ambiguity_resolution', False))
        
        return {
            'total_problems': total_problems,
            'winner_statistics': {
                'improved_wins': improved_wins,
                'robust_wins': robust_wins,
                'ties': ties,
                'improved_win_rate': improved_wins / total_problems,
                'robust_win_rate': robust_wins / total_problems
            },
            'accuracy_statistics': {
                'improved_accuracy': improved_correct / total_problems,
                'robust_accuracy': robust_correct / total_problems,
                'improved_correct_count': improved_correct,
                'robust_correct_count': robust_correct
            },
            'semantic_analysis_statistics': {
                'problems_with_intent_analysis': problems_with_intent,
                'problems_with_ambiguity_resolution': problems_with_ambiguity_resolution,
                'intent_analysis_rate': problems_with_intent / total_problems,
                'ambiguity_resolution_rate': problems_with_ambiguity_resolution / total_problems
            }
        }
    
    def _generate_comparison_report(self, results: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
        """生成比较报告"""
        
        report = f"""
=== Improved vs Robust System Comparison Report ===
Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Problems Tested: {stats['total_problems']}

=== OVERALL WINNER STATISTICS ===
🏆 Improved System Wins: {stats['winner_statistics']['improved_wins']} ({stats['winner_statistics']['improved_win_rate']:.1%})
🏆 Robust System Wins: {stats['winner_statistics']['robust_wins']} ({stats['winner_statistics']['robust_win_rate']:.1%})
🤝 Ties: {stats['winner_statistics']['ties']}

=== ACCURACY COMPARISON ===
✅ Improved System Accuracy: {stats['accuracy_statistics']['improved_accuracy']:.1%} ({stats['accuracy_statistics']['improved_correct_count']}/{stats['total_problems']})
✅ Robust System Accuracy: {stats['accuracy_statistics']['robust_accuracy']:.1%} ({stats['accuracy_statistics']['robust_correct_count']}/{stats['total_problems']})
📈 Accuracy Improvement: {(stats['accuracy_statistics']['improved_accuracy'] - stats['accuracy_statistics']['robust_accuracy']):.1%}

=== SEMANTIC ANALYSIS ENHANCEMENT ===
🧠 Problems with Intent Analysis: {stats['semantic_analysis_statistics']['problems_with_intent_analysis']} ({stats['semantic_analysis_statistics']['intent_analysis_rate']:.1%})
🔍 Problems with Ambiguity Resolution: {stats['semantic_analysis_statistics']['problems_with_ambiguity_resolution']} ({stats['semantic_analysis_statistics']['ambiguity_resolution_rate']:.1%})

=== DETAILED PROBLEM ANALYSIS ==="""
        
        # 详细问题分析
        for result in results:
            report += f"\n\nProblem {result['problem_id']}: {result['question'][:60]}..."
            report += f"\nExpected: {result['expected_answer']}"
            
            if result['improved_system']['success']:
                improved_answer = result['improved_system']['result'].get('final_answer', 'N/A')
                report += f"\nImproved: {improved_answer} ({'✅' if result['comparison']['accuracy_comparison'].get('improved_correct', False) else '❌'})"
            
            if result['robust_system']['success']:
                robust_answer = result['robust_system']['result'].get('final_answer', 'N/A')
                report += f"\nRobust: {robust_answer} ({'✅' if result['comparison']['accuracy_comparison'].get('robust_correct', False) else '❌'})"
            
            report += f"\nWinner: {result['comparison']['winner']}"
        
        return report

def main():
    """主测试函数"""
    print("=== Improved vs Robust System Comparison Test ===")
    print("Testing improvements in:")
    print("1. 语义理解准确性")
    print("2. 推理逻辑连贯性")
    print("3. 歧义解决能力")
    print("4. 多步推理准确性")
    print("5. 答案准确性")
    print("=" * 55)
    
    # 创建比较测试器
    tester = SystemComparisonTester()
    
    # 运行比较测试
    results = tester.run_comprehensive_comparison()
    
    print(results['comparison_report'])
    
    print("\n=== Comparison Test Completed Successfully ===")
    stats = results['overall_statistics']
    print(f"Improved System Win Rate: {stats['winner_statistics']['improved_win_rate']:.1%}")
    print(f"Accuracy Improvement: {(stats['accuracy_statistics']['improved_accuracy'] - stats['accuracy_statistics']['robust_accuracy']):.1%}")
    print(f"Semantic Analysis Coverage: {stats['semantic_analysis_statistics']['intent_analysis_rate']:.1%}")

if __name__ == "__main__":
    main() 