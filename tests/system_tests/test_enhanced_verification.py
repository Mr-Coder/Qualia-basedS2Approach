"""
Enhanced Verification System Test Suite
======================================

测试增强验证系统在GSM8K数据集上的表现，验证：
1. 推理逻辑准确性优化
2. 语义理解增强
3. 多步推理链验证
4. 答案合理性检查
5. 自适应阈值调整

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from enhanced_verification_system import (EnhancedVerificationSystem,
                                          HighPrecisionCalculator,
                                          MultiStepReasoningValidator,
                                          SemanticUnderstandingEngine,
                                          VerificationLevel)
from robust_reasoning_system import RobustMathematicalReasoningSystem

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_verification_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedVerificationTester:
    """增强验证系统测试器"""
    
    def __init__(self):
        self.robust_system = RobustMathematicalReasoningSystem()
        self.verifier = EnhancedVerificationSystem(VerificationLevel.STANDARD)
        self.test_results = []
        
    def load_gsm8k_sample(self, num_problems: int = 20) -> List[Dict[str, Any]]:
        """加载GSM8K测试样本"""
        try:
            gsm8k_path = Path("Data/GSM8K/test.jsonl")
            if not gsm8k_path.exists():
                logger.error(f"GSM8K test file not found: {gsm8k_path}")
                return []
            
            problems = []
            with open(gsm8k_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_problems:
                        break
                    data = json.loads(line.strip())
                    problems.append({
                        'id': i,
                        'question': data['question'],
                        'answer': data['answer'],
                        'expected_numeric': self._extract_numeric_answer(data['answer'])
                    })
            
            logger.info(f"Loaded {len(problems)} GSM8K problems for testing")
            return problems
            
        except Exception as e:
            logger.error(f"Error loading GSM8K sample: {e}")
            return []
    
    def _extract_numeric_answer(self, answer_text: str) -> float:
        """从答案文本中提取数值"""
        import re

        # 查找最后一个数字（通常是最终答案）
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return 0.0
    
    def test_single_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个问题的增强验证"""
        start_time = time.time()
        
        try:
            # 1. 使用robust系统求解问题
            solution = self.robust_system.solve_mathematical_problem(problem['question'])
            
            # 2. 增强验证分析
            verification_result = self.verifier.comprehensive_verification(
                problem['question'],
                solution.get('reasoning_steps', []),
                solution.get('final_answer')
            )
            
            processing_time = time.time() - start_time
            
            # 3. 准确性检查
            predicted_answer = solution.get('final_answer', 0)
            expected_answer = problem['expected_numeric']
            is_correct = abs(predicted_answer - expected_answer) < 0.01 if isinstance(predicted_answer, (int, float)) else False
            
            # 4. 编译测试结果
            test_result = {
                'problem_id': problem['id'],
                'question': problem['question'],
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'processing_time': processing_time,
                'reasoning_steps_count': len(solution.get('reasoning_steps', [])),
                
                # 验证结果
                'verification_score': verification_result['overall_verification_score'],
                'semantic_analysis': {
                    'problem_type': verification_result['semantic_analysis'].problem_type,
                    'understanding_score': verification_result['semantic_analysis'].understanding_score,
                    'semantic_confidence': verification_result['semantic_analysis'].semantic_confidence,
                    'ambiguity_flags': verification_result['semantic_analysis'].ambiguity_flags
                },
                'chain_validation': {
                    'is_logically_consistent': verification_result['chain_validation'].is_logically_consistent,
                    'consistency_score': verification_result['chain_validation'].consistency_score,
                    'logical_errors_count': len(verification_result['chain_validation'].logical_errors),
                    'semantic_coherence': verification_result['chain_validation'].semantic_coherence
                },
                'precision_analysis': {
                    'overall_precision_score': verification_result['precision_results']['overall_precision_score'],
                    'precision_errors_count': len(verification_result['precision_results']['precision_errors'])
                },
                'answer_reasonableness': verification_result['answer_reasonableness'],
                'recommendations': verification_result['recommendations']
            }
            
            logger.info(f"Problem {problem['id']}: {'✓' if is_correct else '✗'} "
                       f"(Verification: {verification_result['overall_verification_score']:.3f})")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing problem {problem['id']}: {e}")
            return {
                'problem_id': problem['id'],
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def run_comprehensive_test(self, num_problems: int = 20) -> Dict[str, Any]:
        """运行综合测试"""
        logger.info(f"Starting comprehensive enhanced verification test on {num_problems} problems")
        
        # 加载测试数据
        problems = self.load_gsm8k_sample(num_problems)
        if not problems:
            return {'error': 'No test data available'}
        
        # 测试每个问题
        test_results = []
        for problem in problems:
            result = self.test_single_problem(problem)
            test_results.append(result)
            self.test_results.append(result)
        
        # 计算统计结果
        stats = self._calculate_test_statistics(test_results)
        
        # 生成详细报告
        report = self._generate_detailed_report(test_results, stats)
        
        # 保存结果
        self._save_test_results(test_results, stats, report)
        
        return {
            'test_results': test_results,
            'statistics': stats,
            'report': report
        }
    
    def _calculate_test_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算测试统计"""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        # 基础统计
        total_problems = len(valid_results)
        correct_answers = sum(1 for r in valid_results if r.get('is_correct', False))
        accuracy = correct_answers / total_problems
        
        # 处理时间统计
        processing_times = [r['processing_time'] for r in valid_results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # 推理步骤统计
        reasoning_steps = [r['reasoning_steps_count'] for r in valid_results]
        avg_reasoning_steps = sum(reasoning_steps) / len(reasoning_steps)
        
        # 验证分数统计
        verification_scores = [r['verification_score'] for r in valid_results]
        avg_verification_score = sum(verification_scores) / len(verification_scores)
        
        # 语义理解统计
        understanding_scores = [r['semantic_analysis']['understanding_score'] for r in valid_results]
        avg_understanding_score = sum(understanding_scores) / len(understanding_scores)
        
        # 逻辑一致性统计
        consistent_chains = sum(1 for r in valid_results if r['chain_validation']['is_logically_consistent'])
        consistency_rate = consistent_chains / total_problems
        
        # 精度统计
        precision_scores = [r['precision_analysis']['overall_precision_score'] for r in valid_results]
        avg_precision_score = sum(precision_scores) / len(precision_scores)
        
        # 问题类型分布
        problem_types = {}
        for r in valid_results:
            ptype = r['semantic_analysis']['problem_type']
            problem_types[ptype] = problem_types.get(ptype, 0) + 1
        
        return {
            'total_problems': total_problems,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'avg_processing_time': avg_processing_time,
            'avg_reasoning_steps': avg_reasoning_steps,
            'avg_verification_score': avg_verification_score,
            'avg_understanding_score': avg_understanding_score,
            'consistency_rate': consistency_rate,
            'avg_precision_score': avg_precision_score,
            'problem_type_distribution': problem_types,
            'reasoning_steps_distribution': {
                '1-2 steps': sum(1 for r in reasoning_steps if 1 <= r <= 2),
                '3-5 steps': sum(1 for r in reasoning_steps if 3 <= r <= 5),
                '6+ steps': sum(1 for r in reasoning_steps if r >= 6)
            }
        }
    
    def _generate_detailed_report(self, results: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
        """生成详细报告"""
        report = f"""
=== Enhanced Verification System Test Report ===
Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Problems Tested: {stats['total_problems']}

=== CORE PERFORMANCE METRICS ===
✓ Answer Accuracy: {stats['accuracy']:.1%} ({stats['correct_answers']}/{stats['total_problems']})
✓ Average Processing Time: {stats['avg_processing_time']:.4f}s
✓ Average Reasoning Steps: {stats['avg_reasoning_steps']:.1f}

=== ENHANCED VERIFICATION METRICS ===
🔍 Overall Verification Score: {stats['avg_verification_score']:.3f}
🧠 Semantic Understanding Score: {stats['avg_understanding_score']:.3f}
⚡ Logic Consistency Rate: {stats['consistency_rate']:.1%}
🎯 Calculation Precision Score: {stats['avg_precision_score']:.3f}

=== PROBLEM TYPE DISTRIBUTION ==="""
        
        for ptype, count in stats['problem_type_distribution'].items():
            percentage = count / stats['total_problems'] * 100
            report += f"\n• {ptype}: {count} problems ({percentage:.1f}%)"
        
        report += f"""

=== REASONING COMPLEXITY ANALYSIS ===
• Simple (1-2 steps): {stats['reasoning_steps_distribution']['1-2 steps']} problems
• Medium (3-5 steps): {stats['reasoning_steps_distribution']['3-5 steps']} problems  
• Complex (6+ steps): {stats['reasoning_steps_distribution']['6+ steps']} problems

=== VERIFICATION QUALITY INSIGHTS ==="""
        
        # 分析验证质量
        high_verification = sum(1 for r in results if r.get('verification_score', 0) > 0.8)
        report += f"\n• High Verification Quality (>0.8): {high_verification}/{stats['total_problems']} ({high_verification/stats['total_problems']:.1%})"
        
        # 逻辑错误分析
        total_logical_errors = sum(r['chain_validation']['logical_errors_count'] for r in results if 'error' not in r)
        report += f"\n• Total Logical Errors Detected: {total_logical_errors}"
        
        # 精度错误分析
        total_precision_errors = sum(r['precision_analysis']['precision_errors_count'] for r in results if 'error' not in r)
        report += f"\n• Total Precision Errors Detected: {total_precision_errors}"
        
        # 语义歧义分析
        ambiguous_problems = sum(1 for r in results if r.get('semantic_analysis', {}).get('ambiguity_flags', []))
        report += f"\n• Problems with Semantic Ambiguities: {ambiguous_problems}"
        
        report += f"""

=== SYSTEM RECOMMENDATIONS ==="""
        
        # 收集推荐建议
        all_recommendations = []
        for r in results:
            if 'error' not in r:
                all_recommendations.extend(r.get('recommendations', []))
        
        # 统计推荐频率
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        for rec, count in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                report += f"\n• {rec} (出现 {count} 次)"
        
        return report
    
    def _save_test_results(self, results: List[Dict[str, Any]], stats: Dict[str, Any], report: str):
        """保存测试结果"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        results_file = f"enhanced_verification_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_results': results,
                'statistics': stats,
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        # 保存报告
        report_file = f"enhanced_verification_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Test results saved to {results_file}")
        logger.info(f"Test report saved to {report_file}")
        
        # 打印报告到控制台
        print(report)

def main():
    """主测试函数"""
    print("=== Enhanced Verification System Test Suite ===")
    print("Testing improvements in:")
    print("1. 推理逻辑准确性优化")
    print("2. 语义理解增强") 
    print("3. 多步推理链验证")
    print("4. 答案合理性检查")
    print("5. 自适应阈值调整")
    print("=" * 50)
    
    # 创建测试器
    tester = EnhancedVerificationTester()
    
    # 运行测试
    results = tester.run_comprehensive_test(num_problems=20)
    
    if 'error' in results:
        print(f"Test failed: {results['error']}")
        return
    
    print("\n=== Test Completed Successfully ===")
    print(f"Accuracy: {results['statistics']['accuracy']:.1%}")
    print(f"Verification Score: {results['statistics']['avg_verification_score']:.3f}")
    print(f"Processing Time: {results['statistics']['avg_processing_time']:.4f}s")

if __name__ == "__main__":
    main() 