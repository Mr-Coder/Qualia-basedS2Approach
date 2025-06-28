#!/usr/bin/env python3
"""
测试新的GSM8K题目 - 验证增强验证系统和改进推理系统的性能
"""

import json
import logging
import time
from typing import Any, Dict, List

from enhanced_verification_system import EnhancedVerificationSystem
from improved_reasoning_system import ImprovedMathematicalReasoningSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('new_gsm8k_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewGSM8KTester:
    def __init__(self):
        self.verification_system = EnhancedVerificationSystem()
        self.improved_system = ImprovedMathematicalReasoningSystem()
    
    def get_test_problems(self) -> List[Dict[str, Any]]:
        """获取新的测试题目"""
        return [
            {
                "id": 1,
                "question": "Dean has 30 marbles. He gives 1/5 of them to Jamie and gives 10 to Donald. How many marbles are left for Dean?",
                "expected_answer": 14.0,
                "solution": "Dean gives 30 x 1/5 = 6 marbles to Jamie. He gives a total of 6 + 10 = 16 marbles in all. So, Dean has 30 - 16 = 14 marbles left.",
                "complexity": "L2"
            },
            {
                "id": 2,
                "question": "Duncan's age eight years ago was two times Adam's age four years ago. If Duncan's age is 60 now, how old will Adam be in 8 years?",
                "expected_answer": 38.0,
                "solution": "Eight years ago, Duncan was 60-8 = 52 years old. Four years ago, Adam's age was 52/2 = 26 years. Currently, Adam is 26+4 = 30 years old. In 8 years from now, Adam will be 30+8 = 38 years old.",
                "complexity": "L3"
            },
            {
                "id": 3,
                "question": "Farmer Brown's farm is 200 acres, and Farmer Smith's farm is 100 acres more than twice that. How many acres do the two farms have, together?",
                "expected_answer": 700.0,
                "solution": "Farmer Smith has 2*200+100 = 500 acres. The total is 200+500 = 700.",
                "complexity": "L2"
            },
            {
                "id": 4,
                "question": "Colby works for a manufacturing company in the packaging division. He gets paid $0.20 for every package he completes. If he completes 10 less than 50 packages per hour, how much money, in dollars, does he earn in a typical eight-hour workday?",
                "expected_answer": 64.0,
                "solution": "Ten less than fifty is 50-10 = 40 packages per hour. At $0.20 per completed package, if he completes 40 packages per hour he earns 40*$0.20 = $8 per hour. Thus, in a typical eight-hour day, he earns 8*$8 = $64.",
                "complexity": "L3"
            },
            {
                "id": 5,
                "question": "John drinks a bottle of water every half hour. A normal sudoku puzzle takes him 45 minutes. An extreme sudoku takes 4 times that long. How many bottles of water does he drink in that time?",
                "expected_answer": 6.0,
                "solution": "He drinks a bottle of water every 60/2 = 30 minutes. It takes him 45*4 = 180 minutes to solve an extreme sudoku. So he drinks 180/30 = 6 bottles in that time.",
                "complexity": "L3"
            },
            {
                "id": 6,
                "question": "Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?",
                "expected_answer": 70.0,
                "solution": "Let x be the number of silver coins. Gretchen has x+30 gold coins. x+x+30=110, 2*x=80, x=40. Gretchen has 40+30=70 gold coins.",
                "complexity": "L3"
            }
        ]
    
    def test_single_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个问题"""
        question = problem["question"]
        expected_answer = problem["expected_answer"]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"测试问题 {problem['id']}: {question}")
        logger.info(f"期望答案: {expected_answer}")
        logger.info(f"复杂度: {problem['complexity']}")
        
        # 使用改进推理系统求解
        start_time = time.time()
        improved_result = self.improved_system.solve_mathematical_problem(question)
        processing_time = time.time() - start_time
        
        computed_answer = improved_result.get('final_answer')
        is_correct = abs(float(computed_answer or 0) - float(expected_answer)) < 0.01 if computed_answer is not None else False
        
        logger.info(f"改进系统答案: {computed_answer}")
        logger.info(f"是否正确: {'✅' if is_correct else '❌'}")
        logger.info(f"处理时间: {processing_time:.4f}秒")
        
        # 使用增强验证系统进行验证
        verification_start = time.time()
        verification_result = self.verification_system.comprehensive_verification(
            problem_text=question,
            reasoning_steps=improved_result.get('reasoning_steps', []),
            final_answer=computed_answer
        )
        verification_time = time.time() - verification_start
        
        logger.info(f"验证时间: {verification_time:.4f}秒")
        logger.info(f"验证得分: {verification_result.get('overall_score', 0):.3f}")
        logger.info(f"语义理解得分: {verification_result.get('semantic_score', 0):.3f}")
        logger.info(f"逻辑一致性: {verification_result.get('logic_consistency', 0):.3f}")
        logger.info(f"计算精度: {verification_result.get('calculation_precision', 0):.3f}")
        
        # 显示推理步骤
        reasoning_steps = improved_result.get('reasoning_steps', [])
        if reasoning_steps:
            logger.info(f"\n推理步骤:")
            for i, step in enumerate(reasoning_steps, 1):
                logger.info(f"  {i}. {step}")
        
        # 显示验证建议
        recommendations = verification_result.get('recommendations', [])
        if recommendations:
            logger.info(f"\n验证建议:")
            for rec in recommendations:
                logger.info(f"  - {rec}")
        
        return {
            "problem_id": problem["id"],
            "question": question,
            "expected_answer": expected_answer,
            "computed_answer": computed_answer,
            "is_correct": is_correct,
            "complexity": problem["complexity"],
            "processing_time": processing_time,
            "verification_time": verification_time,
            "verification_score": verification_result.get('overall_score', 0),
            "semantic_score": verification_result.get('semantic_score', 0),
            "logic_consistency": verification_result.get('logic_consistency', 0),
            "calculation_precision": verification_result.get('calculation_precision', 0),
            "reasoning_steps": reasoning_steps,
            "verification_result": verification_result,
            "improved_result": improved_result
        }
    
    def run_test_suite(self):
        """运行完整测试套件"""
        logger.info("开始新GSM8K题目测试...")
        
        test_problems = self.get_test_problems()
        results = []
        
        # 统计变量
        total_problems = len(test_problems)
        correct_answers = 0
        total_processing_time = 0
        total_verification_time = 0
        total_verification_score = 0
        
        complexity_stats = {
            "L1": {"total": 0, "correct": 0},
            "L2": {"total": 0, "correct": 0},
            "L3": {"total": 0, "correct": 0}
        }
        
        for problem in test_problems:
            try:
                result = self.test_single_problem(problem)
                results.append(result)
                
                # 更新统计信息
                if result["is_correct"]:
                    correct_answers += 1
                
                total_processing_time += result["processing_time"]
                total_verification_time += result["verification_time"]
                total_verification_score += result["verification_score"]
                
                complexity = result["complexity"]
                complexity_stats[complexity]["total"] += 1
                if result["is_correct"]:
                    complexity_stats[complexity]["correct"] += 1
                    
            except Exception as e:
                logger.error(f"测试问题 {problem['id']} 时出错: {e}")
                results.append({
                    "problem_id": problem["id"],
                    "question": problem["question"],
                    "expected_answer": problem["expected_answer"],
                    "computed_answer": None,
                    "is_correct": False,
                    "error": str(e)
                })
        
        # 生成总结报告
        accuracy = (correct_answers / total_problems) * 100 if total_problems > 0 else 0
        avg_processing_time = total_processing_time / total_problems if total_problems > 0 else 0
        avg_verification_time = total_verification_time / total_problems if total_problems > 0 else 0
        avg_verification_score = total_verification_score / total_problems if total_problems > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info("测试结果总结:")
        logger.info(f"总题目数: {total_problems}")
        logger.info(f"正确答案数: {correct_answers}")
        logger.info(f"总准确率: {accuracy:.1f}%")
        logger.info(f"平均处理时间: {avg_processing_time:.4f}秒")
        logger.info(f"平均验证时间: {avg_verification_time:.4f}秒")
        logger.info(f"平均验证得分: {avg_verification_score:.3f}")
        
        logger.info(f"\n复杂度分析:")
        for complexity, stats in complexity_stats.items():
            if stats["total"] > 0:
                comp_accuracy = (stats["correct"] / stats["total"]) * 100
                logger.info(f"{complexity}: {stats['correct']}/{stats['total']} ({comp_accuracy:.1f}%)")
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"new_gsm8k_results_{timestamp}.json"
        summary_file = f"new_gsm8k_summary_{timestamp}.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_info": {
                    "total_problems": total_problems,
                    "correct_answers": correct_answers,
                    "accuracy": accuracy,
                    "avg_processing_time": avg_processing_time,
                    "avg_verification_time": avg_verification_time,
                    "avg_verification_score": avg_verification_score,
                    "complexity_stats": complexity_stats
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"新GSM8K测试结果总结\n")
            f.write(f"总题目数: {total_problems}\n")
            f.write(f"正确答案数: {correct_answers}\n")
            f.write(f"总准确率: {accuracy:.1f}%\n")
            f.write(f"平均处理时间: {avg_processing_time:.4f}秒\n")
            f.write(f"平均验证时间: {avg_verification_time:.4f}秒\n")
            f.write(f"平均验证得分: {avg_verification_score:.3f}\n\n")
            f.write(f"复杂度分析:\n")
            for complexity, stats in complexity_stats.items():
                if stats["total"] > 0:
                    comp_accuracy = (stats["correct"] / stats["total"]) * 100
                    f.write(f"{complexity}: {stats['correct']}/{stats['total']} ({comp_accuracy:.1f}%)\n")
        
        logger.info(f"\n结果已保存到:")
        logger.info(f"详细结果: {results_file}")
        logger.info(f"总结报告: {summary_file}")
        
        return results

def main():
    """主函数"""
    tester = NewGSM8KTester()
    results = tester.run_test_suite()
    
    logger.info("新GSM8K测试完成!")
    return results

if __name__ == "__main__":
    main() 