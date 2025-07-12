#!/usr/bin/env python3
"""
测试修复版推理系统
"""

import json
import logging
import time
from typing import Any, Dict, List

from fixed_reasoning_system import FixedMathematicalReasoningSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixed_system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FixedSystemTester:
    def __init__(self):
        self.fixed_system = FixedMathematicalReasoningSystem()
    
    def get_test_problems(self) -> List[Dict[str, Any]]:
        """获取测试题目"""
        return [
            {
                "id": 1,
                "question": "Dean has 30 marbles. He gives 1/5 of them to Jamie and gives 10 to Donald. How many marbles are left for Dean?",
                "expected_answer": 14.0,
                "solution": "Dean gives 30 × 1/5 = 6 marbles to Jamie. He gives a total of 6 + 10 = 16 marbles in all. So, Dean has 30 - 16 = 14 marbles left.",
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
        
        # 使用修复系统求解
        start_time = time.time()
        fixed_result = self.fixed_system.solve_mathematical_problem(question)
        processing_time = time.time() - start_time
        
        computed_answer = fixed_result.get('final_answer')
        is_correct = abs(float(computed_answer or 0) - float(expected_answer)) < 0.01 if computed_answer is not None else False
        
        logger.info(f"修复系统答案: {computed_answer}")
        logger.info(f"是否正确: {'✅' if is_correct else '❌'}")
        logger.info(f"处理时间: {processing_time:.4f}秒")
        logger.info(f"推理策略: {fixed_result.get('reasoning_type', 'unknown')}")
        logger.info(f"置信度: {fixed_result.get('confidence', 0):.3f}")
        
        # 显示推理步骤
        reasoning_steps = fixed_result.get('reasoning_steps', [])
        if reasoning_steps:
            logger.info(f"\n推理步骤:")
            for step in reasoning_steps:
                logger.info(f"  {step['step_id']}: {step['description']} → {step['output']}")
                if step.get('semantic_justification'):
                    logger.info(f"      理由: {step['semantic_justification']}")
        
        # 显示语义分析
        semantic_analysis = fixed_result.get('semantic_analysis', {})
        if semantic_analysis:
            fractions = semantic_analysis.get('fraction_expressions', [])
            compounds = semantic_analysis.get('compound_expressions', [])
            
            if fractions:
                logger.info(f"\n检测到分数表达式:")
                for frac in fractions:
                    logger.info(f"  - {frac['text']}: {frac['calculation']}")
            
            if compounds:
                logger.info(f"\n检测到复合表达式:")
                for comp in compounds:
                    logger.info(f"  - {comp['text']}: {comp['calculation']}")
        
        return {
            "problem_id": problem["id"],
            "question": question,
            "expected_answer": expected_answer,
            "computed_answer": computed_answer,
            "is_correct": is_correct,
            "complexity": problem["complexity"],
            "processing_time": processing_time,
            "reasoning_strategy": fixed_result.get('reasoning_type'),
            "confidence": fixed_result.get('confidence', 0),
            "reasoning_steps": reasoning_steps,
            "error": fixed_result.get('error')
        }
    
    def run_test_suite(self):
        """运行完整测试套件"""
        logger.info("开始修复系统测试...")
        
        test_problems = self.get_test_problems()
        results = []
        
        # 统计变量
        total_problems = len(test_problems)
        correct_answers = 0
        total_processing_time = 0
        
        complexity_stats = {
            "L1": {"total": 0, "correct": 0},
            "L2": {"total": 0, "correct": 0},
            "L3": {"total": 0, "correct": 0}
        }
        
        strategy_stats = {}
        
        for problem in test_problems:
            try:
                result = self.test_single_problem(problem)
                results.append(result)
                
                # 更新统计信息
                if result["is_correct"]:
                    correct_answers += 1
                
                total_processing_time += result["processing_time"]
                
                complexity = result["complexity"]
                complexity_stats[complexity]["total"] += 1
                if result["is_correct"]:
                    complexity_stats[complexity]["correct"] += 1
                
                # 策略统计
                strategy = result.get("reasoning_strategy", "unknown")
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {"total": 0, "correct": 0}
                strategy_stats[strategy]["total"] += 1
                if result["is_correct"]:
                    strategy_stats[strategy]["correct"] += 1
                    
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
        
        logger.info(f"\n{'='*80}")
        logger.info("修复系统测试结果总结:")
        logger.info(f"总题目数: {total_problems}")
        logger.info(f"正确答案数: {correct_answers}")
        logger.info(f"总准确率: {accuracy:.1f}%")
        logger.info(f"平均处理时间: {avg_processing_time:.4f}秒")
        
        logger.info(f"\n复杂度分析:")
        for complexity, stats in complexity_stats.items():
            if stats["total"] > 0:
                comp_accuracy = (stats["correct"] / stats["total"]) * 100
                logger.info(f"{complexity}: {stats['correct']}/{stats['total']} ({comp_accuracy:.1f}%)")
        
        logger.info(f"\n推理策略分析:")
        for strategy, stats in strategy_stats.items():
            if stats["total"] > 0:
                strategy_accuracy = (stats["correct"] / stats["total"]) * 100
                logger.info(f"{strategy}: {stats['correct']}/{stats['total']} ({strategy_accuracy:.1f}%)")
        
        # 详细分析正确和错误的问题
        correct_problems = [r for r in results if r["is_correct"]]
        incorrect_problems = [r for r in results if not r["is_correct"]]
        
        if correct_problems:
            logger.info(f"\n✅ 正确解答的问题 ({len(correct_problems)}):")
            for result in correct_problems:
                logger.info(f"  问题{result['problem_id']}: {result['reasoning_strategy']} → {result['computed_answer']}")
        
        if incorrect_problems:
            logger.info(f"\n❌ 错误解答的问题 ({len(incorrect_problems)}):")
            for result in incorrect_problems:
                expected = result['expected_answer']
                computed = result['computed_answer']
                logger.info(f"  问题{result['problem_id']}: 期望{expected} 得到{computed} (策略: {result.get('reasoning_strategy', 'unknown')})")
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"fixed_system_results_{timestamp}.json"
        
        # 创建可序列化的结果
        serializable_results = []
        for result in results:
            clean_result = {k: v for k, v in result.items() if k not in ['semantic_analysis']}
            serializable_results.append(clean_result)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_info": {
                    "total_problems": total_problems,
                    "correct_answers": correct_answers,
                    "accuracy": accuracy,
                    "avg_processing_time": avg_processing_time,
                    "complexity_stats": complexity_stats,
                    "strategy_stats": strategy_stats
                },
                "results": serializable_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n结果已保存到: {results_file}")
        
        return results

def main():
    """主函数"""
    tester = FixedSystemTester()
    results = tester.run_test_suite()
    
    logger.info("修复系统测试完成!")
    return results

if __name__ == "__main__":
    main() 