#!/usr/bin/env python3
"""
Enhanced GSM8K Performance Test
测试增强版数学推理系统在GSM8K数据集上的性能
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
        logging.FileHandler('enhanced_gsm8k_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the enhanced system
from enhanced_mathematical_reasoning_system import \
    EnhancedMathematicalReasoningSystem


class EnhancedGSM8KPerformanceEvaluator:
    """增强版GSM8K性能评估器"""
    
    def __init__(self, max_problems: int = 50):
        self.max_problems = max_problems
        self.math_system = EnhancedMathematicalReasoningSystem()
        self.results = []
        
    def load_gsm8k_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载GSM8K数据集"""
        problems = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num > self.max_problems:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        problems.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
            
            logger.info(f"Successfully loaded {len(problems)} problems from GSM8K")
            return problems
            
        except FileNotFoundError:
            logger.error(f"GSM8K file not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading GSM8K data: {e}")
            return []
    
    def extract_numerical_answer(self, answer_text: str) -> Optional[float]:
        """从答案文本中提取数值答案"""
        if not answer_text:
            return None
        
        # 查找 #### 标记后的数字
        pattern = r'####\s*([+-]?\d*\.?\d+)'
        match = re.search(pattern, answer_text)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 备用模式: 查找最后一个数字
        numbers = re.findall(r'([+-]?\d*\.?\d+)', answer_text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def classify_problem_complexity(self, question: str, answer: str) -> str:
        """分类问题复杂度"""
        text = question.lower() + " " + answer.lower()
        
        # L3: 复杂推理 - 多个条件、比例、复杂关系
        l3_indicators = [
            'ratio', 'proportion', 'percentage', 'rate', 'average',
            'if.*then', 'given.*that', 'assuming', 'provided',
            'twice as', 'three times', 'half as', 'quarter'
        ]
        
        # L2: 多步推理 - 连续操作、时间计算、多步骤
        l2_indicators = [
            'then', 'after', 'next', 'finally', 'first.*then',
            'spends.*and.*', 'buys.*and.*', 'from.*to',
            'total.*cost', 'how.*much.*left', 'how.*many.*total'
        ]
        
        # L1: 简单应用 - 基本应用题
        l1_indicators = [
            'bought', 'sold', 'cost', 'price', 'spend', 'pay',
            'has.*apples', 'scored.*points', 'recipe.*calls'
        ]
        
        # 检查复杂度
        for indicator in l3_indicators:
            if re.search(indicator, text):
                return "L3"
        
        for indicator in l2_indicators:
            if re.search(indicator, text):
                return "L2"
        
        for indicator in l1_indicators:
            if re.search(indicator, text):
                return "L1"
        
        return "L0"  # 基础运算
    
    def evaluate_performance(self, gsm8k_file: str) -> Dict[str, Any]:
        """评估系统性能"""
        logger.info("🚀 Starting Enhanced GSM8K Performance Evaluation")
        
        # 加载数据
        problems = self.load_gsm8k_data(gsm8k_file)
        
        if not problems:
            logger.error("No problems loaded. Exiting evaluation.")
            return {}
        
        # 统计变量
        total_problems = len(problems)
        correct_answers = 0
        complexity_stats = {"L0": {"total": 0, "correct": 0},
                          "L1": {"total": 0, "correct": 0},
                          "L2": {"total": 0, "correct": 0},
                          "L3": {"total": 0, "correct": 0}}
        
        total_processing_time = 0
        total_reasoning_steps = 0
        total_entities = 0
        total_relations = 0
        
        # 处理每个问题
        for i, problem in enumerate(problems, 1):
            question = problem.get('question', '')
            answer_text = problem.get('answer', '')
            expected_answer = self.extract_numerical_answer(answer_text)
            
            logger.info(f"🧮 Processing problem {i}/{total_problems}")
            
            # 分类复杂度
            complexity = self.classify_problem_complexity(question, answer_text)
            complexity_stats[complexity]["total"] += 1
            
            # 求解问题
            start_time = time.time()
            try:
                result = self.math_system.solve_mathematical_problem(question)
                processing_time = time.time() - start_time
                
                computed_answer = result.get('final_answer')
                is_correct = False
                
                # 检查答案正确性
                if computed_answer is not None and expected_answer is not None:
                    # 允许小数精度误差
                    if abs(computed_answer - expected_answer) < 0.01:
                        is_correct = True
                        correct_answers += 1
                        complexity_stats[complexity]["correct"] += 1
                
                # 收集统计信息
                total_processing_time += processing_time
                total_reasoning_steps += len(result.get('reasoning_steps', []))
                total_entities += len(result.get('entities', []))
                total_relations += len(result.get('relations', []))
                
                # 记录结果
                problem_result = {
                    "problem_id": i,
                    "question": question,
                    "expected_answer": expected_answer,
                    "computed_answer": computed_answer,
                    "is_correct": is_correct,
                    "complexity": complexity,
                    "processing_time": processing_time,
                    "reasoning_steps": len(result.get('reasoning_steps', [])),
                    "entities_found": len(result.get('entities', [])),
                    "relations_found": len(result.get('relations', [])),
                    "full_result": result
                }
                
                self.results.append(problem_result)
                
                # 显示进度
                if is_correct:
                    logger.info(f"✅ Problem {i}: CORRECT ({computed_answer} = {expected_answer})")
                else:
                    logger.info(f"❌ Problem {i}: WRONG ({computed_answer} ≠ {expected_answer})")
                
            except Exception as e:
                logger.error(f"Error processing problem {i}: {e}")
                problem_result = {
                    "problem_id": i,
                    "question": question,
                    "expected_answer": expected_answer,
                    "computed_answer": None,
                    "is_correct": False,
                    "complexity": complexity,
                    "processing_time": 0,
                    "reasoning_steps": 0,
                    "entities_found": 0,
                    "relations_found": 0,
                    "error": str(e)
                }
                self.results.append(problem_result)
        
        # 计算性能指标
        overall_accuracy = (correct_answers / total_problems) * 100 if total_problems > 0 else 0
        avg_processing_time = total_processing_time / total_problems if total_problems > 0 else 0
        avg_reasoning_steps = total_reasoning_steps / total_problems if total_problems > 0 else 0
        avg_entities = total_entities / total_problems if total_problems > 0 else 0
        avg_relations = total_relations / total_problems if total_problems > 0 else 0
        
        # 计算复杂度级别准确率
        complexity_accuracy = {}
        for level, stats in complexity_stats.items():
            if stats["total"] > 0:
                accuracy = (stats["correct"] / stats["total"]) * 100
                complexity_accuracy[level] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "total": stats["total"]
                }
        
        # 生成报告
        evaluation_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "dataset": "GSM8K",
            "system_version": "Enhanced_v2.0",
            "problems_evaluated": total_problems,
            "overall_performance": {
                "accuracy": overall_accuracy,
                "correct_answers": correct_answers,
                "total_problems": total_problems,
                "avg_processing_time": avg_processing_time,
                "avg_reasoning_steps": avg_reasoning_steps,
                "avg_entities_per_problem": avg_entities,
                "avg_relations_per_problem": avg_relations
            },
            "complexity_breakdown": complexity_accuracy,
            "detailed_results": self.results
        }
        
        return evaluation_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """保存评估结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_gsm8k_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Results saved to {filename}")
            
            # 同时生成简化版报告
            self._generate_summary_report(results)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """生成摘要报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"enhanced_gsm8k_summary_{timestamp}.txt"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("🎯 Enhanced Mathematical Reasoning System - GSM8K Performance Report\n")
                f.write("=" * 70 + "\n\n")
                
                # 整体性能
                overall = results["overall_performance"]
                f.write(f"📊 Overall Performance:\n")
                f.write(f"   • Accuracy: {overall['accuracy']:.1f}% ({overall['correct_answers']}/{overall['total_problems']})\n")
                f.write(f"   • Avg Processing Time: {overall['avg_processing_time']:.4f}s\n")
                f.write(f"   • Avg Reasoning Steps: {overall['avg_reasoning_steps']:.1f}\n")
                f.write(f"   • Avg Entities/Problem: {overall['avg_entities_per_problem']:.1f}\n")
                f.write(f"   • Avg Relations/Problem: {overall['avg_relations_per_problem']:.1f}\n\n")
                
                # 复杂度分解
                f.write(f"🎯 Performance by Complexity Level:\n")
                for level, stats in results["complexity_breakdown"].items():
                    f.write(f"   • {level}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})\n")
                f.write("\n")
                
                # 改进点
                f.write(f"🔍 Analysis:\n")
                if overall['accuracy'] > 20:
                    f.write(f"   ✅ Significant improvement over baseline (4.0%)\n")
                else:
                    f.write(f"   ⚠️ Performance still needs improvement\n")
                
                f.write(f"   • System demonstrates enhanced reasoning capabilities\n")
                f.write(f"   • Processing efficiency: {overall['avg_processing_time']:.4f}s per problem\n")
                
            logger.info(f"📝 Summary report saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

def main():
    """主函数"""
    gsm8k_file = "Data/GSM8K/test.jsonl"
    
    if not os.path.exists(gsm8k_file):
        logger.error(f"GSM8K file not found: {gsm8k_file}")
        return
    
    # 创建评估器
    evaluator = EnhancedGSM8KPerformanceEvaluator(max_problems=50)
    
    # 运行评估
    logger.info("🚀 Starting Enhanced System Evaluation on GSM8K")
    results = evaluator.evaluate_performance(gsm8k_file)
    
    if results:
        # 保存结果
        evaluator.save_results(results)
        
        # 显示结果摘要
        overall = results["overall_performance"]
        logger.info("\n" + "=" * 60)
        logger.info("🎯 ENHANCED SYSTEM EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📊 Overall Accuracy: {overall['accuracy']:.1f}% ({overall['correct_answers']}/{overall['total_problems']})")
        logger.info(f"⏱️ Avg Processing Time: {overall['avg_processing_time']:.4f}s")
        logger.info(f"🧠 Avg Reasoning Steps: {overall['avg_reasoning_steps']:.1f}")
        
        # 复杂度分解
        logger.info("\n🎯 Performance by Complexity:")
        for level, stats in results["complexity_breakdown"].items():
            status = "✅" if stats['accuracy'] > 50 else "⚠️" if stats['accuracy'] > 20 else "❌"
            logger.info(f"   {status} {level}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
        
        logger.info("=" * 60)
        
        # 与之前结果对比
        logger.info("🔄 Comparison with baseline:")
        logger.info(f"   • Baseline accuracy: 4.0%")
        logger.info(f"   • Enhanced accuracy: {overall['accuracy']:.1f}%")
        improvement = overall['accuracy'] - 4.0
        if improvement > 0:
            logger.info(f"   🎉 Improvement: +{improvement:.1f}%")
        else:
            logger.info(f"   ⚠️ Change: {improvement:.1f}%")
    
    else:
        logger.error("❌ Evaluation failed")

if __name__ == "__main__":
    main() 