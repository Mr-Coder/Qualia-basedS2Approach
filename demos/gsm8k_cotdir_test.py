"""
GSM8K数据集上的COT-DIR+MLR集成系统测试
测试真实数学推理能力和性能指标

运行方式：
python gsm8k_cotdir_test.py --num_samples 20 --output_file gsm8k_results.json
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加src路径
sys.path.append('src')

def load_gsm8k_dataset(file_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """加载GSM8K数据集"""
    problems = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        
                        # 提取答案数字
                        answer_text = data.get('answer', '')
                        answer_match = re.search(r'#### (\d+)', answer_text)
                        if answer_match:
                            answer_value = int(answer_match.group(1))
                        else:
                            # 尝试从答案文本中提取最后一个数字
                            numbers = re.findall(r'\d+', answer_text)
                            answer_value = int(numbers[-1]) if numbers else 0
                        
                        problem = {
                            'id': line_num + 1,
                            'question': data.get('question', ''),
                            'answer': answer_value,
                            'solution_steps': answer_text,
                            'difficulty': 'medium'  # GSM8K问题通常是中等难度
                        }
                        problems.append(problem)
                        
                        if num_samples and len(problems) >= num_samples:
                            break
                            
                    except json.JSONDecodeError as e:
                        logging.warning(f"第{line_num + 1}行JSON解析失败: {e}")
                        continue
                        
    except FileNotFoundError:
        logging.error(f"GSM8K数据集文件未找到: {file_path}")
        return []
    
    if num_samples:
        problems = random.sample(problems, min(num_samples, len(problems)))
    
    return problems

class GSM8KProcessor:
    """GSM8K问题处理器"""
    
    def __init__(self):
        # 尝试导入完整系统
        try:
            from reasoning_engine.cotdir_integration import \
                COTDIRIntegratedWorkflow
            self.workflow = COTDIRIntegratedWorkflow()
            self.use_full_system = True
            print("✓ 使用完整COT-DIR+MLR集成系统")
        except ImportError:
            print("⚠️ 使用简化数学推理系统")
            self.workflow = None
            self.use_full_system = False
    
    def process_problem(self, problem: Dict) -> Dict[str, Any]:
        """处理单个GSM8K问题"""
        question = problem['question']
        expected_answer = problem['answer']
        
        start_time = time.time()
        
        if self.use_full_system and self.workflow:
            # 使用完整系统
            try:
                result = self.workflow.process(question, "word_problem")
                
                return {
                    'problem_id': problem['id'],
                    'question': question,
                    'expected_answer': expected_answer,
                    'predicted_answer': result['answer']['value'],
                    'confidence': result['overall_confidence'],
                    'processing_time': time.time() - start_time,
                    'reasoning_steps': result['reasoning_process']['total_steps'],
                    'discovered_relations': len(result.get('discovered_relations', [])),
                    'validation_score': self._calculate_validation_score(result.get('validation_report', {})),
                    'is_correct': result['answer']['value'] == expected_answer,
                    'detailed_result': result,
                    'system_type': 'full_cotdir_mlr'
                }
                
            except Exception as e:
                logging.error(f"处理问题{problem['id']}时出错: {e}")
                return self._create_error_result(problem, str(e), time.time() - start_time)
        else:
            # 使用简化系统
            return self._process_with_simple_system(problem, start_time)
    
    def _process_with_simple_system(self, problem: Dict, start_time: float) -> Dict[str, Any]:
        """使用简化系统处理问题"""
        question = problem['question']
        expected_answer = problem['answer']
        
        # 简化的数学推理逻辑
        numbers = self._extract_numbers(question)
        predicted_answer = self._simple_reasoning(question, numbers)
        
        return {
            'problem_id': problem['id'],
            'question': question,
            'expected_answer': expected_answer,
            'predicted_answer': predicted_answer,
            'confidence': 0.7,  # 简化系统的默认置信度
            'processing_time': time.time() - start_time,
            'reasoning_steps': 3,
            'discovered_relations': 1,
            'validation_score': 0.75,
            'is_correct': predicted_answer == expected_answer,
            'system_type': 'simple_math'
        }
    
    def _extract_numbers(self, text: str) -> List[int]:
        """从文本中提取数字"""
        numbers = re.findall(r'\d+', text)
        return [int(num) for num in numbers]
    
    def _simple_reasoning(self, question: str, numbers: List[int]) -> int:
        """简化的数学推理"""
        if not numbers:
            return 0
        
        # 基于关键词的简单推理
        if any(keyword in question.lower() for keyword in ['total', 'altogether', 'sum', '总共', '一共']):
            return sum(numbers)
        elif any(keyword in question.lower() for keyword in ['difference', 'more than', 'less than', '多', '少']):
            return max(numbers) - min(numbers) if len(numbers) >= 2 else numbers[0]
        elif any(keyword in question.lower() for keyword in ['times', 'multiply', '倍', '乘']):
            result = 1
            for num in numbers[:2]:  # 只取前两个数字
                result *= num
            return result
        else:
            # 默认返回最大数字或数字和
            return max(numbers) if len(numbers) == 1 else sum(numbers[:2])
    
    def _calculate_validation_score(self, validation_report: Dict) -> float:
        """计算验证分数"""
        if not validation_report:
            return 0.0
        
        scores = []
        for dimension, result in validation_report.items():
            if isinstance(result, dict) and 'score' in result:
                scores.append(result['score'])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _create_error_result(self, problem: Dict, error_msg: str, processing_time: float) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'problem_id': problem['id'],
            'question': problem['question'],
            'expected_answer': problem['answer'],
            'predicted_answer': 'ERROR',
            'confidence': 0.0,
            'processing_time': processing_time,
            'reasoning_steps': 0,
            'discovered_relations': 0,
            'validation_score': 0.0,
            'is_correct': False,
            'error': error_msg,
            'system_type': 'error'
        }

def evaluate_results(results: List[Dict]) -> Dict[str, Any]:
    """评估测试结果"""
    total_problems = len(results)
    correct_answers = sum(1 for r in results if r['is_correct'])
    accuracy = correct_answers / total_problems if total_problems > 0 else 0
    
    # 计算各种指标
    avg_confidence = sum(r['confidence'] for r in results) / total_problems if total_problems > 0 else 0
    avg_processing_time = sum(r['processing_time'] for r in results) / total_problems if total_problems > 0 else 0
    avg_reasoning_steps = sum(r['reasoning_steps'] for r in results) / total_problems if total_problems > 0 else 0
    avg_validation_score = sum(r.get('validation_score', 0) for r in results) / total_problems if total_problems > 0 else 0
    
    # 错误分析
    error_count = sum(1 for r in results if r['predicted_answer'] == 'ERROR')
    error_rate = error_count / total_problems if total_problems > 0 else 0
    
    # 置信度分析
    confidence_bins = {'high': 0, 'medium': 0, 'low': 0}
    for result in results:
        conf = result['confidence']
        if conf >= 0.8:
            confidence_bins['high'] += 1
        elif conf >= 0.6:
            confidence_bins['medium'] += 1
        else:
            confidence_bins['low'] += 1
    
    return {
        'total_problems': total_problems,
        'correct_answers': correct_answers,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'metrics': {
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time,
            'average_reasoning_steps': avg_reasoning_steps,
            'average_validation_score': avg_validation_score
        },
        'confidence_distribution': confidence_bins,
        'performance_summary': {
            'excellent': sum(1 for r in results if r['is_correct'] and r['confidence'] >= 0.8),
            'good': sum(1 for r in results if r['is_correct'] and 0.6 <= r['confidence'] < 0.8),
            'fair': sum(1 for r in results if r['is_correct'] and r['confidence'] < 0.6),
            'incorrect': sum(1 for r in results if not r['is_correct'])
        }
    }

def display_results(evaluation: Dict, results: List[Dict]):
    """显示测试结果"""
    print("\n" + "="*80)
    print("🧮 GSM8K 数据集测试结果")
    print("="*80)
    
    # 基本统计
    print(f"\n📊 基本统计:")
    print(f"测试问题总数: {evaluation['total_problems']}")
    print(f"正确答案数: {evaluation['correct_answers']}")
    print(f"准确率: {evaluation['accuracy']:.2%}")
    print(f"错误率: {evaluation['error_rate']:.2%}")
    
    # 性能指标
    metrics = evaluation['metrics']
    print(f"\n⚡ 性能指标:")
    print(f"平均置信度: {metrics['average_confidence']:.3f}")
    print(f"平均处理时间: {metrics['average_processing_time']:.3f}秒")
    print(f"平均推理步骤: {metrics['average_reasoning_steps']:.1f}")
    print(f"平均验证分数: {metrics['average_validation_score']:.3f}")
    
    # 置信度分布
    conf_dist = evaluation['confidence_distribution']
    print(f"\n🎯 置信度分布:")
    print(f"高置信度 (≥0.8): {conf_dist['high']} ({conf_dist['high']/evaluation['total_problems']:.1%})")
    print(f"中等置信度 (0.6-0.8): {conf_dist['medium']} ({conf_dist['medium']/evaluation['total_problems']:.1%})")
    print(f"低置信度 (<0.6): {conf_dist['low']} ({conf_dist['low']/evaluation['total_problems']:.1%})")
    
    # 性能分级
    perf = evaluation['performance_summary']
    print(f"\n🏆 性能分级:")
    print(f"优秀 (正确+高置信度): {perf['excellent']}")
    print(f"良好 (正确+中等置信度): {perf['good']}")
    print(f"一般 (正确+低置信度): {perf['fair']}")
    print(f"错误: {perf['incorrect']}")
    
    # 展示一些具体例子
    print(f"\n🔍 测试样例:")
    correct_samples = [r for r in results if r['is_correct']][:2]
    incorrect_samples = [r for r in results if not r['is_correct']][:2]
    
    for i, sample in enumerate(correct_samples, 1):
        print(f"\n✓ 正确样例 {i}:")
        print(f"  问题: {sample['question'][:60]}...")
        print(f"  预期答案: {sample['expected_answer']}")
        print(f"  预测答案: {sample['predicted_answer']}")
        print(f"  置信度: {sample['confidence']:.2%}")
    
    for i, sample in enumerate(incorrect_samples, 1):
        print(f"\n✗ 错误样例 {i}:")
        print(f"  问题: {sample['question'][:60]}...")
        print(f"  预期答案: {sample['expected_answer']}")
        print(f"  预测答案: {sample['predicted_answer']}")
        print(f"  置信度: {sample['confidence']:.2%}")

def save_results(results: List[Dict], evaluation: Dict, output_file: str):
    """保存测试结果"""
    output_data = {
        'metadata': {
            'framework': 'COT-DIR + MLR Integration',
            'dataset': 'GSM8K',
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(results)
        },
        'evaluation': evaluation,
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存至: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GSM8K数据集上的COT-DIR+MLR测试')
    parser.add_argument('--dataset_path', default='Data/GSM8K/test.jsonl', help='GSM8K数据集路径')
    parser.add_argument('--num_samples', type=int, default=20, help='测试样本数量')
    parser.add_argument('--output_file', default=None, help='结果输出文件')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print(f"🧮 GSM8K数据集 COT-DIR+MLR 集成系统测试")
    print(f"数据集路径: {args.dataset_path}")
    print(f"测试样本数: {args.num_samples}")
    
    # 加载数据集
    print(f"\n📚 加载GSM8K数据集...")
    problems = load_gsm8k_dataset(args.dataset_path, args.num_samples)
    
    if not problems:
        print("❌ 数据集加载失败！请检查文件路径。")
        return
    
    print(f"✓ 成功加载 {len(problems)} 个问题")
    
    # 初始化处理器
    print(f"\n🔧 初始化处理系统...")
    processor = GSM8KProcessor()
    
    # 处理问题
    print(f"\n🚀 开始处理问题...")
    results = []
    
    for i, problem in enumerate(problems, 1):
        if args.verbose:
            print(f"\n处理问题 {i}/{len(problems)}: {problem['question'][:50]}...")
        else:
            if i % 5 == 0:
                print(f"进度: {i}/{len(problems)}")
        
        result = processor.process_problem(problem)
        results.append(result)
        
        if args.verbose:
            status = "✓" if result['is_correct'] else "✗"
            print(f"  {status} 预测: {result['predicted_answer']} (期望: {result['expected_answer']})")
    
    # 评估结果
    print(f"\n📊 评估测试结果...")
    evaluation = evaluate_results(results)
    
    # 显示结果
    display_results(evaluation, results)
    
    # 保存结果
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f'gsm8k_cotdir_results_{timestamp}.json'
    
    save_results(results, evaluation, output_file)
    
    print(f"\n✨ 测试完成！")
    print("="*80)

if __name__ == "__main__":
    main() 