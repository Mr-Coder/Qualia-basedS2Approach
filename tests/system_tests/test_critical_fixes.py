#!/usr/bin/env python3
"""
测试关键修复后的数学推理系统
"""

import json
import sys
from datetime import datetime

from critical_fixes_reasoning_system import CriticalMathematicalReasoningSystem


def load_gsm8k_samples(count=10, start_idx=0):
    """加载GSM8K样本"""
    problems = []
    try:
        with open('Data/GSM8K/test.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i in range(start_idx, min(start_idx + count, len(lines))):
            line = lines[i]
            data = json.loads(line.strip())
            
            # 提取答案
            answer_text = data['answer']
            expected = float(answer_text.split('#### ')[-1])
            
            problems.append({
                'id': f'gsm8k_{i}',
                'question': data['question'],
                'expected_answer': expected
            })
                
        return problems
        
    except Exception as e:
        print(f"❌ 加载GSM8K题目失败: {e}")
        return []

def test_critical_fixes_comprehensive(num_problems=20, start_from=0):
    """测试修复后系统的综合性能"""
    
    print(f"🔧 测试关键修复后的系统 - {num_problems}道题目 (从第{start_from}题开始)")
    
    # 初始化修复后的系统
    system = CriticalMathematicalReasoningSystem()
    
    # 加载题目
    problems = load_gsm8k_samples(num_problems, start_from)
    if not problems:
        print("❌ 没有加载到题目")
        return
    
    # 测试结果统计
    correct = 0
    total = len(problems)
    results = []
    
    # 策略统计
    strategy_stats = {}
    complexity_stats = {}
    error_count = 0
    none_answers = 0
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*60}")
        print(f"📝 题目 {i}/{total}: {problem['id']}")
        print(f"❓ 问题: {problem['question'][:80]}...")
        print(f"🎯 期望答案: {problem['expected_answer']}")
        
        try:
            # 解决问题
            result = system.solve_mathematical_problem(problem['question'])
            predicted = result.get('final_answer', 0.0)
            strategy = result.get('strategy_used', 'unknown')
            complexity = result.get('complexity_level', 'unknown')
            confidence = result.get('confidence', 0.0)
            verification_score = result.get('verification_score', 0.0)
            
            # 检查是否有错误
            if 'error' in result:
                error_count += 1
                print(f"💥 错误: {result['error']}")
            
            # 检查None答案问题
            if predicted is None:
                none_answers += 1
                predicted = 0.0
            
            is_correct = abs(predicted - problem['expected_answer']) < 0.01
            
            if is_correct:
                correct += 1
                status = "✅ 正确"
            else:
                status = "❌ 错误"
            
            print(f"💡 系统答案: {predicted}")
            print(f"📊 结果: {status}")
            print(f"🎲 策略: {strategy}")
            print(f"📏 复杂度: {complexity}")
            print(f"🔍 置信度: {confidence:.3f}")
            print(f"✅ 验证分数: {verification_score:.3f}")
            
            # 统计策略使用情况
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'correct': 0, 'total': 0}
            strategy_stats[strategy]['total'] += 1
            if is_correct:
                strategy_stats[strategy]['correct'] += 1
            
            # 统计复杂度情况
            if complexity not in complexity_stats:
                complexity_stats[complexity] = {'correct': 0, 'total': 0}
            complexity_stats[complexity]['total'] += 1
            if is_correct:
                complexity_stats[complexity]['correct'] += 1
            
            results.append({
                'problem_id': problem['id'],
                'expected': problem['expected_answer'],
                'predicted': predicted,
                'correct': is_correct,
                'strategy': strategy,
                'complexity': complexity,
                'confidence': confidence,
                'verification_score': verification_score,
                'question': problem['question']
            })
            
        except Exception as e:
            error_count += 1
            print(f"💥 处理错误: {e}")
            results.append({
                'problem_id': problem['id'],
                'expected': problem['expected_answer'],
                'predicted': 0.0,
                'correct': False,
                'error': str(e),
                'strategy': 'error',
                'complexity': 'unknown',
                'confidence': 0.0,
                'verification_score': 0.0,
                'question': problem['question']
            })
    
    # 统计结果
    accuracy = (correct / total) * 100
    
    print(f"\n{'='*80}")
    print(f"🔧 关键修复后系统测试结果")
    print(f"{'='*80}")
    print(f"总题目数: {total}")
    print(f"正确题目: {correct}")
    print(f"准确率: {accuracy:.1f}%")
    print(f"系统错误: {error_count} 题")
    print(f"None答案: {none_answers} 题")
    
    # 四个核心问题的修复状态
    print(f"\n📊 四个核心问题修复状态:")
    print(f"1. 系统稳定性: {'✅ 已修复' if error_count == 0 and none_answers == 0 else f'❌ 仍有问题 ({error_count}错误, {none_answers}None答案)'}")
    print(f"2. 策略识别: {'✅ 已修复' if 'unknown' not in [r['strategy'] for r in results] else '❌ 仍有问题'}")
    
    avg_verification = sum(r['verification_score'] for r in results if 'verification_score' in r) / len(results)
    print(f"3. 验证机制: {'✅ 已修复' if avg_verification > 0.1 else '❌ 仍有问题'} (平均验证分数: {avg_verification:.3f})")
    
    # 分析不同策略的表现
    print(f"\n📈 策略表现分析:")
    for strategy, stats in strategy_stats.items():
        accuracy_pct = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {strategy}: {stats['correct']}/{stats['total']} ({accuracy_pct:.1f}%)")
    
    # 分析不同复杂度的表现
    print(f"\n📊 复杂度表现分析:")
    for complexity, stats in complexity_stats.items():
        accuracy_pct = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {complexity}: {stats['correct']}/{stats['total']} ({accuracy_pct:.1f}%)")
    
    # 显示错误题目
    wrong_problems = [r for r in results if not r['correct']]
    if wrong_problems:
        print(f"\n❌ 错误题目分析 ({len(wrong_problems)}道):")
        for prob in wrong_problems[:5]:  # 只显示前5个
            print(f"  - {prob['problem_id']}: 期望 {prob['expected']}, 得到 {prob['predicted']} (策略: {prob['strategy']})")
        if len(wrong_problems) > 5:
            print(f"  ... 还有 {len(wrong_problems) - 5} 道错误题目")
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"critical_fixes_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': {
                'timestamp': timestamp,
                'num_problems': total,
                'start_from': start_from,
                'correct_count': correct,
                'accuracy': accuracy,
                'error_count': error_count,
                'none_answers': none_answers
            },
            'strategy_stats': strategy_stats,
            'complexity_stats': complexity_stats,
            'core_issues_status': {
                'system_stability': error_count == 0 and none_answers == 0,
                'strategy_identification': 'unknown' not in [r['strategy'] for r in results],
                'verification_mechanism': avg_verification > 0.1,
                'avg_verification_score': avg_verification
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {filename}")
    
    # 给出改进建议
    if accuracy >= 80:
        print("\n🎉 系统表现优秀！四个核心问题基本解决")
    elif accuracy >= 60:
        print("\n👍 系统表现良好，核心问题有显著改善")
    elif accuracy >= 40:
        print("\n⚠️  系统有改进，但仍需进一步优化")
    else:
        print("\n💀 系统仍需要大幅改进")
    
    return accuracy, strategy_stats, complexity_stats

def main():
    """主函数"""
    
    # 默认参数
    num_problems = 20
    start_from = 0
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        try:
            num_problems = int(sys.argv[1])
        except ValueError:
            print("❌ 题目数量必须是整数")
            return
    
    if len(sys.argv) > 2:
        try:
            start_from = int(sys.argv[2])
        except ValueError:
            print("❌ 起始位置必须是整数")
            return
    
    # 运行测试
    accuracy, strategy_stats, complexity_stats = test_critical_fixes_comprehensive(num_problems, start_from)
    
    return accuracy

if __name__ == "__main__":
    print("🔧 关键修复数学推理系统测试工具")
    print("用法: python test_critical_fixes.py [题目数量] [起始位置]")
    print("示例: python test_critical_fixes.py 30 10")
    print()
    
    main() 