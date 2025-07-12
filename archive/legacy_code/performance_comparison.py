#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数学问题求解器性能对比
====================

对比原版和优化版求解器的性能差异
"""

import json
import time
from typing import Any, Dict, List

from math_problem_solver import MathProblemSolver
from math_problem_solver_optimized import OptimizedMathSolver


def run_performance_comparison():
    """运行性能对比测试"""
    
    # 测试问题集
    test_problems = [
        {
            'id': 1,
            'problem': "A rectangle has length 8 cm and width 5 cm. What is its area?",
            'expected_answer': 40.0,
            'type': 'geometry'
        },
        {
            'id': 2,
            'problem': "A car travels 120 km in 2 hours. What is its average speed?",
            'expected_answer': 60.0,
            'type': 'motion'
        },
        {
            'id': 3,
            'problem': "What is 15 plus 27?",
            'expected_answer': 42.0,
            'type': 'arithmetic'
        },
        {
            'id': 4,
            'problem': "A circle has radius 3 cm. What is its area?",
            'expected_answer': 28.27,
            'type': 'geometry'
        },
        {
            'id': 5,
            'problem': "Ice cubes, each with a volume of 200 cm³, are dropped into a tank containing 5 L of water at a rate of one cube per minute. Simultaneously, water is leaking from the tank at 2 mL/s. How long will it take for the water level to rise to 9 L?",
            'expected_answer': 50.0,
            'type': 'tank'
        },
        {
            'id': 6,
            'problem': "What is 100 divided by 4?",
            'expected_answer': 25.0,
            'type': 'arithmetic'
        },
        {
            'id': 7,
            'problem': "A square has side length 6 cm. What is its perimeter?",
            'expected_answer': 24.0,
            'type': 'geometry'
        }
    ]
    
    print("=" * 80)
    print("数学问题求解器性能对比报告")
    print("=" * 80)
    
    # 初始化求解器
    print("\n🔧 初始化求解器...")
    
    try:
        original_solver = MathProblemSolver()
        print("✅ 原版求解器初始化成功")
    except Exception as e:
        print(f"❌ 原版求解器初始化失败: {e}")
        original_solver = None
    
    optimized_solver = OptimizedMathSolver()
    print("✅ 优化版求解器初始化成功")
    
    # 运行测试
    results = {
        'original': [],
        'optimized': [],
        'summary': {}
    }
    
    print(f"\n📊 开始测试 {len(test_problems)} 个问题...")
    print("-" * 80)
    
    for problem in test_problems:
        print(f"\n问题 {problem['id']}: {problem['problem'][:60]}...")
        print(f"类型: {problem['type']} | 期望答案: {problem['expected_answer']}")
        
        # 测试原版求解器
        if original_solver:
            try:
                start_time = time.time()
                original_result = original_solver.solve(problem['problem'])
                original_time = time.time() - start_time
                
                original_answer = original_result.get('answer')
                original_status = original_result.get('status', 'unknown')
                
                # 判断正确性
                is_correct = False
                if original_answer is not None:
                    if abs(float(original_answer) - problem['expected_answer']) < 0.1:
                        is_correct = True
                
                results['original'].append({
                    'problem_id': problem['id'],
                    'answer': original_answer,
                    'expected': problem['expected_answer'],
                    'correct': is_correct,
                    'time': original_time,
                    'status': original_status
                })
                
                print(f"  原版: {original_answer} ({'✅' if is_correct else '❌'}) - {original_time:.3f}s")
                
            except Exception as e:
                results['original'].append({
                    'problem_id': problem['id'],
                    'answer': None,
                    'expected': problem['expected_answer'],
                    'correct': False,
                    'time': 0,
                    'status': 'error',
                    'error': str(e)
                })
                print(f"  原版: 错误 - {e}")
        else:
            print("  原版: 跳过（初始化失败）")
        
        # 测试优化版求解器
        try:
            start_time = time.time()
            optimized_result = optimized_solver.solve(problem['problem'])
            optimized_time = time.time() - start_time
            
            optimized_answer = optimized_result.get('answer')
            optimized_status = optimized_result.get('status', 'unknown')
            
            # 判断正确性
            is_correct = False
            if optimized_answer is not None:
                if abs(float(optimized_answer) - problem['expected_answer']) < 0.1:
                    is_correct = True
            
            results['optimized'].append({
                'problem_id': problem['id'],
                'answer': optimized_answer,
                'expected': problem['expected_answer'],
                'correct': is_correct,
                'time': optimized_time,
                'status': optimized_status
            })
            
            print(f"  优化版: {optimized_answer} ({'✅' if is_correct else '❌'}) - {optimized_time:.3f}s")
            
        except Exception as e:
            results['optimized'].append({
                'problem_id': problem['id'],
                'answer': None,
                'expected': problem['expected_answer'],
                'correct': False,
                'time': 0,
                'status': 'error',
                'error': str(e)
            })
            print(f"  优化版: 错误 - {e}")
    
    # 生成汇总报告
    print("\n" + "=" * 80)
    print("📈 性能汇总报告")
    print("=" * 80)
    
    # 原版统计
    if results['original']:
        original_correct = sum(1 for r in results['original'] if r['correct'])
        original_total_time = sum(r['time'] for r in results['original'])
        original_avg_time = original_total_time / len(results['original'])
        original_success_rate = original_correct / len(results['original']) * 100
        
        print(f"\n🔵 原版求解器:")
        print(f"  正确率: {original_correct}/{len(results['original'])} ({original_success_rate:.1f}%)")
        print(f"  总耗时: {original_total_time:.3f}s")
        print(f"  平均耗时: {original_avg_time:.3f}s")
    else:
        print(f"\n🔵 原版求解器: 无法测试")
    
    # 优化版统计
    optimized_correct = sum(1 for r in results['optimized'] if r['correct'])
    optimized_total_time = sum(r['time'] for r in results['optimized'])
    optimized_avg_time = optimized_total_time / len(results['optimized'])
    optimized_success_rate = optimized_correct / len(results['optimized']) * 100
    
    print(f"\n🟢 优化版求解器:")
    print(f"  正确率: {optimized_correct}/{len(results['optimized'])} ({optimized_success_rate:.1f}%)")
    print(f"  总耗时: {optimized_total_time:.3f}s")
    print(f"  平均耗时: {optimized_avg_time:.3f}s")
    
    # 性能提升
    if results['original']:
        accuracy_improvement = optimized_success_rate - original_success_rate
        speed_improvement = (original_avg_time - optimized_avg_time) / original_avg_time * 100 if original_avg_time > 0 else 0
        
        print(f"\n🚀 性能提升:")
        print(f"  准确率提升: {accuracy_improvement:+.1f}%")
        print(f"  速度提升: {speed_improvement:+.1f}%")
    
    # 按问题类型分析
    print(f"\n📊 按问题类型分析:")
    problem_types = set(p['type'] for p in test_problems)
    
    for ptype in problem_types:
        type_problems = [p for p in test_problems if p['type'] == ptype]
        type_results = [r for r in results['optimized'] if r['problem_id'] in [p['id'] for p in type_problems]]
        
        if type_results:
            type_correct = sum(1 for r in type_results if r['correct'])
            type_success_rate = type_correct / len(type_results) * 100
            type_avg_time = sum(r['time'] for r in type_results) / len(type_results)
            
            print(f"  {ptype}: {type_correct}/{len(type_results)} ({type_success_rate:.1f}%) - 平均 {type_avg_time:.3f}s")
    
    # 保存详细结果
    with open('performance_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 详细结果已保存到 performance_comparison_results.json")
    
    return results

if __name__ == "__main__":
    run_performance_comparison() 