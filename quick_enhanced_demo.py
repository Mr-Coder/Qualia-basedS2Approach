#!/usr/bin/env python3
"""
快速增强版演示 - Day 1 & Day 2 优化展示
"""

import json

from enhanced_case_results_generator import EnhancedCaseResultsGenerator


def main():
    print("🎯 快速演示：Day 1 & Day 2 优化成果")
    print("=" * 50)
    
    # 创建生成器 - 小规模测试
    generator = EnhancedCaseResultsGenerator(
        dataset_names=['Math23K', 'GSM8K'],  # 2个数据集
        sample_size_per_dataset=5,           # 每个5题
        total_target_problems=10             # 总共10题
    )
    
    print("\n📊 第一步：动态加载测试用例")
    test_cases = generator.load_dynamic_test_cases()
    
    if test_cases:
        print(f"\n✅ 成功加载 {len(test_cases)} 个测试用例")
        
        # 显示加载的题目样例
        print("\n📝 题目样例展示:")
        for i, case in enumerate(test_cases[:3]):
            print(f"\n🔹 样例 {i+1}:")
            print(f"   ID: {case['id']}")
            print(f"   数据集: {case['source']}")
            print(f"   语言: {case['language']}")
            print(f"   类型: {case['type']}")
            print(f"   难度: {case['difficulty']}")
            print(f"   复杂度: {case['complexity_level']}")
            print(f"   问题: {case['problem'][:80]}...")
            print(f"   答案: {case['expected_answer']}")
        
        print(f"\n🎯 第二步：生成详细推理结果")
        
        # 只处理前3题作为演示
        print("📝 处理前3题进行演示...")
        demo_cases = test_cases[:3]
        
        detailed_results = []
        for i, case in enumerate(demo_cases):
            print(f"\n🔍 处理题目 {i+1}: {case['id']}")
            
            try:
                result = generator._process_single_case(case)
                if result:
                    detailed_results.append(result)
                    
                    # 显示处理结果
                    print(f"   ✅ 推理完成")
                    print(f"   预测答案: {result['final_result']['predicted_answer']}")
                    print(f"   正确答案: {result['final_result']['expected_answer']}")
                    print(f"   是否正确: {'✅' if result['final_result']['is_correct'] else '❌'}")
                    print(f"   置信度: {result['final_result']['confidence_score']:.1f}%")
                    print(f"   质量评分: {result['quality_assessment']['overall_score']}/10")
                    print(f"   质量等级: {result['quality_assessment']['grade']}")
                    
                    # 显示解题过程
                    solution = result['solution_process']
                    print(f"   📋 解题分析: {solution['problem_analysis']}")
                    print(f"   🔄 解题步骤: {len(solution['solution_steps'])}步")
                    
            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
        
        # 统计结果
        if detailed_results:
            print(f"\n📊 演示结果统计:")
            total_demo = len(detailed_results)
            correct_demo = sum(1 for r in detailed_results if r['final_result']['is_correct'])
            accuracy_demo = correct_demo / total_demo * 100 if total_demo > 0 else 0
            
            avg_confidence = sum(r['final_result']['confidence_score'] for r in detailed_results) / total_demo
            avg_quality = sum(r['quality_assessment']['overall_score'] for r in detailed_results) / total_demo
            
            print(f"   处理题目: {total_demo} 题")
            print(f"   正确率: {accuracy_demo:.1f}% ({correct_demo}/{total_demo})")
            print(f"   平均置信度: {avg_confidence:.1f}%")
            print(f"   平均质量分: {avg_quality:.1f}/10")
            
            # 按数据集统计
            dataset_stats = {}
            for result in detailed_results:
                dataset = result['case_info']['source_dataset']
                if dataset not in dataset_stats:
                    dataset_stats[dataset] = {'total': 0, 'correct': 0}
                dataset_stats[dataset]['total'] += 1
                if result['final_result']['is_correct']:
                    dataset_stats[dataset]['correct'] += 1
            
            print(f"\n   按数据集分布:")
            for dataset, stats in dataset_stats.items():
                acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"     {dataset}: {stats['total']}题 (正确率{acc:.1f}%)")
        
        print(f"\n🎉 演示完成！")
        print("💡 提示：运行 enhanced_case_results_generator.py 可以生成完整的30题结果")
        
    else:
        print("❌ 没有加载到测试用例")


if __name__ == "__main__":
    main() 