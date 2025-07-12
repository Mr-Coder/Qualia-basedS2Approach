"""
🔗 COT-DIR 全量关系解答生成器
Full Relation Generator - 为全部14,097道题目生成基于关系的解答

核心功能：
- 显性关系识别
- L1/L2/L3隐含关系推理
- 关系链分析
- 完整解答生成
"""

import concurrent.futures
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 复用关系生成器的核心逻辑
from relation_based_solution_generator import RelationBasedSolutionGenerator


class FullRelationGenerator(RelationBasedSolutionGenerator):
    """全量关系解答生成器"""
    
    def __init__(self):
        """初始化全量关系生成器"""
        super().__init__()
        print("🚀 初始化COT-DIR全量关系解答生成器")
        print("🎯 目标: 为全部14,097道题目生成基于关系的解答")
    
    def process_all_problems_with_relations(self, use_parallel: bool = True, max_workers: int = 8):
        """处理所有题目并生成关系解答"""
        print("🔗 开始全量关系解答生成...")
        
        # 加载原始解答数据
        solution_files = list(Path(".").glob("maximum_solutions_*.json"))
        if not solution_files:
            print("❌ 未找到原始解答文件")
            return []
        
        latest_file = max(solution_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 加载原始解答文件: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            original_solutions = data.get('solutions', [])
        
        print(f"📊 开始为{len(original_solutions):,}个问题生成基于关系的解答...")
        
        start_time = time.time()
        relation_solutions = []
        
        if use_parallel and len(original_solutions) > 100:
            print(f"⚡ 使用{max_workers}个并行进程处理")
            
            # 准备问题数据
            problems = []
            for solution in original_solutions:
                problem = {
                    'problem_id': solution.get('problem_id'),
                    'question': solution.get('question'),
                    'answer': solution.get('final_answer'),
                    'dataset_source': solution.get('dataset_source')
                }
                problems.append(problem)
            
            # 并行处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_problem = {executor.submit(self.generate_relation_based_solution, problem): problem 
                                   for problem in problems}
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_problem):
                    relation_solution = future.result()
                    relation_solutions.append(relation_solution)
                    completed += 1
                    
                    if completed % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (len(problems) - completed) / rate if rate > 0 else 0
                        print(f"   进度: {completed:,}/{len(problems):,} ({completed/len(problems)*100:.1f}%) - "
                              f"速度: {rate:.0f}题/秒 - 预计剩余: {eta:.0f}秒")
        else:
            print("🔄 使用串行处理")
            for i, solution in enumerate(original_solutions):
                problem = {
                    'problem_id': solution.get('problem_id'),
                    'question': solution.get('question'),
                    'answer': solution.get('final_answer'),
                    'dataset_source': solution.get('dataset_source')
                }
                
                relation_solution = self.generate_relation_based_solution(problem)
                relation_solutions.append(relation_solution)
                
                if (i + 1) % 1000 == 0:
                    print(f"   已完成: {i + 1}/{len(original_solutions)} 题")
        
        total_time = time.time() - start_time
        
        self.processing_stats = {
            'total_processed': len(relation_solutions),
            'successful_solutions': sum(1 for s in relation_solutions if s.confidence_score > 0),
            'total_time': total_time,
            'avg_time_per_problem': total_time / len(relation_solutions) if relation_solutions else 0,
            'processing_rate': len(relation_solutions) / total_time if total_time > 0 else 0
        }
        
        self.generated_solutions = relation_solutions
        
        print(f"✅ 全量关系解答生成完成!")
        self._print_comprehensive_relation_summary()
        
        return relation_solutions
    
    def _print_comprehensive_relation_summary(self):
        """打印全面的关系分析摘要"""
        stats = self.processing_stats
        
        print(f"\n🔗 全量关系解答生成摘要:")
        print("=" * 80)
        print(f"🔢 总处理题目: {stats['total_processed']:,} 题")
        print(f"✅ 成功生成解答: {stats['successful_solutions']:,} 题")
        print(f"📈 成功率: {stats['successful_solutions']/stats['total_processed']*100:.1f}%")
        print(f"⏱️  总处理时间: {stats['total_time']:.1f} 秒")
        print(f"⚡ 平均处理速度: {stats['processing_rate']:.0f} 题/秒")
        print(f"🎯 平均每题时间: {stats['avg_time_per_problem']*1000:.2f} 毫秒")
        
        # 关系统计
        total_explicit = sum(len(s.explicit_relations) for s in self.generated_solutions)
        total_L1 = sum(len(s.implicit_relations_L1) for s in self.generated_solutions)
        total_L2 = sum(len(s.implicit_relations_L2) for s in self.generated_solutions)
        total_L3 = sum(len(s.implicit_relations_L3) for s in self.generated_solutions)
        total_relations = total_explicit + total_L1 + total_L2 + total_L3
        
        print(f"\n🔍 关系发现总体统计:")
        print(f"   显性关系: {total_explicit:,} 个 ({total_explicit/total_relations*100:.1f}%)")
        print(f"   L1隐含关系: {total_L1:,} 个 ({total_L1/total_relations*100:.1f}%)")
        print(f"   L2隐含关系: {total_L2:,} 个 ({total_L2/total_relations*100:.1f}%)")
        print(f"   L3隐含关系: {total_L3:,} 个 ({total_L3/total_relations*100:.1f}%)")
        print(f"   关系总数: {total_relations:,} 个")
        print(f"   平均每题关系数: {total_relations/len(self.generated_solutions):.1f} 个")
        
        # 层次分布详细统计
        L1_problems = sum(1 for s in self.generated_solutions if len(s.implicit_relations_L1) > 0)
        L2_problems = sum(1 for s in self.generated_solutions if len(s.implicit_relations_L2) > 0)
        L3_problems = sum(1 for s in self.generated_solutions if len(s.implicit_relations_L3) > 0)
        
        print(f"\n📊 关系层次覆盖统计:")
        print(f"   涉及L1关系的题目: {L1_problems:,} ({L1_problems/len(self.generated_solutions)*100:.1f}%)")
        print(f"   涉及L2关系的题目: {L2_problems:,} ({L2_problems/len(self.generated_solutions)*100:.1f}%)")
        print(f"   涉及L3关系的题目: {L3_problems:,} ({L3_problems/len(self.generated_solutions)*100:.1f}%)")
        
        # 复杂度分布
        complex_problems = sum(1 for s in self.generated_solutions 
                             if len(s.implicit_relations_L1) > 0 and len(s.implicit_relations_L2) > 0)
        advanced_problems = sum(1 for s in self.generated_solutions 
                              if len(s.implicit_relations_L1) > 0 and len(s.implicit_relations_L2) > 0 and len(s.implicit_relations_L3) > 0)
        
        print(f"\n🌟 关系复杂度分布:")
        print(f"   复杂题目 (L1+L2): {complex_problems:,} ({complex_problems/len(self.generated_solutions)*100:.1f}%)")
        print(f"   高级题目 (L1+L2+L3): {advanced_problems:,} ({advanced_problems/len(self.generated_solutions)*100:.1f}%)")
        
        # 按数据集的关系分析
        dataset_stats = {}
        for solution in self.generated_solutions:
            dataset = solution.dataset_source
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {
                    'count': 0, 'total_relations': 0, 'L1_count': 0, 'L2_count': 0, 'L3_count': 0
                }
            
            stats_item = dataset_stats[dataset]
            stats_item['count'] += 1
            stats_item['total_relations'] += (len(solution.explicit_relations) + 
                                            len(solution.implicit_relations_L1) + 
                                            len(solution.implicit_relations_L2) + 
                                            len(solution.implicit_relations_L3))
            if len(solution.implicit_relations_L1) > 0:
                stats_item['L1_count'] += 1
            if len(solution.implicit_relations_L2) > 0:
                stats_item['L2_count'] += 1
            if len(solution.implicit_relations_L3) > 0:
                stats_item['L3_count'] += 1
        
        print(f"\n📈 按数据集关系分析 (Top 8):")
        sorted_datasets = sorted(dataset_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:8]
        for dataset, stats_item in sorted_datasets:
            avg_relations = stats_item['total_relations'] / stats_item['count']
            L3_rate = stats_item['L3_count'] / stats_item['count'] * 100
            print(f"   {dataset}: {stats_item['count']:,}题, 平均{avg_relations:.1f}关系/题, L3覆盖{L3_rate:.1f}%")
    
    def save_full_relation_solutions(self, output_file: str = None):
        """保存全量关系解答"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"full_relation_solutions_{timestamp}.json"
        
        print(f"💾 保存{len(self.generated_solutions):,}个基于关系的解答到 {output_file}...")
        
        solutions_data = []
        for solution in self.generated_solutions:
            solutions_data.append({
                'problem_id': solution.problem_id,
                'question': solution.question,
                'problem_type': solution.problem_type,
                'explicit_relations': solution.explicit_relations,
                'implicit_relations_L1': solution.implicit_relations_L1,
                'implicit_relations_L2': solution.implicit_relations_L2,
                'implicit_relations_L3': solution.implicit_relations_L3,
                'relation_discovery_steps': solution.relation_discovery_steps,
                'relation_reasoning_chain': solution.relation_reasoning_chain,
                'relation_based_solution_steps': solution.relation_based_solution_steps,
                'mathematical_analysis': solution.mathematical_analysis,
                'final_answer': solution.final_answer,
                'verification_process': solution.verification_process,
                'confidence_score': solution.confidence_score,
                'processing_time': solution.processing_time,
                'dataset_source': solution.dataset_source,
                'generated_at': datetime.now().isoformat()
            })
        
        # 计算关系统计信息
        relation_summary = {
            'total_explicit_relations': sum(len(s.explicit_relations) for s in self.generated_solutions),
            'total_L1_relations': sum(len(s.implicit_relations_L1) for s in self.generated_solutions),
            'total_L2_relations': sum(len(s.implicit_relations_L2) for s in self.generated_solutions),
            'total_L3_relations': sum(len(s.implicit_relations_L3) for s in self.generated_solutions),
            'problems_with_L1': sum(1 for s in self.generated_solutions if len(s.implicit_relations_L1) > 0),
            'problems_with_L2': sum(1 for s in self.generated_solutions if len(s.implicit_relations_L2) > 0),
            'problems_with_L3': sum(1 for s in self.generated_solutions if len(s.implicit_relations_L3) > 0),
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'generator_type': 'full_relation_based',
                    'total_solutions': len(solutions_data),
                    'generation_stats': self.processing_stats,
                    'relation_summary': relation_summary,
                    'generated_at': datetime.now().isoformat(),
                    'description': 'COT-DIR全量基于关系的解答生成结果 - 突出显性关系和L1/L2/L3隐含关系推理'
                },
                'solutions': solutions_data
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存 {len(solutions_data):,} 个基于关系的解答")
        print(f"📁 文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
        
        return output_file

def main():
    """主函数"""
    print("🔗 COT-DIR 全量关系解答生成系统")
    print("=" * 80)
    print("🎯 目标: 为全部14,097道题目生成基于关系的解答")
    print("🧠 核心: 显性关系 + L1/L2/L3隐含关系推理")
    print("⚡ 特点: 并行处理、关系链分析、多层推理")
    print("=" * 80)
    
    generator = FullRelationGenerator()
    
    # 处理全部题目
    print("🚀 开始处理全部14,097道题目...")
    solutions = generator.process_all_problems_with_relations(use_parallel=True, max_workers=8)
    
    # 保存结果
    output_file = generator.save_full_relation_solutions()
    
    # 显示最终统计
    print(f"\n🎉 全量关系解答生成任务完成!")
    print(f"📊 总共生成了 {len(solutions):,} 个基于关系的详细解答")
    print(f"📁 输出文件: {output_file}")
    print(f"🏆 这是COT-DIR系统完整的关系推理能力展示")

if __name__ == "__main__":
    main() 