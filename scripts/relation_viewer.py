"""
🔍 COT-DIR 关系分析查看器
Relation Viewer - 专门展示和分析基于关系的解答过程

核心功能：
- 显性关系 + L1/L2/L3隐含关系展示
- 关系推理链可视化
- 层次推理分析
- 关系统计报告
"""

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class RelationViewer:
    """关系分析查看器"""
    
    def __init__(self, solution_file: str = None):
        """初始化关系查看器"""
        print("🔍 初始化COT-DIR关系分析查看器")
        
        if solution_file is None:
            # 自动查找最新的关系解答文件
            relation_files = list(Path(".").glob("*relation_solutions_*.json"))
            if not relation_files:
                print("❌ 未找到关系解答文件")
                self.solutions = []
                self.metadata = {}
                return
            
            latest_file = max(relation_files, key=lambda p: p.stat().st_mtime)
            solution_file = str(latest_file)
        
        print(f"📁 加载关系解答文件: {solution_file}")
        
        with open(solution_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.solutions = data.get('solutions', [])
            self.metadata = data.get('metadata', {})
        
        print(f"📊 已加载 {len(self.solutions):,} 个基于关系的解答")
        self._print_loading_summary()
    
    def _print_loading_summary(self):
        """打印加载摘要"""
        if not self.solutions:
            return
        
        # 基本统计
        total_relations = sum(
            len(s.get('explicit_relations', [])) + 
            len(s.get('implicit_relations_L1', [])) + 
            len(s.get('implicit_relations_L2', [])) + 
            len(s.get('implicit_relations_L3', []))
            for s in self.solutions
        )
        
        print(f"🔗 关系总数: {total_relations:,} 个")
        print(f"📈 平均每题关系数: {total_relations/len(self.solutions):.1f} 个")
        
        # 层次统计
        L1_count = sum(1 for s in self.solutions if len(s.get('implicit_relations_L1', [])) > 0)
        L2_count = sum(1 for s in self.solutions if len(s.get('implicit_relations_L2', [])) > 0)
        L3_count = sum(1 for s in self.solutions if len(s.get('implicit_relations_L3', [])) > 0)
        
        print(f"🧠 L1关系覆盖: {L1_count} 题 ({L1_count/len(self.solutions)*100:.1f}%)")
        print(f"🔗 L2关系覆盖: {L2_count} 题 ({L2_count/len(self.solutions)*100:.1f}%)")
        print(f"🌟 L3关系覆盖: {L3_count} 题 ({L3_count/len(self.solutions)*100:.1f}%)")
    
    def show_relation_examples(self, num_examples: int = 5):
        """展示关系推理示例"""
        print(f"\n🔍 关系推理示例展示 (随机选择{num_examples}个)")
        print("=" * 100)
        
        # 随机选择有丰富关系的题目
        rich_solutions = [
            s for s in self.solutions 
            if (len(s.get('explicit_relations', [])) + 
                len(s.get('implicit_relations_L1', [])) + 
                len(s.get('implicit_relations_L2', [])) + 
                len(s.get('implicit_relations_L3', []))) >= 3
        ]
        
        examples = random.sample(rich_solutions, min(num_examples, len(rich_solutions)))
        
        for i, solution in enumerate(examples, 1):
            print(f"\n📋 示例 {i}: {solution.get('problem_id', 'unknown')}")
            print(f"💬 题目: {solution.get('question', '')[:150]}...")
            print(f"📊 题目类型: {solution.get('problem_type', 'unknown')}")
            print(f"📁 数据集: {solution.get('dataset_source', 'unknown')}")
            
            # 显性关系
            explicit_relations = solution.get('explicit_relations', [])
            if explicit_relations:
                print(f"\n🔍 显性关系 ({len(explicit_relations)}个):")
                for j, rel in enumerate(explicit_relations[:3], 1):
                    print(f"   {j}. {rel.get('description', '')}")
                    if 'evidence' in rel:
                        print(f"      证据: {rel['evidence']}")
            
            # L1隐含关系
            L1_relations = solution.get('implicit_relations_L1', [])
            if L1_relations:
                print(f"\n🧠 L1隐含关系 ({len(L1_relations)}个):")
                for j, rel in enumerate(L1_relations[:3], 1):
                    print(f"   {j}. {rel.get('description', '')}")
                    print(f"      推理: {rel.get('reasoning', '')}")
                    print(f"      数学含义: {rel.get('mathematical_implication', '')}")
            
            # L2隐含关系
            L2_relations = solution.get('implicit_relations_L2', [])
            if L2_relations:
                print(f"\n🔗 L2隐含关系 ({len(L2_relations)}个):")
                for j, rel in enumerate(L2_relations[:2], 1):
                    print(f"   {j}. {rel.get('description', '')}")
                    print(f"      推理: {rel.get('reasoning', '')}")
                    print(f"      数学含义: {rel.get('mathematical_implication', '')}")
                    if 'dependency' in rel:
                        print(f"      依赖关系: {rel['dependency']}")
            
            # L3隐含关系
            L3_relations = solution.get('implicit_relations_L3', [])
            if L3_relations:
                print(f"\n🌟 L3隐含关系 ({len(L3_relations)}个):")
                for j, rel in enumerate(L3_relations[:2], 1):
                    print(f"   {j}. {rel.get('description', '')}")
                    print(f"      推理: {rel.get('reasoning', '')}")
                    print(f"      数学含义: {rel.get('mathematical_implication', '')}")
                    if 'dependency' in rel:
                        print(f"      关系链: {rel['dependency']}")
            
            # 关系推理链
            reasoning_chain = solution.get('relation_reasoning_chain', [])
            if reasoning_chain:
                print(f"\n🔄 关系推理链:")
                for step in reasoning_chain[:5]:
                    print(f"   • {step}")
            
            # 基于关系的解题步骤
            solution_steps = solution.get('relation_based_solution_steps', [])
            if solution_steps:
                print(f"\n🎯 基于关系的解题过程:")
                for step in solution_steps[:5]:
                    print(f"   • {step}")
            
            print(f"\n📈 置信度: {solution.get('confidence_score', 0):.2f}")
            print(f"💡 最终答案: {solution.get('final_answer', '')}")
            print("─" * 100)
    
    def analyze_relation_patterns(self):
        """分析关系模式"""
        print(f"\n🔬 关系模式深度分析")
        print("=" * 80)
        
        # 显性关系类型统计
        explicit_types = {}
        L1_types = {}
        L2_types = {}
        L3_types = {}
        
        for solution in self.solutions:
            # 统计显性关系类型
            for rel in solution.get('explicit_relations', []):
                rel_type = rel.get('type', 'unknown')
                explicit_types[rel_type] = explicit_types.get(rel_type, 0) + 1
            
            # 统计L1关系类型
            for rel in solution.get('implicit_relations_L1', []):
                rel_type = rel.get('type', 'unknown')
                L1_types[rel_type] = L1_types.get(rel_type, 0) + 1
            
            # 统计L2关系类型
            for rel in solution.get('implicit_relations_L2', []):
                rel_type = rel.get('type', 'unknown')
                L2_types[rel_type] = L2_types.get(rel_type, 0) + 1
            
            # 统计L3关系类型
            for rel in solution.get('implicit_relations_L3', []):
                rel_type = rel.get('type', 'unknown')
                L3_types[rel_type] = L3_types.get(rel_type, 0) + 1
        
        # 打印显性关系统计
        print(f"\n🔍 显性关系类型分布:")
        sorted_explicit = sorted(explicit_types.items(), key=lambda x: x[1], reverse=True)
        for rel_type, count in sorted_explicit:
            percentage = count / sum(explicit_types.values()) * 100
            print(f"   {rel_type}: {count:,} 次 ({percentage:.1f}%)")
        
        # 打印L1关系统计
        if L1_types:
            print(f"\n🧠 L1隐含关系类型分布:")
            sorted_L1 = sorted(L1_types.items(), key=lambda x: x[1], reverse=True)
            for rel_type, count in sorted_L1:
                percentage = count / sum(L1_types.values()) * 100
                print(f"   {rel_type}: {count:,} 次 ({percentage:.1f}%)")
        
        # 打印L2关系统计
        if L2_types:
            print(f"\n🔗 L2隐含关系类型分布:")
            sorted_L2 = sorted(L2_types.items(), key=lambda x: x[1], reverse=True)
            for rel_type, count in sorted_L2:
                percentage = count / sum(L2_types.values()) * 100
                print(f"   {rel_type}: {count:,} 次 ({percentage:.1f}%)")
        
        # 打印L3关系统计
        if L3_types:
            print(f"\n🌟 L3隐含关系类型分布:")
            sorted_L3 = sorted(L3_types.items(), key=lambda x: x[1], reverse=True)
            for rel_type, count in sorted_L3:
                percentage = count / sum(L3_types.values()) * 100
                print(f"   {rel_type}: {count:,} 次 ({percentage:.1f}%)")
    
    def analyze_dataset_relations(self):
        """按数据集分析关系分布"""
        print(f"\n📊 按数据集关系分析")
        print("=" * 80)
        
        dataset_stats = {}
        
        for solution in self.solutions:
            dataset = solution.get('dataset_source', 'unknown')
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {
                    'count': 0,
                    'explicit_count': 0,
                    'L1_count': 0,
                    'L2_count': 0,
                    'L3_count': 0,
                    'total_relations': 0,
                    'avg_confidence': 0
                }
            
            stats = dataset_stats[dataset]
            stats['count'] += 1
            stats['explicit_count'] += len(solution.get('explicit_relations', []))
            stats['L1_count'] += len(solution.get('implicit_relations_L1', []))
            stats['L2_count'] += len(solution.get('implicit_relations_L2', []))
            stats['L3_count'] += len(solution.get('implicit_relations_L3', []))
            stats['total_relations'] += (stats['explicit_count'] + stats['L1_count'] + 
                                       stats['L2_count'] + stats['L3_count'])
            stats['avg_confidence'] += solution.get('confidence_score', 0)
        
        # 计算平均值
        for dataset, stats in dataset_stats.items():
            if stats['count'] > 0:
                stats['avg_confidence'] /= stats['count']
                stats['avg_relations'] = stats['total_relations'] / stats['count']
        
        # 按题目数量排序显示
        sorted_datasets = sorted(dataset_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print(f"{'数据集':<15} {'题目数':<8} {'平均关系':<10} {'L1%':<8} {'L2%':<8} {'L3%':<8} {'置信度':<8}")
        print("-" * 80)
        
        for dataset, stats in sorted_datasets[:12]:  # 显示前12个数据集
            L1_rate = (stats['L1_count'] / stats['count']) * 100 if stats['count'] > 0 else 0
            L2_rate = (stats['L2_count'] / stats['count']) * 100 if stats['count'] > 0 else 0
            L3_rate = (stats['L3_count'] / stats['count']) * 100 if stats['count'] > 0 else 0
            
            print(f"{dataset:<15} {stats['count']:<8} {stats['avg_relations']:<10.1f} "
                  f"{L1_rate:<8.1f} {L2_rate:<8.1f} {L3_rate:<8.1f} {stats['avg_confidence']:<8.2f}")
    
    def find_complex_problems(self, min_relations: int = 5):
        """查找复杂关系题目"""
        print(f"\n🌟 复杂关系题目分析 (关系数≥{min_relations})")
        print("=" * 80)
        
        complex_problems = []
        
        for solution in self.solutions:
            total_relations = (
                len(solution.get('explicit_relations', [])) + 
                len(solution.get('implicit_relations_L1', [])) + 
                len(solution.get('implicit_relations_L2', [])) + 
                len(solution.get('implicit_relations_L3', []))
            )
            
            if total_relations >= min_relations:
                complex_problems.append({
                    'solution': solution,
                    'total_relations': total_relations,
                    'L3_relations': len(solution.get('implicit_relations_L3', []))
                })
        
        # 按关系总数排序
        complex_problems.sort(key=lambda x: (x['L3_relations'], x['total_relations']), reverse=True)
        
        print(f"🔍 找到 {len(complex_problems)} 个复杂关系题目")
        
        # 显示前10个最复杂的题目
        print(f"\n🏆 Top 10 最复杂关系题目:")
        for i, item in enumerate(complex_problems[:10], 1):
            solution = item['solution']
            print(f"\n{i}. 题目ID: {solution.get('problem_id', 'unknown')}")
            print(f"   数据集: {solution.get('dataset_source', 'unknown')}")
            print(f"   题目: {solution.get('question', '')[:100]}...")
            print(f"   关系统计: 总{item['total_relations']}个 "
                  f"(显性{len(solution.get('explicit_relations', []))} "
                  f"L1:{len(solution.get('implicit_relations_L1', []))} "
                  f"L2:{len(solution.get('implicit_relations_L2', []))} "
                  f"L3:{len(solution.get('implicit_relations_L3', []))})")
            print(f"   置信度: {solution.get('confidence_score', 0):.2f}")
        
        return complex_problems
    
    def show_L3_examples(self):
        """展示L3关系示例"""
        print(f"\n🌟 L3隐含关系详细示例")
        print("=" * 80)
        
        L3_solutions = [s for s in self.solutions if len(s.get('implicit_relations_L3', [])) > 0]
        
        if not L3_solutions:
            print("❌ 未找到包含L3关系的题目")
            return
        
        print(f"🔍 找到 {len(L3_solutions)} 个包含L3关系的题目")
        
        # 显示前5个L3关系示例
        for i, solution in enumerate(L3_solutions[:5], 1):
            print(f"\n🌟 L3示例 {i}:")
            print(f"   题目: {solution.get('question', '')[:120]}...")
            print(f"   数据集: {solution.get('dataset_source', 'unknown')}")
            
            for j, rel in enumerate(solution.get('implicit_relations_L3', []), 1):
                print(f"\n   L3关系 {j}: {rel.get('description', '')}")
                print(f"   抽象推理: {rel.get('reasoning', '')}")
                print(f"   数学含义: {rel.get('mathematical_implication', '')}")
                if 'dependency' in rel:
                    print(f"   关系链: {rel['dependency']}")
                print(f"   置信度: {rel.get('confidence', 'N/A')}")
            
            print("-" * 60)
    
    def generate_relation_report(self, output_file: str = None):
        """生成关系分析报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"relation_analysis_report_{timestamp}.md"
        
        print(f"\n📝 生成关系分析报告: {output_file}")
        
        report_lines = []
        report_lines.append("# COT-DIR 关系分析报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"")
        
        # 基本统计
        total_relations = sum(
            len(s.get('explicit_relations', [])) + 
            len(s.get('implicit_relations_L1', [])) + 
            len(s.get('implicit_relations_L2', [])) + 
            len(s.get('implicit_relations_L3', []))
            for s in self.solutions
        )
        
        report_lines.append("## 总体统计")
        report_lines.append(f"- 总题目数: {len(self.solutions):,}")
        report_lines.append(f"- 总关系数: {total_relations:,}")
        report_lines.append(f"- 平均每题关系数: {total_relations/len(self.solutions):.1f}")
        report_lines.append("")
        
        # 层次统计
        L1_count = sum(1 for s in self.solutions if len(s.get('implicit_relations_L1', [])) > 0)
        L2_count = sum(1 for s in self.solutions if len(s.get('implicit_relations_L2', [])) > 0)
        L3_count = sum(1 for s in self.solutions if len(s.get('implicit_relations_L3', [])) > 0)
        
        report_lines.append("## 关系层次分布")
        report_lines.append(f"- L1关系题目: {L1_count:,} ({L1_count/len(self.solutions)*100:.1f}%)")
        report_lines.append(f"- L2关系题目: {L2_count:,} ({L2_count/len(self.solutions)*100:.1f}%)")
        report_lines.append(f"- L3关系题目: {L3_count:,} ({L3_count/len(self.solutions)*100:.1f}%)")
        report_lines.append("")
        
        # 复杂度分析
        complex_problems = sum(1 for s in self.solutions 
                             if (len(s.get('explicit_relations', [])) + 
                                 len(s.get('implicit_relations_L1', [])) + 
                                 len(s.get('implicit_relations_L2', [])) + 
                                 len(s.get('implicit_relations_L3', []))) >= 5)
        
        report_lines.append("## 复杂度分析")
        report_lines.append(f"- 复杂题目(≥5关系): {complex_problems:,} ({complex_problems/len(self.solutions)*100:.1f}%)")
        report_lines.append("")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ 关系分析报告已保存: {output_file}")
    
    def interactive_browser(self):
        """交互式关系浏览器"""
        print(f"\n🔍 COT-DIR 关系交互式浏览器")
        print("=" * 60)
        print("可用命令:")
        print("  1. examples - 显示关系推理示例")
        print("  2. patterns - 分析关系模式")
        print("  3. datasets - 按数据集分析")
        print("  4. complex - 查找复杂关系题目")
        print("  5. L3 - 显示L3关系示例")
        print("  6. report - 生成分析报告")
        print("  7. exit - 退出浏览器")
        print("=" * 60)
        
        while True:
            try:
                command = input("\n🔍 请输入命令 (1-7): ").strip().lower()
                
                if command in ['1', 'examples']:
                    self.show_relation_examples()
                elif command in ['2', 'patterns']:
                    self.analyze_relation_patterns()
                elif command in ['3', 'datasets']:
                    self.analyze_dataset_relations()
                elif command in ['4', 'complex']:
                    self.find_complex_problems()
                elif command in ['5', 'l3']:
                    self.show_L3_examples()
                elif command in ['6', 'report']:
                    self.generate_relation_report()
                elif command in ['7', 'exit']:
                    print("👋 退出关系浏览器")
                    break
                else:
                    print("❌ 无效命令，请输入1-7")
            
            except KeyboardInterrupt:
                print("\n👋 退出关系浏览器")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

def main():
    """主函数"""
    print("🔍 COT-DIR 关系分析查看器")
    print("=" * 60)
    print("🎯 专门分析基于关系的解答过程")
    print("🔗 显性关系 + L1/L2/L3隐含关系")
    print("=" * 60)
    
    # 初始化查看器
    viewer = RelationViewer()
    
    if not viewer.solutions:
        print("❌ 无法加载关系解答数据")
        return
    
    # 显示基本示例
    print("\n🔍 快速预览...")
    viewer.show_relation_examples(num_examples=3)
    
    # 启动交互式浏览器
    viewer.interactive_browser()

if __name__ == "__main__":
    main()