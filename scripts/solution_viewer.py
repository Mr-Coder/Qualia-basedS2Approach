"""
🔍 COT-DIR 解答查看器
Solution Viewer - 查看和分析生成的数学解答过程

提供多种方式浏览14,097个详细解答
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SolutionViewer:
    """解答查看器"""
    
    def __init__(self):
        """初始化解答查看器"""
        print("🔍 初始化COT-DIR解答查看器")
        self.solutions = []
        self.metadata = {}
        
    def load_solutions(self, file_path: str = None) -> bool:
        """加载解答文件"""
        if file_path is None:
            # 查找最新的解答文件
            solution_files = list(Path(".").glob("maximum_solutions_*.json"))
            if not solution_files:
                solution_files = list(Path(".").glob("*solutions*.json"))
            
            if not solution_files:
                print("❌ 没有找到解答文件")
                return False
            
            # 选择最新的文件
            file_path = max(solution_files, key=lambda p: p.stat().st_mtime)
            print(f"📁 自动选择最新文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data.get('metadata', {})
                self.solutions = data.get('solutions', [])
                
            print(f"✅ 成功加载 {len(self.solutions):,} 个解答")
            print(f"📊 生成时间: {self.metadata.get('generated_at', '未知')}")
            print(f"🎯 生成器类型: {self.metadata.get('generator_type', '未知')}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return False
    
    def show_overview(self):
        """显示总览"""
        if not self.solutions:
            print("❌ 没有可用的解答数据")
            return
        
        print(f"\n📊 解答数据总览:")
        print("=" * 60)
        print(f"总解答数: {len(self.solutions):,}")
        
        # 按数据集统计
        dataset_stats = {}
        type_stats = {}
        difficulty_stats = {}
        
        for solution in self.solutions:
            # 数据集统计
            dataset = solution.get('dataset_source', 'unknown')
            dataset_stats[dataset] = dataset_stats.get(dataset, 0) + 1
            
            # 类型统计
            ptype = solution.get('problem_type', 'unknown')
            type_stats[ptype] = type_stats.get(ptype, 0) + 1
            
            # 难度统计
            difficulty = solution.get('difficulty_level', 'unknown')
            difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
        
        print(f"\n📈 数据集分布 (Top 10):")
        for dataset, count in sorted(dataset_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = count / len(self.solutions) * 100
            print(f"   {dataset}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n🎯 题目类型分布:")
        for ptype, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(self.solutions) * 100
            print(f"   {ptype}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📚 难度分布:")
        for difficulty, count in sorted(difficulty_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(self.solutions) * 100
            print(f"   {difficulty}: {count:,} ({percentage:.1f}%)")
    
    def search_solutions(self, keyword: str = None, dataset: str = None, 
                        problem_type: str = None, difficulty: str = None, 
                        limit: int = 10) -> List[Dict]:
        """搜索解答"""
        if not self.solutions:
            print("❌ 没有可用的解答数据")
            return []
        
        filtered_solutions = self.solutions.copy()
        
        # 按关键词过滤
        if keyword:
            filtered_solutions = [s for s in filtered_solutions 
                                if keyword.lower() in s.get('question', '').lower()]
        
        # 按数据集过滤
        if dataset:
            filtered_solutions = [s for s in filtered_solutions 
                                if s.get('dataset_source', '').lower() == dataset.lower()]
        
        # 按类型过滤
        if problem_type:
            filtered_solutions = [s for s in filtered_solutions 
                                if s.get('problem_type', '').lower() == problem_type.lower()]
        
        # 按难度过滤
        if difficulty:
            filtered_solutions = [s for s in filtered_solutions 
                                if s.get('difficulty_level', '').lower() == difficulty.lower()]
        
        print(f"🔍 搜索结果: 找到 {len(filtered_solutions)} 个匹配的解答")
        
        return filtered_solutions[:limit]
    
    def display_solution(self, solution: Dict, detailed: bool = True):
        """显示单个解答"""
        print("\n" + "=" * 80)
        print(f"📋 题目ID: {solution.get('problem_id', 'unknown')}")
        print(f"📊 数据集: {solution.get('dataset_source', 'unknown')}")
        print(f"🎯 类型: {solution.get('problem_type', 'unknown')}")
        print(f"📚 难度: {solution.get('difficulty_level', 'unknown')}")
        
        if 'complexity_score' in solution:
            print(f"🔢 复杂度: {solution['complexity_score']}/10")
        
        print(f"📝 题目:")
        question = solution.get('question', '无题目')
        if len(question) > 200:
            print(f"   {question[:200]}...")
        else:
            print(f"   {question}")
        
        if detailed:
            # 显示解答步骤
            if 'solution_steps' in solution:
                print(f"\n🔧 解答步骤:")
                for step in solution['solution_steps']:
                    print(f"   • {step}")
            
            # 显示数学分析
            if 'mathematical_analysis' in solution:
                print(f"\n🧮 数学分析:")
                print(f"   {solution['mathematical_analysis']}")
            
            # 显示计算步骤
            if 'computational_steps' in solution:
                print(f"\n⚙️ 计算过程:")
                for step in solution['computational_steps']:
                    print(f"   步骤{step.get('step', '?')}: {step.get('action', '未知')} - {step.get('description', '无描述')}")
            
            # 显示验证过程
            if 'verification_process' in solution:
                print(f"\n✅ 验证过程:")
                print(f"   {solution['verification_process']}")
        
        print(f"\n🎯 最终答案: {solution.get('final_answer', '无答案')}")
        
        if 'confidence_score' in solution:
            confidence = solution['confidence_score']
            print(f"🔮 置信度: {confidence:.2f} ({'高' if confidence > 0.9 else '中' if confidence > 0.7 else '低'})")
        
        print("=" * 80)
    
    def show_random_samples(self, count: int = 5):
        """显示随机样本"""
        if not self.solutions:
            print("❌ 没有可用的解答数据")
            return
        
        print(f"\n🎲 随机显示 {count} 个解答样本:")
        
        samples = random.sample(self.solutions, min(count, len(self.solutions)))
        
        for i, solution in enumerate(samples, 1):
            print(f"\n【样本 {i}】")
            self.display_solution(solution, detailed=True)
    
    def show_by_difficulty(self, difficulty: str, count: int = 3):
        """按难度显示解答"""
        solutions = self.search_solutions(difficulty=difficulty, limit=count)
        
        if not solutions:
            print(f"❌ 没有找到难度为'{difficulty}'的题目")
            return
        
        print(f"\n📚 难度'{difficulty}'的解答示例:")
        
        for i, solution in enumerate(solutions, 1):
            print(f"\n【示例 {i}】")
            self.display_solution(solution, detailed=True)
    
    def show_by_type(self, problem_type: str, count: int = 3):
        """按类型显示解答"""
        solutions = self.search_solutions(problem_type=problem_type, limit=count)
        
        if not solutions:
            print(f"❌ 没有找到类型为'{problem_type}'的题目")
            return
        
        print(f"\n🎯 类型'{problem_type}'的解答示例:")
        
        for i, solution in enumerate(solutions, 1):
            print(f"\n【示例 {i}】")
            self.display_solution(solution, detailed=True)
    
    def analyze_quality(self):
        """分析解答质量"""
        if not self.solutions:
            print("❌ 没有可用的解答数据")
            return
        
        print(f"\n📊 解答质量分析:")
        print("=" * 60)
        
        # 置信度分析
        confidences = [s.get('confidence_score', 0) for s in self.solutions if 'confidence_score' in s]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            high_confidence = sum(1 for c in confidences if c > 0.9)
            medium_confidence = sum(1 for c in confidences if 0.7 < c <= 0.9)
            low_confidence = sum(1 for c in confidences if c <= 0.7)
            
            print(f"🔮 置信度统计:")
            print(f"   平均置信度: {avg_confidence:.3f}")
            print(f"   高置信度 (>0.9): {high_confidence:,} ({high_confidence/len(confidences)*100:.1f}%)")
            print(f"   中置信度 (0.7-0.9): {medium_confidence:,} ({medium_confidence/len(confidences)*100:.1f}%)")
            print(f"   低置信度 (<0.7): {low_confidence:,} ({low_confidence/len(confidences)*100:.1f}%)")
        
        # 复杂度分析
        complexities = [s.get('complexity_score', 0) for s in self.solutions if 'complexity_score' in s]
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            print(f"\n🔢 复杂度统计:")
            print(f"   平均复杂度: {avg_complexity:.1f}/10")
            
            complexity_dist = {}
            for c in complexities:
                complexity_dist[c] = complexity_dist.get(c, 0) + 1
            
            print(f"   复杂度分布:")
            for level in sorted(complexity_dist.keys()):
                count = complexity_dist[level]
                percentage = count / len(complexities) * 100
                print(f"     {level}/10: {count:,} 题 ({percentage:.1f}%)")
        
        # 处理性能分析
        if 'generation_stats' in self.metadata:
            stats = self.metadata['generation_stats']
            print(f"\n⚡ 处理性能:")
            print(f"   总处理时间: {stats.get('total_time', 0):.1f} 秒")
            print(f"   处理速度: {stats.get('processing_rate', 0):.0f} 题/秒")
            print(f"   平均每题时间: {stats.get('avg_time_per_problem', 0)*1000:.2f} 毫秒")
    
    def interactive_mode(self):
        """交互模式"""
        if not self.solutions:
            print("❌ 没有可用的解答数据，请先加载文件")
            return
        
        print(f"\n🎮 进入交互模式 (共有 {len(self.solutions):,} 个解答)")
        print("=" * 60)
        
        while True:
            print(f"\n🔍 选择操作:")
            print("1. 显示总览")
            print("2. 随机查看解答")
            print("3. 按难度查看")
            print("4. 按类型查看")
            print("5. 关键词搜索")
            print("6. 质量分析")
            print("7. 退出")
            
            try:
                choice = input("\n请选择 (1-7): ").strip()
                
                if choice == '1':
                    self.show_overview()
                
                elif choice == '2':
                    count = input("显示几个样本 (默认5): ").strip()
                    count = int(count) if count.isdigit() else 5
                    self.show_random_samples(count)
                
                elif choice == '3':
                    print("可选难度: easy, medium, hard, expert")
                    difficulty = input("输入难度级别: ").strip()
                    if difficulty:
                        self.show_by_difficulty(difficulty)
                
                elif choice == '4':
                    print("可选类型: arithmetic, word_problem, algebra, geometry, statistics_probability")
                    ptype = input("输入题目类型: ").strip()
                    if ptype:
                        self.show_by_type(ptype)
                
                elif choice == '5':
                    keyword = input("输入搜索关键词: ").strip()
                    if keyword:
                        results = self.search_solutions(keyword=keyword, limit=5)
                        for i, solution in enumerate(results, 1):
                            print(f"\n【搜索结果 {i}】")
                            self.display_solution(solution, detailed=True)
                
                elif choice == '6':
                    self.analyze_quality()
                
                elif choice == '7':
                    print("👋 退出交互模式")
                    break
                
                else:
                    print("❌ 无效选择，请重试")
                    
            except KeyboardInterrupt:
                print("\n👋 退出交互模式")
                break
            except Exception as e:
                print(f"❌ 操作出错: {e}")

def main():
    """主函数"""
    print("🔍 COT-DIR 解答查看器")
    print("=" * 60)
    
    viewer = SolutionViewer()
    
    # 加载解答数据
    if not viewer.load_solutions():
        return
    
    # 显示基本信息
    viewer.show_overview()
    
    # 显示一些样本
    print(f"\n🎲 随机显示3个解答样本:")
    viewer.show_random_samples(3)
    
    # 进入交互模式
    print(f"\n🎮 是否进入交互模式？(y/n)")
    choice = input().strip().lower()
    if choice in ['y', 'yes', '是']:
        viewer.interactive_mode()
    
    print(f"\n🎉 解答查看完成!")

if __name__ == "__main__":
    main() 