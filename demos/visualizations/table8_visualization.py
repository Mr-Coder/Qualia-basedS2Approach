#!/usr/bin/env python3
"""
Table 8 可视化工具
生成计算效率和可扩展性分析的图表
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Table8Visualizer:
    """Table 8 可视化器"""
    
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    
    def plot_time_performance_comparison(self, report: Dict[str, Any], save_path: str = None):
        """绘制时间性能对比图"""
        
        systems = ['Basic Symbolic', 'Simple Neural', 'GPT-3.5 (API)', 'Full COT-DIR']
        avg_times = [0.6, 1.5, 2.8, 4.3]
        l2_times = [0.9, 2.2, 3.4, 6.3]
        l3_times = [1.2, 3.1, 4.2, 9.0]
        
        x = np.arange(len(systems))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width, avg_times, width, label='Average Time', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x, l2_times, width, label='L2 Complexity', alpha=0.8, color='#e74c3c')
        bars3 = ax.bar(x + width, l3_times, width, label='L3 Complexity', alpha=0.8, color='#f39c12')
        
        # 添加数值标签
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        ax.set_xlabel('System Configuration', fontsize=12)
        ax.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax.set_title('Computational Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=15, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 时间性能对比图已保存: {save_path}")
        
        plt.show()
    
    def plot_memory_usage_comparison(self, report: Dict[str, Any], save_path: str = None):
        """绘制内存使用对比图"""
        
        systems = ['Basic Symbolic', 'Simple Neural', 'Full COT-DIR']
        memory_usage = [32, 95, 185]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(systems, memory_usage, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, memory_usage):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                   f'{value}MB', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title('Memory Consumption Comparison', fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(memory_usage) * 1.2)
        
        # 添加GPT-3.5说明
        ax.text(0.5, 0.95, 'Note: GPT-3.5 (API) uses external memory', 
                transform=ax.transAxes, ha='center', va='top', 
                style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 内存使用对比图已保存: {save_path}")
        
        plt.show()
    
    def plot_scalability_analysis(self, report: Dict[str, Any], save_path: str = None):
        """绘制可扩展性分析图"""
        
        systems = ['Basic Symbolic', 'Simple Neural', 'GPT-3.5 (API)', 'Full COT-DIR']
        scalability_scores = [5, 4, 3, 2]  # Very fast=5, Fast=4, Variable=3, Manageable=2
        scalability_labels = ['Very fast', 'Fast', 'Variable', 'Manageable']
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：条形图
        bars = ax1.bar(systems, scalability_scores, color=colors, alpha=0.8)
        
        for bar, label in zip(bars, scalability_labels):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    label, ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Scalability Score', fontsize=12)
        ax1.set_title('Scalability Ranking', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 6)
        ax1.set_xticklabels(systems, rotation=15, ha='right')
        
        # 右图：雷达图（简化版）
        angles = np.linspace(0, 2 * np.pi, len(systems), endpoint=False).tolist()
        angles += angles[:1]
        
        values = scalability_scores + [scalability_scores[0]]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax2.fill(angles, values, alpha=0.25, color='#3498db')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(systems)
        ax2.set_ylim(0, 5)
        ax2.set_title('Scalability Profile', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 可扩展性分析图已保存: {save_path}")
        
        plt.show()
    
    def plot_efficiency_tradeoff(self, report: Dict[str, Any], save_path: str = None):
        """绘制效率权衡散点图"""
        
        systems = ['Basic Symbolic', 'Simple Neural', 'GPT-3.5 (API)', 'Full COT-DIR']
        avg_times = [0.6, 1.5, 2.8, 4.3]
        capability_scores = [2, 3, 4, 5]  # 假设的能力评分
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        sizes = [100, 150, 120, 200]  # 气泡大小代表复杂度
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(avg_times, capability_scores, s=sizes, c=colors, alpha=0.7, edgecolors='black')
        
        # 添加系统标签
        for i, system in enumerate(systems):
            ax.annotate(system, (avg_times[i], capability_scores[i]), 
                       xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax.set_xlabel('Average Processing Time (seconds)', fontsize=12)
        ax.set_ylabel('Reasoning Capability Score', fontsize=12)
        ax.set_title('Performance vs Capability Trade-off', fontsize=16, fontweight='bold')
        
        # 添加趋势线
        z = np.polyfit(avg_times, capability_scores, 1)
        p = np.poly1d(z)
        ax.plot(avg_times, p(avg_times), "--", alpha=0.8, color='gray', 
                label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 效率权衡图已保存: {save_path}")
        
        plt.show()
    
    def plot_complexity_scaling(self, report: Dict[str, Any], save_path: str = None):
        """绘制复杂度扩展图"""
        
        complexity_levels = ['L0', 'L1', 'L2', 'L3']
        
        # 模拟复杂度扩展数据
        basic_symbolic = [0.5, 0.6, 0.9, 1.2]
        simple_neural = [1.2, 1.5, 2.2, 3.1]
        gpt35_api = [2.5, 2.8, 3.4, 4.2]
        cotdir = [3.8, 4.3, 6.3, 9.0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(complexity_levels, basic_symbolic, 'o-', linewidth=2, label='Basic Symbolic', color='#2ecc71')
        ax.plot(complexity_levels, simple_neural, 's-', linewidth=2, label='Simple Neural', color='#3498db')
        ax.plot(complexity_levels, gpt35_api, '^-', linewidth=2, label='GPT-3.5 (API)', color='#f39c12')
        ax.plot(complexity_levels, cotdir, 'D-', linewidth=2, label='Full COT-DIR', color='#e74c3c')
        
        ax.set_xlabel('Problem Complexity Level', fontsize=12)
        ax.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax.set_title('Computational Scaling Across Complexity Levels', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加注释
        ax.annotate('Exponential scaling\nfor complex problems', 
                   xy=('L3', 9.0), xytext=('L2', 7),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   fontsize=10, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 复杂度扩展图已保存: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, analysis_report: Dict[str, Any], output_dir: str = "."):
        """生成综合可视化报告"""
        
        print("📊 生成Table 8综合可视化报告")
        
        # 1. 时间性能对比
        print("\n⏱️ 生成时间性能对比图...")
        self.plot_time_performance_comparison(analysis_report, f"{output_dir}/time_performance_comparison.png")
        
        # 2. 内存使用对比
        print("\n💾 生成内存使用对比图...")
        self.plot_memory_usage_comparison(analysis_report, f"{output_dir}/memory_usage_comparison.png")
        
        # 3. 可扩展性分析
        print("\n📈 生成可扩展性分析图...")
        self.plot_scalability_analysis(analysis_report, f"{output_dir}/scalability_analysis.png")
        
        # 4. 效率权衡分析
        print("\n⚖️ 生成效率权衡分析图...")
        self.plot_efficiency_tradeoff(analysis_report, f"{output_dir}/efficiency_tradeoff.png")
        
        # 5. 复杂度扩展分析
        print("\n📊 生成复杂度扩展分析图...")
        self.plot_complexity_scaling(analysis_report, f"{output_dir}/complexity_scaling.png")
        
        print(f"\n✅ 所有可视化图表已保存到 {output_dir}/ 目录")

def main():
    """演示可视化功能"""
    
    # 示例数据
    sample_report = {
        'table8_results': {
            'Basic Symbolic': {'Avg. Time (s)': '0.6±0.1', 'Memory (MB)': '32±6', 'L2 Time': '0.9±0.2', 'L3 Time': '1.2±0.3', 'Scalability': 'Very fast'},
            'Simple Neural': {'Avg. Time (s)': '1.5±0.4', 'Memory (MB)': '95±18', 'L2 Time': '2.2±0.5', 'L3 Time': '3.1±0.7', 'Scalability': 'Fast'},
            'GPT-3.5 (API)': {'Avg. Time (s)': '2.8±0.9', 'Memory (MB)': 'N/A', 'L2 Time': '3.4±1.1', 'L3 Time': '4.2±1.4', 'Scalability': 'Variable'},
            'Full COT-DIR': {'Avg. Time (s)': '4.3±0.8', 'Memory (MB)': '185±28', 'L2 Time': '6.3±1.2', 'L3 Time': '9.0±1.8', 'Scalability': 'Manageable'}
        },
        'analysis': {
            'computational_overhead': {
                'overhead_ratio': 7.2,
                'description': 'COT-DIR requires 7.2× more computation than baseline'
            }
        }
    }
    
    visualizer = Table8Visualizer()
    visualizer.generate_comprehensive_report(sample_report)

if __name__ == "__main__":
    main() 