#!/usr/bin/env python3
"""
Table 5 数据可视化工具
生成COT-DIR框架性能验证的美观表格和图表
"""

import json
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Table5Visualizer:
    """Table 5 可视化器"""
    
    def __init__(self):
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def create_table5_dataframe(self, table_data: Dict[str, Any]) -> pd.DataFrame:
        """创建Table 5的DataFrame"""
        
        # 转换数据为DataFrame格式
        rows = []
        for method, data in table_data.items():
            row = {
                'Method': method,
                'L0': float(data['L0'].rstrip('%')) / 100,
                'L1': float(data['L1'].rstrip('%')) / 100,
                'L2': float(data['L2'].rstrip('%')) / 100,
                'L3': float(data['L3'].rstrip('%')) / 100,
                'Overall': float(data['Overall'].rstrip('%')) / 100
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.set_index('Method')
        
        return df
    
    def print_formatted_table5(self, table_data: Dict[str, Any]):
        """打印格式化的Table 5"""
        
        print("\n" + "="*85)
        print("Table 5: Framework Performance Validation (n=200)")
        print("="*85)
        
        # 表头
        print(f"{'Method':<20} {'L0':<10} {'L1':<10} {'L2':<10} {'L3':<10} {'Overall':<10}")
        print("-" * 85)
        
        # 按性能排序
        sorted_methods = sorted(table_data.items(), 
                              key=lambda x: x[1]['raw_overall'], 
                              reverse=True)
        
        # 数据行
        for method, data in sorted_methods:
            print(f"{method:<20} {data['L0']:<10} {data['L1']:<10} {data['L2']:<10} {data['L3']:<10} {data['Overall']:<10}")
        
        print("="*85)
    
    def plot_performance_heatmap(self, df: pd.DataFrame, save_path: str = None):
        """绘制性能热力图"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建热力图
        sns.heatmap(df, 
                   annot=True, 
                   fmt='.1%', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Accuracy'},
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_title('Table 5: Framework Performance Validation Heatmap', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Complexity Levels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Methods', fontsize=12, fontweight='bold')
        
        # 旋转x轴标签
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 热力图已保存到: {save_path}")
        
        plt.show()
    
    def plot_complexity_degradation(self, df: pd.DataFrame, save_path: str = None):
        """绘制复杂度性能退化图"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 选择关键方法进行比较
        key_methods = ['Basic Symbolic', 'GPT-3.5 + CoT', 'IRD + MLR', 'Full COT-DIR']
        complexity_levels = ['L0', 'L1', 'L2', 'L3']
        
        for method in key_methods:
            if method in df.index:
                values = [df.loc[method, level] for level in complexity_levels]
                ax.plot(complexity_levels, values, 
                       marker='o', linewidth=2.5, markersize=8, 
                       label=method)
        
        ax.set_title('Performance Degradation Across Complexity Levels', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Complexity Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 复杂度退化图已保存到: {save_path}")
        
        plt.show()
    
    def plot_component_comparison(self, df: pd.DataFrame, save_path: str = None):
        """绘制组件对比图"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 组件方法分组
        single_components = ['IRD only', 'MLR only', 'CV only']
        combinations = ['IRD + MLR', 'Full COT-DIR']
        baselines = ['Basic Symbolic', 'Simple Neural', 'GPT-3.5 + CoT']
        
        # 设置位置
        x_pos = np.arange(len(df.index))
        bar_width = 0.6
        
        # 绘制条形图
        bars = ax.bar(x_pos, df['Overall'], 
                     width=bar_width, alpha=0.8)
        
        # 根据方法类型着色
        colors = []
        for method in df.index:
            if method in baselines:
                colors.append('#FF6B6B')  # 红色 - 基准方法
            elif method in single_components:
                colors.append('#4ECDC4')  # 青色 - 单组件
            elif method in combinations:
                colors.append('#45B7D1')  # 蓝色 - 组合方法
            else:
                colors.append('#96CEB4')  # 绿色 - 其他
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Framework Component Performance Comparison', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
        ax.set_ylabel('Overall Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # 设置x轴标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.8, label='Baseline Methods'),
            plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.8, label='Single Components'),
            plt.Rectangle((0,0),1,1, facecolor='#45B7D1', alpha=0.8, label='Combined Methods')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 组件对比图已保存到: {save_path}")
        
        plt.show()
    
    def plot_synergy_analysis(self, df: pd.DataFrame, save_path: str = None):
        """绘制协同效应分析图"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图: 组件累积效应
        cumulative_methods = ['IRD only', 'MLR only', 'IRD + MLR', 'Full COT-DIR']
        cumulative_data = []
        
        for method in cumulative_methods:
            if method in df.index:
                cumulative_data.append(df.loc[method, 'Overall'])
        
        ax1.plot(range(len(cumulative_data)), cumulative_data, 
                marker='o', linewidth=3, markersize=10, color='#45B7D1')
        
        for i, val in enumerate(cumulative_data):
            ax1.annotate(f'{val:.1%}', (i, val), 
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontweight='bold', fontsize=11)
        
        ax1.set_title('Component Synergy Effect', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Component Combination', fontsize=12)
        ax1.set_ylabel('Overall Accuracy', fontsize=12)
        ax1.set_xticks(range(len(cumulative_methods)))
        ax1.set_xticklabels([m.replace(' ', '\n') for m in cumulative_methods], fontsize=10)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax1.grid(True, alpha=0.3)
        
        # 右图: L3深度推理性能对比
        l3_methods = ['Basic Symbolic', 'Simple Neural', 'GPT-3.5 + CoT', 'Full COT-DIR']
        l3_values = []
        
        for method in l3_methods:
            if method in df.index:
                l3_values.append(df.loc[method, 'L3'])
        
        bars = ax2.bar(range(len(l3_values)), l3_values, 
                      color=['#FF6B6B', '#FF6B6B', '#FFD93D', '#45B7D1'],
                      alpha=0.8)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('L3 Deep Reasoning Performance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Methods', fontsize=12)
        ax2.set_ylabel('L3 Accuracy', fontsize=12)
        ax2.set_xticks(range(len(l3_methods)))
        ax2.set_xticklabels([m.replace(' ', '\n') for m in l3_methods], fontsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 协同效应分析图已保存到: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, evaluation_report: Dict[str, Any], output_dir: str = "."):
        """生成综合报告"""
        
        table_data = evaluation_report['table5_results']
        analysis = evaluation_report['analysis']
        
        # 创建DataFrame
        df = self.create_table5_dataframe(table_data)
        
        print("📊 生成Table 5综合可视化报告")
        
        # 1. 打印格式化表格
        self.print_formatted_table5(table_data)
        
        # 2. 生成热力图
        print("\n🎨 生成性能热力图...")
        self.plot_performance_heatmap(df, f"{output_dir}/table5_heatmap.png")
        
        # 3. 生成复杂度退化图
        print("\n📉 生成复杂度退化分析图...")
        self.plot_complexity_degradation(df, f"{output_dir}/complexity_degradation.png")
        
        # 4. 生成组件对比图
        print("\n🔧 生成组件性能对比图...")
        self.plot_component_comparison(df, f"{output_dir}/component_comparison.png")
        
        # 5. 生成协同效应分析图
        print("\n🤝 生成协同效应分析图...")
        self.plot_synergy_analysis(df, f"{output_dir}/synergy_analysis.png")
        
        # 6. 打印关键发现
        print("\n🔍 关键发现:")
        for finding in analysis['key_findings']:
            print(f"   • {finding}")
        
        # 7. 协同效应统计
        if 'synergistic_effects' in analysis:
            synergy = analysis['synergistic_effects']
            print(f"\n🚀 协同效应统计:")
            print(f"   • 完整COT-DIR框架: {synergy['full_framework']:.1%}")
            print(f"   • 最佳双组件组合: {synergy['best_two_component']:.1%}")
            print(f"   • 协同提升效果: +{synergy['synergy_boost']:.1%} ({synergy['synergy_percentage']:.1f}%)")
        
        print(f"\n✅ 所有可视化图表已保存到 {output_dir}/ 目录")

def main():
    """主函数 - 运行可视化演示"""
    
    # 示例数据 - 对应原始Table 5
    sample_table_data = {
        'Basic Symbolic': {'L0': '90.0%', 'L1': '64.0%', 'L2': '35.0%', 'L3': '15.0%', 'Overall': '46.5%', 'raw_overall': 0.465},
        'Simple Neural': {'L0': '83.0%', 'L1': '70.0%', 'L2': '40.0%', 'L3': '25.0%', 'Overall': '51.0%', 'raw_overall': 0.51},
        'GPT-3.5 + CoT': {'L0': '87.0%', 'L1': '76.0%', 'L2': '52.5%', 'L3': '45.0%', 'Overall': '62.0%', 'raw_overall': 0.62},
        'IRD only': {'L0': '73.0%', 'L1': '56.0%', 'L2': '42.5%', 'L3': '27.5%', 'Overall': '47.5%', 'raw_overall': 0.475},
        'MLR only': {'L0': '80.0%', 'L1': '62.0%', 'L2': '37.5%', 'L3': '22.5%', 'Overall': '47.0%', 'raw_overall': 0.47},
        'CV only': {'L0': '67.0%', 'L1': '52.0%', 'L2': '32.5%', 'L3': '17.5%', 'Overall': '39.5%', 'raw_overall': 0.395},
        'IRD + MLR': {'L0': '93.0%', 'L1': '84.0%', 'L2': '65.0%', 'L3': '55.0%', 'Overall': '72.0%', 'raw_overall': 0.72},
        'Full COT-DIR': {'L0': '96.7%', 'L1': '90.0%', 'L2': '72.5%', 'L3': '65.0%', 'Overall': '79.0%', 'raw_overall': 0.79}
    }
    
    # 示例分析数据
    sample_analysis = {
        'key_findings': [
            'Best performing method: Full COT-DIR with 79.0% overall accuracy',
            'Synergistic effect: Full COT-DIR (79.0%) vs best two-component (72.0%), boost: +7.0%',
            'Performance degradation from L0 to L3: 96.7% → 65.0% (-31.7%)'
        ],
        'synergistic_effects': {
            'full_framework': 0.79,
            'best_two_component': 0.72,
            'synergy_boost': 0.07,
            'synergy_percentage': 9.7
        }
    }
    
    sample_report = {
        'table5_results': sample_table_data,
        'analysis': sample_analysis
    }
    
    # 创建可视化器并生成报告
    visualizer = Table5Visualizer()
    visualizer.generate_comprehensive_report(sample_report)

if __name__ == "__main__":
    main() 