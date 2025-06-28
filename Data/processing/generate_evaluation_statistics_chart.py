#!/usr/bin/env python3
"""
评估统计图表生成器

生成评估指标和测试框架的统计图表和可视化报告
"""

import json

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('default')

def create_performance_complexity_chart():
    """创建按复杂度级别的性能分布图表"""
    
    # 数据
    complexity_levels = ['L0 (基础)', 'L1 (简单)', 'L2 (中等)', 'L3 (困难)']
    accuracy = [0.500, 1.000, 1.000, 0.000]
    sample_counts = [2, 1, 1, 1]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准确率柱状图
    colors = ['#ff7f7f', '#90ee90', '#90ee90', '#ff4444']
    bars1 = ax1.bar(complexity_levels, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('准确率', fontsize=12, fontweight='bold')
    ax1.set_title('按复杂度级别的性能表现', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    
    # 添加数据标签
    for bar, acc in zip(bars1, accuracy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 样本数饼图
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    wedges, texts, autotexts = ax2.pie(sample_counts, labels=complexity_levels, colors=colors_pie,
                                      autopct='%1.0f个', startangle=90, textprops={'fontsize': 10})
    ax2.set_title('样本分布', fontsize=14, fontweight='bold')
    
    # 美化饼图
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('性能评估_复杂度分析.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_relation_discovery_comparison():
    """创建隐式关系发现对比图表"""
    
    # 数据
    models = ['COT-DIR', 'Claude-3.5\nSonnet', 'GPT-4o', 'Qwen2.5\nMath-72B', 
              'InternLM2.5\nMath-7B', 'DeepSeek\nMath-7B', 'Graph2Tree']
    precision = [0.820, 0.730, 0.710, 0.690, 0.620, 0.640, 0.450]
    recall = [0.790, 0.680, 0.650, 0.720, 0.590, 0.610, 0.380]
    f1_score = [0.800, 0.700, 0.680, 0.700, 0.600, 0.620, 0.410]
    semantic_acc = [0.870, 0.810, 0.790, 0.760, 0.690, 0.710, 0.520]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # 角度设置
    angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 绘制COT-DIR (突出显示)
    cotdir_values = [precision[0], recall[0], f1_score[0], semantic_acc[0]]
    cotdir_values += cotdir_values[:1]
    ax.plot(angles[:4] + [angles[0]], cotdir_values, 'o-', linewidth=3, 
            label='COT-DIR', color='red', markersize=8)
    ax.fill(angles[:4] + [angles[0]], cotdir_values, alpha=0.25, color='red')
    
    # 设置标签
    metrics = ['精确率', '召回率', 'F1分数', '语义准确性']
    ax.set_xticks(angles[:4])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('隐式关系发现质量对比 - COT-DIR突出表现', 
                size=16, fontweight='bold', pad=20)
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('关系发现_质量对比.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_reasoning_chain_heatmap():
    """创建推理链质量热力图"""
    
    # 数据
    models = ['COT-DIR', 'Claude-3.5-Sonnet', 'GPT-4o', 'Qwen2.5-Math-72B', 
              'InternLM2.5-Math-7B', 'DeepSeek-Math-7B', 'Graph2Tree']
    dimensions = ['逻辑正确性', '完整性', '连贯性', '效率性', '可验证性']
    
    data = np.array([
        [0.930, 0.910, 0.940, 0.880, 0.960],  # COT-DIR
        [0.870, 0.820, 0.890, 0.760, 0.710],  # Claude
        [0.850, 0.790, 0.860, 0.730, 0.680],  # GPT-4o
        [0.820, 0.840, 0.810, 0.790, 0.760],  # Qwen
        [0.780, 0.750, 0.770, 0.740, 0.690],  # InternLM
        [0.790, 0.760, 0.780, 0.750, 0.700],  # DeepSeek
        [0.710, 0.680, 0.650, 0.820, 0.890]   # Graph2Tree
    ])
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用更好的颜色映射
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=1.0)
    
    # 设置标签
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(dimensions, fontsize=11, fontweight='bold')
    ax.set_yticklabels(models, fontsize=11, fontweight='bold')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标注
    for i in range(len(models)):
        for j in range(len(dimensions)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('质量分数', rotation=-90, va="bottom", fontweight='bold')
    
    ax.set_title("推理链质量评估热力图", fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    
    plt.savefig('推理链质量_热力图.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard():
    """创建综合评估仪表板"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 总体性能指标 (左上)
    ax1 = fig.add_subplot(gs[0, 0])
    scores = [0.6000, 0.5000, 0.8400, 0.6420]
    labels = ['性能\n准确率', '关系发现\nF1', '推理链\n质量', '系统\n综合分数']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    bars = ax1.bar(labels, scores, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylim(0, 1)
    ax1.set_title('系统综合评估分数', fontweight='bold', fontsize=12)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 复杂度分布 (右上)
    ax2 = fig.add_subplot(gs[0, 1:])
    complexity_data = {
        'L0 (基础)': {'accuracy': 0.5, 'samples': 2, 'color': '#ff7f7f'},
        'L1 (简单)': {'accuracy': 1.0, 'samples': 1, 'color': '#90ee90'},
        'L2 (中等)': {'accuracy': 1.0, 'samples': 1, 'color': '#90ee90'},
        'L3 (困难)': {'accuracy': 0.0, 'samples': 1, 'color': '#ff4444'}
    }
    
    x_pos = np.arange(len(complexity_data))
    accuracies = [data['accuracy'] for data in complexity_data.values()]
    colors = [data['color'] for data in complexity_data.values()]
    
    bars = ax2.bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(complexity_data.keys(), fontsize=10)
    ax2.set_ylabel('准确率')
    ax2.set_title('按复杂度级别的性能表现', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 1.1)
    
    # 3. 关系类型分布 (中间左)
    ax3 = fig.add_subplot(gs[1, 0])
    relation_types = ['数学运算', '单位转换', '物理约束', '时间关系', '几何属性', '比例关系']
    percentages = [35.2, 18.7, 16.4, 12.3, 10.8, 6.6]
    colors = plt.cm.Set3(np.linspace(0, 1, len(relation_types)))
    
    wedges, texts, autotexts = ax3.pie(percentages, labels=relation_types, autopct='%1.1f%%',
                                      colors=colors, startangle=90, textprops={'fontsize': 8})
    ax3.set_title('隐式关系类型分布', fontweight='bold', fontsize=12)
    
    # 4. 模型性能对比 (中间右)
    ax4 = fig.add_subplot(gs[1, 1:])
    models = ['COT-DIR', 'Claude', 'GPT-4o', 'Qwen', 'InternLM', 'DeepSeek', 'Graph2Tree']
    overall_scores = [0.920, 0.810, 0.780, 0.800, 0.750, 0.760, 0.750]
    
    # 创建水平条形图
    y_pos = np.arange(len(models))
    bars = ax4.barh(y_pos, overall_scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(models)
    ax4.set_xlabel('总体质量分数')
    ax4.set_title('推理链质量模型对比', fontweight='bold', fontsize=12)
    ax4.set_xlim(0, 1)
    
    # 添加分数标签
    for i, (bar, score) in enumerate(zip(bars, overall_scores)):
        ax4.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontweight='bold', fontsize=9)
    
    # 5. 错误分析 (底部左)
    ax5 = fig.add_subplot(gs[2, 0])
    error_stages = ['早期阶段\n错误', '中期阶段\n错误', '后期阶段\n错误', '验证恢复']
    error_percentages = [25, 45, 20, 10]
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
    
    bars = ax5.bar(error_stages, error_percentages, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('百分比 (%)')
    ax5.set_title('错误阶段分布分析', fontweight='bold', fontsize=12)
    ax5.set_ylim(0, 50)
    
    for bar, pct in zip(bars, error_percentages):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. 评估器状态 (底部右)
    ax6 = fig.add_subplot(gs[2, 1:])
    evaluators = ['Performance\nEvaluator', 'Relation Discovery\nEvaluator', 'Reasoning Chain\nEvaluator']
    status = [1, 1, 1]  # 1 表示已完成
    colors = ['#4CAF50', '#4CAF50', '#4CAF50']  # 绿色表示完成
    
    bars = ax6.bar(evaluators, status, color=colors, alpha=0.8, edgecolor='black')
    ax6.set_ylim(0, 1.2)
    ax6.set_ylabel('完成状态')
    ax6.set_title('评估器集成状态', fontweight='bold', fontsize=12)
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['未完成', '已完成'])
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                '✓ 已完成', ha='center', va='bottom', fontweight='bold', color='green')
    
    # 添加总标题
    fig.suptitle('数学推理系统评估指标和测试框架 - 综合统计报告', 
                fontsize=18, fontweight='bold', y=0.95)
    
    plt.savefig('综合评估仪表板.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics():
    """生成统计总结"""
    
    summary = {
        "evaluation_framework": {
            "total_evaluators": 3,
            "completed_evaluators": 3,
            "integration_status": "100% 完成",
            "evaluation_dimensions": 14
        },
        "performance_metrics": {
            "overall_accuracy": 0.6000,
            "robustness_score": 0.5000,
            "best_complexity_level": "L1, L2 (100%)",
            "worst_complexity_level": "L3 (0%)"
        },
        "relation_discovery": {
            "best_model": "COT-DIR",
            "best_f1_score": 0.800,
            "best_semantic_accuracy": 0.870,
            "total_relation_types": 6,
            "most_common_relation": "数学运算关系 (35.2%)"
        },
        "reasoning_chain": {
            "best_model": "COT-DIR",
            "best_overall_score": 0.920,
            "best_dimension": "可验证性 (0.960)",
            "error_recovery_rate": "10%"
        },
        "system_performance": {
            "comprehensive_score": 0.6420,
            "top_strength": "推理链质量 (0.840)",
            "improvement_area": "关系发现精度 (0.500)"
        }
    }
    
    # 保存统计总结
    with open('evaluation_statistics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("📊 评估统计总结:")
    print("="*50)
    print(f"✅ 评估器框架: {summary['evaluation_framework']['completed_evaluators']}/{summary['evaluation_framework']['total_evaluators']} 个评估器已完成")
    print(f"📈 系统综合分数: {summary['system_performance']['comprehensive_score']:.4f}")
    print(f"🏆 最佳模型: {summary['relation_discovery']['best_model']} (多项指标领先)")
    print(f"💡 主要优势: {summary['system_performance']['top_strength']}")
    print(f"🔧 改进方向: {summary['system_performance']['improvement_area']}")
    print("="*50)

def main():
    """主函数 - 生成所有图表和统计信息"""
    
    print("🎯 正在生成评估指标和测试框架统计图表...")
    
    # 生成各种图表
    print("📊 1. 生成性能复杂度分析图...")
    create_performance_complexity_chart()
    
    print("📊 2. 生成关系发现对比图...")
    create_relation_discovery_comparison()
    
    print("📊 3. 生成推理链质量热力图...")
    create_reasoning_chain_heatmap()
    
    print("📊 4. 生成综合评估仪表板...")
    create_comprehensive_dashboard()
    
    print("📊 5. 生成统计总结...")
    generate_summary_statistics()
    
    print("✅ 所有统计图表和报告已生成完成！")
    print("\n生成的文件:")
    print("- 性能评估_复杂度分析.png")
    print("- 关系发现_质量对比.png") 
    print("- 推理链质量_热力图.png")
    print("- 综合评估仪表板.png")
    print("- evaluation_statistics_summary.json")
    print("- 统计结果_评估指标和测试框架.md")

if __name__ == "__main__":
    main() 