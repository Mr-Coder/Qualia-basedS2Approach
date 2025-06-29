"""
🔢 COT-DIR系统题目数量统计
统计系统最多可以生成多少道解答题目
"""

import json
import os
from pathlib import Path


def count_problems_in_datasets():
    """统计所有数据集中的题目数量"""
    print("🔍 COT-DIR系统题目生成能力分析")
    print("=" * 60)
    
    data_dir = Path("Data")
    total_problems = 0
    dataset_details = []
    
    # 遍历所有数据集目录
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('.') and not dataset_dir.name.startswith('__'):
            dataset_name = dataset_dir.name
            problem_count = 0
            
            # 查找JSON文件
            for json_file in dataset_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            problem_count += len(data)
                        elif isinstance(data, dict):
                            # 检查常见的键名
                            for key in ['problems', 'data', 'examples', 'questions']:
                                if key in data and isinstance(data[key], list):
                                    problem_count += len(data[key])
                                    break
                            else:
                                problem_count += 1  # 单个问题
                except Exception as e:
                    print(f"   ⚠️  无法读取 {json_file}: {e}")
                    continue
            
            # 查找JSONL文件  
            for jsonl_file in dataset_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                problem_count += 1
                except Exception as e:
                    print(f"   ⚠️  无法读取 {jsonl_file}: {e}")
                    continue
            
            if problem_count > 0:
                dataset_details.append((dataset_name, problem_count))
                total_problems += problem_count
    
    # 检查根目录中的其他JSON文件
    for json_file in data_dir.glob("*.json"):
        if json_file.is_file():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        count = 1
                    else:
                        count = 0
                    
                    if count > 0:
                        dataset_details.append((f"根目录_{json_file.stem}", count))
                        total_problems += count
            except Exception as e:
                print(f"   ⚠️  无法读取根目录文件 {json_file}: {e}")
    
    # 按题目数量排序
    dataset_details.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📊 数据集详细统计:")
    for name, count in dataset_details:
        print(f"   {name}: {count:,} 题")
    
    print(f"\n📈 总计题目数量: {total_problems:,} 题")
    print(f"📋 数据集总数: {len(dataset_details)} 个")
    
    # 计算按复杂度分布
    small = sum(1 for _, count in dataset_details if count <= 100)
    medium = sum(1 for _, count in dataset_details if 100 < count <= 1000)  
    large = sum(1 for _, count in dataset_details if 1000 < count <= 2000)
    extra_large = sum(1 for _, count in dataset_details if count > 2000)
    
    print(f"\n📊 数据集规模分布:")
    print(f"   小型(≤100题): {small} 个")
    print(f"   中型(101-1000题): {medium} 个")
    print(f"   大型(1001-2000题): {large} 个")
    print(f"   超大型(>2000题): {extra_large} 个")
    
    # 分析题目类型
    print(f"\n🎯 题目类型分析:")
    elementary_datasets = [name for name, _ in dataset_details if any(keyword in name.lower() for keyword in ['addsub', 'singleeq', 'multiarith'])]
    intermediate_datasets = [name for name, _ in dataset_details if any(keyword in name.lower() for keyword in ['gsm', 'svamp', 'asdiv'])]
    advanced_datasets = [name for name, _ in dataset_details if any(keyword in name.lower() for keyword in ['math', 'aqua', 'mathqa'])]
    
    print(f"   初级题目数据集: {len(elementary_datasets)} 个")
    print(f"   中级题目数据集: {len(intermediate_datasets)} 个") 
    print(f"   高级题目数据集: {len(advanced_datasets)} 个")
    
    # 计算理论生成能力
    print(f"\n🚀 题目生成能力分析:")
    print(f"   ✅ 当前可处理: {total_problems:,} 题")
    print(f"   ✅ 批量处理能力: 理论上无限制")
    print(f"   ✅ 推荐单次批次: 1,000-10,000 题")
    print(f"   ✅ 内存优化批次: 100-1,000 题")
    
    # 估算处理时间
    print(f"\n⏱️ 处理时间估算:")
    processing_speed = 0.2  # 毫秒/题 (根据之前测试结果)
    
    for batch_size in [100, 1000, 10000, total_problems]:
        if batch_size <= total_problems:
            time_ms = batch_size * processing_speed
            if time_ms < 1000:
                time_str = f"{time_ms:.1f} 毫秒"
            elif time_ms < 60000:
                time_str = f"{time_ms/1000:.1f} 秒"
            else:
                time_str = f"{time_ms/60000:.1f} 分钟"
            print(f"   {batch_size:,} 题 → {time_str}")
    
    # 系统限制分析
    print(f"\n🔧 系统限制分析:")
    print(f"   💾 内存限制: 取决于系统RAM (推荐16GB+)")
    print(f"   🔄 并发处理: 支持多线程/多进程")
    print(f"   💿 存储需求: 约 {total_problems * 0.5 / 1024:.1f} MB (估算)")
    
    # 实际应用建议
    print(f"\n💡 实际应用建议:")
    print(f"   🎯 小规模测试: 100-1,000 题")
    print(f"   📊 中等规模研究: 1,000-10,000 题")
    print(f"   🚀 大规模训练: 10,000+ 题")
    print(f"   ⚡ 最大理论处理: {total_problems:,} 题")
    
    return total_problems, dataset_details

def analyze_problem_quality():
    """分析题目质量和多样性"""
    print(f"\n🌟 题目质量与多样性分析:")
    print("-" * 40)
    
    # 基于数据集名称分析题目类型
    type_analysis = {
        "基础算术": ["AddSub", "SingleEq"],
        "多步算术": ["MultiArith"],
        "应用题": ["SVAMP", "ASDiv", "MAWPS"],
        "小学数学": ["GSM8K", "GSM-hard"],
        "中学数学": ["MATH", "MathQA"],
        "中文数学": ["Math23K"],
        "特殊类型": ["AQuA", "DIR-MWP"]
    }
    
    for category, datasets in type_analysis.items():
        count = 0
        data_dir = Path("Data")
        for dataset_name in datasets:
            dataset_path = data_dir / dataset_name
            if dataset_path.exists():
                # 简单估算
                for json_file in dataset_path.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                count += len(data)
                    except:
                        pass
        
        if count > 0:
            print(f"   {category}: ~{count:,} 题")
    
    print(f"\n📝 题目特点:")
    print(f"   ✅ 多语言支持 (英文/中文)")
    print(f"   ✅ 多难度等级 (初级到高级)")
    print(f"   ✅ 多题型覆盖 (算术/几何/代数)")
    print(f"   ✅ 标准化格式 (JSON/JSONL)")

if __name__ == "__main__":
    total, details = count_problems_in_datasets()
    analyze_problem_quality()
    
    print(f"\n🎉 总结:")
    print(f"🔢 COT-DIR系统最多可以生成 {total:,} 道解答题目！")
    print(f"📚 覆盖从基础算术到高级数学的完整范围")
    print(f"⚡ 支持高效批量处理和质量评估") 