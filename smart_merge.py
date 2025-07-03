#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import json

def smart_merge():
    """智能合并 newfile 到 cot-dir1"""
    newfile_dir = Path("~/Desktop/newfile").expanduser()
    target_dir = Path(".")
    
    # 读取分析结果
    with open('merge_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    # 高优先级文件列表
    high_priority_files = [
        'batch_test_hybrid.py',
        'demo_enhanced_integration.py', 
        'demo_intelligent_tutor.py',
        'failure_sample_classifier.py',
        'meta_pattern_generator.py',
        'pattern_library_enhancer.py',
        'detailed_solution_demo.py',
        'batch_test_runner.py'
    ]
    
    # src 目录的重要文件
    src_priority_files = [
        'src/config/advanced_config.py',
        'src/math_problem_solver.py',
        'src/math_problem_solver_v2.py',
        'src/models/enhanced_hybrid_tutor.py',
        'src/models/intelligent_tutor.py',
        'src/models/hybrid_model.py',
        'src/models/simple_pattern_model.py',
        'src/reasoning_engine/pattern_based_solver.py',
        'src/utils/error_handling.py',
        'src/utils/performance_optimizer.py',
        'src/tests/test_math_solver_v2.py'
    ]
    
    print("🚀 **开始智能合并...**")
    
    # 合并高优先级文件
    print("\n📁 **第1阶段：合并根目录核心功能文件**")
    merged_root = 0
    for file in high_priority_files:
        if file in analysis['only_in_newfile']:
            source = newfile_dir / file
            target = target_dir / file
            if source.exists():
                print(f"✅ 复制: {file}")
                shutil.copy2(source, target)
                merged_root += 1
            else:
                print(f"⚠️  未找到: {file}")
    
    # 合并 src 目录结构
    print("\n📁 **第2阶段：合并 src/ 目录结构**")
    merged_src = 0
    for file in src_priority_files:
        if file in analysis['only_in_newfile']:
            source = newfile_dir / file
            target = target_dir / file
            if source.exists():
                # 确保目录存在
                target.parent.mkdir(parents=True, exist_ok=True)
                print(f"✅ 复制: {file}")
                shutil.copy2(source, target)
                merged_src += 1
            else:
                print(f"⚠️  未找到: {file}")
    
    print(f"\n✅ **智能合并完成！**")
    print(f"📊 **合并统计:**")
    print(f"  - 根目录文件: {merged_root}")
    print(f"  - src目录文件: {merged_src}")
    print(f"  - 总计: {merged_root + merged_src}")
    
    return merged_root + merged_src

if __name__ == "__main__":
    smart_merge()
