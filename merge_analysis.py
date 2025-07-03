#!/usr/bin/env python3
import os
import json
from pathlib import Path
from datetime import datetime

def analyze_directories():
    """分析两个目录的差异"""
    cot_dir = Path(".")
    newfile_dir = Path("~/Desktop/newfile").expanduser()
    
    print(f"🔍 分析目录:")
    print(f"  cot-dir1: {cot_dir.absolute()}")
    print(f"  newfile: {newfile_dir.absolute()}")
    print(f"  newfile 存在: {newfile_dir.exists()}")
    
    # 获取所有文件
    cot_files = set()
    newfile_files = set()
    
    for root, dirs, files in os.walk(cot_dir):
        # 跳过 .git 和 __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                rel_path = os.path.relpath(os.path.join(root, file), cot_dir)
                cot_files.add(rel_path)
    
    if newfile_dir.exists():
        for root, dirs, files in os.walk(newfile_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    rel_path = os.path.relpath(os.path.join(root, file), newfile_dir)
                    newfile_files.add(rel_path)
    
    # 分析差异
    only_in_cot = cot_files - newfile_files
    only_in_newfile = newfile_files - cot_files
    common_files = cot_files & newfile_files
    
    print("\n📊 **文件分析报告**")
    print(f"cot-dir1 独有文件: {len(only_in_cot)}")
    print(f"newfile 独有文件: {len(only_in_newfile)}")
    print(f"共同文件: {len(common_files)}")
    
    print("\n🔍 **newfile 独有的重要 Python 文件:**")
    important_new_files = [f for f in only_in_newfile if f.endswith('.py') and not f.startswith('test_')]
    for f in sorted(important_new_files)[:30]:  # 显示前30个
        print(f"  - {f}")
    
    print("\n📁 **newfile 独有的目录结构:**")
    new_dirs = set()
    for f in only_in_newfile:
        parts = Path(f).parts
        if len(parts) > 1:
            new_dirs.add(parts[0])
    for d in sorted(new_dirs):
        print(f"  - {d}/")
    
    # 保存分析结果
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'only_in_cot': sorted(list(only_in_cot)),
        'only_in_newfile': sorted(list(only_in_newfile)),
        'common_files': sorted(list(common_files))
    }
    
    with open('merge_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ 分析结果已保存到 merge_analysis.json")
    return analysis

if __name__ == "__main__":
    analyze_directories()
