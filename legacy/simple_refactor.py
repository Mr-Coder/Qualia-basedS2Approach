#!/usr/bin/env python3
"""
简化项目重构脚本
只删除明确不需要的文件，保持项目功能完整

使用方法：
python simple_refactor.py [--dry-run]
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


class SimpleRefactor:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.project_root = Path(".")
        
        self.stats = {
            "files_deleted": 0,
            "dirs_deleted": 0,
            "bytes_freed": 0
        }
    
    def log(self, message, level="INFO"):
        prefix = "🔥 [预览]" if self.dry_run else "✅ [执行]"
        print(f"{prefix} {message}")
    
    def get_file_size(self, path):
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        except:
            return 0
        return 0
    
    def delete_file_or_dir(self, path):
        if not path.exists():
            return
        
        size = self.get_file_size(path)
        self.stats["bytes_freed"] += size
        
        if path.is_file():
            self.log(f"删除文件: {path}")
            self.stats["files_deleted"] += 1
            if not self.dry_run:
                path.unlink()
        elif path.is_dir():
            file_count = len(list(path.rglob('*')))
            self.log(f"删除目录: {path} ({file_count} 个文件)")
            self.stats["dirs_deleted"] += 1
            if not self.dry_run:
                shutil.rmtree(path)
    
    def clean_legacy_code(self):
        """清理遗留代码目录"""
        self.log("步骤 1: 清理遗留代码...")
        
        legacy_dir = self.project_root / "legacy"
        if legacy_dir.exists():
            self.delete_file_or_dir(legacy_dir)
        else:
            self.log("legacy目录不存在，跳过")
    
    def clean_temp_directories(self):
        """清理临时目录"""
        self.log("步骤 2: 清理临时目录...")
        
        temp_dirs = ["temp", "logs"]
        for temp_dir in temp_dirs:
            temp_path = self.project_root / temp_dir
            if temp_path.exists():
                self.delete_file_or_dir(temp_path)
            else:
                self.log(f"{temp_dir}目录不存在，跳过")
    
    def clean_cache_files(self):
        """清理缓存文件"""
        self.log("步骤 3: 清理Python缓存文件...")
        
        # 删除__pycache__目录
        for pycache_dir in self.project_root.rglob("__pycache__"):
            self.delete_file_or_dir(pycache_dir)
        
        # 删除.pyc文件
        for pyc_file in self.project_root.rglob("*.pyc"):
            self.delete_file_or_dir(pyc_file)
        
        # 删除.DS_Store文件
        for ds_store in self.project_root.rglob(".DS_Store"):
            self.delete_file_or_dir(ds_store)
    
    def clean_report_files(self):
        """清理生成的报告文件"""
        self.log("步骤 4: 清理生成的报告文件...")
        
        # 删除演示生成的JSON报告
        report_patterns = [
            "*_report_*.json",
            "*_results_*.json", 
            "*.log"
        ]
        
        for pattern in report_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    self.delete_file_or_dir(file_path)
    
    def clean_experimental_data(self):
        """清理实验数据（可选）"""
        self.log("步骤 5: 清理实验数据...")
        
        # 询问用户是否删除实验数据
        if not self.dry_run:
            response = input("是否删除experiments和analysis目录？(y/N): ")
            if response.lower() != 'y':
                self.log("跳过删除实验数据")
                return
        
        exp_dirs = ["experiments", "analysis"]
        for exp_dir in exp_dirs:
            exp_path = self.project_root / exp_dir
            if exp_path.exists():
                self.delete_file_or_dir(exp_path)
            else:
                self.log(f"{exp_dir}目录不存在，跳过")
    
    def organize_core_files(self):
        """整理核心文件信息"""
        self.log("步骤 6: 识别保留的核心文件...")
        
        core_files = [
            "interactive_demo.py",
            "detailed_step_by_step_demo.py",
            "quick_test.py",
            "演示使用说明.md",
            "演示总结.md",
            "cotdir_mlr_integration_demo.py",
            "gsm8k_cotdir_test.py",
            "AI_COLLABORATIVE_IMPLEMENTATION_SUMMARY.md",
            "COTDIR_MLR_FINAL_INTEGRATION_REPORT.md"
        ]
        
        existing_core = []
        for file_name in core_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                existing_core.append(file_name)
        
        self.log(f"核心文件总数: {len(existing_core)}")
        for file_name in existing_core:
            self.log(f"  保留: {file_name}")
    
    def print_statistics(self):
        """打印统计信息"""
        self.log("=" * 50)
        self.log("重构统计信息:")
        self.log(f"删除文件: {self.stats['files_deleted']}")
        self.log(f"删除目录: {self.stats['dirs_deleted']}")
        self.log(f"释放空间: {self.stats['bytes_freed'] / 1024 / 1024:.2f} MB")
        self.log("=" * 50)
    
    def run_refactor(self):
        """执行简化重构"""
        self.log("开始简化项目重构...")
        self.log(f"项目路径: {self.project_root.absolute()}")
        self.log(f"模式: {'预览模式' if self.dry_run else '实际执行'}")
        self.log("=" * 50)
        
        # 执行清理步骤
        self.clean_legacy_code()
        self.clean_temp_directories() 
        self.clean_cache_files()
        self.clean_report_files()
        self.clean_experimental_data()
        self.organize_core_files()
        
        # 打印统计信息
        self.print_statistics()
        
        if self.dry_run:
            self.log("这是预览模式，没有实际修改文件")
            self.log("要执行实际重构，请运行: python simple_refactor.py")
        else:
            self.log("重构完成!")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="简化项目重构脚本")
    parser.add_argument("--dry-run", action="store_true", 
                       help="仅显示操作，不实际执行")
    
    args = parser.parse_args()
    
    refactor = SimpleRefactor(dry_run=args.dry_run)
    
    try:
        success = refactor.run_refactor()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"重构过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 