#!/usr/bin/env python3
"""
项目重构自动化脚本
自动整理和重构 newfile 项目，删除不相关文件，保留核心功能

使用方法：
python project_refactor.py [--dry-run] [--backup]

选项：
  --dry-run    仅显示操作，不实际执行
  --backup     重构前创建备份
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


class ProjectRefactor:
    def __init__(self, dry_run=False, backup=True):
        self.dry_run = dry_run
        self.backup = backup
        self.project_root = Path(".")
        self.backup_dir = None
        
        # 统计信息
        self.stats = {
            "files_kept": 0,
            "files_deleted": 0,
            "dirs_deleted": 0,
            "bytes_freed": 0
        }
    
    def log(self, message, level="INFO"):
        """打印日志信息"""
        prefix = "🔥 [DRY-RUN]" if self.dry_run else "✅"
        print(f"{prefix} {level}: {message}")
    
    def get_file_size(self, path):
        """获取文件大小"""
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        except:
            return 0
        return 0
    
    def create_backup(self):
        """创建项目备份"""
        if not self.backup:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.project_root.parent / f"newfile_backup_{timestamp}"
        
        self.log(f"创建备份到: {self.backup_dir}")
        
        if not self.dry_run:
            try:
                shutil.copytree(self.project_root, self.backup_dir)
                self.log(f"备份创建成功: {self.backup_dir}")
            except Exception as e:
                self.log(f"备份创建失败: {e}", "ERROR")
                return False
        
        return True
    
    def delete_file_or_dir(self, path):
        """删除文件或目录"""
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
    
    def keep_file(self, path):
        """标记保留文件"""
        self.log(f"保留: {path}")
        self.stats["files_kept"] += 1
    
    def clean_legacy_files(self):
        """清理遗留代码"""
        self.log("=" * 60)
        self.log("清理遗留代码...")
        
        legacy_dir = self.project_root / "legacy"
        if legacy_dir.exists():
            self.delete_file_or_dir(legacy_dir)
    
    def clean_experimental_data(self):
        """清理实验数据"""
        self.log("=" * 60)
        self.log("清理实验数据...")
        
        # 删除实验目录
        for exp_dir in ["experiments", "analysis", "visualizations"]:
            exp_path = self.project_root / exp_dir
            if exp_path.exists():
                self.delete_file_or_dir(exp_path)
    
    def clean_temp_files(self):
        """清理临时文件"""
        self.log("=" * 60)
        self.log("清理临时文件...")
        
        # 删除临时目录
        temp_dirs = ["temp", "logs", "media", "__pycache__"]
        for temp_dir in temp_dirs:
            temp_path = self.project_root / temp_dir
            if temp_path.exists():
                self.delete_file_or_dir(temp_path)
        
        # 删除特定的临时文件
        temp_patterns = [
            "*_report_*.json",
            "*_results_*.json",
            "*.log",
            ".DS_Store",
            "*.pyc"
        ]
        
        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    self.delete_file_or_dir(file_path)
    
    def clean_llm_code(self):
        """清理LLM训练代码"""
        self.log("=" * 60)
        self.log("清理LLM训练代码...")
        
        llm_dir = self.project_root / "LLM-code"
        if llm_dir.exists():
            self.log("LLM-code目录可以移动到独立项目")
            # 不自动删除，用户可以手动处理
            # self.delete_file_or_dir(llm_dir)
    
    def clean_old_versions(self):
        """清理过时版本"""
        self.log("=" * 60)
        self.log("清理过时版本...")
        
        # src目录下的过时文件
        src_dir = self.project_root / "src"
        old_files = [
            "math_problem_solver.py",
            "math_problem_solver_optimized.py", 
            "math_problem_solver_v2.py",
            "performance_comparison.py",
            "test_optimized_solver.py"
        ]
        
        for old_file in old_files:
            file_path = src_dir / old_file
            if file_path.exists():
                self.delete_file_or_dir(file_path)
    
    def clean_documentation(self):
        """清理多余文档"""
        self.log("=" * 60)
        self.log("清理多余文档...")
        
        doc_dir = self.project_root / "documentation"
        if not doc_dir.exists():
            return
        
        # 保留的重要文档
        keep_docs = [
            "ALGORITHM_IMPLEMENTATION_REPORT.md",
            "COMPREHENSIVE_GSM8K_ANALYSIS_REPORT.md",
            "PROJECT_IMPLEMENTATION_SUMMARY.md"
        ]
        
        # 删除TABLE相关的详细文档
        for doc_file in doc_dir.rglob("TABLE*_RAW_DATA_FINAL_SUMMARY.md"):
            self.delete_file_or_dir(doc_file)
        
        for doc_file in doc_dir.rglob("table*_data_verification.md"):
            self.delete_file_or_dir(doc_file)
    
    def organize_core_files(self):
        """整理核心文件"""
        self.log("=" * 60)
        self.log("标记保留核心文件...")
        
        # 核心演示文件
        core_demos = [
            "interactive_demo.py",
            "detailed_step_by_step_demo.py", 
            "quick_test.py",
            "演示使用说明.md",
            "演示总结.md"
        ]
        
        for demo_file in core_demos:
            file_path = self.project_root / demo_file
            if file_path.exists():
                self.keep_file(file_path)
        
        # 核心集成文件
        core_integration = [
            "cotdir_mlr_integration_demo.py",
            "gsm8k_cotdir_test.py",
            "mlr_demo_final.py",
            "mlr_enhanced_demo_final.py",
            "ai_collaborative_demo.py"
        ]
        
        for integration_file in core_integration:
            file_path = self.project_root / integration_file
            if file_path.exists():
                self.keep_file(file_path)
        
        # 核心文档
        core_docs = [
            "README_COTDIR_MLR.md",
            "COTDIR_MLR_FINAL_INTEGRATION_REPORT.md",
            "MLR_OPTIMIZATION_FINAL_REPORT.md",
            "AI_COLLABORATIVE_IMPLEMENTATION_SUMMARY.md"
        ]
        
        for doc_file in core_docs:
            file_path = self.project_root / doc_file
            if file_path.exists():
                self.keep_file(file_path)
    
    def create_new_structure(self):
        """创建新的项目结构"""
        self.log("=" * 60)
        self.log("建议的新项目结构:")
        
        new_structure = {
            "math_reasoning_system": {
                "core": [
                    "cotdir_mlr_integration.py",
                    "mathematical_reasoning_system.py",
                    "data_structures.py"
                ],
                "demos": [
                    "interactive_demo.py",
                    "detailed_demo.py", 
                    "quick_test.py"
                ],
                "data": [
                    "gsm8k/",
                    "addsub/",
                    "svamp/"
                ],
                "config": [
                    "config.json"
                ],
                "tests": [
                    "system_tests/"
                ],
                "docs": [
                    "README.md",
                    "INTEGRATION_REPORT.md",
                    "USER_GUIDE.md"
                ]
            }
        }
        
        def print_structure(structure, indent=0):
            for key, value in structure.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    self.log(f"{prefix}📁 {key}/")
                    print_structure(value, indent + 1)
                elif isinstance(value, list):
                    self.log(f"{prefix}📁 {key}/")
                    for item in value:
                        self.log(f"{prefix}  ├── {item}")
                else:
                    self.log(f"{prefix}📄 {value}")
        
        print_structure(new_structure)
    
    def print_statistics(self):
        """打印统计信息"""
        self.log("=" * 60)
        self.log("重构统计信息:")
        self.log(f"保留文件: {self.stats['files_kept']}")
        self.log(f"删除文件: {self.stats['files_deleted']}")
        self.log(f"删除目录: {self.stats['dirs_deleted']}")
        self.log(f"释放空间: {self.stats['bytes_freed'] / 1024 / 1024:.2f} MB")
        
        if self.backup and self.backup_dir:
            self.log(f"备份位置: {self.backup_dir}")
    
    def run_refactor(self):
        """执行重构"""
        self.log("开始项目重构...")
        self.log(f"项目路径: {self.project_root.absolute()}")
        self.log(f"模式: {'预览模式' if self.dry_run else '实际执行'}")
        
        # 创建备份
        if not self.create_backup():
            return False
        
        # 标记保留的核心文件
        self.organize_core_files()
        
        # 执行清理操作
        self.clean_legacy_files()
        self.clean_experimental_data()
        self.clean_temp_files()
        self.clean_llm_code()
        self.clean_old_versions()
        self.clean_documentation()
        
        # 显示新结构建议
        self.create_new_structure()
        
        # 打印统计信息
        self.print_statistics()
        
        self.log("重构完成!")
        
        if self.dry_run:
            self.log("这是预览模式，没有实际修改文件")
            self.log("要执行实际重构，请运行: python project_refactor.py")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="项目重构自动化脚本")
    parser.add_argument("--dry-run", action="store_true", 
                       help="仅显示操作，不实际执行")
    parser.add_argument("--no-backup", action="store_true",
                       help="跳过备份创建")
    
    args = parser.parse_args()
    
    refactor = ProjectRefactor(
        dry_run=args.dry_run,
        backup=not args.no_backup
    )
    
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