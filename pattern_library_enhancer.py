#!/usr/bin/env python3
"""
模式库增强工具
将基于失败样本分析生成的新模式合并到现有模式库中
"""

import json
import os
from typing import Any, Dict, List


class PatternLibraryEnhancer:
    def __init__(self):
        self.original_patterns_file = "src/reasoning_engine/patterns.json"
        self.enhanced_patterns_file = "src/reasoning_engine/enhanced_patterns.json"
        self.backup_patterns_file = "src/reasoning_engine/patterns_backup.json"
        
    def load_patterns(self, file_path: str) -> Dict[str, Any]:
        """加载模式文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 文件 {file_path} 不存在")
            return {"patterns": []}
        except json.JSONDecodeError as e:
            print(f"错误: 解析 {file_path} 时出错: {e}")
            return {"patterns": []}
    
    def save_patterns(self, patterns: Dict[str, Any], file_path: str):
        """保存模式到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=4, ensure_ascii=False)
    
    def merge_patterns(self, original_patterns: List[Dict], new_patterns: List[Dict]) -> List[Dict]:
        """合并模式，避免重复"""
        merged = original_patterns.copy()
        existing_names = {p.get("name", "") for p in original_patterns}
        
        for new_pattern in new_patterns:
            name = new_pattern.get("name", "")
            if name not in existing_names:
                merged.append(new_pattern)
                existing_names.add(name)
                print(f"添加新模式: {name}")
            else:
                print(f"跳过重复模式: {name}")
        
        return merged
    
    def enhance_pattern_library(self):
        """增强模式库"""
        print("开始增强模式库...")
        
        # 加载原始模式
        original_data = self.load_patterns(self.original_patterns_file)
        original_patterns = original_data.get("patterns", [])
        print(f"原始模式数量: {len(original_patterns)}")
        
        # 加载增强模式
        enhanced_data = self.load_patterns(self.enhanced_patterns_file)
        enhanced_patterns = enhanced_data.get("patterns", [])
        print(f"增强模式数量: {len(enhanced_patterns)}")
        
        # 备份原始模式
        self.save_patterns(original_data, self.backup_patterns_file)
        print(f"原始模式已备份到: {self.backup_patterns_file}")
        
        # 合并模式
        merged_patterns = self.merge_patterns(original_patterns, enhanced_patterns)
        
        # 创建增强后的模式库
        enhanced_library = {
            "patterns": merged_patterns,
            "metadata": {
                "total_patterns": len(merged_patterns),
                "original_patterns": len(original_patterns),
                "new_patterns": len(enhanced_patterns),
                "enhanced_date": "2024-12-30"
            }
        }
        
        # 保存增强后的模式库
        self.save_patterns(enhanced_library, self.original_patterns_file)
        print(f"增强模式库已保存到: {self.original_patterns_file}")
        print(f"总模式数量: {len(merged_patterns)}")
        
        return enhanced_library
    
    def create_pattern_summary(self, patterns: List[Dict]) -> Dict[str, Any]:
        """创建模式摘要"""
        summary = {
            "total_patterns": len(patterns),
            "pattern_types": {},
            "priority_distribution": {},
            "pattern_names": []
        }
        
        for pattern in patterns:
            # 统计类型
            pattern_type = pattern.get("type", "unknown")
            summary["pattern_types"][pattern_type] = summary["pattern_types"].get(pattern_type, 0) + 1
            
            # 统计优先级
            priority = pattern.get("priority", "low")
            summary["priority_distribution"][priority] = summary["priority_distribution"].get(priority, 0) + 1
            
            # 收集模式名称
            name = pattern.get("name", "unnamed")
            summary["pattern_names"].append(name)
        
        return summary
    
    def generate_enhancement_report(self, original_count: int, enhanced_count: int) -> str:
        """生成增强报告"""
        report = f"""
# 模式库增强报告

## 增强统计
- 原始模式数量: {original_count}
- 新增模式数量: {enhanced_count - original_count}
- 增强后总数量: {enhanced_count}
- 增长率: {((enhanced_count - original_count) / original_count * 100):.1f}%

## 新增模式类型
基于失败样本分析，新增了以下类型的模式：

1. **多步骤模式 (multi_step)**: 处理复杂的状态变化问题
2. **计算模式 (calculation)**: 处理各种数学计算问题
3. **二元操作模式 (binary_operation)**: 处理两个实体之间的关系

## 优先级分布
- 高优先级模式: 处理核心失败场景
- 中优先级模式: 处理常见问题类型
- 低优先级模式: 处理边缘情况

## 建议
1. 测试新增强的模式库性能
2. 根据测试结果进一步优化模式
3. 定期分析失败样本并更新模式库
"""
        return report

def main():
    enhancer = PatternLibraryEnhancer()
    
    # 增强模式库
    enhanced_library = enhancer.enhance_pattern_library()
    
    # 生成摘要
    summary = enhancer.create_pattern_summary(enhanced_library["patterns"])
    
    # 生成报告
    report = enhancer.generate_enhancement_report(
        enhanced_library["metadata"]["original_patterns"],
        enhanced_library["metadata"]["total_patterns"]
    )
    
    # 保存报告
    with open("pattern_enhancement_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n=== 模式库增强完成 ===")
    print(f"模式类型分布: {summary['pattern_types']}")
    print(f"优先级分布: {summary['priority_distribution']}")
    print(f"增强报告已保存到: pattern_enhancement_report.md")

if __name__ == "__main__":
    main() 