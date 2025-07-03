#!/usr/bin/env python3
"""
自动归类失败样本并生成模式建议
分析失败样本的常见模式，为模式库补充提供建议
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


class FailureSampleClassifier:
    def __init__(self):
        self.failure_categories = {
            "missing_pattern": "缺少对应的模式匹配",
            "wrong_arithmetic": "算术运算错误",
            "entity_extraction_error": "实体提取错误", 
            "logic_error": "逻辑推理错误",
            "answer_format_error": "答案格式错误",
            "complex_multi_step": "复杂多步骤问题",
            "context_understanding_error": "上下文理解错误"
        }
        
    def classify_failure_reason(self, sample: Dict[str, Any]) -> str:
        """分类单个失败样本的原因"""
        problem = sample.get("problem", "")
        expected = sample.get("expected_answer", "")
        predicted = sample.get("predicted_answer", "")
        reasoning_steps = sample.get("reasoning_steps", [])
        
        # 检查是否完全没有答案
        if not predicted or predicted == "":
            return "missing_pattern"
            
        # 检查算术错误
        try:
            expected_num = float(expected)
            predicted_num = float(predicted)
            if abs(expected_num - predicted_num) > 0.1:
                return "wrong_arithmetic"
        except:
            pass
            
        # 检查模式匹配失败
        if any("No matching pattern found" in step for step in reasoning_steps):
            return "missing_pattern"
            
        # 检查实体提取问题
        if any("Extracted entities" in step for step in reasoning_steps):
            return "entity_extraction_error"
            
        # 检查复杂多步骤
        if len(reasoning_steps) > 5:
            return "complex_multi_step"
            
        return "logic_error"
    
    def extract_common_patterns(self, samples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """提取常见的问题模式"""
        patterns = defaultdict(list)
        
        for sample in samples:
            problem = sample.get("problem", "")
            
            # 提取数字和实体
            numbers = re.findall(r'\d+', problem)
            entities = re.findall(r'\b[A-Z][a-z]+\b', problem)
            
            # 提取问题类型
            if "How many" in problem:
                if "more" in problem:
                    patterns["comparison_questions"].append(problem)
                elif "total" in problem or "all" in problem:
                    patterns["summation_questions"].append(problem)
                else:
                    patterns["quantity_questions"].append(problem)
                    
            if "How much" in problem:
                patterns["money_questions"].append(problem)
                
            if "each" in problem:
                patterns["distribution_questions"].append(problem)
                
            if "left" in problem or "remaining" in problem:
                patterns["remaining_questions"].append(problem)
                
            if "initially" in problem or "originally" in problem:
                patterns["initial_state_questions"].append(problem)
                
            if "after" in problem and "before" in problem:
                patterns["temporal_questions"].append(problem)
                
        return dict(patterns)
    
    def generate_pattern_suggestions(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于失败样本生成模式建议"""
        suggestions = []
        
        # 分析常见问题模式
        common_patterns = self.extract_common_patterns(samples)
        
        # 为每种问题类型生成模式建议
        for pattern_type, problems in common_patterns.items():
            if len(problems) >= 3:  # 至少3个相似问题才建议模式
                suggestion = self._create_pattern_suggestion(pattern_type, problems)
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _create_pattern_suggestion(self, pattern_type: str, problems: List[str]) -> Dict[str, Any]:
        """为特定问题类型创建模式建议"""
        
        if pattern_type == "comparison_questions":
            return {
                "name": "comparison_difference",
                "type": "calculation",
                "pattern": "How many more \\w+ (?:than|did) (\\w+)",
                "template": "difference_between_entities",
                "description": "比较两个实体之间的差异",
                "examples": problems[:3]
            }
            
        elif pattern_type == "summation_questions":
            return {
                "name": "total_sum_calculation", 
                "type": "calculation",
                "pattern": "How many \\w+ (?:in all|in total|altogether)",
                "template": "sum_all_entities",
                "description": "计算所有实体的总和",
                "examples": problems[:3]
            }
            
        elif pattern_type == "money_questions":
            return {
                "name": "money_calculation",
                "type": "calculation", 
                "pattern": "How much (?:money|dollars) (?:did|does) \\w+",
                "template": "money_operation",
                "description": "金钱相关的计算",
                "examples": problems[:3]
            }
            
        elif pattern_type == "distribution_questions":
            return {
                "name": "equal_distribution",
                "type": "calculation",
                "pattern": "How many \\w+ did each \\w+",
                "template": "total / number_of_recipients",
                "description": "平均分配问题",
                "examples": problems[:3]
            }
            
        elif pattern_type == "remaining_questions":
            return {
                "name": "remaining_after_operation",
                "type": "calculation",
                "pattern": "How many \\w+ (?:left|remaining)",
                "template": "initial - used + gained",
                "description": "剩余数量计算",
                "examples": problems[:3]
            }
            
        elif pattern_type == "initial_state_questions":
            return {
                "name": "initial_state_calculation",
                "type": "calculation", 
                "pattern": "How many \\w+ (?:did|had) \\w+ (?:initially|originally)",
                "template": "final - changes",
                "description": "初始状态计算",
                "examples": problems[:3]
            }
            
        elif pattern_type == "temporal_questions":
            return {
                "name": "temporal_sequence",
                "type": "multi_step",
                "pattern": "\\w+ (?:before|after) \\w+",
                "template": "sequence_of_operations",
                "description": "时间序列问题",
                "examples": problems[:3]
            }
            
        return None
    
    def analyze_failures(self, failure_file: str) -> Dict[str, Any]:
        """分析失败样本文件"""
        with open(failure_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        # 分类失败原因
        failure_reasons = Counter()
        for sample in samples:
            reason = self.classify_failure_reason(sample)
            failure_reasons[reason] += 1
        
        # 生成模式建议
        pattern_suggestions = self.generate_pattern_suggestions(samples)
        
        # 提取常见关键词
        all_problems = [s.get("problem", "") for s in samples]
        keywords = self._extract_keywords(all_problems)
        
        return {
            "total_failures": len(samples),
            "failure_reasons": dict(failure_reasons),
            "pattern_suggestions": pattern_suggestions,
            "common_keywords": keywords,
            "sample_problems": all_problems[:10]  # 前10个样本问题
        }
    
    def _extract_keywords(self, problems: List[str]) -> Dict[str, int]:
        """提取常见关键词"""
        keyword_counter = Counter()
        
        for problem in problems:
            # 提取常见数学词汇
            math_keywords = re.findall(r'\b(?:more|less|total|each|left|remaining|initially|originally|after|before|times|divided|multiplied|added|subtracted)\b', problem.lower())
            keyword_counter.update(math_keywords)
            
            # 提取常见实体词汇
            entity_keywords = re.findall(r'\b(?:dollars|marbles|roses|crackers|trees|customers|points|games|kids|friends)\b', problem.lower())
            keyword_counter.update(entity_keywords)
        
        return dict(keyword_counter.most_common(20))
    
    def generate_enhanced_patterns(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于分析结果生成增强模式"""
        enhanced_patterns = []
        
        # 基于失败原因添加模式
        failure_reasons = analysis_result.get("failure_reasons", {})
        
        if failure_reasons.get("missing_pattern", 0) > 0:
            enhanced_patterns.extend([
                {
                    "name": "complex_initial_state",
                    "type": "multi_step",
                    "pattern": "(\\w+) had (\\d+) \\w+ initially",
                    "template": "{arg1}_initial = {arg2}",
                    "priority": "high"
                },
                {
                    "name": "final_state_given",
                    "type": "multi_step", 
                    "pattern": "(\\w+) has (\\d+) \\w+ now",
                    "template": "{arg1}_final = {arg2}",
                    "priority": "high"
                },
                {
                    "name": "change_operation",
                    "type": "multi_step",
                    "pattern": "(\\w+) (?:gained|lost|added|removed) (\\d+) \\w+",
                    "template": "{arg1}_change = {arg2}",
                    "priority": "high"
                }
            ])
        
        if failure_reasons.get("wrong_arithmetic", 0) > 0:
            enhanced_patterns.extend([
                {
                    "name": "spending_calculation",
                    "type": "calculation",
                    "pattern": "How much did \\w+ spend",
                    "template": "initial_money - remaining_money",
                    "priority": "medium"
                },
                {
                    "name": "earning_calculation", 
                    "type": "calculation",
                    "pattern": "How much money did \\w+ make",
                    "template": "final_money - initial_money",
                    "priority": "medium"
                }
            ])
        
        # 基于关键词添加模式
        keywords = analysis_result.get("common_keywords", {})
        
        if "times" in keywords:
            enhanced_patterns.append({
                "name": "multiplication_operation",
                "type": "binary_operation",
                "pattern": "(\\w+) (?:sold|made|has) (\\d+) times (?:as many|more) \\w+",
                "template": "{arg1} = {arg2} * base_value",
                "priority": "medium"
            })
        
        if "each" in keywords:
            enhanced_patterns.append({
                "name": "per_unit_calculation",
                "type": "calculation",
                "pattern": "How many \\w+ (?:per|each) \\w+",
                "template": "total_quantity / number_of_units",
                "priority": "medium"
            })
        
        return enhanced_patterns

def main():
    parser = argparse.ArgumentParser(description="自动归类失败样本并生成模式建议")
    parser.add_argument("--failure_file", default="failure_analysis.json", help="失败样本文件路径")
    parser.add_argument("--output_file", default="failure_analysis_report.json", help="输出报告文件路径")
    parser.add_argument("--enhanced_patterns_file", default="enhanced_patterns.json", help="增强模式输出文件")
    
    args = parser.parse_args()
    
    classifier = FailureSampleClassifier()
    
    print("开始分析失败样本...")
    analysis_result = classifier.analyze_failures(args.failure_file)
    
    # 生成增强模式
    enhanced_patterns = classifier.generate_enhanced_patterns(analysis_result)
    analysis_result["enhanced_patterns"] = enhanced_patterns
    
    # 保存分析报告
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    # 保存增强模式
    with open(args.enhanced_patterns_file, 'w', encoding='utf-8') as f:
        json.dump({"patterns": enhanced_patterns}, f, indent=2, ensure_ascii=False)
    
    print(f"分析完成！")
    print(f"总失败样本数: {analysis_result['total_failures']}")
    print(f"失败原因分布: {analysis_result['failure_reasons']}")
    print(f"生成模式建议数: {len(analysis_result['pattern_suggestions'])}")
    print(f"生成增强模式数: {len(enhanced_patterns)}")
    print(f"报告已保存到: {args.output_file}")
    print(f"增强模式已保存到: {args.enhanced_patterns_file}")

if __name__ == "__main__":
    main() 