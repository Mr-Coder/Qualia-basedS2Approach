"""
Complexity Classifier Module
===========================

This module provides functionality to classify mathematical problem complexity
and calculate DIR (Depth of Implicit Reasoning) scores.

The classifier categorizes problems into four levels:
- L0: Explicit problems (no implicit reasoning required)
- L1: Shallow implicit problems (minimal inference needed)
- L2: Medium implicit problems (moderate inference required)
- L3: Deep implicit problems (complex reasoning chains)

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ComplexityClassifier:
    """
    数学问题复杂度分类器
    
    根据推理深度(δ)和知识依赖(κ)对数学问题进行复杂度分级：
    - L0: 显式问题 (δ=0, κ=0)
    - L1: 浅层隐式 (δ=1, κ≤1)
    - L2: 中等隐式 (1<δ≤3, κ≤2)
    - L3: 深度隐式 (δ>3 或 κ>2)
    """
    
    def __init__(self):
        """初始化复杂度分类器"""
        self.implicit_relation_patterns = self.load_implicit_patterns()
        self.domain_knowledge_base = self.load_domain_knowledge()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ComplexityClassifier initialized")
    
    def load_implicit_patterns(self) -> Dict[str, List[str]]:
        """
        加载隐式关系模式
        
        Returns:
            Dict: 隐式关系模式字典
        """
        patterns = {
            "mathematical_operations": [
                r"总共|一共|合计",
                r"剩下|还剩|余下",
                r"每.*需要|每.*用",
                r"平均|均匀",
                r"增加.*倍|减少.*倍",
                r"比.*多|比.*少"
            ],
            "unit_conversions": [
                r"\d+\s*(米|厘米|毫米|千米)",
                r"\d+\s*(千克|克|吨)",
                r"\d+\s*(小时|分钟|秒)",
                r"\d+\s*(元|角|分)",
                r"\d+\s*(升|毫升)"
            ],
            "physical_constraints": [
                r"装满|盛满",
                r"最多|最少|至多|至少",
                r"不能超过|不得少于",
                r"容量|体积|面积"
            ],
            "temporal_relations": [
                r"之前|之后|同时",
                r"先.*后|首先.*然后",
                r"开始.*结束",
                r"第.*天|第.*小时"
            ],
            "geometric_properties": [
                r"长方形|正方形|圆形|三角形",
                r"长.*宽|半径|直径",
                r"周长|面积|体积",
                r"平行|垂直|相交"
            ],
            "proportional_relations": [
                r"正比|反比|成比例",
                r"速度.*时间|工作效率",
                r"单价.*数量",
                r"密度.*体积"
            ]
        }
        return patterns
    
    def load_domain_knowledge(self) -> Dict[str, List[str]]:
        """
        加载领域知识库
        
        Returns:
            Dict: 领域知识字典
        """
        knowledge_base = {
            "arithmetic": [
                "加法交换律", "乘法分配律", "除法性质"
            ],
            "geometry": [
                "面积公式", "周长公式", "体积公式", "相似三角形"
            ],
            "algebra": [
                "方程求解", "不等式", "函数关系"
            ],
            "physics": [
                "速度公式", "密度公式", "功率公式"
            ],
            "economics": [
                "利润计算", "折扣计算", "利息计算"
            ]
        }
        return knowledge_base
    
    def classify_problem_complexity(self, problem_text: str, solution_steps: Optional[List[str]] = None) -> str:
        """
        分类问题复杂度 (L0-L3)
        
        Args:
            problem_text: 问题文本
            solution_steps: 解题步骤（可选）
            
        Returns:
            str: 复杂度级别 ("L0", "L1", "L2", "L3")
        """
        try:
            # 分析推理深度 δ
            inference_depth = self.calculate_inference_depth(problem_text, solution_steps)
            
            # 分析知识依赖 κ
            knowledge_dependency = self.calculate_knowledge_dependency(problem_text)
            
            # 识别隐式关系数量
            implicit_relations = self.identify_implicit_relations(problem_text)
            
            self.logger.debug(f"Problem analysis - Inference depth: {inference_depth}, "
                            f"Knowledge dependency: {knowledge_dependency}, "
                            f"Implicit relations: {len(implicit_relations)}")
            
            # 计算复杂度级别
            if inference_depth == 0 and knowledge_dependency == 0:
                return "L0"  # 显式问题
            elif inference_depth == 1 and knowledge_dependency <= 1:
                return "L1"  # 浅层隐式
            elif 1 < inference_depth <= 3 and knowledge_dependency <= 2:
                return "L2"  # 中等隐式
            else:
                return "L3"  # 深度隐式
                
        except Exception as e:
            self.logger.error(f"Error classifying problem complexity: {e}")
            return "L0"  # 默认返回L0
    
    def calculate_inference_depth(self, problem_text: str, solution_steps: Optional[List[str]] = None) -> int:
        """
        计算推理深度 δ
        
        Args:
            problem_text: 问题文本
            solution_steps: 解题步骤
            
        Returns:
            int: 推理深度
        """
        depth = 0
        
        # 基于解题步骤计算深度
        if solution_steps:
            depth = len(solution_steps) - 1  # 减1因为最后一步通常是答案
            return max(0, depth)
        
        # 基于文本特征估算深度
        inference_indicators = [
            r"需要.*计算",
            r"首先.*然后.*最后",
            r"分.*步",
            r"根据.*可知.*所以",
            r"因为.*所以.*因此"
        ]
        
        for pattern in inference_indicators:
            matches = re.findall(pattern, problem_text)
            depth += len(matches)
        
        # 基于问句数量
        question_count = len(re.findall(r'[？?]', problem_text))
        if question_count > 1:
            depth += question_count - 1
        
        return min(depth, 5)  # 限制最大深度为5
    
    def calculate_knowledge_dependency(self, problem_text: str) -> int:
        """
        计算知识依赖 κ
        
        Args:
            problem_text: 问题文本
            
        Returns:
            int: 知识依赖级别
        """
        dependency_count = 0
        
        # 检查各领域知识依赖
        for domain, concepts in self.domain_knowledge_base.items():
            for concept in concepts:
                # 简化的概念匹配
                concept_keywords = concept.split()
                if any(keyword in problem_text for keyword in concept_keywords):
                    dependency_count += 1
                    break  # 每个领域最多计算一次
        
        # 检查公式依赖
        formula_patterns = [
            r"面积.*=.*长.*宽",
            r"速度.*=.*路程.*时间",
            r"总价.*=.*单价.*数量",
            r"利润.*=.*售价.*成本"
        ]
        
        for pattern in formula_patterns:
            if re.search(pattern, problem_text):
                dependency_count += 1
        
        return min(dependency_count, 3)  # 限制最大依赖为3
    
    def identify_implicit_relations(self, problem_text: str) -> List[Dict[str, Any]]:
        """
        识别隐式关系
        
        Args:
            problem_text: 问题文本
            
        Returns:
            List[Dict]: 识别到的隐式关系列表
        """
        relations = []
        
        for relation_type, patterns in self.implicit_relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, problem_text)
                for match in matches:
                    relations.append({
                        "type": relation_type,
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.span()
                    })
        
        return relations
    
    def calculate_dir_score(self, dataset: List[Dict[str, Any]]) -> Tuple[float, Dict[str, int]]:
        """
        计算数据集的DIR分数
        
        Args:
            dataset: 数据集
            
        Returns:
            Tuple[float, Dict]: DIR分数和各级别统计
        """
        level_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        
        for problem in dataset:
            problem_text = problem.get("question", problem.get("text", ""))
            solution_steps = problem.get("solution_steps")
            
            level = self.classify_problem_complexity(problem_text, solution_steps)
            level_counts[level] += 1
        
        total = len(dataset)
        if total == 0:
            return 0.0, level_counts
        
        # 计算加权DIR分数
        dir_score = (0 * level_counts["L0"] + 
                    1 * level_counts["L1"] + 
                    2 * level_counts["L2"] + 
                    3 * level_counts["L3"]) / total
        
        self.logger.info(f"DIR Score: {dir_score:.3f}, Distribution: {level_counts}")
        return dir_score, level_counts
    
    def analyze_dataset_complexity(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析数据集复杂度分布
        
        Args:
            dataset: 数据集
            
        Returns:
            Dict: 复杂度分析结果
        """
        dir_score, level_counts = self.calculate_dir_score(dataset)
        total = len(dataset)
        
        analysis = {
            "dir_score": dir_score,
            "total_problems": total,
            "level_distribution": level_counts,
            "level_percentages": {
                level: (count / total * 100) if total > 0 else 0
                for level, count in level_counts.items()
            },
            "complexity_summary": self._generate_complexity_summary(level_counts, total)
        }
        
        return analysis
    
    def batch_classify_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量分类问题复杂度
        
        Args:
            problems: 问题列表
            
        Returns:
            List[Dict]: 带有复杂度标签的问题列表
        """
        classified_problems = []
        
        for problem in problems:
            problem_text = problem.get("question", problem.get("text", ""))
            solution_steps = problem.get("solution_steps")
            
            complexity_level = self.classify_problem_complexity(problem_text, solution_steps)
            inference_depth = self.calculate_inference_depth(problem_text, solution_steps)
            knowledge_dependency = self.calculate_knowledge_dependency(problem_text)
            implicit_relations = self.identify_implicit_relations(problem_text)
            
            # 添加复杂度信息
            enhanced_problem = problem.copy()
            enhanced_problem.update({
                "complexity_level": complexity_level,
                "inference_depth": inference_depth,
                "knowledge_dependency": knowledge_dependency,
                "implicit_relations_count": len(implicit_relations),
                "implicit_relations": implicit_relations
            })
            
            classified_problems.append(enhanced_problem)
        
        return classified_problems
    
    def _generate_complexity_summary(self, level_counts: Dict[str, int], total: int) -> str:
        """
        生成复杂度摘要
        
        Args:
            level_counts: 各级别统计
            total: 总数
            
        Returns:
            str: 复杂度摘要
        """
        if total == 0:
            return "无数据"
        
        l3_percentage = level_counts["L3"] / total * 100
        l0_percentage = level_counts["L0"] / total * 100
        
        if l3_percentage > 30:
            return "高复杂度数据集（深度隐式问题占主导）"
        elif l0_percentage > 60:
            return "低复杂度数据集（显式问题占主导）"
        else:
            return "中等复杂度数据集（隐式问题分布均匀）"
    
    def export_complexity_analysis(self, analysis: Dict[str, Any], output_path: str) -> None:
        """
        导出复杂度分析结果
        
        Args:
            analysis: 分析结果
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Complexity analysis exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting complexity analysis: {e}")
            raise 