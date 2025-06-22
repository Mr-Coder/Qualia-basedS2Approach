"""
Reasoning Chain Evaluator Module
================================

This module provides functionality to evaluate the quality of reasoning chains
in mathematical problem solving systems.

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReasoningChainEvaluator:
    """
    推理链评估器
    
    用于评估推理链的质量，包括：
    - 逻辑正确性评估
    - 完整性检查
    - 连贯性分析
    - 效率性评估
    - 可验证性检查
    - 错误传播分析
    """
    
    def __init__(self):
        """初始化推理链评估器"""
        self.quality_dimensions = [
            "logical_correctness",
            "completeness", 
            "coherence",
            "efficiency",
            "verifiability"
        ]
        
        self.evaluation_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ReasoningChainEvaluator initialized")
    
    def evaluate_reasoning_chain_quality(self, reasoning_chain: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估推理链质量
        
        Args:
            reasoning_chain: 推理链，包含推理步骤的列表
            
        Returns:
            Dict[str, float]: 各维度的质量分数
        """
        if not reasoning_chain:
            self.logger.warning("Empty reasoning chain provided")
            return {dim: 0.0 for dim in self.quality_dimensions + ["overall"]}
        
        try:
            scores = {}
            
            # 逻辑正确性
            scores["logical_correctness"] = self.check_logical_correctness(reasoning_chain)
            
            # 完整性
            scores["completeness"] = self.check_completeness(reasoning_chain)
            
            # 连贯性
            scores["coherence"] = self.check_coherence(reasoning_chain)
            
            # 效率性
            scores["efficiency"] = self.check_efficiency(reasoning_chain)
            
            # 可验证性
            scores["verifiability"] = self.check_verifiability(reasoning_chain)
            
            # 总体分数
            scores["overall"] = sum(scores.values()) / len(scores)
            
            self.logger.info(f"Reasoning chain quality: Overall={scores['overall']:.3f}")
            return scores
            
        except Exception as e:
            self.logger.error(f"Error evaluating reasoning chain quality: {e}")
            return {dim: 0.0 for dim in self.quality_dimensions + ["overall"]}
    
    def check_logical_correctness(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """
        检查逻辑正确性
        
        Args:
            reasoning_chain: 推理链
            
        Returns:
            float: 逻辑正确性分数 (0-1)
        """
        try:
            if not reasoning_chain:
                return 0.0
            
            correct_steps = 0
            total_steps = len(reasoning_chain)
            
            for i, step in enumerate(reasoning_chain):
                # 检查步骤的逻辑有效性
                if self._is_logically_valid_step(step, reasoning_chain[:i]):
                    correct_steps += 1
                else:
                    self.logger.debug(f"Logical error detected in step {i}: {step.get('description', '')}")
            
            score = correct_steps / total_steps
            self.logger.debug(f"Logical correctness: {score:.3f} ({correct_steps}/{total_steps})")
            return score
            
        except Exception as e:
            self.logger.error(f"Error checking logical correctness: {e}")
            return 0.0
    
    def check_completeness(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """
        检查完整性
        
        Args:
            reasoning_chain: 推理链
            
        Returns:
            float: 完整性分数 (0-1)
        """
        try:
            if not reasoning_chain:
                return 0.0
            
            completeness_factors = {
                "has_problem_analysis": 0.0,
                "has_solution_steps": 0.0,
                "has_verification": 0.0,
                "covers_all_requirements": 0.0,
                "reaches_conclusion": 0.0
            }
            
            # 检查是否有问题分析
            if any("analysis" in step.get("type", "") or "understand" in step.get("type", "") 
                   for step in reasoning_chain):
                completeness_factors["has_problem_analysis"] = 1.0
            
            # 检查是否有解题步骤
            solution_steps = [step for step in reasoning_chain 
                            if step.get("type", "") in ["calculation", "operation", "solve"]]
            if solution_steps:
                completeness_factors["has_solution_steps"] = min(1.0, len(solution_steps) / 3)
            
            # 检查是否有验证步骤
            if any("verify" in step.get("type", "") or "check" in step.get("type", "") 
                   for step in reasoning_chain):
                completeness_factors["has_verification"] = 1.0
            
            # 检查是否覆盖所有要求
            completeness_factors["covers_all_requirements"] = self._check_requirement_coverage(reasoning_chain)
            
            # 检查是否达到结论
            if reasoning_chain and "conclusion" in reasoning_chain[-1].get("type", ""):
                completeness_factors["reaches_conclusion"] = 1.0
            
            score = sum(completeness_factors.values()) / len(completeness_factors)
            self.logger.debug(f"Completeness: {score:.3f}")
            return score
            
        except Exception as e:
            self.logger.error(f"Error checking completeness: {e}")
            return 0.0
    
    def check_coherence(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """
        检查连贯性
        
        Args:
            reasoning_chain: 推理链
            
        Returns:
            float: 连贯性分数 (0-1)
        """
        try:
            if len(reasoning_chain) <= 1:
                return 1.0
            
            coherence_score = 0.0
            transition_count = 0
            
            for i in range(1, len(reasoning_chain)):
                prev_step = reasoning_chain[i-1]
                curr_step = reasoning_chain[i]
                
                # 检查步骤间的逻辑连接
                if self._check_step_transition(prev_step, curr_step):
                    coherence_score += 1.0
                
                transition_count += 1
            
            score = coherence_score / transition_count if transition_count > 0 else 1.0
            self.logger.debug(f"Coherence: {score:.3f}")
            return score
            
        except Exception as e:
            self.logger.error(f"Error checking coherence: {e}")
            return 0.0
    
    def check_efficiency(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """
        检查效率性
        
        Args:
            reasoning_chain: 推理链
            
        Returns:
            float: 效率性分数 (0-1)
        """
        try:
            if not reasoning_chain:
                return 0.0
            
            # 步骤数量评估（适中的步骤数更好）
            step_count = len(reasoning_chain)
            if 3 <= step_count <= 8:
                step_score = 1.0
            elif step_count < 3:
                step_score = step_count / 3
            else:
                step_score = max(0.2, 8 / step_count)
            
            # 冗余度评估
            redundancy_score = 1.0 - self._calculate_redundancy(reasoning_chain)
            
            # 综合效率分数
            score = (step_score + redundancy_score) / 2
            self.logger.debug(f"Efficiency: {score:.3f}")
            return score
            
        except Exception as e:
            self.logger.error(f"Error checking efficiency: {e}")
            return 0.0
    
    def check_verifiability(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """
        检查可验证性
        
        Args:
            reasoning_chain: 推理链
            
        Returns:
            float: 可验证性分数 (0-1)
        """
        try:
            if not reasoning_chain:
                return 0.0
            
            verifiable_steps = 0
            total_steps = len(reasoning_chain)
            
            for step in reasoning_chain:
                if self._is_verifiable_step(step):
                    verifiable_steps += 1
            
            score = verifiable_steps / total_steps
            self.logger.debug(f"Verifiability: {score:.3f} ({verifiable_steps}/{total_steps})")
            return score
            
        except Exception as e:
            self.logger.error(f"Error checking verifiability: {e}")
            return 0.0
    
    def analyze_error_propagation(self, failed_chains: List[List[Dict[str, Any]]]) -> Dict[str, int]:
        """
        分析错误传播模式
        
        Args:
            failed_chains: 失败的推理链列表
            
        Returns:
            Dict[str, int]: 错误模式统计
        """
        error_patterns = {
            "early_stage": 0,    # 早期阶段错误
            "middle_stage": 0,   # 中期阶段错误
            "late_stage": 0,     # 后期阶段错误
            "verification_recovery": 0  # 验证恢复
        }
        
        try:
            for chain in failed_chains:
                if not chain:
                    continue
                
                error_stage = self.identify_error_stage(chain)
                error_patterns[error_stage] += 1
                
                # 检查是否有验证恢复
                if self._has_verification_recovery(chain):
                    error_patterns["verification_recovery"] += 1
            
            self.logger.info(f"Error propagation analysis: {error_patterns}")
            return error_patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing error propagation: {e}")
            return error_patterns
    
    def identify_error_stage(self, reasoning_chain: List[Dict[str, Any]]) -> str:
        """
        识别错误发生的阶段
        
        Args:
            reasoning_chain: 推理链
            
        Returns:
            str: 错误阶段
        """
        try:
            if not reasoning_chain:
                return "early_stage"
            
            chain_length = len(reasoning_chain)
            
            # 查找第一个错误步骤
            for i, step in enumerate(reasoning_chain):
                if step.get("has_error", False) or not self._is_logically_valid_step(step, reasoning_chain[:i]):
                    if i < chain_length * 0.33:
                        return "early_stage"
                    elif i < chain_length * 0.67:
                        return "middle_stage"
                    else:
                        return "late_stage"
            
            return "late_stage"  # 默认为后期错误
            
        except Exception as e:
            self.logger.warning(f"Error identifying error stage: {e}")
            return "early_stage"
    
    def _is_logically_valid_step(self, step: Dict[str, Any], 
                               previous_steps: List[Dict[str, Any]]) -> bool:
        """检查步骤的逻辑有效性"""
        try:
            step_type = step.get("type", "")
            step_content = step.get("content", "")
            
            # 基本有效性检查
            if not step_content or step_type == "error":
                return False
            
            # 检查是否有明确的错误标记
            if step.get("has_error", False) or step.get("is_incorrect", False):
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating step: {e}")
            return False
    
    def _check_requirement_coverage(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """检查是否覆盖所有要求"""
        required_elements = ["problem_understanding", "solution_method", "calculation", "answer"]
        covered_elements = set()
        
        for step in reasoning_chain:
            step_type = step.get("type", "")
            if "understand" in step_type or "analysis" in step_type:
                covered_elements.add("problem_understanding")
            elif "method" in step_type or "approach" in step_type:
                covered_elements.add("solution_method")
            elif "calculation" in step_type or "compute" in step_type:
                covered_elements.add("calculation")
            elif "answer" in step_type or "result" in step_type:
                covered_elements.add("answer")
        
        return len(covered_elements) / len(required_elements)
    
    def _check_step_transition(self, prev_step: Dict[str, Any], 
                             curr_step: Dict[str, Any]) -> bool:
        """检查步骤间的转换是否合理"""
        try:
            prev_type = prev_step.get("type", "")
            curr_type = curr_step.get("type", "")
            
            # 定义合理的转换模式
            valid_transitions = {
                "analysis": ["method", "calculation", "inference"],
                "method": ["calculation", "operation"],
                "calculation": ["calculation", "verification", "answer"],
                "operation": ["operation", "verification", "answer"],
                "inference": ["calculation", "verification", "conclusion"],
                "verification": ["answer", "conclusion", "correction"],
                "answer": ["verification", "conclusion"]
            }
            
            return curr_type in valid_transitions.get(prev_type, [curr_type])
            
        except Exception:
            return True
    
    def _calculate_redundancy(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """计算冗余度"""
        if len(reasoning_chain) <= 1:
            return 0.0
        
        try:
            unique_operations = set()
            total_operations = 0
            
            for step in reasoning_chain:
                operation = step.get("operation", step.get("type", ""))
                if operation:
                    unique_operations.add(operation)
                    total_operations += 1
            
            if total_operations == 0:
                return 0.0
            
            redundancy = 1.0 - (len(unique_operations) / total_operations)
            return max(0.0, redundancy)
            
        except Exception:
            return 0.0
    
    def _is_verifiable_step(self, step: Dict[str, Any]) -> bool:
        """检查步骤是否可验证"""
        try:
            # 检查是否有足够的信息进行验证
            has_input = bool(step.get("input") or step.get("operands"))
            has_operation = bool(step.get("operation") or step.get("method"))
            has_output = bool(step.get("output") or step.get("result"))
            
            return has_input and has_operation and has_output
            
        except Exception:
            return False
    
    def _has_verification_recovery(self, reasoning_chain: List[Dict[str, Any]]) -> bool:
        """检查是否有验证恢复"""
        try:
            has_error = False
            has_recovery = False
            
            for step in reasoning_chain:
                if step.get("has_error", False):
                    has_error = True
                elif has_error and step.get("type", "") in ["verification", "correction"]:
                    has_recovery = True
                    break
            
            return has_error and has_recovery
            
        except Exception:
            return False
    
    def export_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        """导出评估结果"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"Evaluation results exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting evaluation results: {e}")
            raise 