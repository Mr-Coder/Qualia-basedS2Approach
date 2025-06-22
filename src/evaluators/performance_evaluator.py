"""
Performance Evaluator Module
============================

This module provides functionality to evaluate the performance of mathematical problem solving systems.

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    性能评估器
    
    用于评估数学问题求解系统的性能，包括：
    - 整体准确率评估
    - 按复杂度级别的性能分析
    - 鲁棒性分数计算
    """
    
    def __init__(self):
        """初始化性能评估器"""
        self.metrics = {}
        self.evaluation_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("PerformanceEvaluator initialized")
    
    def evaluate_overall_accuracy(self, predictions: List[Any], ground_truth: List[Any]) -> float:
        """
        整体准确率评估
        
        Args:
            predictions: 预测结果列表
            ground_truth: 真实答案列表
            
        Returns:
            float: 整体准确率 (0-1之间)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        if len(predictions) == 0:
            self.logger.warning("Empty prediction list provided")
            return 0.0
        
        try:
            correct = sum(1 for p, g in zip(predictions, ground_truth) if self._compare_answers(p, g))
            accuracy = correct / len(predictions)
            
            self.logger.info(f"Overall accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error calculating overall accuracy: {e}")
            return 0.0
    
    def evaluate_by_complexity_level(self, predictions: List[Any], ground_truth: List[Any], 
                                   complexity_labels: List[str]) -> Dict[str, float]:
        """
        按复杂度级别评估
        
        Args:
            predictions: 预测结果列表
            ground_truth: 真实答案列表
            complexity_labels: 复杂度标签列表 (L0, L1, L2, L3)
            
        Returns:
            Dict[str, float]: 各复杂度级别的准确率
        """
        if not (len(predictions) == len(ground_truth) == len(complexity_labels)):
            raise ValueError("All input lists must have the same length")
        
        level_results = {}
        level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        # 统计各级别的结果
        for pred, truth, level in zip(predictions, ground_truth, complexity_labels):
            level_stats[level]["total"] += 1
            if self._compare_answers(pred, truth):
                level_stats[level]["correct"] += 1
        
        # 计算各级别准确率
        for level in ["L0", "L1", "L2", "L3"]:
            if level in level_stats and level_stats[level]["total"] > 0:
                accuracy = level_stats[level]["correct"] / level_stats[level]["total"]
                level_results[level] = accuracy
                self.logger.info(f"Level {level} accuracy: {accuracy:.4f} "
                               f"({level_stats[level]['correct']}/{level_stats[level]['total']})")
            else:
                level_results[level] = 0.0
                self.logger.info(f"Level {level}: No samples found")
        
        return level_results
    
    def calculate_robustness_score(self, level_results: Dict[str, float]) -> float:
        """
        计算鲁棒性分数
        
        鲁棒性分数定义为最低性能级别与最高性能级别的比值
        
        Args:
            level_results: 各复杂度级别的准确率
            
        Returns:
            float: 鲁棒性分数 (0-1之间)
        """
        valid_scores = [score for score in level_results.values() if score > 0]
        
        if not valid_scores:
            self.logger.warning("No valid scores for robustness calculation")
            return 0.0
        
        if len(valid_scores) == 1:
            return 1.0
        
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        
        robustness = min_score / max_score if max_score > 0 else 0.0
        
        self.logger.info(f"Robustness score: {robustness:.4f} "
                        f"(min: {min_score:.4f}, max: {max_score:.4f})")
        
        return robustness
    
    def calculate_detailed_metrics(self, predictions: List[Any], ground_truth: List[Any], 
                                 complexity_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        计算详细的性能指标
        
        Args:
            predictions: 预测结果列表
            ground_truth: 真实答案列表
            complexity_labels: 复杂度标签列表（可选）
            
        Returns:
            Dict: 详细的性能指标
        """
        metrics = {}
        
        # 基础指标
        metrics["overall_accuracy"] = self.evaluate_overall_accuracy(predictions, ground_truth)
        metrics["total_samples"] = len(predictions)
        metrics["correct_predictions"] = sum(1 for p, g in zip(predictions, ground_truth) 
                                           if self._compare_answers(p, g))
        
        # 按复杂度级别的指标
        if complexity_labels:
            level_results = self.evaluate_by_complexity_level(predictions, ground_truth, complexity_labels)
            metrics["complexity_level_accuracy"] = level_results
            metrics["robustness_score"] = self.calculate_robustness_score(level_results)
            
            # 复杂度级别分布
            level_distribution = defaultdict(int)
            for level in complexity_labels:
                level_distribution[level] += 1
            metrics["complexity_distribution"] = dict(level_distribution)
        
        # 错误分析
        metrics["error_analysis"] = self._analyze_errors(predictions, ground_truth, complexity_labels)
        
        return metrics
    
    def generate_performance_report(self, metrics: Dict[str, Any], 
                                  output_path: Optional[str] = None) -> str:
        """
        生成性能评估报告
        
        Args:
            metrics: 性能指标
            output_path: 输出文件路径（可选）
            
        Returns:
            str: 报告内容
        """
        report_lines = [
            "=" * 60,
            "数学问题求解系统性能评估报告",
            "=" * 60,
            "",
            f"总样本数: {metrics.get('total_samples', 0)}",
            f"正确预测数: {metrics.get('correct_predictions', 0)}",
            f"整体准确率: {metrics.get('overall_accuracy', 0):.4f}",
            ""
        ]
        
        # 复杂度级别性能
        if "complexity_level_accuracy" in metrics:
            report_lines.extend([
                "按复杂度级别的性能:",
                "-" * 30
            ])
            
            for level, accuracy in metrics["complexity_level_accuracy"].items():
                count = metrics.get("complexity_distribution", {}).get(level, 0)
                report_lines.append(f"{level}: {accuracy:.4f} (样本数: {count})")
            
            report_lines.extend([
                "",
                f"鲁棒性分数: {metrics.get('robustness_score', 0):.4f}",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                self.logger.info(f"Performance report saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving report: {e}")
        
        return report_content
    
    def _compare_answers(self, prediction: Any, ground_truth: Any) -> bool:
        """比较预测答案和真实答案"""
        try:
            # 处理数值比较
            if isinstance(prediction, (int, float)) and isinstance(ground_truth, (int, float)):
                return abs(prediction - ground_truth) < 1e-6
            
            # 处理字符串比较
            if isinstance(prediction, str) and isinstance(ground_truth, str):
                return prediction.strip().lower() == ground_truth.strip().lower()
            
            # 默认比较
            return prediction == ground_truth
            
        except Exception as e:
            self.logger.warning(f"Error comparing answers: {e}")
            return False
    
    def _analyze_errors(self, predictions: List[Any], ground_truth: List[Any], 
                       complexity_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """分析错误模式"""
        error_analysis = {
            "total_errors": 0,
            "error_types": defaultdict(int),
            "error_by_complexity": defaultdict(int) if complexity_labels else None
        }
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if not self._compare_answers(pred, truth):
                error_analysis["total_errors"] += 1
                
                # 分类错误类型
                error_type = self._classify_error_type(pred, truth)
                error_analysis["error_types"][error_type] += 1
                
                # 按复杂度分析错误
                if complexity_labels and i < len(complexity_labels):
                    error_analysis["error_by_complexity"][complexity_labels[i]] += 1
        
        return dict(error_analysis)
    
    def _classify_error_type(self, prediction: Any, ground_truth: Any) -> str:
        """分类错误类型"""
        if prediction is None:
            return "no_prediction"
        
        if isinstance(prediction, str) and isinstance(ground_truth, str):
            if prediction.strip() == "":
                return "empty_prediction"
            return "text_mismatch"
        
        if isinstance(prediction, (int, float)) and isinstance(ground_truth, (int, float)):
            if abs(prediction - ground_truth) < 0.1:
                return "minor_numerical_error"
            else:
                return "major_numerical_error"
        
        return "type_mismatch"
    
    def export_metrics(self, metrics: Dict[str, Any], output_path: str) -> None:
        """导出性能指标到文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"Performance metrics exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            raise 