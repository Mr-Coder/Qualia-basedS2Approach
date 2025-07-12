"""
Dataset Loader Module
====================

This module provides functionality to load various mathematical problem datasets
including Math23K, GSM8K, MAWPS, MathQA, MATH, SVAMP, and ASDiv.

Author: Math Problem Solver Team
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    数据集加载器，支持加载多种数学题数据集
    
    支持的数据集:
    - Math23K: 中文数学题数据集 (23,162题)
    - GSM8K: 英文小学数学题 (8,500题)
    - MAWPS: 多领域数学题 (2,373题)
    - MathQA: 竞赛数学题 (37,297题)
    - MATH: 竞赛数学题 (12,500题)
    - SVAMP: 小学数学题 (1,000题)
    - ASDiv: 小学数学题 (2,305题)
    """
    
    def __init__(self):
        """初始化数据集加载器"""
        self.datasets = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("DatasetLoader initialized")
    
    def load_math23k(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载Math23K中文数学题数据集 (23,162题)
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            List[Dict]: 包含问题、方程和答案的字典列表
            数据格式: {"question": "...", "equation": "...", "answer": "..."}
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix == '.json':
                    data = json.load(f)
                elif path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # 标准化数据格式
            standardized_data = []
            for item in data:
                standardized_item = {
                    "question": item.get("question", item.get("text", "")),
                    "equation": item.get("equation", item.get("formula", "")),
                    "answer": item.get("answer", item.get("result", "")),
                    "dataset": "Math23K",
                    "language": "zh"
                }
                standardized_data.append(standardized_item)
            
            self.datasets["Math23K"] = standardized_data
            self.logger.info(f"Loaded Math23K dataset: {len(standardized_data)} problems")
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"Error loading Math23K dataset: {e}")
            raise
    
    def load_gsm8k(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载GSM8K英文小学数学题 (8,500题)
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            List[Dict]: 标准化的问题数据
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix == '.json':
                    data = json.load(f)
                elif path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # 标准化数据格式
            standardized_data = []
            for item in data:
                standardized_item = {
                    "question": item.get("question", item.get("problem", "")),
                    "equation": self._extract_equation_from_solution(item.get("answer", "")),
                    "answer": self._extract_final_answer(item.get("answer", "")),
                    "dataset": "GSM8K",
                    "language": "en",
                    "grade_level": "elementary"
                }
                standardized_data.append(standardized_item)
            
            self.datasets["GSM8K"] = standardized_data
            self.logger.info(f"Loaded GSM8K dataset: {len(standardized_data)} problems")
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"Error loading GSM8K dataset: {e}")
            raise
    
    def load_mawps(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载MAWPS多领域数学题 (2,373题)
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            List[Dict]: 标准化的问题数据
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 标准化数据格式
            standardized_data = []
            for item in data:
                standardized_item = {
                    "question": item.get("sQuestion", item.get("question", "")),
                    "equation": item.get("lEquations", item.get("equation", "")),
                    "answer": item.get("lSolutions", item.get("answer", "")),
                    "dataset": "MAWPS",
                    "language": "en",
                    "domain": item.get("iIndex", "general")
                }
                standardized_data.append(standardized_item)
            
            self.datasets["MAWPS"] = standardized_data
            self.logger.info(f"Loaded MAWPS dataset: {len(standardized_data)} problems")
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"Error loading MAWPS dataset: {e}")
            raise
    
    def load_mathqa(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载MathQA竞赛数学题 (37,297题)
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            List[Dict]: 标准化的问题数据
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            # 标准化数据格式
            standardized_data = []
            for item in data:
                standardized_item = {
                    "question": item.get("Problem", item.get("question", "")),
                    "equation": item.get("linear_formula", item.get("equation", "")),
                    "answer": item.get("correct", item.get("answer", "")),
                    "dataset": "MathQA",
                    "language": "en",
                    "category": item.get("category", "competition"),
                    "options": item.get("options", [])
                }
                standardized_data.append(standardized_item)
            
            self.datasets["MathQA"] = standardized_data
            self.logger.info(f"Loaded MathQA dataset: {len(standardized_data)} problems")
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"Error loading MathQA dataset: {e}")
            raise
    
    def load_math_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载MATH竞赛数学题 (12,500题)
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            List[Dict]: 标准化的问题数据
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            # 标准化数据格式
            standardized_data = []
            for item in data:
                standardized_item = {
                    "question": item.get("problem", item.get("question", "")),
                    "equation": self._extract_equation_from_solution(item.get("solution", "")),
                    "answer": item.get("solution", item.get("answer", "")),
                    "dataset": "MATH",
                    "language": "en",
                    "level": item.get("level", "unknown"),
                    "type": item.get("type", "competition")
                }
                standardized_data.append(standardized_item)
            
            self.datasets["MATH"] = standardized_data
            self.logger.info(f"Loaded MATH dataset: {len(standardized_data)} problems")
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"Error loading MATH dataset: {e}")
            raise
    
    def load_svamp(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载SVAMP小学数学题 (1,000题)
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            List[Dict]: 标准化的问题数据
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 标准化数据格式
            standardized_data = []
            for item in data:
                standardized_item = {
                    "question": item.get("Body", item.get("question", "")),
                    "equation": item.get("Equation", item.get("equation", "")),
                    "answer": item.get("Answer", item.get("answer", "")),
                    "dataset": "SVAMP",
                    "language": "en",
                    "grade_level": "elementary",
                    "question_id": item.get("ID", "")
                }
                standardized_data.append(standardized_item)
            
            self.datasets["SVAMP"] = standardized_data
            self.logger.info(f"Loaded SVAMP dataset: {len(standardized_data)} problems")
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"Error loading SVAMP dataset: {e}")
            raise
    
    def load_asdiv(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载ASDiv小学数学题 (2,305题)
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            List[Dict]: 标准化的问题数据
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 标准化数据格式
            standardized_data = []
            for item in data:
                standardized_item = {
                    "question": item.get("sQuestion", item.get("question", "")),
                    "equation": item.get("lEquations", item.get("equation", "")),
                    "answer": item.get("lSolutions", item.get("answer", "")),
                    "dataset": "ASDiv",
                    "language": "en",
                    "grade_level": item.get("grade", "elementary")
                }
                standardized_data.append(standardized_item)
            
            self.datasets["ASDiv"] = standardized_data
            self.logger.info(f"Loaded ASDiv dataset: {len(standardized_data)} problems")
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"Error loading ASDiv dataset: {e}")
            raise
    
    def get_dataset(self, dataset_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        获取已加载的数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集数据或None
        """
        return self.datasets.get(dataset_name)
    
    def get_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取所有已加载的数据集
        
        Returns:
            所有数据集的字典
        """
        return self.datasets.copy()
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取数据集统计信息
        
        Returns:
            每个数据集的统计信息
        """
        stats = {}
        for name, dataset in self.datasets.items():
            stats[name] = {
                "count": len(dataset),
                "has_equations": sum(1 for item in dataset if item.get("equation")),
                "has_answers": sum(1 for item in dataset if item.get("answer")),
                "languages": list(set(item.get("language", "unknown") for item in dataset))
            }
        return stats
    
    def _extract_equation_from_solution(self, solution: str) -> str:
        """从解答中提取方程"""
        # 这里可以实现更复杂的方程提取逻辑
        # 目前返回简化版本
        if not solution:
            return ""
        
        # 简单的方程提取逻辑
        import re
        equation_patterns = [
            r'(\d+[\+\-\*\/]\d+(?:[\+\-\*\/]\d+)*)',
            r'([a-zA-Z]\s*=\s*\d+(?:[\+\-\*\/]\d+)*)',
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, solution)
            if matches:
                return matches[0]
        
        return ""
    
    def _extract_final_answer(self, solution: str) -> str:
        """从解答中提取最终答案"""
        if not solution:
            return ""
        
        # 简单的答案提取逻辑
        import re
        answer_patterns = [
            r'答案是\s*(\d+(?:\.\d+)?)',
            r'答案：\s*(\d+(?:\.\d+)?)',
            r'=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*$'
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, solution)
            if matches:
                return matches[-1]  # 返回最后一个匹配的数字
        
        return "" 