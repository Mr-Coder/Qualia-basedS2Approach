"""
统一数据集加载工具
支持加载所有数学推理数据集
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union


class MathDatasetLoader:
    """数学推理数据集统一加载器"""
    
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = Path(data_dir)
        self.available_datasets = self._scan_available_datasets()
    
    def _scan_available_datasets(self) -> Dict[str, str]:
        """扫描可用的数据集"""
        datasets = {}
        
        # 定义数据集文件映射
        dataset_files = {
            "AddSub": "AddSub/AddSub.json",
            "SingleEq": "SingleEq/SingleEq.json", 
            "MultiArith": "MultiArith/MultiArith.json",
            "GSM8K": "GSM8K/test.jsonl",
            "GSM-hard": "GSM-hard/gsmhard.jsonl",
            "SVAMP": "SVAMP/SVAMP.json",
            "AQuA": "AQuA/AQuA.json",
            "MAWPS": "MAWPS/mawps.json",
            "MathQA": "MathQA/mathqa.json",
            "MATH": "MATH/math_dataset.json",
            "Math23K": "Math23K/math23k.json",
            "ASDiv": "ASDiv/asdiv.json",
            "DIR-MWP": "DIR-MWP/dir_mwp_complete_dataset.json"
        }
        
        for name, file_path in dataset_files.items():
            full_path = self.data_dir / file_path
            if full_path.exists():
                datasets[name] = str(full_path)
        
        return datasets
    
    def list_datasets(self) -> List[str]:
        """列出所有可用的数据集"""
        return list(self.available_datasets.keys())
    
    def load_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        加载指定数据集
        
        Args:
            dataset_name: 数据集名称
            max_samples: 最大样本数，None表示加载全部
            
        Returns:
            数据集样本列表
        """
        if dataset_name not in self.available_datasets:
            raise ValueError(f"数据集 '{dataset_name}' 不存在。可用数据集: {self.list_datasets()}")
        
        file_path = self.available_datasets[dataset_name]
        
        # 根据文件扩展名选择加载方法
        if file_path.endswith('.jsonl'):
            data = self._load_jsonl(file_path)
        else:
            data = self._load_json(file_path)
        
        # 限制样本数
        if max_samples is not None:
            data = data[:max_samples]
        
        return data
    
    def _load_json(self, file_path: str) -> List[Dict]:
        """加载JSON格式文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是单个字典，转换为列表
        if isinstance(data, dict):
            return [data]
        
        return data
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL格式文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"数据集 '{dataset_name}' 不存在")
        
        data = self.load_dataset(dataset_name, max_samples=1)
        sample = data[0] if data else {}
        
        return {
            "name": dataset_name,
            "file_path": self.available_datasets[dataset_name],
            "sample_count": len(self.load_dataset(dataset_name)),
            "sample_fields": list(sample.keys()),
            "sample_example": sample
        }
    
    def get_all_datasets_info(self) -> Dict[str, Dict]:
        """获取所有数据集信息"""
        info = {}
        for dataset_name in self.list_datasets():
            try:
                info[dataset_name] = self.get_dataset_info(dataset_name)
            except Exception as e:
                info[dataset_name] = {"error": str(e)}
        return info
    
    def load_multiple_datasets(self, dataset_names: List[str], 
                             max_samples_per_dataset: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        加载多个数据集
        
        Args:
            dataset_names: 数据集名称列表
            max_samples_per_dataset: 每个数据集的最大样本数
            
        Returns:
            字典，键为数据集名称，值为样本列表
        """
        datasets = {}
        for name in dataset_names:
            try:
                datasets[name] = self.load_dataset(name, max_samples_per_dataset)
            except Exception as e:
                print(f"加载数据集 '{name}' 失败: {e}")
                datasets[name] = []
        
        return datasets
    
    def create_unified_format(self, dataset_name: str) -> List[Dict]:
        """
        将数据集转换为统一格式
        
        统一格式:
        {
            "id": "唯一标识",
            "problem": "问题描述", 
            "answer": "答案",
            "metadata": {"原始字段": "值"}
        }
        """
        data = self.load_dataset(dataset_name)
        unified_data = []
        
        for i, item in enumerate(data):
            # 提取问题描述
            problem = self._extract_problem(item)
            
            # 提取答案
            answer = self._extract_answer(item)
            
            # 生成统一格式
            unified_item = {
                "id": item.get("id", f"{dataset_name}_{i}"),
                "problem": problem,
                "answer": str(answer) if answer is not None else "",
                "dataset": dataset_name,
                "metadata": item
            }
            
            unified_data.append(unified_item)
        
        return unified_data
    
    def _extract_problem(self, item: Dict) -> str:
        """从数据项中提取问题描述"""
        # 常见的问题字段名
        problem_fields = ["problem", "question", "body", "text"]
        
        for field in problem_fields:
            if field in item:
                return str(item[field])
        
        # 如果有body和question，组合它们
        if "body" in item and "question" in item:
            return f"{item['body']} {item['question']}"
        
        return str(item)
    
    def _extract_answer(self, item: Dict) -> Union[str, int, float]:
        """从数据项中提取答案"""
        # 常见的答案字段名
        answer_fields = ["answer", "solution", "correct", "target"]
        
        for field in answer_fields:
            if field in item:
                return item[field]
        
        return None


def demo_usage():
    """演示使用方法"""
    loader = MathDatasetLoader()
    
    print("📊 可用数据集:")
    for dataset in loader.list_datasets():
        print(f"  • {dataset}")
    
    print("\n🔍 数据集详细信息:")
    for name, info in loader.get_all_datasets_info().items():
        if "error" not in info:
            print(f"  {name}: {info['sample_count']} 个样本")
        else:
            print(f"  {name}: 加载失败 - {info['error']}")
    
    # 加载示例数据集
    if "Math23K" in loader.list_datasets():
        print(f"\n📖 Math23K 示例:")
        samples = loader.load_dataset("Math23K", max_samples=2)
        for sample in samples:
            print(f"  问题: {sample.get('problem', 'N/A')}")
            print(f"  答案: {sample.get('answer', 'N/A')}")
            print()


if __name__ == "__main__":
    demo_usage() 