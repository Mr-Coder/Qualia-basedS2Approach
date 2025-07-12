import json
import os
from typing import Dict, List, Optional


class DataLoader:
    """
    Minimal DataLoader for loading math datasets from Data/ directory.
    """
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = data_dir

    def load(self, dataset_name: Optional[str] = None, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load dataset by name (from Data/ directory) or by path. Supports .json and .jsonl files.
        Args:
            dataset_name: Name of the dataset (e.g., 'Math23K')
            path: Direct path to dataset file
            max_samples: Maximum number of samples to load
        Returns:
            List of dict samples
        """
        if not dataset_name and not path:
            raise ValueError("Either dataset_name or path must be provided")
        if dataset_name and path:
            raise ValueError("Cannot specify both dataset_name and path")
        
        if path:
            # Load from specific path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset file not found: {path}")
            
            if path.endswith('.jsonl'):
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    samples = [json.loads(line) for line in lines]
            elif path.endswith('.json'):
                with open(path, "r", encoding="utf-8") as f:
                    samples = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            # Extract dataset name from path for ID generation
            dataset_name = os.path.basename(os.path.dirname(path))
        else:
            # Load by dataset name
            jsonl_path = os.path.join(self.data_dir, dataset_name, f"{dataset_name.lower()}.jsonl")
            json_path = os.path.join(self.data_dir, dataset_name, f"{dataset_name.lower()}.json")
            
            if os.path.exists(jsonl_path):
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    samples = [json.loads(line) for line in lines]
            elif os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    samples = json.load(f)
            else:
                raise FileNotFoundError(f"Dataset file not found: {jsonl_path} or {json_path}")
        
        if max_samples:
            samples = samples[:max_samples]
        
        # 标准化输出
        for i, s in enumerate(samples):
            s.setdefault("id", f"{dataset_name}_{i}")
            s.setdefault("dataset", dataset_name)
        
        return samples 