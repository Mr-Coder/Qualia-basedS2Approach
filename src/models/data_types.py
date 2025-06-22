"""定义共享的数据类型"""
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ExtractionResult:
    """提取结果"""
    explicit_relations: List[Dict]
    implicit_relations: List[Dict]
