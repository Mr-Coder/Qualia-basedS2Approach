"""
Models Module - Utility Functions
=================================

工具函数：提供模型相关的辅助功能

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def validate_model_structure(model: Dict[str, Any]) -> bool:
    """验证模型结构"""
    try:
        required_fields = ["type", "data"]
        return all(field in model for field in required_fields)
    except Exception as e:
        logger.error(f"模型结构验证失败: {e}")
        return False


def format_model_output(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """格式化模型输出"""
    try:
        return {
            "model": model_data,
            "formatted": True,
            "timestamp": "2024-07-13"
        }
    except Exception as e:
        logger.error(f"模型输出格式化失败: {e}")
        return {"error": str(e)}


def merge_model_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并模型结果"""
    try:
        return {
            "merged_results": results,
            "count": len(results),
            "status": "merged"
        }
    except Exception as e:
        logger.error(f"模型结果合并失败: {e}")
        return {"error": str(e)}


# 工具函数集合
model_utils = {
    "validate_model_structure": validate_model_structure,
    "format_model_output": format_model_output, 
    "merge_model_results": merge_model_results
} 