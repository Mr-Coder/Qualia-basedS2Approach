"""
Processors Module - Utility Functions
====================================

工具函数：提供各种辅助功能和工具方法

Author: AI Assistant
Date: 2024-07-13
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    清理文本数据
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    if not isinstance(text, str):
        return str(text)
    
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    # 移除特殊字符但保留数学符号
    text = re.sub(r'[^\w\s\+\-\*/\=\(\)\[\]\{\}\.,:;!?]', '', text)
    
    return text


def extract_numbers(text: str) -> List[float]:
    """
    从文本中提取数字
    
    Args:
        text: 输入文本
        
    Returns:
        提取的数字列表
    """
    try:
        # 匹配整数和小数
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                if '.' in match:
                    numbers.append(float(match))
                else:
                    numbers.append(float(match))
            except ValueError:
                continue
                
        return numbers
        
    except Exception as e:
        logger.error(f"数字提取失败: {e}")
        return []


def extract_mathematical_expressions(text: str) -> List[str]:
    """
    从文本中提取数学表达式
    
    Args:
        text: 输入文本
        
    Returns:
        数学表达式列表
    """
    try:
        # 匹配数学表达式的模式
        patterns = [
            r'\d+\s*[\+\-\*/]\s*\d+',  # 基本运算
            r'\d+\s*=\s*\d+',          # 等式
            r'\w+\s*=\s*\d+',          # 变量赋值
            r'\d+\s*[\+\-\*/]\s*\w+',  # 含变量的运算
        ]
        
        expressions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            expressions.extend(matches)
            
        return list(set(expressions))  # 去重
        
    except Exception as e:
        logger.error(f"数学表达式提取失败: {e}")
        return []


def normalize_text(text: str) -> str:
    """
    标准化文本
    
    Args:
        text: 输入文本
        
    Returns:
        标准化后的文本
    """
    try:
        # 转换为小写
        text = text.lower()
        
        # 标准化数字表达式
        text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 + \2', text)
        text = re.sub(r'(\d+)\s*\-\s*(\d+)', r'\1 - \2', text)
        text = re.sub(r'(\d+)\s*\*\s*(\d+)', r'\1 * \2', text)
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1 / \2', text)
        text = re.sub(r'(\d+)\s*=\s*(\d+)', r'\1 = \2', text)
        
        # 标准化问号
        text = re.sub(r'\s*\?\s*', ' ? ', text)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"文本标准化失败: {e}")
        return text


def format_processing_result(result: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
    """
    格式化处理结果
    
    Args:
        result: 处理结果
        include_metadata: 是否包含元数据
        
    Returns:
        格式化后的结果
    """
    try:
        formatted_result = {
            "status": result.get("status", "unknown"),
            "data": result.get("processed_data", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        if include_metadata:
            formatted_result["metadata"] = {
                "processing_time": result.get("processing_time", 0),
                "data_size": len(str(result.get("processed_data", {}))),
                "original_input": result.get("original_text", ""),
                "version": "1.0.0"
            }
            
        return formatted_result
        
    except Exception as e:
        logger.error(f"结果格式化失败: {e}")
        return {"status": "error", "error_message": str(e)}


def validate_and_clean_input(input_data: Any) -> Dict[str, Any]:
    """
    验证和清理输入数据
    
    Args:
        input_data: 输入数据
        
    Returns:
        清理后的数据和验证结果
    """
    try:
        result = {
            "is_valid": True,
            "cleaned_data": None,
            "issues": []
        }
        
        if input_data is None:
            result["is_valid"] = False
            result["issues"].append("输入数据为空")
            return result
            
        if isinstance(input_data, str):
            result["cleaned_data"] = clean_text(input_data)
            
        elif isinstance(input_data, dict):
            result["cleaned_data"] = {}
            for key, value in input_data.items():
                if isinstance(value, str):
                    result["cleaned_data"][key] = clean_text(value)
                else:
                    result["cleaned_data"][key] = value
                    
        elif isinstance(input_data, list):
            result["cleaned_data"] = []
            for item in input_data:
                if isinstance(item, str):
                    result["cleaned_data"].append(clean_text(item))
                else:
                    result["cleaned_data"].append(item)
                    
        else:
            result["cleaned_data"] = input_data
            result["issues"].append(f"未知数据类型: {type(input_data)}")
            
        return result
        
    except Exception as e:
        logger.error(f"输入验证清理失败: {e}")
        return {
            "is_valid": False,
            "cleaned_data": None,
            "issues": [f"处理失败: {str(e)}"]
        }


def merge_processing_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个处理结果
    
    Args:
        results: 结果列表
        
    Returns:
        合并后的结果
    """
    try:
        merged_result = {
            "status": "success",
            "combined_data": {},
            "individual_results": results,
            "statistics": {}
        }
        
        # 统计成功和失败的结果
        successful_results = [r for r in results if r.get("status") == "success"]
        failed_results = [r for r in results if r.get("status") != "success"]
        
        merged_result["statistics"] = {
            "total_results": len(results),
            "successful_results": len(successful_results),
            "failed_results": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0
        }
        
        # 合并成功结果的数据
        for result in successful_results:
            processed_data = result.get("processed_data", {})
            for key, value in processed_data.items():
                if key not in merged_result["combined_data"]:
                    merged_result["combined_data"][key] = []
                merged_result["combined_data"][key].append(value)
                
        # 如果大部分结果失败，设置总体状态为失败
        if len(failed_results) > len(successful_results):
            merged_result["status"] = "partial_failure"
            
        return merged_result
        
    except Exception as e:
        logger.error(f"结果合并失败: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "individual_results": results
        }


def save_processing_results(results: Dict[str, Any], filename: str) -> bool:
    """
    保存处理结果到文件
    
    Args:
        results: 处理结果
        filename: 文件名
        
    Returns:
        是否保存成功
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理结果已保存到: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"保存处理结果失败: {e}")
        return False


def load_processing_results(filename: str) -> Optional[Dict[str, Any]]:
    """
    从文件加载处理结果
    
    Args:
        filename: 文件名
        
    Returns:
        处理结果或None
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"处理结果已从文件加载: {filename}")
        return results
        
    except Exception as e:
        logger.error(f"加载处理结果失败: {e}")
        return None


def get_processing_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取处理结果摘要
    
    Args:
        results: 处理结果
        
    Returns:
        结果摘要
    """
    try:
        summary = {
            "status": results.get("status", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "data_summary": {}
        }
        
        # 分析处理的数据
        processed_data = results.get("processed_data", {})
        for key, value in processed_data.items():
            if isinstance(value, list):
                summary["data_summary"][key] = {
                    "type": "list",
                    "count": len(value),
                    "sample": value[:3] if value else []
                }
            elif isinstance(value, dict):
                summary["data_summary"][key] = {
                    "type": "dict",
                    "keys": list(value.keys()),
                    "size": len(value)
                }
            else:
                summary["data_summary"][key] = {
                    "type": type(value).__name__,
                    "value": str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                }
                
        return summary
        
    except Exception as e:
        logger.error(f"生成处理摘要失败: {e}")
        return {"status": "error", "error_message": str(e)}


# 工具函数集合
processing_utils = {
    "clean_text": clean_text,
    "extract_numbers": extract_numbers,
    "extract_mathematical_expressions": extract_mathematical_expressions,
    "normalize_text": normalize_text,
    "format_processing_result": format_processing_result,
    "validate_and_clean_input": validate_and_clean_input,
    "merge_processing_results": merge_processing_results,
    "save_processing_results": save_processing_results,
    "load_processing_results": load_processing_results,
    "get_processing_summary": get_processing_summary
} 