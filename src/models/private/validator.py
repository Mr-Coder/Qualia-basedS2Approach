"""
Models Module - Data Validator
==============================

数据验证器：负责验证模型相关数据的有效性

Author: AI Assistant  
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ModelValidator:
    """模型数据验证器"""
    
    def __init__(self):
        self.logger = logger
        
    def validate_model_input(self, model_data: Any) -> Dict[str, Any]:
        """验证模型输入数据"""
        try:
            result = {"is_valid": True, "error_messages": []}
            
            if model_data is None:
                result["is_valid"] = False
                result["error_messages"].append("模型数据不能为空")
                return result
                
            # 验证数据结构
            if isinstance(model_data, dict):
                required_fields = ["type", "data"]
                for field in required_fields:
                    if field not in model_data:
                        result["is_valid"] = False
                        result["error_messages"].append(f"缺少必要字段: {field}")
                        
            return result
            
        except Exception as e:
            self.logger.error(f"模型验证失败: {e}")
            return {"is_valid": False, "error_messages": [f"验证过程出错: {str(e)}"]}
    
    def validate_equation_input(self, equation: Any) -> Dict[str, Any]:
        """验证方程输入"""
        try:
            result = {"is_valid": True, "error_messages": []}
            
            if equation is None:
                result["is_valid"] = False
                result["error_messages"].append("方程不能为空")
                
            return result
            
        except Exception as e:
            self.logger.error(f"方程验证失败: {e}")
            return {"is_valid": False, "error_messages": [f"验证过程出错: {str(e)}"]}


# 全局验证器实例
model_validator = ModelValidator() 