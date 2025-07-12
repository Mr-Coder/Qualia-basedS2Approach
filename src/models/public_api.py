"""
Models Module - Public API
==========================

模型模块公共API：提供统一的模型接口

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .private.processor import model_core_processor
from .private.utils import model_utils
from .private.validator import model_validator

logger = logging.getLogger(__name__)


class ModelsAPI:
    """模型模块公共API"""
    
    def __init__(self):
        self.logger = logger
        self.validator = model_validator
        self.processor = model_core_processor
        self.utils = model_utils
        self._initialized = False
        
    def initialize(self) -> bool:
        """初始化模型模块"""
        try:
            self.logger.info("初始化模型模块...")
            self._initialized = True
            self.logger.info("模型模块初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"模型模块初始化失败: {e}")
            return False
    
    def create_model(self, model_data: Union[str, Dict], 
                    model_type: str = "general", 
                    config: Optional[Dict] = None) -> Dict[str, Any]:
        """创建模型"""
        try:
            if not self._initialized:
                return {
                    "status": "error",
                    "error_message": "模型模块未初始化，请先调用initialize()"
                }
            
            # 验证输入
            validation_result = self.validator.validate_model_input(model_data)
            if not validation_result["is_valid"]:
                return {
                    "status": "error",
                    "error_message": "模型数据验证失败",
                    "validation_errors": validation_result["error_messages"]
                }
            
            # 设置配置
            config = config or {}
            config["model_type"] = model_type
            
            # 处理模型数据
            result = self.processor.process_model_data(model_data, config)
            
            # 格式化输出
            formatted_result = self.utils["format_model_output"](result)
            
            return {
                "status": "success",
                "model": formatted_result,
                "model_type": model_type,
                "timestamp": "2024-07-13"
            }
            
        except Exception as e:
            self.logger.error(f"模型创建失败: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def process_equations(self, equations: Any, config: Optional[Dict] = None) -> Dict[str, Any]:
        """处理方程"""
        try:
            config = config or {}
            config["model_type"] = "equation"
            result = self.create_model(equations, "equation", config)
            if result["status"] == "success":
                result["model"] = {"equation_model": result["model"], "processed_equations": equations}
            return result
        except Exception as e:
            self.logger.error(f"方程处理失败: {e}")
            return {"status": "error", "error_message": str(e)}
    
    def process_relations(self, relations: Any, config: Optional[Dict] = None) -> Dict[str, Any]:
        """处理关系"""
        try:
            config = config or {}
            config["model_type"] = "relation" 
            result = self.create_model(relations, "relation", config)
            if result["status"] == "success":
                result["model"] = {"relation_model": result["model"], "processed_relations": relations}
            return result
        except Exception as e:
            self.logger.error(f"关系处理失败: {e}")
            return {"status": "error", "error_message": str(e)}
    
    def get_module_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        return {
            "module_name": "models",
            "initialized": self._initialized,
            "version": "1.0.0",
            "components": {
                "validator": "active",
                "processor": "active", 
                "utils": "active"
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            overall_health = "healthy" if self._initialized else "unhealthy"
            return {
                "overall_health": overall_health,
                "initialized": self._initialized
            }
        except Exception as e:
            return {"overall_health": "unhealthy", "error_message": str(e)}
    
    def shutdown(self) -> bool:
        """关闭模型模块"""
        try:
            self._initialized = False
            self.logger.info("模型模块已关闭")
            return True
        except Exception as e:
            self.logger.error(f"关闭模型模块失败: {e}")
            return False


# 全局API实例
models_api = ModelsAPI() 