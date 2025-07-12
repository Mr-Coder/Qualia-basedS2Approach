"""
Data Module - Public API
========================

数据模块公共API：提供统一的数据接口

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List, Optional

# 导入现有的数据处理类
from .dataset_characteristics import get_all_datasets, get_dataset_info
from .performance_analysis import get_method_performance

logger = logging.getLogger(__name__)


class DataAPI:
    """数据模块公共API"""
    
    def __init__(self):
        self.logger = logger
        self._initialized = False
        
    def initialize(self) -> bool:
        """初始化数据模块"""
        try:
            self._initialized = True
            self.logger.info("数据模块初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"数据模块初始化失败: {e}")
            return False
    
    def get_dataset_information(self, dataset_name: Any = None) -> Dict[str, Any]:
        """获取数据集信息"""
        try:
            if not self._initialized:
                return {
                    "status": "error",
                    "error_message": "数据模块未初始化，请先调用initialize()"
                }
            
            # 验证输入类型
            if dataset_name is not None and not isinstance(dataset_name, str):
                return {
                    "status": "error",
                    "error_message": "数据集名称必须是字符串类型"
                }
            
            if dataset_name:
                dataset_info = get_dataset_info(dataset_name)
                if dataset_info:
                    return {
                        "status": "success",
                        "dataset_info": dataset_info,
                        "dataset_name": dataset_name,
                        "timestamp": "2024-07-13"
                    }
                else:
                    return {
                        "status": "error",
                        "error_message": f"数据集 '{dataset_name}' 不存在"
                    }
            else:
                all_datasets = get_all_datasets()
                return {
                    "status": "success",
                    "all_datasets": all_datasets,
                    "total_datasets": len(all_datasets),
                    "timestamp": "2024-07-13"
                }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    def get_performance_data(self, method_name: Any = None) -> Dict[str, Any]:
        """获取性能数据"""
        try:
            if not self._initialized:
                return {
                    "status": "error",
                    "error_message": "数据模块未初始化，请先调用initialize()"
                }
            
            # 验证输入类型
            if method_name is not None and not isinstance(method_name, str):
                return {
                    "status": "error",
                    "error_message": "方法名称必须是字符串类型"
                }
            
            if method_name:
                performance_data = get_method_performance(method_name)
                if performance_data:
                    return {
                        "status": "success",
                        "performance_data": performance_data,
                        "method_name": method_name,
                        "timestamp": "2024-07-13"
                    }
                else:
                    return {
                        "status": "success",
                        "message": f"方法 '{method_name}' 的性能数据不存在",
                        "performance_data": None
                    }
            else:
                return {
                    "status": "success", 
                    "message": "请指定方法名称获取具体性能数据",
                    "available_methods": ["method_1", "method_2", "method_3"]
                }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    def get_module_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        return {
            "module_name": "data",
            "initialized": self._initialized,
            "version": "1.0.0"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            overall_health = "healthy" if self._initialized else "unhealthy"
            return {
                "overall_health": overall_health,
                "initialized": self._initialized,
                "module_name": "data",
                "timestamp": "2024-07-13"
            }
        except Exception as e:
            return {"overall_health": "unhealthy", "error_message": str(e)}
    
    def shutdown(self) -> bool:
        """关闭数据模块"""
        try:
            self._initialized = False
            self.logger.info("数据模块已关闭")
            return True
        except Exception as e:
            self.logger.error(f"关闭数据模块失败: {e}")
            return False


# 全局API实例
data_api = DataAPI() 