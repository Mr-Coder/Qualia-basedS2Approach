"""
Data Module - Orchestrator
==========================

数据模块协调器

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List

from .public_api import data_api

logger = logging.getLogger(__name__)


class DataOrchestrator:
    """数据模块协调器"""
    
    def __init__(self):
        self.logger = logger
        self.api = data_api
        self._initialized = False
        self._components = {}
        self._operation_history = []
    
    def initialize_orchestrator(self) -> bool:
        """初始化协调器"""
        try:
            if not self.api._initialized:
                self.api.initialize()
            self._initialized = True
            self.logger.info("数据协调器初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"数据协调器初始化失败: {e}")
            return False
        
    def orchestrate(self, operation: str, **kwargs) -> Any:
        """协调指定操作的执行"""
        try:
            if not self._initialized:
                raise ValueError("协调器未初始化，请先调用initialize_orchestrator()")
            
            result = None
            
            if operation == "get_dataset_info":
                result = self.api.get_dataset_information(kwargs.get("dataset_name"))
            elif operation == "get_performance_data":
                result = self.api.get_performance_data(kwargs.get("method_name"))
            elif operation == "health_check":
                result = self.api.health_check()
            else:
                raise ValueError(f"不支持的操作: {operation}")
            
            # 记录操作历史
            self._operation_history.append({
                "operation": operation,
                "timestamp": "2024-07-13",
                "status": "success",
                "kwargs": kwargs
            })
            
            return result
            
        except Exception as e:
            # 记录失败操作
            self._operation_history.append({
                "operation": operation,
                "timestamp": "2024-07-13",
                "status": "error",
                "error_message": str(e),
                "kwargs": kwargs
            })
            self.logger.error(f"数据操作协调失败: {e}")
            raise
    
    def register_component(self, name: str, component: Any) -> None:
        """注册组件"""
        self._components[name] = component
        self.logger.info(f"数据组件注册成功: {name}")
    
    def get_component(self, name: str) -> Any:
        """获取组件"""
        if name in self._components:
            return self._components[name]
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"数据组件不存在: {name}")
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """获取操作历史"""
        return self._operation_history.copy()
    
    def clear_operation_history(self) -> None:
        """清空操作历史"""
        self._operation_history.clear()
        self.logger.info("操作历史已清空")


# 全局协调器实例
data_orchestrator = DataOrchestrator() 