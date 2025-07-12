"""
Models Module - Orchestrator
============================

模型模块协调器：负责协调模型相关操作

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .public_api import models_api

logger = logging.getLogger(__name__)


class ModelsOrchestrator:
    """模型模块协调器"""
    
    def __init__(self):
        self.logger = logger
        self.api = models_api
        self._initialized = False
        self._components = {}
        self._operation_history = []
    
    def initialize_orchestrator(self) -> bool:
        """初始化协调器"""
        try:
            if not self.api._initialized:
                self.api.initialize()
            self._initialized = True
            self.logger.info("模型协调器初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"模型协调器初始化失败: {e}")
            return False
        
    def orchestrate(self, operation: str, **kwargs) -> Any:
        """协调指定操作的执行"""
        try:
            if not self._initialized:
                raise ValueError("协调器未初始化，请先调用initialize_orchestrator()")
            
            self.logger.info(f"开始协调模型操作: {operation}")
            
            result = None
            
            if operation == "create_model":
                result = self._orchestrate_model_creation(**kwargs)
            elif operation == "process_equations":
                result = self._orchestrate_equation_processing(**kwargs)
            elif operation == "process_relations":
                result = self._orchestrate_relation_processing(**kwargs)
            elif operation == "health_check":
                result = self._orchestrate_health_check(**kwargs)
            else:
                raise ValueError(f"不支持的操作: {operation}")
            
            # 记录操作历史
            self._operation_history.append({
                "operation": operation,
                "timestamp": "2024-07-13",
                "status": "success",
                "kwargs": kwargs
            })
            
            self.logger.info(f"模型操作协调完成: {operation}")
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
            self.logger.error(f"模型操作协调失败 [{operation}]: {e}")
            raise
    
    def _orchestrate_model_creation(self, model_data: Any, model_type: str = "general", 
                                  config: Optional[Dict] = None) -> Dict[str, Any]:
        """协调模型创建"""
        if not self.api._initialized:
            self.api.initialize()
        return self.api.create_model(model_data, model_type, config)
    
    def _orchestrate_equation_processing(self, equations: Any, 
                                       config: Optional[Dict] = None) -> Dict[str, Any]:
        """协调方程处理"""
        if not self.api._initialized:
            self.api.initialize()
        return self.api.process_equations(equations, config)
    
    def _orchestrate_relation_processing(self, relations: Any,
                                       config: Optional[Dict] = None) -> Dict[str, Any]:
        """协调关系处理"""
        if not self.api._initialized:
            self.api.initialize()
        return self.api.process_relations(relations, config)
    
    def _orchestrate_health_check(self) -> Dict[str, Any]:
        """协调健康检查"""
        if not self.api._initialized:
            self.api.initialize()
        return self.api.health_check()
    
    def register_component(self, name: str, component: Any) -> None:
        """注册组件"""
        self._components[name] = component
        self.logger.info(f"模型组件注册成功: {name}")
    
    def get_component(self, name: str) -> Any:
        """获取组件"""
        if name in self._components:
            return self._components[name]
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"模型组件不存在: {name}")
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """获取操作历史"""
        return self._operation_history.copy()
    
    def clear_operation_history(self) -> None:
        """清空操作历史"""
        self._operation_history.clear()
        self.logger.info("操作历史已清空")


# 全局协调器实例
models_orchestrator = ModelsOrchestrator() 