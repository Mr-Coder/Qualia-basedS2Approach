"""
Processors Module - Orchestrator
===============================

处理器模块协调器：负责协调各种处理操作

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .private.processor import core_processor
from .private.utils import processing_utils
from .private.validator import validator
from .public_api import processors_api

logger = logging.getLogger(__name__)


class ProcessorsOrchestrator:
    """处理器模块协调器"""
    
    def __init__(self):
        self.logger = logger
        self.api = processors_api
        self.validator = validator
        self.processor = core_processor
        self.utils = processing_utils
        self._operation_history = []
        
    def orchestrate(self, operation: str, **kwargs) -> Any:
        """
        协调指定操作的执行
        
        Args:
            operation: 操作名称
            **kwargs: 操作参数
            
        Returns:
            操作结果
        """
        try:
            self.logger.info(f"开始协调操作: {operation}")
            
            # 记录操作
            operation_record = {
                "operation": operation,
                "timestamp": processing_utils["format_processing_result"]({"status": "start"})["timestamp"],
                "parameters": kwargs
            }
            
            result = None
            
            # 根据操作类型进行协调
            if operation == "process_text":
                result = self._orchestrate_text_processing(**kwargs)
                
            elif operation == "process_dataset":
                result = self._orchestrate_dataset_processing(**kwargs)
                
            elif operation == "extract_relations":
                result = self._orchestrate_relation_extraction(**kwargs)
                
            elif operation == "classify_complexity":
                result = self._orchestrate_complexity_classification(**kwargs)
                
            elif operation == "process_nlp":
                result = self._orchestrate_nlp_processing(**kwargs)
                
            elif operation == "batch_process":
                result = self._orchestrate_batch_processing(**kwargs)
                
            elif operation == "validate_input":
                result = self._orchestrate_validation(**kwargs)
                
            elif operation == "health_check":
                result = self._orchestrate_health_check(**kwargs)
                
            else:
                raise ValueError(f"不支持的操作: {operation}")
            
            # 完成操作记录
            operation_record["result"] = "success"
            operation_record["end_timestamp"] = processing_utils["format_processing_result"]({"status": "end"})["timestamp"]
            self._operation_history.append(operation_record)
            
            self.logger.info(f"操作协调完成: {operation}")
            return result
            
        except Exception as e:
            self.logger.error(f"操作协调失败 [{operation}]: {e}")
            
            # 记录失败
            operation_record["result"] = "failed"
            operation_record["error"] = str(e)
            operation_record["end_timestamp"] = processing_utils["format_processing_result"]({"status": "end"})["timestamp"]
            self._operation_history.append(operation_record)
            
            raise
    
    def _orchestrate_text_processing(self, text: Union[str, Dict], 
                                   config: Optional[Dict] = None, 
                                   validate_input: bool = True) -> Dict[str, Any]:
        """协调文本处理操作"""
        try:
            # 确保API已初始化
            if not self.api._initialized:
                self.api.initialize()
            
            # 执行文本处理
            result = self.api.process_text(text, config, validate_input)
            
            return result
            
        except Exception as e:
            self.logger.error(f"文本处理协调失败: {e}")
            raise
    
    def _orchestrate_dataset_processing(self, dataset: Union[List, Dict], 
                                      config: Optional[Dict] = None,
                                      validate_input: bool = True) -> Dict[str, Any]:
        """协调数据集处理操作"""
        try:
            # 确保API已初始化
            if not self.api._initialized:
                self.api.initialize()
            
            # 执行数据集处理
            result = self.api.process_dataset(dataset, config, validate_input)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据集处理协调失败: {e}")
            raise
    
    def _orchestrate_relation_extraction(self, text: Union[str, Dict], 
                                       config: Optional[Dict] = None) -> Dict[str, Any]:
        """协调关系提取操作"""
        try:
            # 确保API已初始化
            if not self.api._initialized:
                self.api.initialize()
            
            # 执行关系提取
            result = self.api.extract_relations(text, config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"关系提取协调失败: {e}")
            raise
    
    def _orchestrate_complexity_classification(self, text: Union[str, Dict], 
                                             config: Optional[Dict] = None) -> Dict[str, Any]:
        """协调复杂度分类操作"""
        try:
            # 确保API已初始化
            if not self.api._initialized:
                self.api.initialize()
            
            # 执行复杂度分类
            result = self.api.classify_complexity(text, config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"复杂度分类协调失败: {e}")
            raise
    
    def _orchestrate_nlp_processing(self, text: Union[str, Dict], 
                                  config: Optional[Dict] = None) -> Dict[str, Any]:
        """协调NLP处理操作"""
        try:
            # 确保API已初始化
            if not self.api._initialized:
                self.api.initialize()
            
            # 执行NLP处理
            result = self.api.process_nlp(text, config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"NLP处理协调失败: {e}")
            raise
    
    def _orchestrate_batch_processing(self, inputs: List[Union[str, Dict]], 
                                    config: Optional[Dict] = None) -> Dict[str, Any]:
        """协调批量处理操作"""
        try:
            # 确保API已初始化
            if not self.api._initialized:
                self.api.initialize()
            
            # 执行批量处理
            result = self.api.batch_process(inputs, config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"批量处理协调失败: {e}")
            raise
    
    def _orchestrate_validation(self, data: Any, 
                              validation_type: str = "text") -> Dict[str, Any]:
        """协调验证操作"""
        try:
            if validation_type == "text":
                result = self.validator.validate_text_input(data)
            elif validation_type == "dataset":
                result = self.validator.validate_dataset_input(data)
            elif validation_type == "relation":
                result = self.validator.validate_relation_input(data)
            else:
                raise ValueError(f"不支持的验证类型: {validation_type}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"验证协调失败: {e}")
            raise
    
    def _orchestrate_health_check(self) -> Dict[str, Any]:
        """协调健康检查操作"""
        try:
            # 确保API已初始化
            if not self.api._initialized:
                self.api.initialize()
            
            # 执行健康检查
            result = self.api.health_check()
            
            return result
            
        except Exception as e:
            self.logger.error(f"健康检查协调失败: {e}")
            raise
    
    def register_component(self, name: str, component: Any) -> None:
        """
        注册组件
        
        Args:
            name: 组件名称
            component: 组件实例
        """
        try:
            setattr(self, name, component)
            self.logger.info(f"组件注册成功: {name}")
            
        except Exception as e:
            self.logger.error(f"组件注册失败 [{name}]: {e}")
            raise
    
    def get_component(self, name: str) -> Any:
        """
        获取组件
        
        Args:
            name: 组件名称
            
        Returns:
            组件实例
        """
        try:
            if hasattr(self, name):
                return getattr(self, name)
            else:
                raise AttributeError(f"组件不存在: {name}")
                
        except Exception as e:
            self.logger.error(f"获取组件失败 [{name}]: {e}")
            raise
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """获取操作历史"""
        return self._operation_history.copy()
    
    def clear_operation_history(self) -> None:
        """清空操作历史"""
        self._operation_history.clear()
        self.logger.info("操作历史已清空")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        try:
            return {
                "orchestrator_name": "processors",
                "components": {
                    "api": "registered",
                    "validator": "registered",
                    "processor": "registered",
                    "utils": "registered"
                },
                "operation_history_count": len(self._operation_history),
                "supported_operations": [
                    "process_text",
                    "process_dataset",
                    "extract_relations",
                    "classify_complexity",
                    "process_nlp",
                    "batch_process",
                    "validate_input",
                    "health_check"
                ],
                "api_status": self.api.get_module_status()
            }
            
        except Exception as e:
            self.logger.error(f"获取协调器状态失败: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def initialize_orchestrator(self) -> bool:
        """初始化协调器"""
        try:
            self.logger.info("初始化处理器协调器...")
            
            # 初始化API
            if not self.api.initialize():
                raise Exception("API初始化失败")
            
            self.logger.info("处理器协调器初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"协调器初始化失败: {e}")
            return False
    
    def shutdown_orchestrator(self) -> bool:
        """关闭协调器"""
        try:
            self.logger.info("关闭处理器协调器...")
            
            # 关闭API
            self.api.shutdown()
            
            # 清空历史记录
            self.clear_operation_history()
            
            self.logger.info("处理器协调器已关闭")
            return True
            
        except Exception as e:
            self.logger.error(f"协调器关闭失败: {e}")
            return False


# 全局协调器实例
processors_orchestrator = ProcessorsOrchestrator() 