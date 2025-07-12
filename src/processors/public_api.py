"""
Processors Module - Public API
==============================

处理器模块公共API：提供统一的处理器接口

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .private.processor import core_processor
from .private.utils import processing_utils
from .private.validator import validator

logger = logging.getLogger(__name__)


class ProcessorsAPI:
    """处理器模块公共API"""
    
    def __init__(self):
        self.logger = logger
        self.validator = validator
        self.processor = core_processor
        self.utils = processing_utils
        self._initialized = False
        
    def initialize(self) -> bool:
        """初始化处理器模块"""
        try:
            self.logger.info("初始化处理器模块...")
            self._initialized = True
            self.logger.info("处理器模块初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"处理器模块初始化失败: {e}")
            return False
    
    def process_text(self, text: Union[str, Dict], 
                    config: Optional[Dict] = None, 
                    validate_input: bool = True) -> Dict[str, Any]:
        """
        处理文本数据的主要接口
        
        Args:
            text: 输入文本或字典
            config: 处理配置
            validate_input: 是否验证输入
            
        Returns:
            处理结果
        """
        try:
            if not self._initialized:
                raise Exception("处理器模块未初始化，请先调用initialize()")
            
            # 输入验证
            if validate_input:
                validation_result = self.validator.validate_text_input(text)
                if not validation_result["is_valid"]:
                    return {
                        "status": "error",
                        "error_message": "输入验证失败",
                        "validation_errors": validation_result["error_messages"]
                    }
            
            # 清理输入数据
            cleaned_input = self.utils["validate_and_clean_input"](text)
            if not cleaned_input["is_valid"]:
                return {
                    "status": "error",
                    "error_message": "输入清理失败",
                    "issues": cleaned_input["issues"]
                }
            
            # 核心处理
            result = self.processor.process_text(cleaned_input["cleaned_data"], config)
            
            # 格式化结果
            formatted_result = self.utils["format_processing_result"](result)
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"文本处理失败: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "original_input": text
            }
    
    def process_dataset(self, dataset: Union[List, Dict], 
                       config: Optional[Dict] = None,
                       validate_input: bool = True) -> Dict[str, Any]:
        """
        处理数据集的主要接口
        
        Args:
            dataset: 数据集
            config: 处理配置
            validate_input: 是否验证输入
            
        Returns:
            处理结果
        """
        try:
            if not self._initialized:
                raise Exception("处理器模块未初始化，请先调用initialize()")
            
            # 输入验证
            if validate_input:
                validation_result = self.validator.validate_dataset_input(dataset)
                if not validation_result["is_valid"]:
                    return {
                        "status": "error",
                        "error_message": "数据集验证失败",
                        "validation_errors": validation_result["error_messages"]
                    }
            
            # 核心处理
            result = self.processor.process_dataset(dataset, config)
            
            # 格式化结果
            formatted_result = self.utils["format_processing_result"](result)
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"数据集处理失败: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def extract_relations(self, text: Union[str, Dict], 
                         config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        关系提取接口
        
        Args:
            text: 输入文本
            config: 处理配置
            
        Returns:
            关系提取结果
        """
        try:
            if not self._initialized:
                raise Exception("处理器模块未初始化，请先调用initialize()")
            
            # 设置关系处理模式
            relation_config = config or {}
            relation_config["processing_mode"] = "relation"
            
            return self.process_text(text, relation_config)
            
        except Exception as e:
            self.logger.error(f"关系提取失败: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def classify_complexity(self, text: Union[str, Dict], 
                           config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        复杂度分类接口
        
        Args:
            text: 输入文本
            config: 处理配置
            
        Returns:
            分类结果
        """
        try:
            if not self._initialized:
                raise Exception("处理器模块未初始化，请先调用initialize()")
            
            # 设置分类处理模式
            classification_config = config or {}
            classification_config["processing_mode"] = "classification"
            
            return self.process_text(text, classification_config)
            
        except Exception as e:
            self.logger.error(f"复杂度分类失败: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def process_nlp(self, text: Union[str, Dict], 
                   config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        NLP处理接口
        
        Args:
            text: 输入文本
            config: 处理配置
            
        Returns:
            NLP处理结果
        """
        try:
            if not self._initialized:
                raise Exception("处理器模块未初始化，请先调用initialize()")
            
            # 设置NLP处理模式
            nlp_config = config or {}
            nlp_config["processing_mode"] = "nlp"
            
            return self.process_text(text, nlp_config)
            
        except Exception as e:
            self.logger.error(f"NLP处理失败: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def batch_process(self, inputs: List[Union[str, Dict]], 
                     config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        批量处理接口
        
        Args:
            inputs: 输入列表
            config: 处理配置
            
        Returns:
            批量处理结果
        """
        try:
            if not self._initialized:
                raise Exception("处理器模块未初始化，请先调用initialize()")
            
            results = []
            for i, input_item in enumerate(inputs):
                try:
                    result = self.process_text(input_item, config)
                    result["batch_id"] = i
                    results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"批量处理项 {i} 失败: {e}")
                    results.append({
                        "batch_id": i,
                        "status": "error",
                        "error_message": str(e)
                    })
            
            # 合并结果
            merged_result = self.utils["merge_processing_results"](results)
            
            return merged_result
            
        except Exception as e:
            self.logger.error(f"批量处理失败: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def get_module_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        try:
            return {
                "module_name": "processors",
                "initialized": self._initialized,
                "version": "1.0.0",
                "components": {
                    "validator": "active",
                    "processor": "active",
                    "utils": "active"
                },
                "capabilities": [
                    "text_processing",
                    "dataset_processing", 
                    "relation_extraction",
                    "complexity_classification",
                    "nlp_processing",
                    "batch_processing"
                ],
                "processing_statistics": self.processor.get_processing_statistics()
            }
            
        except Exception as e:
            self.logger.error(f"获取模块状态失败: {e}")
            return {
                "module_name": "processors",
                "status": "error",
                "error_message": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试各个组件
            test_text = "测试文本：2 + 3 = ?"
            
            # 测试验证器
            validation_result = self.validator.validate_text_input(test_text)
            validator_health = "healthy" if validation_result["is_valid"] else "unhealthy"
            
            # 测试处理器（轻量级测试）
            try:
                test_config = {"processing_mode": "nlp"}
                _ = self.processor.process_text(test_text, test_config)
                processor_health = "healthy"
            except:
                processor_health = "unhealthy"
            
            # 测试工具函数
            try:
                _ = self.utils["clean_text"](test_text)
                utils_health = "healthy"
            except:
                utils_health = "unhealthy"
            
            overall_health = "healthy" if all([
                validator_health == "healthy",
                processor_health == "healthy", 
                utils_health == "healthy",
                self._initialized
            ]) else "unhealthy"
            
            return {
                "overall_health": overall_health,
                "initialized": self._initialized,
                "components": {
                    "validator": validator_health,
                    "processor": processor_health,
                    "utils": utils_health
                },
                "timestamp": self.utils["format_processing_result"]({"status": "test"})["timestamp"]
            }
            
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
            return {
                "overall_health": "unhealthy",
                "error_message": str(e)
            }
    
    def shutdown(self) -> bool:
        """关闭处理器模块"""
        try:
            self.logger.info("关闭处理器模块...")
            self._initialized = False
            self.logger.info("处理器模块已关闭")
            return True
            
        except Exception as e:
            self.logger.error(f"关闭处理器模块失败: {e}")
            return False


# 全局API实例
processors_api = ProcessorsAPI() 