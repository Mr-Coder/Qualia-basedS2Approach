"""
Processors Module - Data Validator
=================================

数据验证器：负责验证输入数据的有效性和完整性

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ProcessorValidator:
    """处理器数据验证器"""
    
    def __init__(self):
        self.logger = logger
        
    def validate_text_input(self, text: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        验证文本输入数据
        
        Args:
            text: 输入的文本数据
            
        Returns:
            验证结果，包含is_valid和error_messages
        """
        try:
            result = {"is_valid": True, "error_messages": []}
            
            if text is None:
                result["is_valid"] = False
                result["error_messages"].append("输入文本不能为空")
                return result
                
            if isinstance(text, str):
                if len(text.strip()) == 0:
                    result["is_valid"] = False
                    result["error_messages"].append("输入文本不能为空字符串")
                    
            elif isinstance(text, dict):
                if "problem" not in text and "text" not in text:
                    result["is_valid"] = False
                    result["error_messages"].append("字典输入必须包含'problem'或'text'字段")
                    
            elif isinstance(text, list):
                if len(text) == 0:
                    result["is_valid"] = False
                    result["error_messages"].append("列表输入不能为空")
                    
            else:
                result["is_valid"] = False
                result["error_messages"].append(f"不支持的输入类型: {type(text)}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"文本验证失败: {e}")
            return {"is_valid": False, "error_messages": [f"验证过程出错: {str(e)}"]}
    
    def validate_dataset_input(self, dataset: Any) -> Dict[str, Any]:
        """
        验证数据集输入
        
        Args:
            dataset: 数据集数据
            
        Returns:
            验证结果
        """
        try:
            result = {"is_valid": True, "error_messages": []}
            
            if dataset is None:
                result["is_valid"] = False
                result["error_messages"].append("数据集不能为空")
                return result
                
            if isinstance(dataset, list):
                if len(dataset) == 0:
                    result["is_valid"] = False
                    result["error_messages"].append("数据集不能为空列表")
                    
                # 验证每个样本
                for i, sample in enumerate(dataset):
                    if not isinstance(sample, dict):
                        result["is_valid"] = False
                        result["error_messages"].append(f"样本 {i} 必须是字典格式")
                        
            elif isinstance(dataset, dict):
                if "samples" not in dataset and "data" not in dataset:
                    result["is_valid"] = False
                    result["error_messages"].append("数据集字典必须包含'samples'或'data'字段")
                    
            else:
                result["is_valid"] = False
                result["error_messages"].append(f"不支持的数据集类型: {type(dataset)}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"数据集验证失败: {e}")
            return {"is_valid": False, "error_messages": [f"验证过程出错: {str(e)}"]}
    
    def validate_relation_input(self, relations: Any) -> Dict[str, Any]:
        """
        验证关系数据输入
        
        Args:
            relations: 关系数据
            
        Returns:
            验证结果
        """
        try:
            result = {"is_valid": True, "error_messages": []}
            
            if relations is None:
                result["is_valid"] = False
                result["error_messages"].append("关系数据不能为空")
                return result
                
            if isinstance(relations, list):
                for i, relation in enumerate(relations):
                    if not isinstance(relation, dict):
                        result["is_valid"] = False
                        result["error_messages"].append(f"关系 {i} 必须是字典格式")
                        continue
                        
                    # 验证关系必要字段
                    required_fields = ["source", "target", "type"]
                    for field in required_fields:
                        if field not in relation:
                            result["is_valid"] = False
                            result["error_messages"].append(f"关系 {i} 缺少必要字段: {field}")
                            
            elif isinstance(relations, dict):
                if "relations" not in relations:
                    result["is_valid"] = False
                    result["error_messages"].append("关系字典必须包含'relations'字段")
                    
            else:
                result["is_valid"] = False
                result["error_messages"].append(f"不支持的关系数据类型: {type(relations)}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"关系验证失败: {e}")
            return {"is_valid": False, "error_messages": [f"验证过程出错: {str(e)}"]}
    
    def validate_processing_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证处理配置
        
        Args:
            config: 处理配置
            
        Returns:
            验证结果
        """
        try:
            result = {"is_valid": True, "error_messages": []}
            
            if not isinstance(config, dict):
                result["is_valid"] = False
                result["error_messages"].append("配置必须是字典格式")
                return result
                
            # 验证必要的配置项
            required_configs = ["processing_mode", "output_format"]
            for config_item in required_configs:
                if config_item not in config:
                    result["is_valid"] = False
                    result["error_messages"].append(f"缺少必要配置项: {config_item}")
                    
            # 验证处理模式
            if "processing_mode" in config:
                valid_modes = ["nlp", "relation", "classification", "annotation"]
                if config["processing_mode"] not in valid_modes:
                    result["is_valid"] = False
                    result["error_messages"].append(f"无效的处理模式: {config['processing_mode']}")
                    
            return result
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return {"is_valid": False, "error_messages": [f"验证过程出错: {str(e)}"]}


# 全局验证器实例
validator = ProcessorValidator() 