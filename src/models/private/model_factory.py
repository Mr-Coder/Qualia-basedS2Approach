"""
模型工厂 (Model Factory)

专注于模型的创建、配置和初始化。
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union


class ModelCreationError(Exception):
    """模型创建错误"""
    pass


class ModelConfigurationError(Exception):
    """模型配置错误"""
    pass


# 简化的基础模型类（用于演示）
class BaseModel:
    """基础模型类"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """解决问题的基础方法"""
        return {
            "final_answer": "基础模型无法求解",
            "confidence": 0.0,
            "success": False
        }

class BaselineModel(BaseModel):
    """基线模型基类"""
    pass

class LLMModel(BaseModel):
    """大语言模型基类"""
    pass

class ProposedModel(BaseModel):
    """提出模型基类"""
    pass

# 简化的模型实现（用于演示）
class TemplateBasedModel(BaselineModel):
    """模板基线模型"""
    pass

class EquationBasedModel(BaselineModel):
    """方程基线模型"""
    pass

class RuleBasedModel(BaselineModel):
    """规则基线模型"""
    pass

class SimplePatternModel(BaselineModel):
    """简单模式模型"""
    pass

class OpenAIGPTModel(LLMModel):
    """OpenAI GPT模型"""
    pass

class ClaudeModel(LLMModel):
    """Claude模型"""
    pass

class QwenModel(LLMModel):
    """Qwen模型"""
    pass

class InternLMModel(LLMModel):
    """InternLM模型"""
    pass

class DeepSeekMathModel(LLMModel):
    """DeepSeek数学模型"""
    pass

class COTDIRModel(ProposedModel):
    """COT-DIR模型"""
    pass


class ModelFactory:
    """模型工厂 - 负责创建和配置模型实例"""
    
    def __init__(self, default_config_path: Optional[str] = None):
        """初始化模型工厂"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 注册的模型类
        self.model_classes = {}
        
        # 默认配置
        self.default_configs = {}
        
        # 加载默认配置
        if default_config_path:
            self._load_default_configs(default_config_path)
        
        # 注册默认模型
        self._register_default_models()
        
        # 创建统计
        self.creation_stats = {
            "total_created": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "models_by_type": {
                "baseline": 0,
                "llm": 0,
                "proposed": 0
            }
        }
        
        self.logger.info("模型工厂初始化完成")
    
    def _register_default_models(self):
        """注册默认模型类"""
        
        # 基线模型
        self.register_model_class("template_baseline", TemplateBasedModel, "baseline")
        self.register_model_class("equation_baseline", EquationBasedModel, "baseline")  
        self.register_model_class("rule_baseline", RuleBasedModel, "baseline")
        self.register_model_class("simple_pattern_solver", SimplePatternModel, "baseline")
        
        # LLM模型
        self.register_model_class("gpt4o", OpenAIGPTModel, "llm")
        self.register_model_class("claude", ClaudeModel, "llm")
        self.register_model_class("qwen", QwenModel, "llm")
        self.register_model_class("internlm", InternLMModel, "llm")
        self.register_model_class("deepseek", DeepSeekMathModel, "llm")
        
        # 提出的模型
        self.register_model_class("cotdir", COTDIRModel, "proposed")
        
        self.logger.info(f"注册了{len(self.model_classes)}个默认模型类")
    
    def register_model_class(self, name: str, model_class: Type[BaseModel], model_type: str):
        """
        注册模型类
        
        Args:
            name: 模型名称
            model_class: 模型类
            model_type: 模型类型 (baseline, llm, proposed)
        """
        if not issubclass(model_class, BaseModel):
            raise ModelCreationError(f"模型类{model_class}必须继承自BaseModel")
        
        self.model_classes[name] = {
            "class": model_class,
            "type": model_type,
            "name": name
        }
        
        self.logger.debug(f"注册模型类: {name} ({model_type})")
    
    def create_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        创建模型实例
        
        Args:
            model_name: 模型名称
            config: 模型配置
            
        Returns:
            创建的模型实例
        """
        try:
            self.logger.info(f"创建模型: {model_name}")
            
            # 获取模型类信息
            if model_name not in self.model_classes:
                available_models = list(self.model_classes.keys())
                raise ModelCreationError(f"未知模型: {model_name}, 可用模型: {available_models}")
            
            model_info = self.model_classes[model_name]
            model_class = model_info["class"]
            model_type = model_info["type"]
            
            # 准备配置
            final_config = self._prepare_model_config(model_name, config)
            
            # 验证配置
            self._validate_model_config(model_name, final_config)
            
            # 创建模型实例
            model_instance = self._instantiate_model(model_class, final_config)
            
            # 更新统计
            self._update_creation_stats(model_type, True)
            
            self.logger.info(f"模型{model_name}创建成功")
            return model_instance
            
        except Exception as e:
            self._update_creation_stats("unknown", False)
            self.logger.error(f"模型{model_name}创建失败: {str(e)}")
            raise ModelCreationError(f"创建模型{model_name}失败: {str(e)}")
    
    def _prepare_model_config(self, model_name: str, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """准备模型配置"""
        
        # 从默认配置开始
        default_config = self.default_configs.get(model_name, {})
        
        # 合并用户配置
        final_config = default_config.copy()
        if user_config:
            final_config.update(user_config)
        
        # 添加模型名称
        final_config["model_name"] = model_name
        
        return final_config
    
    def _validate_model_config(self, model_name: str, config: Dict[str, Any]):
        """验证模型配置"""
        
        if not isinstance(config, dict):
            raise ModelConfigurationError(f"模型{model_name}的配置必须是字典")
        
        # 获取模型类型
        model_info = self.model_classes.get(model_name, {})
        model_type = model_info.get("type", "unknown")
        
        # 类型特定的验证
        if model_type == "llm":
            self._validate_llm_config(model_name, config)
        elif model_type == "proposed":
            self._validate_proposed_config(model_name, config)
        elif model_type == "baseline":
            self._validate_baseline_config(model_name, config)
    
    def _validate_llm_config(self, model_name: str, config: Dict[str, Any]):
        """验证LLM配置"""
        
        # 检查API相关配置
        if model_name in ["gpt4o", "claude"]:
            if "api_key" not in config and "base_url" not in config:
                self.logger.warning(f"LLM模型{model_name}缺少API配置，可能需要环境变量")
        
        # 检查模型参数
        if "temperature" in config:
            temp = config["temperature"]
            if not (0.0 <= temp <= 2.0):
                raise ModelConfigurationError(f"temperature必须在0.0-2.0之间，当前值: {temp}")
        
        if "max_tokens" in config:
            max_tokens = config["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ModelConfigurationError(f"max_tokens必须是正整数，当前值: {max_tokens}")
    
    def _validate_proposed_config(self, model_name: str, config: Dict[str, Any]):
        """验证提出模型配置"""
        
        if model_name == "cotdir":
            # 验证COT-DIR特定配置
            if "enable_ird" in config and not isinstance(config["enable_ird"], bool):
                raise ModelConfigurationError("enable_ird必须是布尔值")
            
            if "enable_mlr" in config and not isinstance(config["enable_mlr"], bool):
                raise ModelConfigurationError("enable_mlr必须是布尔值")
            
            if "enable_cv" in config and not isinstance(config["enable_cv"], bool):
                raise ModelConfigurationError("enable_cv必须是布尔值")
            
            if "confidence_threshold" in config:
                threshold = config["confidence_threshold"]
                if not (0.0 <= threshold <= 1.0):
                    raise ModelConfigurationError(f"confidence_threshold必须在0.0-1.0之间，当前值: {threshold}")
    
    def _validate_baseline_config(self, model_name: str, config: Dict[str, Any]):
        """验证基线模型配置"""
        
        # 基线模型通常不需要特殊验证
        pass
    
    def _instantiate_model(self, model_class: Type[BaseModel], config: Dict[str, Any]) -> BaseModel:
        """实例化模型"""
        
        try:
            # 尝试使用config参数创建
            return model_class(config)
        except TypeError:
            try:
                # 尝试无参数创建
                instance = model_class()
                # 如果有configure方法，调用它
                if hasattr(instance, 'configure') and callable(getattr(instance, 'configure')):
                    instance.configure(config)
                return instance
            except Exception as e:
                raise ModelCreationError(f"无法实例化模型: {str(e)}")
    
    def batch_create_models(self, model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseModel]:
        """
        批量创建模型
        
        Args:
            model_configs: 模型名称到配置的映射
            
        Returns:
            模型名称到模型实例的映射
        """
        models = {}
        
        for model_name, config in model_configs.items():
            try:
                models[model_name] = self.create_model(model_name, config)
            except Exception as e:
                self.logger.error(f"批量创建中模型{model_name}失败: {str(e)}")
                # 继续创建其他模型
        
        self.logger.info(f"批量创建完成: {len(models)}/{len(model_configs)}个模型成功")
        return models
    
    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        """获取可用模型列表"""
        return {
            name: {
                "type": info["type"],
                "class_name": info["class"].__name__
            }
            for name, info in self.model_classes.items()
        }
    
    def get_models_by_type(self, model_type: str) -> Dict[str, Type[BaseModel]]:
        """按类型获取模型"""
        return {
            name: info["class"]
            for name, info in self.model_classes.items()
            if info["type"] == model_type
        }
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if model_name not in self.model_classes:
            return None
        
        model_info = self.model_classes[model_name]
        return {
            "name": model_name,
            "type": model_info["type"],
            "class_name": model_info["class"].__name__,
            "default_config": self.default_configs.get(model_name, {}),
            "description": getattr(model_info["class"], "__doc__", "无描述")
        }
    
    def _load_default_configs(self, config_path: str):
        """加载默认配置"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    all_configs = json.load(f)
                    
                    # 提取模型配置
                    if "models" in all_configs:
                        self.default_configs = all_configs["models"]
                    else:
                        self.default_configs = all_configs
                        
                self.logger.info(f"加载了{len(self.default_configs)}个默认模型配置")
            else:
                self.logger.warning(f"配置文件不存在: {config_path}")
                
        except Exception as e:
            self.logger.error(f"加载默认配置失败: {str(e)}")
    
    def _update_creation_stats(self, model_type: str, success: bool):
        """更新创建统计"""
        self.creation_stats["total_created"] += 1
        
        if success:
            self.creation_stats["successful_creations"] += 1
            if model_type in self.creation_stats["models_by_type"]:
                self.creation_stats["models_by_type"][model_type] += 1
        else:
            self.creation_stats["failed_creations"] += 1
    
    def get_creation_stats(self) -> Dict[str, Any]:
        """获取创建统计"""
        stats = self.creation_stats.copy()
        
        # 计算成功率
        if stats["total_created"] > 0:
            stats["success_rate"] = stats["successful_creations"] / stats["total_created"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.creation_stats = {
            "total_created": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "models_by_type": {
                "baseline": 0,
                "llm": 0,
                "proposed": 0
            }
        }
        self.logger.info("模型工厂统计信息已重置")