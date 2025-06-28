"""
配置管理器模块
~~~~~~~~~~

这个模块提供了一个统一的配置管理接口。
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """配置管理器类（单例）"""
    
    _instance = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        
        # 设置默认配置路径
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent.parent.parent, 'config', 'config.json')
            
        self.config_path = config_path
        self.config = self._load_config()
        self._initialized = True
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            self.logger.info(f"正在加载配置文件: {self.config_path}")
            
            if not os.path.exists(self.config_path):
                self.logger.warning(f"未找到配置文件，使用默认配置")
                return self._get_default_config()
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            self.logger.info("成功加载配置文件")
            return config
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            self.logger.warning("使用默认配置")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "version": "1.0.0",
            "log_level": "INFO",
            "log_file": "logging.yaml",
            "use_gpu": False,
            "solver": {
                "max_iterations": 1000,
                "tolerance": 1.0
            },
            "nlp": {
                "device": "cpu",
                "model_path": "models/nlp"
            }
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置项键名
            default: 默认值
            
        Returns:
            Any: 配置项值
        """
        try:
            if '.' in key:
                # 处理多级键名
                parts = key.split('.')
                value = self.config
                for part in parts:
                    if part not in value:
                        return default
                    value = value[part]
                return value
            else:
                # 处理单级键名
                return self.config.get(key, default)
                
        except Exception as e:
            self.logger.error(f"获取配置项失败: {str(e)}")
            return default
            
    def get_examples_path(self) -> str:
        """获取示例问题文件路径"""
        examples = self.get('examples', {})
        path = examples.get('path', 'examples/problems.json')
        return os.path.join(Path(__file__).parent.parent.parent, path)
        
    def reload(self) -> None:
        """重新加载配置文件"""
        self.config = self._load_config()
        self.logger.info("配置已重新加载")
        
    def get_config(self) -> Dict[str, Any]:
        """获取配置
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return self.config 