#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置加载器
~~~~~~~~~

这个模块负责加载和管理系统配置，提供统一的配置访问接口。

"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

class ConfigLoader:
    """配置加载器类"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的config/config.json
        """
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        
        # 设置默认配置路径
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, 'config', 'config.json')
            
        self.config_path = config_path
        self.config = self._load_config()
        self._initialized = True
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        try:
            self.logger.info(f"正在加载配置文件: {self.config_path}")
            
            if not os.path.exists(self.config_path):
                self.logger.warning(f"未指定配置文件路径，使用默认配置")
                return self._get_default_config()
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            self.logger.info(f"成功加载配置文件")
            return config
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            self.logger.warning("使用默认配置")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置
        
        Returns:
            Dict[str, Any]: 默认配置字典
        """
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
            key: 配置项键名，支持点号分隔的多级键名，如"solver.max_iterations"
            default: 默认值，当配置项不存在时返回
            
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
            
    def get_problem_types(self) -> Dict[str, Dict[str, Any]]:
        """获取问题类型配置
        
        Returns:
            Dict[str, Dict[str, Any]]: 问题类型配置字典
        """
        return self.get('problem_types', {})
        
    def get_problem_type_keywords(self, problem_type: str) -> List[str]:
        """获取问题类型的关键词
        
        Args:
            problem_type: 问题类型
            
        Returns:
            List[str]: 关键词列表
        """
        problem_types = self.get_problem_types()
        if problem_type in problem_types:
            return problem_types[problem_type].get('keywords', [])
        return []
        
    def get_problem_type_units(self, problem_type: str) -> List[str]:
        """获取问题类型的单位
        
        Args:
            problem_type: 问题类型
            
        Returns:
            List[str]: 单位列表
        """
        problem_types = self.get_problem_types()
        if problem_type in problem_types:
            return problem_types[problem_type].get('units', [])
        return []
        
    def get_problem_type_subtypes(self, problem_type: str) -> List[str]:
        """获取问题类型的子类型
        
        Args:
            problem_type: 问题类型
            
        Returns:
            List[str]: 子类型列表
        """
        problem_types = self.get_problem_types()
        if problem_type in problem_types:
            return problem_types[problem_type].get('subtypes', [])
        return []
        
    def get_problem_type_exclusion_keywords(self, problem_type: str) -> List[str]:
        """获取问题类型的排除关键词
        
        Args:
            problem_type: 问题类型
            
        Returns:
            List[str]: 排除关键词列表
        """
        problem_types = self.get_problem_types()
        if problem_type in problem_types:
            return problem_types[problem_type].get('exclusion_keywords', [])
        return []
        
    def get_problem_type_priority(self, problem_type: str) -> int:
        """获取问题类型的优先级
        
        Args:
            problem_type: 问题类型
            
        Returns:
            int: 优先级，数字越小优先级越高
        """
        problem_types = self.get_problem_types()
        if problem_type in problem_types:
            return problem_types[problem_type].get('priority', 999)
        return 999
        
    def get_all_problem_types(self) -> List[str]:
        """获取所有问题类型
        
        Returns:
            List[str]: 问题类型列表，按优先级排序
        """
        problem_types = self.get_problem_types()
        return sorted(problem_types.keys(), 
                     key=lambda x: problem_types[x].get('priority', 999))
                     
    def get_patterns_path(self) -> str:
        """获取模式文件路径
        
        Returns:
            str: 模式文件路径
        """
        patterns = self.get('patterns', {})
        path = patterns.get('path', 'src/models/patterns/')
        return os.path.join(PROJECT_ROOT, path)
        
    def get_default_pattern_file(self) -> str:
        """获取默认模式文件
        
        Returns:
            str: 默认模式文件路径
        """
        patterns = self.get('patterns', {})
        default_pattern = patterns.get('default_pattern', 'default_patterns.json')
        return os.path.join(self.get_patterns_path(), default_pattern)
        
    def get_examples_path(self) -> str:
        """获取示例问题文件路径
        
        Returns:
            str: 示例问题文件路径
        """
        examples = self.get('examples', {})
        path = examples.get('path', 'examples/problems.json')
        return os.path.join(PROJECT_ROOT, path)
        
    def reload(self) -> None:
        """重新加载配置文件"""
        self.config = self._load_config()
        self.logger.info("配置已重新加载")
        
    def __str__(self) -> str:
        """返回配置字符串表示"""
        return f"ConfigLoader(config_path={self.config_path})"
        
    def __repr__(self) -> str:
        """返回配置字符串表示"""
        return self.__str__()


# 创建全局配置加载器实例
config = ConfigLoader()

def get_config() -> ConfigLoader:
    """获取配置加载器实例
    
    Returns:
        ConfigLoader: 配置加载器实例
    """
    return config 