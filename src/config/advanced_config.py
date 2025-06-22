#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理改进模块
~~~~~~~~~~~~~~~~

提供统一的配置管理和验证功能

Author: [Hao Meng]
Date: [2025-05-29]
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """性能配置"""
    enable_caching: bool = True
    max_cache_size: int = 128
    timeout_seconds: int = 30
    enable_performance_tracking: bool = True
    enable_parallel_processing: bool = False
    max_workers: int = 4


@dataclass
class VisualizationConfig:
    """可视化配置"""
    enabled: bool = True
    chinese_font_support: bool = True
    output_dir: str = "visualization"
    save_graphs: bool = True
    show_interactive: bool = False
    figure_size: tuple = (12, 8)
    dpi: int = 300


@dataclass
class NLPConfig:
    """NLP处理配置"""
    language: str = "zh"
    enable_ner: bool = True
    enable_pos_tagging: bool = True
    enable_dependency_parsing: bool = True
    max_text_length: int = 10000


@dataclass
class SolverConfig:
    """主求解器配置"""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    
    # 求解器特定配置
    default_example_enabled: bool = True
    strict_validation: bool = True
    auto_save_results: bool = True
    results_dir: str = "results"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SolverConfig':
        """从字典创建配置"""
        # 处理嵌套配置
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        performance_config = PerformanceConfig(**config_dict.get('performance', {}))
        visualization_config = VisualizationConfig(**config_dict.get('visualization', {}))
        nlp_config = NLPConfig(**config_dict.get('nlp', {}))
        
        # 提取主配置
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['logging', 'performance', 'visualization', 'nlp']}
        
        return cls(
            logging=logging_config,
            performance=performance_config,
            visualization=visualization_config,
            nlp=nlp_config,
            **main_config
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SolverConfig':
        """从文件加载配置"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            return cls.from_dict(config_dict)
            
        except Exception as e:
            raise ValueError(f"配置文件加载失败: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, ensure_ascii=False)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
                    
        except Exception as e:
            raise ValueError(f"配置文件保存失败: {e}")
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 验证日志级别
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level not in valid_log_levels:
            errors.append(f"无效的日志级别: {self.logging.level}")
        
        # 验证性能配置
        if self.performance.max_cache_size <= 0:
            errors.append("缓存大小必须大于0")
        
        if self.performance.timeout_seconds <= 0:
            errors.append("超时时间必须大于0")
        
        if self.performance.max_workers <= 0:
            errors.append("工作线程数必须大于0")
        
        # 验证可视化配置
        if self.visualization.figure_size[0] <= 0 or self.visualization.figure_size[1] <= 0:
            errors.append("图表尺寸必须大于0")
        
        if self.visualization.dpi <= 0:
            errors.append("DPI必须大于0")
        
        # 验证NLP配置
        valid_languages = ['zh', 'en']
        if self.nlp.language not in valid_languages:
            errors.append(f"不支持的语言: {self.nlp.language}")
        
        if self.nlp.max_text_length <= 0:
            errors.append("最大文本长度必须大于0")
        
        return errors
    
    def setup_logging(self):
        """设置日志"""
        import logging
        logger = logging.getLogger('math_solver')
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 设置日志级别
        logger.setLevel(getattr(logging, self.logging.level))
        
        # 创建格式器
        formatter = logging.Formatter(self.logging.format)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果指定了文件路径）
        if self.logging.file_path:
            from logging.handlers import RotatingFileHandler
            
            # 确保日志目录存在
            log_path = Path(self.logging.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.results_dir,
            self.visualization.output_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器"""
        self.config_path = config_path
        self._config = None
        self._logger = None
    
    @property
    def config(self) -> SolverConfig:
        """获取配置"""
        if self._config is None:
            self.load_config()
        return self._config
    
    @property
    def logger(self) -> logging.Logger:
        """获取日志器"""
        if self._logger is None:
            self._logger = self.config.setup_logging()
        return self._logger
    
    def load_config(self, config_path: Optional[str] = None):
        """加载配置"""
        config_path = config_path or self.config_path
        
        if config_path and Path(config_path).exists():
            self._config = SolverConfig.from_file(config_path)
        else:
            # 使用默认配置
            self._config = SolverConfig()
        
        # 验证配置
        errors = self._config.validate()
        if errors:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
        
        # 创建必要的目录
        self._config.create_directories()
        
        # 重新设置日志器
        self._logger = None
    
    def save_config(self, config_path: Optional[str] = None):
        """保存配置"""
        config_path = config_path or self.config_path
        if not config_path:
            raise ValueError("未指定配置文件路径")
        
        self.config.save_to_file(config_path)
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        current_dict = self.config.to_dict()
        
        # 深度更新字典
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(current_dict, updates)
        self._config = SolverConfig.from_dict(current_dict)
        
        # 重新验证
        errors = self._config.validate()
        if errors:
            raise ValueError(f"更新配置验证失败: {'; '.join(errors)}")
        
        # 重新设置日志器
        self._logger = None


def create_default_config_file(config_path: str = "config/solver_config.yaml"):
    """创建默认配置文件"""
    config = SolverConfig()
    config.save_to_file(config_path)
    print(f"默认配置文件已创建: {config_path}")


if __name__ == "__main__":
    # 演示用法
    create_default_config_file()
    
    # 加载配置
    manager = ConfigManager("config/solver_config.yaml")
    print("配置加载成功")
    print(f"日志级别: {manager.config.logging.level}")
    print(f"启用缓存: {manager.config.performance.enable_caching}")
