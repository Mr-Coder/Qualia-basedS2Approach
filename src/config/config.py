#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理模块
~~~~~~~~~~

提供配置文件的加载、验证和管理功能。
支持 YAML 和 JSON 格式的配置文件。

Usage:
    config = load_config('config.yaml')
    # 或者使用默认配置
    config = load_config()
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# 默认配置
DEFAULT_CONFIG = {
    # 系统配置
    'version': '1.0.0',
    'log_level': 'INFO',
    'log_file': 'logging.log',
    
    # 性能配置
    'use_gpu': False,  # 关闭 GPU
    'use_mps': True,   # 启用 MPS
    'num_threads': 8,  # M3 芯片建议使用更多线程
    'batch_size': 32,
    
    # 模型配置
    'nlp': {
        'model_path': 'models/nlp',
        'language': 'zh',
        'max_length': 512,
        'device_settings': {
            'use_mps': True,           # 启用 MPS
            'use_gpu': False,          # 禁用 GPU
            'fallback_to_cpu': True,   # 如果 MPS 不可用则回退到 CPU
            'device_priority': ['mps', 'cpu']  # 移除 cuda，因为 Mac 不支持
        }
    },
    
    # 输出配置
    'output': {
        'save_steps': True,
        'save_path': 'results',
        'format': 'json'
    },
    
    # solver 配置（之前缺少的必需字段）
    'solver': {
        'max_iterations': 1000,
        'tolerance': 1.0,  # 更新了容差值
        'device': 'mps'  # 指定求解器使用 MPS
    }
}

class ConfigError(Exception):
    """配置相关错误"""
    pass

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        Dict: 配置字典
        
    Raises:
        ConfigError: 当配置文件无效或无法加载时
    """
    logger = logging.getLogger(__name__)
    
    # 默认配置
    default_config = {
        'version': '1.0.0',
        'log_level': 'INFO',
        'log_file': 'logging.log',
        'use_gpu': False,
        'solver': {
            'max_iterations': 1000,
            'tolerance': 1.0
        },
        'nlp': {
            'device': 'mps',
            'model_path': 'models/nlp'
        }
    }
    
    if config_path is None:
        logger.warning("未指定配置文件路径，使用默认配置")
        return default_config
        
    try:
        # 尝试加载配置文件
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 更新默认配置
        default_config.update(config)
        logger.info(f"成功加载配置文件: {config_path}")
        return default_config
        
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return default_config
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误: {str(e)}")
        raise ConfigError(f"配置文件格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise ConfigError(f"加载配置文件失败: {str(e)}")

def validate_config(config: Dict) -> Dict:
    """验证配置有效性
    
    Args:
        config: 待验证的配置字典
        
    Returns:
        Dict: 验证后的配置
        
    Raises:
        ConfigError: 当配置无效时
    """
    required_fields = {
        'version',
        'log_level',
        'solver'
    }
    
    # 检查必需字段
    missing_fields = required_fields - set(config.keys())
    if missing_fields:
        raise ConfigError(f"配置缺少必需字段: {missing_fields}")
        
    # 验证字段类型和值
    if 'log_level' in config:
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if config['log_level'] not in valid_levels:
            raise ConfigError(f"无效的日志级别: {config['log_level']}")
            
    if 'solver' in config:
        solver_config = config['solver']
        if 'max_iterations' in solver_config:
            if not isinstance(solver_config['max_iterations'], int) or solver_config['max_iterations'] <= 0:
                raise ConfigError("max_iterations 必须是正整数")
                
        if 'tolerance' in solver_config:
            if not isinstance(solver_config['tolerance'], (int, float)) or solver_config['tolerance'] <= 0:
                raise ConfigError("tolerance 必须是正数")
                
    return config

def merge_configs(default_config: Dict, user_config: Dict) -> Dict:
    """递归合并配置字典
    
    用户配置会覆盖默认配置中的相应值
    
    Args:
        default_config: 默认配置字典
        user_config: 用户配置字典
        
    Returns:
        Dict: 合并后的配置
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        # 如果值是字典，递归合并
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def get_config_path() -> Path:
    """获取默认配置文件路径"""
    # 首先检查环境变量
    config_path = os.environ.get('MATH_SOLVER_CONFIG')
    if config_path:
        return Path(config_path)
        
    # 然后检查常见位置
    common_locations = [
        Path.cwd() / 'config.json',
        Path.home() / '.math_solver' / 'config.json',
        Path(__file__).parent.parent.parent / 'config' / 'default_config.json'
    ]
    
    for path in common_locations:
        if path.exists():
            return path
            
    return common_locations[-1]  # 返回默认位置

def save_config(config: Dict, path: str) -> None:
    """保存配置到文件
    
    Args:
        config: 配置字典
        path: 保存路径
    """
    path = Path(path)
    
    # 创建目录（如果不存在）
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据文件扩展名选择保存格式
    if path.suffix.lower() == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    else:
        raise ConfigError(f"不支持的配置文件格式: {path.suffix}")

def setup_logger(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """设置日志记录器
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        
    Returns:
        logging.Logger: 日志记录器
    """
    try:
        # 移除所有现有的处理器
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers:
                root.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler('math_solver.log', mode='w')
        file_handler.setFormatter(formatter)
        
        # 设置根日志记录器
        root.setLevel(logging.INFO)
        root.addHandler(console_handler)
        root.addHandler(file_handler)
        
    except Exception as e:
        print(f"设置日志时出错: {e}")
        # 确保至少有基本的日志功能
        logging.basicConfig(level=logging.INFO)
    
    # 基本配置
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 设置格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.handlers = []  # 清除所有已有的处理器
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"无法创建日志文件处理器: {str(e)}")
    
    return logger
