"""
配置管理子模块

AI_CONTEXT: 配置系统的核心组件
RESPONSIBILITY: 提供AI友好的配置管理功能
"""

from .config_manager import (AICollaborativeConfigManager, ConfigurationSchema,
                             create_default_config_manager,
                             create_sample_config_file)

__all__ = [
    'AICollaborativeConfigManager',
    'ConfigurationSchema',
    'create_default_config_manager', 
    'create_sample_config_file'
] 