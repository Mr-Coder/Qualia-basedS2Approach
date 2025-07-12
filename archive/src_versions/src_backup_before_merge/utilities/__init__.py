"""
AI协作友好的工具模块

这个模块提供了系统所需的各种工具和辅助功能。

AI_CONTEXT: 通用工具集合，支持整个系统的运行
RESPONSIBILITY: 配置管理、日志系统、测试工具等
"""

from .configuration.config_manager import (AICollaborativeConfigManager,
                                           ConfigurationSchema,
                                           create_default_config_manager,
                                           create_sample_config_file)

__all__ = [
    'AICollaborativeConfigManager',
    'ConfigurationSchema', 
    'create_default_config_manager',
    'create_sample_config_file'
]

# AI_HINT: 这个模块提供系统基础设施工具 