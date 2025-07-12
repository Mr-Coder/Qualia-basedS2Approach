"""
Processors Module - Modular Architecture
========================================

处理器模块：采用模块化架构设计

结构:
- private/: 私有实现
  - validator.py: 数据验证
  - processor.py: 业务处理
  - utils.py: 工具函数
- public_api.py: 公共接口
- orchestrator.py: 模块协调器

Author: AI Assistant
Date: 2024-07-13
"""

import logging

# 配置日志
logger = logging.getLogger(__name__)

from .orchestrator import ProcessorsOrchestrator, processors_orchestrator
from .private.processor import CoreProcessor, core_processor
from .private.utils import processing_utils
from .private.validator import ProcessorValidator, validator
# 导入新的模块化架构
from .public_api import ProcessorsAPI, processors_api

# 导入原有的处理器类以保持兼容性
try:
    from .complexity_classifier import ComplexityClassifier
    from .dataset_loader import DatasetLoader
    from .implicit_relation_annotator import ImplicitRelationAnnotator
    from .inference_tracker import InferenceTracker
    from .relation_extractor import RelationExtractor
    from .relation_matcher import RelationMatcher

    # 标记原有类可用
    legacy_processors_available = True
    
except ImportError as e:
    logger.warning(f"某些原有处理器类无法导入: {e}")
    legacy_processors_available = False

# 尝试导入可选的处理器
try:
    from .solution_generator import SolutionGenerator
    has_solution_generator = True
except ImportError:
    has_solution_generator = False

# 新模块化架构的主要导出
__all__ = [
    # 新架构核心组件
    'ProcessorsAPI',
    'ProcessorsOrchestrator', 
    'ProcessorValidator',
    'CoreProcessor',
    'processing_utils',
    
    # 全局实例
    'processors_api',
    'processors_orchestrator',
    'validator',
    'core_processor',
]

# 如果原有处理器可用，添加到导出列表
if legacy_processors_available:
    __all__.extend([
        'RelationExtractor',
        'RelationMatcher', 
        'ComplexityClassifier',
        'DatasetLoader',
        'ImplicitRelationAnnotator',
        'InferenceTracker'
    ])

# 如果SolutionGenerator可用，添加到导出列表
if has_solution_generator:
    __all__.append('SolutionGenerator')

# 模块元数据
__version__ = '2.0.0'  # 升级版本号以反映模块化架构
__author__ = 'AI Assistant'
__license__ = 'MIT'

# 初始化日志
logger.info(f"Processors Module v{__version__} - Modular Architecture Initialized")
logger.info(f"Legacy processors available: {legacy_processors_available}")
logger.info(f"Solution generator available: {has_solution_generator}")

def get_version() -> str:
    """返回模块版本信息"""
    return __version__

def get_module_info() -> dict:
    """返回模块详细信息"""
    return {
        "name": "processors",
        "version": __version__,
        "architecture": "modular",
        "components": {
            "public_api": "ProcessorsAPI",
            "orchestrator": "ProcessorsOrchestrator", 
            "validator": "ProcessorValidator",
            "core_processor": "CoreProcessor",
            "utils": "processing_utils"
        },
        "legacy_support": legacy_processors_available,
        "optional_components": {
            "solution_generator": has_solution_generator
        }
    }

def initialize_module(auto_init: bool = True) -> bool:
    """
    初始化处理器模块
    
    Args:
        auto_init: 是否自动初始化
        
    Returns:
        初始化是否成功
    """
    if auto_init:
        return processors_api.initialize()
    return True

def get_module_status() -> dict:
    """获取模块状态"""
    return processors_api.get_module_status()

def shutdown_module() -> bool:
    """关闭模块"""
    return processors_api.shutdown()

# 便捷函数：直接访问主要功能
def process_text(text, config=None, validate_input=True):
    """便捷函数：处理文本"""
    if not processors_api._initialized:
        processors_api.initialize()
    return processors_api.process_text(text, config, validate_input)

def process_dataset(dataset, config=None, validate_input=True):
    """便捷函数：处理数据集"""
    if not processors_api._initialized:
        processors_api.initialize()
    return processors_api.process_dataset(dataset, config, validate_input)

def extract_relations(text, config=None):
    """便捷函数：提取关系"""
    if not processors_api._initialized:
        processors_api.initialize()
    return processors_api.extract_relations(text, config)

def classify_complexity(text, config=None):
    """便捷函数：分类复杂度"""
    if not processors_api._initialized:
        processors_api.initialize()
    return processors_api.classify_complexity(text, config)

def process_nlp(text, config=None):
    """便捷函数：NLP处理"""
    if not processors_api._initialized:
        processors_api.initialize()
    return processors_api.process_nlp(text, config)

def batch_process(inputs, config=None):
    """便捷函数：批量处理"""
    if not processors_api._initialized:
        processors_api.initialize()
    return processors_api.batch_process(inputs, config)

# 导出便捷函数
__all__.extend([
    'get_version',
    'get_module_info', 
    'initialize_module',
    'get_module_status',
    'shutdown_module',
    'process_text',
    'process_dataset',
    'extract_relations',
    'classify_complexity', 
    'process_nlp',
    'batch_process'
])

# 包初始化时的设置
logger.info("Processors module initialization complete")
