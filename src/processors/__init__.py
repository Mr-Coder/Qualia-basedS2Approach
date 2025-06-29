"""
处理器模块
~~~~~~~~~

这个模块包含了所有的处理器类，用于处理和解决数学应用题。

Author: [Your Name]
Date: [Current Date]
"""

import logging

# 配置日志
logger = logging.getLogger(__name__)

# from .equation_builder import EquationBuilder  # 临时注释掉
# 使用相对导入
# from .nlp_processor import NLPProcessor  # 临时注释掉，有导入问题
from .relation_extractor import RelationExtractor
from .relation_matcher import RelationMatcher

# 查看是否存在SolutionGenerator模块，如果不存在则跳过导入
try:
    from .solution_generator import SolutionGenerator
    has_solution_generator = True
except ImportError:
    has_solution_generator = False

from .complexity_classifier import ComplexityClassifier
# 导入新的处理器类
from .dataset_loader import DatasetLoader
from .implicit_relation_annotator import ImplicitRelationAnnotator
from .inference_tracker import InferenceTracker

# 为每个处理器类添加日志记录器
processor_classes = [
    # EquationBuilder,  # 临时注释掉
    InferenceTracker,
    # NLPProcessor,  # 临时注释掉，有导入问题
    RelationExtractor,
    RelationMatcher,
    DatasetLoader,
    ComplexityClassifier,
    ImplicitRelationAnnotator
]

# 如果SolutionGenerator存在则添加
if has_solution_generator:
    processor_classes.append(SolutionGenerator)

for cls in processor_classes:
    if not hasattr(cls, 'logger'):
        setattr(cls, 'logger', logging.getLogger(f"{__name__}.{cls.__name__}"))

# 导出的类
__all__ = [
    # 'NLPProcessor',  # 临时注释掉，有导入问题
    'RelationExtractor', 
    # 'EquationBuilder',  # 临时注释掉
    'RelationMatcher',
    'InferenceTracker',
    'DatasetLoader',
    'ComplexityClassifier',
    'ImplicitRelationAnnotator'
]

# 如果SolutionGenerator存在则添加到导出列表
if has_solution_generator:
    __all__.append('SolutionGenerator')
