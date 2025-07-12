"""
推理策略模块

AI_CONTEXT: 包含各种推理策略的实现
RESPONSIBILITY: 提供不同的推理方法和策略
"""

from .mlr_strategy import MLRMultiLayerReasoner, MLRReasoningStrategy

__all__ = [
    'MLRReasoningStrategy',
    'MLRMultiLayerReasoner'
] 