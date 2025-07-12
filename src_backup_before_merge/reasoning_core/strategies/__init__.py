"""
Reasoning Strategies Module
==========================

精简后的推理策略模块 - 只保留核心功能
注意：大部分策略功能已集成到核心API中

核心推理功能请使用：
- src.reasoning_engine.cotdir_integration.COTDIRIntegratedWorkflow  
- src.reasoning_engine.strategies.mlr_strategy.MLRMultiLayerReasoner
"""

# 策略功能已整合到核心API中，无需单独导入
# 推荐直接使用：
# from src.reasoning_engine.cotdir_integration import COTDIRIntegratedWorkflow

__all__ = []

# 核心推理API使用说明：
"""
使用示例：
from src.reasoning_engine.cotdir_integration import COTDIRIntegratedWorkflow

workflow = COTDIRIntegratedWorkflow()  
result = workflow.process("数学问题")
"""
