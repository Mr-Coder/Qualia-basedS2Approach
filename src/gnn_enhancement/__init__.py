"""
GNN Enhancement Module for COT-DIR1
===================================

图神经网络增强模块，用于优化COT-DIR1数学推理算法

主要组件:
- core/: 核心GNN组件
  - concept_gnn/: 数学概念图神经网络
  - reasoning_gnn/: 推理过程图神经网络
  - verification_gnn/: 验证图神经网络
- graph_builders/: 图构建器
- models/: GNN模型实现
- utils/: 工具函数
- integration/: 与现有模块的集成

Author: AI Assistant
Date: 2024-07-13
"""

import logging
from typing import Any, Dict, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 版本信息
__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "GNN Enhancement for COT-DIR1 Mathematical Reasoning"

# 模块可用性标志
_gnn_available = True
_torch_available = True
_dgl_available = True

# 检查依赖
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    _torch_available = False
    logger.warning("PyTorch not available. GNN functionality will be limited.")

try:
    import dgl
    import dgl.nn as dglnn
except ImportError:
    _dgl_available = False
    logger.warning("DGL not available. Using alternative graph library.")

# 核心组件导入
from .core.concept_gnn import MathConceptGNN
from .core.reasoning_gnn import ReasoningGNN
from .core.verification_gnn import VerificationGNN
from .graph_builders import GraphBuilder
from .integration import GNNIntegrator
from .utils import GNNUtils

# 主要导出
__all__ = [
    'MathConceptGNN',
    'ReasoningGNN', 
    'VerificationGNN',
    'GraphBuilder',
    'GNNIntegrator',
    'GNNUtils',
    'get_gnn_status',
    'initialize_gnn_module'
]

def get_gnn_status() -> Dict[str, Any]:
    """获取GNN模块状态信息"""
    return {
        "version": __version__,
        "gnn_available": _gnn_available,
        "torch_available": _torch_available,
        "dgl_available": _dgl_available,
        "components": {
            "concept_gnn": True,
            "reasoning_gnn": True,
            "verification_gnn": True,
            "graph_builders": True,
            "integration": True
        }
    }

def initialize_gnn_module(config: Optional[Dict[str, Any]] = None) -> bool:
    """初始化GNN模块"""
    try:
        config = config or {}
        
        # 检查必要依赖
        if not _torch_available:
            logger.error("PyTorch is required for GNN functionality")
            return False
            
        logger.info(f"GNN Enhancement Module v{__version__} initialized successfully")
        logger.info(f"PyTorch available: {_torch_available}")
        logger.info(f"DGL available: {_dgl_available}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize GNN module: {e}")
        return False

# 模块初始化
logger.info("GNN Enhancement Module loaded") 