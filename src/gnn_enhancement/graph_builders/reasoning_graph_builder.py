"""
Reasoning Graph Builder
======================

推理图构建器，封装ReasoningGNN的功能
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReasoningGraphBuilder:
    """推理图构建器"""
    
    def __init__(self):
        """初始化推理图构建器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ReasoningGraphBuilder initialized")
    
    def build_reasoning_graph(self, reasoning_steps: List[Dict[str, Any]], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建推理图
        
        Args:
            reasoning_steps: 推理步骤列表
            context: 上下文信息
            
        Returns:
            推理图信息
        """
        try:
            # 导入ReasoningGNN（延迟导入避免循环依赖）
            from ..core.reasoning_gnn import ReasoningGNN

            # 创建ReasoningGNN实例
            reasoning_gnn = ReasoningGNN()
            
            # 使用ReasoningGNN构建推理图
            reasoning_graph = reasoning_gnn.build_reasoning_graph(reasoning_steps, context)
            
            return reasoning_graph
            
        except Exception as e:
            self.logger.error(f"Failed to build reasoning graph: {e}")
            return {"error": str(e)} 