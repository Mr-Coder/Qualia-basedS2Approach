"""
Verification Graph Builder
=========================

验证图构建器，封装VerificationGNN的功能
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VerificationGraphBuilder:
    """验证图构建器"""
    
    def __init__(self):
        """初始化验证图构建器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("VerificationGraphBuilder initialized")
    
    def build_verification_graph(self, reasoning_steps: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建验证图
        
        Args:
            reasoning_steps: 推理步骤列表
            context: 上下文信息
            
        Returns:
            验证图信息
        """
        try:
            # 导入VerificationGNN（延迟导入避免循环依赖）
            from ..core.verification_gnn import VerificationGNN

            # 创建VerificationGNN实例
            verification_gnn = VerificationGNN()
            
            # 使用VerificationGNN构建验证图
            verification_graph = verification_gnn.build_verification_graph(reasoning_steps, context)
            
            return verification_graph
            
        except Exception as e:
            self.logger.error(f"Failed to build verification graph: {e}")
            return {"error": str(e)} 