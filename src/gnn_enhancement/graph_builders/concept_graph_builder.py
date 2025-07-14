"""
Concept Graph Builder
====================

概念图构建器，封装MathConceptGNN的功能
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConceptGraphBuilder:
    """概念图构建器"""
    
    def __init__(self):
        """初始化概念图构建器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ConceptGraphBuilder initialized")
    
    def build_concept_graph(self, problem_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建概念图
        
        Args:
            problem_text: 问题文本
            context: 上下文信息
            
        Returns:
            概念图信息
        """
        try:
            # 导入MathConceptGNN（延迟导入避免循环依赖）
            from ..core.concept_gnn import MathConceptGNN

            # 创建MathConceptGNN实例
            concept_gnn = MathConceptGNN()
            
            # 提取实体
            entities = self._extract_entities(problem_text)
            
            # 使用MathConceptGNN构建概念图
            concept_graph = concept_gnn.build_concept_graph(problem_text, entities)
            
            return concept_graph
            
        except Exception as e:
            self.logger.error(f"Failed to build concept graph: {e}")
            return {"error": str(e)}
    
    def _extract_entities(self, problem_text: str) -> List[str]:
        """提取实体"""
        import re
        
        entities = []
        
        # 提取数字
        numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
        entities.extend(numbers)
        
        # 提取单位
        units = re.findall(r'(cm|m|km|mm|l|ml|L|kg|g|mg|s|min|h)', problem_text)
        entities.extend(units)
        
        # 提取关键词
        keywords = ['area', 'volume', 'length', 'width', 'height', 'speed', 'time', 'distance']
        for keyword in keywords:
            if keyword.lower() in problem_text.lower():
                entities.append(keyword)
        
        return list(set(entities)) 