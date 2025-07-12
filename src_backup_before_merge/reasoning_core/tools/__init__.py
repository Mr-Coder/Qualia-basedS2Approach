"""
Reasoning Tools - 精简版
========================

精简后的推理工具模块，只保留核心工具：
- ComplexityAnalyzer: 复杂度分析工具（核心功能）
- RelationDiscoveryTool: 关系发现工具（IRD模块核心）
- BaseTool: 工具基类
- VisualizationTool: 可视化工具（保留用于结果展示）

已移除的工具：
- NumericalComputeTool: 功能简单，可用标准库替代
- SymbolicMathTool: 使用频率低
"""

from .base_tool import BaseTool
from .complexity_analyzer import ComplexityAnalyzer
from .relation_discovery import RelationDiscoveryTool
from .visualization import VisualizationTool

__all__ = [
    'BaseTool',
    'ComplexityAnalyzer',
    'RelationDiscoveryTool', 
    'VisualizationTool'
]

# 使用说明：
"""
核心工具使用示例：

# 1. 复杂度分析
from src.reasoning_core.tools import ComplexityAnalyzer
analyzer = ComplexityAnalyzer()
complexity = analyzer.analyze_complexity(problem_text)

# 2. 关系发现
from src.reasoning_core.tools import RelationDiscoveryTool
tool = RelationDiscoveryTool()
relations = tool.discover_relations(entities, context)
"""
