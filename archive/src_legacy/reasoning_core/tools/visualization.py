"""
Visualization Tool
=================

Tool for creating visualizations and plots.
"""

import logging
from typing import Any, Dict, List

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class VisualizationTool(BaseTool):
    """Tool for creating visualizations using matplotlib"""
    
    def __init__(self, config: dict = None):
        super().__init__("visualization", config)
        self.matplotlib = None
        self.plt = None
        if self.is_available:
            import matplotlib
            import matplotlib.pyplot as plt
            self.matplotlib = matplotlib
            self.plt = plt
    
    def _check_availability(self) -> bool:
        """Check if matplotlib is available"""
        try:
            import matplotlib
            return True
        except ImportError:
            logger.warning("Matplotlib not available - visualization tools disabled")
            return False
    
    def execute(self, operation: str, *args, **kwargs) -> ToolResult:
        """Execute visualization operation"""
        if not self.is_available:
            return ToolResult(
                success=False,
                result=None,
                error_message="Matplotlib not available"
            )
        
        try:
            if operation == "plot_line":
                return self._plot_line(*args, **kwargs)
            elif operation == "plot_bar":
                return self._plot_bar(*args, **kwargs)
            elif operation == "plot_histogram":
                return self._plot_histogram(*args, **kwargs)
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error_message=f"Unsupported operation: {operation}"
                )
                
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return ToolResult(
                success=False,
                result=None,
                error_message=str(e)
            )
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations"""
        return [
            "plot_line",
            "plot_bar",
            "plot_histogram"
        ]
    
    def _plot_line(self, x_data: List[float], y_data: List[float], 
                   title: str = "Line Plot", save_path: str = None) -> ToolResult:
        """Create a line plot"""
        self.plt.figure(figsize=(8, 6))
        self.plt.plot(x_data, y_data)
        self.plt.title(title)
        self.plt.xlabel("X")
        self.plt.ylabel("Y")
        self.plt.grid(True)
        
        if save_path:
            self.plt.savefig(save_path)
            result = f"Plot saved to {save_path}"
        else:
            result = "Plot created (not saved)"
        
        self.plt.close()
        
        return ToolResult(
            success=True,
            result=result,
            metadata={
                'plot_type': 'line',
                'data_points': len(x_data),
                'title': title,
                'saved': bool(save_path)
            }
        )
    
    def _plot_bar(self, categories: List[str], values: List[float],
                  title: str = "Bar Plot", save_path: str = None) -> ToolResult:
        """Create a bar plot"""
        self.plt.figure(figsize=(8, 6))
        self.plt.bar(categories, values)
        self.plt.title(title)
        self.plt.xlabel("Categories")
        self.plt.ylabel("Values")
        self.plt.xticks(rotation=45)
        
        if save_path:
            self.plt.savefig(save_path, bbox_inches='tight')
            result = f"Plot saved to {save_path}"
        else:
            result = "Plot created (not saved)"
        
        self.plt.close()
        
        return ToolResult(
            success=True,
            result=result,
            metadata={
                'plot_type': 'bar',
                'categories': len(categories),
                'title': title,
                'saved': bool(save_path)
            }
        )
    
    def _plot_histogram(self, data: List[float], bins: int = 10,
                       title: str = "Histogram", save_path: str = None) -> ToolResult:
        """Create a histogram"""
        self.plt.figure(figsize=(8, 6))
        self.plt.hist(data, bins=bins, alpha=0.7)
        self.plt.title(title)
        self.plt.xlabel("Values")
        self.plt.ylabel("Frequency")
        
        if save_path:
            self.plt.savefig(save_path)
            result = f"Plot saved to {save_path}"
        else:
            result = "Plot created (not saved)"
        
        self.plt.close()
        
        return ToolResult(
            success=True,
            result=result,
            metadata={
                'plot_type': 'histogram',
                'data_points': len(data),
                'bins': bins,
                'title': title,
                'saved': bool(save_path)
            }
        ) 