import datetime
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class InferenceTracker:
    """推理过程跟踪器，记录每一步推理的输入输出和历史"""
    
    def __init__(self, max_history_size: int = 100):
        """初始化推理跟踪器
        
        Args:
            max_history_size: 最大历史记录大小
        """
        self.history = []
        self.start_time = None
        self.end_time = None
        self.is_tracking = False
        self.max_history_size = max_history_size
        self.step_count = 0
        self.logger = logging.getLogger(__name__)

    def start_tracking(self):
        """开始跟踪推理过程"""
        self.history = []
        self.start_time = datetime.datetime.now()
        self.is_tracking = True
        self.step_count = 0
        self.logger.info("开始推理跟踪")

    def add_inference(self, step_name: str, input_data: Any, output_data: Any):
        """添加一步推理记录
        
        Args:
            step_name: 步骤名称
            input_data: 输入数据
            output_data: 输出数据
        """
        if not self.is_tracking:
            self.logger.warning("推理跟踪未开始，请先调用 start_tracking()")
            return
            
        self.step_count += 1
        
        # 防止历史记录过大
        if len(self.history) >= self.max_history_size:
            self.logger.warning(f"历史记录已达到最大大小 {self.max_history_size}，将删除最早的记录")
            self.history.pop(0)
        
        # 记录推理步骤
        step = {
            'step_id': self.step_count,
            'step_name': step_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'input': self._safe_serialize(input_data),
            'output': self._safe_serialize(output_data)
        }
        
        self.history.append(step)
        self.logger.debug(f"添加推理步骤: {step_name} (步骤 {self.step_count})")

    def get_inference_history(self) -> List[Dict[str, Any]]:
        """获取推理历史记录
        
        Returns:
            List[Dict]: 推理历史记录列表
        """
        return self.history

    def get_inference_summary(self) -> str:
        """获取推理摘要
        
        Returns:
            str: 推理摘要文本
        """
        if not self.history:
            return "无推理历史记录"
            
        summary_lines = ["推理过程摘要:"]
        
        # 添加时间信息
        if self.start_time:
            start_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
            summary_lines.append(f"开始时间: {start_str}")
        if self.end_time:
            end_str = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = (self.end_time - self.start_time).total_seconds()
            summary_lines.append(f"结束时间: {end_str}")
            summary_lines.append(f"总耗时: {duration:.2f} 秒")
        
        # 添加步骤摘要
        summary_lines.append(f"总步骤数: {len(self.history)}")
        for i, step in enumerate(self.history):
            summary_lines.append(f"步骤 {i+1}: {step['step_name']}")
        
        return "\n".join(summary_lines)

    def end_tracking(self):
        """结束推理跟踪"""
        if not self.is_tracking:
            self.logger.warning("推理跟踪未开始，无法结束")
            return
            
        self.end_time = datetime.datetime.now()
        self.is_tracking = False
        
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"结束推理跟踪，共 {self.step_count} 步，耗时 {duration:.2f} 秒")

    def export_history(self, filepath: str):
        """导出推理历史到文件
        
        Args:
            filepath: 导出文件路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"推理历史已导出到 {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"导出推理历史失败: {str(e)}")
            return False
    
    def _safe_serialize(self, data: Any, depth: int = 0, max_depth: int = 10, seen=None) -> Any:
        """安全序列化数据，处理不可序列化的对象
        
        Args:
            data: 输入数据
            depth: 当前递归深度
            max_depth: 最大递归深度
            seen: 已处理对象的集合，用于检测循环引用
            
        Returns:
            可序列化的数据
        """
        # 检查递归深度
        if depth > max_depth:
            return f"<最大递归深度 {max_depth} 已达到>"
            
        # 检测循环引用
        if seen is None:
            seen = set()
            
        # 对于可能有循环引用的对象，使用id检测
        data_id = id(data)
        if data_id in seen:
            return "<循环引用>"
            
        if data is None:
            return None
            
        if isinstance(data, (str, int, float, bool)):
            return data
            
        # 记录当前对象，防止循环引用
        seen.add(data_id)
        
        try:
            if isinstance(data, (list, tuple)):
                return [self._safe_serialize(item, depth + 1, max_depth, seen.copy()) for item in data]
                
            if isinstance(data, dict):
                return {str(k): self._safe_serialize(v, depth + 1, max_depth, seen.copy()) 
                        for k, v in data.items()}
                
            # 处理对象
            if hasattr(data, '__dict__'):
                # 避免处理特殊对象
                if isinstance(data, type):
                    return str(data)
                return self._safe_serialize(data.__dict__, depth + 1, max_depth, seen.copy())
                
            # 其他类型转为字符串
            return str(data)
        except Exception as e:
            return f"<序列化错误: {type(data).__name__}, {str(e)}>" 