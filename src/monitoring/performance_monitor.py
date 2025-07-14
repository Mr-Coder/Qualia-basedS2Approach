"""
性能监控系统
提供系统性能指标收集和监控功能
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil

from ..config.config_manager import get_config
from ..core.exceptions import PerformanceError, TimeoutError

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "tags": self.tags
        }

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        try:
            self.config = get_config()
        except Exception:
            self.config = None
            
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.active_timers = {}
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.custom_metrics = {}
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_monitoring = False
        
        # 性能阈值
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "processing_time": 30.0,
            "error_rate": 0.1
        }
        
        self._init_system_monitoring()
    
    def _init_system_monitoring(self):
        """初始化系统监控"""
        enable_monitoring = self._get_config_value("performance.enable_monitoring", True)
        if enable_monitoring:
            self.start_system_monitoring()
    
    def _get_config_value(self, key: str, default: Any) -> Any:
        """安全获取配置值"""
        if self.config:
            return self.config.get(key, default)
        return default
    
    def start_timer(self, name: str, tags: Dict[str, str] = None) -> str:
        """开始计时器"""
        timer_id = f"{name}_{int(time.time() * 1000000)}"
        
        with self._lock:
            self.active_timers[timer_id] = {
                "name": name,
                "start_time": time.time(),
                "tags": tags or {}
            }
        
        return timer_id
    
    def stop_timer(self, timer_id: str) -> Optional[float]:
        """停止计时器并记录性能指标"""
        with self._lock:
            if timer_id not in self.active_timers:
                logger.warning(f"Timer {timer_id} not found")
                return None
            
            timer_info = self.active_timers.pop(timer_id)
            duration = time.time() - timer_info["start_time"]
            
            # 记录指标
            metric = PerformanceMetric(
                name=f"{timer_info['name']}_duration",
                value=duration,
                timestamp=datetime.now(),
                unit="seconds",
                tags=timer_info["tags"]
            )
            
            self.record_metric(metric)
            
            # 检查性能阈值
            self._check_performance_threshold(timer_info["name"], duration)
            
            return duration
    
    def record_metric(self, metric: PerformanceMetric):
        """记录性能指标"""
        with self._lock:
            self.metrics_history[metric.name].append(metric)
            
            # 记录到日志
            logger.debug(f"Metric recorded: {metric.name}={metric.value}{metric.unit}")
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """增加计数器"""
        with self._lock:
            self.counters[name] += value
            
        # 记录为指标
        metric = PerformanceMetric(
            name=f"{name}_count",
            value=self.counters[name],
            timestamp=datetime.now(),
            unit="count",
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """设置仪表值"""
        with self._lock:
            self.gauges[name] = value
            
        # 记录为指标
        metric = PerformanceMetric(
            name=f"{name}_gauge",
            value=value,
            timestamp=datetime.now(),
            unit="value",
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def get_metrics_summary(self, time_window: timedelta = None) -> Dict[str, Any]:
        """获取性能指标摘要"""
        if time_window is None:
            time_window = timedelta(minutes=10)
        
        cutoff_time = datetime.now() - time_window
        summary = {
            "time_window": str(time_window),
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "system": self._get_system_metrics(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }
        
        with self._lock:
            for metric_name, metrics_list in self.metrics_history.items():
                # 过滤时间窗口内的指标
                recent_metrics = [
                    m for m in metrics_list 
                    if m.timestamp >= cutoff_time
                ]
                
                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    summary["metrics"][metric_name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "latest": values[-1] if values else None
                    }
        
        return summary
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """获取系统性能指标"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                "disk_free_gb": disk.free / (1024 * 1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            return {}
    
    def _check_performance_threshold(self, operation_name: str, duration: float):
        """检查性能阈值"""
        threshold_key = f"{operation_name}_time"
        threshold = self.thresholds.get(threshold_key, self.thresholds.get("processing_time", 30.0))
        
        if duration > threshold:
            logger.warning(f"Performance threshold exceeded for {operation_name}: {duration:.2f}s > {threshold}s")
            
            # 记录性能警告指标
            metric = PerformanceMetric(
                name="performance_warning",
                value=duration,
                timestamp=datetime.now(),
                unit="seconds",
                tags={
                    "operation": operation_name,
                    "threshold": str(threshold),
                    "severity": "warning"
                }
            )
            self.record_metric(metric)
    
    def start_system_monitoring(self):
        """开始系统监控线程"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self._monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_system_monitoring(self):
        """停止系统监控"""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_system(self):
        """系统监控循环"""
        interval = self._get_config_value("monitoring.metrics_interval", 60)
        
        while not self._stop_monitoring:
            try:
                # 收集系统指标
                system_metrics = self._get_system_metrics()
                
                for metric_name, value in system_metrics.items():
                    metric = PerformanceMetric(
                        name=f"system_{metric_name}",
                        value=value,
                        timestamp=datetime.now(),
                        unit=self._get_metric_unit(metric_name),
                        tags={"source": "system"}
                    )
                    self.record_metric(metric)
                
                # 检查系统阈值
                self._check_system_thresholds(system_metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {str(e)}")
                time.sleep(10)  # 出错时短暂等待
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """获取指标单位"""
        unit_map = {
            "cpu_percent": "%",
            "memory_percent": "%",
            "memory_used_mb": "MB",
            "memory_available_mb": "MB",
            "disk_percent": "%",
            "disk_used_gb": "GB",
            "disk_free_gb": "GB"
        }
        return unit_map.get(metric_name, "")
    
    def _check_system_thresholds(self, system_metrics: Dict[str, Any]):
        """检查系统阈值"""
        for metric_name, value in system_metrics.items():
            threshold = self.thresholds.get(metric_name)
            if threshold and value > threshold:
                logger.warning(f"System threshold exceeded: {metric_name}={value} > {threshold}")
                
                # 记录系统警告
                metric = PerformanceMetric(
                    name="system_warning",
                    value=value,
                    timestamp=datetime.now(),
                    unit=self._get_metric_unit(metric_name),
                    tags={
                        "metric": metric_name,
                        "threshold": str(threshold),
                        "severity": "warning"
                    }
                )
                self.record_metric(metric)
    
    def export_metrics(self, format_type: str = "json") -> str:
        """导出性能指标"""
        summary = self.get_metrics_summary()
        
        if format_type == "json":
            return json.dumps(summary, indent=2, default=str)
        elif format_type == "csv":
            return self._export_as_csv(summary)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_as_csv(self, summary: Dict[str, Any]) -> str:
        """导出为CSV格式"""
        lines = ["metric_name,value,unit,timestamp"]
        
        for metric_name, metric_data in summary["metrics"].items():
            lines.append(f"{metric_name},{metric_data['latest']},value,{summary['timestamp']}")
        
        for counter_name, value in summary["counters"].items():
            lines.append(f"{counter_name},{value},count,{summary['timestamp']}")
        
        for gauge_name, value in summary["gauges"].items():
            lines.append(f"{gauge_name},{value},gauge,{summary['timestamp']}")
        
        return "\n".join(lines)

# 性能监控装饰器
def monitor_performance(operation_name: str = None, tags: Dict[str, str] = None):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            timer_id = monitor.start_timer(op_name, tags)
            
            try:
                result = func(*args, **kwargs)
                monitor.increment_counter(f"{op_name}_success")
                return result
            except Exception as e:
                monitor.increment_counter(f"{op_name}_error")
                raise
            finally:
                monitor.stop_timer(timer_id)
        
        return wrapper
    return decorator

# 超时监控装饰器
def timeout_monitor(timeout_seconds: float, operation_name: str = None):
    """超时监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > timeout_seconds:
                    raise TimeoutError(
                        f"Operation {op_name} exceeded timeout: {duration:.2f}s > {timeout_seconds}s",
                        timeout_seconds=timeout_seconds,
                        context={"operation": op_name, "actual_duration": duration}
                    )
                
                return result
            except TimeoutError:
                raise
            except Exception as e:
                duration = time.time() - start_time
                raise PerformanceError(
                    f"Operation {op_name} failed after {duration:.2f}s",
                    operation=op_name,
                    duration=duration,
                    cause=e
                )
        
        return wrapper
    return decorator

# 全局监控器实例
_global_monitor = None

def get_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def init_monitor() -> PerformanceMonitor:
    """初始化全局性能监控器"""
    global _global_monitor
    _global_monitor = PerformanceMonitor()
    return _global_monitor 