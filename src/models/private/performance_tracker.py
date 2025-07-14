"""
性能监控器 (Performance Monitor)

专注于模型性能的监控、分析和报告。
"""

import logging
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import statistics
from datetime import datetime, timedelta


@dataclass
class PerformanceMetric:
    """性能指标"""
    timestamp: float
    model_name: str
    operation: str
    duration: float
    success: bool
    input_size: int
    output_size: int
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "operation": self.operation,
            "duration": self.duration,
            "success": self.success,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "error_message": self.error_message
        }


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, max_metrics: int = 10000):
        """初始化性能跟踪器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 性能指标存储
        self.metrics = deque(maxlen=max_metrics)
        self._lock = threading.RLock()
        
        # 实时统计
        self.model_stats = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "min_duration": float('inf'),
            "max_duration": 0.0,
            "last_call": None
        })
        
        # 操作统计
        self.operation_stats = defaultdict(lambda: {
            "total_calls": 0,
            "avg_duration": 0.0,
            "success_rate": 0.0
        })
        
        self.logger.info("性能跟踪器初始化完成")
    
    def record_metric(self, metric: PerformanceMetric):
        """记录性能指标"""
        with self._lock:
            self.metrics.append(metric)
            self._update_model_stats(metric)
            self._update_operation_stats(metric)
    
    def _update_model_stats(self, metric: PerformanceMetric):
        """更新模型统计"""
        stats = self.model_stats[metric.model_name]
        
        stats["total_calls"] += 1
        stats["last_call"] = metric.timestamp
        
        if metric.success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
        
        # 更新持续时间统计
        stats["total_duration"] += metric.duration
        stats["avg_duration"] = stats["total_duration"] / stats["total_calls"]
        stats["min_duration"] = min(stats["min_duration"], metric.duration)
        stats["max_duration"] = max(stats["max_duration"], metric.duration)
    
    def _update_operation_stats(self, metric: PerformanceMetric):
        """更新操作统计"""
        stats = self.operation_stats[metric.operation]
        
        total_calls = stats["total_calls"]
        current_avg = stats["avg_duration"]
        
        stats["total_calls"] += 1
        stats["avg_duration"] = ((current_avg * total_calls) + metric.duration) / stats["total_calls"]
        
        # 更新成功率
        if stats["total_calls"] > 0:
            success_count = sum(1 for m in self.metrics 
                              if m.operation == metric.operation and m.success)
            stats["success_rate"] = success_count / stats["total_calls"]
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """获取模型性能统计"""
        with self._lock:
            if model_name not in self.model_stats:
                return {"error": f"模型 {model_name} 没有性能数据"}
            
            stats = self.model_stats[model_name].copy()
            
            # 计算成功率
            if stats["total_calls"] > 0:
                stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
            else:
                stats["success_rate"] = 0.0
            
            # 添加最近性能趋势
            recent_metrics = [m for m in self.metrics 
                            if m.model_name == model_name and 
                            m.timestamp > time.time() - 3600]  # 最近1小时
            
            if recent_metrics:
                stats["recent_avg_duration"] = statistics.mean([m.duration for m in recent_metrics])
                stats["recent_success_rate"] = sum(m.success for m in recent_metrics) / len(recent_metrics)
            
            return stats
    
    def get_operation_performance(self, operation: str) -> Dict[str, Any]:
        """获取操作性能统计"""
        with self._lock:
            if operation not in self.operation_stats:
                return {"error": f"操作 {operation} 没有性能数据"}
            
            return self.operation_stats[operation].copy()
    
    def get_performance_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """获取性能趋势（时间窗口内）"""
        with self._lock:
            cutoff_time = time.time() - time_window
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {"message": "没有最近的性能数据"}
            
            # 按模型分组
            model_trends = defaultdict(list)
            for metric in recent_metrics:
                model_trends[metric.model_name].append(metric.duration)
            
            trends = {}
            for model_name, durations in model_trends.items():
                trends[model_name] = {
                    "avg_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                    "call_count": len(durations)
                }
            
            return trends


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化性能监控器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 配置参数
        config = config or {}
        self.enable_monitoring = config.get("enable_monitoring", True)
        self.max_metrics = config.get("max_metrics", 10000)
        self.alert_threshold = config.get("alert_threshold", 10.0)  # 响应时间告警阈值（秒）
        self.error_rate_threshold = config.get("error_rate_threshold", 0.1)  # 错误率告警阈值
        
        # 性能跟踪器
        self.tracker = PerformanceTracker(self.max_metrics)
        
        # 告警状态
        self.alerts = []
        self.alert_history = deque(maxlen=1000)
        
        # 系统指标
        self.system_metrics = {
            "start_time": time.time(),
            "total_operations": 0,
            "total_errors": 0,
            "total_processing_time": 0.0
        }
        
        self.logger.info(f"性能监控器初始化完成，监控状态: {self.enable_monitoring}")
    
    def monitor_model_call(
        self, 
        model_name: str, 
        operation: str, 
        start_time: float, 
        end_time: float,
        success: bool,
        input_size: int = 0,
        output_size: int = 0,
        error_message: Optional[str] = None
    ):
        """监控模型调用"""
        if not self.enable_monitoring:
            return
        
        duration = end_time - start_time
        
        # 创建性能指标
        metric = PerformanceMetric(
            timestamp=start_time,
            model_name=model_name,
            operation=operation,
            duration=duration,
            success=success,
            input_size=input_size,
            output_size=output_size,
            error_message=error_message
        )
        
        # 记录指标
        self.tracker.record_metric(metric)
        
        # 更新系统指标
        self._update_system_metrics(metric)
        
        # 检查告警条件
        self._check_alerts(metric)
    
    def _update_system_metrics(self, metric: PerformanceMetric):
        """更新系统指标"""
        self.system_metrics["total_operations"] += 1
        self.system_metrics["total_processing_time"] += metric.duration
        
        if not metric.success:
            self.system_metrics["total_errors"] += 1
    
    def _check_alerts(self, metric: PerformanceMetric):
        """检查告警条件"""
        
        # 检查响应时间告警
        if metric.duration > self.alert_threshold:
            alert = {
                "type": "slow_response",
                "model_name": metric.model_name,
                "operation": metric.operation,
                "duration": metric.duration,
                "threshold": self.alert_threshold,
                "timestamp": metric.timestamp
            }
            self._trigger_alert(alert)
        
        # 检查错误率告警
        if not metric.success:
            recent_calls = [m for m in self.tracker.metrics 
                          if m.model_name == metric.model_name and 
                          m.timestamp > time.time() - 300]  # 最近5分钟
            
            if len(recent_calls) >= 10:  # 至少10次调用
                error_rate = sum(1 for m in recent_calls if not m.success) / len(recent_calls)
                
                if error_rate > self.error_rate_threshold:
                    alert = {
                        "type": "high_error_rate",
                        "model_name": metric.model_name,
                        "error_rate": error_rate,
                        "threshold": self.error_rate_threshold,
                        "recent_calls": len(recent_calls),
                        "timestamp": metric.timestamp
                    }
                    self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """触发告警"""
        alert["id"] = len(self.alert_history)
        alert["triggered_at"] = time.time()
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        self.logger.warning(f"性能告警: {alert['type']} - {alert}")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        uptime = time.time() - self.system_metrics["start_time"]
        
        overview = {
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "total_operations": self.system_metrics["total_operations"],
            "total_errors": self.system_metrics["total_errors"],
            "total_processing_time": self.system_metrics["total_processing_time"],
            "active_alerts": len(self.alerts),
            "monitoring_enabled": self.enable_monitoring
        }
        
        # 计算平均值
        if overview["total_operations"] > 0:
            overview["avg_processing_time"] = (
                overview["total_processing_time"] / overview["total_operations"]
            )
            overview["error_rate"] = overview["total_errors"] / overview["total_operations"]
            overview["operations_per_hour"] = overview["total_operations"] / (uptime / 3600)
        else:
            overview["avg_processing_time"] = 0.0
            overview["error_rate"] = 0.0
            overview["operations_per_hour"] = 0.0
        
        return overview
    
    def get_model_ranking(self, metric: str = "avg_duration") -> List[Dict[str, Any]]:
        """获取模型性能排名"""
        rankings = []
        
        for model_name, stats in self.tracker.model_stats.items():
            ranking_entry = {
                "model_name": model_name,
                "metric_value": stats.get(metric, 0),
                "total_calls": stats["total_calls"],
                "success_rate": stats["successful_calls"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
            }
            rankings.append(ranking_entry)
        
        # 根据指标排序
        reverse = metric not in ["avg_duration", "min_duration", "failed_calls"]
        rankings.sort(key=lambda x: x["metric_value"], reverse=reverse)
        
        return rankings
    
    def get_performance_report(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            "generated_at": time.time(),
            "system_overview": self.get_system_overview(),
            "alerts": {
                "active": self.alerts,
                "total_triggered": len(self.alert_history)
            }
        }
        
        if model_name:
            # 特定模型的详细报告
            report["model_performance"] = self.tracker.get_model_performance(model_name)
        else:
            # 所有模型的概要报告
            report["model_rankings"] = {
                "by_speed": self.get_model_ranking("avg_duration"),
                "by_success_rate": self.get_model_ranking("success_rate"),
                "by_usage": self.get_model_ranking("total_calls")
            }
            
            # 趋势分析
            report["trends"] = self.tracker.get_performance_trends()
        
        return report
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        system_overview = self.get_system_overview()
        
        # 健康评分 (0-100)
        health_score = 100
        health_issues = []
        
        # 检查错误率
        if system_overview["error_rate"] > self.error_rate_threshold:
            health_score -= 30
            health_issues.append(f"高错误率: {system_overview['error_rate']:.2%}")
        
        # 检查活跃告警
        if system_overview["active_alerts"] > 0:
            health_score -= 20
            health_issues.append(f"活跃告警: {system_overview['active_alerts']}个")
        
        # 检查平均响应时间
        if system_overview["avg_processing_time"] > self.alert_threshold:
            health_score -= 25
            health_issues.append(f"响应时间慢: {system_overview['avg_processing_time']:.2f}s")
        
        # 确定健康状态
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": max(0, health_score),
            "issues": health_issues,
            "recommendations": self._get_health_recommendations(health_issues),
            "last_updated": time.time()
        }
    
    def _get_health_recommendations(self, issues: List[str]) -> List[str]:
        """获取健康建议"""
        recommendations = []
        
        for issue in issues:
            if "错误率" in issue:
                recommendations.append("检查模型配置和输入数据质量")
            elif "告警" in issue:
                recommendations.append("查看告警详情并采取相应措施")
            elif "响应时间" in issue:
                recommendations.append("考虑优化模型或增加缓存")
        
        if not recommendations:
            recommendations.append("系统运行良好，继续监控")
        
        return recommendations
    
    def clear_alerts(self, alert_type: Optional[str] = None):
        """清除告警"""
        if alert_type:
            self.alerts = [a for a in self.alerts if a["type"] != alert_type]
            self.logger.info(f"已清除类型为 {alert_type} 的告警")
        else:
            self.alerts.clear()
            self.logger.info("已清除所有告警")
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """导出性能指标"""
        metrics_data = {
            "export_time": time.time(),
            "metrics_count": len(self.tracker.metrics),
            "metrics": [metric.to_dict() for metric in self.tracker.metrics]
        }
        
        if format == "json":
            import json
            return json.dumps(metrics_data, indent=2, ensure_ascii=False)
        else:
            return metrics_data
    
    def reset_metrics(self):
        """重置所有性能指标"""
        self.tracker.metrics.clear()
        self.tracker.model_stats.clear()
        self.tracker.operation_stats.clear()
        self.alerts.clear()
        
        self.system_metrics = {
            "start_time": time.time(),
            "total_operations": 0,
            "total_errors": 0,
            "total_processing_time": 0.0
        }
        
        self.logger.info("性能指标已重置")
    
    def shutdown(self):
        """关闭性能监控器"""
        self.enable_monitoring = False
        self.logger.info("性能监控器已关闭")