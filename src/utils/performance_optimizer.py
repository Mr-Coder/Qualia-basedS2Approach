#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能优化模块
~~~~~~~~~~

提供缓存、并发处理和性能监控功能

Author: [Hao Meng]
Date: [2025-05-29]
"""

import functools
import hashlib
import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, name: str):
        """开始计时"""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """停止计时并返回耗时"""
        if name not in self.start_times:
            raise ValueError(f"计时器 '{name}' 未启动")
        
        duration = time.time() - self.start_times[name]
        self.metrics[name] = duration
        del self.start_times[name]
        
        self.logger.debug(f"{name} 耗时: {duration:.3f}秒")
        return duration
    
    def time_function(self, func_name: Optional[str] = None):
        """函数计时装饰器"""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_timer(name)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.stop_timer(name)
            
            return wrapper
        return decorator
    
    def get_metrics(self) -> Dict[str, float]:
        """获取所有性能指标"""
        return self.metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics:
            return {"total_time": 0, "step_count": 0, "average_time": 0}
        
        total_time = sum(self.metrics.values())
        step_count = len(self.metrics)
        average_time = total_time / step_count
        
        return {
            "total_time": total_time,
            "step_count": step_count,
            "average_time": average_time,
            "longest_step": max(self.metrics.items(), key=lambda x: x[1]),
            "shortest_step": min(self.metrics.items(), key=lambda x: x[1]),
            "steps": self.metrics
        }
    
    def reset(self):
        """重置所有指标"""
        self.metrics.clear()
        self.start_times.clear()


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.logger = logging.getLogger(__name__)
    
    def _make_key(self, args: Tuple, kwargs: Dict) -> str:
        """生成缓存键"""
        # 创建一个可哈希的键
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        # 序列化为JSON字符串并计算哈希
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """获取缓存值"""
        if key in self.cache:
            # 更新访问顺序
            self.access_order.remove(key)
            self.access_order.append(key)
            return True, self.cache[key]
        
        return False, None
    
    def put(self, key: str, value: Any):
        """存储缓存值"""
        if key in self.cache:
            # 更新现有值
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # 移除最久未使用的项
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            self.logger.debug(f"缓存已满，移除最久未使用的项: {oldest_key}")
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    def hit_rate(self, hits: int, total: int) -> float:
        """计算命中率"""
        return hits / total if total > 0 else 0.0


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 128, enable_persistence: bool = False, 
                 cache_dir: str = "cache"):
        self.lru_cache = LRUCache(max_size)
        self.enable_persistence = enable_persistence
        self.cache_dir = Path(cache_dir)
        self.hit_count = 0
        self.miss_count = 0
        self.logger = logging.getLogger(__name__)
        
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
    
    def cached_function(self, func: Callable) -> Callable:
        """函数缓存装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = self.lru_cache._make_key(args, kwargs)
            
            # 尝试从缓存获取
            hit, value = self.lru_cache.get(cache_key)
            
            if hit:
                self.hit_count += 1
                self.logger.debug(f"缓存命中: {func.__name__}")
                return value
            
            # 缓存未命中，执行函数
            self.miss_count += 1
            result = func(*args, **kwargs)
            
            # 存储到缓存
            self.lru_cache.put(cache_key, result)
            self.logger.debug(f"缓存存储: {func.__name__}")
            
            # 持久化缓存（如果启用）
            if self.enable_persistence:
                self._save_to_persistent_cache(cache_key, result)
            
            return result
        
        return wrapper
    
    def _save_to_persistent_cache(self, key: str, value: Any):
        """保存到持久化缓存"""
        try:
            cache_file = self.cache_dir / f"{key}.pickle"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self.logger.warning(f"持久化缓存保存失败: {e}")
    
    def _load_persistent_cache(self):
        """加载持久化缓存"""
        try:
            for cache_file in self.cache_dir.glob("*.pickle"):
                key = cache_file.stem
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                self.lru_cache.put(key, value)
                
            self.logger.info(f"加载了 {self.lru_cache.size()} 个持久化缓存项")
        except Exception as e:
            self.logger.warning(f"持久化缓存加载失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.lru_cache.hit_rate(self.hit_count, total_requests)
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": self.lru_cache.size(),
            "max_size": self.lru_cache.max_size
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.lru_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        
        if self.enable_persistence:
            # 清理持久化缓存文件
            for cache_file in self.cache_dir.glob("*.pickle"):
                cache_file.unlink()


class ParallelProcessor:
    """并行处理器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, psutil.cpu_count())
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, func: Callable, items: List[Any], 
                     description: str = "批处理") -> List[Any]:
        """批量并行处理"""
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(func, item): i 
                for i, item in enumerate(items)
            }
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    completed_count += 1
                    
                    if completed_count % 10 == 0:  # 每10个任务报告一次进度
                        progress = (completed_count / len(items)) * 100
                        self.logger.info(f"{description} 进度: {progress:.1f}% ({completed_count}/{len(items)})")
                        
                except Exception as e:
                    self.logger.error(f"{description} 任务 {index} 失败: {e}")
                    results[index] = None
        
        self.logger.info(f"{description} 完成，共处理 {len(items)} 个项目")
        return results
    
    def process_with_callback(self, func: Callable, items: List[Any],
                            callback: Callable[[int, Any], None] = None,
                            description: str = "回调处理") -> List[Any]:
        """带回调的并行处理"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(func, item): i 
                for i, item in enumerate(items)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append((index, result))
                    
                    if callback:
                        callback(index, result)
                        
                except Exception as e:
                    self.logger.error(f"{description} 任务 {index} 失败: {e}")
                    results.append((index, None))
        
        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]


class ResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        self.initial_memory = None
    
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.initial_memory = psutil.virtual_memory().used
        self.logger.info("开始系统资源监控")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前系统状态"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        stats = {
            "cpu_percent": cpu_percent,
            "memory_total": memory.total,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_percent": (disk.used / disk.total) * 100,
            "timestamp": time.time()
        }
        
        if self.start_time:
            stats["elapsed_time"] = time.time() - self.start_time
        
        if self.initial_memory:
            stats["memory_growth"] = memory.used - self.initial_memory
        
        return stats
    
    def check_resource_limits(self, max_memory_percent: float = 80.0,
                            max_cpu_percent: float = 90.0) -> List[str]:
        """检查资源限制"""
        warnings = []
        stats = self.get_current_stats()
        
        if stats["memory_percent"] > max_memory_percent:
            warnings.append(f"内存使用率过高: {stats['memory_percent']:.1f}%")
        
        if stats["cpu_percent"] > max_cpu_percent:
            warnings.append(f"CPU使用率过高: {stats['cpu_percent']:.1f}%")
        
        return warnings
    
    def log_stats(self):
        """记录当前统计信息"""
        stats = self.get_current_stats()
        self.logger.info(f"系统状态 - CPU: {stats['cpu_percent']:.1f}%, "
                        f"内存: {stats['memory_percent']:.1f}%, "
                        f"磁盘: {stats['disk_percent']:.1f}%")


class OptimizedSolverMixin:
    """优化求解器混入类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_tracker = PerformanceTracker()
        self.cache_manager = CacheManager(
            max_size=getattr(self.config, 'max_cache_size', 128),
            enable_persistence=getattr(self.config, 'enable_cache_persistence', False)
        )
        self.parallel_processor = ParallelProcessor(
            max_workers=getattr(self.config, 'max_workers', 4)
        )
        self.resource_monitor = ResourceMonitor()
    
    def solve_with_optimization(self, problem_text: str, **kwargs) -> Dict[str, Any]:
        """带优化的求解方法"""
        # 开始资源监控
        self.resource_monitor.start_monitoring()
        
        # 使用性能跟踪
        @self.performance_tracker.time_function("total_solve_time")
        @self.cache_manager.cached_function
        def _solve_cached(text: str):
            return self.solve(text, **kwargs)
        
        try:
            result = _solve_cached(problem_text)
            
            # 添加性能和缓存统计
            result["performance_stats"] = self.performance_tracker.get_summary()
            result["cache_stats"] = self.cache_manager.get_stats()
            result["resource_stats"] = self.resource_monitor.get_current_stats()
            
            # 检查资源警告
            resource_warnings = self.resource_monitor.check_resource_limits()
            if resource_warnings:
                result["resource_warnings"] = resource_warnings
            
            return result
            
        except Exception as e:
            self.logger.error(f"优化求解失败: {e}")
            raise
    
    def batch_solve_optimized(self, problems: List[str], 
                            description: str = "批量求解") -> List[Dict[str, Any]]:
        """优化的批量求解"""
        def solve_single(problem: str) -> Dict[str, Any]:
            return self.solve_with_optimization(problem)
        
        return self.parallel_processor.process_batch(
            solve_single, problems, description
        )


if __name__ == "__main__":
    # 演示性能优化功能
    
    # 性能跟踪演示
    tracker = PerformanceTracker()
    
    @tracker.time_function("test_function")
    def test_function():
        time.sleep(0.1)
        return "测试完成"
    
    result = test_function()
    print("性能跟踪演示:")
    print(tracker.get_summary())
    
    # 缓存管理演示
    cache_manager = CacheManager(max_size=5)
    
    @cache_manager.cached_function
    def expensive_function(x: int) -> int:
        time.sleep(0.01)  # 模拟耗时操作
        return x * x
    
    # 测试缓存
    for i in range(10):
        result = expensive_function(i % 3)  # 重复调用相同参数
    
    print("\n缓存管理演示:")
    print(cache_manager.get_stats())
    
    # 资源监控演示
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    monitor.log_stats()
