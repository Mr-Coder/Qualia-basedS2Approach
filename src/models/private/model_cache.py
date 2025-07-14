"""
模型缓存管理器 (Model Cache Manager)

专注于模型结果的缓存、性能优化和内存管理。
"""

import hashlib
import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import threading
from collections import OrderedDict


class CacheEntry:
    """缓存条目"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.ttl = ttl  # Time to live in seconds
        self.size = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """计算值的大小（字节）"""
        try:
            return len(pickle.dumps(value))
        except:
            # 如果无法序列化，使用字符串长度估算
            return len(str(value)) * 2  # Unicode字符平均2字节
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def touch(self):
        """更新访问时间和计数"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "size": self.size,
            "ttl": self.ttl,
            "expired": self.is_expired()
        }


class ModelCacheManager:
    """模型缓存管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化缓存管理器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 配置参数
        config = config or {}
        self.max_size = config.get("max_size", 1000)  # 最大缓存条目数
        self.max_memory_mb = config.get("max_memory_mb", 512)  # 最大内存使用MB
        self.default_ttl = config.get("default_ttl", 3600)  # 默认TTL（秒）
        self.enable_persistence = config.get("enable_persistence", False)
        self.persist_path = config.get("persist_path", "cache/model_cache.pkl")
        self.cleanup_interval = config.get("cleanup_interval", 300)  # 清理间隔（秒）
        
        # 缓存存储
        self._cache = OrderedDict()  # 使用OrderedDict实现LRU
        self._lock = threading.RLock()  # 线程安全锁
        
        # 统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
            "current_size": 0,
            "current_memory_mb": 0.0,
            "cleanup_runs": 0
        }
        
        # 最后清理时间
        self._last_cleanup = time.time()
        
        # 加载持久化缓存
        if self.enable_persistence:
            self._load_cache()
        
        self.logger.info(f"模型缓存管理器初始化完成，最大条目: {self.max_size}, 最大内存: {self.max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，如果不存在或过期则返回None
        """
        with self._lock:
            self.stats["total_requests"] += 1
            
            if key not in self._cache:
                self.stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                self._remove_entry(key)
                self.stats["misses"] += 1
                return None
            
            # 更新访问信息
            entry.touch()
            
            # 移动到最后（LRU）
            self._cache.move_to_end(key)
            
            self.stats["hits"] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        存储缓存值
        
        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 生存时间（秒），None使用默认值
            
        Returns:
            是否成功存储
        """
        with self._lock:
            try:
                # 使用默认TTL
                if ttl is None:
                    ttl = self.default_ttl
                
                # 创建缓存条目
                entry = CacheEntry(key, value, ttl)
                
                # 检查内存限制
                if not self._check_memory_limit(entry):
                    self.logger.warning(f"缓存条目太大，跳过: {key}")
                    return False
                
                # 如果键已存在，更新
                if key in self._cache:
                    old_entry = self._cache[key]
                    self.stats["current_memory_mb"] -= old_entry.size / (1024 * 1024)
                
                # 存储条目
                self._cache[key] = entry
                self.stats["current_memory_mb"] += entry.size / (1024 * 1024)
                
                # 移动到最后
                self._cache.move_to_end(key)
                
                # 检查是否需要清理
                self._enforce_limits()
                
                # 更新统计
                self.stats["current_size"] = len(self._cache)
                
                return True
                
            except Exception as e:
                self.logger.error(f"存储缓存失败: {key}, 错误: {str(e)}")
                return False
    
    def remove(self, key: str) -> bool:
        """删除缓存条目"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self.stats["current_size"] = 0
            self.stats["current_memory_mb"] = 0.0
            self.logger.info("缓存已清空")
    
    def get_problem_hash(self, problem: Dict[str, Any], model_name: str, config: Dict[str, Any] = None) -> str:
        """
        生成问题的哈希键
        
        Args:
            problem: 问题数据
            model_name: 模型名称
            config: 模型配置
            
        Returns:
            哈希键
        """
        # 构建哈希数据
        hash_data = {
            "problem_text": problem.get("problem") or problem.get("cleaned_text", ""),
            "model_name": model_name,
            "config": config or {}
        }
        
        # 序列化并生成哈希
        hash_str = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(hash_str.encode('utf-8')).hexdigest()
    
    def cache_model_result(
        self, 
        problem: Dict[str, Any], 
        model_name: str, 
        result: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ) -> bool:
        """
        缓存模型结果
        
        Args:
            problem: 问题数据
            model_name: 模型名称
            result: 模型结果
            model_config: 模型配置
            ttl: 生存时间
            
        Returns:
            是否成功缓存
        """
        cache_key = self.get_problem_hash(problem, model_name, model_config)
        
        # 添加元数据
        cache_value = {
            "result": result,
            "model_name": model_name,
            "cached_at": time.time(),
            "problem_hash": cache_key
        }
        
        return self.put(cache_key, cache_value, ttl)
    
    def get_cached_model_result(
        self, 
        problem: Dict[str, Any], 
        model_name: str,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存的模型结果
        
        Args:
            problem: 问题数据
            model_name: 模型名称
            model_config: 模型配置
            
        Returns:
            缓存的结果，如果不存在则返回None
        """
        cache_key = self.get_problem_hash(problem, model_name, model_config)
        cached_value = self.get(cache_key)
        
        if cached_value and isinstance(cached_value, dict):
            return cached_value.get("result")
        
        return None
    
    def _check_memory_limit(self, entry: CacheEntry) -> bool:
        """检查内存限制"""
        entry_size_mb = entry.size / (1024 * 1024)
        
        # 单个条目不能超过最大内存的20%
        if entry_size_mb > self.max_memory_mb * 0.2:
            return False
        
        return True
    
    def _enforce_limits(self):
        """强制执行缓存限制"""
        
        # 清理过期条目
        self._cleanup_expired()
        
        # 强制执行大小限制（LRU）
        while len(self._cache) > self.max_size:
            self._evict_lru()
        
        # 强制执行内存限制（LRU）
        while self.stats["current_memory_mb"] > self.max_memory_mb:
            if not self._evict_lru():
                break  # 没有更多条目可以清理
    
    def _cleanup_expired(self):
        """清理过期条目"""
        current_time = time.time()
        
        # 检查是否需要清理
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        self._last_cleanup = current_time
        self.stats["cleanup_runs"] += 1
        
        if expired_keys:
            self.logger.debug(f"清理了{len(expired_keys)}个过期条目")
    
    def _evict_lru(self) -> bool:
        """驱逐最少使用的条目"""
        if not self._cache:
            return False
        
        # OrderedDict的第一个条目是最少使用的
        lru_key = next(iter(self._cache))
        self._remove_entry(lru_key)
        self.stats["evictions"] += 1
        
        return True
    
    def _remove_entry(self, key: str):
        """删除缓存条目"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats["current_memory_mb"] -= entry.size / (1024 * 1024)
            self.stats["current_size"] = len(self._cache)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            stats = self.stats.copy()
            
            # 计算命中率
            if stats["total_requests"] > 0:
                stats["hit_rate"] = stats["hits"] / stats["total_requests"]
                stats["miss_rate"] = stats["misses"] / stats["total_requests"]
            else:
                stats["hit_rate"] = 0.0
                stats["miss_rate"] = 0.0
            
            # 内存使用率
            stats["memory_usage_rate"] = stats["current_memory_mb"] / self.max_memory_mb
            
            # 缓存使用率
            stats["cache_usage_rate"] = stats["current_size"] / self.max_size
            
            return stats
    
    def get_cache_entries_info(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取缓存条目信息"""
        with self._lock:
            entries_info = []
            
            for key, entry in list(self._cache.items())[:limit]:
                entries_info.append(entry.to_dict())
            
            return entries_info
    
    def optimize_cache(self):
        """优化缓存"""
        with self._lock:
            self.logger.info("开始缓存优化...")
            
            # 清理过期条目
            self._cleanup_expired()
            
            # 如果内存使用率超过80%，清理低频访问的条目
            if self.stats["current_memory_mb"] / self.max_memory_mb > 0.8:
                self._cleanup_low_frequency_entries()
            
            self.logger.info("缓存优化完成")
    
    def _cleanup_low_frequency_entries(self):
        """清理低频访问的条目"""
        
        # 计算平均访问次数
        if not self._cache:
            return
        
        total_access = sum(entry.access_count for entry in self._cache.values())
        avg_access = total_access / len(self._cache)
        
        # 找出访问次数低于平均值一半的条目
        low_freq_keys = []
        for key, entry in self._cache.items():
            if entry.access_count < avg_access * 0.5:
                low_freq_keys.append(key)
        
        # 删除这些条目
        for key in low_freq_keys[:len(low_freq_keys)//2]:  # 只删除一半
            self._remove_entry(key)
        
        if low_freq_keys:
            self.logger.debug(f"清理了{len(low_freq_keys)}个低频访问条目")
    
    def _save_cache(self):
        """保存缓存到文件"""
        if not self.enable_persistence:
            return
        
        try:
            cache_path = Path(self.persist_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 只保存未过期的条目
            valid_cache = {}
            for key, entry in self._cache.items():
                if not entry.is_expired():
                    valid_cache[key] = entry
            
            with open(cache_path, 'wb') as f:
                pickle.dump(valid_cache, f)
            
            self.logger.info(f"缓存已保存到 {cache_path}")
            
        except Exception as e:
            self.logger.error(f"保存缓存失败: {str(e)}")
    
    def _load_cache(self):
        """从文件加载缓存"""
        try:
            cache_path = Path(self.persist_path)
            if not cache_path.exists():
                return
            
            with open(cache_path, 'rb') as f:
                saved_cache = pickle.load(f)
            
            # 恢复有效的缓存条目
            valid_count = 0
            for key, entry in saved_cache.items():
                if not entry.is_expired():
                    self._cache[key] = entry
                    self.stats["current_memory_mb"] += entry.size / (1024 * 1024)
                    valid_count += 1
            
            self.stats["current_size"] = len(self._cache)
            
            self.logger.info(f"从缓存文件加载了{valid_count}个有效条目")
            
        except Exception as e:
            self.logger.error(f"加载缓存失败: {str(e)}")
    
    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_requests": 0,
                "current_size": len(self._cache),
                "current_memory_mb": self.stats["current_memory_mb"],  # 保持当前内存使用
                "cleanup_runs": 0
            }
            self.logger.info("缓存统计信息已重置")
    
    def shutdown(self):
        """关闭缓存管理器"""
        with self._lock:
            if self.enable_persistence:
                self._save_cache()
            
            self.logger.info("模型缓存管理器已关闭")