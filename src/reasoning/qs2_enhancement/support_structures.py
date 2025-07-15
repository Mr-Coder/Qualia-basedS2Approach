"""
QS²增强系统支持的数据结构和模型
============================

为QS²增强的隐式关系发现系统提供额外的数据结构、
工具函数和配置模型。

核心功能：
1. 配置管理和验证
2. 数据序列化和反序列化
3. 批量处理工具
4. 性能监控和优化
"""

import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
from pathlib import Path

from .qualia_constructor import QualiaStructure
from .compatibility_engine import CompatibilityResult
from .enhanced_ird_engine import EnhancedRelation, DiscoveryResult, RelationType, RelationStrength

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class SerializationError(Exception):
    """序列化错误"""
    pass


@dataclass
class QS2Config:
    """QS²系统配置"""
    # 语义结构构建配置
    qualia_config: Dict[str, Any] = field(default_factory=dict)
    
    # 兼容性计算配置
    compatibility_config: Dict[str, Any] = field(default_factory=lambda: {
        "compatibility_weights": {
            "formal": 0.25,
            "telic": 0.35,
            "agentive": 0.15,
            "constitutive": 0.15,
            "contextual": 0.10
        },
        "compatibility_threshold": 0.6,
        "similarity_method": "jaccard"
    })
    
    # 关系发现配置
    discovery_config: Dict[str, Any] = field(default_factory=lambda: {
        "min_strength_threshold": 0.3,
        "max_relations_per_entity": 10,
        "enable_parallel_processing": True,
        "max_workers": 4,
        "relation_type_weights": {
            "semantic": 0.3,
            "functional": 0.35,
            "contextual": 0.15,
            "structural": 0.1,
            "quantitative": 0.1
        }
    })
    
    # 性能配置
    performance_config: Dict[str, Any] = field(default_factory=lambda: {
        "cache_enabled": True,
        "cache_size": 1000,
        "cache_ttl": 3600,
        "enable_monitoring": True,
        "log_level": "INFO"
    })
    
    # 输出配置
    output_config: Dict[str, Any] = field(default_factory=lambda: {
        "export_format": "json",
        "include_statistics": True,
        "include_evidence": True,
        "round_precision": 3
    })
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证权重配置
            comp_weights = self.compatibility_config.get("compatibility_weights", {})
            if abs(sum(comp_weights.values()) - 1.0) > 0.01:
                raise ConfigValidationError("兼容性权重总和必须为1.0")
            
            rel_weights = self.discovery_config.get("relation_type_weights", {})
            if abs(sum(rel_weights.values()) - 1.0) > 0.01:
                raise ConfigValidationError("关系类型权重总和必须为1.0")
            
            # 验证阈值
            if not (0.0 <= self.compatibility_config.get("compatibility_threshold", 0.6) <= 1.0):
                raise ConfigValidationError("兼容性阈值必须在[0.0, 1.0]范围内")
            
            if not (0.0 <= self.discovery_config.get("min_strength_threshold", 0.3) <= 1.0):
                raise ConfigValidationError("最小强度阈值必须在[0.0, 1.0]范围内")
            
            # 验证性能配置
            if self.performance_config.get("cache_size", 1000) <= 0:
                raise ConfigValidationError("缓存大小必须大于0")
            
            if self.performance_config.get("cache_ttl", 3600) <= 0:
                raise ConfigValidationError("缓存TTL必须大于0")
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {str(e)}")
            raise ConfigValidationError(f"配置验证失败: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QS2Config':
        """从字典创建配置"""
        return cls(**data)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """保存配置到文件"""
        file_path = Path(file_path)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            raise SerializationError(f"保存配置失败: {str(e)}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'QS2Config':
        """从文件加载配置"""
        file_path = Path(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config = cls.from_dict(data)
            config.validate()
            logger.info(f"配置已从文件加载: {file_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            raise SerializationError(f"加载配置失败: {str(e)}")


@dataclass
class ProcessingCache:
    """处理缓存"""
    cache_data: Dict[str, Any] = field(default_factory=dict)
    cache_timestamps: Dict[str, float] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    ttl: float = 3600  # 1小时
    max_size: int = 1000
    
    def _generate_key(self, *args) -> str:
        """生成缓存键"""
        key_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key not in self.cache_data:
            self.cache_misses += 1
            return None
        
        # 检查是否过期
        if time.time() - self.cache_timestamps[key] > self.ttl:
            del self.cache_data[key]
            del self.cache_timestamps[key]
            self.cache_misses += 1
            return None
        
        self.cache_hits += 1
        return self.cache_data[key]
    
    def set(self, key: str, value: Any):
        """设置缓存数据"""
        # 如果缓存已满，删除最旧的条目
        if len(self.cache_data) >= self.max_size:
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.cache_data[oldest_key]
            del self.cache_timestamps[oldest_key]
        
        self.cache_data[key] = value
        self.cache_timestamps[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        self.cache_data.clear()
        self.cache_timestamps.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache_data),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


@dataclass
class PerformanceMonitor:
    """性能监控器"""
    operation_times: Dict[str, List[float]] = field(default_factory=dict)
    operation_counts: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def record_operation(self, operation: str, duration: float):
        """记录操作时间"""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
    
    def record_error(self, operation: str):
        """记录错误"""
        if operation not in self.error_counts:
            self.error_counts[operation] = 0
        self.error_counts[operation] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            "uptime": time.time() - self.start_time,
            "operations": {}
        }
        
        for operation, times in self.operation_times.items():
            stats["operations"][operation] = {
                "count": self.operation_counts[operation],
                "total_time": sum(times),
                "average_time": sum(times) / len(times) if times else 0,
                "min_time": min(times) if times else 0,
                "max_time": max(times) if times else 0,
                "error_count": self.error_counts.get(operation, 0)
            }
        
        return stats
    
    def reset(self):
        """重置监控数据"""
        self.operation_times.clear()
        self.operation_counts.clear()
        self.error_counts.clear()
        self.start_time = time.time()


class DataSerializer:
    """数据序列化工具"""
    
    @staticmethod
    def serialize_qualia_structure(structure: QualiaStructure) -> Dict[str, Any]:
        """序列化语义结构"""
        return structure.to_dict()
    
    @staticmethod
    def deserialize_qualia_structure(data: Dict[str, Any]) -> QualiaStructure:
        """反序列化语义结构"""
        return QualiaStructure(
            entity=data["entity"],
            entity_type=data["entity_type"],
            formal_roles=data["qualia_roles"]["formal"],
            telic_roles=data["qualia_roles"]["telic"],
            agentive_roles=data["qualia_roles"]["agentive"],
            constitutive_roles=data["qualia_roles"]["constitutive"],
            context_features=data["context_features"],
            confidence=data["confidence"]
        )
    
    @staticmethod
    def serialize_enhanced_relation(relation: EnhancedRelation) -> Dict[str, Any]:
        """序列化增强关系"""
        return relation.to_dict()
    
    @staticmethod
    def deserialize_enhanced_relation(data: Dict[str, Any]) -> EnhancedRelation:
        """反序列化增强关系"""
        # 重建CompatibilityResult
        comp_data = data["compatibility_result"]
        compatibility_result = CompatibilityResult(
            entity1=comp_data["entity1"],
            entity2=comp_data["entity2"],
            overall_score=comp_data["overall_score"],
            detailed_scores=comp_data["detailed_scores"],
            compatibility_reasons=comp_data["compatibility_reasons"],
            incompatibility_reasons=comp_data["incompatibility_reasons"],
            confidence=comp_data["confidence"]
        )
        
        return EnhancedRelation(
            entity1=data["entity1"],
            entity2=data["entity2"],
            relation_type=RelationType(data["relation_type"]),
            strength=data["strength"],
            strength_level=RelationStrength(data["strength_level"]),
            compatibility_result=compatibility_result,
            evidence=data["evidence"],
            confidence=data["confidence"],
            discovery_method=data["discovery_method"],
            timestamp=data["timestamp"]
        )
    
    @staticmethod
    def serialize_discovery_result(result: DiscoveryResult) -> Dict[str, Any]:
        """序列化发现结果"""
        return result.to_dict()
    
    @staticmethod
    def deserialize_discovery_result(data: Dict[str, Any]) -> DiscoveryResult:
        """反序列化发现结果"""
        relations = [
            DataSerializer.deserialize_enhanced_relation(rel_data)
            for rel_data in data["relations"]
        ]
        
        return DiscoveryResult(
            relations=relations,
            processing_time=data["processing_time"],
            entity_count=data["entity_count"],
            total_pairs_evaluated=data["total_pairs_evaluated"],
            high_strength_relations=data["high_strength_relations"],
            statistics=data["statistics"]
        )
    
    @staticmethod
    def save_to_file(data: Any, file_path: Union[str, Path], format: str = "json"):
        """保存数据到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"数据已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            raise SerializationError(f"保存数据失败: {str(e)}")
    
    @staticmethod
    def load_from_file(file_path: Union[str, Path], format: str = "json") -> Any:
        """从文件加载数据"""
        file_path = Path(file_path)
        
        try:
            if format == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif format == "pickle":
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"数据已从文件加载: {file_path}")
            return data
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise SerializationError(f"加载数据失败: {str(e)}")


class BatchProcessor:
    """批量处理工具"""
    
    def __init__(self, config: QS2Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def process_problems_batch(
        self, 
        problems: List[Union[str, Dict[str, Any]]],
        enhanced_engine: 'EnhancedIRDEngine'
    ) -> List[DiscoveryResult]:
        """批量处理问题"""
        results = []
        
        for i, problem in enumerate(problems):
            try:
                self.logger.info(f"处理问题 {i+1}/{len(problems)}")
                result = enhanced_engine.discover_relations(problem)
                results.append(result)
            except Exception as e:
                self.logger.error(f"处理问题 {i+1} 失败: {str(e)}")
                # 创建空结果
                empty_result = DiscoveryResult(
                    relations=[],
                    processing_time=0.0,
                    entity_count=0,
                    total_pairs_evaluated=0,
                    high_strength_relations=0,
                    statistics={"error": str(e)}
                )
                results.append(empty_result)
        
        return results
    
    def aggregate_results(self, results: List[DiscoveryResult]) -> Dict[str, Any]:
        """聚合处理结果"""
        total_relations = sum(len(r.relations) for r in results)
        total_processing_time = sum(r.processing_time for r in results)
        total_entities = sum(r.entity_count for r in results)
        
        # 聚合统计信息
        aggregated_stats = {
            "total_problems": len(results),
            "total_relations": total_relations,
            "total_processing_time": total_processing_time,
            "total_entities": total_entities,
            "average_relations_per_problem": total_relations / len(results) if results else 0,
            "average_processing_time": total_processing_time / len(results) if results else 0,
            "relation_type_distribution": {},
            "strength_distribution": {}
        }
        
        # 聚合关系类型分布
        for result in results:
            for rel_type, count in result.statistics.get("relation_type_distribution", {}).items():
                aggregated_stats["relation_type_distribution"][rel_type] = \
                    aggregated_stats["relation_type_distribution"].get(rel_type, 0) + count
        
        # 聚合强度分布
        for result in results:
            for strength, count in result.statistics.get("strength_distribution", {}).items():
                aggregated_stats["strength_distribution"][strength] = \
                    aggregated_stats["strength_distribution"].get(strength, 0) + count
        
        return aggregated_stats
    
    def export_batch_results(
        self, 
        results: List[DiscoveryResult],
        output_dir: Union[str, Path],
        format: str = "json"
    ):
        """导出批量结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出单个结果
        for i, result in enumerate(results):
            file_path = output_dir / f"result_{i+1}.{format}"
            DataSerializer.save_to_file(
                DataSerializer.serialize_discovery_result(result),
                file_path,
                format
            )
        
        # 导出聚合统计
        aggregated_stats = self.aggregate_results(results)
        stats_file = output_dir / f"aggregated_stats.{format}"
        DataSerializer.save_to_file(aggregated_stats, stats_file, format)
        
        self.logger.info(f"批量结果已导出到: {output_dir}")


class ValidationUtils:
    """验证工具"""
    
    @staticmethod
    def validate_qualia_structure(structure: QualiaStructure) -> bool:
        """验证语义结构"""
        if not structure.entity or not structure.entity_type:
            return False
        
        if not (0.0 <= structure.confidence <= 1.0):
            return False
        
        return True
    
    @staticmethod
    def validate_enhanced_relation(relation: EnhancedRelation) -> bool:
        """验证增强关系"""
        if not relation.entity1 or not relation.entity2:
            return False
        
        if not (0.0 <= relation.strength <= 1.0):
            return False
        
        if not (0.0 <= relation.confidence <= 1.0):
            return False
        
        return True
    
    @staticmethod
    def validate_discovery_result(result: DiscoveryResult) -> bool:
        """验证发现结果"""
        if result.entity_count < 0 or result.total_pairs_evaluated < 0:
            return False
        
        if result.processing_time < 0:
            return False
        
        # 验证所有关系
        for relation in result.relations:
            if not ValidationUtils.validate_enhanced_relation(relation):
                return False
        
        return True


# 导出常用函数
def create_default_config() -> QS2Config:
    """创建默认配置"""
    return QS2Config()


def create_performance_monitor() -> PerformanceMonitor:
    """创建性能监控器"""
    return PerformanceMonitor()


def create_cache(ttl: float = 3600, max_size: int = 1000) -> ProcessingCache:
    """创建处理缓存"""
    return ProcessingCache(ttl=ttl, max_size=max_size)


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )