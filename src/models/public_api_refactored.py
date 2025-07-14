"""
模型管理重构版公共API

整合模型工厂、缓存管理和性能监控，提供统一的模型管理接口。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import APIError, handle_module_error
from ..core.interfaces import ModuleInfo, ModuleType, PublicAPI
from .private.model_factory import ModelFactory, ModelCreationError
from .private.model_cache import ModelCacheManager
from .private.performance_tracker import PerformanceMonitor


class ModelAPI(PublicAPI):
    """模型管理重构版公共API"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化模型API"""
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 核心组件
        self.model_factory = None
        self.cache_manager = None
        self.performance_monitor = None
        
        # 活跃模型实例
        self.active_models = {}
        
        # 状态管理
        self._initialized = False
        
        # API统计
        self.api_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "models_created": 0
        }
    
    def initialize(self) -> bool:
        """初始化模型管理模块"""
        try:
            self._logger.info("初始化模型管理模块...")
            
            # 初始化模型工厂
            factory_config = self.config.get("factory", {})
            config_path = self.config.get("model_config_path")
            self.model_factory = ModelFactory(config_path)
            
            # 初始化缓存管理器
            cache_config = self.config.get("cache", {})
            self.cache_manager = ModelCacheManager(cache_config)
            
            # 初始化性能监控器
            monitor_config = self.config.get("performance", {})
            self.performance_monitor = PerformanceMonitor(monitor_config)
            
            self._initialized = True
            self._logger.info("模型管理模块初始化完成")
            return True
            
        except Exception as e:
            error = handle_module_error(e, "models", "initialization")
            self._logger.error(f"模型管理模块初始化失败: {error}")
            raise error
    
    def get_module_info(self) -> ModuleInfo:
        """获取模块信息"""
        return ModuleInfo(
            name="models",
            type=ModuleType.MODEL_MANAGEMENT,
            version="2.0.0",  # 重构版本
            dependencies=[],
            public_api_class="ModelAPI",
            orchestrator_class="ModelOrchestrator"
        )
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = {
                "status": "healthy" if self._initialized else "not_initialized",
                "initialized": self._initialized,
                "components": {
                    "model_factory": self.model_factory is not None,
                    "cache_manager": self.cache_manager is not None,
                    "performance_monitor": self.performance_monitor is not None
                }
            }
            
            if self._initialized:
                # 检查各组件状态
                status["component_stats"] = {
                    "factory": self.model_factory.get_creation_stats(),
                    "cache": self.cache_manager.get_cache_stats(),
                    "performance": self.performance_monitor.get_system_overview()
                }
                
                # 活跃模型信息
                status["active_models"] = {
                    "count": len(self.active_models),
                    "models": list(self.active_models.keys())
                }
            
            return status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def solve_with_model(
        self, 
        model_name: str, 
        problem: Dict[str, Any], 
        model_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        使用指定模型解决问题
        
        Args:
            model_name: 模型名称
            problem: 问题数据
            model_config: 模型配置
            use_cache: 是否使用缓存
            
        Returns:
            模型求解结果
        """
        start_time = time.time()
        
        try:
            if not self._initialized:
                raise APIError("Model management module not initialized", module_name="models")
            
            self._validate_solve_input(model_name, problem)
            self.api_stats["total_requests"] += 1
            
            # 检查缓存
            cached_result = None
            if use_cache:
                cached_result = self.cache_manager.get_cached_model_result(
                    problem, model_name, model_config
                )
                if cached_result:
                    self.api_stats["cache_hits"] += 1
                    self._logger.debug(f"缓存命中: {model_name}")
                    
                    # 记录性能（缓存命中）
                    end_time = time.time()
                    self.performance_monitor.monitor_model_call(
                        model_name, "solve_cached", start_time, end_time, True,
                        len(str(problem)), len(str(cached_result))
                    )
                    
                    result = {
                        "final_answer": cached_result.get("final_answer"),
                        "confidence": cached_result.get("confidence", 0.0),
                        "cached": True,
                        "processing_time": end_time - start_time,
                        "model_name": model_name
                    }
                    
                    self.api_stats["successful_requests"] += 1
                    return result
                else:
                    self.api_stats["cache_misses"] += 1
            
            # 获取或创建模型实例
            model_instance = self._get_model_instance(model_name, model_config)
            
            # 调用模型求解
            model_result = self._call_model_solve(model_instance, problem, model_name, start_time)
            
            # 缓存结果
            if use_cache and model_result.get("success", False):
                self.cache_manager.cache_model_result(
                    problem, model_name, model_result, model_config
                )
            
            self.api_stats["successful_requests"] += 1
            return model_result
            
        except Exception as e:
            end_time = time.time()
            self.api_stats["failed_requests"] += 1
            
            # 记录性能（失败）
            self.performance_monitor.monitor_model_call(
                model_name, "solve_failed", start_time, end_time, False,
                len(str(problem)), 0, str(e)
            )
            
            error = handle_module_error(e, "models", f"solve_with_model_{model_name}")
            self._logger.error(f"模型求解失败: {error}")
            
            return {
                "final_answer": "求解失败",
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "processing_time": end_time - start_time,
                "model_name": model_name
            }
    
    def _get_model_instance(self, model_name: str, model_config: Optional[Dict[str, Any]]):
        """获取模型实例"""
        
        # 生成模型实例键
        instance_key = f"{model_name}_{hash(str(model_config))}"
        
        # 检查是否已存在实例
        if instance_key in self.active_models:
            return self.active_models[instance_key]
        
        # 创建新实例
        try:
            model_instance = self.model_factory.create_model(model_name, model_config)
            self.active_models[instance_key] = model_instance
            self.api_stats["models_created"] += 1
            
            self._logger.debug(f"创建模型实例: {model_name}")
            return model_instance
            
        except ModelCreationError as e:
            raise APIError(f"Failed to create model {model_name}: {str(e)}", module_name="models")
    
    def _call_model_solve(self, model_instance, problem: Dict[str, Any], model_name: str, start_time: float) -> Dict[str, Any]:
        """调用模型求解"""
        
        try:
            # 准备模型输入
            if hasattr(model_instance, 'solve_problem'):
                # 新版本接口
                result = model_instance.solve_problem(problem)
            elif hasattr(model_instance, 'solve'):
                # 旧版本接口
                problem_text = problem.get("problem") or problem.get("cleaned_text", "")
                result = model_instance.solve(problem_text)
            else:
                raise APIError(f"Model {model_name} has no solve method", module_name="models")
            
            end_time = time.time()
            
            # 标准化结果格式
            standardized_result = self._standardize_model_result(result, model_name, start_time, end_time)
            
            # 记录性能
            self.performance_monitor.monitor_model_call(
                model_name, "solve", start_time, end_time, 
                standardized_result.get("success", True),
                len(str(problem)), len(str(result))
            )
            
            return standardized_result
            
        except Exception as e:
            end_time = time.time()
            self.performance_monitor.monitor_model_call(
                model_name, "solve", start_time, end_time, False,
                len(str(problem)), 0, str(e)
            )
            raise
    
    def _standardize_model_result(self, result: Any, model_name: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """标准化模型结果格式"""
        
        if isinstance(result, dict):
            # 如果已经是字典格式，进行标准化
            standardized = {
                "final_answer": result.get("final_answer") or result.get("answer") or result.get("result"),
                "confidence": result.get("confidence", 0.8),
                "success": result.get("success", True),
                "reasoning_steps": result.get("reasoning_steps", []),
                "cached": False,
                "processing_time": end_time - start_time,
                "model_name": model_name
            }
            
            # 保留其他字段
            for key, value in result.items():
                if key not in standardized:
                    standardized[key] = value
            
            return standardized
        
        else:
            # 如果是简单值，包装为标准格式
            return {
                "final_answer": str(result),
                "confidence": 0.8,
                "success": True,
                "reasoning_steps": [],
                "cached": False,
                "processing_time": end_time - start_time,
                "model_name": model_name
            }
    
    def batch_solve(
        self, 
        model_name: str, 
        problems: List[Dict[str, Any]], 
        model_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        批量求解问题
        
        Args:
            model_name: 模型名称
            problems: 问题列表
            model_config: 模型配置
            use_cache: 是否使用缓存
            max_workers: 最大并发数
            
        Returns:
            结果列表
        """
        try:
            if not self._initialized:
                raise APIError("Model management module not initialized", module_name="models")
            
            self._logger.info(f"开始批量求解: {len(problems)}个问题，模型: {model_name}")
            
            results = []
            
            # 简化版批量处理（串行）
            for i, problem in enumerate(problems):
                try:
                    result = self.solve_with_model(model_name, problem, model_config, use_cache)
                    result["problem_index"] = i
                    results.append(result)
                except Exception as e:
                    self._logger.warning(f"问题{i}处理失败: {e}")
                    results.append({
                        "problem_index": i,
                        "error": str(e),
                        "final_answer": "处理失败",
                        "confidence": 0.0,
                        "success": False,
                        "model_name": model_name
                    })
            
            self._logger.info(f"批量求解完成: {len(results)}个结果")
            return results
            
        except Exception as e:
            error = handle_module_error(e, "models", "batch_solve")
            self._logger.error(f"批量求解失败: {error}")
            raise error
    
    def compare_models(
        self, 
        model_names: List[str], 
        problems: List[Dict[str, Any]],
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        模型性能比较
        
        Args:
            model_names: 模型名称列表
            problems: 测试问题列表
            model_configs: 各模型配置
            
        Returns:
            比较结果
        """
        try:
            if not self._initialized:
                raise APIError("Model management module not initialized", module_name="models")
            
            self._logger.info(f"开始模型比较: {len(model_names)}个模型，{len(problems)}个问题")
            
            comparison_results = {
                "models": model_names,
                "problem_count": len(problems),
                "results": {},
                "summary": {}
            }
            
            # 为每个模型运行测试
            for model_name in model_names:
                config = model_configs.get(model_name) if model_configs else None
                
                model_results = self.batch_solve(model_name, problems, config, use_cache=False)
                
                # 计算统计信息
                successful_results = [r for r in model_results if r.get("success", False)]
                
                model_summary = {
                    "total_problems": len(problems),
                    "successful_problems": len(successful_results),
                    "success_rate": len(successful_results) / len(problems),
                    "avg_processing_time": sum(r.get("processing_time", 0) for r in model_results) / len(model_results),
                    "avg_confidence": sum(r.get("confidence", 0) for r in successful_results) / len(successful_results) if successful_results else 0
                }
                
                comparison_results["results"][model_name] = model_results
                comparison_results["summary"][model_name] = model_summary
            
            # 生成排名
            comparison_results["rankings"] = self._generate_model_rankings(comparison_results["summary"])
            
            self._logger.info("模型比较完成")
            return comparison_results
            
        except Exception as e:
            error = handle_module_error(e, "models", "compare_models")
            self._logger.error(f"模型比较失败: {error}")
            raise error
    
    def _generate_model_rankings(self, summary: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """生成模型排名"""
        rankings = {}
        
        # 按成功率排名
        rankings["by_success_rate"] = sorted(
            summary.keys(), 
            key=lambda x: summary[x]["success_rate"], 
            reverse=True
        )
        
        # 按速度排名
        rankings["by_speed"] = sorted(
            summary.keys(),
            key=lambda x: summary[x]["avg_processing_time"]
        )
        
        # 按置信度排名
        rankings["by_confidence"] = sorted(
            summary.keys(),
            key=lambda x: summary[x]["avg_confidence"],
            reverse=True
        )
        
        return rankings
    
    def get_available_models(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        try:
            if not self._initialized:
                return {"error": "Module not initialized"}
            
            return self.model_factory.get_available_models()
            
        except Exception as e:
            self._logger.error(f"获取可用模型失败: {e}")
            return {"error": str(e)}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        try:
            if not self._initialized:
                return {"error": "Module not initialized"}
            
            model_info = self.model_factory.get_model_info(model_name)
            if not model_info:
                return {"error": f"Model {model_name} not found"}
            
            # 添加性能统计
            performance_info = self.performance_monitor.tracker.get_model_performance(model_name)
            if "error" not in performance_info:
                model_info["performance"] = performance_info
            
            return model_info
            
        except Exception as e:
            self._logger.error(f"获取模型信息失败: {e}")
            return {"error": str(e)}
    
    def get_performance_report(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            if not self._initialized:
                return {"error": "Module not initialized"}
            
            return self.performance_monitor.get_performance_report(model_name)
            
        except Exception as e:
            self._logger.error(f"获取性能报告失败: {e}")
            return {"error": str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        try:
            if not self._initialized:
                return {"error": "Module not initialized"}
            
            return self.cache_manager.get_cache_stats()
            
        except Exception as e:
            self._logger.error(f"获取缓存统计失败: {e}")
            return {"error": str(e)}
    
    def clear_cache(self, model_name: Optional[str] = None):
        """清空缓存"""
        try:
            if not self._initialized:
                raise APIError("Module not initialized", module_name="models")
            
            if model_name:
                # TODO: 实现特定模型的缓存清理
                self._logger.warning("特定模型缓存清理功能尚未实现")
            else:
                self.cache_manager.clear()
            
            self._logger.info("缓存已清理")
            
        except Exception as e:
            error = handle_module_error(e, "models", "clear_cache")
            self._logger.error(f"清理缓存失败: {error}")
            raise error
    
    def optimize_cache(self):
        """优化缓存"""
        try:
            if not self._initialized:
                raise APIError("Module not initialized", module_name="models")
            
            self.cache_manager.optimize_cache()
            self._logger.info("缓存优化完成")
            
        except Exception as e:
            error = handle_module_error(e, "models", "optimize_cache")
            self._logger.error(f"缓存优化失败: {error}")
            raise error
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取模块统计信息"""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}
            
            stats = self.api_stats.copy()
            
            # 计算成功率
            if stats["total_requests"] > 0:
                stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
                stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
            else:
                stats["success_rate"] = 0.0
                stats["cache_hit_rate"] = 0.0
            
            # 组件统计
            stats["component_stats"] = {
                "factory": self.model_factory.get_creation_stats(),
                "cache": self.cache_manager.get_cache_stats(),
                "performance": self.performance_monitor.get_system_overview()
            }
            
            return stats
            
        except Exception as e:
            self._logger.warning(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def reset_statistics(self):
        """重置统计信息"""
        try:
            self.api_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "models_created": 0
            }
            
            if self._initialized:
                self.model_factory.reset_stats()
                self.cache_manager.reset_stats()
                self.performance_monitor.reset_metrics()
            
            self._logger.info("模型管理模块统计信息已重置")
            
        except Exception as e:
            self._logger.error(f"重置统计信息失败: {e}")
    
    def _validate_solve_input(self, model_name: str, problem: Dict[str, Any]):
        """验证求解输入"""
        if not isinstance(model_name, str) or not model_name.strip():
            raise APIError("Model name must be a non-empty string", module_name="models")
        
        if not isinstance(problem, dict):
            raise APIError("Problem must be a dictionary", module_name="models")
        
        problem_text = problem.get("problem") or problem.get("cleaned_text")
        if not problem_text:
            raise APIError("Problem must contain 'problem' or 'cleaned_text' field", module_name="models")
    
    def shutdown(self) -> bool:
        """关闭模型管理模块"""
        try:
            # 清理活跃模型
            self.active_models.clear()
            
            # 关闭组件
            if self.cache_manager:
                self.cache_manager.shutdown()
            
            if self.performance_monitor:
                self.performance_monitor.shutdown()
            
            self._initialized = False
            self._logger.info("模型管理模块已关闭")
            return True
            
        except Exception as e:
            self._logger.error(f"模型管理模块关闭失败: {e}")
            return False