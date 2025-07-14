"""
模型管理异步版公共API

在原有功能基础上添加异步支持，提高模型调用并发性能。
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from ..core.exceptions import APIError, handle_module_error
from ..core.interfaces import ModuleInfo, ModuleType, PublicAPI
from .private.model_factory import ModelFactory, ModelCreationError
from .private.model_cache import ModelCacheManager
from .private.performance_tracker import PerformanceMonitor


class AsyncModelAPI(PublicAPI):
    """模型管理异步版公共API"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化异步模型API"""
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 核心组件
        self.model_factory = None
        self.cache_manager = None
        self.performance_monitor = None
        
        # 活跃模型实例（线程安全）
        self.active_models = {}
        self._model_lock = asyncio.Lock()
        
        # 异步执行器
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 8),
            thread_name_prefix="model"
        )
        
        # 状态管理
        self._initialized = False
        self._semaphore = None
        
        # API统计（线程安全）
        self.api_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "models_created": 0,
            "concurrent_requests": 0
        }
        self._stats_lock = asyncio.Lock()
    
    async def initialize_async(self) -> bool:
        """异步初始化模型管理模块"""
        try:
            self._logger.info("初始化异步模型管理模块...")
            
            # 创建信号量控制并发
            max_concurrent = self.config.get("max_concurrent_requests", 20)
            self._semaphore = asyncio.Semaphore(max_concurrent)
            
            # 在执行器中初始化组件（避免阻塞）
            loop = asyncio.get_event_loop()
            
            # 初始化模型工厂
            factory_config = self.config.get("factory", {})
            config_path = self.config.get("model_config_path")
            self.model_factory = await loop.run_in_executor(
                self.executor,
                lambda: ModelFactory(config_path)
            )
            
            # 初始化缓存管理器
            cache_config = self.config.get("cache", {})
            self.cache_manager = await loop.run_in_executor(
                self.executor,
                lambda: ModelCacheManager(cache_config)
            )
            
            # 初始化性能监控器
            monitor_config = self.config.get("performance", {})
            self.performance_monitor = await loop.run_in_executor(
                self.executor,
                lambda: PerformanceMonitor(monitor_config)
            )
            
            self._initialized = True
            self._logger.info("异步模型管理模块初始化完成")
            return True
            
        except Exception as e:
            error = handle_module_error(e, "models", "async_initialization")
            self._logger.error(f"异步模型管理模块初始化失败: {error}")
            raise error
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """同步初始化接口（保持兼容性）"""
        if config:
            self.config.update(config)
        
        # 运行异步初始化
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.initialize_async())
    
    async def solve_with_model_async(
        self, 
        model_name: str, 
        problem: Dict[str, Any], 
        model_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        异步使用指定模型解决问题
        
        Args:
            model_name: 模型名称
            problem: 问题数据
            model_config: 模型配置
            use_cache: 是否使用缓存
            
        Returns:
            模型求解结果
        """
        if not self._initialized:
            raise APIError("Async model management module not initialized", module_name="models")
        
        # 使用信号量控制并发
        async with self._semaphore:
            start_time = time.time()
            
            try:
                await self._update_concurrent_stats(1)
                self._validate_solve_input(model_name, problem)
                
                async with self._stats_lock:
                    self.api_stats["total_requests"] += 1
                
                # 检查缓存
                cached_result = None
                if use_cache:
                    cached_result = await self._get_cached_result_async(problem, model_name, model_config)
                    if cached_result:
                        async with self._stats_lock:
                            self.api_stats["cache_hits"] += 1
                        
                        self._logger.debug(f"异步缓存命中: {model_name}")
                        
                        # 记录性能（缓存命中）
                        end_time = time.time()
                        await self._monitor_performance_async(
                            model_name, "solve_cached", start_time, end_time, True,
                            len(str(problem)), len(str(cached_result))
                        )
                        
                        result = {
                            "final_answer": cached_result.get("final_answer"),
                            "confidence": cached_result.get("confidence", 0.0),
                            "cached": True,
                            "processing_time": end_time - start_time,
                            "model_name": model_name,
                            "async_mode": True
                        }
                        
                        async with self._stats_lock:
                            self.api_stats["successful_requests"] += 1
                        
                        return result
                    else:
                        async with self._stats_lock:
                            self.api_stats["cache_misses"] += 1
                
                # 获取或创建模型实例
                model_instance = await self._get_model_instance_async(model_name, model_config)
                
                # 异步调用模型求解
                model_result = await self._call_model_solve_async(model_instance, problem, model_name, start_time)
                
                # 缓存结果
                if use_cache and model_result.get("success", False):
                    await self._cache_result_async(problem, model_name, model_result, model_config)
                
                async with self._stats_lock:
                    self.api_stats["successful_requests"] += 1
                
                return model_result
                
            except Exception as e:
                end_time = time.time()
                async with self._stats_lock:
                    self.api_stats["failed_requests"] += 1
                
                # 记录性能（失败）
                await self._monitor_performance_async(
                    model_name, "solve_failed", start_time, end_time, False,
                    len(str(problem)), 0, str(e)
                )
                
                error = handle_module_error(e, "models", f"async_solve_with_model_{model_name}")
                self._logger.error(f"异步模型求解失败: {error}")
                
                return {
                    "final_answer": "求解失败",
                    "confidence": 0.0,
                    "success": False,
                    "error": str(e),
                    "processing_time": end_time - start_time,
                    "model_name": model_name,
                    "async_mode": True
                }
            finally:
                await self._update_concurrent_stats(-1)
    
    def solve_with_model(
        self, 
        model_name: str, 
        problem: Dict[str, Any], 
        model_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """同步解决问题接口（保持兼容性）"""
        if not self._initialized:
            raise APIError("Model management module not initialized", module_name="models")
        
        # 运行异步版本
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.solve_with_model_async(model_name, problem, model_config, use_cache)
        )
    
    async def _get_cached_result_async(
        self, 
        problem: Dict[str, Any], 
        model_name: str, 
        model_config: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """异步获取缓存结果"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.cache_manager.get_cached_model_result(problem, model_name, model_config)
        )
    
    async def _get_model_instance_async(self, model_name: str, model_config: Optional[Dict[str, Any]]):
        """异步获取模型实例"""
        
        # 生成模型实例键
        instance_key = f"{model_name}_{hash(str(model_config))}"
        
        # 使用锁保证线程安全
        async with self._model_lock:
            # 检查是否已存在实例
            if instance_key in self.active_models:
                return self.active_models[instance_key]
            
            # 在执行器中创建新实例
            try:
                loop = asyncio.get_event_loop()
                model_instance = await loop.run_in_executor(
                    self.executor,
                    lambda: self.model_factory.create_model(model_name, model_config)
                )
                
                self.active_models[instance_key] = model_instance
                async with self._stats_lock:
                    self.api_stats["models_created"] += 1
                
                self._logger.debug(f"异步创建模型实例: {model_name}")
                return model_instance
                
            except ModelCreationError as e:
                raise APIError(f"Failed to create model {model_name}: {str(e)}", module_name="models")
    
    async def _call_model_solve_async(
        self, 
        model_instance, 
        problem: Dict[str, Any], 
        model_name: str, 
        start_time: float
    ) -> Dict[str, Any]:
        """异步调用模型求解"""
        
        try:
            # 在执行器中运行模型求解
            loop = asyncio.get_event_loop()
            
            def solve_in_executor():
                # 准备模型输入
                if hasattr(model_instance, 'solve_problem'):
                    # 新版本接口
                    return model_instance.solve_problem(problem)
                elif hasattr(model_instance, 'solve'):
                    # 旧版本接口
                    problem_text = problem.get("problem") or problem.get("cleaned_text", "")
                    return model_instance.solve(problem_text)
                else:
                    raise APIError(f"Model {model_name} has no solve method", module_name="models")
            
            result = await loop.run_in_executor(self.executor, solve_in_executor)
            end_time = time.time()
            
            # 标准化结果格式
            standardized_result = self._standardize_model_result(result, model_name, start_time, end_time)
            standardized_result["async_mode"] = True
            
            # 记录性能
            await self._monitor_performance_async(
                model_name, "solve", start_time, end_time, 
                standardized_result.get("success", True),
                len(str(problem)), len(str(result))
            )
            
            return standardized_result
            
        except Exception as e:
            end_time = time.time()
            await self._monitor_performance_async(
                model_name, "solve", start_time, end_time, False,
                len(str(problem)), 0, str(e)
            )
            raise
    
    async def _cache_result_async(
        self, 
        problem: Dict[str, Any], 
        model_name: str, 
        result: Dict[str, Any], 
        model_config: Optional[Dict[str, Any]]
    ):
        """异步缓存结果"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            lambda: self.cache_manager.cache_model_result(problem, model_name, result, model_config)
        )
    
    async def _monitor_performance_async(
        self, 
        model_name: str, 
        operation: str, 
        start_time: float, 
        end_time: float, 
        success: bool,
        input_size: int, 
        output_size: int, 
        error: Optional[str] = None
    ):
        """异步记录性能"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            lambda: self.performance_monitor.monitor_model_call(
                model_name, operation, start_time, end_time, success, input_size, output_size, error
            )
        )
    
    async def batch_solve_async(
        self, 
        model_name: str, 
        problems: List[Dict[str, Any]], 
        model_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        异步批量求解问题
        
        Args:
            model_name: 模型名称
            problems: 问题列表
            model_config: 模型配置
            use_cache: 是否使用缓存
            max_concurrent: 最大并发数
            
        Returns:
            结果列表
        """
        try:
            if not self._initialized:
                raise APIError("Async model management module not initialized", module_name="models")
            
            # 设置并发限制
            if max_concurrent is None:
                max_concurrent = self.config.get("batch_max_concurrent", 10)
            
            batch_semaphore = asyncio.Semaphore(max_concurrent)
            
            self._logger.info(f"开始异步批量求解: {len(problems)}个问题，模型: {model_name}，并发数: {max_concurrent}")
            
            async def solve_with_semaphore(problem, index):
                async with batch_semaphore:
                    try:
                        result = await self.solve_with_model_async(model_name, problem, model_config, use_cache)
                        result["problem_index"] = index
                        return result
                    except Exception as e:
                        self._logger.warning(f"问题{index}异步处理失败: {e}")
                        return {
                            "problem_index": index,
                            "error": str(e),
                            "final_answer": "处理失败",
                            "confidence": 0.0,
                            "success": False,
                            "model_name": model_name,
                            "async_mode": True
                        }
            
            # 创建所有任务
            tasks = [solve_with_semaphore(problem, i) for i, problem in enumerate(problems)]
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "problem_index": i,
                        "error": str(result),
                        "final_answer": "处理失败",
                        "confidence": 0.0,
                        "success": False,
                        "model_name": model_name,
                        "async_mode": True
                    })
                else:
                    processed_results.append(result)
            
            self._logger.info(f"异步批量求解完成: {len(problems)}个问题, {len(processed_results)}个结果")
            return processed_results
            
        except Exception as e:
            error = handle_module_error(e, "models", "async_batch_solve")
            self._logger.error(f"异步批量求解失败: {error}")
            raise error
    
    def batch_solve(
        self, 
        model_name: str, 
        problems: List[Dict[str, Any]], 
        model_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """同步批量求解接口（保持兼容性）"""
        if not self._initialized:
            raise APIError("Model management module not initialized", module_name="models")
        
        # 运行异步版本
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.batch_solve_async(model_name, problems, model_config, use_cache, max_workers)
        )
    
    async def compare_models_async(
        self, 
        model_names: List[str], 
        problems: List[Dict[str, Any]],
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        异步模型性能比较
        
        Args:
            model_names: 模型名称列表
            problems: 测试问题列表
            model_configs: 各模型配置
            max_concurrent: 最大并发数
            
        Returns:
            比较结果
        """
        try:
            if not self._initialized:
                raise APIError("Async model management module not initialized", module_name="models")
            
            self._logger.info(f"开始异步模型比较: {len(model_names)}个模型，{len(problems)}个问题")
            
            comparison_results = {
                "models": model_names,
                "problem_count": len(problems),
                "results": {},
                "summary": {},
                "async_mode": True
            }
            
            # 并行为每个模型运行测试
            async def test_model(model_name):
                config = model_configs.get(model_name) if model_configs else None
                
                model_results = await self.batch_solve_async(
                    model_name, problems, config, use_cache=False, max_concurrent=max_concurrent
                )
                
                # 计算统计信息
                successful_results = [r for r in model_results if r.get("success", False)]
                
                model_summary = {
                    "total_problems": len(problems),
                    "successful_problems": len(successful_results),
                    "success_rate": len(successful_results) / len(problems),
                    "avg_processing_time": sum(r.get("processing_time", 0) for r in model_results) / len(model_results),
                    "avg_confidence": sum(r.get("confidence", 0) for r in successful_results) / len(successful_results) if successful_results else 0
                }
                
                return model_name, model_results, model_summary
            
            # 并行测试所有模型
            model_comparison_semaphore = asyncio.Semaphore(len(model_names))
            
            async def test_with_semaphore(model_name):
                async with model_comparison_semaphore:
                    return await test_model(model_name)
            
            tasks = [test_with_semaphore(model_name) for model_name in model_names]
            model_test_results = await asyncio.gather(*tasks)
            
            # 整理结果
            for model_name, model_results, model_summary in model_test_results:
                comparison_results["results"][model_name] = model_results
                comparison_results["summary"][model_name] = model_summary
            
            # 生成排名
            comparison_results["rankings"] = self._generate_model_rankings(comparison_results["summary"])
            
            self._logger.info("异步模型比较完成")
            return comparison_results
            
        except Exception as e:
            error = handle_module_error(e, "models", "async_compare_models")
            self._logger.error(f"异步模型比较失败: {error}")
            raise error
    
    def compare_models(
        self, 
        model_names: List[str], 
        problems: List[Dict[str, Any]],
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """同步模型比较接口（保持兼容性）"""
        if not self._initialized:
            raise APIError("Model management module not initialized", module_name="models")
        
        # 运行异步版本
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.compare_models_async(model_names, problems, model_configs)
        )
    
    def _standardize_model_result(self, result: Any, model_name: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """标准化模型结果格式（复用原有逻辑）"""
        
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
    
    def _generate_model_rankings(self, summary: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """生成模型排名（复用原有逻辑）"""
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
    
    def _validate_solve_input(self, model_name: str, problem: Dict[str, Any]):
        """验证求解输入"""
        if not isinstance(model_name, str) or not model_name.strip():
            raise APIError("Model name must be a non-empty string", module_name="models")
        
        if not isinstance(problem, dict):
            raise APIError("Problem must be a dictionary", module_name="models")
        
        problem_text = problem.get("problem") or problem.get("cleaned_text")
        if not problem_text:
            raise APIError("Problem must contain 'problem' or 'cleaned_text' field", module_name="models")
    
    async def _update_concurrent_stats(self, delta: int):
        """更新并发统计"""
        async with self._stats_lock:
            self.api_stats["concurrent_requests"] += delta
    
    def get_module_info(self) -> ModuleInfo:
        """获取模块信息"""
        return ModuleInfo(
            name="models",
            type=ModuleType.MODELS,
            version="2.1.0",  # 异步版本
            dependencies=[],
            public_api_class="AsyncModelAPI",
            orchestrator_class="ModelOrchestrator"
        )
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = {
                "status": "healthy" if self._initialized else "not_initialized",
                "initialized": self._initialized,
                "async_enabled": True,
                "max_workers": self.executor._max_workers if self.executor else 0,
                "max_concurrent": self._semaphore._value if self._semaphore else 0,
                "current_concurrent": self.api_stats.get("concurrent_requests", 0),
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
    
    async def get_statistics_async(self) -> Dict[str, Any]:
        """异步获取模块统计信息"""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}
            
            async with self._stats_lock:
                stats = self.api_stats.copy()
            
            # 计算成功率
            if stats["total_requests"] > 0:
                stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
                stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
            else:
                stats["success_rate"] = 0.0
                stats["cache_hit_rate"] = 0.0
            
            # 组件统计
            loop = asyncio.get_event_loop()
            component_stats = await loop.run_in_executor(
                self.executor,
                lambda: {
                    "factory": self.model_factory.get_creation_stats(),
                    "cache": self.cache_manager.get_cache_stats(),
                    "performance": self.performance_monitor.get_system_overview()
                }
            )
            stats["component_stats"] = component_stats
            
            return stats
            
        except Exception as e:
            self._logger.warning(f"获取异步统计信息失败: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """同步获取统计信息接口（保持兼容性）"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.get_statistics_async())
    
    # 其他方法保持同步版本的实现...
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
    
    async def shutdown_async(self) -> bool:
        """异步关闭模型管理模块"""
        try:
            # 清理活跃模型
            async with self._model_lock:
                self.active_models.clear()
            
            # 关闭执行器
            if self.executor:
                self.executor.shutdown(wait=True)
            
            # 关闭组件
            if self.cache_manager:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.cache_manager.shutdown)
            
            if self.performance_monitor:
                await loop.run_in_executor(self.executor, self.performance_monitor.shutdown)
            
            self._initialized = False
            self._logger.info("异步模型管理模块已关闭")
            return True
            
        except Exception as e:
            self._logger.error(f"异步模型管理模块关闭失败: {e}")
            return False
    
    def shutdown(self) -> None:
        """同步关闭接口（保持兼容性）"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self.shutdown_async())


# 创建全局实例
async_model_api = AsyncModelAPI()

# 保持向后兼容
model_api = async_model_api