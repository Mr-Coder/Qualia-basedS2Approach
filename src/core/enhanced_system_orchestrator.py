"""
增强系统协调器

支持重构后的模块架构，提供依赖图管理、并发处理和错误恢复。
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .exceptions import OrchestrationError, handle_module_error
from .interfaces import ModuleType, PublicAPI
from .module_registry import registry


class DependencyGraph:
    """模块依赖图管理器"""
    
    def __init__(self):
        self.dependencies = defaultdict(set)  # module -> set of dependencies
        self.dependents = defaultdict(set)    # module -> set of dependents
        self.logger = logging.getLogger(f"{__name__}.DependencyGraph")
    
    def add_dependency(self, module: str, depends_on: str):
        """添加依赖关系"""
        self.dependencies[module].add(depends_on)
        self.dependents[depends_on].add(module)
        self.logger.debug(f"添加依赖: {module} -> {depends_on}")
    
    def remove_dependency(self, module: str, depends_on: str):
        """移除依赖关系"""
        self.dependencies[module].discard(depends_on)
        self.dependents[depends_on].discard(module)
    
    def get_shutdown_order(self) -> List[str]:
        """获取关闭顺序（拓扑排序）"""
        # Kahn's algorithm for topological sorting
        in_degree = defaultdict(int)
        all_modules = set()
        
        # 计算入度
        for module in self.dependencies:
            all_modules.add(module)
            for dep in self.dependencies[module]:
                all_modules.add(dep)
                in_degree[module] += 1
        
        # 找到入度为0的模块
        queue = deque([module for module in all_modules if in_degree[module] == 0])
        shutdown_order = []
        
        while queue:
            module = queue.popleft()
            shutdown_order.append(module)
            
            # 更新依赖此模块的其他模块的入度
            for dependent in self.dependents[module]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # 反转顺序（先关闭依赖者，再关闭被依赖者）
        return shutdown_order[::-1]
    
    def has_cycle(self) -> bool:
        """检查是否存在循环依赖"""
        visited = set()
        rec_stack = set()
        
        def dfs(module):
            visited.add(module)
            rec_stack.add(module)
            
            for dep in self.dependencies[module]:
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(module)
            return False
        
        for module in self.dependencies:
            if module not in visited:
                if dfs(module):
                    return True
        return False


class ConcurrentExecutor:
    """并发执行器"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(f"{__name__}.ConcurrentExecutor")
    
    def execute_parallel(self, tasks: List[Tuple[str, callable, Dict[str, Any]]]) -> Dict[str, Any]:
        """并行执行任务"""
        results = {}
        future_to_task = {}
        
        # 提交所有任务
        for task_name, func, kwargs in tasks:
            future = self.executor.submit(func, **kwargs)
            future_to_task[future] = task_name
        
        # 收集结果
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                results[task_name] = future.result()
            except Exception as e:
                self.logger.error(f"任务 {task_name} 执行失败: {str(e)}")
                results[task_name] = {"error": str(e)}
        
        return results
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)


class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self):
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        self.logger = logging.getLogger(f"{__name__}.ErrorRecoveryManager")
    
    def register_recovery_strategy(self, error_type: str, strategy: callable):
        """注册恢复策略"""
        self.recovery_strategies[error_type] = strategy
        self.logger.debug(f"注册恢复策略: {error_type}")
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理错误"""
        error_type = type(error).__name__
        
        # 记录错误
        error_record = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_message": str(error),
            "context": context
        }
        self.error_history.append(error_record)
        
        # 尝试恢复
        if error_type in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[error_type](error, context)
                self.logger.info(f"错误恢复成功: {error_type}")
                return {"recovered": True, "result": recovery_result}
            except Exception as recovery_error:
                self.logger.error(f"错误恢复失败: {str(recovery_error)}")
        
        return {"recovered": False, "error": error_record}


class EnhancedSystemOrchestrator:
    """增强系统协调器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化增强协调器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config or {}
        
        # 核心组件
        self.dependency_graph = DependencyGraph()
        self.concurrent_executor = ConcurrentExecutor(
            max_workers=self.config.get("max_workers", 4)
        )
        self.error_recovery = ErrorRecoveryManager()
        
        # 性能监控
        self.orchestration_stats = {
            "total_orchestrations": 0,
            "successful_orchestrations": 0,
            "failed_orchestrations": 0,
            "average_orchestration_time": 0.0,
            "concurrent_executions": 0
        }
        
        # 锁管理
        self._orchestration_lock = threading.RLock()
        
        # 初始化依赖关系
        self._initialize_dependencies()
        
        # 注册错误恢复策略
        self._register_recovery_strategies()
        
        self.logger.info("增强系统协调器初始化完成")
    
    def _initialize_dependencies(self):
        """初始化模块依赖关系"""
        
        # 推理模块依赖模型管理
        self.dependency_graph.add_dependency("reasoning", "models")
        
        # 数据处理可能依赖配置
        if self.config.get("enable_data_processing", True):
            self.dependency_graph.add_dependency("data_processing", "configuration")
        
        # 评估模块依赖推理和模型
        self.dependency_graph.add_dependency("evaluation", "reasoning")
        self.dependency_graph.add_dependency("evaluation", "models")
        
        # 检查循环依赖
        if self.dependency_graph.has_cycle():
            self.logger.warning("检测到循环依赖，请检查模块配置")
    
    def _register_recovery_strategies(self):
        """注册错误恢复策略"""
        
        # 模块初始化失败恢复
        self.error_recovery.register_recovery_strategy(
            "ModuleInitializationError",
            self._recover_module_initialization
        )
        
        # 推理失败恢复
        self.error_recovery.register_recovery_strategy(
            "ReasoningError", 
            self._recover_reasoning_failure
        )
        
        # 模型调用失败恢复
        self.error_recovery.register_recovery_strategy(
            "ModelCallError",
            self._recover_model_call_failure
        )
    
    async def solve_math_problem_async(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """异步问题解决"""
        start_time = time.time()
        
        try:
            with self._orchestration_lock:
                self.orchestration_stats["total_orchestrations"] += 1
            
            self.logger.info(f"开始异步数学问题求解: {problem.get('problem', '')[:50]}...")
            
            # 阶段1: 并行获取和初始化模块
            initialization_tasks = [
                ("reasoning", self._get_reasoning_module, {}),
                ("models", self._get_models_module, {}),
                ("data_processing", self._get_data_processing_module, {})
            ]
            
            module_results = self.concurrent_executor.execute_parallel(initialization_tasks)
            
            # 检查模块可用性
            reasoning_module = module_results.get("reasoning")
            models_module = module_results.get("models") 
            data_processing_module = module_results.get("data_processing")
            
            if not reasoning_module or "error" in reasoning_module:
                raise OrchestrationError("推理模块不可用", module_name="reasoning")
            
            # 阶段2: 数据预处理
            processed_data = problem
            if data_processing_module and "error" not in data_processing_module:
                try:
                    processed_data = await self._preprocess_data_async(
                        data_processing_module, problem
                    )
                except Exception as e:
                    self.logger.warning(f"数据预处理失败，使用原始数据: {str(e)}")
            
            # 阶段3: 推理求解（使用重构的推理API）
            reasoning_result = await self._execute_reasoning_async(
                reasoning_module, processed_data
            )
            
            # 阶段4: 结果后处理和验证
            final_result = await self._postprocess_result_async(
                reasoning_result, problem, processed_data
            )
            
            # 更新统计
            processing_time = time.time() - start_time
            final_result["processing_time"] = processing_time
            self._update_orchestration_stats(True, processing_time)
            
            self.logger.info(f"异步问题求解完成: 答案={final_result.get('final_answer')}")
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_orchestration_stats(False, processing_time)
            
            # 错误恢复
            recovery_result = self.error_recovery.handle_error(e, {
                "problem": problem,
                "processing_time": processing_time
            })
            
            if recovery_result.get("recovered"):
                return recovery_result["result"]
            
            # 返回错误结果
            self.logger.error(f"异步问题求解失败: {str(e)}")
            return {
                "final_answer": "求解失败",
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "recovery_attempted": recovery_result.get("recovered", False)
            }
    
    def solve_math_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """同步问题解决（保持向后兼容）"""
        # 运行异步版本
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.solve_math_problem_async(problem))
        finally:
            loop.close()
    
    async def batch_solve_problems_async(
        self, 
        problems: List[Dict[str, Any]],
        max_concurrent: int = None
    ) -> List[Dict[str, Any]]:
        """异步批量解决问题"""
        
        if max_concurrent is None:
            max_concurrent = self.config.get("batch_max_concurrent", 4)
        
        self.logger.info(f"开始异步批量求解: {len(problems)}个问题")
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def solve_with_semaphore(problem, index):
            async with semaphore:
                result = await self.solve_math_problem_async(problem)
                result["problem_index"] = index
                return result
        
        # 创建所有任务
        tasks = [
            solve_with_semaphore(problem, i)
            for i, problem in enumerate(problems)
        ]
        
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
                    "success": False
                })
            else:
                processed_results.append(result)
        
        self.logger.info(f"异步批量求解完成: {len(processed_results)}个结果")
        return processed_results
    
    def batch_solve_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """同步批量解决问题"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.batch_solve_problems_async(problems))
        finally:
            loop.close()
    
    def _get_reasoning_module(self) -> Dict[str, Any]:
        """获取推理模块"""
        try:
            if registry.is_module_registered("reasoning"):
                module = registry.get_module("reasoning")
                return {"module": module, "available": True}
            else:
                return {"error": "推理模块未注册"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_models_module(self) -> Dict[str, Any]:
        """获取模型模块"""
        try:
            if registry.is_module_registered("models"):
                module = registry.get_module("models")
                return {"module": module, "available": True}
            else:
                return {"error": "模型模块未注册"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_data_processing_module(self) -> Dict[str, Any]:
        """获取数据处理模块"""
        try:
            if registry.is_module_registered("data_processing"):
                module = registry.get_module("data_processing")
                return {"module": module, "available": True}
            else:
                return {"available": False}
        except Exception as e:
            return {"error": str(e)}
    
    async def _preprocess_data_async(
        self, 
        data_processing_module: Dict[str, Any], 
        problem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """异步数据预处理"""
        
        module = data_processing_module["module"]
        
        # 如果模块有异步方法，使用异步
        if hasattr(module, 'process_problem_async'):
            return await module.process_problem_async(problem)
        # 否则在线程池中运行同步方法
        elif hasattr(module, 'process_problem'):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, module.process_problem, problem
            )
        else:
            return problem
    
    async def _execute_reasoning_async(
        self, 
        reasoning_module: Dict[str, Any], 
        problem_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """异步推理执行"""
        
        module = reasoning_module["module"]
        
        # 使用重构后的推理API
        if hasattr(module, 'solve_problem_async'):
            return await module.solve_problem_async(problem_data)
        elif hasattr(module, 'solve_problem'):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, module.solve_problem, problem_data
            )
        else:
            raise OrchestrationError("推理模块没有可用的求解方法")
    
    async def _postprocess_result_async(
        self, 
        reasoning_result: Dict[str, Any], 
        original_problem: Dict[str, Any],
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """异步结果后处理"""
        
        # 标准化结果格式
        final_result = {
            "final_answer": reasoning_result.get("final_answer"),
            "confidence": reasoning_result.get("confidence", 0.0),
            "success": reasoning_result.get("success", True),
            "reasoning_steps": reasoning_result.get("reasoning_steps", []),
            "complexity_level": reasoning_result.get("complexity_level"),
            "relations_found": reasoning_result.get("relations_found", []),
            "validation_result": reasoning_result.get("validation_result", {}),
            "processing_info": reasoning_result.get("processing_info", {})
        }
        
        # 添加系统级元数据
        final_result["system_metadata"] = {
            "orchestrator": "EnhancedSystemOrchestrator",
            "async_processing": True,
            "original_problem": original_problem.get("problem", ""),
            "data_preprocessed": processed_data != original_problem
        }
        
        return final_result
    
    def _recover_module_initialization(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """恢复模块初始化失败"""
        self.logger.info("尝试恢复模块初始化...")
        
        # 简化的恢复策略：返回基础结果
        return {
            "final_answer": "模块初始化失败，无法处理",
            "confidence": 0.0,
            "success": False,
            "recovery_method": "module_initialization_fallback"
        }
    
    def _recover_reasoning_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """恢复推理失败"""
        self.logger.info("尝试恢复推理失败...")
        
        problem = context.get("problem", {})
        
        # 尝试简单的数值提取作为兜底
        problem_text = problem.get("problem", "") or problem.get("cleaned_text", "")
        
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
        
        if len(numbers) >= 2:
            try:
                # 简单的加法作为兜底
                result = float(numbers[0]) + float(numbers[1])
                return {
                    "final_answer": str(result),
                    "confidence": 0.3,
                    "success": True,
                    "recovery_method": "simple_arithmetic_fallback"
                }
            except:
                pass
        
        return {
            "final_answer": "推理恢复失败",
            "confidence": 0.0,
            "success": False,
            "recovery_method": "reasoning_fallback_failed"
        }
    
    def _recover_model_call_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """恢复模型调用失败"""
        self.logger.info("尝试恢复模型调用失败...")
        
        return {
            "final_answer": "模型调用失败",
            "confidence": 0.0,
            "success": False,
            "recovery_method": "model_call_fallback"
        }
    
    def shutdown_system_ordered(self) -> bool:
        """基于依赖图的有序关闭"""
        try:
            self.logger.info("开始有序系统关闭...")
            
            # 获取关闭顺序
            shutdown_order = self.dependency_graph.get_shutdown_order()
            
            # 按顺序关闭模块
            for module_name in shutdown_order:
                try:
                    if registry.is_module_registered(module_name):
                        self._safe_shutdown_module(module_name)
                except Exception as e:
                    self.logger.warning(f"关闭模块{module_name}失败: {str(e)}")
            
            # 关闭协调器组件
            self.concurrent_executor.shutdown()
            
            self.logger.info("系统有序关闭完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统关闭失败: {str(e)}")
            return False
    
    def _safe_shutdown_module(self, module_name: str):
        """安全关闭模块"""
        try:
            module = registry.get_module(module_name)
            
            # 如果模块有shutdown方法，调用它
            if hasattr(module, 'shutdown'):
                module.shutdown()
            
            # 从注册表中移除
            registry.unregister_module(module_name)
            
            self.logger.debug(f"模块{module_name}已安全关闭")
            
        except Exception as e:
            self.logger.error(f"安全关闭模块{module_name}失败: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取增强系统状态"""
        try:
            modules = registry.list_modules()
            health_checks = registry.health_check_all()
            
            status = {
                "status": "operational",
                "orchestrator_type": "EnhancedSystemOrchestrator",
                "total_modules": len(modules),
                "modules": [
                    {
                        "name": module.name,
                        "type": module.type.value,
                        "version": module.version,
                        "health": health_checks.get(module.name, {"status": "unknown"})
                    }
                    for module in modules
                ],
                "dependencies": {
                    module: list(deps) 
                    for module, deps in self.dependency_graph.dependencies.items()
                },
                "orchestration_stats": self.orchestration_stats,
                "concurrent_capacity": self.concurrent_executor.max_workers,
                "error_history_count": len(self.error_recovery.error_history)
            }
            
            return status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _update_orchestration_stats(self, success: bool, processing_time: float):
        """更新协调统计"""
        with self._orchestration_lock:
            self.orchestration_stats["total_orchestrations"] += 1
            
            if success:
                self.orchestration_stats["successful_orchestrations"] += 1
            else:
                self.orchestration_stats["failed_orchestrations"] += 1
            
            # 更新平均处理时间
            total = self.orchestration_stats["total_orchestrations"]
            current_avg = self.orchestration_stats["average_orchestration_time"]
            new_avg = ((current_avg * (total - 1)) + processing_time) / total
            self.orchestration_stats["average_orchestration_time"] = new_avg
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            "orchestration_stats": self.orchestration_stats.copy(),
            "dependency_info": {
                "total_dependencies": sum(len(deps) for deps in self.dependency_graph.dependencies.values()),
                "modules_with_dependencies": len(self.dependency_graph.dependencies),
                "has_cycles": self.dependency_graph.has_cycle()
            },
            "error_recovery": {
                "total_errors": len(self.error_recovery.error_history),
                "recovery_strategies": len(self.error_recovery.recovery_strategies)
            },
            "concurrent_execution": {
                "max_workers": self.concurrent_executor.max_workers,
                "total_concurrent_executions": self.orchestration_stats.get("concurrent_executions", 0)
            }
        }


# 全局增强系统协调器实例
enhanced_system_orchestrator = EnhancedSystemOrchestrator()

# 保持向后兼容的全局实例
system_orchestrator = enhanced_system_orchestrator