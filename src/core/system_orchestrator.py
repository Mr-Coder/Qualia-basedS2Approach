"""
系统级协调器

管理所有模块间的协作和系统级操作。
"""

import logging
from typing import Any, Dict, List, Optional

from .exceptions import OrchestrationError, handle_module_error
from .interfaces import ModuleType, PublicAPI
from .module_registry import registry


class SystemOrchestrator:
    """系统级协调器"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._message_handlers = {}
        
    def solve_math_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        解决数学问题的系统级流程协调
        
        这是系统的主要入口点，协调多个模块来解决数学问题。
        """
        try:
            self._logger.info("Starting math problem solving process")
            
            # Step 1: 获取必要的模块
            reasoning_module = registry.get_module("reasoning")
            
            # 安全获取可选模块
            template_module = None
            if registry.is_module_registered("template_management"):
                template_module = registry.get_module("template_management")
            
            data_processing_module = None
            if registry.is_module_registered("data_processing"):
                data_processing_module = registry.get_module("data_processing")
            
            meta_knowledge_module = None
            if registry.is_module_registered("meta_knowledge"):
                meta_knowledge_module = registry.get_module("meta_knowledge")
            
            # Step 2: 数据预处理
            if data_processing_module:
                processed_data = data_processing_module.process_problem(problem)
            else:
                processed_data = problem
                
            # Step 3: 模板识别
            template_info = None
            if template_module:
                template_info = template_module.identify_template(
                    processed_data.get("cleaned_text", processed_data.get("problem", ""))
                )
            
            # Step 4: 元知识增强
            knowledge_context = None
            if meta_knowledge_module:
                knowledge_context = meta_knowledge_module.analyze_problem(processed_data)
            
            # Step 5: 推理求解
            reasoning_result = reasoning_module.solve_problem({
                **processed_data,
                "template_info": template_info,
                "knowledge_context": knowledge_context
            })
            
            # Step 6: 结果整合
            final_result = {
                "final_answer": reasoning_result.get("final_answer"),
                "confidence": reasoning_result.get("confidence"),
                "reasoning_steps": reasoning_result.get("reasoning_steps", []),
                "template_used": template_info.get("type") if template_info else None,
                "knowledge_enhancement": knowledge_context,
                "processing_info": {
                    "modules_used": [
                        module for module in ["reasoning", "template_management", 
                                            "data_processing", "meta_knowledge"]
                        if registry.is_module_registered(module)
                    ]
                }
            }
            
            self._logger.info("Math problem solving process completed successfully")
            return final_result
            
        except Exception as e:
            error = handle_module_error(e, "system", "solve_math_problem")
            self._logger.error(f"System orchestration failed: {error}")
            raise error
    
    def batch_solve_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量解决数学问题"""
        try:
            self._logger.info(f"Starting batch solving for {len(problems)} problems")
            
            results = []
            for i, problem in enumerate(problems):
                try:
                    result = self.solve_math_problem(problem)
                    result["problem_index"] = i
                    results.append(result)
                except Exception as e:
                    self._logger.warning(f"Failed to solve problem {i}: {e}")
                    results.append({
                        "problem_index": i,
                        "error": str(e),
                        "final_answer": "unknown"
                    })
            
            self._logger.info(f"Batch solving completed: {len(results)} results")
            return results
            
        except Exception as e:
            error = handle_module_error(e, "system", "batch_solve_problems")
            self._logger.error(f"Batch solving failed: {error}")
            raise error
    
    def initialize_system(self) -> bool:
        """初始化整个系统"""
        try:
            self._logger.info("Initializing modular system")
            
            # 检查核心模块是否已注册
            required_modules = ["reasoning"]
            optional_modules = ["template_management", "data_processing", 
                              "meta_knowledge", "evaluation", "configuration"]
            
            missing_required = []
            for module_name in required_modules:
                if not registry.is_module_registered(module_name):
                    missing_required.append(module_name)
            
            if missing_required:
                raise OrchestrationError(
                    f"Required modules not registered: {missing_required}",
                    module_name="system"
                )
            
            # 检查可选模块
            available_optional = []
            for module_name in optional_modules:
                if registry.is_module_registered(module_name):
                    available_optional.append(module_name)
            
            self._logger.info(f"System initialized with modules: {required_modules + available_optional}")
            return True
            
        except Exception as e:
            error = handle_module_error(e, "system", "initialization")
            self._logger.error(f"System initialization failed: {error}")
            raise error
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            modules = registry.list_modules()
            health_checks = registry.health_check_all()
            
            return {
                "status": "operational",
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
                "capabilities": self._get_system_capabilities()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def shutdown_system(self) -> bool:
        """关闭系统"""
        try:
            self._logger.info("Shutting down modular system")
            
            # 获取所有模块并按依赖关系逆序关闭
            modules = registry.list_modules()
            
            # 简单逆序关闭（可以改进为基于依赖图的关闭顺序）
            for module in reversed(modules):
                try:
                    registry.unregister_module(module.name)
                except Exception as e:
                    self._logger.warning(f"Failed to unregister module {module.name}: {e}")
            
            self._logger.info("System shutdown completed")
            return True
            
        except Exception as e:
            error = handle_module_error(e, "system", "shutdown")
            self._logger.error(f"System shutdown failed: {error}")
            return False
    
    def _get_system_capabilities(self) -> List[str]:
        """获取系统能力列表"""
        capabilities = ["basic_reasoning"]
        
        if registry.is_module_registered("template_management"):
            capabilities.append("template_matching")
        
        if registry.is_module_registered("meta_knowledge"):
            capabilities.append("knowledge_enhancement")
        
        if registry.is_module_registered("data_processing"):
            capabilities.append("data_preprocessing")
        
        if registry.is_module_registered("evaluation"):
            capabilities.append("performance_evaluation")
        
        return capabilities


# 全局系统协调器实例
system_orchestrator = SystemOrchestrator() 