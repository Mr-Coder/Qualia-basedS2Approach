"""
推理模块协调器

管理推理模块内部组件的协调和流程控制。
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加src_new到路径
src_new_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_new_path))

from core.exceptions import OrchestrationError, handle_module_error
from core.interfaces import BaseOrchestrator

from .private.confidence_calc import ConfidenceCalculator
from .private.processor import ReasoningProcessor
from .private.step_builder import StepBuilder
from .private.utils import ReasoningUtils
from .private.validator import ReasoningValidator


class ReasoningOrchestrator(BaseOrchestrator):
    """推理模块协调器"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._components = {}
        self._config = {}
        self._statistics = {
            "problems_solved": 0,
            "total_processing_time": 0.0,
            "success_rate": 0.0,
            "average_confidence": 0.0
        }
        self._initialized = False
        
    def initialize(self) -> bool:
        """初始化协调器和所有组件"""
        try:
            # 初始化核心组件
            self._components["validator"] = ReasoningValidator()
            self._components["processor"] = ReasoningProcessor()
            self._components["step_builder"] = StepBuilder()
            self._components["confidence_calc"] = ConfidenceCalculator()
            
            # 加载默认配置
            self._config = self._load_default_config()
            
            self._initialized = True
            self._logger.info("Reasoning orchestrator initialized successfully")
            return True
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "orchestrator_initialization")
            self._logger.error(f"Orchestrator initialization failed: {error}")
            raise error
    
    def orchestrate(self, operation: str, **kwargs) -> Any:
        """协调指定操作的执行"""
        if not self._initialized:
            raise OrchestrationError("Orchestrator not initialized", module_name="reasoning")
        
        operation_map = {
            "solve_problem": self._orchestrate_solve_problem,
            "validate_result": self._orchestrate_validate_result,
            "explain_reasoning": self._orchestrate_explain_reasoning,
            "set_configuration": self._orchestrate_set_configuration,
            "get_configuration": self._orchestrate_get_configuration,
            "get_statistics": self._orchestrate_get_statistics
        }
        
        if operation not in operation_map:
            raise OrchestrationError(f"Unknown operation: {operation}", module_name="reasoning")
        
        try:
            return operation_map[operation](**kwargs)
        except Exception as e:
            error = handle_module_error(e, "reasoning", f"orchestrate_{operation}")
            self._logger.error(f"Operation '{operation}' failed: {error}")
            raise error
    
    def register_component(self, name: str, component: Any) -> None:
        """注册组件"""
        self._components[name] = component
        self._logger.debug(f"Registered component: {name}")
    
    def get_component(self, name: str) -> Any:
        """获取组件"""
        return self._components.get(name)
    
    def get_component_status(self) -> Dict[str, Any]:
        """获取所有组件状态"""
        status = {}
        for name, component in self._components.items():
            try:
                if hasattr(component, 'health_check'):
                    status[name] = component.health_check()
                else:
                    status[name] = {"status": "available", "type": type(component).__name__}
            except Exception as e:
                status[name] = {"status": "error", "error": str(e)}
        return status
    
    def shutdown(self) -> bool:
        """关闭协调器"""
        try:
            # 清理组件
            for name, component in self._components.items():
                if hasattr(component, 'shutdown'):
                    try:
                        component.shutdown()
                    except Exception as e:
                        self._logger.warning(f"Failed to shutdown component {name}: {e}")
            
            self._components.clear()
            self._initialized = False
            self._logger.info("Reasoning orchestrator shutdown successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Orchestrator shutdown failed: {e}")
            return False
    
    def _orchestrate_solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """协调问题解决流程"""
        import time
        start_time = time.time()
        
        try:
            # Step 1: 输入验证
            validator = self._components["validator"]
            input_validation = validator.validate(problem)
            
            if not input_validation["is_valid"]:
                return {
                    "final_answer": "invalid_input",
                    "confidence": 0.0,
                    "reasoning_steps": [{
                        "step": 1,
                        "action": "input_validation",
                        "description": f"输入验证失败: {', '.join(input_validation['issues'])}",
                        "confidence": 0.1
                    }],
                    "strategy_used": "validation_failed",
                    "validation_issues": input_validation["issues"]
                }
            
            # Step 2: 核心推理处理
            processor = self._components["processor"]
            reasoning_result = processor.process(problem)
            
            # Step 3: 结果验证
            result_validation = validator.validate(reasoning_result)
            
            # Step 4: 计算最终置信度
            confidence_calc = self._components["confidence_calc"]
            final_confidence = confidence_calc.calculate_overall_confidence(
                reasoning_result.get("reasoning_steps", []),
                result_validation,
                problem.get("knowledge_context")
            )
            
            # Step 5: 构建最终结果
            final_result = {
                **reasoning_result,
                "confidence": final_confidence,
                "input_validation": input_validation,
                "result_validation": result_validation,
                "processing_time": time.time() - start_time
            }
            
            # 更新统计信息
            self._update_statistics(final_result)
            
            return final_result
            
        except Exception as e:
            error_result = {
                "final_answer": "error",
                "confidence": 0.0,
                "reasoning_steps": [{
                    "step": 1,
                    "action": "error_handling",
                    "description": f"推理过程出错: {str(e)}",
                    "confidence": 0.1
                }],
                "strategy_used": "error_fallback",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
            self._update_statistics(error_result)
            return error_result
    
    def _orchestrate_validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """协调结果验证"""
        validator = self._components["validator"]
        return validator.validate(result)
    
    def _orchestrate_explain_reasoning(self, result: Dict[str, Any]) -> str:
        """协调推理解释生成"""
        return ReasoningUtils.format_reasoning_output(result)
    
    def _orchestrate_set_configuration(self, config: Dict[str, Any]) -> bool:
        """协调配置设置"""
        try:
            # 验证配置
            valid_keys = ["confidence_threshold", "max_steps", "enable_validation", "timeout"]
            filtered_config = {k: v for k, v in config.items() if k in valid_keys}
            
            self._config.update(filtered_config)
            self._logger.info(f"Configuration updated: {filtered_config}")
            return True
            
        except Exception as e:
            self._logger.error(f"Configuration update failed: {e}")
            return False
    
    def _orchestrate_get_configuration(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._config.copy()
    
    def _orchestrate_get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._statistics,
            "component_count": len(self._components),
            "initialized": self._initialized
        }
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "confidence_threshold": 0.5,
            "max_steps": 10,
            "enable_validation": True,
            "timeout": 30.0,
            "strategy_priority": ["DIR", "TBR", "COT"],
            "logging_level": "INFO"
        }
    
    def _update_statistics(self, result: Dict[str, Any]) -> None:
        """更新统计信息"""
        try:
            self._statistics["problems_solved"] += 1
            
            # 更新处理时间
            processing_time = result.get("processing_time", 0.0)
            self._statistics["total_processing_time"] += processing_time
            
            # 更新成功率
            is_success = result.get("final_answer", "unknown") != "error"
            current_success_rate = self._statistics["success_rate"]
            total_problems = self._statistics["problems_solved"]
            
            self._statistics["success_rate"] = (
                (current_success_rate * (total_problems - 1) + (1.0 if is_success else 0.0)) 
                / total_problems
            )
            
            # 更新平均置信度
            confidence = result.get("confidence", 0.0)
            current_avg_confidence = self._statistics["average_confidence"]
            
            self._statistics["average_confidence"] = (
                (current_avg_confidence * (total_problems - 1) + confidence) 
                / total_problems
            )
            
        except Exception as e:
            self._logger.warning(f"Failed to update statistics: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        total_problems = self._statistics["problems_solved"]
        total_time = self._statistics["total_processing_time"]
        
        return {
            "total_problems_solved": total_problems,
            "success_rate": self._statistics["success_rate"],
            "average_confidence": self._statistics["average_confidence"],
            "average_processing_time": total_time / total_problems if total_problems > 0 else 0.0,
            "total_processing_time": total_time,
            "problems_per_second": total_problems / total_time if total_time > 0 else 0.0
        } 