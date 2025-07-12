"""
推理模块公共API

提供标准化的推理接口，是外部访问推理功能的唯一入口。
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加src_new到路径
src_new_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_new_path))

from core.exceptions import APIError, handle_module_error
from core.interfaces import ModuleInfo, ModuleType, PublicAPI

from .orchestrator import ReasoningOrchestrator


class ReasoningAPI(PublicAPI):
    """推理模块公共API"""
    
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.orchestrator = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """初始化推理模块"""
        try:
            self.orchestrator = ReasoningOrchestrator()
            
            # 初始化协调器
            if not self.orchestrator.initialize():
                raise APIError("Failed to initialize reasoning orchestrator", module_name="reasoning")
            
            self._initialized = True
            self._logger.info("Reasoning module initialized successfully")
            return True
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "initialization")
            self._logger.error(f"Reasoning module initialization failed: {error}")
            raise error
    
    def get_module_info(self) -> ModuleInfo:
        """获取模块信息"""
        return ModuleInfo(
            name="reasoning",
            type=ModuleType.REASONING,
            version="1.0.0",
            dependencies=[],
            public_api_class="ReasoningAPI",
            orchestrator_class="ReasoningOrchestrator"
        )
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = {
                "status": "healthy" if self._initialized else "not_initialized",
                "initialized": self._initialized,
                "components": {}
            }
            
            if self.orchestrator:
                status["components"] = self.orchestrator.get_component_status()
            
            return status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        解决数学问题
        
        Args:
            problem: 问题数据，包含 'problem' 或 'cleaned_text' 字段
            
        Returns:
            包含推理结果的字典，包含：
            - final_answer: 最终答案
            - confidence: 置信度 (0-1)
            - reasoning_steps: 推理步骤列表
            - strategy_used: 使用的推理策略
        """
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            self._validate_problem_input(problem)
            
            result = self.orchestrator.orchestrate("solve_problem", problem=problem)
            self._logger.debug(f"Problem solved with answer: {result.get('final_answer')}")
            
            return result
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "solve_problem")
            self._logger.error(f"Problem solving failed: {error}")
            raise error
    
    def batch_solve(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量解决数学问题
        
        Args:
            problems: 问题列表
            
        Returns:
            结果列表，每个元素包含单个问题的推理结果
        """
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            if not isinstance(problems, list):
                raise APIError("Problems must be a list", module_name="reasoning")
            
            results = []
            for i, problem in enumerate(problems):
                try:
                    result = self.solve_problem(problem)
                    result["problem_index"] = i
                    results.append(result)
                except Exception as e:
                    self._logger.warning(f"Failed to solve problem {i}: {e}")
                    results.append({
                        "problem_index": i,
                        "error": str(e),
                        "final_answer": "unknown",
                        "confidence": 0.0
                    })
            
            self._logger.info(f"Batch solved {len(problems)} problems, {len(results)} results")
            return results
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "batch_solve")
            self._logger.error(f"Batch solving failed: {error}")
            raise error
    
    def get_reasoning_steps(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        获取详细推理步骤（不包含最终答案）
        
        Args:
            problem: 问题数据
            
        Returns:
            推理步骤列表
        """
        try:
            result = self.solve_problem(problem)
            return result.get("reasoning_steps", [])
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "get_reasoning_steps")
            self._logger.error(f"Getting reasoning steps failed: {error}")
            raise error
    
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证推理结果
        
        Args:
            result: 推理结果
            
        Returns:
            验证结果，包含 is_valid, confidence, issues 等字段
        """
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            return self.orchestrator.orchestrate("validate_result", result=result)
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "validate_result")
            self._logger.error(f"Result validation failed: {error}")
            raise error
    
    def explain_reasoning(self, result: Dict[str, Any]) -> str:
        """
        生成推理过程的文本解释
        
        Args:
            result: 推理结果
            
        Returns:
            推理过程的文本描述
        """
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            return self.orchestrator.orchestrate("explain_reasoning", result=result)
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "explain_reasoning")
            self._logger.error(f"Reasoning explanation failed: {error}")
            raise error
    
    def set_configuration(self, config: Dict[str, Any]) -> bool:
        """
        设置推理模块配置
        
        Args:
            config: 配置字典
            
        Returns:
            设置是否成功
        """
        try:
            if not self._initialized:
                raise APIError("Reasoning module not initialized", module_name="reasoning")
            
            return self.orchestrator.orchestrate("set_configuration", config=config)
            
        except Exception as e:
            error = handle_module_error(e, "reasoning", "set_configuration")
            self._logger.error(f"Configuration setting failed: {error}")
            raise error
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        Returns:
            当前配置字典
        """
        try:
            if not self._initialized:
                return {}
            
            return self.orchestrator.orchestrate("get_configuration")
            
        except Exception as e:
            self._logger.warning(f"Getting configuration failed: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取推理模块统计信息
        
        Returns:
            统计信息字典
        """
        try:
            if not self._initialized:
                return {"status": "not_initialized"}
            
            return self.orchestrator.orchestrate("get_statistics")
            
        except Exception as e:
            self._logger.warning(f"Getting statistics failed: {e}")
            return {"error": str(e)}
    
    def _validate_problem_input(self, problem: Dict[str, Any]) -> None:
        """验证问题输入"""
        if not isinstance(problem, dict):
            raise APIError("Problem must be a dictionary", module_name="reasoning")
        
        # 检查必要字段
        problem_text = problem.get("problem") or problem.get("cleaned_text")
        if not problem_text:
            raise APIError("Problem must contain 'problem' or 'cleaned_text' field", module_name="reasoning")
        
        if not isinstance(problem_text, str) or not problem_text.strip():
            raise APIError("Problem text must be a non-empty string", module_name="reasoning")
    
    def shutdown(self) -> bool:
        """关闭推理模块"""
        try:
            if self.orchestrator:
                self.orchestrator.shutdown()
            
            self._initialized = False
            self._logger.info("Reasoning module shutdown successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Reasoning module shutdown failed: {e}")
            return False 