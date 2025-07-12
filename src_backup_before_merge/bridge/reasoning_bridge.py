"""
推理引擎桥接层 - 激活重构后的代码
将旧版本ReasoningEngine接口桥接到新版本ReasoningAPI
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict

# 添加src_new到Python路径
project_root = Path(__file__).parent.parent.parent
src_new_path = project_root / "src_new"
sys.path.insert(0, str(src_new_path))

class ReasoningEngineBridge:
    """桥接旧版本ReasoningEngine到新版本ReasoningAPI"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self._api = None
        self._old_engine = None
        self._setup_system()
    
    def _setup_system(self):
        """设置系统 - 优先使用新版本，失败则降级到旧版本"""
        # 首先尝试新版本
        if self._try_new_system():
            print("✅ 使用新版本推理系统")
            return
        
        # 新版本失败，尝试旧版本
        if self._try_old_system():
            print("⚠️ 降级到旧版本推理系统")
            return
        
        # 都失败了
        raise Exception("无法初始化任何推理系统")
    
    def _try_new_system(self):
        """尝试初始化新版本系统"""
        try:
            # 导入新版本模块
            from core import (ModuleInfo, ModuleType, registry,
                              system_orchestrator)
            from reasoning import ReasoningAPI

            # 检查是否已经初始化
            if registry.is_module_registered("reasoning"):
                print("✅ 新版本系统已初始化")
                return True
            
            # 注册推理模块
            reasoning_info = ModuleInfo(
                name="reasoning",
                type=ModuleType.REASONING,
                version="1.0.0",
                dependencies=[],
                public_api_class="ReasoningAPI",
                orchestrator_class="ReasoningOrchestrator"
            )
            
            # 创建API实例
            self._api = ReasoningAPI()
            
            # 注册到系统
            registry.register_module(reasoning_info, self._api)
            system_orchestrator.initialize_system()
            
            return True
            
        except Exception as e:
            print(f"新版本系统初始化失败: {e}")
            return False
    
    def _try_old_system(self):
        """尝试初始化旧版本系统"""
        try:
            # 添加旧版本路径
            old_src_path = project_root / "src"
            if str(old_src_path) not in sys.path:
                sys.path.insert(0, str(old_src_path))
            
            from reasoning_core.reasoning_engine import ReasoningEngine
            self._old_engine = ReasoningEngine()
            return True
            
        except Exception as e:
            print(f"旧版本系统初始化失败: {e}")
            return False
    
    def solve(self, sample: Dict) -> Dict:
        """兼容旧版本的solve方法"""
        if self._api:
            # 使用新版本API
            return self._solve_with_new_api(sample)
        elif self._old_engine:
            # 使用旧版本引擎
            return self._old_engine.solve(sample)
        else:
            # 都不可用，返回错误
            return {
                "final_answer": "System Error",
                "strategy_used": "ERROR",
                "confidence": 0.0,
                "reasoning_steps": ["System initialization failed"],
                "intermediate_variables": {},
                "template_used": "",
                "meta_knowledge_enhancement": {},
                "solution_validation": {}
            }
    
    def _solve_with_new_api(self, sample: Dict) -> Dict:
        """使用新版本API解决问题"""
        try:
            from core import system_orchestrator

            # 适配输入格式
            problem = {
                "problem": sample.get("cleaned_text") or sample.get("problem", ""),
                "context": sample
            }
            
            # 调用新版本
            result = system_orchestrator.solve_math_problem(problem)
            
            # 适配返回格式到旧版本期望的格式
            return {
                "final_answer": result.get("final_answer", ""),
                "strategy_used": result.get("strategy_used", "COT"),
                "confidence": result.get("confidence", 0.5),
                "reasoning_steps": result.get("reasoning_steps", []),
                "intermediate_variables": {},
                "template_used": result.get("template_used", ""),
                "meta_knowledge_enhancement": result.get("knowledge_enhancement", {}),
                "solution_validation": result.get("result_validation", {})
            }
            
        except Exception as e:
            print(f"新版本API调用失败: {e}")
            # 如果有旧版本，降级使用
            if self._old_engine:
                return self._old_engine.solve(sample)
            
            # 返回错误结果
            return {
                "final_answer": f"Error: {str(e)}",
                "strategy_used": "ERROR",
                "confidence": 0.0,
                "reasoning_steps": [f"Error: {str(e)}"],
                "intermediate_variables": {},
                "template_used": "",
                "meta_knowledge_enhancement": {},
                "solution_validation": {}
            }

# 为了兼容性，也提供旧的类名
ReasoningEngine = ReasoningEngineBridge 