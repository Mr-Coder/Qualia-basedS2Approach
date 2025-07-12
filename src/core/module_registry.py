"""
模块注册表实现

管理所有模块的注册、发现和生命周期。
"""

import logging
from typing import Dict, List, Optional

from .exceptions import (ModuleDependencyError, ModuleNotFoundError,
                         ModuleRegistrationError, handle_module_error)
from .interfaces import ModuleInfo, ModuleType, PublicAPI


class ModuleRegistryImpl:
    """模块注册表实现"""
    
    def __init__(self):
        self._modules: Dict[str, PublicAPI] = {}
        self._module_info: Dict[str, ModuleInfo] = {}
        self._logger = logging.getLogger(__name__)
        
    def register_module(self, module_info: ModuleInfo, api_instance: PublicAPI) -> bool:
        """注册模块"""
        try:
            # 检查模块是否已存在
            if module_info.name in self._modules:
                raise ModuleRegistrationError(
                    f"Module '{module_info.name}' is already registered",
                    module_name=module_info.name
                )
            
            # 检查依赖关系
            self._check_dependencies(module_info)
            
            # 初始化模块
            if not api_instance.initialize():
                raise ModuleRegistrationError(
                    f"Failed to initialize module '{module_info.name}'",
                    module_name=module_info.name
                )
            
            # 注册模块
            self._modules[module_info.name] = api_instance
            self._module_info[module_info.name] = module_info
            
            self._logger.info(f"Successfully registered module: {module_info.name}")
            return True
            
        except Exception as e:
            error = handle_module_error(e, module_info.name, "registration")
            self._logger.error(f"Module registration failed: {error}")
            raise error
    
    def get_module(self, module_name: str) -> Optional[PublicAPI]:
        """获取模块实例"""
        if module_name not in self._modules:
            raise ModuleNotFoundError(
                f"Module '{module_name}' not found",
                module_name=module_name
            )
        return self._modules[module_name]
    
    def list_modules(self) -> List[ModuleInfo]:
        """列出所有已注册模块"""
        return list(self._module_info.values())
    
    def unregister_module(self, module_name: str) -> bool:
        """注销模块"""
        try:
            if module_name not in self._modules:
                raise ModuleNotFoundError(
                    f"Module '{module_name}' not found",
                    module_name=module_name
                )
            
            # 检查是否有其他模块依赖此模块
            self._check_dependents(module_name)
            
            # 移除模块
            del self._modules[module_name]
            del self._module_info[module_name]
            
            self._logger.info(f"Successfully unregistered module: {module_name}")
            return True
            
        except Exception as e:
            error = handle_module_error(e, module_name, "unregistration")
            self._logger.error(f"Module unregistration failed: {error}")
            raise error
    
    def get_module_info(self, module_name: str) -> Optional[ModuleInfo]:
        """获取模块信息"""
        return self._module_info.get(module_name)
    
    def is_module_registered(self, module_name: str) -> bool:
        """检查模块是否已注册"""
        return module_name in self._modules
    
    def get_modules_by_type(self, module_type: ModuleType) -> List[PublicAPI]:
        """根据类型获取模块"""
        result = []
        for name, info in self._module_info.items():
            if info.type == module_type:
                result.append(self._modules[name])
        return result
    
    def health_check_all(self) -> Dict[str, Dict[str, any]]:
        """对所有模块进行健康检查"""
        results = {}
        for name, module in self._modules.items():
            try:
                results[name] = module.health_check()
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
        return results
    
    def _check_dependencies(self, module_info: ModuleInfo) -> None:
        """检查模块依赖关系"""
        for dependency in module_info.dependencies:
            if dependency not in self._modules:
                raise ModuleDependencyError(
                    f"Dependency '{dependency}' not found for module '{module_info.name}'",
                    module_name=module_info.name
                )
    
    def _check_dependents(self, module_name: str) -> None:
        """检查是否有其他模块依赖此模块"""
        dependents = []
        for name, info in self._module_info.items():
            if module_name in info.dependencies:
                dependents.append(name)
        
        if dependents:
            raise ModuleDependencyError(
                f"Cannot unregister module '{module_name}': modules {dependents} depend on it",
                module_name=module_name
            )


# 全局模块注册表实例
registry = ModuleRegistryImpl() 