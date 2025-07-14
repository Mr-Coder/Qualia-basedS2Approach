"""
共享安全服务

提供单例的安全计算器和其他安全工具，消除代码重复。
"""

import logging
import threading
from typing import Any, Dict, Optional, Union

# 延迟导入以避免循环依赖
_secure_evaluator = None
_security_service_lock = threading.Lock()


class SecurityService:
    """安全服务单例类"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SecurityService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(f"{__name__}.SecurityService")
            self._secure_evaluator = None
            self._file_manager = None
            self._config_manager = None
            self._initialized = True
            self.logger.info("安全服务单例初始化完成")
    
    def get_secure_evaluator(self):
        """获取安全数学计算器"""
        if self._secure_evaluator is None:
            try:
                # 动态导入避免循环依赖
                from ..tools.security_hardening import SecureMathEvaluator
                self._secure_evaluator = SecureMathEvaluator()
                self.logger.debug("安全计算器创建完成")
            except ImportError:
                # 回退到本地实现
                self._secure_evaluator = self._create_fallback_evaluator()
                self.logger.warning("使用回退安全计算器")
        
        return self._secure_evaluator
    
    def get_secure_file_manager(self):
        """获取安全文件管理器"""
        if self._file_manager is None:
            try:
                from ..tools.security_hardening import SecureFileManager
                self._file_manager = SecureFileManager()
                self.logger.debug("安全文件管理器创建完成")
            except ImportError:
                self.logger.warning("安全文件管理器不可用")
                self._file_manager = None
        
        return self._file_manager
    
    def get_secure_config_manager(self, config_dir: Optional[str] = None):
        """获取安全配置管理器"""
        if self._config_manager is None and config_dir:
            try:
                from ..tools.security_hardening import SecureConfigManager
                self._config_manager = SecureConfigManager(config_dir)
                self.logger.debug("安全配置管理器创建完成")
            except ImportError:
                self.logger.warning("安全配置管理器不可用")
                self._config_manager = None
        
        return self._config_manager
    
    def safe_eval(self, expression: str, allowed_names: Optional[Dict[str, Any]] = None) -> Union[float, int]:
        """安全计算数学表达式"""
        evaluator = self.get_secure_evaluator()
        if evaluator:
            return evaluator.safe_eval(expression, allowed_names)
        else:
            # 简单的后备实现
            return self._fallback_eval(expression)
    
    def _create_fallback_evaluator(self):
        """创建后备安全计算器"""
        class FallbackSecureEvaluator:
            def __init__(self):
                self.logger = logging.getLogger("FallbackSecureEvaluator")
            
            def safe_eval(self, expression: str, allowed_names: Optional[Dict[str, Any]] = None) -> Union[float, int]:
                # 简单的安全检查
                if any(dangerous in expression for dangerous in ['import', '__', 'exec', 'eval', 'open']):
                    self.logger.warning(f"拒绝执行危险表达式: {expression}")
                    return 0.0
                
                # 尝试简单的数字解析
                try:
                    return float(expression.strip())
                except ValueError:
                    self.logger.warning(f"无法解析表达式: {expression}")
                    return 0.0
        
        return FallbackSecureEvaluator()
    
    def _fallback_eval(self, expression: str) -> Union[float, int]:
        """后备数学计算"""
        try:
            # 只允许简单的数字
            return float(expression.strip())
        except ValueError:
            self.logger.warning(f"后备计算失败: {expression}")
            return 0.0


# 全局安全服务实例
_security_service = None

def get_security_service() -> SecurityService:
    """获取全局安全服务实例"""
    global _security_service
    if _security_service is None:
        with _security_service_lock:
            if _security_service is None:
                _security_service = SecurityService()
    return _security_service


def get_secure_evaluator():
    """便利函数：获取安全计算器"""
    return get_security_service().get_secure_evaluator()


def safe_eval(expression: str, allowed_names: Optional[Dict[str, Any]] = None) -> Union[float, int]:
    """便利函数：安全计算数学表达式"""
    return get_security_service().safe_eval(expression, allowed_names)


def get_secure_file_manager():
    """便利函数：获取安全文件管理器"""
    return get_security_service().get_secure_file_manager()


def get_secure_config_manager(config_dir: Optional[str] = None):
    """便利函数：获取安全配置管理器"""
    return get_security_service().get_secure_config_manager(config_dir)


# 向后兼容的全局变量
def _get_global_secure_evaluator():
    """向后兼容：获取全局安全计算器"""
    global _secure_evaluator
    if _secure_evaluator is None:
        _secure_evaluator = get_secure_evaluator()
    return _secure_evaluator


# 向后兼容的导出
_secure_evaluator = _get_global_secure_evaluator()