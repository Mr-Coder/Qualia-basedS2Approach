"""
系统级接口定义

定义所有模块必须遵循的基础协议，为模块化架构提供统一的接口标准。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union


class ModuleType(Enum):
    """模块类型枚举"""
    REASONING = "reasoning"
    TEMPLATE_MANAGEMENT = "template_management"
    DATA_PROCESSING = "data_processing"
    META_KNOWLEDGE = "meta_knowledge"
    EVALUATION = "evaluation"
    CONFIGURATION = "configuration"


@dataclass
class ModuleInfo:
    """模块信息数据类"""
    name: str
    type: ModuleType
    version: str
    dependencies: List[str]
    public_api_class: str
    orchestrator_class: str


class BaseValidator(ABC):
    """基础验证器协议"""
    
    @abstractmethod
    def validate(self, data: Any) -> Dict[str, Any]:
        """验证数据并返回验证结果"""
        pass


class BaseProcessor(ABC):
    """基础处理器协议"""
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """处理输入数据并返回结果"""
        pass


class BaseOrchestrator(ABC):
    """基础协调器协议"""
    
    @abstractmethod
    def orchestrate(self, operation: str, **kwargs) -> Any:
        """协调指定操作的执行"""
        pass
    
    @abstractmethod
    def register_component(self, name: str, component: Any) -> None:
        """注册组件"""
        pass
    
    @abstractmethod
    def get_component(self, name: str) -> Any:
        """获取组件"""
        pass


class PublicAPI(ABC):
    """公共API基类"""
    
    def __init__(self):
        self.orchestrator: BaseOrchestrator = None
        self.module_info: ModuleInfo = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化模块"""
        pass
    
    @abstractmethod
    def get_module_info(self) -> ModuleInfo:
        """获取模块信息"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass


class SystemMessage:
    """系统消息定义"""
    
    def __init__(self, sender: str, receiver: str, message_type: str, data: Any):
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.data = data
        self.timestamp = None


class ModuleRegistry(Protocol):
    """模块注册表协议"""
    
    def register_module(self, module_info: ModuleInfo, api_instance: PublicAPI) -> bool:
        """注册模块"""
        ...
    
    def get_module(self, module_name: str) -> Optional[PublicAPI]:
        """获取模块实例"""
        ...
    
    def list_modules(self) -> List[ModuleInfo]:
        """列出所有已注册模块"""
        ...
    
    def unregister_module(self, module_name: str) -> bool:
        """注销模块"""
        ... 