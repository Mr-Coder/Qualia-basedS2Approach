"""
UI模块接口定义
定义UI系统的输入输出接口和核心抽象
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import json
from datetime import datetime


class UIEventType(Enum):
    """UI事件类型枚举"""
    PROBLEM_SUBMIT = "problem_submit"
    PROBLEM_CLEAR = "problem_clear"
    REASONING_START = "reasoning_start"
    REASONING_STEP = "reasoning_step"
    REASONING_COMPLETE = "reasoning_complete"
    RESULT_DISPLAY = "result_display"
    ERROR_OCCURRED = "error_occurred"
    CONFIG_CHANGE = "config_change"
    STATISTICS_UPDATE = "statistics_update"


class UIComponentType(Enum):
    """UI组件类型枚举"""
    INPUT = "input"
    OUTPUT = "output"
    DISPLAY = "display"
    CONTROL = "control"
    NAVIGATION = "navigation"
    MODAL = "modal"


class UIResponseType(Enum):
    """UI响应类型枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    LOADING = "loading"


@dataclass
class UIRequest:
    """UI请求数据结构"""
    request_id: str
    component_id: str
    action: str
    data: Dict[str, Any]
    timestamp: datetime
    user_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "component_id": self.component_id,
            "action": self.action,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "user_context": self.user_context or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIRequest':
        """从字典创建请求"""
        return cls(
            request_id=data["request_id"],
            component_id=data["component_id"],
            action=data["action"],
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_context=data.get("user_context")
        )


@dataclass
class UIResponse:
    """UI响应数据结构"""
    request_id: str
    response_type: UIResponseType
    data: Dict[str, Any]
    message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "response_type": self.response_type.value,
            "data": self.data,
            "message": self.message,
            "error_details": self.error_details,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIResponse':
        """从字典创建响应"""
        return cls(
            request_id=data["request_id"],
            response_type=UIResponseType(data["response_type"]),
            data=data["data"],
            message=data.get("message"),
            error_details=data.get("error_details"),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class UIEvent:
    """UI事件数据结构"""
    event_id: str
    event_type: UIEventType
    source_component: str
    data: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source_component": self.source_component,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class UIComponentState:
    """UI组件状态数据结构"""
    component_id: str
    component_type: UIComponentType
    state: Dict[str, Any]
    visible: bool = True
    enabled: bool = True
    error_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "state": self.state,
            "visible": self.visible,
            "enabled": self.enabled,
            "error_state": self.error_state
        }


class IUIComponent(ABC):
    """UI组件接口"""
    
    @abstractmethod
    def get_component_id(self) -> str:
        """获取组件ID"""
        pass
    
    @abstractmethod
    def get_component_type(self) -> UIComponentType:
        """获取组件类型"""
        pass
    
    @abstractmethod
    def render(self, state: UIComponentState) -> Dict[str, Any]:
        """渲染组件"""
        pass
    
    @abstractmethod
    def handle_event(self, event: UIEvent) -> Optional[UIResponse]:
        """处理事件"""
        pass
    
    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证输入数据"""
        pass
    
    @abstractmethod
    def get_state(self) -> UIComponentState:
        """获取组件状态"""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> bool:
        """设置组件状态"""
        pass


class IUIEventHandler(ABC):
    """UI事件处理器接口"""
    
    @abstractmethod
    def handle_event(self, event: UIEvent) -> UIResponse:
        """处理UI事件"""
        pass
    
    @abstractmethod
    def get_supported_events(self) -> List[UIEventType]:
        """获取支持的事件类型"""
        pass
    
    @abstractmethod
    def validate_event_data(self, event_type: UIEventType, data: Dict[str, Any]) -> bool:
        """验证事件数据"""
        pass


class IUIRenderer(ABC):
    """UI渲染器接口"""
    
    @abstractmethod
    def render_component(self, component: IUIComponent, state: UIComponentState) -> Dict[str, Any]:
        """渲染单个组件"""
        pass
    
    @abstractmethod
    def render_layout(self, components: List[IUIComponent], layout_config: Dict[str, Any]) -> Dict[str, Any]:
        """渲染布局"""
        pass
    
    @abstractmethod
    def render_page(self, page_config: Dict[str, Any]) -> Dict[str, Any]:
        """渲染页面"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的渲染格式"""
        pass


class IUIManager(ABC):
    """UI管理器接口"""
    
    @abstractmethod
    def register_component(self, component: IUIComponent) -> bool:
        """注册UI组件"""
        pass
    
    @abstractmethod
    def unregister_component(self, component_id: str) -> bool:
        """注销UI组件"""
        pass
    
    @abstractmethod
    def get_component(self, component_id: str) -> Optional[IUIComponent]:
        """获取UI组件"""
        pass
    
    @abstractmethod
    def process_request(self, request: UIRequest) -> UIResponse:
        """处理UI请求"""
        pass
    
    @abstractmethod
    def handle_event(self, event: UIEvent) -> None:
        """处理UI事件"""
        pass
    
    @abstractmethod
    def get_component_states(self) -> Dict[str, UIComponentState]:
        """获取所有组件状态"""
        pass
    
    @abstractmethod
    def update_component_state(self, component_id: str, state: Dict[str, Any]) -> bool:
        """更新组件状态"""
        pass


class IUIStateManager(ABC):
    """UI状态管理器接口"""
    
    @abstractmethod
    def get_state(self, component_id: str) -> Optional[UIComponentState]:
        """获取组件状态"""
        pass
    
    @abstractmethod
    def set_state(self, component_id: str, state: UIComponentState) -> bool:
        """设置组件状态"""
        pass
    
    @abstractmethod
    def subscribe_to_state_changes(self, component_id: str, callback: Callable) -> str:
        """订阅状态变化"""
        pass
    
    @abstractmethod
    def unsubscribe_from_state_changes(self, subscription_id: str) -> bool:
        """取消订阅状态变化"""
        pass
    
    @abstractmethod
    def get_global_state(self) -> Dict[str, Any]:
        """获取全局状态"""
        pass
    
    @abstractmethod
    def set_global_state(self, key: str, value: Any) -> bool:
        """设置全局状态"""
        pass


class IUIValidator(ABC):
    """UI验证器接口"""
    
    @abstractmethod
    def validate_request(self, request: UIRequest) -> Dict[str, Any]:
        """验证UI请求"""
        pass
    
    @abstractmethod
    def validate_component_data(self, component_type: UIComponentType, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证组件数据"""
        pass
    
    @abstractmethod
    def validate_event_data(self, event_type: UIEventType, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证事件数据"""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """获取验证规则"""
        pass


# 特定业务接口
class IProblemInputComponent(IUIComponent):
    """问题输入组件接口"""
    
    @abstractmethod
    def get_problem_text(self) -> str:
        """获取问题文本"""
        pass
    
    @abstractmethod
    def set_problem_text(self, text: str) -> bool:
        """设置问题文本"""
        pass
    
    @abstractmethod
    def clear_input(self) -> bool:
        """清空输入"""
        pass
    
    @abstractmethod
    def validate_problem_format(self, text: str) -> Dict[str, Any]:
        """验证问题格式"""
        pass


class IReasoningDisplayComponent(IUIComponent):
    """推理过程显示组件接口"""
    
    @abstractmethod
    def display_reasoning_steps(self, steps: List[Dict[str, Any]]) -> bool:
        """显示推理步骤"""
        pass
    
    @abstractmethod
    def update_current_step(self, step_index: int, step_data: Dict[str, Any]) -> bool:
        """更新当前步骤"""
        pass
    
    @abstractmethod
    def highlight_step(self, step_index: int) -> bool:
        """高亮步骤"""
        pass
    
    @abstractmethod
    def clear_display(self) -> bool:
        """清空显示"""
        pass


class IResultDisplayComponent(IUIComponent):
    """结果显示组件接口"""
    
    @abstractmethod
    def display_result(self, result: Dict[str, Any]) -> bool:
        """显示结果"""
        pass
    
    @abstractmethod
    def display_confidence(self, confidence: float) -> bool:
        """显示置信度"""
        pass
    
    @abstractmethod
    def display_explanation(self, explanation: str) -> bool:
        """显示解释"""
        pass
    
    @abstractmethod
    def clear_result(self) -> bool:
        """清空结果"""
        pass


# 工具类
class UIUtils:
    """UI工具类"""
    
    @staticmethod
    def generate_request_id() -> str:
        """生成请求ID"""
        import uuid
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_event_id() -> str:
        """生成事件ID"""
        import uuid
        return str(uuid.uuid4())
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """清理输入文本"""
        if not isinstance(text, str):
            return ""
        
        # 移除危险字符
        dangerous_chars = ['<', '>', '&', '"', "'", '`']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        return text.strip()
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """格式化置信度"""
        if not isinstance(confidence, (int, float)):
            return "0.00%"
        
        percentage = confidence * 100
        return f"{percentage:.1f}%"
    
    @staticmethod
    def format_processing_time(time_ms: float) -> str:
        """格式化处理时间"""
        if time_ms < 1000:
            return f"{time_ms:.0f}ms"
        elif time_ms < 60000:
            return f"{time_ms/1000:.1f}s"
        else:
            return f"{time_ms/60000:.1f}min"
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """验证JSON模式"""
        try:
            # 简化的模式验证
            errors = []
            
            # 检查必需字段
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            # 检查字段类型
            properties = schema.get("properties", {})
            for field, field_schema in properties.items():
                if field in data:
                    expected_type = field_schema.get("type")
                    actual_value = data[field]
                    
                    if expected_type == "string" and not isinstance(actual_value, str):
                        errors.append(f"Field '{field}' must be string")
                    elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                        errors.append(f"Field '{field}' must be number")
                    elif expected_type == "boolean" and not isinstance(actual_value, bool):
                        errors.append(f"Field '{field}' must be boolean")
                    elif expected_type == "array" and not isinstance(actual_value, list):
                        errors.append(f"Field '{field}' must be array")
                    elif expected_type == "object" and not isinstance(actual_value, dict):
                        errors.append(f"Field '{field}' must be object")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Schema validation error: {str(e)}"]
            }


# 预定义的UI模式
class UISchemas:
    """UI数据模式定义"""
    
    PROBLEM_INPUT_SCHEMA = {
        "type": "object",
        "required": ["problem_text"],
        "properties": {
            "problem_text": {"type": "string", "minLength": 1},
            "problem_type": {"type": "string"},
            "difficulty": {"type": "string"},
            "context": {"type": "object"}
        }
    }
    
    REASONING_STEP_SCHEMA = {
        "type": "object",
        "required": ["step_index", "step_type", "description"],
        "properties": {
            "step_index": {"type": "number"},
            "step_type": {"type": "string"},
            "description": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "data": {"type": "object"}
        }
    }
    
    RESULT_SCHEMA = {
        "type": "object",
        "required": ["final_answer", "confidence"],
        "properties": {
            "final_answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning_steps": {"type": "array"},
            "strategy_used": {"type": "string"},
            "processing_time": {"type": "number"},
            "explanation": {"type": "string"}
        }
    }
    
    UI_REQUEST_SCHEMA = {
        "type": "object",
        "required": ["request_id", "component_id", "action", "data"],
        "properties": {
            "request_id": {"type": "string"},
            "component_id": {"type": "string"},
            "action": {"type": "string"},
            "data": {"type": "object"},
            "timestamp": {"type": "string"},
            "user_context": {"type": "object"}
        }
    }
    
    UI_RESPONSE_SCHEMA = {
        "type": "object",
        "required": ["request_id", "response_type", "data"],
        "properties": {
            "request_id": {"type": "string"},
            "response_type": {"type": "string"},
            "data": {"type": "object"},
            "message": {"type": "string"},
            "error_details": {"type": "object"},
            "timestamp": {"type": "string"}
        }
    }