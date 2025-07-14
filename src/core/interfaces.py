"""
核心接口定义
提供系统各组件的标准接口
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ProcessingStatus(Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class ModuleType(Enum):
    """模块类型枚举"""
    REASONING = "reasoning"
    MODELS = "models"
    DATA_PROCESSING = "data_processing"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    CONFIGURATION = "configuration"
    EVALUATION = "evaluation"

class ReasoningStep(Enum):
    """推理步骤类型"""
    PARSE = "parse"
    EXTRACT = "extract"
    REASON = "reason"
    CALCULATE = "calculate"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"

@dataclass
class ProcessingResult:
    """处理结果数据类"""
    success: bool
    result: Any
    confidence: float
    processing_time: float
    status: ProcessingStatus
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "status": self.status.value,
            "error_message": self.error_message,
            "metadata": self.metadata or {}
        }

@dataclass
class ReasoningContext:
    """推理上下文"""
    problem_text: str
    problem_type: str
    parameters: Dict[str, Any]
    history: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_text": self.problem_text,
            "problem_type": self.problem_type,
            "parameters": self.parameters,
            "history": self.history,
            "constraints": self.constraints
        }

@dataclass 
class ModuleInfo:
    """模块信息数据类"""
    name: str
    type: 'ModuleType'
    version: str
    description: str
    dependencies: List[str] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}

class IProcessor(ABC):
    """处理器接口"""
    
    @abstractmethod
    def process(self, input_data: Any, context: Optional[ReasoningContext] = None) -> ProcessingResult:
        """处理输入数据"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """获取处理器能力描述"""
        pass

# 基础处理器类
class BaseProcessor(ABC):
    """基础处理器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod 
    def process(self, input_data: Any, context: Optional[ReasoningContext] = None) -> ProcessingResult:
        """处理输入数据"""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据（默认实现）"""
        return input_data is not None
    
    def get_capabilities(self) -> Dict[str, Any]:
        """获取处理器能力描述（默认实现）"""
        return {
            "name": self.__class__.__name__,
            "type": "base_processor",
            "config": self.config
        }

class IValidator(ABC):
    """验证器接口"""
    
    @abstractmethod
    def validate(self, data: Any) -> Dict[str, Any]:
        """验证数据"""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """获取验证规则"""
        pass

class IReasoningEngine(ABC):
    """推理引擎接口"""
    
    @abstractmethod
    def reason(self, problem: str, context: Optional[ReasoningContext] = None) -> ProcessingResult:
        """执行推理"""
        pass
    
    @abstractmethod
    def get_reasoning_steps(self) -> List[Dict[str, Any]]:
        """获取推理步骤"""
        pass
    
    @abstractmethod
    def set_reasoning_strategy(self, strategy: str) -> None:
        """设置推理策略"""
        pass

class ITemplateManager(ABC):
    """模板管理器接口"""
    
    @abstractmethod
    def match_template(self, text: str) -> Optional[Dict[str, Any]]:
        """匹配模板"""
        pass
    
    @abstractmethod
    def get_templates(self) -> List[Dict[str, Any]]:
        """获取所有模板"""
        pass
    
    @abstractmethod
    def add_template(self, template: Dict[str, Any]) -> bool:
        """添加模板"""
        pass

class INumberExtractor(ABC):
    """数字提取器接口"""
    
    @abstractmethod
    def extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """提取数字"""
        pass
    
    @abstractmethod
    def identify_number_patterns(self, text: str) -> List[str]:
        """识别数字模式"""
        pass

class IConfidenceCalculator(ABC):
    """置信度计算器接口"""
    
    @abstractmethod
    def calculate_confidence(self, reasoning_steps: List[Dict[str, Any]], result: Any) -> float:
        """计算置信度"""
        pass
    
    @abstractmethod
    def get_confidence_factors(self) -> List[str]:
        """获取置信度影响因素"""
        pass

class ICacheManager(ABC):
    """缓存管理器接口"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass

class IMonitor(ABC):
    """监控器接口"""
    
    @abstractmethod
    def start_timer(self, name: str) -> str:
        """开始计时"""
        pass
    
    @abstractmethod
    def stop_timer(self, timer_id: str) -> Optional[float]:
        """停止计时"""
        pass
    
    @abstractmethod
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""
        pass
    
    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        pass

class IGNNEnhancer(ABC):
    """GNN增强器接口"""
    
    @abstractmethod
    def enhance_reasoning(self, reasoning_data: Dict[str, Any]) -> Dict[str, Any]:
        """增强推理"""
        pass
    
    @abstractmethod
    def build_graph(self, input_data: Any) -> Any:
        """构建图"""
        pass
    
    @abstractmethod
    def predict(self, graph_data: Any) -> Dict[str, Any]:
        """执行预测"""
        pass

class IConfigManager(ABC):
    """配置管理器接口"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """设置配置"""
        pass
    
    @abstractmethod
    def reload_config(self) -> None:
        """重新加载配置"""
        pass

class IResultFormatter(ABC):
    """结果格式化器接口"""
    
    @abstractmethod
    def format_result(self, result: ProcessingResult, format_type: str = "json") -> str:
        """格式化结果"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        pass

# 组件注册接口
class IComponentRegistry(ABC):
    """组件注册表接口"""
    
    @abstractmethod
    def register_component(self, name: str, component: Any, component_type: str) -> bool:
        """注册组件"""
        pass
    
    @abstractmethod
    def get_component(self, name: str) -> Optional[Any]:
        """获取组件"""
        pass
    
    @abstractmethod
    def list_components(self, component_type: Optional[str] = None) -> List[str]:
        """列出组件"""
        pass
    
    @abstractmethod
    def unregister_component(self, name: str) -> bool:
        """注销组件"""
        pass

# 事件系统接口
class IEventHandler(ABC):
    """事件处理器接口"""
    
    @abstractmethod
    def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """处理事件"""
        pass
    
    @abstractmethod
    def get_supported_events(self) -> List[str]:
        """获取支持的事件类型"""
        pass

class IEventBus(ABC):
    """事件总线接口"""
    
    @abstractmethod
    def subscribe(self, event_type: str, handler: IEventHandler) -> None:
        """订阅事件"""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: str, handler: IEventHandler) -> None:
        """取消订阅"""
        pass
    
    @abstractmethod
    def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """发布事件"""
        pass

# 插件系统接口
class IPlugin(ABC):
    """插件接口"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        pass
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理插件资源"""
        pass

class IPluginManager(ABC):
    """插件管理器接口"""
    
    @abstractmethod
    def load_plugin(self, plugin_path: str) -> bool:
        """加载插件"""
        pass
    
    @abstractmethod
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        pass
    
    @abstractmethod
    def get_loaded_plugins(self) -> List[str]:
        """获取已加载的插件"""
        pass
    
    @abstractmethod
    def get_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """获取插件实例"""
        pass

# 系统健康检查接口
class IHealthChecker(ABC):
    """健康检查器接口"""
    
    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> str:
        """获取健康状态"""
        pass
    
    @abstractmethod
    def register_health_check(self, name: str, check_func: callable) -> None:
        """注册健康检查函数"""
        pass

# 公共API接口
class PublicAPI(ABC):
    """公共API基类"""
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化API"""
        pass
    
    @abstractmethod
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""
        pass
    
    @abstractmethod
    def get_api_info(self) -> Dict[str, Any]:
        """获取API信息"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """关闭API"""
        pass

# 基础编排器接口
class BaseOrchestrator(ABC):
    """基础系统编排器接口"""
    
    @abstractmethod
    def initialize_system(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化系统"""
        pass
    
    @abstractmethod
    def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """解决问题"""
        pass
    
    @abstractmethod
    def batch_solve_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量解决问题"""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        pass
    
    @abstractmethod
    def shutdown_system(self) -> bool:
        """关闭系统"""
        pass 