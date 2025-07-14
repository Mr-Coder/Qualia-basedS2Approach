
# 安全改进：导入安全计算器
import sys
import os
sys.path.append(os.path.dirname(__file__))
from secure_components import SecureMathEvaluator, SecurityError

# 初始化安全计算器
_secure_evaluator = SecureMathEvaluator()

"""
🔧 Scalable Architecture - 高度可扩展架构
模块化设计，插件系统，动态扩展能力
"""

import abc
import importlib
import inspect
import json
import logging
import threading
import weakref
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Type, Union


class ModuleType(Enum):
    """模块类型"""
    PROCESSOR = "处理器"
    EVALUATOR = "评估器"
    CLASSIFIER = "分类器"
    SOLVER = "求解器"
    ANALYZER = "分析器"
    EXTRACTOR = "提取器"
    VALIDATOR = "验证器"
    OPTIMIZER = "优化器"


class PluginStatus(Enum):
    """插件状态"""
    REGISTERED = "已注册"
    LOADED = "已加载"
    ACTIVE = "活跃"
    INACTIVE = "非活跃"
    ERROR = "错误"


@dataclass
class PluginInfo:
    """插件信息"""
    plugin_id: str
    name: str
    version: str
    description: str
    module_type: ModuleType
    author: str = ""
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None
    status: PluginStatus = PluginStatus.REGISTERED
    load_order: int = 100
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config_schema is None:
            self.config_schema = {}


class ProcessorProtocol(Protocol):
    """处理器协议"""
    
    def process(self, input_data: Any, config: Dict[str, Any] = None) -> Any:
        """处理数据"""
        ...
    
    def get_info(self) -> PluginInfo:
        """获取插件信息"""
        ...


class EvaluatorProtocol(Protocol):
    """评估器协议"""
    
    def evaluate(self, input_data: Any, expected_output: Any = None) -> Dict[str, float]:
        """评估结果"""
        ...


class BasePlugin(abc.ABC):
    """插件基类"""
    
    @abc.abstractmethod
    def get_info(self) -> PluginInfo:
        """获取插件信息"""
        pass
    
    def initialize(self, config: Dict[str, Any] = None):
        """初始化插件"""
        pass
    
    def cleanup(self):
        """清理资源"""
        pass


class PluginManager:
    """🔧 插件管理器"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        """
        初始化插件管理器
        
        Args:
            plugins_dir: 插件目录
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(exist_ok=True)
        
        # 插件注册表
        self.registered_plugins: Dict[str, PluginInfo] = {}
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.plugin_instances: Dict[str, Any] = {}
        
        # 类型映射
        self.type_registry: Dict[ModuleType, List[str]] = defaultdict(list)
        
        # 配置
        self.config: Dict[str, Any] = {}
        
        # 事件系统
        self.event_handlers: Dict[str, List[callable]] = defaultdict(list)
        
        # 设置日志
        self.logger = logging.getLogger('PluginManager')
        
        print(f"🔧 插件管理器已初始化，插件目录: {self.plugins_dir}")
    
    def register_plugin(self, plugin_class: Type[BasePlugin], 
                       config: Dict[str, Any] = None) -> bool:
        """
        📋 注册插件
        
        Args:
            plugin_class: 插件类
            config: 插件配置
            
        Returns:
            是否注册成功
        """
        try:
            # 创建临时实例获取信息
            temp_instance = plugin_class()
            plugin_info = temp_instance.get_info()
            
            # 检查依赖
            if not self._check_dependencies(plugin_info.dependencies):
                self.logger.warning(f"插件 {plugin_info.name} 依赖未满足")
                return False
            
            # 注册插件
            self.registered_plugins[plugin_info.plugin_id] = plugin_info
            self.type_registry[plugin_info.module_type].append(plugin_info.plugin_id)
            
            # 保存插件类引用
            setattr(temp_instance, '_plugin_class', plugin_class)
            self.plugin_instances[plugin_info.plugin_id] = plugin_class
            
            self.logger.info(f"✅ 注册插件: {plugin_info.name} (ID: {plugin_info.plugin_id})")
            
            # 触发事件
            self._trigger_event('plugin_registered', plugin_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 注册插件失败: {e}")
            return False
    
    def load_plugin(self, plugin_id: str, config: Dict[str, Any] = None) -> bool:
        """
        🔄 加载插件
        
        Args:
            plugin_id: 插件ID
            config: 插件配置
            
        Returns:
            是否加载成功
        """
        if plugin_id not in self.registered_plugins:
            self.logger.error(f"插件 {plugin_id} 未注册")
            return False
        
        if plugin_id in self.loaded_plugins:
            self.logger.warning(f"插件 {plugin_id} 已加载")
            return True
        
        try:
            plugin_info = self.registered_plugins[plugin_id]
            plugin_class = self.plugin_instances[plugin_id]
            
            # 创建插件实例
            plugin_instance = plugin_class()
            
            # 初始化插件
            if config is None:
                config = self.config.get(plugin_id, {})
            
            plugin_instance.initialize(config)
            
            # 保存实例
            self.loaded_plugins[plugin_id] = plugin_instance
            
            # 更新状态
            plugin_info.status = PluginStatus.LOADED
            
            self.logger.info(f"✅ 加载插件: {plugin_info.name}")
            
            # 触发事件
            self._trigger_event('plugin_loaded', plugin_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 加载插件失败 {plugin_id}: {e}")
            if plugin_id in self.registered_plugins:
                self.registered_plugins[plugin_id].status = PluginStatus.ERROR
            return False
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """
        ⏹️ 卸载插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            是否卸载成功
        """
        if plugin_id not in self.loaded_plugins:
            return True
        
        try:
            plugin_instance = self.loaded_plugins[plugin_id]
            plugin_info = self.registered_plugins[plugin_id]
            
            # 清理插件
            plugin_instance.cleanup()
            
            # 移除实例
            del self.loaded_plugins[plugin_id]
            
            # 更新状态
            plugin_info.status = PluginStatus.REGISTERED
            
            self.logger.info(f"⏹️ 卸载插件: {plugin_info.name}")
            
            # 触发事件
            self._trigger_event('plugin_unloaded', plugin_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 卸载插件失败 {plugin_id}: {e}")
            return False
    
    def get_plugins_by_type(self, module_type: ModuleType) -> List[str]:
        """📋 按类型获取插件"""
        return self.type_registry.get(module_type, [])
    
    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """🔍 获取插件实例"""
        return self.loaded_plugins.get(plugin_id)
    
    def execute_plugin(self, plugin_id: str, method: str, *args, **kwargs) -> Any:
        """▶️ 执行插件方法"""
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            raise ValueError(f"插件 {plugin_id} 未加载")
        
        if not hasattr(plugin, method):
            raise AttributeError(f"插件 {plugin_id} 没有方法 {method}")
        
        return getattr(plugin, method)(*args, **kwargs)
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """检查依赖"""
        for dep in dependencies:
            if dep not in self.registered_plugins:
                return False
        return True
    
    def _trigger_event(self, event_name: str, data: Any):
        """触发事件"""
        for handler in self.event_handlers.get(event_name, []):
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"事件处理器错误: {e}")
    
    def add_event_handler(self, event_name: str, handler: callable):
        """添加事件处理器"""
        self.event_handlers[event_name].append(handler)
    
    def get_registry_info(self) -> Dict[str, Any]:
        """📊 获取注册表信息"""
        return {
            "total_registered": len(self.registered_plugins),
            "total_loaded": len(self.loaded_plugins),
            "by_type": {mtype.value: len(plugins) for mtype, plugins in self.type_registry.items()},
            "by_status": {status.value: sum(1 for p in self.registered_plugins.values() 
                                          if p.status == status) for status in PluginStatus}
        }


class ModularFramework:
    """🏗️ 模块化框架"""
    
    def __init__(self):
        """初始化模块化框架"""
        self.plugin_manager = PluginManager()
        self.pipelines: Dict[str, List[str]] = {}
        self.configurations: Dict[str, Dict[str, Any]] = {}
        
        # 注册内置插件
        self._register_builtin_plugins()
        
        print("🏗️ 模块化框架已初始化")
    
    def _register_builtin_plugins(self):
        """注册内置插件"""
        # 这里可以注册一些内置的基础插件
        pass
    
    def create_pipeline(self, pipeline_name: str, plugin_sequence: List[str]) -> bool:
        """
        🔗 创建处理管道
        
        Args:
            pipeline_name: 管道名称
            plugin_sequence: 插件序列
            
        Returns:
            是否创建成功
        """
        # 验证插件存在
        for plugin_id in plugin_sequence:
            if plugin_id not in self.plugin_manager.registered_plugins:
                print(f"❌ 插件 {plugin_id} 未注册")
                return False
        
        self.pipelines[pipeline_name] = plugin_sequence
        print(f"🔗 创建管道: {pipeline_name} -> {' -> '.join(plugin_sequence)}")
        return True
    
    def execute_pipeline(self, pipeline_name: str, input_data: Any, 
                        config: Dict[str, Any] = None) -> Any:
        """
        ▶️ 执行处理管道
        
        Args:
            pipeline_name: 管道名称
            input_data: 输入数据
            config: 配置
            
        Returns:
            处理结果
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"管道 {pipeline_name} 不存在")
        
        plugin_sequence = self.pipelines[pipeline_name]
        current_data = input_data
        
        print(f"▶️ 执行管道: {pipeline_name}")
        
        for i, plugin_id in enumerate(plugin_sequence):
            try:
                # 加载插件（如果未加载）
                if plugin_id not in self.plugin_manager.loaded_plugins:
                    self.plugin_manager.load_plugin(plugin_id)
                
                # 获取插件配置
                plugin_config = {}
                if config and plugin_id in config:
                    plugin_config = config[plugin_id]
                
                # 执行插件
                print(f"  步骤 {i+1}: {plugin_id}")
                current_data = self.plugin_manager.execute_plugin(
                    plugin_id, 'process', current_data, plugin_config
                )
                
            except Exception as e:
                print(f"❌ 管道执行失败在步骤 {i+1} ({plugin_id}): {e}")
                raise
        
        print(f"✅ 管道执行完成: {pipeline_name}")
        return current_data
    
    def register_processor(self, processor_class: Type[BasePlugin]):
        """注册处理器"""
        return self.plugin_manager.register_plugin(processor_class)
    
    def get_available_processors(self) -> List[str]:
        """获取可用处理器"""
        return self.plugin_manager.get_plugins_by_type(ModuleType.PROCESSOR)
    
    def save_configuration(self, config_path: str):
        """💾 保存配置"""
        config_data = {
            "pipelines": self.pipelines,
            "configurations": self.configurations,
            "plugins": {pid: asdict(info) for pid, info in self.plugin_manager.registered_plugins.items()}
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 配置已保存到: {config_path}")
    
    def load_configuration(self, config_path: str):
        """📂 加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.pipelines = config_data.get("pipelines", {})
        self.configurations = config_data.get("configurations", {})
        
        print(f"📂 配置已从 {config_path} 加载")


# 示例插件实现
class SimpleArithmeticProcessor(BasePlugin):
    """简单算术处理器插件示例"""
    
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            plugin_id="simple_arithmetic",
            name="简单算术处理器",
            version="1.0.0",
            description="处理基本的算术表达式",
            module_type=ModuleType.PROCESSOR,
            author="System"
        )
    
    def process(self, input_data: Any, config: Dict[str, Any] = None) -> Any:
        """处理算术表达式"""
        if isinstance(input_data, str):
            try:
                # 简单的表达式计算
                result = _secure_evaluator.safe__secure_evaluator.safe_eval(input_data)
                return {
                    "expression": input_data,
                    "result": result,
                    "success": True
                }
            except:
                return {
                    "expression": input_data,
                    "result": None,
                    "success": False,
                    "error": "计算失败"
                }
        
        return input_data


class ProblemClassifierProcessor(BasePlugin):
    """问题分类处理器插件示例"""
    
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            plugin_id="problem_classifier",
            name="问题分类器",
            version="1.0.0",
            description="对数学问题进行分类",
            module_type=ModuleType.CLASSIFIER,
            author="System"
        )
    
    def process(self, input_data: Any, config: Dict[str, Any] = None) -> Any:
        """分类数学问题"""
        if isinstance(input_data, dict) and 'expression' in input_data:
            expression = input_data['expression']
            
            # 简单的分类逻辑
            if '+' in expression or '-' in expression:
                problem_type = "加减运算"
            elif '*' in expression or '/' in expression:
                problem_type = "乘除运算"
            else:
                problem_type = "其他"
            
            input_data['problem_type'] = problem_type
            input_data['classification_confidence'] = 0.8
        
        return input_data


# 使用示例
def demo_scalable_architecture():
    """演示可扩展架构"""
    print("🔧 Scalable Architecture Demo")
    print("=" * 50)
    
    # 创建框架
    framework = ModularFramework()
    
    # 注册插件
    framework.register_processor(SimpleArithmeticProcessor)
    framework.register_processor(ProblemClassifierProcessor)
    
    # 创建处理管道
    framework.create_pipeline("math_pipeline", [
        "simple_arithmetic",
        "problem_classifier"
    ])
    
    # 测试数据
    test_expressions = ["2 + 3", "5 * 4", "10 / 2"]
    
    print(f"\n🧪 测试处理管道:")
    for expr in test_expressions:
        print(f"\n输入: {expr}")
        result = framework.execute_pipeline("math_pipeline", expr)
        print(f"输出: {result}")
    
    # 显示框架信息
    registry_info = framework.plugin_manager.get_registry_info()
    print(f"\n📊 插件注册表信息:")
    for key, value in registry_info.items():
        print(f"  {key}: {value}")
    
    return framework


if __name__ == "__main__":
    demo_scalable_architecture() 