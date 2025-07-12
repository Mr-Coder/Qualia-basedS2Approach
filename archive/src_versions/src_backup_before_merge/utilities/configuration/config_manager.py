"""
AI协作友好的配置管理器

这个模块提供了一个灵活的配置管理系统，让AI助手可以轻松理解和修改系统配置。

AI_CONTEXT: 配置管理是系统的核心基础设施
RESPONSIBILITY: 提供统一的配置加载、验证和管理功能
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...ai_core.interfaces import ConfigurationError

# 可选的YAML支持
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class ConfigurationSchema:
    """
    配置模式定义 - AI可以理解的配置结构
    
    AI_CONTEXT: 定义配置项的类型、默认值和验证规则
    RESPONSIBILITY: 确保配置的类型安全和完整性
    """
    
    name: str = field(metadata={"ai_hint": "配置项名称"})
    type: str = field(metadata={"ai_hint": "数据类型: str/int/float/bool/list/dict"})
    default: Any = field(metadata={"ai_hint": "默认值"})
    required: bool = field(default=True, metadata={"ai_hint": "是否必需"})
    description: str = field(default="", metadata={"ai_hint": "配置项描述"})
    validation_rules: List[str] = field(
        default_factory=list,
        metadata={"ai_hint": "验证规则列表"}
    )
    
    def validate_value(self, value: Any) -> bool:
        """
        验证配置值是否符合模式要求
        
        Args:
            value: 待验证的值
            
        Returns:
            bool: 验证是否通过
            
        AI_HINT: 根据类型和规则验证配置值
        """
        # 类型检查
        if self.type == "str" and not isinstance(value, str):
            return False
        elif self.type == "int" and not isinstance(value, int):
            return False
        elif self.type == "float" and not isinstance(value, (int, float)):
            return False
        elif self.type == "bool" and not isinstance(value, bool):
            return False
        elif self.type == "list" and not isinstance(value, list):
            return False
        elif self.type == "dict" and not isinstance(value, dict):
            return False
        
        # 自定义验证规则
        for rule in self.validation_rules:
            if not self._apply_validation_rule(value, rule):
                return False
        
        return True
    
    def _apply_validation_rule(self, value: Any, rule: str) -> bool:
        """应用验证规则"""
        # AI_HINT: 这里可以扩展更多验证规则
        if rule.startswith("min_length:"):
            min_len = int(rule.split(":")[1])
            return len(str(value)) >= min_len
        elif rule.startswith("max_length:"):
            max_len = int(rule.split(":")[1])
            return len(str(value)) <= max_len
        elif rule.startswith("range:"):
            min_val, max_val = map(float, rule.split(":")[1].split(","))
            return min_val <= float(value) <= max_val
        
        return True


class AICollaborativeConfigManager:
    """
    AI协作友好的配置管理器
    
    AI_CONTEXT: 统一的配置管理入口，支持多种配置格式
    RESPONSIBILITY: 加载、验证、管理系统配置
    
    AI_INSTRUCTION: 使用这个类来管理系统配置：
    1. 定义配置模式 - 使用 define_schema()
    2. 加载配置文件 - 使用 load_config()
    3. 获取配置值 - 使用 get()
    4. 验证配置 - 自动验证或使用 validate()
    """
    
    def __init__(self, config_name: str = "default"):
        """
        初始化配置管理器
        
        Args:
            config_name: 配置名称，用于标识不同的配置实例
            
        AI_HINT: 可以创建多个配置实例管理不同模块的配置
        """
        self.config_name = config_name
        self.config_data: Dict[str, Any] = {}
        self.schema: Dict[str, ConfigurationSchema] = {}
        self.config_file_path: Optional[Path] = None
        
        # AI友好的默认配置模式
        self._setup_default_schema()
    
    def _setup_default_schema(self) -> None:
        """设置默认的配置模式"""
        default_schemas = [
            ConfigurationSchema(
                name="logging.level",
                type="str",
                default="INFO",
                description="日志级别设置",
                validation_rules=["choices:DEBUG,INFO,WARNING,ERROR,CRITICAL"]
            ),
            ConfigurationSchema(
                name="reasoning.max_steps",
                type="int",
                default=10,
                description="推理最大步数",
                validation_rules=["range:1,100"]
            ),
            ConfigurationSchema(
                name="reasoning.confidence_threshold",
                type="float",
                default=0.8,
                description="推理置信度阈值",
                validation_rules=["range:0.0,1.0"]
            ),
            ConfigurationSchema(
                name="performance.enable_tracking",
                type="bool",
                default=True,
                description="是否启用性能跟踪"
            )
        ]
        
        for schema in default_schemas:
            self.schema[schema.name] = schema
    
    def define_schema(self, schema: ConfigurationSchema) -> None:
        """
        定义配置项模式
        
        Args:
            schema: 配置模式定义
            
        AI_HINT: 使用此方法添加新的配置项定义
        """
        self.schema[schema.name] = schema
    
    def load_config(self, file_path: Union[str, Path]) -> None:
        """
        加载配置文件
        
        Args:
            file_path: 配置文件路径，支持 .json 和 .yaml/.yml 格式
            
        Raises:
            ConfigurationError: 配置文件格式错误或验证失败
            
        AI_HINT: 自动检测文件格式并加载配置
        """
        self.config_file_path = Path(file_path)
        
        if not self.config_file_path.exists():
            raise ConfigurationError(
                f"配置文件不存在: {file_path}",
                config_file=str(file_path),
                suggestions=["检查文件路径是否正确", "确认文件是否存在"]
            )
        
        try:
            if self.config_file_path.suffix.lower() == '.json':
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            elif self.config_file_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ConfigurationError(
                        "YAML支持不可用，请安装PyYAML: pip install PyYAML",
                        config_file=str(file_path),
                        suggestions=["安装PyYAML", "使用JSON格式配置文件"]
                    )
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f)
            else:
                suggestions = ["使用 .json 格式"]
                if YAML_AVAILABLE:
                    suggestions.append("使用 .yaml/.yml 格式")
                raise ConfigurationError(
                    f"不支持的配置文件格式: {self.config_file_path.suffix}",
                    config_file=str(file_path),
                    suggestions=suggestions
                )
        except Exception as e:
            raise ConfigurationError(
                f"配置文件加载失败: {str(e)}",
                config_file=str(file_path),
                context={"original_error": str(e)}
            )
        
        # 验证配置
        self.validate()
        
        # 应用默认值
        self._apply_defaults()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持嵌套键如 'section.subsection.key'
            default: 默认值
            
        Returns:
            配置值或默认值
            
        AI_HINT: 使用点号分隔的键名访问嵌套配置
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            # 尝试从模式获取默认值
            if key in self.schema:
                return self.schema[key].default
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            
        Raises:
            ConfigurationError: 配置值验证失败
            
        AI_HINT: 设置值时会自动进行验证
        """
        # 验证值
        if key in self.schema:
            if not self.schema[key].validate_value(value):
                raise ConfigurationError(
                    f"配置值验证失败: {key} = {value}",
                    config_key=key,
                    expected_type=self.schema[key].type,
                    actual_value=value
                )
        
        # 设置嵌套值
        keys = key.split('.')
        data = self.config_data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value
    
    def validate(self) -> List[str]:
        """
        验证所有配置项
        
        Returns:
            List[str]: 验证错误列表，空列表表示验证通过
            
        AI_HINT: 全面验证配置的完整性和正确性
        """
        errors = []
        
        for schema_key, schema in self.schema.items():
            value = self.get(schema_key)
            
            # 检查必需项
            if schema.required and value is None:
                errors.append(f"缺少必需配置项: {schema_key}")
                continue
            
            # 验证值
            if value is not None and not schema.validate_value(value):
                errors.append(f"配置值验证失败: {schema_key} = {value}")
        
        if errors:
            raise ConfigurationError(
                f"配置验证失败，发现 {len(errors)} 个错误",
                context={"validation_errors": errors}
            )
        
        return errors
    
    def _apply_defaults(self) -> None:
        """应用默认值到缺失的配置项"""
        for schema_key, schema in self.schema.items():
            if self.get(schema_key) is None and not schema.required:
                self.set(schema_key, schema.default)
    
    def save_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        保存配置到文件
        
        Args:
            file_path: 可选的保存路径，默认使用加载时的路径
            
        AI_HINT: 将当前配置保存到文件
        """
        save_path = Path(file_path) if file_path else self.config_file_path
        
        if not save_path:
            raise ConfigurationError(
                "未指定保存路径且没有加载过配置文件",
                suggestions=["指定保存路径参数"]
            )
        
        try:
            if save_path.suffix.lower() == '.json':
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            elif save_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ConfigurationError(
                        "YAML支持不可用，无法保存YAML格式文件",
                        config_file=str(save_path),
                        suggestions=["安装PyYAML", "使用JSON格式保存"]
                    )
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
        except Exception as e:
            raise ConfigurationError(
                f"配置文件保存失败: {str(e)}",
                config_file=str(save_path),
                context={"original_error": str(e)}
            )
    
    def get_ai_friendly_summary(self) -> Dict[str, Any]:
        """
        获取AI友好的配置摘要
        
        Returns:
            Dict: 包含配置状态、模式和当前值的摘要
            
        AI_HINT: 使用此方法获取配置的完整概览
        """
        return {
            "config_name": self.config_name,
            "config_file": str(self.config_file_path) if self.config_file_path else None,
            "schema_count": len(self.schema),
            "config_items": {
                key: {
                    "value": self.get(key),
                    "type": schema.type,
                    "required": schema.required,
                    "description": schema.description
                }
                for key, schema in self.schema.items()
            },
            "validation_status": "valid",  # 如果到这里说明验证通过
            "ai_instructions": [
                "使用 get(key) 获取配置值",
                "使用 set(key, value) 设置配置值",
                "使用 validate() 验证配置完整性",
                "使用 save_config() 保存配置更改"
            ]
        }


# AI_HELPER: 便捷函数
def create_default_config_manager() -> AICollaborativeConfigManager:
    """
    创建默认的配置管理器实例
    
    Returns:
        AICollaborativeConfigManager: 配置好默认模式的管理器
        
    AI_HINT: 快速创建配置管理器的便捷方法
    """
    return AICollaborativeConfigManager("ai_collaborative_system")


def create_sample_config_file(file_path: Union[str, Path]) -> None:
    """
    创建示例配置文件
    
    Args:
        file_path: 配置文件保存路径
        
    AI_HINT: 生成一个包含所有默认配置的示例文件
    """
    sample_config = {
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "reasoning": {
            "max_steps": 10,
            "confidence_threshold": 0.8,
            "strategies": [
                {
                    "name": "algebraic_solver",
                    "priority": 1,
                    "enabled": True
                },
                {
                    "name": "geometric_solver",
                    "priority": 2,
                    "enabled": True
                }
            ]
        },
        "performance": {
            "enable_tracking": True,
            "metrics_collection_interval": 60,
            "max_memory_usage": 1000
        },
        "experimental": {
            "enable_advanced_features": False,
            "debug_mode": False
        }
    }
    
    path = Path(file_path)
    
    if path.suffix.lower() == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
    elif path.suffix.lower() in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            # 如果没有YAML支持，则创建JSON文件
            json_path = path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, indent=2, ensure_ascii=False)
            print(f"注意: YAML不可用，已创建JSON配置文件: {json_path}")
        else:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(sample_config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)


# AI_HINT: 配置管理器使用指南
"""
AI_USAGE_GUIDE:

这个配置管理器设计为AI友好，具有以下特点：

1. 类型安全：每个配置项都有明确的类型定义
2. 自动验证：加载和设置配置时自动验证
3. 默认值：为所有配置项提供合理的默认值
4. 嵌套支持：支持使用点号访问嵌套配置
5. 多格式：支持JSON和YAML配置文件
6. 错误友好：提供详细的错误信息和修复建议

示例用法：
```python
# 创建配置管理器
config = create_default_config_manager()

# 加载配置文件
config.load_config("config.yaml")

# 获取配置值
log_level = config.get("logging.level")
max_steps = config.get("reasoning.max_steps")

# 设置配置值
config.set("reasoning.confidence_threshold", 0.9)

# 保存配置
config.save_config()
```
""" 