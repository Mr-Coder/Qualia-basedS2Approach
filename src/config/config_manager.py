"""
COT-DIR 增强配置管理系统

提供分层配置管理、环境隔离、安全加密、动态重载和配置监听功能。
整合原有功能并添加高级特性。
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

import yaml

try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigLevel(Enum):
    """配置级别枚举"""
    DEFAULT = "default"
    ENVIRONMENT = "environment"  
    USER = "user"
    RUNTIME = "runtime"


@dataclass
class ConfigSchema:
    """配置模式定义"""
    key: str
    required: bool = True
    default: Any = None
    validator: Optional[callable] = None
    description: str = ""
    config_level: ConfigLevel = ConfigLevel.DEFAULT


@dataclass
class ConfigSource:
    """配置源定义"""
    name: str
    level: ConfigLevel
    path: Optional[Path] = None
    encrypted: bool = False
    required: bool = False
    reload_interval: Optional[int] = None
    format: str = "yaml"  # yaml, json, env


class EnhancedConfigurationManager:
    """增强配置管理器"""
    
    def __init__(self, env: str = None, config_dir: Union[str, Path] = "config"):
        self.env = env or os.getenv("COT_DIR_ENV", "development")
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 分层配置存储
        self._config_data = {level: {} for level in ConfigLevel}
        self._config_sources = []
        
        # 加密相关
        self.cipher = None
        self._encryption_key = None
        
        # 线程安全
        self._lock = threading.RLock()
        self._reload_threads = {}
        
        # 配置监听器
        self._listeners = []
        self._change_callbacks = {}
        
        # 初始化
        self._init_encryption()
        self._register_default_sources()
        self._load_all_configs()
        
        logger.info(f"增强配置管理器初始化完成，环境: {self.env}")
    
    def _init_encryption(self):
        """初始化配置加密"""
        if not ENCRYPTION_AVAILABLE:
            logger.warning("cryptography包未安装，配置加密功能不可用")
            return
        
        # 尝试从环境变量加载密钥
        encryption_key = os.getenv("COT_CONFIG_KEY")
        if encryption_key:
            try:
                self.cipher = Fernet(encryption_key.encode())
                self._encryption_key = encryption_key.encode()
                logger.info("从环境变量加载配置加密密钥")
                return
            except Exception as e:
                logger.error(f"环境变量配置加密密钥无效: {e}")
        
        # 尝试从文件加载密钥
        key_file = self.config_dir / ".config_key"
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    self._encryption_key = f.read()
                self.cipher = Fernet(self._encryption_key)
                logger.info("从文件加载配置加密密钥")
                return
            except Exception as e:
                logger.error(f"文件配置加密密钥无效: {e}")
        
        # 生成新密钥
        try:
            self._encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self._encryption_key)
            
            # 保存密钥到文件
            with open(key_file, 'wb') as f:
                f.write(self._encryption_key)
            os.chmod(key_file, 0o600)
            
            logger.info("生成新的配置加密密钥")
        except Exception as e:
            logger.error(f"生成配置加密密钥失败: {e}")
    
    def _register_default_sources(self):
        """注册默认配置源"""
        # 默认配置
        self.register_config_source(ConfigSource(
            name="default",
            level=ConfigLevel.DEFAULT,
            path=self.config_dir / "default.yaml",
            required=False
        ))
        
        # 基础配置（兼容性）
        self.register_config_source(ConfigSource(
            name="base",
            level=ConfigLevel.DEFAULT,
            path=self.config_dir / "base.yaml",
            required=False
        ))
        
        # 环境配置
        self.register_config_source(ConfigSource(
            name=f"env_{self.env}",
            level=ConfigLevel.ENVIRONMENT,
            path=self.config_dir / "environments" / f"{self.env}.yaml",
            required=False
        ))
        
        # 用户配置
        self.register_config_source(ConfigSource(
            name="user",
            level=ConfigLevel.USER,
            path=self.config_dir / "user.yaml",
            encrypted=False,
            required=False
        ))
        
        # 安全配置（加密）
        self.register_config_source(ConfigSource(
            name="secure",
            level=ConfigLevel.USER,
            path=self.config_dir / "security" / f"{self.env}_secure.enc",
            encrypted=True,
            required=False
        ))
        
        # 环境变量
        self.register_config_source(ConfigSource(
            name="environment_variables",
            level=ConfigLevel.RUNTIME,
            format="env",
            required=False
        ))
    
    def register_config_source(self, source: ConfigSource):
        """注册配置源"""
        with self._lock:
            self._config_sources.append(source)
            logger.debug(f"注册配置源: {source.name} ({source.level.value})")
            
            # 立即加载配置
            self._load_config_source(source)
            
            # 设置自动重载
            if source.reload_interval:
                self._setup_auto_reload(source)
    
    def _load_all_configs(self):
        """加载所有配置"""
        with self._lock:
            for source in self._config_sources:
                self._load_config_source(source)
    
    def _load_config_source(self, source: ConfigSource):
        """加载单个配置源"""
        try:
            if source.format == "env":
                config_data = self._load_environment_variables()
            elif source.path and source.path.exists():
                config_data = self._load_config_file(source.path, source.format, source.encrypted)
            else:
                if source.required:
                    raise FileNotFoundError(f"必需的配置文件不存在: {source.path}")
                else:
                    logger.debug(f"可选配置文件不存在: {source.path}")
                    return
            
            # 更新配置数据
            with self._lock:
                old_data = self._config_data[source.level].copy()
                if source.level == ConfigLevel.RUNTIME:
                    # 运行时配置直接更新
                    self._config_data[source.level].update(config_data)
                else:
                    # 其他级别使用深度合并
                    self._config_data[source.level] = self._merge_configs(
                        self._config_data[source.level], config_data
                    )
                
                # 通知监听器
                new_data = self._config_data[source.level]
                if old_data != new_data:
                    self._notify_listeners(source.level, config_data)
                
        except Exception as e:
            logger.error(f"加载配置源失败 {source.name}: {e}")
            if source.required:
                raise ConfigurationError(f"加载必需配置源失败 {source.name}: {e}")
    
    def _load_config_file(self, file_path: Path, format: str = "yaml", encrypted: bool = False) -> Dict[str, Any]:
        """加载配置文件"""
        with open(file_path, 'rb' if encrypted else 'r', encoding=None if encrypted else 'utf-8') as f:
            if encrypted:
                if not self.cipher:
                    raise ConfigurationError("配置加密未初始化，无法读取加密配置")
                
                encrypted_data = f.read()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                content = decrypted_data.decode('utf-8')
                
                # 加密文件通常是JSON格式
                return json.loads(content)
            else:
                content = f.read()
        
        if format == "yaml" or file_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(content) or {}
        elif format == "json" or file_path.suffix == '.json':
            return json.loads(content) if content.strip() else {}
        else:
            raise ConfigurationError(f"不支持的配置文件格式: {format}")
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """加载环境变量配置"""
        config = {}
        prefix = "COT_DIR_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 转换键名：COT_DIR_MAX_WORKERS -> max.workers
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # 类型转换
                converted_value = self._convert_env_value(value)
                
                # 支持嵌套键
                self._set_nested_value(config, config_key, converted_value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """转换环境变量值类型"""
        # 布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """设置嵌套值"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _setup_auto_reload(self, source: ConfigSource):
        """设置自动重载"""
        def reload_worker():
            while True:
                time.sleep(source.reload_interval)
                try:
                    old_data = self._config_data[source.level].copy()
                    self._load_config_source(source)
                    new_data = self._config_data[source.level]
                    
                    if old_data != new_data:
                        logger.info(f"配置源 {source.name} 已自动重载")
                except Exception as e:
                    logger.error(f"自动重载配置失败 {source.name}: {e}")
        
        thread = threading.Thread(target=reload_worker, daemon=True, name=f"ConfigReload-{source.name}")
        thread.start()
        self._reload_threads[source.name] = thread
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None, level: Optional[ConfigLevel] = None) -> Any:
        """获取配置值"""
        with self._lock:
            if level:
                # 从指定级别获取
                return self._get_nested_value(self._config_data[level], key, default)
            else:
                # 按优先级顺序获取（RUNTIME > USER > ENVIRONMENT > DEFAULT）
                for config_level in [ConfigLevel.RUNTIME, ConfigLevel.USER, 
                                   ConfigLevel.ENVIRONMENT, ConfigLevel.DEFAULT]:
                    value = self._get_nested_value(self._config_data[config_level], key, None)
                    if value is not None:
                        return value
                return default
    
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any) -> Any:
        """获取嵌套键值"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any, level: ConfigLevel = ConfigLevel.RUNTIME, persist: bool = False):
        """设置配置值"""
        with self._lock:
            old_value = self.get(key)
            
            # 设置值
            self._set_nested_value(self._config_data[level], key, value)
            
            logger.debug(f"设置配置 {key} = {value} (级别: {level.value})")
            
            # 可选持久化
            if persist:
                self._persist_config_level(level)
            
            # 通知监听器
            if old_value != value:
                self._notify_listeners(level, {key: value})
                self._notify_change_callbacks(key, old_value, value)
    
    def _persist_config_level(self, level: ConfigLevel):
        """持久化指定级别的配置"""
        if level == ConfigLevel.RUNTIME:
            logger.warning("运行时配置不支持持久化")
            return
        
        # 找到对应的配置源
        source = None
        for s in self._config_sources:
            if s.level == level and s.path and not s.encrypted:
                source = s
                break
        
        if not source:
            logger.warning(f"未找到级别 {level.value} 的可持久化配置源")
            return
        
        try:
            config_data = self._config_data[level]
            self._save_config_file(source.path, config_data, source.format, source.encrypted)
            logger.info(f"配置级别 {level.value} 已持久化到 {source.path}")
        except Exception as e:
            logger.error(f"持久化配置失败 {level.value}: {e}")
    
    def _save_config_file(self, file_path: Path, data: Dict[str, Any], 
                         format: str = "yaml", encrypted: bool = False):
        """保存配置文件"""
        # 序列化数据
        if format == "json":
            content = json.dumps(data, indent=2, ensure_ascii=False)
        elif format == "yaml":
            content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持保存格式: {format}")
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if encrypted:
            if not self.cipher:
                raise ConfigurationError("配置加密未初始化")
            
            # 加密保存
            encrypted_data = self.cipher.encrypt(content.encode('utf-8'))
            with open(file_path, 'wb') as f:
                f.write(encrypted_data)
        else:
            # 明文保存
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 设置安全权限
        os.chmod(file_path, 0o600)
    
    def get_all(self, level: Optional[ConfigLevel] = None) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            if level:
                return self._config_data[level].copy()
            else:
                # 合并所有级别的配置
                merged = {}
                for config_level in [ConfigLevel.DEFAULT, ConfigLevel.ENVIRONMENT,
                                   ConfigLevel.USER, ConfigLevel.RUNTIME]:
                    merged = self._merge_configs(merged, self._config_data[config_level])
                return merged
    
    def reload_config(self):
        """重新加载所有配置"""
        logger.info("重新加载所有配置...")
        
        with self._lock:
            # 清空现有配置
            for level in ConfigLevel:
                self._config_data[level].clear()
            
            # 重新加载所有源
            self._load_all_configs()
        
        logger.info("配置重载完成")
    
    def add_listener(self, callback: Callable[[ConfigLevel, Dict[str, Any]], None]):
        """添加配置变更监听器"""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[ConfigLevel, Dict[str, Any]], None]):
        """移除配置变更监听器"""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def add_change_callback(self, key: str, callback: Callable[[str, Any, Any], None]):
        """添加特定键的变更回调"""
        if key not in self._change_callbacks:
            self._change_callbacks[key] = []
        self._change_callbacks[key].append(callback)
    
    def remove_change_callback(self, key: str, callback: Callable[[str, Any, Any], None]):
        """移除特定键的变更回调"""
        if key in self._change_callbacks and callback in self._change_callbacks[key]:
            self._change_callbacks[key].remove(callback)
    
    def _notify_listeners(self, level: ConfigLevel, changes: Dict[str, Any]):
        """通知配置变更监听器"""
        for callback in self._listeners:
            try:
                callback(level, changes)
            except Exception as e:
                logger.error(f"配置监听器回调失败: {e}")
    
    def _notify_change_callbacks(self, key: str, old_value: Any, new_value: Any):
        """通知特定键的变更回调"""
        if key in self._change_callbacks:
            for callback in self._change_callbacks[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"配置变更回调失败 {key}: {e}")
    
    @contextmanager
    def override(self, overrides: Dict[str, Any]):
        """临时覆盖配置"""
        # 保存原配置
        original = {}
        for key in overrides:
            original[key] = self.get(key)
        
        try:
            # 应用覆盖
            for key, value in overrides.items():
                self.set(key, value, ConfigLevel.RUNTIME)
            yield
        finally:
            # 恢复原配置
            for key, value in original.items():
                if value is not None:
                    self.set(key, value, ConfigLevel.RUNTIME)
                else:
                    # 删除键
                    self._delete_key(key, ConfigLevel.RUNTIME)
    
    def _delete_key(self, key: str, level: ConfigLevel):
        """删除配置键"""
        keys = key.split('.')
        current = self._config_data[level]
        
        for k in keys[:-1]:
            if k in current:
                current = current[k]
            else:
                return  # 键不存在
        
        if keys[-1] in current:
            del current[keys[-1]]
    
    def validate_config(self, schema: Dict[str, ConfigSchema]) -> bool:
        """验证配置"""
        errors = []
        
        for key, schema_item in schema.items():
            value = self.get(key)
            
            # 检查必需项
            if schema_item.required and value is None:
                if schema_item.default is not None:
                    # 使用默认值
                    self.set(key, schema_item.default, schema_item.config_level)
                    value = schema_item.default
                else:
                    errors.append(f"必需配置项缺失: {key}")
                    continue
            
            # 运行验证器
            if value is not None and schema_item.validator:
                try:
                    if not schema_item.validator(value):
                        errors.append(f"配置验证失败: {key} = {value}")
                except Exception as e:
                    errors.append(f"配置验证器错误 {key}: {str(e)}")
        
        if errors:
            raise ConfigurationError("配置验证失败: " + "; ".join(errors))
        
        return True
    
    def encrypt_and_save_secure_config(self, config: Dict[str, Any]):
        """加密并保存敏感配置"""
        if not self.cipher:
            raise ConfigurationError("未初始化加密密钥，无法保存敏感配置")
        
        try:
            secure_file = self.config_dir / "security" / f"{self.env}_secure.enc"
            self._save_config_file(secure_file, config, "json", encrypted=True)
            logger.info("敏感配置已加密保存")
        except Exception as e:
            raise ConfigurationError(f"保存敏感配置失败: {str(e)}", cause=e)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要（屏蔽敏感信息）"""
        def mask_sensitive(obj, path=""):
            """递归屏蔽敏感信息"""
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if any(sensitive in key.lower() for sensitive in ['password', 'key', 'secret', 'token']):
                        result[key] = "***MASKED***"
                    else:
                        result[key] = mask_sensitive(value, current_path)
                return result
            elif isinstance(obj, list):
                return [mask_sensitive(item, path) for item in obj]
            else:
                return obj
        
        return {
            "environment": self.env,
            "config_levels": {level.value: mask_sensitive(data) 
                            for level, data in self._config_data.items()},
            "sources_count": len(self._config_sources),
            "listeners_count": len(self._listeners),
            "encryption_available": self.cipher is not None
        }
    
    def create_default_configs(self):
        """创建默认配置文件"""
        default_config = {
            "system": {
                "max_workers": 4,
                "timeout": 30,
                "log_level": "INFO"
            },
            "orchestration": {
                "strategy": "unified",
                "max_concurrent_requests": 20,
                "enable_recovery": True
            },
            "reasoning": {
                "default_strategy": "chain_of_thought",
                "confidence_threshold": 0.7,
                "max_reasoning_steps": 10
            },
            "models": {
                "cache": {
                    "enabled": True,
                    "max_size": 1000,
                    "ttl": 3600
                },
                "performance": {
                    "enable_monitoring": True,
                    "metrics_retention": 86400
                }
            },
            "security": {
                "enable_input_validation": True,
                "enable_output_filtering": True,
                "max_expression_length": 1000
            }
        }
        
        # 环境特定配置
        dev_config = {
            "system": {"log_level": "DEBUG"},
            "models": {"cache": {"enabled": False}}
        }
        
        prod_config = {
            "system": {"max_workers": 8, "log_level": "WARNING"},
            "security": {"enable_input_validation": True}
        }
        
        # 保存配置文件
        configs = [
            (self.config_dir / "default.yaml", default_config),
            (self.config_dir / "environments" / "development.yaml", dev_config),
            (self.config_dir / "environments" / "production.yaml", prod_config)
        ]
        
        for file_path, config_data in configs:
            if not file_path.exists():
                self._save_config_file(file_path, config_data, "yaml")
                logger.info(f"创建默认配置文件: {file_path}")


# 兼容性：保留原有的ConfigurationManager类
class ConfigurationManager(EnhancedConfigurationManager):
    """配置管理器（兼容性类）"""
    
    def __init__(self, env: str = None, config_dir: str = "config"):
        super().__init__(env, config_dir)


# 全局配置实例
_global_config = None
_config_lock = threading.Lock()


def get_config() -> EnhancedConfigurationManager:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = EnhancedConfigurationManager()
    return _global_config


def init_config(env: str = None, config_dir: str = "config") -> EnhancedConfigurationManager:
    """初始化全局配置"""
    global _global_config
    with _config_lock:
        _global_config = EnhancedConfigurationManager(env=env, config_dir=config_dir)
    return _global_config


def get_config_value(key: str, default: Any = None) -> Any:
    """获取配置值的便利函数"""
    return get_config().get(key, default)


def set_config_value(key: str, value: Any, persist: bool = False):
    """设置配置值的便利函数"""
    get_config().set(key, value, persist=persist)


@contextmanager
def config_override(**overrides):
    """配置覆盖上下文管理器"""
    with get_config().override(overrides):
        yield


# 增强配置模式定义
ENHANCED_CONFIG_SCHEMA = {
    "system.max_workers": ConfigSchema(
        key="system.max_workers",
        required=True,
        default=4,
        validator=lambda x: isinstance(x, int) and 1 <= x <= 32,
        description="系统最大工作线程数",
        config_level=ConfigLevel.ENVIRONMENT
    ),
    "system.timeout": ConfigSchema(
        key="system.timeout",
        required=True,
        default=30,
        validator=lambda x: isinstance(x, (int, float)) and x > 0,
        description="系统超时时间（秒）",
        config_level=ConfigLevel.ENVIRONMENT
    ),
    "orchestration.strategy": ConfigSchema(
        key="orchestration.strategy",
        required=True,
        default="unified",
        validator=lambda x: x in ["unified", "reasoning", "processing"],
        description="协调器策略",
        config_level=ConfigLevel.ENVIRONMENT
    ),
    "reasoning.confidence_threshold": ConfigSchema(
        key="reasoning.confidence_threshold", 
        required=True,
        default=0.7,
        validator=lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
        description="推理置信度阈值",
        config_level=ConfigLevel.USER
    ),
    "security.enable_input_validation": ConfigSchema(
        key="security.enable_input_validation",
        required=True,
        default=True,
        validator=lambda x: isinstance(x, bool),
        description="是否启用输入验证",
        config_level=ConfigLevel.ENVIRONMENT
    )
}


# 原有模式定义保持兼容性
REASONING_CONFIG_SCHEMA = {
    "reasoning.max_steps": ConfigSchema(
        key="reasoning.max_steps",
        required=True,
        default=15,
        validator=lambda x: isinstance(x, int) and 1 <= x <= 100,
        description="最大推理步数"
    ),
    "reasoning.confidence_threshold": ConfigSchema(
        key="reasoning.confidence_threshold", 
        required=True,
        default=0.7,
        validator=lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
        description="置信度阈值"
    ),
    "reasoning.timeout_seconds": ConfigSchema(
        key="reasoning.timeout_seconds",
        required=True,
        default=30.0,
        validator=lambda x: isinstance(x, (int, float)) and x > 0,
        description="推理超时时间（秒）"
    ),
    "validation.max_input_length": ConfigSchema(
        key="validation.max_input_length",
        required=True,
        default=10000,
        validator=lambda x: isinstance(x, int) and x > 0,
        description="最大输入长度"
    ),
    "performance.enable_monitoring": ConfigSchema(
        key="performance.enable_monitoring",
        required=False,
        default=True,
        validator=lambda x: isinstance(x, bool),
        description="是否启用性能监控"
    )
} 