"""
模板加载器
从外部文件加载模板，支持热重载
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .template_registry import TemplateDefinition, TemplateRegistry


class TemplateLoader:
    """模板加载器"""
    
    def __init__(self, registry: TemplateRegistry):
        """
        初始化模板加载器
        
        Args:
            registry: 模板注册表
        """
        self.registry = registry
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 加载配置
        self.template_dirs = ["config/templates", "templates", "data/templates"]
        self.supported_formats = {'.json', '.yaml', '.yml'}
        
        # 文件监控
        self.file_timestamps = {}
        self.last_scan = 0
        self.scan_interval = 60  # 60秒扫描一次
        
        # 加载统计
        self.load_stats = {
            "files_loaded": 0,
            "templates_loaded": 0,
            "last_load": None,
            "errors": []
        }
    
    def load_external_templates(self) -> int:
        """
        加载外部模板文件
        
        Returns:
            加载的模板数量
        """
        try:
            loaded_count = 0
            
            # 扫描所有模板目录
            for template_dir in self.template_dirs:
                dir_path = Path(template_dir)
                if dir_path.exists():
                    loaded_count += self._load_templates_from_directory(dir_path)
            
            self.load_stats["last_load"] = time.time()
            self.load_stats["templates_loaded"] += loaded_count
            
            self.logger.info(f"加载外部模板完成: {loaded_count} 个模板")
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"加载外部模板失败: {e}")
            self.load_stats["errors"].append(str(e))
            return 0
    
    def load_templates_from_file(self, file_path: str) -> int:
        """
        从指定文件加载模板
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的模板数量
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"文件不存在: {file_path}")
                return 0
            
            if file_path.suffix.lower() not in self.supported_formats:
                self.logger.error(f"不支持的文件格式: {file_path.suffix}")
                return 0
            
            templates = self._parse_template_file(file_path)
            loaded_count = 0
            
            for template_data in templates:
                try:
                    template = self.registry._create_template_from_dict(template_data)
                    if self.registry.register_template(template):
                        loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"加载模板失败: {e}")
            
            self.logger.info(f"从文件加载模板完成: {file_path}, {loaded_count} 个模板")
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"从文件加载模板失败: {e}")
            return 0
    
    def watch_for_changes(self) -> bool:
        """
        监控文件变更
        
        Returns:
            是否有变更
        """
        current_time = time.time()
        
        # 检查是否需要扫描
        if current_time - self.last_scan < self.scan_interval:
            return False
        
        self.last_scan = current_time
        changes_detected = False
        
        try:
            for template_dir in self.template_dirs:
                dir_path = Path(template_dir)
                if dir_path.exists():
                    if self._check_directory_changes(dir_path):
                        changes_detected = True
            
            if changes_detected:
                self.logger.info("检测到模板文件变更，重新加载")
                self.load_external_templates()
            
            return changes_detected
            
        except Exception as e:
            self.logger.error(f"监控文件变更失败: {e}")
            return False
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """
        获取加载统计信息
        
        Returns:
            统计信息字典
        """
        return {
            **self.load_stats,
            "template_dirs": self.template_dirs,
            "supported_formats": list(self.supported_formats),
            "file_timestamps": len(self.file_timestamps),
            "last_scan": self.last_scan
        }
    
    def _load_templates_from_directory(self, dir_path: Path) -> int:
        """
        从目录加载模板
        
        Args:
            dir_path: 目录路径
            
        Returns:
            加载的模板数量
        """
        loaded_count = 0
        
        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    try:
                        count = self.load_templates_from_file(str(file_path))
                        loaded_count += count
                        self.load_stats["files_loaded"] += 1
                        
                        # 记录文件时间戳
                        self.file_timestamps[str(file_path)] = file_path.stat().st_mtime
                        
                    except Exception as e:
                        self.logger.warning(f"加载文件失败 {file_path}: {e}")
                        self.load_stats["errors"].append(f"{file_path}: {e}")
            
        except Exception as e:
            self.logger.error(f"加载目录失败 {dir_path}: {e}")
        
        return loaded_count
    
    def _parse_template_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        解析模板文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            模板数据列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif file_path.suffix.lower() in ['.yaml', '.yml']:
                    if YAML_AVAILABLE:
                        data = yaml.safe_load(f)
                    else:
                        raise ImportError("PyYAML未安装，无法解析YAML文件")
                else:
                    raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
            # 处理不同的数据格式
            if isinstance(data, dict):
                if "templates" in data:
                    return data["templates"]
                elif "data" in data:
                    return data["data"]
                else:
                    return [data]
            elif isinstance(data, list):
                return data
            else:
                self.logger.warning(f"无效的数据格式: {file_path}")
                return []
                
        except Exception as e:
            self.logger.error(f"解析文件失败 {file_path}: {e}")
            return []
    
    def _check_directory_changes(self, dir_path: Path) -> bool:
        """
        检查目录变更
        
        Args:
            dir_path: 目录路径
            
        Returns:
            是否有变更
        """
        changes_detected = False
        
        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    file_str = str(file_path)
                    current_mtime = file_path.stat().st_mtime
                    last_mtime = self.file_timestamps.get(file_str, 0)
                    
                    if current_mtime > last_mtime:
                        self.logger.info(f"检测到文件变更: {file_path}")
                        self.file_timestamps[file_str] = current_mtime
                        changes_detected = True
            
        except Exception as e:
            self.logger.error(f"检查目录变更失败 {dir_path}: {e}")
        
        return changes_detected
    
    def create_template_file(self, file_path: str, templates: List[Dict[str, Any]]) -> bool:
        """
        创建模板文件
        
        Args:
            file_path: 文件路径
            templates: 模板列表
            
        Returns:
            是否创建成功
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "export_time": time.time(),
                "templates": templates
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    json.dump(data, f, ensure_ascii=False, indent=2)
                elif file_path.suffix.lower() in ['.yaml', '.yml']:
                    if YAML_AVAILABLE:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                    else:
                        raise ImportError("PyYAML未安装，无法创建YAML文件")
                else:
                    raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
            self.logger.info(f"创建模板文件成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建模板文件失败: {e}")
            return False
    
    def backup_templates(self, backup_dir: str = "backups/templates") -> bool:
        """
        备份模板
        
        Args:
            backup_dir: 备份目录
            
        Returns:
            是否备份成功
        """
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 获取所有模板
            templates = self.registry.get_all_templates()
            template_data = []
            
            for template in templates:
                template_dict = {
                    "template_id": template.template_id,
                    "name": template.name,
                    "category": template.category,
                    "patterns": [
                        {
                            "pattern_id": p.pattern_id,
                            "regex_pattern": p.regex_pattern,
                            "confidence_weight": p.confidence_weight,
                            "description": p.description,
                            "examples": p.examples
                        }
                        for p in template.patterns
                    ],
                    "solution_template": template.solution_template,
                    "variables": template.variables,
                    "validation_rules": template.validation_rules,
                    "metadata": {
                        "version": template.metadata.version,
                        "author": template.metadata.author,
                        "description": template.metadata.description,
                        "tags": template.metadata.tags,
                        "enabled": template.metadata.enabled,
                        "priority": template.metadata.priority
                    }
                }
                template_data.append(template_dict)
            
            # 创建备份文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"templates_backup_{timestamp}.json"
            
            return self.create_template_file(str(backup_file), template_data)
            
        except Exception as e:
            self.logger.error(f"备份模板失败: {e}")
            return False 