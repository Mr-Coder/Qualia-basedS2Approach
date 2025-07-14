"""
模板管理器
实现ITemplateManager接口，协调模板注册表、匹配器和验证器
"""

import logging
import time
from typing import Any, Dict, List, Optional

try:
    from src.core.exceptions import TemplateError
except ImportError:
    # Fallback for when src is not in path
    class TemplateError(Exception):
        def __init__(self, message: str, cause: Exception = None):
            super().__init__(message)
            self.message = message
            self.cause = cause

try:
    from src.core.interfaces import ITemplateManager
except ImportError:
    # Fallback for when src is not in path
    class ITemplateManager:
        pass

try:
    from src.monitoring.performance_monitor import monitor_performance
except ImportError:
    # Fallback for when src is not in path
    def monitor_performance(func):
        return func

from .template_loader import TemplateLoader
from .template_matcher import MatchResult, TemplateMatcher
from .template_registry import TemplateDefinition, TemplateRegistry
from .template_validator import TemplateValidator


class TemplateManager(ITemplateManager):
    """模板管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模板管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化组件
        self.registry = TemplateRegistry(
            storage_path=self.config.get("storage_path", "config/templates")
        )
        self.matcher = TemplateMatcher(self.registry)
        self.validator = TemplateValidator()
        self.loader = TemplateLoader(self.registry)
        
        # 性能监控
        self.performance_stats = {
            "total_operations": 0,
            "average_response_time": 0.0,
            "success_rate": 0.0,
            "last_operation": None
        }
        
        # 热更新配置
        self.auto_reload = self.config.get("auto_reload", True)
        self.reload_interval = self.config.get("reload_interval", 300)  # 5分钟
        self.last_reload = time.time()
        
        self.logger.info("模板管理器初始化完成")
    
    @monitor_performance
    def match_template(self, text: str) -> Optional[Dict[str, Any]]:
        """
        匹配模板
        
        Args:
            text: 待匹配文本
            
        Returns:
            匹配结果字典
        """
        try:
            # 检查是否需要重新加载
            if self.auto_reload and self._should_reload():
                self._reload_templates()
            
            # 执行匹配
            match_result = self.matcher.match_text_best(text)
            
            if match_result:
                # 更新模板使用统计
                self.registry.update_template_usage(match_result.template_id, success=True)
                
                # 转换为字典格式
                result_dict = {
                    "template_id": match_result.template_id,
                    "template_name": match_result.template_name,
                    "category": match_result.category,
                    "confidence": match_result.confidence,
                    "matched_pattern": match_result.matched_pattern,
                    "extracted_values": match_result.extracted_values,
                    "solution_template": match_result.solution_template,
                    "variables": match_result.variables,
                    "timestamp": time.time()
                }
                
                self._update_performance_stats(success=True)
                return result_dict
            else:
                self._update_performance_stats(success=False)
                return None
                
        except Exception as e:
            self.logger.error(f"模板匹配失败: {e}")
            self._update_performance_stats(success=False)
            raise TemplateError(f"模板匹配失败: {e}", cause=e)
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """
        获取所有模板
        
        Returns:
            模板列表
        """
        try:
            templates = self.registry.get_all_templates()
            return [
                {
                    "template_id": t.template_id,
                    "name": t.name,
                    "category": t.category,
                    "patterns": [
                        {
                            "pattern_id": p.pattern_id,
                            "regex_pattern": p.regex_pattern,
                            "confidence_weight": p.confidence_weight,
                            "description": p.description,
                            "examples": p.examples
                        }
                        for p in t.patterns
                    ],
                    "solution_template": t.solution_template,
                    "variables": t.variables,
                    "metadata": {
                        "version": t.metadata.version,
                        "author": t.metadata.author,
                        "description": t.metadata.description,
                        "tags": t.metadata.tags,
                        "enabled": t.metadata.enabled,
                        "priority": t.metadata.priority,
                        "usage_count": t.metadata.usage_count,
                        "success_rate": t.metadata.success_rate,
                        "last_used": t.metadata.last_used.isoformat() if t.metadata.last_used else None
                    }
                }
                for t in templates
            ]
        except Exception as e:
            self.logger.error(f"获取模板列表失败: {e}")
            raise TemplateError(f"获取模板列表失败: {e}", cause=e)
    
    def add_template(self, template: Dict[str, Any]) -> bool:
        """
        添加模板
        
        Args:
            template: 模板定义字典
            
        Returns:
            是否添加成功
        """
        try:
            # 验证模板
            if not self.validator.validate_template_dict(template):
                self.logger.error("模板验证失败")
                return False
            
            # 创建模板定义
            template_def = self._create_template_from_dict(template)
            
            # 注册模板
            success = self.registry.register_template(template_def)
            
            if success:
                self.logger.info(f"添加模板成功: {template_def.template_id}")
            else:
                self.logger.error(f"添加模板失败: {template_def.template_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"添加模板失败: {e}")
            return False
    
    def remove_template(self, template_id: str) -> bool:
        """
        移除模板
        
        Args:
            template_id: 模板ID
            
        Returns:
            是否移除成功
        """
        try:
            success = self.registry.unregister_template(template_id)
            
            if success:
                self.logger.info(f"移除模板成功: {template_id}")
            else:
                self.logger.warning(f"移除模板失败: {template_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"移除模板失败: {e}")
            return False
    
    def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新模板
        
        Args:
            template_id: 模板ID
            updates: 更新内容
            
        Returns:
            是否更新成功
        """
        try:
            # 获取现有模板
            existing_template = self.registry.get_template(template_id)
            if not existing_template:
                self.logger.error(f"模板不存在: {template_id}")
                return False
            
            # 创建更新后的模板
            updated_template = self._update_template_definition(existing_template, updates)
            
            # 验证更新后的模板
            if not self.validator.validate_template(updated_template):
                self.logger.error(f"更新后的模板验证失败: {template_id}")
                return False
            
            # 先移除旧模板
            self.registry.unregister_template(template_id)
            
            # 注册新模板
            success = self.registry.register_template(updated_template)
            
            if success:
                self.logger.info(f"更新模板成功: {template_id}")
            else:
                self.logger.error(f"更新模板失败: {template_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"更新模板失败: {e}")
            return False
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """
        获取模板统计信息
        
        Returns:
            统计信息字典
        """
        try:
            registry_stats = self.registry.get_stats()
            matcher_stats = self.matcher.get_match_statistics()
            
            return {
                **registry_stats,
                **matcher_stats,
                **self.performance_stats,
                "auto_reload_enabled": self.auto_reload,
                "last_reload": self.last_reload
            }
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def search_templates(self, query: str, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        搜索模板
        
        Args:
            query: 搜索查询
            categories: 分类限制
            
        Returns:
            匹配的模板列表
        """
        try:
            templates = self.registry.search_templates(query)
            
            if categories:
                templates = [t for t in templates if t.category in categories]
            
            return [
                {
                    "template_id": t.template_id,
                    "name": t.name,
                    "category": t.category,
                    "description": t.metadata.description,
                    "tags": t.metadata.tags,
                    "enabled": t.metadata.enabled,
                    "usage_count": t.metadata.usage_count,
                    "success_rate": t.metadata.success_rate
                }
                for t in templates
            ]
        except Exception as e:
            self.logger.error(f"搜索模板失败: {e}")
            return []
    
    def export_templates(self, file_path: str, categories: Optional[List[str]] = None) -> bool:
        """
        导出模板
        
        Args:
            file_path: 导出文件路径
            categories: 分类限制
            
        Returns:
            是否导出成功
        """
        try:
            if categories:
                # 导出指定分类的模板
                templates = []
                for category in categories:
                    templates.extend(self.registry.get_templates_by_category(category))
                
                # 创建临时注册表
                temp_registry = TemplateRegistry()
                for template in templates:
                    temp_registry.register_template(template)
                
                return temp_registry.export_templates(file_path)
            else:
                # 导出所有模板
                return self.registry.export_templates(file_path)
                
        except Exception as e:
            self.logger.error(f"导出模板失败: {e}")
            return False
    
    def import_templates(self, file_path: str, overwrite: bool = False) -> int:
        """
        导入模板
        
        Args:
            file_path: 导入文件路径
            overwrite: 是否覆盖现有模板
            
        Returns:
            导入的模板数量
        """
        try:
            if overwrite:
                # 清空现有模板
                all_templates = list(self.registry.get_all_templates())
                for template in all_templates:
                    self.registry.unregister_template(template.template_id)
            
            # 导入新模板
            imported_count = self.registry.import_templates(file_path)
            
            self.logger.info(f"导入模板完成: {imported_count} 个模板")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"导入模板失败: {e}")
            return 0
    
    def reload_templates(self) -> bool:
        """
        重新加载模板
        
        Returns:
            是否重新加载成功
        """
        try:
            success = self._reload_templates()
            if success:
                self.last_reload = time.time()
                self.logger.info("模板重新加载成功")
            return success
        except Exception as e:
            self.logger.error(f"重新加载模板失败: {e}")
            return False
    
    def _should_reload(self) -> bool:
        """检查是否需要重新加载"""
        return time.time() - self.last_reload > self.reload_interval
    
    def _reload_templates(self) -> bool:
        """重新加载模板"""
        try:
            # 重新加载默认模板
            self.registry._load_default_templates()
            
            # 重新加载外部模板文件
            self.loader.load_external_templates()
            
            return True
        except Exception as e:
            self.logger.error(f"重新加载模板失败: {e}")
            return False
    
    def _create_template_from_dict(self, template_dict: Dict[str, Any]) -> TemplateDefinition:
        """从字典创建模板定义"""
        return self.registry._create_template_from_dict(template_dict)
    
    def _update_template_definition(self, template: TemplateDefinition, updates: Dict[str, Any]) -> TemplateDefinition:
        """更新模板定义"""
        # 创建新的模板定义，应用更新
        updated_template = TemplateDefinition(
            template_id=template.template_id,
            name=updates.get("name", template.name),
            category=updates.get("category", template.category),
            patterns=updates.get("patterns", template.patterns),
            solution_template=updates.get("solution_template", template.solution_template),
            variables=updates.get("variables", template.variables),
            validation_rules=updates.get("validation_rules", template.validation_rules),
            metadata=template.metadata
        )
        
        # 更新元数据
        if "metadata" in updates:
            metadata_updates = updates["metadata"]
            updated_template.metadata.description = metadata_updates.get("description", template.metadata.description)
            updated_template.metadata.tags = metadata_updates.get("tags", template.metadata.tags)
            updated_template.metadata.enabled = metadata_updates.get("enabled", template.metadata.enabled)
            updated_template.metadata.priority = metadata_updates.get("priority", template.metadata.priority)
        
        return updated_template
    
    def _update_performance_stats(self, success: bool):
        """更新性能统计"""
        self.performance_stats["total_operations"] += 1
        
        # 更新成功率
        total_ops = self.performance_stats["total_operations"]
        current_success_rate = self.performance_stats["success_rate"]
        
        if success:
            self.performance_stats["success_rate"] = (
                (current_success_rate * (total_ops - 1) + 1) / total_ops
            )
        else:
            self.performance_stats["success_rate"] = (
                (current_success_rate * (total_ops - 1)) / total_ops
            )
        
        self.performance_stats["last_operation"] = time.time() 