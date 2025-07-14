"""
模板验证器
验证模板定义的有效性和质量
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .template_registry import TemplateDefinition, TemplatePattern


class TemplateValidator:
    """模板验证器"""
    
    def __init__(self):
        """初始化模板验证器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 验证规则
        self.validation_rules = {
            "template_id": {
                "required": True,
                "pattern": r"^[a-zA-Z][a-zA-Z0-9_]*$",
                "max_length": 50,
                "description": "模板ID必须以字母开头，只能包含字母、数字和下划线"
            },
            "name": {
                "required": True,
                "min_length": 1,
                "max_length": 100,
                "description": "模板名称不能为空，长度不超过100字符"
            },
            "category": {
                "required": True,
                "pattern": r"^[a-zA-Z][a-zA-Z0-9_]*$",
                "max_length": 30,
                "description": "分类必须以字母开头，只能包含字母、数字和下划线"
            },
            "regex_pattern": {
                "required": True,
                "min_length": 1,
                "description": "正则表达式不能为空"
            },
            "solution_template": {
                "required": True,
                "min_length": 1,
                "max_length": 500,
                "description": "解题模板不能为空，长度不超过500字符"
            }
        }
    
    def validate_template(self, template: TemplateDefinition) -> bool:
        """
        验证模板定义
        
        Args:
            template: 模板定义
            
        Returns:
            是否验证通过
        """
        try:
            # 验证基本字段
            if not self._validate_basic_fields(template):
                return False
            
            # 验证模式
            if not self._validate_patterns(template.patterns):
                return False
            
            # 验证解题模板
            if not self._validate_solution_template(template.solution_template, template.variables):
                return False
            
            # 验证变量一致性
            if not self._validate_variables_consistency(template):
                return False
            
            # 验证元数据
            if not self._validate_metadata(template.metadata):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"模板验证失败: {e}")
            return False
    
    def validate_template_dict(self, template_dict: Dict[str, Any]) -> bool:
        """
        验证模板字典
        
        Args:
            template_dict: 模板字典
            
        Returns:
            是否验证通过
        """
        try:
            # 验证必需字段
            required_fields = ["template_id", "name", "category", "patterns", "solution_template"]
            for field in required_fields:
                if field not in template_dict:
                    self.logger.error(f"缺少必需字段: {field}")
                    return False
            
            # 验证字段格式
            if not self._validate_dict_fields(template_dict):
                return False
            
            # 验证模式列表
            patterns = template_dict.get("patterns", [])
            if not isinstance(patterns, list) or not patterns:
                self.logger.error("模式列表不能为空")
                return False
            
            for pattern in patterns:
                if not self._validate_pattern_dict(pattern):
                    return False
            
            # 验证解题模板
            solution_template = template_dict.get("solution_template", "")
            variables = template_dict.get("variables", [])
            if not self._validate_solution_template(solution_template, variables):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"模板字典验证失败: {e}")
            return False
    
    def validate_pattern(self, pattern: TemplatePattern) -> bool:
        """
        验证单个模式
        
        Args:
            pattern: 模式定义
            
        Returns:
            是否验证通过
        """
        try:
            # 验证基本字段
            if not pattern.pattern_id or not pattern.regex_pattern:
                self.logger.error("模式ID和正则表达式不能为空")
                return False
            
            # 验证模式ID格式
            if not re.match(self.validation_rules["template_id"]["pattern"], pattern.pattern_id):
                self.logger.error(f"模式ID格式无效: {pattern.pattern_id}")
                return False
            
            # 验证正则表达式
            if not self._validate_regex_pattern(pattern.regex_pattern):
                return False
            
            # 验证置信度权重
            if not (0.0 <= pattern.confidence_weight <= 1.0):
                self.logger.error(f"置信度权重必须在0.0到1.0之间: {pattern.confidence_weight}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"模式验证失败: {e}")
            return False
    
    def _validate_basic_fields(self, template: TemplateDefinition) -> bool:
        """验证基本字段"""
        # 验证模板ID
        if not re.match(self.validation_rules["template_id"]["pattern"], template.template_id):
            self.logger.error(f"模板ID格式无效: {template.template_id}")
            return False
        
        # 验证名称
        name = template.name
        if not name or len(name) > self.validation_rules["name"]["max_length"]:
            self.logger.error(f"模板名称无效: {name}")
            return False
        
        # 验证分类
        if not re.match(self.validation_rules["category"]["pattern"], template.category):
            self.logger.error(f"分类格式无效: {template.category}")
            return False
        
        return True
    
    def _validate_patterns(self, patterns: List[TemplatePattern]) -> bool:
        """验证模式列表"""
        if not patterns:
            self.logger.error("模式列表不能为空")
            return False
        
        # 检查模式ID唯一性
        pattern_ids = [p.pattern_id for p in patterns]
        if len(pattern_ids) != len(set(pattern_ids)):
            self.logger.error("模式ID必须唯一")
            return False
        
        # 验证每个模式
        for pattern in patterns:
            if not self.validate_pattern(pattern):
                return False
        
        return True
    
    def _validate_solution_template(self, solution_template: str, variables: List[str]) -> bool:
        """验证解题模板"""
        if not solution_template:
            self.logger.error("解题模板不能为空")
            return False
        
        if len(solution_template) > self.validation_rules["solution_template"]["max_length"]:
            self.logger.error(f"解题模板过长: {len(solution_template)}")
            return False
        
        # 检查模板中的变量引用
        template_vars = re.findall(r'\{([^}]+)\}', solution_template)
        for var in template_vars:
            if var not in variables:
                self.logger.warning(f"模板中引用了未定义的变量: {var}")
        
        return True
    
    def _validate_variables_consistency(self, template: TemplateDefinition) -> bool:
        """验证变量一致性"""
        # 检查变量名格式
        for var in template.variables:
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', var):
                self.logger.error(f"变量名格式无效: {var}")
                return False
        
        # 检查变量名唯一性
        if len(template.variables) != len(set(template.variables)):
            self.logger.error("变量名必须唯一")
            return False
        
        return True
    
    def _validate_metadata(self, metadata) -> bool:
        """验证元数据"""
        # 验证版本号格式
        if not re.match(r'^\d+\.\d+\.\d+$', metadata.version):
            self.logger.error(f"版本号格式无效: {metadata.version}")
            return False
        
        # 验证优先级
        if not (0 <= metadata.priority <= 100):
            self.logger.error(f"优先级必须在0到100之间: {metadata.priority}")
            return False
        
        # 验证成功率
        if not (0.0 <= metadata.success_rate <= 1.0):
            self.logger.error(f"成功率必须在0.0到1.0之间: {metadata.success_rate}")
            return False
        
        return True
    
    def _validate_dict_fields(self, template_dict: Dict[str, Any]) -> bool:
        """验证字典字段"""
        # 验证模板ID
        template_id = template_dict.get("template_id", "")
        if not re.match(self.validation_rules["template_id"]["pattern"], template_id):
            self.logger.error(f"模板ID格式无效: {template_id}")
            return False
        
        # 验证名称
        name = template_dict.get("name", "")
        if not name or len(name) > self.validation_rules["name"]["max_length"]:
            self.logger.error(f"模板名称无效: {name}")
            return False
        
        # 验证分类
        category = template_dict.get("category", "")
        if not re.match(self.validation_rules["category"]["pattern"], category):
            self.logger.error(f"分类格式无效: {category}")
            return False
        
        return True
    
    def _validate_pattern_dict(self, pattern_dict: Dict[str, Any]) -> bool:
        """验证模式字典"""
        # 验证必需字段
        required_fields = ["pattern_id", "regex_pattern"]
        for field in required_fields:
            if field not in pattern_dict:
                self.logger.error(f"模式缺少必需字段: {field}")
                return False
        
        # 验证模式ID
        pattern_id = pattern_dict.get("pattern_id", "")
        if not re.match(self.validation_rules["template_id"]["pattern"], pattern_id):
            self.logger.error(f"模式ID格式无效: {pattern_id}")
            return False
        
        # 验证正则表达式
        regex_pattern = pattern_dict.get("regex_pattern", "")
        if not self._validate_regex_pattern(regex_pattern):
            return False
        
        # 验证置信度权重
        confidence_weight = pattern_dict.get("confidence_weight", 1.0)
        if not (0.0 <= confidence_weight <= 1.0):
            self.logger.error(f"置信度权重必须在0.0到1.0之间: {confidence_weight}")
            return False
        
        return True
    
    def _validate_regex_pattern(self, pattern: str) -> bool:
        """验证正则表达式"""
        if not pattern:
            self.logger.error("正则表达式不能为空")
            return False
        
        try:
            # 尝试编译正则表达式
            re.compile(pattern)
            return True
        except re.error as e:
            self.logger.error(f"正则表达式无效: {pattern}, 错误: {e}")
            return False
    
    def get_validation_errors(self, template: TemplateDefinition) -> List[str]:
        """
        获取验证错误列表
        
        Args:
            template: 模板定义
            
        Returns:
            错误信息列表
        """
        errors = []
        
        try:
            # 验证基本字段
            if not re.match(self.validation_rules["template_id"]["pattern"], template.template_id):
                errors.append(f"模板ID格式无效: {template.template_id}")
            
            if not template.name or len(template.name) > self.validation_rules["name"]["max_length"]:
                errors.append(f"模板名称无效: {template.name}")
            
            if not re.match(self.validation_rules["category"]["pattern"], template.category):
                errors.append(f"分类格式无效: {template.category}")
            
            # 验证模式
            if not template.patterns:
                errors.append("模式列表不能为空")
            else:
                pattern_ids = [p.pattern_id for p in template.patterns]
                if len(pattern_ids) != len(set(pattern_ids)):
                    errors.append("模式ID必须唯一")
                
                for pattern in template.patterns:
                    if not self.validate_pattern(pattern):
                        errors.append(f"模式验证失败: {pattern.pattern_id}")
            
            # 验证解题模板
            if not template.solution_template:
                errors.append("解题模板不能为空")
            elif len(template.solution_template) > self.validation_rules["solution_template"]["max_length"]:
                errors.append(f"解题模板过长: {len(template.solution_template)}")
            
            # 验证变量
            for var in template.variables:
                if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', var):
                    errors.append(f"变量名格式无效: {var}")
            
            if len(template.variables) != len(set(template.variables)):
                errors.append("变量名必须唯一")
            
        except Exception as e:
            errors.append(f"验证过程中发生错误: {e}")
        
        return errors 