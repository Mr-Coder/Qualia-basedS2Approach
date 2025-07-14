"""
模板注册表
动态管理所有模板，支持模板的注册、查询、更新和删除
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


@dataclass
class TemplateMetadata:
    """模板元数据"""
    template_id: str
    name: str
    category: str
    version: str = "1.0.0"
    author: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class TemplatePattern:
    """模板模式"""
    pattern_id: str
    regex_pattern: str
    confidence_weight: float = 1.0
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class TemplateDefinition:
    """模板定义"""
    template_id: str
    name: str
    category: str
    patterns: List[TemplatePattern]
    solution_template: str
    variables: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: TemplateMetadata = field(default_factory=TemplateMetadata)


class TemplateRegistry:
    """模板注册表"""
    
    def __init__(self, storage_path: str = "config/templates"):
        """
        初始化模板注册表
        
        Args:
            storage_path: 模板存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 模板存储
        self.templates: Dict[str, TemplateDefinition] = {}
        self.categories: Dict[str, Set[str]] = {}
        self.pattern_index: Dict[str, List[str]] = {}
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "total_templates": 0,
            "active_templates": 0,
            "categories": 0,
            "last_updated": None
        }
        
        # 日志
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 加载默认模板
        self._load_default_templates()
        
        self.logger.info(f"模板注册表初始化完成，存储路径: {self.storage_path}")
    
    def register_template(self, template: TemplateDefinition) -> bool:
        """
        注册模板
        
        Args:
            template: 模板定义
            
        Returns:
            是否注册成功
        """
        with self._lock:
            try:
                # 验证模板
                if not self._validate_template(template):
                    return False
                
                # 注册模板
                self.templates[template.template_id] = template
                
                # 更新分类索引
                if template.category not in self.categories:
                    self.categories[template.category] = set()
                self.categories[template.category].add(template.template_id)
                
                # 更新模式索引
                for pattern in template.patterns:
                    if pattern.regex_pattern not in self.pattern_index:
                        self.pattern_index[pattern.regex_pattern] = []
                    self.pattern_index[pattern.regex_pattern].append(template.template_id)
                
                # 更新统计
                self._update_stats()
                
                self.logger.info(f"注册模板成功: {template.template_id} ({template.name})")
                return True
                
            except Exception as e:
                self.logger.error(f"注册模板失败 {template.template_id}: {e}")
                return False
    
    def unregister_template(self, template_id: str) -> bool:
        """
        注销模板
        
        Args:
            template_id: 模板ID
            
        Returns:
            是否注销成功
        """
        with self._lock:
            try:
                if template_id not in self.templates:
                    self.logger.warning(f"模板不存在: {template_id}")
                    return False
                
                template = self.templates[template_id]
                
                # 从分类索引中移除
                if template.category in self.categories:
                    self.categories[template.category].discard(template_id)
                    if not self.categories[template.category]:
                        del self.categories[template.category]
                
                # 从模式索引中移除
                for pattern in template.patterns:
                    if pattern.regex_pattern in self.pattern_index:
                        if template_id in self.pattern_index[pattern.regex_pattern]:
                            self.pattern_index[pattern.regex_pattern].remove(template_id)
                        if not self.pattern_index[pattern.regex_pattern]:
                            del self.pattern_index[pattern.regex_pattern]
                
                # 从模板存储中移除
                del self.templates[template_id]
                
                # 更新统计
                self._update_stats()
                
                self.logger.info(f"注销模板成功: {template_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"注销模板失败 {template_id}: {e}")
                return False
    
    def get_template(self, template_id: str) -> Optional[TemplateDefinition]:
        """
        获取模板
        
        Args:
            template_id: 模板ID
            
        Returns:
            模板定义
        """
        return self.templates.get(template_id)
    
    def get_all_templates(self) -> List[TemplateDefinition]:
        """
        获取所有模板
        
        Returns:
            模板列表
        """
        return list(self.templates.values())
    
    def get_active_templates(self) -> List[TemplateDefinition]:
        """
        获取活跃模板
        
        Returns:
            活跃模板列表
        """
        return [t for t in self.templates.values() if t.metadata.enabled]
    
    def get_templates_by_category(self, category: str) -> List[TemplateDefinition]:
        """
        根据分类获取模板
        
        Args:
            category: 分类名称
            
        Returns:
            模板列表
        """
        template_ids = self.categories.get(category, set())
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def search_templates(self, query: str) -> List[TemplateDefinition]:
        """
        搜索模板
        
        Args:
            query: 搜索查询
            
        Returns:
            匹配的模板列表
        """
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            # 搜索名称
            if query_lower in template.name.lower():
                results.append(template)
                continue
            
            # 搜索描述
            if query_lower in template.metadata.description.lower():
                results.append(template)
                continue
            
            # 搜索标签
            for tag in template.metadata.tags:
                if query_lower in tag.lower():
                    results.append(template)
                    break
        
        return results
    
    def update_template_usage(self, template_id: str, success: bool = True):
        """
        更新模板使用统计
        
        Args:
            template_id: 模板ID
            success: 是否成功匹配
        """
        with self._lock:
            if template_id in self.templates:
                template = self.templates[template_id]
                template.metadata.usage_count += 1
                template.metadata.last_used = datetime.now()
                
                # 更新成功率
                if template.metadata.usage_count == 1:
                    template.metadata.success_rate = 1.0 if success else 0.0
                else:
                    # 简单的移动平均
                    current_rate = template.metadata.success_rate
                    new_rate = (current_rate * (template.metadata.usage_count - 1) + (1.0 if success else 0.0)) / template.metadata.usage_count
                    template.metadata.success_rate = new_rate
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            **self.stats,
            "categories": list(self.categories.keys()),
            "template_count_by_category": {
                category: len(templates) 
                for category, templates in self.categories.items()
            }
        }
    
    def export_templates(self, file_path: str) -> bool:
        """
        导出模板
        
        Args:
            file_path: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_templates": len(self.templates),
                "templates": []
            }
            
            for template in self.templates.values():
                template_data = {
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
                        "priority": template.metadata.priority,
                        "usage_count": template.metadata.usage_count,
                        "success_rate": template.metadata.success_rate,
                        "last_used": template.metadata.last_used.isoformat() if template.metadata.last_used else None
                    }
                }
                export_data["templates"].append(template_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"模板导出成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"模板导出失败: {e}")
            return False
    
    def import_templates(self, file_path: str) -> int:
        """
        导入模板
        
        Args:
            file_path: 导入文件路径
            
        Returns:
            导入的模板数量
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_count = 0
            templates_data = import_data.get("templates", [])
            
            for template_data in templates_data:
                try:
                    template = self._create_template_from_dict(template_data)
                    if self.register_template(template):
                        imported_count += 1
                except Exception as e:
                    self.logger.error(f"导入模板失败 {template_data.get('template_id', 'unknown')}: {e}")
            
            self.logger.info(f"模板导入完成: {imported_count}/{len(templates_data)} 个模板")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"模板导入失败: {e}")
            return 0
    
    def _validate_template(self, template: TemplateDefinition) -> bool:
        """验证模板"""
        # 基本验证
        if not template.template_id or not template.name or not template.category:
            return False
        
        # 检查ID唯一性
        if template.template_id in self.templates:
            self.logger.warning(f"模板ID已存在: {template.template_id}")
            return False
        
        # 验证模式
        if not template.patterns:
            return False
        
        return True
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats["total_templates"] = len(self.templates)
        self.stats["active_templates"] = len([t for t in self.templates.values() if t.metadata.enabled])
        self.stats["categories"] = len(self.categories)
        self.stats["last_updated"] = datetime.now()
    
    def _load_default_templates(self):
        """加载默认模板"""
        # 尝试从外部文件加载
        if self._load_external_templates():
            return
        
        # 如果外部文件不存在，创建默认模板文件
        self._create_default_template_files()
        self._load_external_templates()
    
    def _load_external_templates(self) -> bool:
        """从外部文件加载模板"""
        try:
            # 查找模板文件
            template_files = list(self.storage_path.glob("*.json")) + list(self.storage_path.glob("*.yaml"))
            
            if not template_files:
                return False
            
            loaded_count = 0
            for file_path in template_files:
                try:
                    if file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    else:  # yaml
                        import yaml
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f)
                    
                    templates_data = data.get("templates", [])
                    for template_data in templates_data:
                        template = self._create_template_from_dict(template_data)
                        if self.register_template(template):
                            loaded_count += 1
                            
                except Exception as e:
                    self.logger.error(f"加载模板文件失败 {file_path}: {e}")
            
            self.logger.info(f"从外部文件加载了 {loaded_count} 个模板")
            return loaded_count > 0
            
        except Exception as e:
            self.logger.error(f"加载外部模板失败: {e}")
            return False
    
    def _create_default_template_files(self):
        """创建默认模板文件"""
        try:
            # 算术模板
            arithmetic_templates = {
                "templates": [
                    {
                        "template_id": "arithmetic_addition",
                        "name": "加法运算",
                        "category": "arithmetic",
                        "patterns": [
                            {
                                "pattern_id": "add_total",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?(\d+(?:\.\d+)?).+?total",
                                "confidence_weight": 0.9,
                                "description": "求总数",
                                "examples": ["5 and 3 total", "10 plus 5 total"]
                            },
                            {
                                "pattern_id": "add_altogether", 
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?(\d+(?:\.\d+)?).+?altogether",
                                "confidence_weight": 0.9,
                                "description": "求总和",
                                "examples": ["7 and 4 altogether", "12 plus 8 altogether"]
                            },
                            {
                                "pattern_id": "add_plus",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?plus.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.95,
                                "description": "加号运算",
                                "examples": ["5 plus 3", "10 plus 5"]
                            }
                        ],
                        "solution_template": "{operand1} + {operand2} = {result}",
                        "variables": ["operand1", "operand2", "result"],
                        "metadata": {
                            "version": "1.0.0",
                            "author": "system",
                            "description": "基本加法运算模板",
                            "tags": ["加法", "算术"],
                            "enabled": True,
                            "priority": 1
                        }
                    },
                    {
                        "template_id": "arithmetic_subtraction",
                        "name": "减法运算",
                        "category": "arithmetic",
                        "patterns": [
                            {
                                "pattern_id": "sub_minus",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?minus.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.95,
                                "description": "减号运算",
                                "examples": ["10 minus 4", "15 minus 7"]
                            },
                            {
                                "pattern_id": "sub_take_away",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?take away.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.9,
                                "description": "拿走运算",
                                "examples": ["8 take away 3", "12 take away 5"]
                            },
                            {
                                "pattern_id": "sub_left",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?left.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.85,
                                "description": "剩余运算",
                                "examples": ["10 left 3", "20 left 8"]
                            }
                        ],
                        "solution_template": "{operand1} - {operand2} = {result}",
                        "variables": ["operand1", "operand2", "result"],
                        "metadata": {
                            "version": "1.0.0",
                            "author": "system",
                            "description": "基本减法运算模板",
                            "tags": ["减法", "算术"],
                            "enabled": True,
                            "priority": 1
                        }
                    },
                    {
                        "template_id": "arithmetic_multiplication",
                        "name": "乘法运算",
                        "category": "arithmetic",
                        "patterns": [
                            {
                                "pattern_id": "mul_times",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?times.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.95,
                                "description": "倍数运算",
                                "examples": ["6 times 7", "4 times 5"]
                            },
                            {
                                "pattern_id": "mul_each",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?each.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.9,
                                "description": "每个运算",
                                "examples": ["3 each 4", "5 each 6"]
                            },
                            {
                                "pattern_id": "mul_groups",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?groups.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.85,
                                "description": "分组运算",
                                "examples": ["2 groups 3", "4 groups 5"]
                            }
                        ],
                        "solution_template": "{operand1} × {operand2} = {result}",
                        "variables": ["operand1", "operand2", "result"],
                        "metadata": {
                            "version": "1.0.0",
                            "author": "system",
                            "description": "基本乘法运算模板",
                            "tags": ["乘法", "算术"],
                            "enabled": True,
                            "priority": 1
                        }
                    },
                    {
                        "template_id": "arithmetic_division",
                        "name": "除法运算",
                        "category": "arithmetic",
                        "patterns": [
                            {
                                "pattern_id": "div_divided",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?divided.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.95,
                                "description": "除法运算",
                                "examples": ["20 divided by 5", "15 divided by 3"]
                            },
                            {
                                "pattern_id": "div_share",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?share.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.9,
                                "description": "分享运算",
                                "examples": ["12 share 4", "18 share 6"]
                            },
                            {
                                "pattern_id": "div_each",
                                "regex_pattern": r"(\d+(?:\.\d+)?).+?each.+?(\d+(?:\.\d+)?)",
                                "confidence_weight": 0.85,
                                "description": "每个运算",
                                "examples": ["10 each 2", "16 each 4"]
                            }
                        ],
                        "solution_template": "{operand1} ÷ {operand2} = {result}",
                        "variables": ["operand1", "operand2", "result"],
                        "metadata": {
                            "version": "1.0.0",
                            "author": "system",
                            "description": "基本除法运算模板",
                            "tags": ["除法", "算术"],
                            "enabled": True,
                            "priority": 1
                        }
                    }
                ]
            }
            
            # 应用题模板
            word_problem_templates = {
                "templates": [
                    {
                        "template_id": "word_problem_discount",
                        "name": "折扣问题",
                        "category": "word_problem",
                        "patterns": [
                            {
                                "pattern_id": "discount_chinese",
                                "regex_pattern": r"打(\d+)折",
                                "confidence_weight": 0.95,
                                "description": "中文折扣",
                                "examples": ["打8折", "打9折"]
                            },
                            {
                                "pattern_id": "discount_percent",
                                "regex_pattern": r"(\d+)%折扣",
                                "confidence_weight": 0.9,
                                "description": "百分比折扣",
                                "examples": ["20%折扣", "30%折扣"]
                            },
                            {
                                "pattern_id": "discount_simple",
                                "regex_pattern": r"(\d+)折",
                                "confidence_weight": 0.85,
                                "description": "简单折扣",
                                "examples": ["8折", "9折"]
                            }
                        ],
                        "solution_template": "原价 × (折扣/10) = 现价",
                        "variables": ["原价", "折扣", "现价"],
                        "metadata": {
                            "version": "1.0.0",
                            "author": "system",
                            "description": "折扣计算问题",
                            "tags": ["折扣", "应用题"],
                            "enabled": True,
                            "priority": 2
                        }
                    },
                    {
                        "template_id": "word_problem_percentage",
                        "name": "百分比问题",
                        "category": "word_problem",
                        "patterns": [
                            {
                                "pattern_id": "percent_symbol",
                                "regex_pattern": r"(\d+)%",
                                "confidence_weight": 0.9,
                                "description": "百分比符号",
                                "examples": ["30%", "50%"]
                            },
                            {
                                "pattern_id": "percent_chinese",
                                "regex_pattern": r"百分之(\d+)",
                                "confidence_weight": 0.95,
                                "description": "中文百分比",
                                "examples": ["百分之三十", "百分之五十"]
                            }
                        ],
                        "solution_template": "总数 × (百分比/100) = 部分",
                        "variables": ["总数", "百分比", "部分"],
                        "metadata": {
                            "version": "1.0.0",
                            "author": "system",
                            "description": "百分比计算问题",
                            "tags": ["百分比", "应用题"],
                            "enabled": True,
                            "priority": 2
                        }
                    },
                    {
                        "template_id": "word_problem_average",
                        "name": "平均值问题",
                        "category": "word_problem",
                        "patterns": [
                            {
                                "pattern_id": "average_chinese",
                                "regex_pattern": r"平均",
                                "confidence_weight": 0.9,
                                "description": "中文平均",
                                "examples": ["平均分", "平均速度"]
                            },
                            {
                                "pattern_id": "average_each",
                                "regex_pattern": r"每",
                                "confidence_weight": 0.85,
                                "description": "每个",
                                "examples": ["每个", "每人"]
                            },
                            {
                                "pattern_id": "average_per",
                                "regex_pattern": r"per",
                                "confidence_weight": 0.8,
                                "description": "英文per",
                                "examples": ["per hour", "per day"]
                            }
                        ],
                        "solution_template": "总和 ÷ 数量 = 平均值",
                        "variables": ["总和", "数量", "平均值"],
                        "metadata": {
                            "version": "1.0.0",
                            "author": "system",
                            "description": "平均值计算问题",
                            "tags": ["平均值", "应用题"],
                            "enabled": True,
                            "priority": 2
                        }
                    }
                ]
            }
            
            # 几何模板
            geometry_templates = {
                "templates": [
                    {
                        "template_id": "geometry_area",
                        "name": "面积问题",
                        "category": "geometry",
                        "patterns": [
                            {
                                "pattern_id": "area_chinese",
                                "regex_pattern": r"面积",
                                "confidence_weight": 0.9,
                                "description": "中文面积",
                                "examples": ["求面积", "计算面积"]
                            },
                            {
                                "pattern_id": "area_square",
                                "regex_pattern": r"平方",
                                "confidence_weight": 0.8,
                                "description": "平方单位",
                                "examples": ["平方米", "平方厘米"]
                            },
                            {
                                "pattern_id": "area_rectangle",
                                "regex_pattern": r"长.*宽",
                                "confidence_weight": 0.9,
                                "description": "长方形",
                                "examples": ["长5宽3", "长10宽6"]
                            }
                        ],
                        "solution_template": "长 × 宽 = 面积",
                        "variables": ["长", "宽", "面积"],
                        "metadata": {
                            "version": "1.0.0",
                            "author": "system",
                            "description": "面积计算问题",
                            "tags": ["面积", "几何"],
                            "enabled": True,
                            "priority": 2
                        }
                    }
                ]
            }
            
            # 保存模板文件
            template_files = [
                ("arithmetic_templates.json", arithmetic_templates),
                ("word_problem_templates.json", word_problem_templates),
                ("geometry_templates.json", geometry_templates)
            ]
            
            for filename, data in template_files:
                file_path = self.storage_path / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info("默认模板文件创建完成")
            
        except Exception as e:
            self.logger.error(f"创建默认模板文件失败: {e}")
    
    def _create_template_from_dict(self, template_data: Dict[str, Any]) -> TemplateDefinition:
        """从字典创建模板定义"""
        patterns = []
        for pattern_data in template_data.get("patterns", []):
            pattern = TemplatePattern(
                pattern_id=pattern_data["pattern_id"],
                regex_pattern=pattern_data["regex_pattern"],
                confidence_weight=pattern_data.get("confidence_weight", 1.0),
                description=pattern_data.get("description", ""),
                examples=pattern_data.get("examples", [])
            )
            patterns.append(pattern)
        
        metadata = TemplateMetadata(
            template_id=template_data["template_id"],
            name=template_data["name"],
            category=template_data["category"],
            version=template_data.get("metadata", {}).get("version", "1.0.0"),
            author=template_data.get("metadata", {}).get("author", "system"),
            description=template_data.get("metadata", {}).get("description", ""),
            tags=template_data.get("metadata", {}).get("tags", []),
            enabled=template_data.get("metadata", {}).get("enabled", True),
            priority=template_data.get("metadata", {}).get("priority", 0)
        )
        
        return TemplateDefinition(
            template_id=template_data["template_id"],
            name=template_data["name"],
            category=template_data["category"],
            patterns=patterns,
            solution_template=template_data["solution_template"],
            variables=template_data.get("variables", []),
            validation_rules=template_data.get("validation_rules", {}),
            metadata=metadata
        ) 