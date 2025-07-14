"""
模板匹配器
动态匹配文本与模板，支持多模式匹配和置信度计算
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .template_registry import TemplateDefinition, TemplatePattern


@dataclass
class MatchResult:
    """匹配结果"""
    template_id: str
    template_name: str
    category: str
    confidence: float
    matched_pattern: str
    extracted_values: Dict[str, Any]
    solution_template: str
    variables: List[str]


class TemplateMatcher:
    """模板匹配器"""
    
    def __init__(self, registry):
        """
        初始化模板匹配器
        
        Args:
            registry: 模板注册表实例
        """
        self.registry = registry
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 缓存编译的正则表达式
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        
        # 匹配统计
        self.match_stats = {
            "total_matches": 0,
            "successful_matches": 0,
            "average_confidence": 0.0,
            "category_stats": {}
        }
    
    def match_text(self, text: str, categories: Optional[List[str]] = None) -> List[MatchResult]:
        """
        匹配文本与模板
        
        Args:
            text: 待匹配文本
            categories: 限制匹配的分类，None表示匹配所有分类
            
        Returns:
            匹配结果列表，按置信度降序排列
        """
        if not text:
            return []
        
        # 获取候选模板
        candidates = self._get_candidate_templates(categories)
        
        matches = []
        text_lower = text.lower()
        
        for template in candidates:
            match_result = self._match_template(text, text_lower, template)
            if match_result:
                matches.append(match_result)
        
        # 按置信度排序
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        # 更新统计
        self._update_match_stats(matches)
        
        return matches
    
    def match_text_best(self, text: str, categories: Optional[List[str]] = None) -> Optional[MatchResult]:
        """
        获取最佳匹配结果
        
        Args:
            text: 待匹配文本
            categories: 限制匹配的分类
            
        Returns:
            最佳匹配结果，如果没有匹配则返回None
        """
        matches = self.match_text(text, categories)
        return matches[0] if matches else None
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        提取文本中的数字
        
        Args:
            text: 文本
            
        Returns:
            数字列表
        """
        pattern = r'\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches]
    
    def extract_variables(self, text: str, template: TemplateDefinition) -> Dict[str, Any]:
        """
        根据模板提取变量
        
        Args:
            text: 文本
            template: 模板定义
            
        Returns:
            变量字典
        """
        variables = {}
        
        # 提取数字
        numbers = self.extract_numbers(text)
        if numbers:
            variables["numbers"] = numbers
            if len(numbers) >= 1:
                variables["first_number"] = numbers[0]
            if len(numbers) >= 2:
                variables["second_number"] = numbers[1]
                variables["operand1"] = numbers[0]
                variables["operand2"] = numbers[1]
        
        # 根据模板变量提取特定值
        for var_name in template.variables:
            if var_name not in variables:
                # 尝试从文本中提取特定变量
                var_value = self._extract_specific_variable(text, var_name)
                if var_value is not None:
                    variables[var_name] = var_value
        
        return variables
    
    def get_match_statistics(self) -> Dict[str, Any]:
        """
        获取匹配统计信息
        
        Returns:
            统计信息字典
        """
        return {
            **self.match_stats,
            "total_templates": len(self.registry.get_all_templates()),
            "active_templates": len(self.registry.get_active_templates()),
            "categories": list(self.registry.categories.keys())
        }
    
    def _get_candidate_templates(self, categories: Optional[List[str]] = None) -> List[TemplateDefinition]:
        """
        获取候选模板
        
        Args:
            categories: 分类限制
            
        Returns:
            候选模板列表
        """
        if categories:
            candidates = []
            for category in categories:
                candidates.extend(self.registry.get_templates_by_category(category))
            return candidates
        else:
            return self.registry.get_active_templates()
    
    def _match_template(self, text: str, text_lower: str, template: TemplateDefinition) -> Optional[MatchResult]:
        """
        匹配单个模板
        
        Args:
            text: 原始文本
            text_lower: 小写文本
            template: 模板定义
            
        Returns:
            匹配结果
        """
        best_match = None
        best_confidence = 0.0
        best_pattern = None
        
        for pattern in template.patterns:
            # 编译正则表达式（带缓存）
            compiled_pattern = self._get_compiled_pattern(pattern.regex_pattern)
            
            # 尝试匹配
            match = compiled_pattern.search(text_lower)
            if match:
                # 计算置信度
                confidence = self._calculate_confidence(pattern, match, text)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = match
                    best_pattern = pattern.regex_pattern
        
        if best_match:
            # 提取变量
            variables = self.extract_variables(text, template)
            
            # 添加匹配组的值
            if best_match.groups():
                for i, group in enumerate(best_match.groups()):
                    if group:
                        variables[f"group_{i+1}"] = float(group)
            
            return MatchResult(
                template_id=template.template_id,
                template_name=template.name,
                category=template.category,
                confidence=best_confidence,
                matched_pattern=best_pattern,
                extracted_values=variables,
                solution_template=template.solution_template,
                variables=template.variables
            )
        
        return None
    
    def _calculate_confidence(self, pattern: TemplatePattern, match: re.Match, text: str) -> float:
        """
        计算匹配置信度
        
        Args:
            pattern: 模式定义
            match: 正则匹配结果
            text: 原始文本
            
        Returns:
            置信度分数
        """
        # 基础置信度
        base_confidence = pattern.confidence_weight
        
        # 匹配长度权重
        match_length = len(match.group(0))
        text_length = len(text)
        length_ratio = match_length / text_length if text_length > 0 else 0
        
        # 位置权重（匹配在文本中的位置）
        position_ratio = match.start() / text_length if text_length > 0 else 0
        
        # 计算综合置信度
        confidence = base_confidence * 0.6 + length_ratio * 0.3 + (1 - position_ratio) * 0.1
        
        return min(confidence, 1.0)
    
    def _extract_specific_variable(self, text: str, var_name: str) -> Optional[Any]:
        """
        提取特定变量
        
        Args:
            text: 文本
            var_name: 变量名
            
        Returns:
            变量值
        """
        # 根据变量名定义提取规则
        extraction_rules = {
            "operand1": lambda t: self.extract_numbers(t)[0] if self.extract_numbers(t) else None,
            "operand2": lambda t: self.extract_numbers(t)[1] if len(self.extract_numbers(t)) > 1 else None,
            "result": lambda t: None,  # 结果需要计算得出
            "原价": lambda t: self.extract_numbers(t)[0] if self.extract_numbers(t) else None,
            "折扣": lambda t: self.extract_numbers(t)[1] if len(self.extract_numbers(t)) > 1 else None,
            "现价": lambda t: None,  # 需要计算
            "总数": lambda t: self.extract_numbers(t)[0] if self.extract_numbers(t) else None,
            "百分比": lambda t: self.extract_numbers(t)[1] if len(self.extract_numbers(t)) > 1 else None,
            "部分": lambda t: None,  # 需要计算
            "长": lambda t: self.extract_numbers(t)[0] if self.extract_numbers(t) else None,
            "宽": lambda t: self.extract_numbers(t)[1] if len(self.extract_numbers(t)) > 1 else None,
            "面积": lambda t: None,  # 需要计算
            "总和": lambda t: self.extract_numbers(t)[0] if self.extract_numbers(t) else None,
            "数量": lambda t: self.extract_numbers(t)[1] if len(self.extract_numbers(t)) > 1 else None,
            "平均值": lambda t: None,  # 需要计算
            "时间": lambda t: self.extract_numbers(t)[0] if self.extract_numbers(t) else None,
            "单位": lambda t: None,  # 需要从文本中提取
            "结果": lambda t: None,  # 需要计算
            "equation": lambda t: self._extract_equation(t),
            "variable": lambda t: self._extract_variable_name(t),
            "solution": lambda t: None  # 需要计算
        }
        
        rule = extraction_rules.get(var_name)
        if rule:
            return rule(text)
        
        return None
    
    def _extract_equation(self, text: str) -> Optional[str]:
        """提取方程"""
        # 简单的方程提取
        equation_patterns = [
            r'[a-zA-Z]\s*[+\-]\s*\d+\s*=\s*\d+',
            r'\d*[a-zA-Z]\s*=\s*\d+',
            r'\d+\s*[+\-×÷]\s*\d+\s*=\s*\d+'
        ]
        
        for pattern in equation_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_variable_name(self, text: str) -> Optional[str]:
        """提取变量名"""
        # 提取字母变量
        var_match = re.search(r'[a-zA-Z]', text)
        if var_match:
            return var_match.group(0)
        
        return None
    
    def _get_compiled_pattern(self, pattern: str) -> re.Pattern:
        """
        获取编译的正则表达式（带缓存）
        
        Args:
            pattern: 正则表达式字符串
            
        Returns:
            编译后的正则表达式
        """
        if pattern not in self._compiled_patterns:
            try:
                self._compiled_patterns[pattern] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                self.logger.warning(f"无效的正则表达式: {pattern}, 错误: {e}")
                # 返回一个不匹配任何内容的模式
                self._compiled_patterns[pattern] = re.compile(r'(?!.*)')
        
        return self._compiled_patterns[pattern]
    
    def _update_match_stats(self, matches: List[MatchResult]):
        """更新匹配统计"""
        self.match_stats["total_matches"] += 1
        
        if matches:
            self.match_stats["successful_matches"] += 1
            
            # 更新平均置信度
            total_confidence = sum(m.confidence for m in matches)
            avg_confidence = total_confidence / len(matches)
            
            current_avg = self.match_stats["average_confidence"]
            total_matches = self.match_stats["successful_matches"]
            
            # 移动平均
            self.match_stats["average_confidence"] = (
                (current_avg * (total_matches - 1) + avg_confidence) / total_matches
            )
            
            # 更新分类统计
            for match in matches:
                category = match.category
                if category not in self.match_stats["category_stats"]:
                    self.match_stats["category_stats"][category] = {
                        "matches": 0,
                        "avg_confidence": 0.0
                    }
                
                cat_stats = self.match_stats["category_stats"][category]
                cat_stats["matches"] += 1
                
                # 更新分类平均置信度
                current_cat_avg = cat_stats["avg_confidence"]
                cat_matches = cat_stats["matches"]
                cat_stats["avg_confidence"] = (
                    (current_cat_avg * (cat_matches - 1) + match.confidence) / cat_matches
                ) 