#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模式加载器
~~~~~~~~

这个模块负责加载和管理模式定义，提供统一的模式访问接口。

"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Union

from src.config.config_loader import get_config


class PatternLoader:
    """模式加载器类"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(PatternLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化模式加载器"""
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # 加载模式
        self.patterns = self._load_patterns()
        self._compiled_patterns = {}
        self._initialize_compiled_patterns()
        
        self._initialized = True
        
    def _load_patterns(self) -> Dict[str, Any]:
        """加载模式定义
        
        Returns:
            Dict[str, Any]: 模式定义字典
        """
        try:
            # 获取默认模式文件路径
            pattern_file = self.config.get_default_pattern_file()
            
            self.logger.info(f"正在加载模式文件: {pattern_file}")
            
            if not os.path.exists(pattern_file):
                self.logger.warning(f"模式文件不存在: {pattern_file}，使用空模式")
                return {}
                
            with open(pattern_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
                
            self.logger.info(f"成功加载模式文件")
            return patterns.get('patterns', {})
            
        except Exception as e:
            self.logger.error(f"加载模式文件失败: {str(e)}")
            return {}
            
    def _initialize_compiled_patterns(self) -> None:
        """初始化编译后的正则表达式模式"""
        try:
            for problem_type, subtypes in self.patterns.items():
                self._compiled_patterns[problem_type] = {}
                
                for subtype, pattern_info in subtypes.items():
                    self._compiled_patterns[problem_type][subtype] = {}
                    
                    if 'regex_patterns' in pattern_info:
                        for key, patterns in pattern_info['regex_patterns'].items():
                            self._compiled_patterns[problem_type][subtype][key] = [
                                re.compile(pattern) for pattern in patterns
                            ]
                            
            self.logger.debug(f"成功初始化编译后的正则表达式模式")
            
        except Exception as e:
            self.logger.error(f"初始化编译后的正则表达式模式失败: {str(e)}")
            
    def get_pattern(self, problem_type: str, subtype: Optional[str] = None) -> Dict[str, Any]:
        """获取模式定义
        
        Args:
            problem_type: 问题类型
            subtype: 子类型，如果为None，则返回所有子类型
            
        Returns:
            Dict[str, Any]: 模式定义字典
        """
        if problem_type not in self.patterns:
            return {}
            
        if subtype is None:
            return self.patterns[problem_type]
            
        return self.patterns[problem_type].get(subtype, {})
        
    def get_compiled_pattern(self, problem_type: str, subtype: str, key: str) -> List[Pattern]:
        """获取编译后的正则表达式模式
        
        Args:
            problem_type: 问题类型
            subtype: 子类型
            key: 模式键名
            
        Returns:
            List[Pattern]: 编译后的正则表达式模式列表
        """
        if problem_type not in self._compiled_patterns:
            return []
            
        if subtype not in self._compiled_patterns[problem_type]:
            return []
            
        return self._compiled_patterns[problem_type][subtype].get(key, [])
        
    def get_keywords(self, problem_type: str, subtype: Optional[str] = None) -> List[str]:
        """获取关键词
        
        Args:
            problem_type: 问题类型
            subtype: 子类型，如果为None，则返回所有子类型的关键词
            
        Returns:
            List[str]: 关键词列表
        """
        if problem_type not in self.patterns:
            return []
            
        if subtype is None:
            # 返回所有子类型的关键词
            keywords = []
            for subtype_info in self.patterns[problem_type].values():
                keywords.extend(subtype_info.get('keywords', []))
            return keywords
            
        if subtype not in self.patterns[problem_type]:
            return []
            
        return self.patterns[problem_type][subtype].get('keywords', [])
        
    def get_equation_template(self, problem_type: str, subtype: str) -> str:
        """获取方程模板
        
        Args:
            problem_type: 问题类型
            subtype: 子类型
            
        Returns:
            str: 方程模板
        """
        if problem_type not in self.patterns:
            return ""
            
        if subtype not in self.patterns[problem_type]:
            return ""
            
        return self.patterns[problem_type][subtype].get('equation_template', "")
        
    def get_subtypes(self, problem_type: str) -> List[str]:
        """获取子类型
        
        Args:
            problem_type: 问题类型
            
        Returns:
            List[str]: 子类型列表
        """
        if problem_type not in self.patterns:
            return []
            
        return list(self.patterns[problem_type].keys())
        
    def match_pattern(self, text: str, problem_type: str, subtype: str, key: str) -> List[Dict[str, Any]]:
        """匹配模式
        
        Args:
            text: 文本
            problem_type: 问题类型
            subtype: 子类型
            key: 模式键名
            
        Returns:
            List[Dict[str, Any]]: 匹配结果列表，每个元素包含匹配的值和单位
        """
        patterns = self.get_compiled_pattern(problem_type, subtype, key)
        if not patterns:
            return []
            
        matches = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = match.group(1)
                unit = match.group(0).replace(value, '').strip()
                matches.append({
                    'value': float(value),
                    'unit': unit,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
                
        return matches
        
    def identify_subtype(self, text: str, problem_type: str) -> str:
        """识别子类型
        
        Args:
            text: 文本
            problem_type: 问题类型
            
        Returns:
            str: 子类型
        """
        if problem_type not in self.patterns:
            return "unknown"
            
        subtypes = self.get_subtypes(problem_type)
        for subtype in subtypes:
            keywords = self.get_keywords(problem_type, subtype)
            if any(keyword in text for keyword in keywords):
                return subtype
                
        # 返回第一个子类型作为默认值
        return subtypes[0] if subtypes else "unknown"
        
    def extract_variables(self, text: str, problem_type: str, subtype: Optional[str] = None) -> Dict[str, Any]:
        """提取变量
        
        Args:
            text: 文本
            problem_type: 问题类型
            subtype: 子类型，如果为None，则自动识别
            
        Returns:
            Dict[str, Any]: 提取的变量、单位和子类型
        """
        if problem_type not in self.patterns:
            return {'variables': {}, 'units': {}, 'subtype': 'unknown'}
            
        # 如果未指定子类型，则自动识别
        if subtype is None or subtype == 'unknown':
            subtype = self.identify_subtype(text, problem_type)
            
        if subtype not in self.patterns[problem_type]:
            # 使用第一个子类型作为默认值
            subtypes = self.get_subtypes(problem_type)
            if subtypes:
                subtype = subtypes[0]
            else:
                return {'variables': {}, 'units': {}, 'subtype': 'unknown'}
                
        # 获取正则表达式模式
        pattern_info = self.patterns[problem_type][subtype]
        regex_patterns = pattern_info.get('regex_patterns', {})
        
        variables = {}
        units = {}
        
        # 提取变量
        for key, patterns in regex_patterns.items():
            for pattern_str in patterns:
                pattern = re.compile(pattern_str)
                matches = list(pattern.finditer(text))
                
                if matches:
                    # 提取数值
                    value = float(matches[0].group(1))
                    
                    # 提取单位
                    unit_text = matches[0].group(0).replace(matches[0].group(1), '').strip()
                    
                    # 标准化单位
                    unit = self._normalize_unit(unit_text, key)
                    
                    # 存储变量和单位
                    if key == 'speed' and 'speed' in variables:
                        # 如果已经有速度了，则添加为第二个速度
                        if isinstance(variables['speed'], list):
                            variables['speed'].append(value)
                        else:
                            variables['speed'] = [variables['speed'], value]
                    elif key == 'time' and 'time_a' in variables:
                        # 如果已经有时间了，则添加为第二个时间
                        variables['time_b'] = value
                        units['time_b'] = unit
                    elif key == 'time' and 'time_a' not in variables:
                        # 第一个时间
                        variables['time_a'] = value
                        units['time_a'] = unit
                    else:
                        variables[key] = value
                        units[key] = unit
                    
                    break
        
        # 特殊处理：运动问题
        if problem_type == 'motion':
            # 提取两个速度
            speed_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:km/h|公里/小时|千米/小时|公里每小时|千米每小时|公里/h|千米/h|km/小时|km每小时)")
            speeds = [float(m.group(1)) for m in speed_pattern.finditer(text)]
            
            if len(speeds) >= 2:
                variables['speeds'] = speeds[:2]
                units['speeds'] = 'km/h'
            elif len(speeds) == 1:
                variables['speed'] = speeds[0]
                units['speed'] = 'km/h'
                
            # 提取距离
            distance_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:km|公里|千米)")
            distances = [float(m.group(1)) for m in distance_pattern.finditer(text)]
            
            if distances:
                variables['distance'] = distances[0]
                units['distance'] = 'km'
        
        # 特殊处理：工作效率问题
        elif problem_type == 'work_efficiency':
            # 提取效率比
            ratio_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*倍")
            ratios = [float(m.group(1)) for m in ratio_pattern.finditer(text)]
            
            if ratios:
                variables['efficiency_ratio'] = ratios[0]
                
            # 提取时间
            time_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:小时|h)")
            times = [float(m.group(1)) for m in time_pattern.finditer(text)]
            
            if len(times) >= 2:
                variables['time_a'] = times[0]
                variables['time_b'] = times[1]
                units['time_a'] = 'h'
                units['time_b'] = 'h'
            elif len(times) == 1:
                variables['time_a'] = times[0]
                units['time_a'] = 'h'
        
        # 特殊处理：速率和体积问题
        elif problem_type == 'rate_and_volume':
            if subtype == 'ice_cube_problem':
                # 提取冰块体积
                volume_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ml|毫升)")
                volumes = [float(m.group(1)) for m in volume_pattern.finditer(text)]
                
                if volumes:
                    variables['ice_cube_volume'] = volumes[0]
                    units['ice_cube_volume'] = 'ml'
                
                # 提取初始水量和目标水量
                water_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:L|升)")
                waters = [float(m.group(1)) for m in water_pattern.finditer(text)]
                
                if len(waters) >= 2:
                    variables['initial_volume'] = waters[0] * 1000  # 转换为毫升
                    variables['target_volume'] = waters[1] * 1000  # 转换为毫升
                    units['initial_volume'] = 'ml'
                    units['target_volume'] = 'ml'
                elif len(waters) == 1:
                    variables['initial_volume'] = waters[0] * 1000  # 转换为毫升
                    units['initial_volume'] = 'ml'
                
                # 提取漏水率
                leak_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ml|毫升)/(?:s|秒)")
                leaks = [float(m.group(1)) for m in leak_pattern.finditer(text)]
                
                if leaks:
                    variables['leak_rate'] = leaks[0]
                    units['leak_rate'] = 'ml/s'
                
                # 提取冰块放入速率
                cube_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*个/(?:min|分钟)")
                cubes = [float(m.group(1)) for m in cube_pattern.finditer(text)]
                
                if cubes:
                    variables['cube_rate'] = cubes[0]
                    units['cube_rate'] = '个/min'
                else:
                    # 默认每分钟放入一个冰块
                    variables['cube_rate'] = 1.0
                    units['cube_rate'] = '个/min'
        
        return {
            'variables': variables,
            'units': units,
            'subtype': subtype
        }
        
    def _normalize_unit(self, unit_text: str, key: str) -> str:
        """标准化单位
        
        Args:
            unit_text: 单位文本
            key: 变量键名
            
        Returns:
            str: 标准化后的单位
        """
        # 速度单位
        if key == 'speed':
            if any(u in unit_text for u in ['km/h', '公里/小时', '千米/小时', '公里每小时', '千米每小时', '公里/h', '千米/h', 'km/小时', 'km每小时']):
                return 'km/h'
            elif any(u in unit_text for u in ['m/s', '米/秒', '米每秒']):
                return 'm/s'
        
        # 距离单位
        elif key == 'distance':
            if any(u in unit_text for u in ['km', '公里', '千米']):
                return 'km'
            elif any(u in unit_text for u in ['m', '米']):
                return 'm'
        
        # 时间单位
        elif key == 'time':
            if any(u in unit_text for u in ['h', '小时', '时']):
                return 'h'
            elif any(u in unit_text for u in ['min', '分钟', '分']):
                return 'min'
            elif any(u in unit_text for u in ['s', '秒']):
                return 's'
        
        # 体积单位
        elif key == 'volume' or key == 'water_volume':
            if any(u in unit_text for u in ['L', '升', '立升', 'l']):
                return 'L'
            elif any(u in unit_text for u in ['ml', 'mL', '毫升']):
                return 'ml'
            elif any(u in unit_text for u in ['cm³', '立方厘米', '立方cm', 'cm3']):
                return 'ml'  # 1立方厘米 = 1毫升
        
        # 速率单位
        elif key == 'rate' or key == 'leak_rate':
            if any(u in unit_text for u in ['ml/s', '毫升/秒', '毫升每秒']):
                return 'ml/s'
            elif any(u in unit_text for u in ['L/min', '升/分钟', '升每分钟']):
                return 'L/min'
        
        # 默认返回原始单位
        return unit_text
        
    def reload(self) -> None:
        """重新加载模式定义"""
        self.patterns = self._load_patterns()
        self._compiled_patterns = {}
        self._initialize_compiled_patterns()
        self.logger.info("模式已重新加载")
        
    def __str__(self) -> str:
        """返回模式加载器字符串表示"""
        return f"PatternLoader(patterns={len(self.patterns)})"
        
    def __repr__(self) -> str:
        """返回模式加载器字符串表示"""
        return self.__str__()


# 创建全局模式加载器实例
pattern_loader = PatternLoader()

def get_pattern_loader() -> PatternLoader:
    """获取模式加载器实例
    
    Returns:
        PatternLoader: 模式加载器实例
    """
    return PatternLoader() 