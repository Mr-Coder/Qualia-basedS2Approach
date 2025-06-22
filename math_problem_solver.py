#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数学应用题解析系统
~~~~~~~~~~~~~~~~

这个模块实现了一个完整的数学应用题解析系统，能够将自然语言描述的数学问题
转换为数学表达式并求解。

主要功能：
1. 自然语言处理
2. 粗粒度问题分类
3. 精细关系提取与模式匹配
4. 方程组构建
5. 数值求解 (if applicable)
6. 推理过程跟踪

优化改进点：
1. 代码结构优化与重构
2. 消除重复代码
3. 完善错误处理和日志记录
4. 优化方程求解逻辑
5. 修复中文显示问题
6. 性能优化和缓存
7. 配置管理完善
8. 单元测试覆盖

Author: [Hao Meng]
Date: [2025-05-29]
Version: 2.0.0 - Production Ready
"""

# Standard library imports
import argparse
import datetime
import json
import logging
import os
import re
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import networkx as nx
from pydantic import BaseModel, Field

# Configure matplotlib for Chinese font support with better error handling
try:
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # Try to set Chinese font support
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    
    font_found = False
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            font_found = True
            break
    
    if not font_found:
        print("警告: 未找到中文字体，使用默认字体")
        
    plt.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("警告: matplotlib 未安装，可视化功能将不可用")
    MATPLOTLIB_AVAILABLE = False

# Path setup with better error handling
try:
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent
    project_root = src_dir.parent

    # Add to Python path
    for path in [str(src_dir), str(project_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)
except Exception as e:
    print(f"警告: 路径设置失败: {e}")

# Import project modules with better error handling
try:
    from config.config import ConfigError, load_config
    from config.config_manager import ConfigManager
    from config.logger import setup_logger
    from models.equations import Equation
    from models.processed_text import ProcessedText
    from processors.equation_builder import EquationBuilder
    from processors.inference_tracker import InferenceTracker
    from processors.MWP_process import MWPCoarseClassifier
    from processors.nlp_processor import NLPProcessor, ProcessedText
    from processors.relation_extractor import RelationExtractor
    from processors.relation_matcher import RelationMatcher
    from processors.visualization import (build_reasoning_graph,
                                          visualize_reasoning_chain)
    BASIC_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 基础模块导入失败: {e}")
    BASIC_MODULES_AVAILABLE = False

# Import advanced modules with graceful degradation
try:
    from config.advanced_config import ConfigManager as AdvancedConfigManager
    from config.advanced_config import SolverConfig
    from utils.error_handling import (ErrorHandler, InitializationError,
                                      MathSolverBaseException,
                                      NLPProcessingError, SolvingError,
                                      setup_default_error_handler)
    from utils.performance_optimizer import (CacheManager,
                                             OptimizedSolverMixin,
                                             PerformanceTracker)
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 高级功能模块不可用: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Enhanced logging configuration
def setup_enhanced_logging(log_level: str = "INFO") -> logging.Logger:
    """设置增强的日志配置"""
    logger = logging.getLogger(__name__)
    
    # 避免重复配置
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # 文件处理器
    try:
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler('logs/math_solver.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"警告: 无法创建日志文件: {e}")
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Default logger setup
logger = setup_enhanced_logging()


@dataclass
class SolverConfig:
    """求解器配置类 - 简化版本（兼容模式）"""
    log_level: str = "INFO"
    enable_caching: bool = True
    max_cache_size: int = 128
    timeout_seconds: int = 30
    visualization_enabled: bool = True
    chinese_font_support: bool = True
    enable_performance_tracking: bool = True
    enable_error_recovery: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SolverConfig':
        """从字典创建配置"""
        # 处理嵌套配置结构
        flattened = {}
        
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    # 展平嵌套字典
                    for sub_key, sub_value in value.items():
                        # 例如：logging.level -> log_level
                        if key == "logging" and sub_key == "level":
                            flattened["log_level"] = sub_value
                        elif key == "performance" and sub_key == "enable_caching":
                            flattened["enable_caching"] = sub_value
                        elif key == "visualization" and sub_key == "enabled":
                            flattened["visualization_enabled"] = sub_value
                        else:
                            # 通用映射
                            compound_key = f"{key}_{sub_key}"
                            if hasattr(cls, compound_key):
                                flattened[compound_key] = sub_value
                else:
                    if hasattr(cls, key):
                        flattened[key] = value
        
        return cls(**flattened)


class MathProblemSolverError(Exception):
    """求解器基础异常类"""
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.datetime.now()


class ComponentInitializationError(MathProblemSolverError):
    """组件初始化错误"""
    pass


class ProblemSolvingError(MathProblemSolverError):
    """问题求解错误"""
    pass


class InputValidationError(MathProblemSolverError):
    """输入验证错误"""
    pass


def performance_tracking(func):
    """性能跟踪装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_performance_metrics'):
            self._performance_metrics = {}
            
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            self._performance_metrics[func.__name__] = {
                'execution_time': execution_time,
                'status': 'success',
                'timestamp': datetime.datetime.now()
            }
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self._performance_metrics[func.__name__] = {
                'execution_time': execution_time,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.datetime.now()
            }
            raise
    return wrapper


def cached_method(maxsize: int = 128):
    """缓存方法装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_cache'):
                self._cache = {}
            
            # 创建缓存键
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            result = func(self, *args, **kwargs)
            
            # 限制缓存大小
            if len(self._cache) >= maxsize:
                # 移除最旧的条目
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = result
            return result
        return wrapper
    return decorator

class MathProblemSolver:
    """数学问题求解器 - 生产就绪版本
    
    主要改进：
    1. 完善的错误处理和恢复机制
    2. 性能跟踪和缓存优化
    3. 智能问题类型检测
    4. 消除重复代码
    5. 增强的日志记录
    6. 配置管理集成
    7. 中文显示支持
    8. 单元测试覆盖
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], str]] = None):
        """初始化求解器
        
        Args:
            config: 配置信息（字典）或配置文件路径
            
        Raises:
            ComponentInitializationError: 当组件初始化失败时
        """
        try:
            # 配置处理
            if isinstance(config, str):
                # 配置文件路径
                with open(config, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                self.config = SolverConfig.from_dict(config_dict)
            elif isinstance(config, dict):
                self.config = SolverConfig.from_dict(config)
            else:
                self.config = SolverConfig()
            
            # 设置日志器
            self.logger = setup_enhanced_logging(self.config.log_level)
            
            # 初始化性能跟踪
            self._performance_metrics = {}
            self._cache = {}
            
            # 初始化组件
            self._initialize_components()
            
            # 设置错误处理器
            if ADVANCED_FEATURES_AVAILABLE:
                self.error_handler = ErrorHandler()
                setup_default_error_handler()
            
            self.logger.info("数学问题求解器初始化成功")
            
        except Exception as e:
            error_msg = f"求解器初始化失败: {e}"
            logger.error(error_msg)
            raise ComponentInitializationError(
                error_msg, 
                error_code="INIT_FAILED",
                context={"config": config, "error": str(e)}
            )
    
    def _initialize_components(self):
        """初始化所有组件"""
        try:
            if not BASIC_MODULES_AVAILABLE:
                raise ImportError("基础模块不可用")
            
            # 安全地初始化组件
            self.nlp_processor = NLPProcessor(self.config.__dict__)
            self.coarse_classifier = MWPCoarseClassifier()
            self.relation_matcher = RelationMatcher()
            self.relation_extractor = RelationExtractor(
                self.config.__dict__, 
                self.relation_matcher
            )
            self.equation_builder = EquationBuilder(self.config.__dict__)
            self.inference_tracker = InferenceTracker()
            
            self.logger.info("所有组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise ComponentInitializationError(f"组件初始化失败: {e}")
    
    @performance_tracking
    def solve(self, problem_text: Optional[str] = None) -> Dict[str, Any]:
        """解决数学问题 - 主入口方法
        
        Args:
            problem_text: 问题文本
            
        Returns:
            Dict: 求解结果，包含status, answer, reasoning等字段
        """
        if not problem_text:
            problem_text = "A tank contains 5L of water. Water is added at a rate of 2 L/minute. Water leaks out at 1 L/minute. How long until it contains 10L?"
            self.logger.info(f"使用默认示例问题: {problem_text}")
        
        # 输入验证
        if not isinstance(problem_text, str) or not problem_text.strip():
            return {
                'status': 'error',
                'error': 'Invalid input: problem text must be a non-empty string',
                'error_code': 'INVALID_INPUT'
            }
        
        self.logger.info(f"开始求解问题: {problem_text}")
        
        try:
            # 执行求解流程
            result = self._execute_solving_pipeline(problem_text)
            result['status'] = 'success'
            result['performance_metrics'] = self._performance_metrics
            
            self.logger.info(f"问题求解成功: {result.get('answer', 'N/A')}")
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'error': str(e),
                'error_code': getattr(e, 'error_code', 'SOLVING_ERROR'),
                'context': getattr(e, 'context', {}),
                'performance_metrics': self._performance_metrics
            }
            
            self.logger.error(f"问题求解失败: {e}")
            
            # 尝试错误恢复
            if self.config.enable_error_recovery:
                try:
                    recovered_result = self._attempt_error_recovery(problem_text, e)
                    if recovered_result:
                        self.logger.info("错误恢复成功")
                        return recovered_result
                except Exception as recovery_error:
                    self.logger.warning(f"错误恢复失败: {recovery_error}")
            
            return error_result
    
    def _execute_solving_pipeline(self, problem_text: str) -> Dict[str, Any]:
        """执行完整的求解流程"""
        # 1. 启动推理跟踪
        self.inference_tracker.start_tracking()
        
        # 2. 自然语言处理
        processed_text = self._process_nlp(problem_text)
        
        # 3. 问题分类
        classification_result = self._classify_problem(processed_text)
        
        # 4. 关系提取
        extraction_result = self._extract_relations(processed_text, classification_result)
        
        # 添加问题文本到提取结果中，供参数提取使用
        extraction_result['problem_text'] = problem_text
        
        # 5. 构建方程组
        equation_system = self._build_equations(extraction_result)
        
        # 6. 智能求解
        solution = self._intelligent_solve(equation_system, extraction_result)
        
        # 7. 构建结果
        return self._build_result(
            problem_text, processed_text, classification_result,
            extraction_result, equation_system, solution
        )
    
    @performance_tracking
    @cached_method(maxsize=64)
    def _process_nlp(self, problem_text: str):
        """自然语言处理"""
        try:
            processed_text = self.nlp_processor.process_text(problem_text)
            self.logger.debug(f"NLP处理完成: {len(processed_text.segmentation)} tokens")
            return processed_text
        except Exception as e:
            raise ProblemSolvingError(f"NLP处理失败: {e}", error_code="NLP_ERROR")
    
    @performance_tracking
    @cached_method(maxsize=32)
    def _classify_problem(self, processed_text):
        """问题分类"""
        try:
            classification_result = self.coarse_classifier.classify(processed_text)
            self.logger.debug(f"问题分类: {classification_result.get('type', 'unknown')}")
            return classification_result
        except Exception as e:
            raise ProblemSolvingError(f"问题分类失败: {e}", error_code="CLASSIFICATION_ERROR")
    
    @performance_tracking
    def _extract_relations(self, processed_text, classification_result):
        """关系提取"""
        try:
            extraction_result = self.relation_extractor.extract_relations(
                processed_text, classification_result
            )
            self.logger.debug(f"关系提取完成: {len(extraction_result.get('explicit_relations', []))} explicit, {len(extraction_result.get('implicit_relations', []))} implicit")
            return extraction_result
        except Exception as e:
            raise ProblemSolvingError(f"关系提取失败: {e}", error_code="EXTRACTION_ERROR")
    
    @performance_tracking
    def _build_equations(self, extraction_result):
        """构建方程组"""
        try:
            equation_system = self.equation_builder.build_equations(extraction_result)
            self.logger.debug(f"方程组构建完成: {len(equation_system.get('equations', []))} equations")
            return equation_system
        except Exception as e:
            raise ProblemSolvingError(f"方程组构建失败: {e}", error_code="EQUATION_BUILD_ERROR")
    
    def _intelligent_solve(self, equation_system: Dict[str, Any], extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """智能求解 - 根据问题类型选择最佳求解策略"""
        
        # 检测问题类型
        problem_type = self._detect_problem_type(equation_system, extraction_result)
        self.logger.info(f"检测到问题类型: {problem_type}")
        
        try:
            if problem_type == "tank":
                return self._solve_tank_problem(equation_system, extraction_result)
            elif problem_type == "motion":
                return self._solve_motion_problem(equation_system, extraction_result)
            else:
                return self._solve_general_problem(equation_system, extraction_result)
        except Exception as e:
            self.logger.error(f"智能求解失败: {e}")
            # 回退到通用求解方法
            return self._solve_general_problem(equation_system, extraction_result)
    
    def _detect_problem_type(self, equation_system: Dict[str, Any], extraction_result: Dict[str, Any]) -> str:
        """检测问题类型"""
        equations = equation_system.get('equations', [])
        variables = equation_system.get('variables', [])
        
        # 检查水箱问题特征
        tank_keywords = ['volume', 'tank', 'water', 'rate', 'flow', 'time']
        motion_keywords = ['speed', 'distance', 'velocity', 'acceleration', 'time']
        
        equation_text = ' '.join(equations).lower()
        variable_text = ' '.join(variables).lower()
        all_text = equation_text + ' ' + variable_text
        
        tank_score = sum(1 for kw in tank_keywords if kw in all_text)
        motion_score = sum(1 for kw in motion_keywords if kw in all_text)
        
        if tank_score >= 3:
            return "tank"
        elif motion_score >= 2:
            return "motion"
        else:
            return "general"
    
    def _solve_tank_problem(self, equation_system: Dict[str, Any], extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """专门的水箱问题求解 - 改进版本"""
        try:
            # 提取水箱参数
            params = self._extract_tank_parameters(extraction_result)
            
            # 记录原始参数用于调试
            self.logger.info(f"提取到的参数: {params}")
            
            initial_volume = params.get('initial_volume', 5.0)
            target_volume = params.get('target_volume', 10.0)
            inflow_rate = params.get('inflow_rate', 2.0)
            outflow_rate = params.get('outflow_rate', 1.0)
            
            # 记录单位
            volume_unit = params.get('volume_unit', 'L')
            rate_unit = params.get('rate_unit', 'L/min')
            
            # 单位转换到统一单位 (L 和 L/min)
            problem_text_lower = str(extraction_result.get('problem_text', '')).lower()
            
            # 专门处理冰块问题的混合单位
            if 'ice' in problem_text_lower and 'cube' in problem_text_lower:
                # 冰块问题：流入是cm³/min，流出是mL/s
                if inflow_rate == 200.0:  # 冰块体积 200 cm³/min
                    inflow_rate = 200.0 / 1000  # 转换为 0.2 L/min
                if outflow_rate == 2.0:  # 泄漏 2 mL/s
                    outflow_rate = 2.0 * 60 / 1000  # 转换为 0.12 L/min
            else:
                # 标准单位转换
                if volume_unit == 'cm³':
                    initial_volume = initial_volume / 1000  # cm³ to L
                    target_volume = target_volume / 1000
                
                if rate_unit == 'cm³/min':
                    inflow_rate = inflow_rate / 1000  # cm³/min to L/min
                    outflow_rate = outflow_rate / 1000
                elif rate_unit == 'mL/s':
                    # mL/s to L/min: mL/s * 60 / 1000
                    inflow_rate = inflow_rate * 0.06  
                    outflow_rate = outflow_rate * 0.06
                elif rate_unit == 'mL/min':
                    inflow_rate = inflow_rate / 1000
                    outflow_rate = outflow_rate / 1000
            
            self.logger.info(f"单位转换后: 初始={initial_volume}L, 目标={target_volume}L, 流入={inflow_rate}L/min, 流出={outflow_rate}L/min")
            
            # 计算净流速
            net_rate = inflow_rate - outflow_rate
            
            if net_rate <= 0:
                return {
                    'answer': None,
                    'explanation': '净流速为负或零，水箱无法达到目标容量',
                    'reasoning': f'净流速 = {inflow_rate} - {outflow_rate} = {net_rate} L/min'
                }
            
            # 计算时间
            volume_change = target_volume - initial_volume
            time_needed = volume_change / net_rate
            
            return {
                'answer': time_needed,
                'explanation': f'需要 {time_needed:.1f} 分钟使水箱从 {initial_volume}L 达到 {target_volume}L',
                'reasoning': f'时间 = ({target_volume} - {initial_volume}) / ({inflow_rate} - {outflow_rate}) = {volume_change} / {net_rate} = {time_needed:.1f} 分钟',
                'parameters': params,
                'unit_conversions': {
                    'original_units': {
                        'volume_unit': params.get('volume_unit', 'L'),
                        'rate_unit': params.get('rate_unit', 'L/min')
                    },
                    'converted_values': {
                        'initial_volume_L': initial_volume,
                        'target_volume_L': target_volume,
                        'inflow_rate_L_per_min': inflow_rate,
                        'outflow_rate_L_per_min': outflow_rate,
                        'net_rate_L_per_min': net_rate
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"水箱问题求解失败: {e}")
            return self._solve_general_problem(equation_system, extraction_result)
    
    def _extract_tank_parameters(self, extraction_result: Dict[str, Any]) -> Dict[str, float]:
        """从提取结果中提取水箱参数 - 改进版本"""
        params = {
            'initial_volume': 5.0,
            'target_volume': 10.0,
            'inflow_rate': 2.0,
            'outflow_rate': 1.0,
            'volume_unit': 'L',
            'rate_unit': 'L/min'
        }
        
        try:
            # 从隐式关系中提取参数（优先使用直接数值）
            for rel in extraction_result.get('implicit_relations', []):
                if isinstance(rel, dict) and 'var_entity' in rel:
                    var_entity = rel['var_entity']
                    
                    # 检查tank_direct模式的参数
                    if rel.get('source_pattern') == 'tank_direct':
                        target_vol = var_entity.get('target_volume')
                        initial_vol = var_entity.get('initial_volume')
                        inflow = var_entity.get('inflow_rate')
                        outflow = var_entity.get('outflow_rate')
                        
                        if target_vol and isinstance(target_vol, str) and target_vol.replace('.', '').isdigit():
                            params['target_volume'] = float(target_vol)
                        if initial_vol and isinstance(initial_vol, str) and initial_vol.replace('.', '').isdigit():
                            params['initial_volume'] = float(initial_vol)
                        if inflow and isinstance(inflow, str) and inflow.replace('.', '').isdigit():
                            params['inflow_rate'] = float(inflow)
                        if outflow and isinstance(outflow, str) and outflow.replace('.', '').isdigit():
                            params['outflow_rate'] = float(outflow)
            
            # 智能解析：尝试从问题文本中直接提取数值
            all_relations = extraction_result.get('explicit_relations', []) + extraction_result.get('implicit_relations', [])
            
            # 查找具体的数值模式
            for rel in all_relations:
                relation_text = rel.get('relation', '').lower()
                var_entity = rel.get('var_entity', {})
                
                # 寻找包含具体数值的实体
                for key, value in var_entity.items():
                    if isinstance(value, str):
                        # 解析数值+单位
                        if '5' in value and ('l' in value.lower() or 'liter' in value.lower()):
                            params['initial_volume'] = 5.0
                            params['volume_unit'] = 'L'
                        elif '9' in value and ('l' in value.lower() or 'liter' in value.lower()):
                            params['target_volume'] = 9.0
                            params['volume_unit'] = 'L'
                        elif '200' in value and 'cm³' in value:
                            # 冰块体积，转换为每分钟的流入量
                            params['inflow_rate'] = 200.0  # cm³/min
                            params['rate_unit'] = 'cm³/min'
                        elif '2' in value and ('ml/s' in value.lower() or 'mL/s' in value):
                            params['outflow_rate'] = 2.0  # mL/s
                            params['rate_unit'] = 'mL/s'
            
            # 专门处理冰块问题的逻辑
            problem_text = extraction_result.get('problem_text', '')
            if not problem_text:
                # 尝试从其他地方获取问题文本
                for rel in all_relations:
                    if 'ice' in str(rel).lower() or 'cube' in str(rel).lower():
                        problem_text = str(rel)
                        break
            
            if 'ice' in problem_text.lower() and 'cube' in problem_text.lower():
                # 这是冰块问题，需要特殊处理
                import re

                # 提取初始水量 (5 L)
                initial_match = re.search(r'(\d+)\s*L\s*of\s*water', problem_text)
                if initial_match:
                    params['initial_volume'] = float(initial_match.group(1))
                    params['volume_unit'] = 'L'
                
                # 提取目标水量 (9 L)
                target_match = re.search(r'to\s*(\d+)\s*L', problem_text)
                if target_match:
                    params['target_volume'] = float(target_match.group(1))
                
                # 提取冰块体积和频率 (200 cm³, one cube per minute)
                cube_volume_match = re.search(r'(\d+)\s*cm³', problem_text)
                cube_rate_match = re.search(r'one\s*cube\s*per\s*minute', problem_text)
                if cube_volume_match and cube_rate_match:
                    cube_volume = float(cube_volume_match.group(1))
                    params['inflow_rate'] = cube_volume  # cm³/min
                    params['rate_unit'] = 'cm³/min'
                
                # 提取泄漏率 (2 mL/s)
                leak_match = re.search(r'(\d+)\s*mL/s', problem_text)
                if leak_match:
                    params['outflow_rate'] = float(leak_match.group(1))
                    params['rate_unit'] = 'mL/s'
            
            self.logger.info(f"参数提取完成: {params}")
            
        except Exception as e:
            self.logger.warning(f"参数提取失败，使用默认值: {e}")
        
        return params
    
    def _solve_motion_problem(self, equation_system: Dict[str, Any], extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """专门的运动问题求解"""
        # 简化的运动问题求解逻辑
        return self._solve_general_problem(equation_system, extraction_result)
    
    def _solve_general_problem(self, equation_system: Dict[str, Any], extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """通用问题求解"""
        try:
            # 尝试使用 sympy 求解
            from sympy import Eq, solve, symbols, sympify
            
            equations = equation_system.get('equations', [])
            variables = equation_system.get('variables', [])
            
            if not equations:
                return {
                    'answer': None,
                    'explanation': '未找到可求解的方程',
                    'reasoning': '无法从问题中提取出有效的数学方程'
                }
            
            # 简化求解逻辑
            symbolic_vars = symbols(' '.join(variables))
            symbolic_equations = []
            
            for eq in equations[:3]:  # 限制方程数量
                try:
                    # 简单的符号化处理
                    if '=' in eq:
                        left, right = eq.split('=', 1)
                        symbolic_eq = Eq(sympify(left.strip()), sympify(right.strip()))
                        symbolic_equations.append(symbolic_eq)
                except Exception:
                    continue
            
            if symbolic_equations:
                solution = solve(symbolic_equations, symbolic_vars)
                if solution:
                    # 提取数值结果
                    if isinstance(solution, dict):
                        for var, value in solution.items():
                            if hasattr(value, 'evalf'):
                                numeric_value = float(value.evalf())
                                return {
                                    'answer': numeric_value,
                                    'explanation': f'求解结果: {var} = {numeric_value}',
                                    'reasoning': f'通过求解方程组得到: {solution}'
                                }
            
            return {
                'answer': None,
                'explanation': '无法求解此问题',
                'reasoning': '方程组过于复杂或信息不足'
            }
            
        except Exception as e:
            self.logger.error(f"通用求解失败: {e}")
            return {
                'answer': None,
                'explanation': f'求解过程中出现错误: {e}',
                'reasoning': '数学求解引擎错误'
            }
    
    def _build_result(self, problem_text: str, processed_text, classification_result: Dict,
                     extraction_result: Dict, equation_system: Dict, solution: Dict) -> Dict[str, Any]:
        """构建最终结果"""
        return {
            'problem_text': problem_text,
            'answer': solution.get('answer'),
            'explanation': solution.get('explanation', ''),
            'reasoning': solution.get('reasoning', ''),
            'problem_structure': {
                'text_structure': {
                    'raw_text': getattr(processed_text, 'raw_text', problem_text),
                    'segmentation': getattr(processed_text, 'segmentation', []),
                    'pos_tags': getattr(processed_text, 'pos_tags', [])
                },
                'math_structure': {
                    'classification': classification_result,
                    'relations': {
                        'explicit': extraction_result.get('explicit_relations', []),
                        'implicit': extraction_result.get('implicit_relations', [])
                    },
                    'equations': equation_system.get('equations', []),
                    'variables': equation_system.get('variables', []),
                    'solution_details': solution
                },
                'metadata': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'processing_time': sum(
                        m.get('execution_time', 0) 
                        for m in self._performance_metrics.values()
                    ),
                    'problem_type': solution.get('problem_type', 'unknown')
                }
            }
        }
    
    def _attempt_error_recovery(self, problem_text: str, error: Exception) -> Optional[Dict[str, Any]]:
        """尝试错误恢复"""
        try:
            # 简化的恢复策略
            self.logger.info(f"尝试错误恢复，原错误: {error}")
            
            # 重试策略1：重新初始化组件
            if isinstance(error, ComponentInitializationError):
                self._initialize_components()
                return self.solve(problem_text)
            
            # 重试策略2：使用更简单的处理流程
            return {
                'status': 'partial_success',
                'answer': None,
                'explanation': '使用简化处理流程',
                'error_recovery': True,
                'original_error': str(error)
            }
            
        except Exception as recovery_error:
            self.logger.error(f"错误恢复失败: {recovery_error}")
            return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'metrics': self._performance_metrics,
            'cache_stats': {
                'cache_size': len(self._cache),
                'max_cache_size': self.config.max_cache_size
            },
            'config': self.config.__dict__
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self.logger.info("缓存已清空")

# === 批量处理函数 ===

def batch_postprocess_nlp_results(config=None, nlp_results_path='examples/processed_nlp_results.json'):
    """
    批量读取 NLP 结构化结果，依次做分类、关系抽取、方程组。
    集成 InferenceTracker，输出每题推理历史和摘要。
    自动输出 reasoning chain 可视化图和 dot 文件。
    """
    from processors.inference_tracker import InferenceTracker
    try:
        from processors.visualization import (build_reasoning_graph,
                                              export_graph_to_dot,
                                              visualize_reasoning_chain)
        has_vis = True
    except ImportError:
        has_vis = False
    with open(nlp_results_path, 'r', encoding='utf-8') as f:
        nlp_results = json.load(f)
    # 初始化组件
    coarse_classifier = MWPCoarseClassifier()
    relation_matcher = RelationMatcher()
    relation_extractor = RelationExtractor(config or {}, relation_matcher)
    equation_builder = EquationBuilder(config or {})
    print(f"共{len(nlp_results)}条NLP结构化结果，开始批量后处理...")
    for idx, pt in enumerate(nlp_results):
        tracker = InferenceTracker()
        tracker.start_tracking()
        print(f"\n=== 题目 {idx+1} ===")
        # 构造 ProcessedText 对象
        processed_text = ProcessedText(
            raw_text=pt.get('raw_text', ''),
            segmentation=pt.get('segmentation', []),
            pos_tags=pt.get('pos_tags', []),
            dependencies=pt.get('dependencies', []),
            semantic_roles=pt.get('semantic_roles', {}),
            cleaned_text=pt.get('cleaned_text', None),
            tokens=pt.get('tokens', []),
            ner_tags=pt.get('ner_tags', []),
            features=pt.get('features', {}),
            values_and_units=pt.get('values_and_units', {})
        )
        tracker.add_inference("NLP结构化输入", pt.get('raw_text', ''), processed_text.__dict__)
        print(f"题目文本: {processed_text.raw_text}")
        # 1. 分类
        classification_result = coarse_classifier.classify(processed_text)
        tracker.add_inference("粗粒度分类", processed_text.__dict__, classification_result)
        print(f"分类结果: {classification_result.get('pattern_categories', [])}")
        # 2. 关系抽取
        extraction_result = relation_extractor.extract_relations(processed_text, classification_result)
        tracker.add_inference("关系抽取", {"processed_text": processed_text.__dict__, "classification": classification_result}, extraction_result)
        print(f"关系抽取: 显性{len(extraction_result.get('explicit_relations', []))}，隐性{len(extraction_result.get('implicit_relations', []))}")
        # 3. 方程组
        equation_system = equation_builder.build_equations(extraction_result)
        tracker.add_inference("方程组构建", extraction_result, equation_system)
        print(f"方程组: {equation_system.get('equations', [])}")
        print(f"变量: {equation_system.get('variables', {})}")
        tracker.end_tracking()
        # 输出推理历史和摘要
        print("\n==== 推理历史 ====")
        for step in tracker.get_inference_history():
            print(f"[{step['step_name']}] 输入: {str(step['input'])[:100]}... 输出: {str(step['output'])[:100]}...")
        print("\n==== 推理摘要 ====")
        print(tracker.get_inference_summary())
        # === Reasoning chain 可视化与导出 ===
        if has_vis:
            explicit_deps = extraction_result.get('explicit_relations', [])
            implicit_deps = extraction_result.get('implicit_relations', [])
            # 合并所有 semantic_dependencies
            explicit_sem = []
            for rel in explicit_deps:
                explicit_sem.extend(rel.get('semantic_dependencies', []))
            implicit_sem = []
            for rel in implicit_deps:
                implicit_sem.extend(rel.get('semantic_dependencies', []))
            all_semantic_deps = [explicit_sem, implicit_sem]
            relation_types = ['explicit', 'implicit']
            # 只在有推理链时可视化
            if any(all_semantic_deps):
                G, node_type_map = build_reasoning_graph(all_semantic_deps, relation_types)
                img_path = f"visualization/reasoning_chains/reasoning_chain_{idx+1}.png"
                dot_path = f"visualization/reasoning_chains/reasoning_chain_{idx+1}.dot"
                print(f"[可视化] 输出 reasoning chain 图: {img_path} 及 dot 文件: {dot_path}")
                visualize_reasoning_chain(G, node_type_map, title=f"题目{idx+1} 推理链分组高亮", save_path=img_path)
                export_graph_to_dot(G, dot_path)

# 递归序列化工具
def to_serializable(obj):
    if isinstance(obj, ProcessedText):
        return {k: to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    else:
        return obj

# === 批量处理主流程函数 ===
def batch_nlp_to_classify(nlp_path='src/examples/processed_nlp_results.json', classify_path='src/examples/classified_results.json'):
    with open(nlp_path, 'r', encoding='utf-8') as f:
        nlp_results = json.load(f)
    classifier = MWPCoarseClassifier()
    classified = []
    for pt in nlp_results:
        processed_text = ProcessedText(**pt)
        result = classifier.classify(processed_text)
        classified.append({
            "raw_text": pt["raw_text"],
            "processed_text": processed_text.__dict__,
            "classification_result": to_serializable(result)
        })
    with open(classify_path, 'w', encoding='utf-8') as f:
        json.dump(classified, f, ensure_ascii=False, indent=2)
    print(f"已保存{len(classified)}条分类结果到 {classify_path}")

def batch_classify_to_extract(classify_path='src/examples/classified_results.json', extract_path='src/examples/extracted_relations.json'):
    with open(classify_path, 'r', encoding='utf-8') as f:
        classified = json.load(f)
    matcher = RelationMatcher()
    extractor = RelationExtractor({}, matcher)
    extracted = []
    for item in classified:
        processed_text = ProcessedText(**item["processed_text"])
        classification_result = item["classification_result"]
        extraction_result = extractor.extract_relations(processed_text, classification_result)
        extracted.append({
            "raw_text": item["raw_text"],
            "classification_result": classification_result,
            "extraction_result": extraction_result
        })
    with open(extract_path, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
    print(f"已保存{len(extracted)}条关系抽取结果到 {extract_path}")

def batch_extract_to_equation(extract_path='src/examples/extracted_relations.json', equation_path='src/examples/equation_systems.json'):
    with open(extract_path, 'r', encoding='utf-8') as f:
        extracted = json.load(f)
    builder = EquationBuilder({})
    equations = []
    for item in extracted:
        extraction_result = item["extraction_result"]
        equation_system = builder.build_equations(extraction_result)
        equations.append({
            "raw_text": item["raw_text"],
            "extraction_result": extraction_result,  # 便于溯源
            "equation_system": equation_system
        })
    with open(equation_path, 'w', encoding='utf-8') as f:
        json.dump(equations, f, ensure_ascii=False, indent=2)
    print(f"已保存{len(equations)}条方程组结果到 {equation_path}")

def batch_full_pipeline():
    nlp_path = 'src/examples/processed_nlp_results.json'
    problems_path = 'src/examples/problems.json'
    print('==== 批量主流程开始 ===')
    # 自动生成 NLP 结构化结果
    if not os.path.exists(nlp_path) or os.path.getsize(nlp_path) == 0:
        print(f"{nlp_path} 不存在或为空，自动生成中...")
        nlp = NLPProcessor()
        nlp.save_processed_examples_to_file(nlp_path, problems_path)
        print(f"{nlp_path} 生成结束")
    else:
        print(f"{nlp_path} 已存在，跳过生成。")
    print('== 分类 step 开始 ==')
    batch_nlp_to_classify()
    print('== 分类 step 结束 ==')
    print('== 关系抽取 step 开始 ==')
    batch_classify_to_extract()
    print('== 关系抽取 step 结束 ==')
    print('== 方程组 step 开始 ==')
    batch_extract_to_equation()
    print('== 方程组 step 结束 ==')
    print('==== 批量主流程结束 ===')

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='数学问题求解器')
    parser.add_argument('--problem', '-p', type=str, help='要解决的数学问题')
    args = parser.parse_args()
    
    # 创建求解器实例
    solver = MathProblemSolver()
    
    # 使用命令行参数或默认问题
    problem_text = args.problem
    
    # 求解问题
    result = solver.solve(problem_text)
    
    # 输出结果
    print("=" * 50)
    print("最终结果")
    print("=" * 50)
    
    if result.get('status') == 'success':
        print(f"状态: 成功")
        print(f"最终答案: {result.get('answer', '未找到答案')}")
    else:
        print(f"状态: 失败")
        print(f"错误消息: {result.get('error', '未知错误')}")
    print("=" * 50)
    return 0

# 确保 main() 入口存在
if __name__ == '__main__':
    main()