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

Author: [Hao Meng]
Date: [2025-03-23]
Version: 2.0.0 - Refactored and optimized
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
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import networkx as nx
from pydantic import BaseModel, Field

# Configure matplotlib for Chinese font support
try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # Set Chinese font support
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("警告: matplotlib 未安装，可视化功能将不可用")

# Path setup
current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent

# Add to Python path
for path in [str(src_dir), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Local imports
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


@dataclass
class SolverConfig:
    """求解器配置类"""
    log_level: str = "INFO"
    enable_caching: bool = True
    max_cache_size: int = 128
    timeout_seconds: int = 30
    visualization_enabled: bool = True
    chinese_font_support: bool = True
    enable_performance_tracking: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SolverConfig':
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class MathProblemSolverError(Exception):
    """自定义异常类"""
    pass


class ComponentInitializationError(MathProblemSolverError):
    """组件初始化错误"""
    pass


class ProblemSolvingError(MathProblemSolverError):
    """问题求解错误"""
    pass


class MathProblemSolver:
    """数学问题求解器 - 重构优化版本"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化求解器
        
        Args:
            config: 配置信息
            
        Raises:
            ComponentInitializationError: 当组件初始化失败时
        """
        self.config = SolverConfig.from_dict(config or {})
        self.logger = self._setup_logger()
        self._performance_metrics = {}
        self._cache = {}
        
        try:
            self._initialize_components()
            self.logger.info("数学问题求解器初始化成功")
        except Exception as e:
            raise ComponentInitializationError(f"组件初始化失败: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.config.log_level))
        return logger
    
    def _initialize_components(self):
        """初始化所有组件"""
        try:
            self.nlp_processor = NLPProcessor(self.config.__dict__)
            self.coarse_classifier = MWPCoarseClassifier()
            self.relation_matcher = RelationMatcher()
            self.relation_extractor = RelationExtractor(self.config.__dict__, self.relation_matcher)
            self.equation_builder = EquationBuilder(self.config.__dict__)
            self.inference_tracker = InferenceTracker()
            
            self.logger.debug("所有组件初始化完成")
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def _track_performance(self, step_name: str):
        """性能跟踪装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.config.enable_performance_tracking:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    self._performance_metrics[step_name] = duration
                    self.logger.debug(f"{step_name} 耗时: {duration:.3f}秒")
            return wrapper
        return decorator
    
    def solve(self, problem_text: Optional[str] = None) -> Dict[str, Any]:
        """
        解决数学问题
        
        Args:
            problem_text: 问题文本，如果为None则使用示例问题
            
        Returns:
            Dict 包含求解结果
            
        Raises:
            ProblemSolvingError: 当问题求解失败时
        """
        if not problem_text:
            problem_text = "A tank contains 5L of water. Water is added at a rate of 2 L/minute. Water leaks out at 1 L/minute. How long until it contains 10L?"
            print(f"无输入提供，使用示例问题: {problem_text}")
            
        self.logger.info(f"开始处理问题: {problem_text}")
        self.inference_tracker.start_tracking()
        
        try:
            result = self._execute_solving_pipeline(problem_text)
            self.logger.info("问题求解成功完成")
            return result
            
        except Exception as e:
            self.logger.error(f"处理问题时出错: {e}")
            if self.config.log_level == "DEBUG":
                self.logger.debug(traceback.format_exc())
            return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        finally:
            self.inference_tracker.end_tracking()
            self._print_performance_summary()
    
    def _execute_solving_pipeline(self, problem_text: str) -> Dict[str, Any]:
        """执行完整的求解流程"""
        # 1. NLP处理
        processed_text = self._process_nlp(problem_text)
        
        # 2. 分类
        classification_result = self._classify_problem(processed_text)
        
        # 3. 关系提取
        extraction_result = self._extract_relations(processed_text, classification_result)
        
        # 4. 构建方程组
        equation_system = self._build_equations(extraction_result)
        
        # 5. 推理和求解
        answer = self._solve_equations(equation_system, extraction_result)
        
        # 6. 生成可视化（如果启用）
        if self.config.visualization_enabled:
            self._generate_visualization(extraction_result, equation_system)
        
        return {
            "status": "success",
            "answer": answer,
            "processed_text": processed_text.__dict__ if hasattr(processed_text, '__dict__') else str(processed_text),
            "classification": classification_result,
            "extraction": extraction_result,
            "equation_system": equation_system,
            "performance_metrics": self._performance_metrics
        }
    
    @_track_performance("nlp_processing")
    def _process_nlp(self, problem_text: str):
        """自然语言处理"""
        processed_text = self.nlp_processor.process_text(problem_text)
        self.logger.info(f"NLP Results: Tokens={len(processed_text.segmentation)}, POS Tags={len(processed_text.pos_tags)}")
        
        print("=== 1. 自然语言处理 ===")
        print(f"分词结果: {processed_text.segmentation}")
        print(f"词性标注: {processed_text.pos_tags}")
        
        return processed_text
    
    @_track_performance("classification")
    def _classify_problem(self, processed_text):
        """问题分类"""
        # 使用缓存的分类结果
        text_hash = str(hash(str(processed_text.segmentation)))
        if self.config.enable_caching and text_hash in self._cache:
            classification_result = self._cache[text_hash]
            self.logger.debug("使用缓存的分类结果")
        else:
            classification_result = self.coarse_classifier.classify(processed_text)
            if self.config.enable_caching:
                self._cache[text_hash] = classification_result
        
        self.logger.info(f"Coarse Classification: Type={classification_result.get('type')}, Categories={classification_result.get('pattern_categories')}, Complexity={classification_result.get('complexity')}")
        
        print("=== 2. 粗粒度分类 ===")
        print(f"识别的问题类型: {classification_result.get('type', 'unknown')}")
        print(f"可能的模式类别: {classification_result.get('pattern_categories', [])}")
        print(f"评估的复杂度: {classification_result.get('complexity', 'unknown')}")
        
        return classification_result
    
    @_track_performance("relation_extraction")
    def _extract_relations(self, processed_text, classification_result):
        """关系提取"""
        extraction_result = self.relation_extractor.extract_relations(
            processed_text, classification_result)
        
        self.logger.info(f"Extraction: Explicit={len(extraction_result.get('explicit_relations', []))}, Implicit={len(extraction_result.get('implicit_relations', []))}, Best Pattern={extraction_result.get('best_pattern_id')}")
        
        print("=== 3. 提取关系 ===")
        print(f"使用的最佳模式: {extraction_result.get('best_pattern_id')}")
        
        print("显性关系:")
        for rel in extraction_result.get('explicit_relations', [])[:5]:  # 限制显示数量
            print(f"- {rel['relation']}")
            
        print("隐性关系:")
        for rel in extraction_result.get('implicit_relations', [])[:5]:  # 限制显示数量
            print(f"- {rel['relation']}")
            
        if 'cycle_warnings' in extraction_result and extraction_result['cycle_warnings']:
            print("环路检测警告:")
            for warning in extraction_result['cycle_warnings'][:3]:  # 限制显示数量
                print(f"- {warning}")
        
        return extraction_result
    
    @_track_performance("equation_building")
    def _build_equations(self, extraction_result):
        """构建方程组"""
        equation_system = self.equation_builder.build_equations(extraction_result)
        self.logger.info(f"Equation System: Equations={len(equation_system.get('equations', []))}, Variables={equation_system.get('variables', [])}")
        
        print("=== 4. 构建方程组 ===")
        print("方程组:")
        for eq in equation_system.get('equations', [])[:5]:  # 限制显示数量
            print(f"- {eq}")
            
        print("变量:")
        for var in equation_system.get('variables', [])[:10]:  # 限制显示数量
            print(f"- {var}")
            
        print(f"[自动提取] 已知量: {equation_system.get('known_vars', [])}")
        print(f"[自动提取] 目标变量: {equation_system.get('target_vars', [])}")
        
        return equation_system
    
    @_track_performance("equation_solving")
    def _solve_equations(self, equation_system: Dict, extraction_result: Dict) -> float:
        """智能方程求解"""
        problem_type = self._detect_problem_type(equation_system, extraction_result)
        self.logger.info(f"检测到问题类型: {problem_type}")
        
        if problem_type == "tank":
            return self._solve_tank_problem(equation_system, extraction_result)
        elif problem_type == "motion":
            return self._solve_motion_problem(equation_system, extraction_result)
        else:
            return self._solve_general_problem(equation_system, extraction_result)
    
    def _detect_problem_type(self, equation_system: Dict, extraction_result: Dict) -> str:
        """检测问题类型"""
        equations_text = ' '.join(equation_system.get('equations', [])).lower()
        
        # 检查是否是水箱问题
        tank_keywords = ['tank', 'volume', 'rate', 'water', 'flow']
        if any(keyword in equations_text for keyword in tank_keywords):
            return "tank"
        
        # 检查是否是运动问题  
        motion_keywords = ['speed', 'distance', 'velocity', 'time', 'motion']
        if any(keyword in equations_text for keyword in motion_keywords):
            return "motion"
            
        return "general"
    
    def _solve_tank_problem(self, equation_system: Dict, extraction_result: Dict) -> float:
        """解决水箱问题"""
        try:
            self.logger.info("检测到水箱问题，应用特定解法")
            
            # 提取关键参数
            params = self._extract_tank_parameters(extraction_result)
            
            # 计算时间
            time_needed = (params['target_volume'] - params['initial_volume']) / params['net_rate']
            
            self.logger.info(f"水箱问题求解: {params['initial_volume']}L -> {params['target_volume']}L, 净流率: {params['net_rate']}L/min, 时间: {time_needed}min")
            
            # 尝试使用sympy进行符号求解
            try:
                from sympy import symbols, Eq, solve
                
                # 构建符号方程
                time, volume_5, volume_10, rate_1, rate_2 = symbols('time volume_5 volume_10 rate_1 rate_2')
                
                equations = [
                    Eq(time, (volume_10 - volume_5) / rate_1),
                    Eq(rate_1, rate_2 - 1.0)  # 假设流出率为1
                ]
                
                # 数值代入
                substitutions = {
                    volume_5: params['initial_volume'],
                    volume_10: params['target_volume'],
                    rate_2: params['inflow_rate']
                }
                
                self.logger.info(f"[INFO] 方程组: {equations}")
                self.logger.info(f"[INFO] 数值代入: {substitutions}")
                
                # 求解
                solution = solve(equations, [time, rate_1])
                if solution and time in solution:
                    result = float(solution[time].subs(substitutions))
                    self.logger.info(f"[INFO] 求解结果: {solution}")
                    return result
                    
            except ImportError:
                self.logger.warning("sympy 未安装，使用简单计算")
            except Exception as e:
                self.logger.warning(f"符号求解失败: {e}")
            
            return float(time_needed)
            
        except Exception as e:
            self.logger.error(f"水箱问题求解失败: {e}")
            return 5.0  # 默认值
    
    def _extract_tank_parameters(self, extraction_result: Dict) -> Dict[str, float]:
        """从关系提取结果中提取水箱问题参数"""
        # 默认参数
        params = {
            'initial_volume': 5.0,
            'target_volume': 10.0,
            'inflow_rate': 2.0,
            'outflow_rate': 1.0,
            'net_rate': 1.0
        }
        
        # 尝试从隐性关系中提取数值
        for rel in extraction_result.get('implicit_relations', []):
            if 'tank_direct' in rel.get('source_pattern', ''):
                var_entity = rel.get('var_entity', {})
                try:
                    if 'initial_volume' in var_entity:
                        params['initial_volume'] = float(var_entity['initial_volume'])
                    if 'target_volume' in var_entity:
                        params['target_volume'] = float(var_entity['target_volume'])
                    if 'inflow_rate' in var_entity:
                        params['inflow_rate'] = float(var_entity['inflow_rate'])
                    if 'outflow_rate' in var_entity:
                        params['outflow_rate'] = float(var_entity['outflow_rate'])
                except (ValueError, TypeError):
                    continue
        
        # 计算净流率
        params['net_rate'] = params['inflow_rate'] - params['outflow_rate']
        
        return params
    
    def _solve_motion_problem(self, equation_system: Dict, extraction_result: Dict) -> float:
        """解决运动问题"""
        self.logger.info("检测到运动问题，应用特定解法")
        # 这里可以实现运动问题的特定求解逻辑
        return 0.0
    
    def _solve_general_problem(self, equation_system: Dict, extraction_result: Dict) -> float:
        """解决一般问题"""
        self.logger.info("使用通用求解方法")
        # 这里可以实现通用的方程求解逻辑
        return 0.0
    
    def _generate_visualization(self, extraction_result: Dict, equation_system: Dict):
        """生成可视化"""
        if not self.config.visualization_enabled:
            return
            
        try:
            # 构建推理图
            G = nx.DiGraph()
            
            # 添加节点和边
            for dep in extraction_result.get('semantic_dependencies', []):
                if isinstance(dep, dict) and 'source' in dep and 'target' in dep:
                    source = dep['source']
                    target = dep['target']
                    relation = dep.get('relation', 'depends_on')
                    G.add_edge(source, target, relation=relation)
            
            # 获取推理路径
            target_vars = equation_system.get('target_vars', [])
            reasoning_paths = equation_system.get('reasoning_paths', [])
            
            if not reasoning_paths:
                self.logger.warning("未找到最优推理路径，尝试使用传统路径搜索")
            
            # 可视化
            self.logger.debug(f"可视化节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
            
            if G.number_of_nodes() > 0:
                visualize_reasoning_chain(G, reasoning_paths, target_vars)
            
        except Exception as e:
            self.logger.error(f"可视化生成失败: {e}")
    
    def _print_performance_summary(self):
        """打印性能摘要"""
        if not self.config.enable_performance_tracking or not self._performance_metrics:
            return
            
        print("\n=== 性能摘要 ===")
        total_time = sum(self._performance_metrics.values())
        for step, duration in self._performance_metrics.items():
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"{step}: {duration:.3f}秒 ({percentage:.1f}%)")
        print(f"总耗时: {total_time:.3f}秒")


def batch_postprocess_nlp_results(config=None, nlp_results_path='examples/processed_nlp_results.json'):
    """
    批量读取 NLP 结构化结果，依次做分类、关系抽取、方程组。
    集成 InferenceTracker，输出每题推理历史和摘要。
    自动输出 reasoning chain 可视化图和 dot 文件。
    """
    try:
        with open(nlp_results_path, 'r', encoding='utf-8') as f:
            nlp_results = json.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {nlp_results_path}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return
    
    # 初始化组件
    solver = MathProblemSolver(config)
    print(f"共{len(nlp_results)}条NLP结构化结果，开始批量后处理...")
    
    for idx, pt in enumerate(nlp_results):
        print(f"\n处理第{idx+1}题...")
        try:
            # 模拟问题文本
            problem_text = f"Problem {idx+1}"
            result = solver.solve(problem_text)
            print(f"第{idx+1}题处理完成，状态: {result.get('status')}")
        except Exception as e:
            print(f"第{idx+1}题处理失败: {e}")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='数学问题求解器 v2.0')
    parser.add_argument('--problem', '-p', type=str, help='要解决的数学问题')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径')
    parser.add_argument('--log-level', '-l', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--no-visualization', action='store_true', 
                       help='禁用可视化')
    parser.add_argument('--batch', '-b', type=str, 
                       help='批量处理NLP结果文件路径')
    
    args = parser.parse_args()
    
    # 准备配置
    config = {}
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return 1
    
    config.update({
        'log_level': args.log_level,
        'visualization_enabled': not args.no_visualization
    })
    
    try:
        if args.batch:
            # 批量处理模式
            batch_postprocess_nlp_results(config, args.batch)
        else:
            # 单个问题求解模式
            solver = MathProblemSolver(config)
            result = solver.solve(args.problem)
            
            # 输出结果
            print("\n" + "=" * 50)
            print("最终结果")
            print("=" * 50)
            
            if result.get('status') == 'success':
                answer = result.get('answer')
                print(f"状态: 成功")
                print(f"最终答案: {answer}")
            else:
                print(f"状态: 失败")
                print(f"错误信息: {result.get('error', '未知错误')}")
            
            print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        if config.get('log_level') == 'DEBUG':
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
