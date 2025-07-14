"""
推理策略管理器
负责推理策略的选择、调度和管理
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type

from ...config.config_manager import get_config
from ...core.exceptions import ConfigurationError, ReasoningError
from ...core.interfaces import ReasoningContext
from ...monitoring.performance_monitor import get_monitor, monitor_performance
from .strategy_base import (ReasoningStrategy, StrategyComplexity,
                            StrategyResult, StrategyType)


class StrategyManager:
    """推理策略管理器"""
    
    def __init__(self):
        """初始化策略管理器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 策略注册表
        self.strategies: Dict[str, ReasoningStrategy] = {}
        self.strategy_types: Dict[StrategyType, List[str]] = defaultdict(list)
        
        # 策略选择配置
        try:
            self.config = get_config()
        except Exception:
            self.config = None
            self.logger.warning("配置管理器不可用，使用默认配置")
        
        # 默认策略选择规则
        self.selection_rules = {
            "complexity_threshold": 0.7,  # 复杂度阈值
            "confidence_threshold": 0.8,  # 置信度阈值
            "fallback_strategy": "cot",   # 默认回退策略
            "enable_multi_strategy": True,  # 是否启用多策略
            "strategy_timeout": 30.0,     # 单策略超时时间
        }
        
        # 策略性能统计
        self.performance_stats = defaultdict(lambda: {
            "total_runs": 0,
            "successful_runs": 0,
            "average_confidence": 0.0,
            "average_execution_time": 0.0,
            "success_rate": 0.0
        })
        
        self.monitor = get_monitor()
        self.logger.info("策略管理器初始化完成")
    
    def register_strategy(self, strategy: ReasoningStrategy) -> bool:
        """
        注册推理策略
        
        Args:
            strategy: 推理策略实例
            
        Returns:
            bool: 注册是否成功
        """
        try:
            strategy_name = strategy.name
            
            if strategy_name in self.strategies:
                self.logger.warning(f"策略 {strategy_name} 已存在，将被覆盖")
            
            self.strategies[strategy_name] = strategy
            self.strategy_types[strategy.strategy_type].append(strategy_name)
            
            self.logger.info(f"策略 {strategy_name} 注册成功")
            return True
            
        except Exception as e:
            self.logger.error(f"策略注册失败: {str(e)}")
            return False
    
    def unregister_strategy(self, strategy_name: str) -> bool:
        """
        注销推理策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 注销是否成功
        """
        try:
            if strategy_name not in self.strategies:
                self.logger.warning(f"策略 {strategy_name} 不存在")
                return False
            
            strategy = self.strategies[strategy_name]
            
            # 从类型映射中移除
            if strategy_name in self.strategy_types[strategy.strategy_type]:
                self.strategy_types[strategy.strategy_type].remove(strategy_name)
            
            # 从策略字典中移除
            del self.strategies[strategy_name]
            
            self.logger.info(f"策略 {strategy_name} 已注销")
            return True
            
        except Exception as e:
            self.logger.error(f"策略注销失败: {str(e)}")
            return False
    
    def get_available_strategies(self) -> List[str]:
        """获取可用策略列表"""
        return list(self.strategies.keys())
    
    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[str]:
        """根据类型获取策略列表"""
        return self.strategy_types.get(strategy_type, [])
    
    @monitor_performance("strategy_selection")
    def select_strategy(self, problem_text: str, context: Optional[ReasoningContext] = None, 
                       preferred_strategy: Optional[str] = None) -> Optional[str]:
        """
        智能选择推理策略
        
        Args:
            problem_text: 问题文本
            context: 推理上下文
            preferred_strategy: 首选策略名称
            
        Returns:
            Optional[str]: 选中的策略名称
        """
        try:
            # 如果指定了首选策略且可用，优先使用
            if preferred_strategy and preferred_strategy in self.strategies:
                strategy = self.strategies[preferred_strategy]
                if strategy.can_handle(problem_text, context):
                    self.logger.info(f"使用首选策略: {preferred_strategy}")
                    return preferred_strategy
                else:
                    self.logger.warning(f"首选策略 {preferred_strategy} 无法处理此问题，将自动选择")
            
            # 智能策略选择
            suitable_strategies = []
            
            for strategy_name, strategy in self.strategies.items():
                if strategy.can_handle(problem_text, context):
                    complexity = strategy.estimate_complexity(problem_text, context)
                    stats = self.performance_stats[strategy_name]
                    
                    # 计算策略评分
                    score = self._calculate_strategy_score(strategy, complexity, stats)
                    
                    suitable_strategies.append({
                        "name": strategy_name,
                        "strategy": strategy,
                        "complexity": complexity,
                        "score": score
                    })
            
            if not suitable_strategies:
                # 没有合适的策略，使用回退策略
                fallback = self.selection_rules["fallback_strategy"]
                if fallback in self.strategies:
                    self.logger.warning(f"没有找到合适策略，使用回退策略: {fallback}")
                    return fallback
                else:
                    self.logger.error("没有可用的推理策略")
                    return None
            
            # 按评分排序，选择最佳策略
            suitable_strategies.sort(key=lambda x: x["score"], reverse=True)
            selected_strategy = suitable_strategies[0]["name"]
            
            self.logger.info(f"智能选择策略: {selected_strategy}, 评分: {suitable_strategies[0]['score']:.2f}")
            return selected_strategy
            
        except Exception as e:
            self.logger.error(f"策略选择失败: {str(e)}")
            return self.selection_rules.get("fallback_strategy")
    
    def _calculate_strategy_score(self, strategy: ReasoningStrategy, complexity: float, 
                                 stats: Dict[str, Any]) -> float:
        """
        计算策略评分
        
        Args:
            strategy: 推理策略
            complexity: 问题复杂度
            stats: 策略统计信息
            
        Returns:
            float: 策略评分
        """
        score = 0.0
        
        # 基础评分 - 策略类型适配度
        type_score = {
            StrategyComplexity.SIMPLE: 0.8 if complexity < 0.3 else 0.4,
            StrategyComplexity.MODERATE: 0.9 if 0.3 <= complexity < 0.7 else 0.6,
            StrategyComplexity.COMPLEX: 0.9 if 0.7 <= complexity < 0.9 else 0.5,
            StrategyComplexity.ADVANCED: 1.0 if complexity >= 0.9 else 0.3
        }
        score += type_score.get(strategy.complexity, 0.5) * 0.4
        
        # 历史成功率
        if stats["total_runs"] > 0:
            success_rate = stats["successful_runs"] / stats["total_runs"]
            score += success_rate * 0.3
        else:
            score += 0.5 * 0.3  # 新策略给中等分数
        
        # 平均置信度
        avg_confidence = stats.get("average_confidence", 0.5)
        score += avg_confidence * 0.2
        
        # 执行效率 (时间越短越好)
        avg_time = stats.get("average_execution_time", 10.0)
        time_score = max(0, 1 - (avg_time / 30.0))  # 30秒为基准
        score += time_score * 0.1
        
        return min(1.0, max(0.0, score))
    
    @monitor_performance("reasoning_execution")
    def execute_reasoning(self, problem_text: str, context: Optional[ReasoningContext] = None,
                         strategy_name: Optional[str] = None, 
                         enable_fallback: bool = True) -> StrategyResult:
        """
        执行推理
        
        Args:
            problem_text: 问题文本
            context: 推理上下文
            strategy_name: 指定策略名称
            enable_fallback: 是否启用回退机制
            
        Returns:
            StrategyResult: 推理结果
        """
        timer_id = self.monitor.start_timer("reasoning_execution")
        
        try:
            # 选择策略
            if not strategy_name:
                strategy_name = self.select_strategy(problem_text, context)
            
            if not strategy_name or strategy_name not in self.strategies:
                raise ReasoningError(
                    f"无法找到可用的推理策略",
                    strategy_name=strategy_name or "unknown",
                    context={"available_strategies": list(self.strategies.keys())}
                )
            
            strategy = self.strategies[strategy_name]
            
            # 执行推理
            self.logger.info(f"开始执行推理，策略: {strategy_name}")
            result = strategy.execute(problem_text, context)
            
            # 更新性能统计
            self._update_performance_stats(strategy_name, result)
            
            # 如果策略失败且启用回退，尝试其他策略
            if not result.success and enable_fallback:
                self.logger.warning(f"策略 {strategy_name} 执行失败，尝试回退策略")
                result = self._try_fallback_strategies(problem_text, context, strategy_name)
            
            self.monitor.increment_counter("reasoning_executions_total")
            if result.success:
                self.monitor.increment_counter("reasoning_executions_success")
            
            return result
            
        except Exception as e:
            self.logger.error(f"推理执行失败: {str(e)}")
            
            # 创建错误结果
            return StrategyResult(
                success=False,
                answer="执行错误",
                confidence=0.0,
                reasoning_steps=[{
                    "step": 1,
                    "action": "execution_error",
                    "description": f"推理执行异常: {str(e)}",
                    "confidence": 0.0
                }],
                strategy_used=strategy_name or "unknown",
                execution_time=time.time() - timer_id if timer_id else 0.0,
                metadata={"error": str(e)}
            )
        
        finally:
            if timer_id:
                self.monitor.stop_timer(timer_id)
    
    def _try_fallback_strategies(self, problem_text: str, context: Optional[ReasoningContext],
                                failed_strategy: str) -> StrategyResult:
        """尝试回退策略"""
        fallback_order = [
            self.selection_rules["fallback_strategy"],
            "cot",  # Chain of Thought 作为最基础的回退
            "dir",  # Direct Implicit Reasoning
        ]
        
        for fallback_strategy in fallback_order:
            if (fallback_strategy != failed_strategy and 
                fallback_strategy in self.strategies):
                
                try:
                    self.logger.info(f"尝试回退策略: {fallback_strategy}")
                    strategy = self.strategies[fallback_strategy]
                    result = strategy.execute(problem_text, context)
                    
                    if result.success:
                        self.logger.info(f"回退策略 {fallback_strategy} 执行成功")
                        result.metadata["fallback_used"] = True
                        result.metadata["original_strategy"] = failed_strategy
                        return result
                        
                except Exception as e:
                    self.logger.warning(f"回退策略 {fallback_strategy} 也失败了: {str(e)}")
                    continue
        
        # 所有策略都失败了
        return StrategyResult(
            success=False,
            answer="所有策略都失败",
            confidence=0.0,
            reasoning_steps=[{
                "step": 1,
                "action": "all_strategies_failed",
                "description": "所有可用的推理策略都无法处理此问题",
                "confidence": 0.0
            }],
            strategy_used="fallback_failed",
            execution_time=0.0,
            metadata={"failed_strategy": failed_strategy}
        )
    
    def _update_performance_stats(self, strategy_name: str, result: StrategyResult):
        """更新策略性能统计"""
        stats = self.performance_stats[strategy_name]
        
        stats["total_runs"] += 1
        if result.success:
            stats["successful_runs"] += 1
        
        # 更新平均置信度
        total_confidence = (stats["average_confidence"] * (stats["total_runs"] - 1) + 
                           result.confidence)
        stats["average_confidence"] = total_confidence / stats["total_runs"]
        
        # 更新平均执行时间
        total_time = (stats["average_execution_time"] * (stats["total_runs"] - 1) + 
                     result.execution_time)
        stats["average_execution_time"] = total_time / stats["total_runs"]
        
        # 更新成功率
        stats["success_rate"] = stats["successful_runs"] / stats["total_runs"]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            "total_strategies": len(self.strategies),
            "strategy_performance": dict(self.performance_stats),
            "selection_rules": self.selection_rules,
            "strategy_info": {
                name: strategy.get_strategy_info() 
                for name, strategy in self.strategies.items()
            }
        }
    
    def update_selection_rules(self, new_rules: Dict[str, Any]):
        """更新策略选择规则"""
        self.selection_rules.update(new_rules)
        self.logger.info("策略选择规则已更新")
    
    def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats.clear()
        for strategy in self.strategies.values():
            strategy.reset_stats()
        self.logger.info("性能统计已重置") 