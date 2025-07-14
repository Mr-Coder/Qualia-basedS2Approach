"""
推理引擎重构测试
验证策略模式、多步推理和置信度计算的功能
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.core.exceptions import ReasoningError
from src.core.interfaces import (ProcessingResult, ProcessingStatus,
                                 ReasoningContext)
from src.reasoning.confidence_calculator import (BasicConfidenceCalculator,
                                                 ConfidenceResult)
from src.reasoning.multi_step_reasoner import (ExecutionResult, StepExecutor,
                                               StepType)
from src.reasoning.new_reasoning_engine import ModernReasoningEngine
from src.reasoning.strategy_manager import (ChainOfThoughtStrategy,
                                            StrategyComplexity,
                                            StrategyManager, StrategyResult,
                                            StrategyType,
                                            TreeOfThoughtsStrategy)


class TestStrategyManager:
    """测试策略管理器"""
    
    def test_strategy_manager_initialization(self):
        """测试策略管理器初始化"""
        manager = StrategyManager()
        
        assert manager is not None
        assert len(manager.get_available_strategies()) == 0
        assert manager.selection_rules["fallback_strategy"] == "cot"
    
    def test_strategy_registration(self):
        """测试策略注册"""
        manager = StrategyManager()
        strategy = ChainOfThoughtStrategy()
        
        # 注册策略
        success = manager.register_strategy(strategy)
        assert success is True
        
        # 检查策略是否可用
        available = manager.get_available_strategies()
        assert strategy.name in available
        
        # 获取指定类型的策略
        cot_strategies = manager.get_strategies_by_type(StrategyType.CHAIN_OF_THOUGHT)
        assert strategy.name in cot_strategies
    
    def test_strategy_selection(self):
        """测试策略选择"""
        manager = StrategyManager()
        
        # 注册多个策略
        strategies = [
            ChainOfThoughtStrategy(),
            TreeOfThoughtsStrategy()
        ]
        
        for strategy in strategies:
            manager.register_strategy(strategy)
        
        # 测试策略选择
        simple_problem = "计算 2 + 3"
        selected = manager.select_strategy(simple_problem)
        assert selected is not None
        assert selected in manager.get_available_strategies()
        
        # 测试复杂问题的策略选择
        complex_problem = "如果小明有10个苹果，小红有5个苹果，当他们把苹果分给3个朋友时，每个朋友分别得到多少个苹果？假设分配是平均的，并且考虑到剩余的苹果数量。"
        selected_complex = manager.select_strategy(complex_problem)
        assert selected_complex is not None
    
    def test_strategy_execution(self):
        """测试策略执行"""
        manager = StrategyManager()
        strategy = ChainOfThoughtStrategy()
        manager.register_strategy(strategy)
        
        # 执行推理
        problem = "小明有3个苹果，小红给了他2个苹果，小明现在有多少个苹果？"
        result = manager.execute_reasoning(problem)
        
        assert isinstance(result, StrategyResult)
        assert result.strategy_used == strategy.name
        assert len(result.reasoning_steps) > 0

class TestStepExecutor:
    """测试步骤执行器"""
    
    def test_step_executor_initialization(self):
        """测试步骤执行器初始化"""
        executor = StepExecutor()
        
        assert executor is not None
        assert len(executor.executors) > 0
        assert StepType.CALCULATE in executor.executors
    
    def test_parse_step_execution(self):
        """测试解析步骤执行"""
        executor = StepExecutor()
        
        step_data = {
            "action": "problem_analysis",
            "text": "计算 5 + 3 的值",
            "description": "分析问题"
        }
        
        result = executor.execute_step(step_data)
        
        assert isinstance(result, ExecutionResult)
        assert result.step_type == StepType.PARSE
        assert "numbers" in result.result
        assert len(result.result["numbers"]) >= 2
    
    def test_extract_step_execution(self):
        """测试提取步骤执行"""
        executor = StepExecutor()
        
        step_data = {
            "action": "number_extraction",
            "text": "小明有10个苹果，买了5个橘子",
            "extract_type": "numbers"
        }
        
        result = executor.execute_step(step_data)
        
        assert result.success is True
        assert result.step_type == StepType.EXTRACT
        assert len(result.result) == 2  # 应该提取到 10 和 5
        assert 10.0 in result.result
        assert 5.0 in result.result
    
    def test_calculate_step_execution(self):
        """测试计算步骤执行"""
        executor = StepExecutor()
        
        # 加法计算
        step_data = {
            "action": "addition",
            "numbers": [5, 3],
            "operation": "add"
        }
        
        result = executor.execute_step(step_data)
        
        assert result.success is True
        assert result.step_type == StepType.CALCULATE
        assert result.result == 8.0
        assert result.confidence > 0.8
        
        # 除法计算
        step_data = {
            "action": "division",
            "numbers": [10, 2],
            "operation": "divide"
        }
        
        result = executor.execute_step(step_data)
        
        assert result.success is True
        assert result.result == 5.0
    
    def test_validate_step_execution(self):
        """测试验证步骤执行"""
        executor = StepExecutor()
        
        step_data = {
            "action": "answer_validation",
            "value": 8,
            "validation_type": "range",
            "min_value": 0,
            "max_value": 100
        }
        
        result = executor.execute_step(step_data)
        
        assert result.success is True
        assert result.step_type == StepType.VALIDATE
        assert result.result["valid"] is True

class TestConfidenceCalculator:
    """测试置信度计算器"""
    
    def test_confidence_calculator_initialization(self):
        """测试置信度计算器初始化"""
        calculator = BasicConfidenceCalculator()
        
        assert calculator is not None
        assert calculator.name == "basic_confidence_calculator"
        assert "step_confidence" in calculator.confidence_weights
    
    def test_step_confidence_calculation(self):
        """测试单步置信度计算"""
        calculator = BasicConfidenceCalculator()
        
        # 高置信度步骤
        high_conf_step = {
            "action": "number_extraction",
            "confidence": 0.9,
            "numbers": [5, 3],
            "result": 8,
            "description": "从问题中提取数字5和3"
        }
        
        confidence = calculator.calculate_step_confidence(high_conf_step)
        assert confidence > 0.9
        
        # 低置信度步骤
        low_conf_step = {
            "action": "reasoning",
            "confidence": 0.5,
            "description": "推理"
        }
        
        confidence = calculator.calculate_step_confidence(low_conf_step)
        assert confidence < 0.6
    
    def test_overall_confidence_calculation(self):
        """测试整体置信度计算"""
        calculator = BasicConfidenceCalculator()
        
        reasoning_steps = [
            {
                "step": 1,
                "action": "number_extraction",
                "confidence": 0.95,
                "numbers": [5, 3],
                "description": "提取数字"
            },
            {
                "step": 2,
                "action": "addition",
                "confidence": 0.95,
                "numbers": [5, 3],
                "result": 8,
                "description": "计算5+3=8"
            },
            {
                "step": 3,
                "action": "answer_validation",
                "confidence": 0.9,
                "validation": {"valid": True, "confidence": 0.9},
                "description": "验证答案"
            }
        ]
        
        result = calculator.calculate_confidence(reasoning_steps, 8)
        
        assert isinstance(result, ConfidenceResult)
        assert result.overall_confidence > 0.8
        assert "step_confidence" in result.component_confidences
        assert len(result.confidence_factors) > 0
    
    def test_logical_consistency_calculation(self):
        """测试逻辑一致性计算"""
        calculator = BasicConfidenceCalculator()
        
        # 逻辑一致的步骤
        consistent_steps = [
            {"action": "addition", "numbers": [5, 3], "result": 8},
            {"action": "validation", "inputs": [8], "result": {"valid": True}}
        ]
        
        consistency = calculator.calculate_logical_consistency(consistent_steps)
        assert consistency > 0.8
        
        # 逻辑不一致的步骤
        inconsistent_steps = [
            {"action": "addition", "numbers": [5, 3], "result": 10},  # 错误结果
        ]
        
        consistency = calculator.calculate_logical_consistency(inconsistent_steps)
        assert consistency < 1.0

class TestModernReasoningEngine:
    """测试现代化推理引擎"""
    
    def test_reasoning_engine_initialization(self):
        """测试推理引擎初始化"""
        engine = ModernReasoningEngine()
        
        assert engine is not None
        assert engine.strategy_manager is not None
        assert engine.step_executor is not None
        assert engine.confidence_calculator is not None
        assert len(engine.get_available_strategies()) > 0
    
    def test_simple_arithmetic_reasoning(self):
        """测试简单算术推理"""
        engine = ModernReasoningEngine()
        
        problem = "计算 7 + 5 的值"
        result = engine.reason(problem)
        
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert "12" in str(result.result) or result.result == 12
        assert result.confidence > 0.6
        assert result.status == ProcessingStatus.COMPLETED
        assert len(result.metadata["reasoning_steps"]) > 0
    
    def test_word_problem_reasoning(self):
        """测试文字应用题推理"""
        engine = ModernReasoningEngine()
        
        problem = "小明有8个苹果，吃了3个，还剩多少个苹果？"
        result = engine.reason(problem)
        
        assert result.success is True
        assert "5" in str(result.result) or result.result == 5
        assert result.confidence > 0.5
        assert "strategy_used" in result.metadata
    
    def test_complex_problem_reasoning(self):
        """测试复杂问题推理"""
        engine = ModernReasoningEngine()
        
        problem = "一个班级有30个学生，其中男生比女生多4人。如果男生人数是女生人数的1.3倍，那么班级里有多少个女生？"
        result = engine.reason(problem)
        
        assert isinstance(result, ProcessingResult)
        # 复杂问题可能失败，但应该有推理步骤
        assert len(result.metadata.get("reasoning_steps", [])) > 0
    
    def test_reasoning_with_context(self):
        """测试带上下文的推理"""
        engine = ModernReasoningEngine()
        
        context = ReasoningContext(
            problem_text="计算面积",
            problem_type="geometry",
            parameters={"precision": 2},
            history=[],
            constraints={"max_steps": 10}
        )
        
        problem = "一个长方形的长是6米，宽是4米，求面积"
        result = engine.reason(problem, context)
        
        assert result.success is True
        assert "24" in str(result.result) or result.result == 24
    
    def test_strategy_selection_and_usage(self):
        """测试策略选择和使用"""
        engine = ModernReasoningEngine()
        
        # 简单问题应该使用Chain of Thought
        simple_problem = "3 + 4 = ?"
        result = engine.reason(simple_problem)
        
        strategy_used = result.metadata.get("strategy_used", "")
        assert "chain_of_thought" in strategy_used.lower() or "cot" in strategy_used.lower()
        
        # 获取推理步骤
        steps = engine.get_reasoning_steps()
        assert len(steps) > 0
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        engine = ModernReasoningEngine()
        
        # 执行几个推理任务
        problems = [
            "2 + 3",
            "10 - 4", 
            "5 × 6",
            "20 ÷ 4"
        ]
        
        for problem in problems:
            engine.reason(problem)
        
        # 检查性能报告
        report = engine.get_performance_report()
        
        assert "reasoning_stats" in report
        assert report["reasoning_stats"]["total_problems"] == len(problems)
        assert report["reasoning_stats"]["successful_problems"] > 0
        assert "success_rate" in report
    
    def test_error_handling(self):
        """测试错误处理"""
        engine = ModernReasoningEngine()
        
        # 测试空问题
        result = engine.reason("")
        assert result.success is False
        
        # 测试无意义问题
        result = engine.reason("这不是一个数学问题")
        # 应该有处理结果，即使失败
        assert isinstance(result, ProcessingResult)
    
    def test_strategy_management(self):
        """测试策略管理"""
        engine = ModernReasoningEngine()
        
        # 获取可用策略
        strategies = engine.get_available_strategies()
        initial_count = len(strategies)
        assert initial_count > 0
        
        # 创建自定义策略
        class TestStrategy(ChainOfThoughtStrategy):
            def __init__(self):
                super().__init__()
                self.name = "test_strategy"
        
        test_strategy = TestStrategy()
        
        # 添加策略
        success = engine.add_strategy(test_strategy)
        assert success is True
        
        new_strategies = engine.get_available_strategies()
        assert len(new_strategies) == initial_count + 1
        assert "test_strategy" in new_strategies
        
        # 移除策略
        success = engine.remove_strategy("test_strategy")
        assert success is True
        
        final_strategies = engine.get_available_strategies()
        assert len(final_strategies) == initial_count

class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_reasoning_flow(self):
        """测试端到端推理流程"""
        engine = ModernReasoningEngine()
        
        # 完整的推理流程测试
        problem = "小红买了3支铅笔，每支2元，又买了2个橡皮，每个1.5元。她一共花了多少钱？"
        
        result = engine.reason(problem)
        
        # 验证结果
        assert result.success is True
        expected_answer = 3 * 2 + 2 * 1.5  # 6 + 3 = 9
        assert abs(float(result.result) - expected_answer) < 0.1
        
        # 验证推理步骤
        steps = result.metadata["reasoning_steps"]
        assert len(steps) >= 3  # 至少应该有数字提取、计算、验证步骤
        
        # 验证置信度
        assert result.confidence > 0.7
        
        # 验证元数据
        assert "strategy_used" in result.metadata
        assert "confidence_details" in result.metadata
        assert "step_count" in result.metadata
    
    def test_multiple_problem_types(self):
        """测试多种问题类型"""
        engine = ModernReasoningEngine()
        
        test_cases = [
            {
                "problem": "5 + 7 = ?",
                "expected": 12,
                "type": "arithmetic"
            },
            {
                "problem": "长方形长8米，宽3米，面积是多少？",
                "expected": 24,
                "type": "geometry"
            },
            {
                "problem": "100元的商品打8折，现价多少？",
                "expected": 80,
                "type": "percentage"
            }
        ]
        
        for case in test_cases:
            result = engine.reason(case["problem"])
            
            # 验证基本成功
            assert result.success is True, f"问题失败: {case['problem']}"
            
            # 验证结果接近预期（允许一定误差）
            try:
                actual = float(result.result)
                expected = case["expected"]
                assert abs(actual - expected) < 0.1, f"结果不匹配: 期望{expected}, 实际{actual}"
            except ValueError:
                # 如果结果不是数字，检查是否包含预期值
                assert str(case["expected"]) in str(result.result)
    
    def test_performance_under_load(self):
        """测试负载下的性能"""
        engine = ModernReasoningEngine()
        
        # 生成测试问题
        problems = [f"计算 {i} + {i+1}" for i in range(1, 21)]
        
        start_time = time.time()
        results = []
        
        for problem in problems:
            result = engine.reason(problem)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # 性能验证
        assert total_time < 30.0, f"处理时间过长: {total_time}秒"
        
        # 成功率验证
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results)
        assert success_rate > 0.8, f"成功率过低: {success_rate}"
        
        # 平均处理时间验证
        avg_time = total_time / len(problems)
        assert avg_time < 2.0, f"平均处理时间过长: {avg_time}秒"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 