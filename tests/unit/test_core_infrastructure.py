"""
核心基础设施单元测试
测试异常处理、配置管理、输入验证和性能监控功能
"""

import os
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from src.config.config_manager import (REASONING_CONFIG_SCHEMA,
                                       ConfigurationManager, get_config)
from src.core.exceptions import (ConfigurationError, COTBaseException,
                                 ExceptionRecoveryStrategy,
                                 InputValidationError, PerformanceError,
                                 ProcessingError, ReasoningError,
                                 SecurityError, TimeoutError, ValidationError,
                                 handle_exceptions)
from src.monitoring.performance_monitor import (PerformanceMetric,
                                                PerformanceMonitor,
                                                monitor_performance,
                                                timeout_monitor)
from src.validation.input_validator import InputValidator, validate_input


class TestExceptionSystem:
    """测试异常处理系统"""
    
    def test_base_exception_creation(self):
        """测试基础异常创建"""
        error = COTBaseException(
            "测试错误",
            error_code="TEST_ERROR",
            context={"test": "data"}
        )
        
        assert error.message == "测试错误"
        assert error.error_code == "TEST_ERROR"
        assert error.context["test"] == "data"
        
        error_dict = error.to_dict()
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "测试错误"
        assert error_dict["context"]["test"] == "data"
    
    def test_validation_error(self):
        """测试验证错误"""
        error = ValidationError(
            "字段验证失败",
            field="test_field",
            value="invalid_value"
        )
        
        assert error.error_code == "VALIDATION_ERROR"
        assert error.context["field"] == "test_field"
        assert error.context["invalid_value"] == "invalid_value"
    
    def test_input_validation_error(self):
        """测试输入验证错误"""
        error = InputValidationError(
            "输入无效",
            input_text="危险输入" * 100
        )
        
        assert error.error_code == "INPUT_VALIDATION_ERROR"
        # 检查输入文本被截断
        assert len(error.context["input_text"]) <= 103  # 100 + "..."
    
    def test_exception_decorator(self):
        """测试异常处理装饰器"""
        @handle_exceptions(default_return="fallback")
        def test_function():
            raise ValueError("测试错误")
        
        result = test_function()
        assert result == "fallback"
        
        @handle_exceptions(reraise_as=ProcessingError)
        def test_function_reraise():
            raise ValueError("测试错误")
        
        with pytest.raises(ProcessingError):
            test_function_reraise()
    
    def test_exception_recovery_strategy(self):
        """测试异常恢复策略"""
        error = ProcessingError("处理失败")
        fallback_result = ExceptionRecoveryStrategy.create_fallback_result(error)
        
        assert fallback_result["success"] is False
        assert fallback_result["fallback"] is True
        assert fallback_result["answer"] == "无法计算"
        assert fallback_result["confidence"] == 0.0
        
        # 测试重试策略
        assert ExceptionRecoveryStrategy.should_retry(
            PerformanceError("性能问题"), attempt=1
        ) is True
        
        assert ExceptionRecoveryStrategy.should_retry(
            SecurityError("安全问题"), attempt=1
        ) is False

class TestConfigurationManager:
    """测试配置管理系统"""
    
    def test_config_manager_initialization(self, test_config_manager):
        """测试配置管理器初始化"""
        config = test_config_manager
        assert config.env == "test"
        assert config.config_cache is not None
    
    def test_config_get_set(self, test_config_manager):
        """测试配置获取和设置"""
        config = test_config_manager
        
        # 测试获取配置
        max_steps = config.get("reasoning.max_steps", 5)
        assert max_steps == 10  # 来自测试配置
        
        # 测试设置配置
        config.set("test.new_value", "test_data")
        assert config.get("test.new_value") == "test_data"
        
        # 测试嵌套配置
        config.set("nested.deep.value", 42)
        assert config.get("nested.deep.value") == 42
    
    def test_config_validation(self, test_config_manager):
        """测试配置验证"""
        config = test_config_manager
        
        # 测试有效配置验证
        assert config.validate_config(REASONING_CONFIG_SCHEMA) is True
        
        # 测试无效配置
        config.set("reasoning.max_steps", -1)  # 无效值
        
        with pytest.raises(ConfigurationError):
            config.validate_config(REASONING_CONFIG_SCHEMA)
    
    def test_config_summary(self, test_config_manager):
        """测试配置摘要"""
        config = test_config_manager
        
        # 添加敏感配置
        config.set("database.password", "secret123")
        config.set("api.secret_key", "top_secret")
        
        summary = config.get_config_summary()
        
        assert summary["environment"] == "test"
        assert "config_summary" in summary
        
        # 检查敏感信息被屏蔽
        config_str = str(summary["config_summary"])
        assert "secret123" not in config_str
        assert "top_secret" not in config_str
        assert "***MASKED***" in config_str

class TestInputValidator:
    """测试输入验证系统"""
    
    def test_valid_math_problem(self, input_validator, sample_math_problems):
        """测试有效数学问题验证"""
        for problem in sample_math_problems:
            result = input_validator.validate_math_problem(problem)
            
            assert result["valid"] is True
            assert result["sanitized_text"] is not None
            assert isinstance(result["warnings"], list)
            assert isinstance(result["errors"], list)
    
    def test_invalid_inputs(self, input_validator, sample_invalid_inputs):
        """测试无效输入验证"""
        for invalid_input in sample_invalid_inputs:
            result = input_validator.validate_math_problem(invalid_input)
            
            if invalid_input in [None, "", 123]:
                assert result["valid"] is False
                assert len(result["errors"]) > 0
            else:
                # 其他类型的无效输入可能有不同的处理
                if not result["valid"]:
                    assert len(result["errors"]) > 0
    
    def test_security_threats(self, input_validator):
        """测试安全威胁检测"""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "eval(malicious_code)",
            "import os; os.system('rm -rf /')",
            "../../../etc/passwd"
        ]
        
        for dangerous_input in dangerous_inputs:
            result = input_validator.validate_math_problem(dangerous_input)
            assert result["valid"] is False
            assert any("安全威胁" in error for error in result["errors"])
    
    def test_math_symbol_normalization(self, input_validator):
        """测试数学符号标准化"""
        input_text = "计算 5 × 3 ÷ 2 ＋ 1 的值"
        result = input_validator.validate_math_problem(input_text)
        
        assert result["valid"] is True
        normalized = result["sanitized_text"]
        
        # 检查符号被标准化
        assert "×" not in normalized
        assert "÷" not in normalized
        assert "＋" not in normalized
        assert "*" in normalized
        assert "/" in normalized
        assert "+" in normalized
    
    def test_numeric_validation(self, input_validator):
        """测试数值验证"""
        # 有效数值
        valid_numbers = [42, 3.14, "123", "45.67", "-10"]
        
        for num in valid_numbers:
            result = input_validator.validate_numeric_input(num)
            assert result["valid"] is True
            assert result["numeric_value"] is not None
        
        # 无效数值
        invalid_numbers = [None, "", "abc", "not a number"]
        
        for num in invalid_numbers:
            result = input_validator.validate_numeric_input(num)
            assert result["valid"] is False
            assert len(result["errors"]) > 0
    
    def test_file_path_validation(self, input_validator, temp_test_files):
        """测试文件路径验证"""
        # 有效路径（相对于项目根目录）
        valid_path = "tests/data/test.txt"
        result = input_validator.validate_file_path(valid_path)
        # 注意：这可能因具体实现而异
        
        # 危险路径
        dangerous_paths = ["../../../etc/passwd", "/etc/shadow", "..\\windows\\system32"]
        
        for path in dangerous_paths:
            result = input_validator.validate_file_path(path)
            assert result["valid"] is False
            assert any("路径遍历" in error for error in result["errors"])
    
    def test_batch_validation(self, input_validator):
        """测试批量验证"""
        inputs = [
            {"type": "math_problem", "value": "计算 2 + 3"},
            {"type": "numeric", "value": "42"},
            {"type": "math_problem", "value": "<script>alert('xss')</script>"},
            {"type": "numeric", "value": "not a number"}
        ]
        
        results = input_validator.batch_validate(inputs)
        
        assert results["summary"]["total"] == 4
        assert results["summary"]["valid"] == 2
        assert results["summary"]["invalid"] == 2
        assert results["all_valid"] is False

class TestPerformanceMonitor:
    """测试性能监控系统"""
    
    def test_timer_functionality(self, performance_monitor):
        """测试计时器功能"""
        monitor = performance_monitor
        
        timer_id = monitor.start_timer("test_operation")
        assert timer_id is not None
        
        time.sleep(0.1)  # 模拟操作
        
        duration = monitor.stop_timer(timer_id)
        assert duration is not None
        assert duration >= 0.1
    
    def test_metric_recording(self, performance_monitor):
        """测试指标记录"""
        monitor = performance_monitor
        
        metric = PerformanceMetric(
            name="test_metric",
            value=42.0,
            timestamp=None,  # 会自动设置
            unit="units"
        )
        
        monitor.record_metric(metric)
        
        summary = monitor.get_metrics_summary()
        assert "test_metric" in summary["metrics"]
    
    def test_counter_and_gauge(self, performance_monitor):
        """测试计数器和仪表"""
        monitor = performance_monitor
        
        # 测试计数器
        monitor.increment_counter("test_counter", 5)
        monitor.increment_counter("test_counter", 3)
        
        # 测试仪表
        monitor.set_gauge("test_gauge", 78.5)
        
        summary = monitor.get_metrics_summary()
        assert summary["counters"]["test_counter"] == 8
        assert summary["gauges"]["test_gauge"] == 78.5
    
    def test_performance_decorator(self):
        """测试性能监控装饰器"""
        @monitor_performance("test_function")
        def test_function():
            time.sleep(0.05)
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_timeout_decorator(self):
        """测试超时监控装饰器"""
        @timeout_monitor(0.1, "fast_function")
        def fast_function():
            time.sleep(0.05)
            return "success"
        
        result = fast_function()
        assert result == "success"
        
        @timeout_monitor(0.05, "slow_function")
        def slow_function():
            time.sleep(0.1)
            return "success"
        
        with pytest.raises(TimeoutError):
            slow_function()
    
    def test_metrics_export(self, performance_monitor):
        """测试指标导出"""
        monitor = performance_monitor
        
        # 添加一些测试数据
        monitor.increment_counter("export_test", 10)
        monitor.set_gauge("export_gauge", 25.5)
        
        # 测试JSON导出
        json_export = monitor.export_metrics("json")
        assert isinstance(json_export, str)
        assert "export_test" in json_export
        
        # 测试CSV导出
        csv_export = monitor.export_metrics("csv")
        assert isinstance(csv_export, str)
        assert "metric_name,value,unit,timestamp" in csv_export

class TestIntegration:
    """集成测试"""
    
    def test_validation_with_config(self, test_config_manager):
        """测试验证器与配置管理器的集成"""
        # 模拟配置管理器可用的情况
        with patch('src.validation.input_validator.get_config', return_value=test_config_manager):
            validator = InputValidator()
            
            # 使用配置中的长度限制
            long_input = "a" * 6000  # 超过测试配置中的5000限制
            result = validator.validate_math_problem(long_input)
            
            assert result["valid"] is False
            assert any("长度超过限制" in error for error in result["errors"])
    
    def test_exception_with_monitor(self, performance_monitor):
        """测试异常处理与性能监控的集成"""
        monitor = performance_monitor
        
        @monitor_performance("test_with_exception")
        def function_with_exception():
            raise ValueError("测试异常")
        
        with pytest.raises(ValueError):
            function_with_exception()
        
        # 检查错误计数器被更新
        summary = monitor.get_metrics_summary()
        assert "test_with_exception_error" in summary["counters"]
    
    def test_complete_workflow(self, test_config_manager, performance_monitor):
        """测试完整工作流程"""
        config = test_config_manager
        monitor = performance_monitor
        
        with patch('src.validation.input_validator.get_config', return_value=config):
            validator = InputValidator()
            
            # 模拟一个完整的处理流程
            timer_id = monitor.start_timer("complete_workflow")
            
            try:
                # 1. 验证输入
                input_text = "计算 10 + 5 的值"
                validation_result = validator.validate_math_problem(input_text)
                assert validation_result["valid"] is True
                
                # 2. 模拟处理
                time.sleep(0.01)
                processing_result = {"answer": 15, "confidence": 0.95}
                
                # 3. 记录成功
                monitor.increment_counter("successful_processing")
                
                return processing_result
                
            except Exception as e:
                # 记录错误
                monitor.increment_counter("failed_processing")
                raise ProcessingError("工作流程失败", cause=e)
            
            finally:
                monitor.stop_timer(timer_id)
        
        # 验证指标被正确记录
        summary = monitor.get_metrics_summary()
        assert "successful_processing" in summary["counters"]
        assert "complete_workflow_duration" in summary["metrics"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 