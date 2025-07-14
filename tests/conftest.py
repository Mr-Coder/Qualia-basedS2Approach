"""
pytest 配置文件
为测试提供统一的fixtures和配置
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_manager import ConfigurationManager, init_config
from src.core.exceptions import COTBaseException
from src.monitoring.performance_monitor import PerformanceMonitor, init_monitor
from src.validation.input_validator import InputValidator

# 配置测试日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session")
def test_config_dir():
    """创建临时测试配置目录"""
    temp_dir = tempfile.mkdtemp(prefix="cot_test_config_")
    config_dir = Path(temp_dir)
    
    # 创建配置目录结构
    (config_dir / "environments").mkdir(parents=True, exist_ok=True)
    (config_dir / "security").mkdir(parents=True, exist_ok=True)
    (config_dir / "templates").mkdir(parents=True, exist_ok=True)
    
    # 创建测试配置文件
    base_config = {
        "reasoning": {
            "max_steps": 10,
            "confidence_threshold": 0.8,
            "timeout_seconds": 20.0
        },
        "validation": {
            "max_input_length": 5000,
            "enable_security_check": True
        },
        "performance": {
            "enable_monitoring": True,
            "enable_caching": False
        },
        "features": {
            "enable_debug_mode": True
        }
    }
    
    # 写入基础配置
    import yaml
    with open(config_dir / "environments" / "base.yaml", 'w') as f:
        yaml.dump(base_config, f)
    
    # 写入测试环境配置
    test_config = {
        "logging": {"level": "DEBUG"},
        "testing": {"mode": True}
    }
    
    with open(config_dir / "environments" / "test.yaml", 'w') as f:
        yaml.dump(test_config, f)
    
    yield str(config_dir)
    
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def test_config_manager(test_config_dir):
    """提供测试配置管理器"""
    os.environ["COT_ENV"] = "test"
    config_manager = init_config(env="test", config_dir=test_config_dir)
    yield config_manager
    # 清理环境变量
    if "COT_ENV" in os.environ:
        del os.environ["COT_ENV"]

@pytest.fixture
def input_validator():
    """提供输入验证器实例"""
    return InputValidator()

@pytest.fixture
def performance_monitor():
    """提供性能监控器实例"""
    monitor = init_monitor()
    yield monitor
    # 清理监控器
    monitor.stop_system_monitoring()

@pytest.fixture
def sample_math_problems():
    """提供示例数学问题"""
    return [
        "小明有3个苹果，小红给了他2个苹果，小明现在有多少个苹果？",
        "一个长方形的长是8米，宽是6米，求这个长方形的面积。",
        "计算 25 + 17 的值",
        "如果一本书有200页，小王每天读20页，他需要多少天才能读完？",
        "班级里有30个学生，其中15个是男生，女生有多少人？"
    ]

@pytest.fixture
def sample_invalid_inputs():
    """提供无效输入示例"""
    return [
        "",  # 空字符串
        "<script>alert('xss')</script>",  # XSS攻击
        "import os; os.system('rm -rf /')",  # 代码注入
        "a" * 20000,  # 超长输入
        None,  # None值
        123,  # 非字符串类型
    ]

@pytest.fixture
def sample_reasoning_context():
    """提供推理上下文示例"""
    from src.core.interfaces import ReasoningContext
    
    return ReasoningContext(
        problem_text="计算 5 + 3 的值",
        problem_type="arithmetic",
        parameters={"precision": 2},
        history=[],
        constraints={"max_steps": 5}
    )

@pytest.fixture
def mock_processing_result():
    """提供模拟处理结果"""
    from src.core.interfaces import ProcessingResult, ProcessingStatus
    
    return ProcessingResult(
        success=True,
        result=8,
        confidence=0.95,
        processing_time=0.1,
        status=ProcessingStatus.COMPLETED,
        metadata={"steps": 2, "method": "addition"}
    )

@pytest.fixture
def temp_test_files():
    """创建临时测试文件"""
    temp_dir = tempfile.mkdtemp(prefix="cot_test_files_")
    temp_path = Path(temp_dir)
    
    # 创建一些测试文件
    (temp_path / "test.txt").write_text("test content")
    (temp_path / "data.json").write_text('{"test": "data"}')
    
    yield temp_path
    
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(autouse=True)
def reset_global_state():
    """重置全局状态（每个测试前后执行）"""
    # 测试开始前的设置
    yield
    
    # 测试结束后的清理
    # 这里可以添加清理全局状态的代码
    pass

# 测试标记定义
def pytest_configure(config):
    """配置测试标记"""
    config.addinivalue_line(
        "markers", "unit: 单元测试"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试"
    )
    config.addinivalue_line(
        "markers", "performance: 性能测试"
    )
    config.addinivalue_line(
        "markers", "security: 安全测试"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试"
    )

# 测试钩子
def pytest_runtest_setup(item):
    """测试运行前的设置"""
    # 为每个测试设置独立的日志
    logger = logging.getLogger(f"test.{item.name}")
    logger.info(f"开始测试: {item.name}")

def pytest_runtest_teardown(item, nextitem):
    """测试运行后的清理"""
    logger = logging.getLogger(f"test.{item.name}")
    logger.info(f"测试完成: {item.name}")

# 异常处理助手
@pytest.fixture
def assert_exception():
    """断言异常的辅助函数"""
    def _assert_exception(exception_class, func, *args, **kwargs):
        """断言函数会抛出指定异常"""
        with pytest.raises(exception_class):
            func(*args, **kwargs)
    
    return _assert_exception

# 性能测试助手
@pytest.fixture
def performance_timer():
    """性能计时器"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# 测试数据生成器
@pytest.fixture
def test_data_generator():
    """测试数据生成器"""
    import random
    import string
    
    class DataGenerator:
        @staticmethod
        def random_string(length=10):
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        
        @staticmethod
        def random_number(min_val=1, max_val=100):
            return random.randint(min_val, max_val)
        
        @staticmethod
        def random_math_problem():
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            operation = random.choice(['+', '-', '*'])
            return f"计算 {a} {operation} {b} 的值"
        
        @staticmethod
        def invalid_input_data():
            return [
                None,
                "",
                " " * 100,
                "<script>",
                "javascript:",
                "eval(",
                "../../../etc/passwd"
            ]
    
    return DataGenerator()

# 模拟工具
@pytest.fixture
def mock_helper():
    """模拟助手"""
    from unittest.mock import MagicMock, Mock, patch
    
    class MockHelper:
        @staticmethod
        def create_mock_config():
            mock_config = Mock()
            mock_config.get.return_value = "default_value"
            return mock_config
        
        @staticmethod
        def create_mock_validator():
            mock_validator = Mock()
            mock_validator.validate_math_problem.return_value = {
                "valid": True,
                "sanitized_text": "test",
                "warnings": [],
                "errors": []
            }
            return mock_validator
        
        @staticmethod
        def create_mock_monitor():
            mock_monitor = Mock()
            mock_monitor.start_timer.return_value = "timer_id"
            mock_monitor.stop_timer.return_value = 0.1
            return mock_monitor
    
    return MockHelper() 