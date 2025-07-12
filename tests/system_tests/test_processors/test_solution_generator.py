import logging

import numpy as np
import pytest

from src.processors.solution_generator import SolutionGenerator


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """设置日志捕获"""
    # 确保捕获所有级别的日志
    caplog.set_level(logging.DEBUG)
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.DEBUG,
        force=True  # 强制重新配置
    )
    return caplog

def test_solve_equation_system(caplog):
    """测试方程组求解"""
    # 使用 with 语句确保在测试期间捕获日志
    with caplog.at_level(logging.DEBUG):
        solver = SolutionGenerator()
        
        # 测试日志是否工作
        solver.logger.info("TEST LOG MESSAGE")
        
        equation_system = {
            'equations': [
                {
                    'left_expr': 'x',
                    'right_expr': '2'
                },
                {
                    'left_expr': 'y',
                    'right_expr': '3'
                }
            ],
            'variables': {'x', 'y'},
            'constraints': []
        }
        
        # 在调用 solve 之前打印一条消息
        print("\nCalling solve method...")
        result = solver.solve(equation_system)
        print("Solve method completed")
        
        # 打印所有捕获的日志
        print("\nCaptured logs:")
        for record in caplog.records:
            print(f"{record.levelname}: {record.message}")
        
        # 打印日志总数
        print(f"\nTotal log records: {len(caplog.records)}")
        
        # 验证求解结果
        assert result['status'] == 'success'
        assert result['values']['x'] == 2
        assert result['values']['y'] == 3
        
        # 验证是否有日志记录
        assert len(caplog.records) > 0, "No logs were captured!"
        
        # 获取所有日志消息
        log_messages = [record.message for record in caplog.records]
        print("\nAll log messages:", log_messages)
        
        # 使用更宽松的断言
        assert any("求解" in msg for msg in log_messages), "No solving-related logs found"

def test_error_handling(caplog):
    """测试错误处理"""
    with caplog.at_level(logging.DEBUG):
        solver = SolutionGenerator()
        invalid_system = {
            'equations': [],
            'variables': set(),
            'constraints': []
        }
        
        result = solver.solve(invalid_system)
        
        # 验证错误处理
        assert result['status'] == 'error'
        
        # 打印捕获的日志
        print("\nCaptured error logs:")
        for record in caplog.records:
            print(f"{record.levelname}: {record.message}")