"""
Modular Architecture - Test Runner
=================================

运行所有模块化架构测试的主程序

Author: AI Assistant
Date: 2024-07-13
"""

import logging
import os
import sys
import time
import unittest
from typing import Any, Dict, List

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from test_data import TestDataIntegration, TestDataModule
from test_evaluation import TestEvaluationIntegration, TestEvaluationModule
from test_integration import (TestModularArchitectureIntegration,
                              TestModularArchitectureStress)
from test_models import TestModelsIntegration, TestModelsModule
# 导入所有测试模块
from test_processors import TestProcessorsIntegration, TestProcessorsModule


def run_all_tests() -> Dict[str, Any]:
    """运行所有测试并返回结果"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加各个模块的测试
    test_classes = [
        # Processors模块测试
        TestProcessorsModule,
        TestProcessorsIntegration,
        
        # Models模块测试
        TestModelsModule,
        TestModelsIntegration,
        
        # Evaluation模块测试
        TestEvaluationModule,
        TestEvaluationIntegration,
        
        # Data模块测试
        TestDataModule,
        TestDataIntegration,
        
        # 集成测试
        TestModularArchitectureIntegration,
        TestModularArchitectureStress,
    ]
    
    # 将测试类添加到套件中
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    print("=" * 80)
    print("开始运行模块化架构测试套件")
    print("=" * 80)
    
    start_time = time.time()
    
    # 创建测试运行器
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # 运行测试
    result = runner.run(test_suite)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 生成测试报告
    test_report = {
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
        "execution_time": execution_time,
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        "details": {
            "failures": [str(failure) for failure in result.failures],
            "errors": [str(error) for error in result.errors]
        }
    }
    
    # 打印测试结果摘要
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    print(f"总测试数: {test_report['total_tests']}")
    print(f"失败数: {test_report['failures']}")
    print(f"错误数: {test_report['errors']}")
    print(f"跳过数: {test_report['skipped']}")
    print(f"执行时间: {test_report['execution_time']:.2f} 秒")
    print(f"成功率: {test_report['success_rate']:.2%}")
    
    if result.failures:
        print("\n失败详情:")
        for failure in result.failures:
            print(f"  - {failure[0]}: {failure[1]}")
    
    if result.errors:
        print("\n错误详情:")
        for error in result.errors:
            print(f"  - {error[0]}: {error[1]}")
    
    print("=" * 80)
    
    return test_report


def run_module_specific_tests(module_name: str) -> Dict[str, Any]:
    """运行特定模块的测试"""
    
    # 模块测试类映射
    module_tests = {
        "processors": [TestProcessorsModule, TestProcessorsIntegration],
        "models": [TestModelsModule, TestModelsIntegration],
        "evaluation": [TestEvaluationModule, TestEvaluationIntegration],
        "data": [TestDataModule, TestDataIntegration],
        "integration": [TestModularArchitectureIntegration, TestModularArchitectureStress],
    }
    
    if module_name not in module_tests:
        print(f"错误: 未知模块 '{module_name}'")
        return {"error": f"未知模块: {module_name}"}
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加指定模块的测试
    for test_class in module_tests[module_name]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    print(f"开始运行 {module_name} 模块测试")
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    end_time = time.time()
    
    # 生成报告
    report = {
        "module": module_name,
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "execution_time": end_time - start_time,
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
    }
    
    print(f"\n{module_name} 模块测试完成:")
    print(f"  总测试数: {report['total_tests']}")
    print(f"  失败数: {report['failures']}")
    print(f"  错误数: {report['errors']}")
    print(f"  执行时间: {report['execution_time']:.2f} 秒")
    print(f"  成功率: {report['success_rate']:.2%}")
    
    return report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行模块化架构测试")
    parser.add_argument(
        "--module", 
        choices=["processors", "models", "evaluation", "data", "integration"],
        help="运行特定模块的测试"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="运行所有测试"
    )
    
    args = parser.parse_args()
    
    if args.module:
        # 运行特定模块的测试
        report = run_module_specific_tests(args.module)
        if "error" in report:
            sys.exit(1)
    elif args.all:
        # 运行所有测试
        report = run_all_tests()
        if report["failures"] > 0 or report["errors"] > 0:
            sys.exit(1)
    else:
        # 默认运行所有测试
        report = run_all_tests()
        if report["failures"] > 0 or report["errors"] > 0:
            sys.exit(1)


if __name__ == '__main__':
    main() 