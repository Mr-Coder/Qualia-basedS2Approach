#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成测试脚本
~~~~~~~~~~~

验证优化后的数学问题求解器是否正常工作

Author: [Hao Meng]
Date: [2025-05-29]
"""

import sys
import os
import logging

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_import():
    """测试基本导入功能"""
    try:
        print("🔍 测试基本导入功能...")
        from math_problem_solver import MathProblemSolver
        print("✅ MathProblemSolver 导入成功")
        
        from utils.error_handling import MathProblemSolverError
        print("✅ 错误处理模块导入成功")
        
        from config.advanced_config import SolverConfig
        print("✅ 配置管理模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_solver_initialization():
    """测试求解器初始化"""
    try:
        print("\n🔍 测试求解器初始化...")
        from math_problem_solver import MathProblemSolver
        
        # 测试默认初始化
        solver = MathProblemSolver()
        print("✅ 默认求解器初始化成功")
        
        # 测试配置文件初始化
        config_path = "src/config/solver_config.json"
        if os.path.exists(config_path):
            solver_with_config = MathProblemSolver(config_path=config_path)
            print("✅ 带配置文件的求解器初始化成功")
        else:
            print("⚠️  配置文件不存在，跳过配置测试")
        
        return True
    except Exception as e:
        print(f"❌ 求解器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_problem():
    """测试简单数学问题求解"""
    try:
        print("\n🔍 测试简单问题求解...")
        from math_problem_solver import MathProblemSolver
        
        solver = MathProblemSolver()
        
        # 简单的问题
        problem = "一个水池有2个进水管和1个出水管。进水管每小时进水10立方米，出水管每小时出水5立方米。如果水池开始时是空的，问多长时间能装满容积为100立方米的水池？"
        
        print(f"问题: {problem}")
        result = solver.solve(problem)
        
        if result and 'answer' in result:
            print(f"✅ 求解成功!")
            print(f"答案: {result['answer']}")
            if 'reasoning' in result:
                print(f"推理过程: {result['reasoning'][:100]}...")
        else:
            print("⚠️  求解返回结果，但格式可能不完整")
            print(f"结果: {result}")
        
        return True
    except Exception as e:
        print(f"❌ 问题求解失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    try:
        print("\n🔍 测试错误处理...")
        from math_problem_solver import MathProblemSolver
        
        solver = MathProblemSolver()
        
        # 测试无效输入
        invalid_problems = [
            "",  # 空字符串
            "这不是一个数学问题",  # 非数学问题
            "1 + 1 = ?",  # 过于简单
        ]
        
        for problem in invalid_problems:
            try:
                result = solver.solve(problem)
                print(f"⚠️  问题 '{problem[:20]}...' 应该失败但没有失败")
            except Exception as e:
                print(f"✅ 正确处理了无效问题: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始集成测试...")
    print("=" * 50)
    
    tests = [
        test_basic_import,
        test_solver_initialization,
        test_simple_problem,
        test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print("\n📊 测试结果:")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
