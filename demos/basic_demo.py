#!/usr/bin/env python3
"""
COT-DIR 基础功能演示
==================

展示COT-DIR数学推理系统的核心功能：
1. 数据加载和预处理
2. 基础推理引擎
3. 简单问题求解演示

Author: COT-DIR Team
Date: 2025-01-31
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import DataLoader
from data.preprocessor import Preprocessor

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.bridge.reasoning_bridge import ReasoningEngine


def demo_basic_functionality():
    """演示基础功能"""
    print("🎯 COT-DIR 基础功能演示")
    print("=" * 50)
    
    # 1. 初始化组件
    print("\n1. 🔧 初始化系统组件...")
    try:
        loader = DataLoader()
        preprocessor = Preprocessor()
        engine = ReasoningEngine()
        print("✅ 系统组件初始化成功")
    except Exception as e:
        print(f"❌ 组件初始化失败: {e}")
        return
    
    # 2. 测试基础推理
    print("\n2. 🧠 测试基础推理功能...")
    test_problems = [
        "小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？",
        "一个长方形的长是8米，宽是6米，面积是多少平方米？",
        "一件衣服原价100元，打8折后多少钱？",
        "班级有30个学生，其中60%是男生，男生有多少人？"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n📝 测试问题 {i}: {problem}")
        
        try:
            # 预处理
            sample = {"problem": problem, "id": f"test_{i}"}
            processed = preprocessor.process(sample)
            
            # 推理求解
            result = engine.solve(processed)
            
            # 输出结果
            print(f"💡 答案: {result.get('final_answer', '未知')}")
            print(f"🎯 策略: {result.get('strategy_used', '未知')}")
            print(f"📊 置信度: {result.get('confidence', 0):.2f}")
            
            # 显示推理步骤（如果有）
            if 'reasoning_steps' in result and result['reasoning_steps']:
                print("🔍 推理步骤:")
                for step in result['reasoning_steps'][:3]:  # 只显示前3步
                    print(f"   - {step.get('description', step)}")
                if len(result['reasoning_steps']) > 3:
                    print(f"   ... 还有 {len(result['reasoning_steps']) - 3} 个步骤")
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
        
        print("-" * 40)
    
    # 3. 系统信息
    print("\n3. ℹ️ 系统信息")
    print(f"推理引擎版本: {getattr(engine, '__version__', '1.0.0')}")
    print(f"支持的策略: DIR, COT, 元知识推理")
    print(f"数据处理能力: 多格式支持")
    
    print("\n✅ 基础功能演示完成！")


def demo_data_loading():
    """演示数据加载功能"""
    print("\n📦 数据加载演示")
    print("-" * 30)
    
    try:
        loader = DataLoader()
        
        # 尝试加载数据集
        datasets_to_try = ["Math23K", "GSM8K", "test"]
        
        for dataset_name in datasets_to_try:
            try:
                print(f"📂 尝试加载数据集: {dataset_name}")
                samples = loader.load(dataset_name=dataset_name, max_samples=2)
                print(f"✅ 成功加载 {len(samples)} 个样本")
                
                if samples:
                    sample = samples[0]
                    print(f"   示例问题: {sample.get('problem', '无')[:50]}...")
                    
            except Exception as e:
                print(f"⚠️ 加载 {dataset_name} 失败: {e}")
                
    except Exception as e:
        print(f"❌ 数据加载器初始化失败: {e}")


def main():
    """主函数"""
    print("🚀 启动 COT-DIR 基础功能演示")
    
    try:
        demo_basic_functionality()
        demo_data_loading()
        
        print("\n🎉 演示完成！")
        print("\n📚 接下来可以尝试:")
        print("   - demos/enhanced_demo.py  (增强功能演示)")
        print("   - demos/validation_demo.py (验证和性能测试)")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 