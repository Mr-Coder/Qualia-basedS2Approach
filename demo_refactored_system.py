#!/usr/bin/env python3
"""
Refactored System Demo
=====================

Demonstration of the refactored mathematical reasoning system with:
1. Modular architecture
2. Comprehensive testing framework
3. Advanced evaluation system
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_reasoning_strategies():
    """Demonstrate the new reasoning strategy architecture"""
    print("\n" + "="*60)
    print("🧠 REASONING STRATEGIES DEMO")
    print("="*60)
    
    try:
        from reasoning_core.strategies.base_strategy import (ReasoningResult,
                                                             ReasoningStep)
        from reasoning_core.strategies.chain_of_thought import \
            ChainOfThoughtStrategy
        from reasoning_core.strategies.enhanced_cotdir_strategy import \
            EnhancedCOTDIRStrategy

        # Test problems with different complexity levels
        problems = [
            {
                "text": "小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？",
                "expected_answer": 8,
                "complexity": "L1"
            },
            {
                "text": "小明原有15个苹果，给了小红5个，又买了8个，小明现在有多少个苹果？",
                "expected_answer": 18,
                "complexity": "L2"
            },
            {
                "text": "班级有28个学生，要平均分成4组，每组有多少人？",
                "expected_answer": 7,
                "complexity": "L2"
            },
            {
                "text": "Janet每天产16个鸡蛋，早餐吃3个，做蛋糕用4个，剩下的以每个2美元卖掉，她每天赚多少钱？",
                "expected_answer": 18,
                "complexity": "L3"
            }
        ]
        
        # Test with basic Chain of Thought
        print("\n🔗 基础思维链策略测试:")
        basic_strategy = ChainOfThoughtStrategy()
        print(f"策略: {basic_strategy.name}")
        
        for i, problem in enumerate(problems[:2], 1):
            print(f"\n--- 问题 {i} (复杂度: {problem['complexity']}) ---")
            print(f"问题: {problem['text']}")
            
            start_time = time.time()
            result = basic_strategy.solve(problem['text'])
            end_time = time.time()
            
            print(f"答案: {result.final_answer}")
            print(f"期望: {problem['expected_answer']}")
            print(f"正确: {'✅' if str(result.final_answer) == str(problem['expected_answer']) else '❌'}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"处理时间: {end_time - start_time:.3f}s")
        
        # Test with Enhanced COT-DIR
        print(f"\n🚀 增强COT-DIR策略测试:")
        enhanced_strategy = EnhancedCOTDIRStrategy({
            'max_steps': 10,
            'confidence_threshold': 0.7,
            'validation_threshold': 0.8
        })
        print(f"策略: {enhanced_strategy.name}")
        
        total_correct = 0
        total_problems = len(problems)
        
        for i, problem in enumerate(problems, 1):
            print(f"\n{'='*80}")
            print(f"🧮 问题 {i} (复杂度: {problem['complexity']})")
            print(f"问题: {problem['text']}")
            print(f"期望答案: {problem['expected_answer']}")
            
            start_time = time.time()
            result = enhanced_strategy.solve(problem['text'])
            end_time = time.time()
            
            # 结果分析
            is_correct = abs(float(result.final_answer or 0) - problem['expected_answer']) < 0.01
            if is_correct:
                total_correct += 1
            
            print(f"\n📊 推理结果:")
            print(f"  最终答案: {result.final_answer}")
            print(f"  正确性: {'✅ 正确' if is_correct else '❌ 错误'}")
            print(f"  总体置信度: {result.confidence:.3f}")
            print(f"  处理时间: {end_time - start_time:.3f}s")
            print(f"  成功状态: {result.success}")
            
            # 显示推理层信息
            if result.metadata and 'reasoning_layers' in result.metadata:
                print(f"\n🏗️ 推理层分析:")
                for layer in result.metadata['reasoning_layers']:
                    print(f"  {layer['level']}: 置信度={layer['confidence']:.3f}, 操作数={layer['operations_count']}")
            
            # 显示验证结果
            if result.metadata and 'validation_results' in result.metadata:
                print(f"\n✅ 5维验证结果:")
                for vr in result.metadata['validation_results']:
                    status = "通过" if vr['passed'] else "失败"
                    print(f"  {vr['dimension']}: {status} (分数: {vr['score']:.3f})")
            
            # 显示详细推理步骤
            print(f"\n🔍 详细推理步骤:")
            for j, step in enumerate(result.reasoning_steps, 1):
                print(f"  步骤{j}: {step.explanation}")
                if step.metadata and "reasoning_level" in step.metadata:
                    print(f"    └─ 层次: {step.metadata['reasoning_level']}, 置信度: {step.confidence:.3f}")
            
            # 显示发现的实体和关系
            if result.metadata:
                print(f"\n🔎 发现信息:")
                print(f"  实体数量: {result.metadata.get('entities_found', 0)}")
                print(f"  关系数量: {result.metadata.get('relations_discovered', 0)}")
                print(f"  复杂度分级: {result.metadata.get('complexity_level', 'Unknown')}")
        
        # 总体性能摘要
        print(f"\n" + "="*80)
        print(f"📈 Enhanced COT-DIR 性能摘要")
        print(f"="*80)
        
        accuracy = total_correct / total_problems * 100
        print(f"准确率: {accuracy:.1f}% ({total_correct}/{total_problems})")
        
        performance_summary = enhanced_strategy.get_performance_summary()
        print(f"平均置信度: {performance_summary['average_confidence']:.3f}")
        print(f"平均处理时间: {performance_summary['average_processing_time']:.3f}s")
        print(f"验证通过率: {performance_summary['validation_pass_rate']:.3f}")
        
        print(f"\n🎯 COT-DIR核心优势展现:")
        print(f"1️⃣ 显式关系发现 vs 隐式推理")
        print(f"   * 传统方法: ChatGPT/Qwen隐式处理关系")
        print(f"   * COT-DIR: 明确识别每个关系，提供数学公式和置信度")
        
        print(f"2️⃣ 结构化多层推理 vs 线性思维")
        print(f"   * 传统方法: 简单步骤序列")
        print(f"   * COT-DIR: L1→L2→L3层次化推理，每层有明确任务")
        
        print(f"3️⃣ 全面置信度验证 vs 无验证")
        print(f"   * 传统方法: 无验证机制")
        print(f"   * COT-DIR: 5维度验证(语法、数学、逻辑、语义、目标)")
        
        print(f"4️⃣ 完整中间结果追踪 vs 黑盒输出")
        print(f"   * 传统方法: 只有最终答案")
        print(f"   * COT-DIR: 每步中间结果全部可见和追踪")
        
        print("\n✅ 推理策略演示完成!")
        
    except ImportError as e:
        print(f"❌ Could not import reasoning modules: {e}")
        return False
    
    return True


def demo_tool_integration():
    """Demonstrate external tool integration"""
    print("\n" + "="*60)
    print("🔧 TOOL INTEGRATION DEMO")
    print("="*60)
    
    try:
        from reasoning_core.tools.symbolic_math import SymbolicMathTool
        
        tool = SymbolicMathTool()
        
        print(f"Tool: {tool.name}")
        print(f"Available: {tool.is_available}")
        print(f"Supported operations: {tool.get_supported_operations()}")
        
        if tool.is_available:
            # Test operations
            test_operations = [
                ("solve_equation", "x + 2 - 5", "x"),
                ("simplify", "x + x + 2*x"),
                ("factor", "x**2 - 4")
            ]
            
            for operation, *args in test_operations:
                print(f"\n--- Testing {operation} ---")
                print(f"Input: {args}")
                
                result = tool.execute(operation, *args)
                print(f"Success: {result.success}")
                if result.success:
                    print(f"Result: {result.result}")
                else:
                    print(f"Error: {result.error_message}")
        
        print("\n✅ Tool integration demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Could not import tool modules: {e}")
        return False
    
    return True


def demo_evaluation_system():
    """Demonstrate the comprehensive evaluation system"""
    print("\n" + "="*60)
    print("📊 EVALUATION SYSTEM DEMO")
    print("="*60)
    
    try:
        from evaluation.evaluator import ComprehensiveEvaluator
        from evaluation.metrics import AccuracyMetric, EfficiencyMetric

        # Sample data for evaluation
        predictions = [8, 120, 4, 15, 25]
        ground_truth = [8, 120, 4, 16, 25]  # One incorrect answer
        
        # Sample metadata
        metadata = {
            'processing_times': [0.5, 1.2, 0.8, 2.1, 0.9],
            'confidence_scores': [0.9, 0.85, 0.92, 0.7, 0.88],
            'reasoning_steps': [
                [{"confidence": 0.9}] * 3,  # 3 steps for first problem
                [{"confidence": 0.85}] * 4,  # 4 steps for second problem  
                [{"confidence": 0.92}] * 2,  # 2 steps for third problem
                [{"confidence": 0.7}] * 5,   # 5 steps for fourth problem
                [{"confidence": 0.88}] * 3   # 3 steps for fifth problem
            ],
            'explanations': [
                "Step by step addition calculation",
                "Distance = speed × time calculation", 
                "Algebraic equation solving",
                "Complex arithmetic problem",
                "Simple multiplication"
            ]
        }
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator()
        
        print("Evaluator initialized with metrics:")
        for metric_name in evaluator.get_available_metrics():
            print(f"  - {metric_name}")
        
        print(f"\nMetric weights:")
        weights = evaluator.get_metric_weights()
        for metric, weight in weights.items():
            print(f"  - {metric}: {weight:.2f}")
        
        # Run evaluation
        print(f"\nRunning evaluation on {len(predictions)} predictions...")
        
        result = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            dataset_name="demo_dataset",
            model_name="chain_of_thought_v1",
            metadata=metadata
        )
        
        print(f"\n--- Evaluation Results ---")
        print(f"Dataset: {result.dataset_name}")
        print(f"Model: {result.model_name}")
        print(f"Overall Score: {result.overall_score:.3f}")
        
        print(f"\nDetailed Metrics:")
        for metric_name, metric_result in result.metric_results.items():
            print(f"  {metric_name}: {metric_result.score:.3f}/{metric_result.max_score}")
            if 'error' not in metric_result.details:
                # Show some details
                if metric_name == 'accuracy':
                    details = metric_result.details
                    print(f"    Accuracy: {details['accuracy_percentage']:.1f}% ({details['correct']}/{details['total']})")
                elif metric_name == 'efficiency':
                    details = metric_result.details
                    print(f"    Avg time: {details['average_time']:.3f}s")
        
        print("\n✅ Evaluation system demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Could not import evaluation modules: {e}")
        return False
    
    return True


def demo_dataset_loading():
    """Demonstrate dataset loading capabilities"""
    print("\n" + "="*60)
    print("📚 DATASET LOADING DEMO")
    print("="*60)
    
    try:
        # Add Data directory to path
        sys.path.insert(0, str(Path(__file__).parent / "Data"))
        from dataset_loader import MathDatasetLoader
        
        loader = MathDatasetLoader()
        
        # List available datasets
        datasets = loader.list_datasets()
        print(f"Available datasets: {len(datasets)}")
        
        for dataset in datasets[:5]:  # Show first 5
            print(f"  - {dataset}")
        
        if datasets:
            # Load a small sample from first dataset
            print(f"\nLoading sample from {datasets[0]}...")
            data = loader.load_dataset(datasets[0], max_samples=3)
            
            print(f"Loaded {len(data)} samples:")
            for i, item in enumerate(data):
                print(f"  Sample {i+1}: {str(item)[:100]}...")
        
        print("\n✅ Dataset loading demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Could not import dataset loader: {e}")
        return False
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return False
    
    return True


def demo_testing_framework():
    """Demonstrate the testing framework"""
    print("\n" + "="*60)
    print("🧪 TESTING FRAMEWORK DEMO")
    print("="*60)
    
    # Show test structure
    test_dir = Path(__file__).parent / "tests"
    if test_dir.exists():
        print("Test directory structure:")
        for item in test_dir.iterdir():
            if item.is_dir():
                test_count = len(list(item.glob("test_*.py")))
                print(f"  {item.name}/: {test_count} test files")
    
    # Show pytest configuration
    pytest_ini = Path(__file__).parent / "pytest.ini"
    if pytest_ini.exists():
        print(f"\nPytest configuration found: {pytest_ini}")
        with open(pytest_ini) as f:
            lines = f.readlines()[:10]  # Show first 10 lines
            for line in lines:
                print(f"  {line.rstrip()}")
    
    # Show available test commands
    print(f"\nAvailable test commands:")
    print(f"  pytest tests/unit_tests/ -v          # Run unit tests")
    print(f"  pytest tests/integration_tests/ -v   # Run integration tests")
    print(f"  pytest tests/performance_tests/ -v   # Run performance tests")
    print(f"  pytest -m smoke                      # Run smoke tests")
    print(f"  pytest -m 'not slow'                 # Run fast tests only")
    
    print("\n✅ Testing framework demo completed successfully!")
    return True


def run_full_demo():
    """Run complete demonstration of refactored system"""
    print("🚀 REFACTORED MATHEMATICAL REASONING SYSTEM DEMO")
    print("=" * 80)
    
    demos = [
        ("Reasoning Strategies", demo_reasoning_strategies),
        ("Tool Integration", demo_tool_integration), 
        ("Evaluation System", demo_evaluation_system),
        ("Dataset Loading", demo_dataset_loading),
        ("Testing Framework", demo_testing_framework)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\nStarting {demo_name} demo...")
            start_time = time.time()
            success = demo_func()
            end_time = time.time()
            
            results[demo_name] = {
                'success': success,
                'time': end_time - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in {demo_name} demo: {e}")
            results[demo_name] = {
                'success': False,
                'time': 0,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "="*80)
    print("📋 DEMO SUMMARY")
    print("="*80)
    
    total_time = sum(r['time'] for r in results.values())
    successful_demos = sum(1 for r in results.values() if r['success'])
    
    print(f"Total demos run: {len(demos)}")
    print(f"Successful demos: {successful_demos}")
    print(f"Total time: {total_time:.2f}s")
    
    print(f"\nDetailed results:")
    for demo_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        time_str = f"{result['time']:.2f}s"
        print(f"  {demo_name:20} {status:8} {time_str:>8}")
        
        if not result['success'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Architecture overview
    print(f"\n" + "="*80)
    print("🏗️  REFACTORED ARCHITECTURE OVERVIEW")
    print("="*80)
    
    print("New modular structure:")
    print("  src/")
    print("  ├── reasoning_core/        # Core reasoning engine")
    print("  │   ├── strategies/       # Different reasoning strategies")
    print("  │   ├── tools/           # External tool integration")
    print("  │   └── validation/      # Validation mechanisms")
    print("  ├── evaluation/           # Comprehensive evaluation system")
    print("  │   ├── metrics.py       # Individual metrics")
    print("  │   ├── evaluator.py     # Main evaluation engine")
    print("  │   └── reports.py       # Report generation")
    print("  └── ... (existing modules)")
    print("")
    print("  tests/")
    print("  ├── unit_tests/          # Unit tests")
    print("  ├── integration_tests/   # Integration tests")
    print("  └── performance_tests/   # Performance benchmarks")
    print("")
    print("  demos/                   # Organized demo files")
    print("  config_files/            # Configuration files")
    print("  legacy/                  # Legacy code")
    
    return successful_demos == len(demos)


if __name__ == "__main__":
    success = run_full_demo()
    
    if success:
        print(f"\n🎉 All demos completed successfully!")
        print(f"The refactored system is ready for use.")
    else:
        print(f"\n⚠️  Some demos failed. Check the error messages above.")
        print(f"You may need to install missing dependencies or fix import paths.")
    
    sys.exit(0 if success else 1) 