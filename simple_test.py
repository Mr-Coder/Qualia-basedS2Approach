"""
🚀 简单测试 - 验证优化模块计算结果
Simple Test - 直接测试单个优化模块
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'processors'))

def test_intelligent_classifier():
    """测试智能分类器"""
    print("🧠 测试智能分类器")
    print("=" * 40)
    
    try:
        # 直接导入
        from intelligent_classifier import IntelligentClassifier, ProblemType
        
        classifier = IntelligentClassifier()
        
        # 测试样本题目
        test_problems = [
            "John has 5 apples and bought 3 more. How many apples does he have now?",
            "A car travels 60 km/h for 2 hours. How far did it travel?", 
            "The ratio of boys to girls in a class is 3:2. If there are 15 boys, how many girls are there?",
            "Find the area of a rectangle with length 8 cm and width 6 cm.",
            "If 3x + 5 = 17, what is the value of x?"
        ]
        
        print(f"📝 分类 {len(test_problems)} 个数学题目:")
        print()
        
        total_confidence = 0
        classification_counts = {}
        
        for i, problem in enumerate(test_problems, 1):
            result = classifier.classify(problem)
            problem_type = result.problem_type.value
            confidence = result.confidence
            
            print(f"{i}. 题目: {problem}")
            print(f"   分类: {problem_type}")
            print(f"   置信度: {confidence:.2f}")
            print(f"   模板: {result.template_match}")
            print()
            
            total_confidence += confidence
            classification_counts[problem_type] = classification_counts.get(problem_type, 0) + 1
        
        # 统计结果
        avg_confidence = total_confidence / len(test_problems)
        print("📊 分类统计结果:")
        print(f"   平均置信度: {avg_confidence:.2f}")
        print("   题型分布:")
        for ptype, count in classification_counts.items():
            print(f"     - {ptype}: {count} 题")
        
        print("✅ 智能分类器测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 智能分类器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_dataset_manager():
    """测试动态数据集管理器"""
    print("\n🚀 测试动态数据集管理器")
    print("=" * 40)
    
    try:
        from dynamic_dataset_manager import DynamicDatasetManager
        
        dataset_manager = DynamicDatasetManager(
            data_dirs=["Data"],
            watch_mode=False,
            auto_reload=False
        )
        
        # 测试数据集发现
        datasets = dataset_manager.discover_datasets()
        print(f"📊 发现数据集: {len(datasets)} 个")
        
        total_problems = 0
        for name, metadata in list(datasets.items())[:5]:  # 显示前5个
            print(f"   - {name}: {metadata.problem_count} 题目")
            total_problems += metadata.problem_count
        
        if len(datasets) > 5:
            remaining = len(datasets) - 5
            for name, metadata in list(datasets.items())[5:]:
                total_problems += metadata.problem_count
            print(f"   ... 另外 {remaining} 个数据集")
        
        print(f"📈 总计: {total_problems} 个题目")
        
        # 测试动态批次生成
        if datasets:
            batch = dataset_manager.get_dynamic_batch(batch_size=3)
            print(f"📦 生成测试批次:")
            print(f"   批次ID: {batch.batch_id}")
            print(f"   源数据集: {batch.source_dataset}")
            print(f"   题目数量: {len(batch.problems)}")
            
            # 显示一个示例题目
            if batch.problems:
                sample = batch.problems[0]
                sample_text = str(sample)[:80] + "..." if len(str(sample)) > 80 else str(sample)
                print(f"   示例题目: {sample_text}")
        
        print("✅ 动态数据集管理器测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 动态数据集管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processor():
    """测试批量处理器"""
    print("\n📊 测试批量处理器")
    print("=" * 40)
    
    try:
        import time

        from batch_processor import BatchProcessor
        
        processor = BatchProcessor(max_workers=2)
        
        # 创建简单的测试数据
        test_items = [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 3, "value": 30},
            {"id": 4, "value": 40},
            {"id": 5, "value": 50}
        ]
        
        def simple_processor(item):
            """简单的处理函数：计算平方"""
            time.sleep(0.1)  # 模拟处理时间
            return {
                "id": item["id"],
                "original": item["value"],
                "squared": item["value"] ** 2,
                "status": "processed"
            }
        
        print(f"🔄 批量处理 {len(test_items)} 个项目...")
        
        start_time = time.time()
        results = processor.process_batch(
            items=test_items,
            processor_func=simple_processor,
            description="平方计算测试"
        )
        processing_time = time.time() - start_time
        
        print("📋 处理结果:")
        print(f"   总项目: {results.total_items}")
        print(f"   成功: {results.success_count}")
        print(f"   失败: {results.failure_count}")
        print(f"   用时: {processing_time:.2f}秒")
        
        # 显示具体结果
        print("   详细结果:")
        for item in results.processed_items[:3]:  # 显示前3个
            print(f"     ID {item['id']}: {item['original']}² = {item['squared']}")
        
        if len(results.processed_items) > 3:
            print(f"     ... 另外 {len(results.processed_items) - 3} 个结果")
        
        print("✅ 批量处理器测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 批量处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scalable_architecture():
    """测试可扩展架构"""
    print("\n🔧 测试可扩展架构")
    print("=" * 40)
    
    try:
        from scalable_architecture import (BasePlugin, ModularFramework,
                                           ModuleType, PluginInfo)
        
        framework = ModularFramework()
        
        # 创建简单的数学插件
        class MathPlugin(BasePlugin):
            def get_info(self):
                return PluginInfo(
                    plugin_id="math_plugin",
                    name="数学计算插件",
                    version="1.0.0",
                    description="执行基本数学运算",
                    module_type=ModuleType.PROCESSOR
                )
            
            def process(self, input_data, config=None):
                if isinstance(input_data, dict) and "operation" in input_data:
                    op = input_data["operation"]
                    a = input_data.get("a", 0)
                    b = input_data.get("b", 0)
                    
                    if op == "add":
                        result = a + b
                    elif op == "multiply":
                        result = a * b
                    elif op == "power":
                        result = a ** b
                    else:
                        result = None
                    
                    return {
                        "input": input_data,
                        "result": result,
                        "processed_by": "math_plugin"
                    }
                return {"error": "无效输入"}
        
        # 注册插件
        framework.register_processor(MathPlugin)
        
        # 测试插件
        test_operations = [
            {"operation": "add", "a": 5, "b": 3},
            {"operation": "multiply", "a": 4, "b": 7},
            {"operation": "power", "a": 2, "b": 3}
        ]
        
        print("🔌 测试数学插件:")
        for i, op_data in enumerate(test_operations, 1):
            result = framework.run_pipeline(["math_plugin"], op_data)
            
            op = op_data["operation"]
            a, b = op_data["a"], op_data["b"]
            calc_result = result.get("result", "错误")
            
            print(f"   {i}. {a} {op} {b} = {calc_result}")
        
        # 检查插件状态
        plugins = framework.list_processors()
        print(f"📋 已注册插件: {len(plugins)} 个")
        for pid, info in plugins.items():
            print(f"   - {info.name} (ID: {pid})")
        
        print("✅ 可扩展架构测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 可扩展架构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 优化模块计算结果测试")
    print("=" * 60)
    
    # 运行各项测试
    test_results = []
    
    test_results.append(("智能分类器", test_intelligent_classifier()))
    test_results.append(("动态数据集管理器", test_dynamic_dataset_manager()))
    test_results.append(("批量处理器", test_batch_processor()))
    test_results.append(("可扩展架构", test_scalable_architecture()))
    
    # 总结测试结果
    print("\n📊 测试总结")
    print("=" * 60)
    
    success_count = 0
    for module_name, success in test_results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{module_name}: {status}")
        if success:
            success_count += 1
    
    success_rate = success_count / len(test_results) * 100
    print(f"\n🎉 测试完成: {success_count}/{len(test_results)} 个模块成功 ({success_rate:.1f}%)")
    
    if success_count == len(test_results):
        print("🎊 所有优化模块的计算功能正常!")
    else:
        print("⚠️  部分模块需要修复")

if __name__ == "__main__":
    main() 