"""
🚀 独立优化演示
Standalone Optimization Demo - 测试四个优化方向

四个优化方向：
🚀 零代码添加新题目 (动态数据集加载)
🧠 智能分类和模板匹配 (10种题型自动识别)
📊 批量处理和质量评估 (标准化流程)
🔧 高度可扩展架构 (模块化设计)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.processors.batch_processor import BatchProcessor
# 直接导入四个优化模块
from src.processors.dynamic_dataset_manager import (DynamicDatasetManager,
                                                    ProblemBatch)
from src.processors.intelligent_classifier import (IntelligentClassifier,
                                                   ProblemType)
from src.processors.scalable_architecture import (BasePlugin, ModularFramework,
                                                  ModuleType, PluginInfo)


class StandaloneOptimizationDemo:
    """🚀 独立优化演示系统"""
    
    def __init__(self):
        """初始化优化系统"""
        print("🚀 独立优化系统初始化")
        print("=" * 60)
        
        # 1. 初始化动态数据集管理器
        print("🚀 1. 初始化动态数据集管理器...")
        try:
            self.dataset_manager = DynamicDatasetManager(
                data_dirs=["Data"],
                watch_mode=False,  # 关闭监控模式避免复杂性
                auto_reload=False
            )
            print("✅ 动态数据集管理器初始化成功")
        except Exception as e:
            print(f"❌ 动态数据集管理器初始化失败: {e}")
            self.dataset_manager = None
        
        # 2. 初始化智能分类器
        print("\n🧠 2. 初始化智能分类器...")
        try:
            self.classifier = IntelligentClassifier()
            print("✅ 智能分类器初始化成功")
        except Exception as e:
            print(f"❌ 智能分类器初始化失败: {e}")
            self.classifier = None
        
        # 3. 初始化批量处理器
        print("\n📊 3. 初始化批量处理器...")
        try:
            self.batch_processor = BatchProcessor(max_workers=2)
            print("✅ 批量处理器初始化成功")
        except Exception as e:
            print(f"❌ 批量处理器初始化失败: {e}")
            self.batch_processor = None
        
        # 4. 初始化模块化框架
        print("\n🔧 4. 初始化模块化框架...")
        try:
            self.framework = ModularFramework()
            print("✅ 模块化框架初始化成功")
        except Exception as e:
            print(f"❌ 模块化框架初始化失败: {e}")
            self.framework = None
        
        print("\n✅ 独立优化系统初始化完成!")
        print("=" * 60)
    
    def test_dynamic_dataset_loading(self):
        """测试动态数据集加载功能"""
        print("\n🚀 测试1: 动态数据集加载")
        print("-" * 40)
        
        if not self.dataset_manager:
            print("❌ 数据集管理器未初始化")
            return
        
        try:
            # 测试数据集发现
            datasets = self.dataset_manager.discover_datasets()
            print(f"📊 发现数据集数量: {len(datasets)}")
            
            for name, metadata in list(datasets.items())[:3]:  # 只显示前3个
                print(f"  - {name}: {metadata.problem_count} 题目")
            
            # 测试动态批次生成
            if datasets:
                batch = self.dataset_manager.get_dynamic_batch(batch_size=5)
                print(f"📦 生成批次: {batch.batch_id}")
                print(f"   源数据集: {batch.source_dataset}")
                print(f"   题目数量: {len(batch.problems)}")
                
                # 显示一个示例题目
                if batch.problems:
                    sample_problem = batch.problems[0]
                    print(f"   示例题目: {str(sample_problem)[:100]}...")
            
            print("✅ 动态数据集加载测试成功")
            
        except Exception as e:
            print(f"❌ 动态数据集加载测试失败: {e}")
    
    def test_intelligent_classification(self):
        """测试智能分类功能"""
        print("\n🧠 测试2: 智能分类")
        print("-" * 40)
        
        if not self.classifier:
            print("❌ 智能分类器未初始化")
            return
        
        try:
            # 测试样本题目
            test_problems = [
                "John has 5 apples and bought 3 more. How many apples does he have now?",
                "A car travels 60 km/h for 2 hours. How far did it travel?",
                "The ratio of boys to girls in a class is 3:2. If there are 15 boys, how many girls are there?",
                "Find the area of a rectangle with length 8 cm and width 6 cm.",
                "If 3x + 5 = 17, what is the value of x?"
            ]
            
            print(f"📝 测试 {len(test_problems)} 个样本题目:")
            
            for i, problem in enumerate(test_problems, 1):
                result = self.classifier.classify(problem)
                print(f"  {i}. 类型: {result.problem_type.value}")
                print(f"     置信度: {result.confidence:.2f}")
                print(f"     题目: {problem[:50]}...")
                print()
            
            print("✅ 智能分类测试成功")
            
        except Exception as e:
            print(f"❌ 智能分类测试失败: {e}")
    
    def test_batch_processing(self):
        """测试批量处理功能"""
        print("\n📊 测试3: 批量处理")
        print("-" * 40)
        
        if not self.batch_processor:
            print("❌ 批量处理器未初始化")
            return
        
        try:
            # 创建测试数据
            test_data = [
                {"problem": "2 + 3 = ?", "answer": 5},
                {"problem": "10 - 4 = ?", "answer": 6},
                {"problem": "3 × 4 = ?", "answer": 12},
                {"problem": "15 ÷ 3 = ?", "answer": 5},
                {"problem": "2² = ?", "answer": 4}
            ]
            
            def simple_processor(item):
                """简单的处理函数"""
                time.sleep(0.1)  # 模拟处理时间
                return {
                    "original": item,
                    "processed": True,
                    "status": "success"
                }
            
            print(f"🔄 批量处理 {len(test_data)} 个项目...")
            
            # 执行批量处理
            results = self.batch_processor.process_batch(
                items=test_data,
                processor_func=simple_processor,
                description="测试批量处理"
            )
            
            print(f"📋 处理报告:")
            print(f"   总项目数: {results.total_items}")
            print(f"   成功数: {results.success_count}")
            print(f"   失败数: {results.failure_count}")
            print(f"   处理时间: {results.total_time:.2f}秒")
            print(f"   平均时间: {results.avg_time_per_item:.3f}秒/项")
            
            print("✅ 批量处理测试成功")
            
        except Exception as e:
            print(f"❌ 批量处理测试失败: {e}")
    
    def test_scalable_architecture(self):
        """测试可扩展架构功能"""
        print("\n🔧 测试4: 可扩展架构")
        print("-" * 40)
        
        if not self.framework:
            print("❌ 模块化框架未初始化")
            return
        
        try:
            # 创建测试插件
            class TestPlugin(BasePlugin):
                def get_info(self) -> PluginInfo:
                    return PluginInfo(
                        plugin_id="test_plugin",
                        name="测试插件",
                        version="1.0.0",
                        description="用于演示的测试插件",
                        module_type=ModuleType.PROCESSOR
                    )
                
                def process(self, input_data: Any, config: Dict[str, Any] = None) -> Any:
                    return {
                        "input": input_data,
                        "output": f"处理结果: {input_data}",
                        "plugin": "test_plugin"
                    }
            
            # 注册插件
            print("🔌 注册测试插件...")
            self.framework.register_processor(TestPlugin)
            
            # 获取已注册的插件
            plugins = self.framework.list_processors()
            print(f"📋 已注册插件数: {len(plugins)}")
            
            for plugin_id, plugin_info in plugins.items():
                print(f"  - {plugin_info.name} (v{plugin_info.version})")
            
            # 测试处理管道
            print("🔄 测试处理管道...")
            pipeline = ["test_plugin"]
            test_input = "测试数据"
            
            result = self.framework.run_pipeline(pipeline, test_input)
            print(f"   输入: {test_input}")
            print(f"   输出: {result}")
            
            print("✅ 可扩展架构测试成功")
            
        except Exception as e:
            print(f"❌ 可扩展架构测试失败: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n🎯 开始全面测试四个优化方向")
        print("=" * 60)
        
        start_time = time.time()
        
        # 运行各项测试
        self.test_dynamic_dataset_loading()
        self.test_intelligent_classification()
        self.test_batch_processing()
        self.test_scalable_architecture()
        
        end_time = time.time()
        
        # 总结
        print(f"\n📊 测试完成总结")
        print("=" * 60)
        print(f"⏱️  总测试时间: {end_time - start_time:.2f}秒")
        
        # 检查各个模块状态
        modules_status = {
            "🚀 动态数据集管理器": "✅" if self.dataset_manager else "❌",
            "🧠 智能分类器": "✅" if self.classifier else "❌",
            "📊 批量处理器": "✅" if self.batch_processor else "❌",
            "🔧 模块化框架": "✅" if self.framework else "❌"
        }
        
        print("📋 模块状态:")
        for module, status in modules_status.items():
            print(f"   {module}: {status}")
        
        success_count = sum(1 for status in modules_status.values() if status == "✅")
        print(f"\n🎉 成功率: {success_count}/{len(modules_status)} ({success_count/len(modules_status)*100:.1f}%)")


def main():
    """主函数"""
    demo = StandaloneOptimizationDemo()
    demo.run_all_tests()


if __name__ == "__main__":
    main() 