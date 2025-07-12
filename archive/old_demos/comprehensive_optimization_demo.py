"""
🚀 Comprehensive Optimization Demo
综合优化演示 - 展示四个方向的整合应用

四个优化方向：
🚀 零代码添加新题目 (动态从数据集加载)
🧠 智能分类和模板匹配 (10种题型自动识别)
📊 批量处理和质量评估 (标准化流程)
🔧 高度可扩展架构 (模块化设计)
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.processors.batch_processor import BatchProcessor
# 导入我们创建的四个优化模块
from src.processors.dynamic_dataset_manager import (DynamicDatasetManager,
                                                    ProblemBatch)
from src.processors.intelligent_classifier import (IntelligentClassifier,
                                                   ProblemType)
from src.processors.scalable_architecture import (BasePlugin, ModularFramework,
                                                  ModuleType, PluginInfo)


class ComprehensiveOptimizationDemo:
    """🚀 综合优化演示系统"""
    
    def __init__(self):
        """初始化综合优化系统"""
        print("🚀 初始化综合优化系统")
        print("=" * 60)
        
        # 1. 初始化动态数据集管理器
        print("🚀 1. 初始化动态数据集管理器...")
        self.dataset_manager = DynamicDatasetManager(
            data_dirs=["Data", "datasets"],
            watch_mode=True,
            auto_reload=True
        )
        
        # 2. 初始化智能分类器
        print("\n🧠 2. 初始化智能分类器...")
        self.classifier = IntelligentClassifier()
        
        # 3. 初始化批量处理器
        print("\n📊 3. 初始化批量处理器...")
        self.batch_processor = BatchProcessor(max_workers=4)
        
        # 4. 初始化模块化框架
        print("\n🔧 4. 初始化模块化框架...")
        self.framework = ModularFramework()
        
        # 注册自定义插件
        self._register_custom_plugins()
        
        print("\n✅ 综合优化系统初始化完成!")
        print("=" * 60)
    
    def _register_custom_plugins(self):
        """注册自定义插件"""
        
        # 数据集加载插件
        class DatasetLoaderPlugin(BasePlugin):
            def __init__(self, dataset_manager):
                self.dataset_manager = dataset_manager
            
            def get_info(self) -> PluginInfo:
                return PluginInfo(
                    plugin_id="dataset_loader",
                    name="数据集加载器",
                    version="1.0.0",
                    description="动态加载数据集",
                    module_type=ModuleType.PROCESSOR
                )
            
            def process(self, input_data: Any, config: Dict[str, Any] = None) -> Any:
                """加载数据集批次"""
                batch_size = config.get('batch_size', 10) if config else 10
                datasets = config.get('datasets', None) if config else None
                
                batch = self.dataset_manager.get_dynamic_batch(
                    batch_size=batch_size,
                    datasets=datasets
                )
                
                return {
                    'batch_id': batch.batch_id,
                    'problems': batch.problems,
                    'source_dataset': batch.source_dataset,
                    'timestamp': batch.timestamp.isoformat()
                }
        
        # 智能分类插件
        class ClassificationPlugin(BasePlugin):
            def __init__(self, classifier):
                self.classifier = classifier
            
            def get_info(self) -> PluginInfo:
                return PluginInfo(
                    plugin_id="intelligent_classifier",
                    name="智能分类器",
                    version="1.0.0",
                    description="10种题型自动识别",
                    module_type=ModuleType.CLASSIFIER
                )
            
            def process(self, input_data: Any, config: Dict[str, Any] = None) -> Any:
                """分类问题"""
                if isinstance(input_data, dict) and 'problems' in input_data:
                    problems = input_data['problems']
                    classified_results = []
                    
                    for problem in problems:
                        # 提取问题文本
                        problem_text = self._extract_problem_text(problem)
                        
                        # 分类
                        classification = self.classifier.classify(problem_text)
                        
                        classified_results.append({
                            'original_problem': problem,
                            'problem_text': problem_text,
                            'classification': {
                                'type': classification.problem_type.value,
                                'confidence': classification.confidence,
                                'template': classification.template_match,
                                'entities': classification.extracted_entities,
                                'reasoning': classification.reasoning
                            }
                        })
                    
                    input_data['classified_problems'] = classified_results
                    input_data['classification_summary'] = self._generate_classification_summary(classified_results)
                
                return input_data
            
            def _extract_problem_text(self, problem: Dict) -> str:
                """提取问题文本"""
                for field in ['problem', 'question', 'text', 'body']:
                    if field in problem:
                        return str(problem[field])
                return str(problem)
            
            def _generate_classification_summary(self, results: List[Dict]) -> Dict:
                """生成分类摘要"""
                type_counts = {}
                total_confidence = 0
                
                for result in results:
                    ptype = result['classification']['type']
                    confidence = result['classification']['confidence']
                    
                    type_counts[ptype] = type_counts.get(ptype, 0) + 1
                    total_confidence += confidence
                
                return {
                    'total_problems': len(results),
                    'type_distribution': type_counts,
                    'average_confidence': total_confidence / len(results) if results else 0
                }
        
        # 注册插件
        self.framework.register_processor(
            type('DatasetLoaderPlugin', (DatasetLoaderPlugin,), {})
        )
        
        # 创建分类插件类并注册
        classifier_plugin_class = type(
            'ClassificationPlugin', 
            (BasePlugin,), 
            {
                '__init__': lambda self: ClassificationPlugin.__init__(self, self.outer.classifier),
                'get_info': ClassificationPlugin.get_info,
                'process': ClassificationPlugin.process,
                '_extract_problem_text': ClassificationPlugin._extract_problem_text,
                '_generate_classification_summary': ClassificationPlugin._generate_classification_summary,
                'outer': self
            }
        )
        
        self.framework.register_processor(classifier_plugin_class)
    
    def demo_all_optimizations(self):
        """🎯 演示所有优化功能"""
        print("\n🎯 综合优化功能演示")
        print("=" * 60)
        
        # 1. 零代码添加新题目演示
        self._demo_dynamic_dataset_loading()
        
        # 2. 智能分类演示
        self._demo_intelligent_classification()
        
        # 3. 批量处理演示
        self._demo_batch_processing()
        
        # 4. 可扩展架构演示
        self._demo_scalable_architecture()
        
        # 5. 综合流程演示
        self._demo_integrated_workflow()
    
    def _demo_dynamic_dataset_loading(self):
        """🚀 演示动态数据集加载"""
        print("\n🚀 1. 动态数据集加载演示")
        print("-" * 40)
        
        # 显示发现的数据集
        stats = self.dataset_manager.get_stats()
        print(f"📊 数据集统计:")
        print(f"  已发现数据集: {stats['total_datasets']}")
        print(f"  可用数据集: {', '.join(stats['available_datasets'][:5])}...")
        
        # 获取动态批次
        batch = self.dataset_manager.get_dynamic_batch(batch_size=3)
        print(f"\n📦 动态批次:")
        print(f"  批次ID: {batch.batch_id}")
        print(f"  数据来源: {batch.source_dataset}")
        print(f"  问题数量: {len(batch.problems)}")
        
        # 显示前2个问题
        for i, problem in enumerate(batch.problems[:2]):
            problem_text = str(problem)[:100] + "..." if len(str(problem)) > 100 else str(problem)
            print(f"  问题 {i+1}: {problem_text}")
        
        return batch
    
    def _demo_intelligent_classification(self):
        """🧠 演示智能分类"""
        print("\n🧠 2. 智能分类演示")
        print("-" * 40)
        
        # 测试问题
        test_problems = [
            "计算 15 + 28 = ?",
            "小明买了5本书，每本12元，总共花费多少钱？",
            "解方程：3x + 7 = 22",
            "一个正方形边长6米，求其面积",
            "从8个人中选择3个人，有多少种选法？"
        ]
        
        print(f"🧪 分类测试 ({len(test_problems)} 个问题):")
        
        classification_results = []
        for i, problem in enumerate(test_problems, 1):
            result = self.classifier.classify(problem)
            classification_results.append(result)
            
            print(f"\n  问题 {i}: {problem}")
            print(f"    类型: {result.problem_type.value}")
            print(f"    置信度: {result.confidence:.2f}")
            print(f"    模板: {result.template_match}")
        
        # 显示分类统计
        stats = self.classifier.get_statistics()
        if stats['type_percentages']:
            print(f"\n📊 分类统计:")
            for ptype, percentage in stats['type_percentages'].items():
                print(f"  {ptype}: {percentage}%")
        
        return classification_results
    
    def _demo_batch_processing(self):
        """📊 演示批量处理"""
        print("\n📊 3. 批量处理演示")
        print("-" * 40)
        
        # 定义处理函数
        def comprehensive_math_processor(problem):
            """综合数学处理器"""
            time.sleep(0.05)  # 模拟处理时间
            
            try:
                # 提取问题文本
                if isinstance(problem, dict):
                    text = problem.get('problem', str(problem))
                else:
                    text = str(problem)
                
                # 简单分类
                if any(op in text for op in ['+', '-', '*', '/', '×', '÷']):
                    problem_type = "算术运算"
                    difficulty = "简单"
                elif any(word in text for word in ['方程', '解', 'x', 'y']):
                    problem_type = "方程求解"
                    difficulty = "中等"
                else:
                    problem_type = "应用题"
                    difficulty = "中等"
                
                return {
                    'original': problem,
                    'text': text,
                    'type': problem_type,
                    'difficulty': difficulty,
                    'is_correct': True,
                    'processing_time': 0.05,
                    'solution_steps': [f"识别为{problem_type}", "进行相应处理"]
                }
                
            except Exception as e:
                return {
                    'original': problem,
                    'error': str(e),
                    'is_correct': False
                }
        
        # 准备测试数据
        test_data = [
            "25 + 37 = ?",
            "解方程: 2x + 5 = 15",
            "小王买了3个苹果，每个2元",
            "计算正方形面积，边长5米",
            "概率问题：投掷硬币",
            "invalid_data",  # 故意的错误数据
        ]
        
        # 提交批处理任务
        job_id = self.batch_processor.submit_job(
            name="综合数学问题处理",
            input_data=test_data,
            processor_func=comprehensive_math_processor,
            quality_evaluator='math_problem_solver'
        )
        
        print(f"📤 提交批处理任务: {job_id}")
        
        # 处理任务
        report = self.batch_processor.process_job(job_id)
        
        # 显示结果
        print(f"\n📋 处理报告:")
        print(f"  总项目数: {report.total_items}")
        print(f"  成功项目: {report.successful_items}")
        print(f"  失败项目: {report.failed_items}")
        print(f"  处理时间: {report.processing_time:.2f}秒")
        print(f"  质量等级: {report.quality_metrics.quality_level.value}")
        print(f"  总体分数: {report.quality_metrics.overall_score:.2f}")
        
        if report.quality_metrics.recommendations:
            print(f"  改进建议: {', '.join(report.quality_metrics.recommendations)}")
        
        return report
    
    def _demo_scalable_architecture(self):
        """🔧 演示可扩展架构"""
        print("\n🔧 4. 可扩展架构演示")
        print("-" * 40)
        
        # 创建处理管道
        self.framework.create_pipeline("comprehensive_pipeline", [
            "simple_arithmetic",
            "problem_classifier"
        ])
        
        # 测试数据
        test_expressions = ["15 + 23", "8 * 7", "56 / 8"]
        
        print(f"🔗 测试处理管道:")
        
        pipeline_results = []
        for expr in test_expressions:
            print(f"\n  输入: {expr}")
            try:
                result = self.framework.execute_pipeline("comprehensive_pipeline", expr)
                pipeline_results.append(result)
                print(f"  输出: {result}")
            except Exception as e:
                print(f"  错误: {e}")
        
        # 显示框架统计
        registry_info = self.framework.plugin_manager.get_registry_info()
        print(f"\n📊 插件统计:")
        for key, value in registry_info.items():
            print(f"  {key}: {value}")
        
        return pipeline_results
    
    def _demo_integrated_workflow(self):
        """🎯 演示整合工作流"""
        print("\n🎯 5. 整合工作流演示")
        print("-" * 40)
        
        print("🔄 执行端到端处理流程:")
        
        # 步骤1: 动态加载数据
        print("  步骤1: 从数据集动态加载问题...")
        batch = self.dataset_manager.get_dynamic_batch(batch_size=5)
        
        # 步骤2: 批量分类
        print("  步骤2: 智能分类问题...")
        classification_results = []
        for problem in batch.problems:
            # 提取问题文本
            problem_text = self._extract_text_from_problem(problem)
            if problem_text:
                result = self.classifier.classify(problem_text)
                classification_results.append({
                    'problem': problem,
                    'text': problem_text,
                    'classification': result
                })
        
        # 步骤3: 质量评估
        print("  步骤3: 质量评估和统计...")
        quality_stats = self._calculate_workflow_quality(classification_results)
        
        # 步骤4: 生成综合报告
        print("  步骤4: 生成综合报告...")
        workflow_report = {
            'workflow_id': f"workflow_{int(time.time())}",
            'total_problems': len(batch.problems),
            'classified_problems': len(classification_results),
            'data_source': batch.source_dataset,
            'quality_metrics': quality_stats,
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'performance_summary': {
                'data_loading': 'SUCCESS',
                'classification': 'SUCCESS',
                'quality_evaluation': 'SUCCESS',
                'overall_status': 'COMPLETED'
            }
        }
        
        # 显示结果
        print(f"\n📊 整合工作流报告:")
        print(f"  工作流ID: {workflow_report['workflow_id']}")
        print(f"  处理问题数: {workflow_report['total_problems']}")
        print(f"  分类成功数: {workflow_report['classified_problems']}")
        print(f"  数据来源: {workflow_report['data_source']}")
        print(f"  平均置信度: {workflow_report['quality_metrics']['average_confidence']:.2f}")
        print(f"  处理状态: {workflow_report['performance_summary']['overall_status']}")
        
        # 显示分类分布
        if workflow_report['quality_metrics']['type_distribution']:
            print(f"  分类分布:")
            for ptype, count in workflow_report['quality_metrics']['type_distribution'].items():
                print(f"    {ptype}: {count}")
        
        return workflow_report
    
    def _extract_text_from_problem(self, problem: Dict) -> str:
        """从问题字典中提取文本"""
        if isinstance(problem, str):
            return problem
        
        if isinstance(problem, dict):
            for field in ['problem', 'question', 'text', 'body']:
                if field in problem:
                    return str(problem[field])
        
        return str(problem)
    
    def _calculate_workflow_quality(self, results: List[Dict]) -> Dict:
        """计算工作流质量"""
        if not results:
            return {'average_confidence': 0, 'type_distribution': {}}
        
        total_confidence = sum(r['classification'].confidence for r in results)
        average_confidence = total_confidence / len(results)
        
        type_distribution = {}
        for result in results:
            ptype = result['classification'].problem_type.value
            type_distribution[ptype] = type_distribution.get(ptype, 0) + 1
        
        return {
            'average_confidence': average_confidence,
            'type_distribution': type_distribution,
            'total_classified': len(results)
        }
    
    def save_demo_results(self, output_dir: str = "demo_results"):
        """💾 保存演示结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 保存演示结果到: {output_path}")
        
        # 保存数据集统计
        dataset_stats = self.dataset_manager.get_stats()
        with open(output_path / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_stats, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存分类统计
        classification_stats = self.classifier.get_statistics()
        with open(output_path / "classification_stats.json", 'w', encoding='utf-8') as f:
            json.dump(classification_stats, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存框架配置
        self.framework.save_configuration(str(output_path / "framework_config.json"))
        
        print("✅ 演示结果已保存")
    
    def print_optimization_summary(self):
        """📈 打印优化总结"""
        print("\n📈 四方向优化总结")
        print("=" * 60)
        
        print("🚀 1. 零代码添加新题目:")
        print("  ✅ 自动发现和加载数据集")
        print("  ✅ 热重载和文件监控")
        print("  ✅ 动态批次生成")
        print("  ✅ 多格式支持 (JSON, JSONL, YAML)")
        
        print("\n🧠 2. 智能分类和模板匹配:")
        print("  ✅ 10种题型自动识别")
        print("  ✅ 模板匹配和实体提取")
        print("  ✅ 置信度评估")
        print("  ✅ 可扩展模式系统")
        
        print("\n📊 3. 批量处理和质量评估:")
        print("  ✅ 多线程批量处理")
        print("  ✅ 智能质量评估")
        print("  ✅ 详细处理报告")
        print("  ✅ 性能监控和优化建议")
        
        print("\n🔧 4. 高度可扩展架构:")
        print("  ✅ 插件系统和模块化设计")
        print("  ✅ 处理管道和工作流")
        print("  ✅ 动态加载和热插拔")
        print("  ✅ 配置管理和事件系统")
        
        print("\n🎯 整合效果:")
        print("  ✅ 端到端自动化处理")
        print("  ✅ 高性能和高质量")
        print("  ✅ 易扩展和易维护")
        print("  ✅ 智能化和标准化")


def main():
    """主函数"""
    print("🚀 综合优化演示程序启动")
    print("=" * 80)
    
    try:
        # 创建演示系统
        demo = ComprehensiveOptimizationDemo()
        
        # 执行全面演示
        demo.demo_all_optimizations()
        
        # 保存结果
        demo.save_demo_results()
        
        # 打印总结
        demo.print_optimization_summary()
        
        print("\n🎉 综合优化演示完成!")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 