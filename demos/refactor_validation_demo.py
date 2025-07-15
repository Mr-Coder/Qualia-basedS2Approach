"""
重构验证演示

验证新架构的功能和性能，展示IRD+MLR+CV集成效果。
"""

import asyncio
import logging
import time
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_ird_engine():
    """测试IRD引擎"""
    print("\n🔍 测试隐式关系发现引擎 (IRD)")
    print("=" * 50)
    
    try:
        import sys
        sys.path.append('/Users/menghao/Documents/GitHub/cot-dir1/src')
        
        from reasoning.qs2_enhancement.enhanced_ird_engine import EnhancedIRDEngine
        
        # 初始化IRD引擎
        ird_config = {
            "confidence_threshold": 0.6,
            "max_relations": 5,
            "enable_advanced_patterns": True
        }
        
        ird_engine = EnhancedIRDEngine(ird_config)
        
        # 测试问题
        test_problems = [
            "小明有10个苹果，给了小红3个，还剩多少个？",
            "一辆汽车以60公里/小时的速度行驶2小时，行驶了多少公里？",
            "班级有40个学生，其中60%是男生，男生有多少人？"
        ]
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n问题 {i}: {problem}")
            
            start_time = time.time()
            result = ird_engine.discover_relations(problem)
            end_time = time.time()
            
            print(f"  处理时间: {end_time - start_time:.3f}秒")
            print(f"  发现关系: {len(result.relations)}个")
            print(f"  置信度: {result.statistics.get('average_confidence', 0.0):.3f}")
            
            for j, relation in enumerate(result.relations, 1):
                print(f"    关系{j}: {relation.description} (置信度: {relation.confidence:.2f})")
        
        # 显示统计信息
        stats = ird_engine.get_global_stats()
        print(f"\nIRD引擎统计:")
        print(f"  总处理数: {stats['total_discoveries']}")
        print(f"  总关系数: {stats['total_relations_found']}")
        print(f"  平均置信度: {stats['average_processing_time']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ IRD引擎测试失败: {str(e)}")
        return False


def test_mlr_processor():
    """测试MLR处理器"""
    print("\n🧠 测试多层级推理处理器 (MLR)")
    print("=" * 50)
    
    try:
        from src.reasoning.qs2_enhancement.enhanced_ird_engine import EnhancedIRDEngine
        from src.reasoning.private.mlr_processor import MultiLevelReasoningProcessor
        
        # 初始化组件
        ird_engine = EnhancedIRDEngine()
        mlr_processor = MultiLevelReasoningProcessor({
            "max_reasoning_depth": 8,
            "confidence_threshold": 0.6
        })
        
        # 测试问题
        problem = "小红买了3支笔，每支5元，又买了2本书，每本12元，总共花了多少钱？"
        
        print(f"问题: {problem}")
        
        # 第一步：IRD
        ird_result = ird_engine.discover_relations(problem)
        print(f"发现 {len(ird_result.relations)} 个隐式关系")
        
        # 第二步：MLR
        start_time = time.time()
        mlr_result = mlr_processor.execute_reasoning(problem, ird_result.relations)
        end_time = time.time()
        
        print(f"\nMLR结果:")
        print(f"  成功: {mlr_result.success}")
        print(f"  复杂度级别: {mlr_result.complexity_level.value}")
        print(f"  推理步骤: {len(mlr_result.reasoning_steps)}步")
        print(f"  最终答案: {mlr_result.final_answer}")
        print(f"  置信度: {mlr_result.confidence_score:.3f}")
        print(f"  处理时间: {end_time - start_time:.3f}秒")
        
        # 显示推理步骤
        print(f"\n推理步骤详情:")
        for i, step in enumerate(mlr_result.reasoning_steps, 1):
            print(f"  步骤{i}: {step.description}")
            print(f"    操作: {step.operation}")
            if step.output_value is not None:
                print(f"    输出: {step.output_value}")
        
        # 显示统计信息
        stats = mlr_processor.get_stats()
        print(f"\nMLR处理器统计:")
        print(f"  总处理数: {stats['total_processed']}")
        print(f"  成功率: {stats['success_rate']:.3f}")
        print(f"  平均步骤数: {stats['average_steps']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ MLR处理器测试失败: {str(e)}")
        return False


def test_cv_validator():
    """测试CV验证器"""
    print("\n✅ 测试链式验证器 (CV)")
    print("=" * 50)
    
    try:
        from src.reasoning.qs2_enhancement.enhanced_ird_engine import EnhancedIRDEngine
        from src.reasoning.private.mlr_processor import MultiLevelReasoningProcessor
        from src.reasoning.private.cv_validator import ChainVerificationValidator
        
        # 初始化组件
        ird_engine = EnhancedIRDEngine()
        mlr_processor = MultiLevelReasoningProcessor()
        cv_validator = ChainVerificationValidator({
            "validation_level": "comprehensive",
            "enable_auto_correction": True
        })
        
        # 测试问题
        problem = "张老师买了4盒粉笔，每盒12支，一共买了多少支粉笔？"
        
        print(f"问题: {problem}")
        
        # 执行完整流程
        ird_result = ird_engine.discover_relations(problem)
        mlr_result = mlr_processor.execute_reasoning(problem, ird_result.relations)
        
        # 验证推理链
        start_time = time.time()
        cv_result = cv_validator.verify_reasoning_chain(
            mlr_result.reasoning_steps,
            {"problem_text": problem}
        )
        end_time = time.time()
        
        print(f"\nCV验证结果:")
        print(f"  验证通过: {cv_result.is_valid}")
        print(f"  一致性分数: {cv_result.consistency_score:.3f}")
        print(f"  发现错误: {len(cv_result.errors)}个")
        print(f"  警告: {len(cv_result.warnings)}个")
        print(f"  建议: {len(cv_result.suggestions)}个")
        print(f"  验证时间: {end_time - start_time:.3f}秒")
        
        # 显示错误和建议
        if cv_result.errors:
            print(f"\n发现的错误:")
            for i, error in enumerate(cv_result.errors, 1):
                print(f"  错误{i}: {error.description} (严重程度: {error.severity:.2f})")
        
        if cv_result.suggestions:
            print(f"\n验证建议:")
            for i, suggestion in enumerate(cv_result.suggestions, 1):
                print(f"  建议{i}: {suggestion}")
        
        # 显示统计信息
        stats = cv_validator.get_stats()
        print(f"\nCV验证器统计:")
        print(f"  总验证数: {stats['total_validations']}")
        print(f"  有效链数: {stats['valid_chains']}")
        print(f"  验证成功率: {stats['valid_chains'] / max(1, stats['total_validations']):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ CV验证器测试失败: {str(e)}")
        return False


def test_integrated_reasoning_api():
    """测试集成推理API"""
    print("\n🚀 测试集成推理API (IRD+MLR+CV)")
    print("=" * 50)
    
    try:
        from src.reasoning.public_api_refactored import ReasoningAPI
        
        # 初始化API
        reasoning_config = {
            "ird": {
                "confidence_threshold": 0.6,
                "enable_advanced_patterns": True
            },
            "mlr": {
                "max_reasoning_depth": 8,
                "confidence_threshold": 0.7
            },
            "cv": {
                "validation_level": "intermediate",
                "enable_auto_correction": True
            }
        }
        
        reasoning_api = ReasoningAPI(reasoning_config)
        reasoning_api.initialize()
        
        # 测试问题集
        test_problems = [
            {
                "problem": "学校买了5箱练习本，每箱24本，一共买了多少本？",
                "type": "arithmetic"
            },
            {
                "problem": "一个班有45个学生，其中40%是女生，女生有多少人？",
                "type": "proportion"
            },
            {
                "problem": "小王每天跑步30分钟，一周跑步多少小时？",
                "type": "time_calculation"
            }
        ]
        
        for i, problem_data in enumerate(test_problems, 1):
            print(f"\n问题 {i}: {problem_data['problem']}")
            print(f"类型: {problem_data['type']}")
            
            start_time = time.time()
            result = reasoning_api.solve_problem(problem_data)
            end_time = time.time()
            
            print(f"  最终答案: {result.get('final_answer')}")
            print(f"  置信度: {result.get('confidence', 0):.3f}")
            print(f"  成功: {result.get('success', False)}")
            print(f"  复杂度: {result.get('complexity_level', 'unknown')}")
            print(f"  发现关系: {len(result.get('relations_found', []))}个")
            print(f"  推理步骤: {len(result.get('reasoning_steps', []))}步")
            print(f"  处理时间: {end_time - start_time:.3f}秒")
            
            # 验证结果
            validation = result.get('validation_result', {})
            print(f"  验证通过: {validation.get('is_valid', False)}")
            if validation.get('errors'):
                print(f"  验证错误: {len(validation['errors'])}个")
        
        # 批量测试
        print(f"\n批量处理测试:")
        start_time = time.time()
        batch_results = reasoning_api.batch_solve(test_problems)
        end_time = time.time()
        
        successful = sum(1 for r in batch_results if r.get('success', False))
        print(f"  批量处理: {len(batch_results)}个问题")
        print(f"  成功数量: {successful}")
        print(f"  成功率: {successful / len(batch_results):.3f}")
        print(f"  总时间: {end_time - start_time:.3f}秒")
        print(f"  平均时间: {(end_time - start_time) / len(batch_results):.3f}秒/问题")
        
        # 显示API统计
        stats = reasoning_api.get_statistics()
        print(f"\n推理API统计:")
        print(f"  总请求: {stats.get('total_problems', 0)}")
        print(f"  成功请求: {stats.get('successful_problems', 0)}")
        print(f"  成功率: {stats.get('success_rate', 0):.3f}")
        print(f"  平均处理时间: {stats.get('average_processing_time', 0):.3f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成推理API测试失败: {str(e)}")
        return False


def test_model_management():
    """测试模型管理"""
    print("\n🤖 测试模型管理系统")
    print("=" * 50)
    
    try:
        from src.models.private.model_factory import ModelFactory
        from src.models.private.model_cache import ModelCacheManager
        from src.models.private.performance_tracker import PerformanceMonitor
        
        # 测试模型工厂
        print("测试模型工厂:")
        factory = ModelFactory()
        
        available_models = factory.get_available_models()
        print(f"  可用模型: {len(available_models)}个")
        for name, info in available_models.items():
            print(f"    {name}: {info['type']}")
        
        # 测试缓存管理器
        print("\n测试缓存管理器:")
        cache_config = {
            "max_size": 100,
            "max_memory_mb": 64,
            "default_ttl": 300
        }
        cache_manager = ModelCacheManager(cache_config)
        
        # 测试缓存操作
        test_problem = {"problem": "1+1=?"}
        test_result = {"final_answer": "2", "confidence": 0.9}
        
        # 缓存结果
        cache_key = cache_manager.get_problem_hash(test_problem, "test_model")
        cache_manager.put(cache_key, test_result)
        
        # 获取缓存
        cached_result = cache_manager.get(cache_key)
        print(f"  缓存测试: {'成功' if cached_result else '失败'}")
        
        cache_stats = cache_manager.get_cache_stats()
        print(f"  缓存条目: {cache_stats['current_size']}")
        print(f"  内存使用: {cache_stats['current_memory_mb']:.2f}MB")
        
        # 测试性能监控器
        print("\n测试性能监控器:")
        monitor = PerformanceMonitor()
        
        # 模拟一些性能数据
        monitor.monitor_model_call(
            "test_model", "solve", time.time(), time.time() + 0.1, True, 100, 50
        )
        
        system_overview = monitor.get_system_overview()
        print(f"  监控启用: {system_overview.get('monitoring_enabled', False)}")
        print(f"  总操作数: {system_overview.get('total_operations', 0)}")
        print(f"  错误率: {system_overview.get('error_rate', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型管理测试失败: {str(e)}")
        return False


async def test_enhanced_orchestrator():
    """测试增强系统协调器"""
    print("\n🎼 测试增强系统协调器")
    print("=" * 50)
    
    try:
        from src.core.enhanced_system_orchestrator import EnhancedSystemOrchestrator
        
        # 初始化协调器
        orchestrator_config = {
            "max_workers": 4,
            "batch_max_concurrent": 3
        }
        
        orchestrator = EnhancedSystemOrchestrator(orchestrator_config)
        
        # 测试单个问题求解（异步）
        print("测试异步问题求解:")
        test_problem = {
            "problem": "图书馆有书架5层，每层放书80本，一共可以放多少本书？"
        }
        
        start_time = time.time()
        try:
            result = await orchestrator.solve_math_problem_async(test_problem)
            end_time = time.time()
            
            print(f"  最终答案: {result.get('final_answer')}")
            print(f"  置信度: {result.get('confidence', 0):.3f}")
            print(f"  成功: {result.get('success', False)}")
            print(f"  处理时间: {end_time - start_time:.3f}秒")
        except Exception as e:
            print(f"  异步求解失败: {str(e)}")
        
        # 测试批量问题求解（异步）
        print(f"\n测试异步批量求解:")
        batch_problems = [
            {"problem": "3 + 5 = ?"},
            {"problem": "10 - 4 = ?"},
            {"problem": "6 × 7 = ?"},
            {"problem": "24 ÷ 8 = ?"}
        ]
        
        start_time = time.time()
        try:
            batch_results = await orchestrator.batch_solve_problems_async(batch_problems, max_concurrent=2)
            end_time = time.time()
            
            successful = sum(1 for r in batch_results if r.get('success', False))
            print(f"  批量问题: {len(batch_problems)}个")
            print(f"  成功数量: {successful}")
            print(f"  总时间: {end_time - start_time:.3f}秒")
        except Exception as e:
            print(f"  异步批量求解失败: {str(e)}")
        
        # 显示系统状态
        print(f"\n系统状态:")
        status = orchestrator.get_system_status()
        print(f"  系统状态: {status.get('status')}")
        print(f"  协调器类型: {status.get('orchestrator_type')}")
        print(f"  并发能力: {status.get('concurrent_capacity')}个工作线程")
        
        # 显示性能报告
        perf_report = orchestrator.get_performance_report()
        orchestration_stats = perf_report.get('orchestration_stats', {})
        print(f"  总协调次数: {orchestration_stats.get('total_orchestrations', 0)}")
        print(f"  成功次数: {orchestration_stats.get('successful_orchestrations', 0)}")
        print(f"  平均处理时间: {orchestration_stats.get('average_orchestration_time', 0):.3f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强系统协调器测试失败: {str(e)}")
        return False


async def main():
    """主测试函数"""
    print("🧪 COT-DIR 重构验证演示")
    print("=" * 60)
    print("测试新架构的IRD+MLR+CV集成和模块化设计")
    print("=" * 60)
    
    test_results = []
    
    # 测试各个组件
    test_functions = [
        ("IRD引擎", test_ird_engine),
        ("MLR处理器", test_mlr_processor), 
        ("CV验证器", test_cv_validator),
        ("集成推理API", test_integrated_reasoning_api),
        ("模型管理", test_model_management)
    ]
    
    # 异步测试
    async_tests = [
        ("增强系统协调器", test_enhanced_orchestrator)
    ]
    
    # 执行同步测试
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'='*60}")
            result = test_func()
            test_results.append((test_name, result))
            if result:
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {str(e)}")
            test_results.append((test_name, False))
    
    # 执行异步测试
    for test_name, test_func in async_tests:
        try:
            print(f"\n{'='*60}")
            result = await test_func()
            test_results.append((test_name, result))
            if result:
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {str(e)}")
            test_results.append((test_name, False))
    
    # 测试总结
    print(f"\n{'='*60}")
    print("📊 测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {test_name}")
    
    print(f"\n总体结果: {passed}/{total} 个测试通过")
    success_rate = passed / total if total > 0 else 0
    print(f"成功率: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("🎉 重构验证成功！新架构运行良好。")
    elif success_rate >= 0.6:
        print("⚠️ 重构基本成功，但还有改进空间。")
    else:
        print("🔧 重构需要进一步调试和完善。")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())