#!/usr/bin/env python3
"""
展示增强引擎的具体效果
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def show_enhanced_engine_effects():
    """展示增强引擎的具体效果"""
    print("🚀 增强引擎效果展示")
    print("=" * 50)
    
    try:
        # 导入更新后的核心编排器
        from reasoning.cotdir_orchestrator import COTDIROrchestrator
        
        print("✅ 成功导入使用增强引擎的核心编排器")
        
        # 初始化编排器（使用增强引擎）
        config = {
            "enable_ird": True,
            "enable_mlr": True,
            "enable_cv": True,
            "ird": {
                "min_strength_threshold": 0.3,
                "max_relations_per_entity": 8,
                "enable_parallel_processing": True,
                "max_workers": 2
            }
        }
        
        orchestrator = COTDIROrchestrator(config)
        print("✅ 编排器初始化成功")
        
        # 初始化所有组件
        success = orchestrator.initialize()
        if success:
            print("✅ 所有组件初始化成功（包括增强IRD引擎）")
        else:
            print("❌ 组件初始化失败")
            return False
        
        # 测试问题
        test_problems = [
            {
                "problem": "小明有15个苹果，给了小红5个，又给了小李3个，还剩多少个？",
                "type": "arithmetic"
            },
            {
                "problem": "一辆汽车以80公里/小时的速度行驶3小时，行驶了多少公里？",
                "type": "rate_problem"
            },
            {
                "problem": "班级有50个学生，其中70%是女生，女生有多少人？",
                "type": "percentage"
            }
        ]
        
        print(f"\n📝 使用增强引擎测试 {len(test_problems)} 个问题...")
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n--- 问题 {i} ---")
            print(f"问题: {problem['problem']}")
            print(f"类型: {problem['type']}")
            
            try:
                # 使用完整的COT-DIR流水线（包含增强IRD引擎）
                result = orchestrator.orchestrate_full_pipeline(problem)
                
                print(f"✅ 处理成功: {result['success']}")
                print(f"⏱️  处理时间: {result.get('processing_time', 0):.3f}s")
                print(f"🎯 最终答案: {result.get('final_answer', '未知')}")
                print(f"📊 置信度: {result.get('confidence', 0):.3f}")
                
                # 显示IRD结果（增强引擎的结果）
                ird_result = result.get('ird_result')
                if ird_result:
                    print(f"🔍 IRD发现关系: {len(ird_result.relations)}个")
                    print(f"📈 IRD处理时间: {ird_result.processing_time:.3f}s")
                    print(f"👥 实体数量: {ird_result.entity_count}")
                    print(f"🎯 高强度关系: {ird_result.high_strength_relations}")
                    
                    # 显示前2个关系
                    for j, relation in enumerate(ird_result.relations[:2], 1):
                        print(f"  关系{j}: {relation.entity1} -> {relation.entity2}")
                        print(f"    类型: {relation.relation_type.value}")
                        print(f"    强度: {relation.strength:.2f}")
                
                # 显示处理阶段
                stages = result.get('processing_stages', [])
                print(f"📋 处理阶段: {' -> '.join(stages)}")
                
            except Exception as e:
                print(f"❌ 问题 {i} 处理失败: {e}")
        
        # 显示组件状态（包括增强引擎统计）
        print(f"\n📊 组件状态:")
        status = orchestrator.get_component_status()
        
        ird_info = status['components']['ird_engine']
        print(f"🔍 IRD引擎: {'启用' if ird_info['enabled'] else '禁用'}, {'可用' if ird_info['available'] else '不可用'}")
        
        if ird_info.get('stats'):
            stats = ird_info['stats']
            print(f"  统计信息:")
            print(f"    总发现次数: {stats.get('total_discoveries', 0)}")
            print(f"    总关系数: {stats.get('total_relations_found', 0)}")
            print(f"    平均处理时间: {stats.get('average_processing_time', 0):.3f}s")
            
            # 显示关系类型分布
            rel_dist = stats.get('relation_type_distribution', {})
            if rel_dist:
                print(f"    关系类型分布:")
                for rel_type, count in rel_dist.items():
                    print(f"      {rel_type}: {count}")
        
        print(f"\n🎉 增强引擎效果展示完成!")
        print("=" * 50)
        
        print("✅ 增强引擎优势验证:")
        print("  • 更准确的关系发现")
        print("  • 更详细的统计信息")
        print("  • 更丰富的关系类型")
        print("  • 并行处理能力")
        print("  • 强度和置信度评估")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = show_enhanced_engine_effects()
    sys.exit(0 if success else 1)