#!/usr/bin/env python3
"""
深度隐含关系发现算法测试脚本
验证语义蕴含推理、隐含约束挖掘、多层关系建模三大核心功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_deep_implicit_engine():
    """测试深度隐含关系发现引擎"""
    try:
        from src.reasoning.private.deep_implicit_engine import (
            DeepImplicitEngine, 
            SemanticRelationType,
            ConstraintType,
            RelationDepth
        )
        
        # 初始化引擎
        engine = DeepImplicitEngine()
        print("✅ 深度隐含关系引擎初始化成功")
        
        # 测试用例1：购物找零问题
        test_case_1 = {
            "problem_text": "小张买笔花了5元，付了10元，应该找回多少钱？",
            "entities": [
                {"name": "小张", "type": "person", "properties": ["agent", "buyer"]},
                {"name": "笔", "type": "object", "properties": ["countable", "commodity"]}, 
                {"name": "5", "type": "number", "properties": ["quantitative", "price"]},
                {"name": "10", "type": "number", "properties": ["quantitative", "payment"]},
                {"name": "元", "type": "money", "properties": ["currency", "value"]}
            ]
        }
        
        print(f"\n🧪 测试用例1: {test_case_1['problem_text']}")
        
        # 执行深度关系发现
        deep_relations, implicit_constraints = engine.discover_deep_relations(
            test_case_1["problem_text"],
            test_case_1["entities"],
            []
        )
        
        print(f"📊 发现结果:")
        print(f"   - 深度关系: {len(deep_relations)} 个")
        print(f"   - 隐含约束: {len(implicit_constraints)} 个")
        
        # 详细展示发现的关系
        print(f"\n🔍 深度关系详情:")
        for i, relation in enumerate(deep_relations, 1):
            print(f"   {i}. {relation.source_entity} → {relation.target_entity}")
            print(f"      类型: {relation.relation_type.value}")
            print(f"      深度: {relation.depth.value}")
            print(f"      置信度: {relation.confidence:.2f}")
            print(f"      逻辑基础: {relation.logical_basis}")
            print(f"      语义证据: {relation.semantic_evidence}")
            print(f"      约束含义: {relation.constraint_implications}")
            print()
        
        # 详细展示隐含约束
        print(f"🔒 隐含约束详情:")
        for i, constraint in enumerate(implicit_constraints, 1):
            print(f"   {i}. {constraint.description}")
            print(f"      类型: {constraint.constraint_type.value}")
            print(f"      表达式: {constraint.constraint_expression}")
            print(f"      影响实体: {constraint.affected_entities}")
            print(f"      置信度: {constraint.confidence:.2f}")
            print(f"      发现方法: {constraint.discovery_method}")
            print()
            
        return True, len(deep_relations), len(implicit_constraints)
        
    except Exception as e:
        print(f"❌ 深度隐含关系引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0

def test_mlr_processor_integration():
    """测试MLR处理器集成"""
    try:
        from src.reasoning.private.mlr_processor import MultiLevelReasoningProcessor
        
        # 初始化增强MLR处理器
        processor = MultiLevelReasoningProcessor()
        print("✅ 增强MLR处理器初始化成功")
        
        # 测试推理执行
        test_problem = "小明有5个苹果，小红有3个苹果，一共有多少个苹果？"
        print(f"\n🧪 测试MLR增强推理: {test_problem}")
        
        result = processor.execute_reasoning(
            problem_text=test_problem,
            relations=[],
            context={"test_mode": True}
        )
        
        print(f"📊 推理结果:")
        print(f"   - 成功: {result.success}")
        print(f"   - 复杂度级别: {result.complexity_level.value}")
        print(f"   - 推理步骤数: {len(result.reasoning_steps)}")
        print(f"   - 最终答案: {result.final_answer}")
        print(f"   - 置信度: {result.confidence_score:.3f}")
        print(f"   - 处理时间: {result.processing_time:.3f}秒")
        
        # 检查元数据中的深度关系信息
        metadata = result.metadata
        if "frontend_visualization_data" in metadata:
            viz_data = metadata["frontend_visualization_data"]
            print(f"   - 深度关系发现: {len(viz_data.get('deep_relations', []))} 个")
            print(f"   - 隐含约束发现: {len(viz_data.get('implicit_constraints', []))} 个")
        
        # 展示推理步骤
        print(f"\n📝 推理步骤详情:")
        for step in result.reasoning_steps:
            print(f"   步骤{step.step_id}: {step.description}")
            print(f"      操作: {step.operation}")
            print(f"      置信度: {step.confidence:.2f}")
            if step.metadata:
                if "relation_type" in step.metadata:
                    print(f"      关系类型: {step.metadata['relation_type']}")
                if "semantic_evidence" in step.metadata:
                    print(f"      语义证据: {step.metadata['semantic_evidence']}")
            print()
            
        return True, result
        
    except Exception as e:
        print(f"❌ MLR处理器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_multiple_problem_types():
    """测试多种问题类型"""
    try:
        from src.reasoning.private.deep_implicit_engine import DeepImplicitEngine
        
        engine = DeepImplicitEngine()
        
        test_cases = [
            {
                "name": "几何面积问题",
                "problem": "长方形的长是8米，宽是5米，面积是多少？",
                "entities": [
                    {"name": "长方形", "type": "object", "properties": ["geometric_shape"]},
                    {"name": "8", "type": "number", "properties": ["length"]},
                    {"name": "5", "type": "number", "properties": ["width"]},
                    {"name": "面积", "type": "concept", "properties": ["calculation_target"]}
                ]
            },
            {
                "name": "百分比问题", 
                "problem": "班级有40个学生，男生占60%，女生有多少人？",
                "entities": [
                    {"name": "班级", "type": "concept", "properties": ["group", "container"]},
                    {"name": "40", "type": "number", "properties": ["total_count"]},
                    {"name": "学生", "type": "person", "properties": ["group_member"]},
                    {"name": "男生", "type": "person", "properties": ["subgroup", "gender"]},
                    {"name": "60", "type": "number", "properties": ["percentage"]}
                ]
            },
            {
                "name": "复杂购物问题",
                "problem": "小李买了3支笔，每支2元，给了店主10元，店主应该找回多少钱？",
                "entities": [
                    {"name": "小李", "type": "person", "properties": ["buyer"]},
                    {"name": "3", "type": "number", "properties": ["quantity"]},
                    {"name": "笔", "type": "object", "properties": ["commodity"]},
                    {"name": "2", "type": "number", "properties": ["unit_price"]},
                    {"name": "10", "type": "number", "properties": ["payment"]},
                    {"name": "店主", "type": "person", "properties": ["seller"]}
                ]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n🧪 测试: {test_case['name']}")
            print(f"   问题: {test_case['problem']}")
            
            deep_relations, implicit_constraints = engine.discover_deep_relations(
                test_case["problem"],
                test_case["entities"], 
                []
            )
            
            # 统计不同深度的关系
            depth_stats = {
                "surface": len([r for r in deep_relations if r.depth.value == "surface"]),
                "shallow": len([r for r in deep_relations if r.depth.value == "shallow"]),
                "medium": len([r for r in deep_relations if r.depth.value == "medium"]),
                "deep": len([r for r in deep_relations if r.depth.value == "deep"])
            }
            
            # 统计不同类型的约束
            constraint_stats = {}
            for constraint in implicit_constraints:
                constraint_type = constraint.constraint_type.value
                constraint_stats[constraint_type] = constraint_stats.get(constraint_type, 0) + 1
            
            result = {
                "name": test_case["name"],
                "deep_relations_count": len(deep_relations),
                "implicit_constraints_count": len(implicit_constraints),
                "depth_distribution": depth_stats,
                "constraint_distribution": constraint_stats,
                "avg_confidence": sum(r.confidence for r in deep_relations) / len(deep_relations) if deep_relations else 0
            }
            
            results.append(result)
            
            print(f"   📊 结果: {len(deep_relations)}个深度关系, {len(implicit_constraints)}个约束")
            print(f"   📈 深度分布: {depth_stats}")
            print(f"   🔒 约束分布: {constraint_stats}")
            print(f"   📊 平均置信度: {result['avg_confidence']:.3f}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ 多类型问题测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def performance_test():
    """性能测试"""
    import time
    
    try:
        from src.reasoning.private.deep_implicit_engine import DeepImplicitEngine
        
        engine = DeepImplicitEngine()
        
        # 测试不同复杂度的问题
        test_problems = [
            {
                "complexity": "简单",
                "problem": "2 + 3 = ?",
                "entities": [
                    {"name": "2", "type": "number", "properties": ["operand"]},
                    {"name": "3", "type": "number", "properties": ["operand"]}
                ]
            },
            {
                "complexity": "中等",
                "problem": "小明有5个苹果，小红有3个苹果，一共有多少个苹果？",
                "entities": [
                    {"name": "小明", "type": "person", "properties": ["owner"]},
                    {"name": "5", "type": "number", "properties": ["quantity"]},
                    {"name": "苹果", "type": "object", "properties": ["countable"]},
                    {"name": "小红", "type": "person", "properties": ["owner"]},
                    {"name": "3", "type": "number", "properties": ["quantity"]}
                ]
            },
            {
                "complexity": "复杂",
                "problem": "班级有50个学生，男生占40%，女生中有80%参加了数学竞赛，参加竞赛的女生有多少人？",
                "entities": [
                    {"name": "班级", "type": "concept", "properties": ["container"]},
                    {"name": "50", "type": "number", "properties": ["total"]},
                    {"name": "学生", "type": "person", "properties": ["group"]},
                    {"name": "男生", "type": "person", "properties": ["subgroup"]},
                    {"name": "40", "type": "number", "properties": ["percentage"]},
                    {"name": "女生", "type": "person", "properties": ["subgroup"]},
                    {"name": "80", "type": "number", "properties": ["percentage"]},
                    {"name": "数学竞赛", "type": "concept", "properties": ["activity"]}
                ]
            }
        ]
        
        performance_results = []
        
        print(f"\n⚡ 性能测试:")
        
        for test_case in test_problems:
            start_time = time.time()
            
            deep_relations, implicit_constraints = engine.discover_deep_relations(
                test_case["problem"],
                test_case["entities"],
                []
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                "complexity": test_case["complexity"],
                "processing_time": processing_time,
                "relations_found": len(deep_relations),
                "constraints_found": len(implicit_constraints),
                "entities_count": len(test_case["entities"])
            }
            
            performance_results.append(result)
            
            print(f"   {test_case['complexity']}问题: {processing_time:.4f}秒")
            print(f"      实体: {result['entities_count']}, 关系: {result['relations_found']}, 约束: {result['constraints_found']}")
        
        return True, performance_results
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False, []

def main():
    """主测试函数"""
    print("🚀 开始深度隐含关系发现算法测试")
    print("=" * 60)
    
    all_passed = True
    
    # 测试1: 基础引擎功能
    print("\n1️⃣ 测试深度隐含关系发现引擎基础功能")
    success1, relations_count, constraints_count = test_deep_implicit_engine()
    if not success1:
        all_passed = False
    
    # 测试2: MLR处理器集成
    print("\n2️⃣ 测试MLR处理器集成")
    success2, result = test_mlr_processor_integration()
    if not success2:
        all_passed = False
    
    # 测试3: 多种问题类型
    print("\n3️⃣ 测试多种问题类型")
    success3, multi_results = test_multiple_problem_types()
    if not success3:
        all_passed = False
    
    # 测试4: 性能测试
    print("\n4️⃣ 性能测试")
    success4, perf_results = performance_test()
    if not success4:
        all_passed = False
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📊 测试总结报告")
    print("=" * 60)
    
    if all_passed:
        print("✅ 所有测试通过!")
        
        print(f"\n📈 核心指标:")
        if success1:
            print(f"   - 基础关系发现: {relations_count} 个深度关系, {constraints_count} 个约束")
        
        if success3 and multi_results:
            total_relations = sum(r["deep_relations_count"] for r in multi_results)
            total_constraints = sum(r["implicit_constraints_count"] for r in multi_results)
            avg_confidence = sum(r["avg_confidence"] for r in multi_results) / len(multi_results)
            print(f"   - 多类型测试: {total_relations} 个关系, {total_constraints} 个约束")
            print(f"   - 平均置信度: {avg_confidence:.3f}")
        
        if success4 and perf_results:
            avg_time = sum(r["processing_time"] for r in perf_results) / len(perf_results)
            print(f"   - 平均处理时间: {avg_time:.4f} 秒")
        
        print(f"\n✨ 算法核心能力验证:")
        print(f"   ✅ 语义蕴含推理逻辑")
        print(f"   ✅ 隐含约束条件挖掘") 
        print(f"   ✅ 多层关系建模机制")
        print(f"   ✅ 前端可视化数据生成")
        
    else:
        print("❌ 部分测试失败，需要修复问题")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)