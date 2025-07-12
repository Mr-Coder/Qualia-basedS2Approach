#!/usr/bin/env python3
"""
COT-DIR 增强功能演示
==================

展示COT-DIR数学推理系统的增强功能：
1. 元知识系统演示
2. 策略推荐系统
3. 复杂推理能力
4. 性能分析功能

Author: COT-DIR Team
Date: 2025-01-31
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time

from data.loader import DataLoader
from data.preprocessor import Preprocessor
from reasoning_core.meta_knowledge import MetaKnowledge, MetaKnowledgeReasoning
from src.bridge.reasoning_bridge import ReasoningEngine


def demo_meta_knowledge_system():
    """演示元知识系统"""
    print("\n🧠 元知识系统演示")
    print("=" * 50)
    
    try:
        meta_knowledge = MetaKnowledge()
        
        # 1. 概念识别演示
        print("\n1. 📚 概念识别能力")
        test_problems = [
            "计算3/4 + 1/2的值",
            "一个商品原价200元，打7折后的价格",
            "长方形的长是10米，宽是8米，求面积",
            "小明的速度是每小时60公里，走120公里需要多长时间？"
        ]
        
        for problem in test_problems:
            concepts = meta_knowledge.identify_concepts_in_text(problem)
            print(f"问题: {problem}")
            print(f"识别概念: {', '.join(concepts) if concepts else '无'}")
            print()
        
        # 2. 策略推荐演示
        print("\n2. 🎯 策略推荐系统")
        strategy_problems = [
            "已知一个数的3倍加上5等于14，求这个数",
            "证明对于任意正整数n，n^2 + n是偶数",
            "在1到100的数字中，能被3整除的数有多少个？",
            "一个班级有男生和女生，男生比女生多5人，总共35人，求男女生各多少人？"
        ]
        
        for problem in strategy_problems:
            strategies = meta_knowledge.suggest_strategies_with_priority(problem)
            print(f"问题: {problem}")
            print("推荐策略:")
            for strategy in strategies[:3]:  # 显示前3个推荐策略
                print(f"  - {strategy['strategy']} (优先级: {strategy['priority']:.2f})")
            print()
        
        # 3. 概念信息查询
        print("\n3. 📖 概念知识库")
        concepts_to_show = ["分数", "百分比", "面积", "速度"]
        
        for concept in concepts_to_show:
            info = meta_knowledge.get_concept_info(concept)
            if info:
                print(f"📚 概念: {concept}")
                print(f"   定义: {info['definition']}")
                print(f"   性质: {', '.join(info['properties'][:2])}...")
                print(f"   常见错误: {', '.join(info['common_mistakes'][:2])}...")
                print()
        
        print("✅ 元知识系统演示完成")
        
    except Exception as e:
        print(f"❌ 元知识系统演示失败: {e}")


def demo_enhanced_reasoning():
    """演示增强推理能力"""
    print("\n🔧 增强推理能力演示")
    print("=" * 50)
    
    try:
        engine = ReasoningEngine()
        preprocessor = Preprocessor()
        
        # 复杂问题测试
        complex_problems = [
            {
                "problem": "一个商店进了一批商品，成本价是每件60元。如果按成本价的150%定价，然后打8折销售，每件商品的利润是多少元？",
                "expected_type": "多步推理"
            },
            {
                "problem": "小明、小红、小李三人的年龄和是45岁。小明比小红大3岁，小红比小李大2岁。问三人各多少岁？",
                "expected_type": "方程组推理"
            },
            {
                "problem": "一个圆形花园的直径是20米，现在要在花园周围建一条宽2米的环形小路，小路的面积是多少平方米？",
                "expected_type": "几何推理"
            }
        ]
        
        for i, test_case in enumerate(complex_problems, 1):
            print(f"\n📝 复杂问题 {i} ({test_case['expected_type']}):")
            print(f"问题: {test_case['problem']}")
            
            try:
                # 预处理
                sample = {
                    "problem": test_case['problem'], 
                    "id": f"complex_{i}"
                }
                processed = preprocessor.process(sample)
                
                # 增强推理
                start_time = time.time()
                result = engine.solve(processed)
                end_time = time.time()
                
                # 输出详细结果
                print(f"💡 答案: {result.get('final_answer', '未解出')}")
                print(f"🎯 使用策略: {result.get('strategy_used', '未知')}")
                print(f"📊 置信度: {result.get('confidence', 0):.2f}")
                print(f"⏱️ 求解时间: {(end_time - start_time)*1000:.1f}ms")
                
                # 显示元知识增强信息
                if 'meta_knowledge_enhancement' in result:
                    enhancement = result['meta_knowledge_enhancement']
                    if 'identified_concepts' in enhancement:
                        print(f"🧠 识别概念: {', '.join(enhancement['identified_concepts'])}")
                    if 'suggested_strategies' in enhancement:
                        print(f"💡 推荐策略: {', '.join(enhancement['suggested_strategies'])}")
                
                # 显示解决方案验证
                if 'solution_validation' in result:
                    validation = result['solution_validation']
                    print(f"✅ 解决方案验证: {validation.get('is_valid', False)}")
                    if 'confidence' in validation:
                        print(f"🎯 验证置信度: {validation['confidence']:.2f}")
                
            except Exception as e:
                print(f"❌ 求解失败: {e}")
            
            print("-" * 60)
        
        print("✅ 增强推理演示完成")
        
    except Exception as e:
        print(f"❌ 增强推理演示失败: {e}")


def demo_strategy_effectiveness():
    """演示策略有效性分析"""
    print("\n📊 策略有效性分析")
    print("=" * 50)
    
    try:
        meta_knowledge = MetaKnowledge()
        
        # 分析不同类型问题的策略推荐
        problem_types = {
            "算术问题": [
                "计算 25 × 4 + 15 ÷ 3",
                "求 2/3 + 3/4 的值"
            ],
            "应用问题": [
                "小明买了3支笔，每支2元，付了10元，找回多少钱？",
                "一个班有40个学生，80%的学生参加了活动，参加活动的学生有多少人？"
            ],
            "几何问题": [
                "正方形的边长是5厘米，面积是多少？",
                "圆的半径是3米，周长是多少？"
            ],
            "代数问题": [
                "解方程 2x + 5 = 13",
                "如果 y = 3x - 2，当 x = 4 时，y 的值是多少？"
            ]
        }
        
        strategy_stats = {}
        
        for problem_type, problems in problem_types.items():
            print(f"\n📚 {problem_type} 策略分析:")
            type_strategies = []
            
            for problem in problems:
                strategies = meta_knowledge.suggest_strategies_with_priority(problem)
                print(f"  问题: {problem[:30]}...")
                print(f"  最佳策略: {strategies[0]['strategy'] if strategies else '无'}")
                
                if strategies:
                    type_strategies.extend([s['strategy'] for s in strategies[:2]])
            
            # 统计策略使用频率
            from collections import Counter
            strategy_counts = Counter(type_strategies)
            strategy_stats[problem_type] = strategy_counts
            
            print(f"  常用策略: {', '.join(list(strategy_counts.keys())[:3])}")
        
        # 输出策略统计
        print(f"\n📈 策略使用统计:")
        all_strategies = set()
        for strategies in strategy_stats.values():
            all_strategies.update(strategies.keys())
        
        for strategy in list(all_strategies)[:5]:  # 显示前5个策略
            total_uses = sum(stats.get(strategy, 0) for stats in strategy_stats.values())
            print(f"  {strategy}: 使用 {total_uses} 次")
        
        print("✅ 策略有效性分析完成")
        
    except Exception as e:
        print(f"❌ 策略分析失败: {e}")


def main():
    """主函数"""
    print("🚀 启动 COT-DIR 增强功能演示")
    print("展示元知识系统、策略推荐、增强推理等高级功能")
    
    try:
        # 1. 元知识系统演示
        demo_meta_knowledge_system()
        
        # 2. 增强推理演示
        demo_enhanced_reasoning()
        
        # 3. 策略有效性分析
        demo_strategy_effectiveness()
        
        print("\n🎉 增强功能演示完成！")
        print("\n📚 更多功能:")
        print("   - demos/validation_demo.py (验证和性能测试)")
        print("   - 查看 validation_results.json 了解详细验证数据")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 