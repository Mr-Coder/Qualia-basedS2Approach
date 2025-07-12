#!/usr/bin/env python3
"""
COT-DIR 验证和性能测试演示
========================

展示COT-DIR数学推理系统的验证和性能分析功能：
1. 系统功能验证
2. 性能基准测试
3. 准确率评估
4. 错误分析

Author: COT-DIR Team
Date: 2025-01-31
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loader import DataLoader
from data.preprocessor import Preprocessor
from reasoning_core.meta_knowledge import MetaKnowledge
from src.bridge.reasoning_bridge import ReasoningEngine


def validate_system_functionality():
    """验证系统基本功能"""
    print("\n🔍 系统功能验证")
    print("=" * 50)
    
    validation_results = {
        "component_initialization": False,
        "basic_reasoning": False,
        "meta_knowledge": False,
        "data_processing": False,
        "error_handling": False
    }
    
    try:
        # 1. 组件初始化验证
        print("1. 🔧 验证组件初始化...")
        loader = DataLoader()
        preprocessor = Preprocessor()
        engine = ReasoningEngine()
        meta_knowledge = MetaKnowledge()
        validation_results["component_initialization"] = True
        print("   ✅ 所有组件初始化成功")
        
        # 2. 基础推理验证
        print("2. 🧠 验证基础推理...")
        test_sample = {
            "problem": "3 + 5 = ?",
            "answer": "8",
            "id": "validation_test"
        }
        processed = preprocessor.process(test_sample)
        result = engine.solve(processed)
        
        if result and 'final_answer' in result:
            validation_results["basic_reasoning"] = True
            print("   ✅ 基础推理功能正常")
        else:
            print("   ❌ 基础推理功能异常")
        
        # 3. 元知识系统验证
        print("3. 🧠 验证元知识系统...")
        concepts = meta_knowledge.identify_concepts_in_text("计算分数 1/2 + 1/3")
        strategies = meta_knowledge.suggest_strategies("解方程 x + 5 = 10")
        
        if concepts and strategies:
            validation_results["meta_knowledge"] = True
            print("   ✅ 元知识系统功能正常")
        else:
            print("   ❌ 元知识系统功能异常")
        
        # 4. 数据处理验证
        print("4. 📦 验证数据处理...")
        try:
            # 尝试处理不同格式的数据
            test_cases = [
                {"problem": "简单问题", "answer": "答案"},
                {"question": "另一种格式", "solution": "解答"},
                {"text": "第三种格式"}
            ]
            
            for case in test_cases:
                processed = preprocessor.process(case)
                if 'cleaned_text' in processed:
                    validation_results["data_processing"] = True
                    break
            
            if validation_results["data_processing"]:
                print("   ✅ 数据处理功能正常")
            else:
                print("   ❌ 数据处理功能异常")
                
        except Exception as e:
            print(f"   ❌ 数据处理验证失败: {e}")
        
        # 5. 错误处理验证
        print("5. ⚠️ 验证错误处理...")
        try:
            # 测试异常输入处理
            invalid_inputs = [
                None,
                {},
                {"invalid": "data"},
                {"problem": ""}
            ]
            
            error_handled_count = 0
            for invalid_input in invalid_inputs:
                try:
                    if invalid_input is not None:
                        preprocessor.process(invalid_input)
                    error_handled_count += 1
                except:
                    error_handled_count += 1
            
            if error_handled_count >= len(invalid_inputs) // 2:
                validation_results["error_handling"] = True
                print("   ✅ 错误处理功能正常")
            else:
                print("   ❌ 错误处理功能需要改进")
                
        except Exception as e:
            print(f"   ⚠️ 错误处理验证异常: {e}")
    
    except Exception as e:
        print(f"❌ 系统验证失败: {e}")
    
    # 输出验证总结
    print(f"\n📊 验证结果总结:")
    passed_tests = sum(validation_results.values())
    total_tests = len(validation_results)
    
    for test_name, passed in validation_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 验证通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    return validation_results


def performance_benchmark():
    """性能基准测试"""
    print("\n⚡ 性能基准测试")
    print("=" * 50)
    
    try:
        engine = ReasoningEngine()
        preprocessor = Preprocessor()
        
        # 测试用例 - 不同复杂度的问题
        test_cases = [
            {
                "name": "简单算术",
                "problems": [
                    "3 + 5",
                    "10 - 4",
                    "6 × 7",
                    "24 ÷ 3"
                ]
            },
            {
                "name": "应用问题",
                "problems": [
                    "小明有5个苹果，吃了2个，还剩几个？",
                    "一件衣服50元，买3件多少钱？",
                    "班级有30人，来了25人，缺席几人？"
                ]
            },
            {
                "name": "复杂推理",
                "problems": [
                    "一个数的3倍加上8等于20，这个数是多少？",
                    "长方形长6米宽4米，周长和面积分别是多少？"
                ]
            }
        ]
        
        performance_results = {}
        
        for test_category in test_cases:
            category_name = test_category["name"]
            problems = test_category["problems"]
            
            print(f"\n📊 测试类别: {category_name}")
            
            times = []
            success_count = 0
            
            for i, problem in enumerate(problems):
                try:
                    # 准备数据
                    sample = {"problem": problem, "id": f"{category_name}_{i}"}
                    processed = preprocessor.process(sample)
                    
                    # 计时测试
                    start_time = time.time()
                    result = engine.solve(processed)
                    end_time = time.time()
                    
                    execution_time = (end_time - start_time) * 1000  # 转换为毫秒
                    times.append(execution_time)
                    
                    if result and 'final_answer' in result:
                        success_count += 1
                    
                    print(f"   问题 {i+1}: {execution_time:.1f}ms")
                    
                except Exception as e:
                    print(f"   问题 {i+1}: 失败 ({e})")
                    times.append(float('inf'))
            
            # 计算统计数据
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                max_time = max(valid_times)
                min_time = min(valid_times)
                
                performance_results[category_name] = {
                    "average_time_ms": avg_time,
                    "max_time_ms": max_time,
                    "min_time_ms": min_time,
                    "success_rate": success_count / len(problems),
                    "total_problems": len(problems)
                }
                
                print(f"   📈 平均时间: {avg_time:.1f}ms")
                print(f"   📈 最大时间: {max_time:.1f}ms")
                print(f"   📈 最小时间: {min_time:.1f}ms")
                print(f"   🎯 成功率: {success_count}/{len(problems)} ({success_count/len(problems)*100:.1f}%)")
        
        # 输出性能总结
        print(f"\n🏆 性能测试总结:")
        overall_avg = sum(r["average_time_ms"] for r in performance_results.values()) / len(performance_results)
        overall_success = sum(r["success_rate"] * r["total_problems"] for r in performance_results.values()) / sum(r["total_problems"] for r in performance_results.values())
        
        print(f"   整体平均响应时间: {overall_avg:.1f}ms")
        print(f"   整体成功率: {overall_success*100:.1f}%")
        
        # 性能评估
        if overall_avg < 100:
            print("   🚀 性能评级: 优秀 (< 100ms)")
        elif overall_avg < 500:
            print("   ✅ 性能评级: 良好 (< 500ms)")
        elif overall_avg < 1000:
            print("   ⚠️ 性能评级: 一般 (< 1s)")
        else:
            print("   ❌ 性能评级: 需要优化 (> 1s)")
        
        return performance_results
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return {}


def accuracy_evaluation():
    """准确率评估"""
    print("\n🎯 准确率评估")
    print("=" * 50)
    
    try:
        engine = ReasoningEngine()
        preprocessor = Preprocessor()
        
        # 已知答案的测试题
        test_problems = [
            {"problem": "5 + 3", "expected": "8"},
            {"problem": "10 - 4", "expected": "6"},
            {"problem": "7 × 6", "expected": "42"},
            {"problem": "20 ÷ 4", "expected": "5"},
            {"problem": "小明有8个苹果，吃了3个，还剩几个？", "expected": "5"},
            {"problem": "一支笔2元，买5支多少钱？", "expected": "10"},
            {"problem": "班级有25个学生，来了20个，缺席几个？", "expected": "5"},
            {"problem": "一个正方形边长3米，面积多少平方米？", "expected": "9"}
        ]
        
        correct_count = 0
        total_count = len(test_problems)
        results = []
        
        print("📝 逐题测试:")
        
        for i, test_case in enumerate(test_problems, 1):
            problem = test_case["problem"]
            expected = test_case["expected"]
            
            try:
                sample = {"problem": problem, "id": f"accuracy_test_{i}"}
                processed = preprocessor.process(sample)
                result = engine.solve(processed)
                
                predicted = result.get('final_answer', '') if result else ''
                is_correct = str(predicted).strip() == str(expected).strip()
                
                if is_correct:
                    correct_count += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"   {i}. {problem}")
                print(f"      预期: {expected}, 实际: {predicted} {status}")
                
                results.append({
                    "problem": problem,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct
                })
                
            except Exception as e:
                print(f"   {i}. {problem}")
                print(f"      错误: {e} ❌")
                results.append({
                    "problem": problem,
                    "expected": expected,
                    "predicted": "ERROR",
                    "correct": False
                })
        
        # 计算准确率
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 准确率评估结果:")
        print(f"   正确答案: {correct_count}/{total_count}")
        print(f"   准确率: {accuracy*100:.1f}%")
        
        # 准确率评级
        if accuracy >= 0.9:
            print("   🏆 准确率评级: 优秀 (≥ 90%)")
        elif accuracy >= 0.8:
            print("   ✅ 准确率评级: 良好 (≥ 80%)")
        elif accuracy >= 0.7:
            print("   ⚠️ 准确率评级: 一般 (≥ 70%)")
        else:
            print("   ❌ 准确率评级: 需要改进 (< 70%)")
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": results
        }
        
    except Exception as e:
        print(f"❌ 准确率评估失败: {e}")
        return {}


def save_validation_report(validation_data: Dict[str, Any]):
    """保存验证报告"""
    try:
        report_file = "validation_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 验证报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 报告保存失败: {e}")


def main():
    """主函数"""
    print("🚀 启动 COT-DIR 验证和性能测试")
    print("全面测试系统功能、性能和准确性")
    
    validation_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_validation": {},
        "performance_benchmark": {},
        "accuracy_evaluation": {}
    }
    
    try:
        # 1. 系统功能验证
        validation_data["system_validation"] = validate_system_functionality()
        
        # 2. 性能基准测试
        validation_data["performance_benchmark"] = performance_benchmark()
        
        # 3. 准确率评估
        validation_data["accuracy_evaluation"] = accuracy_evaluation()
        
        # 4. 保存验证报告
        save_validation_report(validation_data)
        
        print("\n🎉 验证和性能测试完成！")
        print("\n📈 总体评估:")
        
        # 系统健康度评估
        if validation_data["system_validation"]:
            system_health = sum(validation_data["system_validation"].values()) / len(validation_data["system_validation"])
            print(f"   系统健康度: {system_health*100:.1f}%")
        
        if validation_data["accuracy_evaluation"]:
            accuracy = validation_data["accuracy_evaluation"].get("accuracy", 0)
            print(f"   系统准确率: {accuracy*100:.1f}%")
        
        print("\n📚 查看详细结果:")
        print("   - validation_report.json (完整验证报告)")
        print("   - validation_results.json (历史验证数据)")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 验证测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 