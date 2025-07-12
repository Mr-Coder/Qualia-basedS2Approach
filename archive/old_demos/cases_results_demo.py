#!/usr/bin/env python3
"""
案例结果演示程序 - COT-DIR数学推理系统
展示对不同复杂度和类型题目的处理结果
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai_core.interfaces.data_structures import MathProblem, ReasoningResult
from src.models.structures import Relations
from src.processors.complexity_classifier import ComplexityClassifier
from src.reasoning_core.cotdir_method import COTDIRModel
from src.reasoning_engine.cotdir_integration import COTDIRIntegratedWorkflow


class CasesResultsDemo:
    """案例结果演示类"""
    
    def __init__(self):
        """初始化演示系统"""
        print("🚀 初始化COT-DIR案例结果演示系统...")
        
        # 初始化核心组件
        self.cotdir_workflow = COTDIRIntegratedWorkflow()
        self.cotdir_model = COTDIRModel()
        self.complexity_classifier = ComplexityClassifier()
        
        # 定义测试案例
        self.test_cases = self._prepare_test_cases()
        
        print("✅ 系统初始化完成！\n")
    
    def _prepare_test_cases(self) -> List[Dict[str, Any]]:
        """准备测试案例"""
        return [
            # 中文案例 - 从Math23K数据集
            {
                "id": "math23k_001",
                "language": "中文",
                "problem": "小明有15个苹果，他给了小红5个，又买了8个，现在小明有多少个苹果？",
                "expected_answer": "18",
                "type": "加减运算",
                "difficulty": "简单",
                "complexity_level": "L2",
                "source": "Math23K"
            },
            {
                "id": "math23k_003", 
                "language": "中文",
                "problem": "班级里有24名学生，其中男生占3/8，女生有多少名？",
                "expected_answer": "15",
                "type": "分数运算",
                "difficulty": "中等",
                "complexity_level": "L2",
                "source": "Math23K"
            },
            {
                "id": "math23k_004",
                "language": "中文", 
                "problem": "一件衣服原价120元，打8折后的价格是多少元？",
                "expected_answer": "96",
                "type": "百分比计算",
                "difficulty": "中等",
                "complexity_level": "L2",
                "source": "Math23K"
            },
            
            # 英文案例 - 从GSM8K数据集
            {
                "id": "gsm8k_001",
                "language": "英文",
                "problem": "Chenny is 10 years old. Alyana is 4 years younger than Chenny. How old is Anne if she is 2 years older than Alyana?",
                "expected_answer": "8",
                "type": "年龄推理",
                "difficulty": "简单",
                "complexity_level": "L0",
                "source": "GSM8K"
            },
            {
                "id": "gsm8k_004",
                "language": "英文",
                "problem": "Liam is 16 years old now. Two years ago, Liam's age was twice the age of Vince. How old is Vince now?",
                "expected_answer": "9", 
                "type": "时间推理",
                "difficulty": "中等",
                "complexity_level": "L2",
                "source": "GSM8K"
            },
            {
                "id": "gsm8k_complex",
                "language": "英文",
                "problem": "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?",
                "expected_answer": "13",
                "type": "投资回报分析",
                "difficulty": "困难", 
                "complexity_level": "L2",
                "source": "GSM8K"
            }
        ]
    
    def run_single_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个案例"""
        print(f"📝 处理案例: {case['id']} ({case['language']})")
        print(f"   题目: {case['problem']}")
        print(f"   预期答案: {case['expected_answer']}")
        print(f"   复杂度: {case['complexity_level']}")
        print("   " + "="*60)
        
        start_time = time.time()
        
        try:
            # 创建数学问题对象
            math_problem = MathProblem(
                id=case['id'],
                text=case['problem'],
                answer=case['expected_answer'],
                complexity=case['complexity_level']
            )
            
            # 使用COT-DIR系统求解
            reasoning_result = self.cotdir_workflow.process_problem(math_problem)
            
            processing_time = time.time() - start_time
            
            # 处理结果
            result = {
                "case_info": case,
                "processing_time": round(processing_time, 3),
                "reasoning_result": reasoning_result,
                "success": reasoning_result is not None
            }
            
            # 显示推理过程
            self._display_reasoning_process(reasoning_result)
            
            # 显示最终结果
            if reasoning_result and hasattr(reasoning_result, 'final_answer'):
                predicted_answer = str(reasoning_result.final_answer).strip()
                expected_answer = str(case['expected_answer']).strip()
                is_correct = predicted_answer == expected_answer
                
                print(f"\n🎯 结果对比:")
                print(f"   预期答案: {expected_answer}")
                print(f"   系统答案: {predicted_answer}")
                print(f"   是否正确: {'✅ 正确' if is_correct else '❌ 错误'}")
                print(f"   处理时间: {processing_time:.3f}秒")
                
                result["is_correct"] = is_correct
                result["predicted_answer"] = predicted_answer
            else:
                print("❌ 系统未能生成答案")
                result["is_correct"] = False
                result["predicted_answer"] = None
                
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            result = {
                "case_info": case,
                "processing_time": round(time.time() - start_time, 3),
                "reasoning_result": None,
                "success": False,
                "error": str(e),
                "is_correct": False,
                "predicted_answer": None
            }
        
        print("\n" + "="*80 + "\n")
        return result
    
    def _display_reasoning_process(self, reasoning_result: ReasoningResult):
        """显示推理过程"""
        if not reasoning_result:
            return
            
        print("\n🧠 推理过程分析:")
        
        # 显示实体提取
        if hasattr(reasoning_result, 'entities') and reasoning_result.entities:
            print("   📊 实体提取:")
            for entity in reasoning_result.entities[:5]:  # 显示前5个
                print(f"      • {entity.name} ({entity.entity_type}): {entity.value}")
        
        # 显示关系发现
        if hasattr(reasoning_result, 'relations') and reasoning_result.relations:
            print("   🔗 关系发现:")
            for relation in reasoning_result.relations[:5]:  # 显示前5个
                print(f"      • {relation.relation_type}: {relation.source} → {relation.target}")
        
        # 显示推理步骤
        if hasattr(reasoning_result, 'reasoning_steps') and reasoning_result.reasoning_steps:
            print("   🔄 推理步骤:")
            for i, step in enumerate(reasoning_result.reasoning_steps[:5], 1):  # 显示前5步
                if hasattr(step, 'description'):
                    print(f"      {i}. {step.description}")
                else:
                    print(f"      {i}. {str(step)}")
        
        # 显示置信度
        if hasattr(reasoning_result, 'confidence_score'):
            confidence = reasoning_result.confidence_score
            print(f"   💯 置信度评分: {confidence:.2f}%")
            
            # 置信度等级
            if confidence >= 90:
                level = "🟢 极高"
            elif confidence >= 80:
                level = "🔵 高"
            elif confidence >= 70:
                level = "🟡 中等"
            elif confidence >= 60:
                level = "🟠 较低"
            else:
                level = "🔴 低"
            print(f"   📈 置信度等级: {level}")
    
    def run_batch_demo(self):
        """运行批量演示"""
        print("🎯 开始COT-DIR案例结果演示")
        print("="*80)
        
        results = []
        correct_count = 0
        total_count = len(self.test_cases)
        
        for i, case in enumerate(self.test_cases, 1):
            print(f"\n【案例 {i}/{total_count}】")
            result = self.run_single_case(case)
            results.append(result)
            
            if result.get("is_correct", False):
                correct_count += 1
        
        # 生成总结报告
        self._generate_summary_report(results, correct_count, total_count)
        
        return results
    
    def _generate_summary_report(self, results: List[Dict], correct_count: int, total_count: int):
        """生成总结报告"""
        print("📊 演示总结报告")
        print("="*80)
        
        # 整体准确率
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        print(f"🎯 整体准确率: {correct_count}/{total_count} ({accuracy:.1f}%)")
        
        # 按语言分类统计
        chinese_results = [r for r in results if r["case_info"]["language"] == "中文"]
        english_results = [r for r in results if r["case_info"]["language"] == "英文"]
        
        chinese_correct = sum(1 for r in chinese_results if r.get("is_correct", False))
        english_correct = sum(1 for r in english_results if r.get("is_correct", False))
        
        print(f"\n📈 按语言分析:")
        if chinese_results:
            chinese_accuracy = (chinese_correct / len(chinese_results)) * 100
            print(f"   🇨🇳 中文题目: {chinese_correct}/{len(chinese_results)} ({chinese_accuracy:.1f}%)")
        
        if english_results:
            english_accuracy = (english_correct / len(english_results)) * 100
            print(f"   🇺🇸 英文题目: {english_correct}/{len(english_results)} ({english_accuracy:.1f}%)")
        
        # 按复杂度分类统计
        print(f"\n🔢 按复杂度分析:")
        complexity_stats = {}
        for result in results:
            complexity = result["case_info"]["complexity_level"]
            if complexity not in complexity_stats:
                complexity_stats[complexity] = {"total": 0, "correct": 0}
            complexity_stats[complexity]["total"] += 1
            if result.get("is_correct", False):
                complexity_stats[complexity]["correct"] += 1
        
        for complexity, stats in sorted(complexity_stats.items()):
            accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"   {complexity}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
        
        # 平均处理时间
        avg_time = sum(r["processing_time"] for r in results) / len(results) if results else 0
        print(f"\n⏱️  平均处理时间: {avg_time:.3f}秒")
        
        # 系统性能评估
        print(f"\n🏆 系统性能评估:")
        if accuracy >= 90:
            grade = "A+ (优秀)"
        elif accuracy >= 80:
            grade = "A (良好)"
        elif accuracy >= 70:
            grade = "B (一般)"
        elif accuracy >= 60:
            grade = "C (及格)"
        else:
            grade = "D (需改进)"
        
        print(f"   📊 综合评分: {grade}")
        print(f"   🔧 推理能力: {'🟢 强' if accuracy >= 80 else '🟡 中等' if accuracy >= 60 else '🔴 弱'}")
        print(f"   ⚡ 处理速度: {'🟢 快' if avg_time <= 1.0 else '🟡 中等' if avg_time <= 3.0 else '🔴 慢'}")
        
    def save_results_to_file(self, results: List[Dict], filename: str = "case_results.json"):
        """保存结果到文件"""
        try:
            # 处理不可序列化的对象
            serializable_results = []
            for result in results:
                serializable_result = {
                    "case_info": result["case_info"],
                    "processing_time": result["processing_time"],
                    "success": result["success"],
                    "is_correct": result.get("is_correct", False),
                    "predicted_answer": result.get("predicted_answer", None)
                }
                
                # 添加错误信息（如果有）
                if "error" in result:
                    serializable_result["error"] = result["error"]
                
                # 简化推理结果
                if result["reasoning_result"]:
                    reasoning_summary = {
                        "has_final_answer": hasattr(result["reasoning_result"], 'final_answer'),
                        "entities_count": len(getattr(result["reasoning_result"], 'entities', [])),
                        "relations_count": len(getattr(result["reasoning_result"], 'relations', [])),
                        "confidence_score": getattr(result["reasoning_result"], 'confidence_score', 0)
                    }
                    serializable_result["reasoning_summary"] = reasoning_summary
                
                serializable_results.append(serializable_result)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_cases": len(results),
                    "results": serializable_results
                }, f, ensure_ascii=False, indent=2)
            
            print(f"📁 结果已保存到: {filename}")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {str(e)}")

def main():
    """主函数"""
    try:
        # 创建演示实例
        demo = CasesResultsDemo()
        
        # 运行演示
        results = demo.run_batch_demo()
        
        # 保存结果
        demo.save_results_to_file(results)
        
        print("\n🎉 案例结果演示完成！")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 