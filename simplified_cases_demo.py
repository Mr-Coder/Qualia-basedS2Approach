#!/usr/bin/env python3
"""
简化案例结果演示程序 - COT-DIR数学推理系统模拟
展示对不同复杂度和类型题目的处理结果（模拟版本）
"""

import json
import random
import re
import time
from typing import Any, Dict, List


class SimplifiedCOTDIRDemo:
    """简化的COT-DIR演示类"""
    
    def __init__(self):
        """初始化演示系统"""
        print("🚀 初始化COT-DIR案例结果演示系统（模拟版）...")
        
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
    
    def _simulate_cotdir_reasoning(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """模拟COT-DIR推理过程"""
        problem_text = case['problem']
        expected_answer = case['expected_answer']
        
        # 模拟实体提取
        entities = self._extract_entities(problem_text)
        
        # 模拟关系发现
        relations = self._discover_relations(problem_text, entities)
        
        # 模拟多层推理
        reasoning_steps = self._multi_layer_reasoning(problem_text, entities, relations)
        
        # 模拟置信度验证
        confidence_score = self._calculate_confidence(reasoning_steps, case['complexity_level'])
        
        # 模拟最终答案生成
        predicted_answer = self._generate_answer(case)
        
        return {
            "entities": entities,
            "relations": relations,
            "reasoning_steps": reasoning_steps,
            "confidence_score": confidence_score,
            "final_answer": predicted_answer
        }
    
    def _extract_entities(self, problem_text: str) -> List[Dict[str, Any]]:
        """模拟实体提取"""
        entities = []
        
        # 提取数字
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', problem_text)
        for i, num in enumerate(numbers):
            entities.append({
                "name": f"数值_{i+1}",
                "type": "数量",
                "value": num,
                "text": num
            })
        
        # 提取人名（中文）
        chinese_names = re.findall(r'[小大老][明红华丽军文龙凤玉兰美芳]', problem_text)
        for name in chinese_names:
            entities.append({
                "name": name,
                "type": "人物",
                "value": name,
                "text": name
            })
        
        # 提取英文人名
        english_names = re.findall(r'\b[A-Z][a-z]+\b', problem_text)
        for name in english_names:
            if name not in ['How', 'If', 'Two', 'Each', 'The']:  # 排除非人名
                entities.append({
                    "name": name,
                    "type": "人物",
                    "value": name,
                    "text": name
                })
        
        # 提取物品
        items_cn = ['苹果', '学生', '衣服', '元', '折']
        items_en = ['years', 'lemons', 'tree', 'year']
        
        for item in items_cn + items_en:
            if item in problem_text:
                entities.append({
                    "name": item,
                    "type": "物品/概念",
                    "value": item,
                    "text": item
                })
        
        return entities[:10]  # 限制返回数量
    
    def _discover_relations(self, problem_text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """模拟关系发现"""
        relations = []
        
        # 数学运算关系
        if '给了' in problem_text or '给' in problem_text:
            relations.append({
                "type": "转移关系",
                "source": "小明",
                "target": "小红",
                "operation": "减法",
                "description": "给出苹果"
            })
        
        if '买了' in problem_text or 'buy' in problem_text.lower():
            relations.append({
                "type": "获得关系",
                "source": "小明",
                "target": "苹果",
                "operation": "加法",
                "description": "购买获得"
            })
        
        if '占' in problem_text or 'fraction' in problem_text.lower():
            relations.append({
                "type": "比例关系",
                "source": "男生",
                "target": "总数",
                "operation": "乘法",
                "description": "比例计算"
            })
        
        if 'younger' in problem_text.lower() or 'older' in problem_text.lower():
            relations.append({
                "type": "年龄关系",
                "source": "年龄对象1",
                "target": "年龄对象2", 
                "operation": "减法/加法",
                "description": "年龄差异"
            })
        
        if 'cost' in problem_text.lower() or 'sell' in problem_text.lower():
            relations.append({
                "type": "经济关系",
                "source": "成本",
                "target": "收益",
                "operation": "比较",
                "description": "投资回报分析"
            })
        
        return relations
    
    def _multi_layer_reasoning(self, problem_text: str, entities: List[Dict], relations: List[Dict]) -> List[Dict[str, Any]]:
        """模拟多层推理"""
        steps = []
        
        # L1层：基础信息提取
        steps.append({
            "layer": "L1",
            "description": "基础信息提取和解析",
            "operation": "文本分析",
            "details": f"识别出{len(entities)}个实体和{len(relations)}个关系"
        })
        
        # L2层：关系建模
        steps.append({
            "layer": "L2", 
            "description": "关系建模和方程构建",
            "operation": "关系映射",
            "details": "建立实体间的数学关系"
        })
        
        # L3层：推理求解
        if case := self._get_case_from_text(problem_text):
            if case['id'] == 'math23k_001':
                steps.extend([
                    {
                        "layer": "L3",
                        "description": "执行减法操作",
                        "operation": "15 - 5 = 10",
                        "details": "小明给出5个苹果后剩余10个"
                    },
                    {
                        "layer": "L3",
                        "description": "执行加法操作", 
                        "operation": "10 + 8 = 18",
                        "details": "小明买了8个苹果后总共有18个"
                    }
                ])
            elif case['id'] == 'math23k_003':
                steps.extend([
                    {
                        "layer": "L3",
                        "description": "计算男生人数",
                        "operation": "24 × 3/8 = 9",
                        "details": "男生人数为9人"
                    },
                    {
                        "layer": "L3",
                        "description": "计算女生人数",
                        "operation": "24 - 9 = 15", 
                        "details": "女生人数为15人"
                    }
                ])
            elif case['id'] == 'gsm8k_001':
                steps.extend([
                    {
                        "layer": "L3",
                        "description": "计算Alyana年龄",
                        "operation": "10 - 4 = 6",
                        "details": "Alyana比Chenny小4岁，所以6岁"
                    },
                    {
                        "layer": "L3",
                        "description": "计算Anne年龄",
                        "operation": "6 + 2 = 8",
                        "details": "Anne比Alyana大2岁，所以8岁"
                    }
                ])
        
        return steps
    
    def _get_case_from_text(self, problem_text: str) -> Dict[str, Any]:
        """根据问题文本匹配案例"""
        for case in self.test_cases:
            if case['problem'] == problem_text:
                return case
        return {}
    
    def _calculate_confidence(self, reasoning_steps: List[Dict], complexity_level: str) -> float:
        """模拟置信度计算"""
        base_confidence = 85.0
        
        # 根据复杂度调整
        complexity_factors = {
            "L0": 1.1,   # 简单问题置信度更高
            "L1": 1.05,
            "L2": 1.0,   # 中等复杂度基准
            "L3": 0.9    # 复杂问题置信度稍低
        }
        
        confidence = base_confidence * complexity_factors.get(complexity_level, 1.0)
        
        # 根据推理步骤数量调整
        if len(reasoning_steps) >= 4:
            confidence += 5.0  # 推理充分
        elif len(reasoning_steps) <= 2:
            confidence -= 3.0  # 推理不够充分
        
        # 添加随机变化
        confidence += random.uniform(-2, 2)
        
        return min(max(confidence, 60.0), 98.0)  # 限制在合理范围
    
    def _generate_answer(self, case: Dict[str, Any]) -> str:
        """模拟答案生成"""
        # 为了演示效果，大部分情况返回正确答案，偶尔出错
        if random.random() < 0.15:  # 15%概率出错
            # 生成错误答案
            expected = case['expected_answer']
            if expected.isdigit():
                wrong_answer = str(int(expected) + random.choice([-2, -1, 1, 2]))
                return wrong_answer
            else:
                return expected  # 非数字答案不容易生成错误版本
        else:
            return case['expected_answer']  # 返回正确答案
    
    def run_single_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个案例"""
        print(f"📝 处理案例: {case['id']} ({case['language']})")
        print(f"   题目: {case['problem']}")
        print(f"   预期答案: {case['expected_answer']}")
        print(f"   复杂度: {case['complexity_level']}")
        print("   " + "="*60)
        
        start_time = time.time()
        
        try:
            # 使用模拟的COT-DIR系统求解
            reasoning_result = self._simulate_cotdir_reasoning(case)
            
            processing_time = time.time() - start_time
            
            # 显示推理过程
            self._display_reasoning_process(reasoning_result)
            
            # 显示最终结果
            predicted_answer = str(reasoning_result['final_answer']).strip()
            expected_answer = str(case['expected_answer']).strip()
            is_correct = predicted_answer == expected_answer
            
            print(f"\n🎯 结果对比:")
            print(f"   预期答案: {expected_answer}")
            print(f"   系统答案: {predicted_answer}")
            print(f"   是否正确: {'✅ 正确' if is_correct else '❌ 错误'}")
            print(f"   处理时间: {processing_time:.3f}秒")
            
            result = {
                "case_info": case,
                "processing_time": round(processing_time, 3),
                "reasoning_result": reasoning_result,
                "success": True,
                "is_correct": is_correct,
                "predicted_answer": predicted_answer
            }
                
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
    
    def _display_reasoning_process(self, reasoning_result: Dict[str, Any]):
        """显示推理过程"""
        print("\n🧠 推理过程分析:")
        
        # 显示实体提取
        entities = reasoning_result.get('entities', [])
        if entities:
            print("   📊 实体提取:")
            for entity in entities[:5]:  # 显示前5个
                print(f"      • {entity['name']} ({entity['type']}): {entity['value']}")
        
        # 显示关系发现
        relations = reasoning_result.get('relations', [])
        if relations:
            print("   🔗 关系发现:")
            for relation in relations[:5]:  # 显示前5个
                print(f"      • {relation['type']}: {relation['source']} → {relation['target']}")
        
        # 显示推理步骤
        reasoning_steps = reasoning_result.get('reasoning_steps', [])
        if reasoning_steps:
            print("   🔄 推理步骤:")
            for i, step in enumerate(reasoning_steps, 1):
                print(f"      {i}. [{step['layer']}] {step['description']}")
                if 'operation' in step:
                    print(f"         操作: {step['operation']}")
        
        # 显示置信度
        confidence = reasoning_result.get('confidence_score', 0)
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
            accuracy_c = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"   {complexity}: {stats['correct']}/{stats['total']} ({accuracy_c:.1f}%)")
        
        # 平均处理时间
        avg_time = sum(r["processing_time"] for r in results) / len(results) if results else 0
        print(f"\n⏱️  平均处理时间: {avg_time:.3f}秒")
        
        # 平均置信度
        confidences = [r["reasoning_result"]["confidence_score"] for r in results if r["reasoning_result"]]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        print(f"🎯 平均置信度: {avg_confidence:.1f}%")
        
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
        print(f"   ⚡ 处理速度: {'🟢 快' if avg_time <= 0.5 else '🟡 中等' if avg_time <= 1.0 else '🔴 慢'}")
        print(f"   💯 置信度: {'🟢 高' if avg_confidence >= 85 else '🟡 中等' if avg_confidence >= 75 else '🔴 低'}")
        
    def save_results_to_file(self, results: List[Dict], filename: str = "simplified_case_results.json"):
        """保存结果到文件"""
        try:
            serializable_results = []
            for result in results:
                serializable_result = {
                    "case_info": result["case_info"],
                    "processing_time": result["processing_time"],
                    "success": result["success"],
                    "is_correct": result.get("is_correct", False),
                    "predicted_answer": result.get("predicted_answer", None)
                }
                
                if "error" in result:
                    serializable_result["error"] = result["error"]
                
                if result["reasoning_result"]:
                    reasoning_summary = {
                        "entities_count": len(result["reasoning_result"].get('entities', [])),
                        "relations_count": len(result["reasoning_result"].get('relations', [])),
                        "reasoning_steps_count": len(result["reasoning_result"].get('reasoning_steps', [])),
                        "confidence_score": result["reasoning_result"].get('confidence_score', 0)
                    }
                    serializable_result["reasoning_summary"] = reasoning_summary
                
                serializable_results.append(serializable_result)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_cases": len(results),
                    "system_info": "COT-DIR模拟演示系统",
                    "results": serializable_results
                }, f, ensure_ascii=False, indent=2)
            
            print(f"📁 结果已保存到: {filename}")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {str(e)}")

def main():
    """主函数"""
    try:
        # 创建演示实例
        demo = SimplifiedCOTDIRDemo()
        
        # 运行演示
        results = demo.run_batch_demo()
        
        # 保存结果
        demo.save_results_to_file(results)
        
        print("\n🎉 案例结果演示完成！")
        print("\n📝 说明: 这是一个模拟版本的演示，展示了COT-DIR系统的")
        print("   推理流程和结果格式。实际系统会有更复杂的NLP处理")
        print("   和数学推理能力。")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 