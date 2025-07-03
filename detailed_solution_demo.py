#!/usr/bin/env python3
"""
详细解答过程演示程序
展示每道数学题目的完整COT-DIR推理过程
"""

import json
import time
from typing import Dict, List, Any


class DetailedSolutionDemo:
    """详细解答过程演示类"""
    
    def __init__(self):
        """初始化演示系统"""
        print("🚀 初始化详细解答过程演示系统...")
        print("="*60)
        
        # 加载已有的详细案例结果
        self.detailed_results = self._load_detailed_results()
        print(f"✅ 成功加载 {len(self.detailed_results)} 个详细案例")
        print()
    
    def _load_detailed_results(self) -> List[Dict]:
        """加载详细案例结果"""
        try:
            with open('detailed_case_results.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('detailed_cases', [])
        except FileNotFoundError:
            print("⚠️  详细案例结果文件未找到，使用示例数据")
            return self._get_sample_cases()
    
    def _get_sample_cases(self) -> List[Dict]:
        """获取示例案例数据"""
        return [
            {
                "case_id": "sample_001",
                "case_info": {
                    "language": "中文",
                    "problem_statement": "小明有15个苹果，他给了小红5个，又买了8个，现在小明有多少个苹果？",
                    "expected_answer": "18",
                    "problem_type": "加减运算",
                    "difficulty": "简单",
                    "complexity_level": "L2"
                },
                "reasoning_process": {
                    "step_1_entity_extraction": {
                        "entities": [
                            {"name": "小明", "type": "人物", "value": "小明"},
                            {"name": "小红", "type": "人物", "value": "小红"},
                            {"name": "苹果", "type": "物品", "value": "苹果"},
                            {"name": "15", "type": "数量", "value": 15},
                            {"name": "5", "type": "数量", "value": 5},
                            {"name": "8", "type": "数量", "value": 8}
                        ]
                    },
                    "step_2_relation_discovery": {
                        "relations": [
                            {"type": "转移关系", "source": "小明", "target": "小红", "operation": "减法"},
                            {"type": "获得关系", "source": "小明", "target": "苹果", "operation": "加法"}
                        ]
                    },
                    "step_3_multi_layer_reasoning": {
                        "reasoning_steps": [
                            {"layer": "L1", "description": "基础信息提取", "operation": "文本分析"},
                            {"layer": "L2", "description": "关系建模", "operation": "关系映射"},
                            {"layer": "L3", "description": "执行减法", "operation": "15 - 5 = 10"},
                            {"layer": "L3", "description": "执行加法", "operation": "10 + 8 = 18"}
                        ]
                    }
                },
                "solution_process": {
                    "solution_steps": [
                        {"step": 1, "description": "理解题目", "content": "小明最初有15个苹果", "mathematical_expression": "初始 = 15"},
                        {"step": 2, "description": "第一个操作", "content": "给了小红5个", "mathematical_expression": "15 - 5 = 10"},
                        {"step": 3, "description": "第二个操作", "content": "又买了8个", "mathematical_expression": "10 + 8 = 18"},
                        {"step": 4, "description": "最终答案", "content": "现在有18个苹果", "mathematical_expression": "答案 = 18"}
                    ]
                },
                "final_result": {
                    "predicted_answer": "18",
                    "expected_answer": "18",
                    "is_correct": True,
                    "confidence_score": 88.5
                }
            }
        ]
    
    def display_case_solution(self, case: Dict[str, Any], case_index: int):
        """展示单个案例的详细解答过程"""
        case_info = case.get('case_info', {})
        reasoning = case.get('reasoning_process', {})
        solution = case.get('solution_process', {})
        result = case.get('final_result', {})
        
        print(f"【案例 {case_index}】")
        print("="*60)
        
        # 1. 题目信息
        print("📝 题目信息:")
        print(f"   语言: {case_info.get('language', '未知')}")
        print(f"   题目: {case_info.get('problem_statement', '未知')}")
        print(f"   类型: {case_info.get('problem_type', '未知')}")
        print(f"   难度: {case_info.get('difficulty', '未知')}")
        print(f"   复杂度: {case_info.get('complexity_level', '未知')}")
        print(f"   预期答案: {case_info.get('expected_answer', '未知')}")
        print()
        
        # 2. COT-DIR推理过程
        print("🧠 COT-DIR推理过程:")
        print("-"*40)
        
        # 步骤1: 实体提取
        entities = reasoning.get('step_1_entity_extraction', {}).get('entities', [])
        print("📍 步骤1: 实体提取 (IRD模块)")
        if entities:
            print(f"   发现 {len(entities)} 个实体:")
            for entity in entities:
                print(f"     • {entity.get('name', '未知')} ({entity.get('type', '未知')})")
        else:
            print("   未发现实体")
        print()
        
        # 步骤2: 关系发现
        relations = reasoning.get('step_2_relation_discovery', {}).get('relations', [])
        print("📍 步骤2: 关系发现 (IRD模块)")
        if relations:
            print(f"   发现 {len(relations)} 个关系:")
            for relation in relations:
                print(f"     • {relation.get('type', '未知')}: {relation.get('source', '未知')} → {relation.get('target', '未知')} ({relation.get('operation', '未知')})")
        else:
            print("   未发现关系")
        print()
        
        # 步骤3: 多层推理
        reasoning_steps = reasoning.get('step_3_multi_layer_reasoning', {}).get('reasoning_steps', [])
        print("📍 步骤3: 多层推理 (MLR模块)")
        if reasoning_steps:
            print(f"   执行 {len(reasoning_steps)} 个推理步骤:")
            for i, step in enumerate(reasoning_steps, 1):
                layer = step.get('layer', '未知')
                desc = step.get('description', '未知')
                op = step.get('operation', '未知')
                print(f"     {i}. [{layer}] {desc} → {op}")
        else:
            print("   无推理步骤")
        print()
        
        # 3. 详细解答步骤
        print("📖 详细解答步骤:")
        print("-"*40)
        solution_steps = solution.get('solution_steps', [])
        if solution_steps:
            for step in solution_steps:
                step_num = step.get('step', 0)
                desc = step.get('description', '未知')
                content = step.get('content', '未知')
                expr = step.get('mathematical_expression', '未知')
                print(f"   步骤{step_num}: {desc}")
                print(f"     内容: {content}")
                print(f"     数学表达式: {expr}")
                print()
        else:
            print("   无详细解答步骤")
        
        # 4. 最终结果
        print("🎯 最终结果:")
        print("-"*40)
        predicted = result.get('predicted_answer', '未知')
        expected = result.get('expected_answer', '未知')
        is_correct = result.get('is_correct', False)
        confidence = result.get('confidence_score', 0)
        
        print(f"   预测答案: {predicted}")
        print(f"   预期答案: {expected}")
        print(f"   是否正确: {'✅ 正确' if is_correct else '❌ 错误'}")
        print(f"   置信度: {confidence:.2f}%")
        print()
        
        print("="*60)
        print()
    
    def run_demo(self):
        """运行演示"""
        print("🎯 开始详细解答过程演示")
        print("="*60)
        print()
        
        if not self.detailed_results:
            print("❌ 没有可用的案例数据")
            return
        
        for i, case in enumerate(self.detailed_results, 1):
            self.display_case_solution(case, i)
            
            if i < len(self.detailed_results):
                input("按回车键继续下一个案例...")
                print("\n" + "="*60 + "\n")
        
        print("🎉 所有案例演示完成！")


def main():
    """主函数"""
    demo = DetailedSolutionDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
