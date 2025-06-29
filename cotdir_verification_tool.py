"""
🧠 COT-DIR 真实性与一致性验证工具 (已修正)
COT-DIR Authenticity and Consistency Verification Tool (Corrected)

目标:
- 深入审查生成的COT-DIR解答，确保其符合核心思想
- 提供一个诚实的、基于数据的验证结论
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


class COTDIRVerifier:
    """COT-DIR解答验证器"""

    def __init__(self, solutions_file: Path):
        """初始化验证器"""
        print(f"🔬 初始化COT-DIR验证器，加载文件: {solutions_file}")
        self.solutions = self._load_solutions(solutions_file)
        if not self.solutions:
            raise ValueError("未能加载或解析解答文件")
        print(f"📊 已加载 {len(self.solutions):,} 个解答进行验证")
        print("="*80)

    def _load_solutions(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载解答文件"""
        if not file_path.exists():
            print(f"❌ 错误: 文件不存在 {file_path}")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('solutions', [])

    def verify_random_samples(self, num_samples: int = 3):
        """验证随机抽取的样本"""
        if len(self.solutions) < num_samples:
            print("样本数量不足，将验证所有可用解答")
            samples = self.solutions
        else:
            samples = random.sample(self.solutions, num_samples)

        print(f"🕵️ 开始验证 {len(samples)} 个随机样本...")
        for i, solution in enumerate(samples, 1):
            print("\n" + "="*80)
            print(f"🔍 正在验证样本 {i}/{len(samples)} (ID: {solution.get('id', 'N/A')})")
            print("="*80)
            self.run_full_verification(solution)

    def run_full_verification(self, solution: Dict[str, Any]):
        """对单个解答运行完整的验证流程"""
        self._print_problem(solution)
        self._verify_relations_full_print(solution)
        self._verify_reasoning_chain_full_print(solution)
        self._verify_solution_steps_full_print(solution)
        self._print_verification_summary(solution)

    def _print_problem(self, solution: Dict[str, Any]):
        """打印问题原文"""
        print("📋 **1. 问题原文**")
        print("─"*40)
        print(f"   **题目:** {solution.get('question', '无题目信息')}")
        print(f"   **来源:** {solution.get('dataset_source', '未知')}")
        print(f"   **预期答案:** {solution.get('final_answer', '未知')}")
        print("─"*40 + "\n")

    def _verify_relations_full_print(self, solution: Dict[str, Any]):
        """验证所有关系"""
        print("🕵️ **2. 关系发现验证**")
        is_overall_valid, comment = self._verify_relations_summary(solution)
        # Detailed printouts
        for rel in solution.get('explicit_relations', []):
            is_valid, cmt = self._check_explicit_relation(rel, solution.get('question', ''))
            print(f"   - [ {'✅' if is_valid else '❌'} ] 显性: {rel.get('description','')} | {cmt}")
        for level in ["L1", "L2", "L3"]:
            for rel in solution.get(f'implicit_relations_{level}', []):
                is_valid, cmt = self._check_implicit_relation(rel, level)
                print(f"   - [ {'✅' if is_valid else '❌'} ] {level}: {rel.get('description','')} | {cmt}")
        print(f"\n   **关系发现结论:** {'✅' if is_overall_valid else '❌'} {comment}\n")

    def _verify_reasoning_chain_full_print(self, solution: Dict[str, Any]):
        """验证推理链"""
        print("🕵️ **3. COT推理链验证**")
        is_valid, comment = self._verify_reasoning_chain_summary(solution)
        print(f"   - [ {'✅' if is_valid else '❌'} ] 连贯性检查: {comment}")
        print(f"   **推理链:** {solution.get('relation_reasoning_chain', [])}")
        print(f"\n   **推理链验证结论:** {'✅' if is_valid else '❌'} {comment}\n")

    def _verify_solution_steps_full_print(self, solution: Dict[str, Any]):
        """验证解题步骤"""
        print("🕵️ **4. 关系导向解题过程验证**")
        is_valid, comment = self._verify_solution_steps_summary(solution)
        print(f"   - [ {'✅' if is_valid else '❌'} ] 一致性检查: {comment}")
        print(f"   **解题步骤:** {solution.get('relation_based_solution_steps', [])}")
        print(f"   **最终答案:** {solution.get('final_answer', '')}")
        print(f"\n   **解题过程验证结论:** {'✅' if is_valid else '❌'} {comment}\n")

    def _verify_relations_summary(self, solution: Dict[str, Any]) -> Tuple[bool, str]:
        question = solution.get('question', '')
        for rel in solution.get('explicit_relations', []):
            is_valid, comment = self._check_explicit_relation(rel, question)
            if not is_valid: return False, f"显性关系追踪失败 ({comment})"
        
        for level in ["L1", "L2", "L3"]:
            for rel in solution.get(f'implicit_relations_{level}', []):
                is_valid, comment = self._check_implicit_relation(rel, level)
                if not is_valid: return False, f"{level}关系结构无效 ({comment})"
        return True, "关系准确可追溯"

    def _verify_reasoning_chain_summary(self, solution: Dict[str, Any]) -> Tuple[bool, str]:
        chain = solution.get('relation_reasoning_chain', [])
        if not chain or len(chain) < 2 or "构建" in chain[0]: # Check for placeholder
             return False, "推理链为空、过短或为占位符"
        return True, "推理链结构完整"

    def _verify_solution_steps_summary(self, solution: Dict[str, Any]) -> Tuple[bool, str]:
        steps = solution.get('relation_based_solution_steps', [])
        final_answer_str = solution.get('final_answer', '')
        if not steps or "基于关系" in steps[0]: # Check for placeholder
            return False, "解题步骤为空或为占位符"

        last_step = steps[-1]
        # Regex to find the numerical part of answers like <<...>>8 or <<...>>8.0
        answer_nums = re.findall(r'<<[^>]+>>\s*(\d+(?:\.\d+)?)', final_answer_str)
        if not answer_nums:
            return False, "最终答案中无有效数值" # Cannot verify if no number in answer
        
        final_num = answer_nums[-1]
        if final_num not in last_step:
            return False, f"最终答案数值'{final_num}'未在最后一步'{last_step}'中找到"
        return True, "解题步骤与答案一致"

    def _check_explicit_relation(self, rel: Dict[str, Any], question: str) -> Tuple[bool, str]:
        evidence = rel.get('evidence', '')
        if not evidence or '证据' in evidence: # check for placeholder
             return False, "缺乏或为占位符证据"
        
        # A simple check: does the evidence appear in the question?
        # A more complex check could involve tokenization and stemming.
        if evidence.lower() in question.lower():
            return True, "证据可追溯"
        return False, f"证据 '{evidence}' 无法在原文追溯"

    def _check_implicit_relation(self, rel: Dict[str, Any], level: str) -> Tuple[bool, str]:
        if 'reasoning' not in rel or not rel['reasoning']: return False, "缺乏推理过程"
        if 'mathematical_implication' not in rel or not rel['mathematical_implication']: return False, "缺乏数学含义"
        if level in ["L2", "L3"] and 'dependency' not in rel: return False, "缺乏依赖项"
        return True, f"L{level}结构完整"

    def _print_verification_summary(self, solution: Dict[str, Any]):
        """Prints a dynamic, honest verification summary."""
        print("🏆 **5. 最终验证结论**")
        print("─"*40)
        
        relations_ok, rel_comment = self._verify_relations_summary(solution)
        chain_ok, chain_comment = self._verify_reasoning_chain_summary(solution)
        steps_ok, steps_comment = self._verify_solution_steps_summary(solution)

        print(f"   - **关系发现:** {'✅ ' + rel_comment if relations_ok else '❌ ' + rel_comment}")
        print(f"   - **推理链:** {'✅ ' + chain_comment if chain_ok else '❌ ' + chain_comment}")
        print(f"   - **解题过程:** {'✅ ' + steps_comment if steps_ok else '❌ ' + steps_comment}")
        
        if relations_ok and chain_ok and steps_ok:
            final_conclusion = "✅ 该解答符合COT-DIR核心思想。"
        else:
            final_conclusion = "❌ 该解答未能完全符合COT-DIR核心思想，存在明显缺陷。"
            
        print(f"\n   **最终诚信结论: {final_conclusion}**")
        print("─"*40)

def main():
    """主函数"""
    import re  # 引入re模块
    
    print("🚀 COT-DIR 解答真实性与一致性验证程序 🚀")
    
    # 自动查找最新的关系解答文件
    try:
        relation_files = list(Path(".").glob("*relation_solutions_*.json"))
        if not relation_files:
            print("❌ 未找到任何 `*relation_solutions_*.json` 文件。请将脚本放在正确目录下。")
            return
        
        latest_file = max(relation_files, key=lambda p: p.stat().st_mtime)
        verifier = COTDIRVerifier(latest_file)
        verifier.verify_random_samples()
    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")

if __name__ == "__main__":
    main()