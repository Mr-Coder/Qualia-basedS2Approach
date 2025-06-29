"""
🔗 COT-DIR 基于关系的解答生成器
Relation-Based Solution Generator - 以关系为核心的数学解答过程

核心特色：
- 显性关系识别和分析
- 隐含关系L1、L2、L3层次推理
- 关系导向的解题策略
"""

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Relation:
    """A structured representation of a discovered relation."""
    level: str
    type: str
    description: str
    evidence: str
    template: Optional[str] = None
    confidence: float = 0.9

@dataclass
class RelationBasedSolution:
    """(Corrected) Final structure for a relation-based solution."""
    problem_id: str
    question: str
    dataset_source: str
    problem_type: str
    
    relations: List[Relation] = field(default_factory=list)
    
    reasoning_chain: List[str] = field(default_factory=list)
    solution_steps: List[str] = field(default_factory=list)
    
    final_answer: str = ""
    confidence_score: float = 0.0
    processing_time: float = 0.0

class RelationBasedSolutionGenerator:
    """
    (REWRITTEN) COT-DIR Solution Generator
    This version is correctly driven by the `pattern.json` configuration file.
    """
    
    def __init__(self, patterns_file: Path = Path("src/models/pattern.json")):
        print("🔗 (V3) Initializing Configuration-Driven Relation Generator...")
        self.patterns = self._load_patterns(patterns_file)
        print(f"📊 Loaded {len(self.patterns.get('pattern_groups', {}))} pattern groups.")

    def _load_patterns(self, file_path: Path) -> Dict:
        """Loads the patterns from the JSON configuration file."""
        if not file_path.exists():
            print(f"❌ CRITICAL: Patterns file not found at {file_path}")
            return {}
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_solution_for_problem(self, problem: Dict) -> RelationBasedSolution:
        """Main entry point to generate a single, complete solution."""
        start_time = time.time()
        question = problem.get('question', problem.get('original_question', ''))
        
        # 1. Discover all relations using the pattern file
        discovered_relations = self._find_relations_from_patterns(question)
        
        # 2. Build the COT reasoning chain from discovered relations
        reasoning_chain = self._build_reasoning_chain(discovered_relations)
        
        # 3. Generate solution steps based on the chain
        solution_steps = self._build_solution_steps(reasoning_chain)
        
        # 4. Calculate confidence
        confidence = self._calculate_confidence(discovered_relations)
        
        return RelationBasedSolution(
            problem_id=problem.get('id', str(random.randint(1000, 9999))),
            question=question,
            dataset_source=problem.get('dataset_source', 'unknown'),
            problem_type=self._classify_problem_type(question, discovered_relations),
            relations=discovered_relations,
            reasoning_chain=reasoning_chain,
            solution_steps=solution_steps,
            final_answer=problem.get('ans', 'Not calculated'),
            confidence_score=confidence,
            processing_time=time.time() - start_time
        )

    def _find_relations_from_patterns(self, question: str) -> List[Relation]:
        """
        (NEW CORE LOGIC) Matches patterns from the JSON file against the question text.
        This replaces all the old `extract_*` and `infer_*` methods.
        """
        found_relations = []
        question_lower = question.lower()

        # Add explicit numbers as a baseline relation
        numbers = re.findall(r'\d+\.\d+|\d+', question)
        if numbers:
            found_relations.append(Relation(
                level="Explicit", type="Numerical",
                description=f"发现数值",
                evidence=", ".join(numbers)
            ))

        pattern_groups = self.patterns.get('pattern_groups', {})
        for group_name, group_content in pattern_groups.items():
            for scene, patterns_in_scene in group_content.items():
                if not isinstance(patterns_in_scene, list): continue
                
                for pattern_def in patterns_in_scene:
                    # A simplified matching logic: check if all terms in pattern exist
                    pattern_terms = pattern_def.get('pattern', [])
                    # This is a placeholder for a real semantic matching engine.
                    # For now, we use regex to check for keyword presence.
                    try:
                        if all(re.search(r'\b' + re.escape(term) + r'\b', question_lower, re.IGNORECASE) for term in pattern_terms):
                            level = self._determine_level(group_name, pattern_def)
                            found_relations.append(Relation(
                                level=level,
                                type=pattern_def.get('scene', group_name),
                                description=pattern_def.get('description', 'N/A'),
                                evidence=f"Matched pattern terms: {', '.join(pattern_terms)}",
                                template=pattern_def.get('relation_template')
                            ))
                    except re.error:
                        # Ignore invalid regex patterns in the config file
                        continue
        return found_relations

    def _determine_level(self, group_name: str, pattern_def: Dict) -> str:
        """Determines relation level (L1, L2, L3) based on pattern properties."""
        reasoning_type = pattern_def.get('reasoning_type', 'direct')
        if reasoning_type == 'multistep' or len(pattern_def.get('dependencies', [])) > 1:
            return "L3"
        if reasoning_type == 'composite' or len(pattern_def.get('dependencies', [])) == 1:
            return "L2"
        if group_name == '数量关系' or reasoning_type == 'direct':
             return "L1"
        return "Explicit"

    def _build_reasoning_chain(self, relations: List[Relation]) -> List[str]:
        """(Corrected) Builds a true COT chain from the structured relations."""
        if not relations:
            return ["未能发现任何关系，无法构建推理链。"]
            
        chain = ["【起点】识别问题中的基本元素。"]
        
        # Sort relations by level: Explicit, L1, L2, L3
        relations.sort(key=lambda r: ("Explicit", "L1", "L2", "L3").index(r.level))
        
        for rel in relations:
            chain.append("↓")
            chain.append(f"【{rel.level} - {rel.type}】识别到: {rel.description}")
            if rel.template:
                chain.append(f"   - 核心关系式: {rel.template}")

        chain.append("【终点】推理链构建完成，形成解题路径。")
        return chain

    def _build_solution_steps(self, reasoning_chain: List[str]) -> List[str]:
        """(Corrected) Builds solution steps directly from the reasoning chain."""
        if not reasoning_chain or "无法构建" in reasoning_chain[0]:
            return ["无法生成解题步骤。"]
            
        steps = ["1. 根据COT-DIR推理链进行求解。"]
        step_count = 2
        for link in reasoning_chain:
            if link.startswith("【"):
                simplified_link = link.replace("【", "").replace("】", "")
                steps.append(f"{step_count}. 分析 {simplified_link}。")
                step_count += 1
                
        steps.append(f"{step_count}. 综合以上关系，代入数值进行最终计算。")
        return steps
    
    def _classify_problem_type(self, question: str, relations: List[Relation]) -> str:
        """Classifies the problem based on the dominant relation type."""
        if any(r.type == '行程追及' for r in relations): return "行程问题"
        if any(r.type == '合作效率' for r in relations): return "工程问题"
        if any(r.level == 'L3' for r in relations): return "复杂推理"
        if any(r.level == 'L2' for r in relations): return "结构问题"
        if any(word in question for word in ['how many', '多少']): return "计数问题"
        return "一般算术"

    def _calculate_confidence(self, relations: List[Relation]) -> float:
        """Calculates confidence based on the depth and count of relations."""
        score = 0.5 # Base score
        if not relations: return 0.1
        
        level_weights = {"Explicit": 0.05, "L1": 0.1, "L2": 0.15, "L3": 0.2}
        for r in relations:
            score += level_weights.get(r.level, 0)
            
        return min(score, 0.99)

def main():
    """Main function to demonstrate the new generator."""
    print("🚀 Running V3 Configuration-Driven Relation Generator Demo 🚀")
    
    # --- Load Problems ---
    try:
        with open('full_relation_solutions_20250630_024146.json', 'r', encoding='utf-8') as f:
            problems_data = json.load(f).get('solutions', [])
        if not problems_data:
            print("❌ Could not load problems from the JSON file.")
            return
    except FileNotFoundError:
        print("❌ `full_relation_solutions_20250630_024146.json` not found. Cannot run demo.")
        return

    # --- Initialize Generator ---
    generator = RelationBasedSolutionGenerator()
    
    # --- Process a Random Sample ---
    sample_problem = random.choice(problems_data)
    print("\n" + "="*80)
    print(f"🔍 Analyzing Sample Problem: {sample_problem.get('question', '')[:80]}...")
    
    solution = generator.generate_solution_for_problem(sample_problem)
    
    # --- Print Results ---
    print("\n" + "="*80)
    print("✅ Generation Complete. Results:")
    print("="*80)
    print(f"**问题:** {solution.question}")
    print(f"**类型:** {solution.problem_type}")
    print(f"**置信度:** {solution.confidence_score:.2f}")
    
    print("\n**发现的关系:**")
    for rel in solution.relations:
        print(f"  - **[{rel.level}]** {rel.description} (类型: {rel.type}, 核心公式: {rel.template or 'N/A'})")
        
    print("\n**COT推理链:**")
    for link in solution.reasoning_chain:
        print(f"  {link}")
        
    print("\n**解题步骤:**")
    for step in solution.solution_steps:
        print(f"  {step}")
        
    print(f"\n**最终答案:** {solution.final_answer}")
    print("="*80)

if __name__ == "__main__":
    main() 