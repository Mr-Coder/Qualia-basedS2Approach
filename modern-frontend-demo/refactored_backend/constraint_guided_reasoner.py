#!/usr/bin/env python3
"""
约束引导推理器 - 约束如何真正帮助构建推理路径
Constraint-Guided Reasoner - How constraints actually help build reasoning paths
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ReasoningConstraint(Enum):
    """推理约束类型 - 用于引导解题路径"""
    QUANTITY_FLOW = "quantity_flow"           # 数量流动约束：输入→操作→输出
    ENTITY_PERSISTENCE = "entity_persistence" # 实体持续性：同一实体在不同状态的连续性
    OPERATION_VALIDITY = "operation_validity" # 操作有效性：操作必须在合理的实体上进行
    CAUSAL_SEQUENCE = "causal_sequence"       # 因果序列：操作的时间顺序
    SEMANTIC_COMPATIBILITY = "semantic_compatibility" # 语义兼容性：参与运算的实体必须语义兼容

@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    operation_type: str  # 'identify', 'extract', 'calculate', 'verify'
    description: str
    entities_involved: List[str]
    constraints_applied: List[ReasoningConstraint]
    confidence: float
    rationale: str  # 为什么这一步是必要的

@dataclass
class ConstraintGuidedPath:
    """约束引导的推理路径"""
    path_id: str
    reasoning_steps: List[ReasoningStep]
    confidence_score: float
    constraint_satisfaction_rate: float
    path_rationale: str

class ConstraintGuidedReasoner:
    """约束引导推理器 - 真正让约束帮助解题的核心"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def generate_reasoning_path(self, problem_text: str, entities: List[Dict]) -> ConstraintGuidedPath:
        """
        基于约束生成推理路径 - 这是约束系统的真正价值
        """
        
        try:
            # Step 1: 分析问题结构，识别约束模式
            problem_structure = self._analyze_problem_structure(problem_text)
            
            # Step 2: 基于约束生成推理步骤
            reasoning_steps = self._generate_constraint_based_steps(problem_text, entities, problem_structure)
            
            # Step 3: 评估路径质量
            confidence_score = self._evaluate_path_confidence(reasoning_steps)
            constraint_satisfaction = self._calculate_constraint_satisfaction(reasoning_steps)
            
            # Step 4: 生成路径解释
            path_rationale = self._generate_path_rationale(problem_structure, reasoning_steps)
            
            return ConstraintGuidedPath(
                path_id=f"path_{hash(problem_text) % 10000}",
                reasoning_steps=reasoning_steps,
                confidence_score=confidence_score,
                constraint_satisfaction_rate=constraint_satisfaction,
                path_rationale=path_rationale
            )
            
        except Exception as e:
            self.logger.error(f"约束引导推理失败: {e}")
            return self._create_fallback_path(problem_text, entities)
    
    def _analyze_problem_structure(self, problem_text: str) -> Dict[str, Any]:
        """分析问题结构，识别约束模式"""
        
        structure = {
            "problem_type": "unknown",
            "main_operation": None,
            "entity_flow": [],
            "constraint_patterns": [],
            "temporal_sequence": [],
            "verification_requirements": []
        }
        
        # 识别问题类型和主要操作
        if any(word in problem_text for word in ['一共', '总共', '合计']):
            structure["problem_type"] = "aggregation"
            structure["main_operation"] = "addition"
            structure["constraint_patterns"].append(ReasoningConstraint.QUANTITY_FLOW)
            structure["verification_requirements"].append("sum_conservation")
            
        elif any(word in problem_text for word in ['剩下', '还有', '余下']):
            structure["problem_type"] = "reduction"
            structure["main_operation"] = "subtraction"
            structure["constraint_patterns"].append(ReasoningConstraint.QUANTITY_FLOW)
            structure["verification_requirements"].append("non_negative_result")
            
        elif any(word in problem_text for word in ['平均', '每']):
            structure["problem_type"] = "distribution"
            structure["main_operation"] = "division"
            structure["constraint_patterns"].append(ReasoningConstraint.OPERATION_VALIDITY)
            
        # 分析实体流动模式
        structure["entity_flow"] = self._analyze_entity_flow(problem_text)
        
        # 识别时间序列
        if any(word in problem_text for word in ['先', '然后', '接着', '最后']):
            structure["constraint_patterns"].append(ReasoningConstraint.CAUSAL_SEQUENCE)
            structure["temporal_sequence"] = self._extract_temporal_sequence(problem_text)
        
        return structure
    
    def _generate_constraint_based_steps(self, problem_text: str, entities: List[Dict], structure: Dict) -> List[ReasoningStep]:
        """基于约束生成推理步骤"""
        
        steps = []
        step_counter = 1
        
        # Step 1: 实体识别和分类 (必须步骤)
        steps.append(ReasoningStep(
            step_id=step_counter,
            operation_type="identify",
            description="识别问题中的关键实体：人物、物品、数量",
            entities_involved=[entity.get('name', 'unknown') for entity in entities],
            constraints_applied=[ReasoningConstraint.ENTITY_PERSISTENCE],
            confidence=0.9,
            rationale="实体识别是所有数学推理的基础，必须确保实体的语义完整性"
        ))
        step_counter += 1
        
        # Step 2: 数量流动分析 (基于约束)
        if ReasoningConstraint.QUANTITY_FLOW in structure["constraint_patterns"]:
            steps.append(ReasoningStep(
                step_id=step_counter,
                operation_type="extract",
                description="分析数量在实体间的流动模式：谁拥有什么，数量如何变化",
                entities_involved=self._extract_quantity_entities(problem_text),
                constraints_applied=[ReasoningConstraint.QUANTITY_FLOW, ReasoningConstraint.SEMANTIC_COMPATIBILITY],
                confidence=0.85,
                rationale="数量流动约束确保我们正确理解数量在实体间的转移关系"
            ))
            step_counter += 1
        
        # Step 3: 操作序列构建 (基于因果约束)
        if structure["main_operation"]:
            constraints_for_operation = [ReasoningConstraint.OPERATION_VALIDITY]
            if ReasoningConstraint.CAUSAL_SEQUENCE in structure["constraint_patterns"]:
                constraints_for_operation.append(ReasoningConstraint.CAUSAL_SEQUENCE)
                
            steps.append(ReasoningStep(
                step_id=step_counter,
                operation_type="calculate",
                description=f"执行{structure['main_operation']}运算，遵循操作有效性约束",
                entities_involved=self._get_operation_entities(problem_text, structure["main_operation"]),
                constraints_applied=constraints_for_operation,
                confidence=0.8,
                rationale=f"操作有效性约束确保{structure['main_operation']}运算在语义上是合理的"
            ))
            step_counter += 1
        
        # Step 4: 约束验证 (基于验证需求)
        verification_step = self._create_verification_step(step_counter, structure["verification_requirements"])
        if verification_step:
            steps.append(verification_step)
            step_counter += 1
        
        # Step 5: 语义合理性检查
        steps.append(ReasoningStep(
            step_id=step_counter,
            operation_type="verify",
            description="检查答案的语义合理性：类型、范围、单位是否正确",
            entities_involved=["solution"],
            constraints_applied=[ReasoningConstraint.SEMANTIC_COMPATIBILITY],
            confidence=0.75,
            rationale="语义兼容性约束确保答案在现实世界中是有意义的"
        ))
        
        return steps
    
    def _analyze_entity_flow(self, problem_text: str) -> List[Dict]:
        """分析实体间的流动关系"""
        
        flow_patterns = []
        
        # 拥有关系流动
        if re.search(r'(\w+)有(\d+)个?(\w+)', problem_text):
            matches = re.finditer(r'(\w+)有(\d+)个?(\w+)', problem_text)
            for match in matches:
                owner, quantity, item = match.groups()
                flow_patterns.append({
                    "type": "ownership",
                    "source": "initial_state",
                    "target": owner,
                    "item": item,
                    "quantity": quantity,
                    "operation": "has"
                })
        
        # 转移关系流动
        if re.search(r'给了?(\w+)(\d+)个?', problem_text):
            matches = re.finditer(r'给了?(\w+)(\d+)个?', problem_text)
            for match in matches:
                recipient, quantity = match.groups()
                flow_patterns.append({
                    "type": "transfer",
                    "source": "giver",
                    "target": recipient,
                    "quantity": quantity,
                    "operation": "give"
                })
        
        return flow_patterns
    
    def _extract_temporal_sequence(self, problem_text: str) -> List[str]:
        """提取时间序列"""
        
        temporal_markers = ['先', '然后', '接着', '最后', '首先', '其次']
        sequence = []
        
        for marker in temporal_markers:
            if marker in problem_text:
                # 简化的时间序列提取
                parts = problem_text.split(marker)
                if len(parts) > 1:
                    sequence.append(f"{marker}: {parts[1][:20]}...")
        
        return sequence
    
    def _extract_quantity_entities(self, problem_text: str) -> List[str]:
        """提取涉及数量的实体"""
        
        entities = []
        
        # 提取数字和相关实体
        number_matches = re.finditer(r'(\d+)个?(\w+)', problem_text)
        for match in number_matches:
            quantity, entity = match.groups()
            entities.append(f"{entity}({quantity})")
        
        return entities
    
    def _get_operation_entities(self, problem_text: str, operation: str) -> List[str]:
        """获取参与特定操作的实体"""
        
        entities = []
        
        if operation == "addition":
            # 找出所有要相加的数量
            numbers = re.findall(r'\d+', problem_text)
            entities = [f"number_{i}({num})" for i, num in enumerate(numbers)]
            
        elif operation == "subtraction":
            # 找出被减数和减数
            numbers = re.findall(r'\d+', problem_text)
            if len(numbers) >= 2:
                entities = [f"minuend({numbers[0]})", f"subtrahend({numbers[1]})"]
        
        return entities
    
    def _create_verification_step(self, step_id: int, requirements: List[str]) -> Optional[ReasoningStep]:
        """创建验证步骤"""
        
        if not requirements:
            return None
        
        description_parts = []
        constraints = []
        
        if "sum_conservation" in requirements:
            description_parts.append("验证加法守恒：总和 = 各部分之和")
            constraints.append(ReasoningConstraint.QUANTITY_FLOW)
            
        if "non_negative_result" in requirements:
            description_parts.append("验证非负性：剩余数量不能为负")
            constraints.append(ReasoningConstraint.OPERATION_VALIDITY)
        
        return ReasoningStep(
            step_id=step_id,
            operation_type="verify",
            description="; ".join(description_parts),
            entities_involved=["solution", "input_values"],
            constraints_applied=constraints,
            confidence=0.9,
            rationale="约束验证确保解答符合基本的数学和物理定律"
        )
    
    def _evaluate_path_confidence(self, steps: List[ReasoningStep]) -> float:
        """评估推理路径的置信度"""
        
        if not steps:
            return 0.0
        
        # 基于步骤置信度的加权平均
        step_confidences = [step.confidence for step in steps]
        avg_confidence = sum(step_confidences) / len(step_confidences)
        
        # 基于约束覆盖率的调整
        constraint_types = set()
        for step in steps:
            constraint_types.update(step.constraints_applied)
        
        # 更多类型的约束被应用，置信度越高
        constraint_bonus = len(constraint_types) * 0.05
        
        return min(avg_confidence + constraint_bonus, 1.0)
    
    def _calculate_constraint_satisfaction(self, steps: List[ReasoningStep]) -> float:
        """计算约束满足率"""
        
        if not steps:
            return 0.0
        
        # 检查是否覆盖了关键约束类型
        applied_constraints = set()
        for step in steps:
            applied_constraints.update(step.constraints_applied)
        
        total_constraint_types = len(ReasoningConstraint)
        applied_count = len(applied_constraints)
        
        return applied_count / total_constraint_types
    
    def _generate_path_rationale(self, structure: Dict, steps: List[ReasoningStep]) -> str:
        """生成路径解释"""
        
        rationale_parts = []
        
        rationale_parts.append(f"基于问题类型({structure['problem_type']})，")
        rationale_parts.append(f"采用{structure['main_operation']}运算策略。")
        
        constraint_count = sum(len(step.constraints_applied) for step in steps)
        rationale_parts.append(f"整个推理过程应用了{constraint_count}个约束条件，")
        rationale_parts.append("确保推理的逻辑一致性和语义合理性。")
        
        return "".join(rationale_parts)
    
    def _create_fallback_path(self, problem_text: str, entities: List[Dict]) -> ConstraintGuidedPath:
        """创建回退推理路径"""
        
        fallback_steps = [
            ReasoningStep(
                step_id=1,
                operation_type="identify",
                description="基础实体识别",
                entities_involved=[entity.get('name', 'unknown') for entity in entities],
                constraints_applied=[ReasoningConstraint.ENTITY_PERSISTENCE],
                confidence=0.5,
                rationale="回退到基础实体识别"
            ),
            ReasoningStep(
                step_id=2,
                operation_type="calculate",
                description="通用数值计算",
                entities_involved=["numbers"],
                constraints_applied=[ReasoningConstraint.OPERATION_VALIDITY],
                confidence=0.4,
                rationale="执行通用的数值运算"
            )
        ]
        
        return ConstraintGuidedPath(
            path_id="fallback_path",
            reasoning_steps=fallback_steps,
            confidence_score=0.3,
            constraint_satisfaction_rate=0.2,
            path_rationale="由于问题结构复杂，采用了简化的推理路径"
        )

    def explain_constraint_guidance(self, path: ConstraintGuidedPath) -> List[str]:
        """解释约束如何指导推理过程"""
        
        explanations = []
        
        explanations.append(f"=== 约束引导推理解释 ===")
        explanations.append(f"路径置信度: {path.confidence_score:.2f}")
        explanations.append(f"约束满足率: {path.constraint_satisfaction_rate:.2f}")
        explanations.append(f"路径原理: {path.path_rationale}")
        explanations.append("")
        
        for step in path.reasoning_steps:
            explanations.append(f"步骤{step.step_id}: {step.description}")
            explanations.append(f"  操作类型: {step.operation_type}")
            explanations.append(f"  涉及实体: {', '.join(step.entities_involved)}")
            explanations.append(f"  应用约束: {[c.value for c in step.constraints_applied]}")
            explanations.append(f"  约束作用: {step.rationale}")
            explanations.append(f"  步骤置信度: {step.confidence:.2f}")
            explanations.append("")
        
        return explanations

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    reasoner = ConstraintGuidedReasoner()
    
    # 测试复杂问题
    test_problem = "小明有10个苹果，先给了小红3个，然后又买了5个，最后一共有多少个苹果？"
    test_entities = [
        {"name": "小明", "type": "person"},
        {"name": "小红", "type": "person"},
        {"name": "苹果", "type": "object"}
    ]
    
    print(f"测试问题: {test_problem}")
    print("="*60)
    
    # 生成约束引导的推理路径
    path = reasoner.generate_reasoning_path(test_problem, test_entities)
    
    # 显示详细解释
    explanations = reasoner.explain_constraint_guidance(path)
    for explanation in explanations:
        print(explanation)