"""
多层级推理处理器 (Multi-Level Reasoning Processor)

专注于执行L0-L3不同复杂度级别的推理。
这是COT-DIR算法的第二个核心组件。
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..qs2_enhancement.enhanced_ird_engine import EnhancedRelation, RelationType
from .deep_implicit_engine import (
    DeepImplicitEngine, 
    DeepImplicitRelation, 
    ImplicitConstraint,
    SemanticRelationType,
    ConstraintType,
    RelationDepth
)


class ComplexityLevel(Enum):
    """推理复杂度级别"""
    L0_EXPLICIT = "L0"      # 显式推理，无需隐式关系
    L1_SHALLOW = "L1"       # 浅层推理，简单隐式关系
    L2_MEDIUM = "L2"        # 中等推理，多步隐式关系
    L3_DEEP = "L3"          # 深层推理，复杂隐式关系链


class ReasoningStepType(Enum):
    """推理步骤类型"""
    INITIALIZATION = "initialization"    # 初始化
    RELATION_ANALYSIS = "relation_analysis"  # 关系分析
    CALCULATION = "calculation"         # 计算
    VERIFICATION = "verification"       # 验证
    CONCLUSION = "conclusion"           # 结论


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    step_type: ReasoningStepType
    description: str
    operation: str
    input_values: List[Union[float, str]]
    output_value: Union[float, str, None]
    confidence: float
    depends_on: List[int]  # 依赖的前序步骤ID
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "step_id": self.step_id,
            "type": self.step_type.value,
            "description": self.description,
            "operation": self.operation,
            "input_values": self.input_values,
            "output_value": self.output_value,
            "confidence": self.confidence,
            "depends_on": self.depends_on,
            "metadata": self.metadata
        }


@dataclass
class MLRResult:
    """多层级推理结果"""
    success: bool
    complexity_level: ComplexityLevel
    reasoning_steps: List[ReasoningStep]
    final_answer: Union[float, str, None]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    
    def get_steps_by_type(self, step_type: ReasoningStepType) -> List[ReasoningStep]:
        """按类型获取步骤"""
        return [step for step in self.reasoning_steps if step.step_type == step_type]
    
    def get_calculation_chain(self) -> List[ReasoningStep]:
        """获取计算链"""
        return [step for step in self.reasoning_steps 
                if step.step_type == ReasoningStepType.CALCULATION]


class MultiLevelReasoningProcessor:
    """多层级推理处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化MLR处理器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 配置参数
        self.config = config or {}
        self.max_reasoning_depth = self.config.get("max_reasoning_depth", 10)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_verification = self.config.get("enable_verification", True)
        
        # 初始化深度隐含关系引擎
        self.deep_engine = DeepImplicitEngine(self.config.get("deep_engine", {}))
        
        # 推理策略配置
        self.complexity_thresholds = {
            ComplexityLevel.L0_EXPLICIT: 0.2,
            ComplexityLevel.L1_SHALLOW: 0.4,
            ComplexityLevel.L2_MEDIUM: 0.7,
            ComplexityLevel.L3_DEEP: 1.0
        }
        
        # 统计信息
        self.stats = {
            "total_processed": 0,
            "complexity_distribution": {level.value: 0 for level in ComplexityLevel},
            "average_steps": 0.0,
            "average_confidence": 0.0,
            "success_rate": 0.0,
            "deep_relations_discovered": 0,
            "implicit_constraints_found": 0
        }
        
        self.logger.info("多层级推理处理器初始化完成，已集成深度隐含关系引擎")
    
    def execute_reasoning(
        self, 
        problem_text: str, 
        relations: List[Any] = None, 
        context: Optional[Dict[str, Any]] = None
    ) -> MLRResult:
        """
        执行多层级推理
        
        Args:
            problem_text: 问题文本
            relations: 输入关系列表（可选）
            context: 可选的上下文信息
            
        Returns:
            MLRResult: 增强的推理结果，包含深度隐含关系和约束
        """
        start_time = time.time()
        relations = relations or []
        
        try:
            self.logger.info(f"开始增强多层级推理: {problem_text[:50]}...")
            
            # 第零步：提取基础实体信息
            entities = self._extract_basic_entities(problem_text)
            
            # 第一步：深度隐含关系发现
            deep_relations, implicit_constraints = self.deep_engine.discover_deep_relations(
                problem_text, entities, []
            )
            
            # 更新统计
            self.stats["deep_relations_discovered"] += len(deep_relations)
            self.stats["implicit_constraints_found"] += len(implicit_constraints)
            
            # 第二步：确定复杂度级别（考虑深度关系）
            complexity_level = self._determine_enhanced_complexity_level(
                problem_text, relations, deep_relations, implicit_constraints
            )
            
            # 第三步：初始化增强推理上下文
            reasoning_context = self._initialize_enhanced_reasoning_context(
                problem_text, relations, complexity_level, context, 
                deep_relations, implicit_constraints, entities
            )
            
            # 第四步：执行增强分层推理
            reasoning_steps = self._execute_enhanced_layered_reasoning(
                reasoning_context, complexity_level
            )
            
            # 第五步：验证推理结果
            if self.enable_verification:
                verification_steps = self._verify_enhanced_reasoning_chain(
                    reasoning_steps, deep_relations, implicit_constraints
                )
                reasoning_steps.extend(verification_steps)
            
            # 第六步：生成最终答案
            final_answer, answer_confidence = self._generate_enhanced_final_answer(
                reasoning_steps, deep_relations
            )
            
            # 计算整体置信度
            overall_confidence = self._calculate_enhanced_overall_confidence(
                reasoning_steps, deep_relations, implicit_constraints
            )
            
            # 更新统计信息
            self._update_enhanced_stats(
                complexity_level, reasoning_steps, overall_confidence >= self.confidence_threshold,
                deep_relations, implicit_constraints
            )
            
            processing_time = time.time() - start_time
            
            result = MLRResult(
                success=answer_confidence >= self.confidence_threshold,
                complexity_level=complexity_level,
                reasoning_steps=reasoning_steps,
                final_answer=final_answer,
                confidence_score=overall_confidence,
                processing_time=processing_time,
                metadata={
                    "relations_used": len([r for r in relations if hasattr(r, 'confidence') and r.confidence >= 0.5]),
                    "deep_relations_discovered": len(deep_relations),
                    "implicit_constraints_found": len(implicit_constraints),
                    "step_count": len(reasoning_steps),
                    "verification_enabled": self.enable_verification,
                    "entities_extracted": len(entities),
                    "context": context or {},
                    "frontend_visualization_data": self._prepare_frontend_data(
                        entities, deep_relations, implicit_constraints, reasoning_steps
                    )
                }
            )
            
            self.logger.info(f"增强MLR完成: 复杂度{complexity_level.value}, {len(reasoning_steps)}步, "
                           f"置信度{overall_confidence:.3f}, 发现{len(deep_relations)}个深度关系")
            return result
            
        except Exception as e:
            self.logger.error(f"增强多层级推理失败: {str(e)}")
            # 返回失败结果
            return MLRResult(
                success=False,
                complexity_level=ComplexityLevel.L0_EXPLICIT,
                reasoning_steps=[],
                final_answer=None,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e), "deep_relations_discovered": 0}
            )
    
    def _determine_complexity_level(self, problem_text: str, relations: List[ImplicitRelation]) -> ComplexityLevel:
        """确定问题复杂度级别"""
        complexity_score = 0.0
        
        # 基于关系数量和类型
        if relations:
            relation_factor = min(len(relations) / 5, 1.0) * 0.4
            complexity_score += relation_factor
            
            # 关系类型复杂度权重
            type_weights = {
                RelationType.ARITHMETIC: 0.1,
                RelationType.PROPORTION: 0.2,
                RelationType.COMPARISON: 0.2,
                RelationType.TEMPORAL: 0.3,
                RelationType.CAUSAL: 0.4,
                RelationType.CONSTRAINT: 0.5,
                RelationType.FUNCTIONAL: 0.6
            }
            
            type_factor = sum(type_weights.get(r.relation_type, 0.1) for r in relations) / len(relations) if relations else 0
            complexity_score += type_factor * 0.3
        
        # 基于文本特征
        text_features = self._analyze_text_complexity(problem_text)
        complexity_score += text_features * 0.3
        
        # 确定级别
        for level in [ComplexityLevel.L3_DEEP, ComplexityLevel.L2_MEDIUM, 
                     ComplexityLevel.L1_SHALLOW, ComplexityLevel.L0_EXPLICIT]:
            if complexity_score >= self.complexity_thresholds[level]:
                return level
        
        return ComplexityLevel.L0_EXPLICIT
    
    def _analyze_text_complexity(self, text: str) -> float:
        """分析文本复杂度"""
        complexity = 0.0
        
        # 文本长度因子
        length_factor = min(len(text) / 200, 1.0) * 0.2
        complexity += length_factor
        
        # 数字数量因子
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        number_factor = min(len(numbers) / 5, 1.0) * 0.3
        complexity += number_factor
        
        # 复杂关键词
        complex_keywords = [
            "如果", "假设", "当", "因为", "所以", "那么", "否则",
            "比较", "对比", "分析", "推导", "证明"
        ]
        keyword_count = sum(1 for keyword in complex_keywords if keyword in text)
        keyword_factor = min(keyword_count / 3, 1.0) * 0.3
        complexity += keyword_factor
        
        # 数学运算复杂度
        operation_keywords = ["乘以", "除以", "平方", "立方", "开方", "百分比"]
        operation_count = sum(1 for op in operation_keywords if op in text)
        operation_factor = min(operation_count / 2, 1.0) * 0.2
        complexity += operation_factor
        
        return min(1.0, complexity)
    
    def _initialize_reasoning_context(
        self, 
        problem_text: str, 
        relations: List[ImplicitRelation], 
        complexity_level: ComplexityLevel,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """初始化推理上下文"""
        return {
            "problem_text": problem_text,
            "relations": relations,
            "complexity_level": complexity_level,
            "external_context": context or {},
            "variables": {},
            "constraints": [],
            "intermediate_results": {},
            "step_counter": 0
        }
    
    def _execute_layered_reasoning(
        self, 
        reasoning_context: Dict[str, Any], 
        complexity_level: ComplexityLevel
    ) -> List[ReasoningStep]:
        """执行分层推理"""
        steps = []
        
        # L0: 显式推理
        if complexity_level == ComplexityLevel.L0_EXPLICIT:
            steps.extend(self._execute_l0_reasoning(reasoning_context))
        
        # L1: 浅层推理
        elif complexity_level == ComplexityLevel.L1_SHALLOW:
            steps.extend(self._execute_l1_reasoning(reasoning_context))
        
        # L2: 中等推理
        elif complexity_level == ComplexityLevel.L2_MEDIUM:
            steps.extend(self._execute_l2_reasoning(reasoning_context))
        
        # L3: 深层推理
        elif complexity_level == ComplexityLevel.L3_DEEP:
            steps.extend(self._execute_l3_reasoning(reasoning_context))
        
        return steps
    
    def _execute_l0_reasoning(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """执行L0级别推理（显式推理）"""
        steps = []
        problem_text = context["problem_text"]
        
        # 简单的数值提取和计算
        import re
        numbers = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', problem_text)]
        
        if numbers:
            # 初始化步骤
            step_id = len(steps)
            init_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.INITIALIZATION,
                description="提取问题中的数值",
                operation="extract_numbers",
                input_values=[problem_text],
                output_value=numbers,
                confidence=0.9,
                depends_on=[],
                metadata={"numbers_found": len(numbers)}
            )
            steps.append(init_step)
            
            # 简单计算
            if len(numbers) >= 2:
                step_id = len(steps)
                
                # 根据问题类型选择运算
                if "加" in problem_text or "和" in problem_text or "一共" in problem_text:
                    result = sum(numbers)
                    operation = "addition"
                elif "减" in problem_text or "剩" in problem_text or "少" in problem_text:
                    result = numbers[0] - numbers[1] if len(numbers) >= 2 else numbers[0]
                    operation = "subtraction"
                elif "乘" in problem_text or "倍" in problem_text:
                    result = numbers[0] * numbers[1] if len(numbers) >= 2 else numbers[0]
                    operation = "multiplication"
                elif "除" in problem_text or "平均" in problem_text:
                    result = numbers[0] / numbers[1] if len(numbers) >= 2 and numbers[1] != 0 else numbers[0]
                    operation = "division"
                else:
                    result = numbers[0]
                    operation = "identity"
                
                calc_step = ReasoningStep(
                    step_id=step_id,
                    step_type=ReasoningStepType.CALCULATION,
                    description=f"执行{operation}运算",
                    operation=operation,
                    input_values=numbers[:2] if len(numbers) >= 2 else numbers,
                    output_value=result,
                    confidence=0.8,
                    depends_on=[0],
                    metadata={"operation_type": operation}
                )
                steps.append(calc_step)
        
        return steps
    
    def _execute_l1_reasoning(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """执行L1级别推理（浅层推理）"""
        steps = []
        
        # 先执行L0推理
        steps.extend(self._execute_l0_reasoning(context))
        
        # 使用简单的隐式关系
        relations = context["relations"]
        arithmetic_relations = [r for r in relations if r.relation_type == RelationType.ARITHMETIC]
        
        for relation in arithmetic_relations[:3]:  # 限制处理的关系数量
            step_id = len(steps)
            
            # 关系分析步骤
            analysis_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.RELATION_ANALYSIS,
                description=f"分析{relation.relation_type.value}关系",
                operation="analyze_relation",
                input_values=relation.entities,
                output_value=relation.mathematical_expression,
                confidence=relation.confidence,
                depends_on=list(range(len(steps))),
                metadata={"relation": relation.to_dict()}
            )
            steps.append(analysis_step)
        
        return steps
    
    def _execute_l2_reasoning(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """执行L2级别推理（中等推理）"""
        steps = []
        
        # 先执行L1推理
        steps.extend(self._execute_l1_reasoning(context))
        
        # 处理比例和比较关系
        relations = context["relations"]
        proportion_relations = [r for r in relations if r.relation_type == RelationType.PROPORTION]
        comparison_relations = [r for r in relations if r.relation_type == RelationType.COMPARISON]
        
        # 比例推理
        for relation in proportion_relations:
            step_id = len(steps)
            
            prop_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.CALCULATION,
                description=f"计算比例关系: {relation.description}",
                operation="proportion_calculation",
                input_values=relation.entities,
                output_value=self._calculate_proportion(relation),
                confidence=relation.confidence * 0.9,
                depends_on=list(range(max(0, len(steps)-3), len(steps))),
                metadata={"relation_type": "proportion"}
            )
            steps.append(prop_step)
        
        # 比较推理
        for relation in comparison_relations:
            step_id = len(steps)
            
            comp_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.CALCULATION,
                description=f"比较分析: {relation.description}",
                operation="comparison_analysis",
                input_values=relation.entities,
                output_value=self._analyze_comparison(relation),
                confidence=relation.confidence * 0.8,
                depends_on=list(range(max(0, len(steps)-2), len(steps))),
                metadata={"relation_type": "comparison"}
            )
            steps.append(comp_step)
        
        return steps
    
    def _execute_l3_reasoning(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """执行L3级别推理（深层推理）"""
        steps = []
        
        # 先执行L2推理
        steps.extend(self._execute_l2_reasoning(context))
        
        # 处理复杂关系（时间、因果、约束）
        relations = context["relations"]
        complex_relations = [r for r in relations if r.relation_type in 
                           [RelationType.TEMPORAL, RelationType.CAUSAL, RelationType.CONSTRAINT]]
        
        # 构建推理链
        reasoning_chain = self._build_reasoning_chain(complex_relations)
        
        for i, chain_step in enumerate(reasoning_chain):
            step_id = len(steps)
            
            chain_reasoning_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.CALCULATION,
                description=f"复杂推理链步骤 {i+1}: {chain_step['description']}",
                operation="complex_reasoning",
                input_values=chain_step.get("inputs", []),
                output_value=chain_step.get("output"),
                confidence=chain_step.get("confidence", 0.7),
                depends_on=chain_step.get("dependencies", []),
                metadata={"chain_step": i+1, "complexity": "L3"}
            )
            steps.append(chain_reasoning_step)
        
        return steps
    
    def _calculate_proportion(self, relation: ImplicitRelation) -> Union[float, str]:
        """计算比例"""
        try:
            if len(relation.entities) >= 2:
                num1, num2 = float(relation.entities[0]), float(relation.entities[1])
                return num1 / num2 if num2 != 0 else "undefined"
        except (ValueError, IndexError):
            pass
        return "无法计算"
    
    def _analyze_comparison(self, relation: ImplicitRelation) -> str:
        """分析比较关系"""
        if len(relation.entities) >= 2:
            try:
                num1, num2 = float(relation.entities[0]), float(relation.entities[1])
                if num1 > num2:
                    return f"{num1} > {num2}"
                elif num1 < num2:
                    return f"{num1} < {num2}"
                else:
                    return f"{num1} = {num2}"
            except ValueError:
                return f"{relation.entities[0]} vs {relation.entities[1]}"
        return "比较关系"
    
    def _build_reasoning_chain(self, relations: List[ImplicitRelation]) -> List[Dict[str, Any]]:
        """构建复杂推理链"""
        chain = []
        
        for i, relation in enumerate(relations):
            chain_step = {
                "description": relation.description,
                "inputs": relation.entities,
                "output": f"推理结果_{i+1}",
                "confidence": relation.confidence * 0.8,
                "dependencies": list(range(max(0, i-1), i)) if i > 0 else []
            }
            chain.append(chain_step)
        
        return chain
    
    def _verify_reasoning_chain(self, steps: List[ReasoningStep]) -> List[ReasoningStep]:
        """验证推理链"""
        verification_steps = []
        
        # 检查计算步骤的一致性
        calc_steps = [s for s in steps if s.step_type == ReasoningStepType.CALCULATION]
        
        if calc_steps:
            step_id = len(steps)
            
            # 创建验证步骤
            verification_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.VERIFICATION,
                description="验证推理链的一致性",
                operation="chain_verification",
                input_values=[s.output_value for s in calc_steps if s.output_value is not None],
                output_value=self._check_consistency(calc_steps),
                confidence=0.8,
                depends_on=[s.step_id for s in calc_steps],
                metadata={"verified_steps": len(calc_steps)}
            )
            verification_steps.append(verification_step)
        
        return verification_steps
    
    def _check_consistency(self, calc_steps: List[ReasoningStep]) -> bool:
        """检查计算步骤的一致性"""
        # 简化的一致性检查
        for step in calc_steps:
            if step.confidence < 0.5:
                return False
            if step.output_value is None:
                return False
        return True
    
    def _generate_final_answer(self, steps: List[ReasoningStep]) -> Tuple[Union[float, str, None], float]:
        """生成最终答案"""
        if not steps:
            return None, 0.0
        
        # 查找结论步骤
        conclusion_steps = [s for s in steps if s.step_type == ReasoningStepType.CONCLUSION]
        if conclusion_steps:
            final_step = conclusion_steps[-1]
            return final_step.output_value, final_step.confidence
        
        # 查找最后的计算步骤
        calc_steps = [s for s in steps if s.step_type == ReasoningStepType.CALCULATION]
        if calc_steps:
            final_step = calc_steps[-1]
            return final_step.output_value, final_step.confidence
        
        # 返回最后一个有输出的步骤
        for step in reversed(steps):
            if step.output_value is not None:
                return step.output_value, step.confidence
        
        return None, 0.0
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """计算整体置信度"""
        if not steps:
            return 0.0
        
        # 加权平均置信度
        total_confidence = sum(step.confidence for step in steps)
        avg_confidence = total_confidence / len(steps)
        
        # 基于验证步骤的调整
        verification_steps = [s for s in steps if s.step_type == ReasoningStepType.VERIFICATION]
        if verification_steps:
            verification_factor = sum(s.confidence for s in verification_steps) / len(verification_steps)
            avg_confidence = (avg_confidence + verification_factor) / 2
        
        return avg_confidence
    
    def _update_stats(self, complexity_level: ComplexityLevel, steps: List[ReasoningStep], success: bool):
        """更新统计信息"""
        self.stats["total_processed"] += 1
        self.stats["complexity_distribution"][complexity_level.value] += 1
        
        # 更新平均步骤数
        current_avg_steps = self.stats["average_steps"]
        new_avg_steps = ((current_avg_steps * (self.stats["total_processed"] - 1) + len(steps)) / 
                        self.stats["total_processed"])
        self.stats["average_steps"] = new_avg_steps
        
        # 更新平均置信度
        if steps:
            avg_confidence = sum(s.confidence for s in steps) / len(steps)
            current_avg_conf = self.stats["average_confidence"]
            new_avg_conf = ((current_avg_conf * (self.stats["total_processed"] - 1) + avg_confidence) / 
                           self.stats["total_processed"])
            self.stats["average_confidence"] = new_avg_conf
        
        # 更新成功率
        current_success_count = self.stats["success_rate"] * (self.stats["total_processed"] - 1)
        new_success_count = current_success_count + (1 if success else 0)
        self.stats["success_rate"] = new_success_count / self.stats["total_processed"]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_processed": 0,
            "complexity_distribution": {level.value: 0 for level in ComplexityLevel},
            "average_steps": 0.0,
            "average_confidence": 0.0,
            "success_rate": 0.0
        }
        self.logger.info("MLR处理器统计信息已重置")
    
    # ==================== 新增：深度隐含关系增强方法 ====================
    
    def _extract_basic_entities(self, problem_text: str) -> List[Dict[str, Any]]:
        """提取基础实体信息"""
        entities = []
        
        # 数字实体
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
        for num in numbers:
            entities.append({
                "name": num,
                "type": "number",
                "properties": ["quantitative", "measurable"]
            })
        
        # 人物实体
        people = ['小明', '小红', '小张', '小李', '学生', '老师']
        for person in people:
            if person in problem_text:
                entities.append({
                    "name": person,
                    "type": "person", 
                    "properties": ["agent", "possessor"]
                })
        
        # 物品实体
        objects = ['苹果', '书', '笔', '车', '钱', '元']
        for obj in objects:
            if obj in problem_text:
                entities.append({
                    "name": obj,
                    "type": "object" if obj != "元" and obj != "钱" else "money",
                    "properties": ["countable", "possessed"]
                })
        
        # 概念实体
        concepts = ['面积', '周长', '速度', '时间', '距离', '总共', '一共']
        for concept in concepts:
            if concept in problem_text:
                entities.append({
                    "name": concept,
                    "type": "concept",
                    "properties": ["abstract", "calculable"]
                })
        
        return entities
    
    def _determine_enhanced_complexity_level(
        self, 
        problem_text: str, 
        relations: List[Any], 
        deep_relations: List[DeepImplicitRelation],
        implicit_constraints: List[ImplicitConstraint]
    ) -> ComplexityLevel:
        """确定增强的问题复杂度级别"""
        complexity_score = 0.0
        
        # 基于传统关系数量
        if relations:
            relation_factor = min(len(relations) / 5, 1.0) * 0.2
            complexity_score += relation_factor
        
        # 基于深度关系数量和深度
        if deep_relations:
            deep_factor = 0.0
            for relation in deep_relations:
                depth_weight = {
                    RelationDepth.SURFACE: 0.1,
                    RelationDepth.SHALLOW: 0.2,
                    RelationDepth.MEDIUM: 0.4,
                    RelationDepth.DEEP: 0.6
                }
                deep_factor += depth_weight.get(relation.depth, 0.1) * relation.confidence
            
            deep_factor = min(deep_factor / len(deep_relations), 1.0) * 0.4
            complexity_score += deep_factor
        
        # 基于隐含约束复杂度
        if implicit_constraints:
            constraint_factor = 0.0
            for constraint in implicit_constraints:
                type_weight = {
                    ConstraintType.NON_NEGATIVITY: 0.1,
                    ConstraintType.CONSERVATION_LAW: 0.4,
                    ConstraintType.CONSISTENCY: 0.3,
                    ConstraintType.TYPE_COMPATIBILITY: 0.2
                }
                constraint_factor += type_weight.get(constraint.constraint_type, 0.1) * constraint.confidence
            
            constraint_factor = min(constraint_factor / len(implicit_constraints), 1.0) * 0.3
            complexity_score += constraint_factor
        
        # 基于文本特征
        text_features = self._analyze_text_complexity(problem_text)
        complexity_score += text_features * 0.1
        
        # 确定级别
        for level in [ComplexityLevel.L3_DEEP, ComplexityLevel.L2_MEDIUM, 
                     ComplexityLevel.L1_SHALLOW, ComplexityLevel.L0_EXPLICIT]:
            if complexity_score >= self.complexity_thresholds[level]:
                return level
        
        return ComplexityLevel.L0_EXPLICIT
    
    def _initialize_enhanced_reasoning_context(
        self, 
        problem_text: str, 
        relations: List[Any], 
        complexity_level: ComplexityLevel,
        context: Optional[Dict[str, Any]],
        deep_relations: List[DeepImplicitRelation],
        implicit_constraints: List[ImplicitConstraint],
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """初始化增强推理上下文"""
        return {
            "problem_text": problem_text,
            "relations": relations,
            "deep_relations": deep_relations,
            "implicit_constraints": implicit_constraints,
            "entities": entities,
            "complexity_level": complexity_level,
            "external_context": context or {},
            "variables": {},
            "constraints": [],
            "intermediate_results": {},
            "step_counter": 0,
            "semantic_evidence": [],
            "constraint_violations": []
        }
    
    def _execute_enhanced_layered_reasoning(
        self, 
        reasoning_context: Dict[str, Any], 
        complexity_level: ComplexityLevel
    ) -> List[ReasoningStep]:
        """执行增强分层推理"""
        steps = []
        
        # L0: 显式推理 + 基础深度关系
        if complexity_level == ComplexityLevel.L0_EXPLICIT:
            steps.extend(self._execute_enhanced_l0_reasoning(reasoning_context))
        
        # L1: 浅层推理 + 语义蕴含推理
        elif complexity_level == ComplexityLevel.L1_SHALLOW:
            steps.extend(self._execute_enhanced_l1_reasoning(reasoning_context))
        
        # L2: 中等推理 + 隐含约束推理
        elif complexity_level == ComplexityLevel.L2_MEDIUM:
            steps.extend(self._execute_enhanced_l2_reasoning(reasoning_context))
        
        # L3: 深层推理 + 多层关系建模
        elif complexity_level == ComplexityLevel.L3_DEEP:
            steps.extend(self._execute_enhanced_l3_reasoning(reasoning_context))
        
        return steps
    
    def _execute_enhanced_l0_reasoning(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """执行增强L0级别推理"""
        steps = []
        
        # 基础数值提取和计算
        steps.extend(self._execute_l0_reasoning(context))
        
        # 添加基础深度关系分析
        deep_relations = context["deep_relations"]
        surface_relations = [r for r in deep_relations if r.depth == RelationDepth.SURFACE]
        
        for relation in surface_relations[:2]:  # 限制处理数量
            step_id = len(steps)
            
            deep_analysis_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.RELATION_ANALYSIS,
                description=f"深度关系分析: {relation._generate_display_label()}",
                operation="deep_relation_analysis",
                input_values=[relation.source_entity, relation.target_entity],
                output_value=relation.logical_basis,
                confidence=relation.confidence,
                depends_on=list(range(len(steps))),
                metadata={
                    "relation_type": relation.relation_type.value,
                    "semantic_evidence": relation.semantic_evidence,
                    "depth": relation.depth.value,
                    "frontend_data": relation.frontend_display_data
                }
            )
            steps.append(deep_analysis_step)
        
        return steps
    
    def _execute_enhanced_l1_reasoning(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """执行增强L1级别推理（语义蕴含推理）"""
        steps = []
        
        # 先执行L0推理
        steps.extend(self._execute_enhanced_l0_reasoning(context))
        
        # 添加语义蕴含推理
        deep_relations = context["deep_relations"]
        semantic_relations = [r for r in deep_relations 
                            if r.relation_type in [SemanticRelationType.IMPLICIT_DEPENDENCY, 
                                                 SemanticRelationType.IMPLICIT_EQUIVALENCE]]
        
        for relation in semantic_relations:
            step_id = len(steps)
            
            semantic_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.RELATION_ANALYSIS,
                description=f"语义蕴含推理: {relation.logical_basis}",
                operation="semantic_implication",
                input_values=relation.semantic_evidence,
                output_value=relation.mathematical_expression or relation.logical_basis,
                confidence=relation.confidence * 0.9,
                depends_on=list(range(max(0, len(steps)-2), len(steps))),
                metadata={
                    "semantic_type": relation.relation_type.value,
                    "evidence": relation.semantic_evidence,
                    "implications": relation.constraint_implications,
                    "frontend_data": relation.frontend_display_data
                }
            )
            steps.append(semantic_step)
        
        return steps
    
    def _execute_enhanced_l2_reasoning(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """执行增强L2级别推理（隐含约束推理）"""
        steps = []
        
        # 先执行L1推理
        steps.extend(self._execute_enhanced_l1_reasoning(context))
        
        # 添加隐含约束分析
        implicit_constraints = context["implicit_constraints"]
        
        for constraint in implicit_constraints:
            step_id = len(steps)
            
            constraint_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.RELATION_ANALYSIS,
                description=f"隐含约束分析: {constraint.description}",
                operation="constraint_analysis",
                input_values=constraint.affected_entities,
                output_value=constraint.constraint_expression,
                confidence=constraint.confidence,
                depends_on=list(range(max(0, len(steps)-3), len(steps))),
                metadata={
                    "constraint_type": constraint.constraint_type.value,
                    "discovery_method": constraint.discovery_method,
                    "affected_entities": constraint.affected_entities,
                    "frontend_data": constraint.frontend_visualization
                }
            )
            steps.append(constraint_step)
        
        # 约束验证步骤
        if implicit_constraints:
            step_id = len(steps)
            
            validation_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.VERIFICATION,
                description="隐含约束一致性验证",
                operation="constraint_validation",
                input_values=[c.constraint_expression for c in implicit_constraints],
                output_value=self._validate_constraints(implicit_constraints),
                confidence=0.8,
                depends_on=[s.step_id for s in steps[-len(implicit_constraints):] if s.step_type == ReasoningStepType.RELATION_ANALYSIS],
                metadata={"constraint_count": len(implicit_constraints)}
            )
            steps.append(validation_step)
        
        return steps
    
    def _execute_enhanced_l3_reasoning(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """执行增强L3级别推理（多层关系建模）"""
        steps = []
        
        # 先执行L2推理
        steps.extend(self._execute_enhanced_l2_reasoning(context))
        
        # 添加多层关系建模
        deep_relations = context["deep_relations"]
        deep_level_relations = [r for r in deep_relations if r.depth == RelationDepth.DEEP]
        
        # 构建关系层次图
        relation_layers = self._build_relation_hierarchy(deep_level_relations)
        
        for layer_name, layer_relations in relation_layers.items():
            step_id = len(steps)
            
            layer_step = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.CALCULATION,
                description=f"多层关系建模 - {layer_name}层",
                operation="multilayer_modeling",
                input_values=[r.source_entity for r in layer_relations],
                output_value=f"{layer_name}层包含{len(layer_relations)}个关系",
                confidence=np.mean([r.confidence for r in layer_relations]) if layer_relations else 0.0,
                depends_on=list(range(max(0, len(steps)-5), len(steps))),
                metadata={
                    "layer_name": layer_name,
                    "relation_count": len(layer_relations),
                    "relations": [r.to_frontend_format() for r in layer_relations]
                }
            )
            steps.append(layer_step)
        
        # 整体性推理步骤
        step_id = len(steps)
        
        holistic_step = ReasoningStep(
            step_id=step_id,
            step_type=ReasoningStepType.CALCULATION,
            description="整体性关系推理与综合",
            operation="holistic_reasoning",
            input_values=[f"{k}层" for k in relation_layers.keys()],
            output_value=self._synthesize_holistic_understanding(deep_relations, context["implicit_constraints"]),
            confidence=0.7,
            depends_on=[s.step_id for s in steps[-len(relation_layers):] if s.operation == "multilayer_modeling"],
            metadata={
                "total_deep_relations": len(deep_relations),
                "layer_count": len(relation_layers),
                "synthesis_method": "holistic_integration"
            }
        )
        steps.append(holistic_step)
        
        return steps
    
    def _verify_enhanced_reasoning_chain(
        self, 
        steps: List[ReasoningStep], 
        deep_relations: List[DeepImplicitRelation],
        implicit_constraints: List[ImplicitConstraint]
    ) -> List[ReasoningStep]:
        """验证增强推理链"""
        verification_steps = []
        
        # 原有验证
        verification_steps.extend(self._verify_reasoning_chain(steps))
        
        # 深度关系一致性验证
        if deep_relations:
            step_id = len(steps) + len(verification_steps)
            
            relation_verification = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.VERIFICATION,
                description="深度关系一致性验证",
                operation="deep_relation_verification",
                input_values=[r.id for r in deep_relations],
                output_value=self._verify_deep_relations_consistency(deep_relations),
                confidence=0.85,
                depends_on=[s.step_id for s in steps if "deep_relation" in s.operation],
                metadata={"verified_relations": len(deep_relations)}
            )
            verification_steps.append(relation_verification)
        
        # 约束满足性验证
        if implicit_constraints:
            step_id = len(steps) + len(verification_steps)
            
            constraint_verification = ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.VERIFICATION,
                description="隐含约束满足性验证",
                operation="constraint_satisfaction_verification",
                input_values=[c.id for c in implicit_constraints],
                output_value=self._verify_constraint_satisfaction(implicit_constraints, steps),
                confidence=0.8,
                depends_on=[s.step_id for s in steps if "constraint" in s.operation],
                metadata={"verified_constraints": len(implicit_constraints)}
            )
            verification_steps.append(constraint_verification)
        
        return verification_steps
    
    def _generate_enhanced_final_answer(
        self, 
        steps: List[ReasoningStep], 
        deep_relations: List[DeepImplicitRelation]
    ) -> Tuple[Union[float, str, None], float]:
        """生成增强的最终答案"""
        # 首先尝试基础方法
        base_answer, base_confidence = self._generate_final_answer(steps)
        
        # 如果有深度关系，尝试提升答案质量
        if deep_relations and base_answer is not None:
            # 基于深度关系调整置信度
            relation_confidence_boost = 0.0
            high_confidence_relations = [r for r in deep_relations if r.confidence > 0.8]
            
            if high_confidence_relations:
                relation_confidence_boost = min(0.1, len(high_confidence_relations) * 0.02)
            
            enhanced_confidence = min(1.0, base_confidence + relation_confidence_boost)
            
            return base_answer, enhanced_confidence
        
        return base_answer, base_confidence
    
    def _calculate_enhanced_overall_confidence(
        self, 
        steps: List[ReasoningStep], 
        deep_relations: List[DeepImplicitRelation],
        implicit_constraints: List[ImplicitConstraint]
    ) -> float:
        """计算增强的整体置信度"""
        # 基础置信度
        base_confidence = self._calculate_overall_confidence(steps)
        
        # 深度关系贡献
        if deep_relations:
            relation_factor = sum(r.confidence for r in deep_relations) / len(deep_relations)
            relation_weight = min(0.2, len(deep_relations) * 0.05)
            base_confidence = (base_confidence + relation_factor * relation_weight) / (1 + relation_weight)
        
        # 约束一致性贡献
        if implicit_constraints:
            constraint_factor = sum(c.confidence for c in implicit_constraints) / len(implicit_constraints)
            constraint_weight = min(0.1, len(implicit_constraints) * 0.02)
            base_confidence = (base_confidence + constraint_factor * constraint_weight) / (1 + constraint_weight)
        
        return min(1.0, base_confidence)
    
    def _update_enhanced_stats(
        self, 
        complexity_level: ComplexityLevel, 
        steps: List[ReasoningStep], 
        success: bool,
        deep_relations: List[DeepImplicitRelation],
        implicit_constraints: List[ImplicitConstraint]
    ):
        """更新增强统计信息"""
        # 调用原有统计更新
        self._update_stats(complexity_level, steps, success)
        
        # 更新深度关系统计
        self.stats["deep_relations_discovered"] += len(deep_relations)
        self.stats["implicit_constraints_found"] += len(implicit_constraints)
    
    def _prepare_frontend_data(
        self, 
        entities: List[Dict[str, Any]], 
        deep_relations: List[DeepImplicitRelation],
        implicit_constraints: List[ImplicitConstraint],
        reasoning_steps: List[ReasoningStep]
    ) -> Dict[str, Any]:
        """准备前端可视化数据"""
        return {
            "entities": entities,
            "deep_relations": [r.to_frontend_format() for r in deep_relations],
            "implicit_constraints": [c.to_frontend_format() for c in implicit_constraints],
            "reasoning_layers": self._extract_reasoning_layers(reasoning_steps),
            "visualization_config": {
                "show_depth_indicators": True,
                "show_constraint_panels": True,
                "enable_interactive_exploration": True,
                "animation_sequence": True
            }
        }
    
    # 辅助方法
    def _validate_constraints(self, constraints: List[ImplicitConstraint]) -> bool:
        """验证约束一致性"""
        # 简化的约束验证逻辑
        for constraint in constraints:
            if constraint.confidence < 0.3:
                return False
        return True
    
    def _build_relation_hierarchy(self, relations: List[DeepImplicitRelation]) -> Dict[str, List[DeepImplicitRelation]]:
        """构建关系层次结构"""
        hierarchy = {
            "causality": [],
            "conservation": [],
            "dependency": [],
            "equivalence": []
        }
        
        for relation in relations:
            if relation.relation_type == SemanticRelationType.DEEP_CAUSALITY:
                hierarchy["causality"].append(relation)
            elif relation.relation_type == SemanticRelationType.DEEP_CONSERVATION:
                hierarchy["conservation"].append(relation)
            elif relation.relation_type == SemanticRelationType.IMPLICIT_DEPENDENCY:
                hierarchy["dependency"].append(relation)
            elif relation.relation_type == SemanticRelationType.IMPLICIT_EQUIVALENCE:
                hierarchy["equivalence"].append(relation)
        
        return hierarchy
    
    def _synthesize_holistic_understanding(
        self, 
        deep_relations: List[DeepImplicitRelation], 
        constraints: List[ImplicitConstraint]
    ) -> str:
        """综合整体性理解"""
        relation_count = len(deep_relations)
        constraint_count = len(constraints)
        
        if relation_count > 5 and constraint_count > 3:
            return f"复杂整体性系统：{relation_count}个深度关系与{constraint_count}个约束形成完整推理网络"
        elif relation_count > 2:
            return f"中等复杂度系统：{relation_count}个深度关系构成推理基础"
        else:
            return f"简单系统：{relation_count}个关系，{constraint_count}个约束"
    
    def _verify_deep_relations_consistency(self, relations: List[DeepImplicitRelation]) -> bool:
        """验证深度关系一致性"""
        # 检查关系间的逻辑一致性
        for i, rel1 in enumerate(relations):
            for rel2 in relations[i+1:]:
                if rel1.source_entity == rel2.target_entity and rel1.target_entity == rel2.source_entity:
                    # 双向关系检查
                    if abs(rel1.confidence - rel2.confidence) > 0.5:
                        return False
        return True
    
    def _verify_constraint_satisfaction(
        self, 
        constraints: List[ImplicitConstraint], 
        steps: List[ReasoningStep]
    ) -> bool:
        """验证约束满足性"""
        # 检查推理步骤是否违反约束
        for constraint in constraints:
            constraint_type = constraint.constraint_type
            
            if constraint_type == ConstraintType.NON_NEGATIVITY:
                # 检查非负性约束
                for step in steps:
                    if step.output_value is not None and isinstance(step.output_value, (int, float)):
                        if step.output_value < 0:
                            return False
        
        return True
    
    def _extract_reasoning_layers(self, steps: List[ReasoningStep]) -> Dict[str, List[Dict[str, Any]]]:
        """提取推理层次"""
        layers = {
            "initialization": [],
            "relation_analysis": [],
            "calculation": [],
            "verification": [],
            "conclusion": []
        }
        
        for step in steps:
            layer_key = step.step_type.value
            if layer_key in layers:
                layers[layer_key].append({
                    "step_id": step.step_id,
                    "description": step.description,
                    "confidence": step.confidence,
                    "metadata": step.metadata
                })
        
        return layers