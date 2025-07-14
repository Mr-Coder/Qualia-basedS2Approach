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

from .ird_engine import ImplicitRelation, RelationType


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
            "success_rate": 0.0
        }
        
        self.logger.info("多层级推理处理器初始化完成")
    
    def execute_reasoning(
        self, 
        problem_text: str, 
        relations: List[ImplicitRelation], 
        context: Optional[Dict[str, Any]] = None
    ) -> MLRResult:
        """
        执行多层级推理
        
        Args:
            problem_text: 问题文本
            relations: 隐式关系列表
            context: 可选的上下文信息
            
        Returns:
            MLRResult: 推理结果
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始多层级推理: {problem_text[:50]}...")
            
            # 第一步：确定复杂度级别
            complexity_level = self._determine_complexity_level(problem_text, relations)
            
            # 第二步：初始化推理上下文
            reasoning_context = self._initialize_reasoning_context(
                problem_text, relations, complexity_level, context
            )
            
            # 第三步：执行分层推理
            reasoning_steps = self._execute_layered_reasoning(
                reasoning_context, complexity_level
            )
            
            # 第四步：验证推理结果
            if self.enable_verification:
                verification_steps = self._verify_reasoning_chain(reasoning_steps)
                reasoning_steps.extend(verification_steps)
            
            # 第五步：生成最终答案
            final_answer, answer_confidence = self._generate_final_answer(reasoning_steps)
            
            # 计算整体置信度
            overall_confidence = self._calculate_overall_confidence(reasoning_steps)
            
            # 更新统计信息
            self._update_stats(complexity_level, reasoning_steps, overall_confidence >= self.confidence_threshold)
            
            processing_time = time.time() - start_time
            
            result = MLRResult(
                success=answer_confidence >= self.confidence_threshold,
                complexity_level=complexity_level,
                reasoning_steps=reasoning_steps,
                final_answer=final_answer,
                confidence_score=overall_confidence,
                processing_time=processing_time,
                metadata={
                    "relations_used": len([r for r in relations if r.confidence >= 0.5]),
                    "step_count": len(reasoning_steps),
                    "verification_enabled": self.enable_verification,
                    "context": context or {}
                }
            )
            
            self.logger.info(f"MLR完成: 复杂度{complexity_level.value}, {len(reasoning_steps)}步, 置信度{overall_confidence:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"多层级推理失败: {str(e)}")
            # 返回失败结果
            return MLRResult(
                success=False,
                complexity_level=ComplexityLevel.L0_EXPLICIT,
                reasoning_steps=[],
                final_answer=None,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
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