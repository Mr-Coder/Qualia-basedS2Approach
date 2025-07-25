#!/usr/bin/env python3
"""
COT-DIR推理链模块
Chain-of-Thought Directed Implicit Reasoning
结合显式推理和隐式推理构建完整推理链
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from problem_preprocessor import ProcessedProblem
from qs2_semantic_analyzer import SemanticEntity
from ird_relation_discovery import RelationNetwork, ImplicitRelation

logger = logging.getLogger(__name__)

class ReasoningStepType(Enum):
    """推理步骤类型"""
    ENTITY_EXTRACTION = "entity_extraction"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    RELATION_DISCOVERY = "relation_discovery"
    MATHEMATICAL_COMPUTATION = "mathematical_computation"
    LOGIC_INFERENCE = "logic_inference"
    RESULT_SYNTHESIS = "result_synthesis"

@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    step_type: ReasoningStepType
    step_name: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    execution_time: float
    reasoning_method: str
    evidence: List[str]

@dataclass
class ReasoningChain:
    """推理链"""
    chain_id: str
    problem_text: str
    steps: List[ReasoningStep]
    final_answer: Any
    overall_confidence: float
    total_execution_time: float
    chain_metrics: Dict[str, float]

class COTDIRReasoningChain:
    """COT-DIR推理链构建器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 推理策略配置
        self.reasoning_config = {
            "enable_explicit_reasoning": True,
            "enable_implicit_reasoning": True,
            "enable_semantic_enhancement": True,
            "confidence_threshold": 0.3,
            "max_inference_depth": 5
        }
        
        # 数学运算器
        self.math_processor = MathematicalProcessor()
        
        # 逻辑推理器  
        self.logic_processor = LogicInferenceProcessor()

    def build_reasoning_chain(self, processed_problem: ProcessedProblem,
                            semantic_entities: List[SemanticEntity],
                            relation_network: RelationNetwork) -> ReasoningChain:
        """
        构建COT-DIR推理链
        
        Args:
            processed_problem: 预处理后的问题
            semantic_entities: 语义实体列表
            relation_network: 关系网络
            
        Returns:
            ReasoningChain: 完整的推理链
        """
        try:
            start_time = time.time()
            chain_id = f"cotdir_{int(start_time)}"
            
            self.logger.info(f"开始构建COT-DIR推理链: {chain_id}")
            
            steps = []
            
            # Step 1: 实体提取与分析
            step1 = self._create_entity_extraction_step(processed_problem, semantic_entities)
            steps.append(step1)
            
            # Step 2: 语义结构分析
            step2 = self._create_semantic_analysis_step(semantic_entities)
            steps.append(step2)
            
            # Step 3: 隐式关系发现
            step3 = self._create_relation_discovery_step(relation_network)
            steps.append(step3)
            
            # Step 4: 数学计算
            step4 = self._create_mathematical_computation_step(processed_problem, relation_network)
            steps.append(step4)
            
            # Step 5: 逻辑推理
            step5 = self._create_logic_inference_step(steps, relation_network)
            steps.append(step5)
            
            # Step 6: 结果综合
            step6 = self._create_result_synthesis_step(steps, processed_problem)
            steps.append(step6)
            
            # 计算整体指标
            final_answer = step6.output_data.get("final_answer", "计算失败")
            overall_confidence = self._calculate_overall_confidence(steps)
            total_execution_time = time.time() - start_time
            chain_metrics = self._calculate_chain_metrics(steps)
            
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                problem_text=processed_problem.original_text,
                steps=steps,
                final_answer=final_answer,
                overall_confidence=overall_confidence,
                total_execution_time=total_execution_time,
                chain_metrics=chain_metrics
            )
            
            self.logger.info(f"COT-DIR推理链构建完成，包含{len(steps)}个步骤")
            return reasoning_chain
            
        except Exception as e:
            self.logger.error(f"推理链构建失败: {e}")
            return self._create_fallback_chain(processed_problem, str(e))

    def _create_entity_extraction_step(self, processed_problem: ProcessedProblem,
                                     semantic_entities: List[SemanticEntity]) -> ReasoningStep:
        """创建实体提取步骤"""
        
        start_time = time.time()
        
        input_data = {
            "problem_text": processed_problem.cleaned_text,
            "preprocessing_results": {
                "entity_count": len(processed_problem.entities),
                "number_count": len(processed_problem.numbers),
                "keyword_count": len(processed_problem.keywords)
            }
        }
        
        # 分析提取的实体
        extracted_entities = []
        qualia_analysis = {}
        
        for entity in semantic_entities:
            entity_info = {
                "id": entity.entity_id,
                "name": entity.name,
                "type": entity.entity_type,
                "confidence": entity.confidence
            }
            extracted_entities.append(entity_info)
            
            # Qualia结构分析
            qualia_analysis[entity.name] = {
                "formal_roles": len(entity.qualia.formal),
                "telic_roles": len(entity.qualia.telic),
                "agentive_roles": len(entity.qualia.agentive),
                "constitutive_roles": len(entity.qualia.constitutive)
            }
        
        output_data = {
            "entities_extracted": extracted_entities,
            "qualia_analysis": qualia_analysis,
            "semantic_richness": sum(sum(q.values()) for q in qualia_analysis.values()) / max(len(qualia_analysis), 1)
        }
        
        execution_time = time.time() - start_time
        
        return ReasoningStep(
            step_id="step_1_entity_extraction",
            step_type=ReasoningStepType.ENTITY_EXTRACTION,
            step_name="实体提取与Qualia构建",
            description=f"从问题文本中提取{len(extracted_entities)}个语义实体，构建四维Qualia结构",
            input_data=input_data,
            output_data=output_data,
            confidence=min(sum(e["confidence"] for e in extracted_entities) / max(len(extracted_entities), 1), 1.0),
            execution_time=execution_time,
            reasoning_method="qs2_semantic_extraction",
            evidence=[
                f"提取实体: {[e['name'] for e in extracted_entities]}",
                f"平均语义丰富度: {output_data['semantic_richness']:.2f}"
            ]
        )

    def _create_semantic_analysis_step(self, semantic_entities: List[SemanticEntity]) -> ReasoningStep:
        """创建语义分析步骤"""
        
        start_time = time.time()
        
        input_data = {
            "semantic_entities": len(semantic_entities),
            "total_qualia_items": sum(
                len(e.qualia.formal) + len(e.qualia.telic) + 
                len(e.qualia.agentive) + len(e.qualia.constitutive)
                for e in semantic_entities
            )
        }
        
        # 语义分析
        semantic_patterns = []
        for entity in semantic_entities:
            # 分析语义模式
            if "拥有" in entity.qualia.telic or "被拥有" in entity.qualia.telic:
                semantic_patterns.append("ownership_pattern")
            if "计算" in entity.qualia.telic or "运算" in entity.qualia.telic:
                semantic_patterns.append("mathematical_pattern")
            if "人物" in entity.qualia.formal:
                semantic_patterns.append("agent_pattern")
            if "可数" in entity.qualia.formal:
                semantic_patterns.append("countable_pattern")
        
        semantic_categories = self._categorize_entities(semantic_entities)
        
        output_data = {
            "semantic_patterns": list(set(semantic_patterns)),
            "semantic_categories": semantic_categories,
            "pattern_confidence": len(set(semantic_patterns)) / 4.0  # 基于模式数量
        }
        
        execution_time = time.time() - start_time
        
        return ReasoningStep(
            step_id="step_2_semantic_analysis",
            step_type=ReasoningStepType.SEMANTIC_ANALYSIS,
            step_name="四维语义结构分析",
            description=f"分析{len(semantic_entities)}个实体的Qualia语义结构，识别{len(output_data['semantic_patterns'])}种语义模式",
            input_data=input_data,
            output_data=output_data,
            confidence=output_data["pattern_confidence"],
            execution_time=execution_time,
            reasoning_method="qs2_qualia_analysis",
            evidence=[
                f"识别语义模式: {output_data['semantic_patterns']}",
                f"语义分类: {list(semantic_categories.keys())}"
            ]
        )

    def _categorize_entities(self, entities: List[SemanticEntity]) -> Dict[str, List[str]]:
        """对实体进行语义分类"""
        
        categories = {
            "agents": [],      # 主体
            "objects": [],     # 客体
            "quantities": [],  # 数量
            "concepts": []     # 概念
        }
        
        for entity in entities:
            if entity.entity_type == "person":
                categories["agents"].append(entity.name)
            elif entity.entity_type == "object":
                categories["objects"].append(entity.name)
            elif entity.entity_type == "number":
                categories["quantities"].append(entity.name)
            else:
                categories["concepts"].append(entity.name)
        
        return categories

    def _create_relation_discovery_step(self, relation_network: RelationNetwork) -> ReasoningStep:
        """创建关系发现步骤"""
        
        start_time = time.time()
        
        relations = relation_network.relations if relation_network else []
        
        input_data = {
            "entity_count": len(relation_network.entities) if relation_network else 0,
            "potential_relations": len(relation_network.entities) * (len(relation_network.entities) - 1) // 2 if relation_network else 0
        }
        
        # 分析发现的关系
        relation_analysis = {}
        relation_types = {}
        strong_relations = []
        
        for relation in relations:
            # 按类型统计
            rel_type = relation.relation_type
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
            
            # 强关系识别
            if relation.strength > 0.7:
                strong_relations.append({
                    "from": relation.source_entity_id,
                    "to": relation.target_entity_id,
                    "type": rel_type,
                    "strength": relation.strength
                })
            
            # 关系证据分析
            relation_analysis[relation.relation_id] = {
                "type": rel_type,
                "strength": relation.strength,
                "evidence_count": len(relation.evidence)
            }
        
        output_data = {
            "relations_discovered": len(relations),
            "relation_types": relation_types,
            "strong_relations": strong_relations,
            "average_relation_strength": sum(r.strength for r in relations) / max(len(relations), 1),
            "relation_analysis": relation_analysis
        }
        
        execution_time = time.time() - start_time
        
        return ReasoningStep(
            step_id="step_3_relation_discovery",
            step_type=ReasoningStepType.RELATION_DISCOVERY,
            step_name="隐式关系发现",
            description=f"基于QS²兼容性分析发现{len(relations)}个实体间隐式关系",
            input_data=input_data,
            output_data=output_data,
            confidence=min(output_data["average_relation_strength"], 1.0),
            execution_time=execution_time,
            reasoning_method="ird_enhanced",
            evidence=[
                f"关系类型分布: {relation_types}",
                f"强关系数量: {len(strong_relations)}",
                f"平均关系强度: {output_data['average_relation_strength']:.3f}"
            ]
        )

    def _create_mathematical_computation_step(self, processed_problem: ProcessedProblem,
                                            relation_network: RelationNetwork) -> ReasoningStep:
        """创建数学计算步骤"""
        
        start_time = time.time()
        
        numbers = processed_problem.numbers
        keywords = processed_problem.keywords
        
        input_data = {
            "numbers": numbers,
            "keywords": keywords,
            "problem_type": processed_problem.problem_type
        }
        
        # 使用数学处理器计算
        computation_result = self.math_processor.compute(numbers, keywords, relation_network)
        
        output_data = {
            "computation_result": computation_result["result"],
            "operation_performed": computation_result["operation"],
            "computation_confidence": computation_result["confidence"],
            "mathematical_evidence": computation_result["evidence"]
        }
        
        execution_time = time.time() - start_time
        
        return ReasoningStep(
            step_id="step_4_mathematical_computation",
            step_type=ReasoningStepType.MATHEMATICAL_COMPUTATION,
            step_name="数学运算执行",
            description=f"基于识别的数量关系执行{computation_result['operation']}运算",
            input_data=input_data,
            output_data=output_data,
            confidence=computation_result["confidence"],
            execution_time=execution_time,
            reasoning_method="relation_guided_computation",
            evidence=computation_result["evidence"]
        )

    def _create_logic_inference_step(self, previous_steps: List[ReasoningStep],
                                   relation_network: RelationNetwork) -> ReasoningStep:
        """创建逻辑推理步骤"""
        
        start_time = time.time()
        
        input_data = {
            "previous_steps": len(previous_steps),
            "available_relations": len(relation_network.relations) if relation_network else 0
        }
        
        # 使用逻辑处理器推理
        inference_result = self.logic_processor.infer(previous_steps, relation_network)
        
        output_data = {
            "inference_conclusions": inference_result["conclusions"],
            "logical_consistency": inference_result["consistency"],
            "inference_confidence": inference_result["confidence"]
        }
        
        execution_time = time.time() - start_time
        
        return ReasoningStep(
            step_id="step_5_logic_inference",
            step_type=ReasoningStepType.LOGIC_INFERENCE,
            step_name="逻辑推理验证",
            description="基于前序推理步骤进行逻辑一致性验证和结论推导",
            input_data=input_data,
            output_data=output_data,
            confidence=inference_result["confidence"],
            execution_time=execution_time,
            reasoning_method="multi_step_logic_inference",
            evidence=inference_result["evidence"]
        )

    def _create_result_synthesis_step(self, previous_steps: List[ReasoningStep],
                                    processed_problem: ProcessedProblem) -> ReasoningStep:
        """创建结果综合步骤"""
        
        start_time = time.time()
        
        input_data = {
            "steps_to_synthesize": len(previous_steps),
            "problem_type": processed_problem.problem_type
        }
        
        # 综合前面步骤的结果
        math_step = next((step for step in previous_steps 
                         if step.step_type == ReasoningStepType.MATHEMATICAL_COMPUTATION), None)
        logic_step = next((step for step in previous_steps 
                          if step.step_type == ReasoningStepType.LOGIC_INFERENCE), None)
        
        final_answer = "计算失败"
        synthesis_confidence = 0.0
        
        if math_step and math_step.output_data.get("computation_result") is not None:
            result = math_step.output_data["computation_result"]
            
            # 确定答案格式
            if "个" in processed_problem.cleaned_text:
                final_answer = f"{result}个"
            elif "元" in processed_problem.cleaned_text:
                final_answer = f"{result}元"
            elif "米" in processed_problem.cleaned_text:
                final_answer = f"{result}米"
            else:
                final_answer = str(result)
            
            synthesis_confidence = math_step.confidence
            
            # 结合逻辑推理的置信度
            if logic_step:
                logic_confidence = logic_step.output_data.get("inference_confidence", 0.0)
                synthesis_confidence = (synthesis_confidence + logic_confidence) / 2
        
        output_data = {
            "final_answer": final_answer,
            "synthesis_confidence": synthesis_confidence,
            "contributing_steps": [step.step_id for step in previous_steps],
            "answer_format": self._determine_answer_format(processed_problem)
        }
        
        execution_time = time.time() - start_time
        
        return ReasoningStep(
            step_id="step_6_result_synthesis",
            step_type=ReasoningStepType.RESULT_SYNTHESIS,
            step_name="结果综合",
            description=f"综合{len(previous_steps)}个推理步骤的结果，得出最终答案",
            input_data=input_data,
            output_data=output_data,
            confidence=synthesis_confidence,
            execution_time=execution_time,
            reasoning_method="multi_step_synthesis",
            evidence=[
                f"最终答案: {final_answer}",
                f"综合置信度: {synthesis_confidence:.3f}"
            ]
        )

    def _determine_answer_format(self, processed_problem: ProcessedProblem) -> str:
        """确定答案格式"""
        
        text = processed_problem.cleaned_text
        
        if any(unit in text for unit in ["个", "只", "本", "支"]):
            return "countable_units"
        elif any(unit in text for unit in ["元", "角", "分", "块"]):
            return "currency"
        elif any(unit in text for unit in ["米", "厘米", "公里"]):
            return "length"
        elif any(unit in text for unit in ["平方米", "面积"]):
            return "area"
        else:
            return "numeric"

    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """计算整体置信度"""
        
        if not steps:
            return 0.0
        
        # 加权平均，重要步骤权重更高
        weights = {
            ReasoningStepType.MATHEMATICAL_COMPUTATION: 0.3,
            ReasoningStepType.RELATION_DISCOVERY: 0.25,
            ReasoningStepType.LOGIC_INFERENCE: 0.2,
            ReasoningStepType.RESULT_SYNTHESIS: 0.15,
            ReasoningStepType.SEMANTIC_ANALYSIS: 0.05,
            ReasoningStepType.ENTITY_EXTRACTION: 0.05
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for step in steps:
            weight = weights.get(step.step_type, 0.1)
            weighted_confidence += step.confidence * weight
            total_weight += weight
        
        return weighted_confidence / max(total_weight, 1.0)

    def _calculate_chain_metrics(self, steps: List[ReasoningStep]) -> Dict[str, float]:
        """计算推理链指标"""
        
        metrics = {}
        
        # 基础指标
        metrics["total_steps"] = len(steps)
        metrics["average_step_confidence"] = sum(step.confidence for step in steps) / max(len(steps), 1)
        metrics["total_execution_time"] = sum(step.execution_time for step in steps)
        metrics["average_step_time"] = metrics["total_execution_time"] / max(len(steps), 1)
        
        # 推理质量指标
        high_confidence_steps = sum(1 for step in steps if step.confidence > 0.8)
        metrics["high_confidence_ratio"] = high_confidence_steps / max(len(steps), 1)
        
        # 推理方法多样性
        methods = set(step.reasoning_method for step in steps)
        metrics["method_diversity"] = len(methods)
        
        return metrics

    def _create_fallback_chain(self, processed_problem: ProcessedProblem, error_msg: str) -> ReasoningChain:
        """创建fallback推理链"""
        
        fallback_step = ReasoningStep(
            step_id="fallback_step",
            step_type=ReasoningStepType.RESULT_SYNTHESIS,
            step_name="fallback推理",
            description=f"推理链构建失败，使用fallback逻辑: {error_msg}",
            input_data={"error": error_msg},
            output_data={"final_answer": "推理失败"},
            confidence=0.1,
            execution_time=0.01,
            reasoning_method="fallback",
            evidence=[f"错误信息: {error_msg}"]
        )
        
        return ReasoningChain(
            chain_id="fallback_chain",
            problem_text=processed_problem.original_text,
            steps=[fallback_step],
            final_answer="推理失败",
            overall_confidence=0.1,
            total_execution_time=0.01,
            chain_metrics={"total_steps": 1, "error": True}
        )

class MathematicalProcessor:
    """数学运算处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def compute(self, numbers: List[float], keywords: List[str], 
               relation_network: RelationNetwork) -> Dict[str, Any]:
        """执行数学计算"""
        
        if len(numbers) < 2:
            return {
                "result": 0,
                "operation": "no_operation",
                "confidence": 0.1,
                "evidence": ["数字不足，无法计算"]
            }
        
        # 基于关键词确定运算类型
        if any(kw in keywords for kw in ["一共", "总共", "合计", "总数", "加起来"]):
            result = sum(numbers)
            operation = "addition"
            confidence = 0.95
            evidence = [f"关键词指示加法: {'+'.join(map(str, numbers))} = {result}"]
            
        elif any(kw in keywords for kw in ["还剩", "剩余", "少了", "减去"]):
            result = numbers[0] - sum(numbers[1:])
            operation = "subtraction"
            confidence = 0.90
            evidence = [f"关键词指示减法: {numbers[0]} - {sum(numbers[1:])} = {result}"]
            
        elif any(kw in keywords for kw in ["倍", "乘以", "每", "速度"]):
            result = numbers[0] * numbers[1] if len(numbers) >= 2 else numbers[0]
            operation = "multiplication"
            confidence = 0.85
            evidence = [f"关键词指示乘法: {numbers[0]} × {numbers[1]} = {result}"]
            
        else:
            # 基于关系网络判断
            if relation_network and relation_network.relations:
                math_relations = [r for r in relation_network.relations 
                                if r.relation_type in ["mathematical", "quantity"]]
                if math_relations:
                    result = sum(numbers)  # 默认求和
                    operation = "relation_guided_addition"
                    confidence = 0.80
                    evidence = [f"关系指导的加法: {'+'.join(map(str, numbers))} = {result}"]
                else:
                    result = sum(numbers)
                    operation = "default_addition"
                    confidence = 0.70
                    evidence = [f"默认加法: {'+'.join(map(str, numbers))} = {result}"]
            else:
                result = sum(numbers)
                operation = "default_addition"
                confidence = 0.70
                evidence = [f"默认加法: {'+'.join(map(str, numbers))} = {result}"]
        
        return {
            "result": result,
            "operation": operation,
            "confidence": confidence,
            "evidence": evidence
        }

class LogicInferenceProcessor:
    """逻辑推理处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def infer(self, previous_steps: List[ReasoningStep], 
             relation_network: RelationNetwork) -> Dict[str, Any]:
        """执行逻辑推理"""
        
        conclusions = []
        consistency_score = 1.0
        evidence = []
        
        # 检查步骤间的逻辑一致性
        entity_step = next((step for step in previous_steps 
                           if step.step_type == ReasoningStepType.ENTITY_EXTRACTION), None)
        math_step = next((step for step in previous_steps 
                         if step.step_type == ReasoningStepType.MATHEMATICAL_COMPUTATION), None)
        
        if entity_step and math_step:
            # 检查实体数量与数学计算的一致性
            entity_count = len(entity_step.output_data.get("entities_extracted", []))
            numbers_used = len(math_step.input_data.get("numbers", []))
            
            if entity_count >= numbers_used:
                conclusions.append("实体数量与数值数量匹配")
                evidence.append(f"实体{entity_count}个，数值{numbers_used}个")
            else:
                consistency_score *= 0.8
                conclusions.append("实体数量与数值数量不完全匹配")
                evidence.append(f"实体不足：实体{entity_count}个，数值{numbers_used}个")
        
        # 检查关系的逻辑合理性
        if relation_network and relation_network.relations:
            strong_relations = [r for r in relation_network.relations if r.strength > 0.7]
            if strong_relations:
                conclusions.append(f"发现{len(strong_relations)}个强关系支持推理")
                evidence.append(f"强关系类型: {list(set(r.relation_type for r in strong_relations))}")
            else:
                consistency_score *= 0.9
                conclusions.append("关系强度普遍较低")
        
        # 检查数学运算的合理性
        if math_step:
            operation = math_step.output_data.get("operation_performed", "")
            confidence = math_step.output_data.get("computation_confidence", 0.0)
            
            if confidence > 0.8:
                conclusions.append(f"{operation}运算置信度高")
                evidence.append(f"运算置信度: {confidence:.3f}")
            else:
                consistency_score *= 0.85
                conclusions.append(f"{operation}运算置信度中等")
        
        overall_confidence = consistency_score * min(
            sum(step.confidence for step in previous_steps) / max(len(previous_steps), 1), 1.0
        )
        
        return {
            "conclusions": conclusions,
            "consistency": consistency_score,
            "confidence": overall_confidence,
            "evidence": evidence
        }

# 测试函数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from problem_preprocessor import ProblemPreprocessor
    from qs2_semantic_analyzer import QS2SemanticAnalyzer
    from ird_relation_discovery import IRDRelationDiscovery
    
    # 创建组件
    preprocessor = ProblemPreprocessor()
    qs2_analyzer = QS2SemanticAnalyzer()
    ird_discovery = IRDRelationDiscovery(qs2_analyzer)
    cotdir_chain = COTDIRReasoningChain()
    
    # 测试问题
    test_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    print(f"测试问题: {test_problem}")
    print("="*60)
    
    # 执行完整推理流程
    processed = preprocessor.preprocess(test_problem)
    semantic_entities = qs2_analyzer.analyze_semantics(processed)
    relation_network = ird_discovery.discover_relations(semantic_entities, test_problem)
    
    # 构建推理链
    reasoning_chain = cotdir_chain.build_reasoning_chain(
        processed, semantic_entities, relation_network
    )
    
    print(f"推理链ID: {reasoning_chain.chain_id}")
    print(f"最终答案: {reasoning_chain.final_answer}")
    print(f"整体置信度: {reasoning_chain.overall_confidence:.3f}")
    print(f"总执行时间: {reasoning_chain.total_execution_time:.3f}s")
    
    print(f"\n推理步骤详情:")
    for i, step in enumerate(reasoning_chain.steps, 1):
        print(f"\n步骤{i}: {step.step_name}")
        print(f"  描述: {step.description}")
        print(f"  置信度: {step.confidence:.3f}")
        print(f"  执行时间: {step.execution_time:.3f}s")
        print(f"  推理方法: {step.reasoning_method}")
        if step.evidence:
            print(f"  证据: {', '.join(step.evidence[:2])}")
    
    print(f"\n推理链指标:")
    for metric, value in reasoning_chain.chain_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")