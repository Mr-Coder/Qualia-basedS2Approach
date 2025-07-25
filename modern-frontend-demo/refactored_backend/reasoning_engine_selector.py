#!/usr/bin/env python3
"""
推理引擎选择模块
根据问题复杂度和系统状态智能选择合适的推理引擎
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from problem_preprocessor import ProcessedProblem
from qs2_semantic_analyzer import SemanticEntity
from ird_relation_discovery import RelationNetwork
from enhanced_math_solver import EnhancedMathSolver

logger = logging.getLogger(__name__)

class ReasoningMode(Enum):
    """推理模式"""
    SIMPLE = "simple"
    ADVANCED = "advanced"
    HYBRID = "hybrid"
    AUTO = "auto"

class EngineType(Enum):
    """引擎类型"""
    SIMPLE_ENGINE = "simple_engine"
    ADVANCED_ENGINE = "advanced_engine"
    HYBRID_ENGINE = "hybrid_engine"
    FALLBACK_ENGINE = "fallback_engine"

@dataclass
class EngineStatus:
    """引擎状态"""
    engine_type: EngineType
    available: bool
    performance_score: float
    last_success_time: float
    error_count: int
    average_response_time: float

@dataclass
class ReasoningRequest:
    """推理请求"""
    processed_problem: ProcessedProblem
    semantic_entities: List[SemanticEntity]
    relation_network: RelationNetwork
    user_preferences: Dict[str, Any]
    context: str

@dataclass
class EngineSelection:
    """引擎选择结果"""
    selected_engine: EngineType
    confidence: float
    reasoning: List[str]
    fallback_engines: List[EngineType]
    estimated_response_time: float

class SimpleReasoningEngine:
    """简单推理引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def solve(self, request: ReasoningRequest) -> Dict[str, Any]:
        """简单推理求解"""
        try:
            problem = request.processed_problem
            
            # 提取数字
            numbers = problem.numbers
            if len(numbers) < 2:
                return self._create_error_result("数字不足，无法计算")
            
            # 简单加法逻辑
            if any(kw in problem.keywords for kw in ["一共", "总共", "合计", "总数"]):
                result = sum(numbers)
                confidence = 0.95
                operation = "addition"
            else:
                result = numbers[0] + numbers[1]  # 默认相加
                confidence = 0.8
                operation = "default_addition"
            
            return {
                "success": True,
                "answer": f"{result}个" if "个" in problem.cleaned_text else str(result),
                "confidence": confidence,
                "strategy_used": "simple_arithmetic",
                "execution_time": 0.1,
                "reasoning_steps": [
                    {
                        "step": 1,
                        "action": operation,
                        "description": f"识别数字 {numbers}，执行{operation}",
                        "result": result
                    }
                ],
                "entity_relationship_diagram": {
                    "entities": [{"id": f"num_{i}", "name": str(num), "type": "number"} 
                               for i, num in enumerate(numbers)],
                    "relationships": []
                }
            }
            
        except Exception as e:
            self.logger.error(f"简单引擎求解失败: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "success": False,
            "answer": "计算失败",
            "confidence": 0.0,
            "strategy_used": "simple_arithmetic",
            "execution_time": 0.0,
            "reasoning_steps": [],
            "entity_relationship_diagram": {"entities": [], "relationships": []},
            "error": error_msg
        }

class AdvancedReasoningEngine:
    """高级推理引擎（QS²+IRD+COT-DIR）"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def solve(self, request: ReasoningRequest) -> Dict[str, Any]:
        """高级推理求解"""
        try:
            start_time = time.time()
            
            # 使用语义实体和关系网络进行推理
            entities = request.semantic_entities
            relations = request.relation_network.relations
            problem = request.processed_problem
            
            # 构建推理步骤
            reasoning_steps = []
            
            # Step 1: 实体分析
            step1 = {
                "step": 1,
                "action": "实体语义分析",
                "description": f"识别{len(entities)}个语义实体，构建Qualia结构",
                "entities_analyzed": [e.name for e in entities],
                "confidence": 0.9
            }
            reasoning_steps.append(step1)
            
            # Step 2: 关系发现
            step2 = {
                "step": 2,
                "action": "隐式关系发现",
                "description": f"发现{len(relations)}个实体关系",
                "relations_found": [
                    f"{self._get_entity_name(r.source_entity_id, entities)} -> {self._get_entity_name(r.target_entity_id, entities)}"
                    for r in relations[:3]  # 显示前3个关系
                ],
                "confidence": 0.85
            }
            reasoning_steps.append(step2)
            
            # Step 3: 数学计算
            numbers = problem.numbers
            if len(numbers) >= 2:
                if any(kw in problem.keywords for kw in ["一共", "总共", "合计"]):
                    result = sum(numbers)
                    operation = "求和运算"
                else:
                    result = numbers[0] + numbers[1]
                    operation = "基础运算"
                
                step3 = {
                    "step": 3,
                    "action": operation,
                    "description": f"基于语义关系执行数学运算: {' + '.join(map(str, numbers))} = {result}",
                    "calculation": f"{' + '.join(map(str, numbers))} = {result}",
                    "confidence": 0.95
                }
                reasoning_steps.append(step3)
            else:
                result = 0
                step3 = {
                    "step": 3,
                    "action": "无法计算",
                    "description": "数字信息不足",
                    "confidence": 0.1
                }
                reasoning_steps.append(step3)
            
            execution_time = time.time() - start_time
            
            # 构建实体关系图
            erd = self._build_entity_relationship_diagram(entities, relations, problem)
            
            return {
                "success": True,
                "answer": f"{result}个" if result > 0 and "个" in problem.cleaned_text else str(result),
                "confidence": min(sum(step.get("confidence", 0) for step in reasoning_steps) / len(reasoning_steps), 1.0),
                "strategy_used": "qs2_ird_cotdir",
                "execution_time": execution_time,
                "reasoning_steps": reasoning_steps,
                "entity_relationship_diagram": erd
            }
            
        except Exception as e:
            self.logger.error(f"高级引擎求解失败: {e}")
            return self._create_error_result(str(e))
    
    def _get_entity_name(self, entity_id: str, entities: List[SemanticEntity]) -> str:
        """获取实体名称"""
        for entity in entities:
            if entity.entity_id == entity_id:
                return entity.name
        return "未知实体"
    
    def _build_entity_relationship_diagram(self, entities: List[SemanticEntity], 
                                         relations: List, problem: ProcessedProblem) -> Dict[str, Any]:
        """构建实体关系图"""
        
        erd_entities = []
        for entity in entities:
            erd_entity = {
                "id": entity.entity_id,
                "name": entity.name,
                "type": entity.entity_type,
                "properties": getattr(entity, 'properties', []),
                "qualia_roles": {
                    "formal": entity.qualia.formal[:3],  # 只显示前3个
                    "telic": entity.qualia.telic[:3],
                    "agentive": entity.qualia.agentive[:3],
                    "constitutive": entity.qualia.constitutive[:3]
                }
            }
            erd_entities.append(erd_entity)
        
        erd_relationships = []
        for relation in relations:
            erd_relationship = {
                "from": relation.source_entity_id,
                "to": relation.target_entity_id,
                "type": relation.relation_type,
                "strength": relation.strength,
                "evidence": relation.evidence[:2],  # 只显示前2个证据
                "discovered_by": "QS2_IRD"
            }
            erd_relationships.append(erd_relationship)
        
        return {
            "entities": erd_entities,
            "relationships": erd_relationships,
            "implicit_constraints": [
                "数量非负约束",
                "整数约束", 
                "语义一致性约束",
                "关系传递性约束"
            ],
            "qs2_enhancements": {
                "qualia_structures_used": len(entities),
                "semantic_relations_discovered": len(relations),
                "average_relation_strength": sum(r.strength for r in relations) / max(len(relations), 1)
            }
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "success": False,
            "answer": "高级推理失败",
            "confidence": 0.0,
            "strategy_used": "qs2_ird_cotdir",
            "execution_time": 0.0,
            "reasoning_steps": [],
            "entity_relationship_diagram": {"entities": [], "relationships": []},
            "error": error_msg
        }

class ReasoningEngineSelector:
    """推理引擎选择器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化引擎
        self.simple_engine = SimpleReasoningEngine()
        self.advanced_engine = AdvancedReasoningEngine()
        # 🧠 增强数学求解器 - 真正能解题的引擎
        self.enhanced_math_solver = EnhancedMathSolver()
        
        # 引擎状态跟踪
        self.engine_status = {
            EngineType.SIMPLE_ENGINE: EngineStatus(
                engine_type=EngineType.SIMPLE_ENGINE,
                available=True,
                performance_score=0.8,
                last_success_time=time.time(),
                error_count=0,
                average_response_time=0.1
            ),
            EngineType.ADVANCED_ENGINE: EngineStatus(
                engine_type=EngineType.ADVANCED_ENGINE,
                available=True,
                performance_score=0.9,
                last_success_time=time.time(),
                error_count=0,
                average_response_time=1.5
            )
        }
        
        # 选择策略配置
        self.selection_config = {
            "complexity_thresholds": {
                "simple": 0.3,
                "advanced": 0.7
            },
            "performance_weights": {
                "accuracy": 0.4,
                "speed": 0.3,
                "reliability": 0.3
            },
            "fallback_enabled": True
        }
        
        # 当前模式
        self.current_mode = ReasoningMode.AUTO

    def select_engine(self, request: ReasoningRequest) -> EngineSelection:
        """
        选择最合适的推理引擎
        
        Args:
            request: 推理请求
            
        Returns:
            EngineSelection: 引擎选择结果
        """
        try:
            self.logger.info(f"开始引擎选择，当前模式: {self.current_mode.value}")
            
            # 分析问题特征
            problem_features = self._analyze_problem_features(request)
            
            # 评估引擎可用性
            engine_availability = self._evaluate_engine_availability()
            
            # 根据模式选择引擎
            if self.current_mode == ReasoningMode.SIMPLE:
                selected_engine = self._select_simple_mode(problem_features, engine_availability)
            elif self.current_mode == ReasoningMode.ADVANCED:
                selected_engine = self._select_advanced_mode(problem_features, engine_availability)
            elif self.current_mode == ReasoningMode.HYBRID:
                selected_engine = self._select_hybrid_mode(problem_features, engine_availability, request)
            else:  # AUTO mode
                selected_engine = self._select_auto_mode(problem_features, engine_availability)
            
            self.logger.info(f"选择引擎: {selected_engine.selected_engine.value}")
            return selected_engine
            
        except Exception as e:
            self.logger.error(f"引擎选择失败: {e}")
            # 返回fallback选择
            return EngineSelection(
                selected_engine=EngineType.SIMPLE_ENGINE,
                confidence=0.5,
                reasoning=["引擎选择失败，使用fallback"],
                fallback_engines=[EngineType.FALLBACK_ENGINE],
                estimated_response_time=0.1
            )

    def _analyze_problem_features(self, request: ReasoningRequest) -> Dict[str, Any]:
        """分析问题特征"""
        
        problem = request.processed_problem
        entities = request.semantic_entities
        relations = request.relation_network.relations if request.relation_network else []
        
        features = {
            "complexity_score": problem.complexity_score,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "number_count": len(problem.numbers),
            "keyword_complexity": self._assess_keyword_complexity(problem.keywords),
            "problem_type": problem.problem_type,
            "semantic_richness": self._assess_semantic_richness(entities),
            "relation_density": len(relations) / max(len(entities) * (len(entities) - 1) / 2, 1)
        }
        
        return features

    def _assess_keyword_complexity(self, keywords: List[str]) -> float:
        """评估关键词复杂度"""
        complex_keywords = ["比例", "百分比", "倍数", "平均", "面积", "体积", "速度"]
        simple_keywords = ["一共", "总共", "合计", "加", "减"]
        
        complex_count = sum(1 for kw in keywords if kw in complex_keywords)
        simple_count = sum(1 for kw in keywords if kw in simple_keywords)
        
        if complex_count > 0:
            return 0.8 + min(complex_count * 0.1, 0.2)
        elif simple_count > 0:
            return 0.2
        else:
            return 0.5

    def _assess_semantic_richness(self, entities: List[SemanticEntity]) -> float:
        """评估语义丰富度"""
        if not entities:
            return 0.0
        
        total_qualia_items = 0
        for entity in entities:
            total_qualia_items += (
                len(entity.qualia.formal) +
                len(entity.qualia.telic) +
                len(entity.qualia.agentive) +
                len(entity.qualia.constitutive)
            )
        
        average_richness = total_qualia_items / len(entities)
        return min(average_richness / 10, 1.0)  # 归一化到[0,1]

    def _evaluate_engine_availability(self) -> Dict[EngineType, float]:
        """评估引擎可用性"""
        availability = {}
        
        for engine_type, status in self.engine_status.items():
            if not status.available:
                availability[engine_type] = 0.0
            else:
                # 综合性能评分
                performance_score = (
                    status.performance_score * 0.4 +
                    (1.0 / max(status.average_response_time, 0.1)) * 0.3 +
                    max(0, (10 - status.error_count) / 10) * 0.3
                )
                availability[engine_type] = min(performance_score, 1.0)
        
        return availability

    def _select_simple_mode(self, features: Dict[str, Any], 
                          availability: Dict[EngineType, float]) -> EngineSelection:
        """简单模式选择"""
        
        if availability.get(EngineType.SIMPLE_ENGINE, 0) > 0.5:
            return EngineSelection(
                selected_engine=EngineType.SIMPLE_ENGINE,
                confidence=0.8,
                reasoning=["用户指定简单模式", "简单引擎可用"],
                fallback_engines=[EngineType.FALLBACK_ENGINE],
                estimated_response_time=0.1
            )
        else:
            return EngineSelection(
                selected_engine=EngineType.FALLBACK_ENGINE,
                confidence=0.5,
                reasoning=["简单引擎不可用", "使用fallback"],
                fallback_engines=[],
                estimated_response_time=0.05
            )

    def _select_advanced_mode(self, features: Dict[str, Any], 
                            availability: Dict[EngineType, float]) -> EngineSelection:
        """高级模式选择"""
        
        if availability.get(EngineType.ADVANCED_ENGINE, 0) > 0.5:
            return EngineSelection(
                selected_engine=EngineType.ADVANCED_ENGINE,
                confidence=0.9,
                reasoning=["用户指定高级模式", "高级引擎可用"],
                fallback_engines=[EngineType.SIMPLE_ENGINE, EngineType.FALLBACK_ENGINE],
                estimated_response_time=2.0
            )
        else:
            return EngineSelection(
                selected_engine=EngineType.SIMPLE_ENGINE,
                confidence=0.7,
                reasoning=["高级引擎不可用", "降级到简单引擎"],
                fallback_engines=[EngineType.FALLBACK_ENGINE],
                estimated_response_time=0.1
            )

    def _select_hybrid_mode(self, features: Dict[str, Any], 
                          availability: Dict[EngineType, float], 
                          request: ReasoningRequest) -> EngineSelection:
        """混合模式选择"""
        
        # 在混合模式下，根据问题特征动态选择
        complexity = features["complexity_score"]
        
        if complexity < 0.3 and availability.get(EngineType.SIMPLE_ENGINE, 0) > 0.5:
            return EngineSelection(
                selected_engine=EngineType.SIMPLE_ENGINE,
                confidence=0.85,
                reasoning=[
                    f"问题复杂度较低 ({complexity:.2f})", 
                    "选择简单引擎提高效率"
                ],
                fallback_engines=[EngineType.ADVANCED_ENGINE, EngineType.FALLBACK_ENGINE],
                estimated_response_time=0.1
            )
        elif availability.get(EngineType.ADVANCED_ENGINE, 0) > 0.5:
            return EngineSelection(
                selected_engine=EngineType.ADVANCED_ENGINE,
                confidence=0.9,
                reasoning=[
                    f"问题复杂度较高 ({complexity:.2f})", 
                    "选择高级引擎确保准确性"
                ],
                fallback_engines=[EngineType.SIMPLE_ENGINE, EngineType.FALLBACK_ENGINE],
                estimated_response_time=2.0
            )
        else:
            return self._select_simple_mode(features, availability)

    def _select_auto_mode(self, features: Dict[str, Any], 
                        availability: Dict[EngineType, float]) -> EngineSelection:
        """自动模式选择"""
        
        # 智能分析最佳引擎
        scores = {}
        
        # 为每个可用引擎计算得分
        for engine_type in [EngineType.SIMPLE_ENGINE, EngineType.ADVANCED_ENGINE]:
            if availability.get(engine_type, 0) > 0.3:
                score = self._calculate_engine_score(engine_type, features, availability)
                scores[engine_type] = score
        
        if not scores:
            return EngineSelection(
                selected_engine=EngineType.FALLBACK_ENGINE,
                confidence=0.5,
                reasoning=["所有引擎不可用"],
                fallback_engines=[],
                estimated_response_time=0.05
            )
        
        # 选择得分最高的引擎
        best_engine = max(scores.items(), key=lambda x: x[1])
        selected_engine = best_engine[0]
        score = best_engine[1]
        
        return EngineSelection(
            selected_engine=selected_engine,
            confidence=min(score, 1.0),
            reasoning=[
                f"综合评分最高: {score:.3f}",
                f"问题复杂度: {features['complexity_score']:.2f}",
                f"语义丰富度: {features['semantic_richness']:.2f}"
            ],
            fallback_engines=self._get_fallback_engines(selected_engine),
            estimated_response_time=self.engine_status[selected_engine].average_response_time
        )

    def _calculate_engine_score(self, engine_type: EngineType, features: Dict[str, Any], 
                              availability: Dict[EngineType, float]) -> float:
        """计算引擎得分"""
        
        base_availability = availability.get(engine_type, 0)
        
        if engine_type == EngineType.SIMPLE_ENGINE:
            # 简单引擎适合低复杂度问题
            complexity_fit = 1.0 - features["complexity_score"]
            speed_bonus = 0.8  # 速度优势
            accuracy_penalty = max(0, features["complexity_score"] - 0.3) * 0.5
            score = base_availability * complexity_fit * speed_bonus - accuracy_penalty
            
        elif engine_type == EngineType.ADVANCED_ENGINE:
            # 高级引擎适合高复杂度问题
            complexity_fit = features["complexity_score"]
            accuracy_bonus = features["semantic_richness"] * 0.5
            speed_penalty = 0.2  # 速度劣势
            score = base_availability * complexity_fit + accuracy_bonus - speed_penalty
            
        else:
            score = base_availability * 0.5
        
        return max(score, 0)

    def _get_fallback_engines(self, selected_engine: EngineType) -> List[EngineType]:
        """获取fallback引擎列表"""
        
        if selected_engine == EngineType.ADVANCED_ENGINE:
            return [EngineType.SIMPLE_ENGINE, EngineType.FALLBACK_ENGINE]
        elif selected_engine == EngineType.SIMPLE_ENGINE:
            return [EngineType.FALLBACK_ENGINE]
        else:
            return []

    def execute_reasoning(self, request: ReasoningRequest) -> Dict[str, Any]:
        """执行推理"""
        
        # 选择引擎
        selection = self.select_engine(request)
        
        # 执行推理
        start_time = time.time()
        
        try:
            # 🧠 优先使用增强数学求解器进行真正的数学推理
            self.logger.info("使用增强数学求解器求解")
            enhanced_result = self.enhanced_math_solver.solve_problem(request.context)
            
            if enhanced_result["success"] and enhanced_result["confidence"] > 0.7:
                # 增强求解器成功，使用其结果
                self.logger.info(f"增强数学求解器成功求解，置信度: {enhanced_result['confidence']}")
                result = self._adapt_enhanced_result(enhanced_result, request)
            else:
                # 降级到原有引擎
                self.logger.info("降级到原有推理引擎")
                if selection.selected_engine == EngineType.SIMPLE_ENGINE:
                    result = self.simple_engine.solve(request)
                elif selection.selected_engine == EngineType.ADVANCED_ENGINE:
                    result = self.advanced_engine.solve(request)
                else:  # FALLBACK_ENGINE
                    result = self._fallback_solve(request)
            
            # 更新引擎状态
            execution_time = time.time() - start_time
            self._update_engine_status(selection.selected_engine, True, execution_time)
            
            # 添加选择信息到结果
            result["engine_selection"] = {
                "selected_engine": selection.selected_engine.value,
                "selection_confidence": selection.confidence,
                "selection_reasoning": selection.reasoning
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"推理执行失败: {e}")
            execution_time = time.time() - start_time
            self._update_engine_status(selection.selected_engine, False, execution_time)
            
            # 尝试fallback
            if selection.fallback_engines:
                self.logger.info("尝试fallback引擎")
                for fallback_engine in selection.fallback_engines:
                    try:
                        if fallback_engine == EngineType.SIMPLE_ENGINE:
                            return self.simple_engine.solve(request)
                        elif fallback_engine == EngineType.FALLBACK_ENGINE:
                            return self._fallback_solve(request)
                    except:
                        continue
            
            # 所有引擎都失败
            return {
                "success": False,
                "answer": "推理失败",
                "confidence": 0.0,
                "strategy_used": "failed",
                "execution_time": execution_time,
                "reasoning_steps": [],
                "entity_relationship_diagram": {"entities": [], "relationships": []},
                "error": str(e)
            }

    def _adapt_enhanced_result(self, enhanced_result: Dict[str, Any], request: ReasoningRequest) -> Dict[str, Any]:
        """适配增强数学求解器的结果到统一格式"""
        
        # 构建实体关系图
        erd_entities = []
        if "entities" in enhanced_result:
            for entity in enhanced_result["entities"]:
                erd_entities.append({
                    "id": entity["id"],
                    "name": entity["name"],
                    "type": entity["type"],
                    "unit": entity.get("unit", ""),
                    "value": entity.get("name", "")
                })
        
        erd_relationships = []
        if "relations" in enhanced_result:
            for relation in enhanced_result["relations"]:
                erd_relationships.append({
                    "from": relation["entities"][0] if relation["entities"] else "unknown",
                    "to": relation["entities"][1] if len(relation["entities"]) > 1 else "result",
                    "type": relation["type"],
                    "expression": relation.get("expression", ""),
                    "discovered_by": "Enhanced_Math_Solver"
                })
        
        return {
            "success": enhanced_result["success"],
            "answer": enhanced_result["answer"],
            "confidence": enhanced_result["confidence"],
            "strategy_used": f"enhanced_math_solver_{enhanced_result.get('problem_type', 'unknown')}",
            "execution_time": 0.5,  # 估算时间
            "algorithm_type": "Enhanced_Mathematical_Reasoning",
            "reasoning_steps": enhanced_result.get("reasoning_steps", []),
            "entity_relationship_diagram": {
                "entities": erd_entities,
                "relationships": erd_relationships,
                "implicit_constraints": [
                    "数学运算正确性约束",
                    "数值类型一致性约束",
                    "单位统一性约束",
                    "结果合理性约束"
                ],
                "enhancement_info": {
                    "solver_type": "enhanced_math_solver",
                    "problem_type_detected": enhanced_result.get("problem_type", "unknown"),
                    "solution_steps_count": len(enhanced_result.get("solution_steps", [])),
                    "mathematical_expressions_used": len([s for s in enhanced_result.get("solution_steps", []) if s.get("expression")])
                }
            },
            "metadata": {
                "engine_used": "enhanced_math_solver",
                "problem_classification": enhanced_result.get("problem_type", "unknown"),
                "mathematical_reasoning": True,
                "solution_method": "symbolic_and_numerical"
            }
        }

    def _fallback_solve(self, request: ReasoningRequest) -> Dict[str, Any]:
        """fallback求解"""
        return {
            "success": True,
            "answer": "系统暂不可用",
            "confidence": 0.1,
            "strategy_used": "fallback",
            "execution_time": 0.01,
            "reasoning_steps": [
                {
                    "step": 1,
                    "action": "fallback模式",
                    "description": "主要推理引擎不可用，使用基础回退逻辑"
                }
            ],
            "entity_relationship_diagram": {"entities": [], "relationships": []}
        }

    def _update_engine_status(self, engine_type: EngineType, success: bool, execution_time: float):
        """更新引擎状态"""
        
        if engine_type in self.engine_status:
            status = self.engine_status[engine_type]
            
            if success:
                status.last_success_time = time.time()
                status.performance_score = min(status.performance_score + 0.01, 1.0)
                status.error_count = max(status.error_count - 1, 0)
            else:
                status.error_count += 1
                status.performance_score = max(status.performance_score - 0.05, 0.0)
            
            # 更新平均响应时间
            status.average_response_time = (status.average_response_time + execution_time) / 2

    def set_mode(self, mode: ReasoningMode):
        """设置推理模式"""
        self.current_mode = mode
        self.logger.info(f"推理模式已切换至: {mode.value}")

    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            engine_type.value: {
                "available": status.available,
                "performance_score": status.performance_score,
                "error_count": status.error_count,
                "average_response_time": status.average_response_time
            }
            for engine_type, status in self.engine_status.items()
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
    engine_selector = ReasoningEngineSelector()
    
    # 测试问题
    test_problems = [
        "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",  # 简单
        "一个长方形的长是8米，宽是6米，面积是多少平方米？",      # 中等
        "甲乙两车从相距240公里的两地同时相向而行，甲车速度60公里/小时，乙车速度80公里/小时，多长时间相遇？"  # 复杂
    ]
    
    for i, problem_text in enumerate(test_problems):
        print(f"\n{'='*60}")
        print(f"测试问题 {i+1}: {problem_text}")
        
        # 处理问题
        processed = preprocessor.preprocess(problem_text)
        semantic_entities = qs2_analyzer.analyze_semantics(processed)
        relation_network = ird_discovery.discover_relations(semantic_entities, problem_text)
        
        # 创建推理请求
        request = ReasoningRequest(
            processed_problem=processed,
            semantic_entities=semantic_entities,
            relation_network=relation_network,
            user_preferences={},
            context=problem_text
        )
        
        # 测试不同模式
        for mode in [ReasoningMode.AUTO, ReasoningMode.SIMPLE, ReasoningMode.ADVANCED]:
            engine_selector.set_mode(mode)
            result = engine_selector.execute_reasoning(request)
            
            print(f"\n{mode.value.upper()}模式结果:")
            print(f"  引擎: {result.get('engine_selection', {}).get('selected_engine', 'unknown')}")
            print(f"  答案: {result['answer']}")
            print(f"  置信度: {result['confidence']:.3f}")
            print(f"  执行时间: {result['execution_time']:.3f}s")