"""
思维图推理策略 (Graph of Thoughts)
实现图状结构的推理策略，适合最复杂的数学问题
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ...core.exceptions import ReasoningError
from ...core.interfaces import ReasoningContext
from .strategy_base import (ReasoningStrategy, StrategyComplexity,
                            StrategyResult, StrategyType)


@dataclass
class ConceptNode:
    """概念节点"""
    id: str
    concept: str
    value: Optional[Any]
    confidence: float
    node_type: str  # 'input', 'intermediate', 'output'
    reasoning_step: Dict[str, Any]
    
@dataclass
class ReasoningEdge:
    """推理边"""
    from_node: str
    to_node: str
    relation_type: str  # 'implies', 'calculates', 'validates', 'transforms'
    weight: float
    reasoning: str

class GraphOfThoughtsStrategy(ReasoningStrategy):
    """思维图推理策略"""
    
    def __init__(self):
        super().__init__(
            name="graph_of_thoughts",
            strategy_type=StrategyType.GRAPH_OF_THOUGHTS,
            complexity=StrategyComplexity.ADVANCED
        )
        
        # 图结构
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: List[ReasoningEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        
        # 推理配置
        self.max_iterations = 10
        self.convergence_threshold = 0.95
        self.min_path_confidence = 0.6
        
        # 概念类型定义
        self.concept_types = {
            "numerical": ["数字", "数量", "值", "结果"],
            "operational": ["加法", "减法", "乘法", "除法", "运算"],
            "relational": ["等于", "大于", "小于", "比例", "关系"],
            "geometric": ["面积", "周长", "体积", "长度", "角度"],
            "temporal": ["时间", "速度", "持续", "频率"],
            "logical": ["条件", "假设", "因此", "所以", "如果"]
        }
    
    def can_handle(self, problem_text: str, context: Optional[ReasoningContext] = None) -> bool:
        """
        判断是否能处理该问题
        
        思维图策略适合：
        - 极其复杂的多步骤问题
        - 包含多种概念关系的问题
        - 需要综合推理的问题
        - 有循环依赖或复杂约束的问题
        """
        try:
            complexity_indicators = 0
            
            # 多概念指标
            concept_count = 0
            for concept_type, keywords in self.concept_types.items():
                if any(keyword in problem_text for keyword in keywords):
                    concept_count += 1
            
            if concept_count >= 4:
                complexity_indicators += 2
            elif concept_count >= 3:
                complexity_indicators += 1
            
            # 复杂关系指标
            complex_relations = ["因为", "由于", "根据", "基于", "考虑到", "结合"]
            if any(relation in problem_text for relation in complex_relations):
                complexity_indicators += 2
            
            # 多层嵌套指标
            nesting_keywords = ["其中", "包括", "分别", "各自", "同时"]
            if any(keyword in problem_text for keyword in nesting_keywords):
                complexity_indicators += 1
            
            # 约束条件指标
            constraint_keywords = ["约束", "限制", "条件", "要求", "满足"]
            if any(keyword in problem_text for keyword in constraint_keywords):
                complexity_indicators += 1
            
            # 问题长度指标
            if len(problem_text) > 300:
                complexity_indicators += 1
            
            # 需要至少5个复杂度指标才适合图推理
            return complexity_indicators >= 5
            
        except Exception as e:
            self.logger.error(f"能力判断失败: {str(e)}")
            return False
    
    def estimate_complexity(self, problem_text: str, context: Optional[ReasoningContext] = None) -> float:
        """估计问题复杂度"""
        try:
            complexity = 0.7  # 图策略的基础复杂度较高
            
            # 概念多样性 (0.0-0.15)
            concept_diversity = len([ct for ct, keywords in self.concept_types.items() 
                                   if any(kw in problem_text for kw in keywords)])
            complexity += min(concept_diversity * 0.025, 0.15)
            
            # 关系复杂度 (0.0-0.1)
            relations = ["因此", "所以", "由于", "基于", "结合", "考虑"]
            relation_count = sum(1 for rel in relations if rel in problem_text)
            complexity += min(relation_count * 0.02, 0.1)
            
            # 约束条件 (0.0-0.1)
            constraints = ["约束", "限制", "条件", "要求", "满足", "必须"]
            constraint_count = sum(1 for con in constraints if con in problem_text)
            complexity += min(constraint_count * 0.02, 0.1)
            
            # 数值复杂度 (0.0-0.05)
            numbers = self._extract_numbers(problem_text)
            if len(numbers) > 5:
                complexity += 0.05
            elif len(numbers) > 3:
                complexity += 0.03
            
            return min(1.0, complexity)
            
        except Exception as e:
            self.logger.error(f"复杂度估计失败: {str(e)}")
            return 0.8  # 默认高复杂度
    
    def _execute_reasoning(self, problem_text: str, context: Optional[ReasoningContext] = None) -> StrategyResult:
        """执行思维图推理"""
        start_time = time.time()
        
        try:
            # 初始化图结构
            self._initialize_graph()
            
            # 构建概念图
            graph_construction = self._construct_concept_graph(problem_text)
            
            # 传播推理
            propagation_result = self._propagate_reasoning()
            
            # 寻找解决方案路径
            solution_paths = self._find_solution_paths()
            
            # 选择最佳路径
            best_path = self._select_best_path(solution_paths)
            
            # 构建推理步骤
            reasoning_steps = self._build_reasoning_steps(best_path)
            
            # 提取最终答案
            final_answer = self._extract_final_answer(best_path)
            
            # 计算置信度
            confidence = self._calculate_graph_confidence(best_path)
            
            execution_time = time.time() - start_time
            
            return StrategyResult(
                success=True,
                answer=str(final_answer),
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                strategy_used=self.name,
                execution_time=execution_time,
                metadata={
                    "nodes_created": len(self.nodes),
                    "edges_created": len(self.edges),
                    "solution_paths": len(solution_paths),
                    "iterations": propagation_result.get("iterations", 0),
                    "convergence": propagation_result.get("converged", False)
                }
            )
            
        except Exception as e:
            self.logger.error(f"思维图推理失败: {str(e)}")
            
            return StrategyResult(
                success=False,
                answer="图推理失败",
                confidence=0.0,
                reasoning_steps=[{
                    "step": 1,
                    "action": "graph_reasoning_error",
                    "description": f"思维图推理失败: {str(e)}",
                    "confidence": 0.0
                }],
                strategy_used=self.name,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _initialize_graph(self):
        """初始化图结构"""
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
    
    def _construct_concept_graph(self, problem_text: str) -> Dict[str, Any]:
        """构建概念图"""
        construction_stats = {
            "input_nodes": 0,
            "intermediate_nodes": 0,
            "output_nodes": 0
        }
        
        # 1. 创建输入节点 (数字、已知条件)
        numbers = self._extract_numbers(problem_text)
        for i, number in enumerate(numbers):
            node_id = f"input_num_{i}"
            node = ConceptNode(
                id=node_id,
                concept=f"数字_{i}",
                value=number,
                confidence=0.95,
                node_type="input",
                reasoning_step={
                    "step": i + 1,
                    "action": "input_extraction",
                    "description": f"提取输入数字: {number}",
                    "confidence": 0.95
                }
            )
            self.nodes[node_id] = node
            construction_stats["input_nodes"] += 1
        
        # 2. 创建概念节点
        concepts = self._extract_concepts(problem_text)
        for i, concept in enumerate(concepts):
            node_id = f"concept_{i}"
            node = ConceptNode(
                id=node_id,
                concept=concept["name"],
                value=concept.get("value"),
                confidence=concept.get("confidence", 0.8),
                node_type="intermediate",
                reasoning_step={
                    "step": len(numbers) + i + 1,
                    "action": "concept_identification",
                    "description": f"识别概念: {concept['name']}",
                    "confidence": concept.get("confidence", 0.8)
                }
            )
            self.nodes[node_id] = node
            construction_stats["intermediate_nodes"] += 1
        
        # 3. 创建输出节点 (目标)
        targets = self._identify_targets(problem_text)
        for i, target in enumerate(targets):
            node_id = f"output_{i}"
            node = ConceptNode(
                id=node_id,
                concept=target["name"],
                value=None,  # 待计算
                confidence=0.5,  # 初始较低置信度
                node_type="output",
                reasoning_step={
                    "step": len(numbers) + len(concepts) + i + 1,
                    "action": "target_identification",
                    "description": f"识别目标: {target['name']}",
                    "confidence": 0.8
                }
            )
            self.nodes[node_id] = node
            construction_stats["output_nodes"] += 1
        
        # 4. 创建连接边
        self._create_reasoning_edges(problem_text)
        
        return construction_stats
    
    def _extract_concepts(self, problem_text: str) -> List[Dict[str, Any]]:
        """提取概念"""
        concepts = []
        
        for concept_type, keywords in self.concept_types.items():
            for keyword in keywords:
                if keyword in problem_text:
                    concepts.append({
                        "name": keyword,
                        "type": concept_type,
                        "confidence": 0.8
                    })
        
        return concepts
    
    def _identify_targets(self, problem_text: str) -> List[Dict[str, Any]]:
        """识别目标"""
        targets = []
        
        target_keywords = ["求", "计算", "多少", "几", "什么", "答案"]
        for keyword in target_keywords:
            if keyword in problem_text:
                targets.append({
                    "name": f"目标_{keyword}",
                    "keyword": keyword
                })
        
        if not targets:
            targets.append({"name": "默认目标", "keyword": "求解"})
        
        return targets
    
    def _create_reasoning_edges(self, problem_text: str):
        """创建推理边"""
        node_ids = list(self.nodes.keys())
        
        # 基于节点类型创建边
        input_nodes = [nid for nid, node in self.nodes.items() if node.node_type == "input"]
        intermediate_nodes = [nid for nid, node in self.nodes.items() if node.node_type == "intermediate"]
        output_nodes = [nid for nid, node in self.nodes.items() if node.node_type == "output"]
        
        # 输入 -> 中间节点
        for input_id in input_nodes:
            for inter_id in intermediate_nodes:
                edge = ReasoningEdge(
                    from_node=input_id,
                    to_node=inter_id,
                    relation_type="provides",
                    weight=0.8,
                    reasoning=f"{input_id} 提供数据给 {inter_id}"
                )
                self.edges.append(edge)
                self.adjacency[input_id].append(inter_id)
        
        # 中间节点 -> 输出节点
        for inter_id in intermediate_nodes:
            for output_id in output_nodes:
                edge = ReasoningEdge(
                    from_node=inter_id,
                    to_node=output_id,
                    relation_type="calculates",
                    weight=0.9,
                    reasoning=f"{inter_id} 计算得到 {output_id}"
                )
                self.edges.append(edge)
                self.adjacency[inter_id].append(output_id)
        
        # 中间节点之间的运算关系
        if len(intermediate_nodes) >= 2:
            for i in range(len(intermediate_nodes) - 1):
                edge = ReasoningEdge(
                    from_node=intermediate_nodes[i],
                    to_node=intermediate_nodes[i + 1],
                    relation_type="transforms",
                    weight=0.7,
                    reasoning=f"{intermediate_nodes[i]} 转换为 {intermediate_nodes[i + 1]}"
                )
                self.edges.append(edge)
                self.adjacency[intermediate_nodes[i]].append(intermediate_nodes[i + 1])
    
    def _propagate_reasoning(self) -> Dict[str, Any]:
        """传播推理信息"""
        result = {
            "iterations": 0,
            "converged": False,
            "confidence_changes": []
        }
        
        for iteration in range(self.max_iterations):
            result["iterations"] = iteration + 1
            confidence_changed = False
            max_change = 0.0
            
            # 遍历所有节点，更新置信度和值
            for node_id, node in self.nodes.items():
                old_confidence = node.confidence
                
                # 基于邻居节点更新置信度
                new_confidence = self._update_node_confidence(node_id)
                
                if abs(new_confidence - old_confidence) > 0.01:
                    confidence_changed = True
                    max_change = max(max_change, abs(new_confidence - old_confidence))
                    node.confidence = new_confidence
                
                # 尝试计算节点值
                if node.value is None and node.node_type == "output":
                    computed_value = self._compute_node_value(node_id)
                    if computed_value is not None:
                        node.value = computed_value
            
            result["confidence_changes"].append(max_change)
            
            # 检查收敛
            if not confidence_changed or max_change < 0.01:
                result["converged"] = True
                break
        
        return result
    
    def _update_node_confidence(self, node_id: str) -> float:
        """更新节点置信度"""
        node = self.nodes[node_id]
        
        if node.node_type == "input":
            return node.confidence  # 输入节点置信度固定
        
        # 收集前驱节点的置信度
        predecessor_confidences = []
        for edge in self.edges:
            if edge.to_node == node_id:
                pred_node = self.nodes[edge.from_node]
                weighted_confidence = pred_node.confidence * edge.weight
                predecessor_confidences.append(weighted_confidence)
        
        if not predecessor_confidences:
            return node.confidence
        
        # 使用加权平均
        return sum(predecessor_confidences) / len(predecessor_confidences)
    
    def _compute_node_value(self, node_id: str) -> Optional[Any]:
        """计算节点值"""
        node = self.nodes[node_id]
        
        if node.node_type == "input":
            return node.value  # 输入节点值已知
        
        # 收集前驱节点的值
        predecessor_values = []
        for edge in self.edges:
            if edge.to_node == node_id and edge.relation_type in ["calculates", "transforms"]:
                pred_node = self.nodes[edge.from_node]
                if pred_node.value is not None:
                    predecessor_values.append(pred_node.value)
        
        if len(predecessor_values) >= 2:
            # 简单的运算逻辑
            if "加" in node.concept or "总" in node.concept:
                return sum(predecessor_values)
            elif "减" in node.concept:
                return predecessor_values[0] - sum(predecessor_values[1:])
            elif "乘" in node.concept:
                result = 1
                for val in predecessor_values:
                    result *= val
                return result
            elif "除" in node.concept and len(predecessor_values) >= 2:
                if predecessor_values[1] != 0:
                    return predecessor_values[0] / predecessor_values[1]
        
        return None
    
    def _find_solution_paths(self) -> List[List[str]]:
        """寻找解决方案路径"""
        paths = []
        
        input_nodes = [nid for nid, node in self.nodes.items() if node.node_type == "input"]
        output_nodes = [nid for nid, node in self.nodes.items() if node.node_type == "output"]
        
        # 从每个输入节点到每个输出节点寻找路径
        for input_id in input_nodes:
            for output_id in output_nodes:
                path = self._find_path(input_id, output_id)
                if path:
                    paths.append(path)
        
        return paths
    
    def _find_path(self, start: str, end: str) -> Optional[List[str]]:
        """使用BFS寻找路径"""
        if start == end:
            return [start]
        
        visited = set()
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _select_best_path(self, paths: List[List[str]]) -> List[str]:
        """选择最佳路径"""
        if not paths:
            return list(self.nodes.keys())[:3]  # 返回前3个节点作为默认路径
        
        # 计算每条路径的得分
        path_scores = []
        for path in paths:
            score = self._calculate_path_score(path)
            path_scores.append((path, score))
        
        # 返回得分最高的路径
        best_path = max(path_scores, key=lambda x: x[1])[0]
        return best_path
    
    def _calculate_path_score(self, path: List[str]) -> float:
        """计算路径得分"""
        if not path:
            return 0.0
        
        # 基于路径上节点的置信度
        confidences = [self.nodes[node_id].confidence for node_id in path]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 基于路径长度 (适中的长度更好)
        length_penalty = abs(len(path) - 4) * 0.1  # 理想长度为4
        
        # 基于是否有计算结果
        has_result = any(self.nodes[node_id].value is not None for node_id in path)
        result_bonus = 0.2 if has_result else 0.0
        
        return avg_confidence - length_penalty + result_bonus
    
    def _build_reasoning_steps(self, path: List[str]) -> List[Dict[str, Any]]:
        """构建推理步骤"""
        steps = []
        
        for i, node_id in enumerate(path):
            node = self.nodes[node_id]
            step = {
                "step": i + 1,
                "action": node.reasoning_step.get("action", "reasoning"),
                "description": node.reasoning_step.get("description", f"处理概念: {node.concept}"),
                "concept": node.concept,
                "value": node.value,
                "confidence": node.confidence,
                "node_type": node.node_type
            }
            steps.append(step)
        
        return steps
    
    def _extract_final_answer(self, path: List[str]) -> Any:
        """提取最终答案"""
        # 寻找路径中最后一个有值的节点
        for node_id in reversed(path):
            node = self.nodes[node_id]
            if node.value is not None:
                return node.value
        
        # 如果没有找到，返回默认值
        return "未找到答案"
    
    def _calculate_graph_confidence(self, path: List[str]) -> float:
        """计算图推理的整体置信度"""
        if not path:
            return 0.0
        
        # 收集路径上所有节点的置信度
        confidences = [self.nodes[node_id].confidence for node_id in path]
        
        # 收集相关边的权重
        edge_weights = []
        for i in range(len(path) - 1):
            for edge in self.edges:
                if edge.from_node == path[i] and edge.to_node == path[i + 1]:
                    edge_weights.append(edge.weight)
                    break
        
        # 综合置信度计算
        node_confidence = sum(confidences) / len(confidences)
        edge_confidence = sum(edge_weights) / len(edge_weights) if edge_weights else 0.5
        
        # 加权组合
        overall_confidence = 0.7 * node_confidence + 0.3 * edge_confidence
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _extract_numbers(self, text: str) -> List[float]:
        """提取数字"""
        import re
        pattern = r'\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches] 