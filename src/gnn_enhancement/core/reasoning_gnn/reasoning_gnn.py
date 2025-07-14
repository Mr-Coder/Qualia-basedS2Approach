"""
Reasoning Graph Neural Network
==============================

推理过程图神经网络实现

用于优化多层级推理（MLR）过程，构建推理步骤之间的依赖图，
学习最优推理路径，提高推理效率和准确性。

主要功能:
1. 构建推理步骤依赖图
2. 学习推理路径优化
3. 提供推理步骤排序
4. 支持动态推理调整

Author: AI Assistant
Date: 2024-07-13
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# 条件导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ReasoningGNN:
    """
    推理过程图神经网络
    
    核心功能：
    - 构建推理步骤之间的依赖图
    - 学习最优推理路径
    - 提供推理步骤排序和优化
    - 支持动态推理调整
    """
    
    def __init__(self, step_dim: int = 128, hidden_dim: int = 256, 
                 num_layers: int = 3, dropout: float = 0.1):
        """
        初始化推理GNN
        
        Args:
            step_dim: 推理步骤嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout率
        """
        self.step_dim = step_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 推理步骤嵌入
        self.step_embeddings = {}
        
        # 推理步骤类型定义
        self.step_types = {
            "extraction": 0,      # 信息提取
            "calculation": 1,     # 数值计算
            "reasoning": 2,       # 逻辑推理
            "verification": 3,    # 结果验证
            "transformation": 4,  # 数据转换
            "comparison": 5,      # 比较分析
            "synthesis": 6,       # 综合判断
            "application": 7      # 规则应用
        }
        
        # 推理操作类型
        self.operation_types = {
            "arithmetic": ["add", "subtract", "multiply", "divide"],
            "logical": ["and", "or", "not", "implies"],
            "comparison": ["greater", "less", "equal", "between"],
            "transformation": ["convert", "normalize", "scale", "round"],
            "extraction": ["identify", "extract", "parse", "recognize"],
            "verification": ["check", "validate", "confirm", "test"]
        }
        
        # 初始化GNN模型
        self.model = None
        if TORCH_AVAILABLE:
            self.model = self._build_reasoning_gnn_model()
        
        # 推理图
        self.reasoning_graph = None
        if DGL_AVAILABLE:
            self.reasoning_graph = self._initialize_reasoning_graph()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"ReasoningGNN initialized with dim={step_dim}, hidden={hidden_dim}")
    
    def _build_reasoning_gnn_model(self):
        """构建推理GNN模型"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, using fallback implementation")
            return None
            
        class ReasoningGNNModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
                super(ReasoningGNNModel, self).__init__()
                self.num_layers = num_layers
                self.dropout = dropout
                
                # 输入投影层
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                
                # GNN层 - 使用GAT进行注意力机制
                self.gnn_layers = nn.ModuleList()
                for i in range(num_layers):
                    if DGL_AVAILABLE:
                        self.gnn_layers.append(dglnn.GATConv(hidden_dim, hidden_dim, num_heads=4))
                    else:
                        # 使用自注意力作为fallback
                        self.gnn_layers.append(nn.MultiheadAttention(hidden_dim, num_heads=4))
                
                # 输出层
                self.output_proj = nn.Linear(hidden_dim, output_dim)
                self.dropout_layer = nn.Dropout(dropout)
                
                # 推理路径预测器
                self.path_predictor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, graph, features):
                h = self.input_proj(features)
                
                for i, layer in enumerate(self.gnn_layers):
                    if DGL_AVAILABLE and hasattr(layer, 'forward'):
                        h = layer(graph, h).mean(dim=1)  # 平均多头注意力
                    else:
                        h, _ = layer(h, h, h)
                    
                    if i < len(self.gnn_layers) - 1:
                        h = F.relu(h)
                        h = self.dropout_layer(h)
                
                output = self.output_proj(h)
                path_scores = self.path_predictor(h)
                
                return output, path_scores
        
        return ReasoningGNNModel(
            input_dim=self.step_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.step_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
    
    def _initialize_reasoning_graph(self):
        """初始化推理图"""
        if not DGL_AVAILABLE:
            self.logger.warning("DGL not available, using alternative graph representation")
            return {"nodes": [], "edges": []}
        
        # 创建空图
        graph = dgl.graph(([], []))
        return graph
    
    def build_reasoning_graph(self, reasoning_steps: List[Dict[str, Any]], 
                            problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建推理步骤依赖图
        
        Args:
            reasoning_steps: 推理步骤列表
            problem_context: 问题上下文
            
        Returns:
            推理图信息
        """
        try:
            # 1. 分析推理步骤
            analyzed_steps = self._analyze_reasoning_steps(reasoning_steps)
            
            # 2. 构建步骤依赖关系
            dependencies = self._build_step_dependencies(analyzed_steps, problem_context)
            
            # 3. 构建推理图
            graph_info = self._construct_reasoning_graph(analyzed_steps, dependencies)
            
            # 4. 更新推理图
            if DGL_AVAILABLE and self.reasoning_graph is not None:
                self.reasoning_graph = self._update_reasoning_dgl_graph(analyzed_steps, dependencies)
            
            return {
                "steps": analyzed_steps,
                "dependencies": dependencies,
                "graph_info": graph_info,
                "num_steps": len(analyzed_steps),
                "num_dependencies": len(dependencies)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to build reasoning graph: {e}")
            return {"error": str(e)}
    
    def _analyze_reasoning_steps(self, reasoning_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析推理步骤"""
        analyzed_steps = []
        
        for i, step in enumerate(reasoning_steps):
            # 提取步骤信息
            step_info = {
                "id": i,
                "description": step.get("description", ""),
                "action": step.get("action", "unknown"),
                "inputs": step.get("inputs", []),
                "outputs": step.get("outputs", []),
                "confidence": step.get("confidence", 0.5),
                "step_type": self._classify_step_type(step),
                "operation_type": self._classify_operation_type(step),
                "embedding": self._get_step_embedding(step)
            }
            
            # 计算步骤复杂度
            step_info["complexity"] = self._calculate_step_complexity(step_info)
            
            analyzed_steps.append(step_info)
        
        return analyzed_steps
    
    def _classify_step_type(self, step: Dict[str, Any]) -> str:
        """分类推理步骤类型"""
        action = step.get("action", "").lower()
        description = step.get("description", "").lower()
        
        # 基于动作和描述分类
        if any(keyword in action for keyword in ["extract", "identify", "parse"]):
            return "extraction"
        elif any(keyword in action for keyword in ["calculate", "compute", "solve"]):
            return "calculation"
        elif any(keyword in action for keyword in ["reason", "infer", "deduce"]):
            return "reasoning"
        elif any(keyword in action for keyword in ["verify", "check", "validate"]):
            return "verification"
        elif any(keyword in action for keyword in ["convert", "transform", "normalize"]):
            return "transformation"
        elif any(keyword in action for keyword in ["compare", "contrast", "analyze"]):
            return "comparison"
        elif any(keyword in action for keyword in ["synthesize", "combine", "integrate"]):
            return "synthesis"
        elif any(keyword in action for keyword in ["apply", "use", "implement"]):
            return "application"
        
        return "reasoning"  # 默认类型
    
    def _classify_operation_type(self, step: Dict[str, Any]) -> str:
        """分类操作类型"""
        action = step.get("action", "").lower()
        description = step.get("description", "").lower()
        
        # 检查各种操作类型
        for op_type, keywords in self.operation_types.items():
            if any(keyword in action or keyword in description for keyword in keywords):
                return op_type
        
        return "unknown"
    
    def _get_step_embedding(self, step: Dict[str, Any]) -> np.ndarray:
        """获取步骤嵌入"""
        step_key = f"{step.get('action', 'unknown')}_{step.get('description', '')[:50]}"
        
        if step_key in self.step_embeddings:
            return self.step_embeddings[step_key]
        
        # 生成步骤嵌入
        embedding = np.random.normal(0, 0.1, self.step_dim)
        
        # 基于步骤类型调整嵌入
        step_type = self._classify_step_type(step)
        if step_type in self.step_types:
            type_idx = self.step_types[step_type]
            embedding[type_idx] = 1.0  # 类型标识
        
        # 基于置信度调整
        confidence = step.get("confidence", 0.5)
        embedding[-1] = confidence
        
        self.step_embeddings[step_key] = embedding
        return embedding
    
    def _calculate_step_complexity(self, step_info: Dict[str, Any]) -> float:
        """计算步骤复杂度"""
        complexity = 0.0
        
        # 基于输入输出数量
        num_inputs = len(step_info.get("inputs", []))
        num_outputs = len(step_info.get("outputs", []))
        complexity += (num_inputs + num_outputs) * 0.1
        
        # 基于步骤类型
        type_complexity = {
            "extraction": 0.2,
            "calculation": 0.4,
            "reasoning": 0.8,
            "verification": 0.3,
            "transformation": 0.5,
            "comparison": 0.6,
            "synthesis": 0.9,
            "application": 0.7
        }
        complexity += type_complexity.get(step_info["step_type"], 0.5)
        
        # 基于描述长度
        desc_length = len(step_info.get("description", ""))
        complexity += min(desc_length / 100.0, 0.3)
        
        return min(complexity, 1.0)
    
    def _build_step_dependencies(self, steps: List[Dict[str, Any]], 
                                problem_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """构建步骤依赖关系"""
        dependencies = []
        
        # 基于输入输出构建依赖
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps):
                if i >= j:  # 避免自依赖和重复
                    continue
                
                # 检查数据依赖
                dependency_type, strength = self._check_step_dependency(step1, step2)
                
                if strength > 0.1:
                    dependencies.append({
                        "source": i,
                        "target": j,
                        "type": dependency_type,
                        "strength": strength,
                        "source_step": step1,
                        "target_step": step2
                    })
        
        # 基于逻辑顺序构建依赖
        logical_deps = self._infer_logical_dependencies(steps)
        dependencies.extend(logical_deps)
        
        return dependencies
    
    def _check_step_dependency(self, step1: Dict[str, Any], 
                              step2: Dict[str, Any]) -> Tuple[str, float]:
        """检查步骤间依赖关系"""
        # 数据依赖：step1的输出是step2的输入
        outputs1 = set(step1.get("outputs", []))
        inputs2 = set(step2.get("inputs", []))
        
        if outputs1.intersection(inputs2):
            return "data_dependency", 0.9
        
        # 类型依赖：某些类型的步骤需要特定顺序
        type1 = step1["step_type"]
        type2 = step2["step_type"]
        
        type_dependencies = {
            ("extraction", "calculation"): 0.7,
            ("calculation", "reasoning"): 0.6,
            ("reasoning", "verification"): 0.8,
            ("transformation", "calculation"): 0.6,
            ("comparison", "synthesis"): 0.5
        }
        
        if (type1, type2) in type_dependencies:
            return "type_dependency", type_dependencies[(type1, type2)]
        
        # 语义依赖：基于描述的相似性
        desc1 = step1.get("description", "").lower()
        desc2 = step2.get("description", "").lower()
        
        # 简单的关键词匹配
        common_keywords = self._find_common_keywords(desc1, desc2)
        if len(common_keywords) > 0:
            return "semantic_dependency", min(len(common_keywords) * 0.2, 0.6)
        
        return "no_dependency", 0.0
    
    def _find_common_keywords(self, text1: str, text2: str) -> List[str]:
        """查找共同关键词"""
        import re

        # 提取关键词
        keywords1 = set(re.findall(r'\b\w+\b', text1.lower()))
        keywords2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # 过滤停用词
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords1 -= stopwords
        keywords2 -= stopwords
        
        return list(keywords1.intersection(keywords2))
    
    def _infer_logical_dependencies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """推断逻辑依赖关系"""
        logical_deps = []
        
        # 基于步骤顺序的逻辑依赖
        for i in range(len(steps) - 1):
            step1 = steps[i]
            step2 = steps[i + 1]
            
            # 检查是否需要严格顺序
            if self._requires_strict_order(step1, step2):
                logical_deps.append({
                    "source": i,
                    "target": i + 1,
                    "type": "sequential_dependency",
                    "strength": 0.8,
                    "source_step": step1,
                    "target_step": step2
                })
        
        return logical_deps
    
    def _requires_strict_order(self, step1: Dict[str, Any], step2: Dict[str, Any]) -> bool:
        """检查是否需要严格顺序"""
        type1 = step1["step_type"]
        type2 = step2["step_type"]
        
        # 某些类型组合需要严格顺序
        strict_orders = [
            ("extraction", "calculation"),
            ("calculation", "verification"),
            ("transformation", "application"),
            ("reasoning", "synthesis")
        ]
        
        return (type1, type2) in strict_orders
    
    def _construct_reasoning_graph(self, steps: List[Dict[str, Any]], 
                                  dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建推理图结构"""
        graph_info = {
            "nodes": [],
            "edges": [],
            "adjacency_matrix": None
        }
        
        # 构建节点
        for step in steps:
            graph_info["nodes"].append({
                "id": step["id"],
                "description": step["description"],
                "step_type": step["step_type"],
                "operation_type": step["operation_type"],
                "complexity": step["complexity"],
                "confidence": step["confidence"],
                "embedding": step["embedding"].tolist()
            })
        
        # 构建边
        for dep in dependencies:
            graph_info["edges"].append({
                "source": dep["source"],
                "target": dep["target"],
                "type": dep["type"],
                "strength": dep["strength"]
            })
        
        # 构建邻接矩阵
        n_steps = len(steps)
        if n_steps > 0:
            adj_matrix = np.zeros((n_steps, n_steps))
            for dep in dependencies:
                i, j = dep["source"], dep["target"]
                adj_matrix[i][j] = dep["strength"]
            graph_info["adjacency_matrix"] = adj_matrix.tolist()
        
        return graph_info
    
    def _update_reasoning_dgl_graph(self, steps: List[Dict[str, Any]], 
                                   dependencies: List[Dict[str, Any]]):
        """更新推理DGL图"""
        if not DGL_AVAILABLE:
            return None
        
        # 构建边列表
        src_nodes = []
        dst_nodes = []
        edge_weights = []
        
        for dep in dependencies:
            src_nodes.append(dep["source"])
            dst_nodes.append(dep["target"])
            edge_weights.append(dep["strength"])
        
        if not src_nodes:  # 没有边的情况
            return dgl.graph(([], []))
        
        # 创建图
        graph = dgl.graph((src_nodes, dst_nodes))
        
        # 添加节点特征
        node_features = []
        for step in steps:
            node_features.append(step["embedding"])
        
        if node_features:
            graph.ndata['feat'] = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # 添加边权重
        if edge_weights:
            graph.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
        
        return graph
    
    def optimize_reasoning_path(self, reasoning_steps: List[Dict[str, Any]], 
                               problem_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        优化推理路径
        
        Args:
            reasoning_steps: 原始推理步骤
            problem_context: 问题上下文
            
        Returns:
            优化后的推理步骤
        """
        try:
            # 1. 构建推理图
            reasoning_graph = self.build_reasoning_graph(reasoning_steps, problem_context)
            
            # 2. 计算最优路径
            optimal_path = self._find_optimal_path(reasoning_graph)
            
            # 3. 重排推理步骤
            optimized_steps = self._reorder_steps(reasoning_steps, optimal_path)
            
            # 4. 添加优化信息
            for i, step in enumerate(optimized_steps):
                step["optimization_score"] = optimal_path.get(i, 0.5)
                step["original_order"] = reasoning_steps.index(step) if step in reasoning_steps else i
            
            return optimized_steps
            
        except Exception as e:
            self.logger.error(f"Failed to optimize reasoning path: {e}")
            return reasoning_steps
    
    def _find_optimal_path(self, reasoning_graph: Dict[str, Any]) -> Dict[int, float]:
        """寻找最优推理路径"""
        if "graph_info" not in reasoning_graph:
            return {}
        
        graph_info = reasoning_graph["graph_info"]
        nodes = graph_info["nodes"]
        edges = graph_info["edges"]
        
        # 计算节点重要性分数
        node_scores = {}
        for node in nodes:
            score = 0.0
            
            # 基于复杂度和置信度
            complexity = node["complexity"]
            confidence = node["confidence"]
            score += confidence * (1.0 - complexity * 0.5)
            
            # 基于连接度
            in_degree = sum(1 for edge in edges if edge["target"] == node["id"])
            out_degree = sum(1 for edge in edges if edge["source"] == node["id"])
            score += (in_degree + out_degree) * 0.1
            
            node_scores[node["id"]] = score
        
        return node_scores
    
    def _reorder_steps(self, original_steps: List[Dict[str, Any]], 
                      path_scores: Dict[int, float]) -> List[Dict[str, Any]]:
        """根据路径分数重排步骤"""
        # 为每个步骤分配分数
        step_scores = []
        for i, step in enumerate(original_steps):
            score = path_scores.get(i, 0.5)
            step_scores.append((step, score, i))
        
        # 按分数排序，但保持依赖关系
        sorted_steps = sorted(step_scores, key=lambda x: x[1], reverse=True)
        
        # 重新排列，确保依赖关系
        reordered_steps = []
        used_indices = set()
        
        for step, score, original_idx in sorted_steps:
            if original_idx not in used_indices:
                reordered_steps.append(step)
                used_indices.add(original_idx)
        
        return reordered_steps
    
    def get_reasoning_quality_score(self, reasoning_steps: List[Dict[str, Any]], 
                                   problem_context: Dict[str, Any]) -> float:
        """
        计算推理质量分数
        
        Args:
            reasoning_steps: 推理步骤
            problem_context: 问题上下文
            
        Returns:
            推理质量分数 (0-1)
        """
        try:
            # 构建推理图
            reasoning_graph = self.build_reasoning_graph(reasoning_steps, problem_context)
            
            if "steps" not in reasoning_graph:
                return 0.5
            
            steps = reasoning_graph["steps"]
            dependencies = reasoning_graph["dependencies"]
            
            # 计算各项质量指标
            coherence_score = self._calculate_coherence_score(steps, dependencies)
            completeness_score = self._calculate_completeness_score(steps, problem_context)
            efficiency_score = self._calculate_efficiency_score(steps, dependencies)
            
            # 加权平均
            quality_score = (
                coherence_score * 0.4 +
                completeness_score * 0.3 +
                efficiency_score * 0.3
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate reasoning quality: {e}")
            return 0.5
    
    def _calculate_coherence_score(self, steps: List[Dict[str, Any]], 
                                  dependencies: List[Dict[str, Any]]) -> float:
        """计算推理连贯性分数"""
        if not steps:
            return 0.0
        
        # 计算依赖关系的强度
        total_strength = sum(dep["strength"] for dep in dependencies)
        avg_strength = total_strength / max(len(dependencies), 1)
        
        # 计算步骤间的逻辑连贯性
        coherence = 0.0
        for i in range(len(steps) - 1):
            step1, step2 = steps[i], steps[i + 1]
            
            # 检查类型连贯性
            type1, type2 = step1["step_type"], step2["step_type"]
            if self._are_types_coherent(type1, type2):
                coherence += 0.2
            
            # 检查操作连贯性
            op1, op2 = step1["operation_type"], step2["operation_type"]
            if self._are_operations_coherent(op1, op2):
                coherence += 0.1
        
        coherence /= max(len(steps) - 1, 1)
        
        return (avg_strength + coherence) / 2.0
    
    def _are_types_coherent(self, type1: str, type2: str) -> bool:
        """检查类型是否连贯"""
        coherent_sequences = [
            ("extraction", "calculation"),
            ("calculation", "reasoning"),
            ("reasoning", "verification"),
            ("transformation", "application"),
            ("comparison", "synthesis")
        ]
        
        return (type1, type2) in coherent_sequences
    
    def _are_operations_coherent(self, op1: str, op2: str) -> bool:
        """检查操作是否连贯"""
        # 同类型操作是连贯的
        if op1 == op2:
            return True
        
        # 特定操作序列是连贯的
        coherent_ops = [
            ("extraction", "arithmetic"),
            ("arithmetic", "comparison"),
            ("comparison", "logical"),
            ("transformation", "verification")
        ]
        
        return (op1, op2) in coherent_ops
    
    def _calculate_completeness_score(self, steps: List[Dict[str, Any]], 
                                     problem_context: Dict[str, Any]) -> float:
        """计算推理完整性分数"""
        if not steps:
            return 0.0
        
        # 检查必要步骤类型是否存在
        required_types = {"extraction", "calculation", "reasoning"}
        present_types = set(step["step_type"] for step in steps)
        
        completeness = len(required_types.intersection(present_types)) / len(required_types)
        
        # 检查是否有验证步骤
        if "verification" in present_types:
            completeness += 0.2
        
        return min(1.0, completeness)
    
    def _calculate_efficiency_score(self, steps: List[Dict[str, Any]], 
                                   dependencies: List[Dict[str, Any]]) -> float:
        """计算推理效率分数"""
        if not steps:
            return 0.0
        
        # 计算平均复杂度
        avg_complexity = sum(step["complexity"] for step in steps) / len(steps)
        
        # 计算冗余度
        redundancy = self._calculate_redundancy(steps)
        
        # 效率分数 = 1 - 复杂度 - 冗余度
        efficiency = 1.0 - avg_complexity - redundancy
        
        return max(0.0, efficiency)
    
    def _calculate_redundancy(self, steps: List[Dict[str, Any]]) -> float:
        """计算冗余度"""
        if len(steps) <= 1:
            return 0.0
        
        # 检查重复的步骤类型
        type_counts = {}
        for step in steps:
            step_type = step["step_type"]
            type_counts[step_type] = type_counts.get(step_type, 0) + 1
        
        # 计算冗余度
        redundancy = 0.0
        for count in type_counts.values():
            if count > 1:
                redundancy += (count - 1) * 0.1
        
        return min(redundancy, 0.5)
    
    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            "name": "ReasoningGNN",
            "version": "1.0.0",
            "step_dim": self.step_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_step_types": len(self.step_types),
            "num_operation_types": len(self.operation_types),
            "torch_available": TORCH_AVAILABLE,
            "dgl_available": DGL_AVAILABLE,
            "model_loaded": self.model is not None
        } 