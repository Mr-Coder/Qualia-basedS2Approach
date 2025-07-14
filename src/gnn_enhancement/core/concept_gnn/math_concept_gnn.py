"""
Math Concept Graph Neural Network
=================================

数学概念图神经网络实现

用于构建数学概念之间的关系图，学习概念间的隐式关系，
增强IRD（隐式关系发现）模块的能力。

主要功能:
1. 构建数学概念图
2. 学习概念间的隐式关系
3. 提供概念相似度计算
4. 支持动态概念图更新

Author: AI Assistant
Date: 2024-07-13
"""

import logging
import math
import re
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


class MathConceptGNN:
    """
    数学概念图神经网络
    
    核心功能：
    - 构建数学概念之间的关系图
    - 学习概念间的隐式关系
    - 增强关系发现能力
    - 提供概念嵌入和相似度计算
    """
    
    def __init__(self, concept_dim: int = 128, hidden_dim: int = 256, 
                 num_layers: int = 3, dropout: float = 0.1):
        """
        初始化数学概念GNN
        
        Args:
            concept_dim: 概念嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout率
        """
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 概念嵌入字典
        self.concept_embeddings = {}
        
        # 关系类型定义
        self.relation_types = {
            "arithmetic_relation": 0,    # 算术关系
            "geometric_relation": 1,     # 几何关系
            "algebraic_relation": 2,     # 代数关系
            "logical_relation": 3,       # 逻辑关系
            "temporal_relation": 4,      # 时间关系
            "unit_relation": 5,          # 单位关系
            "proportional_relation": 6,  # 比例关系
            "physical_relation": 7       # 物理关系
        }
        
        # 数学概念词典
        self.math_concepts = {
            # 基础概念
            "number": ["数字", "数值", "数量", "个数", "amount", "quantity"],
            "operation": ["运算", "计算", "操作", "operation", "calculation"],
            "equation": ["方程", "等式", "equation", "formula"],
            "variable": ["变量", "未知数", "variable", "unknown"],
            
            # 几何概念
            "area": ["面积", "area", "surface"],
            "volume": ["体积", "容积", "volume", "capacity"],
            "length": ["长度", "长", "length", "distance"],
            "width": ["宽度", "宽", "width"],
            "height": ["高度", "高", "height"],
            "radius": ["半径", "radius"],
            "diameter": ["直径", "diameter"],
            
            # 时间概念
            "time": ["时间", "时刻", "time", "moment"],
            "speed": ["速度", "速率", "speed", "velocity", "rate"],
            "duration": ["持续时间", "duration", "period"],
            
            # 单位概念
            "unit": ["单位", "unit", "measurement"],
            "meter": ["米", "m", "meter"],
            "liter": ["升", "L", "l", "liter"],
            "kilogram": ["千克", "kg", "kilogram"],
            "second": ["秒", "s", "second"],
            "minute": ["分钟", "min", "minute"],
            "hour": ["小时", "h", "hour"]
        }
        
        # 初始化GNN模型
        self.model = None
        if TORCH_AVAILABLE:
            self.model = self._build_gnn_model()
        
        # 概念图
        self.concept_graph = None
        if DGL_AVAILABLE:
            self.concept_graph = self._initialize_concept_graph()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"MathConceptGNN initialized with dim={concept_dim}, hidden={hidden_dim}")
    
    def _build_gnn_model(self):
        """构建GNN模型"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, using fallback implementation")
            return None
            
        class ConceptGNNModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
                super(ConceptGNNModel, self).__init__()
                self.num_layers = num_layers
                self.dropout = dropout
                
                # 输入投影层
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                
                # GNN层
                self.gnn_layers = nn.ModuleList()
                for i in range(num_layers):
                    if DGL_AVAILABLE:
                        self.gnn_layers.append(dglnn.GraphConv(hidden_dim, hidden_dim))
                    else:
                        # 使用简单的线性层作为fallback
                        self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
                
                # 输出层
                self.output_proj = nn.Linear(hidden_dim, output_dim)
                self.dropout_layer = nn.Dropout(dropout)
                
            def forward(self, graph, features):
                h = self.input_proj(features)
                
                for i, layer in enumerate(self.gnn_layers):
                    if DGL_AVAILABLE and hasattr(layer, 'forward'):
                        h = layer(graph, h)
                    else:
                        h = layer(h)
                    
                    if i < len(self.gnn_layers) - 1:
                        h = F.relu(h)
                        h = self.dropout_layer(h)
                
                return self.output_proj(h)
        
        return ConceptGNNModel(
            input_dim=self.concept_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.concept_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
    
    def _initialize_concept_graph(self):
        """初始化概念图"""
        if not DGL_AVAILABLE:
            self.logger.warning("DGL not available, using alternative graph representation")
            return {"nodes": [], "edges": []}
        
        # 创建空图
        graph = dgl.graph(([], []))
        return graph
    
    def build_concept_graph(self, problem_text: str, entities: List[str]) -> Dict[str, Any]:
        """
        从问题文本构建概念图
        
        Args:
            problem_text: 问题文本
            entities: 识别出的实体列表
            
        Returns:
            概念图信息
        """
        try:
            # 1. 概念识别
            concepts = self._identify_concepts(problem_text, entities)
            
            # 2. 关系构建
            relations = self._build_concept_relations(concepts, problem_text)
            
            # 3. 图构建
            graph_info = self._construct_graph(concepts, relations)
            
            # 4. 更新概念图
            if DGL_AVAILABLE and self.concept_graph is not None:
                self.concept_graph = self._update_dgl_graph(concepts, relations)
            
            return {
                "concepts": concepts,
                "relations": relations,
                "graph_info": graph_info,
                "num_concepts": len(concepts),
                "num_relations": len(relations)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to build concept graph: {e}")
            return {"error": str(e)}
    
    def _identify_concepts(self, problem_text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """识别数学概念"""
        concepts = []
        text_lower = problem_text.lower()
        
        # 从实体中识别概念
        for entity in entities:
            entity_lower = entity.lower()
            concept_type = self._classify_concept(entity_lower)
            if concept_type:
                concepts.append({
                    "text": entity,
                    "type": concept_type,
                    "embedding": self._get_concept_embedding(entity_lower),
                    "confidence": 0.8
                })
        
        # 从预定义概念词典中识别
        for concept_type, keywords in self.math_concepts.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    concepts.append({
                        "text": keyword,
                        "type": concept_type,
                        "embedding": self._get_concept_embedding(keyword.lower()),
                        "confidence": 0.9
                    })
        
        # 去重
        unique_concepts = []
        seen_texts = set()
        for concept in concepts:
            if concept["text"] not in seen_texts:
                unique_concepts.append(concept)
                seen_texts.add(concept["text"])
        
        return unique_concepts
    
    def _classify_concept(self, concept_text: str) -> Optional[str]:
        """分类概念类型"""
        # 数字模式
        if re.match(r'\d+(\.\d+)?', concept_text):
            return "number"
        
        # 单位模式
        unit_patterns = [
            r'(cm|m|km|mm)$',  # 长度单位
            r'(l|ml|L)$',      # 体积单位
            r'(kg|g|mg)$',     # 重量单位
            r'(s|min|h)$',     # 时间单位
        ]
        for pattern in unit_patterns:
            if re.search(pattern, concept_text):
                return "unit"
        
        # 从概念词典中查找
        for concept_type, keywords in self.math_concepts.items():
            if concept_text in [kw.lower() for kw in keywords]:
                return concept_type
        
        return None
    
    def _get_concept_embedding(self, concept: str) -> np.ndarray:
        """获取概念嵌入"""
        if concept in self.concept_embeddings:
            return self.concept_embeddings[concept]
        
        # 简单的嵌入生成（实际应用中可以使用预训练的词向量）
        embedding = np.random.normal(0, 0.1, self.concept_dim)
        
        # 根据概念类型调整嵌入
        concept_type = self._classify_concept(concept)
        if concept_type:
            type_bias = hash(concept_type) % 100 / 100.0
            embedding[0] = type_bias
        
        self.concept_embeddings[concept] = embedding
        return embedding
    
    def _build_concept_relations(self, concepts: List[Dict[str, Any]], 
                               problem_text: str) -> List[Dict[str, Any]]:
        """构建概念间关系"""
        relations = []
        
        # 两两概念间构建关系
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:  # 避免重复和自关系
                    continue
                
                # 计算关系类型和强度
                relation_type, strength = self._compute_relation(
                    concept1, concept2, problem_text
                )
                
                if strength > 0.1:  # 只保留强度较高的关系
                    relations.append({
                        "source": concept1["text"],
                        "target": concept2["text"],
                        "type": relation_type,
                        "strength": strength,
                        "source_idx": i,
                        "target_idx": j
                    })
        
        return relations
    
    def _compute_relation(self, concept1: Dict[str, Any], concept2: Dict[str, Any], 
                         problem_text: str) -> Tuple[str, float]:
        """计算两个概念间的关系类型和强度"""
        type1, type2 = concept1["type"], concept2["type"]
        text1, text2 = concept1["text"], concept2["text"]
        
        # 基于概念类型的关系推理
        if type1 == "number" and type2 == "unit":
            return "unit_relation", 0.9
        elif type1 == "area" and type2 == "length":
            return "geometric_relation", 0.8
        elif type1 == "speed" and type2 == "time":
            return "temporal_relation", 0.8
        elif type1 == "volume" and type2 == "liter":
            return "unit_relation", 0.9
        
        # 基于文本共现的关系推理
        text_lower = problem_text.lower()
        if text1.lower() in text_lower and text2.lower() in text_lower:
            # 计算文本距离
            pos1 = text_lower.find(text1.lower())
            pos2 = text_lower.find(text2.lower())
            if pos1 != -1 and pos2 != -1:
                distance = abs(pos1 - pos2)
                if distance < 50:  # 距离较近
                    return "logical_relation", max(0.5, 1.0 - distance / 100.0)
        
        # 基于嵌入相似度的关系推理
        emb1, emb2 = concept1["embedding"], concept2["embedding"]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        if similarity > 0.7:
            return "logical_relation", similarity
        
        return "unknown_relation", 0.0
    
    def _construct_graph(self, concepts: List[Dict[str, Any]], 
                        relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建图结构"""
        graph_info = {
            "nodes": [],
            "edges": [],
            "adjacency_matrix": None
        }
        
        # 构建节点
        for i, concept in enumerate(concepts):
            graph_info["nodes"].append({
                "id": i,
                "text": concept["text"],
                "type": concept["type"],
                "embedding": concept["embedding"].tolist()
            })
        
        # 构建边
        for relation in relations:
            graph_info["edges"].append({
                "source": relation["source_idx"],
                "target": relation["target_idx"],
                "type": relation["type"],
                "weight": relation["strength"]
            })
        
        # 构建邻接矩阵
        n_nodes = len(concepts)
        if n_nodes > 0:
            adj_matrix = np.zeros((n_nodes, n_nodes))
            for relation in relations:
                i, j = relation["source_idx"], relation["target_idx"]
                adj_matrix[i][j] = relation["strength"]
                adj_matrix[j][i] = relation["strength"]  # 无向图
            graph_info["adjacency_matrix"] = adj_matrix.tolist()
        
        return graph_info
    
    def _update_dgl_graph(self, concepts: List[Dict[str, Any]], 
                         relations: List[Dict[str, Any]]):
        """更新DGL图"""
        if not DGL_AVAILABLE:
            return None
        
        # 构建边列表
        src_nodes = []
        dst_nodes = []
        edge_weights = []
        
        for relation in relations:
            src_nodes.append(relation["source_idx"])
            dst_nodes.append(relation["target_idx"])
            edge_weights.append(relation["strength"])
        
        if not src_nodes:  # 没有边的情况
            return dgl.graph(([], []))
        
        # 创建图
        graph = dgl.graph((src_nodes, dst_nodes))
        
        # 添加节点特征
        node_features = []
        for concept in concepts:
            node_features.append(concept["embedding"])
        
        if node_features:
            graph.ndata['feat'] = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # 添加边权重
        if edge_weights:
            graph.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
        
        return graph
    
    def enhance_implicit_relations(self, problem_text: str, 
                                 existing_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        增强隐式关系发现
        
        Args:
            problem_text: 问题文本
            existing_relations: 现有关系列表
            
        Returns:
            增强后的关系列表
        """
        try:
            # 1. 构建概念图
            entities = self._extract_entities_from_text(problem_text)
            concept_graph = self.build_concept_graph(problem_text, entities)
            
            # 2. 基于GNN推理隐式关系
            implicit_relations = self._infer_implicit_relations(concept_graph)
            
            # 3. 合并现有关系和隐式关系
            enhanced_relations = existing_relations.copy()
            for relation in implicit_relations:
                if not self._relation_exists(relation, enhanced_relations):
                    enhanced_relations.append(relation)
            
            # 4. 关系验证和评分
            validated_relations = self._validate_relations(enhanced_relations, problem_text)
            
            return validated_relations
            
        except Exception as e:
            self.logger.error(f"Failed to enhance implicit relations: {e}")
            return existing_relations
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """从文本中提取实体"""
        entities = []
        
        # 数字实体
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        entities.extend(numbers)
        
        # 单位实体
        units = re.findall(r'(cm|m|km|mm|l|ml|L|kg|g|mg|s|min|h)', text)
        entities.extend(units)
        
        # 关键词实体
        for concept_type, keywords in self.math_concepts.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    entities.append(keyword)
        
        return list(set(entities))
    
    def _infer_implicit_relations(self, concept_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于GNN推理隐式关系"""
        implicit_relations = []
        
        if not concept_graph.get("concepts"):
            return implicit_relations
        
        concepts = concept_graph["concepts"]
        
        # 基于概念类型的隐式关系推理
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:
                    continue
                
                # 推理规则
                implicit_type, confidence = self._apply_inference_rules(concept1, concept2)
                
                if confidence > 0.6:
                    implicit_relations.append({
                        "source": concept1["text"],
                        "target": concept2["text"],
                        "type": implicit_type,
                        "confidence": confidence,
                        "source_type": "gnn_inference"
                    })
        
        return implicit_relations
    
    def _apply_inference_rules(self, concept1: Dict[str, Any], 
                              concept2: Dict[str, Any]) -> Tuple[str, float]:
        """应用推理规则"""
        type1, type2 = concept1["type"], concept2["type"]
        
        # 推理规则库
        rules = {
            ("number", "unit"): ("unit_relation", 0.9),
            ("area", "length"): ("geometric_relation", 0.8),
            ("volume", "liter"): ("unit_relation", 0.9),
            ("speed", "time"): ("temporal_relation", 0.8),
            ("speed", "length"): ("physical_relation", 0.7),
            ("time", "operation"): ("temporal_relation", 0.6),
            ("number", "operation"): ("arithmetic_relation", 0.7)
        }
        
        # 检查正向和反向规则
        if (type1, type2) in rules:
            return rules[(type1, type2)]
        elif (type2, type1) in rules:
            return rules[(type2, type1)]
        
        return ("unknown_relation", 0.0)
    
    def _relation_exists(self, relation: Dict[str, Any], 
                        relation_list: List[Dict[str, Any]]) -> bool:
        """检查关系是否已存在"""
        for existing in relation_list:
            if (existing.get("source") == relation["source"] and 
                existing.get("target") == relation["target"]) or \
               (existing.get("source") == relation["target"] and 
                existing.get("target") == relation["source"]):
                return True
        return False
    
    def _validate_relations(self, relations: List[Dict[str, Any]], 
                           problem_text: str) -> List[Dict[str, Any]]:
        """验证和评分关系"""
        validated = []
        
        for relation in relations:
            # 计算关系置信度
            confidence = self._calculate_relation_confidence(relation, problem_text)
            
            if confidence > 0.3:  # 置信度阈值
                relation["confidence"] = confidence
                validated.append(relation)
        
        # 按置信度排序
        validated.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return validated
    
    def _calculate_relation_confidence(self, relation: Dict[str, Any], 
                                     problem_text: str) -> float:
        """计算关系置信度"""
        base_confidence = relation.get("confidence", 0.5)
        
        # 基于文本共现的置信度调整
        source = relation["source"]
        target = relation["target"]
        
        if source.lower() in problem_text.lower() and target.lower() in problem_text.lower():
            base_confidence += 0.2
        
        # 基于关系类型的置信度调整
        relation_type = relation.get("type", "unknown_relation")
        type_weights = {
            "unit_relation": 1.0,
            "geometric_relation": 0.9,
            "temporal_relation": 0.8,
            "arithmetic_relation": 0.8,
            "physical_relation": 0.7,
            "logical_relation": 0.6,
            "unknown_relation": 0.3
        }
        
        type_weight = type_weights.get(relation_type, 0.5)
        base_confidence *= type_weight
        
        return min(1.0, base_confidence)
    
    def get_concept_similarity(self, concept1: str, concept2: str) -> float:
        """计算概念相似度"""
        try:
            emb1 = self._get_concept_embedding(concept1.lower())
            emb2 = self._get_concept_embedding(concept2.lower())
            
            # 余弦相似度
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate concept similarity: {e}")
            return 0.0
    
    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            "name": "MathConceptGNN",
            "version": "1.0.0",
            "concept_dim": self.concept_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_concepts": len(self.concept_embeddings),
            "relation_types": list(self.relation_types.keys()),
            "torch_available": TORCH_AVAILABLE,
            "dgl_available": DGL_AVAILABLE,
            "model_loaded": self.model is not None
        } 