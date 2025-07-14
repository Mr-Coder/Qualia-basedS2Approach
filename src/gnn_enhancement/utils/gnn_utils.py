"""
GNN Utils
=========

GNN工具类，提供通用的工具函数和辅助方法
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GNNUtils:
    """GNN工具类"""
    
    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """归一化嵌入向量"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        return embeddings / norms
    
    @staticmethod
    def calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def build_adjacency_matrix(nodes: List[Dict[str, Any]], 
                              edges: List[Dict[str, Any]]) -> np.ndarray:
        """构建邻接矩阵"""
        n_nodes = len(nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # 创建节点ID到索引的映射
        id_to_idx = {node.get("id", i): i for i, node in enumerate(nodes)}
        
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            weight = edge.get("weight", 1.0)
            
            if source in id_to_idx and target in id_to_idx:
                i, j = id_to_idx[source], id_to_idx[target]
                adj_matrix[i][j] = weight
                adj_matrix[j][i] = weight  # 无向图
        
        return adj_matrix
    
    @staticmethod
    def extract_graph_features(graph_info: Dict[str, Any]) -> Dict[str, Any]:
        """提取图特征"""
        features = {}
        
        nodes = graph_info.get("nodes", [])
        edges = graph_info.get("edges", [])
        
        # 基本统计
        features["num_nodes"] = len(nodes)
        features["num_edges"] = len(edges)
        features["density"] = len(edges) / max(len(nodes) * (len(nodes) - 1) / 2, 1)
        
        # 度分布
        if nodes and edges:
            degree_counts = {i: 0 for i in range(len(nodes))}
            id_to_idx = {node.get("id", i): i for i, node in enumerate(nodes)}
            
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                if source in id_to_idx:
                    degree_counts[id_to_idx[source]] += 1
                if target in id_to_idx:
                    degree_counts[id_to_idx[target]] += 1
            
            degrees = list(degree_counts.values())
            features["avg_degree"] = np.mean(degrees) if degrees else 0
            features["max_degree"] = max(degrees) if degrees else 0
            features["min_degree"] = min(degrees) if degrees else 0
        
        return features
    
    @staticmethod
    def validate_graph_structure(graph_info: Dict[str, Any]) -> Dict[str, Any]:
        """验证图结构"""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 检查必要字段
        if "nodes" not in graph_info:
            validation["valid"] = False
            validation["errors"].append("Missing 'nodes' field")
        
        if "edges" not in graph_info:
            validation["warnings"].append("Missing 'edges' field")
        
        nodes = graph_info.get("nodes", [])
        edges = graph_info.get("edges", [])
        
        # 验证节点
        node_ids = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                validation["errors"].append(f"Node {i} is not a dictionary")
                validation["valid"] = False
                continue
            
            node_id = node.get("id", i)
            if node_id in node_ids:
                validation["errors"].append(f"Duplicate node ID: {node_id}")
                validation["valid"] = False
            node_ids.add(node_id)
        
        # 验证边
        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                validation["errors"].append(f"Edge {i} is not a dictionary")
                validation["valid"] = False
                continue
            
            source = edge.get("source")
            target = edge.get("target")
            
            if source not in node_ids:
                validation["errors"].append(f"Edge {i} source {source} not found in nodes")
                validation["valid"] = False
            
            if target not in node_ids:
                validation["errors"].append(f"Edge {i} target {target} not found in nodes")
                validation["valid"] = False
        
        return validation
    
    @staticmethod
    def merge_graphs(graph1: Dict[str, Any], graph2: Dict[str, Any]) -> Dict[str, Any]:
        """合并两个图"""
        merged = {
            "nodes": [],
            "edges": [],
            "metadata": {}
        }
        
        # 合并节点
        nodes1 = graph1.get("nodes", [])
        nodes2 = graph2.get("nodes", [])
        
        # 重新分配节点ID以避免冲突
        id_mapping = {}
        next_id = 0
        
        for node in nodes1:
            old_id = node.get("id", len(merged["nodes"]))
            new_node = node.copy()
            new_node["id"] = next_id
            id_mapping[old_id] = next_id
            merged["nodes"].append(new_node)
            next_id += 1
        
        for node in nodes2:
            old_id = node.get("id", len(merged["nodes"]))
            new_node = node.copy()
            new_node["id"] = next_id
            id_mapping[old_id] = next_id
            merged["nodes"].append(new_node)
            next_id += 1
        
        # 合并边
        edges1 = graph1.get("edges", [])
        edges2 = graph2.get("edges", [])
        
        for edge in edges1:
            new_edge = edge.copy()
            if edge.get("source") in id_mapping:
                new_edge["source"] = id_mapping[edge["source"]]
            if edge.get("target") in id_mapping:
                new_edge["target"] = id_mapping[edge["target"]]
            merged["edges"].append(new_edge)
        
        for edge in edges2:
            new_edge = edge.copy()
            if edge.get("source") in id_mapping:
                new_edge["source"] = id_mapping[edge["source"]]
            if edge.get("target") in id_mapping:
                new_edge["target"] = id_mapping[edge["target"]]
            merged["edges"].append(new_edge)
        
        # 合并元数据
        merged["metadata"] = {
            "source_graphs": [graph1.get("metadata", {}), graph2.get("metadata", {})],
            "merge_timestamp": "2024-07-13",
            "num_nodes": len(merged["nodes"]),
            "num_edges": len(merged["edges"])
        }
        
        return merged
    
    @staticmethod
    def calculate_graph_metrics(graph_info: Dict[str, Any]) -> Dict[str, Any]:
        """计算图度量指标"""
        metrics = {}
        
        nodes = graph_info.get("nodes", [])
        edges = graph_info.get("edges", [])
        
        if not nodes:
            return metrics
        
        # 基本度量
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        metrics["num_nodes"] = n_nodes
        metrics["num_edges"] = n_edges
        metrics["density"] = n_edges / max(n_nodes * (n_nodes - 1) / 2, 1)
        
        # 度分布
        degree_counts = {i: 0 for i in range(n_nodes)}
        id_to_idx = {node.get("id", i): i for i, node in enumerate(nodes)}
        
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source in id_to_idx:
                degree_counts[id_to_idx[source]] += 1
            if target in id_to_idx:
                degree_counts[id_to_idx[target]] += 1
        
        degrees = list(degree_counts.values())
        if degrees:
            metrics["avg_degree"] = np.mean(degrees)
            metrics["max_degree"] = max(degrees)
            metrics["min_degree"] = min(degrees)
            metrics["degree_std"] = np.std(degrees)
        
        # 连通性
        if n_edges > 0:
            adj_matrix = GNNUtils.build_adjacency_matrix(nodes, edges)
            metrics["is_connected"] = GNNUtils._is_connected(adj_matrix)
        else:
            metrics["is_connected"] = n_nodes <= 1
        
        return metrics
    
    @staticmethod
    def _is_connected(adj_matrix: np.ndarray) -> bool:
        """检查图是否连通"""
        n = adj_matrix.shape[0]
        if n <= 1:
            return True
        
        # 使用DFS检查连通性
        visited = [False] * n
        
        def dfs(node):
            visited[node] = True
            for neighbor in range(n):
                if adj_matrix[node][neighbor] > 0 and not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(0)
        return all(visited)
    
    @staticmethod
    def format_graph_for_visualization(graph_info: Dict[str, Any]) -> Dict[str, Any]:
        """格式化图数据用于可视化"""
        viz_data = {
            "nodes": [],
            "links": []
        }
        
        nodes = graph_info.get("nodes", [])
        edges = graph_info.get("edges", [])
        
        # 格式化节点
        for node in nodes:
            viz_node = {
                "id": node.get("id", "unknown"),
                "label": node.get("text", node.get("description", "Node")),
                "type": node.get("type", "default"),
                "size": 10,
                "color": GNNUtils._get_node_color(node.get("type", "default"))
            }
            viz_data["nodes"].append(viz_node)
        
        # 格式化边
        for edge in edges:
            viz_edge = {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "weight": edge.get("weight", edge.get("strength", 1.0)),
                "type": edge.get("type", "default"),
                "color": GNNUtils._get_edge_color(edge.get("type", "default"))
            }
            viz_data["links"].append(viz_edge)
        
        return viz_data
    
    @staticmethod
    def _get_node_color(node_type: str) -> str:
        """获取节点颜色"""
        colors = {
            "concept": "#FF6B6B",
            "reasoning_step": "#4ECDC4", 
            "verification_step": "#45B7D1",
            "number": "#96CEB4",
            "operation": "#FFEAA7",
            "default": "#DDA0DD"
        }
        return colors.get(node_type, colors["default"])
    
    @staticmethod
    def _get_edge_color(edge_type: str) -> str:
        """获取边颜色"""
        colors = {
            "data_dependency": "#FF6B6B",
            "type_dependency": "#4ECDC4",
            "semantic_dependency": "#45B7D1",
            "unit_relation": "#96CEB4",
            "logical_relation": "#FFEAA7",
            "default": "#999999"
        }
        return colors.get(edge_type, colors["default"])
    
    @staticmethod
    def get_utils_info() -> Dict[str, Any]:
        """获取工具信息"""
        return {
            "name": "GNNUtils",
            "version": "1.0.0",
            "functions": [
                "normalize_embeddings",
                "calculate_cosine_similarity",
                "build_adjacency_matrix",
                "extract_graph_features",
                "validate_graph_structure",
                "merge_graphs",
                "calculate_graph_metrics",
                "format_graph_for_visualization"
            ]
        } 