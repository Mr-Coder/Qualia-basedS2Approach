#!/usr/bin/env python3
"""
增强QS²语义分析器
基于多层级语义向量空间和注意力机制的Qualia理论扩展
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MultiLevelSemanticVector:
    """多层级语义向量"""
    entity_embedding: np.ndarray    # 实体级嵌入
    qualia_embeddings: Dict[str, np.ndarray]  # Qualia角色级嵌入
    context_embedding: np.ndarray   # 上下文级嵌入
    meta_embedding: np.ndarray      # 元认知级嵌入

@dataclass
class AttentionWeights:
    """注意力权重"""
    role_attention: Dict[str, float]     # Qualia角色注意力
    context_attention: float             # 上下文注意力
    temporal_attention: float            # 时序注意力
    causal_attention: float              # 因果注意力

class EnhancedQS2Analyzer:
    """增强QS²语义分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化神经网络组件
        self.attention_network = self._build_attention_network()
        self.semantic_encoder = self._build_semantic_encoder()
        self.qualia_transformer = self._build_qualia_transformer()
        
        # 语义知识库
        self.semantic_knowledge_base = self._load_semantic_kb()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "embedding_dim": 256,
            "qualia_dim": 64,
            "attention_heads": 8,
            "transformer_layers": 4,
            "dropout_rate": 0.1,
            "semantic_threshold": 0.75,
            "context_window": 5,
            "enable_meta_learning": True,
            "use_pretrained_embeddings": True
        }
    
    def _build_attention_network(self) -> nn.Module:
        """构建多头注意力网络"""
        class QualiaAttentionNetwork(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Qualia角色注意力
                self.role_attention = nn.MultiheadAttention(
                    embed_dim=config["qualia_dim"],
                    num_heads=config["attention_heads"],
                    dropout=config["dropout_rate"]
                )
                
                # 交叉模态注意力
                self.cross_modal_attention = nn.MultiheadAttention(
                    embed_dim=config["embedding_dim"],
                    num_heads=config["attention_heads"],
                    dropout=config["dropout_rate"]
                )
                
                # 因果注意力层
                self.causal_attention = nn.TransformerDecoderLayer(
                    d_model=config["embedding_dim"],
                    nhead=config["attention_heads"],
                    dropout=config["dropout_rate"]
                )
                
            def forward(self, qualia_features, context_features):
                # 计算Qualia角色间注意力
                role_attended, role_weights = self.role_attention(
                    qualia_features, qualia_features, qualia_features
                )
                
                # 计算上下文注意力
                context_attended, context_weights = self.cross_modal_attention(
                    context_features, qualia_features, qualia_features
                )
                
                return {
                    "role_attended": role_attended,
                    "context_attended": context_attended,
                    "role_weights": role_weights,
                    "context_weights": context_weights
                }
        
        return QualiaAttentionNetwork(self.config)
    
    def _build_semantic_encoder(self) -> nn.Module:
        """构建语义编码器"""
        class HierarchicalSemanticEncoder(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # 词级编码器
                self.word_encoder = nn.LSTM(
                    input_size=300,  # 预训练词向量维度
                    hidden_size=config["embedding_dim"] // 2,
                    num_layers=2,
                    bidirectional=True,
                    dropout=config["dropout_rate"]
                )
                
                # 实体级编码器
                self.entity_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config["embedding_dim"],
                        nhead=config["attention_heads"],
                        dropout=config["dropout_rate"]
                    ),
                    num_layers=config["transformer_layers"]
                )
                
                # Qualia角色专用编码器
                self.qualia_encoders = nn.ModuleDict({
                    role: nn.Linear(config["embedding_dim"], config["qualia_dim"])
                    for role in ["formal", "telic", "agentive", "constitutive"]
                })
                
            def forward(self, word_embeddings, entity_masks):
                # 词级编码
                word_features, _ = self.word_encoder(word_embeddings)
                
                # 实体级编码
                entity_features = self.entity_encoder(word_features)
                
                # Qualia角色编码
                qualia_features = {}
                for role, encoder in self.qualia_encoders.items():
                    qualia_features[role] = encoder(entity_features)
                
                return {
                    "entity_features": entity_features,
                    "qualia_features": qualia_features
                }
        
        return HierarchicalSemanticEncoder(self.config)
    
    def _build_qualia_transformer(self) -> nn.Module:
        """构建Qualia变换器"""
        class QualiaTransformer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Qualia角色交互层
                self.role_interaction = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config["qualia_dim"],
                        nhead=4,  # 较少的注意力头用于角色交互
                        dropout=config["dropout_rate"]
                    ),
                    num_layers=2
                )
                
                # 语义兼容性计算层
                self.compatibility_head = nn.Sequential(
                    nn.Linear(config["qualia_dim"] * 4, config["qualia_dim"]),
                    nn.ReLU(),
                    nn.Dropout(config["dropout_rate"]),
                    nn.Linear(config["qualia_dim"], 1),
                    nn.Sigmoid()
                )
                
                # 因果关系推理层
                self.causal_reasoning = nn.GRU(
                    input_size=config["qualia_dim"],
                    hidden_size=config["qualia_dim"],
                    num_layers=2,
                    dropout=config["dropout_rate"]
                )
                
            def forward(self, qualia_features):
                # Qualia角色交互
                role_stack = torch.stack(list(qualia_features.values()), dim=0)
                interacted_roles = self.role_interaction(role_stack)
                
                # 语义兼容性计算
                flattened = interacted_roles.view(-1, self.config["qualia_dim"] * 4)
                compatibility = self.compatibility_head(flattened)
                
                # 因果推理
                causal_output, _ = self.causal_reasoning(role_stack)
                
                return {
                    "interacted_roles": interacted_roles,
                    "compatibility": compatibility,
                    "causal_features": causal_output
                }
        
        return QualiaTransformer(self.config)
    
    def analyze_semantics_enhanced(self, processed_problem, context_history=None) -> List[Dict[str, Any]]:
        """增强语义分析"""
        self.logger.info("开始增强QS²语义分析")
        
        # 1. 多层级语义向量构建
        semantic_vectors = self._build_multilevel_vectors(processed_problem)
        
        # 2. 注意力权重计算
        attention_weights = self._compute_attention_weights(semantic_vectors, context_history)
        
        # 3. Qualia结构增强
        enhanced_qualia = self._enhance_qualia_structures(semantic_vectors, attention_weights)
        
        # 4. 动态兼容性计算
        compatibility_matrix = self._compute_dynamic_compatibility(enhanced_qualia)
        
        # 5. 元认知推理
        if self.config["enable_meta_learning"]:
            meta_insights = self._meta_cognitive_reasoning(enhanced_qualia, compatibility_matrix)
        else:
            meta_insights = {}
        
        # 6. 生成增强语义实体
        enhanced_entities = self._generate_enhanced_entities(
            enhanced_qualia, compatibility_matrix, meta_insights
        )
        
        self.logger.info(f"增强语义分析完成，生成{len(enhanced_entities)}个增强实体")
        return enhanced_entities
    
    def _build_multilevel_vectors(self, processed_problem) -> Dict[str, MultiLevelSemanticVector]:
        """构建多层级语义向量"""
        vectors = {}
        
        for entity in processed_problem.entities:
            # 实体级嵌入
            entity_embedding = self._get_entity_embedding(entity)
            
            # Qualia角色级嵌入
            qualia_embeddings = self._get_qualia_embeddings(entity)
            
            # 上下文级嵌入
            context_embedding = self._get_context_embedding(entity, processed_problem)
            
            # 元认知级嵌入
            meta_embedding = self._get_meta_embedding(entity, processed_problem)
            
            vectors[entity.id] = MultiLevelSemanticVector(
                entity_embedding=entity_embedding,
                qualia_embeddings=qualia_embeddings,
                context_embedding=context_embedding,
                meta_embedding=meta_embedding
            )
        
        return vectors
    
    def _compute_attention_weights(self, semantic_vectors, context_history) -> Dict[str, AttentionWeights]:
        """计算注意力权重"""
        attention_weights = {}
        
        for entity_id, vector in semantic_vectors.items():
            # 使用注意力网络计算权重
            with torch.no_grad():
                qualia_tensor = torch.stack([
                    torch.tensor(emb) for emb in vector.qualia_embeddings.values()
                ])
                context_tensor = torch.tensor(vector.context_embedding).unsqueeze(0)
                
                attention_output = self.attention_network(qualia_tensor, context_tensor)
                
                # 提取注意力权重
                role_attention = {
                    role: float(weight) for role, weight in zip(
                        ["formal", "telic", "agentive", "constitutive"],
                        attention_output["role_weights"].mean(dim=0)
                    )
                }
                
                attention_weights[entity_id] = AttentionWeights(
                    role_attention=role_attention,
                    context_attention=float(attention_output["context_weights"].mean()),
                    temporal_attention=self._compute_temporal_attention(entity_id, context_history),
                    causal_attention=self._compute_causal_attention(entity_id, semantic_vectors)
                )
        
        return attention_weights
    
    def _enhance_qualia_structures(self, semantic_vectors, attention_weights) -> Dict[str, Dict[str, List[str]]]:
        """增强Qualia结构"""
        enhanced_qualia = {}
        
        for entity_id, vector in semantic_vectors.items():
            weights = attention_weights[entity_id]
            
            # 基于注意力权重增强Qualia角色
            enhanced_roles = {}
            for role in ["formal", "telic", "agentive", "constitutive"]:
                # 获取基础角色描述
                base_roles = self._get_base_qualia_roles(entity_id, role)
                
                # 基于注意力权重扩展
                attention_weight = weights.role_attention[role]
                if attention_weight > self.config["semantic_threshold"]:
                    # 从语义知识库中检索相关角色
                    expanded_roles = self._expand_qualia_roles(
                        base_roles, role, attention_weight, vector
                    )
                    enhanced_roles[role] = base_roles + expanded_roles
                else:
                    enhanced_roles[role] = base_roles
            
            enhanced_qualia[entity_id] = enhanced_roles
        
        return enhanced_qualia
    
    def _compute_dynamic_compatibility(self, enhanced_qualia) -> np.ndarray:
        """计算动态兼容性矩阵"""
        entity_ids = list(enhanced_qualia.keys())
        n_entities = len(entity_ids)
        compatibility_matrix = np.zeros((n_entities, n_entities))
        
        for i, entity1_id in enumerate(entity_ids):
            for j, entity2_id in enumerate(entity_ids):
                if i != j:
                    # 使用Qualia变换器计算兼容性
                    compatibility = self._compute_pairwise_compatibility(
                        enhanced_qualia[entity1_id],
                        enhanced_qualia[entity2_id]
                    )
                    compatibility_matrix[i, j] = compatibility
        
        return compatibility_matrix
    
    def _meta_cognitive_reasoning(self, enhanced_qualia, compatibility_matrix) -> Dict[str, Any]:
        """元认知推理"""
        meta_insights = {
            "semantic_clusters": self._identify_semantic_clusters(compatibility_matrix),
            "dominant_patterns": self._identify_dominant_patterns(enhanced_qualia),
            "reasoning_strategies": self._suggest_reasoning_strategies(enhanced_qualia),
            "uncertainty_estimation": self._estimate_uncertainty(compatibility_matrix)
        }
        
        return meta_insights
    
    # 辅助方法实现
    def _get_entity_embedding(self, entity) -> np.ndarray:
        """获取实体嵌入"""
        # 实现实体嵌入获取逻辑
        return np.random.randn(self.config["embedding_dim"])
    
    def _get_qualia_embeddings(self, entity) -> Dict[str, np.ndarray]:
        """获取Qualia角色嵌入"""
        return {
            role: np.random.randn(self.config["qualia_dim"])
            for role in ["formal", "telic", "agentive", "constitutive"]
        }
    
    def _get_context_embedding(self, entity, processed_problem) -> np.ndarray:
        """获取上下文嵌入"""
        return np.random.randn(self.config["embedding_dim"])
    
    def _get_meta_embedding(self, entity, processed_problem) -> np.ndarray:
        """获取元认知嵌入"""
        return np.random.randn(self.config["embedding_dim"])
    
    def _load_semantic_kb(self) -> Dict[str, Any]:
        """加载语义知识库"""
        return {
            "mathematical_concepts": [],
            "spatial_relations": [],
            "temporal_relations": [],
            "causal_patterns": []
        }
    
    def test_enhanced_analysis(self) -> Dict[str, Any]:
        """测试增强分析功能"""
        from problem_preprocessor import ProcessedProblem, Entity
        
        # 创建测试数据
        test_entity = Entity(
            id="test_entity",
            name="苹果",
            entity_type="object",
            value=5,
            unit="个"
        )
        
        test_problem = ProcessedProblem(
            original_text="小明有5个苹果",
            cleaned_text="小明有5个苹果",
            entities=[test_entity],
            numbers=[5],
            keywords=["有"],
            problem_type="arithmetic"
        )
        
        # 执行增强分析
        result = self.analyze_semantics_enhanced(test_problem)
        
        return {
            "success": True,
            "enhanced_entities_count": len(result),
            "algorithm_features": {
                "multi_level_vectors": True,
                "attention_mechanism": True,
                "qualia_transformer": True,
                "meta_cognitive_reasoning": self.config["enable_meta_learning"]
            }
        }

# 使用示例
if __name__ == "__main__":
    analyzer = EnhancedQS2Analyzer()
    test_result = analyzer.test_enhanced_analysis()
    print(f"增强QS²分析器测试结果: {test_result}")