#!/usr/bin/env python3
"""
强化学习增强的COT-DIR推理链
基于深度强化学习的动态推理策略优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from collections import deque, namedtuple
import logging
import random

logger = logging.getLogger(__name__)

# 强化学习状态和动作定义
State = namedtuple('State', ['entities', 'relations', 'step_history', 'context'])
Action = namedtuple('Action', ['step_type', 'target_entities', 'reasoning_method'])
Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'done'])

@dataclass
class ReasoningStrategy:
    """推理策略"""
    strategy_id: str
    step_sequence: List[str]
    priority_weights: Dict[str, float]
    success_rate: float
    avg_confidence: float

@dataclass
class ReasoningState:
    """推理状态"""
    current_step: int
    available_entities: List[str]
    discovered_relations: List[str]
    step_confidence: List[float]
    global_context: Dict[str, Any]

class RLEnhancedCOTDIR:
    """强化学习增强的COT-DIR推理链"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 强化学习组件
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        self.experience_buffer = deque(maxlen=self.config["buffer_size"])
        
        # 推理策略库
        self.strategy_library = self._initialize_strategy_library()
        
        # 奖励函数
        self.reward_calculator = self._build_reward_calculator()
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), 
            lr=self.config["learning_rate"]
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), 
            lr=self.config["learning_rate"]
        )
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "state_dim": 256,
            "action_dim": 64,
            "hidden_dim": 512,
            "num_layers": 3,
            "learning_rate": 1e-4,
            "buffer_size": 10000,
            "batch_size": 64,
            "gamma": 0.99,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "target_update_freq": 100,
            "max_episode_length": 20,
            "reward_shaping": True,
            "curiosity_driven": True,
            "meta_learning": True
        }
    
    def _build_policy_network(self) -> nn.Module:
        """构建策略网络"""
        class PolicyNetwork(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # 状态编码器
                self.state_encoder = nn.Sequential(
                    nn.Linear(config["state_dim"], config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                    nn.ReLU()
                )
                
                # 注意力机制用于动态关注不同状态组件
                self.attention = nn.MultiheadAttention(
                    embed_dim=config["hidden_dim"],
                    num_heads=8,
                    dropout=0.1
                )
                
                # 动作预测头
                self.action_heads = nn.ModuleDict({
                    "step_type": nn.Sequential(
                        nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                        nn.ReLU(),
                        nn.Linear(config["hidden_dim"] // 2, 6)  # 6种推理步骤类型
                    ),
                    "reasoning_method": nn.Sequential(
                        nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                        nn.ReLU(),
                        nn.Linear(config["hidden_dim"] // 2, 10)  # 10种推理方法
                    ),
                    "priority": nn.Sequential(
                        nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                        nn.ReLU(),
                        nn.Linear(config["hidden_dim"] // 2, 1),
                        nn.Sigmoid()
                    )
                })
                
            def forward(self, state):
                # 编码状态
                encoded_state = self.state_encoder(state)
                
                # 注意力机制
                attended_state, attention_weights = self.attention(
                    encoded_state.unsqueeze(0),
                    encoded_state.unsqueeze(0),
                    encoded_state.unsqueeze(0)
                )
                attended_state = attended_state.squeeze(0)
                
                # 预测各种动作组件
                actions = {}
                for head_name, head in self.action_heads.items():
                    if head_name in ["step_type", "reasoning_method"]:
                        actions[head_name] = F.softmax(head(attended_state), dim=-1)
                    else:
                        actions[head_name] = head(attended_state)
                
                return actions, attention_weights
        
        return PolicyNetwork(self.config)
    
    def _build_value_network(self) -> nn.Module:
        """构建价值网络"""
        class ValueNetwork(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                self.value_net = nn.Sequential(
                    nn.Linear(config["state_dim"], config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                    nn.ReLU(),
                    nn.Linear(config["hidden_dim"] // 2, 1)
                )
                
            def forward(self, state):
                return self.value_net(state)
        
        return ValueNetwork(self.config)
    
    def _build_reward_calculator(self) -> nn.Module:
        """构建奖励计算器"""
        class RewardCalculator(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # 多维度奖励计算
                self.reward_components = nn.ModuleDict({
                    "accuracy": nn.Linear(1, 1),
                    "efficiency": nn.Linear(1, 1),
                    "novelty": nn.Linear(1, 1),
                    "consistency": nn.Linear(1, 1)
                })
                
                # 奖励融合网络
                self.reward_fusion = nn.Sequential(
                    nn.Linear(4, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Tanh()
                )
                
            def calculate_reward(self, step_result, context):
                """计算多维度奖励"""
                # 准确性奖励
                accuracy_reward = self._calculate_accuracy_reward(step_result)
                
                # 效率奖励
                efficiency_reward = self._calculate_efficiency_reward(step_result, context)
                
                # 新颖性奖励（好奇心驱动）
                novelty_reward = self._calculate_novelty_reward(step_result, context)
                
                # 一致性奖励
                consistency_reward = self._calculate_consistency_reward(step_result, context)
                
                # 融合奖励
                reward_vector = torch.tensor([
                    accuracy_reward, efficiency_reward, 
                    novelty_reward, consistency_reward
                ], dtype=torch.float)
                
                total_reward = self.reward_fusion(reward_vector)
                
                return {
                    "total_reward": float(total_reward),
                    "accuracy": accuracy_reward,
                    "efficiency": efficiency_reward,
                    "novelty": novelty_reward,
                    "consistency": consistency_reward
                }
            
            def _calculate_accuracy_reward(self, step_result):
                """计算准确性奖励"""
                confidence = step_result.get("confidence", 0.0)
                return confidence * 2 - 1  # 映射到[-1, 1]
            
            def _calculate_efficiency_reward(self, step_result, context):
                """计算效率奖励"""
                execution_time = step_result.get("execution_time", 1.0)
                baseline_time = context.get("baseline_time", 1.0)
                efficiency = baseline_time / max(execution_time, 0.001)
                return min(efficiency, 2.0) - 1  # 映射到[-1, 1]
            
            def _calculate_novelty_reward(self, step_result, context):
                """计算新颖性奖励（好奇心）"""
                # 基于发现的新关系数量
                new_relations = step_result.get("new_relations_count", 0)
                return min(new_relations / 5.0, 1.0)  # 映射到[0, 1]
            
            def _calculate_consistency_reward(self, step_result, context):
                """计算一致性奖励"""
                logical_consistency = step_result.get("logical_consistency", 1.0)
                return logical_consistency * 2 - 1  # 映射到[-1, 1]
        
        return RewardCalculator(self.config)
    
    def _initialize_strategy_library(self) -> Dict[str, ReasoningStrategy]:
        """初始化策略库"""
        strategies = {
            "analytical": ReasoningStrategy(
                strategy_id="analytical",
                step_sequence=["entity_extraction", "semantic_analysis", "relation_discovery", 
                             "mathematical_computation", "logic_verification", "result_synthesis"],
                priority_weights={"accuracy": 0.8, "efficiency": 0.2},
                success_rate=0.85,
                avg_confidence=0.82
            ),
            "intuitive": ReasoningStrategy(
                strategy_id="intuitive",
                step_sequence=["entity_extraction", "relation_discovery", "mathematical_computation", 
                             "result_synthesis"],
                priority_weights={"efficiency": 0.7, "accuracy": 0.3},
                success_rate=0.75,
                avg_confidence=0.70
            ),
            "exploratory": ReasoningStrategy(
                strategy_id="exploratory",
                step_sequence=["entity_extraction", "semantic_analysis", "relation_discovery", 
                             "alternative_reasoning", "verification", "result_synthesis"],
                priority_weights={"novelty": 0.6, "accuracy": 0.4},
                success_rate=0.70,
                avg_confidence=0.65
            )
        }
        return strategies
    
    def build_reasoning_chain_rl(self, processed_problem, semantic_entities, 
                                relation_network, learning_mode=True) -> Dict[str, Any]:
        """使用强化学习构建推理链"""
        self.logger.info("开始RL增强推理链构建")
        
        # 初始化环境状态
        initial_state = self._initialize_state(processed_problem, semantic_entities, relation_network)
        
        # 选择初始策略
        selected_strategy = self._select_strategy(initial_state)
        
        # 执行推理链构建
        if learning_mode:
            chain_result = self._build_chain_with_learning(
                initial_state, selected_strategy, processed_problem
            )
        else:
            chain_result = self._build_chain_inference_only(
                initial_state, selected_strategy, processed_problem
            )
        
        # 更新策略库
        self._update_strategy_library(selected_strategy, chain_result)
        
        return chain_result
    
    def _initialize_state(self, processed_problem, semantic_entities, relation_network) -> torch.Tensor:
        """初始化推理状态"""
        # 构建状态向量
        state_components = []
        
        # 问题特征
        problem_features = self._encode_problem_features(processed_problem)
        state_components.extend(problem_features)
        
        # 实体特征
        entity_features = self._encode_entity_features(semantic_entities)
        state_components.extend(entity_features)
        
        # 关系特征
        relation_features = self._encode_relation_features(relation_network)
        state_components.extend(relation_features)
        
        # 填充到固定维度
        while len(state_components) < self.config["state_dim"]:
            state_components.append(0.0)
        
        return torch.tensor(state_components[:self.config["state_dim"]], dtype=torch.float)
    
    def _select_strategy(self, state) -> ReasoningStrategy:
        """选择推理策略"""
        # 使用策略网络选择动作
        with torch.no_grad():
            actions, attention_weights = self.policy_network(state)
        
        # 基于注意力权重和历史成功率选择策略
        strategy_scores = {}
        for strategy_id, strategy in self.strategy_library.items():
            # 计算策略适配度
            score = (strategy.success_rate * 0.6 + 
                    strategy.avg_confidence * 0.4 + 
                    np.random.normal(0, 0.1))  # 添加探索噪声
            strategy_scores[strategy_id] = score
        
        # 选择最高分策略
        best_strategy_id = max(strategy_scores, key=strategy_scores.get)
        return self.strategy_library[best_strategy_id]
    
    def _build_chain_with_learning(self, initial_state, strategy, processed_problem) -> Dict[str, Any]:
        """在学习模式下构建推理链"""
        episode_transitions = []
        current_state = initial_state
        total_reward = 0.0
        step_results = []
        
        for step_idx, step_type in enumerate(strategy.step_sequence):
            # 策略网络选择具体动作
            actions, attention_weights = self.policy_network(current_state)
            
            # 执行推理步骤
            step_result = self._execute_reasoning_step(
                step_type, current_state, processed_problem, step_idx
            )
            step_results.append(step_result)
            
            # 计算奖励
            reward_info = self.reward_calculator.calculate_reward(step_result, {
                "step_index": step_idx,
                "strategy": strategy,
                "baseline_time": 0.1
            })
            
            # 更新状态
            next_state = self._update_state(current_state, step_result, actions)
            
            # 记录转换
            transition = Transition(
                state=current_state,
                action=actions,
                next_state=next_state,
                reward=reward_info["total_reward"],
                done=(step_idx == len(strategy.step_sequence) - 1)
            )
            episode_transitions.append(transition)
            
            current_state = next_state
            total_reward += reward_info["total_reward"]
        
        # 存储经验
        self.experience_buffer.extend(episode_transitions)
        
        # 训练网络
        if len(self.experience_buffer) >= self.config["batch_size"]:
            self._train_networks()
        
        return {
            "success": True,
            "strategy_used": strategy.strategy_id,
            "total_reward": total_reward,
            "step_results": step_results,
            "attention_weights": attention_weights.detach().numpy() if attention_weights is not None else None,
            "learning_stats": {
                "episode_length": len(step_results),
                "avg_step_reward": total_reward / len(step_results),
                "buffer_size": len(self.experience_buffer)
            }
        }
    
    def _build_chain_inference_only(self, initial_state, strategy, processed_problem) -> Dict[str, Any]:
        """仅推理模式构建推理链"""
        current_state = initial_state
        step_results = []
        
        with torch.no_grad():
            for step_idx, step_type in enumerate(strategy.step_sequence):
                # 执行推理步骤
                step_result = self._execute_reasoning_step(
                    step_type, current_state, processed_problem, step_idx
                )
                step_results.append(step_result)
                
                # 更新状态
                actions, _ = self.policy_network(current_state)
                current_state = self._update_state(current_state, step_result, actions)
        
        return {
            "success": True,
            "strategy_used": strategy.strategy_id,
            "step_results": step_results,
            "inference_mode": True
        }
    
    def _execute_reasoning_step(self, step_type, state, processed_problem, step_index) -> Dict[str, Any]:
        """执行具体的推理步骤"""
        # 这里应该调用相应的推理模块
        # 为了示例，我们模拟推理结果
        
        mock_results = {
            "entity_extraction": {
                "confidence": 0.9,
                "execution_time": 0.05,
                "new_relations_count": 0,
                "logical_consistency": 1.0,
                "entities_found": 4
            },
            "semantic_analysis": {
                "confidence": 0.85,
                "execution_time": 0.08,
                "new_relations_count": 2,
                "logical_consistency": 0.95,
                "semantic_patterns": 3
            },
            "relation_discovery": {
                "confidence": 0.80,
                "execution_time": 0.12,
                "new_relations_count": 6,
                "logical_consistency": 0.90,
                "relations_found": 6
            },
            "mathematical_computation": {
                "confidence": 0.95,
                "execution_time": 0.03,
                "new_relations_count": 0,
                "logical_consistency": 1.0,
                "computation_result": 8.0
            },
            "logic_verification": {
                "confidence": 0.88,
                "execution_time": 0.06,
                "new_relations_count": 1,
                "logical_consistency": 0.98,
                "verification_passed": True
            },
            "result_synthesis": {
                "confidence": 0.92,
                "execution_time": 0.04,
                "new_relations_count": 0,
                "logical_consistency": 0.96,
                "final_answer": "8个"
            }
        }
        
        return mock_results.get(step_type, {
            "confidence": 0.5,
            "execution_time": 0.1,
            "new_relations_count": 0,
            "logical_consistency": 0.5
        })
    
    def _train_networks(self):
        """训练策略和价值网络"""
        if len(self.experience_buffer) < self.config["batch_size"]:
            return
        
        # 采样批次
        batch = random.sample(self.experience_buffer, self.config["batch_size"])
        
        # 提取批次数据
        states = torch.stack([t.state for t in batch])
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float)
        next_states = torch.stack([t.next_state for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool)
        
        # 训练价值网络
        current_values = self.value_network(states).squeeze()
        next_values = self.value_network(next_states).squeeze()
        target_values = rewards + self.config["gamma"] * next_values * (~dones)
        
        value_loss = F.mse_loss(current_values, target_values.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 训练策略网络（简化的策略梯度）
        advantages = target_values - current_values
        
        policy_loss = 0
        for i, transition in enumerate(batch):
            actions, _ = self.policy_network(transition.state)
            # 简化的策略损失计算
            log_prob = torch.log(actions["priority"] + 1e-8)
            policy_loss += -log_prob * advantages[i].detach()
        
        policy_loss = policy_loss.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
    # 辅助方法
    def _encode_problem_features(self, processed_problem) -> List[float]:
        """编码问题特征"""
        return [
            len(processed_problem.entities),
            len(processed_problem.numbers),
            len(processed_problem.keywords),
            hash(processed_problem.problem_type) % 100 / 100.0
        ]
    
    def _encode_entity_features(self, semantic_entities) -> List[float]:
        """编码实体特征"""
        if not semantic_entities:
            return [0.0] * 20
        
        features = []
        for entity in semantic_entities[:5]:  # 最多编码5个实体
            features.extend([
                entity.confidence,
                hash(entity.entity_type) % 10 / 10.0,
                len(entity.name) / 10.0,
                1.0  # 存在标志
            ])
        
        # 填充到固定长度
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _encode_relation_features(self, relation_network) -> List[float]:
        """编码关系特征"""
        if not relation_network or not relation_network.relations:
            return [0.0] * 10
        
        features = [
            len(relation_network.relations),
            np.mean([r.confidence for r in relation_network.relations]),
            len(set(r.relation_type for r in relation_network.relations)),
        ]
        
        # 填充到固定长度
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]
    
    def _update_state(self, current_state, step_result, actions) -> torch.Tensor:
        """更新状态"""
        # 简化的状态更新：在当前状态基础上添加步骤结果信息
        state_update = torch.tensor([
            step_result.get("confidence", 0.0),
            step_result.get("new_relations_count", 0.0) / 10.0,
            step_result.get("logical_consistency", 0.0),
            float(actions["priority"]) if "priority" in actions else 0.0
        ], dtype=torch.float)
        
        # 将更新信息添加到状态向量的末尾，截断以保持固定维度
        updated_state = torch.cat([current_state[4:], state_update])
        return updated_state[:self.config["state_dim"]]
    
    def _update_strategy_library(self, strategy, chain_result):
        """更新策略库"""
        if chain_result["success"]:
            # 更新成功率（指数移动平均）
            alpha = 0.1
            old_success_rate = strategy.success_rate
            strategy.success_rate = old_success_rate * (1 - alpha) + 1.0 * alpha
            
            # 更新平均置信度
            if "step_results" in chain_result:
                avg_confidence = np.mean([
                    step.get("confidence", 0.0) for step in chain_result["step_results"]
                ])
                strategy.avg_confidence = strategy.avg_confidence * (1 - alpha) + avg_confidence * alpha
    
    def test_rl_cotdir(self) -> Dict[str, Any]:
        """测试RL增强COT-DIR"""
        from problem_preprocessor import ProcessedProblem, Entity
        from qs2_semantic_analyzer import SemanticEntity, QualiaStructure
        from ird_relation_discovery import RelationNetwork, Relation
        
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
        
        test_semantic_entity = SemanticEntity(
            entity_id="semantic_test",
            name="苹果",
            entity_type="object",
            qualia=QualiaStructure([], [], [], []),
            semantic_vector=[],
            confidence=0.9
        )
        
        test_relation_network = RelationNetwork([])
        
        # 执行RL增强推理链构建
        result = self.build_reasoning_chain_rl(
            test_problem, 
            [test_semantic_entity], 
            test_relation_network,
            learning_mode=True
        )
        
        return {
            "success": True,
            "rl_features": {
                "policy_network": True,
                "value_network": True,
                "experience_replay": len(self.experience_buffer) > 0,
                "strategy_library": len(self.strategy_library),
                "reward_shaping": self.config["reward_shaping"]
            },
            "chain_result": result
        }

# 使用示例
if __name__ == "__main__":
    rl_cotdir = RLEnhancedCOTDIR()
    test_result = rl_cotdir.test_rl_cotdir()
    print(f"RL增强COT-DIR测试结果: {test_result}")