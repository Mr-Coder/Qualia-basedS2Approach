#!/usr/bin/env python3
"""
激活扩散推理引擎 - 基于交互式物性图谱的核心理念
Activation Diffusion Reasoning Engine - Based on Interactive Property Graph
"""

import logging
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """节点类型"""
    CONCEPT = "concept"      # 概念节点（实体、关系、属性）
    STRATEGY = "strategy"    # 策略节点（COT、GOT、TOT）
    DOMAIN = "domain"        # 领域节点（算术、几何、代数）
    SKILL = "skill"          # 技能节点（分解、建模、验证）

class ActivationState(Enum):
    """激活状态"""
    INACTIVE = "inactive"    # 未激活
    PRIMED = "primed"       # 预激活
    ACTIVE = "active"       # 激活
    DECAYING = "decaying"   # 衰减中

@dataclass
class PropertyNode:
    """物性节点 - 交互式物性图谱的基本单元"""
    id: str
    name: str
    description: str
    node_type: NodeType
    details: List[str]
    
    # 激活扩散相关属性
    activation_level: float = 0.0  # 激活强度 [0, 1]
    activation_state: ActivationState = ActivationState.INACTIVE
    activation_history: List[Tuple[float, float]] = field(default_factory=list)  # (time, level)
    
    # 空间属性（用于可视化）
    x: float = 0.0
    y: float = 0.0
    
    # 连接关系
    connections: List[str] = field(default_factory=list)
    
    def activate(self, strength: float, timestamp: float = None):
        """激活节点"""
        if timestamp is None:
            timestamp = time.time()
        
        # 更新激活强度
        self.activation_level = min(1.0, self.activation_level + strength)
        
        # 更新激活状态
        if self.activation_level > 0.8:
            self.activation_state = ActivationState.ACTIVE
        elif self.activation_level > 0.3:
            self.activation_state = ActivationState.PRIMED
        else:
            self.activation_state = ActivationState.INACTIVE
        
        # 记录激活历史
        self.activation_history.append((timestamp, self.activation_level))
        
        # 保持历史记录长度
        if len(self.activation_history) > 50:
            self.activation_history = self.activation_history[-50:]
    
    def decay(self, decay_rate: float = 0.1):
        """激活衰减"""
        self.activation_level = max(0.0, self.activation_level - decay_rate)
        
        if self.activation_level < 0.1:
            self.activation_state = ActivationState.INACTIVE
        elif self.activation_level < 0.5:
            self.activation_state = ActivationState.DECAYING

@dataclass
class PropertyConnection:
    """物性连接 - 节点间的关系"""
    from_node: str
    to_node: str
    connection_type: str
    weight: float
    label: str
    bidirectional: bool = True
    
    def get_activation_transfer(self, source_activation: float) -> float:
        """计算激活传递强度"""
        return source_activation * self.weight * 0.5  # 传递损失50%

class ActivationDiffusionEngine:
    """激活扩散推理引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 节点和连接
        self.nodes: Dict[str, PropertyNode] = {}
        self.connections: List[PropertyConnection] = []
        
        # 扩散参数
        self.diffusion_threshold = 0.3  # 扩散阈值
        self.decay_rate = 0.05  # 衰减率
        self.max_diffusion_depth = 3  # 最大扩散深度
        
        # 初始化物性图谱
        self._initialize_property_graph()
        
        self.logger.info("激活扩散推理引擎初始化完成")
    
    def _initialize_property_graph(self):
        """初始化物性图谱 - 基于真实的数学推理结构"""
        
        # 核心概念节点
        concept_nodes = [
            PropertyNode(
                id="entity", name="实体", description="问题中的基本对象",
                node_type=NodeType.CONCEPT, x=150, y=100,
                details=["人物", "物品", "数量", "单位"],
                connections=["relation", "modeling", "arithmetic"]
            ),
            PropertyNode(
                id="relation", name="关系", description="实体间的连接",
                node_type=NodeType.CONCEPT, x=350, y=100,
                details=["数量关系", "空间关系", "因果关系"],
                connections=["property", "reasoning", "got"]
            ),
            PropertyNode(
                id="property", name="属性", description="实体的特征",
                node_type=NodeType.CONCEPT, x=550, y=100,
                details=["数值属性", "类别属性", "约束条件"],
                connections=["constraint", "verification"]
            ),
            PropertyNode(
                id="constraint", name="约束", description="问题的限制条件",
                node_type=NodeType.CONCEPT, x=750, y=100,
                details=["非负约束", "整数约束", "守恒约束"],
                connections=["reasoning", "tot"]
            )
        ]
        
        # 推理策略节点
        strategy_nodes = [
            PropertyNode(
                id="cot", name="链式思维", description="逐步推理",
                node_type=NodeType.STRATEGY, x=150, y=300,
                details=["步骤分解", "逻辑链条", "顺序执行"],
                connections=["decomposition", "verification"]
            ),
            PropertyNode(
                id="got", name="图式思维", description="关系网络推理",
                node_type=NodeType.STRATEGY, x=350, y=300,
                details=["网络分析", "关系发现", "并行推理"],
                connections=["relation", "analysis"]
            ),
            PropertyNode(
                id="tot", name="树式思维", description="多路径探索",
                node_type=NodeType.STRATEGY, x=550, y=300,
                details=["路径搜索", "方案评估", "最优选择"],
                connections=["exploration", "evaluation"]
            )
        ]
        
        # 领域知识节点
        domain_nodes = [
            PropertyNode(
                id="arithmetic", name="算术", description="基本数学运算",
                node_type=NodeType.DOMAIN, x=150, y=500,
                details=["加减乘除", "数值计算", "运算规则"],
                connections=["entity", "modeling"]
            ),
            PropertyNode(
                id="geometry", name="几何", description="空间形状关系",
                node_type=NodeType.DOMAIN, x=350, y=500,
                details=["图形计算", "空间推理", "测量分析"],
                connections=["property", "analysis"]
            ),
            PropertyNode(
                id="algebra", name="代数", description="符号化数学",
                node_type=NodeType.DOMAIN, x=550, y=500,
                details=["方程求解", "符号运算", "函数关系"],
                connections=["relation", "reasoning"]
            )
        ]
        
        # 技能节点
        skill_nodes = [
            PropertyNode(
                id="decomposition", name="分解", description="问题分解",
                node_type=NodeType.SKILL, x=150, y=700,
                details=["结构分析", "子问题识别", "分解策略"],
                connections=["cot"]
            ),
            PropertyNode(
                id="modeling", name="建模", description="数学建模",
                node_type=NodeType.SKILL, x=350, y=700,
                details=["抽象建模", "参数确定", "模型验证"],
                connections=["entity", "arithmetic"]
            ),
            PropertyNode(
                id="analysis", name="分析", description="深度分析",
                node_type=NodeType.SKILL, x=550, y=700,
                details=["模式识别", "关系分析", "逻辑推理"],
                connections=["got", "geometry"]
            ),
            PropertyNode(
                id="verification", name="验证", description="结果验证",
                node_type=NodeType.SKILL, x=750, y=700,
                details=["结果检查", "约束验证", "合理性评估"],
                connections=["property", "cot"]
            ),
            PropertyNode(
                id="exploration", name="探索", description="方案探索",
                node_type=NodeType.SKILL, x=250, y=900,
                details=["多方案尝试", "创新思维", "路径优化"],
                connections=["tot"]
            ),
            PropertyNode(
                id="evaluation", name="评估", description="方案评估",
                node_type=NodeType.SKILL, x=450, y=900,
                details=["方案比较", "优劣分析", "决策制定"],
                connections=["tot"]
            )
        ]
        
        # 添加所有节点
        all_nodes = concept_nodes + strategy_nodes + domain_nodes + skill_nodes
        for node in all_nodes:
            self.nodes[node.id] = node
        
        # 建立连接关系
        self._build_connections()
    
    def _build_connections(self):
        """构建节点间的连接关系"""
        
        connection_definitions = [
            # 概念间的基础连接
            ("entity", "relation", "dependency", 0.8, "依赖"),
            ("relation", "property", "enhancement", 0.7, "增强"),
            ("property", "constraint", "application", 0.9, "应用"),
            ("constraint", "reasoning", "dependency", 0.8, "依赖"),
            
            # 策略间的关系
            ("cot", "got", "enhancement", 0.6, "增强"),
            ("got", "tot", "enhancement", 0.6, "增强"),
            ("cot", "tot", "application", 0.5, "应用"),
            
            # 概念到策略的映射
            ("entity", "cot", "application", 0.7, "应用"),
            ("relation", "got", "application", 0.9, "应用"),
            ("constraint", "tot", "application", 0.8, "应用"),
            
            # 领域知识连接
            ("arithmetic", "entity", "dependency", 0.8, "依赖"),
            ("geometry", "property", "dependency", 0.7, "依赖"),
            ("algebra", "relation", "dependency", 0.8, "依赖"),
            
            # 技能连接
            ("decomposition", "cot", "enhancement", 0.9, "增强"),
            ("modeling", "entity", "application", 0.8, "应用"),
            ("analysis", "got", "enhancement", 0.9, "增强"),
            ("verification", "constraint", "application", 0.9, "应用"),
            ("exploration", "tot", "enhancement", 0.8, "增强"),
            ("evaluation", "tot", "enhancement", 0.8, "增强"),
            
            # 跨层连接
            ("reasoning", "cot", "dependency", 0.7, "依赖"),
            ("reasoning", "got", "dependency", 0.7, "依赖"),
            ("reasoning", "tot", "dependency", 0.7, "依赖")
        ]
        
        for from_id, to_id, conn_type, weight, label in connection_definitions:
            if from_id in self.nodes and to_id in self.nodes:
                self.connections.append(PropertyConnection(
                    from_node=from_id,
                    to_node=to_id,
                    connection_type=conn_type,
                    weight=weight,
                    label=label
                ))
    
    def activate_nodes_from_problem(self, problem_text: str, entities: List[Dict]) -> Dict[str, float]:
        """根据问题激活相关节点"""
        
        activations = {}
        
        # 重置所有节点激活状态
        for node in self.nodes.values():
            node.activation_level = 0.0
            node.activation_state = ActivationState.INACTIVE
        
        # 基于问题文本激活节点
        if any(word in problem_text for word in ["一共", "总共"]):
            self.nodes["arithmetic"].activate(0.9)
            self.nodes["cot"].activate(0.8)
            activations["arithmetic"] = 0.9
            activations["cot"] = 0.8
        
        if any(word in problem_text for word in ["关系", "比", "相互"]):
            self.nodes["relation"].activate(0.8)
            self.nodes["got"].activate(0.9)
            activations["relation"] = 0.8
            activations["got"] = 0.9
        
        if any(word in problem_text for word in ["如果", "假设", "条件"]):
            self.nodes["constraint"].activate(0.9)
            self.nodes["tot"].activate(0.8)
            activations["constraint"] = 0.9
            activations["tot"] = 0.8
        
        # 基于实体激活节点
        if entities:
            self.nodes["entity"].activate(0.8)
            activations["entity"] = 0.8
        
        # 检查是否需要验证
        if any(word in problem_text for word in ["验证", "检查", "合理"]):
            self.nodes["verification"].activate(0.7)
            activations["verification"] = 0.7
        
        # 执行激活扩散
        diffusion_result = self._perform_activation_diffusion()
        activations.update(diffusion_result)
        
        return activations
    
    def _perform_activation_diffusion(self) -> Dict[str, float]:
        """执行激活扩散过程"""
        
        diffusion_result = {}
        
        # 多轮扩散
        for depth in range(self.max_diffusion_depth):
            new_activations = {}
            
            # 对每个激活的节点，向相邻节点扩散
            for node_id, node in self.nodes.items():
                if node.activation_level > self.diffusion_threshold:
                    
                    # 找到所有连接的节点
                    for connection in self.connections:
                        target_id = None
                        if connection.from_node == node_id:
                            target_id = connection.to_node
                        elif connection.to_node == node_id and connection.bidirectional:
                            target_id = connection.from_node
                        
                        if target_id and target_id in self.nodes:
                            # 计算扩散强度
                            transfer_strength = connection.get_activation_transfer(node.activation_level)
                            
                            # 累加激活强度
                            if target_id not in new_activations:
                                new_activations[target_id] = 0.0
                            new_activations[target_id] += transfer_strength
            
            # 应用新的激活
            for node_id, activation in new_activations.items():
                if activation > 0.1:  # 过滤微弱激活
                    self.nodes[node_id].activate(activation)
                    diffusion_result[node_id] = self.nodes[node_id].activation_level
            
            # 应用衰减
            for node in self.nodes.values():
                node.decay(self.decay_rate)
        
        return diffusion_result
    
    def get_activated_reasoning_path(self) -> List[Dict[str, Any]]:
        """获取激活的推理路径"""
        
        path = []
        
        # 获取所有激活的节点，按激活强度排序
        activated_nodes = [
            (node_id, node) for node_id, node in self.nodes.items()
            if node.activation_level > 0.2
        ]
        activated_nodes.sort(key=lambda x: x[1].activation_level, reverse=True)
        
        # 构建推理路径
        for i, (node_id, node) in enumerate(activated_nodes):
            step = {
                "step_id": i + 1,
                "node_id": node_id,
                "node_name": node.name,
                "node_type": node.node_type.value,
                "activation_level": node.activation_level,
                "activation_state": node.activation_state.value,
                "description": node.description,
                "details": node.details,
                "reasoning": self._generate_step_reasoning(node)
            }
            path.append(step)
        
        return path
    
    def _generate_step_reasoning(self, node: PropertyNode) -> str:
        """为节点生成推理解释"""
        
        activation_strength = "强" if node.activation_level > 0.7 else ("中" if node.activation_level > 0.4 else "弱")
        
        reasoning_templates = {
            NodeType.CONCEPT: f"概念'{node.name}'被{activation_strength}激活，表明问题涉及{node.description}相关内容",
            NodeType.STRATEGY: f"策略'{node.name}'被{activation_strength}激活，建议采用{node.description}方法求解",
            NodeType.DOMAIN: f"领域'{node.name}'被{activation_strength}激活，问题属于{node.description}范畴",
            NodeType.SKILL: f"技能'{node.name}'被{activation_strength}激活，需要运用{node.description}能力"
        }
        
        return reasoning_templates.get(node.node_type, f"节点'{node.name}'被{activation_strength}激活")
    
    def get_network_state(self) -> Dict[str, Any]:
        """获取当前网络状态"""
        
        return {
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "description": node.description,
                    "type": node.node_type.value,
                    "activation_level": node.activation_level,
                    "activation_state": node.activation_state.value,
                    "x": node.x,
                    "y": node.y,
                    "details": node.details
                }
                for node in self.nodes.values()
            ],
            "connections": [
                {
                    "from": conn.from_node,
                    "to": conn.to_node,
                    "type": conn.connection_type,
                    "weight": conn.weight,
                    "label": conn.label
                }
                for conn in self.connections
            ],
            "total_activation": sum(node.activation_level for node in self.nodes.values()),
            "active_nodes_count": len([n for n in self.nodes.values() if n.activation_level > 0.3])
        }

# 创建全局实例
activation_engine = ActivationDiffusionEngine()

def get_activation_engine() -> ActivationDiffusionEngine:
    """获取激活扩散引擎实例"""
    return activation_engine

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = ActivationDiffusionEngine()
    
    # 测试激活扩散
    test_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    test_entities = [{"name": "小明", "type": "person"}, {"name": "苹果", "type": "object"}]
    
    print("=== 激活扩散测试 ===")
    activations = engine.activate_nodes_from_problem(test_problem, test_entities)
    
    print(f"激活的节点数量: {len(activations)}")
    for node_id, level in sorted(activations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node_id}: {level:.3f} ({engine.nodes[node_id].name})")
    
    print("\n=== 推理路径 ===")
    reasoning_path = engine.get_activated_reasoning_path()
    for step in reasoning_path:
        print(f"步骤{step['step_id']}: {step['reasoning']}")
    
    print("\n=== 网络状态 ===")
    network_state = engine.get_network_state()
    print(f"总激活强度: {network_state['total_activation']:.3f}")
    print(f"激活节点数: {network_state['active_nodes_count']}")