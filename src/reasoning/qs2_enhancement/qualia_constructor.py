"""
语义结构构建器 - QS²算法核心组件
==============================

基于Qualia Structure理论构建实体的四维语义结构，
为隐式关系发现提供语义基础。

核心功能：
1. 构建实体的四维语义结构（Formal, Telic, Agentive, Constitutive）
2. 语义角色识别和分类
3. 上下文相关的语义特征提取
4. 与现有实体识别系统的无缝集成
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# 尝试导入现有系统组件
try:
    from ...processors.nlp_processor import NLPProcessor
    from ...models.structures import Entity
except ImportError:
    # 如果现有组件不可用，提供基础替代
    Entity = Dict[str, Any]
    NLPProcessor = None

logger = logging.getLogger(__name__)


class QualiaRole(Enum):
    """四维语义角色类型"""
    FORMAL = "formal"         # 形式角色：区分实体的特征
    TELIC = "telic"          # 目的角色：实体的功能和目的
    AGENTIVE = "agentive"    # 施事角色：实体的起源和创建方式
    CONSTITUTIVE = "constitutive"  # 构成角色：实体的构成和材料


@dataclass
class QualiaStructure:
    """实体的四维语义结构"""
    entity: str
    entity_type: str
    formal_roles: List[str]      # 形式特征
    telic_roles: List[str]       # 功能目的
    agentive_roles: List[str]    # 起源方式
    constitutive_roles: List[str] # 构成成分
    context_features: Dict[str, Any]  # 上下文特征
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entity": self.entity,
            "entity_type": self.entity_type,
            "qualia_roles": {
                "formal": self.formal_roles,
                "telic": self.telic_roles,
                "agentive": self.agentive_roles,
                "constitutive": self.constitutive_roles
            },
            "context_features": self.context_features,
            "confidence": self.confidence
        }


class QualiaStructureConstructor:
    """语义结构构建器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化NLP处理器（如果可用）
        self.nlp_processor = NLPProcessor() if NLPProcessor else None
        
        # 语义角色识别规则
        self.role_patterns = self._initialize_role_patterns()
        
        # 数学实体特定的语义规则
        self.math_entity_rules = self._initialize_math_entity_rules()
        
        self.logger.info("语义结构构建器初始化完成")
    
    def construct_qualia_structure(
        self, 
        entity: Entity, 
        context: str,
        entity_type: Optional[str] = None
    ) -> QualiaStructure:
        """
        构建实体的四维语义结构
        
        Args:
            entity: 实体对象或实体字符串
            context: 问题上下文
            entity_type: 实体类型（可选）
            
        Returns:
            QualiaStructure: 四维语义结构
        """
        try:
            # 标准化实体输入
            entity_str = self._normalize_entity(entity)
            entity_type = entity_type or self._infer_entity_type(entity_str, context)
            
            # 构建四维语义角色
            formal_roles = self._extract_formal_roles(entity_str, context, entity_type)
            telic_roles = self._extract_telic_roles(entity_str, context, entity_type)
            agentive_roles = self._extract_agentive_roles(entity_str, context, entity_type)
            constitutive_roles = self._extract_constitutive_roles(entity_str, context, entity_type)
            
            # 提取上下文特征
            context_features = self._extract_context_features(entity_str, context)
            
            # 计算置信度
            confidence = self._calculate_construction_confidence(
                formal_roles, telic_roles, agentive_roles, constitutive_roles
            )
            
            structure = QualiaStructure(
                entity=entity_str,
                entity_type=entity_type,
                formal_roles=formal_roles,
                telic_roles=telic_roles,
                agentive_roles=agentive_roles,
                constitutive_roles=constitutive_roles,
                context_features=context_features,
                confidence=confidence
            )
            
            self.logger.debug(f"构建实体 '{entity_str}' 的语义结构，置信度: {confidence:.2f}")
            return structure
            
        except Exception as e:
            self.logger.error(f"构建语义结构失败: {str(e)}")
            # 返回基础结构
            return QualiaStructure(
                entity=str(entity),
                entity_type=entity_type or "unknown",
                formal_roles=[],
                telic_roles=[],
                agentive_roles=[],
                constitutive_roles=[],
                context_features={},
                confidence=0.0
            )
    
    def _normalize_entity(self, entity: Entity) -> str:
        """标准化实体表示"""
        if isinstance(entity, str):
            return entity.strip()
        elif isinstance(entity, dict):
            return entity.get("name", entity.get("entity", str(entity)))
        else:
            return str(entity)
    
    def _infer_entity_type(self, entity: str, context: str) -> str:
        """推断实体类型"""
        # 数字实体
        if re.match(r'^\d+(?:\.\d+)?$', entity):
            return "number"
        
        # 单位实体
        if entity in ["米", "厘米", "千米", "元", "角", "分", "小时", "分钟", "秒", "天", "个", "只", "本"]:
            return "unit"
        
        # 数学概念
        math_concepts = ["面积", "周长", "体积", "速度", "时间", "距离", "价格", "成本", "利润"]
        if entity in math_concepts:
            return "math_concept"
        
        # 物理对象
        physical_objects = ["苹果", "书", "车", "房子", "桌子", "水", "油", "钱"]
        if entity in physical_objects:
            return "physical_object"
        
        # 抽象概念
        abstract_concepts = ["问题", "方法", "结果", "答案", "关系"]
        if entity in abstract_concepts:
            return "abstract_concept"
        
        # 基于上下文推断
        if any(word in context for word in ["计算", "求", "算"]):
            return "math_related"
        
        return "general"
    
    def _extract_formal_roles(self, entity: str, context: str, entity_type: str) -> List[str]:
        """提取形式角色（区分特征）"""
        formal_roles = []
        
        # 基于实体类型的形式特征
        if entity_type == "number":
            # 数字的形式特征
            if "." in entity:
                formal_roles.append("小数")
            else:
                formal_roles.append("整数")
            
            # 数值范围特征
            try:
                value = float(entity)
                if value > 0:
                    formal_roles.append("正数")
                elif value < 0:
                    formal_roles.append("负数")
                else:
                    formal_roles.append("零")
            except ValueError:
                pass
        
        elif entity_type == "unit":
            # 单位的形式特征
            unit_categories = {
                "长度单位": ["米", "厘米", "千米", "毫米"],
                "时间单位": ["小时", "分钟", "秒", "天", "年"],
                "货币单位": ["元", "角", "分"],
                "计数单位": ["个", "只", "本", "张"]
            }
            
            for category, units in unit_categories.items():
                if entity in units:
                    formal_roles.append(category)
        
        elif entity_type == "physical_object":
            # 物理对象的形式特征
            if entity in ["苹果", "橙子", "香蕉"]:
                formal_roles.append("水果")
            elif entity in ["书", "本子", "笔"]:
                formal_roles.append("学习用品")
            elif entity in ["车", "船", "飞机"]:
                formal_roles.append("交通工具")
        
        # 从上下文中提取形式特征
        context_formal = self._extract_contextual_formal_features(entity, context)
        formal_roles.extend(context_formal)
        
        return list(set(formal_roles))  # 去重
    
    def _extract_telic_roles(self, entity: str, context: str, entity_type: str) -> List[str]:
        """提取目的角色（功能和用途）"""
        telic_roles = []
        
        # 基于实体类型的目的特征
        if entity_type == "number":
            # 数字在数学问题中的目的
            if "计算" in context:
                telic_roles.append("用于计算")
            if "比较" in context:
                telic_roles.append("用于比较")
            if "测量" in context:
                telic_roles.append("用于测量")
        
        elif entity_type == "unit":
            # 单位的目的
            telic_roles.append("用于度量")
            if entity in ["米", "厘米", "千米"]:
                telic_roles.append("测量长度")
            elif entity in ["小时", "分钟", "秒"]:
                telic_roles.append("测量时间")
            elif entity in ["元", "角", "分"]:
                telic_roles.append("表示价值")
        
        elif entity_type == "physical_object":
            # 物理对象的目的
            if entity == "苹果":
                telic_roles.extend(["食用", "营养供给"])
            elif entity == "书":
                telic_roles.extend(["阅读", "学习"])
            elif entity == "车":
                telic_roles.extend(["运输", "出行"])
        
        # 从上下文中推断目的
        purpose_patterns = {
            "用于计算": ["计算", "算", "求"],
            "用于比较": ["比较", "大于", "小于", "相等"],
            "用于测量": ["测量", "量", "长度", "面积", "体积"],
            "用于购买": ["买", "购买", "价格", "成本"],
            "用于解决问题": ["解决", "求解", "答案"]
        }
        
        for purpose, patterns in purpose_patterns.items():
            if any(pattern in context for pattern in patterns):
                telic_roles.append(purpose)
        
        return list(set(telic_roles))
    
    def _extract_agentive_roles(self, entity: str, context: str, entity_type: str) -> List[str]:
        """提取施事角色（起源和创建方式）"""
        agentive_roles = []
        
        # 基于实体类型的起源特征
        if entity_type == "number":
            # 数字的来源
            if "给定" in context or "已知" in context:
                agentive_roles.append("题目给定")
            if "计算得出" in context:
                agentive_roles.append("计算结果")
            if "测量" in context:
                agentive_roles.append("测量获得")
        
        elif entity_type == "physical_object":
            # 物理对象的来源
            if entity == "苹果":
                agentive_roles.append("自然生长")
            elif entity == "书":
                agentive_roles.append("人工制造")
            elif entity == "钱":
                agentive_roles.append("经济系统")
        
        # 从上下文中识别创建方式
        creation_patterns = {
            "自然形成": ["自然", "生长", "形成"],
            "人工制造": ["制造", "生产", "制作"],
            "计算得出": ["计算", "算出", "得出"],
            "测量获得": ["测量", "量得", "测得"],
            "购买获得": ["买", "购买", "买到"]
        }
        
        for creation, patterns in creation_patterns.items():
            if any(pattern in context for pattern in patterns):
                agentive_roles.append(creation)
        
        return list(set(agentive_roles))
    
    def _extract_constitutive_roles(self, entity: str, context: str, entity_type: str) -> List[str]:
        """提取构成角色（构成和材料）"""
        constitutive_roles = []
        
        # 基于实体类型的构成特征
        if entity_type == "number":
            # 数字的构成
            if "." in entity:
                constitutive_roles.append("整数部分和小数部分")
            constitutive_roles.append("数字符号")
        
        elif entity_type == "physical_object":
            # 物理对象的构成
            if entity == "苹果":
                constitutive_roles.extend(["果皮", "果肉", "果核"])
            elif entity == "书":
                constitutive_roles.extend(["纸张", "文字", "封面"])
            elif entity == "水":
                constitutive_roles.append("H2O分子")
        
        elif entity_type == "math_concept":
            # 数学概念的构成
            if entity == "面积":
                constitutive_roles.append("长度的平方")
            elif entity == "速度":
                constitutive_roles.append("距离除以时间")
            elif entity == "利润":
                constitutive_roles.append("收入减去成本")
        
        # 从上下文中识别构成关系
        composition_patterns = {
            "由...组成": ["组成", "构成", "包含"],
            "由...制成": ["制成", "做成", "材料"],
            "包括": ["包括", "含有", "具有"]
        }
        
        for composition, patterns in composition_patterns.items():
            if any(pattern in context for pattern in patterns):
                constitutive_roles.append(composition)
        
        return list(set(constitutive_roles))
    
    def _extract_context_features(self, entity: str, context: str) -> Dict[str, Any]:
        """提取上下文特征"""
        features = {}
        
        # 实体在上下文中的位置
        entity_positions = [i for i, word in enumerate(context.split()) if entity in word]
        if entity_positions:
            features["first_position"] = entity_positions[0]
            features["last_position"] = entity_positions[-1]
            features["occurrence_count"] = len(entity_positions)
        
        # 实体周围的词汇
        words = context.split()
        surrounding_words = []
        for pos in entity_positions:
            # 前后各取2个词
            start = max(0, pos - 2)
            end = min(len(words), pos + 3)
            surrounding_words.extend(words[start:end])
        
        features["surrounding_words"] = list(set(surrounding_words))
        
        # 实体相关的动词
        verbs = ["计算", "求", "买", "卖", "给", "有", "是", "做", "用", "需要"]
        related_verbs = [verb for verb in verbs if verb in context]
        features["related_verbs"] = related_verbs
        
        # 实体相关的数字
        numbers = re.findall(r'\d+(?:\.\d+)?', context)
        features["related_numbers"] = numbers
        
        # 问题类型特征
        if "面积" in context or "周长" in context:
            features["problem_type"] = "geometry"
        elif "速度" in context or "时间" in context:
            features["problem_type"] = "motion"
        elif "价格" in context or "钱" in context:
            features["problem_type"] = "economics"
        else:
            features["problem_type"] = "general"
        
        return features
    
    def _extract_contextual_formal_features(self, entity: str, context: str) -> List[str]:
        """从上下文中提取形式特征"""
        features = []
        
        # 数量修饰
        if re.search(rf'\d+\s*{re.escape(entity)}', context):
            features.append("可数的")
        
        # 形容词修饰
        adjectives = ["大", "小", "多", "少", "新", "旧", "好", "坏", "快", "慢"]
        for adj in adjectives:
            if f"{adj}{entity}" in context or f"{entity}{adj}" in context:
                features.append(f"具有{adj}的属性")
        
        # 比较关系
        if "比" in context:
            features.append("可比较的")
        
        return features
    
    def _calculate_construction_confidence(
        self, 
        formal: List[str], 
        telic: List[str], 
        agentive: List[str], 
        constitutive: List[str]
    ) -> float:
        """计算语义结构构建的置信度"""
        # 基于角色数量的置信度
        role_counts = [len(formal), len(telic), len(agentive), len(constitutive)]
        non_empty_roles = sum(1 for count in role_counts if count > 0)
        
        # 基础置信度
        base_confidence = non_empty_roles / 4.0
        
        # 角色平衡度加权
        total_roles = sum(role_counts)
        if total_roles > 0:
            balance_score = 1.0 - (max(role_counts) - min(role_counts)) / total_roles
            base_confidence = base_confidence * 0.7 + balance_score * 0.3
        
        return min(1.0, max(0.0, base_confidence))
    
    def _initialize_role_patterns(self) -> Dict[str, List[str]]:
        """初始化语义角色识别模式"""
        return {
            "formal_patterns": [
                r"是\s*(\w+)",
                r"(\w+)的",
                r"属于\s*(\w+)",
                r"(\w+)类"
            ],
            "telic_patterns": [
                r"用于\s*(\w+)",
                r"为了\s*(\w+)",
                r"目的是\s*(\w+)",
                r"可以\s*(\w+)"
            ],
            "agentive_patterns": [
                r"由\s*(\w+)\s*制成",
                r"(\w+)生产",
                r"来自\s*(\w+)",
                r"(\w+)创造"
            ],
            "constitutive_patterns": [
                r"包含\s*(\w+)",
                r"由\s*(\w+)\s*组成",
                r"(\w+)构成",
                r"含有\s*(\w+)"
            ]
        }
    
    def _initialize_math_entity_rules(self) -> Dict[str, Dict[str, List[str]]]:
        """初始化数学实体的特定语义规则"""
        return {
            "number": {
                "formal": ["数值", "量", "大小"],
                "telic": ["计算", "比较", "测量"],
                "agentive": ["给定", "计算", "测量"],
                "constitutive": ["数字", "符号"]
            },
            "unit": {
                "formal": ["单位", "标准", "度量"],
                "telic": ["测量", "标准化", "比较"],
                "agentive": ["约定", "标准化"],
                "constitutive": ["符号", "名称"]
            },
            "geometry": {
                "formal": ["图形", "形状", "几何"],
                "telic": ["计算", "测量", "描述"],
                "agentive": ["绘制", "构造"],
                "constitutive": ["点", "线", "面"]
            }
        }
    
    def batch_construct_structures(
        self, 
        entities: List[Entity], 
        context: str
    ) -> List[QualiaStructure]:
        """批量构建语义结构"""
        structures = []
        
        for entity in entities:
            try:
                structure = self.construct_qualia_structure(entity, context)
                structures.append(structure)
            except Exception as e:
                self.logger.error(f"构建实体 {entity} 的语义结构失败: {e}")
                continue
        
        self.logger.info(f"批量构建完成，成功构建 {len(structures)} 个语义结构")
        return structures
    
    def get_construction_statistics(self) -> Dict[str, Any]:
        """获取构建统计信息"""
        # 这里可以添加统计信息收集逻辑
        return {
            "total_constructions": 0,
            "successful_constructions": 0,
            "average_confidence": 0.0,
            "common_entity_types": [],
            "construction_time_stats": {}
        }