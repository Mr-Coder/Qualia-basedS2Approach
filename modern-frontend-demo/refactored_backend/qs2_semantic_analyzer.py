#!/usr/bin/env python3
"""
QS²语义分析模块
基于Qualia理论的四维语义空间建模
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from problem_preprocessor import Entity, ProcessedProblem

logger = logging.getLogger(__name__)

@dataclass
class QualiaStructure:
    """Qualia语义结构"""
    formal: List[str]       # 形式角色：类型、类别
    telic: List[str]        # 目的角色：功能、用途
    agentive: List[str]     # 施事角色：来源、创造方式
    constitutive: List[str] # 构成角色：组成部分、内在结构

@dataclass
class SemanticEntity:
    """语义增强的实体"""
    entity_id: str
    name: str
    entity_type: str
    qualia: QualiaStructure
    semantic_vector: List[float]
    confidence: float

@dataclass
class CompatibilityResult:
    """兼容性计算结果"""
    entity1_id: str
    entity2_id: str
    compatibility_score: float
    role_similarities: Dict[str, float]
    context_boost: float

class QS2SemanticAnalyzer:
    """QS²语义分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Qualia角色权重
        self.role_weights = {
            "formal": 0.3,
            "telic": 0.4,
            "agentive": 0.1,
            "constitutive": 0.2
        }
        
        # 预定义的语义知识库
        self.semantic_knowledge = self._initialize_semantic_knowledge()
        
        # 上下文分析器
        self.context_analyzer = ContextAnalyzer()

    def _initialize_semantic_knowledge(self) -> Dict[str, QualiaStructure]:
        """初始化语义知识库"""
        knowledge = {}
        
        # 人物实体的Qualia结构
        person_qualia = QualiaStructure(
            formal=["人物实体", "主体", "个体", "认知主体"],
            telic=["拥有物品", "参与活动", "进行计算", "执行动作"],
            agentive=["题目设定", "问题构造", "故事角色"],
            constitutive=["意识", "行为能力", "认知能力"]
        )
        
        # 物品实体的Qualia结构
        object_qualia = QualiaStructure(
            formal=["物理实体", "可数对象", "有形物体"],
            telic=["被拥有", "被使用", "参与计算", "作为计数单位"],
            agentive=["自然存在", "人工制造", "题目元素"],
            constitutive=["物理属性", "可分离性", "独立性"]
        )
        
        # 数量实体的Qualia结构
        number_qualia = QualiaStructure(
            formal=["数值概念", "抽象实体", "数学对象"],
            telic=["表示数量", "参与运算", "描述关系", "量化属性"],
            agentive=["数学抽象", "计数结果", "测量产生"],
            constitutive=["数值大小", "单位属性", "精度信息"]
        )
        
        # 具体实体的Qualia结构
        specific_entities = {
            "苹果": QualiaStructure(
                formal=["水果", "可数物体", "圆形物体", "食物"],
                telic=["食用", "计算对象", "交换媒介", "营养供给"],
                agentive=["树木生长", "自然产物", "农业产品"],
                constitutive=["果肉", "果皮", "果核", "营养成分"]
            ),
            "书": QualiaStructure(
                formal=["出版物", "知识载体", "矩形物体"],
                telic=["阅读", "学习", "计算对象", "知识传递"],
                agentive=["编写出版", "印刷制作", "知识整理"],
                constitutive=["纸张", "文字", "图片", "封面"]
            ),
            "钱": QualiaStructure(
                formal=["货币", "价值载体", "交换媒介"],
                telic=["购买", "计算价值", "储存财富", "交易媒介"],
                agentive=["经济制度", "价值抽象", "社会约定"],
                constitutive=["面额", "材质", "防伪特征"]
            )
        }
        
        # 将预定义结构加入知识库
        knowledge.update({
            "person": person_qualia,
            "object": object_qualia,
            "number": number_qualia,
            **specific_entities
        })
        
        return knowledge

    def analyze_semantics(self, processed_problem: ProcessedProblem) -> List[SemanticEntity]:
        """
        对预处理后的问题进行语义分析
        
        Args:
            processed_problem: 预处理后的问题数据
            
        Returns:
            List[SemanticEntity]: 语义增强的实体列表
        """
        try:
            self.logger.info(f"开始QS²语义分析，实体数量: {len(processed_problem.entities)}")
            
            semantic_entities = []
            
            for entity in processed_problem.entities:
                # 构建Qualia结构
                qualia = self._build_qualia_structure(entity, processed_problem.cleaned_text)
                
                # 生成语义向量
                semantic_vector = self._generate_semantic_vector(qualia)
                
                # 创建语义实体
                semantic_entity = SemanticEntity(
                    entity_id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    qualia=qualia,
                    semantic_vector=semantic_vector,
                    confidence=entity.confidence
                )
                
                semantic_entities.append(semantic_entity)
                
                self.logger.debug(f"构建语义实体: {entity.name} -> {len(qualia.formal)}个formal角色")
            
            self.logger.info(f"QS²语义分析完成，生成{len(semantic_entities)}个语义实体")
            return semantic_entities
            
        except Exception as e:
            self.logger.error(f"QS²语义分析失败: {e}")
            return []

    def _build_qualia_structure(self, entity: Entity, context: str) -> QualiaStructure:
        """构建实体的Qualia结构"""
        
        # 从知识库获取基础Qualia结构
        base_qualia = self._get_base_qualia(entity)
        
        # 上下文增强
        context_enhanced_qualia = self._enhance_with_context(base_qualia, entity, context)
        
        return context_enhanced_qualia

    def _get_base_qualia(self, entity: Entity) -> QualiaStructure:
        """从知识库获取基础Qualia结构"""
        
        # 优先查找具体实体
        if entity.name in self.semantic_knowledge:
            return self.semantic_knowledge[entity.name]
        
        # 根据实体类型查找
        if entity.entity_type in self.semantic_knowledge:
            return self.semantic_knowledge[entity.entity_type]
        
        # 默认通用结构
        return QualiaStructure(
            formal=["通用实体", "题目元素"],
            telic=["参与计算", "表达信息"],
            agentive=["题目设定"],
            constitutive=["基本属性"]
        )

    def _enhance_with_context(self, base_qualia: QualiaStructure, 
                            entity: Entity, context: str) -> QualiaStructure:
        """基于上下文增强Qualia结构"""
        
        enhanced_formal = list(base_qualia.formal)
        enhanced_telic = list(base_qualia.telic)
        enhanced_agentive = list(base_qualia.agentive)
        enhanced_constitutive = list(base_qualia.constitutive)
        
        # 基于上下文的关键词分析
        if "拥有" in context or "有" in context:
            enhanced_telic.append("拥有关系对象")
        
        if "一共" in context or "总共" in context:
            enhanced_telic.append("求和计算对象")
        
        if "买" in context or "购买" in context:
            enhanced_telic.append("交易对象")
            enhanced_formal.append("商品")
        
        if "速度" in context or "时间" in context:
            enhanced_formal.append("运动相关")
            enhanced_telic.append("物理量计算")
        
        # 数量相关的特殊处理
        if entity.entity_type == "number":
            if "个" in context:
                enhanced_constitutive.append("计数单位")
            if "元" in context or "钱" in context:
                enhanced_constitutive.append("货币单位")
        
        return QualiaStructure(
            formal=list(set(enhanced_formal)),
            telic=list(set(enhanced_telic)),
            agentive=list(set(enhanced_agentive)),
            constitutive=list(set(enhanced_constitutive))
        )

    def _generate_semantic_vector(self, qualia: QualiaStructure) -> List[float]:
        """生成语义向量表示"""
        
        # 简化的语义向量生成（实际应用中可以使用词向量模型）
        vector = []
        
        # 将Qualia角色转换为数值特征
        for role_name, role_values in [
            ("formal", qualia.formal),
            ("telic", qualia.telic),
            ("agentive", qualia.agentive),
            ("constitutive", qualia.constitutive)
        ]:
            # 基于角色内容计算特征值
            role_feature = len(role_values) * self.role_weights[role_name]
            vector.append(role_feature)
            
            # 添加角色内容的语义特征
            content_feature = sum(hash(item) % 100 / 100.0 for item in role_values) / max(len(role_values), 1)
            vector.append(content_feature)
        
        return vector

    def compute_compatibility(self, entity1: SemanticEntity, 
                            entity2: SemanticEntity, context: str = "") -> CompatibilityResult:
        """
        计算两个实体间的语义兼容性
        
        Args:
            entity1: 第一个语义实体
            entity2: 第二个语义实体
            context: 上下文信息
            
        Returns:
            CompatibilityResult: 兼容性计算结果
        """
        
        try:
            # 计算各个Qualia角色的相似度
            role_similarities = {}
            
            role_similarities["formal"] = self._calculate_role_similarity(
                entity1.qualia.formal, entity2.qualia.formal
            )
            role_similarities["telic"] = self._calculate_role_similarity(
                entity1.qualia.telic, entity2.qualia.telic
            )
            role_similarities["agentive"] = self._calculate_role_similarity(
                entity1.qualia.agentive, entity2.qualia.agentive
            )
            role_similarities["constitutive"] = self._calculate_role_similarity(
                entity1.qualia.constitutive, entity2.qualia.constitutive
            )
            
            # 加权计算总体兼容性
            compatibility_score = sum(
                self.role_weights[role] * similarity 
                for role, similarity in role_similarities.items()
            )
            
            # 上下文增强
            context_boost = self.context_analyzer.analyze_context_connection(
                entity1, entity2, context
            )
            
            # 最终兼容性分数
            final_compatibility = compatibility_score * (1 + context_boost)
            final_compatibility = min(final_compatibility, 1.0)
            
            return CompatibilityResult(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                compatibility_score=final_compatibility,
                role_similarities=role_similarities,
                context_boost=context_boost
            )
            
        except Exception as e:
            self.logger.error(f"兼容性计算失败: {e}")
            return CompatibilityResult(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                compatibility_score=0.0,
                role_similarities={},
                context_boost=0.0
            )

    def _calculate_role_similarity(self, role1: List[str], role2: List[str]) -> float:
        """计算Qualia角色相似度"""
        
        if not role1 or not role2:
            return 0.0
        
        # 计算交集比例
        set1, set2 = set(role1), set(role2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # 语义相似度增强（简化版本）
        semantic_similarity = 0.0
        for item1 in role1:
            for item2 in role2:
                if self._are_semantically_related(item1, item2):
                    semantic_similarity += 0.1
        
        semantic_similarity = min(semantic_similarity, 0.5)
        
        return min(jaccard_similarity + semantic_similarity, 1.0)

    def _are_semantically_related(self, concept1: str, concept2: str) -> bool:
        """判断两个概念是否语义相关（简化版本）"""
        
        # 简化的语义关联判断
        related_groups = [
            ["人物实体", "主体", "个体", "认知主体"],
            ["物理实体", "可数对象", "有形物体"],
            ["拥有", "被拥有", "所有权", "归属"],
            ["计算", "运算", "数学", "数值"],
            ["食物", "水果", "营养", "食用"],
            ["知识", "学习", "教育", "书籍"]
        ]
        
        for group in related_groups:
            if concept1 in group and concept2 in group:
                return True
        
        return False

class ContextAnalyzer:
    """上下文分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze_context_connection(self, entity1: SemanticEntity, 
                                 entity2: SemanticEntity, context: str) -> float:
        """分析实体在上下文中的连接强度"""
        
        connection_boost = 0.0
        
        # 基于实体名称在上下文中的距离
        if entity1.name in context and entity2.name in context:
            context_distance = self._calculate_text_distance(entity1.name, entity2.name, context)
            if context_distance < 20:  # 字符距离较近
                connection_boost += 0.2
        
        # 基于关键词连接
        connection_keywords = ["有", "拥有", "买", "卖", "给", "从", "和", "与", "共同"]
        for keyword in connection_keywords:
            if keyword in context:
                connection_boost += 0.1
        
        # 基于数学关系词汇
        math_keywords = ["一共", "总共", "合计", "相加", "加起来"]
        for keyword in math_keywords:
            if keyword in context:
                connection_boost += 0.15
        
        return min(connection_boost, 0.5)  # 最大增强50%

    def _calculate_text_distance(self, name1: str, name2: str, text: str) -> int:
        """计算两个实体名称在文本中的距离"""
        try:
            pos1 = text.find(name1)
            pos2 = text.find(name2)
            if pos1 == -1 or pos2 == -1:
                return float('inf')
            return abs(pos1 - pos2)
        except:
            return float('inf')

# 测试函数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from problem_preprocessor import ProblemPreprocessor
    
    # 创建分析器
    preprocessor = ProblemPreprocessor()
    qs2_analyzer = QS2SemanticAnalyzer()
    
    # 测试问题
    test_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    print(f"测试问题: {test_problem}")
    print("="*50)
    
    # 预处理
    processed = preprocessor.preprocess(test_problem)
    print(f"预处理实体: {[e.name for e in processed.entities]}")
    
    # QS²语义分析
    semantic_entities = qs2_analyzer.analyze_semantics(processed)
    print(f"语义实体数量: {len(semantic_entities)}")
    
    # 显示语义结构
    for entity in semantic_entities:
        print(f"\n实体: {entity.name}")
        print(f"  Formal: {entity.qualia.formal}")
        print(f"  Telic: {entity.qualia.telic}")
        print(f"  Agentive: {entity.qualia.agentive}")
        print(f"  Constitutive: {entity.qualia.constitutive}")
    
    # 计算兼容性
    if len(semantic_entities) >= 2:
        print(f"\n兼容性分析:")
        for i in range(len(semantic_entities)):
            for j in range(i+1, len(semantic_entities)):
                compatibility = qs2_analyzer.compute_compatibility(
                    semantic_entities[i], semantic_entities[j], test_problem
                )
                print(f"{semantic_entities[i].name} <-> {semantic_entities[j].name}: {compatibility.compatibility_score:.3f}")