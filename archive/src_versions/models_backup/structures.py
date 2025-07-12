"""
数据结构定义模块

This module contains all the data structure definitions used in the project,
including text processing, relation extraction, equation solving and inference models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class MatchedModel(BaseModel):
    id: str
    pattern: str
    relation_template: str
    var_slot_val: str
    var_slot_index: Dict[str, str]

class RelationEntity(BaseModel):
    relation: str
    var_entity: Dict[str, str]
    sent_text: str
    matched_token: List[List[str]]
    matched_model: MatchedModel

class Relations(BaseModel):
    explicit_relations: List[RelationEntity]
    implicit_relations: List[RelationEntity]

    @property
    def explicit(self):
        return self.explicit_relations

    @property
    def implicit(self):
        return self.implicit_relations


# Text Processing Models
@dataclass
class ProcessedTextData:
    """处理后的文本结构
    
    属性:
        raw_text: 原始文本
        segmentation: 分词结果
        pos_tags: 词性标注
        dependencies: 依存关系
    """
    raw_text: str
    segmentation: List[str]
    pos_tags: List[str]
    dependencies: List[Any] = field(default_factory=list)

class Relations(BaseModel):
    """关系提取结果的集合"""
    explicit_relations: List[RelationEntity] = Field(
        default_factory=list,
        description="显式关系列表"
    )
    implicit_relations: List[RelationEntity] = Field(
        default_factory=list,
        description="隐式关系列表"
    )

    @property
    def explicit(self) -> List[RelationEntity]:
        """获取显式关系列表"""
        return self.explicit_relations

    @property
    def implicit(self) -> List[RelationEntity]:
        """获取隐式关系列表"""
        return self.implicit_relations

    def __len__(self) -> int:
        """返回所有关系的总数"""
        return len(self.explicit_relations) + len(self.implicit_relations)

@dataclass
class ExtractionResult:
    """提取结果类
    
    属性:
        status: 提取状态
        explicit_relations: 显式关系列表
        implicit_relations: 隐式关系列表
        message: 错误信息（如果有）
    """
    status: str = 'success'
    explicit_relations: List[Dict] = field(default_factory=list)
    implicit_relations: List[Dict] = field(default_factory=list)
    message: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 字典形式的提取结果
        """
        return {
            'status': self.status,
            'explicit_relations': self.explicit_relations,
            'implicit_relations': self.implicit_relations,
            'message': self.message
        }

class RelationCollection(BaseModel):
    """关系集合，用于存储和组织提取的关系"""
    explicit: List[Dict[str, Any]] = Field(..., description="显式关系集合")
    implicit: List[Dict[str, Any]] = Field(..., description="隐式关系集合")

# Equation Models
class Context(BaseModel):
    """方程式上下文信息"""
    sentence: str = Field(..., description="相关句子")
    position: Tuple[int, int] = Field(..., description="在文本中的位置")

class Equation(BaseModel):
    """数学方程式表示"""
    type: str = Field(..., description="方程类型")
    variables: List[str] = Field(..., description="变量列表")
    coefficients: List[float] = Field(..., description="系数列表")
    constant: float = Field(..., description="常数项")
    context: Context = Field(..., description="方程上下文")

class Equations(BaseModel):
    """方程组系统"""
    system: List[Dict[str, Any]] = Field(..., description="方程系统")
    variables: Dict[str, Any] = Field(..., description="变量信息")
    constraints: List[Dict[str, Any]] = Field(..., description="约束条件")

# Solution Models
class Solution(BaseModel):
    """问题求解结果"""
    result: Dict[str, Any] = Field(..., description="求解结果")
    steps: List[Dict[str, Any]] = Field(..., description="求解步骤")

# Entity and Attribute Models
class Entities(BaseModel):
    """实体关系"""
    source: str = Field(..., description="源实体")
    target: str = Field(..., description="目标实体")

class Attributes(BaseModel):
    """属性信息"""
    quantity: float = Field(..., description="数量值")
    unit: str = Field(..., description="单位")

# Inference Models
class InferenceResult(BaseModel):
    """推理结果"""
    conclusion: str = Field(..., description="推理结论")
    evidence: List[str] = Field(..., description="支持证据")

class InferenceStep(BaseModel):
    """推理步骤"""
    step_number: int = Field(..., description="步骤编号")
    description: str = Field(..., description="步骤描述")
    reasoning: str = Field(..., description="推理过程")
    result: InferenceResult = Field(..., description="推理结果")

# Problem Structure
 
class ProblemStructure(BaseModel):
    """问题整体结构"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text: str = Field(..., description="原始问题文本")
    processed: ProcessedTextData = Field(..., description="处理后的文本")
    relations: RelationCollection = Field(..., description="提取的关系")
    equations: Equations = Field(..., description="方程系统")
    solution: Solution = Field(..., description="问题解答")

@dataclass
class FeatureSet:
    """问题特征集
    
    属性:
        math_complexity: 数学复杂度特征
        linguistic_structure: 语言结构特征
        relation_type: 关系类型特征
        domain_indicators: 问题领域特征
        question_target: 问题目标特征
    """
    math_complexity: Dict[str, Any] = field(default_factory=dict)
    linguistic_structure: Dict[str, Any] = field(default_factory=dict)
    relation_type: Dict[str, Any] = field(default_factory=dict)
    domain_indicators: Dict[str, Any] = field(default_factory=dict)
    question_target: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternMatch:
    """模式匹配结果
    
    属性:
        pattern_id: 模式ID
        matched_text: 匹配到的文本
        score: 匹配分数
        variables: 提取的变量
    """
    pattern_id: str
    matched_text: str
    score: float
    variables: Dict[str, Any] = field(default_factory=dict)