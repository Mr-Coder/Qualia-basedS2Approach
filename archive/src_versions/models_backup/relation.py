# src/models/relation.py
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class RelationType(Enum):
    # """关系类型枚举"""
    EXPLICIT = "explicit"      # 显式关系
    IMPLICIT = "implicit"      # 隐式关系
    CAUSAL = "causal"         # 因果关系
    TEMPORAL = "temporal"      # 时序关系
    SPATIAL = "spatial"       # 空间关系
    QUANTITATIVE = "quantitative"  # 数量关系
    LOGICAL = "logical"       # 逻辑关系
    OTHER = "other"           # 其他关系
@dataclass
class Entity:
    """实体类"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = None

@dataclass
class Relation:
    """关系类"""
    id: str
    type: RelationType
    entities: List[Entity]
    attributes: Dict[str, Any]
    description: str = ""
    
    def __init__(self, id: str, type: RelationType, entities: List[Entity], 
                 attributes: Dict[str, Any], description: str = ""):
        self.id = id
        self.type = type
        self.entities = entities
        self.attributes = attributes
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'type': self.type.value,
            'entities': [entity.__dict__ for entity in self.entities],
            'attributes': self.attributes,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """从字典创建关系实例"""
        entities = [Entity(**entity_data) for entity_data in data['entities']]
        return cls(
            id=data['id'],
            type=RelationType(data['type']),
            entities=entities,
            attributes=data['attributes'],
            description=data.get('description', '')
        )

@dataclass
class Relations:
    # """关系集合类"""
    explicit_relations: List[Relation]
    implicit_relations: List[Relation]
    
    def __init__(self, explicit: List[Relation] = None, implicit: List[Relation] = None):
        self.explicit_relations = explicit if explicit is not None else []
        self.implicit_relations = implicit if implicit is not None else []
    
    @property
    def relations(self) -> List[Relation]:
        # """获取所有关系"""
        return self.explicit_relations + self.implicit_relations
    
    def add_relation(self, relation: Relation, is_explicit: bool = True):
        # """添加一个关系
        # Args:
        #     relation: 要添加的关系
        #     is_explicit: 是否为显式关系
        # """
        if is_explicit:
            self.explicit_relations.append(relation)
        else:
            self.implicit_relations.append(relation)
    
    def get_relations(self, relation_type: str = 'all') -> List[Relation]:
        # """获取关系
        # Args:
        #     relation_type: 'all', 'explicit' 或 'implicit'
        # Returns:
        #     指定类型的关系列表
        # """
        if relation_type == 'explicit':
            return self.explicit_relations
        elif relation_type == 'implicit':
            return self.implicit_relations
        return self.relations
    
    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        # """转换为字典格式"""
        return {
            'explicit': [relation.to_dict() for relation in self.explicit_relations],
            'implicit': [relation.to_dict() for relation in self.implicit_relations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[Dict[str, Any]]]) -> 'Relations':
        # """从字典创建关系集合实例"""
        explicit = [Relation.from_dict(r) for r in data.get('explicit', [])]
        implicit = [Relation.from_dict(r) for r in data.get('implicit', [])]
        return cls(explicit=explicit, implicit=implicit)
