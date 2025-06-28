import unittest
from typing import Dict, List, Tuple

from models.relation import Entity, Relation, RelationType
from src.models.relation import Entity, Relation, RelationType
from src.processors.relation_matcher import RelationMatcher  # 添加这行导入语句


class TestRelationMatcher(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.matcher = RelationMatcher()
        
        # 创建测试用的实体
        self.entity1 = Entity(id="e1", name="水箱", type="container")
        self.entity2 = Entity(id="e2", name="冰块", type="object")
        self.entity3 = Entity(id="e3", name="水", type="liquid")
        
        # 创建基准关系
        self.base_relation = Relation(
            id="r1",
            type=RelationType.QUANTITATIVE,
            entities=[self.entity1, self.entity2],
            attributes={"rate": "1 cube/minute"},
            description="冰块放入水箱的速率关系"
        )

    def test_exact_match(self):
        """测试完全相同的关系"""
        identical_relation = Relation(
            id="r2",
            type=RelationType.QUANTITATIVE,
            entities=[self.entity1, self.entity2],
            attributes={"rate": "1 cube/minute"},
            description="相同的关系"
        )
        
        result = self.matcher.match(self.base_relation, identical_relation)
        
        self.assertTrue(result['matched'])
        self.assertEqual(result['similarity_score'], 1.0)
        self.assertIn('type', result['matching_details']['matches'])
        self.assertIn('entities', result['matching_details']['matches'])
        self.assertIn('attributes', result['matching_details']['matches'])
        
    def test_different_type(self):
        """测试不同类型的关系"""
        different_type_relation = Relation(
            id="r3",
            type=RelationType.SPATIAL,  # 不同的类型
            entities=[self.entity1, self.entity2],
            attributes={"rate": "1 cube/minute"},
            description="不同类型的关系"
        )
        
        result = self.matcher.match(self.base_relation, different_type_relation)
        
        self.assertFalse(result['matched'])
        self.assertLess(result['similarity_score'], 1.0)
        self.assertIn('type', result['matching_details']['differences'])
        
    def test_different_entities(self):
        """测试不同实体的关系"""
        different_entities_relation = Relation(
            id="r4",
            type=RelationType.QUANTITATIVE,
            entities=[self.entity1, self.entity3],  # 不同的实体
            attributes={"rate": "1 cube/minute"},
            description="不同实体的关系"
        )
        
        result = self.matcher.match(self.base_relation, different_entities_relation)
        
        self.assertFalse(result['matched'])
        self.assertLess(result['similarity_score'], 1.0)
        self.assertIn('entities', result['matching_details']['differences'])
        
    def test_different_attributes(self):
        """测试不同属性的关系"""
        different_attributes_relation = Relation(
            id="r5",
            type=RelationType.QUANTITATIVE,
            entities=[self.entity1, self.entity2],
            attributes={"rate": "2 cubes/minute"},  # 不同的属性值
            description="不同属性的关系"
        )
        
        result = self.matcher.match(self.base_relation, different_attributes_relation)
        
        self.assertFalse(result['matched'])
        self.assertLess(result['similarity_score'], 1.0)
        self.assertIn('attributes', result['matching_details']['differences'])
        
    def test_empty_relations(self):
        """测试空关系"""
        empty_relation1 = Relation(
            id="r6",
            type=RelationType.QUANTITATIVE,
            entities=[],
            attributes={},
            description="空关系1"
        )
        
        empty_relation2 = Relation(
            id="r7",
            type=RelationType.QUANTITATIVE,
            entities=[],
            attributes={},
            description="空关系2"
        )
        
        result = self.matcher.match(empty_relation1, empty_relation2)
        
        self.assertTrue(result['matched'])
        self.assertEqual(result['similarity_score'], 1.0)

if __name__ == '__main__':
    unittest.main()
