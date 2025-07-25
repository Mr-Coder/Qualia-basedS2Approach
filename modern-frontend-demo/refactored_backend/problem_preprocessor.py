#!/usr/bin/env python3
"""
问题预处理模块
负责文本清理、实体识别、复杂度评估
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体数据结构"""
    id: str
    name: str
    entity_type: str
    properties: List[str]
    confidence: float = 0.0

@dataclass
class ProcessedProblem:
    """预处理后的问题数据结构"""
    original_text: str
    cleaned_text: str
    entities: List[Entity]
    numbers: List[float]
    complexity_score: float
    keywords: List[str]
    problem_type: str

class ProblemPreprocessor:
    """问题预处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 复杂关键词词典
        self.complex_keywords = {
            '比例': 0.4, '百分比': 0.4, '倍数': 0.3, '平均': 0.3,
            '面积': 0.5, '体积': 0.5, '速度': 0.4, '时间': 0.3,
            '距离': 0.3, '工程': 0.6, '行程': 0.5, '浓度': 0.6,
            '利润': 0.4, '折扣': 0.4, '增长': 0.3, '减少': 0.3
        }
        
        # 简单运算关键词
        self.simple_keywords = {
            '一共': 0.1, '总共': 0.1, '合计': 0.1, '总数': 0.1,
            '总计': 0.1, '共有': 0.1, '总和': 0.1, '相加': 0.1,
            '加起来': 0.1, '多少': 0.1
        }
        
        # 人名词典
        self.person_names = [
            '小明', '小红', '小华', '小李', '小王', '小张', '小刘', '小陈',
            '张三', '李四', '王五', '赵六', '孙七', '周八', '吴九', '郑十',
            '甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'
        ]
        
        # 物品词典
        self.object_names = [
            '苹果', '橘子', '香蕉', '梨子', '桃子', '葡萄', '西瓜', '草莓',
            '书', '笔', '本子', '铅笔', '橡皮', '尺子', '文具', '玩具',
            '球', '车', '花', '树', '房子', '桌子', '椅子', '电脑',
            '钱', '元', '角', '分', '块', '毛'
        ]

    def preprocess(self, problem_text: str) -> ProcessedProblem:
        """
        主要的预处理入口
        
        Args:
            problem_text: 原始问题文本
            
        Returns:
            ProcessedProblem: 预处理后的问题数据
        """
        try:
            self.logger.info(f"开始预处理问题: {problem_text[:50]}...")
            
            # 1. 文本清理
            cleaned_text = self._clean_text(problem_text)
            
            # 2. 实体识别
            entities = self._extract_entities(cleaned_text)
            
            # 3. 数字提取
            numbers = self._extract_numbers(cleaned_text)
            
            # 4. 关键词提取
            keywords = self._extract_keywords(cleaned_text)
            
            # 5. 复杂度评估
            complexity_score = self._estimate_complexity(cleaned_text, entities, numbers, keywords)
            
            # 6. 问题类型判断
            problem_type = self._classify_problem_type(keywords, entities, numbers)
            
            processed_problem = ProcessedProblem(
                original_text=problem_text,
                cleaned_text=cleaned_text,
                entities=entities,
                numbers=numbers,
                complexity_score=complexity_score,
                keywords=keywords,
                problem_type=problem_type
            )
            
            self.logger.info(f"预处理完成: 复杂度={complexity_score:.2f}, 类型={problem_type}")
            return processed_problem
            
        except Exception as e:
            self.logger.error(f"预处理失败: {e}")
            # 返回基础的处理结果
            return ProcessedProblem(
                original_text=problem_text,
                cleaned_text=problem_text,
                entities=[],
                numbers=[],
                complexity_score=0.5,
                keywords=[],
                problem_type="unknown"
            )

    def _clean_text(self, text: str) -> str:
        """文本清理"""
        # 去除多余空格
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # 标准化标点符号
        cleaned = cleaned.replace('，', ',').replace('。', '.').replace('？', '?')
        
        # 移除特殊字符但保留中文、数字、基本标点
        cleaned = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z,.\?!，。？！、：；]', ' ', cleaned)
        
        return cleaned.strip()

    def _extract_entities(self, text: str) -> List[Entity]:
        """实体识别"""
        entities = []
        entity_id = 1
        
        # 识别人名
        for person in self.person_names:
            if person in text:
                entity = Entity(
                    id=f"person_{entity_id}",
                    name=person,
                    entity_type="person",
                    properties=["human", "subject"],
                    confidence=0.95
                )
                entities.append(entity)
                entity_id += 1
        
        # 识别物品
        for obj in self.object_names:
            if obj in text:
                entity = Entity(
                    id=f"object_{entity_id}",
                    name=obj,
                    entity_type="object",
                    properties=["countable", "physical"],
                    confidence=0.90
                )
                entities.append(entity)
                entity_id += 1
        
        # 识别数量概念
        numbers = re.findall(r'\d+', text)
        for i, num in enumerate(numbers):
            entity = Entity(
                id=f"number_{i+1}",
                name=num,
                entity_type="number",
                properties=["quantity", "mathematical"],
                confidence=0.98
            )
            entities.append(entity)
        
        return entities

    def _extract_numbers(self, text: str) -> List[float]:
        """提取数字"""
        # 提取整数和小数
        number_pattern = r'\d+\.?\d*'
        numbers = re.findall(number_pattern, text)
        return [float(num) for num in numbers if num]

    def _extract_keywords(self, text: str) -> List[str]:
        """关键词提取"""
        keywords = []
        
        # 检查复杂关键词
        for keyword in self.complex_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        # 检查简单关键词
        for keyword in self.simple_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        # 检查运算关键词
        operation_keywords = ['加', '减', '乘', '除', '求', '计算', '多少']
        for keyword in operation_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return list(set(keywords))  # 去重

    def _estimate_complexity(self, text: str, entities: List[Entity], 
                           numbers: List[float], keywords: List[str]) -> float:
        """复杂度评估"""
        complexity = 0.0
        
        # 基于文本长度
        text_complexity = min(len(text) / 200, 0.3)
        complexity += text_complexity
        
        # 基于数字数量
        number_complexity = min(len(numbers) / 5, 0.3)
        complexity += number_complexity
        
        # 基于关键词复杂度
        keyword_complexity = 0.0
        for keyword in keywords:
            if keyword in self.complex_keywords:
                keyword_complexity += self.complex_keywords[keyword]
            elif keyword in self.simple_keywords:
                keyword_complexity += self.simple_keywords[keyword]
        keyword_complexity = min(keyword_complexity, 0.4)
        complexity += keyword_complexity
        
        # 基于实体关系密度
        if len(entities) > 0:
            relation_density = min(len(entities) * 0.1, 0.2)
            complexity += relation_density
        
        return min(complexity, 1.0)

    def _classify_problem_type(self, keywords: List[str], entities: List[Entity], 
                             numbers: List[float]) -> str:
        """问题类型分类"""
        
        # 检查是否是简单算术题
        simple_ops = ['一共', '总共', '合计', '总数', '加起来']
        if any(kw in keywords for kw in simple_ops) and len(numbers) <= 3:
            return "simple_arithmetic"
        
        # 检查是否是应用题
        complex_kws = ['比例', '百分比', '速度', '时间', '距离', '工程']
        if any(kw in keywords for kw in complex_kws):
            return "application_problem"
        
        # 检查是否是几何题
        geo_kws = ['面积', '体积', '周长', '长度', '宽度', '高度']
        if any(kw in keywords for kw in geo_kws):
            return "geometry_problem"
        
        # 检查是否是概率统计题
        stat_kws = ['平均', '概率', '统计', '可能性']
        if any(kw in keywords for kw in stat_kws):
            return "statistics_problem"
        
        # 默认分类
        if len(numbers) > 3 or len(entities) > 5:
            return "complex_problem"
        else:
            return "basic_problem"

    def get_problem_summary(self, processed_problem: ProcessedProblem) -> Dict[str, Any]:
        """获取问题摘要信息"""
        return {
            "complexity_level": "low" if processed_problem.complexity_score < 0.3 
                               else "medium" if processed_problem.complexity_score < 0.7 
                               else "high",
            "entity_count": len(processed_problem.entities),
            "number_count": len(processed_problem.numbers),
            "keyword_count": len(processed_problem.keywords),
            "problem_type": processed_problem.problem_type,
            "recommended_engine": self._recommend_engine(processed_problem)
        }

    def _recommend_engine(self, processed_problem: ProcessedProblem) -> str:
        """推荐推理引擎"""
        if processed_problem.complexity_score < 0.3:
            return "simple_engine"
        elif processed_problem.complexity_score < 0.7:
            return "hybrid_engine"
        else:
            return "advanced_engine"

# 测试函数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = ProblemPreprocessor()
    
    # 测试用例
    test_problems = [
        "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
        "一个长方形的长是8米，宽是6米，求这个长方形的面积。",
        "张三以每小时60公里的速度开车，行驶了2小时，他总共行驶了多少公里？"
    ]
    
    for problem in test_problems:
        print(f"\n{'='*50}")
        print(f"原问题: {problem}")
        processed = preprocessor.preprocess(problem)
        summary = preprocessor.get_problem_summary(processed)
        
        print(f"清理后: {processed.cleaned_text}")
        print(f"实体: {[e.name for e in processed.entities]}")
        print(f"数字: {processed.numbers}")
        print(f"关键词: {processed.keywords}")
        print(f"复杂度: {processed.complexity_score:.2f}")
        print(f"类型: {processed.problem_type}")
        print(f"推荐引擎: {summary['recommended_engine']}")