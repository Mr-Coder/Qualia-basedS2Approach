#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自然语言处理器
~~~~~~~~~~

这个模块负责对输入文本进行自然语言处理，包括分词、词性标注等。

Author: [Your Name]
Date: [Current Date]
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import spacy

# Import ProcessedText from proper location
from ..models.processed_text import ProcessedText
# from ..models.structures import FeatureSet, PatternMatch

logger = logging.getLogger(__name__)


class NLPProcessor:
    """自然语言处理器"""
    def __init__(self, config=None):
        """初始化自然语言处理器
        
        Args:
            config: 配置信息
        """
        # 加载spaCy的英文模型用于自然语言处理
        self.nlp = spacy.load("en_core_web_sm")
        self.config = config
        # 预定义的词性映射，用于简化词性标注
        self.pos_mapping = {
            'NOUN': 'n',   # 名词
            'VERB': 'v',   # 动词
            'NUM': 'm',    # 数字
            'SYM': 'q',    # 符号（可能是单位）
            'ADV': 'adv',  # 副词
            'ADJ': 'adj',  # 形容词
            'ADP': 'p',    # 介词
            'DET': 'det',  # 冠词
            'PRON': 'pron', # 代词
            'PUNCT': 'w'   # 标点
        }
        # 预定义的单位词列表
        self.unit_terms = ['cm³', 'cm', 'm³', 'm', 'L', 'l', 'mL', 'ml', 'kg', 'g', 's', 'second', 'seconds', 'minute', 'minutes', 'hour', 'hours']
        # 预定义的数量模式
        self.quantity_patterns = {
            'number_unit': r'(\d+(?:\.\d+)?)\s*([a-zA-Z一-龥]+)',
            'rate': r'(\d+(?:\.\d+)?)\s*([a-zA-Z一-龥]+/[a-zA-Z一-龥]+)',
            'percent': r'(\d+(?:\.\d+)?)%'
        }
    
    def analyze(self, problem_text: str) -> Dict:
        """分析问题文本，提取关键信息和问题类型"""
        # 处理文本
        doc = self.nlp(problem_text)
        
        # 提取问题类型
        problem_type = self._identify_problem_type(problem_text)
        
        # 提取数量和关系
        quantities = self._extract_quantities(problem_text)
        
        # 识别问题目标
        question_target = self._identify_question_target(doc)
        
        return {
            "problem_type": problem_type,
            "quantities": quantities,
            "question_target": question_target,
            "doc": doc  # 返回spaCy文档对象以供进一步分析
        }

    def _identify_problem_type(self, text: str) -> str:
        """识别问题类型
        
        Args:
            text: 问题文本
        
        Returns:
            str: 问题类型
        """
        # 这里可以添加if判断不同类型的启发式规则
        # 目前直接返回unknown
        return 'unknown'

    def process_text(self, text: str) -> Dict:
        """处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            ProcessedText: 处理后的文本结构
            
        Raises:
            ValueError: 当输入文本为空或None时
        """
        if not text:
            raise ValueError("输入文本不能为空")
            
        logger.info(f"[NLPProcessor] 输入: {text}")
        # 分词
        tokens = self._tokenize(text)
        
        # 词性标注
        pos_tags = self._pos_tag(tokens)
        
        # 依存关系分析
        dependencies = self._dependency_parse(tokens)

        # 提取特征
        features = self._extract_features(text, tokens, pos_tags)
        
        # 提取数值和单位
        values_and_units = self._extract_values_and_units(text)
        
        result = {
            'raw_text': text,
            'segmentation': tokens,
            'pos_tags': pos_tags,
            'dependencies': dependencies,
            'semantic_roles': {},
            'features': features,
            'values_and_units': values_and_units
        }
        logger.info(f"[NLPProcessor] 输出: {result}")
        return result

    def _extract_quantities(self, text: str) -> Dict:
        """从文本中提取数量信息"""
        quantities = {}
        for quantity_name, pattern in self.quantity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        value, unit = match[0], match[1]
                        if len(match) > 2:  # 如果有第三个元素（如速率单位的分母）
                            unit = f"{unit}/{match[2]}"
                    else:
                        value, unit = match, ""
                    quantities[quantity_name] = {
                        "value": float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value,
                        "unit": unit
                    }
        return quantities

    def _tokenize(self, text: str) -> List[str]:
        """分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        # 使用spaCy进行分词
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        
        # 处理特殊情况：数值和单位分开处理
        processed_tokens = []
        i = 0
        while i < len(tokens):
            # 当前词是数字，并且下一个词可能是单位
            if i < len(tokens) - 1 and tokens[i].isdigit() and tokens[i+1] in self.unit_terms:
                processed_tokens.append(tokens[i])
                processed_tokens.append(tokens[i+1])
                i += 2
            else:
                processed_tokens.append(tokens[i])
                i += 1
                
        return processed_tokens
        
    def _pos_tag(self, tokens: List[str]) -> List[str]:
        """词性标注
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[str]: 词性标注结果
        """
        # 使用spaCy重新处理文本以获取词性标注
        text = ' '.join(tokens)
        doc = self.nlp(text)
        
        # 使用简化的词性标注
        pos_tags = []
        for token in doc:
            # 映射spaCy的词性标签到简化的标签
            if token.pos_ in self.pos_mapping:
                pos_tag = self.pos_mapping[token.pos_]
            else:
                pos_tag = 'n'  # 默认为名词
            
            # 特殊处理数字
            if token.text.isdigit() or token.text.replace('.', '', 1).isdigit():
                pos_tag = 'm'
                
            # 特殊处理单位
            if token.text in self.unit_terms:
                pos_tag = 'q'
                
            pos_tags.append(pos_tag)
            
        return pos_tags
        
    def _dependency_parse(self, tokens: List[str]) -> List[Tuple[str, str, str]]:
        """依存关系分析
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[Tuple[str, str, str]]: 依存关系列表，每个元素为(head, relation, dependent)
        """
        # 使用spaCy获取依存关系
        text = ' '.join(tokens)
        doc = self.nlp(text)
        
        dependencies = []
        for token in doc:
            if token.dep_ != 'ROOT':  # 跳过根节点
                head = token.head.text
                relation = token.dep_
                dependent = token.text
                dependencies.append((head, relation, dependent))
                
        return dependencies

    def _extract_features(self, text: str, tokens: List[str], pos_tags: List[str]) -> Dict:
        """提取问题特征
        
        Args:
            text: 原始文本
            tokens: 分词结果
            pos_tags: 词性标注结果
            
        Returns:
            Dict: 问题特征集
        """
        features = {}
        
        # 1. 数学复杂度特征
        features['math_complexity'] = {
            'variable_count': len(self._extract_values_and_units(text)),
            'operation_indicators': self._detect_operations(text),
            'function_indicators': self._detect_functions(text)
        }
        
        # 2. 语言表达结构特征
        features['linguistic_structure'] = {
            'pos_distribution': self._analyze_pos_distribution(pos_tags),
            'pos_sequence': ','.join(pos_tags),
            'sentence_structure': self._analyze_sentence_structure(text)
        }
        
        # 3. 关系类型特征
        features['relation_type'] = {
            'explicit_relations': self._detect_explicit_relations(text),
            'implicit_relations': self._detect_implicit_relations(text),
            'dominant_type': self._determine_dominant_relation_type(text)
        }
        
        # 4. 问题领域特征
        features['domain_indicators'] = {
            'liquid_related': self._contains_domain_terms(text, 'liquid_related'),
            'motion_related': self._contains_domain_terms(text, 'motion_related'),
            'work_related': self._contains_domain_terms(text, 'work_related'),
            'growth_related': self._contains_domain_terms(text, 'growth_related')
        }
        
        # 5. 问题目标特征
        features['question_target'] = {
            'target_variable': self._identify_target_variable(text),
            'question_type': self._classify_question_type(text)
        }
        
        return features
    
    def _extract_values_and_units(self, text: str) -> Dict[str, Dict[str, Any]]:
        """增强版：结合依存分析、上下文和NER，将数值与最近的名词/实体/上下文绑定，提升准确率"""
        doc = self.nlp(text)
        values_and_units = {}
        used_indices = set()
        for token in doc:
            # 找到数值
            if token.like_num and token.i not in used_indices:
                # 查找最近的单位
                unit = ""
                next_token = token.nbor(1) if token.i+1 < len(doc) else None
                if next_token and (next_token.text in self.unit_terms or next_token.pos_ == "NOUN"):
                    unit = next_token.text
                    used_indices.add(next_token.i)
                # 查找最近的名词实体
                entity = ""
                # 1. 依存树找修饰名词
                for child in token.children:
                    if child.pos_ == "NOUN":
                        entity = child.text
                        break
                # 2. 向左找最近名词
                if not entity:
                    for left in reversed(list(doc[:token.i])):
                        if left.pos_ == "NOUN":
                            entity = left.text
                            break
                # 3. NER补充
                if not entity:
                    for ent in doc.ents:
                        if token.i >= ent.start and token.i < ent.end:
                            entity = ent.text
                            break
                # 4. 上下文补充：如前一句的主语/名词
                if not entity:
                    for sent in doc.sents:
                        if token.i >= sent.start and token.i < sent.end:
                            for t in reversed(list(doc[sent.start:token.i])):
                                if t.pos_ == "NOUN":
                                    entity = t.text
                                    break
                # 变量名
                var_name = f"{entity}_{token.text}{unit}".strip("_")
                values_and_units[var_name] = {
                    "value": float(token.text) if token.text.replace('.', '', 1).isdigit() else token.text,
                    "unit": unit,
                    "entity": entity
                }
                used_indices.add(token.i)
        return values_and_units
    
    def _count_potential_variables(self, text: str, tokens: List[str]) -> int:
        """计算潜在变量数量"""
        return 0
    
    def _detect_operations(self, text: str) -> List[str]:
        """检测操作符"""
        return []
    
    def _detect_functions(self, text: str) -> List[str]:
        """检测函数调用"""
        return []
    
    def _analyze_pos_distribution(self, pos_tags: List[str]) -> Dict[str, int]:
        """分析词性分布"""
        pos_distribution = {}
        for pos in pos_tags:
            if pos in pos_distribution:
                pos_distribution[pos] += 1
            else:
                pos_distribution[pos] = 1
        return pos_distribution
    
    def _analyze_sentence_structure(self, text: str) -> str:
        """分析句子结构"""
        return ""
    
    def _detect_explicit_relations(self, text: str) -> List[str]:
        """检测显式关系"""
        return []
    
    def _detect_implicit_relations(self, text: str) -> List[str]:
        """检测隐式关系"""
        return []
    
    def _determine_dominant_relation_type(self, text: str) -> str:
        """确定主导关系类型"""
        return "unknown"
    
    def _contains_domain_terms(self, text: str, domain: str) -> bool:
        """检查是否包含领域特定术语"""
        domain_terms = {
            'liquid_related': ['water', 'liquid', 'tank', 'volume', 'flow', 'leak', 'fill'],
            'motion_related': ['speed', 'distance', 'time', 'travel', 'meet', 'train', 'car'],
            'work_related': ['work', 'task', 'complete', 'efficiency', 'rate', 'worker'],
            'growth_related': ['increase', 'decrease', 'grow', 'rate', 'percent', 'interest', 'investment']
        }
        
        if domain in domain_terms:
            terms = domain_terms[domain]
            return any(term.lower() in text.lower() for term in terms)
            
        return False
    
    def _identify_target_variable(self, text: str) -> str:
        """增强版：结合问句、关键词、依存结构和上下文，提升目标变量识别率"""
        doc = self.nlp(text)
        # 1. 优先找问句中的 how/what/which/求/多少/多久/多长时间/需要多少时间
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if sent.text.strip().endswith("?") or any(q in sent.text for q in ["求", "多少", "what", "how many", "how much", "多久", "多长时间", "需要多少时间", "需要多长时间", "how long"]):
                # 英文 how long/how much time/how many minutes/hours
                if any(kw in sent_text for kw in ["how long", "how much time", "how many minutes", "how many hours", "需要多少时间", "多长时间", "多久"]):
                    return "time"
                # 中文"多久""多长时间""需要多少时间"
                if any(kw in sent_text for kw in ["多久", "多长时间", "需要多少时间"]):
                    return "time"
                # 找下一个名词
                for token in sent:
                    if token.text.lower() in ["how", "what", "which", "求", "多少"]:
                        for right in sent[token.i+1:]:
                            if right.pos_ == "NOUN":
                                return right.text
                        for right in sent[token.i+1:]:
                            if right.pos_ in ("NUM", "SYM", "ADJ"):
                                return right.text
                for token in sent:
                    if token.dep_ in ("dobj", "attr", "nsubj") and token.pos_ == "NOUN":
                        return token.text
        # 2. 结合依存结构：找 root 的宾语/补语
        for token in doc:
            if token.dep_ in ("dobj", "attr", "nsubj") and token.pos_ == "NOUN":
                return token.text
        # 3. 结合上下文：找全文最后一个名词
        nouns = [t.text for t in doc if t.pos_ == "NOUN"]
        if nouns:
            return nouns[-1]
        # 4. 兜底：找 how/what/which 后的 token
        for token in doc:
            if token.text.lower() in ["how", "what", "which"]:
                next_token = token.nbor(1) if token.i+1 < len(doc) else None
                if next_token:
                    return next_token.text
        return "unknown"
    
    def _classify_question_type(self, text: str) -> str:
        """分类问题类型"""
        return "unknown"
    
    def _identify_question_target(self, doc) -> str:
        """识别问题目标"""
        for sent in doc.sents:
            if sent.text.strip().endswith("?"):
                # 简单判断：看问句包含哪些关键词
                if any(token.text.lower() in ["how", "what", "when"] for token in sent):
                    if any(token.text.lower() in ["long", "time", "minutes", "hours"] for token in sent):
                        return "time"
                    elif any(token.text.lower() in ["many", "much", "value", "amount"] for token in sent):
                        return "value"
                    elif any(token.text.lower() in ["rate", "speed", "velocity"] for token in sent):
                        return "rate"
        
        return "unknown"

    def load_examples_from_json(self, json_path: str = None):
        """批量读取examples/problems.json中的问题并NLP处理，返回ProcessedText对象列表"""
        if json_path is None:
            # 默认路径
            json_path = str(Path(__file__).parent.parent / 'examples' / 'problems.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        problems = data.get('problems', [])
        processed_list = []
        for prob in problems:
            text = prob.get('text', '')
            if text:
                processed = self.process_text(text)
                processed_list.append(processed)
        return processed_list

    def save_processed_examples_to_file(self, output_path: str, json_path: str = None):
        """
        批量处理examples/problems.json中的问题，并将所有ProcessedText对象的结构化结果保存到指定文件（JSON格式）。
        Args:
            output_path: 输出文件路径
            json_path: 输入的problems.json路径（可选）
        """
        processed_list = self.load_examples_from_json(json_path)
        # 只保存结构化的主要字段，避免对象不可序列化
        result = []
        for p in processed_list:
            result.append({
                'raw_text': p.raw_text,
                'segmentation': p.segmentation,
                'pos_tags': p.pos_tags,
                'dependencies': p.dependencies,
                'semantic_roles': p.semantic_roles,
                'cleaned_text': p.cleaned_text,
                'tokens': p.tokens,
                'ner_tags': p.ner_tags
            })
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已保存{len(result)}条结构化NLP结果到 {output_path}")


def analyze_problem(problem_text):
    analyzer = NLPProcessor()
    analysis = analyzer.analyze(problem_text)
    
    print("问题类型:", analysis["problem_type"])
    print("提取的数量:")
    for name, info in analysis["quantities"].items():
        print(f"  - {name}: {info['value']} {info['unit']}")
    print("问题目标:", analysis["question_target"])
    
    return analysis