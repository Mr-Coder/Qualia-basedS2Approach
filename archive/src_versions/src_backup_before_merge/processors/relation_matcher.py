import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass
class MatchedPattern:
    """匹配到的模式"""
    pattern_id: str
    matches: List[Tuple]
    pattern_type: str

class RelationMatcher:
    """关系匹配器"""
    
    def __init__(self, pattern_path: str = None, problem_type_path: str = None):
        """初始化关系匹配器"""
        self.logger = logging.getLogger(__name__)
        if pattern_path is None:
            pattern_path = str(Path(__file__).parent.parent / 'models' / 'pattern.json')
        self.pattern_path = pattern_path
        self.problem_type_path = problem_type_path
        self.patterns = self._load_patterns()
        self.problem_type_patterns = self._init_problem_type_patterns()
    
    def _normalize_patterns(self, patterns):
        # 如果是 dict 且有 pattern_groups，递归提取所有 pattern
        if isinstance(patterns, dict) and 'pattern_groups' in patterns:
            all_patterns = []
            def collect(group):
                if isinstance(group, list):
                    for p in group:
                        if isinstance(p, dict):
                            all_patterns.append(p)
                elif isinstance(group, dict):
                    for v in group.values():
                        collect(v)
            collect(patterns['pattern_groups'])
            return all_patterns
        # 如果本身就是 list，直接用
        elif isinstance(patterns, list):
            return patterns
        else:
            return []

    def _load_patterns(self) -> list:
        with open(self.pattern_path, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        return self._normalize_patterns(patterns)
    
    def _init_problem_type_patterns(self) -> Dict[str, List[str]]:
        # 直接遍历 self.patterns，收集 scene->id 映射，只处理 dict 类型
        scene_to_ids = defaultdict(list)
        for pattern in self.patterns:
            if isinstance(pattern, dict):
                scene = pattern.get("scene")
                pid = pattern.get("id")
                if scene and pid:
                    scene_to_ids[scene].append(str(pid))
        return dict(scene_to_ids)
    
    def _init_patterns(self) -> List[Dict]:
        """初始化关系模式"""
        return [
            # 模式1：名词-数字-单位模式 （用于提取物理量）
            {
                'id': '1',
                'pattern': 'n,m,q',
                'relation_template': 'a=b*c',
                'var_slot_val': 'n,m,q',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2'},
                'description': '提取冰块体积，识别形如"volume of 200 cm³"的表达'
            },

            # 模式2：动词-数字-单位模式（用于提取状态或变化）
            {
                'id': '2',
                'pattern': 'v,m,q',
                'relation_template': 'a=b*c',
                'var_slot_val': 'v,m,q',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2'},
                'description': '提取初始水量，识别形如"containing 5L"的表达'
            },

            # 模式3：名词-数字-单位-名词模式 （用于提取速率关系）
            {
                'id': '3',
                'pattern': 'n,m,q,n',
                'relation_template': 'a=b*c/d',
                'var_slot_val': 'n,m,q,n',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2', 'd': '3'},
                'description': '提取漏水速率，识别形如"rate of 2 mL per second"的表达'
            },
            
            # 模式4：动词-数字-单位模式 （用于提取目标值）
            {
                'id': '4',
                'pattern': 'v,m,q',
                'relation_template': 'a=b*c',
                'var_slot_val': 'v,m,q',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2'},
                'description': '提取目标水量，识别形如"rise to 9L"的表达'
            },

            # 模式5：名词-属性-数字，名词-属性-名词模式 （用于提取复杂关系）
            {
                'id': '5',
                'pattern': 'n->ATT->m,n->ATT->n',
                'relation_template': 'a=b/c',
                'var_slot_val': 'n,m,n',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2'},
                'description': '提取冰块加入速率，识别形如"rate of one cube per minute"的复杂表达'
            },
            
            # 模式6：动词-动词-主谓关系（用于提取变化关系）
            {
                'id': '6',
                'pattern': 'n->ATT->n,v->SBV->q',
                'relation_template': 'a=b-c',
                'var_slot_val': 'n,q,q',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2'},
                'description': '提取水量变化关系，识别形如"water level...rise to 9L"的复杂表达'
            },
            
            # 模式7：形容词-动词-主谓关系（用于提取时间计算关系）
            {
                'id': '7',
                'pattern': 'adv->ADV->adj,v->SBV->v',
                'relation_template': 'a=(b-c)/(d*e-f*60)',
                'var_slot_val': 'adv,v,v',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2', 'd': '3', 'e': '4', 'f': '5'},
                'description': '提取时间计算关系，识别形如"How long will it take for...to rise"的复杂问句'
            },

            # 模式8：工作效率模式
            {
                'id': '8',
                'pattern': 'n,v,m,n',
                'relation_template': 'a=1/b',
                'var_slot_val': 'n,v,m,n',
                'var_slot_index': {'a': '0', 'b': '2'},
                'description': '识别形如"Worker A can complete a task in 6 hours"的表达'
            },

            # 模式9：合作效率模式
            {
                'id': '9',
                'pattern': 'n,v,adv,v',
                'relation_template': 'a=b+c',
                'var_slot_val': 'n,v,adv,v',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '3'},
                'description': '识别形如"If they work together"的表达'
            },

            # 模式10：增长率模式
            {
                'id': '10',
                'pattern': 'n,v,prep,m,n',
                'relation_template': 'a=b*(1+c)^d',
                'var_slot_val': 'n,v,m,n',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2', 'd': '3'},
                'description': '识别形如"investment grows at an annual rate of 8%"的表达'
            },

            # 模式11：运动相遇模式
            {
                'id': '11',
                'pattern': 'n,v,prep,m,q',
                'relation_template': 'a=b/(c+d)',
                'var_slot_val': 'n,v,m,q',
                'var_slot_index': {'a': '0', 'b': '2', 'c': '3', 'd': '4'},
                'description': '识别形如"trains start from stations that are 300 kilometers apart"的表达'
            },

            # 模式12：液体增减速率模式
            {
                'id': '12',
                'pattern': 'v,m,n,prep,m,q',
                'relation_template': 'a=b*c-d',
                'var_slot_val': 'v,m,n,m,q',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2', 'd': '3'},
                'description': '识别形如"water flows through a pipe at a rate of 10 liters per second"的表达'
            },

            # 模式13：时间计算模式
            {
                'id': '13',
                'pattern': 'adv,v,n,v,prep,m,q',
                'relation_template': 'a=(b-c)/d',
                'var_slot_val': 'adv,n,m,q',
                'var_slot_index': {'a': '0', 'b': '2', 'c': '1', 'd': '3'},
                'description': '识别形如"多长时间后水池的水量会达到200升"的问句'
            },
            
            # 模式14：单位转换模式
            {
                'id': '14',
                'pattern': 'm,q1,prep,q2',
                'relation_template': 'a=b*c',
                'var_slot_val': 'm,q1,q2',
                'var_slot_index': {'a': '0', 'b': '1', 'c': '2'},
                'description': '识别形如"1000米等于多少公里"的问句'
            }
        ]
    
    def match_patterns(self, tokens, pos_tags, text, problem_type=None, dependencies=None, scene=None, reasoning_type=None) -> list:
        """增强：支持 scene/reasoning_type 优先筛选"""
        matches = []
        # scene/reasoning_type 优先筛选
        relevant_patterns = []
        if scene and reasoning_type:
            relevant_patterns = [p for p in self.patterns if p.get('scene') == scene and p.get('reasoning_type') == reasoning_type]
        if not relevant_patterns and scene:
            relevant_patterns = [p for p in self.patterns if p.get('scene') == scene]
        if not relevant_patterns:
            relevant_patterns = self.patterns
        # 评估每个模式的匹配度
        for pattern in relevant_patterns:
            score = self._evaluate_pattern_match(pattern, tokens, pos_tags, dependencies)
            matches.append((pattern, score))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _evaluate_pattern_match(self, pattern, tokens, pos_tags, dependencies=None):
        score = 0
        pattern_elements = pattern["pattern"].split(",")
        # 1. 检查语法结构匹配
        if "->" in pattern["pattern"]:
            if dependencies is not None:
                mapping = self._match_dependency_pattern(pattern["pattern"], dependencies, tokens, pos_tags)
                score += len(mapping)
            else:
                score += self._check_dependency_pattern(pattern["pattern"], tokens, pos_tags)
        else:
            score += self._check_sequence_pattern(pattern_elements, pos_tags)
        return score
    
    def _check_sequence_pattern(self, pattern_elements, pos_tags):
        """检查简单序列模式的匹配度"""
        score = 0
        pos_counts = {"n": 0, "v": 0, "m": 0, "q": 0, "adv": 0, "adj": 0, "prep": 0}
        
        # 统计文档中的词性
        for pos in pos_tags:
            if pos in pos_counts:
                pos_counts[pos] += 1
                
        # 检查模式中的每个元素
        for element in pattern_elements:
            if element in pos_counts and pos_counts[element] > 0:
                score += 1
                
        return score
    
    def _check_dependency_pattern(self, pattern, tokens, pos_tags):
        """检查依存关系模式的匹配度
        
        简化版实现，仅检查序列中是否包含所需的词性
        """
        score = 0
        
        # 解析依存关系模式中的词性
        pos_types = set()
        for part in pattern.split(","):
            if "->" in part:
                parts = part.split("->")
                pos_types.add(parts[0])
                if len(parts) > 1:
                    rel_target = parts[1]
                    if "->" in rel_target:
                        pos_types.add(rel_target.split("->")[1])
        
        # 检查是否所有所需词性都存在
        for pos_type in pos_types:
            if pos_type in pos_tags:
                score += 1
                
        return score
    
    def _match_pos(self, pos, pattern_pos):
        """匹配词性标签和模式中的词性类型"""
        return pos == pattern_pos
    
    def _match_dependency_pattern(self, pattern_str, dependencies, tokens, pos_tags):
        """依存结构 pattern 匹配，返回 pattern 变量与文本实体的对齐映射"""
        mapping = {}
        for part in pattern_str.split(','):
            if '->' in part:
                try:
                    src_type, rel, tgt_type = part.split('->')
                except Exception:
                    continue
                for head, dep_rel, dep in dependencies:
                    head_idx = tokens.index(head) if head in tokens else -1
                    dep_idx = tokens.index(dep) if dep in tokens else -1
                    if dep_rel.lower() == rel.lower() and head_idx >= 0 and dep_idx >= 0:
                        if pos_tags[head_idx].startswith(src_type[0]) and pos_tags[dep_idx].startswith(tgt_type[0]):
                            mapping[src_type] = head
                            mapping[tgt_type] = dep
        return mapping
    
    def match_problem_patterns(self, problem_analysis, tokens, pos_tags, text=None):
        """匹配问题模式
        
        Args:
            problem_analysis: 问题分析结果
            tokens: 分词结果
            pos_tags: 词性标注
            text: 原始文本
            
        Returns:
            List[Tuple[Dict, float]]: 匹配到的模式及其分数
        """
        matches = self.match_patterns(tokens, pos_tags, text or "")
        
        self.logger.info("\n模式匹配结果:")
        for i, (pattern, score) in enumerate(matches[:3]):  # 显示前3个最佳匹配
            self.logger.info(f"{i+1}. 模式ID: {pattern['id']}, 匹配分数: {score}")
            self.logger.info(f"   描述: {pattern['description']}")
            self.logger.info(f"   模式: {pattern['pattern']}")
            self.logger.info(f"   关系模板: {pattern['relation_template']}")
            self.logger.info("")
        
        return matches




# 模式分类体系

# 基于模式的特征，建立了以下多维度分类体系：

# 维度 1：数学关系复杂度
# 1.基础运算模式（CI < 5）
#    - 特点：简单的加减乘除关系。
#    - 示例：模式 1、模式 5、模式 14。

# 2.中等复杂度模式（5 ≤ CI < 10）
#    - 特点：包含复合运算的关系。
#    - 示例：模式 11、模式 12、模式 13。

# 3.高复杂度模式（CI ≥ 10）
#    - 特点：包含指数、对数等高级运算。
#    - 示例：模式 10。


# 维度 2：语言表达结构
# 1.对象描述型模式（SPV[0] > SPV[1]）
#    - 特点：以名词为主导的模式。
#    - 应用：描述对象属性的问题。

# 2.过程描述型模式（SPV[1] ≥ SPV[0]）
#    - 特点：以动词为主导的模式。
#    - 应用：描述动作和变化的问题。

# 3.关系描述型模式（SPV[4] > 1）
#    - 特点：包含多个介词的模式。
#    - 应用：描述对象间关系的问题。



# 维度 3：变量间的依赖关系
# 1.简单依赖模式
#    - 特点：变量间存在直接的一对一关系。
#    - 示例：`a = b c`（简单乘法关系）。
#    - 应用：基础的数量关系问题。

# 2.复合依赖模式
#    - 特点：变量间存在多重依赖关系。
#    - 示例：`a = (b - c) / d`（模式 13，时间计算模式）。
#    - 应用：多步骤计算问题。

# 3.递归依赖模式
#    - 特点：变量定义中包含自身或循环依赖。
#    - 示例：模式 10 中的复利计算。
#    - 应用：迭代计算和递归问题。



# 模式分类示例

# 以下是几个模式的分类示例：

# 1.模式 8：`a = 1 / b`
#    CI = 3 （1 个变量 + 2 个运算符）。
#    SPV = `[2, 1, 0, 0, 0, 1]`。
#    RTI = 1 （分数）。
#    分类：基础运算模式 + 对象描述型模式。

# 2.模式 10：`a = b (1 + c)^d`
#    CI = 11 （3 个变量 + 4 个运算符 + 1 层嵌套）。
#    SPV = `[2, 1, 0, 0, 1, 1]`。
#    RTI = 2 （指数）。
#    分类：高复杂度模式 + 对象描述型模式。

# 3.模式 12：`a = b c - d`
#    CI = 7 （3 个变量 + 2 个运算符）。
#    SPV = `[1, 1, 0, 0, 1, 2]`。
#    RTI = 0 （线性）。
#    分类：中等复杂度模式 + 过程描述型模式。
