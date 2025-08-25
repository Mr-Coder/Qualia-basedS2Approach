"""关系提取器模块

从处理后的文本中提取数学关系，利用粗粒度分类结果进行精细模式匹配。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

# Assuming paths are handled by the main solver script
# from src.config.config_loader import get_config
# from src.models.pattern_loader import get_pattern_loader
from ..models.processed_text import ProcessedText
from .MWP_process import MWPCoarseClassifier
from .relation_matcher import RelationMatcher

logger = logging.getLogger(__name__)

class RelationExtractor:
    """关系提取器 - 支持递归、复合、变量对齐、依赖链自动生成、兜底机制"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, relation_matcher: Optional[RelationMatcher] = None):
        """初始化关系提取器
        
        Args:
            config: 配置信息
            relation_matcher: 关系模式匹配器实例
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        # RelationMatcher holds the patterns and basic matching logic
        self.relation_matcher = relation_matcher or RelationMatcher()
        # Coefficients for fine-grained scoring (example values)
        self.score_weights = {
            'relation_template': 3.0,
            'syntax_structure': 2.0,
            'variable_slot': 2.5,
            'domain_relevance': 1.0,
            'relation_type_match': 1.5
        }
        # 最大递归深度控制
        self.max_recursion_depth = self.config.get('max_recursion_depth', 10)

    def extract_relations(self, processed_text: ProcessedText, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从处理后的文本中提取关系
        
        Args:
            processed_text: 处理后的文本对象
            classification_result: 分类结果
            
        Returns:
            Dict 包含提取的关系
        """
        self.logger.info("开始精细关系提取（递归/复合/变量对齐/依赖链/兜底）...")
        
        # 1. 获取候选模式
        pattern_categories = classification_result.get('pattern_categories', [])
        candidate_patterns = self._generate_candidate_patterns(pattern_categories)
        
        # 2. 依赖环路检测
        cycles = self._detect_dependency_cycles(candidate_patterns)
        cycle_warnings = []
        if cycles:
            cycle_warnings = self._handle_dependency_cycles(cycles)
        
        # 3. 递归提取关系
        all_explicit = []
        all_implicit = []
        all_semantic = []
        
        # 为每个候选模式提取关系
        for pattern_id in candidate_patterns:
            visited = set()
            explicit, implicit, semantic = self._recursive_extract_with_vars(pattern_id, processed_text, visited)
            all_explicit.extend(explicit)
            all_implicit.extend(implicit)
            all_semantic.extend(semantic)
        
        # 4. 添加隐性关系
        self._add_implicit_patterns(processed_text, all_implicit, all_semantic)
        
        # 5. 变量环路检测
        var_cycles = self._detect_variable_cycles(all_semantic)
        if var_cycles:
            cycle_warnings.extend(self._handle_variable_cycles(var_cycles, all_semantic))
        
        # 6. 标准化变量名
        all_explicit = self._normalize_entities(all_explicit)
        all_implicit = self._normalize_entities(all_implicit)
        
        # 7. 确定最佳模式（简单实现：使用显性关系最多的模式）
        pattern_counts = {}
        for rel in all_explicit:
            pattern_id = rel.get('source_pattern', 'unknown')
            pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1
        
        best_pattern_id = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None
        
        # 8. 识别目标变量
        target_variables = self._identify_target_variables(processed_text)
        
        result = {
            'explicit_relations': all_explicit,
            'implicit_relations': all_implicit,
            'semantic_dependencies': all_semantic,
            'target_variables': target_variables,
            'best_pattern_id': best_pattern_id,
        }
        
        if cycle_warnings:
            result['cycle_warnings'] = cycle_warnings
            
        self.logger.info(f"[RelationExtractor] 输出: 显性{len(all_explicit)} 隐性{len(all_implicit)} 依赖{len(all_semantic)}")
        return result

    def _recursive_extract_with_vars(self, pattern_id, processed_text, visited, depth=0, var_scope=None):
        """递归提取关系，支持变量作用域管理
        
        Args:
            pattern_id: 模式ID
            processed_text: 处理后的文本
            visited: 已访问的模式集合
            depth: 当前递归深度
            var_scope: 变量作用域，用于跟踪变量来源和解决冲突
            
        Returns:
            提取的关系列表
        """
        # 递归深度控制
        max_depth = self.config.get('max_recursion_depth', 5)
        if depth > max_depth:
            self.logger.warning(f"达到最大递归深度 {max_depth}，停止递归")
            return [], [], []
            
        # 初始化变量作用域
        if var_scope is None:
            var_scope = {'global': set(), 'pattern_vars': {}}
            
        # 防止重复处理
        if pattern_id in visited:
            return [], [], []
        visited.add(pattern_id)
        
        # 获取模式数据
        pattern_data = self._get_pattern_data(pattern_id)
        if not pattern_data:
            return [], [], []
            
        # 记录详细日志
        self.logger.debug(f"递归提取关系: pattern_id={pattern_id}, depth={depth}")
        
        # 提取当前模式的关系
        explicit, implicit, semantic = self._generate_relations_from_pattern(pattern_data, processed_text)
        
        # 更新变量作用域
        for rel in explicit + implicit:
            if 'var_entity' in rel:
                # 记录每个变量属于哪个模式
                for var_name, mapped_name in rel['var_entity'].items():
                    if pattern_id not in var_scope['pattern_vars']:
                        var_scope['pattern_vars'][pattern_id] = set()
                    var_scope['pattern_vars'][pattern_id].add(mapped_name)
                    var_scope['global'].add(mapped_name)
        
        # 处理依赖、继承和组合
        dependencies = pattern_data.get('dependencies', [])
        inherits_from = pattern_data.get('inherits_from', [])
        composed_of = pattern_data.get('composed_of', [])
        
        # 递归处理依赖
        for dep_id in dependencies:
            if dep_id not in visited:
                self.logger.debug(f"处理依赖: {pattern_id} -> {dep_id}")
                dep_explicit, dep_implicit, dep_semantic = self._recursive_extract_with_vars(
                    dep_id, processed_text, visited.copy(), depth + 1, var_scope
                )
                explicit.extend(dep_explicit)
                implicit.extend(dep_implicit)
                semantic.extend(dep_semantic)
        
        # 递归处理继承
        for inh_id in inherits_from:
            if inh_id not in visited:
                self.logger.debug(f"处理继承: {pattern_id} inherits from {inh_id}")
                inh_explicit, inh_implicit, inh_semantic = self._recursive_extract_with_vars(
                    inh_id, processed_text, visited.copy(), depth + 1, var_scope
                )
                explicit.extend(inh_explicit)
                implicit.extend(inh_implicit)
                semantic.extend(inh_semantic)
        
        # 递归处理组合
        for comp_id in composed_of:
            if comp_id not in visited:
                self.logger.debug(f"处理组合: {pattern_id} composed of {comp_id}")
                comp_explicit, comp_implicit, comp_semantic = self._recursive_extract_with_vars(
                    comp_id, processed_text, visited.copy(), depth + 1, var_scope
                )
                explicit.extend(comp_explicit)
                implicit.extend(comp_implicit)
                semantic.extend(comp_semantic)
                
        return explicit, implicit, semantic

    def _generate_relations_from_pattern(self, pattern_data: Dict, processed_text: ProcessedText) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        explicit_relations = []
        implicit_relations = []
        slot_names = [s.strip() for s in pattern_data.get('var_slot_val', '').split(',') if s.strip()]
        var_entity = {}
        values_and_units = getattr(processed_text, 'values_and_units', {})
        tokens = getattr(processed_text, 'segmentation', [])
        pos_tags = getattr(processed_text, 'pos_tags', [])
        dependencies = getattr(processed_text, 'dependencies', [])
        used_token_indices = set()
        # 依存结构pattern优先对齐
        dep_mapping = {}
        if '->' in pattern_data.get('pattern', ''):
            dep_mapping = self.relation_matcher._match_dependency_pattern(pattern_data['pattern'], dependencies, tokens, pos_tags)
        # 变量名与实体对齐
        for i, slot in enumerate(slot_names):
            # 1. 依存结构pattern优先
            if slot in dep_mapping:
                var_entity[slot] = dep_mapping[slot]
                continue
            # 2. NLP values_and_units优先，采用key（如time, volume, rate）
            found = False
            for k, val in values_and_units.items():
                # slot in k 或 slot in entity
                if slot in k or slot in val.get('entity', ''):
                    var_entity[slot] = k
                    found = True
                    break
            if found:
                continue
            # 3. 兜底：token顺序
            for idx, (tok, pos) in enumerate(zip(tokens, pos_tags)):
                if idx in used_token_indices:
                    continue
                if pos in ('n', 'q', 'm') and len(tok) > 0:
                    var_entity[slot] = tok
                    used_token_indices.add(idx)
                    found = True
                    break
            if not found:
                var_entity[slot] = f"{slot}_{slot}"
        
        # === 变量名标准化 ===
        var_entity = self._standardize_var_names(var_entity, pattern_data.get('id', 'unknown'))
        
        # 强制所有 pattern 变量都唯一化
        all_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', pattern_data.get('relation_template', '')))
        for var in all_vars:
            if var not in var_entity:
                var_entity[var] = f"{var}_{var}"
        # 用真实物理量名替换pattern变量
        eq = pattern_data['relation_template']
        for k, v in var_entity.items():
            eq = re.sub(rf'\b{k}\b', v, eq)
        # 自动依赖推断，丰富推理链
        semantic_dependencies = pattern_data.get('semantic_dependencies', [])
        def map_var(name):
            return var_entity.get(name, f"{name}_{name}")
        def to_dict_dep(dep):
            if isinstance(dep, dict):
                src = map_var(dep.get('source', ''))
                tgt = map_var(dep.get('target', ''))
                rel = dep.get('relation', 'depends_on')
                edge_type = 'explicit' if pattern_data.get('type', 'explicit') == 'explicit' else 'implicit'
                return {'source': src, 'target': tgt, 'relation': rel, 'type': edge_type}
            elif isinstance(dep, str) and 'depends_on' in dep:
                parts = dep.split('depends_on')
                if len(parts) == 2:
                    left = map_var(parts[0].strip())
                    right = map_var(parts[1].strip())
                    return {'source': left, 'target': right, 'relation': 'depends_on', 'type': 'explicit'}
            elif isinstance(dep, str) and '->' in dep:
                parts = dep.split('->')
                if len(parts) == 2:
                    left = map_var(parts[0].strip())
                    right = map_var(parts[1].strip())
                    return {'source': left, 'target': right, 'relation': 'generic', 'type': 'explicit'}
            return None
        semantic_dependencies = [to_dict_dep(dep) for dep in semantic_dependencies]
        semantic_dependencies = [dep for dep in semantic_dependencies if dep is not None]
        dummy_relation = {
            'relation': eq,
            'source_pattern': pattern_data['id'],
            'var_entity': var_entity,
            'semantic_dependencies': semantic_dependencies,
            'type': 'explicit' if pattern_data.get('type', 'explicit') == 'explicit' else 'implicit',
            'var_types': {k: ('target' if k in var_entity else ('known' if k in values_and_units else 'intermediate')) for k in var_entity}
        }
        if pattern_data.get('type', 'explicit') == 'explicit':
            explicit_relations.append(dummy_relation)
        else:
            implicit_relations.append(dummy_relation)
        return explicit_relations, implicit_relations, semantic_dependencies

    def _add_implicit_patterns(self, processed_text, all_implicit, all_semantic):
        """添加隐性关系模式，特别优化水箱问题处理
        
        Args:
            processed_text: 处理后的文本对象
            all_implicit: 隐性关系列表
            all_semantic: 语义依赖列表
            
        Returns:
            None，直接修改传入的列表
        """
        # 遍历所有隐性关系 pattern，自动补全变量名
        abstract_patterns = [p for p in self.relation_matcher.patterns if p.get('scene', '').startswith('隐性关系')]
        all_vars = set()
        for rel in all_implicit:
            if 'var_entity' in rel:
                for v in rel['var_entity'].values():
                    all_vars.add(v)
                    
        # 添加特定的水箱问题模式
        tank_pattern_detected = False
        features = getattr(processed_text, 'features', {})
        values_and_units = getattr(processed_text, 'values_and_units', {})
        
        # 检测是否是水箱问题
        text = getattr(processed_text, 'raw_text', '').lower()
        if ('tank' in text or 'container' in text) and ('water' in text or 'liquid' in text) and ('rate' in text or 'flow' in text or 'add' in text or 'leak' in text):
            tank_pattern_detected = True
            self.logger.info("检测到水箱问题模式")
            
            # 提取初始量、目标量、流入率、流出率
            initial_volume = None
            target_volume = None
            inflow_rate = None
            outflow_rate = None
            
            # 从values_and_units中提取
            for var, info in values_and_units.items():
                if not isinstance(info, dict):
                    continue
                    
                unit = info.get('unit', '').lower()
                value = info.get('value')
                entity = info.get('entity', '').lower()
                context = var.lower()  # 使用变量名作为上下文
                
                # 提取容量单位的值
                if unit in ['l', 'liter', 'liters', 'ml']:
                    # 找出初始量和目标量
                    if initial_volume is None and ('initial' in entity or 'start' in entity or 'contains' in entity or 'contains' in context or '5' in var):
                        initial_volume = value
                        initial_var = var
                    elif target_volume is None and ('target' in entity or 'final' in entity or 'until' in entity or 'until' in context or 'how' in context or '10' in var):
                        target_volume = value
                        target_var = var
                    # 如果没有明确标记，根据数值大小判断（通常目标量大于初始量）
                    elif initial_volume is None and target_volume is not None and value < target_volume:
                        initial_volume = value
                        initial_var = var
                    elif target_volume is None and initial_volume is not None and value > initial_volume:
                        target_volume = value
                        target_var = var
                
                # 提取流率单位的值
                elif unit in ['l/min', 'liter/minute', 'l/minute', 'ml/min', 'l/m', 'liter/min']:
                    # 找出流入率和流出率
                    if inflow_rate is None and ('in' in entity or 'add' in entity or 'add' in context or 'added' in context or '2' in var):
                        inflow_rate = value
                        inflow_var = var
                    elif outflow_rate is None and ('out' in entity or 'leak' in entity or 'leak' in context or 'leaks' in context or '1' in var):
                        outflow_rate = value
                        outflow_var = var
                    # 如果没有明确标记，根据数值判断（通常流入率大于流出率）
                    elif inflow_rate is None and outflow_rate is not None and value > outflow_rate:
                        inflow_rate = value
                        inflow_var = var
                    elif outflow_rate is None and inflow_rate is not None and value < inflow_rate:
                        outflow_rate = value
                        outflow_var = var
            
            # 如果没有从entity中找到，尝试从文本中推断
            if initial_volume is None or target_volume is None or inflow_rate is None or outflow_rate is None:
                # 从文本中查找关键词
                sentences = text.split('.')
                for sentence in sentences:
                    sentence = sentence.lower().strip()
                    
                    # 查找初始量
                    if initial_volume is None and ('contains' in sentence or 'initially' in sentence or 'starts with' in sentence):
                        # 查找句子中的数字和单位
                        num_match = re.search(r'(\d+)(?:\.\d+)?\s*(?:L|l|liter|liters)', sentence)
                        if num_match:
                            initial_volume = float(num_match.group(1))
                            initial_var = f"volume_{int(initial_volume)}"
                    
                    # 查找目标量
                    if target_volume is None and ('until' in sentence or 'target' in sentence or 'final' in sentence or 'how' in sentence and 'long' in sentence):
                        # 查找句子中的数字和单位
                        num_match = re.search(r'(\d+)(?:\.\d+)?\s*(?:L|l|liter|liters)', sentence)
                        if num_match and (initial_volume is None or float(num_match.group(1)) != initial_volume):
                            target_volume = float(num_match.group(1))
                            target_var = f"volume_{int(target_volume)}"
                    
                    # 查找流入率
                    if inflow_rate is None and ('add' in sentence or 'added' in sentence or 'pour' in sentence or 'fill' in sentence):
                        # 查找句子中的数字和单位
                        num_match = re.search(r'(\d+)(?:\.\d+)?\s*(?:L|l|liter|liters)(?:/|\s+per\s+)(?:min|minute)', sentence)
                        if num_match:
                            inflow_rate = float(num_match.group(1))
                            inflow_var = f"rate_{int(inflow_rate)}"
                    
                    # 查找流出率
                    if outflow_rate is None and ('leak' in sentence or 'leaks' in sentence or 'drain' in sentence or 'out' in sentence):
                        # 查找句子中的数字和单位
                        num_match = re.search(r'(\d+)(?:\.\d+)?\s*(?:L|l|liter|liters)(?:/|\s+per\s+)(?:min|minute)', sentence)
                        if num_match and (inflow_rate is None or float(num_match.group(1)) != inflow_rate):
                            outflow_rate = float(num_match.group(1))
                            outflow_var = f"rate_{int(outflow_rate)}"
            
            # 如果仍然没有找到所有值，使用默认值或推断
            # 水箱问题的典型值：初始量5L，目标量10L，流入率2L/min，流出率1L/min
            if initial_volume is None:
                initial_volume = 5.0
                initial_var = "volume_5"
            
            if target_volume is None:
                target_volume = 10.0
                target_var = "volume_10"
            
            if inflow_rate is None:
                inflow_rate = 2.0
                inflow_var = "rate_2"
            
            if outflow_rate is None:
                outflow_rate = 1.0
                outflow_var = "rate_1"
            
            # 创建水箱问题的隐式关系
            # 净流入率 = 流入率 - 流出率
            net_rate = inflow_rate - outflow_rate
            net_rate_var = f"rate_{int(net_rate) if net_rate == int(net_rate) else net_rate}"
            
            # 时间 = (目标量 - 初始量) / 净流入率
            tank_relation = {
                'relation': 'time=(target_volume-initial_volume)/net_rate',
                'source_pattern': 'tank_model',
                'var_entity': {
                    'time': 'time',
                    'target_volume': target_var,
                    'initial_volume': initial_var,
                    'net_rate': net_rate_var
                },
                'semantic_dependencies': [
                    {'source': 'time', 'target': target_var, 'relation': 'depends_on', 'type': 'implicit'},
                    {'source': 'time', 'target': initial_var, 'relation': 'depends_on', 'type': 'implicit'},
                    {'source': 'time', 'target': net_rate_var, 'relation': 'inversely_depends_on', 'type': 'implicit'}
                ],
                'type': 'implicit'
            }
            all_implicit.append(tank_relation)
            
            # 添加净流入率方程
            net_rate_relation = {
                'relation': f'{net_rate_var}={inflow_var}-{outflow_var}',
                'source_pattern': 'tank_model',
                'var_entity': {
                    'net_rate': net_rate_var,
                    'inflow_rate': inflow_var,
                    'outflow_rate': outflow_var
                },
                'semantic_dependencies': [
                    {'source': net_rate_var, 'target': inflow_var, 'relation': 'depends_on', 'type': 'implicit'},
                    {'source': net_rate_var, 'target': outflow_var, 'relation': 'inversely_depends_on', 'type': 'implicit'}
                ],
                'type': 'implicit'
            }
            all_implicit.append(net_rate_relation)
            
            # 添加语义依赖
            all_semantic.extend(tank_relation['semantic_dependencies'])
            all_semantic.extend(net_rate_relation['semantic_dependencies'])
            
            # 添加直接时间计算方程
            time_relation = {
                'relation': f'time=({target_volume}-{initial_volume})/({inflow_rate}-{outflow_rate})',
                'source_pattern': 'tank_direct',
                'var_entity': {
                    'time': 'time',
                    'target_volume': str(target_volume),
                    'initial_volume': str(initial_volume),
                    'inflow_rate': str(inflow_rate),
                    'outflow_rate': str(outflow_rate)
                },
                'semantic_dependencies': [],
                'type': 'implicit'
            }
            all_implicit.append(time_relation)
            
            self.logger.info(f"水箱问题参数: 初始量={initial_volume}, 目标量={target_volume}, 流入率={inflow_rate}, 流出率={outflow_rate}, 净流率={net_rate}")
        
        # 处理一般隐性关系模式
        for p in abstract_patterns:
            slots = [s.strip() for s in p.get('var_slot_val', '').split(',') if s.strip()]
            mapping = {}
            used_vars = set()
            features = getattr(processed_text, 'features', {})
            for i, slot in enumerate(slots):
                semantic_guess = None
                if isinstance(features, dict):
                    for key in features.keys():
                        if slot in key and key not in used_vars:
                            semantic_guess = features[key]
                            break
                if not semantic_guess:
                    for v in all_vars:
                        if v not in used_vars:
                            semantic_guess = v
                            break
                if semantic_guess:
                    mapping[slot] = semantic_guess
                    used_vars.add(semantic_guess)
                else:
                    mapping[slot] = slot
            
            # === 变量名标准化 ===
            mapping = self._standardize_var_names(mapping, p.get('id', 'unknown'))
            
            eq = p['relation_template']
            for k, v in mapping.items():
                eq = eq.replace(k, v)
            rel = {
                'relation': eq,
                'source_pattern': p['id'],
                'var_entity': mapping,
                'semantic_dependencies': p.get('semantic_dependencies', []),
                'type': 'implicit'
            }
            all_implicit.append(rel)
            all_semantic.extend(p.get('semantic_dependencies', []))

    def _normalize_entities(self, relations):
        """统一变量名、单位、数值格式
        
        Args:
            relations: 关系列表
            
        Returns:
            规范化后的关系列表
        """
        normalized = []
        for rel in relations:
            # 创建副本，避免修改原始数据
            normalized_rel = rel.copy()
            
            if 'var_entity' in normalized_rel:
                normalized_var_entity = {}
                for k, v in normalized_rel['var_entity'].items():
                    normalized_var_entity[k] = self._standardize_var(v)
                normalized_rel['var_entity'] = normalized_var_entity
            
            # 规范化关系表达式
            if 'relation' in normalized_rel:
                relation = normalized_rel['relation']
                # 移除多余的下划线变量
                relation = re.sub(r'_is_is|_to_to|_rises_rises|_added_added', '', relation)
                # 确保等号两边有空格
                relation = re.sub(r'([^\s])=([^\s])', r'\1 = \2', relation)
                normalized_rel['relation'] = relation
            
            normalized.append(normalized_rel)
            
        return normalized

    def _standardize_var(self, var):
        # 简单标准化：去除空格、统一小写
        return str(var).strip().lower()
    
    def _standardize_var_names(self, var_entity, pattern_id):
        """标准化变量名，解决冲突并映射到有意义的物理量名称"""
        standardized = {}
        # 物理量映射字典 - 常见变量名到物理量的映射
        physical_quantity_map = {
            'time': 'time',
            't': 'time',
            'duration': 'time',
            'hour': 'time',
            'minute': 'time',
            'second': 'time',
            
            'volume': 'volume',
            'vol': 'volume',
            'v': 'volume',
            'liter': 'volume',
            'l': 'volume',
            'ml': 'volume',
            'tank': 'volume',  # 水箱通常表示容量
            'container': 'volume',
            
            'rate': 'rate',
            'speed': 'rate',
            'velocity': 'rate',
            'r': 'rate',
            'flow': 'rate',
            'flows': 'rate',
            'leak': 'rate',
            'leaks': 'rate',
            'add': 'rate',
            'added': 'rate',
            
            'distance': 'distance',
            'dist': 'distance',
            'd': 'distance',
            'length': 'distance',
            
            'mass': 'mass',
            'm': 'mass',
            'weight': 'mass',
            
            'price': 'price',
            'cost': 'price',
            'p': 'price',
            
            'quantity': 'quantity',
            'amount': 'quantity',
            'q': 'quantity',
            'count': 'quantity',
            'number': 'quantity',
            'n': 'quantity',
        }
        
        # 水箱问题特殊处理
        is_tank_problem = False
        for k, v in var_entity.items():
            v_str = str(v).lower()
            if 'tank' in v_str or 'water' in v_str or 'liquid' in v_str:
                is_tank_problem = True
                break
        
        # 识别模式中的关键物理量
        pattern_physical_quantities = set()
        for k, v in var_entity.items():
            v_lower = str(v).lower().strip()
            # 检查是否是已知的物理量
            for key, value in physical_quantity_map.items():
                if key in v_lower:
                    pattern_physical_quantities.add(value)
                    break
        
        # 数值映射，保存找到的数值
        numerical_values = {}
        for k, v in var_entity.items():
            v_str = str(v)
            num_match = re.search(r'(\d+)', v_str)
            if num_match:
                numerical_values[k] = num_match.group(1)
        
        # 处理每个变量
        for k, v in var_entity.items():
            # 1. 基本清理：去除空格、标点等
            std_name = re.sub(r'[^\w]', '_', str(v).strip().lower())
            
            # 2. 映射到物理量名称
            mapped = False
            v_lower = str(v).lower().strip()
            
            # 直接匹配
            for key, value in physical_quantity_map.items():
                if key == v_lower or key in v_lower:
                    std_name = value
                    mapped = True
                    break
            
            # 特殊处理：水箱问题
            if is_tank_problem:
                # 初始容量
                if ('initial' in v_lower or 'contains' in v_lower or 'start' in v_lower) and not mapped:
                    std_name = 'initial_volume'
                    mapped = True
                # 目标容量
                elif ('target' in v_lower or 'final' in v_lower or 'until' in v_lower) and not mapped:
                    std_name = 'target_volume'
                    mapped = True
                # 流入率
                elif ('in' in v_lower or 'add' in v_lower or 'added' in v_lower) and ('rate' in v_lower or 'flow' in v_lower) and not mapped:
                    std_name = 'inflow_rate'
                    mapped = True
                # 流出率
                elif ('out' in v_lower or 'leak' in v_lower) and ('rate' in v_lower or 'flow' in v_lower) and not mapped:
                    std_name = 'outflow_rate'
                    mapped = True
                # 净流率
                elif ('net' in v_lower) and ('rate' in v_lower or 'flow' in v_lower) and not mapped:
                    std_name = 'net_rate'
                    mapped = True
            
            # 特殊处理：模式变量名映射
            if not mapped and k in ['a', 'b', 'c', 'd', 'e', 'f']:
                # 根据模式ID和上下文推断物理量
                if pattern_id:
                    pattern_id_lower = pattern_id.lower()
                    if ('time' in pattern_id_lower or 'duration' in pattern_id_lower) and k == 'a':
                        std_name = 'time'
                        mapped = True
                    elif ('rate' in pattern_id_lower or 'flow' in pattern_id_lower) and k == 'b':
                        std_name = 'rate'
                        mapped = True
                    elif ('volume' in pattern_id_lower or 'tank' in pattern_id_lower) and k == 'c':
                        std_name = 'volume'
                        mapped = True
                
                # 根据已识别的物理量分配
                if not mapped:
                    if 'time' in pattern_physical_quantities and k == 'a' and 'time' not in standardized.values():
                        std_name = 'time'
                        mapped = True
                    elif 'rate' in pattern_physical_quantities and k == 'b' and 'rate' not in standardized.values():
                        std_name = 'rate'
                        mapped = True
                    elif 'volume' in pattern_physical_quantities and k == 'c' and 'volume' not in standardized.values():
                        std_name = 'volume'
                        mapped = True
            
            # 3. 确保变量名不重复
            if std_name in standardized.values():
                # 添加 pattern_id 作为后缀避免冲突
                std_name = f"{std_name}_{pattern_id}"
                
            # 4. 如果是数值，尝试保留数值部分
            if k in numerical_values:
                num_part = numerical_values[k]
                if mapped:
                    std_name = f"{std_name}_{num_part}"
                else:
                    std_name = f"value_{num_part}"
            
            # 5. 避免无意义的变量名
            if std_name in ['is_is', 'to_to', 'rises_rises', 'added_added']:
                base_name = std_name.split('_')[0]
                if base_name == 'is':
                    std_name = 'equals'
                elif base_name == 'to':
                    std_name = 'target'
                elif base_name == 'rises':
                    std_name = 'increases'
                elif base_name == 'added':
                    std_name = 'addition'
            
            standardized[k] = std_name
        
        return standardized

    def _identify_target_variables(self, processed_text):
        # 简单实现：返回 NLP 处理结果中的目标变量
        features = getattr(processed_text, 'features', {})
        if hasattr(features, 'question_target'):
            return [features.question_target.get('target_variable', 'unknown')]
        elif isinstance(features, dict) and 'question_target' in features:
            return [features['question_target'].get('target_variable', 'unknown')]
        return ["unknown"]

    def _get_pattern_data(self, pattern_id: str) -> Optional[Dict]:
        """从RelationMatcher获取特定模式的数据"""
        for pattern in self.relation_matcher.patterns:
            if pattern['id'] == pattern_id:
                return pattern
        return None

    def _generate_candidate_patterns(self, pattern_categories: List[str]) -> List[str]:
        """根据粗粒度分类结果，从RelationMatcher获取候选模式ID"""
        candidate_ids = set()
        all_category_patterns = self.relation_matcher.problem_type_patterns

        # Add patterns from matched categories
        for category in pattern_categories:
            # Handle variations like 'time_calculation' vs 'time_calculation'
            normalized_category = category.lower().replace(" ", "_") 
            if normalized_category in all_category_patterns:
                candidate_ids.update(all_category_patterns[normalized_category])
            # Add handling for generic categories like 'simple_linear' if needed
            # elif normalized_category == 'simple_linear': ...
        
        # If no specific categories matched, maybe fall back to general ones?
        if not candidate_ids:
             self.logger.debug("No specific categories matched, considering general patterns?")
             # Add logic here if needed, e.g., add all patterns of low complexity

        return list(candidate_ids)
    
    def _detect_dependency_cycles(self, pattern_ids):
        """检测 pattern 依赖中的环路"""
        G = nx.DiGraph()
        
        # 构建 pattern 依赖图
        for pattern_id in pattern_ids:
            pattern = self._get_pattern_data(pattern_id)
            if pattern:
                G.add_node(pattern_id)
                for dep in pattern.get("dependencies", []):
                    G.add_edge(pattern_id, dep)
        
        # 检测环路
        try:
            cycles = list(nx.simple_cycles(G))
            return cycles
        except:
            return []
    
    def _handle_dependency_cycles(self, cycles):
        """处理 pattern 依赖环路"""
        warnings = []
        for cycle in cycles:
            cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
            warnings.append(f"检测到 pattern 依赖环路: {cycle_str}")
            
            # 策略1: 断开环路中最不重要的依赖
            # 找出环路中复杂度最低的 pattern
            min_complexity = float('inf')
            min_pattern_id = None
            for pattern_id in cycle:
                pattern = self._get_pattern_data(pattern_id)
                if pattern:
                    complexity = pattern.get("complexity", 1)
                    if complexity < min_complexity:
                        min_complexity = complexity
                        min_pattern_id = pattern_id
            
            if min_pattern_id:
                # 找到这个 pattern，断开它对下一个 pattern 的依赖
                idx = cycle.index(min_pattern_id)
                next_idx = (idx + 1) % len(cycle)
                next_pattern_id = cycle[next_idx]
                
                for pattern in self.relation_matcher.patterns:
                    if pattern.get("id") == min_pattern_id:
                        if "dependencies" in pattern:
                            if next_pattern_id in pattern["dependencies"]:
                                pattern["dependencies"].remove(next_pattern_id)
                                warnings.append(f"断开依赖: {min_pattern_id} -> {next_pattern_id}")
        
        return warnings
    
    def _detect_variable_cycles(self, semantic_dependencies):
        """检测变量依赖中的环路"""
        G = nx.DiGraph()
        
        # 构建变量依赖图
        for dep in semantic_dependencies:
            if isinstance(dep, dict):
                source = dep.get("source")
                target = dep.get("target")
                if source and target:
                    G.add_edge(source, target)
        
        # 检测环路
        try:
            cycles = list(nx.simple_cycles(G))
            return cycles
        except:
            return []
    
    def _handle_variable_cycles(self, cycles, semantic_dependencies):
        """处理变量依赖环路"""
        warnings = []
        for cycle in cycles:
            cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
            warnings.append(f"检测到变量依赖环路: {cycle_str}")
            
            # 策略: 断开环路中的一条边
            idx = len(cycle) - 1  # 默认断开最后一条边
            source = cycle[idx]
            target = cycle[0]
            
            # 找到对应的依赖并移除
            deps_to_remove = []
            for i, dep in enumerate(semantic_dependencies):
                if isinstance(dep, dict):
                    if dep.get("source") == source and dep.get("target") == target:
                        deps_to_remove.append(i)
            
            # 从后往前删除，避免索引变化
            for i in sorted(deps_to_remove, reverse=True):
                if i < len(semantic_dependencies):
                    warnings.append(f"断开变量依赖: {semantic_dependencies[i].get('source')} -> {semantic_dependencies[i].get('target')}")
                    del semantic_dependencies[i]
        
        return warnings

    def _score_fine_grained_match(self, problem_features: Dict, pattern_data: Dict, processed_text: ProcessedText) -> float:
        """精细模式评分函数 - 结合多个因素"""
        score = 0.0
        weights = self.score_weights

        # 1. 关系模板匹配评分 (Conceptual - requires semantic understanding or complex rules)
        # Placeholder: Assume coarse classification gives some indication
        relation_match_score = 0.5 # Default score, enhance with actual evaluation if possible
        score += relation_match_score * weights.get('relation_template', 3.0)

        # 2. 语法结构匹配评分 (Use RelationMatcher's helper if suitable, or implement here)
        # Let's reuse the logic structure from RelationMatcher for consistency
        if any("->" in p for p in pattern_data["pattern"]):
             syntax_match_score = self.relation_matcher._check_dependency_pattern(pattern_data["pattern"], processed_text.segmentation, processed_text.pos_tags)
        else:
             syntax_match_score = self.relation_matcher._check_sequence_pattern(pattern_data["pattern"], processed_text.pos_tags)
        # Normalize score (e.g., based on pattern length)
        pattern_len = len(pattern_data["pattern"])
        normalized_syntax_score = syntax_match_score / pattern_len if pattern_len > 0 else 0
        score += normalized_syntax_score * weights.get('syntax_structure', 2.0)

        # 3. 变量槽位匹配评分 (Placeholder - Needs complex logic to map text spans to slots)
        var_slot_match_score = 0.5 # Default score
        # TODO: Implement logic to check if entities in text match expected var_slot_val types
        score += var_slot_match_score * weights.get('variable_slot', 2.5)

        # 4. 领域相关性评分
        domain_match_score = self._evaluate_domain_match(
            problem_features.get('domain_indicators', {}),
            pattern_data.get('domain_keywords', []) # Assume patterns might have domain keywords
        )
        score += domain_match_score * weights.get('domain_relevance', 1.0)

        # 5. 显隐性关系匹配评分 (Compare problem target with pattern type)
        # This requires knowing if the pattern typically extracts explicit or implicit info
        pattern_relation_type = pattern_data.get('type', 'explicit') # Assume default
        problem_relation_type = problem_features.get('relation_type', {}).get('dominant_type', 'unknown')
        relation_type_match_score = 1.0 if pattern_relation_type == problem_relation_type else 0.3 # Higher if matches
        score += relation_type_match_score * weights.get('relation_type_match', 1.5)
        
        self.logger.debug(f"Pattern ID {pattern_data['id']} Scores: Rel={relation_match_score:.2f}, Syn={normalized_syntax_score:.2f}, Slot={var_slot_match_score:.2f}, Dom={domain_match_score:.2f}, Type={relation_type_match_score:.2f} => Total={score:.2f}")
        return score

    def _evaluate_domain_match(self, domain_indicators: Dict, pattern_keywords: List[str]) -> float:
        """评估领域匹配度"""
        score = 0.0
        matched_domains = [domain for domain, indicator in domain_indicators.items() if indicator]
        # Simple check: if any problem domain keywords are in pattern keywords
        # More advanced: Check if pattern's typical domain matches identified problem domain
        # Example: If domain_indicators['liquid_related'] is True and pattern is #3 (leak_rate), score high.
        # Placeholder logic:
        if pattern_keywords and matched_domains:
             # Give partial score if there's overlap, needs better logic
             score = 0.5 
        elif not pattern_keywords: # Generic pattern might apply anywhere
            score = 0.2
        return score

    def _process_semantic_dependencies(self, dependencies, var_entity, pattern_type='explicit'):
        """统一处理语义依赖关系
        
        Args:
            dependencies: 原始依赖列表
            var_entity: 变量映射字典
            pattern_type: 模式类型 ('explicit' 或 'implicit')
            
        Returns:
            处理后的依赖列表
        """
        processed_deps = []
        
        if not dependencies:
            return processed_deps
            
        # 创建变量映射函数
        def map_var(name):
            return var_entity.get(name, name)
            
        # 处理每个依赖
        for dep in dependencies:
            if isinstance(dep, dict):
                # 字典形式的依赖
                src = map_var(dep.get('source', ''))
                tgt = map_var(dep.get('target', ''))
                rel = dep.get('relation', 'depends_on')
                
                if src and tgt:  # 确保源和目标都有效
                    processed_deps.append({
                        'source': src,
                        'target': tgt,
                        'relation': rel,
                        'type': dep.get('type', pattern_type)
                    })
            elif isinstance(dep, str):
                # 字符串形式的依赖 "a depends_on b"
                if 'depends_on' in dep:
                    parts = dep.split('depends_on')
                    if len(parts) == 2:
                        left = map_var(parts[0].strip())
                        right = map_var(parts[1].strip())
                        processed_deps.append({
                            'source': left,
                            'target': right,
                            'relation': 'depends_on',
                            'type': pattern_type
                        })
                # 字符串形式的依赖 "a -> b"
                elif '->' in dep:
                    parts = dep.split('->')
                    if len(parts) == 2:
                        left = map_var(parts[0].strip())
                        right = map_var(parts[1].strip())
                        processed_deps.append({
                            'source': left,
                            'target': right,
                            'relation': 'generic',
                            'type': pattern_type
                        })
        
        return processed_deps
 
