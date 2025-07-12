import logging
import re
from typing import Any, Dict

import networkx as nx
from sympy import Eq, solve, symbols, sympify

logger = logging.getLogger(__name__)


class EquationBuilder:
    def __init__(self, config=None):
        self.config = config or {}

    def solve_equation_system(self, equations, target_vars, values_and_units):
        """自动单位换算、优先求解目标变量、支持多目标/多步递推"""
        # 1. 单位换算表（可扩展）
        unit_table = {
            ('ml', 'l'): lambda x: x / 1000,
            ('l', 'ml'): lambda x: x * 1000,
            ('minutes', 'hours'): lambda x: x / 60,
            ('hours', 'minutes'): lambda x: x * 60,
        }
        # 2. 统一单位（以第一个目标变量的单位为主）
        main_unit = None
        for t in target_vars:
            for k, v in values_and_units.items():
                if t in k and v.get('unit'):
                    main_unit = v['unit'].lower()
                    break
            if main_unit:
                break
        # 3. 单位换算
        converted_values = {}
        for k, v in values_and_units.items():
            val = v['value']
            unit = v.get('unit', '').lower()
            entity = v.get('entity', '')
            if main_unit and unit and unit != main_unit:
                func = unit_table.get((unit, main_unit))
                if func:
                    val = func(val)
                    unit = main_unit
            converted_values[k] = {'value': val, 'unit': unit, 'entity': entity}
        # 4. 构建符号
        all_vars = set()
        for eq in equations:
            all_vars.update(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', eq))
        sym_vars = {v: symbols(v) for v in all_vars}
        # 5. 组装方程
        sym_eqs = []
        for eq in equations:
            try:
                sym_eqs.append(sympify(eq, locals=sym_vars))
            except Exception:
                pass
        # 6. 数值代入
        subs = {}
        for k, v in converted_values.items():
            if k in sym_vars:
                subs[sym_vars[k]] = v['value']
        # 7. 多目标/多步递推求解
        results = {}
        for t in target_vars:
            if t in sym_vars:
                try:
                    sol = solve(sym_eqs, sym_vars[t], dict=True)
                    if sol:
                        val = sol[0].get(sym_vars[t], None)
                        if val is not None and subs:
                            val = val.subs(subs)
                        results[t] = val
                except Exception as e:
                    results[t] = f'求解失败: {e}'
        return results

    def build_equations(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据关系提取结果构建方程组
        Args:
            extraction_result: 关系提取结果
        Returns:
            Dict 包含方程组和变量信息
        """
        logger.info(f"[EquationBuilder] 输入: {extraction_result}")
        equations = []
        variables = {}
        
        # 1. 构建方程组
        all_equations = []
        for rel in extraction_result.get('explicit_relations', []):
            eq = self._map_pattern_vars(rel['relation'], rel.get('var_entity', {}))
            all_equations.append({
                'equation': eq,
                'source_pattern': rel.get('source_pattern', 'unknown'),
                'type': 'explicit',
                'semantic_dependencies': rel.get('semantic_dependencies', []),
                'priority': 5  # 默认优先级
            })
        for rel in extraction_result.get('implicit_relations', []):
            eq = self._map_pattern_vars(rel['relation'], rel.get('var_entity', {}))
            # 水箱问题模式的隐性关系优先级更高
            priority = 8 if rel.get('source_pattern') == 'tank_model' else 3
            all_equations.append({
                'equation': eq,
                'source_pattern': rel.get('source_pattern', 'unknown'),
                'type': 'implicit',
                'semantic_dependencies': rel.get('semantic_dependencies', []),
                'priority': priority
            })
            
        # 2. 去重和简化方程组
        unique_equations = []
        seen_eqs = set()
        for eq_data in all_equations:
            eq = eq_data['equation']
            # 标准化方程形式
            eq = self._normalize_equation(eq)
            if eq and eq not in seen_eqs:
                seen_eqs.add(eq)
                unique_equations.append({**eq_data, 'equation': eq})
        
        # 3. 按照模式复杂度、优先级和依赖关系排序
        def get_equation_score(eq_data):
            # 基础优先级
            priority_score = eq_data.get('priority', 5) * 10
            
            # 水箱问题特殊模式加分
            if eq_data.get('source_pattern') == 'tank_model':
                priority_score += 50
            
            # 显性关系优先
            type_score = 10 if eq_data['type'] == 'explicit' else 5
            
            # 依赖数量越少越简单
            dep_score = 10 - min(len(eq_data['semantic_dependencies']), 10)
            
            # 方程中变量数量越少越简单
            var_count = len(self._extract_variables([eq_data['equation']]))
            var_score = 10 - min(var_count, 10)
            
            # 避免无意义变量的方程
            meaningless_vars = ['is_is', 'to_to', 'rises_rises', 'added_added']
            for var in meaningless_vars:
                if var in eq_data['equation']:
                    priority_score -= 30
            
            # 包含目标变量的方程加分
            target_vars = extraction_result.get('target_variables', [])
            for target in target_vars:
                if target in eq_data['equation']:
                    priority_score += 20
            
            return priority_score + type_score + dep_score + var_score
            
        sorted_equations = sorted(unique_equations, key=get_equation_score, reverse=True)
        
        # 提取最终方程组（去除冗余和无意义方程）
        final_equations = []
        for eq_data in sorted_equations:
            eq = eq_data['equation']
            # 过滤掉包含无意义变量的方程
            if any(var in eq for var in ['is_is', 'to_to', 'rises_rises', 'added_added']):
                continue
                
            # 过滤掉函数调用形式但没有被正确转换的方程
            if '(' in eq and ')' in eq and '=' in eq:
                parts = eq.split('=')
                if len(parts) == 2 and '(' in parts[0] and ')' in parts[0]:
                    continue
            
            # 添加到最终方程组
            final_equations.append(eq)
            
            # 限制方程数量，避免过多冗余
            if len(final_equations) >= 7:  # 一般问题不需要太多方程
                break
        
        # 4. 构建推理链有向图
        G = nx.DiGraph()
        eq_to_vars = {}
        for eq_data in sorted_equations:
            eq = eq_data['equation']
            var_set = self._extract_variables([eq])
            eq_to_vars[eq] = var_set
            for v in var_set:
                G.add_node(v)
            # 以依赖为边
            for dep in eq_data.get('semantic_dependencies', []):
                if isinstance(dep, dict):
                    source = dep.get('source')
                    target = dep.get('target')
                    if source and target:
                        G.add_edge(target, source, 
                                 relation=dep.get('relation', 'depends_on'), 
                                 type=dep.get('type', 'explicit'))
        
        # 5. 识别关键变量
        known_vars = set()
        target_vars = set()
        
        # 从NLP结果中提取已知变量
        values_and_units = extraction_result.get('values_and_units', {})
        for var in values_and_units:
            # 标准化变量名
            std_var = var.lower().replace(' ', '_')
            known_vars.add(std_var)
            
        # 从提取结果中获取目标变量
        for var in extraction_result.get('target_variables', []):
            if var != 'unknown':
                # 标准化变量名
                std_var = var.lower().replace(' ', '_')
                target_vars.add(std_var)
                
        # 如果没有明确的目标变量，尝试从问题特征中推断
        if not target_vars:
            # 查找常见目标变量名
            var_names = self._extract_variables(final_equations)
            for var in var_names:
                var_lower = var.lower()
                if 'time' in var_lower or 'duration' in var_lower:
                    target_vars.add(var)
                    break
        
        # 特殊处理：水箱问题
        is_tank_problem = False
        for eq in final_equations:
            if 'tank' in eq or 'volume' in eq or 'rate' in eq:
                is_tank_problem = True
                break
                
        if is_tank_problem and not target_vars:
            # 水箱问题通常求时间
            var_names = self._extract_variables(final_equations)
            for var in var_names:
                var_lower = var.lower()
                if 'time' in var_lower:
                    target_vars.add(var)
                    break
                    
        # 6. 推理链路径提取
        all_paths = []
        for s in known_vars:
            for t in target_vars:
                if s in G.nodes() and t in G.nodes():
                    try:
                        for path in nx.all_simple_paths(G, source=s, target=t):
                            all_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        # 7. 结构类型分析
        structure_type = "未知"
        if nx.is_tree(G):
            structure_type = "树"
        elif nx.is_forest(G):
            structure_type = "森林"
        elif nx.is_directed_acyclic_graph(G):
            structure_type = "有向无环图"
        elif list(nx.simple_cycles(G)):
            structure_type = "环"
        else:
            structure_type = "链"
            
        # 8. 可解性检查
        var_names = self._extract_variables(final_equations)
        is_solvable = False
        if structure_type in ("树", "链", "森林", "有向无环图"):
            # 检查方程数是否足够
            if len(final_equations) >= len(var_names) - len(known_vars):
                is_solvable = True
        
        # 9. 冗余性分析
        redundancy = len(final_equations) > len(var_names) - len(known_vars)
        
        # 10. 自动求解目标变量
        solve_results = self.solve_equation_system(final_equations, list(target_vars), values_and_units)
        
        result = {
            'equations': final_equations,
            'variables': list(var_names),
            'known_vars': list(known_vars),
            'target_vars': list(target_vars),
            'structure_type': structure_type,
            'is_solvable': is_solvable,
            'redundancy': redundancy,
            'reasoning_paths': all_paths,
            'solve_results': solve_results
        }
        logger.info(f"[EquationBuilder] 输出: {result}")
        return result
        
    def _normalize_equation(self, eq):
        """标准化方程形式，移除无用部分，简化表达式"""
        if not eq:
            return ""
            
        # 移除函数调用形式，保留等式部分
        if '(' in eq and ')' in eq and '=' in eq:
            parts = eq.split('=')
            if len(parts) == 2:
                left, right = parts
                # 如果左边是函数调用，尝试提取变量
                if '(' in left and ')' in left:
                    var_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', left)
                    if var_match:
                        var_name = var_match.group(1)
                        return f"{var_name}={right}"
        
        # 移除无意义的is/to等词语
        eq = re.sub(r'_is|_to|_added|_rises', '', eq)
        
        # 如果没有等号，不是有效方程
        if '=' not in eq:
            return ""
            
        return eq

    def _map_pattern_vars(self, eq, var_entity):
        # eq: 'a=b*c/d', var_entity: {'a': 'time', 'b': 'volume', 'c': 'rate'}
        # 用实际变量名替换 pattern 变量
        for k, v in var_entity.items():
            eq = re.sub(rf'\b{k}\b', v, eq)
        return eq

    def _extract_variables(self, equations):
        # 用正则提取所有变量名（只取字母开头的变量）
        var_names = set()
        for eq in equations:
            for var in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', eq):
                # 排除数字和常见函数名
                if not var.isdigit() and var not in {'sin','cos','tan','exp','log','sqrt'}:
                    var_names.add(var)
        return var_names 