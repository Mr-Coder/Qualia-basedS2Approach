import os
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

__all__ = [
    'build_reasoning_graph',
    'visualize_reasoning_chain',
    'export_graph_to_dot',
    'group_nodes_by_relation_type',
    'visualize_reasoning_chain_interactive',
    'annotate_reasoning_steps',
]

def build_reasoning_graph(semantic_dependencies_list, relation_types=None):
    """
    构建 reasoning chain 的有向图，支持多 pattern 合并。
    :param semantic_dependencies_list: List[List[str|dict]] 或 List[str|dict]
    :param relation_types: 可选，List[str]，与每组 semantic_dependencies 对应（如 'explicit', 'implicit'）
    :return: networkx.DiGraph, node_type_map
    """
    G = nx.DiGraph()
    node_type_map = {}  # 节点: 类型（如 explicit/implicit）
    if isinstance(semantic_dependencies_list, list) and semantic_dependencies_list and isinstance(semantic_dependencies_list[0], str):
        semantic_dependencies_list = [semantic_dependencies_list]
    for i, semantic_dependencies in enumerate(semantic_dependencies_list):
        rel_type = relation_types[i] if relation_types and i < len(relation_types) else None
        for dep in semantic_dependencies:
            if isinstance(dep, dict):
                src = dep.get("source")
                tgt = dep.get("target")
                rel = dep.get("relation", "depends_on")
                if src and tgt:
                    G.add_edge(tgt, src)  # 依赖方向与原逻辑一致
                    if rel_type:
                        node_type_map[src] = rel_type
                        node_type_map[tgt] = rel_type
            elif isinstance(dep, str):
                if "depends_on" in dep:
                    left, right = dep.split("depends_on")
                    left = left.strip()
                    right = right.strip()
                    G.add_edge(right, left)
                    if rel_type:
                        node_type_map[left] = rel_type
                        node_type_map[right] = rel_type
                elif "->" in dep:
                    left, right = dep.split("->")
                    G.add_edge(left.strip(), right.strip())
                    if rel_type:
                        node_type_map[left.strip()] = rel_type
                        node_type_map[right.strip()] = rel_type
    return G, node_type_map

def group_nodes_by_relation_type(node_type_map):
    """
    将节点按类型分组，返回 {type: [nodes]} 字典
    """
    groups = defaultdict(list)
    for node, typ in node_type_map.items():
        groups[typ].append(node)
    return groups

def visualize_reasoning_chain(G, node_type_map=None, title="Reasoning Chain", save_path=None, reasoning_paths=None, known_vars=None, target_vars=None, show=True):
    """
    增强版可视化推理链，支持多目标变量高亮和路径显示
    
    Args:
        G: networkx.DiGraph 推理链图
        node_type_map: 节点类型映射 {node: type}
        title: 图表标题
        save_path: 保存路径
        reasoning_paths: 推理路径列表，每个路径是节点列表
        known_vars: 已知变量列表
        target_vars: 目标变量列表
        show: 是否显示图表
    """
    print(f"[DEBUG] 可视化节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    if G.number_of_nodes() == 0:
        print("[WARNING] 推理链图为空，不可视化。")
        return
    
    # 为每个目标变量的路径使用不同颜色
    path_colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f1c40f']
    
    # 绘制基础图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # 基础节点和边
    nx.draw(G, pos, with_labels=True, node_color='#b0d4ec', 
            edge_color='#7f8c8d', font_weight='bold', node_size=1200)
    
    # 高亮已知量节点
    if known_vars:
        known_nodes = [n for n in G.nodes() if n in known_vars]
        if known_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=known_nodes, 
                                  node_color='#6bd46b', node_size=1400)
    
    # 高亮目标变量节点
    if target_vars:
        target_nodes = [n for n in G.nodes() if n in target_vars]
        if target_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, 
                                  node_color='#e74c3c', node_size=1400)
    
    # 高亮每条推理路径
    if reasoning_paths:
        for i, path in enumerate(reasoning_paths):
            path_edges = []
            for j in range(len(path)-1):
                if G.has_edge(path[j], path[j+1]):
                    path_edges.append((path[j], path[j+1]))
            if path_edges:
                color = path_colors[i % len(path_colors)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                      width=3.0, edge_color=color, arrows=True)
    
    # 结构分析
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
    
    plt.title(f"{title} - 结构类型: {structure_type}")
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def visualize_reasoning_chain_interactive(G, node_type_map=None, title="Reasoning Chain", 
                                         save_path=None, reasoning_paths=None, 
                                         known_vars=None, target_vars=None):
    """
    交互式可视化推理链
    
    Args:
        G: networkx.DiGraph 推理链图
        node_type_map: 节点类型映射 {node: type}
        title: 图表标题
        save_path: 保存路径
        reasoning_paths: 推理路径列表，每个路径是节点列表
        known_vars: 已知变量列表
        target_vars: 目标变量列表
        
    Returns:
        Network: pyvis.network.Network 对象
    """
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", heading=title)
    
    # 添加节点
    for node in G.nodes():
        color = "#b0d4ec"  # 默认颜色
        if known_vars and node in known_vars:
            color = "#6bd46b"  # 已知量绿色
        elif target_vars and node in target_vars:
            color = "#e74c3c"  # 目标变量红色
        net.add_node(node, label=node, color=color, title=f"变量: {node}")
    
    # 添加边
    path_edges = set()
    if reasoning_paths:
        for path in reasoning_paths:
            for i in range(len(path)-1):
                if G.has_edge(path[i], path[i+1]):
                    path_edges.add((path[i], path[i+1]))
    
    for u, v in G.edges():
        if (u, v) in path_edges:
            net.add_edge(u, v, color="#e74c3c", width=3, title="推理路径")  # 路径边红色加粗
        else:
            net.add_edge(u, v, color="#7f8c8d", title="依赖关系")
    
    # 设置物理布局
    net.set_options("""
    var options = {
        "nodes": {
            "font": {"size": 16, "face": "sans", "color": "#000000"},
            "shape": "dot",
            "size": 20
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 1}},
            "color": {"inherit": false},
            "smooth": {"type": "continuous"}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        }
    }
    """)
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        net.save_graph(save_path)
    
    return net

def export_graph_to_dot(G, dot_path):
    """
    导出 networkx.DiGraph 为 graphviz dot 文件。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(dot_path) if os.path.dirname(dot_path) else '.', exist_ok=True)
    nx.drawing.nx_pydot.write_dot(G, dot_path)

def annotate_reasoning_steps(G, reasoning_path, equations):
    """
    为推理路径上的每一步添加方程标注
    
    Args:
        G: networkx.DiGraph 推理链图
        reasoning_path: 推理路径，节点列表
        equations: 方程列表
        
    Returns:
        List[Dict]: 推理步骤列表，每个步骤包含 step, from, to, equation
    """
    steps = []
    for i in range(len(reasoning_path)-1):
        source = reasoning_path[i]
        target = reasoning_path[i+1]
        # 找到这一步对应的方程
        step_eq = None
        for eq in equations:
            if source in eq and target in eq:
                step_eq = eq
                break
        steps.append({
            "step": i+1,
            "from": source,
            "to": target,
            "equation": step_eq
        })
    return steps

def detect_cycles(G):
    """
    检测图中的环路
    
    Args:
        G: networkx.DiGraph
        
    Returns:
        List[List]: 环路列表，每个环路是节点列表
    """
    try:
        return list(nx.simple_cycles(G))
    except:
        return []

def find_all_reasoning_paths(G, known_vars, target_vars, max_paths=10, max_length=15):
    """
    查找从已知变量到目标变量的所有可能推理路径
    
    Args:
        G: networkx.DiGraph 推理链图
        known_vars: 已知变量列表
        target_vars: 目标变量列表
        max_paths: 每个起点-终点对返回的最大路径数
        max_length: 路径的最大长度限制
        
    Returns:
        List[List]: 所有推理路径列表
    """
    all_paths = []
    
    # 确保输入变量在图中
    valid_known = [v for v in known_vars if v in G.nodes()]
    valid_targets = [v for v in target_vars if v in G.nodes()]
    
    if not valid_known or not valid_targets:
        return all_paths
    
    # 对每个起点-终点对查找路径
    for source in valid_known:
        for target in valid_targets:
            try:
                # 使用all_simple_paths查找所有简单路径
                paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_length))
                
                # 限制每对起点-终点的路径数
                if paths:
                    paths = paths[:max_paths]
                    all_paths.extend(paths)
            except nx.NetworkXNoPath:
                continue
    
    return all_paths

def rank_reasoning_paths(G, paths, known_vars, target_vars):
    """
    对推理路径进行排序，选择最优路径
    
    Args:
        G: networkx.DiGraph 推理链图
        paths: 推理路径列表
        known_vars: 已知变量列表
        target_vars: 目标变量列表
        
    Returns:
        List[Dict]: 排序后的路径及其评分
    """
    if not paths:
        return []
    
    ranked_paths = []
    
    for path in paths:
        # 评估路径质量
        quality = evaluate_reasoning_chain(path, G)
        
        # 计算路径覆盖率（覆盖已知变量和目标变量的比例）
        known_coverage = len([v for v in path if v in known_vars]) / max(1, len(known_vars))
        target_coverage = len([v for v in path if v in target_vars]) / max(1, len(target_vars))
        
        # 计算路径直接性（路径长度的倒数）
        directness = 1.0 / max(1, len(path) - 1) if len(path) > 1 else 1.0
        
        # 综合评分
        combined_score = (
            quality['score'] * 0.4 +  # 基础质量评分
            known_coverage * 0.2 +    # 已知变量覆盖
            target_coverage * 0.3 +   # 目标变量覆盖
            directness * 0.1          # 路径直接性
        )
        
        ranked_paths.append({
            'path': path,
            'score': combined_score,
            'quality': quality,
            'known_coverage': known_coverage,
            'target_coverage': target_coverage,
            'directness': directness
        })
    
    # 按评分降序排序
    ranked_paths.sort(key=lambda x: x['score'], reverse=True)
    
    return ranked_paths

def select_optimal_reasoning_paths(G, known_vars, target_vars, max_paths=5):
    """
    选择最优的推理路径集合
    
    Args:
        G: networkx.DiGraph 推理链图
        known_vars: 已知变量列表
        target_vars: 目标变量列表
        max_paths: 返回的最大路径数
        
    Returns:
        List[List]: 最优推理路径列表
    """
    # 查找所有可能路径
    all_paths = find_all_reasoning_paths(G, known_vars, target_vars)
    
    if not all_paths:
        return []
    
    # 对路径进行排序
    ranked_paths = rank_reasoning_paths(G, all_paths, known_vars, target_vars)
    
    # 选择前N条最优路径
    optimal_paths = [p['path'] for p in ranked_paths[:max_paths]]
    
    return optimal_paths

def evaluate_reasoning_chain(reasoning_path, G):
    """
    评估推理链的质量
    
    Args:
        reasoning_path: 推理路径，节点列表
        G: networkx.DiGraph 推理链图
        
    Returns:
        Dict: 评估结果，包含 score, path_length, is_direct, missing_nodes, completeness
    """
    if not reasoning_path:
        return {"score": 0, "reason": "空路径"}
    
    # 计算路径长度
    path_length = len(reasoning_path) - 1
    
    # 检查是否包含所有必要节点
    missing_nodes = []
    for node in G.nodes():
        if G.in_degree(node) == 0:  # 入度为0的节点（可能是已知量）
            if node not in reasoning_path:
                missing_nodes.append(node)
    
    # 检查路径是否直接
    is_direct = True
    for i in range(len(reasoning_path)-1):
        if not G.has_edge(reasoning_path[i], reasoning_path[i+1]):
            is_direct = False
            break
    
    # 计算完整性（路径中节点数与图中总节点数的比例）
    completeness = len(set(reasoning_path)) / max(1, len(G.nodes()))
    
    # 计算得分
    score = 100
    if not is_direct:
        score -= 30
    score -= len(missing_nodes) * 10
    if path_length > 5:  # 路径过长可能不是最优
        score -= (path_length - 5) * 5
    
    # 加上完整性得分
    score += completeness * 20
    
    return {
        "score": max(0, min(100, score)),  # 限制在0-100范围
        "path_length": path_length,
        "is_direct": is_direct,
        "missing_nodes": missing_nodes,
        "completeness": completeness
    }

# 主流程集成示例
if __name__ == '__main__':
    # 示例：合并显性/隐性关系的 semantic_dependencies
    explicit = ["a depends_on b", "a depends_on c"]
    implicit = ["d depends_on a", "d depends_on e"]
    all_deps = [explicit, implicit]
    rel_types = ['explicit', 'implicit']
    G, node_type_map = build_reasoning_graph(all_deps, rel_types)
    print("节点分组:", group_nodes_by_relation_type(node_type_map))
    
    # 测试增强可视化
    reasoning_paths = [["b", "a", "d"], ["c", "a", "d"]]
    known_vars = ["b", "c", "e"]
    target_vars = ["d"]
    
    visualize_reasoning_chain(G, node_type_map, 
                             title="推理链分组高亮", 
                             save_path="reasoning_chain_demo.png",
                             reasoning_paths=reasoning_paths,
                             known_vars=known_vars,
                             target_vars=target_vars)
    
    # 测试交互式可视化
    net = visualize_reasoning_chain_interactive(G, node_type_map,
                                              title="交互式推理链",
                                              save_path="interactive_reasoning_chain.html",
                                              reasoning_paths=reasoning_paths,
                                              known_vars=known_vars,
                                              target_vars=target_vars)
    
    # 测试推理步骤标注
    equations = ["a = b + c", "d = a * e"]
    steps = annotate_reasoning_steps(G, reasoning_paths[0], equations)
    print("推理步骤:", steps)
    
    # 测试环路检测
    cycles = detect_cycles(G)
    print("检测到的环路:", cycles)
    
    # 测试推理链评估
    quality = evaluate_reasoning_chain(reasoning_paths[0], G)
    print("推理链质量:", quality)
    
    export_graph_to_dot(G, "reasoning_chain_demo.dot") 