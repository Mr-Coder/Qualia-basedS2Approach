import json
import re
from pathlib import Path

from processors.visualization import (build_reasoning_graph,
                                      visualize_reasoning_chain)


def extract_dependencies_from_relation(relation):
    # 只支持简单的 a = b + c 形式
    if '=' not in relation:
        return []
    left, right = relation.split('=', 1)
    left = left.strip()
    # 提取右侧所有变量名
    right_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', right)
    # 排除数字
    right_vars = [v for v in right_vars if not v.isdigit()]
    return [f"{left} depends_on {v}" for v in right_vars if v != left]

def main():
    # 1. 读取抽取结果
    json_path = Path("src/examples/extracted_relations.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 取第一个样例
    extraction_result = data[0]["extraction_result"]

    # 2. 收集所有 semantic_dependencies
    explicit_deps = []
    for rel in extraction_result.get('explicit_relations', []):
        if rel.get('semantic_dependencies'):
            # pattern.json 结构化依赖
            for dep in rel['semantic_dependencies']:
                if isinstance(dep, dict):
                    # 结构化依赖
                    explicit_deps.append(f"{dep['source']} depends_on {dep['target']}")
                else:
                    explicit_deps.append(dep)
        else:
            # 自动从 relation 解析
            explicit_deps.extend(extract_dependencies_from_relation(rel.get('relation', '')))
    implicit_deps = []
    for rel in extraction_result.get('implicit_relations', []):
        if rel.get('semantic_dependencies'):
            for dep in rel['semantic_dependencies']:
                if isinstance(dep, dict):
                    implicit_deps.append(f"{dep['source']} depends_on {dep['target']}")
                else:
                    implicit_deps.append(dep)
        else:
            implicit_deps.extend(extract_dependencies_from_relation(rel.get('relation', '')))

    all_deps = [explicit_deps, implicit_deps]
    relation_types = ['explicit', 'implicit']

    # 3. 构建有向图
    G, node_type_map = build_reasoning_graph(all_deps, relation_types)

    # 4. 可视化
    if G.number_of_nodes() == 0:
        print("没有可视化的推理链（semantic_dependencies 为空且 relation 也无法推断）")
    else:
        visualize_reasoning_chain(G, node_type_map, title="推理链分组高亮", save_path="reasoning_chain_demo.png")
        print("推理链图已保存为 reasoning_chain_demo.png")

if __name__ == "__main__":
    main() 