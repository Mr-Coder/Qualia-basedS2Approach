import json
import os
import re
from pathlib import Path


def extract_semantic_dependencies(relation_template):
    """
    从 relation_template 自动生成 semantic_dependencies
    """
    if not relation_template or '=' not in relation_template:
        return []
    left, right = relation_template.split('=', 1)
    left = left.strip()
    right = right.strip()
    # 提取右侧所有变量名
    right_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', right)
    # 排除数字和常见函数名
    right_vars = [v for v in right_vars if not v.isdigit() and v not in {'sin','cos','tan','exp','log','sqrt'}]
    # 检查运算符
    dependencies = []
    # 检查是否有除号
    if '/' in right:
        # 只处理最外层的除法
        parts = right.split('/')
        numerator = parts[0]
        denominator = '/'.join(parts[1:])
        num_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', numerator)
        den_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', denominator)
        for v in num_vars:
            if v != left:
                dependencies.append({"source": left, "target": v, "relation": "depends_on"})
        for v in den_vars:
            if v != left:
                dependencies.append({"source": left, "target": v, "relation": "inversely_depends_on"})
        return dependencies
    # 检查是否有减号
    if '-' in right:
        parts = right.split('-')
        first = parts[0]
        rest = parts[1:]
        first_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', first)
        for v in first_vars:
            if v != left:
                dependencies.append({"source": left, "target": v, "relation": "depends_on"})
        for part in rest:
            for v in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', part):
                if v != left:
                    dependencies.append({"source": left, "target": v, "relation": "inversely_depends_on"})
        return dependencies
    # 检查是否有加号或乘号
    for v in right_vars:
        if v != left:
            dependencies.append({"source": left, "target": v, "relation": "depends_on"})
    return dependencies

def process_pattern_obj(obj):
    if isinstance(obj, dict):
        # 只处理有 relation_template 的 dict
        if 'relation_template' in obj:
            if 'semantic_dependencies' not in obj or not obj['semantic_dependencies']:
                deps = extract_semantic_dependencies(obj['relation_template'])
                if deps:
                    obj['semantic_dependencies'] = deps
        # 递归处理所有子字段
        for k, v in obj.items():
            process_pattern_obj(v)
    elif isinstance(obj, list):
        for item in obj:
            process_pattern_obj(item)

def main():
    # 修正路径，确保指向 src/models/pattern.json
    pattern_path = Path(__file__).parent.parent / 'models' / 'pattern.json'
    with open(pattern_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    process_pattern_obj(data)
    with open(pattern_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('已自动补全缺失的 semantic_dependencies 字段')

if __name__ == '__main__':
    main() 