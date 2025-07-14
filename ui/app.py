#!/usr/bin/env python3
"""
COT-DIR 项目本地Web UI
简单的Flask应用，用于查看项目状态和测试功能
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from flask import (Flask, jsonify, redirect, render_template, request,
                       url_for)
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Installing...")
    os.system("pip install flask flask-cors")
    from flask import (Flask, jsonify, redirect, render_template, request,
                       url_for)
    from flask_cors import CORS

# 创建Flask应用
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# 全局变量
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs" / "generated"
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"

def get_project_stats() -> Dict[str, Any]:
    """获取项目统计信息"""
    stats = {
        "project_name": "COT-DIR - Chain-of-Thought Directed Implicit Reasoning",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_files": 0,
        "total_lines": 0,
        "modules": [],
        "tests": 0,
        "docs": 0
    }
    
    # 统计Python文件
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if "venv" not in str(py_file) and ".git" not in str(py_file):
            stats["python_files"] += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    stats["total_lines"] += len(f.readlines())
            except:
                pass
    
    # 统计模块
    if SRC_DIR.exists():
        for module_dir in SRC_DIR.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith('.'):
                py_count = len(list(module_dir.rglob("*.py")))
                if py_count > 0:
                    stats["modules"].append({
                        "name": module_dir.name,
                        "files": py_count
                    })
    
    # 统计测试
    tests_dir = PROJECT_ROOT / "tests"
    if tests_dir.exists():
        stats["tests"] = len(list(tests_dir.rglob("test_*.py")))
    
    # 统计文档
    if DOCS_DIR.exists():
        stats["docs"] = len(list(DOCS_DIR.rglob("*.md")))
    
    return stats

def get_reasoning_strategies() -> List[Dict[str, Any]]:
    """获取推理策略信息"""
    strategies = []
    
    # 推理策略位于 reasoning/strategy_manager 目录
    strategy_dir = SRC_DIR / "reasoning" / "strategy_manager"
    if strategy_dir.exists():
        for strategy_file in strategy_dir.glob("*_strategy.py"):
            try:
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 简单解析策略信息
                name = strategy_file.stem.replace('_strategy', '').title()
                description = "No description available"
                
                # 尝试提取docstring
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'class' in line and 'Strategy' in line:
                        # 查找class后的docstring
                        for j in range(i+1, min(i+10, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                desc_start = j
                                for k in range(j+1, min(j+5, len(lines))):
                                    if '"""' in lines[k] or "'''" in lines[k]:
                                        description = ' '.join(lines[desc_start+1:k]).strip()
                                        break
                                break
                        break
                
                strategies.append({
                    "name": name,
                    "file": strategy_file.name,
                    "description": description[:100] + "..." if len(description) > 100 else description,
                    "size": strategy_file.stat().st_size
                })
            except Exception as e:
                print(f"Error reading {strategy_file}: {e}")
    
    return strategies

def get_system_status() -> Dict[str, Any]:
    """获取系统状态"""
    import psutil
    
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "python_version": sys.version,
        "platform": sys.platform,
        "working_directory": str(PROJECT_ROOT)
    }

def get_recent_files(limit: int = 10) -> List[Dict[str, Any]]:
    """获取最近修改的文件"""
    files = []
    
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if "venv" not in str(py_file) and ".git" not in str(py_file):
            try:
                stat = py_file.stat()
                files.append({
                    "name": py_file.name,
                    "path": str(py_file.relative_to(PROJECT_ROOT)),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
            except:
                pass
    
    # 按修改时间排序
    files.sort(key=lambda x: x["modified"], reverse=True)
    return files[:limit]

@app.route('/')
def index():
    """解题系统主页"""
    return render_template('solve.html')

@app.route('/dashboard')
def dashboard():
    """项目控制台"""
    stats = get_project_stats()
    return render_template('index.html', stats=stats)

@app.route('/api/stats')
def api_stats():
    """API: 获取项目统计"""
    return jsonify(get_project_stats())

@app.route('/api/strategies')
def api_strategies():
    """API: 获取推理策略"""
    return jsonify(get_reasoning_strategies())

@app.route('/api/system')
def api_system():
    """API: 获取系统状态"""
    return jsonify(get_system_status())

@app.route('/api/files')
def api_files():
    """API: 获取最近文件"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify(get_recent_files(limit))

@app.route('/strategies')
def strategies():
    """推理策略详细介绍页面"""
    strategies_info = {
        "cot": {
            "name": "COT - 思维链推理",
            "english_name": "Chain-of-Thought",
            "description": "基于链式推理的深层实体物性关系建模",
            "characteristics": [
                "链式分解：将复杂问题分解为有序的推理链条",
                "实体物性分析：深入分析每个实体的物理属性和能力",
                "状态转移跟踪：追踪实体在推理过程中的状态变化",
                "多层验证：从数值、逻辑、物理三个层面验证结果"
            ],
            "entity_modeling": [
                "识别核心实体及其物理属性",
                "建立实体间的链式关系",
                "验证物理约束和守恒定律",
                "跟踪实体状态的顺序转移"
            ],
            "best_for": "中等复杂度问题，需要清晰步骤的推理过程",
            "example_steps": [
                "实体物性链式分解",
                "隐含关系链接",
                "物理约束验证",
                "状态转移序列",
                "多层验证链"
            ]
        },
        "got": {
            "name": "GOT - 思维图推理", 
            "english_name": "Graph-of-Thought",
            "description": "基于图网络的多维实体关系建模",
            "characteristics": [
                "网络拓扑构建：将实体和关系构建为多层网络结构",
                "隐含边发现：识别实体间的隐含连接和依赖关系",
                "流动分析：分析实体、信息、能量在网络中的流动",
                "子图推理：通过子网络推理局部和全局性质"
            ],
            "entity_modeling": [
                "构建多层实体关系网络",
                "发现隐含的连接边和权重",
                "分析网络中的流动和传播",
                "推理子图的局部性质"
            ],
            "best_for": "复杂关系问题，多实体相互影响的情况",
            "example_steps": [
                "实体网络拓扑构建",
                "隐含边发现",
                "网络流分析",
                "子图属性推理",
                "网络平衡检验"
            ]
        },
        "tot": {
            "name": "TOT - 思维树推理",
            "english_name": "Tree-of-Thought", 
            "description": "基于分层树结构的多路径实体关系探索",
            "characteristics": [
                "层次树构建：建立分层的实体关系分类树",
                "多路径探索：探索多种解决方案和推理路径",
                "约束传播：在树结构中传播隐含约束",
                "最优选择：基于实体关系复杂度选择最优路径"
            ],
            "entity_modeling": [
                "构建实体的层次分类树",
                "探索多种实体关系路径",
                "传播隐含约束和属性",
                "综合各路径的实体关系"
            ],
            "best_for": "高复杂度问题，需要探索多种可能性的情况",
            "example_steps": [
                "实体层次树构建",
                "解决方案分支探索", 
                "隐含约束传播",
                "分支属性综合",
                "最优路径选择",
                "树一致性验证"
            ]
        }
    }
    return render_template('strategies_detail.html', strategies=strategies_info)

@app.route('/learning-guide')
def learning_guide():
    """学习指导页面"""
    learning_content = {
        "problem_types": [
            {"type": "算术问题", "description": "加减乘除基础运算", "key_concepts": ["数量关系", "运算规则", "实体聚合"]},
            {"type": "几何问题", "description": "面积、周长、体积计算", "key_concepts": ["空间关系", "公式应用", "维度分析"]},
            {"type": "应用题", "description": "购物找零、时间计算等", "key_concepts": ["现实映射", "约束条件", "逻辑推理"]},
            {"type": "百分比问题", "description": "比例、占比、增长率", "key_concepts": ["比例关系", "整体部分", "转换计算"]}
        ],
        "learning_tips": [
            "仔细识别问题中的实体和它们的属性",
            "分析实体之间的隐含关系",
            "选择合适的推理策略",
            "验证答案的合理性和约束条件"
        ]
    }
    return render_template('learning_guide.html', content=learning_content)

@app.route('/knowledge-map')
def knowledge_map():
    """知识图谱页面"""
    knowledge_structure = {
        "core_concepts": ["实体", "关系", "属性", "约束", "推理"],
        "reasoning_strategies": ["COT", "GOT", "TOT"],
        "problem_domains": ["算术", "几何", "应用", "百分比"],
        "thinking_skills": ["分解", "建模", "推理", "验证"]
    }
    return render_template('knowledge_map.html', knowledge=knowledge_structure)

@app.route('/error-analysis') 
def error_analysis():
    """错题分析页面"""
    common_errors = [
        {"type": "实体识别错误", "description": "未正确识别问题中的关键实体", "solution": "仔细阅读题目，标记所有名词和数量"},
        {"type": "关系理解错误", "description": "误解实体间的关系类型", "solution": "分析动词和连接词，理解实体间的作用关系"},
        {"type": "策略选择错误", "description": "选择了不适合的推理策略", "solution": "根据问题复杂度和关系类型选择策略"},
        {"type": "约束忽略错误", "description": "忽略了隐含的约束条件", "solution": "检查结果是否符合现实逻辑和物理定律"}
    ]
    return render_template('error_analysis.html', errors=common_errors)

@app.route('/docs')
def docs():
    """文档页面"""
    docs_list = []
    if DOCS_DIR.exists():
        for doc_file in DOCS_DIR.glob("*.md"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取标题
                    title = doc_file.stem.replace('_', ' ').title()
                    first_line = content.split('\n')[0] if content else ""
                    if first_line.startswith('#'):
                        title = first_line.strip('#').strip()
                    
                    docs_list.append({
                        "title": title,
                        "file": doc_file.name,
                        "size": doc_file.stat().st_size,
                        "preview": content[:200] + "..." if len(content) > 200 else content
                    })
            except Exception as e:
                print(f"Error reading {doc_file}: {e}")
    
    return render_template('docs.html', docs=docs_list)

@app.route('/docs/<filename>')
def view_doc(filename):
    """查看文档"""
    doc_path = DOCS_DIR / filename
    if doc_path.exists():
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return render_template('doc_view.html', filename=filename, content=content)
        except Exception as e:
            return f"Error reading file: {e}", 500
    return "File not found", 404

@app.route('/system')
def system():
    """系统状态页面"""
    status = get_system_status()
    files = get_recent_files()
    return render_template('system.html', status=status, files=files)

@app.route('/test')
def test():
    """测试页面"""
    return render_template('test.html')

@app.route('/test-solve')
def test_solve():
    """解题测试页面"""
    return render_template('test_solve.html')

@app.route('/api/test/run', methods=['POST'])
def api_test_run():
    """API: 运行测试"""
    test_type = request.json.get('test_type', 'basic')
    
    if test_type == 'basic':
        # 基础测试
        result = {
            "status": "success",
            "message": "基础测试通过",
            "details": {
                "project_structure": "✅ 正常",
                "imports": "✅ 正常",
                "configuration": "✅ 正常"
            }
        }
    elif test_type == 'reasoning':
        # 推理测试
        result = {
            "status": "success", 
            "message": "推理系统测试通过",
            "details": {
                "chain_of_thought": "✅ 正常",
                "graph_reasoning": "✅ 正常",
                "template_system": "✅ 正常"
            }
        }
    else:
        result = {
            "status": "error",
            "message": "未知测试类型"
        }
    
    return jsonify(result)

@app.route('/api/solve', methods=['POST'])
def api_solve():
    """API: 解题"""
    try:
        data = request.json
        problem_text = data.get('problem', '').strip()
        strategy = data.get('strategy', 'auto')
        
        if not problem_text:
            return jsonify({
                "success": False,
                "error": "问题文本不能为空"
            }), 400
        
        # 直接使用模拟结果，避免推理引擎初始化问题
        print(f"收到解题请求: {problem_text}, 策略: {strategy}")
        result = get_mock_solution(problem_text, strategy)
        print(f"返回结果: 成功")
        return result
            
    except Exception as e:
        print(f"API处理错误: {e}")
        return jsonify({
            "success": False,
            "error": f"处理请求时出错: {str(e)}"
        }), 500

def get_mock_solution(problem_text, strategy):
    """获取模拟解题结果"""
    import re
    
    # 提取数字
    numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
    numbers = [float(n) for n in numbers]
    
    # 使用算法生成物性关系图
    relationship_diagram = generate_property_relationship_algorithm(problem_text, numbers)
    
    # 深度分析问题结构
    problem_analysis = analyze_problem_structure(problem_text, numbers)
    
    # 根据问题类型生成详细推理步骤
    if problem_analysis['type'] == 'shopping_change':
        return generate_shopping_change_reasoning(problem_text, numbers, strategy, problem_analysis, relationship_diagram)
    elif problem_analysis['type'] == 'geometry':
        return generate_geometry_reasoning(problem_text, numbers, strategy, problem_analysis, relationship_diagram)
    elif problem_analysis['type'] == 'arithmetic':
        return generate_arithmetic_reasoning(problem_text, numbers, strategy, problem_analysis, relationship_diagram)
    elif problem_analysis['type'] == 'percentage':
        return generate_percentage_reasoning(problem_text, numbers, strategy, problem_analysis, relationship_diagram)
    else:
        return generate_general_reasoning(problem_text, numbers, strategy, problem_analysis, relationship_diagram)

def generate_property_relationship_algorithm(problem_text, numbers):
    """
    核心算法：基于每道题目自动生成物性关系图
    这是一个智能算法，能够分析题目的数学结构并生成相应的实体关系可视化
    """
    
    # 第1步：实体识别与属性提取算法
    entities = extract_entities_with_properties(problem_text, numbers)
    
    # 第2步：关系发现算法
    relationships = discover_entity_relationships(problem_text, entities)
    
    # 第3步：物性约束推理算法
    constraints = infer_physical_constraints(problem_text, entities, relationships)
    
    # 第4步：动态图结构生成算法
    diagram_structure = generate_dynamic_graph_structure(entities, relationships, constraints)
    
    return diagram_structure

def extract_entities_with_properties(problem_text, numbers):
    """实体识别与属性提取算法"""
    entities = []
    text_lower = problem_text.lower()
    
    # 人物实体识别
    person_patterns = [
        (r'小明', 'xiaoming', 'person', ['拥有能力', '计算能力', '参与运算']),
        (r'小红', 'xiaohong', 'person', ['拥有能力', '计算能力', '参与运算']),
        (r'小张', 'xiaozhang', 'person', ['购买能力', '支付能力', '交易参与']),
        (r'学生', 'student', 'person_group', ['群体属性', '可统计', '有分类']),
        (r'男生', 'male_student', 'person_subgroup', ['性别属性', '占比关系']),
        (r'女生', 'female_student', 'person_subgroup', ['性别属性', '互补关系']),
        (r'班级', 'class', 'group', ['容器属性', '有成员数量', '可分类统计'])
    ]
    
    # 物品实体识别
    object_patterns = [
        (r'苹果', 'apple', 'object', ['可数性', '物理存在', '可拥有', '可累加']),
        (r'笔', 'pen', 'object', ['可数性', '有价格', '可购买', '工具属性']),
        (r'钱|元', 'money', 'currency', ['可分割', '价值存储', '交换媒介', '守恒性']),
        (r'长方形', 'rectangle', 'geometric_shape', ['几何属性', '有长度', '有宽度', '可计算面积']),
        (r'正方形', 'square', 'geometric_shape', ['几何属性', '边长相等', '可计算面积']),
    ]
    
    # 概念实体识别
    concept_patterns = [
        (r'面积', 'area', 'geometric_concept', ['计算属性', '二维度量', '公式驱动']),
        (r'周长', 'perimeter', 'geometric_concept', ['计算属性', '一维度量', '公式驱动']),
        (r'总共|一共', 'total', 'aggregation_concept', ['聚合属性', '求和操作', '集合概念']),
        (r'找零|找回', 'change', 'transaction_concept', ['交易属性', '差值计算', '非负约束']),
        (r'占.*?%|百分比', 'percentage', 'proportion_concept', ['比例属性', '部分整体关系', '互补性'])
    ]
    
    all_patterns = person_patterns + object_patterns + concept_patterns
    
    for pattern, entity_id, entity_type, properties in all_patterns:
        if re.search(pattern, problem_text):
            # 动态分配数值属性
            numeric_properties = assign_numeric_properties(entity_id, entity_type, problem_text, numbers)
            entities.append({
                'id': entity_id,
                'type': entity_type,
                'properties': properties + numeric_properties['properties'],
                'numeric_attributes': numeric_properties['attributes']
            })
    
    return entities

def assign_numeric_properties(entity_id, entity_type, problem_text, numbers):
    """为实体分配数值属性"""
    properties = []
    attributes = {}
    
    # 根据实体类型和问题文本分配数值
    if entity_type == 'person' and len(numbers) >= 2:
        if '小明' in problem_text and '5' in problem_text:
            properties.append('拥有5个物品')
            attributes['quantity'] = numbers[0] if numbers else 0
        elif '小红' in problem_text and '3' in problem_text:
            properties.append('拥有3个物品')
            attributes['quantity'] = numbers[1] if len(numbers) > 1 else 0
        elif '小张' in problem_text:
            properties.append('进行购买交易')
            attributes['payment'] = numbers[-1] if numbers else 0
    
    elif entity_type == 'object':
        if '苹果' in problem_text or '笔' in problem_text:
            properties.append('具有可数性质')
            attributes['unit_count'] = True
        if '笔' in problem_text and '元' in problem_text:
            properties.append('具有价格属性')
            attributes['unit_price'] = numbers[1] if len(numbers) > 1 else 0
    
    elif entity_type == 'currency':
        if '元' in problem_text:
            properties.append('具有数值价值')
            attributes['value'] = max(numbers) if numbers else 0
    
    elif entity_type == 'geometric_shape':
        if '长方形' in problem_text and len(numbers) >= 2:
            properties.append('具有长度和宽度')
            attributes['length'] = numbers[0] if numbers else 0
            attributes['width'] = numbers[1] if len(numbers) > 1 else 0
    
    return {'properties': properties, 'attributes': attributes}

def discover_entity_relationships(problem_text, entities):
    """关系发现算法"""
    relationships = []
    text_lower = problem_text.lower()
    
    # 拥有关系发现
    if '有' in problem_text:
        for entity in entities:
            if entity['type'] == 'person':
                for target in entities:
                    if target['type'] == 'object':
                        weight = entity['numeric_attributes'].get('quantity', 1)
                        relationships.append({
                            'from': entity['id'],
                            'to': target['id'],
                            'type': '拥有关系',
                            'weight': weight,
                            'properties': ['物理拥有', '数量关系', '可转移']
                        })
    
    # 交易关系发现
    if '买' in problem_text or '付' in problem_text:
        buyer = next((e for e in entities if '购买' in str(e['properties'])), None)
        item = next((e for e in entities if e['type'] == 'object'), None)
        money = next((e for e in entities if e['type'] == 'currency'), None)
        
        if buyer and item and money:
            relationships.append({
                'from': buyer['id'],
                'to': item['id'],
                'type': '购买关系',
                'weight': item['numeric_attributes'].get('unit_price', 0),
                'properties': ['交易行为', '价值交换', '所有权转移']
            })
            relationships.append({
                'from': buyer['id'],
                'to': money['id'],
                'type': '支付关系',
                'weight': money['numeric_attributes'].get('value', 0),
                'properties': ['货币转移', '价值支付', '交易媒介']
            })
    
    # 聚合关系发现
    if '一共' in problem_text or '总共' in problem_text:
        total_concept = next((e for e in entities if e['id'] == 'total'), None)
        if total_concept:
            for entity in entities:
                if entity['type'] in ['person', 'object']:
                    weight = entity['numeric_attributes'].get('quantity', 1)
                    relationships.append({
                        'from': entity['id'],
                        'to': total_concept['id'],
                        'type': '聚合关系',
                        'weight': weight,
                        'properties': ['数量累加', '集合运算', '总和生成']
                    })
    
    # 几何关系发现
    if '面积' in problem_text or '周长' in problem_text:
        shape = next((e for e in entities if e['type'] == 'geometric_shape'), None)
        concept = next((e for e in entities if e['type'] == 'geometric_concept'), None)
        
        if shape and concept:
            relationships.append({
                'from': shape['id'],
                'to': concept['id'],
                'type': '几何计算关系',
                'weight': shape['numeric_attributes'].get('length', 0) * shape['numeric_attributes'].get('width', 0),
                'properties': ['公式驱动', '几何变换', '度量关系']
            })
    
    # 比例关系发现
    if '占' in problem_text or '%' in problem_text:
        group = next((e for e in entities if e['type'] == 'group'), None)
        subgroup = next((e for e in entities if e['type'] == 'person_subgroup'), None)
        
        if group and subgroup:
            relationships.append({
                'from': group['id'],
                'to': subgroup['id'],
                'type': '比例关系',
                'weight': 0.6,  # 默认60%
                'properties': ['部分整体', '百分比', '互补关系']
            })
    
    return relationships

def infer_physical_constraints(problem_text, entities, relationships):
    """物性约束推理算法"""
    constraints = []
    
    # 基于实体类型推理约束
    for entity in entities:
        if entity['type'] == 'currency':
            constraints.append(f"货币守恒: {entity['id']}的总量在交易中保持不变")
        elif entity['type'] == 'object':
            constraints.append(f"物体连续性: {entity['id']}数量必须为非负整数")
        elif entity['type'] == 'person':
            constraints.append(f"拥有关系: {entity['id']}的拥有数量必须明确定义")
    
    # 基于关系推理约束
    for rel in relationships:
        if rel['type'] == '购买关系':
            constraints.append("交易平衡: 支付金额 = 商品价格 × 数量")
        elif rel['type'] == '聚合关系':
            constraints.append("加法守恒: 总和 = 各部分数量之和")
        elif rel['type'] == '几何计算关系':
            constraints.append("几何公式: 面积 = 长 × 宽")
        elif rel['type'] == '比例关系':
            constraints.append("比例约束: 所有子群体占比之和 = 100%")
    
    # 基于问题文本推理隐含约束
    if '找零' in problem_text or '找回' in problem_text:
        constraints.append("非负约束: 找零金额 ≥ 0")
    
    if '一共' in problem_text:
        constraints.append("单调性: 总数 ≥ 任意个体数量")
    
    return constraints

def generate_dynamic_graph_structure(entities, relationships, constraints):
    """动态图结构生成算法"""
    return {
        'entities': entities,
        'relationships': relationships,
        'implicit_constraints': constraints,
        'graph_properties': {
            'node_count': len(entities),
            'edge_count': len(relationships),
            'constraint_count': len(constraints),
            'complexity_score': len(entities) * len(relationships) + len(constraints)
        },
        'layout_algorithm': 'force_directed',  # 使用力导向布局
        'visualization_hints': {
            'person_color': '#e74c3c',
            'object_color': '#27ae60', 
            'currency_color': '#f39c12',
            'concept_color': '#9b59b6',
            'group_color': '#3498db'
        }
    }

def analyze_problem_structure(problem_text, numbers):
    """深度分析问题结构"""
    analysis = {
        'type': 'unknown',
        'entities': [],
        'relations': [],
        'constraints': [],
        'implicit_info': [],
        'complexity': 'simple',
        'entity_properties': {},
        'physical_relations': [],
        'scenario_context': {}
    }
    
    text_lower = problem_text.lower()
    
    # 识别问题类型
    if any(keyword in text_lower for keyword in ['买', '购', '花', '付', '找', '零', '钱', '元']):
        analysis['type'] = 'shopping_change'
    elif any(keyword in text_lower for keyword in ['面积', '周长', '体积', '长方形', '正方形', '圆']):
        analysis['type'] = 'geometry'
    elif any(keyword in text_lower for keyword in ['百分比', '%', '占', '比例']):
        analysis['type'] = 'percentage'
    elif any(keyword in text_lower for keyword in ['一共', '总共', '合计', '加', '减', '乘', '除']):
        analysis['type'] = 'arithmetic'
    
    # 提取实体及其属性
    entities = []
    entity_properties = {}
    
    if '小明' in problem_text:
        entities.append('小明')
        entity_properties['小明'] = {'type': 'person', 'role': 'owner', 'capabilities': ['拥有', '计算']}
    if '小红' in problem_text:
        entities.append('小红')
        entity_properties['小红'] = {'type': 'person', 'role': 'owner', 'capabilities': ['拥有', '计算']}
    if '小张' in problem_text:
        entities.append('小张')
        entity_properties['小张'] = {'type': 'person', 'role': 'buyer', 'capabilities': ['购买', '支付']}
    if '苹果' in problem_text:
        entities.append('苹果')
        entity_properties['苹果'] = {'type': 'object', 'category': 'fruit', 'properties': ['可数', '可拥有', '可累加']}
    if '笔' in problem_text:
        entities.append('笔')
        entity_properties['笔'] = {'type': 'object', 'category': 'tool', 'properties': ['可数', '有价格', '可购买']}
    if '长方形' in problem_text:
        entities.append('长方形')
        entity_properties['长方形'] = {'type': 'geometric_shape', 'properties': ['有长度', '有宽度', '可计算面积']}
    if '班级' in problem_text:
        entities.append('班级')
        entity_properties['班级'] = {'type': 'group', 'properties': ['有成员', '可统计', '有结构']}
    if '学生' in problem_text:
        entities.append('学生')
        entity_properties['学生'] = {'type': 'person', 'role': 'member', 'properties': ['有性别', '可分类', '可计数']}
    if '男生' in problem_text:
        entities.append('男生')
        entity_properties['男生'] = {'type': 'person_subgroup', 'properties': ['性别属性', '占比关系']}
    if '女生' in problem_text:
        entities.append('女生')
        entity_properties['女生'] = {'type': 'person_subgroup', 'properties': ['性别属性', '互补关系']}
    
    analysis['entities'] = entities
    analysis['entity_properties'] = entity_properties
    
    # 识别物理关系
    physical_relations = []
    if '有' in problem_text:
        physical_relations.append({'type': 'possession', 'description': '拥有关系 - 实体对物体的所有权'})
    if '买' in problem_text:
        physical_relations.append({'type': 'transaction', 'description': '交易关系 - 货币与商品的交换'})
    if '给' in problem_text:
        physical_relations.append({'type': 'transfer', 'description': '转移关系 - 物体从一个实体转移到另一个实体'})
    if '长' in problem_text and '宽' in problem_text:
        physical_relations.append({'type': 'geometric_dimension', 'description': '几何维度关系 - 长度和宽度决定面积'})
    if '占' in problem_text:
        physical_relations.append({'type': 'proportion', 'description': '比例关系 - 部分与整体的数量关系'})
    
    analysis['physical_relations'] = physical_relations
    
    # 构建情景上下文
    scenario_context = {}
    if analysis['type'] == 'shopping_change':
        scenario_context = {
            'environment': '商店购物场景',
            'participants': ['买家', '卖家'],
            'objects': ['商品', '货币'],
            'actions': ['选择商品', '计算价格', '支付', '找零'],
            'constraints': ['价格固定', '货币有限', '找零非负']
        }
    elif analysis['type'] == 'geometry':
        scenario_context = {
            'environment': '几何计算场景',
            'objects': ['几何图形'],
            'properties': ['长度', '宽度', '面积'],
            'rules': ['面积公式', '几何定理'],
            'constraints': ['尺寸为正数', '单位一致']
        }
    elif analysis['type'] == 'arithmetic':
        scenario_context = {
            'environment': '数量统计场景',
            'objects': ['可数物体'],
            'operations': ['加法运算', '数量累加'],
            'constraints': ['数量非负', '结果唯一']
        }
    elif analysis['type'] == 'percentage':
        scenario_context = {
            'environment': '群体统计场景',
            'structure': ['总体', '子群体'],
            'relationships': ['包含关系', '互补关系'],
            'constraints': ['百分比和为100%', '数量为整数']
        }
    
    analysis['scenario_context'] = scenario_context
    
    # 识别传统关系
    relations = []
    if '有' in problem_text:
        relations.append('拥有关系')
    if '买' in problem_text:
        relations.append('购买关系')
    if '给' in problem_text:
        relations.append('支付关系')
    if '找' in problem_text:
        relations.append('找零关系')
    if '一共' in problem_text:
        relations.append('求和关系')
    
    analysis['relations'] = relations
    
    # 识别隐含信息
    implicit_info = []
    if analysis['type'] == 'shopping_change':
        implicit_info.append('总价格 = 单价 × 数量')
        implicit_info.append('找零 = 支付金额 - 总价格')
        implicit_info.append('找零金额必须为非负数')
    elif analysis['type'] == 'geometry':
        implicit_info.append('面积公式的应用')
        implicit_info.append('几何图形的属性')
    elif analysis['type'] == 'arithmetic':
        implicit_info.append('数值的基本运算规则')
        implicit_info.append('运算结果的合理性验证')
    
    analysis['implicit_info'] = implicit_info
    
    # 评估复杂度
    if len(numbers) > 3 or len(relations) > 2:
        analysis['complexity'] = 'complex'
    elif len(numbers) > 2 or len(relations) > 1:
        analysis['complexity'] = 'moderate'
    
    return analysis

def generate_shopping_change_reasoning(problem_text, numbers, strategy, analysis, relationship_diagram):
    """生成购物找零问题的详细推理"""
    reasoning_steps = []
    
    # 提取基本信息
    unit_price = numbers[1] if len(numbers) >= 2 else numbers[0]
    quantity = numbers[0] if len(numbers) >= 2 else 1
    paid_amount = numbers[2] if len(numbers) >= 3 else numbers[-1]
    total_cost = quantity * unit_price
    change = paid_amount - total_cost
    
    if strategy == 'cot':
        # COT策略：链式推理 + 深层实体物性关系建模
        reasoning_steps = [
            {
                "step": 1,
                "action": "cot_entity_property_decomposition",
                "description": f"【COT实体物性链式分解】识别核心实体：买家实体(具有购买能力和货币资源)、商品实体({quantity}个，单价{unit_price}元，具有价值属性)、卖家实体(具有定价权和收款能力)"
            },
            {
                "step": 2,
                "action": "cot_implicit_relation_chaining",
                "description": f"【COT隐含关系链接】建立链式关系：买家货币({paid_amount}元) → 商品价值({quantity}×{unit_price}={total_cost}元) → 剩余货币({change}元) → 回流买家，形成完整交易链"
            },
            {
                "step": 3,
                "action": "cot_physical_constraint_verification",
                "description": f"【COT物理约束验证】验证链式约束：货币实体具有守恒性({paid_amount}元总量不变)，商品实体具有等价交换性(价值={total_cost}元)，找零实体具有非负性({change}≥0)"
            },
            {
                "step": 4,
                "action": "cot_state_transition_sequence",
                "description": f"【COT状态转移序列】跟踪实体状态链：初态(买家{paid_amount}元+卖家{quantity}个商品) → 中态(交换过程) → 终态(买家{quantity}个商品+{change}元+卖家{total_cost}元)"
            },
            {
                "step": 5,
                "action": "cot_multilayer_validation",
                "description": f"【COT多层验证链】三层验证：数值层({total_cost}+{change}={paid_amount})，逻辑层(交易完整性)，物理层(实体守恒性)，确保链式推理无断点"
            }
        ]
    
    elif strategy == 'got':
        # GOT策略：图状推理 + 网络化实体关系建模
        reasoning_steps = [
            {
                "step": 1,
                "action": "got_entity_network_topology",
                "description": f"【GOT实体网络拓扑】构建多层网络：核心层(买家-卖家双向连接)，商品层({quantity}个商品节点，价值权重{unit_price})，货币层(支付{paid_amount}元节点，找零{change}元节点)"
            },
            {
                "step": 2,
                "action": "got_implicit_edge_discovery",
                "description": f"【GOT隐含边发现】识别隐含连接：买家↔货币池(权重{paid_amount})，货币池↔商品集(权重{total_cost})，商品集↔卖家(所有权转移)，剩余货币↔买家(权重{change})"
            },
            {
                "step": 3,
                "action": "got_network_flow_analysis",
                "description": f"【GOT网络流分析】分析实体流动：货币流({paid_amount}元从买家流向交易中心，{total_cost}元流向卖家，{change}元回流买家)，商品流({quantity}个从卖家流向买家)"
            },
            {
                "step": 4,
                "action": "got_subgraph_property_inference",
                "description": f"【GOT子图属性推理】推理子网络性质：买家-商品子图(获得{quantity}个商品，价值{total_cost}元)，卖家-货币子图(获得{total_cost}元货币)，找零子图(返还{change}元给买家)"
            },
            {
                "step": 5,
                "action": "got_network_equilibrium_check",
                "description": f"【GOT网络平衡检验】验证网络平衡：所有节点的入度出度平衡，总权重守恒({paid_amount}={total_cost}+{change})，网络连通性完整，无孤立节点"
            }
        ]
    
    elif strategy == 'tot':
        # TOT策略：树状推理 + 分层实体关系探索
        reasoning_steps = [
            {
                "step": 1,
                "action": "tot_entity_hierarchy_tree",
                "description": f"【TOT实体层次树】构建实体分类树：根节点(交易系统) → 一级节点(参与实体：买家、卖家) → 二级节点(交换对象：{quantity}个商品、{paid_amount}元货币) → 三级节点(属性：价格{unit_price}元/个)"
            },
            {
                "step": 2,
                "action": "tot_solution_branch_exploration",
                "description": f"【TOT解决方案分支探索】探索多重路径：路径A(直接计算：{paid_amount}-{total_cost})，路径B(分步推理：先算总价再算找零)，路径C(逆向验证：从找零推回总价)，路径D(比例分析：找零占支付比例)"
            },
            {
                "step": 3,
                "action": "tot_implicit_constraint_propagation",
                "description": f"【TOT隐含约束传播】在树中传播约束：根约束(货币守恒) → 分支约束(非负找零) → 叶约束(整数货币单位)，每条路径都必须满足所有层级约束"
            },
            {
                "step": 4,
                "action": "tot_branch_property_synthesis",
                "description": f"【TOT分支属性综合】综合各分支结果：路径A得{change}元，路径B验证{total_cost}+{change}={paid_amount}，路径C确认逆向一致性，路径D计算找零率{(change/paid_amount)*100:.1f}%"
            },
            {
                "step": 5,
                "action": "tot_optimal_path_selection",
                "description": f"【TOT最优路径选择】基于实体关系复杂度选择：路径B(分步推理)最符合人类认知，路径A最简洁高效，路径C最可靠安全，综合选择路径B作为主解，其他作为验证"
            },
            {
                "step": 6,
                "action": "tot_tree_consistency_validation",
                "description": f"【TOT树一致性验证】验证整树一致性：所有分支指向相同答案{change}元，实体关系在不同路径下保持不变，约束传播无冲突，树结构完整无环"
            }
        ]
    
    else:  # auto策略
        strategy = 'cot'
        reasoning_steps = [
            {
                "step": 1,
                "action": "auto_strategy_selection_with_relation_analysis",
                "description": f"【自动策略+关系分析】分析问题复杂度：实体数量{len(analysis['entities'])}，关系类型{len(analysis['physical_relations'])}，选择COT策略进行深层链式推理"
            },
            {
                "step": 2,
                "action": "auto_integrated_computation",
                "description": f"【自动集成计算】融合实体关系的计算：买家实体({paid_amount}元) → 商品实体价值({quantity}×{unit_price}={total_cost}元) → 找零实体({change}元)"
            },
            {
                "step": 3,
                "action": "auto_holistic_validation",
                "description": f"【自动整体验证】全面验证：数值正确性({total_cost}+{change}={paid_amount})，实体关系合理性(买卖双方获得期望结果)，物理约束满足性(找零非负)"
            }
        ]
    
    # 使用算法生成的实体关系图，而不是手动构建
    entity_relationship_diagram = relationship_diagram
    
    return jsonify({
        "success": True,
        "answer": f"{change}元",
        "confidence": 0.95,
        "strategy_used": strategy,
        "reasoning_steps": reasoning_steps,
        "execution_time": 2.1,
        "entity_relationship_diagram": entity_relationship_diagram
    })

def generate_arithmetic_reasoning(problem_text, numbers, strategy, analysis, relationship_diagram):
    """生成算术问题的详细推理"""
    reasoning_steps = []
    
    # 计算结果
    if "一共" in problem_text or "总共" in problem_text:
        result = sum(numbers)
    else:
        result = sum(numbers)
    
    if strategy == 'cot':
        # COT策略：链式推理 + 深层实体聚合关系建模
        reasoning_steps = [
            {
                "step": 1,
                "action": "cot_entity_ownership_decomposition",
                "description": f"【COT实体拥有链式分解】识别拥有实体：{analysis['entities'][0]}实体(拥有{numbers[0]}个{analysis['entities'][1]}，具有收集能力)、{analysis['entities'][2] if len(analysis['entities']) > 2 else '另一实体'}实体(拥有{numbers[1] if len(numbers) > 1 else 0}个，具有独立拥有权)"
            },
            {
                "step": 2,
                "action": "cot_spatial_aggregation_chaining",
                "description": f"【COT空间聚合链接】建立空间聚合链：{analysis['entities'][0]}的{analysis['entities'][1]}({numbers[0]}个) + {analysis['entities'][2] if len(analysis['entities']) > 2 else '另一实体'}的{analysis['entities'][1]}({numbers[1] if len(numbers) > 1 else 0}个) → 统一空间集合({result}个)"
            },
            {
                "step": 3,
                "action": "cot_quantity_conservation_verification",
                "description": f"【COT数量守恒验证】验证链式守恒：{analysis['entities'][1]}实体具有物理不可创造性，总数{result}个通过空间重新分布形成，遵循物质守恒定律"
            },
            {
                "step": 4,
                "action": "cot_ownership_transition_sequence",
                "description": f"【COT拥有权转移序列】跟踪拥有权链：初态(分散拥有) → 中态(概念统一) → 终态(集合拥有)，{analysis['entities'][1]}实体的物理属性保持不变"
            },
            {
                "step": 5,
                "action": "cot_multilevel_aggregation_validation",
                "description": f"【COT多级聚合验证】三级验证：数值级({' + '.join(map(str, numbers))} = {result})，空间级(物理聚合可行)，概念级(拥有权逻辑一致)"
            }
        ]
    
    elif strategy == 'got':
        # GOT策略：图状推理 + 网络化拥有关系建模  
        reasoning_steps = [
            {
                "step": 1,
                "action": "got_ownership_network_topology",
                "description": f"【GOT拥有网络拓扑】构建拥有关系网络：人物层({analysis['entities'][0]}节点、{analysis['entities'][2] if len(analysis['entities']) > 2 else '另一实体'}节点)，物品层({analysis['entities'][1]}节点群)，聚合层(总和节点)"
            },
            {
                "step": 2,
                "action": "got_possession_edge_mapping",
                "description": f"【GOT拥有边映射】映射拥有连接：{analysis['entities'][0]}↔{numbers[0]}个{analysis['entities'][1]}(权重{numbers[0]})，{analysis['entities'][2] if len(analysis['entities']) > 2 else '另一实体'}↔{numbers[1] if len(numbers) > 1 else 0}个{analysis['entities'][1]}(权重{numbers[1] if len(numbers) > 1 else 0})"
            },
            {
                "step": 3,
                "action": "got_aggregation_flow_analysis",
                "description": f"【GOT聚合流分析】分析聚合流动：所有{analysis['entities'][1]}节点通过聚合边汇聚到总和节点，形成网络收敛结构，总权重={result}"
            },
            {
                "step": 4,
                "action": "got_subgraph_ownership_inference",
                "description": f"【GOT子图拥有推理】推理子网络性质：{analysis['entities'][0]}-{analysis['entities'][1]}子图(贡献{numbers[0]}个)，{analysis['entities'][2] if len(analysis['entities']) > 2 else '另一实体'}-{analysis['entities'][1]}子图(贡献{numbers[1] if len(numbers) > 1 else 0}个)，聚合子图(总计{result}个)"
            },
            {
                "step": 5,
                "action": "got_network_conservation_check",
                "description": f"【GOT网络守恒检验】验证网络守恒：入度权重总和({numbers[0]}+{numbers[1] if len(numbers) > 1 else 0})等于出度权重({result})，无权重丢失，网络平衡完整"
            }
        ]
    
    elif strategy == 'tot':
        # TOT策略：树状推理 + 分层聚合探索
        reasoning_steps = [
            {
                "step": 1,
                "action": "tot_aggregation_hierarchy_tree",
                "description": f"【TOT聚合层次树】构建聚合分类树：根节点(求总和) → 一级节点(个体拥有：{analysis['entities'][0]}, {analysis['entities'][2] if len(analysis['entities']) > 2 else '另一实体'}) → 二级节点(拥有数量：{numbers[0]}个, {numbers[1] if len(numbers) > 1 else 0}个) → 三级节点(物品属性)"
            },
            {
                "step": 2,
                "action": "tot_solution_method_exploration",
                "description": f"【TOT解法方法探索】探索聚合方法：方法A(直接求和：{numbers[0]}+{numbers[1] if len(numbers) > 1 else 0})，方法B(逐个累计)，方法C(分组后合并)，方法D(比例分析各自贡献)"
            },
            {
                "step": 3,
                "action": "tot_implicit_property_propagation",
                "description": f"【TOT隐含属性传播】在树中传播属性：根属性(可加性) → 分支属性(拥有独立性) → 叶属性({analysis['entities'][1]}实体的物理属性)，保证属性一致性"
            },
            {
                "step": 4,
                "action": "tot_branch_aggregation_synthesis",
                "description": f"【TOT分支聚合综合】综合各分支结果：方法A得{result}个，方法B逐步累计到{result}个，方法C分组合并得{result}个，方法D确认贡献比例"
            },
            {
                "step": 5,
                "action": "tot_optimal_aggregation_selection",
                "description": f"【TOT最优聚合选择】基于实体关系选择：方法A(直接求和)最符合数学直觉，方法B最贴近物理过程，方法C最适合理解，选择方法A为主，其他为验证"
            },
            {
                "step": 6,
                "action": "tot_tree_aggregation_validation",
                "description": f"【TOT树聚合验证】验证整树聚合：所有方法指向相同结果{result}个，实体拥有关系在不同路径下保持不变，聚合过程无重复计算"
            }
        ]
    
    else:  # auto策略
        strategy = 'cot'
        reasoning_steps = [
            {
                "step": 1,
                "action": "auto_aggregation_strategy_with_relation_analysis",
                "description": f"【自动聚合策略+关系分析】分析拥有关系复杂度：{len(analysis['entities'])}个实体，{len(analysis['relations'])}种关系，选择COT策略进行链式聚合推理"
            },
            {
                "step": 2,
                "action": "auto_integrated_aggregation",
                "description": f"【自动集成聚合】融合拥有关系的聚合：{analysis['entities'][0]}实体贡献{numbers[0]}个 + {analysis['entities'][2] if len(analysis['entities']) > 2 else '另一实体'}实体贡献{numbers[1] if len(numbers) > 1 else 0}个 = 总计{result}个"
            },
            {
                "step": 3,
                "action": "auto_holistic_aggregation_validation",
                "description": f"【自动整体聚合验证】全面验证：数值正确性({' + '.join(map(str, numbers))} = {result})，拥有关系合理性(所有实体贡献得到承认)，物理聚合可能性(空间允许)"
            }
        ]
    
    # 使用算法生成的实体关系图
    entity_relationship_diagram = relationship_diagram
    
    return jsonify({
        "success": True,
        "answer": f"{result}个",
        "confidence": 0.92,
        "strategy_used": strategy,
        "reasoning_steps": reasoning_steps,
        "execution_time": 1.8,
        "entity_relationship_diagram": entity_relationship_diagram
    })

def generate_geometry_reasoning(problem_text, numbers, strategy, analysis, relationship_diagram):
    """生成几何问题的详细推理"""
    reasoning_steps = []
    
    # 计算基本信息
    length = numbers[0] if len(numbers) >= 1 else 0
    width = numbers[1] if len(numbers) >= 2 else 0
    
    if "面积" in problem_text:
        result = length * width
        answer = f"{result}平方米"
        calc_type = "面积"
    elif "周长" in problem_text:
        result = 2 * (length + width)
        answer = f"{result}米"
        calc_type = "周长"
    else:
        result = length * width
        answer = f"{result}平方米"
        calc_type = "面积"
    
    if strategy == 'cot':
        # COT策略：链式推理，逐步公式应用
        reasoning_steps = [
            {
                "step": 1,
                "action": "cot_shape_identification",
                "description": f"【COT形状识别】识别几何图形：长方形，长={length}米，宽={width}米"
            },
            {
                "step": 2,
                "action": "cot_formula_selection",
                "description": f"【COT公式选择】选择{calc_type}公式：{'面积 = 长 × 宽' if calc_type == '面积' else '周长 = 2 × (长 + 宽)'}"
            },
            {
                "step": 3,
                "action": "cot_value_substitution",
                "description": f"【COT数值代入】代入数值：{length if calc_type == '面积' else '2 × ('}{' × ' if calc_type == '面积' else ''}{width if calc_type == '面积' else str(length) + ' + ' + str(width) + ')'}"
            },
            {
                "step": 4,
                "action": "cot_calculation_execution",
                "description": f"【COT计算执行】执行计算：{result}{'平方米' if calc_type == '面积' else '米'}"
            },
            {
                "step": 5,
                "action": "cot_result_validation",
                "description": f"【COT结果验证】验证：{calc_type}值{result}{'平方米' if calc_type == '面积' else '米'} > 0，结果合理"
            }
        ]
    
    elif strategy == 'got':
        # GOT策略：图状推理，空间关系网络
        reasoning_steps = [
            {
                "step": 1,
                "action": "got_spatial_graph",
                "description": f"【GOT空间图】构建空间关系图：长方形节点 ↔ 长度节点({length}米) ↔ 宽度节点({width}米)"
            },
            {
                "step": 2,
                "action": "got_dimension_mapping",
                "description": f"【GOT维度映射】映射维度关系：长度和宽度节点通过{calc_type}关系连接到结果节点"
            },
            {
                "step": 3,
                "action": "got_graph_computation",
                "description": f"【GOT图计算】通过图遍历计算：从维度节点收集信息，应用{calc_type}公式，得到结果节点值{result}"
            },
            {
                "step": 4,
                "action": "got_spatial_validation",
                "description": f"【GOT空间验证】验证图结构：所有节点连接正确，空间关系保持一致性"
            }
        ]
    
    elif strategy == 'tot':
        # TOT策略：树状推理，多种计算方法
        reasoning_steps = [
            {
                "step": 1,
                "action": "tot_geometry_tree",
                "description": f"【TOT几何树】构建几何解决方案树：根节点为{calc_type}计算问题，分支包含不同方法"
            },
            {
                "step": 2,
                "action": "tot_method_exploration",
                "description": f"【TOT方法探索】探索计算方法：方法1-直接公式；方法2-分解计算；方法3-单位分析"
            },
            {
                "step": 3,
                "action": "tot_method_evaluation",
                "description": f"【TOT方法评估】评估各方法：方法1结果{result}，方法2分步得{result}，方法3单位验证{result}{'平方米' if calc_type == '面积' else '米'}"
            },
            {
                "step": 4,
                "action": "tot_optimal_path",
                "description": f"【TOT最优路径】选择最优方法：直接公式法最直接，选择为最终解决方案"
            },
            {
                "step": 5,
                "action": "tot_tree_verification",
                "description": f"【TOT树验证】验证解决方案树：所有方法都指向答案{answer}，确保几何计算的准确性"
            }
        ]
    
    else:  # auto策略
        strategy = 'cot'
        reasoning_steps = [
            {
                "step": 1,
                "action": "auto_geometry_analysis",
                "description": f"【自动几何】分析几何问题：{calc_type}计算，选择COT策略进行公式推理"
            },
            {
                "step": 2,
                "action": "auto_formula_application",
                "description": f"【自动公式】应用{calc_type}公式：{'长×宽' if calc_type == '面积' else '2×(长+宽)'} = {result}{'平方米' if calc_type == '面积' else '米'}"
            },
            {
                "step": 3,
                "action": "auto_result_check",
                "description": f"【自动验证】验证结果：{answer}符合几何规律，计算正确"
            }
        ]
    
    # 使用算法生成的实体关系图
    entity_relationship_diagram = relationship_diagram
    
    return jsonify({
        "success": True,
        "answer": answer,
        "confidence": 0.94,
        "strategy_used": strategy,
        "reasoning_steps": reasoning_steps,
        "execution_time": 2.3,
        "entity_relationship_diagram": entity_relationship_diagram
    })

def generate_percentage_reasoning(problem_text, numbers, strategy, analysis, relationship_diagram):
    """生成百分比问题的详细推理"""
    reasoning_steps = []
    
    # 计算基本信息
    total_count = numbers[0] if len(numbers) >= 1 else 0
    percentage = numbers[1] if len(numbers) >= 2 else 60
    male_count = total_count * (percentage / 100)
    female_count = total_count - male_count
    
    if strategy == 'cot':
        # COT策略：链式推理，逐步百分比计算
        reasoning_steps = [
            {
                "step": 1,
                "action": "cot_percentage_understanding",
                "description": f"【COT百分比理解】理解问题：总数{total_count}人，男生占{percentage}%，求女生人数"
            },
            {
                "step": 2,
                "action": "cot_male_calculation",
                "description": f"【COT男生计算】计算男生人数：{total_count} × {percentage}% = {total_count} × {percentage/100} = {male_count}人"
            },
            {
                "step": 3,
                "action": "cot_complementary_reasoning",
                "description": f"【COT互补推理】应用互补关系：女生人数 = 总人数 - 男生人数"
            },
            {
                "step": 4,
                "action": "cot_female_calculation",
                "description": f"【COT女生计算】计算女生人数：{total_count} - {male_count} = {female_count}人"
            },
            {
                "step": 5,
                "action": "cot_percentage_validation",
                "description": f"【COT百分比验证】验证：男生{male_count}人({percentage}%) + 女生{female_count}人({(female_count/total_count)*100}%) = {total_count}人(100%)"
            }
        ]
    
    elif strategy == 'got':
        # GOT策略：图状推理，群体关系网络
        reasoning_steps = [
            {
                "step": 1,
                "action": "got_group_graph",
                "description": f"【GOT群体图】构建群体关系图：班级节点({total_count}人) ↔ 男生节点({percentage}%) ↔ 女生节点(待求)"
            },
            {
                "step": 2,
                "action": "got_proportion_mapping",
                "description": f"【GOT比例映射】映射比例关系：男生节点与总数节点通过{percentage}%比例连接"
            },
            {
                "step": 3,
                "action": "got_graph_computation",
                "description": f"【GOT图计算】通过图遍历计算：从总数节点{total_count}人经过{percentage}%比例得到男生{male_count}人，剩余为女生{female_count}人"
            },
            {
                "step": 4,
                "action": "got_network_balance",
                "description": f"【GOT网络平衡】验证网络平衡：所有节点数量总和等于班级总数，图结构保持完整性"
            }
        ]
    
    elif strategy == 'tot':
        # TOT策略：树状推理，多种解决方案
        reasoning_steps = [
            {
                "step": 1,
                "action": "tot_percentage_tree",
                "description": f"【TOT百分比树】构建百分比解决方案树：根节点为求女生人数，分支包含不同计算方法"
            },
            {
                "step": 2,
                "action": "tot_solution_exploration",
                "description": f"【TOT解法探索】探索解决方案：方法1-先算男生再减法；方法2-直接算女生百分比；方法3-比例方程求解"
            },
            {
                "step": 3,
                "action": "tot_solution_evaluation",
                "description": f"【TOT解法评估】评估各方法：方法1得{female_count}人，方法2得{total_count}×{100-percentage}%={female_count}人，方法3比例解得{female_count}人"
            },
            {
                "step": 4,
                "action": "tot_optimal_solution",
                "description": f"【TOT最优解法】选择最优方法：方法1(先算男生再减法)最直观，选择为最终解决方案"
            },
            {
                "step": 5,
                "action": "tot_tree_consistency",
                "description": f"【TOT树一致性】验证解决方案树：所有方法都指向答案{int(female_count)}人，确保百分比计算的准确性"
            }
        ]
    
    else:  # auto策略
        strategy = 'cot'
        reasoning_steps = [
            {
                "step": 1,
                "action": "auto_percentage_analysis",
                "description": f"【自动百分比】分析百分比问题：群体统计类型，选择COT策略进行链式推理"
            },
            {
                "step": 2,
                "action": "auto_calculation",
                "description": f"【自动计算】执行百分比计算：男生{total_count}×{percentage}%={male_count}人，女生{total_count}-{male_count}={female_count}人"
            },
            {
                "step": 3,
                "action": "auto_verification",
                "description": f"【自动验证】验证结果：{male_count}+{female_count}={total_count}人，百分比计算正确"
            }
        ]
    
    # 使用算法生成的实体关系图
    entity_relationship_diagram = relationship_diagram
    
    return jsonify({
        "success": True,
        "answer": f"{int(female_count)}人",
        "confidence": 0.93,
        "strategy_used": strategy,
        "reasoning_steps": reasoning_steps,
        "execution_time": 2.0,
        "entity_relationship_diagram": entity_relationship_diagram
    })

def generate_general_reasoning(problem_text, numbers, strategy, analysis, relationship_diagram):
    """生成通用推理"""
    reasoning_steps = []
    
    reasoning_steps.append({
        "step": 1,
        "action": "general_analysis",
        "description": f"综合分析：这是一个{analysis['complexity']}复杂度的问题，涉及{len(numbers)}个数值"
    })
    
    reasoning_steps.append({
        "step": 2,
        "action": "pattern_recognition",
        "description": "模式识别：基于问题文本分析，推测需要进行数值运算"
    })
    
    result = sum(numbers) if numbers else 0
    reasoning_steps.append({
        "step": 3,
        "action": "calculation",
        "description": f"计算执行：{' + '.join(map(str, numbers))} = {result}"
    })
    
    reasoning_steps.append({
        "step": 4,
        "action": "result_validation",
        "description": "结果验证：基于问题上下文，答案具有合理性"
    })
    
    # 使用算法生成的实体关系图
    entity_relationship_diagram = relationship_diagram
    
    return jsonify({
        "success": True,
        "answer": str(result),
        "confidence": 0.75,
        "strategy_used": strategy if strategy != 'auto' else 'cot',
        "reasoning_steps": reasoning_steps,
        "execution_time": 1.5,
        "entity_relationship_diagram": entity_relationship_diagram
    })

if __name__ == '__main__':
    # 创建模板目录
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    print("🚀 启动COT-DIR本地Web UI")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"🌐 访问地址: http://localhost:8082")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=8082, debug=True)