#!/usr/bin/env python3
"""
真实推理引擎API服务器
集成COT-DIR推理引擎，提供完整的数学推理功能
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

from flask import Flask, request, jsonify
from flask_cors import CORS

# 导入独立COT-DIR推理器
from standalone_reasoning_api import solve_mathematical_problem, cotdir_reasoner

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用跨域请求支持

# 全局推理引擎状态
reasoning_engine_status = "initialized"

class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self):
        # 尝试找到数据目录
        current_dir = Path(__file__).parent
        # 查找项目根目录的Data文件夹
        self.data_dir = None
        for parent in [current_dir, current_dir.parent, current_dir.parent.parent]:
            data_path = parent / "Data"
            if data_path.exists():
                self.data_dir = data_path
                break
        
        if not self.data_dir:
            logger.warning("未找到Data数据集目录")
        else:
            logger.info(f"找到数据集目录: {self.data_dir}")
    
    def get_available_datasets(self) -> List[str]:
        """获取可用的数据集列表"""
        if not self.data_dir or not self.data_dir.exists():
            return []
        
        datasets = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != 'processing':
                datasets.append(item.name)
        
        return sorted(datasets)
    
    def load_dataset_sample(self, dataset_name: str, count: int = 5) -> List[Dict]:
        """从指定数据集加载样本"""
        if not self.data_dir:
            return []
        
        dataset_path = self.data_dir / dataset_name
        if not dataset_path.exists():
            return []
        
        # 查找数据文件
        json_files = list(dataset_path.glob("*.json"))
        jsonl_files = list(dataset_path.glob("*.jsonl"))
        
        if not json_files and not jsonl_files:
            return []
        
        # 优先使用json文件
        if json_files:
            data_file = json_files[0]
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # 随机选择样本
                        sample_size = min(count, len(data))
                        samples = random.sample(data, sample_size)
                        return samples
            except Exception as e:
                logger.error(f"读取数据集 {dataset_name} 出错: {e}")
        
        # 使用jsonl文件
        elif jsonl_files:
            data_file = jsonl_files[0]
            try:
                samples = []
                with open(data_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # 随机选择样本
                    sample_lines = random.sample(lines, min(count, len(lines)))
                    for line in sample_lines:
                        samples.append(json.loads(line.strip()))
                return samples
            except Exception as e:
                logger.error(f"读取数据集 {dataset_name} 出错: {e}")
        
        return []

    def find_problem_by_id(self, dataset_name: str, problem_id: str) -> Optional[Dict]:
        """通过ID在数据集中查找特定问题"""
        if not self.data_dir:
            return None
        
        dataset_path = self.data_dir / dataset_name
        if not dataset_path.exists():
            return None
        
        # 查找数据文件
        json_files = list(dataset_path.glob("*.json"))
        jsonl_files = list(dataset_path.glob("*.jsonl"))
        
        # 搜索JSON文件
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for problem in data:
                            if problem.get('id') == problem_id:
                                return problem
            except Exception as e:
                logger.error(f"搜索问题时读取文件 {json_file} 出错: {e}")
        
        # 搜索JSONL文件
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        problem = json.loads(line.strip())
                        if problem.get('id') == problem_id:
                            return problem
            except Exception as e:
                logger.error(f"搜索问题时读取文件 {jsonl_file} 出错: {e}")
        
        return None

# 创建数据集加载器实例
dataset_loader = DatasetLoader()

def initialize_reasoning_engine():
    """初始化真实推理引擎"""
    global reasoning_engine_status
    
    try:
        logger.info("正在初始化独立COT-DIR推理引擎...")
        
        # 测试推理器
        test_result = cotdir_reasoner.health_check()
        if test_result['status'] == 'healthy':
            logger.info("✅ 独立COT-DIR推理引擎初始化成功")
            reasoning_engine_status = "healthy"
            return True
        else:
            logger.error("❌ COT-DIR推理引擎健康检查失败")
            reasoning_engine_status = "error"
            return False
            
    except Exception as e:
        logger.error(f"❌ 推理引擎初始化出错: {e}")
        reasoning_engine_status = "error"
        return False

def fallback_reasoning(problem_text: str) -> Dict[str, Any]:
    """
    降级推理方案（当真实引擎不可用时）
    提供基础的模式匹配推理
    """
    import re
    
    entities = []
    relations = []
    
    # 基础实体识别
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', problem_text)
    for i, num in enumerate(numbers):
        entities.append({
            'id': f'num_{i}',
            'text': num,
            'type': 'number',
            'value': float(num),
            'confidence': 0.90
        })
    
    # 基础关系识别
    if 'give' in problem_text.lower() or 'has' in problem_text.lower():
        relations.append({
            'id': 'rel_ownership',
            'source': 'person',
            'target': 'object',
            'type': 'ownership',
            'description': 'Ownership relationship detected',
            'confidence': 0.80
        })
    
    # 生成推理步骤
    steps = [
        {
            'id': 'step_1',
            'step': 1,
            'type': 'entity_recognition',
            'description': f'识别了 {len(entities)} 个数学实体',
            'confidence': 0.85,
            'timestamp': int(time.time() * 1000),
            'details': {'entities': entities[:3]}
        },
        {
            'id': 'step_2', 
            'step': 2,
            'type': 'relation_discovery',
            'description': f'发现了 {len(relations)} 个实体关系',
            'confidence': 0.80,
            'timestamp': int(time.time() * 1000) + 100,
            'details': {'relations': relations}
        }
    ]
    
    # 简单答案生成
    answer = "解决方案已找到"
    confidence = 0.75
    
    # 简单算术识别
    if re.search(r'(\d+)\s*[-]\s*(\d+)', problem_text):
        match = re.search(r'(\d+)\s*[-]\s*(\d+)', problem_text)
        if match:
            result = int(match.group(1)) - int(match.group(2))
            answer = str(result)
            confidence = 0.95
    
    return {
        'answer': answer,
        'final_answer': answer,
        'confidence': confidence,
        'explanation': f'使用降级推理模式解决问题。识别了{len(entities)}个实体和{len(relations)}个关系。',
        'reasoning_steps': steps,
        'entities': entities,
        'relations': relations,
        'complexity': {
            'level': 'L1',
            'sublevel': 'L1.1',
            'reasoning_depth': len(steps)
        },
        'engine_mode': 'fallback'
    }

@app.route('/api/solve', methods=['POST'])
def solve_problem():
    """解决数学问题的主要端点"""
    try:
        data = request.get_json()
        
        if not data or 'problem' not in data:
            return jsonify({
                'error': 'Missing problem text',
                'message': 'Please provide a problem to solve'
            }), 400
        
        problem_text = data['problem'].strip()
        
        if not problem_text:
            return jsonify({
                'error': 'Empty problem text',
                'message': 'Please provide a non-empty problem'
            }), 400
        
        logger.info(f"📝 收到问题: {problem_text[:100]}...")
        
        start_time = time.time()
        
        # 尝试使用独立COT-DIR推理引擎
        if reasoning_engine_status == "healthy":
            try:
                # 调用独立COT-DIR推理引擎
                logger.info("🧠 使用独立COT-DIR推理引擎求解...")
                result = solve_mathematical_problem(problem_text, data.get('options', {}))
                
                processing_time = time.time() - start_time
                logger.info(f"✅ COT-DIR推理完成，用时 {processing_time:.2f}秒，答案: {result.get('final_answer', 'Unknown')}")
                
                return jsonify(result)
                
            except Exception as e:
                logger.warning(f"⚠️ COT-DIR推理引擎出错，使用降级模式: {e}")
                # 降级到简单推理
                pass
        
        # 使用降级推理模式
        logger.info("🔄 使用降级推理模式...")
        result = fallback_reasoning(problem_text)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ 降级推理完成，用时 {processing_time:.2f}秒")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ 解题过程出错: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

def transform_reasoning_result(reasoning_result: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
    """将推理引擎结果转换为前端期望的格式"""
    
    # 基础结果结构
    transformed = {
        'answer': reasoning_result.get('final_answer', 'Unknown'),
        'final_answer': reasoning_result.get('final_answer', 'Unknown'),
        'confidence': reasoning_result.get('confidence', 0.8),
        'explanation': reasoning_result.get('explanation', '使用COT-DIR推理引擎求解'),
        'reasoning_steps': [],
        'entities': [],
        'relations': [],
        'complexity': {
            'level': 'L2',
            'sublevel': 'L2.1', 
            'reasoning_depth': 3
        }
    }
    
    # 转换推理步骤
    steps = reasoning_result.get('reasoning_steps', [])
    for i, step in enumerate(steps):
        transformed_step = {
            'id': step.get('id', f'step_{i+1}'),
            'step': i + 1,
            'type': step.get('type', 'reasoning'),
            'description': step.get('description', f'推理步骤 {i+1}'),
            'confidence': step.get('confidence', 0.8),
            'timestamp': int(time.time() * 1000) + i * 100,
            'details': step.get('details', {})
        }
        transformed['reasoning_steps'].append(transformed_step)
    
    # 提取或生成实体信息
    entities = reasoning_result.get('entities', [])
    if not entities:
        # 从问题中提取基础实体
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', problem_text)
        for i, num in enumerate(numbers[:5]):  # 最多5个数字
            entities.append({
                'id': f'entity_{i}',
                'text': num,
                'type': 'number',
                'value': float(num),
                'confidence': 0.9
            })
    
    transformed['entities'] = entities[:10]  # 最多10个实体
    
    # 提取或生成关系信息
    relations = reasoning_result.get('relations', [])
    if not relations and len(entities) >= 2:
        # 生成基础关系
        relations.append({
            'id': 'rel_math',
            'source': entities[0]['text'],
            'target': entities[1]['text'] if len(entities) > 1 else 'result',
            'type': 'mathematical',
            'description': '数学运算关系',
            'confidence': 0.8
        })
    
    transformed['relations'] = relations[:5]  # 最多5个关系
    
    # 复杂度分析
    if 'complexity' in reasoning_result:
        transformed['complexity'] = reasoning_result['complexity']
    elif len(steps) <= 3:
        transformed['complexity'] = {'level': 'L1', 'sublevel': 'L1.2', 'reasoning_depth': len(steps)}
    elif len(steps) <= 6:
        transformed['complexity'] = {'level': 'L2', 'sublevel': 'L2.1', 'reasoning_depth': len(steps)}
    else:
        transformed['complexity'] = {'level': 'L3', 'sublevel': 'L3.1', 'reasoning_depth': len(steps)}
    
    return transformed

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    health_status = {
        'status': 'healthy',
        'service': 'COT-DIR Mathematical Reasoning API',
        'version': '2.0.0',
        'timestamp': int(time.time()),
        'engine_status': 'unknown'
    }
    
    try:
        if reasoning_engine_status == "healthy":
            engine_health = cotdir_reasoner.health_check()
            health_status['engine_status'] = engine_health.get('status', 'unknown')
            health_status['engine_details'] = engine_health
        else:
            health_status['engine_status'] = reasoning_engine_status
            health_status['message'] = 'Using fallback reasoning mode'
    except Exception as e:
        health_status['engine_status'] = 'error'
        health_status['engine_error'] = str(e)
    
    return jsonify(health_status)

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """获取示例问题"""
    examples = [
        {
            'text': '如果约翰有5个苹果，给了玛丽2个，他还剩多少个苹果？',
            'complexity': 'L1',
            'type': 'Arithmetic',
            'expected_answer': '3'
        },
        {
            'text': '一列火车在2小时内行驶了120公里。它的平均速度是多少？',
            'complexity': 'L2', 
            'type': 'Word Problem',
            'expected_answer': '60公里/小时'
        },
        {
            'text': '求解 x: 2x + 3 = 11',
            'complexity': 'L2',
            'type': 'Algebra',
            'expected_answer': 'x = 4'
        },
        {
            'text': '求半径为5厘米的圆的面积',
            'complexity': 'L2',
            'type': 'Geometry',
            'expected_answer': '78.54平方厘米'
        },
        {
            'text': '如果 f(x) = x² + 2x - 3，求 f\'(x)',
            'complexity': 'L3',
            'type': 'Calculus',
            'expected_answer': 'f\'(x) = 2x + 2'
        }
    ]
    
    return jsonify({
        'examples': examples,
        'count': len(examples)
    })

@app.route('/api/engine/status', methods=['GET'])
def get_engine_status():
    """获取推理引擎状态"""
    try:
        if reasoning_engine_status == "healthy":
            stats = cotdir_reasoner.get_statistics()
            
            return jsonify({
                'engine_type': 'Independent COT-DIR',
                'status': 'active',
                'statistics': stats,
                'configuration': cotdir_reasoner.config
            })
        else:
            return jsonify({
                'engine_type': 'Fallback',
                'status': reasoning_engine_status,
                'message': 'Using simplified reasoning mode'
            })
    except Exception as e:
        return jsonify({
            'engine_type': 'COT-DIR',
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/engine/restart', methods=['POST'])
def restart_engine():
    """重启推理引擎"""
    global reasoning_engine_status
    
    try:
        logger.info("🔄 重启推理引擎...")
        
        # 重新初始化
        success = initialize_reasoning_engine()
        
        return jsonify({
            'success': success,
            'message': '推理引擎重启成功' if success else '推理引擎重启失败，使用降级模式',
            'engine_status': reasoning_engine_status
        })
        
    except Exception as e:
        logger.error(f"重启推理引擎出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """获取可用的数据集列表"""
    try:
        datasets = dataset_loader.get_available_datasets()
        
        return jsonify({
            'datasets': datasets,
            'count': len(datasets),
            'message': f'找到 {len(datasets)} 个数据集' if datasets else '未找到数据集'
        })
        
    except Exception as e:
        logger.error(f"获取数据集列表出错: {e}")
        return jsonify({
            'error': 'Failed to load datasets',
            'message': str(e)
        }), 500

@app.route('/api/datasets/<dataset_name>/problems', methods=['GET'])
def get_dataset_problems(dataset_name: str):
    """从指定数据集获取问题样本"""
    try:
        # 获取参数
        count = request.args.get('count', 5, type=int)
        count = min(max(count, 1), 20)  # 限制在1-20之间
        
        # 加载数据集样本
        problems = dataset_loader.load_dataset_sample(dataset_name, count)
        
        if not problems:
            return jsonify({
                'error': 'Dataset not found or empty',
                'message': f'数据集 {dataset_name} 不存在或为空'
            }), 404
        
        return jsonify({
            'dataset': dataset_name,
            'problems': problems,
            'count': len(problems),
            'message': f'从 {dataset_name} 数据集获取了 {len(problems)} 个问题'
        })
        
    except Exception as e:
        logger.error(f"获取数据集问题出错: {e}")
        return jsonify({
            'error': 'Failed to load dataset problems',
            'message': str(e)
        }), 500

@app.route('/api/datasets/<dataset_name>/solve', methods=['POST'])
def solve_dataset_problem(dataset_name: str):
    """解决来自数据集的问题"""
    try:
        data = request.get_json()
        
        if not data or 'problem_id' not in data:
            return jsonify({
                'error': 'Missing problem_id',
                'message': 'Please provide a problem_id to solve'
            }), 400
        
        problem_id = data['problem_id']
        
        # 直接通过ID查找问题
        problem = dataset_loader.find_problem_by_id(dataset_name, problem_id)
        
        if not problem:
            return jsonify({
                'error': 'Problem not found',
                'message': f'在数据集 {dataset_name} 中未找到问题 {problem_id}'
            }), 404
        
        # 提取问题文本
        problem_text = problem.get('problem', '')
        if not problem_text:
            return jsonify({
                'error': 'Empty problem text',
                'message': '问题文本为空'
            }), 400
        
        # 使用推理引擎求解
        logger.info(f"📝 解决数据集问题: {dataset_name}/{problem_id}")
        
        if reasoning_engine_status == "healthy":
            result = solve_mathematical_problem(problem_text, data.get('options', {}))
        else:
            result = fallback_reasoning(problem_text)
        
        # 添加数据集相关信息
        result['dataset_info'] = {
            'dataset': dataset_name,
            'problem_id': problem_id,
            'original_answer': problem.get('answer', 'unknown'),
            'equation': problem.get('equation', ''),
            'type': problem.get('type', 'unknown')
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"解决数据集问题出错: {e}")
        return jsonify({
            'error': 'Failed to solve dataset problem',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 启动COT-DIR数学推理API服务器")
    print("📡 服务器将运行在 http://localhost:5001")
    print("🔗 API端点:")
    print("   POST /api/solve - 解决数学问题")
    print("   GET /api/health - 健康检查")
    print("   GET /api/examples - 获取示例问题")
    print("   GET /api/engine/status - 推理引擎状态")
    print("   POST /api/engine/restart - 重启推理引擎")
    print("   GET /api/datasets - 获取可用数据集列表")
    print("   GET /api/datasets/<name>/problems - 从数据集获取问题")
    print("   POST /api/datasets/<name>/solve - 解决数据集中的问题")
    print()
    
    # 初始化推理引擎
    initialize_reasoning_engine()
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )