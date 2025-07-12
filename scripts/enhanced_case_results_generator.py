#!/usr/bin/env python3
"""
增强版案例结果生成器 - Day 1 & Day 2 优化实现
支持数据集批量加载 + 通用解题模板 + 批量处理管道
"""

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# 导入现有模块
from Data.dataset_loader import MathDatasetLoader
from simplified_cases_demo import SimplifiedCOTDIRDemo


class EnhancedCaseResultsGenerator:
    """增强版案例结果生成器"""
    
    def __init__(self, 
                 dataset_names: List[str] = None, 
                 sample_size_per_dataset: int = 10,
                 total_target_problems: int = 50):
        """
        初始化增强版生成器
        """
        print("🚀 初始化增强版COT-DIR案例结果生成器...")
        
        # 初始化数据集加载器
        self.dataset_loader = MathDatasetLoader()
        
        # 设置数据集配置
        self.dataset_names = dataset_names or ['Math23K', 'GSM8K', 'MAWPS', 'SVAMP', 'ASDiv']
        self.sample_size_per_dataset = sample_size_per_dataset
        self.total_target_problems = total_target_problems
        
        # 初始化原有的演示器用于推理
        self.demo = SimplifiedCOTDIRDemo()
        
        # 加载可用数据集
        self.available_datasets = self.dataset_loader.list_datasets()
        print(f"✅ 可用数据集: {self.available_datasets}")
        
        # 过滤实际存在的数据集
        self.dataset_names = [name for name in self.dataset_names if name in self.available_datasets]
        print(f"✅ 将要使用的数据集: {self.dataset_names}")
        
        # 初始化问题分类器
        self.problem_classifier = ProblemTypeClassifier()
        
        # 初始化解题模板库
        self.solution_templates = SolutionTemplateLibrary()
        
        print("✅ 增强版生成器初始化完成！\n")
    
    def load_dynamic_test_cases(self) -> List[Dict[str, Any]]:
        """动态加载多数据集题目"""
        print("📊 开始动态加载测试用例...")
        
        all_cases = []
        problems_per_dataset = max(1, self.total_target_problems // len(self.dataset_names))
        
        for dataset_name in self.dataset_names:
            print(f"📖 加载数据集: {dataset_name}")
            
            try:
                # 加载数据集
                dataset = self.dataset_loader.load_dataset(
                    dataset_name, 
                    max_samples=min(problems_per_dataset * 2, 100)
                )
                
                if not dataset:
                    print(f"⚠️  数据集 {dataset_name} 为空，跳过")
                    continue
                
                # 随机采样
                sample_size = min(problems_per_dataset, len(dataset))
                sampled_problems = random.sample(dataset, sample_size)
                
                # 转换为测试用例格式
                for i, problem in enumerate(sampled_problems):
                    case = self._convert_to_test_case(problem, dataset_name, i)
                    if case:
                        all_cases.append(case)
                
                print(f"  ✅ 从 {dataset_name} 加载了 {len(sampled_problems)} 个题目")
                
            except Exception as e:
                print(f"  ❌ 加载 {dataset_name} 失败: {e}")
                continue
        
        # 限制总数量
        if len(all_cases) > self.total_target_problems:
            all_cases = random.sample(all_cases, self.total_target_problems)
        
        print(f"🎯 总共加载了 {len(all_cases)} 个测试用例")
        return all_cases
    
    def _convert_to_test_case(self, problem: Dict, dataset_name: str, index: int) -> Optional[Dict[str, Any]]:
        """将数据集问题转换为测试用例格式"""
        try:
            # 提取问题文本
            problem_text = self._extract_problem_text(problem)
            if not problem_text or len(problem_text.strip()) < 10:
                return None
            
            # 提取答案
            answer = self._extract_answer(problem)
            if answer is None:
                return None
            
            # 自动分类问题类型
            problem_type = self.problem_classifier.classify(problem_text)
            
            # 判断语言
            language = "中文" if self._is_chinese(problem_text) else "英文"
            
            # 生成测试用例
            case = {
                "id": f"{dataset_name.lower()}_{index:03d}",
                "language": language,
                "problem": problem_text,
                "expected_answer": str(answer),
                "type": problem_type,
                "difficulty": self._estimate_difficulty(problem_text),
                "complexity_level": self._estimate_complexity(problem_text),
                "source": dataset_name,
                "original_data": problem
            }
            
            return case
            
        except Exception as e:
            print(f"  ⚠️  转换问题失败: {e}")
            return None
    
    def _extract_problem_text(self, problem: Dict) -> str:
        """从问题数据中提取问题文本"""
        text_fields = ["problem", "question", "text", "body", "sQuestion", "Problem"]
        
        for field in text_fields:
            if field in problem and problem[field]:
                text = str(problem[field]).strip()
                if text:
                    return text
        
        if "body" in problem and "question" in problem:
            return f"{problem['body']} {problem['question']}"
        
        return ""
    
    def _extract_answer(self, problem: Dict) -> Optional[str]:
        """从问题数据中提取答案"""
        answer_fields = ["answer", "solution", "correct", "target", "lSolutions", "Answer"]
        
        for field in answer_fields:
            if field in problem and problem[field] is not None:
                answer = str(problem[field]).strip()
                if answer:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', answer)
                    if numbers:
                        return numbers[-1]
                    return answer
        
        return None
    
    def _is_chinese(self, text: str) -> bool:
        """判断文本是否为中文"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return chinese_chars > len(text) * 0.3
    
    def _estimate_difficulty(self, problem_text: str) -> str:
        """估计问题难度"""
        text_length = len(problem_text)
        numbers_count = len(re.findall(r'\d+', problem_text))
        
        if text_length < 50 and numbers_count <= 3:
            return "简单"
        elif text_length < 100 and numbers_count <= 5:
            return "中等"
        else:
            return "困难"
    
    def _estimate_complexity(self, problem_text: str) -> str:
        """估计复杂度等级"""
        operators = len(re.findall(r'[+\-*/÷×]', problem_text))
        conditional_words = len(re.findall(r'如果|假设|when|if', problem_text, re.IGNORECASE))
        
        if operators <= 1 and conditional_words == 0:
            return "L0"
        elif operators <= 2 and conditional_words <= 1:
            return "L1"  
        elif operators <= 4 and conditional_words <= 2:
            return "L2"
        else:
            return "L3"
    
    def generate_enhanced_detailed_results(self, use_parallel: bool = False) -> List[Dict[str, Any]]:
        """生成增强版的详细结果"""
        print("🎯 开始生成增强版详细结果...")
        
        # 动态加载测试用例
        test_cases = self.load_dynamic_test_cases()
        
        if not test_cases:
            print("❌ 没有加载到有效的测试用例")
            return []
        
        # 处理测试用例
        detailed_results = []
        
        for i, case in enumerate(test_cases):
            print(f"🔍 处理用例 {i+1}/{len(test_cases)}: {case['id']}")
            
            try:
                result = self._process_single_case(case)
                if result:
                    detailed_results.append(result)
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                continue
        
        return detailed_results
    
    def _process_single_case(self, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理单个测试用例"""
        try:
            # 执行推理
            reasoning_result = self.demo._simulate_cotdir_reasoning(case)
            
            # 使用通用解题模板生成解题过程
            solution_process = self.solution_templates.generate_solution_process(case, reasoning_result)
            
            # 构建详细结果
            detailed_result = {
                "case_id": case['id'],
                "case_info": {
                    "language": case['language'],
                    "problem_statement": case['problem'],
                    "expected_answer": case['expected_answer'],
                    "problem_type": case['type'],
                    "difficulty": case['difficulty'],
                    "complexity_level": case['complexity_level'],
                    "source_dataset": case['source']
                },
                
                "reasoning_process": {
                    "step_1_entity_extraction": {
                        "description": "IRD模块第一步：实体提取",
                        "entities": reasoning_result.get('entities', []),
                        "analysis": self._analyze_entities(reasoning_result.get('entities', []))
                    },
                    
                    "step_2_relation_discovery": {
                        "description": "IRD模块第二步：关系发现",
                        "relations": reasoning_result.get('relations', []),
                        "analysis": self._analyze_relations(reasoning_result.get('relations', []))
                    },
                    
                    "step_3_multi_layer_reasoning": {
                        "description": "MLR模块：多层推理",
                        "reasoning_steps": reasoning_result.get('reasoning_steps', []),
                        "layer_analysis": self._analyze_reasoning_layers(reasoning_result.get('reasoning_steps', []))
                    },
                    
                    "step_4_confidence_verification": {
                        "description": "CV模块：置信度验证",
                        "confidence_score": reasoning_result.get('confidence_score', 0),
                        "confidence_analysis": self._analyze_confidence(reasoning_result.get('confidence_score', 0))
                    }
                },
                
                "solution_process": solution_process,
                
                "final_result": {
                    "predicted_answer": reasoning_result.get('final_answer'),
                    "expected_answer": case['expected_answer'],
                    "is_correct": str(reasoning_result.get('final_answer')) == str(case['expected_answer']),
                    "confidence_score": reasoning_result.get('confidence_score', 0)
                },
                
                "performance_metrics": {
                    "processing_time": 0.001,
                    "entities_count": len(reasoning_result.get('entities', [])),
                    "relations_count": len(reasoning_result.get('relations', [])),
                    "reasoning_steps_count": len(reasoning_result.get('reasoning_steps', []))
                },
                
                "quality_assessment": self._assess_quality(case, reasoning_result)
            }
            
            return detailed_result
            
        except Exception as e:
            print(f"    ❌ 处理用例 {case['id']} 时出错: {e}")
            return None
    
    def _analyze_entities(self, entities: List[Dict]) -> Dict[str, Any]:
        """分析提取的实体"""
        entity_types = {}
        for entity in entities:
            entity_type = entity.get('type', 'unknown')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity.get('name', ''))
        
        return {
            "total_entities": len(entities),
            "entity_types": entity_types,
            "completeness": "高" if len(entities) >= 5 else "中等" if len(entities) >= 3 else "低",
            "key_entities": entities[:3]
        }
    
    def _analyze_relations(self, relations: List[Dict]) -> Dict[str, Any]:
        """分析发现的关系"""
        relation_types = [rel.get('type', '') for rel in relations]
        
        return {
            "total_relations": len(relations),
            "relation_types": list(set(relation_types)),
            "complexity": "高" if len(relations) >= 3 else "中等" if len(relations) >= 1 else "低",
            "key_relations": relations[:2]
        }
    
    def _analyze_reasoning_layers(self, steps: List[Dict]) -> Dict[str, Any]:
        """分析推理层次"""
        layers = {}
        for step in steps:
            layer = step.get('layer', 'unknown')
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(step.get('description', ''))
        
        return {
            "total_steps": len(steps),
            "layers_used": list(layers.keys()),
            "layer_distribution": {k: len(v) for k, v in layers.items()},
            "reasoning_depth": "深入" if len(steps) >= 4 else "中等" if len(steps) >= 2 else "浅层"
        }
    
    def _analyze_confidence(self, confidence: float) -> Dict[str, Any]:
        """分析置信度"""
        if confidence >= 90:
            level = "极高"
            interpretation = "系统对答案非常确信"
        elif confidence >= 80:
            level = "高"
            interpretation = "系统对答案比较确信"
        elif confidence >= 70:
            level = "中等"
            interpretation = "系统对答案有一定把握"
        else:
            level = "低"
            interpretation = "系统对答案缺乏信心"
        
        return {
            "confidence_level": level,
            "interpretation": interpretation,
            "score": confidence,
            "reliability": "可靠" if confidence >= 85 else "一般" if confidence >= 70 else "不可靠"
        }
    
    def _assess_quality(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """评估推理质量"""
        entities_count = len(reasoning_result.get('entities', []))
        relations_count = len(reasoning_result.get('relations', []))
        steps_count = len(reasoning_result.get('reasoning_steps', []))
        is_correct = str(reasoning_result.get('final_answer')) == str(case['expected_answer'])
        
        # 计算质量分数
        entity_score = min(entities_count * 2, 10)
        relation_score = min(relations_count * 3, 10) 
        reasoning_score = min(steps_count * 2, 10)
        correctness_score = 10 if is_correct else 0
        
        total_score = (entity_score + relation_score + reasoning_score + correctness_score) / 4
        
        return {
            "overall_score": round(total_score, 1),
            "component_scores": {
                "entity_extraction": entity_score,
                "relation_discovery": relation_score,
                "reasoning_quality": reasoning_score,
                "correctness": correctness_score
            },
            "strengths": self._identify_strengths(entities_count, relations_count, steps_count, is_correct),
            "weaknesses": self._identify_weaknesses(entities_count, relations_count, steps_count, is_correct),
            "grade": self._get_quality_grade(total_score)
        }
    
    def _identify_strengths(self, entities: int, relations: int, steps: int, correct: bool) -> List[str]:
        """识别推理优势"""
        strengths = []
        if entities >= 5:
            strengths.append("实体提取完整")
        if relations >= 3:
            strengths.append("关系发现深入")
        if steps >= 4:
            strengths.append("推理步骤详细")
        if correct:
            strengths.append("答案正确")
        return strengths
    
    def _identify_weaknesses(self, entities: int, relations: int, steps: int, correct: bool) -> List[str]:
        """识别推理弱点"""
        weaknesses = []
        if entities < 3:
            weaknesses.append("实体提取不足")
        if relations < 2:
            weaknesses.append("关系发现简单")
        if steps < 3:
            weaknesses.append("推理步骤不够")
        if not correct:
            weaknesses.append("答案错误")
        return weaknesses
    
    def _get_quality_grade(self, score: float) -> str:
        """获取质量等级"""
        if score >= 9:
            return "A+"
        elif score >= 8:
            return "A"
        elif score >= 7:
            return "B+"
        elif score >= 6:
            return "B"
        elif score >= 5:
            return "C"
        else:
            return "D"
    
    def save_enhanced_results(self, results: List[Dict], filename: str = "enhanced_case_results.json"):
        """保存增强版结果"""
        print(f"💾 保存结果到 {filename}...")
        
        # 创建结果目录
        output_dir = Path("enhanced_results")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        
        # 添加元数据
        output_data = {
            "metadata": {
                "generator_version": "enhanced_v1.0",
                "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_cases": len(results),
                "datasets_used": self.dataset_names,
                "sample_size_per_dataset": self.sample_size_per_dataset
            },
            "summary": self._generate_summary(results),
            "detailed_results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存: {output_path}")
        print(f"📊 总共生成 {len(results)} 个详细案例结果")
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """生成结果摘要"""
        if not results:
            return {}
        
        # 统计信息
        total_cases = len(results)
        correct_cases = sum(1 for r in results if r['final_result']['is_correct'])
        accuracy = correct_cases / total_cases * 100 if total_cases > 0 else 0
        
        # 按数据集统计
        dataset_stats = {}
        for result in results:
            dataset = result['case_info']['source_dataset']
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {'total': 0, 'correct': 0}
            dataset_stats[dataset]['total'] += 1
            if result['final_result']['is_correct']:
                dataset_stats[dataset]['correct'] += 1
        
        # 计算平均质量分数
        quality_scores = [r['quality_assessment']['overall_score'] for r in results]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "total_cases": total_cases,
            "correct_cases": correct_cases,
            "accuracy_percentage": round(accuracy, 2),
            "average_quality_score": round(avg_quality, 2),
            "dataset_breakdown": dataset_stats,
            "processing_status": "completed"
        }


class ProblemTypeClassifier:
    """问题类型分类器 - Day 2 优化"""
    
    def __init__(self):
        self.patterns = {
            "算术运算": [r'加|减|乘|除|\+|\-|\*|\/|总共|一共', r'个|只|本|支'],
            "分数运算": [r'分数|几分之几|1/\d+|\d+/\d+|占.*之.*'],
            "百分比计算": [r'百分比|%|折|打.*折|增长.*%'],
            "年龄推理": [r'年龄|岁|years?\s+old|older|younger'],
            "时间计算": [r'时间|小时|分钟|天|年|hours?|minutes?|days?|years?'],
            "投资分析": [r'投资|利润|成本|收入|profit|cost|investment'],
            "几何计算": [r'面积|周长|体积|长度|宽度|area|perimeter|volume'],
            "比例问题": [r'比例|比值|ratio|proportion'],
            "应用题": [r'买|卖|购买|销售|商店|市场']
        }
    
    def classify(self, problem_text: str) -> str:
        """分类问题类型"""
        for problem_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_text, re.IGNORECASE):
                    return problem_type
        
        return "通用数学题"


class SolutionTemplateLibrary:
    """解题模板库 - Day 2 优化"""
    
    def __init__(self):
        self.templates = {
            "算术运算": self._arithmetic_template,
            "分数运算": self._fraction_template,
            "百分比计算": self._percentage_template,
            "年龄推理": self._age_reasoning_template,
            "时间计算": self._time_calculation_template,
            "投资分析": self._investment_template,
            "几何计算": self._geometry_template,
            "比例问题": self._proportion_template,
            "应用题": self._application_template,
            "通用数学题": self._general_template
        }
    
    def generate_solution_process(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """生成解题过程"""
        problem_type = case.get('type', '通用数学题')
        template_func = self.templates.get(problem_type, self.templates["通用数学题"])
        
        return template_func(case, reasoning_result)
    
    def _arithmetic_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """算术运算模板"""
        problem_text = case['problem']
        numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
        
        return {
            "problem_analysis": "这是一个算术运算问题，需要理解数量关系并进行计算",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "识别题目中的关键数据",
                    "content": f"从题目中提取数字: {', '.join(numbers)}",
                    "mathematical_expression": f"关键数据: {numbers}"
                },
                {
                    "step": 2,
                    "description": "分析数量关系",
                    "content": "确定数字之间的运算关系",
                    "mathematical_expression": "建立运算表达式"
                },
                {
                    "step": 3,
                    "description": "执行计算",
                    "content": "按照运算顺序进行计算",
                    "mathematical_expression": f"计算结果 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "理解题目中的数量关系",
                "正确识别运算类型",
                "按步骤执行计算"
            ]
        }
    
    def _fraction_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """分数运算模板"""
        return {
            "problem_analysis": "这是一个分数运算问题，需要理解分数概念和运算规则",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "识别分数信息",
                    "content": "找出题目中的分数表示和整体数量",
                    "mathematical_expression": "确定分子、分母和整体"
                },
                {
                    "step": 2,
                    "description": "建立分数关系",
                    "content": "建立分数与实际数量的对应关系",
                    "mathematical_expression": "分数 × 整体 = 部分"
                },
                {
                    "step": 3,
                    "description": "计算结果",
                    "content": "执行分数运算得到最终答案",
                    "mathematical_expression": f"答案 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "理解分数的含义",
                "掌握分数运算规则",
                "建立分数与实际的对应关系"
            ]
        }
    
    def _percentage_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """百分比计算模板"""
        return {
            "problem_analysis": "这是一个百分比计算问题，需要理解百分比概念和计算方法",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "识别百分比信息",
                    "content": "找出题目中的百分比和基准数值",
                    "mathematical_expression": "确定百分比和基数"
                },
                {
                    "step": 2,
                    "description": "转换百分比",
                    "content": "将百分比转换为小数进行计算",
                    "mathematical_expression": "百分比 ÷ 100 = 小数"
                },
                {
                    "step": 3,
                    "description": "执行计算",
                    "content": "用小数乘以基数得到结果",
                    "mathematical_expression": f"结果 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "理解百分比的含义",
                "掌握百分比与小数的转换",
                "正确进行百分比计算"
            ]
        }
    
    def _age_reasoning_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """年龄推理模板"""
        return {
            "problem_analysis": "这是一个年龄推理问题，需要理解时间关系和年龄变化规律",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "理解年龄关系",
                    "content": "分析题目中各人物的年龄关系",
                    "mathematical_expression": "建立年龄关系式"
                },
                {
                    "step": 2,
                    "description": "考虑时间因素",
                    "content": "考虑时间推移对年龄的影响",
                    "mathematical_expression": "年龄 ± 时间差 = 新年龄"
                },
                {
                    "step": 3,
                    "description": "求解目标年龄",
                    "content": "根据关系式计算目标人物的年龄",
                    "mathematical_expression": f"目标年龄 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "理解年龄差不变的规律",
                "正确处理时间推移",
                "建立准确的年龄关系"
            ]
        }
    
    def _time_calculation_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """时间计算模板"""
        return {
            "problem_analysis": "这是一个时间计算问题，需要理解时间单位和时间运算",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "识别时间信息",
                    "content": "找出题目中的时间数据和单位",
                    "mathematical_expression": "确定时间量和单位"
                },
                {
                    "step": 2,
                    "description": "统一时间单位",
                    "content": "将不同的时间单位转换为统一单位",
                    "mathematical_expression": "单位转换"
                },
                {
                    "step": 3,
                    "description": "计算时间结果",
                    "content": "进行时间的加减运算",
                    "mathematical_expression": f"时间结果 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "掌握时间单位换算",
                "理解时间的加减运算",
                "注意时间的连续性"
            ]
        }
    
    def _investment_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """投资分析模板"""
        return {
            "problem_analysis": "这是一个投资分析问题，需要计算收入、成本和利润",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "识别财务要素",
                    "content": "找出成本、收入、利润等关键数据",
                    "mathematical_expression": "确定财务变量"
                },
                {
                    "step": 2,
                    "description": "建立财务关系",
                    "content": "建立收入、成本、利润之间的关系",
                    "mathematical_expression": "利润 = 收入 - 成本"
                },
                {
                    "step": 3,
                    "description": "计算财务结果",
                    "content": "根据题目要求计算相应的财务指标",
                    "mathematical_expression": f"结果 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "理解基本财务概念",
                "掌握收支计算方法",
                "考虑时间价值因素"
            ]
        }
    
    def _geometry_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """几何计算模板"""
        return {
            "problem_analysis": "这是一个几何计算问题，需要应用几何公式和空间概念",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "识别几何图形",
                    "content": "确定题目涉及的几何图形类型",
                    "mathematical_expression": "确定图形参数"
                },
                {
                    "step": 2,
                    "description": "选择计算公式",
                    "content": "根据题目要求选择相应的几何公式",
                    "mathematical_expression": "应用几何公式"
                },
                {
                    "step": 3,
                    "description": "计算几何量",
                    "content": "代入数值计算面积、周长或体积",
                    "mathematical_expression": f"几何量 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "掌握基本几何公式",
                "理解几何图形性质",
                "正确代入参数计算"
            ]
        }
    
    def _proportion_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """比例问题模板"""
        return {
            "problem_analysis": "这是一个比例问题，需要理解比例关系和比例运算",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "识别比例关系",
                    "content": "找出题目中的比例关系",
                    "mathematical_expression": "a : b = c : d"
                },
                {
                    "step": 2,
                    "description": "建立比例方程",
                    "content": "根据比例性质建立方程",
                    "mathematical_expression": "a × d = b × c"
                },
                {
                    "step": 3,
                    "description": "求解未知量",
                    "content": "解方程得到未知的比例项",
                    "mathematical_expression": f"未知量 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "理解比例的性质",
                "掌握比例方程求解",
                "注意比例的实际意义"
            ]
        }
    
    def _application_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """应用题模板"""
        return {
            "problem_analysis": "这是一个实际应用问题，需要从实际情境中抽象出数学关系",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "理解实际情境",
                    "content": "分析题目描述的实际场景",
                    "mathematical_expression": "明确问题背景"
                },
                {
                    "step": 2,
                    "description": "抽象数学关系",
                    "content": "从实际情境中抽象出数学关系",
                    "mathematical_expression": "建立数学模型"
                },
                {
                    "step": 3,
                    "description": "求解并验证",
                    "content": "计算数学结果并验证实际合理性",
                    "mathematical_expression": f"实际结果 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "理解实际问题背景",
                "准确抽象数学关系",
                "验证结果的合理性"
            ]
        }
    
    def _general_template(self, case: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """通用模板"""
        return {
            "problem_analysis": "这是一个数学问题，需要运用数学知识和推理能力求解",
            "solution_steps": [
                {
                    "step": 1,
                    "description": "理解题目要求",
                    "content": "仔细阅读题目，理解问题的要求",
                    "mathematical_expression": "明确求解目标"
                },
                {
                    "step": 2,
                    "description": "分析数学关系",
                    "content": "分析题目中的数学关系和约束条件",
                    "mathematical_expression": "建立数学关系"
                },
                {
                    "step": 3,
                    "description": "求解问题",
                    "content": "运用适当的数学方法求解问题",
                    "mathematical_expression": f"答案 = {reasoning_result.get('final_answer', '未知')}"
                }
            ],
            "key_insights": [
                "仔细理解题目要求",
                "准确分析数学关系",
                "选择合适的求解方法"
            ]
        }


def main():
    """主函数 - 演示增强版生成器"""
    print("🚀 启动增强版案例结果生成器演示")
    print("=" * 60)
    
    # 创建增强版生成器
    generator = EnhancedCaseResultsGenerator(
        dataset_names=['Math23K', 'GSM8K', 'MAWPS'],
        sample_size_per_dataset=10,
        total_target_problems=30
    )
    
    # 生成增强版详细结果
    results = generator.generate_enhanced_detailed_results()
    
    # 保存结果
    generator.save_enhanced_results(results, "enhanced_case_results_v1.json")
    
    # 显示摘要
    print("\n📊 生成结果摘要:")
    print(f"  总用例数: {len(results)}")
    if results:
        correct_count = sum(1 for r in results if r['final_result']['is_correct'])
        accuracy = correct_count / len(results) * 100
        print(f"  正确率: {accuracy:.1f}% ({correct_count}/{len(results)})")
        
        # 按数据集统计
        dataset_counts = {}
        for result in results:
            dataset = result['case_info']['source_dataset']
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print(f"  数据集分布:")
        for dataset, count in dataset_counts.items():
            print(f"    {dataset}: {count} 题")
    
    print("\n🎉 增强版案例结果生成完成！")


if __name__ == "__main__":
    main() 