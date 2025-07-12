"""
🧠 Intelligent Problem Classifier - 智能分类和模板匹配
10种题型自动识别，智能模板匹配系统
"""

import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ProblemType(Enum):
    """数学问题类型枚举"""
    ARITHMETIC = "算术运算"          # 基本四则运算
    WORD_PROBLEM = "应用题"         # 实际应用情境
    EQUATION = "方程求解"           # 线性/非线性方程
    GEOMETRY = "几何问题"           # 面积、周长、体积
    RATIO_PROPORTION = "比例问题"    # 比率、比例、百分比
    TIME_DISTANCE = "行程问题"      # 时间、速度、距离
    FINANCE = "金融问题"            # 利息、价格、折扣
    COMBINATION = "排列组合"        # 概率、统计
    PHYSICS_MATH = "物理数学"       # 物理应用数学
    COMPLEX_REASONING = "复杂推理"   # 多步骤推理


@dataclass
class ProblemPattern:
    """问题模式"""
    pattern_id: str
    problem_type: ProblemType
    keywords: List[str]
    regex_patterns: List[str]
    template: str
    confidence_weight: float = 1.0
    examples: List[str] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass 
class ClassificationResult:
    """分类结果"""
    problem_type: ProblemType
    confidence: float
    matched_patterns: List[str]
    template_match: Optional[str] = None
    extracted_entities: Dict[str, Any] = None
    reasoning: str = ""
    
    def __post_init__(self):
        if self.extracted_entities is None:
            self.extracted_entities = {}


class IntelligentClassifier:
    """🧠 智能问题分类器"""
    
    def __init__(self, patterns_file: Optional[str] = None):
        """
        初始化智能分类器
        
        Args:
            patterns_file: 自定义模式文件路径
        """
        self.patterns: Dict[ProblemType, List[ProblemPattern]] = defaultdict(list)
        self.classification_stats = defaultdict(int)
        self.entity_extractors = {}
        
        # 加载预定义模式
        self._load_default_patterns()
        
        # 加载自定义模式
        if patterns_file and Path(patterns_file).exists():
            self._load_custom_patterns(patterns_file)
        
        # 初始化实体提取器
        self._init_entity_extractors()
        
        print(f"🧠 智能分类器已初始化，加载了 {sum(len(patterns) for patterns in self.patterns.values())} 个模式")
    
    def _load_default_patterns(self):
        """加载默认模式"""
        default_patterns = [
            # 1. 算术运算
            ProblemPattern(
                pattern_id="arithmetic_basic",
                problem_type=ProblemType.ARITHMETIC,
                keywords=["加", "减", "乘", "除", "计算", "等于", "+", "-", "×", "÷", "="],
                regex_patterns=[
                    r"\d+\s*[+\-×÷]\s*\d+",
                    r"(\d+)\s*(加|减|乘|除)\s*(\d+)",
                    r"计算.*\d+.*[+\-×÷].*\d+"
                ],
                template="基本四则运算: {operand1} {operator} {operand2} = ?",
                confidence_weight=0.9
            ),
            
            # 2. 应用题
            ProblemPattern(
                pattern_id="word_problem_basic",
                problem_type=ProblemType.WORD_PROBLEM,
                keywords=["买", "卖", "花费", "剩余", "总共", "一共", "每个", "分给"],
                regex_patterns=[
                    r".*买了.*(\d+).*",
                    r".*一共.*(\d+).*",
                    r".*剩下.*(\d+).*"
                ],
                template="应用情境问题: 根据{context}，求{target}",
                confidence_weight=0.8
            ),
            
            # 3. 方程求解
            ProblemPattern(
                pattern_id="equation_linear",
                problem_type=ProblemType.EQUATION,
                keywords=["方程", "解", "未知数", "x", "y", "求解"],
                regex_patterns=[
                    r"[a-zA-Z]\s*[+\-]\s*\d+\s*=\s*\d+",
                    r"\d*[a-zA-Z]\s*=\s*\d+",
                    r"解方程.*[a-zA-Z]"
                ],
                template="方程求解: {equation}，求 {variable}",
                confidence_weight=0.95
            ),
            
            # 4. 几何问题
            ProblemPattern(
                pattern_id="geometry_area",
                problem_type=ProblemType.GEOMETRY,
                keywords=["面积", "周长", "体积", "长方形", "正方形", "圆形", "三角形", "半径", "直径"],
                regex_patterns=[
                    r".*面积.*(\d+).*",
                    r".*周长.*(\d+).*",
                    r".*(长方形|正方形|圆形|三角形).*"
                ],
                template="几何计算: 求{shape}的{property}",
                confidence_weight=0.85
            ),
            
            # 5. 比例问题
            ProblemPattern(
                pattern_id="ratio_percent",
                problem_type=ProblemType.RATIO_PROPORTION,
                keywords=["比例", "比", "百分比", "%", "比率", "成正比", "成反比"],
                regex_patterns=[
                    r"\d+:\d+",
                    r"\d+%",
                    r".*比例.*\d+.*"
                ],
                template="比例计算: {ratio_info}",
                confidence_weight=0.8
            ),
            
            # 6. 行程问题
            ProblemPattern(
                pattern_id="time_distance",
                problem_type=ProblemType.TIME_DISTANCE,
                keywords=["速度", "时间", "距离", "行驶", "走", "跑", "每小时", "公里", "米"],
                regex_patterns=[
                    r".*速度.*(\d+).*",
                    r".*(\d+)\s*(公里|米).*",
                    r".*(\d+)\s*小时.*"
                ],
                template="行程问题: 速度{speed}，时间{time}，求{target}",
                confidence_weight=0.85
            ),
            
            # 7. 金融问题
            ProblemPattern(
                pattern_id="finance_basic",
                problem_type=ProblemType.FINANCE,
                keywords=["价格", "成本", "利润", "折扣", "利息", "元", "钱", "花费"],
                regex_patterns=[
                    r".*(\d+)\s*元.*",
                    r".*价格.*(\d+).*",
                    r".*利息.*(\d+).*"
                ],
                template="金融计算: {financial_context}",
                confidence_weight=0.8
            ),
            
            # 8. 排列组合
            ProblemPattern(
                pattern_id="combination_basic",
                problem_type=ProblemType.COMBINATION,
                keywords=["排列", "组合", "选择", "概率", "可能", "方法", "种"],
                regex_patterns=[
                    r".*(\d+)\s*种.*方法.*",
                    r".*排列.*(\d+).*",
                    r".*组合.*(\d+).*"
                ],
                template="排列组合: 从{total}中选{select}",
                confidence_weight=0.9
            ),
            
            # 9. 物理数学
            ProblemPattern(
                pattern_id="physics_math",
                problem_type=ProblemType.PHYSICS_MATH,
                keywords=["力", "压强", "密度", "重量", "质量", "温度", "电流", "功率"],
                regex_patterns=[
                    r".*力.*(\d+).*牛顿.*",
                    r".*压强.*(\d+).*",
                    r".*密度.*(\d+).*"
                ],
                template="物理数学: {physics_concept}的计算",
                confidence_weight=0.85
            ),
            
            # 10. 复杂推理
            ProblemPattern(
                pattern_id="complex_reasoning",
                problem_type=ProblemType.COMPLEX_REASONING,
                keywords=["如果", "那么", "因为", "所以", "推理", "证明", "假设"],
                regex_patterns=[
                    r"如果.*那么.*",
                    r"因为.*所以.*",
                    r".*推理.*"
                ],
                template="复杂推理: 基于{conditions}推导{conclusion}",
                confidence_weight=0.75
            )
        ]
        
        # 按类型组织模式
        for pattern in default_patterns:
            self.patterns[pattern.problem_type].append(pattern)
    
    def _load_custom_patterns(self, patterns_file: str):
        """加载自定义模式"""
        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                custom_data = json.load(f)
            
            for pattern_data in custom_data.get('patterns', []):
                pattern = ProblemPattern(**pattern_data)
                self.patterns[pattern.problem_type].append(pattern)
            
            print(f"✅ 加载自定义模式: {patterns_file}")
        except Exception as e:
            print(f"⚠️ 加载自定义模式失败: {e}")
    
    def _init_entity_extractors(self):
        """初始化实体提取器"""
        self.entity_extractors = {
            'numbers': re.compile(r'\d+(?:\.\d+)?'),
            'variables': re.compile(r'[a-zA-Z]'),
            'operators': re.compile(r'[+\-×÷=]'),
            'units': re.compile(r'(元|米|公里|小时|分钟|秒|平方米|立方米|千克|克)'),
            'percentages': re.compile(r'\d+%'),
            'ratios': re.compile(r'\d+:\d+')
        }
    
    def classify(self, problem_text: str) -> ClassificationResult:
        """
        🎯 对问题进行智能分类
        
        Args:
            problem_text: 问题文本
            
        Returns:
            分类结果
        """
        # 文本预处理
        cleaned_text = self._preprocess_text(problem_text)
        
        # 计算每种类型的匹配分数
        type_scores = {}
        matched_patterns_all = {}
        
        for problem_type, patterns in self.patterns.items():
            score, matched = self._calculate_type_score(cleaned_text, patterns)
            type_scores[problem_type] = score
            matched_patterns_all[problem_type] = matched
        
        # 选择最佳匹配
        best_type = max(type_scores, key=type_scores.get) if type_scores else ProblemType.COMPLEX_REASONING
        best_score = type_scores.get(best_type, 0.0)
        
        # 提取实体
        entities = self._extract_entities(cleaned_text)
        
        # 生成模板匹配
        template_match = self._generate_template_match(best_type, entities, cleaned_text)
        
        # 生成推理解释
        reasoning = self._generate_reasoning(best_type, matched_patterns_all.get(best_type, []), best_score)
        
        # 更新统计
        self.classification_stats[best_type] += 1
        
        return ClassificationResult(
            problem_type=best_type,
            confidence=min(best_score, 1.0),
            matched_patterns=matched_patterns_all.get(best_type, []),
            template_match=template_match,
            extracted_entities=entities,
            reasoning=reasoning
        )
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 标准化数字表达
        text = re.sub(r'(\d+)\s*(个|只|人|本|支)', r'\1', text)
        
        # 标准化运算符
        text = text.replace('×', '*').replace('÷', '/')
        
        return text
    
    def _calculate_type_score(self, text: str, patterns: List[ProblemPattern]) -> Tuple[float, List[str]]:
        """计算类型匹配分数"""
        total_score = 0.0
        matched_patterns = []
        
        for pattern in patterns:
            pattern_score = 0.0
            
            # 关键词匹配
            keyword_matches = sum(1 for keyword in pattern.keywords if keyword in text)
            keyword_score = (keyword_matches / len(pattern.keywords)) * 0.6 if pattern.keywords else 0
            
            # 正则表达式匹配
            regex_matches = sum(1 for regex in pattern.regex_patterns if re.search(regex, text))
            regex_score = (regex_matches / len(pattern.regex_patterns)) * 0.4 if pattern.regex_patterns else 0
            
            pattern_score = (keyword_score + regex_score) * pattern.confidence_weight
            
            if pattern_score > 0.3:  # 阈值过滤
                total_score += pattern_score
                matched_patterns.append(pattern.pattern_id)
        
        return total_score, matched_patterns
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """提取实体信息"""
        entities = {}
        
        for entity_type, extractor in self.entity_extractors.items():
            matches = extractor.findall(text)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def _generate_template_match(self, problem_type: ProblemType, entities: Dict, text: str) -> str:
        """生成模板匹配"""
        patterns_for_type = self.patterns.get(problem_type, [])
        if not patterns_for_type:
            return f"{problem_type.value}问题"
        
        # 找到最佳匹配的模式
        best_pattern = patterns_for_type[0]  # 使用第一个模式作为默认
        
        # 简单的模板填充
        template = best_pattern.template
        
        # 移除未填充的占位符
        template = re.sub(r'\{[^}]+\}', '...', template)
        
        return template
    
    def _generate_reasoning(self, problem_type: ProblemType, patterns: List[str], confidence: float) -> str:
        """生成推理解释"""
        reasoning_parts = [
            f"识别为{problem_type.value}，置信度: {confidence:.2f}"
        ]
        
        if patterns:
            reasoning_parts.append(f"匹配模式: {', '.join(patterns)}")
        
        if confidence > 0.8:
            reasoning_parts.append("高置信度匹配")
        elif confidence > 0.5:
            reasoning_parts.append("中等置信度匹配")
        else:
            reasoning_parts.append("低置信度匹配，建议人工确认")
        
        return " | ".join(reasoning_parts)
    
    def batch_classify(self, problems: List[str]) -> List[ClassificationResult]:
        """📦 批量分类"""
        results = []
        
        print(f"🔄 批量分类 {len(problems)} 个问题...")
        
        for i, problem in enumerate(problems):
            try:
                result = self.classify(problem)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    print(f"  已处理: {i + 1}/{len(problems)}")
                    
            except Exception as e:
                print(f"  ⚠️ 分类问题 {i+1} 失败: {e}")
                # 创建默认结果
                results.append(ClassificationResult(
                    problem_type=ProblemType.COMPLEX_REASONING,
                    confidence=0.0,
                    matched_patterns=[],
                    reasoning=f"分类失败: {e}"
                ))
        
        print(f"✅ 批量分类完成")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 获取分类统计"""
        total_classified = sum(self.classification_stats.values())
        
        stats = {
            'total_classified': total_classified,
            'type_distribution': dict(self.classification_stats),
            'type_percentages': {}
        }
        
        if total_classified > 0:
            for ptype, count in self.classification_stats.items():
                percentage = (count / total_classified) * 100
                stats['type_percentages'][ptype.value] = round(percentage, 2)
        
        return stats
    
    def save_model(self, model_path: str):
        """💾 保存分类模型"""
        model_data = {
            'patterns': {ptype.value: [asdict(p) for p in patterns] 
                        for ptype, patterns in self.patterns.items()},
            'stats': {ptype.value: count for ptype, count in self.classification_stats.items()}
        }
        
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 模型已保存到: {model_path}")
    
    def add_pattern(self, pattern: ProblemPattern):
        """➕ 添加新模式"""
        self.patterns[pattern.problem_type].append(pattern)
        print(f"✅ 添加新模式: {pattern.pattern_id}")
    
    def analyze_classification_accuracy(self, test_data: List[Tuple[str, ProblemType]]) -> Dict[str, float]:
        """🎯 分析分类准确度"""
        correct = 0
        total = len(test_data)
        type_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for problem_text, true_type in test_data:
            result = self.classify(problem_text)
            predicted_type = result.problem_type
            
            type_accuracy[true_type]['total'] += 1
            
            if predicted_type == true_type:
                correct += 1
                type_accuracy[true_type]['correct'] += 1
        
        overall_accuracy = correct / total if total > 0 else 0
        
        accuracy_report = {
            'overall_accuracy': round(overall_accuracy, 3),
            'total_samples': total,
            'correct_predictions': correct,
            'per_type_accuracy': {}
        }
        
        for ptype, stats in type_accuracy.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            accuracy_report['per_type_accuracy'][ptype.value] = {
                'accuracy': round(acc, 3),
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        return accuracy_report


# 使用示例和测试
def demo_intelligent_classifier():
    """演示智能分类器"""
    print("🧠 Intelligent Problem Classifier Demo")
    print("=" * 50)
    
    # 创建分类器
    classifier = IntelligentClassifier()
    
    # 测试问题
    test_problems = [
        "计算 25 + 17 = ?",
        "小明买了3个苹果，每个2元，一共花了多少钱？",
        "解方程: 2x + 5 = 15",
        "一个正方形的边长是5米，求面积",
        "如果一辆车以60公里/小时的速度行驶2小时，走了多远？",
        "从10个人中选3个人，有多少种选法？",
        "一本书原价20元，打8折后多少钱？",
        "A和B的比例是3:2，如果A是15，B是多少？"
    ]
    
    print(f"🎯 测试分类 {len(test_problems)} 个问题:")
    print("-" * 50)
    
    results = []
    for i, problem in enumerate(test_problems, 1):
        result = classifier.classify(problem)
        results.append(result)
        
        print(f"问题 {i}: {problem}")
        print(f"  类型: {result.problem_type.value}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  模板: {result.template_match}")
        print(f"  推理: {result.reasoning}")
        print()
    
    # 显示统计信息
    stats = classifier.get_statistics()
    print("📊 分类统计:")
    for ptype, percentage in stats['type_percentages'].items():
        print(f"  {ptype}: {percentage}%")
    
    return classifier, results


if __name__ == "__main__":
    demo_intelligent_classifier() 