#!/usr/bin/env python3
"""
批量数据集复杂度分类工具
====================================

使用增强的复杂度分析算法对数学问题数据集进行L0-L3分类
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from Data.dataset_loader import MathDatasetLoader
from src.reasoning_core.data_structures import ProblemComplexity
from src.reasoning_core.tools.complexity_analyzer import ComplexityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchComplexityClassifier:
    """批量复杂度分类器"""
    
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dataset_loader = MathDatasetLoader()
        
        # 增强的分类规则
        self.enhanced_rules = {
            ProblemComplexity.L0_EXPLICIT: {
                'max_numbers': 3,
                'max_operators': 1,
                'forbidden_keywords': ['每', '平均', '比', '分配', '如果'],
                'required_patterns': [r'\d+\s*[+\-×÷]\s*\d+', r'计算.*=.*'],
                'confidence_threshold': 0.8
            },
            ProblemComplexity.L1_SHALLOW: {
                'max_numbers': 5,
                'max_operators': 2,
                'required_keywords': ['一共', '总共', '还有', '剩下', 'total', 'sum'],
                'step_indicators': ['然后', '接着', 'then'],
                'confidence_threshold': 0.7
            },
            ProblemComplexity.L2_MEDIUM: {
                'max_numbers': 8,
                'max_operators': 4,
                'required_keywords': ['每', '平均', '分配', '比例', 'each', 'per', 'average'],
                'reasoning_indicators': ['需要', '可以得出', '因此', 'need', 'can'],
                'confidence_threshold': 0.6
            },
            ProblemComplexity.L3_DEEP: {
                'complex_patterns': [r'首先.*然后.*最后', r'如果.*那么.*否则', r'first.*then.*finally'],
                'domain_knowledge': ['利率', '比例', '几何', '概率', 'rate', 'ratio', 'percentage'],
                'multi_step_indicators': ['第一步', '第二步', '综合考虑', 'step 1', 'step 2'],
                'confidence_threshold': 0.5
            }
        }
    
    def classify_dataset(self, dataset_name: str, sample_size: int = None) -> Dict[str, Any]:
        """
        分类整个数据集
        
        Args:
            dataset_name: 数据集名称 (如 'GSM8K', 'Math23K')
            sample_size: 采样大小，None表示处理全部
            
        Returns:
            分类结果字典
        """
        logger.info(f"开始分类数据集: {dataset_name}")
        
        # 加载数据集
        try:
            dataset = self.dataset_loader.load_dataset(dataset_name)
            if not dataset:
                logger.error(f"无法加载数据集: {dataset_name}")
                return {}
                
            # 采样处理
            if sample_size and len(dataset) > sample_size:
                import random
                dataset = random.sample(dataset, sample_size)
                logger.info(f"采样 {sample_size} 个问题进行分析")
                
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            return {}
        
        # 批量分类
        results = {
            'dataset_name': dataset_name,
            'total_problems': len(dataset),
            'classification_results': [],
            'distribution': {level.value: 0 for level in ProblemComplexity},
            'confidence_stats': {level.value: [] for level in ProblemComplexity}
        }
        
        for i, problem in enumerate(dataset):
            problem_text = self._extract_problem_text(problem)
            if not problem_text:
                continue
                
            # 分类单个问题
            classification = self._classify_single_problem(problem_text)
            
            # 记录结果
            results['classification_results'].append({
                'id': problem.get('id', f'problem_{i}'),
                'problem_text': problem_text[:100] + '...' if len(problem_text) > 100 else problem_text,
                'complexity_level': classification['complexity_level'].value,
                'confidence': classification['confidence'],
                'detailed_scores': classification['detailed_scores']
            })
            
            # 更新统计
            level = classification['complexity_level']
            results['distribution'][level.value] += 1
            results['confidence_stats'][level.value].append(classification['confidence'])
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(dataset)} 个问题")
        
        # 计算最终统计
        results['percentage_distribution'] = self._calculate_percentages(results['distribution'], len(dataset))
        results['average_confidence'] = self._calculate_avg_confidence(results['confidence_stats'])
        results['dir_score'] = self._calculate_dir_score(results['distribution'], len(dataset))
        
        logger.info(f"分类完成: {dataset_name}")
        logger.info(f"分布: {results['percentage_distribution']}")
        logger.info(f"DIR分数: {results['dir_score']:.2f}")
        
        return results
    
    def _extract_problem_text(self, problem: Dict) -> str:
        """从问题字典中提取文本"""
        text_fields = ['question', 'problem', 'text', 'statement', 'body']
        for field in text_fields:
            if field in problem and problem[field]:
                return str(problem[field]).strip()
        return ""
    
    def _classify_single_problem(self, problem_text: str) -> Dict[str, Any]:
        """分类单个问题"""
        # 使用现有的复杂度分析器
        analysis = self.complexity_analyzer.analyze_complexity(problem_text)
        
        # 增强判断逻辑
        enhanced_scores = self._apply_enhanced_rules(problem_text)
        
        # 结合两种分析结果
        final_scores = {}
        for level in ProblemComplexity:
            base_score = analysis['complexity_scores'].get(level.value, 0)
            enhanced_score = enhanced_scores.get(level, 0)
            final_scores[level] = (base_score * 0.6 + enhanced_score * 0.4)
        
        # 确定最终复杂度
        final_complexity = max(final_scores.items(), key=lambda x: x[1])[0]
        final_confidence = final_scores[final_complexity]
        
        return {
            'complexity_level': final_complexity,
            'confidence': final_confidence,
            'detailed_scores': {level.value: score for level, score in final_scores.items()},
            'base_analysis': analysis
        }
    
    def _apply_enhanced_rules(self, problem_text: str) -> Dict[ProblemComplexity, float]:
        """应用增强的分类规则"""
        scores = {}
        
        for level, rules in self.enhanced_rules.items():
            score = 0.0
            factors = 0
            
            # 检查数字和操作符数量
            if 'max_numbers' in rules:
                number_count = len([x for x in problem_text if x.isdigit()])
                if number_count <= rules['max_numbers']:
                    score += 0.3
                factors += 1
            
            # 检查关键词
            if 'required_keywords' in rules:
                keyword_matches = sum(1 for kw in rules['required_keywords'] if kw in problem_text.lower())
                if keyword_matches > 0:
                    score += 0.4 * (keyword_matches / len(rules['required_keywords']))
                factors += 1
            
            # 检查禁用关键词 (仅对L0)
            if 'forbidden_keywords' in rules:
                forbidden_matches = sum(1 for kw in rules['forbidden_keywords'] if kw in problem_text.lower())
                if forbidden_matches == 0:
                    score += 0.3
                factors += 1
            
            # 检查复杂模式 (L3)
            if 'complex_patterns' in rules:
                import re
                pattern_matches = sum(1 for pattern in rules['complex_patterns'] 
                                    if re.search(pattern, problem_text.lower()))
                if pattern_matches > 0:
                    score += 0.5
                factors += 1
            
            # 归一化分数
            scores[level] = score / max(factors, 1) if factors > 0 else 0.0
        
        return scores
    
    def _calculate_percentages(self, distribution: Dict, total: int) -> Dict[str, float]:
        """计算百分比分布"""
        if total == 0:
            return {level: 0.0 for level in distribution}
        return {level: (count / total) * 100 for level, count in distribution.items()}
    
    def _calculate_avg_confidence(self, confidence_stats: Dict) -> Dict[str, float]:
        """计算平均置信度"""
        avg_confidence = {}
        for level, confidences in confidence_stats.items():
            if confidences:
                avg_confidence[level] = sum(confidences) / len(confidences)
            else:
                avg_confidence[level] = 0.0
        return avg_confidence
    
    def _calculate_dir_score(self, distribution: Dict, total: int) -> float:
        """计算DIR分数"""
        if total == 0:
            return 0.0
        
        level_weights = {'L0_显式计算': 0, 'L1_浅层推理': 1, 'L2_中等推理': 2, 'L3_深层推理': 3}
        weighted_sum = sum(distribution.get(level, 0) * weight 
                          for level, weight in level_weights.items())
        return weighted_sum / total
    
    def save_results(self, results: Dict, output_file: str):
        """保存分类结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {output_path}")
    
    def generate_summary_report(self, results_list: List[Dict]) -> str:
        """生成汇总报告"""
        report = "# 数据集复杂度分类汇总报告\n\n"
        
        # 汇总表格
        report += "| 数据集 | 总题数 | L0(%) | L1(%) | L2(%) | L3(%) | DIR分数 |\n"
        report += "|--------|--------|-------|-------|-------|-------|----------|\n"
        
        for result in results_list:
            name = result['dataset_name']
            total = result['total_problems']
            dist = result['percentage_distribution']
            dir_score = result['dir_score']
            
            report += f"| {name} | {total:,} | {dist.get('L0_显式计算', 0):.1f} | "
            report += f"{dist.get('L1_浅层推理', 0):.1f} | {dist.get('L2_中等推理', 0):.1f} | "
            report += f"{dist.get('L3_深层推理', 0):.1f} | {dir_score:.2f} |\n"
        
        # 置信度分析
        report += "\n## 分类置信度分析\n\n"
        for result in results_list:
            name = result['dataset_name']
            avg_conf = result['average_confidence']
            report += f"### {name}\n"
            for level, conf in avg_conf.items():
                report += f"- {level}: {conf:.3f}\n"
            report += "\n"
        
        return report


def main():
    """主函数"""
    classifier = BatchComplexityClassifier()
    
    # 获取可用数据集
    available_datasets = classifier.dataset_loader.list_datasets()
    logger.info(f"可用数据集: {available_datasets}")
    
    # 要分类的数据集列表 (包含MATH和MathQA)
    datasets_to_classify = [
        'GSM8K',
        'Math23K', 
        'MAWPS',
        'ASDiv',
        'SVAMP',
        'MATH',
        'MathQA'
    ]
    
    # 过滤只存在的数据集
    datasets_to_classify = [ds for ds in datasets_to_classify if ds in available_datasets]
    logger.info(f"将要分类的数据集: {datasets_to_classify}")
    
    all_results = []
    
    for dataset_name in datasets_to_classify:
        print(f"\n{'='*50}")
        print(f"开始分类数据集: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # 分类数据集 (采样500个问题用于快速测试)
            results = classifier.classify_dataset(dataset_name, sample_size=500)
            
            if results:
                # 保存结果
                output_file = f"classification_results/{dataset_name}_complexity_classification.json"
                classifier.save_results(results, output_file)
                all_results.append(results)
                
                # 打印简要统计
                print(f"\n{dataset_name} 分类结果:")
                for level, percentage in results['percentage_distribution'].items():
                    print(f"  {level}: {percentage:.1f}%")
                print(f"  DIR分数: {results['dir_score']:.2f}")
                
        except Exception as e:
            logger.error(f"分类 {dataset_name} 时出错: {e}")
            continue
    
    # 生成汇总报告
    if all_results:
        summary_report = classifier.generate_summary_report(all_results)
        
        with open("classification_results/complexity_classification_summary.md", 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"\n{'='*50}")
        print("汇总报告已生成: classification_results/complexity_classification_summary.md")
        print(f"{'='*50}")


if __name__ == "__main__":
    main() 