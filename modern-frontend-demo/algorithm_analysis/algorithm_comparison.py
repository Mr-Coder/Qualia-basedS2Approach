#!/usr/bin/env python3
"""
算法方案对比分析
从泛化性和可解释性角度评估四种算法增强方案
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AlgorithmEvaluation:
    """算法评估指标"""
    name: str
    generalizability: Dict[str, float]  # 泛化性指标
    interpretability: Dict[str, float]  # 可解释性指标
    implementation: Dict[str, float]    # 实现难度指标
    overall_score: float

class AlgorithmComparator:
    """算法对比分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.algorithms = self._initialize_algorithms()
        
    def _initialize_algorithms(self) -> List[AlgorithmEvaluation]:
        """初始化算法评估"""
        
        # 1. 增强QS²语义分析器
        enhanced_qs2 = AlgorithmEvaluation(
            name="增强QS²语义分析器",
            generalizability={
                "cross_domain_adaptability": 0.85,  # 跨领域适应性
                "problem_type_expansion": 0.80,     # 问题类型扩展
                "data_efficiency": 0.60,            # 数据效率（需要大量训练数据）
                "zero_shot_capability": 0.70,       # 零样本能力
                "domain_transfer": 0.75             # 领域迁移能力
            },
            interpretability={
                "reasoning_transparency": 0.70,     # 推理透明度（深度神经网络黑盒）
                "frontend_visualization": 0.85,     # 前端可视化友好度
                "user_comprehension": 0.65,         # 用户理解度
                "educational_value": 0.80,          # 教育价值
                "step_explainability": 0.75         # 步骤可解释性
            },
            implementation={
                "engineering_complexity": 0.30,     # 工程复杂度（高复杂度=低分）
                "computational_cost": 0.40,         # 计算成本
                "maintenance_difficulty": 0.35,     # 维护难度
                "scalability": 0.70,               # 可扩展性
                "stability": 0.60                   # 稳定性
            },
            overall_score=0.0
        )
        
        # 2. GNN增强IRD关系发现器
        gnn_ird = AlgorithmEvaluation(
            name="GNN增强IRD关系发现器",
            generalizability={
                "cross_domain_adaptability": 0.75,  # 图结构适用性强
                "problem_type_expansion": 0.85,     # 关系推理泛化性好
                "data_efficiency": 0.50,            # 需要图结构数据
                "zero_shot_capability": 0.65,       # 图模式泛化
                "domain_transfer": 0.80             # 图结构迁移性好
            },
            interpretability={
                "reasoning_transparency": 0.60,     # GNN相对黑盒
                "frontend_visualization": 0.90,     # 图可视化效果好
                "user_comprehension": 0.70,         # 图结构直观
                "educational_value": 0.85,          # 关系图教育价值高
                "step_explainability": 0.65         # 注意力权重可解释
            },
            implementation={
                "engineering_complexity": 0.25,     # GNN实现复杂
                "computational_cost": 0.35,         # 图计算开销大
                "maintenance_difficulty": 0.30,     # 图数据维护复杂
                "scalability": 0.60,               # 图规模限制
                "stability": 0.55                   # 训练不稳定
            },
            overall_score=0.0
        )
        
        # 3. 强化学习增强COT-DIR
        rl_cotdir = AlgorithmEvaluation(
            name="强化学习增强COT-DIR",
            generalizability={
                "cross_domain_adaptability": 0.90,  # RL适应性极强
                "problem_type_expansion": 0.95,     # 策略学习泛化性好
                "data_efficiency": 0.70,            # 在线学习效率高
                "zero_shot_capability": 0.60,       # 需要探索学习
                "domain_transfer": 0.85             # 策略迁移能力强
            },
            interpretability={
                "reasoning_transparency": 0.45,     # RL策略黑盒化严重
                "frontend_visualization": 0.60,     # 策略可视化困难
                "user_comprehension": 0.40,         # RL概念抽象
                "educational_value": 0.50,          # 教育价值有限
                "step_explainability": 0.55         # 动作选择难解释
            },
            implementation={
                "engineering_complexity": 0.15,     # RL实现最复杂
                "computational_cost": 0.20,         # 训练成本极高
                "maintenance_difficulty": 0.20,     # 超参数调优困难
                "scalability": 0.50,               # 状态空间爆炸
                "stability": 0.40                   # 训练不稳定
            },
            overall_score=0.0
        )
        
        # 4. 物理约束传播网络
        physics_network = AlgorithmEvaluation(
            name="物理约束传播网络",
            generalizability={
                "cross_domain_adaptability": 0.95,  # 物理定律普适性强
                "problem_type_expansion": 0.75,     # 限于物理相关问题
                "data_efficiency": 0.90,            # 基于规则，数据需求低
                "zero_shot_capability": 0.85,       # 物理规则零样本应用
                "domain_transfer": 0.70             # 物理领域内迁移好
            },
            interpretability={
                "reasoning_transparency": 0.95,     # 基于物理定律，透明度极高
                "frontend_visualization": 0.90,     # 约束可视化效果好
                "user_comprehension": 0.85,         # 物理概念易理解
                "educational_value": 0.95,          # 教育价值最高
                "step_explainability": 0.90         # 每步都有物理依据
            },
            implementation={
                "engineering_complexity": 0.75,     # 规则引擎相对简单
                "computational_cost": 0.80,         # 约束求解效率高
                "maintenance_difficulty": 0.70,     # 规则维护相对容易
                "scalability": 0.85,               # 约束数量可控
                "stability": 0.90                   # 基于规则，稳定性高
            },
            overall_score=0.0
        )
        
        algorithms = [enhanced_qs2, gnn_ird, rl_cotdir, physics_network]
        
        # 计算综合得分
        for algo in algorithms:
            generalizability_score = np.mean(list(algo.generalizability.values()))
            interpretability_score = np.mean(list(algo.interpretability.values()))
            implementation_score = np.mean(list(algo.implementation.values()))
            
            # 加权计算总分（泛化性40%，可解释性40%，实现难度20%）
            algo.overall_score = (
                generalizability_score * 0.4 + 
                interpretability_score * 0.4 + 
                implementation_score * 0.2
            )
        
        return algorithms
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """生成对比分析报告"""
        
        self.logger.info("生成算法对比分析报告")
        
        # 1. 综合排名
        ranked_algorithms = sorted(self.algorithms, key=lambda x: x.overall_score, reverse=True)
        
        # 2. 维度分析
        dimension_analysis = self._analyze_by_dimensions()
        
        # 3. 使用场景推荐
        scenario_recommendations = self._generate_scenario_recommendations()
        
        # 4. 实施建议
        implementation_advice = self._generate_implementation_advice()
        
        return {
            "overall_ranking": [
                {
                    "rank": i + 1,
                    "name": algo.name,
                    "overall_score": round(algo.overall_score, 3),
                    "generalizability": round(np.mean(list(algo.generalizability.values())), 3),
                    "interpretability": round(np.mean(list(algo.interpretability.values())), 3),
                    "implementation": round(np.mean(list(algo.implementation.values())), 3)
                }
                for i, algo in enumerate(ranked_algorithms)
            ],
            "dimension_analysis": dimension_analysis,
            "scenario_recommendations": scenario_recommendations,
            "implementation_advice": implementation_advice,
            "detailed_scores": self._get_detailed_scores()
        }
    
    def _analyze_by_dimensions(self) -> Dict[str, Any]:
        """按维度分析"""
        
        # 泛化性最佳
        best_generalizability = max(self.algorithms, 
            key=lambda x: np.mean(list(x.generalizability.values())))
        
        # 可解释性最佳
        best_interpretability = max(self.algorithms, 
            key=lambda x: np.mean(list(x.interpretability.values())))
        
        # 实现难度最低（分数最高）
        easiest_implementation = max(self.algorithms, 
            key=lambda x: np.mean(list(x.implementation.values())))
        
        return {
            "best_generalizability": {
                "algorithm": best_generalizability.name,
                "score": round(np.mean(list(best_generalizability.generalizability.values())), 3),
                "strengths": self._get_top_strengths(best_generalizability.generalizability)
            },
            "best_interpretability": {
                "algorithm": best_interpretability.name,
                "score": round(np.mean(list(best_interpretability.interpretability.values())), 3),
                "strengths": self._get_top_strengths(best_interpretability.interpretability)
            },
            "easiest_implementation": {
                "algorithm": easiest_implementation.name,
                "score": round(np.mean(list(easiest_implementation.implementation.values())), 3),
                "advantages": self._get_top_strengths(easiest_implementation.implementation)
            }
        }
    
    def _generate_scenario_recommendations(self) -> Dict[str, str]:
        """生成使用场景推荐"""
        
        return {
            "教育应用优先": "物理约束传播网络 - 最高的可解释性和教育价值，学生能直观理解推理过程",
            "研究原型开发": "物理约束传播网络 - 实现难度最低，可快速验证概念和算法效果",
            "工业级部署": "增强QS²语义分析器 - 平衡了性能和可解释性，工程化程度较高",
            "前沿研究探索": "强化学习增强COT-DIR - 泛化性最强，但需要长期投入和专业团队",
            "关系挖掘重点": "GNN增强IRD关系发现器 - 在关系发现方面表现突出，图可视化效果好",
            "快速原型验证": "物理约束传播网络 - 基于规则，开发周期短，效果可预期"
        }
    
    def _generate_implementation_advice(self) -> Dict[str, List[str]]:
        """生成实施建议"""
        
        return {
            "短期实施(1-2个月)": [
                "优先选择物理约束传播网络",
                "基于现有QS²+IRD+COT-DIR框架扩展",
                "重点实现核心物理定律约束",
                "完善前端可视化展示"
            ],
            "中期发展(3-6个月)": [
                "集成增强QS²语义分析器的部分功能",
                "添加注意力机制和多层级语义向量",
                "优化物理约束网络的性能",
                "扩展支持的问题类型"
            ],
            "长期规划(6个月以上)": [
                "探索GNN在特定场景下的应用",
                "研究强化学习在推理优化中的潜力",
                "构建完整的多算法融合框架",
                "持续优化和用户反馈迭代"
            ],
            "技术风险控制": [
                "避免过度复杂化系统架构",
                "保持算法结果的可解释性",
                "确保系统稳定性和可维护性",
                "平衡创新性和实用性"
            ]
        }
    
    def _get_top_strengths(self, scores_dict: Dict[str, float], top_k: int = 3) -> List[str]:
        """获取top优势"""
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:top_k]]
    
    def _get_detailed_scores(self) -> Dict[str, Dict[str, float]]:
        """获取详细得分"""
        detailed_scores = {}
        
        for algo in self.algorithms:
            detailed_scores[algo.name] = {
                **algo.generalizability,
                **algo.interpretability,
                **algo.implementation,
                "overall_score": algo.overall_score
            }
        
        return detailed_scores
    
    def visualize_comparison(self) -> Dict[str, Any]:
        """生成可视化对比数据"""
        
        # 准备雷达图数据
        algorithms = [algo.name for algo in self.algorithms]
        
        generalizability_scores = [np.mean(list(algo.generalizability.values())) for algo in self.algorithms]
        interpretability_scores = [np.mean(list(algo.interpretability.values())) for algo in self.algorithms]
        implementation_scores = [np.mean(list(algo.implementation.values())) for algo in self.algorithms]
        
        return {
            "radar_chart_data": {
                "algorithms": algorithms,
                "dimensions": ["泛化性", "可解释性", "实现难度"],
                "scores": [
                    generalizability_scores,
                    interpretability_scores,
                    implementation_scores
                ]
            },
            "bar_chart_data": {
                "algorithms": algorithms,
                "overall_scores": [algo.overall_score for algo in self.algorithms]
            },
            "detailed_metrics": {
                algo.name: {
                    "泛化性指标": algo.generalizability,
                    "可解释性指标": algo.interpretability,
                    "实现难度指标": algo.implementation
                }
                for algo in self.algorithms
            }
        }

# 使用示例和测试
if __name__ == "__main__":
    comparator = AlgorithmComparator()
    
    # 生成对比报告
    report = comparator.generate_comparison_report()
    
    print("=" * 60)
    print("算法对比分析报告")
    print("=" * 60)
    
    print("\n🏆 综合排名:")
    for item in report["overall_ranking"]:
        print(f"{item['rank']}. {item['name']}")
        print(f"   综合得分: {item['overall_score']}")
        print(f"   泛化性: {item['generalizability']} | 可解释性: {item['interpretability']} | 实现难度: {item['implementation']}")
        print()
    
    print("\n📊 维度分析:")
    for dimension, info in report["dimension_analysis"].items():
        print(f"{dimension}: {info['algorithm']} (得分: {info['score']})")
    
    print("\n🎯 使用场景推荐:")
    for scenario, recommendation in report["scenario_recommendations"].items():
        print(f"{scenario}: {recommendation}")
    
    print("\n💡 实施建议:")
    for phase, advice_list in report["implementation_advice"].items():
        print(f"{phase}:")
        for advice in advice_list:
            print(f"  • {advice}")
        print()
    
    # 可视化数据
    viz_data = comparator.visualize_comparison()
    print("\n📈 可视化数据已生成，可用于前端图表展示")