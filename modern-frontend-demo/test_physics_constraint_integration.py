#!/usr/bin/env python3
"""
物理约束传播网络集成测试
测试增强物理约束网络与现有系统的集成效果
"""

import sys
import os
import logging
import time
import json
from typing import Dict, List, Any

# 添加路径以导入后端模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'refactored_backend'))

from enhanced_physical_constraint_network import EnhancedPhysicalConstraintNetwork
from integrated_reasoning_pipeline import IntegratedReasoningPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhysicsConstraintIntegrationTester:
    """物理约束集成测试器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.test_results = []
        
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """运行综合测试"""
        
        print("🧪 物理约束传播网络集成测试")
        print("=" * 60)
        
        test_results = {
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "total_execution_time": 0.0
            },
            "individual_tests": [],
            "performance_metrics": {},
            "integration_analysis": {}
        }
        
        start_time = time.time()
        
        # 测试1: 基础约束网络功能
        self.logger.info("执行测试1: 基础约束网络功能")
        test1_result = self._test_basic_constraint_network()
        test_results["individual_tests"].append(test1_result)
        
        # 测试2: 集成推理管道
        self.logger.info("执行测试2: 集成推理管道")
        test2_result = self._test_integrated_pipeline()
        test_results["individual_tests"].append(test2_result)
        
        # 测试3: 性能基准测试
        self.logger.info("执行测试3: 性能基准测试")
        test3_result = self._test_performance_benchmarks()
        test_results["individual_tests"].append(test3_result)
        
        # 测试4: 错误处理和边界情况
        self.logger.info("执行测试4: 错误处理测试")
        test4_result = self._test_error_handling()
        test_results["individual_tests"].append(test4_result)
        
        # 测试5: 前端数据格式兼容性
        self.logger.info("执行测试5: 前端兼容性测试")
        test5_result = self._test_frontend_compatibility()
        test_results["individual_tests"].append(test5_result)
        
        # 计算汇总结果
        total_time = time.time() - start_time
        passed_tests = sum(1 for test in test_results["individual_tests"] if test["passed"])
        
        test_results["test_summary"].update({
            "total_tests": len(test_results["individual_tests"]),
            "passed_tests": passed_tests,
            "failed_tests": len(test_results["individual_tests"]) - passed_tests,
            "total_execution_time": total_time,
            "success_rate": passed_tests / len(test_results["individual_tests"]) * 100
        })
        
        # 生成性能指标
        test_results["performance_metrics"] = self._generate_performance_metrics(test_results["individual_tests"])
        
        # 生成集成分析
        test_results["integration_analysis"] = self._analyze_integration_quality(test_results["individual_tests"])
        
        return test_results
    
    def _test_basic_constraint_network(self) -> Dict[str, Any]:
        """测试基础约束网络功能"""
        
        test_result = {
            "test_name": "基础约束网络功能",
            "passed": False,
            "execution_time": 0.0,
            "details": {},
            "errors": []
        }
        
        try:
            start_time = time.time()
            
            # 创建约束网络实例
            network = EnhancedPhysicalConstraintNetwork()
            
            # 执行基础测试
            basic_test_result = network.test_constraint_network()
            
            test_result["execution_time"] = time.time() - start_time
            test_result["details"] = {
                "constraint_network_created": True,
                "test_execution_success": basic_test_result["test_success"],
                "laws_identified": basic_test_result["laws_identified"],
                "constraints_generated": basic_test_result["constraints_generated"],
                "satisfaction_rate": basic_test_result["constraint_satisfaction_rate"],
                "physical_consistency": basic_test_result["physical_consistency"]
            }
            
            # 验证结果
            if (basic_test_result["test_success"] and 
                basic_test_result["laws_identified"] > 0 and
                basic_test_result["constraints_generated"] > 0):
                test_result["passed"] = True
            else:
                test_result["errors"].append("基础约束网络测试未通过预期标准")
                
        except Exception as e:
            test_result["errors"].append(f"约束网络测试失败: {str(e)}")
            test_result["execution_time"] = time.time() - start_time
        
        return test_result
    
    def _test_integrated_pipeline(self) -> Dict[str, Any]:
        """测试集成推理管道"""
        
        test_result = {
            "test_name": "集成推理管道",
            "passed": False,
            "execution_time": 0.0,
            "details": {},
            "errors": []
        }
        
        try:
            start_time = time.time()
            
            # 创建集成管道
            pipeline = IntegratedReasoningPipeline()
            
            # 测试问题
            test_problem = "小明有5个苹果，又买了3个苹果，现在总共有多少个苹果？"
            
            # 执行推理
            result = pipeline.solve_problem(test_problem)
            
            test_result["execution_time"] = time.time() - start_time
            test_result["details"] = {
                "pipeline_created": True,
                "reasoning_success": result.success,
                "final_answer": result.final_solution.get("answer"),
                "confidence_score": result.confidence_score,
                "steps_completed": len(result.reasoning_steps),
                "constraint_integration": result.enhanced_constraints.get("success", False),
                "constraint_satisfaction_rate": result.enhanced_constraints.get("network_metrics", {}).get("satisfaction_rate", 0)
            }
            
            # 验证集成效果
            if (result.success and 
                result.final_solution.get("answer") == 8 and  # 5 + 3 = 8
                result.confidence_score > 0.6 and
                result.enhanced_constraints.get("success", False)):
                test_result["passed"] = True
            else:
                test_result["errors"].append(f"集成推理结果不符合预期: 答案={result.final_solution.get('answer')}, 置信度={result.confidence_score}")
                
        except Exception as e:
            test_result["errors"].append(f"集成推理测试失败: {str(e)}")
            test_result["execution_time"] = time.time() - start_time
        
        return test_result
    
    def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """测试性能基准"""
        
        test_result = {
            "test_name": "性能基准测试",
            "passed": False,
            "execution_time": 0.0,
            "details": {},
            "errors": []
        }
        
        try:
            start_time = time.time()
            
            pipeline = IntegratedReasoningPipeline()
            
            # 性能测试用例
            performance_tests = [
                "小明有10个苹果，吃了3个，还剩几个？",
                "教室里有25个学生，其中12个是女生，男生有多少个？", 
                "商店有50个橙子，卖出20个，又进货15个，现在有多少个？",
                "一本书有120页，小红每天读8页，需要多少天读完？",
                "学校有3个班级，每班30个学生，总共有多少学生？"
            ]
            
            execution_times = []
            accuracy_scores = []
            constraint_success_rates = []
            
            for problem in performance_tests:
                problem_start = time.time()
                result = pipeline.solve_problem(problem)
                problem_time = time.time() - problem_start
                
                execution_times.append(problem_time)
                accuracy_scores.append(result.confidence_score)
                
                if result.enhanced_constraints.get("success"):
                    satisfaction_rate = result.enhanced_constraints.get("network_metrics", {}).get("satisfaction_rate", 0)
                    constraint_success_rates.append(satisfaction_rate)
            
            avg_execution_time = sum(execution_times) / len(execution_times)
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            avg_constraint_success = sum(constraint_success_rates) / len(constraint_success_rates) if constraint_success_rates else 0
            
            test_result["execution_time"] = time.time() - start_time
            test_result["details"] = {
                "problems_tested": len(performance_tests),
                "average_execution_time": avg_execution_time,
                "max_execution_time": max(execution_times),
                "min_execution_time": min(execution_times),
                "average_accuracy": avg_accuracy,
                "average_constraint_success_rate": avg_constraint_success,
                "performance_meets_target": avg_execution_time < 1.0  # 目标：1秒内
            }
            
            # 性能验证
            if (avg_execution_time < 1.0 and 
                avg_accuracy > 0.6 and
                avg_constraint_success > 0.8):
                test_result["passed"] = True
            else:
                test_result["errors"].append(f"性能未达标: 平均时间={avg_execution_time:.3f}s, 平均准确度={avg_accuracy:.3f}")
                
        except Exception as e:
            test_result["errors"].append(f"性能测试失败: {str(e)}")
            test_result["execution_time"] = time.time() - start_time
        
        return test_result
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """测试错误处理"""
        
        test_result = {
            "test_name": "错误处理测试",
            "passed": False,
            "execution_time": 0.0,
            "details": {},
            "errors": []
        }
        
        try:
            start_time = time.time()
            
            pipeline = IntegratedReasoningPipeline()
            
            # 错误测试用例
            error_test_cases = [
                "",  # 空字符串
                "这不是一个数学问题",  # 非数学问题
                "小明有苹果个",  # 语法错误
                "1/0 等于多少？",  # 除零错误
                "小明有-5个苹果",  # 负数问题
            ]
            
            error_handling_scores = []
            
            for test_case in error_test_cases:
                try:
                    result = pipeline.solve_problem(test_case)
                    # 检查是否优雅处理错误
                    if not result.success and result.error_message:
                        error_handling_scores.append(1.0)  # 正确处理错误
                    elif result.success:
                        error_handling_scores.append(0.5)  # 可能的假阳性
                    else:
                        error_handling_scores.append(0.0)  # 处理不当
                except Exception:
                    error_handling_scores.append(0.0)  # 未捕获异常
            
            avg_error_handling = sum(error_handling_scores) / len(error_handling_scores)
            
            test_result["execution_time"] = time.time() - start_time
            test_result["details"] = {
                "error_cases_tested": len(error_test_cases),
                "error_handling_score": avg_error_handling,
                "graceful_degradation": avg_error_handling > 0.7
            }
            
            if avg_error_handling > 0.7:
                test_result["passed"] = True
            else:
                test_result["errors"].append(f"错误处理不够健壮: 得分={avg_error_handling:.3f}")
                
        except Exception as e:
            test_result["errors"].append(f"错误处理测试失败: {str(e)}")
            test_result["execution_time"] = time.time() - start_time
        
        return test_result
    
    def _test_frontend_compatibility(self) -> Dict[str, Any]:
        """测试前端兼容性"""
        
        test_result = {
            "test_name": "前端兼容性测试",
            "passed": False,
            "execution_time": 0.0,
            "details": {},
            "errors": []
        }
        
        try:
            start_time = time.time()
            
            pipeline = IntegratedReasoningPipeline()
            result = pipeline.solve_problem("小明有5个苹果，买了3个，总共有多少个？")
            
            # 检查前端所需的数据结构
            required_fields = [
                "enhanced_constraints",
                "final_solution", 
                "reasoning_steps",
                "confidence_score"
            ]
            
            frontend_compatibility_checks = []
            
            # 检查基础字段
            for field in required_fields:
                if hasattr(result, field):
                    frontend_compatibility_checks.append(True)
                else:
                    frontend_compatibility_checks.append(False)
                    test_result["errors"].append(f"缺少前端必需字段: {field}")
            
            # 检查约束数据格式
            if result.enhanced_constraints:
                constraint_format_valid = all(key in result.enhanced_constraints for key in [
                    "applicable_physics_laws", "generated_constraints", "constraint_solution"
                ])
                frontend_compatibility_checks.append(constraint_format_valid)
                if not constraint_format_valid:
                    test_result["errors"].append("约束数据格式不符合前端要求")
            
            # 检查可序列化性
            try:
                json.dumps(result.enhanced_constraints, default=str)
                frontend_compatibility_checks.append(True)
            except Exception as e:
                frontend_compatibility_checks.append(False)
                test_result["errors"].append(f"数据不可序列化: {str(e)}")
            
            compatibility_score = sum(frontend_compatibility_checks) / len(frontend_compatibility_checks)
            
            test_result["execution_time"] = time.time() - start_time
            test_result["details"] = {
                "compatibility_checks": len(frontend_compatibility_checks),
                "compatibility_score": compatibility_score,
                "serializable": len(test_result["errors"]) == 0
            }
            
            if compatibility_score >= 0.9:
                test_result["passed"] = True
            else:
                test_result["errors"].append(f"前端兼容性不足: 得分={compatibility_score:.3f}")
                
        except Exception as e:
            test_result["errors"].append(f"前端兼容性测试失败: {str(e)}")
            test_result["execution_time"] = time.time() - start_time
        
        return test_result
    
    def _generate_performance_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成性能指标"""
        
        total_execution_time = sum(test["execution_time"] for test in test_results)
        passed_tests = [test for test in test_results if test["passed"]]
        
        return {
            "total_execution_time": total_execution_time,
            "average_test_time": total_execution_time / len(test_results),
            "fastest_test": min(test["execution_time"] for test in test_results),
            "slowest_test": max(test["execution_time"] for test in test_results),
            "pass_rate": len(passed_tests) / len(test_results) * 100,
            "performance_grade": self._calculate_performance_grade(test_results)
        }
    
    def _analyze_integration_quality(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析集成质量"""
        
        integration_scores = []
        
        for test in test_results:
            if test["passed"]:
                integration_scores.append(1.0)
            elif test["details"]:
                # 部分成功的测试给予部分分数
                integration_scores.append(0.5)
            else:
                integration_scores.append(0.0)
        
        avg_integration_quality = sum(integration_scores) / len(integration_scores)
        
        return {
            "integration_quality_score": avg_integration_quality,
            "integration_grade": "A" if avg_integration_quality >= 0.9 else 
                               "B" if avg_integration_quality >= 0.7 else
                               "C" if avg_integration_quality >= 0.5 else "D",
            "key_strengths": self._identify_strengths(test_results),
            "improvement_areas": self._identify_improvement_areas(test_results),
            "recommendation": self._generate_recommendation(avg_integration_quality)
        }
    
    def _calculate_performance_grade(self, test_results: List[Dict[str, Any]]) -> str:
        """计算性能等级"""
        
        performance_test = next((test for test in test_results if test["test_name"] == "性能基准测试"), None)
        
        if not performance_test or not performance_test["passed"]:
            return "D"
        
        avg_time = performance_test["details"].get("average_execution_time", 999)
        
        if avg_time < 0.2:
            return "A+"
        elif avg_time < 0.5:
            return "A"
        elif avg_time < 1.0:
            return "B"
        elif avg_time < 2.0:
            return "C"
        else:
            return "D"
    
    def _identify_strengths(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """识别优势"""
        
        strengths = []
        
        for test in test_results:
            if test["passed"]:
                if test["test_name"] == "基础约束网络功能":
                    strengths.append("约束网络核心功能稳定")
                elif test["test_name"] == "集成推理管道":
                    strengths.append("系统集成效果良好")
                elif test["test_name"] == "性能基准测试":
                    strengths.append("性能指标达到预期")
                elif test["test_name"] == "错误处理测试":
                    strengths.append("错误处理机制健壮")
                elif test["test_name"] == "前端兼容性测试":
                    strengths.append("前端集成兼容性好")
        
        return strengths
    
    def _identify_improvement_areas(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """识别改进领域"""
        
        improvements = []
        
        for test in test_results:
            if not test["passed"]:
                improvements.extend(test["errors"])
        
        return improvements
    
    def _generate_recommendation(self, integration_quality: float) -> str:
        """生成建议"""
        
        if integration_quality >= 0.9:
            return "系统集成质量优秀，可以进入生产环境部署"
        elif integration_quality >= 0.7:
            return "系统集成质量良好，建议修复少量问题后部署"
        elif integration_quality >= 0.5:
            return "系统集成质量一般，需要解决主要问题后再考虑部署"
        else:
            return "系统集成质量需要大幅改进，不建议当前部署"

def main():
    """主测试函数"""
    
    tester = PhysicsConstraintIntegrationTester()
    results = tester.run_comprehensive_tests()
    
    # 打印测试结果
    print(f"\n📊 测试汇总报告")
    print("=" * 60)
    
    summary = results["test_summary"]
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过测试: {summary['passed_tests']}")
    print(f"失败测试: {summary['failed_tests']}")
    print(f"成功率: {summary['success_rate']:.1f}%")
    print(f"总执行时间: {summary['total_execution_time']:.3f}秒")
    
    print(f"\n📈 性能指标")
    print("-" * 40)
    perf = results["performance_metrics"]
    print(f"性能等级: {perf['performance_grade']}")
    print(f"平均测试时间: {perf['average_test_time']:.3f}秒")
    print(f"最快测试: {perf['fastest_test']:.3f}秒")
    print(f"最慢测试: {perf['slowest_test']:.3f}秒")
    
    print(f"\n🔍 集成质量分析")
    print("-" * 40)
    integration = results["integration_analysis"]
    print(f"集成质量等级: {integration['integration_grade']}")
    print(f"集成质量得分: {integration['integration_quality_score']:.3f}")
    print(f"建议: {integration['recommendation']}")
    
    print(f"\n✅ 主要优势:")
    for strength in integration["key_strengths"]:
        print(f"  • {strength}")
    
    if integration["improvement_areas"]:
        print(f"\n⚠️  改进领域:")
        for improvement in integration["improvement_areas"][:3]:  # 只显示前3个
            print(f"  • {improvement}")
    
    print(f"\n📋 详细测试结果:")
    for test in results["individual_tests"]:
        status = "✅ 通过" if test["passed"] else "❌ 失败"
        print(f"  {status} - {test['test_name']} ({test['execution_time']:.3f}s)")
        if test["errors"]:
            for error in test["errors"][:2]:  # 只显示前2个错误
                print(f"    ⚠️  {error}")
    
    print(f"\n🎯 结论:")
    if summary['success_rate'] >= 80:
        print("✅ 物理约束传播网络集成成功，系统运行稳定！")
        print("📦 建议: 可以继续进行前端集成和用户测试")
    elif summary['success_rate'] >= 60:
        print("⚠️  物理约束传播网络基本集成成功，存在少量问题")
        print("🔧 建议: 修复发现的问题后进行进一步测试")
    else:
        print("❌ 物理约束传播网络集成存在重大问题")
        print("🚧 建议: 需要解决核心问题后重新测试")

if __name__ == "__main__":
    main()