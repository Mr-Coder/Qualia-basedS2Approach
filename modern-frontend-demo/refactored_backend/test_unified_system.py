#!/usr/bin/env python3
"""
统一系统测试脚本
测试QS²+IRD+COT-DIR完整推理流程
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any

# 导入统一推理系统
from unified_backend_server import UnifiedReasoningSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.reasoning_system = UnifiedReasoningSystem()
        
        # 测试用例
        self.test_cases = [
            {
                "name": "简单加法问题",
                "problem": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
                "expected_answer": "8个",
                "mode": "auto"
            },
            {
                "name": "简单减法问题", 
                "problem": "小明有10个苹果，吃了3个，还剩多少个？",
                "expected_answer": "7个",
                "mode": "auto"
            },
            {
                "name": "面积计算问题",
                "problem": "一个长方形的长是8米，宽是6米，面积是多少平方米？",
                "expected_answer": "48平方米",
                "mode": "advanced"
            },
            {
                "name": "金钱计算问题",
                "problem": "小华有5元钱，妈妈给了他3元钱，他现在有多少钱？",
                "expected_answer": "8元", 
                "mode": "simple"
            }
        ]

    async def run_all_tests(self):
        """运行所有测试"""
        
        self.logger.info("开始系统集成测试")
        self.logger.info("="*80)
        
        results = []
        
        for i, test_case in enumerate(self.test_cases):
            self.logger.info(f"\n测试用例 {i+1}: {test_case['name']}")
            self.logger.info(f"问题: {test_case['problem']}")
            self.logger.info(f"推理模式: {test_case['mode']}")
            self.logger.info(f"期望答案: {test_case['expected_answer']}")
            
            try:
                start_time = time.time()
                
                # 执行推理
                result = await self.reasoning_system.solve_problem(
                    problem_text=test_case['problem'],
                    mode=test_case['mode']
                )
                
                execution_time = time.time() - start_time
                
                # 分析结果
                success = result.get('success', False)
                answer = result.get('answer', '无答案')
                confidence = result.get('confidence', 0.0)
                strategy = result.get('strategy_used', '未知策略')
                
                self.logger.info(f"执行结果:")
                self.logger.info(f"  成功: {success}")
                self.logger.info(f"  答案: {answer}")
                self.logger.info(f"  置信度: {confidence:.3f}")
                self.logger.info(f"  策略: {strategy}")
                self.logger.info(f"  执行时间: {execution_time:.3f}s")
                
                # 验证答案
                answer_correct = self._check_answer(answer, test_case['expected_answer'])
                self.logger.info(f"  答案正确性: {'✓' if answer_correct else '✗'}")
                
                # 分析推理步骤
                steps = result.get('reasoning_steps', [])
                self.logger.info(f"  推理步骤数: {len(steps)}")
                
                if steps:
                    self.logger.info(f"  推理步骤概览:")
                    for j, step in enumerate(steps[:3]):  # 显示前3步
                        self.logger.info(f"    {j+1}. {step.get('action', '未知动作')}: {step.get('description', '无描述')[:50]}...")
                
                # 分析实体关系图
                erd = result.get('entity_relationship_diagram', {})
                entities = erd.get('entities', [])
                relationships = erd.get('relationships', [])
                
                self.logger.info(f"  实体关系图:")
                self.logger.info(f"    实体数: {len(entities)}")
                self.logger.info(f"    关系数: {len(relationships)}")
                
                if entities:
                    self.logger.info(f"    实体列表: {[e.get('name', '未知') for e in entities[:5]]}")
                
                # 记录测试结果
                test_result = {
                    'test_case': test_case['name'],
                    'success': success,
                    'answer_correct': answer_correct,
                    'execution_time': execution_time,
                    'confidence': confidence,
                    'strategy_used': strategy,
                    'entities_count': len(entities),
                    'relationships_count': len(relationships),
                    'reasoning_steps_count': len(steps)
                }
                
                results.append(test_result)
                
            except Exception as e:
                self.logger.error(f"测试失败: {e}")
                results.append({
                    'test_case': test_case['name'],
                    'success': False,
                    'error': str(e)
                })
        
        # 生成测试报告
        self._generate_test_report(results)
        
        return results

    def _check_answer(self, actual_answer: str, expected_answer: str) -> bool:
        """检查答案正确性"""
        
        # 提取数字进行比较
        import re
        
        actual_numbers = re.findall(r'\d+', actual_answer)
        expected_numbers = re.findall(r'\d+', expected_answer)
        
        if actual_numbers and expected_numbers:
            return actual_numbers[0] == expected_numbers[0]
        
        return actual_answer.strip() == expected_answer.strip()

    def _generate_test_report(self, results: List[Dict[str, Any]]):
        """生成测试报告"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("测试报告")
        self.logger.info("="*80)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get('success', False))
        correct_answers = sum(1 for r in results if r.get('answer_correct', False))
        
        self.logger.info(f"总测试数: {total_tests}")
        self.logger.info(f"成功执行: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        self.logger.info(f"答案正确: {correct_answers} ({correct_answers/total_tests*100:.1f}%)")
        
        # 平均执行时间
        avg_time = sum(r.get('execution_time', 0) for r in results) / max(total_tests, 1)
        self.logger.info(f"平均执行时间: {avg_time:.3f}s")
        
        # 平均置信度
        avg_confidence = sum(r.get('confidence', 0) for r in results) / max(total_tests, 1)
        self.logger.info(f"平均置信度: {avg_confidence:.3f}")
        
        # 策略使用分布
        strategies = {}
        for result in results:
            strategy = result.get('strategy_used', '未知')
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        self.logger.info(f"\n策略使用分布:")
        for strategy, count in strategies.items():
            self.logger.info(f"  {strategy}: {count} ({count/total_tests*100:.1f}%)")
        
        # 详细结果
        self.logger.info(f"\n详细测试结果:")
        for i, result in enumerate(results):
            status = "✓" if result.get('success', False) else "✗"
            correct = "✓" if result.get('answer_correct', False) else "✗"
            self.logger.info(f"  {i+1}. {result['test_case']}: 执行{status} 正确{correct} " + 
                          f"时间{result.get('execution_time', 0):.2f}s " +
                          f"置信{result.get('confidence', 0):.2f}")

    async def test_system_performance(self):
        """测试系统性能"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("系统性能测试")
        self.logger.info("="*80)
        
        test_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
        
        # 测试不同模式的性能
        modes = ["simple", "advanced", "auto"]
        
        for mode in modes:
            self.logger.info(f"\n测试{mode}模式性能:")
            
            times = []
            successes = 0
            
            for i in range(5):  # 运行5次
                try:
                    start_time = time.time()
                    result = await self.reasoning_system.solve_problem(test_problem, mode)
                    execution_time = time.time() - start_time
                    
                    times.append(execution_time)
                    if result.get('success', False):
                        successes += 1
                        
                except Exception as e:
                    self.logger.error(f"性能测试失败: {e}")
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                success_rate = successes / len(times)
                
                self.logger.info(f"  平均时间: {avg_time:.3f}s")
                self.logger.info(f"  最短时间: {min_time:.3f}s") 
                self.logger.info(f"  最长时间: {max_time:.3f}s")
                self.logger.info(f"  成功率: {success_rate:.1%}")

    async def test_system_status(self):
        """测试系统状态"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("系统状态测试")
        self.logger.info("="*80)
        
        status = self.reasoning_system.get_system_status()
        
        self.logger.info(f"系统状态: {status['status']}")
        self.logger.info(f"运行时间: {status['uptime_seconds']:.1f}秒")
        self.logger.info(f"处理请求数: {status['requests_processed']}")
        self.logger.info(f"成功率: {status['success_rate']:.1%}")
        
        self.logger.info(f"\n模块状态:")
        for module, state in status['modules'].items():
            self.logger.info(f"  {module}: {state}")
        
        self.logger.info(f"\n引擎状态:")
        for engine, engine_status in status['engine_status'].items():
            self.logger.info(f"  {engine}:")
            self.logger.info(f"    可用: {engine_status['available']}")
            self.logger.info(f"    性能评分: {engine_status['performance_score']:.3f}")
            self.logger.info(f"    错误计数: {engine_status['error_count']}")
            self.logger.info(f"    平均响应时间: {engine_status['average_response_time']:.3f}s")

async def main():
    """主函数"""
    
    print("QS²+IRD+COT-DIR 统一推理系统集成测试")
    print("="*80)
    
    tester = SystemTester()
    
    try:
        # 系统状态测试
        await tester.test_system_status()
        
        # 功能测试
        await tester.run_all_tests()
        
        # 性能测试
        await tester.test_system_performance()
        
        print("\n" + "="*80)
        print("所有测试完成！")
        print("="*80)
        
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())