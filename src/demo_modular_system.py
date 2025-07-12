#!/usr/bin/env python3
"""
模块化数学推理系统演示

展示新模块化架构的使用方式和功能。
"""

import sys
import time
from typing import Dict, List

# 导入核心系统组件
from core import ModuleInfo, ModuleType, registry, system_orchestrator
from reasoning import ReasoningAPI


def setup_logging():
    """设置日志"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def register_modules():
    """注册系统模块"""
    logger = setup_logging()
    
    # 创建推理模块信息
    reasoning_info = ModuleInfo(
        name="reasoning",
        type=ModuleType.REASONING,
        version="1.0.0",
        dependencies=[],
        public_api_class="ReasoningAPI",
        orchestrator_class="ReasoningOrchestrator"
    )
    
    # 创建并注册推理模块
    reasoning_api = ReasoningAPI()
    
    try:
        registry.register_module(reasoning_info, reasoning_api)
        logger.info("✅ 推理模块注册成功")
        return True
    except Exception as e:
        logger.error(f"❌ 推理模块注册失败: {e}")
        return False


def test_basic_reasoning():
    """测试基础推理功能"""
    logger = setup_logging()
    logger.info("\n🧠 测试基础推理功能")
    
    test_problems = [
        {
            "problem": "小明有100元，买了30元的书，还剩多少钱？",
            "expected": "70"
        },
        {
            "problem": "一个长方形的长是8米，宽是5米，面积是多少？",
            "expected": "40"
        },
        {
            "problem": "15 + 25 = ?",
            "expected": "40"
        },
        {
            "problem": "商品原价200元，打8折，现价多少？",
            "expected": "160"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_problems, 1):
        try:
            logger.info(f"\n📝 测试 {i}: {test_case['problem']}")
            
            # 使用系统级协调器求解
            result = system_orchestrator.solve_math_problem({
                "problem": test_case["problem"]
            })
            
            answer = result.get("final_answer", "unknown")
            confidence = result.get("confidence", 0.0)
            strategy = result.get("strategy_used", "unknown")
            
            logger.info(f"💡 答案: {answer}")
            logger.info(f"🎯 置信度: {confidence:.2f}")
            logger.info(f"📋 策略: {strategy}")
            
            # 检查答案是否正确
            is_correct = str(answer) == test_case["expected"]
            status = "✅ 正确" if is_correct else "❌ 错误"
            logger.info(f"📊 结果: {status} (期望: {test_case['expected']})")
            
            results.append({
                "problem": test_case["problem"],
                "answer": answer,
                "expected": test_case["expected"],
                "correct": is_correct,
                "confidence": confidence,
                "strategy": strategy
            })
            
        except Exception as e:
            logger.error(f"❌ 测试 {i} 失败: {e}")
            results.append({
                "problem": test_case["problem"],
                "error": str(e),
                "correct": False
            })
    
    return results


def test_batch_processing():
    """测试批量处理功能"""
    logger = setup_logging()
    logger.info("\n📦 测试批量处理功能")
    
    problems = [
        {"problem": "3 + 5 = ?"},
        {"problem": "10 - 4 = ?"},
        {"problem": "2 × 6 = ?"},
        {"problem": "15 ÷ 3 = ?"},
        {"problem": "小红有20个苹果，吃了5个，还有多少个？"}
    ]
    
    try:
        start_time = time.time()
        results = system_orchestrator.batch_solve_problems(problems)
        processing_time = time.time() - start_time
        
        logger.info(f"⏱️  批量处理完成，耗时: {processing_time:.2f}秒")
        logger.info(f"📊 处理了 {len(problems)} 个问题，获得 {len(results)} 个结果")
        
        # 显示结果摘要
        correct_answers = sum(1 for r in results if r.get("final_answer") != "unknown" and "error" not in r)
        success_rate = correct_answers / len(results) * 100
        
        logger.info(f"✅ 成功率: {success_rate:.1f}% ({correct_answers}/{len(results)})")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 批量处理失败: {e}")
        return []


def test_system_status():
    """测试系统状态监控"""
    logger = setup_logging()
    logger.info("\n🔍 检查系统状态")
    
    try:
        # 获取系统状态
        status = system_orchestrator.get_system_status()
        
        logger.info(f"🟢 系统状态: {status['status']}")
        logger.info(f"📈 模块数量: {status['total_modules']}")
        logger.info(f"🎯 系统能力: {', '.join(status['capabilities'])}")
        
        # 显示模块详情
        logger.info("\n📋 模块详情:")
        for module in status["modules"]:
            name = module["name"]
            module_type = module["type"]
            version = module["version"]
            health = module["health"].get("status", "unknown")
            
            logger.info(f"  • {name} (v{version}) - {module_type} - {health}")
        
        return status
        
    except Exception as e:
        logger.error(f"❌ 获取系统状态失败: {e}")
        return {}


def generate_report(basic_results: List[Dict], batch_results: List[Dict], 
                   system_status: Dict) -> None:
    """生成测试报告"""
    logger = setup_logging()
    logger.info("\n📊 生成测试报告")
    
    # 基础测试统计
    basic_correct = sum(1 for r in basic_results if r.get("correct", False))
    basic_total = len(basic_results)
    basic_success_rate = basic_correct / basic_total * 100 if basic_total > 0 else 0
    
    # 批量测试统计
    batch_success = sum(1 for r in batch_results if r.get("final_answer") != "unknown" and "error" not in r)
    batch_total = len(batch_results)
    batch_success_rate = batch_success / batch_total * 100 if batch_total > 0 else 0
    
    # 平均置信度
    confidences = [r.get("confidence", 0) for r in basic_results if "confidence" in r]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    report = f"""
🔬 模块化数学推理系统测试报告
{'='*50}

📊 基础推理测试
  • 测试数量: {basic_total}
  • 正确答案: {basic_correct}
  • 成功率: {basic_success_rate:.1f}%
  • 平均置信度: {avg_confidence:.2f}

📦 批量处理测试
  • 测试数量: {batch_total}
  • 成功处理: {batch_success}
  • 成功率: {batch_success_rate:.1f}%

🏗️ 系统架构
  • 系统状态: {system_status.get('status', '未知')}
  • 注册模块: {system_status.get('total_modules', 0)}
  • 系统能力: {len(system_status.get('capabilities', []))}

🎯 推理策略分布
"""
    
    # 策略使用统计
    strategy_counts = {}
    for result in basic_results:
        strategy = result.get("strategy", "unknown")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    for strategy, count in strategy_counts.items():
        percentage = count / basic_total * 100 if basic_total > 0 else 0
        report += f"  • {strategy}: {count} 次 ({percentage:.1f}%)\n"
    
    report += f"\n{'='*50}"
    
    logger.info(report)


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("🚀 启动模块化数学推理系统演示")
    
    try:
        # 1. 注册模块
        if not register_modules():
            logger.error("❌ 模块注册失败，退出演示")
            return False
        
        # 2. 初始化系统
        if not system_orchestrator.initialize_system():
            logger.error("❌ 系统初始化失败，退出演示")
            return False
        
        logger.info("✅ 系统初始化成功")
        
        # 3. 运行测试
        basic_results = test_basic_reasoning()
        batch_results = test_batch_processing()
        system_status = test_system_status()
        
        # 4. 生成报告
        generate_report(basic_results, batch_results, system_status)
        
        # 5. 系统关闭
        logger.info("\n🔄 关闭系统...")
        system_orchestrator.shutdown_system()
        logger.info("✅ 系统已安全关闭")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  用户中断，正在关闭系统...")
        system_orchestrator.shutdown_system()
        return False
        
    except Exception as e:
        logger.error(f"❌ 演示过程发生错误: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 