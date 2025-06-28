#!/usr/bin/env python3
"""
AI协作友好模块设计演示

这个脚本展示了如何使用AI协作友好的模块设计，
让AI助手能够轻松理解和扩展数学推理系统。

AI_CONTEXT: 完整的演示程序，展示AI协作设计的所有核心特性
RESPONSIBILITY: 演示各个模块的使用方法和协作方式

运行方式:
    python ai_collaborative_demo.py

AI_INSTRUCTION: 这个演示展示了：
1. AI友好的数据结构使用
2. 标准化接口的实现
3. 配置管理系统
4. 异常处理机制
5. 模块间的协作
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ai_core.interfaces import (MathProblem, OperationType,
                                    PerformanceMetrics, ProblemComplexity,
                                    ProblemType, ReasoningError,
                                    ReasoningResult, ReasoningStep,
                                    ReasoningStrategy, ValidationError,
                                    ValidationResult, Validator,
                                    handle_ai_collaborative_error)
from src.utilities.configuration.config_manager import (
    AICollaborativeConfigManager, ConfigurationSchema,
    create_default_config_manager, create_sample_config_file)


class DemoAlgebraicStrategy:
    """
    演示用的代数推理策略 - AI协作友好实现
    
    AI_CONTEXT: 实现ReasoningStrategy协议的示例策略
    RESPONSIBILITY: 处理简单的代数问题
    
    AI_INSTRUCTION: 这个类展示了如何实现推理策略接口
    """
    
    def can_handle(self, problem: MathProblem) -> bool:
        """
        判断是否能处理给定问题
        
        AI_HINT: 这个策略处理包含等号的代数问题
        """
        return "=" in problem.text and problem.problem_type == ProblemType.ALGEBRA
    
    def solve(self, problem: MathProblem) -> ReasoningResult:
        """
        解决代数问题
        
        AI_HINT: 简化的代数求解，仅用于演示
        """
        steps = []
        
        # 步骤1：识别问题结构
        step1 = ReasoningStep(
            step_id=1,
            operation=OperationType.LOGICAL_REASONING,
            description="识别代数方程结构",
            inputs={"problem_text": problem.text},
            outputs={"equation_type": "linear"},
            confidence=0.9,
            reasoning="检测到线性方程特征",
            is_verified=True,
            verification_method="pattern_matching"
        )
        steps.append(step1)
        
        # 步骤2：简化求解
        step2 = ReasoningStep(
            step_id=2,
            operation=OperationType.EQUATION_SOLVING,
            description="求解方程",
            inputs={"equation": problem.text},
            outputs={"solution": "x = 10"},
            confidence=0.8,
            reasoning="应用线性方程求解方法",
            is_verified=False,
            verification_method=None
        )
        steps.append(step2)
        
        # 创建结果
        result = ReasoningResult(
            problem_id=problem.id,
            final_answer="x = 10",
            reasoning_steps=steps,
            overall_confidence=0.85,
            execution_time=0.05,
            strategy_used="DemoAlgebraicStrategy",
            alternative_strategies=["geometric_solver", "numerical_solver"],
            is_correct=None,  # 需要外部验证
            validation_details={},
            metadata={
                "demo_mode": True,
                "ai_generated": True
            }
        )
        
        return result
    
    def get_confidence(self, problem: MathProblem) -> float:
        """
        获取对问题的置信度
        
        AI_HINT: 基于问题特征评估置信度
        """
        if "x" in problem.text and "=" in problem.text:
            return 0.9
        elif "=" in problem.text:
            return 0.7
        else:
            return 0.3


class DemoValidator:
    """
    演示用的验证器 - AI协作友好实现
    
    AI_CONTEXT: 实现Validator协议的示例验证器
    RESPONSIBILITY: 验证推理结果的正确性
    """
    
    def validate(self, target: ReasoningResult) -> ValidationResult:
        """
        验证推理结果
        
        AI_HINT: 检查推理过程的逻辑一致性
        """
        errors = []
        warnings = []
        suggestions = []
        
        # 检查推理步骤
        if not target.reasoning_steps:
            errors.append("缺少推理步骤")
        
        # 检查置信度
        if target.overall_confidence < 0.5:
            warnings.append(f"整体置信度较低: {target.overall_confidence}")
        
        # 检查步骤一致性
        for i, step in enumerate(target.reasoning_steps):
            if step.confidence < 0.3:
                warnings.append(f"步骤 {i+1} 置信度过低: {step.confidence}")
        
        # 生成建议
        if warnings:
            suggestions.append("考虑使用其他策略验证结果")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            target_type="ReasoningResult",
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            fix_recommendations=[
                "增加验证步骤",
                "提高置信度阈值",
                "添加交叉验证"
            ] if not is_valid else [],
            validation_method="DemoValidator",
            confidence_score=0.8 if is_valid else 0.4,
            details={
                "total_steps": len(target.reasoning_steps),
                "avg_confidence": sum(s.confidence for s in target.reasoning_steps) / len(target.reasoning_steps) if target.reasoning_steps else 0
            }
        )
    
    def get_error_details(self, target: ReasoningResult) -> list[str]:
        """获取详细错误信息"""
        return self.validate(target).errors
    
    def suggest_fixes(self, target: ReasoningResult) -> list[str]:
        """提供修复建议"""
        return self.validate(target).fix_recommendations


def demonstrate_data_structures():
    """
    演示AI友好的数据结构使用
    
    AI_HINT: 展示如何创建和使用结构化数据
    """
    print("🏗️ AI友好数据结构演示")
    print("=" * 50)
    
    # 创建数学问题
    problem = MathProblem(
        id="demo_001",
        text="如果 2x + 5 = 15，求 x 的值",
        answer=5,
        complexity=ProblemComplexity.L1,
        problem_type=ProblemType.ALGEBRA,
        entities={
            "variables": ["x"],
            "constants": [2, 5, 15],
            "operations": ["+", "="]
        },
        constraints=["x 必须是实数"],
        target_variable="x",
        source="ai_demo",
        difficulty_score=0.3,
        metadata={"created_for": "ai_collaboration_demo"}
    )
    
    print(f"📝 创建问题: {problem.text}")
    print(f"🎯 复杂度: {problem.complexity.value}")
    print(f"📊 类型: {problem.problem_type.value}")
    print(f"🔍 实体: {problem.entities}")
    print()
    
    return problem


def demonstrate_strategy_usage(problem: MathProblem):
    """
    演示推理策略的使用
    
    AI_HINT: 展示如何使用策略解决问题
    """
    print("🧠 AI友好推理策略演示")
    print("=" * 50)
    
    strategy = DemoAlgebraicStrategy()
    
    # 检查策略适用性
    can_handle = strategy.can_handle(problem)
    confidence = strategy.get_confidence(problem)
    
    print(f"✅ 策略适用性: {can_handle}")
    print(f"🎯 置信度: {confidence:.2f}")
    
    if can_handle:
        try:
            # 执行推理
            result = strategy.solve(problem)
            
            print(f"💡 最终答案: {result.final_answer}")
            print(f"⏱️ 执行时间: {result.execution_time:.3f}秒")
            print(f"📈 整体置信度: {result.overall_confidence:.2f}")
            print(f"🔧 使用策略: {result.strategy_used}")
            
            print(f"\n📋 推理步骤:")
            for step in result.reasoning_steps:
                print(f"  {step.step_id}. {step.description}")
                print(f"     操作: {step.operation.value}")
                print(f"     置信度: {step.confidence:.2f}")
                print(f"     推理: {step.reasoning}")
                print()
            
            return result
            
        except ReasoningError as e:
            print(f"❌ 推理失败: {e.message}")
            error_info = handle_ai_collaborative_error(e)
            print(f"🔧 修复建议: {error_info['fix_recommendations']}")
            return None
    
    else:
        print("❌ 策略无法处理此问题")
        return None


def demonstrate_validation(result: ReasoningResult):
    """
    演示验证器的使用
    
    AI_HINT: 展示如何验证推理结果
    """
    print("🔍 AI友好验证器演示")
    print("=" * 50)
    
    validator = DemoValidator()
    
    try:
        # 执行验证
        validation_result = validator.validate(result)
        
        print(f"✅ 验证通过: {validation_result.is_valid}")
        print(f"🎯 验证置信度: {validation_result.confidence_score:.2f}")
        print(f"🔧 验证方法: {validation_result.validation_method}")
        
        if validation_result.errors:
            print(f"\n❌ 错误:")
            for error in validation_result.errors:
                print(f"  - {error}")
        
        if validation_result.warnings:
            print(f"\n⚠️ 警告:")
            for warning in validation_result.warnings:
                print(f"  - {warning}")
        
        if validation_result.suggestions:
            print(f"\n💡 建议:")
            for suggestion in validation_result.suggestions:
                print(f"  - {suggestion}")
        
        if validation_result.fix_recommendations:
            print(f"\n🔧 修复建议:")
            for rec in validation_result.fix_recommendations:
                print(f"  - {rec}")
        
        print(f"\n📊 验证详情: {validation_result.details}")
        
    except ValidationError as e:
        print(f"❌ 验证失败: {e.message}")
        error_info = handle_ai_collaborative_error(e)
        print(f"🔧 处理建议: {error_info['suggestions']}")


def demonstrate_configuration():
    """
    演示配置管理系统
    
    AI_HINT: 展示AI友好的配置管理
    """
    print("⚙️ AI友好配置管理演示")
    print("=" * 50)
    
    # 创建示例配置文件
    config_file = "demo_config.json"
    create_sample_config_file(config_file)
    print(f"📄 创建示例配置文件: {config_file}")
    
    # 创建配置管理器
    config = create_default_config_manager()
    
    # 加载配置
    try:
        config.load_config(config_file)
        print("✅ 配置加载成功")
        
        # 获取配置值
        log_level = config.get("logging.level")
        max_steps = config.get("reasoning.max_steps")
        confidence_threshold = config.get("reasoning.confidence_threshold")
        
        print(f"📊 当前配置:")
        print(f"  日志级别: {log_level}")
        print(f"  最大步数: {max_steps}")
        print(f"  置信度阈值: {confidence_threshold}")
        
        # 修改配置
        config.set("reasoning.confidence_threshold", 0.9)
        print(f"✏️ 修改置信度阈值为: 0.9")
        
        # 获取AI友好摘要
        summary = config.get_ai_friendly_summary()
        print(f"\n🤖 AI友好摘要:")
        print(f"  配置名称: {summary['config_name']}")
        print(f"  配置项数量: {summary['schema_count']}")
        print(f"  验证状态: {summary['validation_status']}")
        
        # 保存配置
        config.save_config()
        print("💾 配置保存成功")
        
    except Exception as e:
        print(f"❌ 配置操作失败: {str(e)}")
    
    finally:
        # 清理演示文件
        if Path(config_file).exists():
            Path(config_file).unlink()


def demonstrate_performance_tracking():
    """
    演示性能跟踪
    
    AI_HINT: 展示性能指标的收集和分析
    """
    print("📈 AI友好性能跟踪演示")
    print("=" * 50)
    
    # 创建性能指标
    metrics = PerformanceMetrics(
        operation_count=100,
        total_duration=5.5,
        average_duration=0.055,
        min_duration=0.01,
        max_duration=0.15,
        std_duration=0.025,
        success_count=95,
        failure_count=5,
        success_rate=0.95,
        memory_usage={"peak": 128.5, "average": 85.2},
        cpu_usage={"peak": 75.0, "average": 45.3},
        measurement_period={
            "start": datetime.now().replace(hour=10, minute=0),
            "end": datetime.now().replace(hour=10, minute=5)
        },
        operation_metrics={
            "reasoning": {"avg_duration": 0.08, "success_rate": 0.92},
            "validation": {"avg_duration": 0.03, "success_rate": 0.98}
        }
    )
    
    print(f"🔢 操作总数: {metrics.operation_count}")
    print(f"⏱️ 总耗时: {metrics.total_duration:.2f}秒")
    print(f"📊 平均耗时: {metrics.average_duration:.3f}秒")
    print(f"✅ 成功率: {metrics.success_rate:.1%}")
    print(f"💾 内存使用 - 峰值: {metrics.memory_usage['peak']:.1f}MB")
    print(f"🖥️ CPU使用 - 峰值: {metrics.cpu_usage['peak']:.1f}%")
    
    print(f"\n📋 分类指标:")
    for operation, op_metrics in metrics.operation_metrics.items():
        print(f"  {operation}:")
        print(f"    平均耗时: {op_metrics['avg_duration']:.3f}秒")
        print(f"    成功率: {op_metrics['success_rate']:.1%}")


def demonstrate_error_handling():
    """
    演示AI友好的错误处理
    
    AI_HINT: 展示结构化异常处理
    """
    print("🚨 AI友好错误处理演示")
    print("=" * 50)
    
    # 模拟推理错误
    try:
        raise ReasoningError(
            "推理策略执行失败",
            strategy_name="DemoFailStrategy",
            problem_id="demo_002",
            reasoning_step=3,
            context={"input_data": "invalid_format"}
        )
    except ReasoningError as e:
        print(f"❌ 捕获推理错误: {e.message}")
        
        error_info = handle_ai_collaborative_error(e)
        print(f"\n🔍 错误分析:")
        print(f"  错误类型: {error_info['error_type']}")
        print(f"  错误代码: {error_info['error_code']}")
        print(f"  严重程度: {error_info['severity']}")
        
        print(f"\n📋 上下文信息:")
        for key, value in error_info['context'].items():
            print(f"  {key}: {value}")
        
        print(f"\n💡 AI建议:")
        for suggestion in error_info['suggestions']:
            print(f"  - {suggestion}")
        
        print(f"\n🔧 修复建议:")
        for recommendation in error_info['fix_recommendations']:
            print(f"  - {recommendation}")
        
        print(f"\n📝 处理步骤:")
        for step in error_info['handling_steps']:
            print(f"  {step}")


def main():
    """
    主演示函数
    
    AI_HINT: 完整的AI协作友好模块设计演示
    """
    print("🤖 AI协作友好模块设计演示")
    print("=" * 60)
    print("这个演示展示了如何构建AI助手能够轻松理解和扩展的模块化系统")
    print("=" * 60)
    print()
    
    # 1. 数据结构演示
    problem = demonstrate_data_structures()
    print()
    
    # 2. 推理策略演示
    result = demonstrate_strategy_usage(problem)
    print()
    
    # 3. 验证器演示
    if result:
        demonstrate_validation(result)
        print()
    
    # 4. 配置管理演示
    demonstrate_configuration()
    print()
    
    # 5. 性能跟踪演示
    demonstrate_performance_tracking()
    print()
    
    # 6. 错误处理演示
    demonstrate_error_handling()
    print()
    
    print("🎉 AI协作友好模块设计演示完成！")
    print("\n📚 AI学习要点:")
    print("1. 使用类型注解和丰富的文档字符串")
    print("2. 实现标准化的协议接口")
    print("3. 提供结构化的错误信息和修复建议")
    print("4. 使用配置驱动的灵活设计")
    print("5. 包含详细的性能和质量指标")
    print("6. 保持模块间的松耦合关系")
    
    print("\n🔧 对AI助手的指导:")
    print("- 所有组件都遵循清晰的协议接口")
    print("- 错误信息包含上下文和修复建议")
    print("- 配置系统支持动态修改和验证")
    print("- 性能指标提供系统状态的完整视图")
    print("- 代码结构便于理解和扩展")


if __name__ == "__main__":
    main() 