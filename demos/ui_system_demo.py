#!/usr/bin/env python3
"""
COT-DIR UI系统演示
展示完整的UI系统功能，包括组件交互、事件处理、错误处理等
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """主演示函数"""
    print("🎨 COT-DIR UI系统演示")
    print("=" * 80)
    
    # 运行各个演示
    demo_sections = [
        ("1. UI接口和数据结构", demo_ui_interfaces),
        ("2. UI核心系统", demo_ui_core_system),
        ("3. UI组件功能", demo_ui_components),
        ("4. UI错误处理", demo_ui_error_handling),
        ("5. UI完整工作流程", demo_complete_workflow),
        ("6. UI性能和监控", demo_performance_monitoring)
    ]
    
    for section_name, demo_func in demo_sections:
        print(f"\n{section_name}")
        print("-" * 60)
        try:
            demo_func()
            print("✅ 演示完成")
        except Exception as e:
            print(f"❌ 演示失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🎉 UI系统演示完成！")
    print("=" * 80)


def demo_ui_interfaces():
    """演示UI接口和数据结构"""
    print("📋 演示UI接口和数据结构...")
    
    try:
        from ui.interfaces import (
            UIRequest, UIResponse, UIEvent, UIComponentState,
            UIEventType, UIResponseType, UIComponentType,
            UIUtils
        )
        
        # 1. 创建UI请求
        print("\n1. 创建UI请求:")
        request = UIRequest(
            request_id=UIUtils.generate_request_id(),
            component_id="problem_input_001",
            action="submit_problem",
            data={
                "problem_text": "小明有10个苹果，给了小红3个，还剩几个？",
                "problem_type": "math_word_problem",
                "difficulty": "easy"
            },
            timestamp=datetime.now()
        )
        
        print(f"  - 请求ID: {request.request_id}")
        print(f"  - 组件ID: {request.component_id}")
        print(f"  - 动作: {request.action}")
        print(f"  - 数据: {json.dumps(request.data, ensure_ascii=False, indent=4)}")
        
        # 2. 创建UI响应
        print("\n2. 创建UI响应:")
        response = UIResponse(
            request_id=request.request_id,
            response_type=UIResponseType.SUCCESS,
            data={
                "validation_result": "valid",
                "processing_started": True
            },
            message="请求处理成功"
        )
        
        print(f"  - 响应类型: {response.response_type.value}")
        print(f"  - 消息: {response.message}")
        print(f"  - 数据: {json.dumps(response.data, ensure_ascii=False, indent=4)}")
        
        # 3. 创建UI事件
        print("\n3. 创建UI事件:")
        event = UIEvent(
            event_id=UIUtils.generate_event_id(),
            event_type=UIEventType.PROBLEM_SUBMIT,
            source_component="problem_input_001",
            data={
                "problem_text": "小明有10个苹果，给了小红3个，还剩几个？",
                "submitted_at": datetime.now().isoformat()
            },
            timestamp=datetime.now()
        )
        
        print(f"  - 事件ID: {event.event_id}")
        print(f"  - 事件类型: {event.event_type.value}")
        print(f"  - 源组件: {event.source_component}")
        
        # 4. 创建组件状态
        print("\n4. 创建组件状态:")
        state = UIComponentState(
            component_id="problem_input_001",
            component_type=UIComponentType.INPUT,
            state={
                "problem_text": "小明有10个苹果，给了小红3个，还剩几个？",
                "validation_status": "valid",
                "submit_count": 1
            },
            visible=True,
            enabled=True
        )
        
        print(f"  - 组件ID: {state.component_id}")
        print(f"  - 组件类型: {state.component_type.value}")
        print(f"  - 可见性: {state.visible}")
        print(f"  - 启用状态: {state.enabled}")
        
        # 5. 演示数据序列化
        print("\n5. 数据序列化:")
        request_dict = request.to_dict()
        print(f"  - 请求序列化: {len(json.dumps(request_dict))} 字符")
        
        # 从字典还原
        restored_request = UIRequest.from_dict(request_dict)
        print(f"  - 数据还原成功: {restored_request.request_id == request.request_id}")
        
        # 6. 演示工具函数
        print("\n6. 工具函数:")
        print(f"  - 生成请求ID: {UIUtils.generate_request_id()}")
        print(f"  - 生成事件ID: {UIUtils.generate_event_id()}")
        print(f"  - 清理输入: '{UIUtils.sanitize_input('<script>alert(1)</script>Hello')}'")
        print(f"  - 格式化置信度: {UIUtils.format_confidence(0.875)}")
        print(f"  - 格式化处理时间: {UIUtils.format_processing_time(1234.5)}")
        
    except Exception as e:
        print(f"❌ 接口演示失败: {e}")
        raise


def demo_ui_core_system():
    """演示UI核心系统"""
    print("⚙️ 演示UI核心系统...")
    
    try:
        from ui.core import UIManager, UIRenderer, UIEventHandler, UIStateManager
        from ui.interfaces import UIComponentType, UIEventType
        
        # 1. 创建UI管理器
        print("\n1. 创建UI管理器:")
        ui_manager = UIManager(config={
            "max_components": 100,
            "enable_performance_monitoring": True
        })
        
        print(f"  - 管理器已创建")
        print(f"  - 注册组件数: {len(ui_manager.components)}")
        
        # 2. 测试状态管理器
        print("\n2. 测试状态管理器:")
        state_manager = UIStateManager()
        
        # 设置全局状态
        state_manager.set_global_state("current_user", "demo_user")
        state_manager.set_global_state("session_id", "demo_session_001")
        
        global_state = state_manager.get_global_state()
        print(f"  - 全局状态: {json.dumps(global_state, ensure_ascii=False, indent=4)}")
        
        # 3. 测试渲染器
        print("\n3. 测试渲染器:")
        renderer = UIRenderer(config={
            "default_format": "json",
            "enable_caching": True
        })
        
        supported_formats = renderer.get_supported_formats()
        print(f"  - 支持的格式: {', '.join(supported_formats)}")
        
        # 渲染页面
        page_config = {
            "title": "COT-DIR 数学推理系统",
            "description": "智能数学问题求解界面",
            "layouts": [
                {
                    "type": "container",
                    "props": {"padding": "20px", "background": "#f5f5f5"}
                }
            ]
        }
        
        rendered_page = renderer.render_page(page_config)
        print(f"  - 页面渲染: {rendered_page['title']}")
        print(f"  - 渲染时间: {rendered_page['metadata']['rendered_at']}")
        
        # 4. 测试事件处理器
        print("\n4. 测试事件处理器:")
        event_handler = UIEventHandler()
        
        supported_events = event_handler.get_supported_events()
        print(f"  - 支持的事件: {len(supported_events)} 种")
        
        # 验证事件数据
        valid_data = event_handler.validate_event_data(
            UIEventType.PROBLEM_SUBMIT,
            {"problem_text": "测试问题"}
        )
        print(f"  - 事件数据验证: {'✅ 通过' if valid_data else '❌ 失败'}")
        
        # 5. 测试性能统计
        print("\n5. 性能统计:")
        performance_stats = ui_manager.get_performance_stats()
        print(f"  - 已处理请求: {performance_stats['requests_processed']}")
        print(f"  - 发生错误: {performance_stats['errors_occurred']}")
        print(f"  - 平均响应时间: {performance_stats['average_response_time']:.2f}ms")
        
    except Exception as e:
        print(f"❌ 核心系统演示失败: {e}")
        raise


def demo_ui_components():
    """演示UI组件功能"""
    print("🧩 演示UI组件功能...")
    
    try:
        from ui.components import (
            BaseProblemInputComponent,
            BaseReasoningDisplayComponent,
            BaseResultDisplayComponent
        )
        from ui.interfaces import UIEvent, UIEventType
        
        # 1. 问题输入组件
        print("\n1. 问题输入组件:")
        problem_input = BaseProblemInputComponent(
            component_id="problem_input_demo",
            config={
                "min_length": 5,
                "max_length": 500,
                "required_patterns": []
            }
        )
        
        # 设置问题文本
        problem_input.set_problem_text("小明有15个苹果，给了小红5个，给了小李3个，还剩几个？")
        
        # 渲染组件
        rendered_input = problem_input.render(problem_input.get_state())
        print(f"  - 组件类型: {rendered_input['type']}")
        print(f"  - 组件标题: {rendered_input['title']}")
        print(f"  - 当前问题: {rendered_input['fields']['problem_text']['value'][:50]}...")
        
        # 验证输入
        validation_result = problem_input.validate_input({
            "problem_text": "小明有15个苹果，给了小红5个，给了小李3个，还剩几个？"
        })
        print(f"  - 输入验证: {'✅ 通过' if validation_result['valid'] else '❌ 失败'}")
        
        # 2. 推理显示组件
        print("\n2. 推理显示组件:")
        reasoning_display = BaseReasoningDisplayComponent(
            component_id="reasoning_display_demo",
            config={"max_steps": 10}
        )
        
        # 模拟推理步骤
        reasoning_steps = [
            {
                "step_index": 0,
                "step_type": "parse",
                "description": "解析问题：小明有15个苹果，给了小红5个，给了小李3个，还剩几个？",
                "confidence": 0.95,
                "data": {"numbers": [15, 5, 3], "operation": "subtraction"}
            },
            {
                "step_index": 1,
                "step_type": "calculate",
                "description": "计算：15 - 5 - 3 = 7",
                "confidence": 1.0,
                "data": {"calculation": "15 - 5 - 3 = 7"}
            },
            {
                "step_index": 2,
                "step_type": "verify",
                "description": "验证答案：7个苹果",
                "confidence": 0.98,
                "data": {"answer": 7, "unit": "个苹果"}
            }
        ]
        
        reasoning_display.display_reasoning_steps(reasoning_steps)
        
        # 渲染推理显示
        rendered_reasoning = reasoning_display.render(reasoning_display.get_state())
        print(f"  - 组件类型: {rendered_reasoning['type']}")
        print(f"  - 推理步骤数: {rendered_reasoning['content']['progress']['total']}")
        print(f"  - 当前进度: {rendered_reasoning['content']['progress']['percentage']:.1f}%")
        
        # 3. 结果显示组件
        print("\n3. 结果显示组件:")
        result_display = BaseResultDisplayComponent(
            component_id="result_display_demo",
            config={"show_confidence": True}
        )
        
        # 设置结果
        result_data = {
            "final_answer": "7个苹果",
            "confidence": 0.95,
            "reasoning_steps": reasoning_steps,
            "strategy_used": "chain_of_thought",
            "processing_time": 1250.5,
            "validation_checks": ["数值计算", "单位一致性", "逻辑合理性"]
        }
        
        result_display.display_result(result_data)
        result_display.display_confidence(0.95)
        result_display.display_explanation("通过逐步解析和计算，确定小明最终剩余7个苹果。")
        
        # 渲染结果显示
        rendered_result = result_display.render(result_display.get_state())
        print(f"  - 组件类型: {rendered_result['type']}")
        print(f"  - 最终答案: {rendered_result['content']['answer']['value']}")
        print(f"  - 置信度: {rendered_result['content']['answer']['confidence']}")
        print(f"  - 处理时间: {rendered_result['content']['answer']['processing_time']}")
        
        # 4. 组件状态管理
        print("\n4. 组件状态管理:")
        print(f"  - 问题输入组件状态: {problem_input.get_state().component_id}")
        print(f"  - 推理显示组件状态: {reasoning_display.get_state().component_id}")
        print(f"  - 结果显示组件状态: {result_display.get_state().component_id}")
        
        # 5. 事件处理
        print("\n5. 事件处理:")
        submit_event = UIEvent(
            event_id="demo_event_001",
            event_type=UIEventType.PROBLEM_SUBMIT,
            source_component="problem_input_demo",
            data={
                "problem_text": "小明有15个苹果，给了小红5个，给了小李3个，还剩几个？",
                "problem_type": "math_word_problem"
            },
            timestamp=datetime.now()
        )
        
        response = problem_input.handle_event(submit_event)
        if response:
            print(f"  - 事件处理结果: {response.response_type.value}")
            print(f"  - 响应消息: {response.message}")
        
    except Exception as e:
        print(f"❌ 组件演示失败: {e}")
        raise


def demo_ui_error_handling():
    """演示UI错误处理"""
    print("⚠️ 演示UI错误处理...")
    
    try:
        from ui.error_handling import (
            UIErrorHandler, UIErrorNotifier, UIErrorRecoveryManager,
            UIError, UIErrorType, UIErrorSeverity,
            handle_ui_error, recover_from_ui_error, get_ui_error_statistics
        )
        
        # 1. 创建错误处理器
        print("\n1. 创建错误处理器:")
        error_handler = UIErrorHandler()
        
        # 2. 处理各种类型的错误
        print("\n2. 处理各种类型的错误:")
        
        # 验证错误
        try:
            raise ValueError("输入数据格式无效")
        except ValueError as e:
            ui_error = handle_ui_error(e, "problem_input_demo", "demo_request_001")
            print(f"  - 验证错误: {ui_error.error_type.value}")
            print(f"  - 错误严重程度: {ui_error.severity.value}")
        
        # 组件错误
        try:
            raise KeyError("组件状态键不存在")
        except KeyError as e:
            ui_error = handle_ui_error(e, "reasoning_display_demo", "demo_request_002")
            print(f"  - 组件错误: {ui_error.error_type.value}")
        
        # 超时错误
        try:
            raise TimeoutError("请求处理超时")
        except TimeoutError as e:
            ui_error = handle_ui_error(e, "result_display_demo", "demo_request_003")
            print(f"  - 超时错误: {ui_error.error_type.value}")
        
        # 3. 错误统计
        print("\n3. 错误统计:")
        error_stats = get_ui_error_statistics()
        print(f"  - 总错误数: {error_stats['total_errors']}")
        print(f"  - 错误类型分布: {json.dumps(error_stats['error_types'], ensure_ascii=False)}")
        print(f"  - 严重程度分布: {json.dumps(error_stats['severity_distribution'], ensure_ascii=False)}")
        
        # 4. 错误恢复
        print("\n4. 错误恢复:")
        recovery_manager = UIErrorRecoveryManager()
        
        # 创建测试错误
        test_error = UIError(
            error_id="demo_error_001",
            error_type=UIErrorType.COMPONENT_ERROR,
            severity=UIErrorSeverity.MEDIUM,
            message="组件状态异常",
            component_id="demo_component"
        )
        
        # 尝试恢复
        recovery_success = recover_from_ui_error(test_error, "component_reset")
        print(f"  - 恢复尝试: {'✅ 成功' if recovery_success else '❌ 失败'}")
        
        # 获取恢复历史
        recovery_history = recovery_manager.get_recovery_history()
        print(f"  - 恢复历史记录: {len(recovery_history)} 条")
        
        # 5. 错误通知
        print("\n5. 错误通知:")
        notifier = UIErrorNotifier()
        
        # 注册通知处理器
        notifications_received = []
        
        def demo_notification_handler(error):
            notifications_received.append(error)
        
        notifier.register_notification_handler(UIErrorSeverity.HIGH, demo_notification_handler)
        
        # 创建高严重程度错误
        high_severity_error = UIError(
            error_id="demo_error_002",
            error_type=UIErrorType.SYSTEM_ERROR,
            severity=UIErrorSeverity.HIGH,
            message="系统关键错误"
        )
        
        notifier.notify_error(high_severity_error)
        print(f"  - 通知已发送: {len(notifications_received)} 条")
        
        # 6. 错误过滤
        print("\n6. 错误过滤:")
        
        def demo_error_filter(error):
            # 过滤掉低严重程度的错误
            return error.severity != UIErrorSeverity.LOW
        
        notifier.add_error_filter(demo_error_filter)
        
        low_severity_error = UIError(
            error_id="demo_error_003",
            error_type=UIErrorType.VALIDATION_ERROR,
            severity=UIErrorSeverity.LOW,
            message="低严重程度错误"
        )
        
        notifier.notify_error(low_severity_error)
        print(f"  - 过滤后通知: {len(notifications_received)} 条（应该还是1条）")
        
    except Exception as e:
        print(f"❌ 错误处理演示失败: {e}")
        raise


def demo_complete_workflow():
    """演示完整工作流程"""
    print("🔄 演示完整工作流程...")
    
    try:
        from ui.core import UIManager
        from ui.components import (
            BaseProblemInputComponent,
            BaseReasoningDisplayComponent,
            BaseResultDisplayComponent
        )
        from ui.interfaces import UIRequest, UIEvent, UIEventType
        
        # 1. 初始化UI系统
        print("\n1. 初始化UI系统:")
        ui_manager = UIManager()
        
        # 创建组件
        problem_input = BaseProblemInputComponent("problem_input", {})
        reasoning_display = BaseReasoningDisplayComponent("reasoning_display", {})
        result_display = BaseResultDisplayComponent("result_display", {})
        
        # 注册组件
        ui_manager.register_component(problem_input)
        ui_manager.register_component(reasoning_display)
        ui_manager.register_component(result_display)
        
        print(f"  - 已注册组件: {len(ui_manager.components)}")
        
        # 2. 模拟用户输入问题
        print("\n2. 模拟用户输入问题:")
        problem_text = "学校买了48支铅笔，平均分给6个班级，每个班级分到几支？"
        
        submit_request = UIRequest(
            request_id="workflow_request_001",
            component_id="problem_input",
            action="submit_problem",
            data={
                "problem_text": problem_text,
                "problem_type": "math_word_problem",
                "difficulty": "medium"
            },
            timestamp=datetime.now()
        )
        
        response = ui_manager.process_request(submit_request)
        print(f"  - 问题提交: {response.response_type.value}")
        print(f"  - 响应消息: {response.message}")
        
        # 3. 模拟推理过程
        print("\n3. 模拟推理过程:")
        
        # 推理开始事件
        reasoning_start_event = UIEvent(
            event_id="workflow_event_001",
            event_type=UIEventType.REASONING_START,
            source_component="reasoning_display",
            data={"strategy": "chain_of_thought"},
            timestamp=datetime.now()
        )
        
        ui_manager.handle_event(reasoning_start_event)
        print("  - 推理开始")
        
        # 模拟推理步骤
        reasoning_steps = [
            {
                "step_index": 0,
                "step_type": "parse",
                "description": "解析问题：学校买了48支铅笔，平均分给6个班级",
                "confidence": 0.95,
                "data": {"total": 48, "groups": 6, "operation": "division"}
            },
            {
                "step_index": 1,
                "step_type": "calculate",
                "description": "计算：48 ÷ 6 = 8",
                "confidence": 1.0,
                "data": {"calculation": "48 ÷ 6 = 8"}
            },
            {
                "step_index": 2,
                "step_type": "verify",
                "description": "验证：8 × 6 = 48 ✓",
                "confidence": 1.0,
                "data": {"verification": "8 × 6 = 48"}
            }
        ]
        
        # 发送推理步骤事件
        for step in reasoning_steps:
            step_event = UIEvent(
                event_id=f"workflow_event_step_{step['step_index']}",
                event_type=UIEventType.REASONING_STEP,
                source_component="reasoning_display",
                data={"step_data": step},
                timestamp=datetime.now()
            )
            
            ui_manager.handle_event(step_event)
            print(f"  - 推理步骤 {step['step_index'] + 1}: {step['step_type']}")
            time.sleep(0.1)  # 模拟处理时间
        
        # 推理完成事件
        reasoning_complete_event = UIEvent(
            event_id="workflow_event_complete",
            event_type=UIEventType.REASONING_COMPLETE,
            source_component="reasoning_display",
            data={
                "total_steps": len(reasoning_steps),
                "processing_time": 1500.0
            },
            timestamp=datetime.now()
        )
        
        ui_manager.handle_event(reasoning_complete_event)
        print("  - 推理完成")
        
        # 4. 显示结果
        print("\n4. 显示结果:")
        
        result_data = {
            "final_answer": "8支铅笔",
            "confidence": 0.98,
            "reasoning_steps": reasoning_steps,
            "strategy_used": "chain_of_thought",
            "processing_time": 1500.0,
            "validation_checks": ["数值计算", "单位一致性", "逻辑合理性"]
        }
        
        result_event = UIEvent(
            event_id="workflow_event_result",
            event_type=UIEventType.RESULT_DISPLAY,
            source_component="result_display",
            data={"result": result_data},
            timestamp=datetime.now()
        )
        
        ui_manager.handle_event(result_event)
        print(f"  - 结果显示: {result_data['final_answer']}")
        print(f"  - 置信度: {result_data['confidence']}")
        
        # 5. 获取系统状态
        print("\n5. 获取系统状态:")
        component_states = ui_manager.get_component_states()
        print(f"  - 组件状态数: {len(component_states)}")
        
        for component_id, state in component_states.items():
            print(f"    * {component_id}: {state.component_type.value}")
        
        # 6. 性能统计
        print("\n6. 性能统计:")
        performance_stats = ui_manager.get_performance_stats()
        print(f"  - 处理的请求数: {performance_stats['requests_processed']}")
        print(f"  - 平均响应时间: {performance_stats['average_response_time']:.2f}ms")
        print(f"  - 错误数: {performance_stats['errors_occurred']}")
        
        print("\n✅ 完整工作流程演示成功！")
        
    except Exception as e:
        print(f"❌ 完整工作流程演示失败: {e}")
        raise


def demo_performance_monitoring():
    """演示性能监控"""
    print("📊 演示性能监控...")
    
    try:
        from ui.core import UIManager
        from ui.components import BaseProblemInputComponent
        from ui.interfaces import UIRequest
        import threading
        
        # 1. 创建UI管理器
        ui_manager = UIManager()
        
        # 注册组件
        problem_input = BaseProblemInputComponent("perf_test_component", {})
        ui_manager.register_component(problem_input)
        
        print("\n1. 并发请求测试:")
        
        # 2. 并发请求测试
        request_count = 20
        results = []
        
        def make_request(request_id):
            request = UIRequest(
                request_id=f"perf_request_{request_id}",
                component_id="perf_test_component",
                action="get_state",
                data={},
                timestamp=datetime.now()
            )
            
            start_time = time.time()
            response = ui_manager.process_request(request)
            end_time = time.time()
            
            results.append({
                "request_id": request_id,
                "response_time": (end_time - start_time) * 1000,
                "success": response.response_type.value == "success"
            })
        
        # 创建并启动线程
        threads = []
        start_time = time.time()
        
        for i in range(request_count):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # 分析结果
        total_time = (end_time - start_time) * 1000
        successful_requests = sum(1 for r in results if r["success"])
        response_times = [r["response_time"] for r in results]
        
        print(f"  - 总请求数: {request_count}")
        print(f"  - 成功请求数: {successful_requests}")
        print(f"  - 总耗时: {total_time:.2f}ms")
        print(f"  - 平均响应时间: {sum(response_times) / len(response_times):.2f}ms")
        print(f"  - 最快响应时间: {min(response_times):.2f}ms")
        print(f"  - 最慢响应时间: {max(response_times):.2f}ms")
        print(f"  - 请求吞吐量: {request_count / (total_time / 1000):.2f} req/s")
        
        # 3. 内存使用监控
        print("\n3. 内存使用监控:")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        print(f"  - 内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
        print(f"  - 虚拟内存: {memory_info.vms / 1024 / 1024:.2f} MB")
        
        # 4. 系统性能统计
        print("\n4. 系统性能统计:")
        performance_stats = ui_manager.get_performance_stats()
        
        print(f"  - 总处理请求数: {performance_stats['requests_processed']}")
        print(f"  - 总错误数: {performance_stats['errors_occurred']}")
        print(f"  - 平均响应时间: {performance_stats['average_response_time']:.2f}ms")
        print(f"  - 成功率: {(performance_stats['requests_processed'] - performance_stats['errors_occurred']) / performance_stats['requests_processed'] * 100:.1f}%")
        
        # 5. 组件状态监控
        print("\n5. 组件状态监控:")
        component_states = ui_manager.get_component_states()
        
        for component_id, state in component_states.items():
            print(f"  - {component_id}:")
            print(f"    * 类型: {state.component_type.value}")
            print(f"    * 可见: {state.visible}")
            print(f"    * 启用: {state.enabled}")
            print(f"    * 状态键数: {len(state.state)}")
        
        print("\n✅ 性能监控演示完成！")
        
    except Exception as e:
        print(f"❌ 性能监控演示失败: {e}")
        raise


if __name__ == "__main__":
    main()