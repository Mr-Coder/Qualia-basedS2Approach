"""
UI模块核心处理逻辑
实现UI管理器、渲染器和事件处理器的核心功能
"""

import logging
import asyncio
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor
import json
import uuid

from .interfaces import (
    IUIManager, IUIRenderer, IUIEventHandler, IUIComponent, IUIStateManager,
    UIRequest, UIResponse, UIEvent, UIComponentState, UIEventType, UIResponseType,
    UIComponentType, UIUtils, UISchemas
)


class UIStateManager(IUIStateManager):
    """UI状态管理器实现"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._states: Dict[str, UIComponentState] = {}
        self._global_state: Dict[str, Any] = {}
        self._subscriptions: Dict[str, Callable] = {}
        self._lock = threading.RLock()
    
    def get_state(self, component_id: str) -> Optional[UIComponentState]:
        """获取组件状态"""
        with self._lock:
            return self._states.get(component_id)
    
    def set_state(self, component_id: str, state: UIComponentState) -> bool:
        """设置组件状态"""
        try:
            with self._lock:
                old_state = self._states.get(component_id)
                self._states[component_id] = state
                
                # 通知订阅者
                self._notify_state_change(component_id, old_state, state)
                
                return True
        except Exception as e:
            self.logger.error(f"Failed to set state for {component_id}: {e}")
            return False
    
    def subscribe_to_state_changes(self, component_id: str, callback: Callable) -> str:
        """订阅状态变化"""
        subscription_id = str(uuid.uuid4())
        self._subscriptions[subscription_id] = {
            "component_id": component_id,
            "callback": callback
        }
        return subscription_id
    
    def unsubscribe_from_state_changes(self, subscription_id: str) -> bool:
        """取消订阅状态变化"""
        return self._subscriptions.pop(subscription_id, None) is not None
    
    def get_global_state(self) -> Dict[str, Any]:
        """获取全局状态"""
        with self._lock:
            return self._global_state.copy()
    
    def set_global_state(self, key: str, value: Any) -> bool:
        """设置全局状态"""
        try:
            with self._lock:
                self._global_state[key] = value
                return True
        except Exception as e:
            self.logger.error(f"Failed to set global state {key}: {e}")
            return False
    
    def _notify_state_change(self, component_id: str, old_state: Optional[UIComponentState], new_state: UIComponentState):
        """通知状态变化"""
        for subscription in self._subscriptions.values():
            if subscription["component_id"] == component_id:
                try:
                    subscription["callback"](component_id, old_state, new_state)
                except Exception as e:
                    self.logger.error(f"Error in state change callback: {e}")


class UIRenderer(IUIRenderer):
    """UI渲染器实现"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.supported_formats = ["json", "html", "react", "vue"]
        self.templates = {}
        self.cache = {}
    
    def render_component(self, component: IUIComponent, state: UIComponentState) -> Dict[str, Any]:
        """渲染单个组件"""
        try:
            # 获取组件渲染结果
            component_data = component.render(state)
            
            # 添加元数据
            rendered_component = {
                "component_id": component.get_component_id(),
                "component_type": component.get_component_type().value,
                "visible": state.visible,
                "enabled": state.enabled,
                "error_state": state.error_state,
                "content": component_data,
                "timestamp": datetime.now().isoformat()
            }
            
            return rendered_component
            
        except Exception as e:
            self.logger.error(f"Failed to render component {component.get_component_id()}: {e}")
            return {
                "component_id": component.get_component_id(),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def render_layout(self, components: List[IUIComponent], layout_config: Dict[str, Any]) -> Dict[str, Any]:
        """渲染布局"""
        try:
            layout_type = layout_config.get("type", "vertical")
            layout_props = layout_config.get("props", {})
            
            rendered_components = []
            
            for component in components:
                # 获取组件状态（这里简化处理）
                state = UIComponentState(
                    component_id=component.get_component_id(),
                    component_type=component.get_component_type(),
                    state={}
                )
                
                rendered_component = self.render_component(component, state)
                rendered_components.append(rendered_component)
            
            layout = {
                "type": layout_type,
                "props": layout_props,
                "components": rendered_components,
                "timestamp": datetime.now().isoformat()
            }
            
            return layout
            
        except Exception as e:
            self.logger.error(f"Failed to render layout: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def render_page(self, page_config: Dict[str, Any]) -> Dict[str, Any]:
        """渲染页面"""
        try:
            page_title = page_config.get("title", "COT-DIR System")
            page_description = page_config.get("description", "Mathematical Reasoning System")
            layouts = page_config.get("layouts", [])
            
            rendered_layouts = []
            for layout_config in layouts:
                # 这里需要实际的组件实例，简化处理
                rendered_layout = {
                    "type": layout_config.get("type", "container"),
                    "props": layout_config.get("props", {}),
                    "components": []
                }
                rendered_layouts.append(rendered_layout)
            
            page = {
                "title": page_title,
                "description": page_description,
                "layouts": rendered_layouts,
                "metadata": {
                    "rendered_at": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            }
            
            return page
            
        except Exception as e:
            self.logger.error(f"Failed to render page: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的渲染格式"""
        return self.supported_formats.copy()


class UIEventHandler(IUIEventHandler):
    """UI事件处理器实现"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_events = [
            UIEventType.PROBLEM_SUBMIT,
            UIEventType.PROBLEM_CLEAR,
            UIEventType.REASONING_START,
            UIEventType.REASONING_STEP,
            UIEventType.REASONING_COMPLETE,
            UIEventType.RESULT_DISPLAY,
            UIEventType.ERROR_OCCURRED,
            UIEventType.CONFIG_CHANGE,
            UIEventType.STATISTICS_UPDATE
        ]
        
        # 事件处理器映射
        self.event_handlers = {
            UIEventType.PROBLEM_SUBMIT: self._handle_problem_submit,
            UIEventType.PROBLEM_CLEAR: self._handle_problem_clear,
            UIEventType.REASONING_START: self._handle_reasoning_start,
            UIEventType.REASONING_STEP: self._handle_reasoning_step,
            UIEventType.REASONING_COMPLETE: self._handle_reasoning_complete,
            UIEventType.RESULT_DISPLAY: self._handle_result_display,
            UIEventType.ERROR_OCCURRED: self._handle_error_occurred,
            UIEventType.CONFIG_CHANGE: self._handle_config_change,
            UIEventType.STATISTICS_UPDATE: self._handle_statistics_update
        }
    
    def handle_event(self, event: UIEvent) -> UIResponse:
        """处理UI事件"""
        try:
            # 验证事件数据
            if not self.validate_event_data(event.event_type, event.data):
                return UIResponse(
                    request_id=event.event_id,
                    response_type=UIResponseType.ERROR,
                    data={},
                    message="Invalid event data"
                )
            
            # 获取对应的处理器
            handler = self.event_handlers.get(event.event_type)
            if not handler:
                return UIResponse(
                    request_id=event.event_id,
                    response_type=UIResponseType.ERROR,
                    data={},
                    message=f"Unsupported event type: {event.event_type}"
                )
            
            # 执行处理器
            response_data = handler(event)
            
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.SUCCESS,
                data=response_data,
                message="Event processed successfully"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling event {event.event_type}: {e}")
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message=str(e),
                error_details={"exception": str(e)}
            )
    
    def get_supported_events(self) -> List[UIEventType]:
        """获取支持的事件类型"""
        return self.supported_events.copy()
    
    def validate_event_data(self, event_type: UIEventType, data: Dict[str, Any]) -> bool:
        """验证事件数据"""
        try:
            # 根据事件类型验证数据
            if event_type == UIEventType.PROBLEM_SUBMIT:
                return "problem_text" in data and isinstance(data["problem_text"], str)
            elif event_type == UIEventType.REASONING_STEP:
                return "step_data" in data and isinstance(data["step_data"], dict)
            elif event_type == UIEventType.RESULT_DISPLAY:
                return "result" in data and isinstance(data["result"], dict)
            elif event_type == UIEventType.ERROR_OCCURRED:
                return "error_message" in data and isinstance(data["error_message"], str)
            elif event_type == UIEventType.CONFIG_CHANGE:
                return "config" in data and isinstance(data["config"], dict)
            else:
                return True  # 其他事件类型暂时不验证
                
        except Exception as e:
            self.logger.error(f"Error validating event data: {e}")
            return False
    
    def _handle_problem_submit(self, event: UIEvent) -> Dict[str, Any]:
        """处理问题提交事件"""
        problem_text = event.data.get("problem_text", "")
        problem_type = event.data.get("problem_type", "math_word_problem")
        
        # 清理输入
        cleaned_text = UIUtils.sanitize_input(problem_text)
        
        return {
            "action": "problem_submitted",
            "problem_text": cleaned_text,
            "problem_type": problem_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_problem_clear(self, event: UIEvent) -> Dict[str, Any]:
        """处理问题清空事件"""
        return {
            "action": "problem_cleared",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_reasoning_start(self, event: UIEvent) -> Dict[str, Any]:
        """处理推理开始事件"""
        return {
            "action": "reasoning_started",
            "strategy": event.data.get("strategy", "chain_of_thought"),
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_reasoning_step(self, event: UIEvent) -> Dict[str, Any]:
        """处理推理步骤事件"""
        step_data = event.data.get("step_data", {})
        
        return {
            "action": "reasoning_step_processed",
            "step_index": step_data.get("step_index", 0),
            "step_type": step_data.get("step_type", "unknown"),
            "confidence": step_data.get("confidence", 0.0),
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_reasoning_complete(self, event: UIEvent) -> Dict[str, Any]:
        """处理推理完成事件"""
        return {
            "action": "reasoning_completed",
            "total_steps": event.data.get("total_steps", 0),
            "processing_time": event.data.get("processing_time", 0.0),
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_result_display(self, event: UIEvent) -> Dict[str, Any]:
        """处理结果显示事件"""
        result = event.data.get("result", {})
        
        return {
            "action": "result_displayed",
            "final_answer": result.get("final_answer", ""),
            "confidence": result.get("confidence", 0.0),
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_error_occurred(self, event: UIEvent) -> Dict[str, Any]:
        """处理错误事件"""
        error_message = event.data.get("error_message", "Unknown error")
        error_type = event.data.get("error_type", "general")
        
        return {
            "action": "error_handled",
            "error_message": error_message,
            "error_type": error_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_config_change(self, event: UIEvent) -> Dict[str, Any]:
        """处理配置变化事件"""
        config = event.data.get("config", {})
        
        return {
            "action": "config_changed",
            "changed_keys": list(config.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_statistics_update(self, event: UIEvent) -> Dict[str, Any]:
        """处理统计更新事件"""
        stats = event.data.get("statistics", {})
        
        return {
            "action": "statistics_updated",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }


class UIManager(IUIManager):
    """UI管理器实现"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 初始化子系统
        self.state_manager = UIStateManager()
        self.renderer = UIRenderer(self.config.get("renderer", {}))
        self.event_handler = UIEventHandler()
        
        # 组件注册表
        self.components: Dict[str, IUIComponent] = {}
        
        # 请求处理
        self.request_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 性能监控
        self.performance_stats = {
            "requests_processed": 0,
            "errors_occurred": 0,
            "average_response_time": 0.0
        }
    
    def register_component(self, component: IUIComponent) -> bool:
        """注册UI组件"""
        try:
            component_id = component.get_component_id()
            
            if component_id in self.components:
                self.logger.warning(f"Component {component_id} already registered, overwriting")
            
            self.components[component_id] = component
            
            # 初始化组件状态
            initial_state = component.get_state()
            self.state_manager.set_state(component_id, initial_state)
            
            self.logger.info(f"Registered component: {component_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register component: {e}")
            return False
    
    def unregister_component(self, component_id: str) -> bool:
        """注销UI组件"""
        try:
            if component_id in self.components:
                del self.components[component_id]
                self.logger.info(f"Unregistered component: {component_id}")
                return True
            else:
                self.logger.warning(f"Component {component_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to unregister component {component_id}: {e}")
            return False
    
    def get_component(self, component_id: str) -> Optional[IUIComponent]:
        """获取UI组件"""
        return self.components.get(component_id)
    
    def process_request(self, request: UIRequest) -> UIResponse:
        """处理UI请求"""
        start_time = datetime.now()
        
        try:
            # 验证请求
            validation_result = self._validate_request(request)
            if not validation_result["valid"]:
                return UIResponse(
                    request_id=request.request_id,
                    response_type=UIResponseType.ERROR,
                    data={},
                    message="Invalid request",
                    error_details={"validation_errors": validation_result["errors"]}
                )
            
            # 获取目标组件
            component = self.get_component(request.component_id)
            if not component:
                return UIResponse(
                    request_id=request.request_id,
                    response_type=UIResponseType.ERROR,
                    data={},
                    message=f"Component {request.component_id} not found"
                )
            
            # 处理请求
            response_data = self._process_component_request(component, request)
            
            # 更新性能统计
            self._update_performance_stats(start_time)
            
            return UIResponse(
                request_id=request.request_id,
                response_type=UIResponseType.SUCCESS,
                data=response_data,
                message="Request processed successfully"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing request {request.request_id}: {e}")
            self.performance_stats["errors_occurred"] += 1
            
            return UIResponse(
                request_id=request.request_id,
                response_type=UIResponseType.ERROR,
                data={},
                message=str(e),
                error_details={"exception": str(e)}
            )
    
    def handle_event(self, event: UIEvent) -> None:
        """处理UI事件"""
        try:
            # 通过事件处理器处理事件
            response = self.event_handler.handle_event(event)
            
            # 根据事件类型更新相关组件状态
            self._update_components_from_event(event, response)
            
        except Exception as e:
            self.logger.error(f"Error handling event {event.event_type}: {e}")
    
    def get_component_states(self) -> Dict[str, UIComponentState]:
        """获取所有组件状态"""
        states = {}
        for component_id in self.components:
            state = self.state_manager.get_state(component_id)
            if state:
                states[component_id] = state
        return states
    
    def update_component_state(self, component_id: str, state: Dict[str, Any]) -> bool:
        """更新组件状态"""
        try:
            component = self.get_component(component_id)
            if not component:
                self.logger.error(f"Component {component_id} not found")
                return False
            
            # 更新组件内部状态
            component.set_state(state)
            
            # 更新状态管理器中的状态
            current_state = self.state_manager.get_state(component_id)
            if current_state:
                current_state.state.update(state)
                self.state_manager.set_state(component_id, current_state)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update component state {component_id}: {e}")
            return False
    
    def _validate_request(self, request: UIRequest) -> Dict[str, Any]:
        """验证请求"""
        return UIUtils.validate_json_schema(request.to_dict(), UISchemas.UI_REQUEST_SCHEMA)
    
    def _process_component_request(self, component: IUIComponent, request: UIRequest) -> Dict[str, Any]:
        """处理组件请求"""
        # 根据请求动作执行相应操作
        if request.action == "get_state":
            return component.get_state().to_dict()
        elif request.action == "set_state":
            success = component.set_state(request.data)
            return {"success": success}
        elif request.action == "render":
            state = component.get_state()
            return self.renderer.render_component(component, state)
        elif request.action == "validate_input":
            return component.validate_input(request.data)
        else:
            # 创建事件并让组件处理
            event = UIEvent(
                event_id=UIUtils.generate_event_id(),
                event_type=UIEventType.PROBLEM_SUBMIT,  # 默认事件类型
                source_component=request.component_id,
                data=request.data,
                timestamp=datetime.now()
            )
            
            response = component.handle_event(event)
            return response.to_dict() if response else {}
    
    def _update_components_from_event(self, event: UIEvent, response: UIResponse) -> None:
        """根据事件更新组件状态"""
        try:
            # 根据事件类型更新相关组件
            if event.event_type == UIEventType.PROBLEM_SUBMIT:
                # 更新问题输入组件状态
                self._update_problem_input_state(event, response)
            elif event.event_type == UIEventType.REASONING_STEP:
                # 更新推理显示组件状态
                self._update_reasoning_display_state(event, response)
            elif event.event_type == UIEventType.RESULT_DISPLAY:
                # 更新结果显示组件状态
                self._update_result_display_state(event, response)
                
        except Exception as e:
            self.logger.error(f"Error updating components from event: {e}")
    
    def _update_problem_input_state(self, event: UIEvent, response: UIResponse) -> None:
        """更新问题输入组件状态"""
        # 查找问题输入组件
        for component_id, component in self.components.items():
            if component.get_component_type() == UIComponentType.INPUT:
                state = self.state_manager.get_state(component_id)
                if state:
                    state.state["last_submitted"] = event.data.get("problem_text", "")
                    state.state["submit_time"] = event.timestamp.isoformat()
                    self.state_manager.set_state(component_id, state)
    
    def _update_reasoning_display_state(self, event: UIEvent, response: UIResponse) -> None:
        """更新推理显示组件状态"""
        # 查找推理显示组件
        for component_id, component in self.components.items():
            if component.get_component_type() == UIComponentType.DISPLAY:
                state = self.state_manager.get_state(component_id)
                if state:
                    if "reasoning_steps" not in state.state:
                        state.state["reasoning_steps"] = []
                    state.state["reasoning_steps"].append(event.data.get("step_data", {}))
                    self.state_manager.set_state(component_id, state)
    
    def _update_result_display_state(self, event: UIEvent, response: UIResponse) -> None:
        """更新结果显示组件状态"""
        # 查找结果显示组件
        for component_id, component in self.components.items():
            if component.get_component_type() == UIComponentType.OUTPUT:
                state = self.state_manager.get_state(component_id)
                if state:
                    state.state["result"] = event.data.get("result", {})
                    state.state["display_time"] = event.timestamp.isoformat()
                    self.state_manager.set_state(component_id, state)
    
    def _update_performance_stats(self, start_time: datetime) -> None:
        """更新性能统计"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        self.performance_stats["requests_processed"] += 1
        
        # 更新平均响应时间
        total_requests = self.performance_stats["requests_processed"]
        current_avg = self.performance_stats["average_response_time"]
        self.performance_stats["average_response_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def shutdown(self) -> None:
        """关闭UI管理器"""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("UI Manager shut down successfully")
        except Exception as e:
            self.logger.error(f"Error shutting down UI Manager: {e}")