"""
UI组件实现
提供具体的UI组件实现，包括问题输入、推理显示、结果显示等组件
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import re

from .interfaces import (
    IUIComponent, IProblemInputComponent, IReasoningDisplayComponent, IResultDisplayComponent,
    UIEvent, UIResponse, UIComponentState, UIEventType, UIResponseType, UIComponentType,
    UIUtils, UISchemas
)
from .error_handling import handle_ui_error, UIErrorType, UIErrorSeverity


class BaseProblemInputComponent(IProblemInputComponent):
    """基础问题输入组件"""
    
    def __init__(self, component_id: str, config: Optional[Dict[str, Any]] = None):
        self.component_id = component_id
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 组件状态
        self.state = UIComponentState(
            component_id=component_id,
            component_type=UIComponentType.INPUT,
            state={
                "problem_text": "",
                "problem_type": "math_word_problem",
                "difficulty": "medium",
                "validation_errors": [],
                "last_submitted": None,
                "submit_count": 0
            },
            visible=True,
            enabled=True
        )
        
        # 验证规则
        self.validation_rules = {
            "min_length": self.config.get("min_length", 10),
            "max_length": self.config.get("max_length", 1000),
            "required_patterns": self.config.get("required_patterns", []),
            "forbidden_patterns": self.config.get("forbidden_patterns", [])
        }
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return self.component_id
    
    def get_component_type(self) -> UIComponentType:
        """获取组件类型"""
        return UIComponentType.INPUT
    
    def render(self, state: UIComponentState) -> Dict[str, Any]:
        """渲染组件"""
        try:
            render_data = {
                "type": "problem_input",
                "title": "数学问题输入",
                "description": "请输入您想要解决的数学问题",
                "fields": {
                    "problem_text": {
                        "type": "textarea",
                        "label": "问题描述",
                        "value": state.state.get("problem_text", ""),
                        "placeholder": "请输入数学问题，例如：小明有10个苹果，吃了3个，还剩几个？",
                        "required": True,
                        "validation": {
                            "min_length": self.validation_rules["min_length"],
                            "max_length": self.validation_rules["max_length"]
                        }
                    },
                    "problem_type": {
                        "type": "select",
                        "label": "问题类型",
                        "value": state.state.get("problem_type", "math_word_problem"),
                        "options": [
                            {"value": "math_word_problem", "label": "数学应用题"},
                            {"value": "algebra", "label": "代数问题"},
                            {"value": "geometry", "label": "几何问题"},
                            {"value": "arithmetic", "label": "算术问题"}
                        ]
                    },
                    "difficulty": {
                        "type": "select",
                        "label": "难度等级",
                        "value": state.state.get("difficulty", "medium"),
                        "options": [
                            {"value": "easy", "label": "简单"},
                            {"value": "medium", "label": "中等"},
                            {"value": "hard", "label": "困难"}
                        ]
                    }
                },
                "actions": {
                    "submit": {
                        "type": "button",
                        "label": "提交问题",
                        "primary": True,
                        "enabled": state.enabled and len(state.state.get("problem_text", "")) > 0
                    },
                    "clear": {
                        "type": "button",
                        "label": "清空",
                        "secondary": True,
                        "enabled": state.enabled
                    }
                },
                "validation_errors": state.state.get("validation_errors", []),
                "statistics": {
                    "submit_count": state.state.get("submit_count", 0),
                    "last_submitted": state.state.get("last_submitted")
                }
            }
            
            return render_data
            
        except Exception as e:
            error = handle_ui_error(e, self.component_id, context={"action": "render"})
            return {
                "type": "error",
                "message": "渲染失败",
                "error_details": error.to_dict()
            }
    
    def handle_event(self, event: UIEvent) -> Optional[UIResponse]:
        """处理事件"""
        try:
            if event.event_type == UIEventType.PROBLEM_SUBMIT:
                return self._handle_problem_submit(event)
            elif event.event_type == UIEventType.PROBLEM_CLEAR:
                return self._handle_problem_clear(event)
            else:
                return None
                
        except Exception as e:
            error = handle_ui_error(e, self.component_id, context={"event": event.event_type})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="事件处理失败",
                error_details=error.to_dict()
            )
    
    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证输入数据"""
        try:
            # 使用预定义的模式验证
            validation_result = UIUtils.validate_json_schema(data, UISchemas.PROBLEM_INPUT_SCHEMA)
            
            if not validation_result["valid"]:
                return validation_result
            
            # 自定义验证规则
            errors = []
            problem_text = data.get("problem_text", "")
            
            # 长度验证
            if len(problem_text) < self.validation_rules["min_length"]:
                errors.append(f"问题描述至少需要{self.validation_rules['min_length']}个字符")
            
            if len(problem_text) > self.validation_rules["max_length"]:
                errors.append(f"问题描述不能超过{self.validation_rules['max_length']}个字符")
            
            # 必需模式验证
            for pattern in self.validation_rules["required_patterns"]:
                if not re.search(pattern, problem_text):
                    errors.append(f"问题描述必须包含模式: {pattern}")
            
            # 禁止模式验证
            for pattern in self.validation_rules["forbidden_patterns"]:
                if re.search(pattern, problem_text):
                    errors.append(f"问题描述不能包含模式: {pattern}")
            
            # 数学问题特定验证
            if not self._validate_math_problem(problem_text):
                errors.append("这似乎不是一个有效的数学问题")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
            
        except Exception as e:
            error = handle_ui_error(e, self.component_id, context={"action": "validate_input"})
            return {
                "valid": False,
                "errors": [f"验证过程出错: {str(e)}"]
            }
    
    def get_state(self) -> UIComponentState:
        """获取组件状态"""
        return self.state
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """设置组件状态"""
        try:
            self.state.state.update(state)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set state: {e}")
            return False
    
    def get_problem_text(self) -> str:
        """获取问题文本"""
        return self.state.state.get("problem_text", "")
    
    def set_problem_text(self, text: str) -> bool:
        """设置问题文本"""
        try:
            cleaned_text = UIUtils.sanitize_input(text)
            self.state.state["problem_text"] = cleaned_text
            return True
        except Exception as e:
            self.logger.error(f"Failed to set problem text: {e}")
            return False
    
    def clear_input(self) -> bool:
        """清空输入"""
        try:
            self.state.state.update({
                "problem_text": "",
                "validation_errors": [],
                "last_submitted": None
            })
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear input: {e}")
            return False
    
    def validate_problem_format(self, text: str) -> Dict[str, Any]:
        """验证问题格式"""
        return self.validate_input({"problem_text": text})
    
    def _handle_problem_submit(self, event: UIEvent) -> UIResponse:
        """处理问题提交事件"""
        try:
            problem_data = event.data
            
            # 验证输入
            validation_result = self.validate_input(problem_data)
            
            if not validation_result["valid"]:
                # 更新状态显示验证错误
                self.state.state["validation_errors"] = validation_result["errors"]
                
                return UIResponse(
                    request_id=event.event_id,
                    response_type=UIResponseType.ERROR,
                    data={"validation_errors": validation_result["errors"]},
                    message="输入验证失败"
                )
            
            # 清空验证错误
            self.state.state["validation_errors"] = []
            
            # 更新状态
            self.set_problem_text(problem_data.get("problem_text", ""))
            self.state.state["problem_type"] = problem_data.get("problem_type", "math_word_problem")
            self.state.state["difficulty"] = problem_data.get("difficulty", "medium")
            self.state.state["last_submitted"] = datetime.now().isoformat()
            self.state.state["submit_count"] += 1
            
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.SUCCESS,
                data={
                    "problem_text": self.get_problem_text(),
                    "problem_type": self.state.state["problem_type"],
                    "difficulty": self.state.state["difficulty"],
                    "submit_count": self.state.state["submit_count"]
                },
                message="问题提交成功"
            )
            
        except Exception as e:
            error = handle_ui_error(e, self.component_id, event.event_id, {"action": "submit"})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="提交失败",
                error_details=error.to_dict()
            )
    
    def _handle_problem_clear(self, event: UIEvent) -> UIResponse:
        """处理问题清空事件"""
        try:
            success = self.clear_input()
            
            if success:
                return UIResponse(
                    request_id=event.event_id,
                    response_type=UIResponseType.SUCCESS,
                    data={},
                    message="输入已清空"
                )
            else:
                return UIResponse(
                    request_id=event.event_id,
                    response_type=UIResponseType.ERROR,
                    data={},
                    message="清空失败"
                )
                
        except Exception as e:
            error = handle_ui_error(e, self.component_id, event.event_id, {"action": "clear"})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="清空失败",
                error_details=error.to_dict()
            )
    
    def _validate_math_problem(self, text: str) -> bool:
        """验证是否为数学问题"""
        # 简单的数学问题检测
        math_indicators = [
            r"\\d+",  # 包含数字
            r"[加减乘除]",  # 包含运算符
            r"[几多少]",  # 包含疑问词
            r"[求解计算]",  # 包含求解词
            r"[总共一共]",  # 包含总和词
            r"[剩余还有]",  # 包含剩余词
        ]
        
        for pattern in math_indicators:
            if re.search(pattern, text):
                return True
        
        return False


class BaseReasoningDisplayComponent(IReasoningDisplayComponent):
    """基础推理显示组件"""
    
    def __init__(self, component_id: str, config: Optional[Dict[str, Any]] = None):
        self.component_id = component_id
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 组件状态
        self.state = UIComponentState(
            component_id=component_id,
            component_type=UIComponentType.DISPLAY,
            state={
                "reasoning_steps": [],
                "current_step": -1,
                "total_steps": 0,
                "reasoning_strategy": "",
                "processing_status": "idle",
                "start_time": None,
                "end_time": None
            },
            visible=True,
            enabled=True
        )
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return self.component_id
    
    def get_component_type(self) -> UIComponentType:
        """获取组件类型"""
        return UIComponentType.DISPLAY
    
    def render(self, state: UIComponentState) -> Dict[str, Any]:
        """渲染组件"""
        try:
            reasoning_steps = state.state.get("reasoning_steps", [])
            current_step = state.state.get("current_step", -1)
            
            render_data = {
                "type": "reasoning_display",
                "title": "推理过程",
                "description": "系统正在逐步分析和解决问题",
                "content": {
                    "strategy": state.state.get("reasoning_strategy", ""),
                    "status": state.state.get("processing_status", "idle"),
                    "progress": {
                        "current": current_step + 1,
                        "total": state.state.get("total_steps", 0),
                        "percentage": self._calculate_progress_percentage(current_step, state.state.get("total_steps", 0))
                    },
                    "steps": self._format_reasoning_steps(reasoning_steps, current_step),
                    "timing": {
                        "start_time": state.state.get("start_time"),
                        "end_time": state.state.get("end_time"),
                        "elapsed_time": self._calculate_elapsed_time(
                            state.state.get("start_time"),
                            state.state.get("end_time")
                        )
                    }
                },
                "controls": {
                    "pause": {
                        "type": "button",
                        "label": "暂停",
                        "enabled": state.state.get("processing_status") == "processing"
                    },
                    "resume": {
                        "type": "button",
                        "label": "继续",
                        "enabled": state.state.get("processing_status") == "paused"
                    },
                    "stop": {
                        "type": "button",
                        "label": "停止",
                        "enabled": state.state.get("processing_status") in ["processing", "paused"]
                    }
                }
            }
            
            return render_data
            
        except Exception as e:
            error = handle_ui_error(e, self.component_id, context={"action": "render"})
            return {
                "type": "error",
                "message": "渲染失败",
                "error_details": error.to_dict()
            }
    
    def handle_event(self, event: UIEvent) -> Optional[UIResponse]:
        """处理事件"""
        try:
            if event.event_type == UIEventType.REASONING_START:
                return self._handle_reasoning_start(event)
            elif event.event_type == UIEventType.REASONING_STEP:
                return self._handle_reasoning_step(event)
            elif event.event_type == UIEventType.REASONING_COMPLETE:
                return self._handle_reasoning_complete(event)
            else:
                return None
                
        except Exception as e:
            error = handle_ui_error(e, self.component_id, context={"event": event.event_type})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="事件处理失败",
                error_details=error.to_dict()
            )
    
    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证输入数据"""
        try:
            # 验证推理步骤数据
            if "step_data" in data:
                step_validation = UIUtils.validate_json_schema(
                    data["step_data"], 
                    UISchemas.REASONING_STEP_SCHEMA
                )
                return step_validation
            
            return {"valid": True, "errors": []}
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"验证过程出错: {str(e)}"]
            }
    
    def get_state(self) -> UIComponentState:
        """获取组件状态"""
        return self.state
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """设置组件状态"""
        try:
            self.state.state.update(state)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set state: {e}")
            return False
    
    def display_reasoning_steps(self, steps: List[Dict[str, Any]]) -> bool:
        """显示推理步骤"""
        try:
            # 验证步骤数据
            for step in steps:
                validation = UIUtils.validate_json_schema(step, UISchemas.REASONING_STEP_SCHEMA)
                if not validation["valid"]:
                    self.logger.warning(f"Invalid step data: {validation['errors']}")
            
            self.state.state["reasoning_steps"] = steps
            self.state.state["total_steps"] = len(steps)
            self.state.state["current_step"] = len(steps) - 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to display reasoning steps: {e}")
            return False
    
    def update_current_step(self, step_index: int, step_data: Dict[str, Any]) -> bool:
        """更新当前步骤"""
        try:
            if step_index < 0 or step_index >= len(self.state.state["reasoning_steps"]):
                return False
            
            self.state.state["reasoning_steps"][step_index].update(step_data)
            self.state.state["current_step"] = step_index
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update current step: {e}")
            return False
    
    def highlight_step(self, step_index: int) -> bool:
        """高亮步骤"""
        try:
            if step_index < 0 or step_index >= len(self.state.state["reasoning_steps"]):
                return False
            
            # 清除之前的高亮
            for step in self.state.state["reasoning_steps"]:
                step["highlighted"] = False
            
            # 设置新的高亮
            self.state.state["reasoning_steps"][step_index]["highlighted"] = True
            self.state.state["current_step"] = step_index
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to highlight step: {e}")
            return False
    
    def clear_display(self) -> bool:
        """清空显示"""
        try:
            self.state.state.update({
                "reasoning_steps": [],
                "current_step": -1,
                "total_steps": 0,
                "reasoning_strategy": "",
                "processing_status": "idle",
                "start_time": None,
                "end_time": None
            })
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear display: {e}")
            return False
    
    def _handle_reasoning_start(self, event: UIEvent) -> UIResponse:
        """处理推理开始事件"""
        try:
            self.state.state.update({
                "processing_status": "processing",
                "reasoning_strategy": event.data.get("strategy", ""),
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "reasoning_steps": [],
                "current_step": -1,
                "total_steps": 0
            })
            
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.SUCCESS,
                data={"status": "started"},
                message="推理开始"
            )
            
        except Exception as e:
            error = handle_ui_error(e, self.component_id, event.event_id, {"action": "start"})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="启动失败",
                error_details=error.to_dict()
            )
    
    def _handle_reasoning_step(self, event: UIEvent) -> UIResponse:
        """处理推理步骤事件"""
        try:
            step_data = event.data.get("step_data", {})
            
            # 添加步骤到列表
            self.state.state["reasoning_steps"].append(step_data)
            self.state.state["total_steps"] = len(self.state.state["reasoning_steps"])
            self.state.state["current_step"] = self.state.state["total_steps"] - 1
            
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.SUCCESS,
                data={
                    "step_added": True,
                    "current_step": self.state.state["current_step"],
                    "total_steps": self.state.state["total_steps"]
                },
                message="步骤已添加"
            )
            
        except Exception as e:
            error = handle_ui_error(e, self.component_id, event.event_id, {"action": "step"})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="步骤处理失败",
                error_details=error.to_dict()
            )
    
    def _handle_reasoning_complete(self, event: UIEvent) -> UIResponse:
        """处理推理完成事件"""
        try:
            self.state.state.update({
                "processing_status": "completed",
                "end_time": datetime.now().isoformat()
            })
            
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.SUCCESS,
                data={
                    "status": "completed",
                    "total_steps": self.state.state["total_steps"],
                    "elapsed_time": self._calculate_elapsed_time(
                        self.state.state.get("start_time"),
                        self.state.state.get("end_time")
                    )
                },
                message="推理完成"
            )
            
        except Exception as e:
            error = handle_ui_error(e, self.component_id, event.event_id, {"action": "complete"})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="完成处理失败",
                error_details=error.to_dict()
            )
    
    def _format_reasoning_steps(self, steps: List[Dict[str, Any]], current_step: int) -> List[Dict[str, Any]]:
        """格式化推理步骤"""
        formatted_steps = []
        
        for i, step in enumerate(steps):
            formatted_step = {
                "index": i,
                "type": step.get("step_type", "unknown"),
                "description": step.get("description", ""),
                "confidence": UIUtils.format_confidence(step.get("confidence", 0.0)),
                "status": "current" if i == current_step else "completed" if i < current_step else "pending",
                "highlighted": step.get("highlighted", False),
                "data": step.get("data", {}),
                "timestamp": step.get("timestamp", "")
            }
            formatted_steps.append(formatted_step)
        
        return formatted_steps
    
    def _calculate_progress_percentage(self, current: int, total: int) -> float:
        """计算进度百分比"""
        if total == 0:
            return 0.0
        return (current + 1) / total * 100
    
    def _calculate_elapsed_time(self, start_time: Optional[str], end_time: Optional[str]) -> Optional[str]:
        """计算经过时间"""
        if not start_time:
            return None
        
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time) if end_time else datetime.now()
            
            elapsed = end - start
            return UIUtils.format_processing_time(elapsed.total_seconds() * 1000)
            
        except Exception as e:
            self.logger.error(f"Error calculating elapsed time: {e}")
            return None


class BaseResultDisplayComponent(IResultDisplayComponent):
    """基础结果显示组件"""
    
    def __init__(self, component_id: str, config: Optional[Dict[str, Any]] = None):
        self.component_id = component_id
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 组件状态
        self.state = UIComponentState(
            component_id=component_id,
            component_type=UIComponentType.OUTPUT,
            state={
                "result": None,
                "confidence": 0.0,
                "explanation": "",
                "processing_time": 0.0,
                "strategy_used": "",
                "display_time": None,
                "validation_status": None
            },
            visible=True,
            enabled=True
        )
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return self.component_id
    
    def get_component_type(self) -> UIComponentType:
        """获取组件类型"""
        return UIComponentType.OUTPUT
    
    def render(self, state: UIComponentState) -> Dict[str, Any]:
        """渲染组件"""
        try:
            result = state.state.get("result")
            confidence = state.state.get("confidence", 0.0)
            
            render_data = {
                "type": "result_display",
                "title": "解答结果",
                "description": "系统为您提供的问题解答",
                "content": {
                    "answer": {
                        "value": result.get("final_answer", "") if result else "",
                        "confidence": UIUtils.format_confidence(confidence),
                        "confidence_level": self._get_confidence_level(confidence),
                        "strategy": state.state.get("strategy_used", ""),
                        "processing_time": UIUtils.format_processing_time(
                            state.state.get("processing_time", 0.0)
                        )
                    },
                    "explanation": {
                        "text": state.state.get("explanation", ""),
                        "reasoning_steps": result.get("reasoning_steps", []) if result else [],
                        "key_insights": self._extract_key_insights(result) if result else []
                    },
                    "validation": {
                        "status": state.state.get("validation_status"),
                        "checks": result.get("validation_checks", []) if result else [],
                        "issues": result.get("validation_issues", []) if result else []
                    },
                    "metadata": {
                        "display_time": state.state.get("display_time"),
                        "result_id": result.get("result_id") if result else None,
                        "version": "1.0.0"
                    }
                },
                "actions": {
                    "explain": {
                        "type": "button",
                        "label": "详细解释",
                        "enabled": bool(result)
                    },
                    "validate": {
                        "type": "button",
                        "label": "验证结果",
                        "enabled": bool(result)
                    },
                    "export": {
                        "type": "button",
                        "label": "导出结果",
                        "enabled": bool(result)
                    },
                    "feedback": {
                        "type": "button",
                        "label": "反馈",
                        "enabled": bool(result)
                    }
                }
            }
            
            return render_data
            
        except Exception as e:
            error = handle_ui_error(e, self.component_id, context={"action": "render"})
            return {
                "type": "error",
                "message": "渲染失败",
                "error_details": error.to_dict()
            }
    
    def handle_event(self, event: UIEvent) -> Optional[UIResponse]:
        """处理事件"""
        try:
            if event.event_type == UIEventType.RESULT_DISPLAY:
                return self._handle_result_display(event)
            else:
                return None
                
        except Exception as e:
            error = handle_ui_error(e, self.component_id, context={"event": event.event_type})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="事件处理失败",
                error_details=error.to_dict()
            )
    
    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证输入数据"""
        try:
            # 验证结果数据
            if "result" in data:
                result_validation = UIUtils.validate_json_schema(
                    data["result"], 
                    UISchemas.RESULT_SCHEMA
                )
                return result_validation
            
            return {"valid": True, "errors": []}
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"验证过程出错: {str(e)}"]
            }
    
    def get_state(self) -> UIComponentState:
        """获取组件状态"""
        return self.state
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """设置组件状态"""
        try:
            self.state.state.update(state)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set state: {e}")
            return False
    
    def display_result(self, result: Dict[str, Any]) -> bool:
        """显示结果"""
        try:
            # 验证结果数据
            validation = self.validate_input({"result": result})
            if not validation["valid"]:
                self.logger.warning(f"Invalid result data: {validation['errors']}")
            
            self.state.state.update({
                "result": result,
                "display_time": datetime.now().isoformat(),
                "confidence": result.get("confidence", 0.0),
                "strategy_used": result.get("strategy_used", ""),
                "processing_time": result.get("processing_time", 0.0)
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to display result: {e}")
            return False
    
    def display_confidence(self, confidence: float) -> bool:
        """显示置信度"""
        try:
            if not 0.0 <= confidence <= 1.0:
                self.logger.warning(f"Invalid confidence value: {confidence}")
                return False
            
            self.state.state["confidence"] = confidence
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to display confidence: {e}")
            return False
    
    def display_explanation(self, explanation: str) -> bool:
        """显示解释"""
        try:
            cleaned_explanation = UIUtils.sanitize_input(explanation)
            self.state.state["explanation"] = cleaned_explanation
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to display explanation: {e}")
            return False
    
    def clear_result(self) -> bool:
        """清空结果"""
        try:
            self.state.state.update({
                "result": None,
                "confidence": 0.0,
                "explanation": "",
                "processing_time": 0.0,
                "strategy_used": "",
                "display_time": None,
                "validation_status": None
            })
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear result: {e}")
            return False
    
    def _handle_result_display(self, event: UIEvent) -> UIResponse:
        """处理结果显示事件"""
        try:
            result_data = event.data.get("result", {})
            
            success = self.display_result(result_data)
            
            if success:
                return UIResponse(
                    request_id=event.event_id,
                    response_type=UIResponseType.SUCCESS,
                    data={
                        "result_displayed": True,
                        "final_answer": result_data.get("final_answer", ""),
                        "confidence": result_data.get("confidence", 0.0)
                    },
                    message="结果已显示"
                )
            else:
                return UIResponse(
                    request_id=event.event_id,
                    response_type=UIResponseType.ERROR,
                    data={},
                    message="结果显示失败"
                )
                
        except Exception as e:
            error = handle_ui_error(e, self.component_id, event.event_id, {"action": "display"})
            return UIResponse(
                request_id=event.event_id,
                response_type=UIResponseType.ERROR,
                data={},
                message="显示失败",
                error_details=error.to_dict()
            )
    
    def _get_confidence_level(self, confidence: float) -> str:
        """获取置信度等级"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def _extract_key_insights(self, result: Dict[str, Any]) -> List[str]:
        """提取关键见解"""
        insights = []
        
        # 从推理步骤中提取关键信息
        reasoning_steps = result.get("reasoning_steps", [])
        for step in reasoning_steps:
            if step.get("confidence", 0) > 0.8:
                insights.append(step.get("description", ""))
        
        # 限制见解数量
        return insights[:5]


# 导出组件类
__all__ = [
    "BaseProblemInputComponent",
    "BaseReasoningDisplayComponent", 
    "BaseResultDisplayComponent"
]