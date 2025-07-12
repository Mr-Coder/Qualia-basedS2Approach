import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from src.processors.inference_tracker import InferenceTracker


@pytest.fixture
def tracker():
    """创建一个新的推理追踪器实例"""
    return InferenceTracker()

def test_initialization(tracker):
    """测试追踪器初始化"""
    assert tracker.inferences == []
    assert tracker.start_time is None
    assert tracker.end_time is None

def test_start_tracking(tracker):
    """测试开始追踪"""
    tracker.start_tracking()
    assert tracker.start_time is not None
    assert isinstance(tracker.start_time, datetime)
    assert tracker.inferences == []

def test_end_tracking(tracker):
    """测试结束追踪"""
    tracker.start_tracking()
    # 等待一小段时间以确保时间差
    import time
    time.sleep(0.1)
    tracker.end_tracking()
    
    assert tracker.end_time is not None
    assert isinstance(tracker.end_time, datetime)
    assert tracker.end_time > tracker.start_time

def test_add_inference(tracker):
    """测试添加推理步骤"""
    step_name = "test_step"
    input_data = {"x": 1}
    output_data = {"y": 2}
    metadata = {"time_taken": 0.5}
    
    tracker.add_inference(step_name, input_data, output_data, metadata)
    
    assert len(tracker.inferences) == 1
    inference = tracker.inferences[0]
    assert inference["step_name"] == step_name
    assert inference["input"] == input_data
    assert inference["output"] == output_data
    assert inference["metadata"] == metadata
    assert isinstance(inference["timestamp"], datetime)

def test_get_inference_history(tracker):
    """测试获取推理历史"""
    step_name = "test_step"
    input_data = {"x": 1}
    output_data = {"y": 2}
    metadata = {"time_taken": 0.5}
    
    tracker.add_inference(step_name, input_data, output_data, metadata)
    
    history = tracker.get_inference_history()
    assert len(history) == 1
    inference = history[0]
    assert inference["step_name"] == step_name
    assert inference["input"] == input_data
    assert inference["output"] == output_data
    assert inference["metadata"] == metadata
    assert isinstance(inference["timestamp"], datetime)

def test_get_inference_summary(tracker):
    """测试获取推理摘要"""
    tracker.start_tracking()
    step_name = "test_step"
    input_data = {"x": 1}
    output_data = {"y": 2}
    metadata = {"time_taken": 0.5}
    
    tracker.add_inference(step_name, input_data, output_data, metadata)
    tracker.end_tracking()
    
    summary = tracker.get_inference_summary()
    assert summary["total_steps"] == 1
    assert summary["start_time"] is not None
    assert summary["end_time"] is not None
    assert summary["duration"] is not None
    assert summary["steps"] == [step_name]

def test_get_step_details(tracker):
    """测试获取步骤详细信息"""
    step_name = "test_step"
    input_data = {"x": 1}
    output_data = {"y": 2}
    metadata = {"time_taken": 0.5}
    
    tracker.add_inference(step_name, input_data, output_data, metadata)
    
    details = tracker.get_step_details(step_name)
    assert len(details) == 1
    inference = details[0]
    assert inference["step_name"] == step_name
    assert inference["input"] == input_data
    assert inference["output"] == output_data
    assert inference["metadata"] == metadata
    assert isinstance(inference["timestamp"], datetime)

def test_clear(tracker):
    """测试清除记录"""
    tracker.start_tracking()
    step_name = "test_step"
    input_data = {"x": 1}
    output_data = {"y": 2}
    metadata = {"time_taken": 0.5}
    
    tracker.add_inference(step_name, input_data, output_data, metadata)
    tracker.end_tracking()
    
    tracker.clear()
    
    assert tracker.inferences == []
    assert tracker.start_time is None
    assert tracker.end_time is None
