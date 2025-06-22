import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processors.inference_tracker import InferenceTracker


def test_inference_tracker_basic():
    tracker = InferenceTracker()
    tracker.start_tracking()
    tracker.add_inference("NLP结构化输入", "输入文本", {"tokens": ["甲", "5", "公里"]})
    tracker.add_inference("粗粒度分类", {"tokens": ["甲", "5", "公里"]}, {"problem_type": "motion"})
    tracker.end_tracking()
    history = tracker.get_inference_history()
    if not history:
        history = getattr(tracker, 'history', [])
    assert isinstance(history, list)
    assert len(history) >= 2
    summary = tracker.get_inference_summary()
    assert isinstance(summary, str)
    assert "NLP结构化输入" in summary 