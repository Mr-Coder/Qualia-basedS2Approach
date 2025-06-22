import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from models.processed_text import ProcessedText
from processors.MWP_process import MWPCoarseClassifier


def test_coarse_classification_basic():
    classifier = MWPCoarseClassifier()
    pt = ProcessedText(
        raw_text="甲以每小时5公里的速度行驶，问2小时后走了多少公里？",
        segmentation=["甲", "以", "每小时", "5", "公里", "的", "速度", "行驶", "，", "问", "2", "小时", "后", "走了", "多少", "公里", "？"],
        pos_tags=["n", "p", "t", "m", "q", "u", "n", "v", "w", "v", "m", "q", "f", "v", "r", "q", "w"],
        dependencies=[],
        semantic_roles={},
        cleaned_text=None,
        tokens=[],
        ner_tags=[],
        features={},
        values_and_units={"速度": {"value": 5, "unit": "km/h"}, "时间": {"value": 2, "unit": "h"}}
    )
    result = classifier.classify(pt)
    assert 'problem_type' in result
    assert 'pattern_categories' in result
    assert isinstance(result['pattern_categories'], list) 