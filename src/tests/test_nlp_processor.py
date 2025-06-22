import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from processors.nlp_processor import NLPProcessor


@pytest.fixture
def nlp():
    return NLPProcessor({})

def test_nlp_basic_tokenization(nlp):
    text = "甲以每小时5公里的速度行驶，问2小时后走了多少公里？"
    result = nlp.process_text(text)
    assert hasattr(result, 'segmentation')
    assert hasattr(result, 'pos_tags')
    assert isinstance(result.segmentation, list)
    assert isinstance(result.pos_tags, list)
    assert len(result.segmentation) == len(result.pos_tags)
    assert '公里' in ''.join(result.segmentation) 