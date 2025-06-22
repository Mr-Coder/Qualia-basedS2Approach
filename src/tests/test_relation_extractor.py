import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from models.processed_text import ProcessedText
from processors.relation_extractor import RelationExtractor
from processors.relation_matcher import RelationMatcher


@pytest.fixture
def extractor():
    matcher = RelationMatcher()
    return RelationExtractor({}, matcher)

def test_relation_extraction_basic(extractor):
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
        values_and_units={"速度": 5, "时间": 2}
    )
    classification_result = {"problem_type": "motion", "pattern_categories": ["速度_距离_时间"]}
    result = extractor.extract_relations(pt, classification_result)
    assert 'explicit_relations' in result
    assert isinstance(result['explicit_relations'], list) 