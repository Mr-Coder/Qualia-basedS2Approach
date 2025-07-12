import re
from typing import Dict


class Preprocessor:
    """
    Minimal Preprocessor for math problem samples.
    Adds cleaned_text, problem_type, classification_confidence, complexity_level.
    """
    def __init__(self):
        pass

    def process(self, sample: Dict) -> Dict:
        text = sample.get("problem") or sample.get("question") or sample.get("text") or ""
        cleaned = re.sub(r"\s+", " ", text.strip())
        # 简单规则分类
        if any(k in cleaned for k in ["一共", "总共", "合计", "total", "together"]):
            problem_type = "arithmetic"
            confidence = 0.9
        elif any(k in cleaned for k in ["平均", "每", "per", "each"]):
            problem_type = "average"
            confidence = 0.8
        else:
            problem_type = "unknown"
            confidence = 0.5
        # 简单复杂度分级
        if len(cleaned) < 20:
            complexity = "L0"
        elif len(cleaned) < 40:
            complexity = "L1"
        elif len(cleaned) < 80:
            complexity = "L2"
        else:
            complexity = "L3"
        sample = dict(sample)
        sample["cleaned_text"] = cleaned
        sample["problem_type"] = problem_type
        sample["classification_confidence"] = confidence
        sample["complexity_level"] = complexity
        return sample 