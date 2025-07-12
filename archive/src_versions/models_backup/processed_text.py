"""处理后的文本类"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProcessedText:
    """处理后的文本类
    
    Attributes:
        raw_text: 原始文本
        segmentation: 分词结果
        pos_tags: 词性标注结果
        dependencies: 依存句法结果
        semantic_roles: 语义角色结果
        cleaned_text: 清理后的文本
        tokens: 分词结果
        ner_tags: 命名实体识别结果
        features: 额外特征字典
        values_and_units: 数值和单位信息
    """
    raw_text: str
    segmentation: List[str] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list)
    dependencies: List[Any] = field(default_factory=list)
    semantic_roles: Dict[str, Any] = field(default_factory=dict)
    cleaned_text: Optional[str] = None
    tokens: List[str] = field(default_factory=list)
    ner_tags: List[Tuple[str, str]] = field(default_factory=list)
    features: Optional[Dict[str, Any]] = None
    values_and_units: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.cleaned_text is None:
            self.cleaned_text = self.raw_text
        if self.features is None:
            self.features = {}
        if self.values_and_units is None:
            self.values_and_units = {}

    def __str__(self) -> str:
        """返回字符串表示
        
        Returns:
            str: 清理后的文本
        """
        return self.cleaned_text
        
    def __repr__(self) -> str:
        """返回字符串表示
        
        Returns:
            str: 详细信息
        """
        return (f"ProcessedText(raw_text='{self.raw_text}', cleaned_text='{self.cleaned_text}', "
                f"tokens={self.tokens}, pos_tags={self.pos_tags}, ner_tags={self.ner_tags})")
