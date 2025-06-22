"""数学应用题粗粒度分类器模块

用于对数学应用题进行初步分类，识别问题类型和特征。
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class MWPCoarseClassifier:
    """数学应用题粗粒度分类器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化分类器
        
        Args:
            config: 配置信息
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def classify(self, processed_text: Any) -> Dict[str, Any]:
        """对处理后的文本进行粗粒度分类
        
        Args:
            processed_text: 处理后的文本对象
            
        Returns:
            Dict 包含分类结果，包括：
            - pattern_categories: 匹配的模式类别列表
            - features: 问题特征字典
        """
        features = getattr(processed_text, 'features', {})
        pattern_categories = set()
        # 1. 领域特征
        domain = features.get('domain_indicators', {}) if isinstance(features, dict) else {}
        if domain.get('liquid_related'):
            pattern_categories.add('数量关系_基础计量')
        if domain.get('motion_related'):
            pattern_categories.add('数量关系_行程追及')
        if domain.get('work_related'):
            pattern_categories.add('数量关系_合作效率')
        if domain.get('growth_related'):
            pattern_categories.add('数量关系_复利')
        # 2. 动词特征
        verbs = set()
        if hasattr(processed_text, 'pos_tags') and hasattr(processed_text, 'segmentation'):
            for word, pos in zip(processed_text.segmentation, processed_text.pos_tags):
                if pos.startswith('v') or pos == 'VERB':
                    verbs.add(word.lower())
        if 'leak' in verbs or 'leaking' in verbs:
            pattern_categories.add('动作关系_流动类')
        if 'add' in verbs or 'added' in verbs:
            pattern_categories.add('动作关系_变化类')
        # 3. 单位特征
        units = set()
        if hasattr(processed_text, 'values_and_units'):
            for v in processed_text.values_and_units.values():
                unit = v.get('unit', '').lower()
                if 'l' in unit or 'liter' in unit:
                    pattern_categories.add('数量关系_基础计量')
                if 'min' in unit or 'minute' in unit:
                    pattern_categories.add('属性关系_时间属性')
        # 4. 目标变量
        target_var = ''
        if hasattr(features, 'question_target'):
            target_var = features.question_target.get('target_variable', '')
        elif isinstance(features, dict):
            target_var = features.get('question_target', {}).get('target_variable', '')
        if target_var:
            if 'time' in target_var.lower() or 't' == target_var.lower():
                pattern_categories.add('属性关系_时间属性')
            if 'concentration' in target_var.lower():
                pattern_categories.add('数量关系_流水混合')
            if 'distance' in target_var.lower():
                pattern_categories.add('数量关系_行程追及')
        # 5. 兜底
        if not pattern_categories:
            pattern_categories.add('数量关系_基础计量')
        return {
            'pattern_categories': list(pattern_categories),
            'features': features if isinstance(features, dict) else features.__dict__
        } 