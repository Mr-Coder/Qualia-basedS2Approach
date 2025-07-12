"""
元知识模块 - 为数学推理提供丰富的背景知识
包含数学概念、解题策略、常见错误模式等元知识
"""

import re
from typing import Dict, List, Optional, Set, Tuple


class MetaKnowledge:
    """
    数学推理的元知识库
    提供概念定义、解题策略、错误模式等知识
    """
    
    def __init__(self):
        # 数学概念定义
        self.concepts = {
            "分数": {
                "definition": "分数表示整体的一部分，形式为 a/b，其中a是分子，b是分母",
                "properties": ["分子分母都是整数", "分母不能为0", "可以约分", "可以通分"],
                "operations": ["加法需要通分", "乘法直接相乘", "除法转换为乘法"],
                "common_mistakes": ["忘记约分", "通分错误", "符号错误"]
            },
            "百分比": {
                "definition": "百分比是分数的特殊形式，分母为100",
                "properties": ["可以转换为小数", "可以转换为分数", "用于表示比例"],
                "operations": ["转换为小数除以100", "转换为分数除以100", "计算部分值"],
                "common_mistakes": ["忘记除以100", "符号错误", "单位错误"]
            },
            "面积": {
                "definition": "面积是二维图形所占的平面大小",
                "properties": ["单位是平方单位", "总是正数", "可以相加"],
                "operations": ["矩形面积 = 长 × 宽", "正方形面积 = 边长²", "三角形面积 = 底 × 高 ÷ 2"],
                "common_mistakes": ["单位错误", "公式错误", "计算错误"]
            },
            "体积": {
                "definition": "体积是三维物体所占的空间大小",
                "properties": ["单位是立方单位", "总是正数", "可以相加"],
                "operations": ["长方体体积 = 长 × 宽 × 高", "正方体体积 = 边长³", "圆柱体积 = πr²h"],
                "common_mistakes": ["单位错误", "公式错误", "π值错误"]
            },
            "速度": {
                "definition": "速度是单位时间内移动的距离",
                "properties": ["单位是距离/时间", "可以是标量或向量", "与时间成反比"],
                "operations": ["速度 = 距离 ÷ 时间", "距离 = 速度 × 时间", "时间 = 距离 ÷ 速度"],
                "common_mistakes": ["单位转换错误", "方向错误", "时间计算错误"]
            },
            "折扣": {
                "definition": "折扣是商品价格的减少，通常以百分比或分数表示",
                "properties": ["折扣率小于等于1", "折扣价格小于原价", "可以转换为百分比"],
                "operations": ["折扣价格 = 原价 × 折扣率", "原价 = 折扣价格 ÷ 折扣率", "折扣率 = 折扣价格 ÷ 原价"],
                "common_mistakes": ["折扣率理解错误", "计算方向错误", "单位错误"]
            },
            "利润": {
                "definition": "利润是收入减去成本后的剩余",
                "properties": ["可以是正数或负数", "利润率 = 利润 ÷ 成本", "总利润 = 单价利润 × 数量"],
                "operations": ["利润 = 收入 - 成本", "利润率 = 利润 ÷ 成本 × 100%", "收入 = 成本 + 利润"],
                "common_mistakes": ["正负号错误", "利润率计算错误", "单位错误"]
            },
            "平均数": {
                "definition": "平均数是所有数值的总和除以数值的个数",
                "properties": ["介于最大值和最小值之间", "受极值影响", "可以加权"],
                "operations": ["平均数 = 总和 ÷ 个数", "总和 = 平均数 × 个数", "加权平均 = Σ(值×权重) ÷ Σ权重"],
                "common_mistakes": ["忘记除以个数", "权重错误", "极值影响"]
            },
            "比例": {
                "definition": "比例表示两个量之间的比值关系",
                "properties": ["可以约分", "可以交叉相乘", "保持相等关系"],
                "operations": ["a:b = c:d 等价于 a×d = b×c", "比例可以约分", "比例可以扩展"],
                "common_mistakes": ["交叉相乘错误", "约分错误", "单位不匹配"]
            },
            "方程": {
                "definition": "方程是含有未知数的等式",
                "properties": ["等式两边相等", "可以移项", "可以合并同类项"],
                "operations": ["移项变号", "合并同类项", "系数化为1"],
                "common_mistakes": ["移项错误", "符号错误", "计算错误"]
            }
        }
        
        # 解题策略
        self.strategies = {
            "逆向思维": {
                "description": "从结果倒推，找到解题路径",
                "applicable_problems": ["已知结果求条件", "验证题", "选择题"],
                "steps": ["确定目标", "分析已知条件", "寻找中间步骤", "验证路径"],
                "examples": ["已知面积求边长", "已知速度求时间"],
                "difficulty": "中等",
                "success_rate": 0.85
            },
            "分类讨论": {
                "description": "根据条件的不同情况分别讨论",
                "applicable_problems": ["绝对值问题", "分段函数", "条件概率"],
                "steps": ["识别分类条件", "列出所有情况", "分别求解", "合并结果"],
                "examples": ["|x| = a 的解", "分段函数求值"],
                "difficulty": "中等",
                "success_rate": 0.80
            },
            "数形结合": {
                "description": "结合数字计算和图形分析",
                "applicable_problems": ["几何问题", "函数问题", "应用题"],
                "steps": ["画图分析", "标注已知量", "建立方程", "求解验证"],
                "examples": ["行程问题", "几何证明"],
                "difficulty": "中等",
                "success_rate": 0.82
            },
            "等量代换": {
                "description": "用等价的表达式替换原表达式",
                "applicable_problems": ["方程求解", "证明题", "化简题"],
                "steps": ["识别等价关系", "选择替换对象", "执行替换", "验证等价性"],
                "examples": ["解方程组", "证明恒等式"],
                "difficulty": "中等",
                "success_rate": 0.78
            },
            "设未知数": {
                "description": "引入未知数建立方程求解",
                "applicable_problems": ["应用题", "几何问题", "行程问题"],
                "steps": ["识别未知量", "设未知数", "建立方程", "求解验证"],
                "examples": ["年龄问题", "行程问题", "几何问题"],
                "difficulty": "中等",
                "success_rate": 0.88
            },
            "列表法": {
                "description": "通过列表整理信息，寻找规律",
                "applicable_problems": ["排列组合", "概率问题", "逻辑推理"],
                "steps": ["列出所有可能", "分析规律", "排除不可能", "确定答案"],
                "examples": ["排列组合", "逻辑推理", "概率计算"],
                "difficulty": "简单",
                "success_rate": 0.90
            },
            "假设法": {
                "description": "通过假设验证答案的正确性",
                "applicable_problems": ["选择题", "验证题", "逻辑题"],
                "steps": ["提出假设", "验证假设", "调整假设", "确定答案"],
                "examples": ["选择题验证", "逻辑推理", "证明题"],
                "difficulty": "中等",
                "success_rate": 0.75
            },
            "整体思想": {
                "description": "将问题作为一个整体考虑，简化计算",
                "applicable_problems": ["复杂计算", "分数问题", "比例问题"],
                "steps": ["识别整体", "简化计算", "整体求解", "验证结果"],
                "examples": ["分数运算", "比例计算", "复杂应用题"],
                "difficulty": "中等",
                "success_rate": 0.85
            },
            "递推法": {
                "description": "通过递推关系逐步求解",
                "applicable_problems": ["数列问题", "递归问题", "动态规划"],
                "steps": ["建立递推关系", "确定初始条件", "逐步计算", "验证结果"],
                "examples": ["斐波那契数列", "等差数列求和", "递归问题"],
                "difficulty": "困难",
                "success_rate": 0.70
            },
            "反证法": {
                "description": "假设结论不成立，推出矛盾",
                "applicable_problems": ["证明题", "存在性问题", "唯一性问题"],
                "steps": ["假设结论不成立", "推导矛盾", "否定假设", "证明原结论"],
                "examples": ["证明无理数", "证明唯一性", "存在性证明"],
                "difficulty": "困难",
                "success_rate": 0.65
            },
            "构造法": {
                "description": "构造满足条件的对象或反例",
                "applicable_problems": ["存在性问题", "构造题", "反例题"],
                "steps": ["分析条件", "设计构造方案", "验证构造", "得出结论"],
                "examples": ["构造反例", "构造满足条件的数", "构造图形"],
                "difficulty": "困难",
                "success_rate": 0.60
            },
            "极值法": {
                "description": "利用极值性质求解问题",
                "applicable_problems": ["最值问题", "不等式", "优化问题"],
                "steps": ["识别极值条件", "建立函数关系", "求极值", "验证结果"],
                "examples": ["求最大值最小值", "不等式证明", "优化问题"],
                "difficulty": "困难",
                "success_rate": 0.72
            },
            "归纳法": {
                "description": "通过数学归纳法证明结论",
                "applicable_problems": ["数列证明", "恒等式证明", "不等式证明"],
                "steps": ["验证基础情况", "假设归纳假设", "证明递推", "得出结论"],
                "examples": ["数列求和公式", "恒等式证明", "不等式证明"],
                "difficulty": "困难",
                "success_rate": 0.68
            },
            "数轴法": {
                "description": "利用数轴分析问题",
                "applicable_problems": ["绝对值问题", "不等式", "区间问题"],
                "steps": ["画数轴", "标注关键点", "分析区间", "确定解集"],
                "examples": ["|x-a| < b", "不等式求解", "区间运算"],
                "difficulty": "简单",
                "success_rate": 0.88
            },
            "换元法": {
                "description": "通过变量替换简化问题",
                "applicable_problems": ["复杂方程", "积分问题", "化简题"],
                "steps": ["识别替换对象", "进行变量替换", "求解新问题", "还原结果"],
                "examples": ["解高次方程", "积分换元", "表达式化简"],
                "difficulty": "困难",
                "success_rate": 0.75
            },
            "配方法": {
                "description": "通过配方简化表达式",
                "applicable_problems": ["二次函数", "完全平方", "化简题"],
                "steps": ["识别配方形式", "进行配方", "简化表达式", "得出结论"],
                "examples": ["二次函数配方", "完全平方公式", "表达式化简"],
                "difficulty": "中等",
                "success_rate": 0.80
            },
            "因式分解": {
                "description": "将多项式分解为因式乘积",
                "applicable_problems": ["多项式化简", "方程求解", "证明题"],
                "steps": ["识别分解方法", "进行因式分解", "简化表达式", "验证结果"],
                "examples": ["多项式因式分解", "解高次方程", "恒等式证明"],
                "difficulty": "中等",
                "success_rate": 0.78
            },
            "配凑法": {
                "description": "通过配凑简化计算",
                "applicable_problems": ["复杂计算", "化简题", "证明题"],
                "steps": ["识别配凑目标", "进行配凑", "简化计算", "验证结果"],
                "examples": ["复杂分数化简", "表达式配凑", "计算简化"],
                "difficulty": "中等",
                "success_rate": 0.82
            }
        }
        
        # 常见错误模式
        self.error_patterns = {
            "计算错误": {
                "types": ["进位错误", "借位错误", "符号错误", "小数点错误"],
                "prevention": ["仔细检查", "使用估算", "反向验证"],
                "examples": ["12 + 8 = 19", "100 - 7 = 93"]
            },
            "概念错误": {
                "types": ["定义理解错误", "性质应用错误", "公式记忆错误"],
                "prevention": ["复习概念", "理解原理", "多做练习"],
                "examples": ["分数加法直接相加", "面积单位错误"]
            },
            "逻辑错误": {
                "types": ["推理步骤错误", "条件使用错误", "结论错误"],
                "prevention": ["检查逻辑链", "验证条件", "确认结论"],
                "examples": ["从A推出B，从B推出C，所以A=C"]
            },
            "单位错误": {
                "types": ["单位转换错误", "单位遗漏", "单位不匹配"],
                "prevention": ["注意单位", "统一单位", "检查单位"],
                "examples": ["1米 = 100厘米", "速度单位错误"]
            }
        }
        
        # 数学符号和术语
        self.symbols = {
            "运算符号": {
                "+": "加法", "-": "减法", "×": "乘法", "÷": "除法",
                "=": "等于", "≠": "不等于", ">": "大于", "<": "小于",
                "≥": "大于等于", "≤": "小于等于", "±": "正负号"
            },
            "几何符号": {
                "∠": "角", "⊥": "垂直", "∥": "平行", "≅": "全等",
                "∽": "相似", "°": "度", "π": "圆周率"
            },
            "集合符号": {
                "∈": "属于", "∉": "不属于", "⊂": "真包含", "⊆": "包含",
                "∪": "并集", "∩": "交集", "∅": "空集"
            }
        }
        
        # 单位转换关系
        self.unit_conversions = {
            "长度": {
                "米": {"厘米": 100, "毫米": 1000, "千米": 0.001},
                "厘米": {"米": 0.01, "毫米": 10, "千米": 0.00001},
                "毫米": {"米": 0.001, "厘米": 0.1, "千米": 0.000001}
            },
            "面积": {
                "平方米": {"平方厘米": 10000, "平方毫米": 1000000},
                "平方厘米": {"平方米": 0.0001, "平方毫米": 100},
                "平方毫米": {"平方米": 0.000001, "平方厘米": 0.01}
            },
            "体积": {
                "立方米": {"立方厘米": 1000000, "立方毫米": 1000000000},
                "立方厘米": {"立方米": 0.000001, "立方毫米": 1000},
                "立方毫米": {"立方米": 0.000000001, "立方厘米": 0.001}
            },
            "重量": {
                "千克": {"克": 1000, "毫克": 1000000, "吨": 0.001},
                "克": {"千克": 0.001, "毫克": 1000, "吨": 0.000001},
                "毫克": {"千克": 0.000001, "克": 0.001, "吨": 0.000000001}
            },
            "时间": {
                "小时": {"分钟": 60, "秒": 3600, "天": 1/24},
                "分钟": {"小时": 1/60, "秒": 60, "天": 1/1440},
                "秒": {"小时": 1/3600, "分钟": 1/60, "天": 1/86400}
            }
        }
    
    def get_concept_info(self, concept_name: str) -> Optional[Dict]:
        """获取概念信息"""
        return self.concepts.get(concept_name)
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict]:
        """获取解题策略信息"""
        return self.strategies.get(strategy_name)
    
    def identify_concepts_in_text(self, text: str) -> List[str]:
        """识别文本中的数学概念"""
        identified_concepts = []
        
        # 概念关键词映射
        concept_keywords = {
            "分数": ["分数", "分之", "/", "÷"],
            "百分比": ["百分比", "%", "百分之"],
            "面积": ["面积", "平方", "长", "宽"],
            "体积": ["体积", "立方", "高"],
            "速度": ["速度", "每小时", "每分钟"],
            "折扣": ["折扣", "打折", "折", "优惠"],
            "利润": ["利润", "盈利", "赚", "亏"],
            "平均数": ["平均", "均值", "每"],
            "比例": ["比例", "比", ":"],
            "方程": ["方程", "等式", "="]
        }
        
        for concept, keywords in concept_keywords.items():
            if any(keyword in text for keyword in keywords):
                identified_concepts.append(concept)
        
        return identified_concepts
    
    def suggest_strategies(self, problem_text: str) -> List[str]:
        """根据问题文本推荐解题策略 - 增强版"""
        suggested_strategies = []
        strategy_scores = {}  # 策略评分
        
        # 基于关键词推荐策略
        keyword_strategies = {
            "逆向思维": ["已知", "求", "验证", "原价", "现价", "结果", "答案"],
            "分类讨论": ["如果", "当", "情况", "可能", "不同", "分别"],
            "数形结合": ["图形", "画", "图", "长方形", "正方形", "三角形", "圆形", "几何"],
            "等量代换": ["等于", "替换", "代换", "等价", "相同"],
            "设未知数": ["多少", "几", "未知", "设", "求", "x", "y"],
            "列表法": ["排列", "组合", "可能", "情况", "所有", "列举"],
            "假设法": ["假设", "验证", "正确", "错误", "判断"],
            "整体思想": ["整体", "全部", "总和", "平均", "综合"],
            "递推法": ["数列", "递推", "规律", "下一个", "第n项"],
            "反证法": ["证明", "不存在", "唯一", "矛盾", "反证"],
            "构造法": ["构造", "存在", "找到", "设计", "创造"],
            "极值法": ["最大", "最小", "最值", "极值", "最优"],
            "归纳法": ["归纳", "证明", "对所有", "数学归纳"],
            "数轴法": ["绝对值", "不等式", "区间", "数轴", "范围"],
            "换元法": ["换元", "替换", "变量", "积分", "复杂"],
            "配方法": ["配方", "完全平方", "二次", "平方"],
            "因式分解": ["因式", "分解", "多项式", "高次"],
            "配凑法": ["配凑", "化简", "简化", "复杂计算"]
        }
        
        # 计算策略得分
        for strategy, keywords in keyword_strategies.items():
            score = 0
            for keyword in keywords:
                if keyword in problem_text:
                    score += 1
            if score > 0:
                strategy_scores[strategy] = score
        
        # 基于概念推荐策略
        concepts = self.identify_concepts_in_text(problem_text)
        concept_strategies = {
            "比例": ["整体思想", "等量代换"],
            "分数": ["整体思想", "配凑法"],
            "方程": ["设未知数", "等量代换"],
            "面积": ["数形结合", "设未知数"],
            "体积": ["数形结合", "设未知数"],
            "速度": ["设未知数", "等量代换"],
            "折扣": ["逆向思维", "等量代换"],
            "利润": ["设未知数", "等量代换"],
            "平均数": ["整体思想", "设未知数"],
            "百分比": ["等量代换", "整体思想"]
        }
        
        for concept in concepts:
            if concept in concept_strategies:
                for strategy in concept_strategies[concept]:
                    strategy_scores[strategy] = strategy_scores.get(strategy, 0) + 2
        
        # 基于问题复杂度推荐策略
        complexity_indicators = {
            "简单": ["列表法", "数轴法", "整体思想"],
            "中等": ["设未知数", "等量代换", "数形结合", "配凑法"],
            "困难": ["递推法", "反证法", "构造法", "极值法", "归纳法", "换元法"]
        }
        
        # 简单判断问题复杂度
        if len(problem_text) < 50 and any(word in problem_text for word in ["简单", "基础", "基本"]):
            for strategy in complexity_indicators["简单"]:
                strategy_scores[strategy] = strategy_scores.get(strategy, 0) + 1
        elif any(word in problem_text for word in ["复杂", "困难", "高级", "证明", "推导"]):
            for strategy in complexity_indicators["困难"]:
                strategy_scores[strategy] = strategy_scores.get(strategy, 0) + 1
        
        # 根据得分排序并选择策略
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择得分最高的策略（最多5个）
        max_strategies = 5
        for strategy, score in sorted_strategies[:max_strategies]:
            if score > 0:
                suggested_strategies.append(strategy)
        
        # 如果没有推荐策略，提供默认策略
        if not suggested_strategies:
            suggested_strategies = ["设未知数", "整体思想"]
        
        return suggested_strategies
    
    def get_strategy_priority(self, strategy_name: str, problem_text: str) -> float:
        """获取策略优先级评分"""
        strategy_info = self.strategies.get(strategy_name)
        if not strategy_info:
            return 0.0
        
        # 基础分数
        base_score = strategy_info.get("success_rate", 0.5)
        
        # 根据问题文本调整分数
        adjustment = 0.0
        
        # 检查策略是否适用于当前问题
        applicable_problems = strategy_info.get("applicable_problems", [])
        for problem_type in applicable_problems:
            if any(keyword in problem_text for keyword in problem_type.split()):
                adjustment += 0.1
        
        # 检查难度匹配
        difficulty = strategy_info.get("difficulty", "中等")
        if difficulty == "简单" and len(problem_text) < 50:
            adjustment += 0.05
        elif difficulty == "困难" and len(problem_text) > 100:
            adjustment += 0.05
        
        return min(1.0, base_score + adjustment)
    
    def suggest_strategies_with_priority(self, problem_text: str) -> List[Dict]:
        """推荐策略并返回优先级信息"""
        strategies = self.suggest_strategies(problem_text)
        result = []
        
        for strategy in strategies:
            priority = self.get_strategy_priority(strategy, problem_text)
            strategy_info = self.strategies.get(strategy, {})
            
            result.append({
                "strategy": strategy,
                "priority": priority,
                "description": strategy_info.get("description", ""),
                "difficulty": strategy_info.get("difficulty", "未知"),
                "success_rate": strategy_info.get("success_rate", 0.0),
                "steps": strategy_info.get("steps", [])
            })
        
        # 按优先级排序
        result.sort(key=lambda x: x["priority"], reverse=True)
        return result
    
    def check_for_common_errors(self, calculation: str, result: str) -> List[Dict]:
        """检查常见错误"""
        errors = []
        
        # 检查计算错误
        try:
            # 简单的计算验证
            if "=" in calculation:
                left, right = calculation.split("=")
                left_eval = eval(left.strip())
                right_eval = eval(right.strip())
                if abs(left_eval - right_eval) > 0.01:
                    errors.append({
                        "type": "计算错误",
                        "description": "等式两边不相等",
                        "severity": "high"
                    })
        except:
            pass
        
        # 检查单位错误
        if any(unit in calculation for unit in ["米", "厘米", "千克", "克"]):
            if "米" in calculation and "厘米" in result:
                errors.append({
                    "type": "单位错误",
                    "description": "单位不匹配",
                    "severity": "medium"
                })
        
        return errors
    
    def convert_units(self, value: float, from_unit: str, to_unit: str, unit_type: str) -> Optional[float]:
        """单位转换"""
        conversions = self.unit_conversions.get(unit_type, {})
        if from_unit in conversions and to_unit in conversions[from_unit]:
            return value * conversions[from_unit][to_unit]
        return None
    
    def get_related_concepts(self, concept_name: str) -> List[str]:
        """获取相关概念"""
        related = []
        concept_info = self.get_concept_info(concept_name)
        if concept_info:
            # 基于概念属性找到相关概念
            for other_concept, other_info in self.concepts.items():
                if other_concept != concept_name:
                    # 检查是否有共同属性
                    common_properties = set(concept_info.get("properties", [])) & set(other_info.get("properties", []))
                    if common_properties:
                        related.append(other_concept)
        return related
    
    def validate_mathematical_expression(self, expression: str) -> Dict:
        """验证数学表达式的合理性"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 检查括号匹配
        if expression.count("(") != expression.count(")"):
            validation_result["is_valid"] = False
            validation_result["errors"].append("括号不匹配")
        
        # 检查除零
        if "/0" in expression or "÷0" in expression:
            validation_result["is_valid"] = False
            validation_result["errors"].append("除零错误")
        
        # 检查连续运算符
        if re.search(r'[+\-×÷*/]{2,}', expression):
            validation_result["warnings"].append("连续运算符可能表示错误")
        
        return validation_result


class MetaKnowledgeReasoning:
    """
    基于元知识的推理增强器
    """
    
    def __init__(self, meta_knowledge: MetaKnowledge):
        self.meta_knowledge = meta_knowledge
    
    def enhance_reasoning(self, problem_text: str, current_reasoning: List[Dict]) -> Dict:
        """使用元知识增强推理过程"""
        enhanced_reasoning = {
            "original_reasoning": current_reasoning,
            "meta_knowledge_enhancements": [],
            "suggested_strategies": [],
            "concept_analysis": {},
            "error_prevention": []
        }
        
        # 1. 概念分析
        concepts = self.meta_knowledge.identify_concepts_in_text(problem_text)
        enhanced_reasoning["concept_analysis"]["identified_concepts"] = concepts
        
        for concept in concepts:
            concept_info = self.meta_knowledge.get_concept_info(concept)
            if concept_info:
                enhanced_reasoning["concept_analysis"][concept] = {
                    "definition": concept_info["definition"],
                    "properties": concept_info["properties"],
                    "common_mistakes": concept_info["common_mistakes"]
                }
        
        # 2. 策略推荐
        strategies = self.meta_knowledge.suggest_strategies(problem_text)
        enhanced_reasoning["suggested_strategies"] = strategies
        
        for strategy in strategies:
            strategy_info = self.meta_knowledge.get_strategy_info(strategy)
            if strategy_info:
                enhanced_reasoning["meta_knowledge_enhancements"].append({
                    "type": "strategy_suggestion",
                    "strategy": strategy,
                    "description": strategy_info["description"],
                    "steps": strategy_info["steps"]
                })
        
        # 3. 错误预防
        for concept in concepts:
            concept_info = self.meta_knowledge.get_concept_info(concept)
            if concept_info and "common_mistakes" in concept_info:
                enhanced_reasoning["error_prevention"].extend([
                    {
                        "concept": concept,
                        "mistake": mistake,
                        "prevention": "注意检查"
                    }
                    for mistake in concept_info["common_mistakes"]
                ])
        
        # 4. 相关概念推荐
        for concept in concepts:
            related = self.meta_knowledge.get_related_concepts(concept)
            if related:
                enhanced_reasoning["meta_knowledge_enhancements"].append({
                    "type": "related_concepts",
                    "concept": concept,
                    "related": related
                })
        
        return enhanced_reasoning
    
    def validate_solution(self, problem_text: str, solution: str, calculation_steps: List[str]) -> Dict:
        """验证解决方案的合理性"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # 检查计算步骤
        for step in calculation_steps:
            errors = self.meta_knowledge.check_for_common_errors(step, solution)
            validation_result["errors"].extend(errors)
        
        # 检查概念一致性
        concepts = self.meta_knowledge.identify_concepts_in_text(problem_text)
        for concept in concepts:
            concept_info = self.meta_knowledge.get_concept_info(concept)
            if concept_info:
                # 检查是否违反了概念的基本性质
                for property_name in concept_info.get("properties", []):
                    if "不能为0" in property_name and "0" in solution:
                        validation_result["warnings"].append(f"{concept}的{property_name}被违反")
        
        # 检查单位一致性
        if any(unit in problem_text for unit in ["米", "厘米", "千克", "克"]):
            validation_result["suggestions"].append("检查单位转换是否正确")
        
        if validation_result["errors"]:
            validation_result["is_valid"] = False
        
        return validation_result 