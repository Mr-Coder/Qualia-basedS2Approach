"""
思维树推理策略 (Tree of Thoughts)
实现多路径探索的推理策略，适合复杂数学问题
"""

import heapq
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...core.exceptions import ReasoningError
from ...core.interfaces import ReasoningContext
from .strategy_base import (ReasoningStrategy, StrategyComplexity,
                            StrategyResult, StrategyType)


@dataclass
class ThoughtNode:
    """思维节点"""
    id: int
    content: str
    parent_id: Optional[int]
    depth: int
    score: float
    confidence: float
    reasoning_step: Dict[str, Any]
    children: List[int]
    is_solution: bool = False
    
    def __lt__(self, other):
        """用于优先队列排序"""
        return self.score > other.score  # 分数高的优先

class TreeOfThoughtsStrategy(ReasoningStrategy):
    """思维树推理策略"""
    
    def __init__(self):
        super().__init__(
            name="tree_of_thoughts",
            strategy_type=StrategyType.TREE_OF_THOUGHTS,
            complexity=StrategyComplexity.COMPLEX
        )
        
        # 树搜索参数
        self.max_depth = 6
        self.max_branches = 4
        self.beam_width = 8
        self.exploration_factor = 0.3
        
        # 节点管理
        self.nodes: Dict[int, ThoughtNode] = {}
        self.next_node_id = 0
        
        # 搜索状态
        self.frontier = []  # 优先队列
        self.explored = set()
        self.solutions = []
    
    def can_handle(self, problem_text: str, context: Optional[ReasoningContext] = None) -> bool:
        """
        判断是否能处理该问题
        
        思维树策略适合：
        - 复杂的多步骤问题
        - 需要探索多种解法的问题
        - 包含条件分支的问题
        """
        try:
            # 检查复杂度指标
            complexity_score = 0
            
            # 数字数量
            numbers = self._extract_numbers(problem_text)
            if len(numbers) >= 3:
                complexity_score += 1
            
            # 条件分支关键词
            branch_keywords = ["如果", "假设", "当", "分情况", "或者", "要么"]
            if any(keyword in problem_text for keyword in branch_keywords):
                complexity_score += 2
            
            # 多步骤关键词
            multi_step_keywords = ["首先", "然后", "接下来", "最后", "第一步", "第二步"]
            if any(keyword in problem_text for keyword in multi_step_keywords):
                complexity_score += 1
            
            # 问题长度
            if len(problem_text) > 150:
                complexity_score += 1
            
            # 复杂数学概念
            complex_concepts = ["方程", "函数", "几何", "概率", "统计", "微积分"]
            if any(concept in problem_text for concept in complex_concepts):
                complexity_score += 2
            
            # 需要至少3分才适合树形搜索
            return complexity_score >= 3
            
        except Exception as e:
            self.logger.error(f"能力判断失败: {str(e)}")
            return False
    
    def estimate_complexity(self, problem_text: str, context: Optional[ReasoningContext] = None) -> float:
        """估计问题复杂度"""
        try:
            complexity = 0.5  # 基础复杂度
            
            # 数字复杂度 (0.0-0.2)
            numbers = self._extract_numbers(problem_text)
            complexity += min(len(numbers) * 0.03, 0.2)
            
            # 条件分支复杂度 (0.0-0.3)
            branch_keywords = ["如果", "假设", "当", "分情况", "或者", "要么"]
            branch_count = sum(1 for keyword in branch_keywords if keyword in problem_text)
            complexity += min(branch_count * 0.1, 0.3)
            
            # 文本长度复杂度 (0.0-0.2)
            text_length = len(problem_text)
            if text_length > 300:
                complexity += 0.2
            elif text_length > 200:
                complexity += 0.15
            elif text_length > 100:
                complexity += 0.1
            
            # 数学概念复杂度 (0.0-0.3)
            complex_concepts = ["方程", "函数", "几何", "概率", "统计", "微积分", "矩阵"]
            concept_count = sum(1 for concept in complex_concepts if concept in problem_text)
            complexity += min(concept_count * 0.1, 0.3)
            
            return min(1.0, complexity)
            
        except Exception as e:
            self.logger.error(f"复杂度估计失败: {str(e)}")
            return 0.7  # 默认较高复杂度
    
    def _execute_reasoning(self, problem_text: str, context: Optional[ReasoningContext] = None) -> StrategyResult:
        """执行思维树推理"""
        start_time = time.time()
        
        try:
            # 初始化搜索状态
            self._initialize_search()
            
            # 创建根节点
            root_node = self._create_root_node(problem_text)
            
            # 执行树搜索
            search_result = self._tree_search(problem_text, root_node)
            
            # 选择最佳解决方案
            best_solution = self._select_best_solution()
            
            # 构建推理步骤
            reasoning_steps = self._build_reasoning_path(best_solution)
            
            # 计算置信度
            confidence = self._calculate_tree_confidence(best_solution)
            
            execution_time = time.time() - start_time
            
            return StrategyResult(
                success=True,
                answer=str(best_solution.reasoning_step.get("result", "未知")),
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                strategy_used=self.name,
                execution_time=execution_time,
                metadata={
                    "nodes_explored": len(self.nodes),
                    "solutions_found": len(self.solutions),
                    "max_depth_reached": max(node.depth for node in self.nodes.values()),
                    "search_complete": search_result["complete"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"思维树推理失败: {str(e)}")
            
            return StrategyResult(
                success=False,
                answer="推理失败",
                confidence=0.0,
                reasoning_steps=[{
                    "step": 1,
                    "action": "tree_search_error",
                    "description": f"思维树搜索失败: {str(e)}",
                    "confidence": 0.0
                }],
                strategy_used=self.name,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _initialize_search(self):
        """初始化搜索状态"""
        self.nodes.clear()
        self.next_node_id = 0
        self.frontier.clear()
        self.explored.clear()
        self.solutions.clear()
    
    def _create_root_node(self, problem_text: str) -> ThoughtNode:
        """创建根节点"""
        root_step = {
            "step": 1,
            "action": "problem_understanding",
            "description": f"理解问题: {problem_text[:100]}...",
            "content": problem_text,
            "confidence": 0.9
        }
        
        root_node = ThoughtNode(
            id=self.next_node_id,
            content=f"问题分析: {problem_text}",
            parent_id=None,
            depth=0,
            score=0.9,
            confidence=0.9,
            reasoning_step=root_step,
            children=[]
        )
        
        self.nodes[self.next_node_id] = root_node
        heapq.heappush(self.frontier, root_node)
        self.next_node_id += 1
        
        return root_node
    
    def _tree_search(self, problem_text: str, root_node: ThoughtNode) -> Dict[str, Any]:
        """执行树搜索"""
        search_stats = {
            "nodes_expanded": 0,
            "solutions_found": 0,
            "complete": False
        }
        
        while self.frontier and search_stats["nodes_expanded"] < 50:  # 限制搜索规模
            # 获取最佳节点
            current_node = heapq.heappop(self.frontier)
            
            if current_node.id in self.explored:
                continue
            
            self.explored.add(current_node.id)
            search_stats["nodes_expanded"] += 1
            
            # 检查是否为解决方案
            if self._is_solution_node(current_node, problem_text):
                current_node.is_solution = True
                self.solutions.append(current_node)
                search_stats["solutions_found"] += 1
                
                # 如果找到足够好的解决方案，可以提前结束
                if current_node.confidence > 0.9:
                    search_stats["complete"] = True
                    break
            
            # 扩展节点
            if current_node.depth < self.max_depth:
                children = self._expand_node(current_node, problem_text)
                
                for child in children:
                    if child.id not in self.explored:
                        heapq.heappush(self.frontier, child)
        
        if not self.solutions:
            search_stats["complete"] = False
        
        return search_stats
    
    def _expand_node(self, node: ThoughtNode, problem_text: str) -> List[ThoughtNode]:
        """扩展节点，生成子节点"""
        children = []
        
        try:
            # 根据节点深度和内容生成不同类型的子节点
            if node.depth == 0:
                # 根节点：生成问题分解子节点
                children.extend(self._generate_decomposition_nodes(node, problem_text))
            elif node.depth == 1:
                # 第一层：生成解题方法子节点
                children.extend(self._generate_method_nodes(node, problem_text))
            elif node.depth == 2:
                # 第二层：生成具体计算子节点
                children.extend(self._generate_calculation_nodes(node, problem_text))
            else:
                # 深层：生成验证和细化子节点
                children.extend(self._generate_refinement_nodes(node, problem_text))
            
            # 限制每个节点的子节点数量
            children = children[:self.max_branches]
            
            # 更新父节点的子节点列表
            for child in children:
                node.children.append(child.id)
            
        except Exception as e:
            self.logger.error(f"节点扩展失败: {str(e)}")
        
        return children
    
    def _generate_decomposition_nodes(self, parent: ThoughtNode, problem_text: str) -> List[ThoughtNode]:
        """生成问题分解子节点"""
        children = []
        
        # 数字提取分解
        numbers = self._extract_numbers(problem_text)
        if numbers:
            child = self._create_child_node(
                parent,
                f"提取数字: {numbers}",
                "number_extraction",
                f"从问题中提取到数字: {numbers}",
                {"numbers": numbers},
                0.9
            )
            children.append(child)
        
        # 关键词分析分解
        keywords = self._extract_keywords(problem_text)
        if keywords:
            child = self._create_child_node(
                parent,
                f"关键词分析: {keywords}",
                "keyword_analysis",
                f"识别关键词: {', '.join(keywords)}",
                {"keywords": keywords},
                0.8
            )
            children.append(child)
        
        # 问题类型识别
        problem_type = self._identify_problem_type(problem_text)
        child = self._create_child_node(
            parent,
            f"问题类型: {problem_type}",
            "type_identification",
            f"识别问题类型为: {problem_type}",
            {"problem_type": problem_type},
            0.85
        )
        children.append(child)
        
        return children
    
    def _generate_method_nodes(self, parent: ThoughtNode, problem_text: str) -> List[ThoughtNode]:
        """生成解题方法子节点"""
        children = []
        
        # 直接计算方法
        child = self._create_child_node(
            parent,
            "直接计算方法",
            "direct_calculation",
            "使用直接计算方法求解",
            {"method": "direct"},
            0.8
        )
        children.append(child)
        
        # 方程求解方法
        if any(keyword in problem_text for keyword in ["方程", "等式", "="]):
            child = self._create_child_node(
                parent,
                "方程求解方法",
                "equation_solving",
                "建立方程求解",
                {"method": "equation"},
                0.9
            )
            children.append(child)
        
        # 比例计算方法
        if any(keyword in problem_text for keyword in ["比例", "比率", "百分比", "%"]):
            child = self._create_child_node(
                parent,
                "比例计算方法",
                "proportion_calculation",
                "使用比例关系计算",
                {"method": "proportion"},
                0.85
            )
            children.append(child)
        
        return children
    
    def _generate_calculation_nodes(self, parent: ThoughtNode, problem_text: str) -> List[ThoughtNode]:
        """生成具体计算子节点"""
        children = []
        
        parent_data = parent.reasoning_step.get("data", {})
        numbers = parent_data.get("numbers", self._extract_numbers(problem_text))
        
        if len(numbers) >= 2:
            # 加法计算
            result = sum(numbers)
            child = self._create_child_node(
                parent,
                f"加法: {' + '.join(map(str, numbers))} = {result}",
                "addition",
                f"计算总和: {result}",
                {"operation": "add", "operands": numbers, "result": result},
                0.9
            )
            children.append(child)
            
            # 减法计算
            result = numbers[0] - sum(numbers[1:])
            child = self._create_child_node(
                parent,
                f"减法: {numbers[0]} - {sum(numbers[1:])} = {result}",
                "subtraction",
                f"计算差值: {result}",
                {"operation": "subtract", "operands": numbers, "result": result},
                0.9
            )
            children.append(child)
            
            # 乘法计算
            if len(numbers) == 2:
                result = numbers[0] * numbers[1]
                child = self._create_child_node(
                    parent,
                    f"乘法: {numbers[0]} × {numbers[1]} = {result}",
                    "multiplication",
                    f"计算乘积: {result}",
                    {"operation": "multiply", "operands": numbers, "result": result},
                    0.9
                )
                children.append(child)
        
        return children
    
    def _generate_refinement_nodes(self, parent: ThoughtNode, problem_text: str) -> List[ThoughtNode]:
        """生成验证和细化子节点"""
        children = []
        
        parent_data = parent.reasoning_step.get("data", {})
        result = parent_data.get("result")
        
        if result is not None:
            # 答案验证
            validation = self._validate_result(result, problem_text)
            child = self._create_child_node(
                parent,
                f"验证结果: {validation['status']}",
                "validation",
                f"答案验证: {validation['description']}",
                {"validation": validation, "final_result": result},
                validation.get("confidence", 0.7)
            )
            children.append(child)
        
        return children
    
    def _create_child_node(self, parent: ThoughtNode, content: str, action: str, 
                          description: str, data: Dict[str, Any], confidence: float) -> ThoughtNode:
        """创建子节点"""
        reasoning_step = {
            "step": parent.depth + 2,
            "action": action,
            "description": description,
            "data": data,
            "confidence": confidence
        }
        
        # 计算节点分数 (结合置信度和深度)
        score = confidence * (1 - parent.depth * 0.1)
        
        child = ThoughtNode(
            id=self.next_node_id,
            content=content,
            parent_id=parent.id,
            depth=parent.depth + 1,
            score=score,
            confidence=confidence,
            reasoning_step=reasoning_step,
            children=[]
        )
        
        self.nodes[self.next_node_id] = child
        self.next_node_id += 1
        
        return child
    
    def _is_solution_node(self, node: ThoughtNode, problem_text: str) -> bool:
        """判断节点是否为解决方案"""
        step_data = node.reasoning_step.get("data", {})
        
        # 如果节点包含最终结果且通过验证
        if "final_result" in step_data:
            validation = step_data.get("validation", {})
            return validation.get("status") == "valid"
        
        # 如果节点包含计算结果且深度足够
        if "result" in step_data and node.depth >= 3:
            return True
        
        return False
    
    def _select_best_solution(self) -> ThoughtNode:
        """选择最佳解决方案"""
        if not self.solutions:
            # 如果没有明确的解决方案，选择分数最高的叶子节点
            leaf_nodes = [node for node in self.nodes.values() 
                         if not node.children and node.depth > 0]
            if leaf_nodes:
                return max(leaf_nodes, key=lambda x: x.score)
            else:
                # 返回根节点作为最后的选择
                return list(self.nodes.values())[0]
        
        # 选择置信度最高的解决方案
        return max(self.solutions, key=lambda x: x.confidence)
    
    def _build_reasoning_path(self, solution_node: ThoughtNode) -> List[Dict[str, Any]]:
        """构建从根到解决方案的推理路径"""
        path = []
        current = solution_node
        
        # 从解决方案回溯到根节点
        while current is not None:
            path.append(current.reasoning_step)
            current = self.nodes.get(current.parent_id) if current.parent_id is not None else None
        
        # 反转路径，使其从根到解决方案
        path.reverse()
        
        return path
    
    def _calculate_tree_confidence(self, solution_node: ThoughtNode) -> float:
        """计算树搜索的整体置信度"""
        # 获取路径上所有节点的置信度
        path_confidences = []
        current = solution_node
        
        while current is not None:
            path_confidences.append(current.confidence)
            current = self.nodes.get(current.parent_id) if current.parent_id is not None else None
        
        if not path_confidences:
            return 0.0
        
        # 使用几何平均计算整体置信度
        product = 1.0
        for conf in path_confidences:
            product *= conf
        
        return product ** (1.0 / len(path_confidences))
    
    def _extract_numbers(self, text: str) -> List[float]:
        """提取数字"""
        import re
        pattern = r'\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = ["加", "减", "乘", "除", "总共", "一共", "剩余", "平均", "每", "比", "面积", "周长"]
        found = [kw for kw in keywords if kw in text]
        return found
    
    def _identify_problem_type(self, text: str) -> str:
        """识别问题类型"""
        if any(kw in text for kw in ["面积", "周长", "体积"]):
            return "geometry"
        elif any(kw in text for kw in ["速度", "时间", "距离"]):
            return "motion"
        elif any(kw in text for kw in ["价格", "成本", "利润"]):
            return "economics"
        elif any(kw in text for kw in ["比例", "百分比", "%"]):
            return "proportion"
        else:
            return "arithmetic"
    
    def _validate_result(self, result: float, problem_text: str) -> Dict[str, Any]:
        """验证计算结果"""
        validation = {
            "status": "valid",
            "confidence": 0.8,
            "description": "结果验证通过"
        }
        
        try:
            # 基本数值检查
            if not isinstance(result, (int, float)):
                validation["status"] = "invalid"
                validation["confidence"] = 0.0
                validation["description"] = "结果不是数字"
                return validation
            
            # 合理性检查
            if abs(result) > 1e10:
                validation["status"] = "suspicious"
                validation["confidence"] = 0.3
                validation["description"] = "结果过大，可能不合理"
            elif result < 0 and any(kw in problem_text for kw in ["数量", "个数", "长度", "面积"]):
                validation["status"] = "suspicious"
                validation["confidence"] = 0.4
                validation["description"] = "数量类问题结果为负数"
            
        except Exception as e:
            validation["status"] = "error"
            validation["confidence"] = 0.0
            validation["description"] = f"验证出错: {str(e)}"
        
        return validation 