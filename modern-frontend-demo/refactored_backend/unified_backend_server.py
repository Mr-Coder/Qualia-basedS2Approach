#!/usr/bin/env python3
"""
统一后端服务器
整合QS²+IRD+COT-DIR完整推理流程的后端服务
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入所有推理模块
from problem_preprocessor import ProblemPreprocessor
from qs2_semantic_analyzer import QS2SemanticAnalyzer  
from ird_relation_discovery import IRDRelationDiscovery
from reasoning_engine_selector import ReasoningEngineSelector, ReasoningRequest
from cotdir_reasoning_chain import COTDIRReasoningChain
from result_generator import ResultGenerator, ResultEnhancer
from physical_property_graph import PhysicalPropertyGraphBuilder
from activation_diffusion_engine import ActivationDiffusionEngine
from enhanced_math_solver import EnhancedMathSolver

class ActivationLearningEngine:
    """基于激活扩散理论的学习指导引擎"""
    
    def __init__(self):
        self.learning_network = {
            'basic_arithmetic': {
                'activation_threshold': 0.3,
                'connected_concepts': ['addition', 'subtraction', 'number_recognition'],
                'difficulty': 'beginner',
                'estimated_time': '30-45分钟'
            },
            'entity_recognition': {
                'activation_threshold': 0.4,
                'connected_concepts': ['pattern_recognition', 'semantic_understanding'],
                'difficulty': 'beginner',
                'estimated_time': '45-60分钟'
            },
            'relationship_analysis': {
                'activation_threshold': 0.5,
                'connected_concepts': ['logical_reasoning', 'graph_theory'],
                'difficulty': 'intermediate',
                'estimated_time': '60-75分钟'
            },
            'strategy_selection': {
                'activation_threshold': 0.6,
                'connected_concepts': ['meta_cognition', 'decision_making'],
                'difficulty': 'intermediate',
                'estimated_time': '75-90分钟'
            },
            'deep_reasoning': {
                'activation_threshold': 0.7,
                'connected_concepts': ['complex_analysis', 'multi_step_thinking'],
                'difficulty': 'advanced',
                'estimated_time': '90-120分钟'
            }
        }
        
    def get_personalized_learning_paths(self, user_level: str, learning_goal: str) -> List[Dict[str, Any]]:
        """获取个性化学习路径"""
        paths = []
        
        if user_level == 'beginner':
            paths.append({
                'id': 'basic_activation',
                'title': '基础算术激活路径',
                'description': '通过激活扩散理论学习基础数学运算',
                'estimatedTime': '2-3小时',
                'difficulty': 'beginner',
                'stages': 4,
                'icon': '🧮',
                'activation_pattern': 'sequential',
                'recommended_for': '初学者和基础薄弱的学习者'
            })
        
        paths.append({
            'id': 'advanced_reasoning',
            'title': '高级推理激活路径',
            'description': '通过函数式思维和网络思维进行复杂推理',
            'estimatedTime': '4-5小时',
            'difficulty': 'advanced',
            'stages': 3,
            'icon': '🧠',
            'activation_pattern': 'parallel',
            'recommended_for': '有一定基础的学习者'
        })
        
        return paths
    
    def get_activation_based_techniques(self, current_problem: str = None) -> List[Dict[str, Any]]:
        """获取基于激活扩散的学习技巧"""
        techniques = [
            {
                'category': '激活扩散识别技巧',
                'icon': '🔍',
                'color': 'blue',
                'techniques': [
                    '通过关键词激活相关概念网络',
                    '利用语义相似性发现隐含实体',
                    '使用激活强度判断实体重要性',
                    '通过激活路径追踪实体关系'
                ],
                'activation_methods': [
                    '概念激活：从已知概念出发激活相关实体',
                    '语义激活：通过语义相似性扩散激活',
                    '结构激活：利用问题结构激活对应实体',
                    '上下文激活：基于问题上下文激活实体'
                ]
            },
            {
                'category': '网络化关系理解方法',
                'icon': '🕸️',
                'color': 'green',
                'techniques': [
                    '构建激活扩散的关系网络',
                    '通过激活强度评估关系重要性',
                    '利用激活路径发现隐式关系',
                    '基于激活模式识别关系类型'
                ],
                'activation_methods': [
                    '双向激活：同时激活关系的两端实体',
                    '层次激活：按关系层次逐步激活',
                    '聚类激活：激活相似关系的集合',
                    '路径激活：沿关系路径扩散激活'
                ]
            }
        ]
        
        return techniques
    
    def generate_learning_insights(self, problem: str, solution_data: Dict[str, Any]) -> Dict[str, Any]:
        """基于解题过程生成学习洞察"""
        insights = {
            'activated_concepts': [],
            'learning_opportunities': [],
            'difficulty_analysis': {},
            'recommended_practice': [],
            'activation_visualization': {}
        }
        
        # 分析激活的概念
        if 'entities' in solution_data:
            insights['activated_concepts'] = [
                f"实体识别激活了{len(solution_data['entities'])}个概念节点"
            ]
            
        if 'relationships' in solution_data:
            insights['activated_concepts'].append(
                f"关系发现激活了{len(solution_data['relationships'])}个连接路径"
            )
        
        # 学习机会分析
        insights['learning_opportunities'] = [
            "可以通过相似问题练习强化激活模式",
            "建议学习相关概念以扩展激活网络",
            "尝试不同策略以激活多样化的推理路径"
        ]
        
        # 难度分析
        entity_count = len(solution_data.get('entities', []))
        relation_count = len(solution_data.get('relationships', []))
        
        if entity_count <= 3 and relation_count <= 2:
            difficulty_level = 'beginner'
            difficulty_desc = '基础问题，适合激活扩散入门练习'
        elif entity_count <= 6 and relation_count <= 5:
            difficulty_level = 'intermediate'  
            difficulty_desc = '中等问题，需要激活多个概念网络'
        else:
            difficulty_level = 'advanced'
            difficulty_desc = '复杂问题，需要激活复杂的关系网络'
            
        insights['difficulty_analysis'] = {
            'level': difficulty_level,
            'description': difficulty_desc,
            'entity_complexity': entity_count,
            'relation_complexity': relation_count
        }
        
        return insights

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic数据模型
class ProblemRequest(BaseModel):
    problem: str
    mode: Optional[str] = "auto"  # auto, simple, advanced
    preferences: Optional[Dict[str, Any]] = {}

class LearningPathRequest(BaseModel):
    user_level: str  # beginner, intermediate, advanced
    learning_goal: str  # specific learning objective
    preferences: Optional[Dict[str, Any]] = {}

class LearningStageRequest(BaseModel):
    stage_id: int
    user_progress: Dict[int, str]  # stage_id -> status

class ActivationLearningResponse(BaseModel):
    recommended_paths: List[Dict[str, Any]]
    personalized_stages: List[Dict[str, Any]]
    activation_based_techniques: List[Dict[str, Any]]
    learning_network_state: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    modules: Dict[str, str]

class SolveResponse(BaseModel):
    success: bool
    answer: str
    confidence: float
    strategy_used: str
    execution_time: float
    algorithm_type: str
    reasoning_steps: List[Dict[str, Any]]
    entity_relationship_diagram: Dict[str, Any]
    metadata: Dict[str, Any]
    learning_insights: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# 创建FastAPI应用
app = FastAPI(
    title="QS²+IRD+COT-DIR统一推理后端",
    description="基于量子语义学和隐式关系发现的数学推理系统",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class UnifiedReasoningSystem:
    """统一推理系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化所有模块
        self.logger.info("初始化推理系统模块...")
        
        self.preprocessor = ProblemPreprocessor()
        self.qs2_analyzer = QS2SemanticAnalyzer()
        self.ird_discovery = IRDRelationDiscovery(self.qs2_analyzer)
        self.engine_selector = ReasoningEngineSelector()
        self.cotdir_chain = COTDIRReasoningChain()
        self.result_generator = ResultGenerator()
        self.result_enhancer = ResultEnhancer()
        self.property_graph_builder = PhysicalPropertyGraphBuilder()
        # 🧠 激活扩散推理引擎 - 基于交互式物性图谱的核心优化
        self.activation_engine = ActivationDiffusionEngine()
        # 🔧 增强数学求解器 - 真正能解题的数学推理引擎
        self.enhanced_math_solver = EnhancedMathSolver()
        # 📚 激活扩散学习引擎 - 基于激活扩散理论的学习指导
        self.learning_engine = ActivationLearningEngine()
        
        # 系统状态
        self.system_status = {
            "initialized": True,
            "start_time": time.time(),
            "requests_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
        
        self.logger.info("推理系统初始化完成")

    async def solve_problem(self, problem_text: str, mode: str = "auto", 
                          preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        解决数学问题的完整流程
        
        Args:
            problem_text: 问题文本
            mode: 推理模式 (auto, simple, advanced)
            preferences: 用户偏好设置
            
        Returns:
            Dict: 标准化结果
        """
        start_time = time.time()
        self.system_status["requests_processed"] += 1
        
        try:
            self.logger.info(f"开始处理问题: {problem_text[:50]}...")
            
            # Step 1: 问题预处理
            self.logger.debug("Step 1: 问题预处理")
            processed_problem = self.preprocessor.preprocess(problem_text)
            
            # Step 2: QS²语义分析
            self.logger.debug("Step 2: QS²语义分析")
            semantic_entities = self.qs2_analyzer.analyze_semantics(processed_problem)
            
            # Step 3: IRD隐式关系发现
            self.logger.debug("Step 3: IRD隐式关系发现")
            relation_network = self.ird_discovery.discover_relations(semantic_entities, problem_text)
            
            # Step 3.5: 🧠 激活扩散分析 - 基于交互式物性图谱的智能激活
            self.logger.debug("Step 3.5: 激活扩散分析")
            activation_result = self.activation_engine.activate_nodes_from_problem(
                problem_text, semantic_entities
            )
            activated_reasoning_path = self.activation_engine.get_activated_reasoning_path()
            network_state = self.activation_engine.get_network_state()
            
            # Step 3.6: 物性图谱构建（集成激活状态）
            self.logger.debug("Step 3.6: 物性图谱构建")
            property_graph = self.property_graph_builder.build_property_graph(
                processed_problem, semantic_entities, relation_network, 
                activation_state=network_state
            )
            
            # Step 4: 推理引擎选择与执行
            self.logger.debug("Step 4: 推理引擎选择")
            
            # 设置推理模式
            if mode == "simple":
                from reasoning_engine_selector import ReasoningMode
                self.engine_selector.set_mode(ReasoningMode.SIMPLE)
            elif mode == "advanced":
                from reasoning_engine_selector import ReasoningMode
                self.engine_selector.set_mode(ReasoningMode.ADVANCED)
            else:
                from reasoning_engine_selector import ReasoningMode
                self.engine_selector.set_mode(ReasoningMode.AUTO)
            
            # 创建推理请求（集成激活扩散信息）
            reasoning_request = ReasoningRequest(
                processed_problem=processed_problem,
                semantic_entities=semantic_entities,
                relation_network=relation_network,
                user_preferences=preferences or {},
                context=problem_text,
                # 🧠 激活扩散增强信息
                activation_result=activation_result,
                reasoning_path=activated_reasoning_path,
                network_state=network_state
            )
            
            # 执行推理
            reasoning_result = self.engine_selector.execute_reasoning(reasoning_request)
            
            # Step 5: COT-DIR推理链构建（如果使用高级引擎，集成激活扩散）
            if reasoning_result.get("strategy_used") in ["qs2_ird_cotdir", "advanced"]:
                self.logger.debug("Step 5: COT-DIR推理链构建（激活扩散增强）")
                reasoning_chain = self.cotdir_chain.build_reasoning_chain(
                    processed_problem, semantic_entities, relation_network,
                    activation_info={
                        "activations": activation_result,
                        "reasoning_path": activated_reasoning_path,
                        "network_state": network_state
                    }
                )
                
                # Step 6: 结果生成与增强（含物性图谱）
                self.logger.debug("Step 6: 结果生成")
                standard_result = self.result_generator.generate_standard_result(
                    reasoning_chain, semantic_entities, relation_network, problem_text, property_graph
                )
                
                # 增强结果为前端格式
                enhanced_result = self.result_enhancer.enhance_for_frontend(standard_result)
                
            else:
                # 简单引擎结果转换
                standard_result = self._convert_simple_result(reasoning_result, problem_text)
                enhanced_result = self.result_enhancer.enhance_for_frontend(standard_result)
            
            execution_time = time.time() - start_time
            enhanced_result["execution_time"] = execution_time
            
            # 添加前端需要的基础字段
            if "reasoning_steps" in enhanced_result:
                enhanced_result["steps"] = [step["description"] for step in enhanced_result["reasoning_steps"]]
            enhanced_result["processingTime"] = execution_time
            
            # 🎓 生成学习洞察
            try:
                learning_insights = self.learning_engine.generate_learning_insights(
                    problem_text, enhanced_result
                )
                enhanced_result["learning_insights"] = learning_insights
            except Exception as le:
                self.logger.warning(f"生成学习洞察失败: {le}")
                enhanced_result["learning_insights"] = None
            
            self.system_status["successful_requests"] += 1
            
            self.logger.info(f"问题处理完成，耗时: {execution_time:.3f}s")
            return enhanced_result
            
        except Exception as e:
            self.system_status["failed_requests"] += 1
            self.logger.error(f"问题处理失败: {e}")
            
            return {
                "success": False,
                "answer": "推理失败",
                "confidence": 0.0,
                "strategy_used": "error_handler",
                "execution_time": time.time() - start_time,
                "algorithm_type": "QS2_Enhanced_Unified",
                "reasoning_steps": [
                    {
                        "step": 1,
                        "action": "错误处理",
                        "description": f"系统处理出现错误: {str(e)}",
                        "confidence": 0.0
                    }
                ],
                "entity_relationship_diagram": {
                    "entities": [],
                    "relationships": [],
                    "implicit_constraints": [],
                    "qs2_enhancements": {}
                },
                "metadata": {
                    "engine_used": "error_handler",
                    "error_occurred": True,
                    "original_problem": problem_text
                },
                "error": str(e)
            }

    def _convert_simple_result(self, reasoning_result: Dict[str, Any], 
                             problem_text: str) -> Dict[str, Any]:
        """将简单引擎结果转换为标准格式"""
        
        return {
            "success": reasoning_result.get("success", False),
            "answer": reasoning_result.get("answer", "计算失败"),
            "confidence": reasoning_result.get("confidence", 0.0),
            "strategy_used": reasoning_result.get("strategy_used", "simple_arithmetic"),
            "execution_time": reasoning_result.get("execution_time", 0.0),
            "algorithm_type": "Simple_Arithmetic",
            "reasoning_steps": reasoning_result.get("reasoning_steps", []),
            "entity_relationship_diagram": reasoning_result.get("entity_relationship_diagram", {}),
            "metadata": {
                "engine_used": "simple_engine",
                "processing_mode": "simplified",
                "original_problem": problem_text
            },
            "error": reasoning_result.get("error")
        }

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        
        uptime = time.time() - self.system_status["start_time"]
        success_rate = (
            self.system_status["successful_requests"] / 
            max(self.system_status["requests_processed"], 1)
        )
        
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "requests_processed": self.system_status["requests_processed"],
            "success_rate": success_rate,
            "engine_status": self.engine_selector.get_engine_status(),
            "modules": {
                "preprocessor": "active",
                "qs2_analyzer": "active", 
                "ird_discovery": "active",
                "engine_selector": "active",
                "cotdir_chain": "active",
                "result_generator": "active",
                "property_graph_builder": "active"
            }
        }

# 创建全局推理系统实例
reasoning_system = UnifiedReasoningSystem()

# API路由定义
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    status = reasoning_system.get_system_status()
    
    return HealthResponse(
        status=status["status"],
        timestamp=time.time(),
        version="2.0.0",
        modules=status["modules"]
    )

@app.post("/api/solve", response_model=SolveResponse)
async def solve_problem(request: ProblemRequest):
    """问题求解接口"""
    
    if not request.problem or not request.problem.strip():
        raise HTTPException(status_code=400, detail="问题文本不能为空")
    
    try:
        result = await reasoning_system.solve_problem(
            problem_text=request.problem,
            mode=request.mode,
            preferences=request.preferences
        )
        
        return SolveResponse(**result)
        
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/api/status")
async def get_system_status():
    """获取系统状态"""
    return reasoning_system.get_system_status()

@app.get("/api/qs2/demo")
async def qs2_demo():
    """QS²算法演示接口"""
    
    demo_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    try:
        result = await reasoning_system.solve_problem(
            problem_text=demo_problem,
            mode="advanced"
        )
        
        return {
            "demo_problem": demo_problem,
            "result": result,
            "algorithm_features": {
                "qs2_semantic_analysis": "基于Qualia理论的四维语义分析",
                "ird_relation_discovery": "三层关系发现：直接、上下文、传递",
                "cotdir_reasoning": "链式推理与隐式推理结合",
                "unified_architecture": "模块化推理引擎架构"
            }
        }
        
    except Exception as e:
        logger.error(f"QS²演示失败: {e}")
        return {
            "demo_problem": demo_problem,
            "error": str(e),
            "algorithm_features": {
                "error": "演示暂不可用"
            }
        }

@app.get("/api/algorithm/execution")
async def get_algorithm_execution():
    """获取算法执行信息"""
    
    return {
        "algorithm_name": "QS²+IRD+COT-DIR统一推理算法",
        "version": "2.0.0",
        "components": [
            {
                "name": "问题预处理器",
                "status": "active",
                "description": "文本清理、实体提取、复杂度评估"
            },
            {
                "name": "QS²语义分析器", 
                "status": "active",
                "description": "基于Qualia理论的四维语义结构分析"
            },
            {
                "name": "IRD关系发现器",
                "status": "active", 
                "description": "三层隐式关系发现算法"
            },
            {
                "name": "推理引擎选择器",
                "status": "active",
                "description": "智能引擎选择与fallback机制"
            },
            {
                "name": "COT-DIR推理链",
                "status": "active",
                "description": "六步推理链构建与执行"
            },
            {
                "name": "结果生成器",
                "status": "active",
                "description": "标准化结果生成与前端增强"
            },
            {
                "name": "物性图谱构建器",
                "status": "active", 
                "description": "基于物理属性和约束的推理图谱构建"
            }
        ],
        "execution_stats": reasoning_system.get_system_status()
    }

@app.get("/api/algorithm/execution/history")
async def get_execution_history(limit: int = 20):
    """获取执行历史"""
    
    # 模拟历史数据
    history = []
    for i in range(min(limit, 10)):
        history.append({
            "id": f"exec_{i+1}",
            "timestamp": time.time() - i * 300,  # 5分钟间隔
            "problem": f"示例问题 {i+1}",
            "result": "8个",
            "confidence": 0.95 - i * 0.02,
            "execution_time": 1.2 + i * 0.1,
            "strategy": "qs2_ird_cotdir"
        })
    
    return {
        "total_executions": reasoning_system.system_status["requests_processed"],
        "success_rate": (
            reasoning_system.system_status["successful_requests"] / 
            max(reasoning_system.system_status["requests_processed"], 1)
        ),
        "recent_history": history
    }

@app.get("/api/physical-property/demo")
async def physical_property_demo():
    """物性图谱演示接口"""
    
    demo_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    try:
        # 执行物性图谱分析
        processed = reasoning_system.preprocessor.preprocess(demo_problem)
        semantic_entities = reasoning_system.qs2_analyzer.analyze_semantics(processed)
        relation_network = reasoning_system.ird_discovery.discover_relations(semantic_entities, demo_problem)
        property_graph = reasoning_system.property_graph_builder.build_property_graph(
            processed, semantic_entities, relation_network
        )
        
        # 提取物性图谱信息
        physical_analysis = {
            "problem": demo_problem,
            "physical_properties": [
                {
                    "id": prop.property_id,
                    "type": prop.property_type.value,
                    "entity": prop.entity_id,
                    "value": prop.value,
                    "unit": prop.unit,
                    "certainty": prop.certainty,
                    "constraints": prop.constraints
                } for prop in property_graph.properties
            ],
            "physical_constraints": [
                {
                    "id": constraint.constraint_id,
                    "type": constraint.constraint_type.value,
                    "description": constraint.description,
                    "expression": constraint.mathematical_expression,
                    "strength": constraint.strength,
                    "entities": constraint.involved_entities
                } for constraint in property_graph.constraints
            ],
            "physical_relations": [
                {
                    "id": relation.relation_id,
                    "source": relation.source_entity_id,
                    "target": relation.target_entity_id,
                    "type": relation.relation_type,
                    "physical_basis": relation.physical_basis,
                    "strength": relation.strength,
                    "causal_direction": relation.causal_direction
                } for relation in property_graph.relations
            ],
            "graph_metrics": property_graph.graph_metrics,
            "consistency_score": property_graph.consistency_score
        }
        
        return {
            "status": "success",
            "demo_type": "physical_property_graph",
            "analysis": physical_analysis,
            "backend_driven_features": {
                "physical_property_analysis": "自动识别实体的物理属性类型",
                "constraint_discovery": "基于物理原理发现约束关系",
                "consistency_checking": "物性一致性验证机制",
                "causal_modeling": "因果关系方向推理",
                "constraint_propagation": "约束传播和冲突检测"
            },
            "frontend_optimization": {
                "property_visualization": "物理属性的类型化可视化",
                "constraint_overlay": "约束关系的图形化展示",
                "consistency_indicator": "一致性得分的实时显示",
                "causal_arrows": "因果方向的动态箭头",
                "interactive_exploration": "可交互的物性探索界面"
            }
        }
        
    except Exception as e:
        logger.error(f"物性图谱演示失败: {e}")
        return {
            "status": "error",
            "demo_type": "physical_property_graph",
            "error": str(e),
            "fallback_info": {
                "description": "物性图谱模块基于物理属性和约束关系构建推理图谱",
                "key_features": [
                    "物理属性自动识别",
                    "约束关系发现",
                    "物性一致性检查",
                    "因果关系建模"
                ]
            }
        }

@app.get("/api/qs2/relations")
async def get_qs2_relations():
    """QS²关系发现演示接口"""
    
    demo_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    try:
        # 执行QS²+IRD完整分析
        processed = reasoning_system.preprocessor.preprocess(demo_problem)
        semantic_entities = reasoning_system.qs2_analyzer.analyze_semantics(processed)
        relation_network = reasoning_system.ird_discovery.discover_relations(semantic_entities, demo_problem)
        property_graph = reasoning_system.property_graph_builder.build_property_graph(
            processed, semantic_entities, relation_network
        )
        
        return {
            "problem": demo_problem,
            "semantic_entities": [
                {
                    "id": entity.entity_id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "confidence": entity.confidence,
                    "qualia": {
                        "formal": entity.qualia.formal[:3],
                        "telic": entity.qualia.telic[:3],
                        "agentive": entity.qualia.agentive[:3], 
                        "constitutive": entity.qualia.constitutive[:3]
                    }
                } for entity in semantic_entities
            ],
            "discovered_relations": [
                {
                    "id": rel.relation_id,
                    "source": rel.source_entity_id,
                    "target": rel.target_entity_id,
                    "type": rel.relation_type,
                    "strength": rel.strength,
                    "confidence": rel.confidence,
                    "discovery_method": rel.discovery_method,
                    "evidence": rel.evidence[:2]
                } for rel in relation_network.relations
            ],
            "physical_relations": [
                {
                    "id": rel.relation_id,
                    "source": rel.source_entity_id,
                    "target": rel.target_entity_id,
                    "type": rel.relation_type,
                    "physical_basis": rel.physical_basis,
                    "strength": rel.strength,
                    "causal_direction": rel.causal_direction
                } for rel in property_graph.relations
            ],
            "network_metrics": relation_network.network_metrics,
            "physical_graph_metrics": property_graph.graph_metrics,
            "consistency_score": property_graph.consistency_score,
            "algorithm_info": {
                "qs2_analysis": "四维语义结构分析(Qualia Structure)",
                "ird_discovery": "三层隐式关系发现算法",
                "physical_modeling": "物性图谱关系建模",
                "relation_types": ["semantic", "contextual", "transitive", "physical"],
                "discovery_methods": ["direct_semantic", "context_based", "transitive_inference", "physical_principle"]
            }
        }
        
    except Exception as e:
        logger.error(f"QS²关系发现演示失败: {e}")
        return {
            "problem": demo_problem,
            "error": str(e),
            "algorithm_info": {
                "description": "QS²+IRD算法专门用于发现实体间的隐式关系",
                "features": [
                    "四维语义结构分析",
                    "三层关系发现机制", 
                    "物理约束建模",
                    "关系强度评估"
                ]
            }
        }

@app.get("/api/activation/diffusion")
async def get_activation_diffusion():
    """🧠 激活扩散分析接口 - 基于交互式物性图谱的智能激活"""
    
    demo_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    try:
        # 执行激活扩散分析
        processed = reasoning_system.preprocessor.preprocess(demo_problem)
        semantic_entities = reasoning_system.qs2_analyzer.analyze_semantics(processed)
        
        # 🧠 激活扩散核心分析
        activation_result = reasoning_system.activation_engine.activate_nodes_from_problem(
            demo_problem, semantic_entities
        )
        activated_reasoning_path = reasoning_system.activation_engine.get_activated_reasoning_path()
        network_state = reasoning_system.activation_engine.get_network_state()
        
        # 生成前端可视化数据
        activation_analysis = {
            "problem": demo_problem,
            "activation_result": activation_result,
            "activated_nodes": [
                {
                    "node_id": step["node_id"],
                    "node_name": step["node_name"], 
                    "node_type": step["node_type"],
                    "activation_level": step["activation_level"],
                    "activation_state": step["activation_state"],
                    "reasoning": step["reasoning"],
                    "details": step["details"]
                }
                for step in activated_reasoning_path
            ],
            "network_state": {
                "total_nodes": len(network_state["nodes"]),
                "active_nodes": network_state["active_nodes_count"],
                "total_activation": network_state["total_activation"],
                "activation_density": network_state["total_activation"] / len(network_state["nodes"])
            },
            "node_network": network_state["nodes"],
            "connection_network": network_state["connections"]
        }
        
        return {
            "status": "success",
            "demo_type": "activation_diffusion",
            "analysis": activation_analysis,
            "algorithm_features": {
                "interactive_property_graph": "基于交互式物性图谱的节点激活机制",
                "activation_diffusion": "智能激活扩散算法，模拟人类思维激活过程",
                "network_propagation": "多层激活传播，发现隐含的知识关联",
                "adaptive_reasoning": "根据激活模式动态选择推理策略",
                "visual_feedback": "实时激活状态可视化反馈"
            },
            "frontend_optimization": {
                "node_activation_visualization": "节点激活状态的动态可视化",
                "diffusion_animation": "激活扩散过程的动画展示",
                "activation_strength_indicators": "激活强度的进度条显示",
                "reasoning_path_highlighting": "推理路径的高亮展示",
                "interactive_exploration": "可交互的激活网络探索"
            }
        }
        
    except Exception as e:
        logger.error(f"激活扩散分析失败: {e}")
        return {
            "status": "error", 
            "demo_type": "activation_diffusion",
            "error": str(e),
            "fallback_info": {
                "description": "激活扩散引擎基于交互式物性图谱理论，模拟智能激活过程",
                "key_features": [
                    "基于交互式物性图谱的节点设计",
                    "智能激活扩散算法",
                    "多层网络传播机制", 
                    "自适应推理路径选择",
                    "实时激活状态反馈"
                ]
            }
        }

@app.post("/api/enhanced-solve")
async def enhanced_solve_direct(request: ProblemRequest):
    """🔧 直接使用增强数学求解器解决问题"""
    
    try:
        # 直接使用增强数学求解器
        result = reasoning_system.enhanced_math_solver.solve_problem(request.problem)
        
        return {
            "success": result["success"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "strategy_used": result["strategy_used"],
            "algorithm_type": "Enhanced_Mathematical_Reasoning",
            "problem_type": result.get("problem_type", "unknown"),
            "execution_time": 0.5,
            "reasoning_steps": result.get("reasoning_steps", []),
            "solution_steps": result.get("solution_steps", []),
            "entities": result.get("entities", []),
            "relations": result.get("relations", []),
            "entity_relationship_diagram": {
                "entities": result.get("entities", []),
                "relationships": result.get("relations", []),
                "solver_info": {
                    "solver_type": "enhanced_math_solver",
                    "mathematical_reasoning": True,
                    "problem_classification": result.get("problem_type", "unknown")
                }
            },
            "metadata": {
                "engine_used": "enhanced_math_solver",
                "mathematical_reasoning": True,
                "original_problem": request.problem
            }
        }
        
    except Exception as e:
        logger.error(f"增强数学求解器直接求解失败: {e}")
        return {
            "success": False,
            "answer": "求解失败",
            "confidence": 0.0,
            "strategy_used": "enhanced_math_solver_error",
            "algorithm_type": "Enhanced_Mathematical_Reasoning",
            "execution_time": 0.0,
            "reasoning_steps": [],
            "entity_relationship_diagram": {"entities": [], "relationships": []},
            "error": str(e)
        }

@app.get("/api/solver/test")
async def test_enhanced_solver():
    """🧪 测试增强数学求解器的能力"""
    
    test_problems = [
        "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
        "书店有30本书，卖了12本，还剩多少本？",
        "一个班级有24个学生，平均分成4组，每组有多少个学生？",
        "长方形的长是8米，宽是5米，面积是多少平方米？",
        "小华买了3包糖，每包有15个，一共有多少个糖？"
    ]
    
    test_results = []
    
    for i, problem in enumerate(test_problems, 1):
        try:
            result = reasoning_system.enhanced_math_solver.solve_problem(problem)
            
            test_results.append({
                "test_id": i,
                "problem": problem,
                "success": result["success"],
                "answer": result["answer"],
                "confidence": result["confidence"],
                "problem_type": result.get("problem_type", "unknown"),
                "solution_steps": len(result.get("solution_steps", [])),
                "entities_found": len(result.get("entities", [])),
                "relations_found": len(result.get("relations", []))
            })
            
        except Exception as e:
            test_results.append({
                "test_id": i,
                "problem": problem,
                "success": False,
                "error": str(e)
            })
    
    success_count = sum(1 for r in test_results if r.get("success", False))
    total_count = len(test_results)
    
    return {
        "status": "completed",
        "test_type": "enhanced_math_solver_capability",
        "summary": {
            "total_tests": total_count,
            "successful_tests": success_count,
            "success_rate": f"{success_count/total_count*100:.1f}%",
            "solver_status": "operational" if success_count > 0 else "needs_attention"
        },
        "test_results": test_results,
        "solver_info": {
            "solver_name": "Enhanced Mathematical Reasoning Engine",
            "capabilities": [
                "基础算术运算 (加减乘除)",
                "应用题求解 (文字题)",
                "几何问题 (面积、周长)",
                "乘除法问题 (分组、分配)",
                "实体关系提取",
                "数学表达式构建",
                "符号和数值计算"
            ],
            "supported_problem_types": [
                "arithmetic",
                "word_problem", 
                "geometry",
                "multiplication",
                "division"
            ]
        }
    }

# 学习指导API端点
@app.post("/api/learning/paths", response_model=ActivationLearningResponse)
async def get_personalized_learning_paths(request: LearningPathRequest):
    """获取个性化学习路径"""
    try:
        # 获取推荐学习路径
        recommended_paths = reasoning_system.learning_engine.get_personalized_learning_paths(
            request.user_level, request.learning_goal
        )
        
        # 获取激活扩散技巧
        activation_techniques = reasoning_system.learning_engine.get_activation_based_techniques()
        
        # 模拟个性化学习阶段（基于用户水平）
        personalized_stages = []
        stage_templates = [
            {
                'id': 1, 'title': '实体识别阶段', 'difficulty': 'beginner',
                'estimatedTime': '30-45分钟', 'status': 'available'
            },
            {
                'id': 2, 'title': '关系理解阶段', 'difficulty': 'beginner', 
                'estimatedTime': '45-60分钟', 'status': 'locked'
            },
            {
                'id': 3, 'title': '策略选择阶段', 'difficulty': 'intermediate',
                'estimatedTime': '60-75分钟', 'status': 'locked'
            }
        ]
        
        # 根据用户水平调整阶段状态
        if request.user_level == 'advanced':
            for stage in stage_templates:
                stage['status'] = 'available' if stage['id'] <= 2 else 'locked'
        elif request.user_level == 'intermediate':
            stage_templates[0]['status'] = 'available'
        
        personalized_stages = stage_templates
        
        # 学习网络状态
        learning_network_state = {
            'activated_concepts': reasoning_system.learning_engine.learning_network.keys(),
            'user_level': request.user_level,
            'activation_strength': 0.7 if request.user_level == 'advanced' else 0.5,
            'recommended_focus': 'entity_recognition' if request.user_level == 'beginner' else 'relationship_analysis'
        }
        
        return ActivationLearningResponse(
            recommended_paths=recommended_paths,
            personalized_stages=personalized_stages,
            activation_based_techniques=activation_techniques,
            learning_network_state=learning_network_state
        )
        
    except Exception as e:
        logger.error(f"获取学习路径失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取学习路径失败: {str(e)}")

@app.get("/api/learning/techniques")
async def get_learning_techniques():
    """获取学习技巧和方法"""
    return reasoning_system.learning_engine.get_activation_based_techniques()

@app.post("/api/learning/insights")
async def generate_learning_insights(request: ProblemRequest):
    """基于问题解答生成学习洞察"""
    try:
        # 先解决问题
        solution_result = await reasoning_system.solve_problem(
            problem_text=request.problem,
            mode=request.mode,
            preferences=request.preferences
        )
        
        # 生成学习洞察
        learning_insights = reasoning_system.learning_engine.generate_learning_insights(
            request.problem, solution_result
        )
        
        return {
            "success": True,
            "problem": request.problem,
            "solution_summary": {
                "answer": solution_result.get("answer"),
                "confidence": solution_result.get("confidence"),
                "entity_count": len(solution_result.get("entities", [])),
                "relation_count": len(solution_result.get("relationships", []))
            },
            "learning_insights": learning_insights,
            "activation_analysis": {
                "concepts_activated": len(solution_result.get("entities", [])) + len(solution_result.get("relationships", [])),
                "activation_pattern": "sequential" if len(solution_result.get("entities", [])) <= 3 else "parallel",
                "complexity_level": learning_insights.get("difficulty_analysis", {}).get("level", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"生成学习洞察失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成学习洞察失败: {str(e)}")

@app.get("/api/learning/network-state")
async def get_learning_network_state():
    """获取学习网络状态"""
    return {
        "network_structure": reasoning_system.learning_engine.learning_network,
        "activation_patterns": {
            "sequential": "适合初学者，逐步激活概念",
            "parallel": "适合高级用户，同时激活多个概念",
            "hierarchical": "适合复杂问题，按层次激活"
        },
        "learning_modes": {
            "guided": "引导式学习，系统推荐路径",
            "exploration": "探索式学习，用户自主选择",
            "adaptive": "自适应学习，根据表现调整"
        }
    }

# 启动函数
def start_server(host: str = "127.0.0.1", port: int = 8000):
    """启动服务器"""
    
    logger.info(f"启动QS²+IRD+COT-DIR统一推理后端服务")
    logger.info(f"服务地址: http://{host}:{port}")
    logger.info(f"API文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        "unified_backend_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()