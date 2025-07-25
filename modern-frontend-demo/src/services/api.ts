// 统一API客户端
import { apiClient, testAPIConnection } from './unifiedAPI'

import { Strategy, SolveResult, DeepRelation, ImplicitConstraint, ReasoningLayer, VisualizationConfig } from '@/stores/problemStore'

// 使用Vite代理路径，避免CORS问题
const API_BASE_URL = ''  // 使用相对路径，通过Vite代理到后端

export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
}

export interface SolveProblemRequest {
  problem: string
  strategy: Strategy
}

// 增强引擎API接口类型
interface EnhancedSolveResponse {
  success: boolean
  answer: string
  confidence: number
  strategy_used: string
  reasoning_steps: Array<{
    step: number
    action: string
    description: string
  }>
  execution_time: number
  entity_relationship_diagram: {
    entities: Array<{
      id: string
      type: string
      properties: string[]
      numeric_attributes: Record<string, any>
    }>
    relationships: Array<{
      from: string
      to: string
      type: string
      weight: number
      properties: string[]
    }>
    implicit_constraints: string[]
    graph_properties: {
      node_count: number
      edge_count: number
      constraint_count: number
      complexity_score: number
    }
    enhanced_discovery?: boolean
    relations_found?: number
    processing_time?: number
    entity_count?: number
    high_strength_relations?: number
    relations?: Array<{
      entity1: string
      entity2: string
      type: string
      strength: number
      semantic_score: number
      functional_score: number
      contextual_score: number
      evidence: string[]
      confidence: number
    }>
    graph_data?: {
      nodes: Array<{ id: string; type: string }>
      edges: Array<{
        source: string
        target: string
        weight: number
        type: string
        evidence: string[]
      }>
    }
    statistics?: Record<string, any>
    enhancement_status?: string
    algorithm_used?: string
  }
  enhanced_analysis?: {
    enhancement_used: boolean
    relations_discovered: number
    processing_method: string
    high_confidence_relations: number
    discovery_algorithm: string
    semantic_depth: number
  }
  error?: string
}

// 转换增强引擎响应为前端格式
const transformEnhancedResponse = (enhancedResponse: EnhancedSolveResponse): SolveResult => {
  const diagram = enhancedResponse.entity_relationship_diagram || {}
  const frontendVisualizationData = enhancedResponse.metadata?.frontend_visualization_data || {}
  
  // 转换实体 - 需要正确映射类型
  const entities = (diagram.entities || []).map(entity => {
    let frontendType: 'person' | 'object' | 'money' | 'concept' = 'concept'
    
    if (entity.type === 'person') frontendType = 'person'
    else if (entity.type === 'object') frontendType = 'object' 
    else if (entity.type === 'currency') frontendType = 'money'
    else if (entity.type.includes && entity.type.includes('concept')) frontendType = 'concept'
    else frontendType = 'concept' // 默认为概念类型
    
    return {
      id: entity.id,
      name: entity.id,
      type: frontendType
    }
  })
  
  // 如果前端可视化数据中有实体，优先使用
  if (frontendVisualizationData.entities && frontendVisualizationData.entities.length > 0) {
    entities.length = 0 // 清空现有实体
    entities.push(...frontendVisualizationData.entities)
  }
  
  // 转换关系 - 优先使用传统关系结构
  const relationships = (diagram.relationships || []).map(rel => ({
    source: rel.from,
    target: rel.to,
    type: rel.type,
    label: rel.type,
    weight: rel.weight
  }))
  
  // 从增强发现的关系中添加更多连接
  if (diagram.graph_data?.edges && Array.isArray(diagram.graph_data.edges)) {
    diagram.graph_data.edges.forEach(edge => {
      // 检查是否已存在相同的关系
      const existingRel = relationships.find(r => 
        r.source === edge.source && r.target === edge.target
      )
      
      if (!existingRel) {
        relationships.push({
          source: edge.source,
          target: edge.target,
          type: edge.type,
          label: `${edge.type} (${(edge.weight * 100).toFixed(1)}%)`,
          weight: edge.weight
        })
      }
    })
  }
  
  // 如果传统实体为空，使用增强发现的实体图
  if (entities.length === 0 && diagram.graph_data?.nodes) {
    diagram.graph_data.nodes.forEach(node => {
      let frontendType: 'person' | 'object' | 'money' | 'concept' = 'concept'
      
      // 根据节点ID推断类型
      if (['小明', '小红', '小张', 'xiaoming', 'xiaohong'].includes(node.id)) {
        frontendType = 'person'
      } else if (['苹果', '笔', 'apple'].includes(node.id)) {
        frontendType = 'object'
      } else if (['元', '钱', 'money'].includes(node.id)) {
        frontendType = 'money'
      } else if (!isNaN(Number(node.id))) {
        frontendType = 'concept' // 数字作为概念
      }
      
      entities.push({
        id: node.id,
        name: node.id,
        type: frontendType
      })
    })
  }
  
  // 转换推理步骤 - FastAPI返回字符串数组
  const steps = Array.isArray(enhancedResponse.reasoning_steps) 
    ? enhancedResponse.reasoning_steps.map((step, index) => {
        // 如果是对象格式，提取描述；如果是字符串，直接使用
        if (typeof step === 'string') {
          return step
        } else if (step.description) {
          return `步骤${step.step || index + 1}: ${step.description}`
        } else {
          return `步骤${index + 1}: ${JSON.stringify(step)}`
        }
      })
    : []
  
  // 组合约束条件
  const constraints = [
    ...(diagram.implicit_constraints || []),
    ...(enhancedResponse.enhanced_analysis ? [
      `使用了${enhancedResponse.enhanced_analysis.discovery_algorithm}算法`,
      `发现了${enhancedResponse.enhanced_analysis.relations_discovered}个增强关系`,
      `语义深度: ${(enhancedResponse.enhanced_analysis.semantic_depth * 100).toFixed(1)}%`
    ] : [])
  ]

  // 提取物性约束
  const physicalConstraints = (diagram.implicit_constraints || []).filter(constraint => 
    constraint.includes('守恒') || constraint.includes('连续性') || constraint.includes('拥有关系') || 
    constraint.includes('单调性') || constraint.includes('非负') || constraint.includes('约束')
  )

  // 分类物性属性
  const physicalProperties = {
    conservationLaws: (diagram.implicit_constraints || []).filter(c => c.includes('守恒')),
    spatialRelations: (diagram.implicit_constraints || []).filter(c => c.includes('空间') || c.includes('位置')),
    temporalConstraints: (diagram.implicit_constraints || []).filter(c => c.includes('时间') || c.includes('顺序')),
    materialProperties: (diagram.implicit_constraints || []).filter(c => c.includes('连续性') || c.includes('物体') || c.includes('拥有'))
  }

  // 新增：处理深度隐含关系数据
  const deepRelations: DeepRelation[] = frontendVisualizationData.deep_relations || []
  const implicitConstraints: ImplicitConstraint[] = frontendVisualizationData.implicit_constraints || []
  const reasoningLayers: ReasoningLayer[] = frontendVisualizationData.reasoning_layers || {}
  const visualizationConfig: VisualizationConfig = frontendVisualizationData.visualization_config || {
    show_depth_indicators: true,
    show_constraint_panels: true,
    enable_interactive_exploration: true,
    animation_sequence: true
  }

  // 新增：处理物性图谱数据（从后端获取）
  const physicalGraph = enhancedResponse.physicalGraph || enhancedResponse.physical_graph
  const physicalAnalysis = enhancedResponse.physicalAnalysis || enhancedResponse.physical_analysis
  
  return {
    answer: enhancedResponse.answer,
    confidence: enhancedResponse.confidence,
    strategy: enhancedResponse.strategy_used as Strategy,
    steps,
    entities,
    relationships,
    constraints,
    physicalConstraints,
    physicalProperties,
    processingTime: enhancedResponse.execution_time,
    enhancedInfo: enhancedResponse.enhanced_analysis ? {
      algorithm: enhancedResponse.enhanced_analysis.discovery_algorithm,
      relationsFound: enhancedResponse.enhanced_analysis.relations_discovered,
      semanticDepth: enhancedResponse.enhanced_analysis.semantic_depth,
      processingMethod: enhancedResponse.enhanced_analysis.processing_method
    } : undefined,
    // 新增深度隐含关系增强数据
    deepRelations,
    implicitConstraints,
    reasoningLayers,
    visualizationConfig,
    // 新增：物性图谱数据
    physicalGraph,
    physicalAnalysis
  }
}

// 主要的解题API函数 - 现在直接使用增强引擎
export const solveProblem = async (request: SolveProblemRequest): Promise<SolveResult> => {
  try {
    console.log('调用增强引擎API:', request)
    
    // 创建超时控制器
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 45000) // 45秒超时
    
    const response = await fetch(`${API_BASE_URL}/api/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        problem: request.problem,
        strategy: request.strategy
      }),
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const enhancedResponse: EnhancedSolveResponse = await response.json()
    console.log('增强引擎响应:', enhancedResponse)
    
    if (!enhancedResponse.success) {
      throw new Error(enhancedResponse.error || '增强引擎返回错误')
    }
    
    return transformEnhancedResponse(enhancedResponse)
    
  } catch (error) {
    console.error('增强引擎调用失败:', error)
    
    // 根据错误类型给出更友好的提示
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error('请求超时，COT-DIR算法处理时间较长，请稍后重试')
      } else if (error.message.includes('Failed to fetch')) {
        throw new Error('网络连接失败，请检查后端服务是否启动')
      } else if (error.message.includes('500')) {
        throw new Error('服务器内部错误，推理引擎可能遇到问题')
      } else {
        throw new Error(`增强引擎错误: ${error.message}`)
      }
    }
    
    throw new Error('增强引擎不可用')
  }
}

// 保留原有API调用接口以兼容
export const solveProblemAPI = async (request: SolveProblemRequest): Promise<ApiResponse<SolveResult>> => {
  try {
    const result = await solveProblem(request)
    return { success: true, data: result }
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }
  }
}

// 新增：获取系统状态
export const getSystemStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/system`)
    if (response.ok) {
      return await response.json()
    }
  } catch (error) {
    console.warn('无法获取系统状态:', error)
  }
  return null
}

// 新增：获取推理策略信息
export const getStrategiesInfo = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/strategies`)
    if (response.ok) {
      return await response.json()
    }
  } catch (error) {
    console.warn('无法获取策略信息:', error)
  }
  return null
}
// 添加API连接测试功能
export { testAPIConnection } from './unifiedAPI'

// 学习指导相关API
export interface LearningPathRequest {
  user_level: 'beginner' | 'intermediate' | 'advanced'
  learning_goal: string
  preferences?: Record<string, any>
}

export interface LearningInsightsRequest {
  problem: string
  mode?: string
  preferences?: Record<string, any>
}

export interface ActivationLearningResponse {
  recommended_paths: Array<{
    id: string
    title: string
    description: string
    estimatedTime: string
    difficulty: string
    stages: number
    icon: string
    activation_pattern: string
    recommended_for: string
  }>
  personalized_stages: Array<{
    id: number
    title: string
    difficulty: string
    estimatedTime: string
    status: string
  }>
  activation_based_techniques: Array<{
    category: string
    icon: string
    color: string
    techniques: string[]
    activation_methods?: string[]
  }>
  learning_network_state: {
    activated_concepts: string[]
    user_level: string
    activation_strength: number
    recommended_focus: string
  }
}

// 获取个性化学习路径
export const getPersonalizedLearningPaths = async (request: LearningPathRequest): Promise<ActivationLearningResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/learning/paths`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('获取学习路径失败:', error)
    throw new Error('获取学习路径失败')
  }
}

// 获取学习技巧
export const getLearningTechniques = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/learning/techniques`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('获取学习技巧失败:', error)
    throw new Error('获取学习技巧失败')
  }
}

// 生成学习洞察
export const generateLearningInsights = async (request: LearningInsightsRequest) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/learning/insights`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('生成学习洞察失败:', error)
    throw new Error('生成学习洞察失败')
  }
}

// 获取学习网络状态
export const getLearningNetworkState = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/learning/network-state`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('获取学习网络状态失败:', error)
    throw new Error('获取学习网络状态失败')
  }
}

// 错题分析相关API
export interface ErrorAnalysisRequest {
  problem: string
  student_answer?: string
  correct_answer?: string
  error_type?: string
}

export interface ErrorAnalysisResponse {
  error_types: Array<{
    id: string
    name: string
    description: string
    confidence: number
    examples: string[]
    correction_methods: string[]
  }>
  strategy_errors: Array<{
    id: string
    strategy: 'COT' | 'GOT' | 'TOT' | 'AUTO'
    name: string
    description: string
    common_issues: string[]
    improvements: string[]
    confidence: number
  }>
  detailed_analysis: {
    error_pattern: string
    root_cause: string
    cognitive_process: string
    solution_steps: string[]
    related_concepts: string[]
  }
  personalized_feedback: {
    error_severity: 'low' | 'medium' | 'high'
    improvement_priority: string[]
    practice_suggestions: string[]
    learning_path_adjustment: string[]
  }
}

// 分析学生错误
export const analyzeStudentErrors = async (request: ErrorAnalysisRequest): Promise<ErrorAnalysisResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/error-analysis/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('错误分析失败:', error)
    throw new Error('错误分析失败')
  }
}

// 获取常见错误类型
export const getCommonErrorTypes = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/error-analysis/common-types`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('获取常见错误类型失败:', error)
    throw new Error('获取常见错误类型失败')
  }
}

// 获取策略专项错误分析
export const getStrategyErrors = async (strategy?: string) => {
  try {
    const url = strategy 
      ? `${API_BASE_URL}/api/error-analysis/strategy-errors?strategy=${strategy}`
      : `${API_BASE_URL}/api/error-analysis/strategy-errors`
    
    const response = await fetch(url)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('获取策略错误分析失败:', error)
    throw new Error('获取策略错误分析失败')
  }
}

// 获取错误详细信息
export const getErrorDetails = async (errorId: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/error-analysis/details/${errorId}`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('获取错误详情失败:', error)
    throw new Error('获取错误详情失败')
  }
}

// 生成个性化改进建议
export const generateImprovementSuggestions = async (request: {
  error_types: string[]
  user_level: string
  problem_context: string
}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/error-analysis/improvement-suggestions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('生成改进建议失败:', error)
    throw new Error('生成改进建议失败')
  }
}

// 在开发模式下自动测试连接
if (process.env.NODE_ENV === 'development') {
  testAPIConnection().then(connected => {
    if (!connected) {
      console.warn('⚠️ 后端API连接失败，请检查服务器状态')
    }
  })
}
