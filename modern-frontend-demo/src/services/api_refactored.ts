// 重构后的组件 - 改进复杂度和可维护性
// TODO: 手动优化具体逻辑

import React, { memo, useCallback, useMemo } from 'react';
import { ErrorBoundary } from '../shared/ErrorBoundary';

// 建议：将以下功能拆分为独立组件
// const HeaderSection = memo(() => { /* 头部逻辑 */ });
// const ContentSection = memo(() => { /* 内容逻辑 */ });
// const FooterSection = memo(() => { /* 底部逻辑 */ });

// 原组件代码 (需要手动优化):
// 统一API客户端
import { apiClient, testAPIConnection } from './unifiedAPI'

import { Strategy, SolveResult, DeepRelation, ImplicitConstraint, ReasoningLayer, VisualizationConfig } from '@/stores/problemStore'

// 更新为FastAPI后端端口
const API_BASE_URL = 'http://localhost:8000'

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
    visualizationConfig
  }
}

// 主要的解题API函数 - 现在直接使用增强引擎
export const solveProblem = async (request: SolveProblemRequest): Promise<SolveResult> => {
  try {
    console.log('调用增强引擎API:', request)
    
    const response = await fetch(`${API_BASE_URL}/api/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        problem: request.problem,
        strategy: request.strategy
      }),
    })
    
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
    
    // 如果增强引擎不可用，返回基础错误信息
    throw new Error(
      error instanceof Error 
        ? `增强引擎错误: ${error.message}` 
        : '增强引擎不可用'
    )
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

// 在开发模式下自动测试连接
if (process.env.NODE_ENV === 'development') {
  testAPIConnection().then(connected => {
    if (!connected) {
      console.warn('⚠️ 后端API连接失败，请检查服务器状态')
    }
  })
}
