// 物性关系图专用API客户端
import { Entity, Relationship, DeepRelation, ImplicitConstraint, VisualizationConfig } from '@/stores/problemStore'

// 物性关系图API的基础URL
const PHYSICAL_API_BASE_URL = process.env.NODE_ENV === 'development' ? '' : 'http://127.0.0.1:5004'

// 物性实体接口
export interface PhysicalEntity extends Entity {
  properties: Record<string, any>
  position?: { x: number; y: number }
}

// 物性关系接口
export interface PhysicalRelationship extends Relationship {
  strength: number
  properties: Record<string, any>
  constraints: string[]
}

// 物性约束接口
export interface PhysicalConstraint {
  id: string
  type: string
  description: string
  expression: string
  entities: string[]
  confidence: number
  category: string
  visualization: {
    color: string
    icon: string
    priority: number
  }
}

// 物性属性接口
export interface PhysicalProperties {
  conservationLaws: string[]
  spatialRelations: string[]
  temporalConstraints: string[]
  materialProperties: string[]
}

// 物性关系分析结果接口
export interface PhysicalAnalysisResult {
  entities: PhysicalEntity[]
  relationships: PhysicalRelationship[]
  physicalConstraints: PhysicalConstraint[]
  physicalProperties: PhysicalProperties
  visualizationConfig: VisualizationConfig
  metadata: {
    analysis_method: string
    complexity_score: number
    physics_categories: string[]
    confidence: number
  }
}

// API响应接口
export interface PhysicalAPIResponse<T> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

// 分析类型
export type AnalysisType = 'complete' | 'entities' | 'relations' | 'constraints'

// 图表类型
export type DiagramType = 'physical_relations' | 'entity_relationship' | 'constraint_diagram'

/**
 * 物性关系图专用API类
 */
export class PhysicalRelationsAPI {
  private baseURL: string

  constructor(baseURL: string = PHYSICAL_API_BASE_URL) {
    this.baseURL = baseURL
  }

  /**
   * 执行完整的物性关系分析
   */
  async analyzePhysicalRelations(problem: string): Promise<PhysicalAnalysisResult> {
    try {
      const response = await fetch(`${this.baseURL}/api/physical-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
          type: 'complete'
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result: PhysicalAPIResponse<PhysicalAnalysisResult> = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || '物性关系分析失败')
      }

      return result.data!
    } catch (error) {
      console.error('物性关系分析失败:', error)
      throw error
    }
  }

  /**
   * 获取物性实体
   */
  async getPhysicalEntities(problem: string): Promise<PhysicalEntity[]> {
    try {
      const response = await fetch(`${this.baseURL}/api/physical-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
          type: 'entities'
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result: PhysicalAPIResponse<PhysicalEntity[]> = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || '实体提取失败')
      }

      return result.data!
    } catch (error) {
      console.error('物性实体提取失败:', error)
      throw error
    }
  }

  /**
   * 获取物性关系
   */
  async getPhysicalRelationships(problem: string): Promise<PhysicalRelationship[]> {
    try {
      const response = await fetch(`${this.baseURL}/api/physical-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
          type: 'relations'
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result: PhysicalAPIResponse<PhysicalRelationship[]> = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || '关系提取失败')
      }

      return result.data!
    } catch (error) {
      console.error('物性关系提取失败:', error)
      throw error
    }
  }

  /**
   * 获取物性约束
   */
  async getPhysicalConstraints(problem: string): Promise<PhysicalConstraint[]> {
    try {
      const response = await fetch(`${this.baseURL}/api/physical-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
          type: 'constraints'
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result: PhysicalAPIResponse<PhysicalConstraint[]> = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || '约束提取失败')
      }

      return result.data!
    } catch (error) {
      console.error('物性约束提取失败:', error)
      throw error
    }
  }

  /**
   * 获取支持的物理学分类
   */
  async getPhysicsCategories(): Promise<Record<string, any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/physics-categories`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || '获取物理学分类失败')
      }

      return result.categories
    } catch (error) {
      console.error('获取物理学分类失败:', error)
      throw error
    }
  }

  /**
   * 获取可视化配置
   */
  async getVisualizationConfig(problem: string, diagramType: DiagramType = 'physical_relations'): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/api/visualization-config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
          diagram_type: diagramType
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || '获取可视化配置失败')
      }

      return result.config
    } catch (error) {
      console.error('获取可视化配置失败:', error)
      throw error
    }
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/api/health`, {
        method: 'GET',
        timeout: 5000,
      })

      if (!response.ok) {
        return false
      }

      const result = await response.json()
      return result.status === 'healthy'
    } catch (error) {
      console.error('物性关系API健康检查失败:', error)
      return false
    }
  }

  /**
   * 获取系统统计信息
   */
  async getStats(): Promise<Record<string, any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/stats`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || '获取统计信息失败')
      }

      return result.stats
    } catch (error) {
      console.error('获取统计信息失败:', error)
      throw error
    }
  }
}

// 创建默认实例
export const physicalRelationsAPI = new PhysicalRelationsAPI()

// 便捷函数
export const analyzePhysicalRelations = (problem: string) => 
  physicalRelationsAPI.analyzePhysicalRelations(problem)

export const getPhysicalEntities = (problem: string) => 
  physicalRelationsAPI.getPhysicalEntities(problem)

export const getPhysicalRelationships = (problem: string) => 
  physicalRelationsAPI.getPhysicalRelationships(problem)

export const getPhysicalConstraints = (problem: string) => 
  physicalRelationsAPI.getPhysicalConstraints(problem)

export const getPhysicsCategories = () => 
  physicalRelationsAPI.getPhysicsCategories()

export const getVisualizationConfig = (problem: string, diagramType?: DiagramType) => 
  physicalRelationsAPI.getVisualizationConfig(problem, diagramType)

export const checkPhysicalAPIHealth = () => 
  physicalRelationsAPI.healthCheck()

export const getPhysicalAPIStats = () => 
  physicalRelationsAPI.getStats()

// 在开发模式下自动测试连接
if (process.env.NODE_ENV === 'development') {
  checkPhysicalAPIHealth().then(healthy => {
    if (healthy) {
      console.log('✅ 物性关系API连接正常')
    } else {
      console.warn('⚠️ 物性关系API连接失败，请检查服务器状态（端口5004）')
    }
  })
}