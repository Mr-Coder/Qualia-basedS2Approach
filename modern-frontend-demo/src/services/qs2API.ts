/**
 * QS²算法专用API服务
 * ==================
 * 
 * 为QS²算法提供专门的API接口，包括：
 * - Qualia语义结构获取
 * - 兼容性分析结果
 * - 算法执行可视化数据
 * - 关系发现结果
 */

export interface QualiaRole {
  formal: string[]      // 形式角色
  telic: string[]       // 目的角色
  agentive: string[]    // 施事角色
  constitutive: string[] // 构成角色
}

export interface QualiaStructure {
  entity: string
  entity_type: string
  qualia_roles: QualiaRole
  context_features: Record<string, any>
  confidence: number
}

export interface QS2Relation {
  source: string
  target: string
  type: string
  strength: number
  confidence: number
  evidence: string[]
  qualia_based: boolean
  compatibility_result?: {
    overall_score: number
    detailed_scores: Record<string, number>
    compatibility_reasons: string[]
  }
}

export interface QS2Entity {
  id: string
  name: string
  type: 'person' | 'object' | 'concept' | 'money' | 'general'
  qualia_roles?: QualiaRole
  confidence: number
}

export interface QS2AlgorithmStage {
  name: string
  duration_ms: number
  confidence: number
  visual_elements: Array<{
    type: string
    id: string
    label?: string
    strength?: number
  }>
}

export interface QS2Data {
  entities: QS2Entity[]
  relationships: QS2Relation[]
  qualia_structures: QualiaStructure[]
  compatibility_results: any[]
  algorithm_stages: QS2AlgorithmStage[]
}

export interface QS2APIResponse {
  success: boolean
  data: QS2Data
  execution_id: string
  problem_text: string
  error?: string
}

class QS2APIService {
  private baseURL: string
  
  constructor() {
    // 使用相对路径，通过代理访问后端
    this.baseURL = '/api'
  }
  
  /**
   * 获取QS²算法关系发现结果
   */
  async getQS2Relations(): Promise<QS2APIResponse> {
    try {
      // 首先尝试获取QS²演示数据
      const demoResponse = await fetch(`${this.baseURL}/qs2/demo`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (demoResponse.ok) {
        const demoData = await demoResponse.json()
        if (demoData.success && demoData.data) {
          console.log('🧠 QS²演示数据获取成功:', demoData)
          return {
            success: true,
            data: demoData.data,
            execution_id: demoData.execution_id,
            problem_text: demoData.problem_text
          }
        }
      }
      
      // 如果演示数据不可用，尝试获取真实QS²数据
      const response = await fetch(`${this.baseURL}/qs2/relations`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      console.log('QS²关系数据获取成功:', data)
      
      return data
      
    } catch (error) {
      console.error('获取QS²关系失败:', error)
      
      // 返回模拟数据作为fallback
      return this.getMockQS2Data()
    }
  }
  
  /**
   * 获取Qualia语义结构
   */
  async getQualiaStructures(): Promise<{success: boolean, data: QualiaStructure[], count: number}> {
    try {
      // 首先尝试从演示数据提取Qualia结构
      const demoResponse = await fetch(`${this.baseURL}/qs2/demo`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (demoResponse.ok) {
        const demoData = await demoResponse.json()
        if (demoData.success && demoData.data && demoData.data.entities) {
          const qualiaStructures = demoData.data.entities.map((entity: any) => ({
            entity: entity.name,
            entity_type: entity.type,
            qualia_roles: entity.qualia_roles,
            context_features: {
              id: entity.id,
              confidence: entity.confidence
            },
            confidence: entity.confidence
          }))
          
          console.log('🧠 从演示数据提取Qualia结构成功:', qualiaStructures)
          return {
            success: true,
            data: qualiaStructures,
            count: qualiaStructures.length
          }
        }
      }
      
      // 如果演示数据不可用，尝试获取真实Qualia结构
      const response = await fetch(`${this.baseURL}/qs2/qualia-structures`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      return await response.json()
      
    } catch (error) {
      console.error('获取Qualia结构失败:', error)
      
      // 返回模拟数据
      return {
        success: true,
        data: this.getMockQualiaStructures(),
        count: 3
      }
    }
  }
  
  /**
   * 获取算法执行数据（增强版）
   */
  async getEnhancedAlgorithmExecution(): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/algorithm/execution`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      
      // 如果是QS²增强数据，添加特殊标记
      if (data.success && data.data) {
        data.data.is_qs2_enhanced = true
        data.data.algorithm_type = 'QS2_Enhanced'
      }
      
      return data
      
    } catch (error) {
      console.error('获取增强算法执行数据失败:', error)
      return { success: false, error: error.message }
    }
  }
  
  /**
   * 创建QS²可视化配置
   */
  createQS2VisualizationConfig() {
    return {
      node_colors: {
        person: '#e74c3c',    // 红色 - 人物实体
        object: '#27ae60',    // 绿色 - 物体实体
        concept: '#9b59b6',   // 紫色 - 概念实体
        money: '#f39c12',     // 橙色 - 货币实体
        general: '#6b7280'    // 灰色 - 通用实体
      },
      
      relation_colors: {
        'semantic': '#FF9FF3',      // 语义关系
        'functional': '#54A0FF',    // 功能关系
        'contextual': '#5F27CD',    // 上下文关系
        'structural': '#00D2D3',    // 结构关系
        'quantitative': '#FF9F43',  // 数量关系
        'qualia_based': '#8B5CF6'   // QS²特有关系
      },
      
      qualia_colors: {
        formal: '#FF6B6B',        // 形式角色
        telic: '#4ECDC4',         // 目的角色
        agentive: '#45B7D1',      // 施事角色
        constitutive: '#96CEB4'   // 构成角色
      },
      
      animation: {
        enable_stage_animation: true,
        stage_duration: 1500,
        relation_discovery_delay: 500,
        confidence_fade_in: true
      }
    }
  }
  
  /**
   * 模拟QS²数据（用于测试和fallback）
   */
  private getMockQS2Data(): QS2APIResponse {
    return {
      success: true,
      execution_id: 'qs2_mock_001',
      problem_text: '小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？',
      data: {
        entities: [
          {
            id: 'entity_1',
            name: '小明',
            type: 'person',
            confidence: 0.95,
            qualia_roles: {
              formal: ['人物', '主体'],
              telic: ['拥有苹果', '参与计算'],
              agentive: ['题目设定'],
              constitutive: ['个体', '主语']
            }
          },
          {
            id: 'entity_2', 
            name: '小红',
            type: 'person',
            confidence: 0.95,
            qualia_roles: {
              formal: ['人物', '主体'],
              telic: ['拥有苹果', '参与计算'],
              agentive: ['题目设定'],
              constitutive: ['个体', '主语']
            }
          },
          {
            id: 'entity_3',
            name: '苹果',
            type: 'object',
            confidence: 0.90,
            qualia_roles: {
              formal: ['可计数物体', '水果'],
              telic: ['被拥有', '被计算'],
              agentive: ['自然生长'],
              constitutive: ['有机物', '可食用']
            }
          },
          {
            id: 'entity_4',
            name: '5',
            type: 'concept',
            confidence: 0.85,
            qualia_roles: {
              formal: ['正整数', '数量'],
              telic: ['表示数量', '参与运算'],
              agentive: ['题目给定'],
              constitutive: ['数字符号']
            }
          },
          {
            id: 'entity_5',
            name: '3',
            type: 'concept', 
            confidence: 0.85,
            qualia_roles: {
              formal: ['正整数', '数量'],
              telic: ['表示数量', '参与运算'],
              agentive: ['题目给定'],
              constitutive: ['数字符号']
            }
          }
        ],
        
        relationships: [
          {
            source: 'entity_1',
            target: 'entity_3',
            type: '拥有关系',
            strength: 0.92,
            confidence: 0.90,
            evidence: ['题目明确表述', '主谓关系'],
            qualia_based: true,
            compatibility_result: {
              overall_score: 0.88,
              detailed_scores: {
                formal: 0.85,
                telic: 0.95,
                agentive: 0.80,
                constitutive: 0.85
              },
              compatibility_reasons: ['目的角色高度兼容', '实体类型互补']
            }
          },
          {
            source: 'entity_2',
            target: 'entity_3',
            type: '拥有关系',
            strength: 0.92,
            confidence: 0.90,
            evidence: ['题目明确表述', '主谓关系'],
            qualia_based: true,
            compatibility_result: {
              overall_score: 0.88,
              detailed_scores: {
                formal: 0.85,
                telic: 0.95,
                agentive: 0.80,
                constitutive: 0.85
              },
              compatibility_reasons: ['目的角色高度兼容', '实体类型互补']
            }
          },
          {
            source: 'entity_1',
            target: 'entity_4',
            type: '数量关系',
            strength: 0.85,
            confidence: 0.88,
            evidence: ['数量对应', '语义关联'],
            qualia_based: true,
            compatibility_result: {
              overall_score: 0.82,
              detailed_scores: {
                formal: 0.90,
                telic: 0.85,
                agentive: 0.75,
                constitutive: 0.80
              },
              compatibility_reasons: ['形式角色兼容', '功能互补']
            }
          },
          {
            source: 'entity_2',
            target: 'entity_5',
            type: '数量关系',
            strength: 0.85,
            confidence: 0.88,
            evidence: ['数量对应', '语义关联'],
            qualia_based: true
          },
          {
            source: 'entity_4',
            target: 'entity_5',
            type: '聚合关系',
            strength: 0.78,
            confidence: 0.85,
            evidence: ['求和运算', '数学关系'],
            qualia_based: true
          }
        ],
        
        qualia_structures: [],
        compatibility_results: [],
        
        algorithm_stages: [
          {
            name: '实体提取',
            duration_ms: 45.2,
            confidence: 0.95,
            visual_elements: [
              { type: 'entity', id: 'entity_1', label: '小明' },
              { type: 'entity', id: 'entity_2', label: '小红' },
              { type: 'entity', id: 'entity_3', label: '苹果' }
            ]
          },
          {
            name: '语义结构构建',
            duration_ms: 128.7,
            confidence: 0.88,
            visual_elements: [
              { type: 'qualia', id: 'qualia_1', label: '小明-语义结构' },
              { type: 'qualia', id: 'qualia_2', label: '小红-语义结构' }
            ]
          },
          {
            name: '兼容性计算',
            duration_ms: 89.3,
            confidence: 0.92,
            visual_elements: [
              { type: 'compatibility', id: 'comp_1', strength: 0.88 },
              { type: 'compatibility', id: 'comp_2', strength: 0.85 }
            ]
          },
          {
            name: '关系发现',
            duration_ms: 156.4,
            confidence: 0.87,
            visual_elements: [
              { type: 'relation', id: 'rel_1', strength: 0.92 },
              { type: 'relation', id: 'rel_2', strength: 0.85 }
            ]
          }
        ]
      }
    }
  }
  
  /**
   * 模拟Qualia结构数据
   */
  private getMockQualiaStructures(): QualiaStructure[] {
    return [
      {
        entity: '小明',
        entity_type: 'person',
        confidence: 0.95,
        qualia_roles: {
          formal: ['人物实体', '主体', '个体'],
          telic: ['拥有物品', '参与计算', '作为主语'],
          agentive: ['题目设定', '概念构建'],
          constitutive: ['认知主体', '语言主语']
        },
        context_features: {
          problem_type: 'arithmetic',
          entity_role: 'subject',
          semantic_weight: 0.9
        }
      },
      {
        entity: '苹果',
        entity_type: 'object',
        confidence: 0.90,
        qualia_roles: {
          formal: ['可计数对象', '物理实体', '水果类'],
          telic: ['被拥有', '被计算', '作为宾语'],
          agentive: ['自然生长', '题目提及'],
          constitutive: ['有机物质', '可分离单元']
        },
        context_features: {
          problem_type: 'arithmetic',
          entity_role: 'object',
          countable: true
        }
      },
      {
        entity: '5',
        entity_type: 'concept',
        confidence: 0.85,
        qualia_roles: {
          formal: ['正整数', '数量概念', '数值'],
          telic: ['表示数量', '参与运算', '量化关系'],
          agentive: ['数学系统', '题目给定'],
          constitutive: ['抽象符号', '数学概念']
        },
        context_features: {
          problem_type: 'arithmetic',
          numeric_value: 5,
          operation_role: 'operand'
        }
      }
    ]
  }
}

// 创建全局实例
export const qs2API = new QS2APIService()

export default qs2API