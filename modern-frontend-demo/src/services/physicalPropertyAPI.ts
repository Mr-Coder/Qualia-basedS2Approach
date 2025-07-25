// 物性图谱API接口
// 专门用于处理后端生成的物性图谱数据

export interface PhysicalProperty {
  id: string
  type: 'conservation' | 'discreteness' | 'continuity' | 'additivity' | 'measurability' | 'locality' | 'temporality' | 'causality'
  entity: string
  value: any
  unit: string
  certainty: number
  constraints: string[]
}

export interface PhysicalConstraint {
  id: string
  type: 'conservation_law' | 'non_negative' | 'integer_constraint' | 'upper_bound' | 'lower_bound' | 'equivalence' | 'ordering' | 'exclusivity'
  description: string
  expression: string
  strength: number
  entities: string[]
}

export interface PhysicalRelation {
  id: string
  source: string
  target: string
  type: string
  physical_basis: string
  strength: number
  causal_direction?: string
}

export interface PhysicalGraphAnalysis {
  problem: string
  physical_properties: PhysicalProperty[]
  physical_constraints: PhysicalConstraint[]
  physical_relations: PhysicalRelation[]
  graph_metrics: Record<string, number>
  consistency_score: number
  backend_driven_features?: Record<string, string>
  frontend_optimization?: Record<string, string>
}

// 获取物性图谱演示数据
export const getPhysicalPropertyDemo = async (): Promise<PhysicalGraphAnalysis | null> => {
  try {
    const response = await fetch('/api/physical-property/demo')
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    const data = await response.json()
    
    if (data.status === 'success' && data.analysis) {
      console.log('✅ 物性图谱数据获取成功:', data.analysis)
      return data.analysis
    } else {
      console.warn('⚠️ 物性图谱数据格式异常:', data)
      return null
    }
  } catch (error) {
    console.error('❌ 获取物性图谱演示数据失败:', error)
    return null
  }
}

// 获取特定问题的物性图谱分析
export const getPhysicalPropertyAnalysis = async (problemText: string): Promise<PhysicalGraphAnalysis | null> => {
  try {
    // 首先尝试求解问题，这会触发完整的推理流程包括物性图谱构建
    const solveResponse = await fetch('/api/solve', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        problem: problemText,
        mode: 'advanced'
      }),
    })
    
    if (solveResponse.ok) {
      const solveData = await solveResponse.json()
      
      // 检查是否包含物性图谱数据
      if (solveData.physicalGraph || solveData.physical_graph) {
        const physicalGraph = solveData.physicalGraph || solveData.physical_graph
        
        return {
          problem: problemText,
          physical_properties: physicalGraph.properties || [],
          physical_constraints: physicalGraph.constraints || [],
          physical_relations: physicalGraph.relations || [],
          graph_metrics: physicalGraph.graph_metrics || {},
          consistency_score: physicalGraph.consistency_score || 0,
          backend_driven_features: solveData.backend_driven_features,
          frontend_optimization: solveData.frontend_optimization
        }
      }
    }
    
    // 如果求解接口没有返回物性图谱数据，回退到演示数据
    console.log('🔄 求解接口未返回物性图谱，使用演示数据')
    return await getPhysicalPropertyDemo()
    
  } catch (error) {
    console.error('❌ 获取物性图谱分析失败:', error)
    return null
  }
}

// 获取后端驱动的前端优化配置
export const getBackendDrivenConfig = async (): Promise<Record<string, any> | null> => {
  try {
    const response = await fetch('/api/physical-property/demo')
    
    if (!response.ok) {
      return null
    }
    
    const data = await response.json()
    
    if (data.frontend_optimization) {
      console.log('🎨 获取到后端驱动的前端优化配置:', data.frontend_optimization)
      return {
        visualization: data.frontend_optimization,
        backend_features: data.backend_driven_features
      }
    }
    
    return null
  } catch (error) {
    console.error('❌ 获取前端优化配置失败:', error)
    return null
  }
}

// 获取物性属性类型说明
export const getPhysicalPropertyTypes = (): Record<string, string> => {
  return {
    conservation: '守恒性 - 量的守恒定律',
    discreteness: '离散性 - 可数性质',
    continuity: '连续性 - 连续变化',
    additivity: '可加性 - 线性叠加',
    measurability: '可测性 - 可量化属性',
    locality: '局域性 - 空间位置',
    temporality: '时序性 - 时间关系',
    causality: '因果性 - 因果关系'
  }
}

// 获取物性约束类型说明
export const getPhysicalConstraintTypes = (): Record<string, string> => {
  return {
    conservation_law: '守恒定律 - 物理量守恒',
    non_negative: '非负约束 - 数值非负',
    integer_constraint: '整数约束 - 离散数值',
    upper_bound: '上界约束 - 最大值限制',
    lower_bound: '下界约束 - 最小值限制',
    equivalence: '等价约束 - 数值相等',
    ordering: '顺序约束 - 大小关系',
    exclusivity: '排斥约束 - 互斥关系'
  }
}

// 验证物性图谱一致性
export const validatePhysicalConsistency = (analysis: PhysicalGraphAnalysis): {
  isConsistent: boolean
  violations: string[]
  warnings: string[]
} => {
  const violations: string[] = []
  const warnings: string[] = []
  
  // 检查守恒定律违背
  const conservationProperties = analysis.physical_properties.filter(p => p.type === 'conservation')
  const conservationConstraints = analysis.physical_constraints.filter(c => c.type === 'conservation_law')
  
  if (conservationProperties.length > 0 && conservationConstraints.length === 0) {
    warnings.push('发现守恒属性但缺少守恒约束')
  }
  
  // 检查非负约束
  const nonNegativeConstraints = analysis.physical_constraints.filter(c => c.type === 'non_negative')
  const numericProperties = analysis.physical_properties.filter(p => typeof p.value === 'number')
  
  numericProperties.forEach(prop => {
    if (prop.value < 0) {
      const hasNonNegConstraint = nonNegativeConstraints.some(c => c.entities.includes(prop.entity))
      if (!hasNonNegConstraint) {
        violations.push(`实体 ${prop.entity} 的数值为负但缺少非负约束`)
      }
    }
  })
  
  // 检查一致性评分
  if (analysis.consistency_score < 0.7) {
    warnings.push(`物性一致性评分较低: ${(analysis.consistency_score * 100).toFixed(1)}%`)
  }
  
  return {
    isConsistent: violations.length === 0,
    violations,
    warnings
  }
}

export default {
  getPhysicalPropertyDemo,
  getPhysicalPropertyAnalysis,
  getBackendDrivenConfig,
  getPhysicalPropertyTypes,
  getPhysicalConstraintTypes,
  validatePhysicalConsistency
}