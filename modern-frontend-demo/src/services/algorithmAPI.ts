/**
 * 算法执行数据API服务
 * 专门处理算法执行追踪和可视化数据
 */

interface AlgorithmStageSnapshot {
  stage_id: string
  stage_name: string
  timestamp: number
  duration_ms: number
  input_data: Record<string, any>
  output_data: Record<string, any>
  algorithm_state: Record<string, any>
  confidence: number
  decisions: Array<Record<string, any>>
  metrics: Record<string, any>
  visual_elements: Array<Record<string, any>>
}

interface AlgorithmExecution {
  execution_id: string
  problem_text: string
  start_time: number
  end_time: number
  total_duration_ms: number
  stages: AlgorithmStageSnapshot[]
  final_result: Record<string, any>
  execution_metrics: Record<string, any>
  error_info?: Record<string, any>
}

interface AlgorithmExecutionHistory {
  execution_id: string
  problem_text: string
  start_time: number
  total_duration_ms: number
  stages_count: number
  execution_metrics: Record<string, any>
  has_error: boolean
}

class AlgorithmAPIService {
  private baseURL = ''  // 使用相对路径，通过Vite代理

  /**
   * 获取最新的算法执行数据
   */
  async getLatestExecution(): Promise<AlgorithmExecution | null> {
    try {
      console.log('🔍 开始获取算法执行数据...')
      
      // 🎯 强制启用QS²模式 - 优先使用QS²演示数据
      console.log('🧠 强制启用QS²模式，获取演示数据...')
      const qs2DemoResponse = await fetch(`${this.baseURL}/api/qs2/demo`)
      if (qs2DemoResponse.ok) {
        const qs2Data = await qs2DemoResponse.json()
        if (qs2Data.success && qs2Data.data) {
          console.log('✅ QS²算法演示数据获取成功，启用QS²模式!')
          return this.convertQS2ToAlgorithmExecution(qs2Data)
        }
      } else {
        console.log('⚠️ QS²演示数据获取失败，尝试其他数据源...')
      }
      
      // 如果演示数据不可用，尝试获取真实QS²算法数据
      const qs2Response = await fetch(`${this.baseURL}/api/qs2/relations`)
      if (qs2Response.ok) {
        const qs2Data = await qs2Response.json()
        if (qs2Data.success && qs2Data.data) {
          console.log('🧠 检测到QS²算法数据，转换为算法执行格式')
          return this.convertQS2ToAlgorithmExecution(qs2Data)
        }
      }
      
      // 如果没有QS²数据，获取标准算法执行数据
      const response = await fetch(`${this.baseURL}/api/algorithm/execution`)
      const data = await response.json()
      
      if (data.success && data.data) {
        // 检查是否是QS²增强算法的特征
        const executionData = data.data
        const hasQualiaStages = executionData.stages?.some((stage: any) => 
          stage.stage_name?.includes('语义结构构建') || 
          stage.output_data?.qualia_structures ||
          stage.decisions?.some((d: any) => d.method === 'qualia_based')
        )
        
        if (hasQualiaStages) {
          console.log('🧠 检测到QS²算法特征，转换为QS²格式')
          console.log('📊 检测到的QS²特征:', {
            stages: executionData.stages?.map(s => s.stage_name),
            hasQualia: executionData.stages?.some(s => s.output_data?.qualia_structures),
            hasQualiaMethod: executionData.stages?.some(s => s.decisions?.some(d => d.method === 'qualia_based'))
          })
          
          // 添加QS²标识
          executionData.final_result = {
            ...executionData.final_result,
            is_qs2_enhanced: true,
            algorithm_type: 'QS2_Enhanced'
          }
          
          // 为每个阶段添加QS²标识
          if (executionData.stages) {
            executionData.stages = executionData.stages.map((stage: any) => ({
              ...stage,
              algorithm_state: {
                ...stage.algorithm_state,
                is_qs2_enhanced: true,
                algorithm_type: 'QS2_Enhanced'
              }
            }))
          }
        }
        
        return executionData as AlgorithmExecution
      }
      
      return null
    } catch (error) {
      console.error('获取算法执行数据失败:', error)
      return null
    }
  }

  /**
   * 获取算法执行历史
   */
  async getExecutionHistory(limit: number = 10): Promise<AlgorithmExecutionHistory[]> {
    try {
      const response = await fetch(`${this.baseURL}/api/algorithm/execution/history?limit=${limit}`)
      const data = await response.json()
      
      if (data.success && data.data) {
        return data.data as AlgorithmExecutionHistory[]
      }
      
      return []
    } catch (error) {
      console.error('获取算法执行历史失败:', error)
      return []
    }
  }

  /**
   * 实时轮询获取算法执行数据
   */
  startExecutionPolling(
    callback: (execution: AlgorithmExecution | null) => void,
    interval: number = 2000
  ): () => void {
    const pollInterval = setInterval(async () => {
      const execution = await this.getLatestExecution()
      callback(execution)
    }, interval)

    // 返回停止轮询的函数
    return () => clearInterval(pollInterval)
  }

  /**
   * 格式化执行阶段数据用于可视化
   */
  formatStagesForVisualization(stages: AlgorithmStageSnapshot[]): Array<{
    id: string
    name: string
    duration: number
    confidence: number
    entities: any[]
    relations: any[]
    decisions: any[]
    visualElements: any[]
  }> {
    return stages.map(stage => ({
      id: stage.stage_id,
      name: stage.stage_name,
      duration: stage.duration_ms,
      confidence: stage.confidence,
      entities: stage.output_data?.entities || stage.visual_elements?.filter(el => el.type === 'entity') || [],
      relations: stage.output_data?.relations || stage.visual_elements?.filter(el => el.type === 'relation') || [],
      decisions: stage.decisions || [],
      visualElements: stage.visual_elements || []
    }))
  }

  /**
   * 将QS²算法数据转换为算法执行格式
   */
  convertQS2ToAlgorithmExecution(qs2Data: any): AlgorithmExecution {
    const currentTime = Date.now() / 1000
    
    // 安全地获取algorithm_stages
    const algorithmStages = qs2Data.data?.algorithm_stages || []
    
    // 构建QS²算法的执行阶段
    const qs2Stages: AlgorithmStageSnapshot[] = algorithmStages.map((stage: any, index: number) => ({
      stage_id: `qs2_stage_${index}`,
      stage_name: stage.name,
      timestamp: currentTime + index,
      duration_ms: stage.duration_ms,
      input_data: {
        problem_text: qs2Data.problem_text,
        stage_index: index
      },
      output_data: {
        entities: stage.name === '实体提取' ? qs2Data.data.entities || [] : [],
        relations: stage.name === '关系发现' ? qs2Data.data.relations || [] : [],
        qualia_structures: stage.name === '语义结构构建' ? (qs2Data.data.entities || []).map((e: any) => e.qualia_roles) : [],
        compatibility_results: stage.name === '兼容性计算' ? (qs2Data.data.relations || []).map((r: any) => r.compatibility_result).filter(Boolean) : []
      },
      algorithm_state: {
        phase: stage.name,
        is_qs2_enhanced: true,
        algorithm_type: 'QS2_Enhanced'
      },
      confidence: stage.confidence,
      decisions: [
        {
          type: 'algorithm_choice',
          method: 'QS2_Qualia_Based',
          threshold: stage.name === '兼容性计算' ? 0.3 : 0.8
        }
      ],
      metrics: {
        processing_time_ms: stage.duration_ms,
        elements_processed: stage.visual_elements?.length || 0,
        confidence_level: stage.confidence
      },
      visual_elements: stage.visual_elements || []
    }))
    
    return {
      execution_id: qs2Data.execution_id || `qs2_${Date.now()}`,
      problem_text: qs2Data.problem_text || '数学问题',
      start_time: currentTime,
      end_time: currentTime + 1,
      total_duration_ms: algorithmStages.reduce((sum: number, stage: any) => sum + (stage.duration_ms || 0), 0),
      stages: qs2Stages,
      final_result: {
        entities: qs2Data.data.entities || [],
        relations: qs2Data.data.relations || [],
        algorithm_type: 'QS2_Enhanced',
        is_qs2_enhanced: true,
        qualia_structures: (qs2Data.data.entities || []).map((e: any) => ({
          entity: e.name,
          qualia_roles: e.qualia_roles,
          confidence: e.confidence
        }))
      },
      execution_metrics: {
        total_entities_discovered: (qs2Data.data.entities || []).length,
        total_relations_discovered: (qs2Data.data.relations || []).length,
        average_confidence: algorithmStages.length > 0 ? algorithmStages.reduce((sum: number, stage: any) => sum + stage.confidence, 0) / algorithmStages.length : 0,
        algorithm_enhancement: 'QS2_Qualia_Based_Semantic_Analysis'
      }
    }
  }

  /**
   * 计算算法执行统计信息
   */
  calculateExecutionStats(execution: AlgorithmExecution): {
    totalStages: number
    averageConfidence: number
    totalEntities: number
    totalRelations: number
    performanceScore: number
    isQS2Enhanced?: boolean
  } {
    const stages = execution.stages
    const avgConfidence = stages.length > 0 
      ? stages.reduce((sum, stage) => sum + stage.confidence, 0) / stages.length 
      : 0

    const finalResult = execution.final_result
    const totalEntities = finalResult?.entities?.length || 0
    const totalRelations = finalResult?.relations?.length || 0
    const isQS2Enhanced = finalResult?.is_qs2_enhanced || false

    // 性能评分：基于执行时间、置信度和发现的关系数量
    const timeScore = Math.max(0, 100 - (execution.total_duration_ms / 10)) // 时间越短分数越高
    const confidenceScore = avgConfidence * 100
    const complexityScore = Math.min(100, (totalRelations * (isQS2Enhanced ? 15 : 10))) // QS²算法发现能力更强
    const performanceScore = (timeScore + confidenceScore + complexityScore) / 3

    return {
      totalStages: stages.length,
      averageConfidence: avgConfidence,
      totalEntities,
      totalRelations,
      performanceScore: Math.round(performanceScore),
      isQS2Enhanced
    }
  }
}

export const algorithmAPI = new AlgorithmAPIService()

export type {
  AlgorithmStageSnapshot,
  AlgorithmExecution,
  AlgorithmExecutionHistory
}