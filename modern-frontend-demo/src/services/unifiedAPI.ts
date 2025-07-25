import { Strategy, SolveResult } from '@/stores/problemStore'

// 统一API配置
class APIConfig {
  private static instance: APIConfig
  
  private constructor() {}
  
  static getInstance(): APIConfig {
    if (!APIConfig.instance) {
      APIConfig.instance = new APIConfig()
    }
    return APIConfig.instance
  }
  
  // 自动检测可用的API端点
  getAPIBaseURL(): string {
    // 安全获取环境变量，避免process未定义错误
    let devMode = false
    let apiUrl = 'http://localhost:8000'
    
    try {
      // 优先使用Vite环境变量
      devMode = import.meta.env?.DEV === true || 
                (typeof process !== 'undefined' && process.env?.NODE_ENV === 'development')
      
      // 获取API URL
      apiUrl = import.meta.env?.VITE_API_URL || 
               (typeof process !== 'undefined' ? process.env?.REACT_APP_API_URL : undefined) || 
               'http://localhost:8000'
    } catch (error) {
      console.warn('获取环境变量失败，使用默认值:', error)
    }
    
    // 开发环境使用Vite代理，避免CORS问题
    if (devMode) {
      // 使用相对路径，通过Vite代理转发到后端
      return ''
    }
    
    // 生产环境
    return apiUrl
  }
  
  getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    
    // 安全获取API密钥，避免process未定义错误
    try {
      const apiKey = import.meta.env?.VITE_API_KEY || (typeof process !== 'undefined' ? process.env?.REACT_APP_API_KEY : undefined)
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`
      }
    } catch (error) {
      console.warn('获取API密钥失败:', error)
    }
    
    return headers
  }
  
  getTimeout(): number {
    return 30000 // 30秒超时
  }
}

// 统一API响应类型
export interface UnifiedAPIResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

// 标准化的求解请求
export interface StandardSolveRequest {
  problem: string
  strategy: Strategy
}

// 标准化的求解响应
export interface StandardSolveResponse {
  success: boolean
  answer: string
  confidence: number
  strategy_used: string
  reasoning_steps: string[]
  execution_time: number
  entity_relationship_diagram?: {
    entities: Array<{
      id: string
      name: string
      type: 'person' | 'object' | 'money' | 'concept'
    }>
    relationships: Array<{
      source: string
      target: string
      type: string
      label: string
      weight?: number
    }>
    implicit_constraints: string[]
  }
  enhanced_analysis?: {
    algorithm?: string
    relations_found?: number
    semantic_depth?: number
    processing_method?: string
  }
  engine_used?: string
  error?: string
}

// 统一API客户端类
class UnifiedAPIClient {
  private config: APIConfig
  private baseURL: string
  
  constructor() {
    this.config = APIConfig.getInstance()
    this.baseURL = this.config.getAPIBaseURL()
  }
  
  // 通用请求方法
  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<UnifiedAPIResponse<T>> {
    try {
      const url = `${this.baseURL}${endpoint}`
      const headers = this.config.getHeaders()
      
      const response = await fetch(url, {
        ...options,
        headers: {
          ...headers,
          ...options.headers,
        },
        timeout: this.config.getTimeout(),
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const data = await response.json()
      
      return {
        success: true,
        data,
      }
      
    } catch (error) {
      console.error(`API请求失败 [${endpoint}]:`, error)
      
      return {
        success: false,
        error: error instanceof Error ? error.message : '未知错误',
      }
    }
  }
  
  // 健康检查
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.makeRequest('/api/health')
      return response.success
    } catch {
      return false
    }
  }
  
  // 主要求解接口
  async solveProblem(request: StandardSolveRequest): Promise<StandardSolveResponse> {
    console.log('🧠 发送求解请求:', request)
    
    const response = await this.makeRequest<StandardSolveResponse>('/api/solve', {
      method: 'POST',
      body: JSON.stringify(request),
    })
    
    if (response.success && response.data) {
      // 标准化响应格式
      const standardized = this.standardizeResponse(response.data)
      console.log('✅ 求解成功:', standardized)
      return standardized
    } else {
      console.error('❌ 求解失败:', response.error)
      return {
        success: false,
        answer: '求解失败',
        confidence: 0,
        strategy_used: 'error',
        reasoning_steps: ['API调用失败'],
        execution_time: 0,
        error: response.error || '未知错误',
      }
    }
  }
  
  // 标准化不同后端的响应格式
  private standardizeResponse(rawResponse: any): StandardSolveResponse {
    // 处理不同的响应格式
    const response: StandardSolveResponse = {
      success: rawResponse.success ?? true,
      answer: rawResponse.answer || '无答案',
      confidence: rawResponse.confidence || 0.5,
      strategy_used: rawResponse.strategy_used || 'auto',
      reasoning_steps: this.normalizeReasoningSteps(rawResponse.reasoning_steps || []),
      execution_time: rawResponse.execution_time || 0,
      engine_used: rawResponse.engine_used || 'unknown',
    }
    
    // 处理实体关系图
    if (rawResponse.entity_relationship_diagram) {
      response.entity_relationship_diagram = this.normalizeERD(
        rawResponse.entity_relationship_diagram
      )
    }
    
    // 处理增强分析
    if (rawResponse.enhanced_analysis) {
      response.enhanced_analysis = rawResponse.enhanced_analysis
    }
    
    return response
  }
  
  private normalizeReasoningSteps(steps: any[]): string[] {
    return steps.map(step => {
      if (typeof step === 'string') {
        return step
      } else if (step && step.description) {
        return step.description
      } else if (step && step.action) {
        return `${step.action}: ${step.description || ''}`
      } else {
        return String(step)
      }
    })
  }
  
  private normalizeERD(erd: any) {
    return {
      entities: (erd.entities || []).map((entity: any) => ({
        id: entity.id,
        name: entity.name || entity.id,
        type: this.normalizeEntityType(entity.type),
      })),
      relationships: (erd.relationships || []).map((rel: any) => ({
        source: rel.source || rel.from,
        target: rel.target || rel.to,
        type: rel.type,
        label: rel.label || rel.type,
        weight: rel.weight || 1,
      })),
      implicit_constraints: erd.implicit_constraints || [],
    }
  }
  
  private normalizeEntityType(type: string): 'person' | 'object' | 'money' | 'concept' {
    const lowerType = type?.toLowerCase() || ''
    
    if (['person', 'people', '人'].some(t => lowerType.includes(t))) {
      return 'person'
    } else if (['object', 'item', '物体', '东西'].some(t => lowerType.includes(t))) {
      return 'object'
    } else if (['money', 'currency', '钱', '元'].some(t => lowerType.includes(t))) {
      return 'money'
    } else {
      return 'concept'
    }
  }
  
  // 获取可用策略
  async getStrategies(): Promise<any[]> {
    const response = await this.makeRequest('/api/strategies')
    return response.data?.strategies || []
  }
  
  // 获取系统信息
  async getSystemInfo(): Promise<any> {
    const response = await this.makeRequest('/api/system')
    return response.data || {}
  }
}

// 导出全局API客户端实例
export const apiClient = new UnifiedAPIClient()

// 向后兼容的函数接口
export const solveProblem = async (request: StandardSolveRequest): Promise<SolveResult> => {
  const response = await apiClient.solveProblem(request)
  
  // 转换为原有的SolveResult格式
  return {
    answer: response.answer,
    confidence: response.confidence,
    strategy: response.strategy_used as Strategy,
    steps: response.reasoning_steps,
    entities: response.entity_relationship_diagram?.entities || [],
    relationships: response.entity_relationship_diagram?.relationships || [],
    constraints: response.entity_relationship_diagram?.implicit_constraints || [],
    processingTime: response.execution_time,
    enhancedInfo: response.enhanced_analysis,
  } as SolveResult
}

export const getSystemStatus = async () => {
  return await apiClient.getSystemInfo()
}

export const getStrategiesInfo = async () => {
  return await apiClient.getStrategies()
}

// 开发工具：API连接测试
export const testAPIConnection = async (): Promise<boolean> => {
  console.log('🔍 测试API连接...')
  
  const isHealthy = await apiClient.healthCheck()
  
  if (isHealthy) {
    console.log('✅ API连接正常')
  } else {
    console.warn('⚠️ API连接失败，可能需要启动后端服务器')
  }
  
  return isHealthy
}

export default apiClient
