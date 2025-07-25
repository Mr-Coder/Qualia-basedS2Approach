import React, { useState } from 'react'
import Layout from '@/components/layout/Layout'
import { useProblemStore } from '@/stores/problemStore'

const SafeAPIProcessor: React.FC = () => {
  const [apiResponse, setApiResponse] = useState<any>(null)
  const [processedData, setProcessedData] = useState<any>(null)
  const [processingError, setProcessingError] = useState<string | null>(null)
  
  const { 
    currentProblem, 
    selectedStrategy, 
    setSolveResult, 
    addToHistory 
  } = useProblemStore()

  const fetchAPIResponse = async () => {
    try {
      setProcessingError(null)
      
      const response = await fetch('/api/solve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem: currentProblem || '小明有3个苹果，小红有2个苹果，一共有多少个苹果？',
          strategy: selectedStrategy
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      setApiResponse(result)
      
    } catch (error) {
      setProcessingError(`API调用失败: ${String(error)}`)
    }
  }

  const processAPIData = () => {
    if (!apiResponse) return

    try {
      setProcessingError(null)
      
      // 安全的数据转换
      const safeResult = {
        answer: apiResponse.answer || '无答案',
        confidence: apiResponse.confidence || 0.5,
        strategy: apiResponse.strategy_used || apiResponse.strategy || 'cot',
        steps: Array.isArray(apiResponse.reasoning_steps) 
          ? apiResponse.reasoning_steps.map((step: any, index: number) => {
              if (typeof step === 'string') return step
              if (step.description) return step.description
              return `步骤${index + 1}: ${JSON.stringify(step)}`
            })
          : ['解题步骤处理中'],
        entities: [],
        relationships: [],
        constraints: [],
        processingTime: apiResponse.execution_time || 0
      }

      // 安全处理实体数据
      try {
        const diagram = apiResponse.entity_relationship_diagram || {}
        if (diagram.entities && Array.isArray(diagram.entities)) {
          safeResult.entities = diagram.entities.map((entity: any) => ({
            id: entity.id || `entity-${Math.random()}`,
            name: entity.id || entity.name || '未知实体',
            type: mapEntityType(entity.type || 'concept')
          }))
        }

        if (diagram.relationships && Array.isArray(diagram.relationships)) {
          safeResult.relationships = diagram.relationships.map((rel: any) => ({
            source: rel.from || rel.source || 'unknown',
            target: rel.to || rel.target || 'unknown',
            type: rel.type || '关系',
            weight: rel.weight || 1
          }))
        }
      } catch (entityError) {
        console.warn('实体数据处理失败:', entityError)
      }

      setProcessedData(safeResult)
      
    } catch (error) {
      setProcessingError(`数据处理失败: ${String(error)}`)
    }
  }

  const mapEntityType = (backendType: string): 'person' | 'object' | 'money' | 'concept' => {
    if (backendType === 'person') return 'person'
    if (backendType === 'object') return 'object'
    if (backendType === 'currency' || backendType === 'money') return 'money'
    return 'concept'
  }

  const applyToStore = () => {
    if (!processedData) return

    try {
      setProcessingError(null)
      
      // 分步应用到Store
      console.log('应用数据到Store:', processedData)
      setSolveResult(processedData)
      
      // 添加到历史记录 - 修复接口匹配问题
      const historyEntry = {
        id: `safe-${Date.now()}`,
        problem: currentProblem || '测试问题',
        answer: processedData.answer,
        strategy: processedData.strategy as 'auto' | 'cot' | 'got' | 'tot',
        timestamp: new Date(),
        confidence: processedData.confidence
      }
      
      console.log('添加历史记录:', historyEntry)
      addToHistory(historyEntry)
      
    } catch (error) {
      setProcessingError(`Store更新失败: ${String(error)}`)
    }
  }

  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔧 安全API数据处理器</h1>
        
        <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', border: '1px solid #0066cc' }}>
          <h3>目标：分步安全处理API响应</h3>
          <p>1. 获取API响应 → 2. 处理数据格式 → 3. 应用到Store</p>
        </div>

        <div style={{ display: 'flex', gap: '10px', margin: '15px 0', flexWrap: 'wrap' }}>
          <button 
            onClick={fetchAPIResponse}
            style={{ padding: '10px 20px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}
          >
            1. 获取API响应
          </button>
          <button 
            onClick={processAPIData}
            disabled={!apiResponse}
            style={{ 
              padding: '10px 20px', 
              backgroundColor: !apiResponse ? '#6c757d' : '#ffc107', 
              color: !apiResponse ? 'white' : 'black', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            2. 处理数据格式
          </button>
          <button 
            onClick={applyToStore}
            disabled={!processedData}
            style={{ 
              padding: '10px 20px', 
              backgroundColor: !processedData ? '#6c757d' : '#28a745', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            3. 应用到Store
          </button>
        </div>

        {processingError && (
          <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
            <h3>❌ 处理错误:</h3>
            <p>{processingError}</p>
          </div>
        )}

        {apiResponse && (
          <div style={{ background: '#e3f2fd', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
            <h3>📥 API原始响应:</h3>
            <details>
              <summary>查看原始数据</summary>
              <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '200px' }}>
                {JSON.stringify(apiResponse, null, 2)}
              </pre>
            </details>
          </div>
        )}

        {processedData && (
          <div style={{ background: '#e8f5e8', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
            <h3>⚙️ 处理后数据:</h3>
            <p>答案: {processedData.answer}</p>
            <p>置信度: {processedData.confidence}</p>
            <p>实体数量: {processedData.entities.length}</p>
            <p>关系数量: {processedData.relationships.length}</p>
            <details>
              <summary>查看处理后数据</summary>
              <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '200px' }}>
                {JSON.stringify(processedData, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>
    </Layout>
  )
}

export default SafeAPIProcessor