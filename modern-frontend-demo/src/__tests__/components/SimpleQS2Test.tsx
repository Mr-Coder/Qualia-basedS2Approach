import React, { useState, useEffect } from 'react'

// 简化的QS²测试组件
const SimpleQS2Test: React.FC = () => {
  const [qs2Data, setQS2Data] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const testQS2API = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      console.log('🧠 开始QS²API测试...')
      
      // 直接测试QS²API
      const response = await fetch('http://localhost:8000/api/qs2/demo')
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const data = await response.json()
      console.log('✅ QS²数据获取成功:', data)
      
      setQS2Data(data)
      
    } catch (err) {
      console.error('❌ QS²API测试失败:', err)
      setError(err instanceof Error ? err.message : '未知错误')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    // 组件加载时自动测试
    testQS2API()
  }, [])

  return (
    <div className="p-6 bg-white rounded-lg border">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">🧠 QS²API测试组件</h2>
        <button
          onClick={testQS2API}
          disabled={isLoading}
          className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
        >
          {isLoading ? '测试中...' : '重新测试'}
        </button>
      </div>

      {/* 加载状态 */}
      {isLoading && (
        <div className="text-center py-8">
          <div className="text-4xl mb-2 animate-pulse">🔄</div>
          <div className="text-purple-600">正在测试QS²API连接...</div>
        </div>
      )}

      {/* 错误状态 */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-4 mb-4">
          <div className="text-red-800 font-medium">❌ API测试失败</div>
          <div className="text-red-600 text-sm mt-1">{error}</div>
        </div>
      )}

      {/* 成功状态 */}
      {qs2Data && (
        <div className="space-y-4">
          <div className="bg-green-50 border border-green-200 rounded p-4">
            <div className="text-green-800 font-medium">✅ QS²API连接成功</div>
            <div className="text-green-600 text-sm mt-1">
              执行ID: {qs2Data.execution_id}
            </div>
          </div>

          {/* 数据概览 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-3 rounded">
              <div className="text-blue-800 font-medium">实体数量</div>
              <div className="text-2xl font-bold text-blue-600">
                {qs2Data.data?.entities?.length || 0}
              </div>
            </div>
            
            <div className="bg-purple-50 p-3 rounded">
              <div className="text-purple-800 font-medium">关系数量</div>
              <div className="text-2xl font-bold text-purple-600">
                {qs2Data.data?.relationships?.length || 0}
              </div>
            </div>
            
            <div className="bg-orange-50 p-3 rounded">
              <div className="text-orange-800 font-medium">算法阶段</div>
              <div className="text-2xl font-bold text-orange-600">
                {qs2Data.data?.algorithm_stages?.length || 0}
              </div>
            </div>
          </div>

          {/* 问题文本 */}
          {qs2Data.problem_text && (
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-gray-800 font-medium mb-2">问题文本:</div>
              <div className="text-gray-600">{qs2Data.problem_text}</div>
            </div>
          )}

          {/* 实体列表 */}
          {qs2Data.data?.entities && qs2Data.data.entities.length > 0 && (
            <div className="bg-white border rounded p-4">
              <h3 className="font-medium mb-3">🎯 QS²实体列表</h3>
              <div className="space-y-2">
                {qs2Data.data.entities.map((entity: any, index: number) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div>
                      <span className="font-medium">{entity.name}</span>
                      <span className="text-sm text-gray-500 ml-2">({entity.type})</span>
                    </div>
                    <div className="text-sm text-purple-600">
                      置信度: {(entity.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 关系列表 */}
          {qs2Data.data?.relationships && qs2Data.data.relationships.length > 0 && (
            <div className="bg-white border rounded p-4">
              <h3 className="font-medium mb-3">🔗 QS²关系列表</h3>
              <div className="space-y-2">
                {qs2Data.data.relationships.map((relation: any, index: number) => (
                  <div key={index} className="p-2 bg-gray-50 rounded">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{relation.type}</span>
                      <span className="text-sm text-purple-600">
                        强度: {(relation.strength * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      {relation.source} → {relation.target}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 算法阶段 */}
          {qs2Data.data?.algorithm_stages && qs2Data.data.algorithm_stages.length > 0 && (
            <div className="bg-white border rounded p-4">
              <h3 className="font-medium mb-3">⚡ QS²算法执行阶段</h3>
              <div className="space-y-2">
                {qs2Data.data.algorithm_stages.map((stage: any, index: number) => (
                  <div key={index} className="p-2 bg-gray-50 rounded">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{stage.name}</span>
                      <span className="text-sm text-green-600">
                        {stage.duration_ms.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      置信度: {(stage.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default SimpleQS2Test