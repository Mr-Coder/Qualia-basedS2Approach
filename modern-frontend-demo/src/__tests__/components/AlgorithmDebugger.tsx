import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { algorithmAPI, AlgorithmExecution, AlgorithmExecutionHistory } from '@/services/algorithmAPI'

interface AlgorithmDebuggerProps {
  onExecutionSelect?: (execution: AlgorithmExecution) => void
}

const AlgorithmDebugger: React.FC<AlgorithmDebuggerProps> = ({ onExecutionSelect }) => {
  const [currentExecution, setCurrentExecution] = useState<AlgorithmExecution | null>(null)
  const [executionHistory, setExecutionHistory] = useState<AlgorithmExecutionHistory[]>([])
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<string | null>(null)
  const [debugMode, setDebugMode] = useState<'overview' | 'step-by-step' | 'performance'>('overview')
  const [currentStepIndex, setCurrentStepIndex] = useState(0)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    const [execution, history] = await Promise.all([
      algorithmAPI.getLatestExecution(),
      algorithmAPI.getExecutionHistory(20)
    ])
    
    if (execution) {
      setCurrentExecution(execution)
      onExecutionSelect?.(execution)
    }
    setExecutionHistory(history)
  }

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString()
  }

  const getPerformanceColor = (value: number, max: number) => {
    const ratio = value / max
    if (ratio < 0.3) return 'text-red-500'
    if (ratio < 0.7) return 'text-yellow-500'
    return 'text-green-500'
  }

  const renderOverview = () => {
    if (!currentExecution) return null

    const stats = algorithmAPI.calculateExecutionStats(currentExecution)

    return (
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* 执行概览 */}
          <div className="bg-blue-50 rounded-lg p-4 border">
            <h4 className="font-semibold text-blue-800 mb-3 flex items-center">
              📊 执行概览
              {stats.isQS2Enhanced && (
                <span className="ml-2 text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded-full">
                  QS²增强
                </span>
              )}
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>执行ID:</span>
                <span className="font-mono">{currentExecution.execution_id.slice(0, 12)}...</span>
              </div>
              <div className="flex justify-between">
                <span>总耗时:</span>
                <span className={getPerformanceColor(currentExecution.total_duration_ms, 1000)}>
                  {currentExecution.total_duration_ms.toFixed(0)}ms
                </span>
              </div>
              <div className="flex justify-between">
                <span>阶段数:</span>
                <span>{stats.totalStages}</span>
              </div>
              <div className="flex justify-between">
                <span>平均置信度:</span>
                <span className={getPerformanceColor(stats.averageConfidence, 1)}>
                  {(stats.averageConfidence * 100).toFixed(1)}%
                </span>
              </div>
              {stats.isQS2Enhanced && (
                <div className="flex justify-between">
                  <span>算法类型:</span>
                  <span className="text-purple-600 font-medium">QS²语义增强</span>
                </div>
              )}
            </div>
          </div>

          {/* 发现统计 */}
          <div className="bg-green-50 rounded-lg p-4 border">
            <h4 className="font-semibold text-green-800 mb-3">🔍 发现统计</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>发现实体:</span>
                <span className="font-bold text-green-600">{stats.totalEntities}</span>
              </div>
              <div className="flex justify-between">
                <span>发现关系:</span>
                <span className="font-bold text-green-600">{stats.totalRelations}</span>
              </div>
              <div className="flex justify-between">
                <span>实体/关系比:</span>
                <span>{stats.totalEntities > 0 ? (stats.totalRelations / stats.totalEntities).toFixed(2) : '0'}</span>
              </div>
              <div className="flex justify-between">
                <span>性能评分:</span>
                <span className={getPerformanceColor(stats.performanceScore, 100)}>
                  {stats.performanceScore}/100
                </span>
              </div>
            </div>
          </div>

          {/* 错误信息 */}
          <div className="bg-red-50 rounded-lg p-4 border">
            <h4 className="font-semibold text-red-800 mb-3">⚠️ 错误与警告</h4>
            {currentExecution.error_info ? (
              <div className="text-sm">
                <div className="text-red-600 font-medium">{currentExecution.error_info.error_type}</div>
                <div className="text-red-500 text-xs mt-1">{currentExecution.error_info.error_message}</div>
              </div>
            ) : (
              <div className="text-green-600 text-sm">✅ 执行成功，无错误</div>
            )}
          </div>
        </div>

        {/* 阶段性能对比 */}
        <div className="bg-white rounded-lg border p-4">
          <h4 className="font-semibold text-gray-800 mb-3">⚡ 阶段性能对比</h4>
          <div className="space-y-3">
            {currentExecution.stages.map((stage, index) => (
              <div key={stage.stage_id} className="flex items-center space-x-4">
                <div className="w-24 text-sm text-gray-600">{stage.stage_name}</div>
                <div className="flex-1 bg-gray-200 rounded-full h-3 relative">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(stage.duration_ms / Math.max(...currentExecution.stages.map(s => s.duration_ms))) * 100}%` }}
                    transition={{ duration: 0.8, delay: index * 0.1 }}
                    className="bg-blue-500 h-3 rounded-full"
                  />
                  <span className="absolute right-2 top-0 text-xs text-gray-600">
                    {stage.duration_ms.toFixed(0)}ms
                  </span>
                </div>
                <div className="w-16 text-sm text-gray-600">
                  {(stage.confidence * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const renderStepByStep = () => {
    if (!currentExecution || currentExecution.stages.length === 0) return null

    const currentStage = currentExecution.stages[currentStepIndex]

    return (
      <div className="space-y-4">
        {/* 步骤导航 */}
        <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4">
          <button
            onClick={() => setCurrentStepIndex(Math.max(0, currentStepIndex - 1))}
            disabled={currentStepIndex === 0}
            className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            ← 上一步
          </button>
          
          <div className="text-center">
            <div className="text-lg font-bold text-gray-800">
              步骤 {currentStepIndex + 1} / {currentExecution.stages.length}
            </div>
            <div className="text-sm text-gray-600">{currentStage.stage_name}</div>
          </div>
          
          <button
            onClick={() => setCurrentStepIndex(Math.min(currentExecution.stages.length - 1, currentStepIndex + 1))}
            disabled={currentStepIndex === currentExecution.stages.length - 1}
            className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            下一步 →
          </button>
        </div>

        {/* 当前步骤详情 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white rounded-lg border p-4">
            <h4 className="font-semibold text-gray-800 mb-3">📥 输入数据</h4>
            <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto max-h-48">
              {JSON.stringify(currentStage.input_data, null, 2)}
            </pre>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <h4 className="font-semibold text-gray-800 mb-3">📤 输出数据</h4>
            <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto max-h-48">
              {JSON.stringify(currentStage.output_data, null, 2)}
            </pre>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <h4 className="font-semibold text-gray-800 mb-3">🧠 算法状态</h4>
            <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto max-h-48">
              {JSON.stringify(currentStage.algorithm_state, null, 2)}
            </pre>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <h4 className="font-semibold text-gray-800 mb-3">🎯 决策记录</h4>
            <div className="space-y-2 max-h-48 overflow-auto">
              {currentStage.decisions.map((decision, idx) => (
                <div key={idx} className="bg-blue-50 rounded p-2 text-sm">
                  <div className="font-medium text-blue-800">{decision.type}</div>
                  <div className="text-blue-600 text-xs">{JSON.stringify(decision, null, 2)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    )
  }

  const renderPerformance = () => {
    return (
      <div className="space-y-4">
        <div className="bg-white rounded-lg border p-4">
          <h4 className="font-semibold text-gray-800 mb-4">📈 执行历史性能分析</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 历史执行列表 */}
            <div>
              <h5 className="font-medium text-gray-700 mb-3">最近执行记录</h5>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {executionHistory.map((item) => (
                  <div
                    key={item.execution_id}
                    className={`p-3 border rounded cursor-pointer transition-colors ${
                      selectedHistoryItem === item.execution_id ? 'bg-blue-50 border-blue-300' : 'hover:bg-gray-50'
                    }`}
                    onClick={() => setSelectedHistoryItem(
                      selectedHistoryItem === item.execution_id ? null : item.execution_id
                    )}
                  >
                    <div className="flex justify-between items-start mb-1">
                      <div className="text-sm font-medium text-gray-800">
                        {formatTimestamp(item.start_time)}
                      </div>
                      <div className={`text-xs px-2 py-1 rounded ${
                        item.has_error ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'
                      }`}>
                        {item.has_error ? '失败' : '成功'}
                      </div>
                    </div>
                    <div className="text-xs text-gray-600 mb-1">
                      {item.problem_text}
                    </div>
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>{item.total_duration_ms.toFixed(0)}ms</span>
                      <span>{item.stages_count} 阶段</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 性能统计 */}
            <div>
              <h5 className="font-medium text-gray-700 mb-3">性能统计</h5>
              <div className="space-y-3">
                <div className="bg-blue-50 rounded p-3">
                  <div className="text-sm font-medium text-blue-800">平均执行时间</div>
                  <div className="text-lg font-bold text-blue-600">
                    {executionHistory.length > 0 
                      ? (executionHistory.reduce((sum, item) => sum + item.total_duration_ms, 0) / executionHistory.length).toFixed(0) 
                      : 0}ms
                  </div>
                </div>
                
                <div className="bg-green-50 rounded p-3">
                  <div className="text-sm font-medium text-green-800">成功率</div>
                  <div className="text-lg font-bold text-green-600">
                    {executionHistory.length > 0 
                      ? ((executionHistory.filter(item => !item.has_error).length / executionHistory.length) * 100).toFixed(1)
                      : 0}%
                  </div>
                </div>
                
                <div className="bg-purple-50 rounded p-3">
                  <div className="text-sm font-medium text-purple-800">最快执行</div>
                  <div className="text-lg font-bold text-purple-600">
                    {executionHistory.length > 0 
                      ? Math.min(...executionHistory.map(item => item.total_duration_ms)).toFixed(0)
                      : 0}ms
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* 调试模式切换 */}
      <div className="flex items-center space-x-1 bg-gray-100 rounded-lg p-1">
        {[
          { key: 'overview', label: '概览', icon: '📊' },
          { key: 'step-by-step', label: '步进调试', icon: '🔍' },
          { key: 'performance', label: '性能分析', icon: '📈' }
        ].map((mode) => (
          <button
            key={mode.key}
            onClick={() => setDebugMode(mode.key as any)}
            className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              debugMode === mode.key
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            {mode.icon} {mode.label}
          </button>
        ))}
      </div>

      {/* 内容区域 */}
      <motion.div
        key={debugMode}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {debugMode === 'overview' && renderOverview()}
        {debugMode === 'step-by-step' && renderStepByStep()}
        {debugMode === 'performance' && renderPerformance()}
      </motion.div>
    </div>
  )
}

export default AlgorithmDebugger