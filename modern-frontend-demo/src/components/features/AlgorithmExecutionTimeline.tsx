import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { algorithmAPI, AlgorithmExecution, AlgorithmStageSnapshot } from '@/services/algorithmAPI'

interface AlgorithmExecutionTimelineProps {
  problemText?: string
  autoRefresh?: boolean
  refreshInterval?: number
}

const AlgorithmExecutionTimeline: React.FC<AlgorithmExecutionTimelineProps> = ({
  problemText,
  autoRefresh = true,
  refreshInterval = 2000
}) => {
  const [execution, setExecution] = useState<AlgorithmExecution | null>(null)
  const [selectedStage, setSelectedStage] = useState<AlgorithmStageSnapshot | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [stats, setStats] = useState<any>(null)
  const stopPollingRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    // 初始加载
    loadLatestExecution()

    // 开始轮询
    if (autoRefresh) {
      stopPollingRef.current = algorithmAPI.startExecutionPolling(
        (newExecution) => {
          if (newExecution) {
            setExecution(newExecution)
            setStats(algorithmAPI.calculateExecutionStats(newExecution))
          }
          setIsLoading(false)
        },
        refreshInterval
      )
    }

    return () => {
      if (stopPollingRef.current) {
        stopPollingRef.current()
      }
    }
  }, [autoRefresh, refreshInterval])

  const loadLatestExecution = async () => {
    setIsLoading(true)
    const latestExecution = await algorithmAPI.getLatestExecution()
    if (latestExecution) {
      setExecution(latestExecution)
      setStats(algorithmAPI.calculateExecutionStats(latestExecution))
    }
    setIsLoading(false)
  }

  const getStageIcon = (stageName: string) => {
    const icons = {
      '实体提取': '🔍',
      '语义结构构建': '🧠', 
      '关系发现': '🔗',
      '后处理优化': '⚡'
    }
    return icons[stageName] || '📊'
  }

  const getStageColor = (stageName: string, confidence: number) => {
    const baseColors = {
      '实体提取': 'bg-blue-500',
      '语义结构构建': 'bg-purple-500',
      '关系发现': 'bg-green-500', 
      '后处理优化': 'bg-orange-500'
    }
    
    const baseColor = baseColors[stageName] || 'bg-gray-500'
    
    // 根据置信度调整透明度
    if (confidence < 0.5) return baseColor.replace('500', '300')
    if (confidence < 0.7) return baseColor.replace('500', '400')
    return baseColor
  }

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
        <span className="ml-3 text-gray-600">加载算法执行数据...</span>
      </div>
    )
  }

  if (!execution) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-500 mb-4">暂无算法执行数据</div>
        <button 
          onClick={loadLatestExecution}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          刷新数据
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* 执行概览 */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 border">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-gray-800">🧠 IRD算法执行时序</h2>
          <div className="flex items-center space-x-4 text-sm">
            <span className="text-gray-600">执行ID: {execution.execution_id.slice(0, 8)}</span>
            <span className="text-gray-600">总耗时: {formatDuration(execution.total_duration_ms)}</span>
            {stats && (
              <span className="text-green-600 font-medium">性能评分: {stats.performanceScore}</span>
            )}
          </div>
        </div>
        
        <div className="text-sm text-gray-700 mb-4">
          <span className="font-medium">问题: </span>
          {execution.problem_text}
        </div>

        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="bg-white rounded p-3 border">
              <div className="text-lg font-bold text-blue-600">{stats.totalStages}</div>
              <div className="text-xs text-gray-600">执行阶段</div>
            </div>
            <div className="bg-white rounded p-3 border">
              <div className="text-lg font-bold text-green-600">{(stats.averageConfidence * 100).toFixed(0)}%</div>
              <div className="text-xs text-gray-600">平均置信度</div>
            </div>
            <div className="bg-white rounded p-3 border">
              <div className="text-lg font-bold text-purple-600">{stats.totalEntities}</div>
              <div className="text-xs text-gray-600">发现实体</div>
            </div>
            <div className="bg-white rounded p-3 border">
              <div className="text-lg font-bold text-orange-600">{stats.totalRelations}</div>
              <div className="text-xs text-gray-600">发现关系</div>
            </div>
          </div>
        )}
      </div>

      {/* 时序图 */}
      <div className="bg-white rounded-lg border p-6">
        <h3 className="font-semibold text-gray-800 mb-4">算法执行时序图</h3>
        
        <div className="relative">
          {/* 时间轴 */}
          <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-300"></div>
          
          <div className="space-y-6">
            {execution.stages.map((stage, index) => (
              <motion.div
                key={stage.stage_id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="relative flex items-start"
              >
                {/* 时间点 */}
                <div className={`w-16 h-16 rounded-full ${getStageColor(stage.stage_name, stage.confidence)} 
                              flex items-center justify-center text-white text-xl font-bold shadow-lg z-10 cursor-pointer
                              hover:scale-110 transition-transform`}
                     onClick={() => setSelectedStage(selectedStage?.stage_id === stage.stage_id ? null : stage)}>
                  {getStageIcon(stage.stage_name)}
                </div>

                {/* 阶段信息 */}
                <div className="ml-6 flex-1 bg-gray-50 rounded-lg p-4 border">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-gray-800">{stage.stage_name}</h4>
                    <div className="flex items-center space-x-3 text-sm">
                      <span className="text-blue-600">耗时: {formatDuration(stage.duration_ms)}</span>
                      <span className="text-green-600">置信度: {(stage.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  {/* 进度条 */}
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
                    <motion.div 
                      className="bg-blue-500 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${stage.confidence * 100}%` }}
                      transition={{ duration: 0.8, delay: index * 0.1 }}
                    />
                  </div>

                  {/* 输出摘要 */}
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">输入: </span>
                      <span className="text-gray-700">
                        {Object.keys(stage.input_data).length} 项
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">输出: </span>
                      <span className="text-gray-700">
                        {Object.keys(stage.output_data).length} 项
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">决策: </span>
                      <span className="text-gray-700">
                        {stage.decisions.length} 个
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* 阶段详情面板 */}
      <AnimatePresence>
        {selectedStage && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-white rounded-lg border p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-gray-800">
                {getStageIcon(selectedStage.stage_name)} {selectedStage.stage_name} - 详细信息
              </h3>
              <button 
                onClick={() => setSelectedStage(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* 算法状态 */}
              <div>
                <h4 className="font-medium text-gray-700 mb-2">算法状态</h4>
                <div className="bg-gray-50 rounded p-3 text-sm">
                  <pre className="whitespace-pre-wrap text-gray-600">
                    {JSON.stringify(selectedStage.algorithm_state, null, 2)}
                  </pre>
                </div>
              </div>

              {/* 决策过程 */}
              <div>
                <h4 className="font-medium text-gray-700 mb-2">决策过程</h4>
                <div className="space-y-2">
                  {selectedStage.decisions.map((decision, idx) => (
                    <div key={idx} className="bg-blue-50 rounded p-2 text-sm">
                      <div className="font-medium text-blue-800">{decision.type}</div>
                      <div className="text-blue-600">{JSON.stringify(decision, null, 2)}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 性能指标 */}
              <div>
                <h4 className="font-medium text-gray-700 mb-2">性能指标</h4>
                <div className="bg-green-50 rounded p-3 text-sm">
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(selectedStage.metrics).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-green-700">{key}:</span>
                        <span className="text-green-800 font-medium">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 可视化元素 */}
              <div>
                <h4 className="font-medium text-gray-700 mb-2">可视化元素</h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {selectedStage.visual_elements.map((element, idx) => (
                    <div key={idx} className="bg-purple-50 rounded p-2 text-sm">
                      <span className="text-purple-600 font-medium">{element.type}: </span>
                      <span className="text-purple-800">{element.id || element.label || 'Unknown'}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default AlgorithmExecutionTimeline