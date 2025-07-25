import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { useProblemStore } from '@/stores/problemStore'

interface ReasoningStepsVisualizationProps {
  className?: string
}

const ReasoningStepsVisualization: React.FC<ReasoningStepsVisualizationProps> = ({ className = '' }) => {
  const { solveResult } = useProblemStore()
  const [expandedStep, setExpandedStep] = useState<number | null>(null)

  // 如果没有解决结果，显示占位符
  if (!solveResult || !solveResult.steps || solveResult.steps.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center">
            🧠 完整推理过程展示
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-gray-500">
            <div className="text-6xl mb-4">🔍</div>
            <div className="text-lg font-medium mb-2">暂无推理步骤</div>
            <div className="text-sm">请先在智能求解模块解决一个数学问题</div>
          </div>
        </CardContent>
      </Card>
    )
  }

  // 步骤类型图标映射
  const getStepIcon = (stepText: string, index: number) => {
    if (stepText.includes('实体提取') || stepText.includes('Qualia构建')) return '🔍'
    if (stepText.includes('语义结构分析') || stepText.includes('语义模式')) return '🧩'
    if (stepText.includes('隐式关系发现') || stepText.includes('关系发现')) return '🔗'
    if (stepText.includes('数学运算') || stepText.includes('运算执行')) return '🧮'
    if (stepText.includes('逻辑推理验证') || stepText.includes('逻辑验证')) return '✅'
    if (stepText.includes('结果综合') || stepText.includes('综合')) return '🎯'
    return `${index + 1}️⃣`
  }

  // 获取步骤颜色
  const getStepColor = (index: number) => {
    const colors = [
      'border-l-blue-500 bg-blue-50',
      'border-l-green-500 bg-green-50', 
      'border-l-purple-500 bg-purple-50',
      'border-l-orange-500 bg-orange-50',
      'border-l-red-500 bg-red-50',
      'border-l-indigo-500 bg-indigo-50'
    ]
    return colors[index % colors.length]
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            🧠 完整推理过程展示
            <span className="ml-2 text-sm text-gray-500">
              ({solveResult.steps.length}步)
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">
              置信度: {(solveResult.confidence * 100).toFixed(1)}%
            </span>
            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
              策略: {solveResult.strategy}
            </span>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {solveResult.steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`border-l-4 ${getStepColor(index)} rounded-lg p-4 cursor-pointer hover:shadow-md transition-shadow`}
              onClick={() => setExpandedStep(expandedStep === index ? null : index)}
            >
              <div className="flex items-start space-x-3">
                <div className="text-2xl flex-shrink-0">
                  {getStepIcon(step, index)}
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium text-gray-800">
                      步骤 {index + 1}
                    </div>
                    <div className="text-xs text-gray-500">
                      {expandedStep === index ? '收起' : '展开详情'}
                    </div>
                  </div>
                  
                  <div className="text-sm text-gray-700 leading-relaxed">
                    {step}
                  </div>
                  
                  {/* 展开的详细信息 */}
                  <AnimatePresence>
                    {expandedStep === index && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-3 pt-3 border-t border-gray-200"
                      >
                        <div className="text-xs text-gray-600 space-y-2">
                          <div>
                            <strong>执行顺序:</strong> 第 {index + 1} 步 / 共 {solveResult.steps.length} 步
                          </div>
                          <div>
                            <strong>推理类型:</strong> {
                              step.includes('实体') ? 'QS²语义实体提取' :
                              step.includes('语义') ? 'Qualia四维结构分析' :
                              step.includes('关系') ? 'IRD隐式关系发现' :
                              step.includes('运算') ? '数学计算执行' :
                              step.includes('验证') ? '逻辑一致性验证' :
                              step.includes('综合') ? 'COT-DIR结果综合' :
                              '通用推理步骤'
                            }
                          </div>
                          {solveResult.processingTime && (
                            <div>
                              <strong>估计耗时:</strong> {(solveResult.processingTime / solveResult.steps.length).toFixed(2)}ms
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* 推理过程总结 */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-800 mb-2">🎯 推理过程总结</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div>
              <span className="font-medium text-gray-700">总步数:</span>
              <span className="text-gray-600 ml-1">{solveResult.steps.length}步</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">推理策略:</span>
              <span className="text-gray-600 ml-1">{solveResult.strategy}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">最终置信度:</span>
              <span className="text-gray-600 ml-1">{(solveResult.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default ReasoningStepsVisualization