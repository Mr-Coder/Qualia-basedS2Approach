import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Textarea'
import { Select, SelectOption } from '@/components/ui/Select'
import { useProblemStore, SolveResult } from '@/stores/problemStore'
import { solveProblem } from '@/services/api'
import { generateId } from '@/utils/helpers'
import EntityRelationshipDiagram from './EntityRelationshipDiagram'

const strategyOptions: SelectOption[] = [
  { value: 'auto', label: '🤖 自动选择' },
  { value: 'cot', label: '🔗 思维链推理 (COT)' },
  { value: 'got', label: '🕸️ 思维图推理 (GOT)' },
  { value: 'tot', label: '🌳 思维树推理 (TOT)' }
]

const exampleProblems = [
  "小明有10个苹果，他给了小红3个，又买了5个，请问小明现在有多少个苹果？",
  "一个长方形的长是12厘米，宽是8厘米，求这个长方形的面积。",
  "班级里有30名学生，其中男生比女生多4人，请问男生和女生各有多少人？"
]

export const ProblemSolver: React.FC = () => {
  const {
    currentProblem,
    selectedStrategy,
    solveResult,
    isLoading,
    error,
    setProblem,
    setStrategy,
    setSolveResult,
    setLoading,
    setError,
    addToHistory
  } = useProblemStore()

  const [localProblem, setLocalProblem] = useState(currentProblem)

  const handleSolve = async () => {
    if (!localProblem.trim()) {
      setError('请输入问题')
      return
    }

    setProblem(localProblem)
    setLoading(true)
    setError(null)

    try {
      const result = await solveProblem({
        problem: localProblem,
        strategy: selectedStrategy
      })

      setSolveResult(result)
      addToHistory({
        id: generateId(),
        problem: localProblem,
        strategy: selectedStrategy,
        result,
        timestamp: new Date()
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : '解题失败')
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (example: string) => {
    setLocalProblem(example)
    setProblem(example)
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* 主要输入区域 */}
      <Card>
        <CardHeader>
          <CardTitle>🧠 智能数学问题解决器</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* 示例问题 */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-700">快速选择示例：</h3>
              <div className="grid grid-cols-1 gap-2">
                {exampleProblems.map((example, index) => (
                  <motion.button
                    key={index}
                    onClick={() => handleExampleClick(example)}
                    className="text-left p-3 rounded-lg border border-gray-200 hover:border-purple-300 hover:bg-purple-50 transition-colors text-sm"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    {example}
                  </motion.button>
                ))}
              </div>
            </div>

            {/* 问题输入 */}
            <Textarea
              label="输入数学问题"
              placeholder="请输入您要解决的数学问题..."
              value={localProblem}
              onChange={(e) => setLocalProblem(e.target.value)}
              className="min-h-[120px]"
              error={error}
            />

            {/* 策略选择和解题按钮 */}
            <div className="flex flex-col sm:flex-row gap-4 items-end">
              <div className="flex-1">
                <Select
                  label="选择推理策略"
                  value={selectedStrategy}
                  onChange={(e) => setStrategy(e.target.value as any)}
                  options={strategyOptions}
                />
              </div>
              
              <Button
                onClick={handleSolve}
                loading={isLoading}
                disabled={!localProblem.trim()}
                size="lg"
                className="w-full sm:w-auto"
              >
                {isLoading ? '正在解题...' : '🚀 开始解题'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 结果展示 */}
      <AnimatePresence>
        {solveResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <div className="grid grid-cols-1 gap-6">
              <SolveResultDisplay result={solveResult} />
              <EntityRelationshipDiagram
                entities={solveResult.entities}
                relationships={solveResult.relationships}
                width={700}
                height={400}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// 结果展示组件
interface SolveResultDisplayProps {
  result: SolveResult
}

const SolveResultDisplay: React.FC<SolveResultDisplayProps> = ({ result }) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* 答案和基本信息 */}
      <Card>
        <CardHeader>
          <CardTitle>✅ 解答结果</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-800">
                答案：{result.answer}
              </div>
              <div className="text-sm text-green-600 mt-2">
                置信度：{(result.confidence * 100).toFixed(1)}%
              </div>
              {result.processingTime && (
                <div className="text-sm text-green-600">
                  处理时间：{result.processingTime.toFixed(2)}秒
                </div>
              )}
            </div>
            
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="text-sm text-blue-600">
                使用策略：{result.strategy === 'auto' ? '自动选择' : 
                        result.strategy === 'cot' ? '思维链推理' :
                        result.strategy === 'got' ? '思维图推理' : '思维树推理'}
              </div>
            </div>

            {/* 增强引擎信息 */}
            {result.enhancedInfo && (
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <div className="text-sm font-medium text-purple-800 mb-2">
                  🚀 增强引擎分析
                </div>
                <div className="space-y-1 text-sm text-purple-600">
                  <div>算法：{result.enhancedInfo.algorithm}</div>
                  <div>发现关系：{result.enhancedInfo.relationsFound}个</div>
                  <div>语义深度：{(result.enhancedInfo.semanticDepth * 100).toFixed(1)}%</div>
                  <div>处理方法：{result.enhancedInfo.processingMethod}</div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 解题步骤 */}
      <Card>
        <CardHeader>
          <CardTitle>📋 解题步骤</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {result.steps.map((step: string, index: number) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-start space-x-3"
              >
                <div className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-medium">
                  {index + 1}
                </div>
                <div className="flex-1 text-sm text-gray-700">
                  {step}
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 实体关系统计 */}
      {(result.entities.length > 0 || result.relationships.length > 0) && (
        <Card>
          <CardHeader>
            <CardTitle>📊 关系分析统计</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-lg font-bold text-gray-800">
                  {result.entities.length}
                </div>
                <div className="text-sm text-gray-600">实体数量</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-lg font-bold text-gray-800">
                  {result.relationships.length}
                </div>
                <div className="text-sm text-gray-600">关系数量</div>
              </div>
            </div>
            
            {/* 实体类型分布 */}
            <div className="mt-4">
              <div className="text-sm font-medium text-gray-700 mb-2">实体类型分布：</div>
              <div className="flex flex-wrap gap-2">
                {Array.from(new Set(result.entities.map(e => e.type))).map(type => {
                  const count = result.entities.filter(e => e.type === type).length
                  return (
                    <span key={type} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                      {type}: {count}
                    </span>
                  )
                })}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 隐含约束 */}
      {result.constraints && result.constraints.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>⚠️ 隐含约束与发现</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {result.constraints.map((constraint: string, index: number) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-start space-x-3"
                >
                  <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
                  <div className="flex-1 text-sm text-gray-700">
                    {constraint}
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default ProblemSolver