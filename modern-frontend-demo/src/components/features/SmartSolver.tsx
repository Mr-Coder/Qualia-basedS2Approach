import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Textarea'
import { useProblemStore } from '@/stores/problemStore'
import { solveProblem } from '@/services/api'
import { generateId } from '@/utils/helpers'
import EntityRelationshipDiagram from './EntityRelationshipDiagram'
import { apiClient, testAPIConnection } from '@/services/unifiedAPI'


const strategyOptions = [
  { 
    id: 'auto', 
    name: '🤖 自动选择', 
    description: '系统根据问题特征自动选择最适合的推理策略',
    color: 'bg-purple-500'
  },
  { 
    id: 'cot', 
    name: '🔗 思维链推理 (COT)', 
    description: '逐步分解问题，建立清晰的推理链条',
    color: 'bg-blue-500'
  },
  { 
    id: 'got', 
    name: '🕸️ 思维图推理 (GOT)', 
    description: '构建网络拓扑，发现隐含连接关系',
    color: 'bg-green-500'
  },
  { 
    id: 'tot', 
    name: '🌳 思维树推理 (TOT)', 
    description: '多路径探索，层次化分析问题',
    color: 'bg-orange-500'
  }
]

const exampleProblems = [
  {
    id: 1,
    problem: "小明有10个苹果，他给了小红3个，又买了5个，请问小明现在有多少个苹果？",
    category: "算术问题",
    difficulty: "简单",
    actualResult: {
      answer: "7个",
      confidence: 78.7,
      executionTime: "0.60ms",
      stagesCompleted: 4,
      strategy: "COT-DIR高级推理",
      performance: "优秀"
    }
  },
  {
    id: 2,
    problem: "一个长方形的长是12厘米，宽是8厘米，求这个长方形的面积。",
    category: "几何问题",
    difficulty: "简单"
  },
  {
    id: 3,
    problem: "班级里有30名学生，其中男生比女生多4人，请问男生和女生各有多少人？",
    category: "应用题",
    difficulty: "中等"
  },
  {
    id: 4,
    problem: "一件商品原价100元，打8折后又减10元，最终价格是多少？这相当于原价的几折？",
    category: "百分比问题",
    difficulty: "中等"
  }
]

export const SmartSolver: React.FC = () => {
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
  const [selectedStrategyLocal, setSelectedStrategyLocal] = useState(selectedStrategy)

  const handleSolve = async () => {
    if (!localProblem.trim()) {
      setError('请输入问题')
      return
    }

    setProblem(localProblem)
    setStrategy(selectedStrategyLocal)
    setLoading(true)
    setError(null)

    try {
      const result = await solveProblem({
        problem: localProblem,
        strategy: selectedStrategyLocal
      })

      setSolveResult(result)
      addToHistory({
        id: generateId(),
        problem: localProblem,
        strategy: selectedStrategyLocal,
        result,
        timestamp: new Date()
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : '解题失败')
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (example: typeof exampleProblems[0]) => {
    setLocalProblem(example.problem)
    setProblem(example.problem)
  }

  const handleClear = () => {
    setLocalProblem('')
    setProblem('')
    setSolveResult(null)
    setError(null)
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* 主要输入区域 */}
      <Card>
        <CardHeader>
          <CardTitle>🧠 智能数学解题系统</CardTitle>
          <p className="text-gray-600 mb-3">
            基于COT-DIR算法的智能推理系统，支持多种推理策略，智能分析数学问题
          </p>
          
          {/* 新增：教育友好和快速实现的介绍 */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 border border-blue-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <span className="text-blue-500">✨</span> 系统优势
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-green-600 text-sm">📚</span>
                </div>
                <div>
                  <div className="font-medium text-gray-800 mb-1">教育友好设计</div>
                  <div className="text-sm text-gray-600 leading-relaxed">
                    基于<strong className="text-blue-600">物性定律的天然可解释性</strong>，每个推理步骤都有清晰的物理依据，
                    帮助学生理解数学概念背后的物理原理，让抽象数学变得直观易懂
                  </div>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-purple-600 text-sm">⚡</span>
                </div>
                <div>
                  <div className="font-medium text-gray-800 mb-1">快速实现部署</div>
                  <div className="text-sm text-gray-600 leading-relaxed">
                    采用<strong className="text-purple-600">基于规则的系统架构</strong>，无需复杂的机器学习训练，
                    开发周期短、技术风险低，可快速集成到现有教育系统中
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-white/70 rounded-lg border border-blue-100">
              <div className="text-xs text-blue-700 flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                <strong>核心技术：</strong> COT-DIR隐式关系发现 + 物理约束传播网络 + OR-Tools约束求解器
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* 示例问题 */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-gray-700">📚 示例问题</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {exampleProblems.map((example) => (
                  <motion.button
                    key={example.id}
                    onClick={() => handleExampleClick(example)}
                    className="text-left p-4 rounded-lg border-2 border-gray-200 hover:border-purple-300 hover:bg-purple-50 transition-colors"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-xs px-2 py-1 bg-blue-100 text-blue-600 rounded-full">
                        {example.category}
                      </span>
                      <span className="text-xs px-2 py-1 bg-green-100 text-green-600 rounded-full">
                        {example.difficulty}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 leading-relaxed mb-3">
                      {example.problem}
                    </p>
                    {example.actualResult && (
                      <div className="mt-2 p-2 bg-green-50 rounded border-l-4 border-green-400">
                        <div className="text-xs text-green-700 mb-1 font-medium">
                          ✅ 实际运行结果
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-gray-500">答案: </span>
                            <span className="font-medium text-green-600">{example.actualResult.answer}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">置信度: </span>
                            <span className="font-medium text-green-600">{example.actualResult.confidence}%</span>
                          </div>
                          <div>
                            <span className="text-gray-500">耗时: </span>
                            <span className="font-medium text-green-600">{example.actualResult.executionTime}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">策略: </span>
                            <span className="font-medium text-green-600">{example.actualResult.strategy}</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </motion.button>
                ))}
              </div>
            </div>

            {/* 问题输入 */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">
                💭 输入数学问题
              </label>
              <Textarea
                placeholder="请输入您要解决的数学问题，系统将进行智能分析..."
                value={localProblem}
                onChange={(e) => setLocalProblem(e.target.value)}
                className="min-h-[120px] text-base"
                error={error}
              />
            </div>

            {/* 策略选择 */}
            <div className="space-y-3">
              <label className="text-sm font-medium text-gray-700">
                🎯 选择推理策略
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {strategyOptions.map((strategy) => (
                  <motion.button
                    key={strategy.id}
                    onClick={() => setSelectedStrategyLocal(strategy.id as any)}
                    className={`p-4 rounded-lg border-2 text-left transition-all ${
                      selectedStrategyLocal === strategy.id
                        ? 'border-purple-500 bg-purple-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-4 h-4 rounded-full ${strategy.color}`} />
                      <div className="flex-1">
                        <div className="font-medium text-gray-900 mb-1">
                          {strategy.name}
                        </div>
                        <div className="text-xs text-gray-600">
                          {strategy.description}
                        </div>
                      </div>
                      {selectedStrategyLocal === strategy.id && (
                        <div className="text-purple-500">✓</div>
                      )}
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* 控制按钮 */}
            <div className="flex gap-4 justify-center">
              <Button
                onClick={handleSolve}
                loading={isLoading}
                disabled={!localProblem.trim()}
                size="lg"
                className="px-8"
              >
                {isLoading ? '🔄 正在解题...' : '🚀 开始解题'}
              </Button>
              <Button
                onClick={handleClear}
                variant="outline"
                size="lg"
                className="px-8"
              >
                🗑️ 清空
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
            className="space-y-6"
          >
            <SolveResultDisplay result={solveResult} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// 结果展示组件
interface SolveResultDisplayProps {
  result: any
}

const SolveResultDisplay: React.FC<SolveResultDisplayProps> = ({ result }) => {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      {/* 答案和基本信息 */}
      <Card className="xl:col-span-1">
        <CardHeader>
          <CardTitle>✅ 解答结果</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="text-3xl font-bold text-green-800 mb-2">
                {result.answer}
              </div>
              <div className="text-sm text-green-600">
                置信度：{(result.confidence * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="text-sm text-blue-600 mb-2">
                <strong>推理策略：</strong>
                {result.strategy === 'auto' ? '自动选择' : 
                 result.strategy === 'cot' ? '思维链推理 (COT)' :
                 result.strategy === 'got' ? '思维图推理 (GOT)' : 
                 result.strategy === 'tot' ? '思维树推理 (TOT)' : result.strategy}
              </div>
              <div className="text-xs text-blue-500">
                基于问题特征自动选择最优策略
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 解题步骤 */}
      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle>📋 详细推理步骤</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {result.steps.map((step: string, index: number) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-start gap-4 p-4 bg-gray-50 rounded-lg"
              >
                <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <div className="text-sm text-gray-800 leading-relaxed">
                    {step}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 简单版实体关系图 - 儿童友好设计 */}
      <div className="xl:col-span-3">
        <EntityRelationshipDiagram
          entities={result.entities}
          relationships={result.relationships}
          width={600}
          height={400}
          diagramMode="simple"
          onEntitySelect={(entity) => console.log('选中实体:', entity)}
        />
      </div>

      {/* 隐含约束 */}
      {result.constraints && result.constraints.length > 0 && (
        <Card className="xl:col-span-3">
          <CardHeader>
            <CardTitle>⚠️ 隐含约束条件</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {result.constraints.map((constraint: string, index: number) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-start gap-3 p-3 bg-orange-50 border border-orange-200 rounded-lg"
                >
                  <div className="w-6 h-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
                    !
                  </div>
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

export default SmartSolver