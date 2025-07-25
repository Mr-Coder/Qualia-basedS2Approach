import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import ActivationPropertyGraph from './ActivationPropertyGraph'

// Icons
import { 
  Target, 
  Zap, 
  TrendingUp, 
  Brain,
  Layers,
  Network,
  GitBranch,
  CheckCircle,
  ArrowRight,
  BarChart3
} from 'lucide-react'

interface StrategyNode {
  id: string
  name: string
  strategy_type: 'cot' | 'got' | 'tot'
  activation_level: number
  effectiveness_score: number
  description: string
  key_features: string[]
  applicable_scenarios: string[]
  activation_pattern: {
    initial_trigger: string
    propagation_path: string[]
    peak_activation: number
  }
}

interface StrategyComparison {
  problem_text: string
  strategies: StrategyNode[]
  optimal_strategy: string
  reasoning: string
  activation_flow: {
    [key: string]: {
      triggered_nodes: string[]
      activation_strength: number[]
      convergence_time: number
    }
  }
}

const ActivationStrategyAnalysis: React.FC = () => {
  const [currentProblem, setCurrentProblem] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [strategyComparison, setStrategyComparison] = useState<StrategyComparison | null>(null)
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [activationAnimation, setActivationAnimation] = useState<string | null>(null)

  // 预设的策略分析示例
  const strategyExamples = [
    {
      problem: "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
      complexity: "简单",
      optimal: "COT"
    },
    {
      problem: "一个图书馆有3层，每层有4个书架，每个书架有5排书，每排有20本书，这个图书馆总共有多少本书？",
      complexity: "中等",
      optimal: "COT"
    },
    {
      problem: "在一个复杂的数学竞赛中，有多种解题路径，需要找到最优解，涉及多个相互关联的概念。",
      complexity: "困难",
      optimal: "TOT"
    }
  ]

  const analyzeStrategies = async () => {
    if (!currentProblem.trim()) return

    setIsAnalyzing(true)
    
    try {
      // 模拟激活扩散策略分析
      await new Promise(resolve => setTimeout(resolve, 2500))
      
      // 基于问题复杂度生成策略分析
      const problemComplexity = currentProblem.length > 100 ? 'complex' : 
                               currentProblem.includes('多少') && currentProblem.includes('总共') ? 'medium' : 'simple'
      
      const mockComparison: StrategyComparison = {
        problem_text: currentProblem,
        strategies: [
          {
            id: 'cot_strategy',
            name: '链式思维 (COT)',
            strategy_type: 'cot',
            activation_level: problemComplexity === 'simple' ? 0.95 : problemComplexity === 'medium' ? 0.85 : 0.65,
            effectiveness_score: problemComplexity === 'simple' ? 0.92 : problemComplexity === 'medium' ? 0.88 : 0.70,
            description: '逐步推理，线性思维链条',
            key_features: [
              '步骤清晰，易于跟踪',
              '逻辑链条明确',
              '适合教学演示',
              '错误容易定位'
            ],
            applicable_scenarios: [
              '简单算术问题',
              '步骤明确的推理',
              '教学演示场景',
              '基础概念应用'
            ],
            activation_pattern: {
              initial_trigger: '问题实体识别',
              propagation_path: ['实体识别', '关系分析', '运算执行', '结果验证'],
              peak_activation: problemComplexity === 'simple' ? 0.95 : 0.85
            }
          },
          {
            id: 'got_strategy',
            name: '图式思维 (GOT)',
            strategy_type: 'got',
            activation_level: problemComplexity === 'simple' ? 0.70 : problemComplexity === 'medium' ? 0.90 : 0.85,
            effectiveness_score: problemComplexity === 'simple' ? 0.75 : problemComplexity === 'medium' ? 0.92 : 0.88,
            description: '关系网络推理，并行处理',
            key_features: [
              '并行处理多个关系',
              '全局视角分析',
              '发现隐含连接',
              '处理复杂关系网络'
            ],
            applicable_scenarios: [
              '多实体关系问题',
              '网络化信息处理',
              '关系发现任务',
              '系统性分析'
            ],
            activation_pattern: {
              initial_trigger: '关系网络构建',
              propagation_path: ['关系识别', '网络构建', '并行分析', '模式发现'],
              peak_activation: problemComplexity === 'medium' ? 0.90 : 0.85
            }
          },
          {
            id: 'tot_strategy',
            name: '树式思维 (TOT)',
            strategy_type: 'tot',
            activation_level: problemComplexity === 'simple' ? 0.40 : problemComplexity === 'medium' ? 0.70 : 0.95,
            effectiveness_score: problemComplexity === 'simple' ? 0.45 : problemComplexity === 'medium' ? 0.75 : 0.95,
            description: '多路径探索，树状搜索',
            key_features: [
              '多方案生成',
              '路径比较评估',
              '最优解搜索',
              '处理复杂决策'
            ],
            applicable_scenarios: [
              '复杂问题求解',
              '多方案比较',
              '优化决策任务',
              '探索性推理'
            ],
            activation_pattern: {
              initial_trigger: '问题空间分析',
              propagation_path: ['方案生成', '并行探索', '评估比较', '最优选择'],
              peak_activation: problemComplexity === 'complex' ? 0.95 : 0.70
            }
          }
        ],
        optimal_strategy: problemComplexity === 'simple' ? 'cot_strategy' : 
                         problemComplexity === 'medium' ? 'got_strategy' : 'tot_strategy',
        reasoning: problemComplexity === 'simple' ? 
                   '简单问题适合使用COT链式推理，步骤清晰，执行效率高' :
                   problemComplexity === 'medium' ?
                   '中等复杂度问题适合GOT图式推理，能够处理多实体关系' :
                   '复杂问题需要TOT树式推理，多路径探索找到最优解',
        activation_flow: {
          'cot_strategy': {
            triggered_nodes: ['entity', 'arithmetic', 'verification', 'decomposition'],
            activation_strength: [0.9, 0.95, 0.8, 0.85],
            convergence_time: 1.2
          },
          'got_strategy': {
            triggered_nodes: ['relation', 'network', 'analysis', 'modeling'],
            activation_strength: [0.85, 0.90, 0.88, 0.82],
            convergence_time: 1.8
          },
          'tot_strategy': {
            triggered_nodes: ['exploration', 'evaluation', 'constraint', 'reasoning'],
            activation_strength: [0.92, 0.90, 0.88, 0.95],
            convergence_time: 2.5
          }
        }
      }
      
      setStrategyComparison(mockComparison)
      
    } finally {
      setIsAnalyzing(false)
    }
  }

  const loadExample = (example: typeof strategyExamples[0]) => {
    setCurrentProblem(example.problem)
  }

  const simulateActivation = (strategyId: string) => {
    setActivationAnimation(strategyId)
    setTimeout(() => setActivationAnimation(null), 3000)
  }

  const getStrategyColor = (strategyType: string) => {
    const colors = {
      cot: 'bg-blue-100 text-blue-800 border-blue-200',
      got: 'bg-green-100 text-green-800 border-green-200',
      tot: 'bg-purple-100 text-purple-800 border-purple-200'
    }
    return colors[strategyType as keyof typeof colors] || colors.cot
  }

  const getStrategyIcon = (strategyType: string) => {
    const icons = {
      cot: <ArrowRight className="h-4 w-4" />,
      got: <Network className="h-4 w-4" />,
      tot: <GitBranch className="h-4 w-4" />
    }
    return icons[strategyType as keyof typeof icons] || icons.cot
  }

  return (
    <div className="space-y-6">
      {/* 策略分析输入 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>🎯 激活扩散策略分析</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-gray-600 text-sm">
              基于激活扩散理论，分析不同推理策略（COT、GOT、TOT）在特定问题上的激活模式和效果比较。
            </p>
            
            {/* 示例问题快速加载 */}
            <div>
              <label className="block text-sm font-medium mb-2">快速加载示例问题</label>
              <div className="flex flex-wrap gap-2">
                {strategyExamples.map((example, index) => (
                  <Button
                    key={index}
                    size="sm"
                    variant="outline"
                    onClick={() => loadExample(example)}
                    className="text-xs"
                  >
                    {example.complexity}问题 (推荐{example.optimal})
                  </Button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">数学问题</label>
              <textarea
                value={currentProblem}
                onChange={(e) => setCurrentProblem(e.target.value)}
                placeholder="输入需要分析推理策略的数学问题..."
                className="w-full p-3 border rounded-lg resize-none"
                rows={4}
              />
            </div>
            
            <div className="text-center">
              <Button
                onClick={analyzeStrategies}
                disabled={isAnalyzing || !currentProblem.trim()}
                className="flex items-center space-x-2"
              >
                {isAnalyzing ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Brain className="h-4 w-4" />
                    </motion.div>
                    <span>策略激活分析中...</span>
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4" />
                    <span>开始策略分析</span>
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 策略比较结果 */}
      <AnimatePresence>
        {strategyComparison && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* 最优策略推荐 */}
            <Card className="border-green-200 bg-green-50">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-green-800">
                  <CheckCircle className="h-5 w-5" />
                  <span>最优策略推荐</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center space-y-2">
                  <div className="text-2xl font-bold text-green-700">
                    {strategyComparison.strategies.find(s => s.id === strategyComparison.optimal_strategy)?.name}
                  </div>
                  <p className="text-green-600">{strategyComparison.reasoning}</p>
                </div>
              </CardContent>
            </Card>

            {/* 策略详细比较 */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center">
                <BarChart3 className="h-5 w-5 mr-2" />
                策略激活对比分析
              </h3>
              
              {strategyComparison.strategies.map((strategy, index) => (
                <motion.div
                  key={strategy.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className={`${
                    strategy.id === strategyComparison.optimal_strategy 
                      ? 'ring-2 ring-green-400 bg-green-50' 
                      : ''
                  } ${activationAnimation === strategy.id ? 'animate-pulse bg-yellow-50' : ''}`}>
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded-lg ${getStrategyColor(strategy.strategy_type)}`}>
                            {getStrategyIcon(strategy.strategy_type)}
                          </div>
                          <div>
                            <h4 className="font-semibold">{strategy.name}</h4>
                            <p className="text-sm text-gray-600">{strategy.description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold text-blue-600">
                            {(strategy.effectiveness_score * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-gray-500">效果评分</div>
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {/* 激活强度条 */}
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>激活强度</span>
                            <span>{(strategy.activation_level * 100).toFixed(0)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <motion.div
                              className={`h-2 rounded-full ${
                                strategy.strategy_type === 'cot' ? 'bg-blue-500' :
                                strategy.strategy_type === 'got' ? 'bg-green-500' : 'bg-purple-500'
                              }`}
                              initial={{ width: 0 }}
                              animate={{ width: `${strategy.activation_level * 100}%` }}
                              transition={{ duration: 1, delay: index * 0.2 }}
                            />
                          </div>
                        </div>

                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <h5 className="font-medium mb-2">核心特征</h5>
                            <ul className="text-sm space-y-1">
                              {strategy.key_features.map((feature, i) => (
                                <li key={i} className="flex items-start">
                                  <CheckCircle className="h-3 w-3 mt-0.5 mr-2 text-green-400" />
                                  {feature}
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h5 className="font-medium mb-2">适用场景</h5>
                            <ul className="text-sm space-y-1">
                              {strategy.applicable_scenarios.map((scenario, i) => (
                                <li key={i} className="flex items-start">
                                  <Target className="h-3 w-3 mt-0.5 mr-2 text-blue-400" />
                                  {scenario}
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>

                        {/* 激活模式 */}
                        <div className="bg-gray-50 p-3 rounded-lg">
                          <h5 className="font-medium mb-2 flex items-center">
                            <Layers className="h-4 w-4 mr-2" />
                            激活传播路径
                          </h5>
                          <div className="flex items-center space-x-2 text-sm">
                            <span className="font-medium text-green-600">
                              {strategy.activation_pattern.initial_trigger}
                            </span>
                            {strategy.activation_pattern.propagation_path.map((step, i) => (
                              <React.Fragment key={i}>
                                <ArrowRight className="h-3 w-3 text-gray-400" />
                                <span className="px-2 py-1 bg-white rounded text-xs">
                                  {step}
                                </span>
                              </React.Fragment>
                            ))}
                          </div>
                          <div className="mt-2 text-xs text-gray-600">
                            峰值激活: {(strategy.activation_pattern.peak_activation * 100).toFixed(0)}%
                          </div>
                        </div>

                        {/* 操作按钮 */}
                        <div className="flex space-x-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => simulateActivation(strategy.id)}
                            className="flex items-center space-x-1"
                          >
                            <Zap className="h-3 w-3" />
                            <span>模拟激活</span>
                          </Button>
                          
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setSelectedStrategy(
                              selectedStrategy === strategy.id ? null : strategy.id
                            )}
                          >
                            {selectedStrategy === strategy.id ? '隐藏详情' : '查看详情'}
                          </Button>
                        </div>

                        {/* 详细激活流程 */}
                        <AnimatePresence>
                          {selectedStrategy === strategy.id && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              className="border-t pt-4"
                            >
                              <h5 className="font-medium mb-2">详细激活流程</h5>
                              <div className="space-y-2">
                                {strategyComparison.activation_flow[strategy.id]?.triggered_nodes.map((node, i) => (
                                  <div key={i} className="flex items-center justify-between text-sm">
                                    <span>{node}</span>
                                    <div className="flex items-center space-x-2">
                                      <div className="w-16 bg-gray-200 rounded-full h-1">
                                        <div 
                                          className="bg-blue-500 h-1 rounded-full"
                                          style={{ 
                                            width: `${(strategyComparison.activation_flow[strategy.id]?.activation_strength[i] || 0) * 100}%` 
                                          }}
                                        />
                                      </div>
                                      <span className="text-xs text-gray-500">
                                        {((strategyComparison.activation_flow[strategy.id]?.activation_strength[i] || 0) * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                              <div className="mt-2 text-xs text-gray-600">
                                收敛时间: {strategyComparison.activation_flow[strategy.id]?.convergence_time}秒
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>

            {/* 策略激活网络可视化 */}
            <Card>
              <CardHeader>
                <CardTitle>🧠 策略激活网络可视化</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-4">
                  动态展示不同推理策略的激活扩散过程和节点连接模式
                </p>
                <ActivationPropertyGraph
                  problemText={strategyComparison.problem_text}
                  entities={strategyComparison.strategies.map(s => ({
                    name: s.name,
                    type: s.strategy_type
                  }))}
                />
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ActivationStrategyAnalysis