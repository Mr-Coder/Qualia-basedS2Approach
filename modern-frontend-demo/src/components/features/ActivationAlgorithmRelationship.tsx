import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import ActivationPropertyGraph from './ActivationPropertyGraph'

// Icons
import { 
  Cpu, 
  GitBranch, 
  Network, 
  Zap,
  Target,
  Settings,
  Brain,
  ArrowRight,
  CheckCircle,
  TrendingUp,
  Layers,
  Clock
} from 'lucide-react'

interface AlgorithmComponent {
  id: string
  name: string
  description: string
  component_type: 'core' | 'engine' | 'processor' | 'validator' | 'optimizer'
  activation_level: number
  activation_state: 'inactive' | 'primed' | 'active' | 'peak' | 'decaying'
  processing_steps: string[]
  input_dependencies: string[]
  output_connections: string[]
  performance_metrics: {
    speed: number
    accuracy: number
    complexity: number
  }
  activation_pattern: {
    trigger_conditions: string[]
    propagation_sequence: string[]
    peak_duration: number
  }
}

interface AlgorithmFlow {
  flow_id: string
  name: string
  description: string
  components: AlgorithmComponent[]
  activation_sequence: string[]
  total_processing_time: number
  confidence_score: number
  optimization_suggestions: string[]
}

const ActivationAlgorithmRelationship: React.FC = () => {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('cotdir_complete')
  const [algorithmFlow, setAlgorithmFlow] = useState<AlgorithmFlow | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null)
  const [activationSimulation, setActivationSimulation] = useState<boolean>(false)
  const [currentActivationStep, setCurrentActivationStep] = useState<number>(0)

  // 预定义的算法流程
  const algorithmFlows: {[key: string]: AlgorithmFlow} = {
    cotdir_complete: {
      flow_id: 'cotdir_complete',
      name: 'COT-DIR 完整算法流',
      description: '链式思维导向的隐含关系发现完整处理流程',
      total_processing_time: 2.5,
      confidence_score: 0.92,
      activation_sequence: ['qs2_analyzer', 'ird_engine', 'constraint_validator', 'cot_reasoner', 'result_optimizer'],
      optimization_suggestions: [
        '优化QS²语义分析的并行处理能力',
        '增强IRD引擎的模式匹配精度',
        '改进约束验证的实时反馈机制'
      ],
      components: [
        {
          id: 'qs2_analyzer',
          name: 'QS² 语义分析器',
          description: 'Qualia Structure Semantic Analyzer - 语义结构质性分析',
          component_type: 'engine',
          activation_level: 0.95,
          activation_state: 'peak',
          processing_steps: [
            '文本预处理和分词',
            '实体识别和分类',
            '语义关系提取',
            '质性结构构建'
          ],
          input_dependencies: [],
          output_connections: ['ird_engine', 'constraint_validator'],
          performance_metrics: {
            speed: 0.88,
            accuracy: 0.94,
            complexity: 0.76
          },
          activation_pattern: {
            trigger_conditions: ['问题文本输入', '语义分析请求'],
            propagation_sequence: ['词法分析', '句法分析', '语义分析', '质性构建'],
            peak_duration: 0.5
          }
        },
        {
          id: 'ird_engine',
          name: 'IRD 隐含关系发现引擎',
          description: 'Implicit Relation Discovery - 隐含关系发现核心引擎',
          component_type: 'core',
          activation_level: 0.90,
          activation_state: 'active',
          processing_steps: [
            '基于regex的模式匹配',
            '隐含关系挖掘',
            '关系强度评估',
            '关系网络构建'
          ],
          input_dependencies: ['qs2_analyzer'],
          output_connections: ['cot_reasoner', 'constraint_validator'],
          performance_metrics: {
            speed: 0.82,
            accuracy: 0.91,
            complexity: 0.85
          },
          activation_pattern: {
            trigger_conditions: ['QS²输出激活', '关系发现需求'],
            propagation_sequence: ['模式识别', '关系抽取', '强度计算', '网络构建'],
            peak_duration: 0.8
          }
        },
        {
          id: 'constraint_validator',
          name: '约束验证器',
          description: '物性约束和逻辑约束的实时验证系统',
          component_type: 'validator',
          activation_level: 0.75,
          activation_state: 'active',
          processing_steps: [
            '约束条件识别',
            '物性规律检查',
            '逻辑一致性验证',
            '违约报告生成'
          ],
          input_dependencies: ['qs2_analyzer', 'ird_engine'],
          output_connections: ['cot_reasoner'],
          performance_metrics: {
            speed: 0.95,
            accuracy: 0.97,
            complexity: 0.45
          },
          activation_pattern: {
            trigger_conditions: ['约束检查需求', '验证请求'],
            propagation_sequence: ['约束识别', '规律检查', '一致性验证', '结果输出'],
            peak_duration: 0.3
          }
        },
        {
          id: 'cot_reasoner',
          name: 'COT 链式推理器',
          description: 'Chain of Thought - 链式思维推理核心处理器',
          component_type: 'processor',
          activation_level: 0.88,
          activation_state: 'active',
          processing_steps: [
            '推理链构建',
            '逐步推理执行',
            '中间结果验证',
            '推理路径优化'
          ],
          input_dependencies: ['ird_engine', 'constraint_validator'],
          output_connections: ['result_optimizer'],
          performance_metrics: {
            speed: 0.78,
            accuracy: 0.89,
            complexity: 0.92
          },
          activation_pattern: {
            trigger_conditions: ['推理任务激活', '逻辑链构建需求'],
            propagation_sequence: ['链构建', '步骤执行', '中间验证', '路径优化'],
            peak_duration: 1.2
          }
        },
        {
          id: 'result_optimizer',
          name: '结果优化器',
          description: '最终结果的优化和置信度评估系统',
          component_type: 'optimizer',
          activation_level: 0.70,
          activation_state: 'active',
          processing_steps: [
            '结果候选生成',
            '质量评估',
            '置信度计算',
            '最优结果选择'
          ],
          input_dependencies: ['cot_reasoner'],
          output_connections: [],
          performance_metrics: {
            speed: 0.85,
            accuracy: 0.93,
            complexity: 0.65
          },
          activation_pattern: {
            trigger_conditions: ['结果优化需求', 'COT输出激活'],
            propagation_sequence: ['候选生成', '质量评估', '置信度计算', '最优选择'],
            peak_duration: 0.4
          }
        }
      ]
    },
    lightweight_flow: {
      flow_id: 'lightweight_flow',
      name: '轻量级快速流',
      description: '优化的快速处理流程，适用于简单问题',
      total_processing_time: 1.2,
      confidence_score: 0.85,
      activation_sequence: ['simple_parser', 'basic_reasoner', 'quick_validator'],
      optimization_suggestions: [
        '进一步优化解析速度',
        '增加缓存机制减少重复计算'
      ],
      components: [
        {
          id: 'simple_parser',
          name: '简化解析器',
          description: '轻量级文本解析和实体识别',
          component_type: 'processor',
          activation_level: 0.80,
          activation_state: 'active',
          processing_steps: ['快速分词', '实体识别', '基础关系提取'],
          input_dependencies: [],
          output_connections: ['basic_reasoner'],
          performance_metrics: { speed: 0.95, accuracy: 0.82, complexity: 0.35 },
          activation_pattern: {
            trigger_conditions: ['简单问题输入'],
            propagation_sequence: ['分词', '识别', '提取'],
            peak_duration: 0.2
          }
        },
        {
          id: 'basic_reasoner',
          name: '基础推理器',
          description: '简化的推理处理器',
          component_type: 'core',
          activation_level: 0.75,
          activation_state: 'active',
          processing_steps: ['模式匹配', '直接推理', '结果生成'],
          input_dependencies: ['simple_parser'],
          output_connections: ['quick_validator'],
          performance_metrics: { speed: 0.88, accuracy: 0.80, complexity: 0.50 },
          activation_pattern: {
            trigger_conditions: ['解析结果激活'],
            propagation_sequence: ['匹配', '推理', '生成'],
            peak_duration: 0.6
          }
        },
        {
          id: 'quick_validator',
          name: '快速验证器',
          description: '基础约束检查',
          component_type: 'validator',
          activation_level: 0.65,
          activation_state: 'primed',
          processing_steps: ['基础检查', '结果验证'],
          input_dependencies: ['basic_reasoner'],
          output_connections: [],
          performance_metrics: { speed: 0.98, accuracy: 0.85, complexity: 0.25 },
          activation_pattern: {
            trigger_conditions: ['验证需求'],
            propagation_sequence: ['检查', '验证'],
            peak_duration: 0.2
          }
        }
      ]
    }
  }

  useEffect(() => {
    if (selectedAlgorithm) {
      analyzeAlgorithmFlow()
    }
  }, [selectedAlgorithm])

  const analyzeAlgorithmFlow = async () => {
    setIsAnalyzing(true)
    try {
      // 模拟算法分析
      await new Promise(resolve => setTimeout(resolve, 1500))
      setAlgorithmFlow(algorithmFlows[selectedAlgorithm])
    } finally {
      setIsAnalyzing(false)
    }
  }

  const simulateActivation = async () => {
    if (!algorithmFlow) return
    
    setActivationSimulation(true)
    setCurrentActivationStep(0)
    
    for (let i = 0; i < algorithmFlow.activation_sequence.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      setCurrentActivationStep(i + 1)
    }
    
    setTimeout(() => {
      setActivationSimulation(false)
      setCurrentActivationStep(0)
    }, 2000)
  }

  const getComponentTypeIcon = (type: string) => {
    const icons = {
      core: '🎯',
      engine: '⚙️', 
      processor: '🔄',
      validator: '✅',
      optimizer: '⚡'
    }
    return icons[type as keyof typeof icons] || '🔧'
  }

  const getComponentTypeColor = (type: string) => {
    const colors = {
      core: 'bg-red-100 text-red-800 border-red-200',
      engine: 'bg-blue-100 text-blue-800 border-blue-200',
      processor: 'bg-green-100 text-green-800 border-green-200',
      validator: 'bg-orange-100 text-orange-800 border-orange-200',
      optimizer: 'bg-purple-100 text-purple-800 border-purple-200'
    }
    return colors[type as keyof typeof colors] || colors.core
  }

  const getActivationStateColor = (state: string) => {
    const colors = {
      inactive: 'text-gray-500',
      primed: 'text-yellow-600',
      active: 'text-blue-600',
      peak: 'text-red-600',
      decaying: 'text-orange-600'
    }
    return colors[state as keyof typeof colors] || colors.inactive
  }

  return (
    <div className="space-y-6">
      {/* 算法选择区域 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Cpu className="h-5 w-5" />
            <span>🔬 算法物性关系图谱</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-gray-600 text-sm">
              基于激活扩散理论，可视化展示COT-DIR算法各组件间的物性关系和激活传播模式。
            </p>
            
            {/* 算法流程选择 */}
            <div>
              <label className="block text-sm font-medium mb-2">选择算法流程</label>
              <div className="flex space-x-3">
                {Object.entries(algorithmFlows).map(([key, flow]) => (
                  <Button
                    key={key}
                    size="sm"
                    variant={selectedAlgorithm === key ? "default" : "outline"}
                    onClick={() => setSelectedAlgorithm(key)}
                    className="flex items-center space-x-2"
                  >
                    <GitBranch className="h-4 w-4" />
                    <span>{flow.name}</span>
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-600">
                动态展示算法组件的激活扩散和物性关系
              </div>
              <div className="flex space-x-2">
                <Button
                  onClick={simulateActivation}
                  disabled={activationSimulation || !algorithmFlow}
                  className="flex items-center space-x-2"
                  size="sm"
                >
                  {activationSimulation ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Zap className="h-4 w-4" />
                      </motion.div>
                      <span>激活中...</span>
                    </>
                  ) : (
                    <>
                      <Target className="h-4 w-4" />
                      <span>模拟激活</span>
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 算法流程概览 */}
      <AnimatePresence>
        {algorithmFlow && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* 流程状态总览 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center space-x-2">
                    <Network className="h-5 w-5" />
                    <span>{algorithmFlow.name}</span>
                  </span>
                  <div className="text-sm text-gray-500">
                    置信度: {(algorithmFlow.confidence_score * 100).toFixed(1)}%
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-gray-600">{algorithmFlow.description}</p>
                  
                  <div className="grid md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {algorithmFlow.components.length}
                      </div>
                      <div className="text-xs text-gray-600">算法组件</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {algorithmFlow.total_processing_time}s
                      </div>
                      <div className="text-xs text-gray-600">处理时间</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {algorithmFlow.components.filter(c => c.activation_state === 'active' || c.activation_state === 'peak').length}
                      </div>
                      <div className="text-xs text-gray-600">激活组件</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">
                        {(algorithmFlow.components.reduce((sum, c) => sum + c.activation_level, 0) / algorithmFlow.components.length * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-600">平均激活度</div>
                    </div>
                  </div>

                  {/* 激活序列展示 */}
                  <div>
                    <h4 className="font-medium mb-2 flex items-center">
                      <ArrowRight className="h-4 w-4 mr-2" />
                      激活传播序列
                    </h4>
                    <div className="flex items-center space-x-2 text-sm">
                      {algorithmFlow.activation_sequence.map((compId, index) => {
                        const component = algorithmFlow.components.find(c => c.id === compId)
                        const isCurrentStep = activationSimulation && index < currentActivationStep
                        
                        return (
                          <React.Fragment key={compId}>
                            <div className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-300 ${
                              isCurrentStep 
                                ? 'bg-blue-500 text-white animate-pulse' 
                                : 'bg-gray-100 text-gray-700'
                            }`}>
                              {component?.name || compId}
                            </div>
                            {index < algorithmFlow.activation_sequence.length - 1 && (
                              <ArrowRight className={`h-3 w-3 transition-colors duration-300 ${
                                isCurrentStep ? 'text-blue-500' : 'text-gray-400'
                              }`} />
                            )}
                          </React.Fragment>
                        )
                      })}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 算法组件详细分析 */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center">
                <Settings className="h-5 w-5 mr-2" />
                组件激活状态分析
              </h3>
              
              {algorithmFlow.components.map((component, index) => (
                <motion.div
                  key={component.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className={`${
                    selectedComponent === component.id 
                      ? 'ring-2 ring-blue-400 bg-blue-50' 
                      : ''
                  } ${
                    activationSimulation && 
                    algorithmFlow.activation_sequence.indexOf(component.id) < currentActivationStep
                      ? 'animate-pulse bg-yellow-50'
                      : ''
                  }`}>
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <span className="text-2xl">{getComponentTypeIcon(component.component_type)}</span>
                          <div>
                            <h4 className="font-semibold">{component.name}</h4>
                            <p className="text-sm text-gray-600">{component.description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`text-lg font-bold ${getActivationStateColor(component.activation_state)}`}>
                            {(component.activation_level * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-gray-500">
                            状态: {component.activation_state}
                          </div>
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {/* 组件类型和激活水平 */}
                        <div className="flex items-center justify-between">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getComponentTypeColor(component.component_type)}`}>
                            {component.component_type}
                          </span>
                          <div className="w-32 bg-gray-200 rounded-full h-2">
                            <motion.div
                              className="bg-blue-500 h-2 rounded-full"
                              initial={{ width: 0 }}
                              animate={{ width: `${component.activation_level * 100}%` }}
                              transition={{ duration: 1, delay: index * 0.2 }}
                            />
                          </div>
                        </div>

                        {/* 处理步骤 */}
                        <div>
                          <h5 className="font-medium mb-2 flex items-center">
                            <Layers className="h-4 w-4 mr-2" />
                            处理步骤
                          </h5>
                          <div className="text-sm space-y-1">
                            {component.processing_steps.map((step, i) => (
                              <div key={i} className="flex items-start">
                                <div className="w-4 h-4 rounded-full bg-blue-100 text-blue-600 text-xs flex items-center justify-center mr-2 mt-0.5">
                                  {i + 1}
                                </div>
                                {step}
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* 性能指标 */}
                        <div>
                          <h5 className="font-medium mb-2 flex items-center">
                            <TrendingUp className="h-4 w-4 mr-2" />
                            性能指标
                          </h5>
                          <div className="grid grid-cols-3 gap-3 text-sm">
                            <div className="text-center">
                              <div className="font-semibold text-green-600">
                                {(component.performance_metrics.speed * 100).toFixed(0)}%
                              </div>
                              <div className="text-xs text-gray-600">速度</div>
                            </div>
                            <div className="text-center">
                              <div className="font-semibold text-blue-600">
                                {(component.performance_metrics.accuracy * 100).toFixed(0)}%
                              </div>
                              <div className="text-xs text-gray-600">准确度</div>
                            </div>
                            <div className="text-center">
                              <div className="font-semibold text-orange-600">
                                {(component.performance_metrics.complexity * 100).toFixed(0)}%
                              </div>
                              <div className="text-xs text-gray-600">复杂度</div>
                            </div>
                          </div>
                        </div>

                        {/* 激活模式 */}
                        <div className="bg-gray-50 p-3 rounded-lg">
                          <h5 className="font-medium mb-2 flex items-center">
                            <Clock className="h-4 w-4 mr-2" />
                            激活模式
                          </h5>
                          <div className="text-sm space-y-1">
                            <div>
                              <span className="font-medium">触发条件: </span>
                              {component.activation_pattern.trigger_conditions.join(', ')}
                            </div>
                            <div>
                              <span className="font-medium">传播序列: </span>
                              {component.activation_pattern.propagation_sequence.join(' → ')}
                            </div>
                            <div>
                              <span className="font-medium">峰值持续: </span>
                              {component.activation_pattern.peak_duration}秒
                            </div>
                          </div>
                        </div>

                        {/* 依赖关系 */}
                        <div className="grid md:grid-cols-2 gap-4 text-sm">
                          <div>
                            <h5 className="font-medium mb-1">输入依赖</h5>
                            {component.input_dependencies.length > 0 ? (
                              component.input_dependencies.map(dep => (
                                <div key={dep} className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded mb-1">
                                  {algorithmFlow.components.find(c => c.id === dep)?.name || dep}
                                </div>
                              ))
                            ) : (
                              <div className="text-xs text-gray-500">无依赖</div>
                            )}
                          </div>
                          <div>
                            <h5 className="font-medium mb-1">输出连接</h5>
                            {component.output_connections.length > 0 ? (
                              component.output_connections.map(conn => (
                                <div key={conn} className="text-xs bg-green-50 text-green-700 px-2 py-1 rounded mb-1">
                                  {algorithmFlow.components.find(c => c.id === conn)?.name || conn}
                                </div>
                              ))
                            ) : (
                              <div className="text-xs text-gray-500">终端输出</div>
                            )}
                          </div>
                        </div>

                        {/* 操作按钮 */}
                        <div className="flex space-x-2 pt-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setSelectedComponent(
                              selectedComponent === component.id ? null : component.id
                            )}
                          >
                            {selectedComponent === component.id ? '取消选择' : '查看详情'}
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>

            {/* 算法物性关系图谱可视化 */}
            <Card>
              <CardHeader>
                <CardTitle>🧠 算法组件激活扩散图谱</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-4">
                  动态展示算法组件间的激活扩散模式和物性关系连接
                </p>
                <ActivationPropertyGraph
                  problemText={`${algorithmFlow.name}: ${algorithmFlow.description}`}
                  entities={algorithmFlow.components.map(comp => ({
                    name: comp.name,
                    type: comp.component_type
                  }))}
                />
              </CardContent>
            </Card>

            {/* 优化建议 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5 text-purple-500" />
                  <span>算法优化建议</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {algorithmFlow.optimization_suggestions.map((suggestion, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 bg-purple-50 rounded-lg">
                      <CheckCircle className="h-5 w-5 text-purple-500 mt-0.5" />
                      <div>
                        <div className="font-medium text-purple-800">优化建议 {index + 1}</div>
                        <div className="text-sm text-purple-700">{suggestion}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ActivationAlgorithmRelationship