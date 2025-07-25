import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useProblemStore } from '@/stores/problemStore'

// Icons
import { 
  Brain, 
  Zap, 
  Network, 
  Target,
  CheckCircle,
  ArrowRight,
  Clock,
  TrendingUp,
  Play,
  Pause
} from 'lucide-react'

// 🧠 基于交互式物性图谱的节点结构
interface PropertyNode {
  id: string
  name: string
  description: string
  category: 'concept' | 'strategy' | 'domain' | 'skill'
  activation_level: number
  activation_state: 'inactive' | 'primed' | 'active' | 'decaying'
  details: string[]
  x: number
  y: number
  connections: string[]
}

interface PropertyConnection {
  from: string
  to: string
  label: string
  type: 'dependency' | 'application' | 'enhancement' | 'example'
  weight: number
}

interface ActivationStep {
  step_id: number
  node_id: string
  node_name: string
  node_type: 'concept' | 'strategy' | 'domain' | 'skill'
  activation_level: number
  activation_state: string
  description: string
  details: string[]
  reasoning: string
}

const ActivationSmartSolver: React.FC = () => {
  const { currentProblem, setSolveResult, solveResult } = useProblemStore()
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [activationSteps, setActivationSteps] = useState<ActivationStep[]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [selectedActivationNode, setSelectedActivationNode] = useState<string | null>(null)
  const [analysisPhase, setAnalysisPhase] = useState<'input' | 'activating' | 'reasoning' | 'complete'>('input')

  const solveProblem = async () => {
    if (!currentProblem.trim()) return

    setIsAnalyzing(true)
    setAnalysisPhase('activating')
    setCurrentStep(0)

    try {
      // 模拟激活扩散分析过程
      const mockActivationSteps: ActivationStep[] = [
        {
          step_id: 1,
          node_id: "entity",
          node_name: "实体识别",
          node_type: "concept",
          activation_level: 0.85,
          activation_state: "active",
          description: "识别问题中的基本对象",
          details: ["人物: 小明、小红", "物品: 苹果", "数量: 5个、3个"],
          reasoning: "概念'实体'被强激活，表明问题涉及基本对象识别相关内容"
        },
        {
          step_id: 2,
          node_id: "arithmetic",
          node_name: "算术运算",
          node_type: "domain",
          activation_level: 0.95,
          activation_state: "active",
          description: "基本数学运算",
          details: ["运算类型: 加法", "操作数: 5, 3", "关键词: 一共"],
          reasoning: "领域'算术'被强激活，问题属于基本数学运算范畴"
        },
        {
          step_id: 3,
          node_id: "cot",
          node_name: "链式推理",
          node_type: "strategy",
          activation_level: 0.90,
          activation_state: "active",
          description: "逐步推理策略",
          details: ["步骤1: 识别数量", "步骤2: 确定运算", "步骤3: 执行计算"],
          reasoning: "策略'链式思维'被强激活，建议采用逐步推理方法求解"
        },
        {
          step_id: 4,
          node_id: "modeling",
          node_name: "数学建模",
          node_type: "skill",
          activation_level: 0.75,
          activation_state: "active",
          description: "构建数学模型",
          details: ["模型: 5 + 3 = ?", "变量定义明确", "运算关系清晰"],
          reasoning: "技能'建模'被强激活，需要运用数学建模能力"
        },
        {
          step_id: 5,
          node_id: "verification",
          node_name: "结果验证",
          node_type: "skill",
          activation_level: 0.70,
          activation_state: "active",
          description: "验证解答正确性",
          details: ["检查: 5 + 3 = 8", "合理性: 符合实际", "约束: 满足非负整数"],
          reasoning: "技能'验证'被强激活，需要运用结果验证能力"
        }
      ]

      // 逐步展示激活过程
      for (let i = 0; i < mockActivationSteps.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 800))
        setActivationSteps(prev => [...prev, mockActivationSteps[i]])
        setCurrentStep(i + 1)
      }

      setAnalysisPhase('reasoning')
      await new Promise(resolve => setTimeout(resolve, 1000))

      // 生成最终解答
      const finalResult = {
        entities: [
          { id: 'xiaoming', name: '小明', type: 'person' },
          { id: 'xiaohong', name: '小红', type: 'person' },
          { id: 'apples', name: '苹果', type: 'object' },
          { id: 'number_5', name: '5', type: 'number' },
          { id: 'number_3', name: '3', type: 'number' }
        ],
        relationships: [
          { from: 'xiaoming', to: 'apples', type: 'has', label: '拥有5个' },
          { from: 'xiaohong', to: 'apples', type: 'has', label: '拥有3个' }
        ],
        reasoning_steps: mockActivationSteps.map((step, index) => ({
          step: index + 1,
          description: step.reasoning,
          confidence: step.activation_level
        })),
        final_answer: "8个苹果",
        confidence: 0.92,
        method: "activation_diffusion_reasoning"
      }

      setSolveResult(finalResult)
      setAnalysisPhase('complete')

    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getNodeIcon = (nodeType: string) => {
    const icons = {
      concept: '💡',
      strategy: '🎯', 
      domain: '📚',
      skill: '🛠️'
    }
    return icons[nodeType as keyof typeof icons] || '⚡'
  }

  const getActivationColor = (level: number) => {
    if (level > 0.8) return 'text-red-600 bg-red-50'
    if (level > 0.6) return 'text-orange-600 bg-orange-50'
    if (level > 0.4) return 'text-yellow-600 bg-yellow-50'
    return 'text-green-600 bg-green-50'
  }

  return (
    <div className="space-y-6">
      {/* 问题输入区域 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>🧠 激活扩散智能求解</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                数学问题 (激活扩散将自动分析物性关系)
              </label>
              <textarea
                value={currentProblem}
                onChange={(e) => useProblemStore.getState().setCurrentProblem(e.target.value)}
                placeholder="输入数学问题，例如：小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
                className="w-full p-3 border rounded-lg resize-none"
                rows={3}
              />
            </div>
            
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-600">
                基于激活扩散理论的智能关联分析
              </div>
              <Button
                onClick={solveProblem}
                disabled={isAnalyzing || !currentProblem.trim()}
                className="flex items-center space-x-2"
              >
                {isAnalyzing ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Zap className="h-4 w-4" />
                    </motion.div>
                    <span>激活分析中...</span>
                  </>
                ) : (
                  <>
                    <Network className="h-4 w-4" />
                    <span>开始激活扩散</span>
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 激活扩散过程展示 */}
      <AnimatePresence>
        {activationSteps.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5" />
                  <span>激活扩散过程</span>
                  <div className="ml-auto text-sm text-gray-500">
                    {currentStep}/{activationSteps.length} 步骤
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {activationSteps.map((step, index) => (
                    <motion.div
                      key={step.step_id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className={`p-4 rounded-lg border-l-4 ${
                        step.activation_level > 0.8 ? 'border-red-500 bg-red-50' :
                        step.activation_level > 0.6 ? 'border-orange-500 bg-orange-50' :
                        'border-green-500 bg-green-50'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <span className="text-lg">{getNodeIcon(step.node_type)}</span>
                            <span className="font-semibold">{step.node_name}</span>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getActivationColor(step.activation_level)}`}>
                              激活度: {(step.activation_level * 100).toFixed(0)}%
                            </span>
                          </div>
                          <p className="text-sm text-gray-700 mb-2">{step.reasoning}</p>
                          <div className="text-xs text-gray-600 space-y-1">
                            {step.details.map((detail, i) => (
                              <div key={i}>• {detail}</div>
                            ))}
                          </div>
                        </div>
                        <div className="ml-4">
                          {index < activationSteps.length - 1 && (
                            <ArrowRight className="h-4 w-4 text-gray-400" />
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 激活扩散图谱可视化 */}
      <AnimatePresence>
        {analysisPhase !== 'input' && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
          >
            <ActivationPropertyGraph
              problemText={currentProblem}
              entities={solveResult?.entities || []}
              onNodeActivation={(nodeId, level) => {
                setSelectedActivationNode(nodeId)
                console.log(`Node ${nodeId} activated with level ${level}`)
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* 最终解答 */}
      <AnimatePresence>
        {analysisPhase === 'complete' && solveResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card className="border-green-200 bg-green-50">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-green-800">
                  <CheckCircle className="h-5 w-5" />
                  <span>激活扩散分析完成</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-700 mb-2">
                      {solveResult.final_answer}
                    </div>
                    <div className="text-sm text-green-600">
                      置信度: {(solveResult.confidence * 100).toFixed(1)}% | 
                      方法: 激活扩散推理
                    </div>
                  </div>
                  
                  <div className="grid md:grid-cols-2 gap-4 mt-4">
                    <div className="bg-white p-4 rounded-lg">
                      <h4 className="font-semibold mb-2 flex items-center">
                        <Brain className="h-4 w-4 mr-2" />
                        激活节点统计
                      </h4>
                      <div className="text-sm space-y-1">
                        <div>强激活节点: {activationSteps.filter(s => s.activation_level > 0.8).length}</div>
                        <div>中激活节点: {activationSteps.filter(s => s.activation_level > 0.6 && s.activation_level <= 0.8).length}</div>
                        <div>总激活强度: {activationSteps.reduce((sum, s) => sum + s.activation_level, 0).toFixed(2)}</div>
                      </div>
                    </div>
                    
                    <div className="bg-white p-4 rounded-lg">
                      <h4 className="font-semibold mb-2 flex items-center">
                        <TrendingUp className="h-4 w-4 mr-2" />
                        推理路径质量
                      </h4>
                      <div className="text-sm space-y-1">
                        <div>推理步骤: {activationSteps.length}</div>
                        <div>平均置信度: {(activationSteps.reduce((sum, s) => sum + s.activation_level, 0) / activationSteps.length * 100).toFixed(1)}%</div>
                        <div>覆盖领域: {new Set(activationSteps.map(s => s.node_type)).size}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ActivationSmartSolver