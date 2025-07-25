import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Textarea'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'

interface ErrorType {
  id: string
  name: string
  description: string
  examples: string[]
  symptoms: string[]
  solutions: string[]
  color: string
  icon: string
}

interface StrategyError {
  strategy: string
  name: string
  description: string
  causes: string[]
  solutions: string[]
  prevention: string[]
}

interface DiagnosisResult {
  errorType: string
  confidence: number
  analysis: string
  recommendations: string[]
  exercises: string[]
}

const errorTypes: ErrorType[] = [
  {
    id: 'entity-identification',
    name: '实体识别错误',
    description: '未能正确识别问题中的关键实体或混淆实体类型',
    examples: [
      '将"苹果的价格"误认为"苹果的数量"',
      '忽略问题中的隐含实体',
      '混淆相似的实体对象',
      '错误理解实体的属性和状态'
    ],
    symptoms: [
      '答案的量纲不正确',
      '计算过程中出现逻辑错误',
      '忽略了关键信息',
      '混淆了不同类型的对象'
    ],
    solutions: [
      '仔细阅读问题，标记所有实体',
      '使用表格整理实体信息',
      '区分实体的类型和属性',
      '检查实体的完整性'
    ],
    color: 'bg-red-500',
    icon: '🔍'
  },
  {
    id: 'relation-understanding',
    name: '关系理解错误',
    description: '未能正确理解实体间的关系或建立错误的关系',
    examples: [
      '混淆"增加"和"减少"的关系',
      '错误理解比例关系',
      '忽略时间序列关系',
      '建立不存在的因果关系'
    ],
    symptoms: [
      '计算方向错误',
      '运算符号使用错误',
      '逻辑推理不一致',
      '结果不符合常理'
    ],
    solutions: [
      '绘制关系图表达实体关系',
      '使用箭头标示关系方向',
      '检查关系的逻辑一致性',
      '验证关系的现实合理性'
    ],
    color: 'bg-orange-500',
    icon: '🔗'
  },
  {
    id: 'strategy-selection',
    name: '策略选择错误',
    description: '选择了不适合当前问题的推理策略',
    examples: [
      '用COT处理复杂网络关系问题',
      '用TOT处理简单算术问题',
      '策略切换不当',
      '缺乏策略组合思维'
    ],
    symptoms: [
      '解题过程过于复杂',
      '推理效率低下',
      '容易陷入推理死循环',
      '无法发现最优解'
    ],
    solutions: [
      '分析问题的复杂度和特征',
      '了解各策略的适用场景',
      '尝试多种策略组合',
      '根据结果调整策略'
    ],
    color: 'bg-blue-500',
    icon: '🎯'
  },
  {
    id: 'constraint-neglect',
    name: '约束忽略错误',
    description: '忽略了问题中的重要约束条件',
    examples: [
      '忽略数值范围限制',
      '忽略物理可能性约束',
      '忽略逻辑一致性要求',
      '忽略现实意义约束'
    ],
    symptoms: [
      '结果超出合理范围',
      '违背物理定律',
      '逻辑自相矛盾',
      '缺乏现实意义'
    ],
    solutions: [
      '列出所有约束条件',
      '检查隐含约束',
      '验证结果的合理性',
      '建立约束检查机制'
    ],
    color: 'bg-purple-500',
    icon: '⚠️'
  }
]

const strategyErrors: StrategyError[] = [
  {
    strategy: 'COT',
    name: '链式推理断点错误',
    description: '在思维链推理过程中出现推理断点或跳跃',
    causes: [
      '推理步骤过大，跳过中间环节',
      '逻辑链条不完整',
      '状态转移不清晰',
      '验证环节缺失'
    ],
    solutions: [
      '细化推理步骤',
      '建立完整的逻辑链',
      '明确状态转移过程',
      '增加中间验证'
    ],
    prevention: [
      '训练逐步推理思维',
      '建立步骤检查习惯',
      '使用推理模板',
      '增强逻辑思维能力'
    ]
  },
  {
    strategy: 'GOT',
    name: '关系网络构建错误',
    description: '构建的关系网络不完整或存在错误连接',
    causes: [
      '遗漏重要关系',
      '建立虚假关系',
      '网络结构不合理',
      '忽略隐含关系'
    ],
    solutions: [
      '系统性分析所有关系',
      '验证关系的真实性',
      '优化网络结构',
      '发掘隐含关系'
    ],
    prevention: [
      '培养系统思维',
      '训练关系识别能力',
      '使用关系检查表',
      '增强网络分析技能'
    ]
  },
  {
    strategy: 'TOT',
    name: '路径选择偏差错误',
    description: '在多路径探索中选择了次优或错误路径',
    causes: [
      '评估标准不当',
      '路径探索不充分',
      '缺乏比较分析',
      '决策过程主观'
    ],
    solutions: [
      '建立客观评估标准',
      '充分探索所有路径',
      '进行系统比较分析',
      '使用决策支持工具'
    ],
    prevention: [
      '训练多元思维',
      '建立评估框架',
      '增强决策能力',
      '培养批判性思维'
    ]
  },
  {
    strategy: 'AUTO',
    name: '策略选择不当错误',
    description: '自动策略选择机制选择了不适合的策略',
    causes: [
      '问题特征分析不准确',
      '策略适用性判断错误',
      '缺乏策略组合考虑',
      '反馈机制不完善'
    ],
    solutions: [
      '改进问题特征分析',
      '优化策略选择算法',
      '考虑策略组合使用',
      '建立反馈调整机制'
    ],
    prevention: [
      '提升问题分析能力',
      '了解策略特点',
      '培养策略思维',
      '建立经验积累'
    ]
  }
]

const improvementSuggestions = [
  {
    category: '基础能力提升',
    suggestions: [
      '加强数学基础知识学习',
      '培养逻辑思维能力',
      '提高阅读理解能力',
      '增强空间想象力'
    ]
  },
  {
    category: '解题技巧改进',
    suggestions: [
      '掌握问题分析方法',
      '学会画图辅助理解',
      '建立解题模板',
      '培养验证习惯'
    ]
  },
  {
    category: '策略应用优化',
    suggestions: [
      '理解各策略特点',
      '练习策略选择',
      '尝试策略组合',
      '建立策略评估机制'
    ]
  },
  {
    category: '学习方法调整',
    suggestions: [
      '制定个性化学习计划',
      '建立错题集',
      '寻求同伴交流',
      '获得专业指导'
    ]
  }
]

export const ErrorAnalysis: React.FC = () => {
  const [selectedErrorType, setSelectedErrorType] = useState<string | null>(null)
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [diagnosisForm, setDiagnosisForm] = useState({
    problem: '',
    wrongAnswer: '',
    correctAnswer: '',
    strategy: 'auto',
    description: ''
  })
  const [diagnosisResult, setDiagnosisResult] = useState<DiagnosisResult | null>(null)
  const [executionHistory, setExecutionHistory] = useState<any[]>([])
  const [errorPatterns, setErrorPatterns] = useState<any>(null)

  // 获取算法执行历史分析错误模式
  useEffect(() => {
    const fetchExecutionHistory = async () => {
      try {
        const response = await fetch('/api/algorithm/execution/history?limit=20')
        const data = await response.json()
        if (data.success && data.data) {
          setExecutionHistory(data.data)
          
          // 分析错误模式
          const patterns = analyzeErrorPatterns(data.data)
          setErrorPatterns(patterns)
        }
      } catch (error) {
        console.error('获取执行历史失败:', error)
      }
    }

    fetchExecutionHistory()
  }, [])

  // 分析错误模式
  const analyzeErrorPatterns = (history: any[]) => {
    const lowConfidenceExecutions = history.filter(exec => 
      exec.execution_metrics?.average_confidence < 0.7
    )
    
    const slowExecutions = history.filter(exec => 
      exec.total_duration_ms > 5
    )

    const errorStages = history.flatMap(exec => 
      exec.stages?.filter((stage: any) => stage.confidence < 0.6) || []
    )

    // 统计最常见的问题阶段
    const stageErrors = errorStages.reduce((acc, stage) => {
      acc[stage.stage_name] = (acc[stage.stage_name] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const mostProblematicStage = Object.entries(stageErrors)
      .sort((a, b) => b[1] - a[1])[0]

    return {
      totalExecutions: history.length,
      lowConfidenceCount: lowConfidenceExecutions.length,
      slowExecutionsCount: slowExecutions.length,
      avgConfidence: (history.reduce((sum, exec) => 
        sum + (exec.execution_metrics?.average_confidence || 0), 0) / history.length * 100).toFixed(1),
      mostProblematicStage: mostProblematicStage ? {
        name: mostProblematicStage[0],
        count: mostProblematicStage[1]
      } : null,
      commonErrors: identifyCommonErrors(lowConfidenceExecutions),
      recommendations: generateErrorRecommendations(lowConfidenceExecutions, slowExecutions, stageErrors)
    }
  }

  // 识别常见错误
  const identifyCommonErrors = (lowConfidenceExecs: any[]) => {
    const errors = []
    
    if (lowConfidenceExecs.length > 0) {
      // 分析问题文本中的模式
      const geometryProblems = lowConfidenceExecs.filter(exec => 
        exec.problem_text.includes('面积') || exec.problem_text.includes('长方形')
      ).length

      const arithmeticProblems = lowConfidenceExecs.filter(exec => 
        /\d+.*[给买拿]+.*\d+/.test(exec.problem_text)
      ).length

      if (geometryProblems > 0) {
        errors.push({
          type: '几何计算错误',
          count: geometryProblems,
          description: '在几何问题中出现计算或公式应用错误'
        })
      }

      if (arithmeticProblems > 0) {
        errors.push({
          type: '数量关系理解错误', 
          count: arithmeticProblems,
          description: '在算术问题中未能正确理解数量变化关系'
        })
      }
    }

    return errors
  }

  // 生成错误改进建议
  const generateErrorRecommendations = (lowConf: any[], slow: any[], stageErrors: Record<string, number>) => {
    const recommendations = []

    if (lowConf.length > 0) {
      recommendations.push({
        title: '🎯 提升置信度',
        description: `有${lowConf.length}次执行置信度较低，建议重点练习基础概念`,
        priority: 'high',
        actions: ['复习基础概念', '练习相似问题', '检查计算步骤']
      })
    }

    if (slow.length > 0) {
      recommendations.push({
        title: '⚡ 提升执行速度',
        description: `有${slow.length}次执行耗时较长，可能存在理解困难`,
        priority: 'medium', 
        actions: ['练习问题识别', '熟悉常见模式', '提高阅读理解']
      })
    }

    if (stageErrors['实体提取'] > 0) {
      recommendations.push({
        title: '🔍 实体识别训练',
        description: '实体提取阶段经常出错，需要加强实体识别能力',
        priority: 'high',
        actions: ['练习关键词识别', '学习实体分类', '提高文本理解']
      })
    }

    return recommendations
  }
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleDiagnosis = async () => {
    if (!diagnosisForm.problem || !diagnosisForm.wrongAnswer) {
      alert('请填写问题内容和错误答案')
      return
    }

    setIsAnalyzing(true)
    
    // 模拟智能诊断
    setTimeout(() => {
      const result: DiagnosisResult = {
        errorType: 'entity-identification',
        confidence: 0.85,
        analysis: '根据您提供的信息，主要问题出现在实体识别环节。您可能混淆了问题中的关键实体或其属性，导致后续推理出现偏差。',
        recommendations: [
          '仔细重读问题，标记所有实体',
          '使用表格整理实体信息',
          '区分实体的类型和属性',
          '检查实体识别的完整性'
        ],
        exercises: [
          '练习实体识别专项题目',
          '做关系图绘制练习',
          '进行问题分析训练',
          '参加实体建模课程'
        ]
      }
      setDiagnosisResult(result)
      setIsAnalyzing(false)
    }, 2000)
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* 页面标题 */}
      <Card>
        <CardHeader>
          <CardTitle>🔍 错题分析</CardTitle>
          <p className="text-gray-600">
            智能分析常见错误类型，提供个性化的改进建议，帮助您提升解题能力
          </p>
        </CardHeader>
      </Card>

      {/* 实际错误模式分析 */}
      {errorPatterns && (
        <Card>
          <CardHeader>
            <CardTitle>📊 实际错误模式分析</CardTitle>
            <p className="text-gray-600">
              基于最近{errorPatterns.totalExecutions}次算法执行的错误分析
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-red-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-red-700">{errorPatterns.lowConfidenceCount}</div>
                <div className="text-sm text-red-600">低置信度执行</div>
              </div>
              <div className="bg-orange-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-orange-700">{errorPatterns.slowExecutionsCount}</div>
                <div className="text-sm text-orange-600">耗时过长执行</div>
              </div>
              <div className="bg-blue-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-blue-700">{errorPatterns.avgConfidence}%</div>
                <div className="text-sm text-blue-600">平均置信度</div>
              </div>
              <div className="bg-purple-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-purple-700">
                  {errorPatterns.mostProblematicStage?.name || 'N/A'}
                </div>
                <div className="text-sm text-purple-600">最频繁错误阶段</div>
              </div>
            </div>

            {errorPatterns.commonErrors.length > 0 && (
              <div className="mb-6">
                <h4 className="font-semibold text-gray-800 mb-3">🔍 发现的常见错误</h4>
                <div className="space-y-2">
                  {errorPatterns.commonErrors.map((error: any, index: number) => (
                    <div key={index} className="flex items-center justify-between bg-red-50 rounded-lg p-3">
                      <div>
                        <div className="font-medium text-red-800">{error.type}</div>
                        <div className="text-sm text-red-600">{error.description}</div>
                      </div>
                      <div className="text-red-700 font-bold">{error.count}次</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {errorPatterns.recommendations.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-3">💡 改进建议</h4>
                <div className="space-y-3">
                  {errorPatterns.recommendations.map((rec: any, index: number) => (
                    <div key={index} className={`p-4 rounded-lg border-l-4 ${
                      rec.priority === 'high' ? 'border-red-400 bg-red-50' :
                      rec.priority === 'medium' ? 'border-yellow-400 bg-yellow-50' :
                      'border-blue-400 bg-blue-50'
                    }`}>
                      <div className="flex items-center justify-between mb-2">
                        <h5 className="font-medium text-gray-800">{rec.title}</h5>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          rec.priority === 'high' ? 'bg-red-200 text-red-700' :
                          rec.priority === 'medium' ? 'bg-yellow-200 text-yellow-700' :
                          'bg-blue-200 text-blue-700'
                        }`}>
                          {rec.priority === 'high' ? '高优先级' : rec.priority === 'medium' ? '中优先级' : '低优先级'}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-3">{rec.description}</p>
                      <div className="flex flex-wrap gap-2">
                        {rec.actions.map((action: string, actionIndex: number) => (
                          <span key={actionIndex} className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded">
                            {action}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 常见错误类型 */}
      <Card>
        <CardHeader>
          <CardTitle>❌ 常见错误类型</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {errorTypes.map((errorType, index) => (
              <motion.div
                key={errorType.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedErrorType === errorType.id
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedErrorType(
                  selectedErrorType === errorType.id ? null : errorType.id
                )}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-10 h-10 ${errorType.color} rounded-lg flex items-center justify-center text-white text-xl`}>
                    {errorType.icon}
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800">{errorType.name}</h3>
                    <p className="text-sm text-gray-600">{errorType.description}</p>
                  </div>
                </div>
                
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full"
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedErrorType(
                      selectedErrorType === errorType.id ? null : errorType.id
                    )
                  }}
                >
                  {selectedErrorType === errorType.id ? '收起详情' : '查看详情'}
                </Button>
              </motion.div>
            ))}
          </div>

          {/* 错误类型详细信息 */}
          {selectedErrorType && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="mt-6 p-4 bg-gray-50 rounded-lg"
            >
              {(() => {
                const errorType = errorTypes.find(e => e.id === selectedErrorType)!
                return (
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 ${errorType.color} rounded-lg flex items-center justify-center text-white`}>
                        {errorType.icon}
                      </div>
                      <h3 className="text-lg font-semibold">{errorType.name}</h3>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">📝 典型例子</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {errorType.examples.map((example, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-red-500 mt-1">•</span>
                              <span>{example}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">🔍 错误症状</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {errorType.symptoms.map((symptom, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-orange-500 mt-1">•</span>
                              <span>{symptom}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">💡 解决方案</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {errorType.solutions.map((solution, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-green-500 mt-1">•</span>
                              <span>{solution}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )
              })()}
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* 策略专项错误分析 */}
      <Card>
        <CardHeader>
          <CardTitle>🎯 策略专项错误分析</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {strategyErrors.map((strategyError, index) => (
              <motion.div
                key={strategyError.strategy}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedStrategy === strategyError.strategy
                    ? 'border-purple-500 bg-purple-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedStrategy(
                  selectedStrategy === strategyError.strategy ? null : strategyError.strategy
                )}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center text-white font-bold">
                    {strategyError.strategy}
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800">{strategyError.name}</h3>
                    <p className="text-sm text-gray-600">{strategyError.description}</p>
                  </div>
                </div>
                
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full"
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedStrategy(
                      selectedStrategy === strategyError.strategy ? null : strategyError.strategy
                    )
                  }}
                >
                  {selectedStrategy === strategyError.strategy ? '收起详情' : '查看详情'}
                </Button>
              </motion.div>
            ))}
          </div>

          {/* 策略错误详细信息 */}
          {selectedStrategy && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="mt-6 p-4 bg-gray-50 rounded-lg"
            >
              {(() => {
                const strategyError = strategyErrors.find(s => s.strategy === selectedStrategy)!
                return (
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-purple-500 rounded-lg flex items-center justify-center text-white font-bold">
                        {strategyError.strategy}
                      </div>
                      <h3 className="text-lg font-semibold">{strategyError.name}</h3>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">🔍 错误原因</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {strategyError.causes.map((cause, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-red-500 mt-1">•</span>
                              <span>{cause}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">💡 解决方案</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {strategyError.solutions.map((solution, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-green-500 mt-1">•</span>
                              <span>{solution}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">🛡️ 预防措施</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {strategyError.prevention.map((prevention, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-blue-500 mt-1">•</span>
                              <span>{prevention}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )
              })()}
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* 智能错题诊断工具 */}
      <Card>
        <CardHeader>
          <CardTitle>🧠 智能错题诊断工具</CardTitle>
          <p className="text-sm text-gray-600">
            输入您的错题信息，系统将进行智能分析并提供个性化建议
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Textarea
                label="问题内容"
                placeholder="请输入原始问题..."
                value={diagnosisForm.problem}
                onChange={(e) => setDiagnosisForm({...diagnosisForm, problem: e.target.value})}
                className="min-h-[100px]"
              />
              
              <div className="space-y-4">
                <Input
                  label="错误答案"
                  placeholder="您的答案..."
                  value={diagnosisForm.wrongAnswer}
                  onChange={(e) => setDiagnosisForm({...diagnosisForm, wrongAnswer: e.target.value})}
                />
                
                <Input
                  label="正确答案（可选）"
                  placeholder="正确答案..."
                  value={diagnosisForm.correctAnswer}
                  onChange={(e) => setDiagnosisForm({...diagnosisForm, correctAnswer: e.target.value})}
                />
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Select
                label="使用策略"
                value={diagnosisForm.strategy}
                onChange={(e) => setDiagnosisForm({...diagnosisForm, strategy: e.target.value})}
                options={[
                  { value: 'auto', label: '自动选择' },
                  { value: 'cot', label: 'COT推理' },
                  { value: 'got', label: 'GOT推理' },
                  { value: 'tot', label: 'TOT推理' }
                ]}
              />
              
              <Textarea
                label="错误描述"
                placeholder="描述您遇到的困难..."
                value={diagnosisForm.description}
                onChange={(e) => setDiagnosisForm({...diagnosisForm, description: e.target.value})}
                className="min-h-[80px]"
              />
            </div>
            
            <div className="flex justify-center">
              <Button
                onClick={handleDiagnosis}
                loading={isAnalyzing}
                size="lg"
                className="px-8"
              >
                {isAnalyzing ? '🔄 正在分析...' : '🔍 开始诊断'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 诊断结果 */}
      {diagnosisResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>📊 诊断结果</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-800 mb-2">🎯 错误类型</h4>
                  <p className="text-sm text-blue-700 mb-2">
                    {errorTypes.find(e => e.id === diagnosisResult.errorType)?.name}
                  </p>
                  <div className="bg-blue-100 rounded-lg p-2">
                    <span className="text-xs text-blue-600">
                      置信度：{(diagnosisResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h4 className="font-semibold text-green-800 mb-2">📝 分析结果</h4>
                  <p className="text-sm text-green-700">
                    {diagnosisResult.analysis}
                  </p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                    <h4 className="font-semibold text-orange-800 mb-2">💡 改进建议</h4>
                    <ul className="text-sm text-orange-700 space-y-1">
                      {diagnosisResult.recommendations.map((rec, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="text-orange-500 mt-1">•</span>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <h4 className="font-semibold text-purple-800 mb-2">📚 练习建议</h4>
                    <ul className="text-sm text-purple-700 space-y-1">
                      {diagnosisResult.exercises.map((exercise, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="text-purple-500 mt-1">•</span>
                          <span>{exercise}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* 个性化改进建议 */}
      <Card>
        <CardHeader>
          <CardTitle>🚀 个性化改进建议</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {improvementSuggestions.map((category, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-gray-50 rounded-lg p-4"
              >
                <h3 className="font-semibold text-gray-800 mb-3">{category.category}</h3>
                <ul className="space-y-2">
                  {category.suggestions.map((suggestion, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                      <span className="text-green-500 mt-1">✓</span>
                      <span>{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ErrorAnalysis