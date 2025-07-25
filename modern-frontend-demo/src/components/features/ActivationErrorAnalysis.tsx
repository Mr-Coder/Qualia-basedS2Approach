import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import ActivationPropertyGraph from './ActivationPropertyGraph'

// Icons
import { 
  AlertTriangle, 
  XCircle, 
  TrendingDown, 
  Zap,
  Target,
  RefreshCw,
  CheckCircle,
  ArrowRight,
  Brain,
  Search,
  Link,
  Settings,
  ExclamationTriangle
} from 'lucide-react'

interface ErrorType {
  id: string
  name: string
  description: string
  icon: string
  color: string
  examples: string[]
  correctionMethods: string[]
}

interface StrategyError {
  id: string
  strategy: 'COT' | 'GOT' | 'TOT' | 'AUTO'
  name: string
  description: string
  icon: string
  color: string
  commonIssues: string[]
  improvements: string[]
}

interface ErrorAnalysisDetail {
  errorType: string
  problemExample: string
  errorDescription: string
  rootCause: string
  solutionSteps: string[]
  relatedConcepts: string[]
}

const ActivationErrorAnalysis: React.FC = () => {
  const [selectedErrorType, setSelectedErrorType] = useState<string | null>(null)
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [showDetail, setShowDetail] = useState<ErrorAnalysisDetail | null>(null)
  const [activeSection, setActiveSection] = useState<'common' | 'strategy'>('common')

  // 常见错误类型数据
  const commonErrorTypes: ErrorType[] = [
    {
      id: 'entity_recognition_error',
      name: '实体识别错误',
      description: '未能正确识别问题中的关键实体或混淆实体类型',
      icon: '🔍',
      color: 'red',
      examples: [
        '将"小明有5个苹果"中的"5"识别为人名',
        '混淆问题中的主体和客体关系',
        '忽略重要的数量词或单位'
      ],
      correctionMethods: [
        '加强实体标注训练',
        '建立实体分类体系',
        '练习关键词识别技巧'
      ]
    },
    {
      id: 'relationship_error',
      name: '关系理解错误',
      description: '未能正确理解实体间的关系或建立错误的关系',
      icon: '🔗',
      color: 'orange',
      examples: [
        '将"比...多"理解为加法而非减法关系',
        '忽略时间先后顺序的因果关系',
        '混淆比较关系的方向性'
      ],
      correctionMethods: [
        '关系图谱绘制练习',
        '语义关系分析训练',
        '逻辑推理能力培养'
      ]
    },
    {
      id: 'strategy_selection_error',
      name: '策略选择错误',
      description: '选择了不适合当前问题特点的推理策略',
      icon: '🎯',
      color: 'blue',
      examples: [
        '复杂问题使用过于简单的策略',
        '简单问题过度复杂化处理',
        '未根据问题类型调整策略'
      ],
      correctionMethods: [
        '问题类型识别训练',
        '策略适用性判断练习',
        '多策略比较分析'
      ]
    },
    {
      id: 'constraint_ignore_error',
      name: '约束忽略错误',
      description: '忽略了问题中的重要约束条件',
      icon: '⚠️',
      color: 'purple',
      examples: [
        '忽略"非负整数"等数值约束',
        '不考虑实际情境的合理性约束',
        '遗漏隐含的逻辑约束条件'
      ],
      correctionMethods: [
        '约束条件识别训练',
        '合理性检验习惯培养',
        '完整性检查方法学习'
      ]
    }
  ]

  // 策略专项错误分析数据
  const strategyErrors: StrategyError[] = [
    {
      id: 'cot_error',
      strategy: 'COT',
      name: '链式推理错误',
      description: '在思维链推理过程中出现推理跳跃',
      icon: 'COT',
      color: 'purple',
      commonIssues: [
        '推理步骤不完整或跳跃',
        '逻辑链条中断或错误',
        '未能保持推理的连贯性'
      ],
      improvements: [
        '强化逐步推理训练',
        '建立完整的推理链条',
        '加强逻辑连贯性检查'
      ]
    },
    {
      id: 'got_error',
      strategy: 'GOT',
      name: '关系网络构建错误',
      description: '构建的关系网络不完整或存在错误连接',
      icon: 'GOT',
      color: 'green',
      commonIssues: [
        '关系网络构建不完整',
        '节点间连接关系错误',
        '图结构理解偏差'
      ],
      improvements: [
        '图网络构建方法训练',
        '关系映射准确性提升',
        '网络完整性验证方法'
      ]
    },
    {
      id: 'tot_error',
      strategy: 'TOT',
      name: '路径选择偏差错误',
      description: '在多路径探索中选择了次优或错误路径',
      icon: 'TOT',
      color: 'yellow',
      commonIssues: [
        '路径探索不够充分',
        '路径评估标准不当',
        '过早收敛到局部最优'
      ],
      improvements: [
        '多路径探索策略训练',
        '路径评估方法改进',
        '全局最优搜索能力提升'
      ]
    },
    {
      id: 'auto_error',
      strategy: 'AUTO',
      name: '策略选择不当错误',
      description: '自动策略选择器选择了不适合的策略',
      icon: 'AUTO',
      color: 'teal',
      commonIssues: [
        '问题特征识别不准确',
        '策略匹配算法偏差',
        '适应性调整能力不足'
      ],
      improvements: [
        '问题特征提取能力训练',
        '策略匹配算法优化',
        '自适应调整机制改进'
      ]
    }
  ]


  // 显示错误详情
  const showErrorDetail = (errorType: ErrorType) => {
    const detail: ErrorAnalysisDetail = {
      errorType: errorType.name,
      problemExample: '示例：小明有8个苹果，给了小红3个，还剩多少个？（学生答案：11）',
      errorDescription: errorType.description,
      rootCause: '认知激活模式错误，相关概念节点激活不足或连接错误',
      solutionSteps: errorType.correctionMethods,
      relatedConcepts: ['概念激活', '语义解析', '逻辑推理', '元认知监控']
    }
    setShowDetail(detail)
  }

  const showStrategyDetail = (strategy: StrategyError) => {
    const detail: ErrorAnalysisDetail = {
      errorType: strategy.name,
      problemExample: `${strategy.strategy}策略应用示例及常见错误模式`,
      errorDescription: strategy.description,
      rootCause: '策略选择或执行过程中的认知偏差',
      solutionSteps: strategy.improvements,
      relatedConcepts: ['策略选择', '元认知', '问题表征', '认知灵活性']
    }
    setShowDetail(detail)
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* 页面标题 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <span className="text-3xl">🔍</span>
            <div>
              <h1 className="text-2xl font-bold">错题分析</h1>
              <p className="text-sm text-gray-600 mt-1">
                智能分析常见错误类型，提供个性化的改进建议，帮助您提升解题能力
              </p>
            </div>
          </CardTitle>
        </CardHeader>
      </Card>
      {/* 常见错误类型 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span className="text-xl">❌</span>
            <span>常见错误类型</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {commonErrorTypes.map((errorType, index) => (
              <motion.div
                key={errorType.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-all cursor-pointer"
                onClick={() => showErrorDetail(errorType)}
              >
                <div className="flex items-start space-x-4">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-2xl bg-${errorType.color}-100`}>
                    {errorType.icon}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-800 mb-2">{errorType.name}</h3>
                    <p className="text-sm text-gray-600 mb-3">{errorType.description}</p>
                    <Button 
                      size="sm" 
                      variant="outline"
                      className="w-full bg-purple-50 border-purple-200 text-purple-700 hover:bg-purple-100"
                    >
                      查看详情
                    </Button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 策略专项错误分析 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span className="text-xl">🧩</span>
            <span>策略专项错误分析</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {strategyErrors.map((strategy, index) => (
              <motion.div
                key={strategy.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-all cursor-pointer"
                onClick={() => showStrategyDetail(strategy)}
              >
                <div className="flex items-start space-x-4">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-sm font-bold bg-${strategy.color}-100 text-${strategy.color}-800`}>
                    {strategy.icon}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-800 mb-2">{strategy.name}</h3>
                    <p className="text-sm text-gray-600 mb-3">{strategy.description}</p>
                    <Button 
                      size="sm" 
                      variant="outline"
                      className="w-full bg-purple-50 border-purple-200 text-purple-700 hover:bg-purple-100"
                    >
                      查看详情
                    </Button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 错误分析详情弹窗 */}
      <AnimatePresence>
        {showDetail && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowDetail(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold">{showDetail.errorType}</h2>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => setShowDetail(null)}
                  >
                    ✕
                  </Button>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold mb-2">问题示例</h3>
                    <div className="bg-gray-50 p-3 rounded text-sm">
                      {showDetail.problemExample}
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">错误描述</h3>
                    <p className="text-gray-700 text-sm">{showDetail.errorDescription}</p>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">根本原因</h3>
                    <p className="text-gray-700 text-sm">{showDetail.rootCause}</p>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">解决步骤</h3>
                    <ul className="text-sm space-y-1">
                      {showDetail.solutionSteps.map((step, i) => (
                        <li key={i} className="flex items-start">
                          <span className="text-blue-500 mr-2">{i + 1}.</span>
                          <span>{step}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">相关概念</h3>
                    <div className="flex flex-wrap gap-2">
                      {showDetail.relatedConcepts.map((concept, i) => (
                        <span key={i} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                          {concept}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ActivationErrorAnalysis